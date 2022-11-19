import os
import glob
import requests
from datetime import datetime
import numpy as np
import pandas as pd
import sqlalchemy as sa
import warnings

from timeit import default_timer as timer

from astroquery.mast import Observations, Catalogs
import astropy.table import Table

from src.source import angle_diff
from src.observatory import VirtualObservatory, ParsObservatory
from src.dataset import DatasetMixin, RawData, Lightcurve
from src.catalog import Catalog

class ParsObsTESS(ParsObservatory):

    # must register this observatory in list of allowed names
    ParsObservatory.add_to_obs_names("TESS")

    def __init__(self, **kwargs):

        super().__init__("tess")

        self._enforce_type_checks = True
        self._enforce_no_new_attrs = True

        config = self.load_then_update(kwargs)

        # apply parameters specific to this class
        self.apply_specific_pars(config)

class VirtualTESS(VirtualObservatory):
    def __init__(self, **kwargs):
        """
        Generate an instance of a VirtualTESS object.
        This can be used to download TESS data
        and run analysis on it.

        Parameters
        ----------
        Are the same as the VirtualObservatory class.
        The only difference is that the obs_name is set to "tess".
        """

        self.pars = ParsObsTESS(**kwargs)
        super().__init__(name="tess")

    def reduce_to_lightcurves(
        self,
        datasets,
        source=None,
        init_kwargs={},
        mag_range=0.75,
        radius=3,
        gap=40,
        drop_bad=False,
        **_,
    ):
        """
        Reduce the datasets to lightcurves.
        Splits up the raw data that corresponds
        to several object IDs (oid) into separate
        lightcurves. Will also split observing seasons
        into separate lightcurves, if there's a gap
        of more than X days (given by the "gap" parameter).

        Parameters
        ----------
        datasets: a list of src.dataset.RawData objects
            The raw data to reduce.
        source: src.source.Source object
            The source to which the dataset belongs.
            If None, the reduction will not use any
            data of the source, such as the expected
            magnitude, the position, etc.
        init_kwargs: dict
            A dictionary of keyword arguments to be
            passed to the constructor of the new dataset.
        mag_range: float or None
            If not None, and if the source is also given,
            this value will be used to remove datasets
            where the median magnitude is outside of this range,
            relative to the source's magnitude.
        radius: float
            The maximum distance (in arcesconds) from the source
            for each oid lightcurve to be considered a match.
            If outside the radius, the oid lightcurve will be
            dropped. Only works if given the source.
        gap: float
            If there is a gap in a lightcurve where there are
            no observations for this many days, split into
            separate lightcurves.
        drop_bad: bool
            If True, any points in the lightcurves will be
            dropped if their flag is non-zero
            or if their magnitude is NaN.
            This reduces the output size but will
            also not let bad data be transferred
            down the pipeline for further review.

        Returns
        -------
        a list of src.dataset.Lightcurve objects
            The reduced datasets, after minimal processing.
            The reduced datasets will have uniform filter,
            each dataset will be sorted by time,
            and some initial processing will be done,
            using the "reducer" parameter (or function inputs).
        """
        allowed_types = "photometry"
        allowed_dataclasses = pd.DataFrame

        for i, d in enumerate(datasets):
            # check the raw input types make sense
            if d.type is None or d.type not in allowed_types:
                raise ValueError(
                    f"Expected RawData to contain {str(allowed_types)}, "
                    f"but dataset {i} was a {d.type} dataset."
                )
            if not isinstance(d.data, allowed_dataclasses):
                raise ValueError(
                    f"Expected RawData to contain {str(allowed_dataclasses)}, "
                    f"but data in dataset {i} was a {type(d.data)} object."
                )

        data = pd.concat([d.data for d in datasets])

        time_col = datasets[0].colmap["time"]
        mjd_conversion = datasets[0].time_info["to mjd"]
        exp_col = datasets[0].colmap["exptime"]
        filt_col = datasets[0].colmap["filter"]
        flag_col = datasets[0].colmap["flag"] if "flag" in datasets[0].colmap else None
        mag_col = datasets[0].colmap["mag"]
        magerr_col = datasets[0].colmap["magerr"]
        ra_col = datasets[0].colmap["ra"]
        dec_col = datasets[0].colmap["dec"]

        # all filters in this dataset
        filters = list(set(data[filt_col]))

        # split the dataset into oids
        oid_dfs = []
        object_ids = list(set(data["oid"]))
        for oid in object_ids:
            new_oid_df = data[data["oid"] == oid]
            bad_idx = (new_oid_df[flag_col] != 0) | (new_oid_df[mag_col].isna())
            df = new_oid_df[~bad_idx].reset_index(
                drop=True, inplace=False
            )  # cleaned data
            # check that the objects are close to the source
            if source and source.ra is not None and source.dec is not None and radius:
                dRA = angle_diff(df[ra_col].median(), source.ra)
                dRA *= 3600 * np.cos(np.radians(source.dec))
                dDec = (df[dec_col].median() - source.dec) * 3600
                dist = np.sqrt(dRA**2 + dDec**2)
                if dist > radius:
                    continue

            # check the source magnitude is within the range
            if source and source.mag is not None and mag_range:
                mag = df[mag_col]
                if np.all(np.isnan(mag)):
                    continue
                mag_diff = abs(source.mag - np.nanmedian(mag))
                if mag_diff > mag_range:
                    continue

            # verify that all data for the same oid
            # has the same filter
            if len(list(set(df[filt_col]))) > 1:
                raise ValueError(
                    f"Expected all data for the same oid to have the same filter, "
                    f"but the oid {df['oid'].iloc[0]} had filters {filters}."
                )
            oid_dfs.append(new_oid_df)

        # split the data into lightcurves
        # based on the gap between observations
        data = pd.concat(oid_dfs)
        data_sort = data.sort_values(by=[time_col], inplace=False)
        data_sort.reset_index(drop=True)

        dt = np.diff(mjd_conversion(data_sort[time_col]))

        gaps = np.where(dt > gap)[0]
        gaps = np.append(gaps, len(data))
        prev_idx = 0

        dfs = []
        for idx in gaps:
            df_gap = data_sort[prev_idx : idx + 1]
            for filt in filters:
                df = df_gap[df_gap[filt_col] == filt].reset_index(
                    drop=True, inplace=False
                )

                # drop the flagged or NaN values
                if drop_bad:
                    bad_idx = (df[flag_col] != 0) | (df[mag_col].isna())
                    df = df[~bad_idx]

                if len(df) > 0:
                    dfs.append(df)

            prev_idx = idx + 1

        new_datasets = []
        keep_columns = [
            time_col,
            exp_col,
            filt_col,
            mag_col,
            magerr_col,
            ra_col,
            dec_col,
            flag_col,
        ]
        for df in dfs:
            df = df[keep_columns]
            if len(df) > 0:
                new_datasets.append(Lightcurve(data=df, **init_kwargs))

        return new_datasets

    def fetch_data_from_observatory(self, cat_row, verbose=0, **kwargs):
        """
        Fetch data from TESS for a given source.
        Must return a dataframe (or equivalent), even if it is an empty one.
        This must be implemented by each observatory subclass.

        Parameters
        ----------
        cat_row: dict like
            A row in the catalog for a specific source.
            In general, this row should contain the following keys:
            name, ra, dec, mag, filter_name (saying which band the mag is in).
        kwargs: dict
            Additional keyword arguments to pass to the fetcher.

        Returns
        -------
        data : pandas.DataFrame or other data structure
            Raw data from the observatory, to be put into a RawData object.
        altdata: dict
            Additional data to be stored in the RawData object.

        """
        # TODO: figure out dstArcSec threshold for flagging close stars
        DST_THRESHOLD = 10
        # TODO: figure out mag diff threshold for flagging similar stars
        MAGDIFF_THRESHOLD = 0.5
        # TODO: figure out optimal radii for queries
        
        # name should be source_id for GAIA catalog entry
        name = cat_row['name']
        dec = cat_row['dec']
        ra = cat_row['ra']
        mag = cat_row['mag']
        if mag > 16:
            # TESS can't see stars fainter than 15 mag
            print(f"Magnitude of {mag} is too faint for TESS.")
            return pd.DataFrame()

        cone_radius = kwargs['cone_radius'] \
                        if 'cone_radius' in kwargs else 0.01
        sign = ' -' if dec < 0 else ' '
        coord_str = str(ra) + sign + str(dec)
        
        kwargs = {
            'coordinates': coord_str,
            'catalog': 'TIC',
            'radius': cone_radius
        }
        catalog_data = self._try_query(Catalogs.query_region, kwargs)
        if len(catalog_data) == 0:
            print("No TESS object found for given catalog row.")
            return pd.DataFrame()
        
        # TODO: check for better way of selecting candidate
        # catalog_data is automatically sorted by dstArcSec, index 0 is closest
        # also, is GAIAmag guaranteed to exist? 
        # what if we're not using a GAIA cat_row?
        candidate = 0
        while abs(catalog_data['GAIAmag'][candidate] - mag) > MAGDIFF_THRESHOLD:
            candidate += 1
        
        # if there are multiple likely candidates, throw warning
        close_mags = 0
        for i in range(len(catalog_data)):
            gm = catalog_data['GAIAmag'][i]
            dst = catalog_data['dstArcSec'][i]
            if abs(gm - mag) < MAGDIFF_THRESHOLD and dst < DST_THRESHOLD:
                close_mags += 1
        if close_mags > 1:
            warnings.warn(f"Multiple ({close_mags}) sources with similar " +
                            "magnitudes in cone search." +
                            " Might select incorrect source.")

        TESS_ID = catalog_data['ID'][candidate]
        kwargs = {
            'objectname': 'TIC ' + TESS_ID,
            'radius': 0.1,
            'obs_collection': 'TESS',
            # 'dataproduct_type': 'timeseries'
        }
        data_query = self._try_query(Observations.query_criteria, kwargs)
        if len(catalog_data) == 0:
            print("No data found for object " + 'TIC ' + TESS_ID + '.')
            return pd.DataFrame()
        if 'timeseries' not in data_query['dataproduct_type']:
            print("No timeseries data found for object " + 
                    'TIC ' + TESS_ID + '.')
            return pd.DataFrame()
        
        lc_indices = []
        for i, uri in enumerate(data_query['dataURL']):
            if isinstance(uri, str) and uri[-5:-7] == 'lc':
                lc_indices.append(i)
        print(f"Found {len(lc_indices)} light curve(s) for this source.")

        base_url = "https://mast.stsci.edu/api/v0.1/Download/file?uri="

        # how to properly aggregate data if more than 1 LC?
        # do we just add rows naively?
        df_list = []
        for i in lc_indices:
            uri = data_query['dataURL'][i]
            table = Table.read(base_url + uri, format='fits')
            df_list.append(table.to_pandas())
        data = pd.concat(df_list, ignore_index=True)

        first_uri = data_query['dataURL'][lc_indices[0]]
        # what other metadata should we save from a FITS file?
        altdata = {}
        with fits.open(base_url + first_uri, cache=False) as hdul:
            altdata['lightcurve columns'] = hdul[1].data.columns.__str__()
            altdata['aperture matrix'] = hdul[2].data
        
        return data, altdata


    def _try_query(self, query_fn, kwargs):
        """
        Makes an astroquery request repeatedly, ignoring any timeout errors.
        Returns first successful response, otherwise raises TimeoutError.
        """
        # maybe try using multiprocessing to terminate after 10 secs?
        tries = 1
        while tries <= 10:
            try:
                print(f"Making query request, attempt {tries}/10 ...")
                ret = query_fn(**kwargs)
                return ret
            except ReadTimeoutError as e:
                # print(e)
                print(f"Request timed out.")

        raise TimeoutError(f"Too many timeouts from query request.")
        
