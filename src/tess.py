import os
import glob
import requests
from datetime import datetime
import numpy as np
import pandas as pd
import sqlalchemy as sa
import warnings
import socket

from timeit import default_timer as timer

from astroquery.mast import Observations, Catalogs
from astropy.table import Table
import astropy.io.fits as fits

from src.source import angle_diff
from src.observatory import VirtualObservatory, ParsObservatory
from src.catalog import Catalog
from src.dataset import DatasetMixin, RawPhotometry, Lightcurve

class ParsObsTESS(ParsObservatory):

    # must register this observatory in list of allowed names
    ParsObservatory.add_to_obs_names("TESS")

    def __init__(self, **kwargs):

        super().__init__("tess")

        self.distance_thresh = self.add_par(
            "distance_thresh",
            10.0, 
            float, 
            "Distance threshold in arcseconds for flagging "
            "close stars while querying TIC."
        )

        self.magdiff_thresh = self.add_par(
            "magdiff_thresh",
            0.75, 
            float, 
            "Magnitude difference threshold for flagging "
            "similar stars while querying TIC."
        )

        self.cat_qry_radius = self.add_par(
            "cat_qry_radius",
            0.01,
            float,
            "Radius in degrees for cone search for MAST Catalogs query."
            "This query is to find the TESS ID for given source."
        )

        self.obs_qry_radius = self.add_par(
            "obs_qry_radius",
            0.1,
            float,
            "Radius in degrees for cone search for MAST Observations query."
            "This query is to find data from TIC for given source."
        )

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
        dataset,
        source=None,
        init_kwargs={},
        mag_range=0.75,
        radius=3,
        gap=40,
        drop_bad=False,
        **_,
    ):
        """
        Not implemented yet. 
        Similar to reduce() from ztf.py.
        """
        raise NotImplementedError()

    def fetch_data_from_observatory(
        self,
        cat_row,
        verbose=0,
    ):
        """
        Fetch data from TESS for a given source.
        Returns a dataframe including all TESS observations of given source.
        Returns empty dataframe if:
        - source magnitude is too high (too faint for TESS)
        - source doesn't exist in TESS catalog
        - TESS has no data for source

        Parameters
        ----------
        cat_row: dict like
            A row in the catalog for a specific source.
            In general, this row should contain the following keys:
            name, ra, dec, mag, filter_name (saying which band the mag is in).

        Returns
        -------
        data : pandas.DataFrame or other data structure
            Raw data from the observatory, to be put into a RawPhotometry object.
        altdata: dict
            Additional data to be stored in the RawPhotometry object.

        """
        
        # name should be source_id for GAIA catalog entry
        name = cat_row['name']
        dec = cat_row['dec']
        ra = cat_row['ra']
        mag = cat_row['mag']
        if mag > 16:
            # TESS can't see stars fainter than 15 mag
            self._print(f"Magnitude of {mag} is too faint for TESS.", verbose)
            return pd.DataFrame()
        
        # DELETE
        print(f"{name} \t {mag} \t {ra} \t {dec}")

        coord_str = str(ra) + ' ' + str(dec)
        cat_params = {
            'coordinates': coord_str,
            'catalog': 'TIC',
            'radius': self.pars.cat_qry_radius
        }
        catalog_data = self._try_query(Catalogs.query_region, cat_params, verbose)
        if len(catalog_data) == 0:
            self._print("No TESS object found for given catalog row.", verbose)
            return pd.DataFrame()
        
        # TODO: check for better way of selecting candidate
        # catalog_data is automatically sorted by dstArcSec, index 0 is closest
        # also, is GAIAmag guaranteed to exist? -> yes, might be NaN though
        # what if we're not using a GAIA cat_row? -> doesn't matter

        cat_data_mags = np.array(catalog_data['GAIAmag'])
        close_mags = np.logical_and(
            ~np.isnan(cat_data_mags),
            abs(cat_data_mags - mag) < self.pars.magdiff_thresh
        )
        if not any(close_mags):
            self._print("No objects found within mag difference "
                        "threshold for TIC query.", verbose)
            return pd.DataFrame()

        cat_data_dists = np.array(catalog_data['dstArcSec'])
        close_dsts = np.logical_and(
            ~np.isnan(cat_data_dists),
            cat_data_dists < self.pars.distance_thresh
        )
        if not any(close_dsts):
            self._print("No objects found within distance "
                        "threshold for TIC query.", verbose)
            return pd.DataFrame()

        close_mags_and_dsts = np.logical_and(close_mags, close_dsts)
        cand_count = 0
        for b in close_mags_and_dsts:
            if b: cand_count += 1
        # if there are multiple likely candidates, throw warning
        if cand_count > 1:
            warnings.warn(f"Multiple ({close_mags}) sources with similar "
                            "magnitudes in cone search. "
                            "Might select incorrect source.")

        candidate_idx = np.nanargmin(cat_data_dists)
        tess_name = 'TIC ' + catalog_data['ID'][candidate_idx]
        obs_params = {
            'objectname': tess_name,
            'radius': self.pars.obs_qry_radius,
            'obs_collection': 'TESS',
        }
        data_query = self._try_query(Observations.query_criteria, obs_params, verbose)
        if len(data_query) == 0:
            self._print(f"No data found for object {tess_name}.", verbose)
            return pd.DataFrame()
        if 'timeseries' not in data_query['dataproduct_type']:
            self._print(
                f"No timeseries data found for object {tess_name}.", verbose)
            return pd.DataFrame()
        
        lc_indices = []
        for i, uri in enumerate(data_query['dataURL']):
            if isinstance(uri, str) and uri[-7:-5] == 'lc':
                lc_indices.append(i)

        if not lc_indices:
            self._print(f"No lightcurve data found for object {tess_name}.", verbose)
            return pd.DataFrame()

        self._print(f"Found {len(lc_indices)} light curve(s) "
                     "for this source.", verbose)

        base_url = "https://mast.stsci.edu/api/v0.1/Download/file?uri="

        header_attributes_to_save = [
            'TICVER', 'RA_OBJ', 'DEC_OBJ',
            'PMRA', 'PMDEC', 'TESSMAG',
            'TEFF', 'LOGG', 'MH',
            'RADIUS', 'CRMITEN',
            'CRSPOC', 'CRBLKSZ',
        ]
        df_list = []
        altdata = {}
        altdata['source mag'] = mag
        altdata['source coordinates'] = coord_str
        sectors = set()
        TICID = set()
        for i in lc_indices:
            uri = data_query['dataURL'][i]
            table = self._try_table_read_fits(base_url + uri)
            hdul0header, hdul2data = self._try_open_fits(base_url + uri)

            TICID.add(hdul0header['TICID'])
            sector = hdul0header['SECTOR']
            sectors.add(sector)
            table["SECTOR"] = sector
            df_list.append(table.to_pandas())

            # TODO: what if below attributes have diff values
            # over all lightcurve files? save all?
            # right now we only save values for first file
            for attribute in header_attributes_to_save:
                if attribute not in altdata:
                    altdata[attribute] = hdul0header[attribute]
            if 'aperture matrix' not in altdata:
                altdata['aperture matrix'] = hdul2data
        
        data = pd.concat(df_list, ignore_index=True)

        altdata['sectors'] = list(sectors)
        if len(TICID) == 1:
            altdata['TICID'] = TICID.pop()
        else:
            altdata['TICID'] = list(TICID)
            # warning or throw error or keep everything?
            warnings.warn("Fetched data is not for a single TIC object "
                            "(multiple TICIDs found in light curve data).")
        
        return data, altdata


    def _try_query(self, query_fn, params, verbose):
        """
        Makes an astroquery request repeatedly, ignoring any timeout errors.
        Returns first successful response, otherwise raises TimeoutError.
        """
        # maybe try using multiprocessing to terminate after 10 secs?
        for tries in range(10):
            try:
                self._print(f"Making query request, "
                            f"attempt {tries + 1}/10 ...", verbose)
                ret = query_fn(**params)
                return ret
            except TimeoutError as e:
                self._print(f"Request timed out.", verbose)

        raise TimeoutError(f"Too many timeouts from query request.")


    def _try_table_read_fits(self, url):
        """
        Tries to read fits file repeatedly, ignoring any timeout errors.
        Returns first successful response, otherwise raises TimeoutError.
        """
        for _ in range(10):
            try:
                return Table.read(url, format="fits")
            except socket.timeout:
                continue

        raise TimeoutError(f"Too many timeouts from trying to read fits.")


    def _try_open_fits(self, url):
        """
        Tries to open fits file repeatedly, ignoring any timeout errors.
        Returns first successful response, otherwise raises TimeoutError.
        """
        for _ in range(10):
            try:
                header0 = None
                data2 = None
                with fits.open(url, cache=False) as hdul:
                    header0 = hdul[0].header
                    data2 = hdul[2].data
                return header0, data2
            except socket.timeout:
                continue

        raise TimeoutError(f"Too many timeouts from trying to open fits.")
                
        
    def _print(self, msg, verbose):
        """
        Verbose print helper.
        """
        if verbose > 0:
            print(msg)


if __name__ == "__main__":
    tess = VirtualTESS(project="testing VirtualTESS")
    white_dwarfs = Catalog(default="wd")
    white_dwarfs.load()

    print("finished loading catalog")

    count = 0
    warning_count = 0
    for i in range(len(white_dwarfs.data)):
        cat_row = white_dwarfs.get_row(i, output="dict")
        if cat_row["mag"] > 16:
            continue

        print(f"\n\nindex {i}, count {count}")
        result = tess.fetch_data_from_observatory(cat_row, verbose=1)
        if type(result) != tuple:
            continue

        lc_data, altdata = result
        count += 1
        if type(altdata['TICID']) == list:
            warning_count += 1
        if count % 100 == 0:
            print(f"\n\nProgress: found {count} lightcurves\n\n")
        
        gaia_source_id = cat_row["name"]
        print("saving to disk...")
        lc_data.to_csv("/Users/felix_3gpdyfd/astro_research/virtualobserver"
            f"/src/tess_data_TEST/tess_lc_{gaia_source_id}.csv", index=False)

    print(f"\n\nCount: {count} \tWarning count: {warning_count}")