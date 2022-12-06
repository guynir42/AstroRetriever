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
from astropy.table import Table
import astropy.io.fits as fits

from src.source import angle_diff
from src.observatory import VirtualObservatory, ParsObservatory
from src.dataset import DatasetMixin, RawData, Lightcurve

class ParsObsTESS(ParsObservatory):

    # must register this observatory in list of allowed names
    ParsObservatory.add_to_obs_names("TESS")

    def __init__(self, **kwargs):

        super().__init__("tess")

        # TODO: figure out dstArcSec threshold for flagging close stars
        self.dst_threshold = self.add_par(
            "dst_threshold",
            10.0, 
            float, 
            "Distance threshold in arcseconds for flagging "
            "close stars while querying TIC."
        )

        # TODO: figure out mag diff threshold for flagging similar stars
        self.magdiff_threshold = self.add_par(
            "magdiff_threshold",
            0.5, 
            float, 
            "Magnitude difference threshold for flagging "
            "similar stars while querying TIC."
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
        cat_qry_rad=0.01,
        obs_qry_rad=0.1
        # TODO: figure out optimal radii for queries
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
        cat_qry_rad: float
            Radius in degrees for cone search for MAST Catalogs query.
            This query is to find the TESS ID for given source.
        obs_qry_rad: float
            Radius in degrees for cone search for MAST Observations query.
            This query is to find data from TIC corresponding to given source.

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

        coord_str = str(ra) + ' ' + str(dec)
        cat_params = {
            'coordinates': coord_str,
            'catalog': 'TIC',
            'radius': cat_qry_rad
        }
        catalog_data = self._try_query(Catalogs.query_region, cat_params)
        if len(catalog_data) == 0:
            self._print("No TESS object found for given catalog row.", verbose)
            return pd.DataFrame()
        
        # TODO: check for better way of selecting candidate
        # catalog_data is automatically sorted by dstArcSec, index 0 is closest
        # also, is GAIAmag guaranteed to exist? -> yes, might be NaN though
        # what if we're not using a GAIA cat_row? -> doesn't matter

        cat_data_mags = np.array(catalog_data['GAIAmag'])
        close_mags = np.where(
            ~np.isnan(cat_data_mags) and \
            abs(cat_data_mags - mag) < self.pars.magdiff_threshold
        )
        if len(close_mags) == 0:
            self._print("No objects found within mag difference "
                        "threshold for TIC query.", verbose)
            return pd.DataFrame()

        cat_data_dsts = np.array(catalog_data['dstArcSec'])
        close_dsts = np.where(
            ~np.isnan(cat_data_dsts) and \
            cat_data_dsts < self.pars.dst_threshold
        )
        if len(close_dsts) == 0:
            self._print("No objects found within distance "
                        "threshold for TIC query.", verbose)
            return pd.DataFrame()

        # close_mags_and_dsts = 0
        # for i in range(len(catalog_data)):
        #     gm = catalog_data['GAIAmag'][i]
        #     dst = catalog_data['dstArcSec'][i]
        #     if np.isnan(gm) or np.isnan(dst):
        #         continue
        #     if abs(gm - mag) < self.pars.magdiff_threshold and \
        #                  dst < self.pars.dst_threshold:
        #         close_mags_and_dsts += 1

        close_mags_and_dsts = np.intersect1d(close_mags, close_dsts)
        # if there are multiple likely candidates, throw warning
        if len(close_mags_and_dsts) > 1:
            warnings.warn(f"Multiple ({close_mags}) sources with similar "
                            "magnitudes in cone search. "
                            "Might select incorrect source.")

        candidate = np.nanargmin(cat_data_dsts)
        tess_name = 'TIC ' + catalog_data['ID'][candidate]
        obs_params = {
            'objectname': tess_name,
            'radius': obs_qry_rad,
            'obs_collection': 'TESS',
        }
        data_query = self._try_query(Observations.query_criteria, obs_params)
        if len(data_query) == 0:
            self._print(f"No data found for object {tess_name}.", verbose)
            return pd.DataFrame()
        if 'timeseries' not in data_query['dataproduct_type']:
            self._print(
                f"No timeseries data found for object {tess_name}.", verbose)
            return pd.DataFrame()
        
        lc_indices = []
        for i, uri in enumerate(data_query['dataURL']):
            if isinstance(uri, str) and uri[-5:-7] == 'lc':
                lc_indices.append(i)
        self._print(f"Found {len(lc_indices)} light curve(s) "
                     "for this source.", verbose)

        base_url = "https://mast.stsci.edu/api/v0.1/Download/file?uri="

        df_list = []
        altdata = {}
        altdata['source mag'] = mag
        altdata['source coordinates'] = coord_str
        sectors = set()
        TICID = set()
        for i in lc_indices:
            uri = data_query['dataURL'][i]
            table = Table.read(base_url + uri, format='fits')
            sector = np.nan
            with fits.open(base_url + uri, cache=False) as hdul:
                sector = hdul[0].header['SECTOR']
                sectors.add(sector)
                TICID.add(hdul[0].header['TICID'])
                # TODO: what if below attributes have diff values
                # over all lightcurve files? save all?
                # right now we only save values for first file
                if 'TICVER' not in altdata:
                    altdata['TICVER'] = hdul[0].header['TICVER']
                if 'RA_OBJ' not in altdata:
                    altdata['RA_OBJ'] = hdul[0].header['RA_OBJ']
                if 'DEC_OBJ' not in altdata:
                    altdata['DEC_OBJ'] = hdul[0].header['DEC_OBJ']
                if 'PMRA' not in altdata:
                    altdata['PMRA'] = hdul[0].header['PMRA']
                if 'PMDEC' not in altdata:
                    altdata['PMDEC'] = hdul[0].header['PMDEC']
                if 'TESSMAG' not in altdata:
                    altdata['TESSMAG'] = hdul[0].header['TESSMAG']
                if 'TEFF' not in altdata:
                    altdata['TEFF'] = hdul[0].header['TEFF']
                if 'LOGG' not in altdata:
                    altdata['LOGG'] = hdul[0].header['LOGG']
                if 'MH' not in altdata:
                    altdata['MH'] = hdul[0].header['MH']
                if 'RADIUS' not in altdata:
                    altdata['RADIUS'] = hdul[0].header['RADIUS']
                if 'CRMITEN' not in altdata:
                    altdata['CRMITEN'] = hdul[0].header['CRMITEN']
                if 'CRSPOC' not in altdata:
                    altdata['CRSPOC'] = hdul[0].header['CRSPOC']
                if 'CRBLKSZ' not in altdata:
                    altdata['CRBLKSZ'] = hdul[0].header['CRBLKSZ']
                if 'lightcurve columns' not in altdata:
                    altdata['lightcurve columns'] = str(hdul[1].data.columns)
                if 'aperture matrix' not in altdata:
                    altdata['aperture matrix'] = hdul[2].data
            table.add_column(sector, "SECTOR")
            df_list.append(table.to_pandas())
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
                # print(e)
                self._print(f"Request timed out.", verbose)

        raise TimeoutError(f"Too many timeouts from query request.")
        
    def _print(self, msg, verbose):
        if verbose > 0:
            print(msg)