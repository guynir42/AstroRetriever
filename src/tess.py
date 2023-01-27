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
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.io.fits as fits

# from src.source import angle_diff
from src.observatory import VirtualObservatory, ParsObservatory
from src.catalog import Catalog
# from src.dataset import DatasetMixin, RawPhotometry, Lightcurve

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

        self.query_radius = self.add_par(
            "cat_query_radius",
            360,
            float,
            "Radius in arcsec for cone search for MAST queries:"
            "Catalog query is to find the TESS ID for given source."
            "Observations query is to find data from TIC for given source."
        )

        self.download_pars_list = [
            "distance_thresh",
            "magdiff_thresh",
            "query_radius"
        ]

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
            # TESS can't see stars fainter than 16 mag
            self._print(f"Magnitude of {mag} is too faint for TESS.", verbose)
            return pd.DataFrame(), {}

        cat_params = {
            'coordinates': SkyCoord(ra, dec, frame="icrs", unit="deg"),
            'catalog': 'TIC',
            'radius': self.pars.query_radius / 3600
        }
        catalog_data = self._try_query(Catalogs.query_region, cat_params, verbose)
        if len(catalog_data) == 0:
            self._print("No TESS object found for given catalog row.", verbose)
            return pd.DataFrame(), {}

        candidate_idx = None
        for i in range(len(catalog_data)):
            # catalog is sorted by distance
            # -> iterating from least to greatest distance
            m = catalog_data['GAIAmag'][i]
            d = catalog_data['dstArcSec'][i]
            if ~np.isnan(m) and abs(m - mag) < self.pars.magdiff_thresh \
                    and ~np.isnan(d) and d < self.pars.distance_thresh:
                candidate_idx = i
                # grab the first candidate within dist and magdiff threshold
                break

        if candidate_idx is None:
            self._print("No objects found within mag difference "
                        "threshold for TIC query.", verbose)
            return pd.DataFrame(), {}

        ticid = catalog_data['ID'][candidate_idx]
        tess_name = 'TIC ' + ticid
        obs_params = {
            'objectname': tess_name,
            'radius': self.pars.query_radius / 3600,
            'obs_collection': 'TESS',
            'dataproduct_type': 'timeseries'
        }
        data_query = self._try_query(Observations.query_criteria, obs_params, verbose)

        if len(data_query) == 0:
            self._print(f"No data found for object {tess_name}.", verbose)
            return pd.DataFrame(), {}
        if ticid not in data_query['target_name']:
            self._print(
                f"No timeseries data found for object {tess_name}.", verbose)
            return pd.DataFrame(), {}
        
        lc_indices = []
        for i in range(len(data_query)):
            uri = data_query['dataURL'][i]
            id = data_query['target_name'][i]
            if isinstance(uri, str) and uri[-7:-5] == 'lc' and id == ticid:
                lc_indices.append(i)

        if not lc_indices:
            self._print(f"No lightcurve data found for object {tess_name}.", verbose)
            return pd.DataFrame(), {}

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
        altdata['TICID'] = ticid
        altdata['source mag'] = mag
        altdata['source coordinates'] = f"{ra} {dec}"
        sectors = set()
        for i in lc_indices:
            uri = data_query['dataURL'][i]
            hdul0header, hdul1data, hdul2data, time_units \
                = self._try_open_fits(base_url + uri)
            table = Table(hdul1data)

            table["TICID"] = hdul0header['TICID']
            sectors.add(hdul0header['SECTOR'])
            table["SECTOR"] = hdul0header['SECTOR']
            table[time_units] = table['TIME']
            table.remove_column("TIME")
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


    def _try_open_fits(self, url):
        """
        Tries to open fits file repeatedly, ignoring any timeout errors.
        Returns first successful response, otherwise raises TimeoutError.
        """
        for _ in range(10):
            try:
                header0 = None
                data1 = None
                data2 = None
                time_units = None
                with fits.open(url, cache=False) as hdul:
                    header0 = hdul[0].header
                    data1 = hdul[1].data
                    data2 = hdul[2].data
                    time_units = hdul[1].header['TUNIT1']
                return header0, data1, data2, time_units
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
    for i in range(len(white_dwarfs.data)):
        cat_row = white_dwarfs.get_row(i, output="dict")
        if cat_row["mag"] > 16:
            continue

        print(f"\n\nindex={i}, count={count}")
        result = tess.fetch_data_from_observatory(cat_row, verbose=1)
        if not result[1]: # failed fetch returns empty dict
            continue

        lc_data, altdata = result
        print(f"TICID = {altdata['TICID']}, GAIA mag = {cat_row['mag']}, TESS mag = {altdata['TESSMAG']}")
        count += 1
        
        ticid = altdata['TICID']
        print("saving to disk...")
        lc_data.to_hdf("/Users/felix_3gpdyfd/astro_research/virtualobserver"
            f"/notebook/tess_data_TEST/tess_lc_{ticid}.h5", key="df")

    print(f"\n\nFinal Count: {count}")