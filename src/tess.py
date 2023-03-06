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
from astropy.time import Time
import astropy.io.fits as fits

# from src.source import angle_diff
from src.observatory import VirtualObservatory, ParsObservatory
from src.dataset import RawPhotometry, Lightcurve
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
            "close stars while querying TIC.",
        )

        self.magdiff_thresh = self.add_par(
            "magdiff_thresh",
            0.75,
            float,
            "Magnitude difference threshold for flagging "
            "similar stars while querying TIC.",
        )

        self.query_radius = self.add_par(
            "cat_query_radius",
            360,
            float,
            "Radius in arcsec for cone search for MAST queries:"
            "Catalog query is to find the TESS ID for given source."
            "Observations query is to find data from TIC for given source.",
        )

        self.use_simple_flux = self.add_par(
            "use_simple_flux",
            False,
            bool,
            "Use simple flux (SAP_FLUX) instead of calibrated flux (PDCSAP_FLUX) for TESS data.",
        )

        self.use_psf_positions = self.add_par(
            "use_psf_positions",
            False,
            bool,
            "Use PSF positions instead of centroids for TESS data.",
        )

        self.download_pars_list = ["distance_thresh", "magdiff_thresh", "query_radius"]

        self._enforce_type_checks = True
        self._enforce_no_new_attrs = True

        config = self.load_then_update(kwargs)

        # apply parameters specific to this class
        self._apply_specific_pars(config)


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

        self.pars = self._make_pars_object(kwargs)
        super().__init__(name="tess")

    @staticmethod
    def _make_pars_object(kwargs):
        """
        Make the ParsObsTESS object.
        When writing a subclass of this class
        that has its own subclassed Parameters,
        this function will allow the constructor
        of the superclass to instantiate the correct
        subclass Parameters object.
        """
        return ParsObsTESS(**kwargs)

    def reduce_photometry(
        self,
        dataset,
        source=None,
        init_kwargs={},
        **_,
    ):
        """
        Reduce the raw photometry to usable lightcurves.
        The data is all from a single filter, but it should
        still be split up into sectors.

        Parameters
        ----------
        dataset: a src.dataset.RawPhotometry object
            The raw data to reduce.
        source: src.source.Source object
            The source to which the dataset belongs.
            If None, the reduction will not use any
            data of the source, such as the expected
            magnitude, the position, etc.
        init_kwargs: dict
            A dictionary of keyword arguments to be
            passed to the constructor of the new dataset.
        # TODO: need to add more parameters for advanced detrending

         Returns
        -------
        a list of src.dataset.Lightcurve objects
            The reduced datasets, after minimal processing.
            The reduced datasets will have uniform filter,
            each dataset will be sorted by time,
            and some initial processing will be done,
            using the "reducer" parameter (or function inputs).
        """
        self._check_dataset(
            dataset, DataClass=RawPhotometry, allowed_dataclasses=[pd.DataFrame]
        )

        # mjd_conversion = dataset.time_info["to mjd"]
        # c = dataset.colmap

        # get the altdata from the init_kwargs (if it is there)
        altdata = init_kwargs.pop("altdata", dataset.altdata)

        # split the dataframe into sectors
        dfs = dataset.data.groupby("SECTOR")
        sectors = [df[0] for df in dfs]
        new_datasets = []
        for df_tuple in dfs:
            sector = df_tuple[0]
            df = df_tuple[1]
            new_altdata = altdata.copy()
            new_altdata["sectors"] = sector
            new_altdata["filter"] = "TESS"
            for i in range(len(sectors)):
                if altdata["file_headers"][i]["SECTOR"] == sectors[i]:
                    new_altdata["file_headers"] = [altdata["file_headers"][i]]
                    new_altdata["lightcurve_headers"] = [
                        altdata["lightcurve_headers"][i]
                    ]
                    new_altdata["aperture_arrays"] = [altdata["aperture_arrays"][i]]
                    new_altdata["aperture_headers"] = [altdata["aperture_headers"][i]]
                    break

            if len(df) > 0:
                new_datasets.append(
                    Lightcurve(data=df, altdata=new_altdata, **init_kwargs)
                )

        # TODO: add more processing here (e.g., detrending)

        return new_datasets

    def get_colmap_time_info(self, data=None, altdata=None):
        """
        Update the column map of the dataset.
        This parses the time and flux columns
        correctly, including the specific time offset
        of this mission and the preferred flux type.

        Parameters
        ----------
        data: pandas.DataFrame
            The raw data to be parsed. Sometimes the raw data
            contains information about the columns or the time format.
        altdata: dict
            The altdata dictionary to be updated.
            Sometimes the altdata contains info like the time offset.

        Returns
        -------
        colmap: dict (optional)
            A dictionary mapping the column names in the raw dataset
            to the standardized names in the raw dataset.
        time_info: dict (optional)
            A dictionary with information about the time column in the raw dataset.
        """
        colmap = {}
        time_info = {}

        time_info["offset"] = 2457000.0  # TODO: get this from the altdata
        time_info["format"] = "jd"
        # time_info["to datetime"] = lambda t: Time(
        #     t - offset, format="jd", scale="utc"
        # ).datetime
        # time_info["to mjd"] = lambda t: Time(
        #     t - offset, format="jd", scale="utc"
        # ).mjd
        colmap["time"] = "TIME"

        colmap["flux"] = "PDCSAP_FLUX"
        colmap["fluxerr"] = "PDCSAP_FLUX_ERR"
        if self.pars.use_simple_flux:
            colmap["flux"] = "SAP_FLUX"
            colmap["fluxerr"] = "SAP_FLUX_ERR"

        colmap["time_corr"] = "TIMECORR"
        colmap["bg"] = "SAP_BKG"
        colmap["bg_err"] = "SAP_BKG_ERR"

        colmap["pos1"] = "MOM_CENTR1"
        colmap["pos1_err"] = "MOM_CENTR1_ERR"
        colmap["pos2"] = "MOM_CENTR2"
        colmap["pos2_err"] = "MOM_CENTR2_ERR"

        if self.pars.use_psf_positions:
            colmap["pos1"] = "PSF_CENTR1"
            colmap["pos1_err"] = "PSF_CENTR1_ERR"
            colmap["pos2"] = "PSF_CENTR2"
            colmap["pos2_err"] = "PSF_CENTR2_ERR"

        colmap["pos_corr1"] = "POS_CORR1"
        colmap["pos_corr2"] = "POS_CORR2"

        return colmap, time_info

    def _append_local_name(self, source):
        """
        Append to the local_names of the source.
        In this case the alias is the TIC ID.
        """
        if self.name.upper() not in source.local_names:
            raw_data = None
            for dt in self.pars.data_types:
                for rd in getattr(source, f"raw_{dt}"):
                    if rd.observatory == self.name:
                        raw_data = rd
                        break

            if raw_data is not None and "file_headers" in raw_data.altdata:
                source.local_names[self.name.upper()] = raw_data.altdata[
                    "file_headers"
                ][0]["TICID"]

    @staticmethod
    def _get_exposure_time(altdata):
        """
        Get the exposure time of the observations
        from the altdata.
        ref: page 20 of https://archive.stsci.edu/files/live/sites/mast/files/home/missions-and-data/active-missions/tess/_documents/EXP-TESS-ARC-ICD-TM-0014-Rev-F.pdf
        Returns
        -------
        exp_time: float
            The exposure time of the observations.
        """

        if "lightcurve_headers" in altdata and len(altdata["lightcurve_headers"]) > 0:
            header = altdata["lightcurve_headers"][0]
            frame_time = header["INT_TIME"]
        else:
            frame_time = 0.98 * 2.0

        if "file_headers" in altdata and len(altdata["file_headers"]) > 0:
            header = altdata["file_headers"][0]
            num_frames = header["CRBLKSZ"]  # this many frames per block
            # brightest and dimmest frames are removed to avoid cosmic rays
            if header["CRMITEN"] or header["CRSPOC"]:
                num_frames -= 2

            # native exposure time for each TESS image is 2.0s minus 0.04s dead time
            exp_time = frame_time * num_frames
        else:
            exp_time = None

        altdata["EXP_TIME"] = exp_time

        return exp_time

    def download_from_observatory(
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
        name = cat_row["name"]
        dec = cat_row["dec"]
        ra = cat_row["ra"]
        mag = cat_row["mag"]

        if mag > 16:
            # TESS can't see stars fainter than 16 mag
            self._print(f"Magnitude of {mag} is too faint for TESS.", verbose)
            return pd.DataFrame(), {}

        cat_params = {
            "coordinates": SkyCoord(ra, dec, frame="icrs", unit="deg"),
            "catalog": "TIC",
            "radius": self.pars.query_radius / 3600,
        }
        catalog_data = self._try_query(Catalogs.query_region, cat_params, verbose)
        if len(catalog_data) == 0:
            self._print("No TESS object found for given catalog row.", verbose)
            return pd.DataFrame(), {}

        candidate_idx = None
        for i in range(len(catalog_data)):
            # catalog is sorted by distance
            # -> iterating from least to greatest distance
            m = catalog_data["GAIAmag"][i]
            d = catalog_data["dstArcSec"][i]
            if (
                ~np.isnan(m)
                and abs(m - mag) < self.pars.magdiff_thresh
                and ~np.isnan(d)
                and d < self.pars.distance_thresh
            ):
                candidate_idx = i
                # grab the first candidate within dist and magdiff threshold
                break

        if candidate_idx is None:
            self._print(
                "No objects found within mag difference " "threshold for TIC query.",
                verbose,
            )
            return pd.DataFrame(), {}

        ticid = catalog_data["ID"][candidate_idx]
        tess_name = "TIC " + ticid
        obs_params = {
            "objectname": tess_name,
            "radius": self.pars.query_radius / 3600,
            "obs_collection": "TESS",
            "dataproduct_type": "timeseries",
        }
        data_query = self._try_query(Observations.query_criteria, obs_params, verbose)

        if len(data_query) == 0:
            self._print(f"No data found for object {tess_name}.", verbose)
            return pd.DataFrame(), {}
        if ticid not in data_query["target_name"]:
            self._print(f"No timeseries data found for object {tess_name}.", verbose)
            return pd.DataFrame(), {}

        lc_indices = []
        for i in range(len(data_query)):
            uri = data_query["dataURL"][i]
            id = data_query["target_name"][i]
            if isinstance(uri, str) and uri[-7:-5] == "lc" and id == ticid:
                lc_indices.append(i)

        if not lc_indices:
            self._print(f"No lightcurve data found for object {tess_name}.", verbose)
            return pd.DataFrame(), {}

        self._print(
            f"Found {len(lc_indices)} light curve(s) " "for this source.", verbose
        )

        base_url = "https://mast.stsci.edu/api/v0.1/Download/file?uri="

        df_list = []
        altdata = {}
        altdata["TICID"] = int(ticid)

        sectors = set()
        for i in lc_indices:
            uri = data_query["dataURL"][i]
            (
                file_header,
                lightcurve_data,
                lightcurve_header,
                aperture_array,
                aperture_header,
            ) = self._try_open_fits(base_url + uri)

            sectors.add(file_header["SECTOR"])
            lightcurve_data["SECTOR"] = file_header["SECTOR"]
            # get the exposure time from the header
            lightcurve_data["EXPTIME"] = lightcurve_header["EXPOSURE"]
            df_list.append(lightcurve_data)

            if "file_headers" not in altdata:
                altdata["file_headers"] = []
            altdata["file_headers"].append(file_header)

            if "lightcurve_headers" not in altdata:
                altdata["lightcurve_headers"] = []
            altdata["lightcurve_headers"].append(lightcurve_header)

            if "aperture_arrays" not in altdata:
                altdata["aperture_arrays"] = []
            # convert the aperture matrix into a nested list
            altdata["aperture_arrays"].append(aperture_array.tolist())
            if "aperture_headers" not in altdata:
                altdata["aperture_headers"] = []
            altdata["aperture_headers"].append(aperture_header)

        self._get_exposure_time(altdata)
        altdata["filter"] = "TESS"

        data = pd.concat(df_list, ignore_index=True)
        altdata["sectors"] = list(sectors)

        return data, altdata

    def download_by_tic(self, tic_number, verbose=0):
        pass  # TODO: finish this

    def _try_query(self, query_fn, params, verbose):
        """
        Makes an astroquery request repeatedly, ignoring any timeout errors.
        Returns first successful response, otherwise raises TimeoutError.
        """
        # maybe try using multiprocessing to terminate after 10 secs?
        for tries in range(10):
            try:
                self._print(
                    f"Making query request, " f"attempt {tries + 1}/10 ...", verbose
                )
                ret = query_fn(**params)
                return ret
            except TimeoutError as e:
                self._print(f"Request timed out.", verbose)

        raise TimeoutError(f"Too many timeouts from query request.")

    def _try_open_fits(self, url):
        """
        Tries to open fits file repeatedly, ignoring any timeout errors.
        Returns first successful response, otherwise raises TimeoutError.

        Returns
        -------
        file_header: dict
            Header of the FITS file.
        lightcurve_data: np.ndarray
            Data from the first extension,
            including the lightcurve for this file.
        lightcurve_header: dict
            Header of the first extension,
            including metadata for the lightcurve.
        aperture_array: nested 2D list
            Data from the second extension,
            including an aperture mask for this source.
        aperture_header: dict
            Header of the second extension,
            with some metadata about the aperture.
        """
        for _ in range(10):
            try:
                # TODO: can we store some of the extra info from FITS
                #  e.g., the units on the data columns?
                with fits.open(url, cache=False) as hdul:
                    file_header = dict(hdul[0].header)
                    lightcurve_data = pd.DataFrame(hdul[1].data)
                    lightcurve_header = dict(hdul[1].header)
                    aperture_array = hdul[2].data
                    aperture_header = dict(hdul[2].header)

                    # rename the TIME column of the lightcurve
                    # this will help make sure we know the units and offset from JD
                    # lightcurve_data.rename(columns={"TIME": lightcurve_header['TUNIT1']}, inplace=True)
                return (
                    file_header,
                    lightcurve_data,
                    lightcurve_header,
                    aperture_array,
                    aperture_header,
                )
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
    tess = VirtualTESS(project="testing VirtualTESS", verbose=0)
    white_dwarfs = Catalog(default="wd")
    white_dwarfs.load()

    print("finished loading catalog")

    count = 0
    for i in range(len(white_dwarfs.data)):
        if i > 20:
            break

        cat_row = white_dwarfs.get_row(i, output="dict")
        if cat_row["mag"] > 16:
            continue

        print(f"index={i}, cat_row: {cat_row}")
        tess.fetch_source(cat_row, reduce=True, save=1)

        # result = tess.download_from_observatory(cat_row, verbose=1)
        # if not result[1]:  # failed fetch returns empty dict
        #     continue
        #
        #
        #
        # lc_data, altdata = result
        # print(
        #     f"TICID = {altdata['TICID']}, GAIA mag = {cat_row['mag']}, TESS mag = {altdata['TESSMAG']}"
        # )
        count += 1
        #
        # ticid = altdata["TICID"]
        # print("saving to disk...")
        # lc_data.to_hdf(
        #     "/Users/felix_3gpdyfd/astro_research/virtualobserver"
        #     f"/notebook/tess_data_TEST/tess_lc_{ticid}.h5",
        #     key="df",
        # )

    print(f"\nFinal Count: {count}")
    print(tess.latest_source.raw_photometry[0].loaded_status)
