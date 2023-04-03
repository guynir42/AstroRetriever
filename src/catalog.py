import os
import requests
import re
import subprocess
import shutil
import hashlib
from datetime import datetime, timezone
import dateutil.parser

import sqlalchemy as sa

from astropy.table import Table
from astropy.coordinates import SkyCoord, Distance
import astropy.units as u
from astropy.time import Time

import numpy as np
import pandas as pd

from astropy.io import fits

from src.parameters import Parameters
from src.source import Source
from src.database import SmartSession, safe_mkdir
from src.utils import help_with_class, help_with_object, ra2sex, dec2sex, legalize


SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_\./\\\\]+|\.\.")


class ParsCatalog(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.catalog_name = self.add_par(
            "catalog_name", None, (None, str), "Name of the catalog"
        )
        self.filename = self.add_par(
            "filename", None, (None, str), "Name of the catalog file"
        )
        self.name_column = self.add_par(
            "name_column", "name", str, "Name of the column with source names"
        )
        self.ra_column = self.add_par(
            "ra_column", "ra", str, "Name of the column with source right ascension"
        )
        self.dec_column = self.add_par(
            "dec_column", "dec", str, "Name of the column with source declination"
        )
        self.pm_ra_column = self.add_par(
            "pm_ra_column",
            None,
            [None, str],
            "Name of the column with source proper motion "
            "in right ascension (times cos declination)",
        )
        self.pm_dec_column = self.add_par(
            "pm_dec_column",
            None,
            [None, str],
            "Name of the column with source proper motion in declination",
        )
        self.parallax_column = self.add_par(
            "parallax_column",
            None,
            [None, str],
            "Name of the column with source parallax",
        )
        self.mag_column = self.add_par(
            "mag_column", "mag", str, "Name of the column with source magnitude"
        )
        self.mag_error_column = self.add_par(
            "mag_error_column",
            "magerr",
            str,
            "Name of the column with source magnitude error",
        )
        self.mag_filter_name = self.add_par(
            "mag_filter_name", "R", str, "Name of the filter for the magnitude column"
        )

        self.coord_frame_data = self.add_par(
            "coord_frame_data",
            "icrs",
            (None, str),
            "Coordinate frame of the catalog's raw data",
        )

        self.coord_frame_output = self.add_par(
            "coord_frame_output",
            "icrs",
            (None, str),
            "Coordinate frame for outputting cat_row, etc.",
        )

        self.catalog_observation_year = self.add_par(
            "catalog_observation_year",
            2000,
            (None, str, int, float),
            "Time of observation for the catalog, for proper motion correction",
        )

        self.alias_column = self.add_par(
            "alias_column", None, (None, str), "Name of the column with source aliases"
        )

        # use default configurations to quickly setup pars
        self.default = self.add_par(
            "default", None, (None, str), "Apply a default configuration"
        )

        self.mag_to_column_map = self.add_par(
            "mag_to_column_map",
            None,
            [None, dict],
            "Map (dict) between the canonical name of the filter (using lower-case legalize) "
            "and the name of the column in the catalog file. ",
        )
        self.mag_to_error_column_map = self.add_par(
            "mag_to_error_column_map",
            None,
            [None, dict],
            "Map (dict) between the canonical name of the filter (using lower-case legalize) "
            "and the name of the column for the mag error in the catalog file. ",
        )

        self.url = self.add_par(
            "url", None, (None, str), "URL to download the catalog file"
        )
        self.reference = self.add_par(
            "reference",
            None,
            (None, str),
            "Reference for the catalog for citations, etc",
        )

        # numpy arrays are faster to read from FITS files because they are big-endian.
        # but for uniformity it may sometimes be better to load into pandas dataframes
        # The WD catalog (~1.3 million rows) takes 1s as array and 11s as pandas dataframe
        self.use_only_pandas = self.add_par(
            "use_only_pandas",
            False,
            bool,
            "Convert any file format to pandas dataframe",
        )

        self._enforce_no_new_attrs = True

        self.load_then_update(kwargs)

    @classmethod
    def _get_default_cfg_key(cls):
        """
        Get the default key to use when loading a config file.
        """
        return "catalog"

    def setup_from_defaults(self, default):
        """
        Set up the catalog using a default keyword.

        Currently supported defaults:
        -"test": make a small catalog with random sources
        -"WD": load the WD catalog from Fusillo et al 2021
        """
        if default is None:
            return

        default = default.lower().replace("_", " ").replace("-", " ")

        if default == "test":
            self.catalog_name = "test catalog"
            self.name_column = "object_id"
            self.ra_column = "ra"
            self.dec_column = "dec"
            self.mag_column = "mag"
            self.mag_error_column = "magerr"
            self.mag_filter_name = "R"
            self.pm_ra_column = None
            self.pm_dec_column = None
            self.parallax_column = None
            self.filename = "test.csv"
            self.mag_to_column_map = {"r": "mag"}
            self.mag_to_error_column_map = {"r": "magerr"}

        elif default in ("wd", "wds", "white dwarf", "white_dwarfs"):
            self.catalog_name = "Gaia eDR3 white dwarfs"
            # file to save inside "catalogs" directory
            self.filename = "GaiaEDR3_WD_main.fits"
            # URL to get this file if missing:
            self.url = (
                "https://warwick.ac.uk/fac/sci/physics/"
                "research/astro/research/catalogues/"
                "gaiaedr3_wd_main.fits.gz"
            )
            # paper citation for this catalog:
            self.reference = (
                "https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.3877G/abstract"
            )

            # self.name_column = "WDJ_name"
            self.name_column = "source_id"
            self.ra_column = "ra"
            self.dec_column = "dec"
            self.mag_column = "phot_g_mean_mag"
            self.mag_error_column = "phot_g_mean_mag_error"
            self.mag_filter_name = "Gaia_G"
            self.pm_ra_column = "pmra"
            self.pm_dec_column = "pmdec"
            self.parallax_column = "parallax"
            self.catalog_observation_year = 2016.5
            self.mag_to_column_map = {
                "gaia_g": "phot_g_mean_mag",
                "gaia_bp": "phot_bp_mean_mag",
                "gaia_rp": "phot_rp_mean_mag",
            }

            self.mag_to_error_column_map = {
                "gaia_g": "phot_g_mean_mag_error",
                "gaia_bp": "phot_bp_mean_mag_error",
                "gaia_rp": "phot_rp_mean_mag_error",
            }

    def __setattr__(self, key, value):
        if key in ("name", "cat_name"):
            super().__setattr__("catalog_name", value)
        else:
            super().__setattr__(key, value)


class Catalog:
    """
    Container for a table with sources and their magnitudes and coordinates.

    The catalog object can be used to load or even download info on
    some assortment of astronomical sources. For example, choose
    default="WD" to download the white dwarf catalog from Fusillo et al 2021.

    If not using the default, please make sure to specify the filename
    and/or URL to download the catalog from.
    Other parameters include names of the columns in the raw catalog data,
    and other definitions useful for finding the sources in the sky.

    The loaded catalog will exist, as a pandas dataframe or numpy named-array,
    inside the attribute "data".

    Some methods exist to translate a row from the raw data into usable
    information for e.g., downloading sources.

    A complete list of default options:
    -"test": make a small catalog with random sources
    -"WD": load the WD catalog from Fusillo et al 2021

    """

    def __init__(self, **kwargs):
        self.pars = self._make_pars_object(kwargs)

        # if loaded test default catalog,
        # make sure it exists, and generate
        # a random one if it doesn't
        if self.pars.default == "test":
            if not os.path.isfile(self.get_fullpath()):
                self.make_test_catalog(self.pars.filename)

        if self.pars.catalog_name is None and self.pars.filename:
            self.pars.catalog_name = self.pars.filename.split(".")[0]

        if self.pars.catalog_name is None:
            raise ValueError("Catalog name not set.")

        self.data = None
        self.inverse_name_index = None
        self.names = None
        self.cat_hash = None
        self.cfg_hash = None

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _make_pars_object(kwargs):
        """
        Make the ParsCatalog object.
        When writing a subclass of this class
        that has its own subclassed Parameters,
        this function will allow the constructor
        of the superclass to instantiate the correct
        subclass Parameters object.
        """
        return ParsCatalog(**kwargs)

    def _guess_file_type(self):
        """
        Guess the file type of the catalog.
        If parameters contain a "filetype" key, use that.
        Otherwise, try to guess the file type from the file extension.

        Returns
        -------
        str
            The file type.
            Can be one of: "fits", "csv".
        """

        if "filetype" in self.pars:
            return self.pars.filetype

        ext = os.path.splitext(self.pars.filename)[1]
        if ext.lower() == ".fits":
            return "fits"
        elif ext.lower() == ".csv":
            return "csv"
        elif ext.lower() in (".h5", ".hdf5"):
            return "h5"
        else:
            raise ValueError(f"Unknown file type: {ext}")

    def get_fullpath(self):
        """
        Get the full path of the catalog.
        If pars.filename is an absolute path,
        just return it. If not, return the path
        relative to the "catalogs" directory.

        Returns
        -------
        str
            The full path of the catalog file on disk.
        """
        if os.path.isabs(self.pars.filename):
            return os.path.abspath(self.pars.filename)
        else:
            return os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "../catalogs", self.pars.filename
                )
            )

    def load(self, force_reload=False, force_redownload=False):
        """
        Load the catalog from disk.
        By default, will not load the catalog
        if it is already loaded to memory
        (to the "data" parameter).
        If the file is not found on disk,
        it will be downloaded.
        The filename should be either an absolute path,
        or a relative path. If relative, the path will be
        relative to the "catalogs" directory.

        Parameters
        ----------
        force_reload: bool
            If True, will reload the catalog from disk,
            even if it is already loaded to memory.
        force_redownload: bool
            If True, will re-download the catalog from the web,
            even if it already exists on disk.

        """
        if self.data is None or force_reload or force_redownload:
            self._load_from_disk(force_redownload)
        if self.data is not None and len(self.data) > 0:
            self._make_inverse_index()

    def _load_from_disk(self, force_redownload=False):
        """
        Load the catalog from disk.
        The filename given in the parameters object
        should be either an absolute path,
        or a relative path. If relative, the path will be
        relative to the "catalogs" directory.

        Parameters
        ----------
        force_redownload: bool
            If True, will re-download the catalog from the web,
            even if it already exists on disk.

        """
        if not os.path.isfile(self.get_fullpath()) or force_redownload:
            self.download_catalog()
            if not os.path.isfile(self.get_fullpath()):
                raise FileNotFoundError(f"File {self.get_fullpath()} not found.")

        self.pars.vprint(f"Loading catalog from {self.get_fullpath()}")

        # do the actual loading:
        type = self._guess_file_type()
        if type == "fits":
            if self.pars.use_only_pandas:
                self.data = Table.read(self.get_fullpath(), format="fits").to_pandas()
            else:
                with fits.open(self.get_fullpath()) as hdul:
                    self.data = np.array(hdul[1].data)
        elif type == "csv":
            self.data = pd.read_csv(self.get_fullpath())
        elif type == "h5":
            self.data = pd.read_hdf(self.get_fullpath())
        else:
            raise ValueError(f"Unknown file type: {type}")

    def download_catalog(self):
        """
        Download the catalog from the web.
        To be able to download the catalog
        the parameters must contain a key
        named "url" with the URL of the catalog.
        """
        if "url" not in self.pars and "URL" not in self.pars:
            raise ValueError("No URL specified for catalog.")

        fullname = self.get_fullpath()
        path = os.path.dirname(os.path.abspath(fullname))

        URL = self.pars.url if "url" in self.pars else self.pars.URL
        downloaded_filename = os.path.split(URL)[-1]
        downloaded_filename = os.path.join(path, downloaded_filename)

        # sanitize inputs:
        downloaded_filename = SANITIZE_RE.sub("", downloaded_filename)

        with requests.get(URL, stream=True) as r:
            safe_mkdir(path)
            with open(downloaded_filename, "wb") as f:
                print(f"Downloading {URL} \n to {downloaded_filename}")
                shutil.copyfileobj(r.raw, f)

        if os.path.splitext(downloaded_filename)[1] in (".gz"):
            print(f"Unzipping {downloaded_filename} \n to {fullname}")
            subprocess.run(["gunzip", downloaded_filename])
            os.rename(os.path.splitext(downloaded_filename)[0], fullname)
            if os.path.isfile(downloaded_filename):
                os.remove(downloaded_filename)

    def make_smaller_catalog(self, idx):
        """
        Get a copy of this catalog object,
        but with only the rows specified by idx.
        The name of the catalog will be appended
        with "_small".

        Parameters
        ----------
        idx: array-like
            Any indexing array that can be used to select
            some rows of the underlying "data" member.
            Could be logical or integer indexing.
            For example, it could be all stars with
            magnitude brighter than 16:
            idx = c.data['phot_g_mean_mag'] < 16.0
            c2 = c.make_smaller_catalog(idx)

        Returns
        -------
        Catalog
            A copy of this catalog object,
            but with only the rows specified by idx.
        """
        c = Catalog(name=self.pars.catalog_name + "_small")
        c.pars = self.pars.copy()
        c.data = self.get_data_slice(idx)
        c._make_inverse_index()
        return c

    def get_all_sources(self, session=None, project=None):
        """
        Get all sources in the catalog that have a
        corresponding row in the database.

        """

        if project is None:
            project = self.pars.project
        hash = self.cfg_hash if self.cfg_hash is not None else ""

        with SmartSession(session) as session:

            names = [str(name) for name in self.data[self.pars.name_column]]

            stmt = sa.select(Source).where(
                Source.name.in_(names),
                Source.project == project,
                Source.cfg_hash == hash,
            )

            sources = session.scalars(stmt).all()

        return sources

    def get_nearest_row(self, ra, dec, radius=2.0, output="raw", obstime=None):
        """
        Get the row in the catalog that is closest
        to the given RA and Dec.
        The coordinates are given in degrees.
        The maximum radius to search is given in arcseconds.
        If radius is set to None, no maximum radius is used.

        Parameters
        ----------
        ra: float
            RA in degrees.
        dec: float
            Dec in degrees.
        radius: float (optional)
            Maximum radius to search in arcseconds.
            If None, no maximum radius is used.
            Default is 2.0 arcseconds.
        output: str
            The type of catalog row to output.
            Can be "raw" or "dict".
        obstime: astropy.time.Time
            The time of observation. If given, will apply
            proper motion to the source based on the time
            the catalog was observed relative to the given time.
            Only works when output="dict".

        Returns
        -------
        The catalog row that is closest to the given RA and Dec.
        If output="raw", the row is returned as a numpy array.
        If output="dict", the row is returned as a dictionary.
        If no matches are found within the given radius,
        None is returned.
        """

        # TODO: can we come up with a faster indexing of the raw catalog data?

        cat_ra = self.data[self.pars.ra_column]
        cat_dec = self.data[self.pars.dec_column]

        delta_ra = 180 - abs(abs(cat_ra - ra) - 180)
        delta_ra *= np.cos(dec * np.pi / 180)
        delta_dec = np.abs(cat_dec - dec)

        dist = np.sqrt(delta_ra**2 + delta_dec**2)
        idx = np.argmin(dist)
        if dist[idx] <= radius / 3600.0:
            return self.get_row(
                idx, index_type="number", output=output, obstime=obstime
            )
        else:
            return None

    @staticmethod
    def check_sanitizer(input_str):
        return SANITIZE_RE.sub("", input_str)

    def _make_inverse_index(self):
        """
        Generate a dictionary that translates
        from object name to row index.
        """
        if self.data is None:
            raise ValueError("Catalog not loaded.")
        self.inverse_name_index = {
            self.name_to_string(name): index
            for name, index in zip(
                self.data[self.pars.name_column], range(len(self.data))
            )
        }

        self.names = list(self.inverse_name_index.keys())
        self.cat_hash = hashlib.sha256("".join(self.names).encode("utf-8")).hexdigest()

    def get_index_from_name(self, name):
        """
        Get the row index of an object in the catalog,
        given its name.

        Parameters
        ----------
        name: str
            The name of the object.

        Returns
        -------
        int
            The row index of the object in the catalog.
        """
        if self.inverse_name_index is None:
            self._make_inverse_index()
        return self.inverse_name_index[self.name_to_string(name)]

    def get_name_from_index(self, number):
        """
        Get the source name from its number (index)
        in the catalog.

        Parameters
        ----------
        number: int
            The index of the source in the catalog.

        Returns
        -------
        str
            The name of the source.
        """
        return self.name_to_string(self.data[self.pars.name_column][number])

    @staticmethod
    def name_to_string(name):
        """
        Convert an object name to a string.
        """

        if isinstance(name, str):
            return name
        elif isinstance(name, bytes):
            return name.decode("utf-8")
        else:
            return str(name)

    def get_columns(self):
        """
        Get the columns of the catalog.

        Returns
        -------
        list
            The list of columns in the catalog.
        """

        if isinstance(self.data, pd.DataFrame):
            return list(self.data.columns)
        else:
            return list(self.data.dtype.names)

    def get_data_slice(self, idx):
        """
        Get a slice of the raw data using indexing
        array idx. If the catalog is a pandas DataFrame,
        then the slice will be a DataFrame, using .iloc.
        Otherwise, just index into the data array.

        Parameters
        ----------
        idx: array-like or integer
            The indexing array or integer to use to slice the data.

        Returns
        -------
        pandas.DataFrame or numpy.ndarray
            The slice of the data.
        """

        if isinstance(self.data, pd.DataFrame):
            new_data = self.data.iloc[idx]
        else:
            new_data = self.data[idx]

        return new_data

    def get_row(
        self, loc, index_type="number", output="raw", obstime=None, preferred_mag=None
    ):
        """
        Get a row from the catalog.
        Can get it by index or by name.
        The output parameter can be "raw" or "dict".
        If "raw", will give the row in the catalog data
        (could be a pandas Series or a numpy array).
        If "dict" will copy only a few columns into a dictionary,
        using the dict_from_row() method.

        Parameters
        ----------
        loc: int or str
            The index or name of the object.
        index_type: str
            Either "number" or "name".
            If "number" will index the catalog based
            on the serial number in the data table.
            If "name" will index the catalog based
            on the name of the object (using the
            inverse index).
        output: str
            Either "raw" or "dict".
            If "raw" will return the row in the catalog data.
            If "dict" will return a dictionary with a few columns.
        obstime: astropy.time.Time
            The time of observation. If given, will apply
            proper motion to the source based on the time
            the catalog was observed relative to the given time.
            Only works when output="dict".
        preferred_mag: str
            The preferred magnitude to use for the source.
            This should name the canonical name of the filter,
            e.g., Gaia_G (and not phot_g_mean_mag).
            The input goes through legalize() so case is ignored
            and spaces are replaced with underscores.

        Returns
        -------
        dict or pandas Series or numpy array
            The row in the catalog data.

        """
        if self.data is None:
            raise ValueError("Catalog not loaded.")
        if len(self.data) == 0:
            raise ValueError("Catalog is empty.")

        if index_type == "number":
            idx = int(loc)
        elif index_type == "name":
            idx = int(self.get_index_from_name(loc))
        else:
            raise ValueError('index_type must be "number" or "name"')

        row = self.get_data_slice(idx)

        if output == "raw":
            return row
        elif output == "dict":
            return self.dict_from_row(row, obstime=obstime, preferred_mag=preferred_mag)
        else:
            raise ValueError('Parameter "output" must be "raw" or "dict"')

    def dict_from_row(self, row, obstime=None, preferred_mag=None):
        """
        Extract the relevant information from a row of the catalog as a dictionary.

        Parameters
        ----------
        row: pandas Series or numpy array
            The row of the catalog, without any processing.
        obstime: astropy Time object
            The time of the observation to use for calculating
            the apparent coordinates for the required survey
            using proper motion of the source.
        preferred_mag: str
            The preferred magnitude to use for the source.
            This should name the canonical name of the filter,
            e.g., Gaia_G (and not phot_g_mean_mag).
            The input goes through legalize() so case is ignored
            and spaces are replaced with underscores.
        """
        index = self.get_index_from_name(row[self.pars.name_column])
        name = self.name_to_string(row[self.pars.name_column])

        ra, dec = self.convert_coords(row, obstime)

        mag = float(row[self.pars.mag_column])

        if "mag_err_column" in self.pars:
            mag_err = float(row[self.pars.mag_err_column])
        else:
            mag_err = None
        filter_name = self.pars.mag_filter_name

        if preferred_mag is not None:
            preferred_mag_legal = legalize(preferred_mag, to_lower=True)
            if self.pars.mag_to_column_map is not None:
                if preferred_mag_legal in self.pars.mag_to_column_map:
                    mag = float(row[self.pars.mag_to_column_map[preferred_mag_legal]])
                    filter_name = preferred_mag
                else:
                    raise ValueError(
                        f'Preferred magnitude "{preferred_mag}" not found in mag_to_column_map.'
                    )
            else:
                raise ValueError(
                    "preferred_mag is not None, but mag_to_column_map is None."
                )
            if self.pars.mag_to_error_column_map is not None:
                if preferred_mag_legal in self.pars.mag_to_error_column_map:
                    mag_err = float(
                        row[self.pars.mag_to_error_column_map[preferred_mag_legal]]
                    )
                else:
                    raise ValueError(
                        f'Preferred magnitude "{preferred_mag}" not found in mag_to_error_column_map.'
                    )

        if self.pars.alias_column:
            alias = self.name_to_string(row[self.pars.alias_column])
        else:
            alias = None

        output = dict(
            cat_index=index,
            name=name,
            ra=ra,
            dec=dec,
            mag=mag,
            mag_err=mag_err,
            mag_filter=filter_name,
            alias=alias,
        )

        if self.pars.pm_ra_column is not None:
            output.update(pmra=float(row[self.pars.pm_ra_column]))

        if self.pars.pm_dec_column is not None:
            output.update(pmra=float(row[self.pars.pm_dec_column]))

        return output

    def values_from_row(self, row):
        """
        Extract the relevant information from a row of the catalog.
        """
        if not isinstance(row, dict):
            d = self.dict_from_row(row)
        return (
            d["index"],
            d["name"],
            d["ra"],
            d["dec"],
            d["mag"],
            d["mag_err"],
            d["filter_name"],
            d["alias"],
        )

    def convert_coords(self, row, obstime=None):
        """
        Convert the coordinates from the catalog's epoch
        (default J2000) to the output coordinate epoch.
        TODO: This should also accept date to propagate the
              coordinates using proper motion info.

        Parameters
        ----------
        row: pandas Series or numpy array
            The row in the catalog data.
        obstime: astropy Time (optional)
            The time of observation to propagate the coordinates
            using proper motion.
            If not given, will not apply proper motion.

        Returns
        -------
        float, float
            The RA and Dec of the object in the output epoch.
        """
        ra = row[self.pars.ra_column]
        dec = row[self.pars.dec_column]
        pm_ra = row[self.pars.pm_ra_column] if self.pars.pm_ra_column else 0.0
        pm_dec = row[self.pars.pm_dec_column] if self.pars.pm_dec_column else 0.0
        parallax = row[self.pars.parallax_column] if self.pars.parallax_column else 0.0
        dist = Distance(parallax=parallax * u.mas) if parallax > 0.0 else 0.0 * u.pc

        coords = SkyCoord(
            ra=ra * u.deg,
            dec=dec * u.deg,
            frame=self.pars.coord_frame_data,
            obstime=Time(
                self.pars.catalog_observation_year, format="jyear", scale="tdb"
            ),  # reference epoch for Gaia DR3
            pm_ra_cosdec=pm_ra * u.mas / u.yr,
            pm_dec=pm_dec * u.mas / u.yr,
            distance=dist,
        )
        if obstime is not None:
            if isinstance(obstime, (str, int, float)):
                obstime = Time(obstime, format="jyear", scale="tdb")
            coords = coords.apply_space_motion(new_obstime=obstime)

        if self.pars.coord_frame_data != self.pars.coord_frame_output:
            coords.transform_to(
                self.pars.coord_frame_output
            )  # TODO: what about equinox?

        return coords.ra.value, coords.dec.value

    @staticmethod
    def make_test_catalog(filename=None, number=10, fmt=None):
        """
        Make a test catalog, save it to catalogs/test.csv
        (or other formats).

        Parameters
        ----------
        filename: str
            Filename to save the test catalog to.
            If None, will use the default filename "catalogs/test.csv"
        number: int
            Number of objects to generate.
        fmt: str
            Format to save the catalog in.
            Options are 'csv', 'fits', 'hdf5'.
            Default is 'csv', or based on file extension.

        """

        ra = np.random.uniform(0, 360, number)
        dec = np.random.uniform(-90, 90, number)
        mag = np.random.uniform(15, 20, number)
        mag_err = np.random.uniform(0.1, 0.5, number)
        filters = np.random.choice(["R", "I", "V"], number)
        names = []
        for i in range(len(ra)):
            names.append(f"J{ra2sex(ra[i])}{dec2sex(dec[i])}")

        data = {
            "object_id": names,
            "ra": ra,
            "dec": dec,
            "mag": mag,
            "magerr": mag_err,
            "filter": filters,
        }

        df = pd.DataFrame(data)
        if filename is None:
            filename = "test"
        filename = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../catalogs/", filename)
        )
        filename, ext = os.path.splitext(filename)

        if fmt is None:
            fmt = ext[1:]

        if fmt == "hdf5":
            fmt = ".h5"

        filename += f".{fmt}"

        path = os.path.dirname(filename)
        safe_mkdir(path)

        if fmt == "csv":
            df.to_csv(filename, index=False, header=True)
        elif fmt == "fits":
            pass  # TODO: need to finish this
        elif fmt == "h5":
            df.to_hdf(filename, key="catalog", mode="w")

    def help(self=None, owner_pars=None):
        """
        Print the help for this object and objects contained in it.
        """
        if isinstance(self, Catalog):
            help_with_object(self, owner_pars)
        elif self is None or self == Catalog:
            help_with_class(Catalog, ParsCatalog)


if __name__ == "__main__":
    c = Catalog(default="wd")
    c.load()
