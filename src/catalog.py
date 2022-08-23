import os
import requests
import re
import subprocess
import shutil
from datetime import datetime, timezone
import dateutil.parser
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time


import numpy as np
import pandas as pd

from astropy.io import fits

from src.parameters import Parameters

SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_\./\\\\]+|\.\.")


class Catalog:
    def __init__(self, filename=None, **kwargs):

        self.pars = Parameters(
            required_pars=[
                "catalog_name",
                "filename",
                "name_column",
                "ra_column",
                "dec_column",
                "mag_column",
                "mag_filter_name",
            ]
        )
        # set default values
        if filename is not None:
            kwargs["filename"] = filename
        # load parameters from user input:
        if "default" in kwargs:
            self.setup_from_defaults(kwargs["default"])
        from pprint import pprint

        # add any additional updates after that
        self.pars.update(kwargs)
        if not hasattr(self.pars, "catalog_name") or self.pars.catalog_name is None:
            self.pars.catalog_name = self.pars.filename.split(".")[0]

        self.pars.verify()

        self.data = None
        self.name_to_row = None

    def guess_file_type(self):
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

        if hasattr(self.pars, "filetype"):
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
            self.load_from_disk(force_redownload)
        if self.data is not None and len(self.data) > 0:
            self.make_locator()

    def load_from_disk(self, force_redownload=False):
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

        print(f"Loading catalog from {self.get_fullpath()}")

        # do the actual loading:
        type = self.guess_file_type()
        if type == "fits":
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
        if not hasattr(self.pars, "url") and not hasattr(self.pars, "url"):
            raise ValueError("No URL specified for catalog.")

        fullname = self.get_fullpath()
        path = os.path.dirname(os.path.abspath(fullname))

        URL = self.pars.url if hasattr(self.pars, "url") else self.pars.URL
        downloaded_filename = os.path.split(URL)[-1]
        downloaded_filename = os.path.join(path, downloaded_filename)

        # sanitize inputs:
        downloaded_filename = SANITIZE_RE.sub("", downloaded_filename)

        with requests.get(URL, stream=True) as r:
            if not os.path.isdir(path):
                os.makedirs(path)
            with open(downloaded_filename, "wb") as f:
                print(f"Downloading {URL} \n to {downloaded_filename}")
                shutil.copyfileobj(r.raw, f)

        if os.path.splitext(downloaded_filename)[1] in (".gz"):
            print(f"Unzipping {downloaded_filename} \n to {fullname}")
            subprocess.run(["gunzip", downloaded_filename])
            os.rename(os.path.splitext(downloaded_filename)[0], fullname)
            if os.path.isfile(downloaded_filename):
                os.remove(downloaded_filename)

    @staticmethod
    def check_sanitizer(input_str):
        return SANITIZE_RE.sub("", input_str)

    def make_locator(self):
        """
        Generate a dictionary that translates
        from object name to row index.
        """
        if self.data is None:
            raise ValueError("Catalog not loaded.")
        self.name_to_row = {
            name: index
            for name, index in zip(
                self.data[self.pars.name_column], range(len(self.data))
            )
        }

    def setup_from_defaults(self, default):
        """
        Set up the catalog using a default keyword.

        """
        default = default.lower().replace("_", " ").replace("-", " ")

        if default == "test":
            self.pars.catalog_name = "test catalog"
            self.pars.name_column = "object_id"
            self.pars.ra_column = "ra"
            self.pars.dec_column = "dec"
            self.pars.mag_column = "mag"
            self.pars.mag_error_column = "magerr"
            self.pars.mag_filter_name = "R"
            self.pars.filename = "test.csv"
            if not os.path.isfile(self.get_fullpath()):
                self.make_test_catalog(self.pars.filename)
        elif default in ("wd", "wds", "white dwarf", "white_dwarfs"):
            self.pars.catalog_name = "Gaia eDR3 white dwarfs"
            # file to save inside "catalogs" directory
            self.pars.filename = "GaiaEDR3_WD_main.fits"
            # URL to get this file if missing:
            self.pars.url = (
                "https://warwick.ac.uk/fac/sci/physics/"
                "research/astro/research/catalogues/"
                "gaiaedr3_wd_main.fits.gz"
            )
            # paper citation for this catalog:
            self.pars.reference = (
                "https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.3877G/abstract"
            )

            self.pars.name_column = "WDJ_name"
            self.pars.ra_column = "ra"
            self.pars.dec_column = "dec"
            self.pars.mag_column = "phot_g_mean_mag"
            self.pars.mag_err_column = "phot_g_mean_mag_error"
            self.pars.mag_filter_name = "Gaia_G"

    def get_row(self, loc, index_type="number"):

        if self.data is None:
            raise ValueError("Catalog not loaded.")
        if len(self.data) == 0:
            raise ValueError("Catalog is empty.")

        if index_type == "number":
            return self.data[loc]
        elif index_type == "name":
            return self.data[self.name_to_row[loc]]
        else:
            raise ValueError('index_type must be "number" or "name"')

    def extract_from_row(self, row):
        """
        Extract the relevant information from a row of the catalog.
        """
        index = self.name_to_row[row[self.pars.name_column]]
        name = self.to_string(row[self.pars.name_column])
        ra = float(row[self.pars.ra_column])
        dec = float(row[self.pars.dec_column])
        mag = float(row[self.pars.mag_column])
        if hasattr(self.pars, "mag_err_column"):
            mag_err = float(row[self.pars.mag_err_column])
        else:
            mag_err = None
        filter_name = self.pars.mag_filter_name
        if hasattr(self.pars, "alias_column"):
            alias = self.to_string(row[self.pars.alias_column])
        else:
            alias = None

        return index, name, ra, dec, mag, mag_err, filter_name, alias

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
            names.append(f"J{Catalog.ra2sex(ra[i])}{Catalog.dec2sex(dec[i])}")

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
        if not os.path.isdir(path):
            os.makedirs(path)

        if fmt == "csv":
            df.to_csv(filename, index=False, header=True)
        elif fmt == "fits":
            pass  # TODO: need to finish this
        elif fmt == "h5":
            df.to_hdf(filename, key="catalog", mode="w")

    @staticmethod
    def to_string(string):
        """
        Convert a string or a bytes object to a string
        """
        if isinstance(string, bytes):
            # can later parametrize the encoding
            return string.decode("utf-8")
        else:
            return string

    @staticmethod
    def ra2sex(ra):
        """
        Convert an RA in degrees to a string in sexagesimal format.
        """
        if ra < 0 or ra > 360:
            raise ValueError("RA out of range.")
        ra /= 15.0  # convert to hours
        return (
            f"{int(ra):02d}:{int((ra % 1) * 60):02d}:{((ra % 1) * 60) % 1 * 60:05.2f}"
        )

    @staticmethod
    def dec2sex(dec):
        """
        Convert a Dec in degrees to a string in sexagesimal format.
        """
        if dec < -90 or dec > 90:
            raise ValueError("Dec out of range.")
        return f"{int(dec):+03d}:{int((dec % 1) * 60):02d}:{((dec % 1) * 60) % 1 * 60:04.1f}"

    @staticmethod
    def ra2deg(ra):
        """
        Convert the input right ascension into a float of decimal degrees.
        The input can be a string (with hour angle units) or a float (degree units!).

        Parameters
        ----------
        ra: scalar float or str
            Input RA (right ascension).
            Can be given in decimal degrees or in sexagesimal string (in hours!)
            Example 1: 271.3
            Example 2: 18:23:21.1

        Returns
        -------
        ra: scalar float
            The RA as a float, in decimal degrees

        """
        if type(ra) == str:
            c = SkyCoord(ra=ra, dec=0, unit=(u.hourangle, u.degree))
            ra = c.ra.value  # output in degrees
        else:
            ra = float(ra)

        if not 0.0 < ra < 360.0:
            raise ValueError(f"Value of RA ({ra}) is outside range (0 -> 360).")

        return ra

    @staticmethod
    def dec2deg(dec):
        """
        Convert the input right ascension into a float of decimal degrees.
        The input can be a string (with hour angle units) or a float (degree units!).

        Parameters
        ----------
        dec: scalar float or str
            Input declination.
            Can be given in decimal degrees or in sexagesimal string (in degrees as well)
            Example 1: +33.21 (northern hemisphere)
            Example 2: -22.56 (southern hemisphere)
            Example 3: +12.34.56.7

        Returns
        -------
        dec: scalar float
            The declination as a float, in decimal degrees

        """
        if type(dec) == str:
            c = SkyCoord(ra=0, dec=dec, unit=(u.degree, u.degree))
            dec = c.dec.value  # output in degrees
        else:
            dec = float(dec)

        if not -90.0 < dec < 90.0:
            raise ValueError(f"Value of dec ({dec}) is outside range (-90 -> +90).")

        return dec

    @staticmethod
    def date2jd(date):
        """
        Parse a string or datetime object into a Julian Date (JD) float.
        If string, will parse using dateutil.parser.parse.
        If datetime, will convert to UTC or add that timezone if is naive.
        If given as float, will just return it as a float.

        Parameters
        ----------
        date: float or string or datetime
            The input date or datetime object.

        Returns
        -------
        jd: scalar float
            The Julian Date associated with the input date.

        """
        if isinstance(date, datetime):
            t = date
        elif isinstance(date, str):
            t = dateutil.parser.parse(date)
        else:
            return float(date)

        if t.tzinfo is None:  # naive datetime (no timezone)
            # turn a naive datetime into a UTC datetime
            t = t.replace(tzinfo=timezone.utc)
        else:  # non naive (has timezone)
            t = t.astimezone(timezone.utc)

        return Time(t).jd


if __name__ == "__main__":
    c = Catalog(default="wd")
    c.load()
