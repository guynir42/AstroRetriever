import os
import requests
import re
import subprocess
import shutil

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

        # add any additional updates after that
        self.pars.update(kwargs)

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
            pass
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

    def check_sanitizer(self, input_str):
        return SANITIZE_RE.sub("", input_str)

    def make_locator(self):
        """
        Generate a dictionary that translates
        from object name to row index.
        """

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
            self.pars.filename = "catalogs/test.csv"
            if not os.path.isfile(self.pars.filename):
                self.make_test_catalog()
        elif default == "wd" or default == "white dwarfs":
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
                "https://ui.adsabs.harvard.edu/abs/" "2021MNRAS.508.3877G/abstract"
            )

            self.pars.name_column = "WDJ_name"
            self.pars.ra_column = "ra"
            self.pars.dec_column = "dec"
            self.pars.mag_column = "phot_g_mean_mag"
            self.pars.mag_err_column = "phot_g_mean_mag_error"
            self.pars.mag_filter_name = "Gaia_G"

    def make_test_catalog(self):
        """
        Make a test catalog, save it to catalogs/test.csv.
        """
        pass
        # TODO: finish this and add tests of this module

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
    def to_string(string):
        """
        Convert a string or a bytes object to a string
        """
        if isinstance(string, bytes):
            # can later parametrize the encoding
            return string.decode("utf-8")
        else:
            return string


if __name__ == "__main__":
    c = Catalog(default="wd")
    c.load()
