import os
import glob
import requests
from datetime import datetime
import numpy as np
import pandas as pd
import sqlalchemy as sa

from timeit import default_timer as timer

from ztfquery import lightcurve

from src import utils
from src.observatory import VirtualObservatory


class VirtualZTF(VirtualObservatory):
    def __init__(self, project_name, config=None, keyname=None):
        """
        Generate an instance of a VirtualZTF object.
        This can be used to download ZTF data
        and run analysis on it.

        Parameters
        ----------
        Are the same as the VirtualObservatory class.
        The only difference is that the obs_name is set to "ztf".

        """

        super().__init__("ztf", project_name, config, keyname)
        self.pars.required_pars += ["credentials"]
        # define any default parameters at the
        # source code level here
        # These could be overridden by the config file.
        self.pars.credentials = {}  # load from the default credentials file
        self.pars.data_folder = "ZTF"
        self.pars.data_glob = project_name + "_ZTF_*.h5"

        self.load_parameters()
        # after loading the parameters
        # the user may override them in external code
        # and then call initialize() to verify
        # that all required parameters are set.

    def initialize(self):
        """
        Verify inputs to the observatory.
        """
        super().initialize()
        # verify parameters have the correct type, etc.


def ztf_forced_photometry(ra, dec, start=None, end=None, **kwargs):
    """
    Call the ZTF forced photometery service to produce high-quality lightcurves
    directly from subtraction images, at the coordinates given by ra/dec.

    Parameters
    ----------
    ra: scalar float or string
        The Right Ascension (RA) of the target.
        Can be given in decimal degrees or in sexagesimal string (in hours!)
        Example 1: 271.3
        Example 2: 18:23:21.1

    dec: scalar float or string
        The declination of the target.
        Can be given in decimal degrees or in sexagesimal string (in degrees also)
        Example 1: +33.21 (northern hemisphere)
        Example 2: -22.56 (southern hemisphere)
        Example 3: +12.34.56.7

    start: scalar float or string or datetime
        Start of the range over which to search for images.
        If None (default) will use the beginning of all time (Jan 1st, 2000).
        Can be given as a float (interpreted as Julian Date),
        or as a string (parsed by )
        or as a datetime object


    kwargs:
        ...

    Returns
    -------


    References
    ----------
    More details on the forced photometry can be found here:
    https://web.ipac.caltech.edu/staff/fmasci/ztf/forcedphot.pdf

    Citation:
    Masci, F. J., Laher, R. R., Rusholme, B., et al. 2018,
    The Zwicky Transient Facility: Data Processing, Products, and Archive, PASP, 131, 995.

    Acknowledgment:
    The ZTF forced-photometry service was funded
    under the Heising-Simons Foundation grant #12540303 (PI: Graham).

    """
    if "verbose" in kwargs and kwargs["verbose"]:
        print(f"Calling the ZTF forced photometry service with coordinates: {ra} {dec}")

    ra = utils.ra2deg(ra)
    dec = utils.dec2deg(dec)

    if start is None:
        start = "2000-01-01"
    start_jd = utils.date2jd(start)

    if end is None:
        end = datetime.utcnow()
    end_jd = utils.date2jd(end)

    credentials = utils.get_username_password("ztf")

    url = "https://ztfweb.ipac.caltech.edu/cgi-bin/requestForcedPhotometry.cgi"
    auth = ("ztffps", "dontgocrazy!")
    params = {
        "ra": ra,
        "dec": dec,
        "jdstart": start_jd,
        "jdend": end_jd,
        "email": credentials[0],
        "userpass": credentials[1],
    }

    res = requests.get(url, params=params, auth=auth)

    return res


if __name__ == "__main__":
    pass
    # res = ztf_forced_photometry(280.0, -45.2, 2458231.891227, 2458345.025359)
    # print(res.content)
