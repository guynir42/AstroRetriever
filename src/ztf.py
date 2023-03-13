import os
import yaml
import requests
from datetime import datetime
import numpy as np
import pandas as pd

from astropy.time import Time

from ztfquery import lightcurve

from src.source import angle_diff
from src.observatory import VirtualObservatory, ParsObservatory
from src.dataset import RawPhotometry, Lightcurve
from src.utils import help_with_class, help_with_object
from src.utils import ra2deg, dec2deg, date2jd


class ParsObsZTF(ParsObservatory):

    # must register this observatory in list of allowed names
    ParsObservatory.add_to_obs_names("ZTF")

    def __init__(self, **kwargs):

        super().__init__("ztf")

        self.minimal_declination = self.add_par(
            "minimal_declination",
            -30.0,
            float,
            "Minimal declination for downloading ZTF observations",
        )

        self.cone_search_radius = self.add_par(
            "cone_search_radius",
            2.0,
            float,
            "Radius of cone search for ZTF observations (arcsec)",
        )

        self.limiting_magnitude = self.add_par(
            "limiting_magnitude",
            20.5,
            float,
            "Limiting magnitude for downloading ZTF observations",
        )

        self.faint_magnitude_difference = self.add_par(
            "faint_magnitude_difference",
            1.0,
            (None, float),
            "Maximum magnitude difference between catalog magnitude "
            "and median measured magnitude, above which that oid is not saved",
        )

        self.bright_magnitude_difference = self.add_par(
            "bright_magnitude_difference",
            1.0,
            (None, float),
            "Maximum magnitude difference between catalog magnitude "
            "and median measured magnitude, below which that oid is not saved",
        )

        self.download_pars_list = [
            "minimal_declination",
            "cone_search_radius",
            "limiting_magnitude",
            "faint_magnitude_difference",
            "bright_magnitude_difference",
        ]

        self._enforce_no_new_attrs = True

        config = self.load_then_update(kwargs)

        # apply parameters specific to this class
        self._apply_specific_pars(config)


class VirtualZTF(VirtualObservatory):
    """
    A virtual observatory sub class for getting ZTF data.
    """

    def __init__(self, **kwargs):
        """
        Generate an instance of a VirtualZTF object.
        This can be used to download ZTF data
        and reduce the raw data.

        Parameters
        ----------
        Are the same as the VirtualObservatory class.
        The only difference is that the obs_name is set to "ztf".

        """

        self.pars = self._make_pars_object(kwargs)
        super().__init__(name="ztf")

    @staticmethod
    def _make_pars_object(kwargs):
        """
        Make the ParsObsZTF object.
        When writing a subclass of this class
        that has its own subclassed Parameters,
        this function will allow the constructor
        of the superclass to instantiate the correct
        subclass Parameters object.
        """
        return ParsObsZTF(**kwargs)

    def download_from_observatory(self, cat_row, verbose=False):
        """
        Fetch data from the ZTF archive for a given source.

        Parameters
        ----------
        cat_row: dict like
            A row in the catalog for a specific source.
            In general, this row should contain the following keys:
            name, ra, dec, mag, filter_name (saying which band the mag is in).
        verbose: bool, optional
            If True, will print out some information about the
            data that is being fetched or simulated.

        Returns
        -------
        data : pandas.DataFrame or other data structure
            Raw data from the observatory, to be put into a RawPhotometry object.
        altdata: dict
            Additional data to be stored in the RawPhotometry object.

        """

        self.pars.vprint(
            f'Fetching data from ZTF observatory for source {cat_row["cat_index"]}'
        )

        if (
            cat_row["dec"] < self.pars.minimal_declination
            or cat_row["mag"] > self.pars.limiting_magnitude
        ):
            data = pd.DataFrame([])
            altdata = {}
        else:
            # get a big enough radius to fit high proper motion stars
            # proper motion is given as milli-arcsec per year
            pmra = cat_row.get("pmra", 0)
            pmdec = cat_row.get("pmdec", 0)
            pm = np.sqrt(pmra**2 + pmdec**2)

            pm_radius_bump = 0.003 * pm  # convert to arcsec, allow 3 years of motion
            radius = self.pars.cone_search_radius + pm_radius_bump  # arcsec
            new_query = lightcurve.LCQuery.from_position(
                cat_row["ra"], cat_row["dec"], radius, auth=self.get_credentials()
            )
            data = new_query.data

            if (
                self.pars.faint_magnitude_difference is not None
                and self.pars.bright_magnitude_difference is not None
            ):
                # filter out objects that are too faint or too bright
                # compared to the catalog magnitude
                short_table = data.groupby("oid").mean("mag")["mag"]

            altdata = {}  # TODO: is there anything we want to put here?

        return data, altdata

    # TODO: these specific parameters should live in the Parameters object
    #  we should do that when splitting it into a separate Reducer class
    def reduce_photometry(
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
        Reduce the raw dataset to lightcurves.
        Splits up the raw data that corresponds
        to several object IDs (oid) into separate
        lightcurves. Will also split observing seasons
        into separate lightcurves, if there's a gap
        of more than X days (given by the "gap" parameter).

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
        mag_range: float or None
            If not None, and if the source is also given,
            this value will be used to remove oid's
            where the median magnitude is outside this range,
            relative to the source's magnitude.
        radius: float
            The maximum distance (in arc-seconds) from the source
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
        self._check_dataset(
            dataset, DataClass=RawPhotometry, allowed_dataclasses=[pd.DataFrame]
        )

        data = dataset.data
        altdata_base = init_kwargs.pop("altdata", dataset.altdata)

        time_col = dataset.colmap["time"]

        def mjd_conversion(t):
            return Time(
                t + dataset.time_info["offset"],
                format=dataset.time_info["format"],
                scale="utc",
            ).mjd

        exp_col = dataset.colmap["exptime"]
        filt_col = dataset.colmap["filter"]
        flag_col = dataset.colmap["flag"] if "flag" in dataset.colmap else None
        mag_col = dataset.colmap["mag"]
        magerr_col = dataset.colmap["magerr"]
        ra_col = dataset.colmap["ra"]
        dec_col = dataset.colmap["dec"]

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
            # df = df.rename(columns={v: k for k, v in dataset.colmap.items()})
            time = Time(df.mjd[0], format="mjd", scale="utc").datetime
            filter = df.loc[0, "filtercode"]
            additional_altdata = dict(
                ra=float(df[ra_col].median()),
                dec=float(df[dec_col].median()),
                series_name=f"{time.year}-{time.month}-{filter}",
                object_id=str(df.loc[0, "oid"]),
                time_stamp_alignment="middle",  # TODO: check this
                exp_time=float(df[exp_col].median()),
            )
            altdata = altdata_base.copy()
            altdata.update(additional_altdata)

            df = df[keep_columns]
            if len(df) > 0:
                new_datasets.append(Lightcurve(data=df, altdata=altdata, **init_kwargs))

        return new_datasets

    @staticmethod
    def get_credentials():
        """Get the username/password for ZTF downloading"""

        username = os.getenv("ZTF_USERNAME")
        password = os.getenv("ZTF_PASSWORD")

        if username is None or password is None:
            basepath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            try:
                with open(os.path.join(basepath, "credentials.yaml")) as f:
                    creds = yaml.safe_load(f)
                    username = creds["ztf"]["username"]
                    password = creds["ztf"]["password"]
            except Exception:
                pass

        if username is None or password is None:
            raise ValueError(
                "ZTF_USERNAME and ZTF_PASSWORD environment variables "
                "must be set to download ZTF data."
            )

        return username, password

    def help(self=None, owner_pars=None):
        """
        Print the help for this object and objects contained in it.
        """
        if isinstance(self, VirtualZTF):
            help_with_object(self, owner_pars)
        elif self is None or self == VirtualZTF:
            help_with_class(VirtualZTF, ParsObsZTF)


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

    ra = ra2deg(ra)
    dec = dec2deg(dec)

    if start is None:
        start = "2000-01-01"
    start_jd = date2jd(start)

    if end is None:
        end = datetime.utcnow()
    end_jd = date2jd(end)

    credentials = utils.get_username_password("ztf")  # TODO: fix this!

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

    from src.catalog import Catalog
    import src.database

    src.database.DATA_ROOT = "/home/guyn/Dropbox/DATA"

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("qt5agg")

    ztf = VirtualZTF(project="test")

    # c0 = Catalog(default='wd')
    # c0.load()
    # idx = (c0.data['phot_g_mean_mag'] < 18) & (c0.data['dec'] > 0)
    # idx = np.where(idx)[0][:5]
    # c = c0.make_smaller_catalog(idx)
    #
    # # download the lightcurve:
    # ztf.catalog = c
    # ztf.download_all_sources()

    cat_row = {
        "cat_index": 6,
        "name": "2873304808599755520",
        "ra": 359.9944299296063,
        "dec": 30.27293792274067,
        "mag": 18.81524658203125,
        "mag_err": None,
        "mag_filter": "Gaia_G",
        "alias": None,
    }

    cat_row = {
        "cat_index": 553834,
        "name": "4068499305485306240",
        "ra": 267.3988352699508,
        "dec": -23.9174116183762,
        "mag": 15.364676475524902,
        "mag_err": None,
        "mag_filter": "Gaia_G",
        "alias": None,
    }

    ztf.pars.cone_search_radius = 10
    data, altdata = ztf.download_from_observatory(cat_row, verbose=True)
    data2 = data[data["filtercode"] == "zg"]
    plt.plot(data2["mjd"], data2["mag"], "o")
