import os
import glob
import requests
from datetime import datetime
import numpy as np
import pandas as pd
import sqlalchemy as sa

from timeit import default_timer as timer

from ztfquery import lightcurve

from src.source import angle_diff
from src.observatory import VirtualObservatory
from src.dataset import DatasetMixin, RawData, Lightcurve


class VirtualZTF(VirtualObservatory):
    def __init__(self, **kwargs):
        """
        Generate an instance of a VirtualZTF object.
        This can be used to download ZTF data
        and run analysis on it.

        Parameters
        ----------
        Are the same as the VirtualObservatory class.
        The only difference is that the obs_name is set to "ztf".

        """

        super().__init__(name="ztf", **kwargs)

        if self.project:
            data_glob = self.project + "_ZTF_*.h5"
        else:
            data_glob = "ZTF_*.h5"

        self.pars.required_pars += ["credentials"]
        self.pars.default_values(
            credentials={},
            data_folder="ZTF",
            data_glob=data_glob,
        )

    def initialize(self):
        """
        Verify inputs to the observatory
        and run any additional setup.
        """
        super().initialize()
        # verify parameters have the correct type, etc.

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

    # TODO: update utils to use Catalog instead
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
