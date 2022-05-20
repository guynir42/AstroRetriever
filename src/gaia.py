import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from astroquery.gaia import GaiaClass
from src import utils


def cone_search(
    ra, dec, radius=2.0, num_matches=1, limmag=20.5, catalog="gaiaedr3.gaia_source"
):
    ra = utils.ra2deg(ra)
    dec = utils.dec2deg(dec)

    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    df = (
        GaiaClass()
        .cone_search(
            target,
            radius * u.arcsec,
            table_name=catalog,
        )
        .get_data()
        .to_pandas()
    )

    # first remove rows that have faint magnitudes
    if limmag:  # do not remove if limmag is None or zero
        df = df[df["phot_g_mean_mag"] < limmag]

    # propagate the stars using Gaia proper motion
    # then choose the closest match(es)
    df["dist_pm"] = 0  # new column
    for index, row in df.iterrows():
        c = SkyCoord(
            ra=row["ra"] * u.deg,
            dec=row["dec"] * u.deg,
            pm_ra_cosdec=row["pmra"] * u.mas / u.yr,
            pm_dec=row["pmdec"] * u.mas / u.yr,
            frame="icrs",
            distance=min(abs(1 / row["parallax"]), 10) * u.kpc,
            obstime=Time(row["ref_epoch"], format="jyear"),
        )
        new_dist = c.separation(target).deg
        df.at[index, "dist_pm"] = new_dist

    df.sort_values(by=["dist_pm"], inplace=True)
    df = df.head(num_matches)

    return df
