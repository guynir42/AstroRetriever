import os

import numpy as np
import pandas as pd
from astropy.time import Time

from src.utils import OnClose
from src.database import Session

import src.dataset
from src.dataset import RawPhotometry
from src.ztf import VirtualZTF

basepath = os.path.abspath(os.path.dirname(__file__))
src.dataset.DATA_ROOT = basepath


def test_ztf_download(ztf_project, wd_cat):

    # make sure the project has ZTF observatory:
    assert len(ztf_project.observatories) == 1
    assert "ztf" in ztf_project.observatories
    ztf = ztf_project.observatories["ztf"]
    assert isinstance(ztf, VirtualZTF)

    # get a small catalog with only 3 bright sources above the equator
    idx = (wd_cat.data["phot_g_mean_mag"] < 18) & (wd_cat.data["dec"] > 0)
    idx = np.where(idx)[0][:3]
    c = wd_cat.make_smaller_catalog(idx)

    # download the lightcurve:
    ztf_project.catalog = c
    ztf.catalog = c
    ztf.download_all_sources()

    def cleanup():  # to be called at the end
        with Session() as session:
            for s in ztf.sources:
                for p in s.raw_photometry:
                    p.delete_data_from_disk()
                    session.delete(p)
                session.delete(s)

            session.commit()

    _ = OnClose(cleanup)  # called even upon exception

    filenames = []

    assert len(ztf.sources) == 3
    for s in ztf.sources:
        assert len(s.raw_photometry) == 1
        p = s.raw_photometry[0]
        assert p.get_fullname() is not None
        assert os.path.exists(p.get_fullname())
        filenames.append(p.get_fullname())

        with pd.HDFStore(p.get_fullname()) as store:
            assert len(store.keys()) == 1
            key = store.keys()[0]
            df = store[key]
            assert len(df) > 0
            assert np.all(df["mjd"] > 0)
            assert np.all(df["mag"] > 0)
            assert all([x in ["zg", "zr", "zi"] for x in df["filtercode"]])

            altdata = store.get_storer(key).attrs["altdata"]
            assert isinstance(altdata, dict)
            assert altdata["cat_row"] == s.cat_row
            assert altdata["download_pars"] == {
                "minimal_declination": -30.0,
                "cone_search_radius": 2.0,
                "limiting_magnitude": 20.5,
                "faint_magnitude_difference": 1.0,
                "bright_magnitude_difference": 1.0,
            }


def test_ztf_reduction(ztf_project, new_source):
    # make sure the project has ZTF observatory:
    assert len(ztf_project.observatories) == 1
    assert "ztf" in ztf_project.observatories
    ztf = ztf_project.observatories["ztf"]
    assert isinstance(ztf, VirtualZTF)

    # load the data into a RawData object
    new_source.project = "test_ZTF"
    raw_data = RawPhotometry(observatory="ztf")
    raw_data.filename = "ZTF_lightcurve.csv"
    raw_data.folder = "DATA"
    raw_data.load()

    # set the source mag to fit the data:
    new_source.mag = raw_data.data.mag.median()
    new_source.ra = raw_data.data.ra.median()
    new_source.dec = raw_data.data.dec.median()
    new_source.raw_photometry.append(raw_data)
    new_lcs = ztf.reduce(source=new_source, data_type="photometry", gap=40)
    new_lc_epochs = np.sum([lc.number for lc in new_lcs])

    assert raw_data.number == new_lc_epochs

    start_mjds = np.array([58240, 58590, 58960, 59325])
    start_times = Time(start_mjds, format="mjd").datetime
    end_mjds = np.array([58530, 58910, 59260, 59530])
    end_times = Time(end_mjds, format="mjd").datetime

    for i, lc in enumerate(new_lcs):
        filt = np.unique(lc.data[lc.colmap["filter"]].values)
        # verify all points have the same filter
        assert filt == lc.filter

        # make sure each light curve starts and ends at the correct time
        assert any((lc.data.mjd.min() > start_mjds) & (lc.data.mjd.max() < end_mjds))
        assert any((lc.time_start > start_times) & (lc.time_end < end_times))

        # make sure the times are in chronological order
        mjds = lc.data.mjd.values
        assert np.array_equal(mjds, np.sort(mjds))

    # check the number of lightcurves is reduced when gap is increased
    filters = np.unique(raw_data.data.filtercode)

    new_lcs = ztf.reduce(source=new_source, data_type="photometry", gap=300)
    new_lc_epochs2 = np.sum([lc.number for lc in new_lcs])

    assert new_lc_epochs == new_lc_epochs2
    assert len(new_lcs) == len(filters)

    # check that flagged points are removed
    flags = (raw_data.data["catflags"] > 0) | np.isnan(raw_data.data.mag)
    new_lcs = ztf.reduce(
        source=new_source, data_type="photometry", gap=40, drop_bad=True
    )
    new_lc_epochs3 = np.sum([lc.number for lc in new_lcs])

    num_bad = np.sum(flags)
    assert new_lc_epochs3 == new_lc_epochs - num_bad

    # try to make some oid's the wrong mag
    oid = raw_data.data.oid.values[0]
    raw_data.data.loc[raw_data.data.oid == oid, "mag"] = 12.0
    num_bad = np.sum(raw_data.data.oid == oid)

    new_lcs = ztf.reduce(source=new_source, data_type="photometry", gap=40)
    new_lc_epochs4 = np.sum([lc.number for lc in new_lcs])

    assert new_lc_epochs4 == new_lc_epochs - num_bad

    # try to make some oid's the wrong coordinates
    oid = raw_data.data.oid.values[-1]
    raw_data.load()  # reload the original data
    raw_data.data.loc[raw_data.data.oid == oid, "ra"] -= 0.1
    num_bad = np.sum(raw_data.data.oid == oid)

    new_lcs = ztf.reduce(source=new_source, data_type="photometry", gap=40)
    new_lc_epochs5 = np.sum([lc.number for lc in new_lcs])

    assert new_lc_epochs5 == new_lc_epochs - num_bad

    # increasing the radius should bring back those points
    new_lcs = ztf.reduce(source=new_source, data_type="photometry", gap=40, radius=500)
    new_lc_epochs6 = np.sum([lc.number for lc in new_lcs])

    assert new_lc_epochs6 == new_lc_epochs

    # not giving the source should also bring back those points
    new_lcs = ztf.reduce(raw_data, data_type="photometry", gap=40)
    new_lc_epochs7 = np.sum([lc.number for lc in new_lcs])

    assert new_lc_epochs7 == new_lc_epochs
