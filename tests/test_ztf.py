import os

import numpy as np
from astropy.time import Time

import src.dataset
from src.dataset import RawData
from src.ztf import VirtualZTF

basepath = os.path.abspath(os.path.dirname(__file__))
src.dataset.DATA_ROOT = basepath


def test_ztf_reduction(ztf_project, new_source):
    # make sure the project has ZTF observatory:
    assert len(ztf_project.observatories) == 1
    assert "ztf" in ztf_project.observatories
    ztf = ztf_project.observatories["ztf"]
    assert isinstance(ztf, VirtualZTF)

    # load the data into a RawData object
    new_source.project = "test_ZTF"
    raw_data = RawData()
    raw_data.filename = "ZTF_lightcurve.csv"
    raw_data.load()

    # set the source mag to fit the data:
    new_source.mag = raw_data.data.mag.median()
    new_source.ra = raw_data.data.ra.median()
    new_source.dec = raw_data.data.dec.median()
    new_source.raw_data.append(raw_data)

    new_lcs = ztf.reduce(raw_data, to="lcs", source=new_source, gap=40)
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

    new_lcs = ztf.reduce(raw_data, to="lcs", source=new_source, gap=300)
    new_lc_epochs2 = np.sum([lc.number for lc in new_lcs])

    assert new_lc_epochs == new_lc_epochs2
    assert len(new_lcs) == len(filters)

    # check that flagged points are removed
    flags = (raw_data.data["catflags"] > 0) | np.isnan(raw_data.data.mag)
    new_lcs = ztf.reduce(raw_data, to="lcs", source=new_source, gap=40, drop_bad=True)
    new_lc_epochs3 = np.sum([lc.number for lc in new_lcs])

    num_bad = np.sum(flags)
    assert new_lc_epochs3 == new_lc_epochs - num_bad

    # try to make some oid's the wrong mag
    oid = raw_data.data.oid.values[0]
    raw_data.data.mag[raw_data.data.oid == oid] = 12.0
    num_bad = np.sum(raw_data.data.oid == oid)

    new_lcs = ztf.reduce(raw_data, to="lcs", source=new_source, gap=40)
    new_lc_epochs4 = np.sum([lc.number for lc in new_lcs])

    assert new_lc_epochs4 == new_lc_epochs - num_bad

    # try to make some oid's the wrong coordinates
    oid = raw_data.data.oid.values[-1]
    raw_data.load()  # reload the original data
    raw_data.data.ra[raw_data.data.oid == oid] -= 0.1
    num_bad = np.sum(raw_data.data.oid == oid)

    new_lcs = ztf.reduce(raw_data, to="lcs", source=new_source, gap=40)
    new_lc_epochs5 = np.sum([lc.number for lc in new_lcs])

    assert new_lc_epochs5 == new_lc_epochs - num_bad

    # increasing the radius should bring back those points
    new_lcs = ztf.reduce(raw_data, to="lcs", source=new_source, gap=40, radius=500)
    new_lc_epochs6 = np.sum([lc.number for lc in new_lcs])

    assert new_lc_epochs6 == new_lc_epochs

    # not giving the source should also bring back those points
    new_lcs = ztf.reduce(raw_data, to="lcs", source=None, gap=40)
    new_lc_epochs7 = np.sum([lc.number for lc in new_lcs])

    assert new_lc_epochs7 == new_lc_epochs
