import pytest

import numpy as np
import pandas as pd
from src.histogram import Histogram


@pytest.mark.flaky(max_runs=3)
def test_histogram():

    h = Histogram()
    # make sure the test does not
    # change if we modify the defaults
    h.pars.dtype = "uint32"
    h.pars.score_coords = {
        "snr": (-10, 10, 0.1),
        "dmag": (-3, 3, 0.1),
    }
    h.pars.source_coords = {
        "mag": (15, 21, 0.5),
    }
    h.pars.obs_coords = {
        "exptime": (30.0, 0.3),
        "filt": (),
    }
    h.initialize()

    num_snr = len(np.arange(-10, 10 + 0.1, 0.1))
    num_dmag = len(np.arange(-3, 3 + 0.1, 0.1))
    num_mag = len(np.arange(15, 21 + 0.5, 0.5))
    num_dynamic = 3  # guess the number of values for dynamic axes
    num_bytes = 4  # uint32

    assert h.get_size() == 0
    estimate_bytes = (num_snr + num_dmag) * num_mag * num_dynamic**2 * num_bytes
    assert h.get_size_estimate("bytes") == estimate_bytes

    # add some data with uniform filter
    num_points1 = 10
    df = pd.DataFrame(
        dict(
            mjd=np.linspace(57000, 58000, num_points1),
            snr=np.random.normal(0, 3, num_points1),
            dmag=0,
            exptime=30.0,
            filt="R",
        )
    )

    # make sure data has well defined dmag values
    df.loc[0:4, "dmag"] = 1.3

    # this will fail because the df doesn't have "mag"
    with pytest.raises(ValueError) as err:
        h.add_data(df)

    assert "Could not find data for axis mag" in str(err.value)

    # throwaway class to make a test source
    class FakeSource:
        pass

    source = FakeSource()
    source.id = np.random.randint(0, 1000)
    source.mag = 18
    h.add_data(df, source)
    assert h.data.coords["filt"] == ["R"]
    assert h.data.coords["exptime"] == [30]
    assert h.get_size("bytes") == (num_snr + num_dmag) * num_mag * num_bytes

    assert np.sum(h.data.snr_counts.values) == num_points1
    assert np.sum(h.data.dmag_counts.values) == num_points1

    # check the dmag values we used get summed correctly
    assert (
        h.data.dmag_counts.sel(dmag=0, method="nearest").sum().values == num_points1 - 5
    )
    assert h.data.dmag_counts.sel(dmag=1.3, method="nearest").sum().values == 5

    # add some data with varying filter
    num_points2 = 10
    df = pd.DataFrame(
        dict(
            mjd=np.linspace(57000, 58000, num_points2),
            snr=np.random.normal(0, 3, num_points2),
            dmag=0,
            exptime=30.0,
            filt=np.random.choice(["V", "I"], num_points2),
        )
    )

    h.add_data(df, source)

    assert set(h.data.coords["filt"].values) == {"R", "V", "I"}
    assert h.data.coords["exptime"] == [30]
    assert h.get_size("bytes") == (num_snr + num_dmag) * num_mag * 3 * num_bytes
    assert (
        h.data.dmag_counts.sel(dmag=0, method="nearest").sum().values == num_points2 + 5
    )
    assert (
        h.data.dmag_counts.sel(dmag=0, method="nearest").sum().values == num_points2 + 5
    )

    # check the filters have the correct counts
    assert h.data.dmag_counts.sel(filt="R").sum().values == num_points1
    assert (
        h.data.dmag_counts.sel(filt="V").sum().values == df[df["filt"] == "V"].shape[0]
    )
    assert (
        h.data.dmag_counts.sel(filt="I").sum().values == df[df["filt"] == "I"].shape[0]
    )
    assert h.data.snr_counts.sel(filt="R").sum().values == num_points1
    assert (
        h.data.snr_counts.sel(filt="V").sum().values == df[df["filt"] == "V"].shape[0]
    )
    assert (
        h.data.snr_counts.sel(filt="I").sum().values == df[df["filt"] == "I"].shape[0]
    )

    num_points3 = 100
    df = pd.DataFrame(
        dict(
            mjd=np.linspace(57000, 58000, num_points3),
            snr=np.random.normal(0, 3, num_points3),
            dmag=0,
            exptime=30.0,
            filt=np.random.choice(["R", "V", "I"], num_points3),
        )
    )
    df.loc[0:4, "exptime"] = 20.3
    df.loc[5:, "exptime"] = 39.8

    h.add_data(df, source)
    assert len(h.data.coords["exptime"]) == len(np.arange(20, 39.8, 0.3))
    assert h.data.sel(exptime=20.3, method="nearest").dmag_counts.sum().values == 5
    assert (
        h.data.sel(exptime=39.8, method="nearest").dmag_counts.sum().values
        == num_points3 - 5
    )
    assert (
        h.data.sel(exptime=30.0, method="nearest").dmag_counts.sum().values
        == num_points1 + num_points2
    )

    # check most of the S/N values are in the middle of the distribution
    snr = h.data.snr
    high_snr = h.data.snr_counts.sel(snr=snr[snr > 5]).sum().values
    low_snr = h.data.snr_counts.sel(snr=snr[snr < -5]).sum().values
    mid_snr = h.data.snr_counts.sel(snr=snr[(-5 <= snr) & (snr <= 5)]).sum().values
    assert mid_snr > low_snr * 5
    assert mid_snr > high_snr * 5

    # add some very high and very low values:
    df.loc[3, "snr"] = 100
    df.loc[4, "snr"] = -100
    df.loc[10:19, "dmag"] = 100

    h.add_data(df, source)
    assert h.data.snr.attrs["overflow"] == 1
    assert h.data.snr.attrs["underflow"] == 1
    assert h.data.dmag.attrs["overflow"] == 10
    assert h.data.dmag.attrs["underflow"] == 0

    # a new source with magnitude above range
    source.mag = 25.3
    h.add_data(df, source)
    assert h.data.mag.attrs["overflow"] == num_points3
