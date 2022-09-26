import uuid
import numpy as np
import pandas as pd

import pytest

from src.source import Source
from src.project import Project
from src.dataset import RawData, Lightcurve
from src.finder import Finder


@pytest.fixture
def new_source():
    source = Source(
        name=str(uuid.uuid4()),
        ra=np.random.uniform(0, 360),
        dec=np.random.uniform(-90, 90),
    )
    return source


@pytest.fixture
def raw_photometry():
    data = RawData(folder="data_temp", altdata=dict(foo="bar"), observatory="demo")
    data.make_random_photometry(number=30)
    return data


@pytest.fixture
def raw_photometry_no_exptime():
    data = RawData(folder="data_temp", altdata=dict(foo="bar"), observatory="demo")
    data.make_random_photometry(number=30, exptime=None)
    return data


@pytest.fixture
def test_project():
    project = Project(name="test_project", catalog_kwargs={"default": "WD"})
    return project


@pytest.fixture
def wd_project():
    project = Project(name="WD", catalog_kwargs={"default": "WD"})
    return project


@pytest.fixture
def ztf_project():
    project = Project(
        name="test_ZTF",
        obs_names="ZTF",  # a single observatory named ZTF
        catalog_kwargs={"default": "test"},
    )
    return project


@pytest.fixture
def simple_finder():

    finder = Finder()
    return finder


@pytest.fixture
def lightcurve_factory():
    def __factory(
        num_points=100,
        bad_indices=[],
        mjd_range=(57000, 58000),
        shuffle_time=False,
        mag_err_range=(0.09, 0.11),
        mean_mag=18,
        exptime=30,
        filter="R",
    ):
        if shuffle_time:
            mjd = np.random.uniform(mjd_range[0], mjd_range[1], num_points)
        else:
            mjd = np.linspace(mjd_range[0], mjd_range[1], num_points)

        mag_err = np.random.uniform(mag_err_range[0], mag_err_range[1], num_points)
        mag = np.random.normal(mean_mag, mag_err, num_points)
        flag = np.zeros(num_points, dtype=bool)
        flag[bad_indices] = True
        test_data = dict(mjd=mjd, mag=mag, mag_err=mag_err, filter=filter, flag=flag)
        df = pd.DataFrame(test_data)
        df["exptime"] = exptime
        lc = Lightcurve(data=df, observatory="demo")

        return lc

    return __factory
