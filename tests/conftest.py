import os
import uuid
import numpy as np
import pandas as pd

import pytest

from src.source import Source
from src.project import Project
from src.dataset import RawData


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
    num_points = 30
    filt = np.random.choice(["r", "g", "i"], num_points)
    mjd = np.random.uniform(57000, 58000, num_points)
    mag = np.random.uniform(15, 20, num_points)
    mag_err = np.random.uniform(0.1, 0.5, num_points)
    oid = np.random.randint(0, 5, num_points)
    test_data = dict(mjd=mjd, mag=mag, mag_err=mag_err, filter=filt, oid=oid)
    df = pd.DataFrame(test_data)
    df["exptime"] = 30
    return RawData(data=df, folder="data_temp", altdata=dict(foo="bar"))


@pytest.fixture
def raw_photometry_no_exptime():
    num_points = 30
    filt = np.random.choice(["r", "g", "i"], num_points)
    mjd = np.random.uniform(57000, 58000, num_points)
    mag = np.random.uniform(15, 20, num_points)
    mag_err = np.random.uniform(0.1, 0.5, num_points)
    oid = np.random.randint(0, 5, num_points)
    test_data = dict(mjd=mjd, mag=mag, mag_err=mag_err, filter=filt, oid=oid)
    df = pd.DataFrame(test_data)
    return RawData(data=df, folder="data_temp", altdata=dict(foo="bar"))


@pytest.fixture
def test_project():
    project = Project(name="test_project", config=False)
    return project


@pytest.fixture
def ztf_project():
    project = Project(
        name="test_ZTF",
        params={
            "observatories": "ZTF",  # a single observatory named ZTF
        },
        config=False,
    )
    return project
