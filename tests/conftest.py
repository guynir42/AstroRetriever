import os
import uuid
import numpy as np
import pandas as pd

import pytest

import sqlalchemy as sa

from src.database import Session
from src.catalog import Catalog
from src.source import Source
from src.project import Project
from src.dataset import RawPhotometry, Lightcurve
from src.finder import Finder
import src.database


@pytest.fixture(scope="session", autouse=True)
def data_dir():
    basepath = os.path.abspath(os.path.dirname(__file__))
    src.database.DATA_ROOT = basepath
    return basepath


@pytest.fixture(scope="session")
def wd_cat():
    c = Catalog(default="wd")
    c.load()
    return c


@pytest.fixture
def new_source():
    source = Source(
        name=str(uuid.uuid4()),
        ra=np.random.uniform(0, 360),
        dec=np.random.uniform(-90, 90),
        test_hash="test",
        mag=np.random.uniform(15, 20),
    )
    yield source
    # with Session() as session:
    #     if source.id:
    #         session.execute(sa.delete(RawPhotometry).where(RawPhotometry.id == source.id))
    #     session.commit()


@pytest.fixture
def new_source2():
    source = Source(
        name=str(uuid.uuid4()),
        ra=np.random.uniform(0, 360),
        dec=np.random.uniform(-90, 90),
        test_hash="test",
    )
    yield source
    # with Session() as session:
    #     if source.id:
    #         session.execute(sa.delete(RawPhotometry).where(RawPhotometry.id == source.id))
    #     session.commit()


@pytest.fixture
def raw_phot():
    df = RawPhotometry.make_random_photometry(number=100)
    data = RawPhotometry(
        data=df,
        folder="data_temp",
        altdata=dict(foo="bar"),
        observatory="demo",
        source_name=str(uuid.uuid4()),
        test_hash="test",
    )

    yield data

    data.delete_data_from_disk()
    with Session() as session:
        if data.id:
            session.execute(sa.delete(RawPhotometry).where(RawPhotometry.id == data.id))
        session.commit()


@pytest.fixture
def raw_phot_no_exptime():
    df = RawPhotometry.make_random_photometry(number=100)
    data = RawPhotometry(
        data=df,
        folder="data_temp",
        altdata=dict(foo="bar"),
        observatory="demo",
        source_name=str(uuid.uuid4()),
        test_hash="test",
    )
    yield data

    data.delete_data_from_disk()
    with Session() as session:
        if data.id:
            session.execute(sa.delete(RawPhotometry).where(RawPhotometry.id == data.id))
        session.commit()


@pytest.fixture
def saved_phot(new_source):
    df = RawPhotometry.make_random_photometry(number=10)
    data = RawPhotometry(
        data=df,
        folder="data_temp",
        altdata=dict(foo="bar"),
        observatory="demo",
        source_name=str(uuid.uuid4()),
        test_hash="test",
    )
    data.source = new_source
    data.save()
    yield data

    data.delete_data_from_disk()
    with Session() as session:
        if data.id:
            session.execute(sa.delete(RawPhotometry).where(RawPhotometry.id == data.id))
        session.commit()


@pytest.fixture
def test_project(data_dir):
    project = Project(
        name="test_project",
        catalog_kwargs={
            "default": "test",
            "filename": os.path.join(data_dir, "test_catalog.csv"),
        },
    )
    yield project

    if os.path.isfile(project.catalog.get_fullpath()):
        os.remove(project.catalog.get_fullpath())


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
def tess_project():
    project = Project(
        name="test_TESS",
        obs_names="TESS",  # a single observatory named TESS
        catalog_kwargs={"default": "test"},
    )
    return project


@pytest.fixture
def simple_finder():

    finder = Finder(project=str(uuid.uuid4()))
    return finder


@pytest.fixture
def analysis():
    from src.analysis import Analysis

    analysis = Analysis(project=str(uuid.uuid4()))
    return analysis


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
        lc = Lightcurve(
            data=df, observatory="demo", project="test_project", test_hash="test"
        )
        return lc

    return __factory
