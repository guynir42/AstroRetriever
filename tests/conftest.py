import os
import shutil
import uuid
import numpy as np
import pandas as pd

import pytest

import sqlalchemy as sa

from src.database import SmartSession, safe_mkdir, clear_test_objects
from src.catalog import Catalog
from src.source import Source
from src.project import Project
from src.dataset import RawPhotometry, Lightcurve
from src.finder import Finder
from src.utils import random_string
import src.database


@pytest.fixture(scope="session", autouse=True)
def data_dir():

    # make sure the DATA_ROOT points at the DATA_TEMP folder
    old_data_root = src.database.DATA_ROOT
    src.database.DATA_ROOT = src.database.DATA_TEMP

    safe_mkdir(src.database.DATA_TEMP)

    yield src.database.DATA_TEMP

    # make sure to remove this folder and all temporary data at the end
    shutil.rmtree(src.database.DATA_TEMP)

    # reset the data root
    src.database.DATA_ROOT = old_data_root


@pytest.fixture(scope="session", autouse=True)
def test_hash():
    # mark all the test objects with this
    # and then delete them at the end
    value = str(uuid.uuid4())

    yield value

    clear_test_objects(value)


@pytest.fixture(scope="session")
def wd_cat():
    c = Catalog(default="wd")
    c.load()
    return c


@pytest.fixture
def new_source(test_hash):
    source = Source(
        name=str(uuid.uuid4()),
        ra=np.random.uniform(0, 360),
        dec=np.random.uniform(-90, 90),
        test_hash=test_hash,
        mag=np.random.uniform(15, 20),
    )
    yield source
    # with SmartSession() as session:
    #     if source.id:
    #         session.execute(sa.delete(RawPhotometry).where(RawPhotometry.id == source.id))
    #     session.commit()


@pytest.fixture
def new_source2(test_hash):
    source = Source(
        name=str(uuid.uuid4()),
        ra=np.random.uniform(0, 360),
        dec=np.random.uniform(-90, 90),
        test_hash=test_hash,
    )
    yield source
    # with SmartSession() as session:
    #     if source.id:
    #         session.execute(sa.delete(RawPhotometry).where(RawPhotometry.id == source.id))
    #     session.commit()


@pytest.fixture
def raw_phot(test_hash):
    df = RawPhotometry.make_random_photometry(number=100)
    data = RawPhotometry(
        data=df,
        folder="data_temp",
        altdata=dict(foo="bar"),
        observatory="demo",
        source_name=str(uuid.uuid4()),
        test_hash=test_hash,
    )

    yield data

    data.delete_data_from_disk()
    with SmartSession() as session:
        if data.id:
            session.execute(sa.delete(RawPhotometry).where(RawPhotometry.id == data.id))
        session.commit()


@pytest.fixture
def raw_phot_no_exptime(test_hash):
    df = RawPhotometry.make_random_photometry(number=100, exptime=None)
    data = RawPhotometry(
        data=df,
        folder="data_temp",
        altdata=dict(foo="bar"),
        observatory="demo",
        source_name=str(uuid.uuid4()),
        test_hash=test_hash,
    )
    yield data

    data.delete_data_from_disk()
    with SmartSession() as session:
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
        test_hash=new_source.test_hash,
    )
    data.source = new_source
    data.save()
    yield data

    data.delete_data_from_disk()
    with SmartSession() as session:
        if data.id:
            session.execute(sa.delete(RawPhotometry).where(RawPhotometry.id == data.id))
        session.commit()


@pytest.fixture
def test_project(data_dir, test_hash):
    project = Project(
        name="test_project",
        catalog_kwargs={
            "default": "test",
            "filename": os.path.join(data_dir, "test_catalog.csv"),
        },
    )
    project._test_hash = test_hash
    yield project

    if os.path.isfile(project.catalog.get_fullpath()):
        os.remove(project.catalog.get_fullpath())


@pytest.fixture
def wd_project(test_hash):
    project = Project(name="WD", catalog_kwargs={"default": "WD"})
    project._test_hash = test_hash
    return project


@pytest.fixture
def ztf_project(test_hash):
    project = Project(
        name="test_ZTF",
        obs_names="ZTF",  # a single observatory named ZTF
        catalog_kwargs={"default": "test"},
    )
    project._test_hash = test_hash
    return project


@pytest.fixture
def tess_project(test_hash):
    project = Project(
        name="test_TESS",
        obs_names="TESS",  # a single observatory named TESS
        catalog_kwargs={"default": "test"},
    )
    project._test_hash = test_hash
    return project


@pytest.fixture
def simple_finder(test_hash):

    finder = Finder(project=random_string(8))
    finder._test_hash = test_hash
    return finder


@pytest.fixture
def analysis(test_hash):
    from src.analysis import Analysis

    analysis = Analysis(project=random_string(8))
    analysis._test_hash = test_hash
    return analysis


@pytest.fixture
def lightcurve_factory(test_hash):
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
            data=df,
            observatory="demo",
            project="test_project",
            test_hash=test_hash,
        )
        return lc

    return __factory
