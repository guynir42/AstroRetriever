import os
import time
import pytest

import numpy as np

from src.database import SmartSession
from src.dataset import simplify, get_time_offset
from src.catalog import Catalog


def test_observatory_filename_conventions(test_project):

    obs = test_project.observatories["demo"]

    # load a big catalog with more than a million rows
    cat = Catalog(default="wds")
    cat.load()

    obs.catalog = cat

    num = np.random.randint(0, 1000)
    col = cat.pars.name_column
    name = cat.name_to_string(cat.data[col][num])
    _ = int(name)  # make sure conversion to int works

    # get some info on the source
    cat_row = cat.get_row(num, "number", "dict")

    # test the filename conventions
    source = obs.fetch_source(cat_row, save=False)
    data = source.raw_photometry[0]
    data._invent_filename(ra_deg=cat_row["ra"])

    assert (
        data.filename
        == f'RA{int(cat_row["ra"]):02d}/DEMO_photometry_{cat_row["name"]}.h5'
    )

    # try it again with higher numbers in the catalog
    num = np.random.randint(100000, 101000)
    col = cat.pars.name_column
    name = cat.name_to_string(cat.data[col][num])
    _ = int(name)  # make sure conversion to int works

    # test the filename conventions
    source = obs.fetch_source(cat_row, save=False)
    data = source.raw_photometry[0]
    data._invent_filename(ra_deg=cat_row["ra"])

    assert (
        data.filename
        == f'RA{int(cat_row["ra"]):02d}/DEMO_photometry_{cat_row["name"]}.h5'
    )

    # test the key conventions:
    data._invent_filekey(source_name=name)
    assert data.filekey == f"{data.type}_{name}"


@pytest.mark.flaky(max_runs=3)
def test_demo_observatory_download_time(test_project):
    test_project.catalog.make_test_catalog()
    test_project.catalog.load()
    obs = test_project.observatories["demo"]

    t0 = time.time()
    obs.pars.num_threads_download = 0  # no multithreading
    obs.fetch_all_sources(0, 10, save=False, download_args={"wait_time": 1})
    assert len(obs.sources) == 10
    assert len(obs.raw_data) == 10
    single_tread_time = time.time() - t0
    assert abs(single_tread_time - 10) < 2  # should take about 10s

    t0 = time.time()
    obs.sources = []
    obs.raw_data = []
    obs.pars.num_threads_download = 5  # five multithreading cores
    obs.fetch_all_sources(0, 10, save=False, download_args={"wait_time": 5})
    assert len(obs.sources) == 10
    assert len(obs.raw_data) == 10
    multitread_time = time.time() - t0
    assert abs(multitread_time - 10) < 2  # should take about 10s


def test_demo_observatory_save_downloaded(test_project):
    obs = test_project.observatories["demo"]
    try:
        obs.fetch_all_sources(0, 10, save=True, download_args={"wait_time": 0})
        # reloading these sources should be quick (no call to fetch should be sent)
        t0 = time.time()
        obs.fetch_all_sources(0, 10, download_args={"wait_time": 3})
        reload_time = time.time() - t0
        assert reload_time < 1  # should take less than 1s

    finally:
        for d in obs.raw_data:
            d.delete_data_from_disk()

        assert not os.path.isfile(obs.raw_data[0].get_fullname())

        with SmartSession() as session:
            for d in obs.raw_data:
                session.delete(d)
            session.commit()


def test_download_pars(test_project):
    # make random sources unique to this test
    test_project.catalog.make_test_catalog()
    test_project.catalog.load()
    obs = test_project.observatories["demo"]
    try:
        # download the first source only
        obs.fetch_all_sources(0, 1, download_args={"wait_time": 0})

        # reloading this source should be quick (no call to fetch should be sent)
        t0 = time.time()
        obs.fetch_all_sources(0, 1, download_args={"wait_time": 3})
        reload_time = time.time() - t0
        assert reload_time < 1  # should take less than 1s

        # now check that download parameters are inconsistent
        obs.pars.check_download_pars = True

        # reloading
        t0 = time.time()
        obs.fetch_all_sources(0, 1, download_args={"wait_time": 3})
        reload_time = time.time() - t0
        assert reload_time > 1  # should take about 3s to re-download

        # reloading this source should be quick (no call to fetch should be sent)
        t0 = time.time()
        obs.fetch_all_sources(0, 1, download_args={"wait_time": 3})
        reload_time = time.time() - t0
        assert reload_time < 1  # should take less than 1s

    finally:
        for d in obs.raw_data:
            d.delete_data_from_disk()

        if len(obs.raw_data) > 0:
            assert not os.path.isfile(obs.raw_data[0].get_fullname())

        with SmartSession() as session:
            for d in obs.raw_data:
                session.delete(d)
            session.commit()


def test_column_names_simplify_and_offset():
    assert simplify("Times") == "time"
    assert simplify("M-JD") == "mjd"
    assert simplify("BJD - 2457000, days") == "bjd"

    assert get_time_offset("BJD - 2457000, days") == -2457000
