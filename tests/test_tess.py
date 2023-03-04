import os

import numpy as np
import pandas as pd
from astropy.time import Time

from src.utils import OnClose
from src.database import Session

import src.dataset
from src.dataset import RawPhotometry
from src.tess import VirtualTESS

basepath = os.path.abspath(os.path.dirname(__file__))
src.dataset.DATA_ROOT = basepath


def test_tess_download(tess_project, wd_cat):

    # make sure the project has TESS observatory:
    assert len(tess_project.observatories) == 1
    assert "tess" in tess_project.observatories
    tess = tess_project.observatories["tess"]
    assert isinstance(tess, VirtualTESS)

    # get a small catalog with only 3 bright sources above the equator
    idx = wd_cat.data["phot_g_mean_mag"] < 10
    idx = np.where(idx)[0][:3]
    c = wd_cat.make_smaller_catalog(idx)

    with Session() as session:
        sources = c.get_all_sources(session)
        for s in sources:
            session.delete(s)
        session.commit()

    # download the lightcurve:
    tess_project.catalog = c
    tess.catalog = c
    tess.fetch_all_sources(
        reduce=False
    )  # TODO: when finished adding reducer, remove this

    def cleanup():  # to be called at the end
        with Session() as session:
            for s in tess.sources:
                for p in s.raw_photometry:
                    p.delete_data_from_disk()
                    session.delete(p)
                session.delete(s)

            session.commit()

    _ = OnClose(cleanup)  # called even upon exception

    filenames = []

    assert len(tess.sources) == 3
    num_sources_with_data = 0
    for s in tess.sources:
        assert len(s.raw_photometry) == 1
        p = s.raw_photometry[0]
        assert p.get_fullname() is not None
        assert os.path.exists(p.get_fullname())
        filenames.append(p.get_fullname())

        if len(p.data):
            with pd.HDFStore(p.get_fullname()) as store:
                assert len(store.keys()) == 1
                key = store.keys()[0]
                df = store[key]
                assert len(df) > 0
                assert np.all(df["TIME"] > 0)
                assert np.all(df.loc[~np.isnan(df["SAP_FLUX"]), "SAP_FLUX"] > 0)

                metadata = store.get_storer(key).attrs["altdata"]
                assert isinstance(metadata, dict)
                assert "cat_row" in metadata
                assert metadata["cat_row"] == s.cat_row
                assert "TICID" in metadata
                assert len(metadata["sectors"]) >= 1

                # should be a list of arrays, each one a nested 2D list
                assert isinstance(metadata["aperture_arrays"], list)
                assert isinstance(metadata["aperture_arrays"][0], list)
                assert isinstance(metadata["aperture_arrays"][0][0], list)

                num_sources_with_data += 1

                assert "EXP_TIME" in metadata
                assert metadata["EXP_TIME"] < 20.0

        assert num_sources_with_data > 0


def test_tess_reduction(tess_project, new_source):
    # make sure the project has tess observatory:
    assert len(tess_project.observatories) == 1
    assert "tess" in tess_project.observatories
    tess = tess_project.observatories["tess"]
    assert isinstance(tess, VirtualTESS)

    tess.pars.save_reduced = False

    # load the data into a RawData object
    new_source.project = "test_TESS"
    colmap, time_info = tess.get_colmap_time_info()
    raw_data = RawPhotometry(observatory="tess", colmap=colmap, time_info=time_info)
    raw_data.filename = "TESS_photometry.h5"
    raw_data.folder = "DATA"
    raw_data.load()
    new_source.raw_photometry.append(raw_data)

    new_lcs = tess.reduce(
        source=new_source, data_type="photometry"
    )  # TODO: add more advanced reduction like detrend
    new_lc_epochs = np.sum([lc.number for lc in new_lcs])

    assert raw_data.number == new_lc_epochs

    # check the raw data was split into two lightcurves, one for each sector
    start_mjds = np.array([58350, 59080])
    start_times = Time(start_mjds, format="mjd").datetime
    end_mjds = np.array([58385, 59115])
    end_times = Time(end_mjds, format="mjd").datetime

    for i, lc in enumerate(new_lcs):
        # verify the altdata sector makes sense:

        # make sure each light curve starts and ends at the correct time
        assert any((lc.data.mjd.min() > start_mjds) & (lc.data.mjd.max() < end_mjds))
        assert any((lc.time_start > start_times) & (lc.time_end < end_times))

        # make sure the times are in chronological order
        mjds = lc.data.mjd.values
        assert np.array_equal(mjds, np.sort(mjds))

    # TODO: more tests for this specific observatory reduction?
