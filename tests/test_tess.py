import os

import numpy as np
import pandas as pd
from astropy.time import Time

import sqlalchemy as sa

from src.utils import OnClose
from src.database import SmartSession

import src.dataset
from src.catalog import Catalog
from src.dataset import RawPhotometry
from src.source import Source
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

    with SmartSession() as session:
        sources = c.get_all_sources(session)
        for s in sources:
            session.delete(s)
        session.commit()

    # download the lightcurve:
    tess_project.catalog = c
    tess.catalog = c
    tess.fetch_all_sources()

    def cleanup():  # to be called at the end
        with SmartSession() as session:
            for s in tess.sources:
                for p in s.raw_photometry:
                    p.delete_data_from_disk()
                    session.delete(p)
                for lc in s.reduced_lightcurves:
                    lc.delete_data_from_disk()
                    session.delete(lc)
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


def test_tess_reduction(tess_project, new_source, test_hash):
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
    raw_data.test_hash = test_hash
    raw_data.filename = "TESS_photometry.h5"
    raw_data.folder = "DATA"
    raw_data.load()
    new_source.raw_photometry.append(raw_data)

    # TODO: add more advanced reduction like detrend
    new_lcs = tess.reduce(source=new_source, data_type="photometry")
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


def test_tess_analysis(tess_project, new_source):
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

    reduced_lcs = tess.reduce(source=new_source, data_type="photometry")

    tess_project.analysis.pars.save_anything = False
    tess_project.analysis.analyze_sources(new_source)
    proc_lcs = new_source.processed_photometry

    assert len(reduced_lcs) == len(proc_lcs)

    # check the number of qflat frames in the first light curve
    num_qflags = np.sum(proc_lcs[0].data["qflag"] > 0)

    # add a fake position offset to the first reduced light curve:
    first_good_bin = np.where(proc_lcs[0].data["flag"] == 0)[0][0]
    mean_pos1 = np.nanmedian(proc_lcs[0].data["pos1"])
    reduced_lcs[0].data.loc[first_good_bin, "pos1"] = mean_pos1 + 3
    new_source.reset_analysis()
    tess_project.analysis.analyze_sources(new_source)
    proc_lcs = new_source.processed_photometry

    new_num_qflags = np.sum(proc_lcs[0].data["qflag"] > 0)
    assert num_qflags + 1 == new_num_qflags

    # TODO: test for other things, like adding a flare


def test_tess_download_by_ticid(tess_project):
    tess = tess_project.tess
    assert isinstance(tess, VirtualTESS)

    cat = Catalog(default="wd")
    cat.load()
    tess.catalog = cat

    # identify a white dwarf in TESS
    cat_idx = np.where(cat.data["phot_g_mean_mag"] < 12)[0][0]
    cat_row = cat.get_row(cat_idx, index_type="number", output="dict")

    # first download this source from the catalog:
    try:
        source = tess.fetch_source(cat_row, reduce=False, save=True)
        source_name = source.name
        assert source.loaded_status == "new"
        assert source.raw_photometry[0].observatory == "tess"
        assert source.raw_photometry[0].loaded_status == "new"

        ticid = str(source.local_names["TESS"])

        # fetch by TICID should get the same object
        source2 = tess.fetch_by_ticid(ticid, download=True, use_catalog=True)
        assert source2.loaded_status == "database"
        assert source2.name == source_name
        assert source2.raw_photometry[0].observatory == "tess"
        assert source2.raw_photometry[0].loaded_status == "database"

        # delete the source and the re-fetch it using the TICID
        with SmartSession() as session:
            session.delete(source)
            session.commit()

        # should have the same source name from Gaia
        source3 = tess.fetch_by_ticid(ticid, download=True, use_catalog=True)
        assert source3.loaded_status == "new"
        assert source3.name == source_name
        assert source3.raw_photometry[0].observatory == "tess"
        assert source3.raw_photometry[0].loaded_status == "database"

        # re-fetch using the TICID, without a catalog (name should be TICID)
        source4 = tess.fetch_by_ticid(ticid, download=True, use_catalog=False)
        assert source4.loaded_status == "new"
        assert source4.name != source_name
        assert source4.name == ticid
        assert source4.raw_photometry[0].observatory == "tess"
        assert source4.raw_photometry[0].loaded_status == "database"

    finally:
        with SmartSession() as session:
            source = session.scalars(
                sa.select(Source).where(
                    Source.name == source_name, Source.project == tess_project.name
                )
            ).first()

            if source is not None:
                session.delete(source)

            raw_phot = session.scalars(
                sa.select(RawPhotometry).where(RawPhotometry.source_name == source_name)
            ).all()
            for rp in raw_phot:
                session.delete(rp)
            session.commit()

    with SmartSession() as session:
        source = session.scalars(
            sa.select(Source).where(
                Source.name == source_name, Source.project == tess_project.name
            )
        ).first()
        assert source is None
        raw_phot = session.scalars(
            sa.select(RawPhotometry).where(RawPhotometry.source_name == source_name)
        ).first()
        assert raw_phot is None


def test_tess_to_skyportal_conversion(tess_project, new_source):
    assert isinstance(tess_project.tess, VirtualTESS)
    colmap, time_info = tess_project.tess.get_colmap_time_info()

    raw_data = RawPhotometry(observatory="tess", colmap=colmap, time_info=time_info)
    raw_data.filename = "TESS_photometry.h5"
    raw_data.folder = "DATA"
    raw_data.load()
    new_source.raw_photometry.append(raw_data)

    lightcurves = tess_project.tess.reduce(source=new_source, data_type="photometry")

    lc = lightcurves[0]

    filename = "test_tess_skyportal_photometry.h5"
    try:  # make sure to remove file at the end
        lc.export_to_skyportal(filename)

        with pd.HDFStore(filename) as store:
            keys = store.keys()
            assert len(keys) == 1
            key = keys[0]
            df = store[key]
            for name in ["mjd", "flux", "fluxerr"]:
                assert name in df.columns

            metadata = store.get_storer(key).attrs["metadata"]

            for name in [
                "series_name",
                "series_obj_id",
                "exp_time",
                "ra",
                "dec",
                "filter",
                "time_stamp_alignment",
            ]:
                assert name in metadata

    finally:
        if os.path.isfile(filename):
            os.remove(filename)
