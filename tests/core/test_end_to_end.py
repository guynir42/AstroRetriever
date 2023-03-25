import os
import yaml
import time
import uuid
import pytest
import datetime
from pprint import pprint


from src.project import Project
from src.observatory import VirtualDemoObs
from src.utils import legalize, random_string


def test_project_histograms(test_hash):

    filenames = []

    try:  # cleanup at end
        proj = Project(
            name="default_test",
            obs_names=["demo"],
            analysis_kwargs={"num_injections": 3},
            obs_kwargs={},
            catalog_kwargs={"default": "test"},
            verbose=0,
        )
        proj.test_hash = test_hash
        num_sources = len(proj.catalog.data)

        proj.demo.pars.check_data_exists = True
        proj.run()

        num_epochs = sum(s.raw_photometry[0].number for s in proj.sources)

        assert len(proj.demo.sources) == num_sources
        assert len(proj.sources) == num_sources
        for s in proj.sources:
            assert s.loaded_status == "new"
            assert len(s.raw_photometry) == 1
            assert s.raw_photometry[0].loaded_status == "new"
            assert len(s.reduced_photometry) == 1
            assert s.reduced_photometry[0].loaded_status == "new"
            assert s.raw_photometry[0].get_fullname() is not None
            assert os.path.exists(s.raw_photometry[0].get_fullname())
            filenames.append(s.raw_photometry[0].get_fullname())

        assert len(proj.get_detections()) == 0

        # check the histograms make sense:
        h = proj.analysis.quality_values
        overflow = h.data.offset.attrs["overflow"] + h.data.offset.attrs["underflow"]
        num_points = h.data.offset_counts.sum()

        assert overflow + num_points == num_epochs
        h.get_sum_scores() == num_epochs

        h = proj.analysis.good_scores
        overflow = h.data.snr.attrs["overflow"] + h.data.snr.attrs["underflow"]
        num_points = h.data.snr_counts.sum()

        assert overflow + num_points == num_epochs
        assert h.get_sum_scores() == num_epochs

        hist_sources = proj.analysis.good_scores.source_names
        proj_sources = {s.name for s in proj.sources}

        assert hist_sources == proj_sources

    finally:  # cleanup
        proj.delete_all_sources(
            remove_associated_data=True, remove_raw_data=True, remove_folder=True
        )
        proj.delete_project_files(remove_folder=True)
    for f in filenames:
        assert not os.path.exists(f)
        assert not os.path.exists(os.path.join(proj.output_folder, "config.yaml"))


def test_project_multiple_runs(test_hash):
    num_sources_with_data = 0
    filenames = []

    try:  # cleanup at end
        proj = Project(
            name="default_test",
            obs_names=["demo"],
            analysis_kwargs={"num_injections": 3},
            obs_kwargs={},
            catalog_kwargs={"default": "test"},
            verbose=0,
            max_num_total_exceptions=1,
        )
        proj.test_hash = test_hash
        num_sources = len(proj.catalog.data)

        assert isinstance(proj.demo, VirtualDemoObs)
        proj.demo.pars.check_data_exists = True
        proj.run()

        num_epochs = sum(s.raw_photometry[0].number for s in proj.sources)

        assert len(proj.demo.sources) == num_sources
        assert len(proj.sources) == num_sources
        for s in proj.sources:
            assert s.loaded_status == "new"
            assert len(s.raw_photometry) == 1
            assert s.raw_photometry[0].loaded_status == "new"
            assert len(s.reduced_photometry) == 1
            assert s.reduced_photometry[0].loaded_status == "new"
            assert s.raw_photometry[0].get_fullname() is not None
            assert os.path.exists(s.raw_photometry[0].get_fullname())
            filenames.append(s.raw_photometry[0].get_fullname())

            if len(s.raw_photometry[0].data):
                num_sources_with_data += 1

        assert num_sources_with_data > 0
        assert len(proj.get_detections()) == 0
        assert proj.analysis.quality_values.get_sum_scores() == num_epochs
        assert proj.analysis.good_scores.get_sum_scores() == num_epochs
        assert proj.analysis.all_scores.get_sum_scores() == num_epochs

        # try running again and make sure all sources are loaded from DB
        proj.run()
        assert len(proj.sources) == num_sources
        assert len(proj.get_detections()) == 0

        for s in proj.sources:
            assert s.loaded_status == "database"
            s.get_data(obs="demo", data_type="photometry", level="raw", append=True)
            assert len(s.raw_photometry) == 1
            assert s.raw_photometry[0].loaded_status == "database"
            s.get_data(obs="demo", data_type="photometry", level="reduced", append=True)
            assert len(s.reduced_photometry) == 1
            assert s.reduced_photometry[0].loaded_status == "database"

        assert proj.analysis.good_scores.get_sum_scores() == num_epochs

        # delete the sources, but leave the data
        proj.delete_all_sources(remove_associated_data=False, remove_raw_data=False)

        # try running again and make sure all sources are loaded from DB
        proj.run()
        assert len(proj.sources) == num_sources
        assert len(proj.get_detections()) == 0

        for s in proj.sources:
            assert s.loaded_status == "new"
            assert len(s.raw_photometry) == 1
            assert s.raw_photometry[0].loaded_status == "database"
            assert len(s.reduced_photometry) == 1
            assert s.reduced_photometry[0].loaded_status == "database"

        # delete the sources and the data
        proj.delete_all_sources(remove_associated_data=True, remove_raw_data=True)
        # try running again and make sure all sources and data are loaded from DB
        proj.run()
        assert len(proj.sources) == num_sources
        assert len(proj.get_detections()) == 0

        for s in proj.sources:
            assert s.loaded_status == "new"
            assert len(s.raw_photometry) == 1
            assert s.raw_photometry[0].loaded_status == "new"
            assert len(s.reduced_photometry) == 1
            assert s.reduced_photometry[0].loaded_status == "new"

    finally:  # cleanup
        proj.delete_all_sources(
            remove_associated_data=True, remove_raw_data=True, remove_folder=True
        )
        proj.delete_project_files(remove_folder=True)
    for f in filenames:
        assert not os.path.exists(f)
        assert not os.path.exists(os.path.join(proj.output_folder, "config.yaml"))


def test_project_with_simulated_events(test_hash):
    filenames = []
    original_name = f" {random_string(8)}-{random_string(8)} 42"
    legal_name = legalize(original_name)

    try:  # cleanup at end
        proj = Project(
            name=original_name,
            obs_names=["demo"],
            analysis_kwargs={"num_injections": 3},
            obs_kwargs={},
            catalog_kwargs={"default": "test"},
            verbose=0,
            max_num_total_exceptions=1,
        )
        proj.test_hash = test_hash
        num_sources = len(proj.catalog.data)
        obs = proj.observatories["demo"]
        assert isinstance(obs, VirtualDemoObs)
        obs.pars.check_data_exists = True
        proj.run()

        assert len(obs.sources) == num_sources
        assert len(proj.sources) == num_sources
        for s in proj.sources:
            assert s.loaded_status == "new"
            assert len(s.raw_photometry) == 1
            assert s.raw_photometry[0].loaded_status == "new"
            assert len(s.reduced_photometry) == 1
            assert s.reduced_photometry[0].loaded_status == "new"
            assert s.raw_photometry[0].get_fullname() is not None
            assert os.path.exists(s.raw_photometry[0].get_fullname())
            filenames.append(s.raw_photometry[0].get_fullname())

            # check project name is legal
            assert s.project == legal_name
            assert s.reduced_photometry[0].project == legal_name
            assert s.properties.project == legal_name

    # TODO: add simulator, check detections are found
    #  check the detections are all labelled as simulated
    #  check we do not make additional detections upon re-running
    #  check each detection can find its source and data on all levels
    #  check we can delete the detections

    finally:  # cleanup
        proj.delete_all_sources(
            remove_associated_data=True, remove_raw_data=True, remove_folder=True
        )
        proj.delete_project_files(remove_folder=True)
    for f in filenames:
        assert not os.path.exists(f)
        assert not os.path.exists(os.path.join(proj.output_folder, "config.yaml"))


def test_project_with_no_db_interaction():
    pass  # TODO: finish this!
