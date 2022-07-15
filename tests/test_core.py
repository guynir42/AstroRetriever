import os
import yaml
import time
import uuid
import pytest

import numpy as np
import pandas as pd
from astropy.time import Time

import sqlalchemy as sa

from src.parameters import Parameters
from src.project import Project
from src.observatory import VirtualDemoObs
from src.ztf import VirtualZTF

from src.database import Session
from src.source import Source
import src.dataset
from src.dataset import RawData
from src.observatory import VirtualDemoObs

basepath = os.path.abspath(os.path.dirname(__file__))
src.dataset.DATA_ROOT = basepath


def test_load_save_parameters():

    filename = "parameters_test.yaml"
    filename = os.path.abspath(os.path.join(basepath, filename))
    print(filename)
    # write an example parameters file
    with open(filename, "w") as file:
        data = {"username": "guy", "password": "12345"}
        yaml.dump(data, file, sort_keys=False)

    try:
        # create some parameters object
        # with a couple of required parameters
        pars = Parameters(["username", "password"])
        # password value should be overriden from file
        pars.password = None
        # extra parameter should remain untouched
        pars.extra_parameter = "test"
        pars.load(filename)

        pars.verify()

        # username was not defined before reading the file
        assert pars.username == "guy"
        assert pars.password == "12345"
        assert pars.extra_parameter == "test"

        # add a new parameter that doesn't exist in the file
        pars.required_pars.append("not_set")

        with pytest.raises(ValueError):
            pars.verify()
    finally:
        # cleanup the test file
        os.remove(filename)

    try:
        # test saving the parameters
        filename = "parameters_test_saved.yaml"
        pars.save(filename)
        with open(filename) as file:
            new_data = yaml.safe_load(file)
        assert {
            "username",
            "password",
            "extra_parameter",
            "required_pars",
            "verbose",
        } == set(new_data.keys())
        assert new_data["username"] == "guy"
        assert new_data["password"] == "12345"
        assert new_data["extra_parameter"] == "test"

    finally:
        # cleanup the test file
        os.remove(filename)


def test_default_project():
    proj = Project("default_test", config=False)
    assert proj.pars.observatories == {"DemoObs"}
    assert "demo" in proj.observatories
    assert isinstance(proj.observatories["demo"], VirtualDemoObs)


def test_project_user_inputs():

    project_str = str(uuid.uuid4())
    proj = Project(
        name="default_test",
        params={
            "project_string": project_str,
            "observatories": ["ZTF"],
            "calibration": {"cal_key": "cal_value"},
            "analysis": {"an_key": "an_value"},
        },
        obs_params={
            "ZTF": {"credentials": {"username": "guy", "password": "12345"}},
        },
        config=False,
    )

    # check the project parameters are loaded correctly
    assert proj.pars.project_string == project_str
    assert proj.pars.observatories == {"ZTF"}
    assert proj.catalog.pars.filename == "test.csv"

    # check the observatory was loaded correctly
    assert "ztf" in proj.observatories
    assert isinstance(proj.observatories["ztf"], VirtualZTF)
    assert proj.observatories["ztf"].calibration.pars.cal_key == "cal_value"
    assert proj.observatories["ztf"].analysis.pars.an_key == "an_value"
    assert proj.observatories["ztf"]._credentials["username"] == "guy"
    assert proj.observatories["ztf"]._credentials["password"] == "12345"


def test_project_config_file():
    project_str1 = str(uuid.uuid4())
    project_str2 = str(uuid.uuid4())

    data = {
        "project": {  # project wide definitions
            "project_string": project_str1,  # random string
            "calibration": {  # should be overriden by observatory calibration
                "cal_key": "project_calibration",
            },
            "analysis": {  # should be overriden by observatory analysis
                "an_key": "project_analysis",
            },
        },
        "demo": {  # demo observatory specific definitions
            "demo_boolean": False,
            "demo_string": "test-string",
        },
        "ztf": {
            "credentials": {
                "filename": os.path.abspath(
                    os.path.join(basepath, "passwords_test.yaml")
                ),
            },
            "calibration": {
                "cal_key": "ztf_calibration",
            },
            "analysis": {
                "an_key": "ztf_analysis",
            },
        },
    }
    # TODO: add Catalog configurations

    # make config and passwords file
    configs_folder = os.path.abspath(os.path.join(basepath, "../configs"))
    if not os.path.isdir(configs_folder):
        os.mkdir(configs_folder)
    filename = os.path.join(configs_folder, "default_test.yaml")
    with open(filename, "w") as file:
        yaml.dump(data, file, sort_keys=False)
    with open(data["ztf"]["credentials"]["filename"], "w") as file:
        password = str(uuid.uuid4())
        yaml.dump(
            {"ztf": {"username": "test-username", "password": password}},
            file,
            sort_keys=False,
        )

    try:
        # do not load the config file
        proj = Project("default_test", config=False)
        assert not hasattr(proj.pars, "project_string")

        proj = Project(
            "default_test", params={"observatories": ["DemoObs", "ZTF"]}, config=True
        )
        assert proj.pars.project_string == project_str1
        assert proj.pars.calibration["cal_key"] == "project_calibration"
        assert proj.pars.analysis["an_key"] == "project_analysis"

        # check the observatories were loaded correctly
        assert "demo" in proj.observatories
        assert isinstance(proj.observatories["demo"], VirtualDemoObs)
        # existing parameters should be overridden by the config file
        assert proj.observatories["demo"].pars.demo_boolean is False
        # new parameter is successfully added
        assert proj.observatories["demo"].pars.demo_string == "test-string"

        # check the ZTF calibration/analysis got their own parameters loaded
        assert "ztf" in proj.observatories
        assert isinstance(proj.observatories["ztf"], VirtualZTF)
        assert proj.observatories["ztf"].calibration.pars.cal_key == "ztf_calibration"
        assert proj.observatories["ztf"].analysis.pars.an_key == "ztf_analysis"

        # check the user inputs override the config file
        proj = Project(
            "default_test", params={"project_string": project_str2}, config=True
        )
        assert proj.pars.project_string == project_str2

    finally:
        os.remove(filename)
        os.remove(data["ztf"]["credentials"]["filename"])


def test_catalog():
    pass


def test_add_source_and_data():
    fullname = ""
    try:  # at end, delete the temp file

        with Session() as session:
            # create a random source
            source_id = str(uuid.uuid4())
            new_source = Source(
                name=source_id,
                ra=np.random.uniform(0, 360),
                dec=np.random.uniform(-90, 90),
            )

            # add some data to the source
            num_points = 10
            filt = np.random.choice(["r", "g", "i", "z"], num_points)
            mjd = np.random.uniform(57000, 58000, num_points)
            mag = np.random.uniform(15, 20, num_points)
            mag_err = np.random.uniform(0.1, 0.5, num_points)
            test_data = dict(mjd=mjd, mag=mag, mag_err=mag_err, filter=filt)
            df = pd.DataFrame(test_data)

            # add the data to a database mapped object
            new_data = RawData(data=df, folder="data_temp", altdata=dict(foo="bar"))

            # check the times make sense
            start_time = Time(min(df.mjd), format="mjd").datetime
            end_time = Time(max(df.mjd), format="mjd").datetime
            assert start_time == new_data.time_start
            assert end_time == new_data.time_end

            new_source.raw_data.append(new_data)
            session.add(new_source)

            # this should not work because
            # no filename was specified
            with pytest.raises(ValueError):
                session.commit()
            session.rollback()
            session.add(new_source)

            new_data.filename = "test_data.h5"
            # this should not work because the file
            # does not yet exist and autosave is False
            assert new_data.autosave is False
            assert new_data.check_file_exists() is False
            with pytest.raises(ValueError):
                session.commit()
            session.rollback()
            new_data.filename = None  # reset the filename

            session.add(new_source)

            # filename should be auto-generated
            new_data.save()  # must save to allow RawData to be added to DB
            session.commit()  # this should now work fine
            assert new_source.id is not None
            assert new_source.id == new_data.source_id

            # try to recover the data
            filename = new_data.filename
            fullname = os.path.join(basepath, "data_temp", filename)

            with pd.HDFStore(fullname) as store:
                key = store.keys()[0]
                df_from_file = store.get(key)
                assert df_from_file.equals(df)
                dict_from_file = store.get_storer(key).attrs
                assert dict_from_file["foo"] == "bar"

        # check that the data is in the database
        with Session() as session:
            sources = session.scalars(
                sa.select(Source).where(Source.name == source_id)
            ).first()
            assert sources is not None
            assert len(sources.raw_data) == 1
            assert sources.raw_data[0].filename == filename
            assert sources.raw_data[0].key == new_data.key
            assert sources.raw_data[0].source_id == new_source.id
            # this autoloads the data:
            assert sources.raw_data[0].data.equals(df)

    finally:
        if os.path.isfile(fullname):
            os.remove(fullname)
    with pytest.raises(FileNotFoundError):
        with open(fullname) as file:
            pass

    # make sure loading this data does not work without file
    with Session() as session:
        sources = session.scalars(
            sa.select(Source).where(Source.name == source_id)
        ).first()
        assert sources is not None
        assert len(sources.raw_data) == 1
        with pytest.raises(FileNotFoundError):
            sources.raw_data[0].data.equals(df)

    # make sure deleting the source also cleans up the data
    with Session() as session:
        session.execute(sa.delete(Source).where(Source.name == source_id))
        session.commit()
        data = session.scalars(
            sa.select(RawData).where(RawData.key == new_data.key)
        ).first()
        assert data is None


def test_data_reduction(test_project, new_source):
    fullname = ""
    with Session() as session:
        try:  # at end, delete the temp file
            # add some data to the source
            num_points = 30
            filt = np.random.choice(["r", "g", "i"], num_points)
            mjd = np.random.uniform(57000, 58000, num_points)
            mag = np.random.uniform(15, 20, num_points)
            mag_err = np.random.uniform(0.1, 0.5, num_points)
            oid = np.random.randint(0, 5, num_points)
            test_data = dict(mjd=mjd, mag=mag, mag_err=mag_err, filter=filt, oid=oid)
            df = pd.DataFrame(test_data)

            # add the data to a database mapped object
            source_id = new_source.id
            new_source.project = test_project.name
            new_data = RawData(data=df, folder="data_temp", altdata=dict(foo="bar"))
            new_source.raw_data.append(new_data)

            # reduce the data use the demo observatory
            assert len(test_project.observatories) == 1
            obs_key = list(test_project.observatories.keys())[0]
            obs = test_project.observatories[obs_key]  # key should be "demo"
            assert isinstance(obs, VirtualDemoObs)
            new_source.lightcurves = obs.reduce(new_data, to="lcs", source=new_source)

            # save to disk
            # session.add(new_source)
            # session.commit()

        finally:
            new_data.delete_data_from_disk()
            assert not os.path.isfile(fullname)

        # make sure deleting the source also cleans up the data
        session.execute(sa.delete(Source).where(Source.name == source_id))
        session.commit()
        data = session.scalars(
            sa.select(RawData).where(RawData.key == new_data.key)
        ).first()
        assert data is None
