import os
import yaml

import uuid
import pytest


from src import utils
from src.parameters import Parameters
from src.project import Project
from src.observatory import VirtualDemoObs
from src.ztf import VirtualZTF

basepath = os.path.abspath(os.path.dirname(__file__))


def test_utils():
    pass  # TODO: add tests to other util functions
    # filename = "passwords_test.yml"
    # with open(filename, "w") as file:
    #     data = {"test": {"username": "guy", "password": "12345"}}
    #     yaml.dump(data, file, sort_keys=False)
    # try:
    #     credentials = utils.get_username_password("test", filename)
    #
    #     assert credentials[0] == "guy"
    #     assert credentials[1] == "12345"
    #
    # finally:
    #     os.remove(filename)


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

    # TODO: make a CVS catalog file to check it finds it ok

    project_str = str(uuid.uuid4())
    proj = Project(
        name="default_test",
        params={
            "project_string": project_str,
            "observatories": ["ZTF"],
            "catalog": {"filename": "demo_catalog.csv"},
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
    assert proj.catalog.pars.filename == "demo_catalog.csv"

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
    configs_folder = os.path.abspath(os.path.join(basepath, "configs"))
    if not os.path.isdir(configs_folder):
        os.mkdir(configs_folder)
    filename = os.path.join(configs_folder, "default_test.yaml")
    with open(filename, "w") as file:
        yaml.dump(data, file, sort_keys=False)
        print(filename)
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
        pass
        os.remove(filename)
        os.remove(data["ztf"]["credentials"]["filename"])
