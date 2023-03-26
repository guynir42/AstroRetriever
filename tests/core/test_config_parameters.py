import os
import yaml
import uuid
import pytest

import numpy as np

from src.parameters import Parameters, ParsDemoSubclass
from src.project import Project
from src.ztf import VirtualZTF

import src.database
from src.database import safe_mkdir
from src.observatory import VirtualDemoObs
from src.properties import Properties
from src.utils import random_string, legalize


def test_load_save_parameters(data_dir):

    filename = "parameters_test.yaml"
    filename = os.path.abspath(os.path.join(data_dir, filename))
    safe_mkdir(os.path.dirname(filename))

    # write an example parameters file
    with open(filename, "w") as file:
        data = {"username": "guy", "password": "12345"}
        yaml.dump(data, file, sort_keys=False)

    try:
        # create some parameters object
        # with a couple of required parameters
        pars = Parameters()
        # password value should be overridden from file
        pars.password = None
        # extra parameter should remain untouched
        pars.extra_parameter = "test"
        config = pars.load(filename)
        pars.read(config)

        # username was not defined before reading the file
        assert pars.username == "guy"
        assert pars.password == "12345"
        assert pars.extra_parameter == "test"

    finally:
        # cleanup the test file
        os.remove(filename)

    try:
        # test saving the parameters
        filename = "parameters_test_saved.yaml"
        pars.save(filename)
        with open(filename) as file:
            new_data = yaml.safe_load(file)
            print(new_data)
        assert {
            "username",
            "password",
            "extra_parameter",
            "data_types",
            "verbose",
            "project",
            "cfg_file",
        } == set(new_data.keys())
        assert new_data["username"] == "guy"
        assert new_data["password"] == "12345"
        assert new_data["extra_parameter"] == "test"

    finally:
        # cleanup the test file
        os.remove(filename)


def test_default_project():
    proj = Project("default_test", catalog_kwargs={"default": "test"})
    assert proj.pars.obs_names == ["DEMO"]
    assert "DEMO" in [obs.name for obs in proj.observatories]
    assert isinstance(proj.observatories[0], VirtualDemoObs)


def test_project_user_inputs():

    proj = Project(
        name="default_test",
        obs_names=["demo", "ZTF"],
        analysis_kwargs={"num_injections": 3},
        obs_kwargs={
            "reduce_kwargs": {"reducer_key": "reducer_value"},
            "ZTF": {"credentials": {"username": "guy", "password": "12345"}},
            "DEMO": {"reduce_kwargs": {"reducer_key": "reducer_value2"}},
        },
        catalog_kwargs={"default": "test"},
    )

    # check the project parameters are loaded correctly
    assert set(proj.pars.obs_names) == {"DEMO", "ZTF"}
    assert proj.catalog.pars.filename == "test.csv"

    # check the observatory was loaded correctly
    assert "ZTF" in [obs.name for obs in proj.observatories]
    idx = None
    for i, item in enumerate(proj.observatories):
        if isinstance(item, VirtualZTF):
            idx = i
            break
    assert idx is not None
    assert isinstance(proj.observatories[idx], VirtualZTF)
    # check that observatories can be referenced using (case-insensitive) strings
    assert isinstance(proj.observatories["ztf"], VirtualZTF)
    assert isinstance(proj.observatories["ZTF"], VirtualZTF)
    assert isinstance(proj.observatories["ZTF"]._credentials, dict)
    assert proj.ztf._credentials["username"] == "guy"
    assert proj.ztf._credentials["password"] == "12345"

    # check the reducer was overriden in the demo observatory
    assert proj.ztf.pars.reduce_kwargs["reducer_key"] == "reducer_value"
    assert proj.demo.pars.reduce_kwargs["reducer_key"] == "reducer_value2"


def test_project_config_file():
    project_str1 = str(uuid.uuid4())
    project_str2 = str(uuid.uuid4())

    configs_folder = os.path.join(src.database.CODE_ROOT, "configs")

    data = {
        "project": {  # project wide definitions
            "description": project_str1,  # random string
            "obs_names": ["demo", "ztf"],  # list of observatory names
        },
        "observatories": {  # general instructions to pass to observatories
            "reduce_kwargs": {  # should be overriden by observatory reducer
                "reducer_key": "project_reduction",
            },
            "demo": {  # demo observatory specific definitions
                "demo_boolean": False,
                "demo_string": "test-string",
            },
            "ztf": {
                "credentials": {
                    "filename": os.path.abspath(
                        os.path.join(configs_folder, "passwords_test.yaml")
                    ),
                },
                "reduce_kwargs": {
                    "reducer_key": "ztf_reduction",
                },
            },
        },
        "catalog": {"default": "test"},  # catalog definitions
        "analysis": {
            "num_injections": 2.5,
        },
    }

    # make config and passwords file
    if not os.path.isdir(configs_folder):
        os.mkdir(configs_folder)
    filename = os.path.join(configs_folder, "default_test.yaml")

    # make the config file
    with open(filename, "w") as file:
        yaml.dump(data, file, sort_keys=False)

    # make the passwords file
    with open(data["observatories"]["ztf"]["credentials"]["filename"], "w") as file:
        password = str(uuid.uuid4())
        yaml.dump(
            {"ztf": {"username": "test-username", "password": password}},
            file,
            sort_keys=False,
        )

    try:
        # do not load the config file
        proj = Project(
            "default_test", catalog_kwargs={"default": "test"}, cfg_file=False
        )
        assert proj.pars.description == ""

        # load the default config file at configs/default_test.yaml
        proj = Project("default_test")
        assert "description" in proj.pars
        assert proj.pars.description == project_str1
        assert proj.analysis.pars.num_injections == 2.5

        # check the observatories were loaded correctly
        assert "demo" in proj.observatories
        assert isinstance(proj.observatories["demo"], VirtualDemoObs)
        # existing parameters should be overridden by the config file
        assert proj.demo.pars.demo_boolean is False
        # new parameter is successfully added
        assert proj.demo.pars.demo_string == "test-string"
        # general project-wide reducer is used by demo observatory:
        assert proj.demo.pars.reduce_kwargs["reducer_key"] == "project_reduction"

        # check the ZTF calibration/analysis got their own parameters loaded
        assert "ztf" in proj.observatories
        assert isinstance(proj.observatories["ztf"], VirtualZTF)
        assert proj.ztf.pars.reduce_kwargs == {"reducer_key": "ztf_reduction"}

        # check the user inputs override the config file
        proj = Project(
            "default_test",
            description=project_str2,
            obs_kwargs={
                "demo": {
                    "demo_string": "new-test-string"
                },  # directly override demo parameters
            },
        )
        assert proj.pars.description == project_str2
        assert proj.demo.pars.demo_string == "new-test-string"

    finally:
        os.remove(filename)
        os.remove(data["observatories"]["ztf"]["credentials"]["filename"])


def test_legal_project_names(new_source, raw_phot):
    original_name = random_string()
    proj = Project(original_name, catalog_kwargs={"default": "test"})
    assert proj.name != original_name
    assert proj.name == original_name.upper()

    original_name = f"  {random_string(8)}-{random_string(8)}12 "
    proj = Project(original_name, catalog_kwargs={"default": "test"})
    assert proj.name != original_name
    assert proj.name == legalize(original_name)
    assert proj.name.endswith("12")
    assert not proj.name.startswith(" ")
    assert "-" not in proj.name
    assert "_" in proj.name

    assert proj.pars.project == legalize(original_name)

    new_source.project = original_name
    assert new_source.project == legalize(original_name)

    raw_phot.project = original_name
    assert raw_phot.project == legalize(original_name)

    prop = Properties(project=original_name)
    assert prop.project == legalize(original_name)

    original_name = f"1{random_string(8)}-{random_string(8)}12 "
    with pytest.raises(ValueError) as e:
        Project(original_name, catalog_kwargs={"default": "test"})
    assert "Cannot legalize name" in str(e.value)

    original_name = f"{random_string(8)}-{random_string(8)} $ "
    with pytest.raises(ValueError) as e:
        Project(original_name, catalog_kwargs={"default": "test"})
    assert "Cannot legalize name" in str(e.value)


def test_parameter_name_matches():
    # cannot initialize parameters subclass (which is locked) with wrong key
    with pytest.raises(AttributeError) as e:
        ParsDemoSubclass(wrong_key="test")
    assert 'Attribute "wrong_key" does not exist.' in str(e)

    int_par = np.random.randint(2, 10)
    float_par = np.random.uniform(2, 10)

    p = ParsDemoSubclass(Int_Par=int_par, FLOAT_P=float_par, plot=False, null=None)

    assert p.Int_Par == int_par
    assert p.integer_parameter == int_par
    assert p.FLOAT_P == float_par
    assert p.float_parameter == float_par
    assert p.plot is False
    assert p.plotting_value is False
    assert p.null is None
    assert p.nullable_parameter is None

    # try to set parameters using aliases with capitals
    p.INTEGER = int_par * 2
    assert p.integer_parameter == int_par * 2
    p.INT = int_par * 3
    assert p.integer_parameter == int_par * 3

    p.FLOAT = float_par * 2
    assert p.float_parameter == float_par * 2

    # an attribute not related to any of these parameters still fails
    with pytest.raises(AttributeError) as e:
        p.wrong_attribute = 1
    assert 'Attribute "wrong_attribute" does not exist.' in str(e)

    # now turn off the partial matches
    p._allow_shorthands = False

    with pytest.raises(AttributeError) as e:
        p.integer = int_par * 2
    assert 'Attribute "integer" does not exist.' in str(e)

    with pytest.raises(AttributeError) as e:
        p.float = float_par * 2
    assert 'Attribute "float" does not exist.' in str(e)

    # still works with capitals
    p.Integer_Parameter = int_par * 2
    assert p.integer_parameter == int_par * 2

    # still works with aliases
    p.INT_PAR = int_par * 3
    assert p.integer_parameter == int_par * 3

    # turn off case-insensitivity
    p._ignore_case = False

    with pytest.raises(AttributeError) as e:
        p.integer_parameteR = int_par * 2
    assert 'Attribute "integer_parameteR" does not exist.' in str(e)

    # aliases still work
    p.int_par = int_par * 3
    assert p.integer_parameter == int_par * 3

    # turn partial matches back on:
    p._allow_shorthands = True
    p.float_ = float_par * 2
    assert p.float_parameter == float_par * 2

    # but capitals are still turned off
    with pytest.raises(AttributeError) as e:
        p.Float_Parameter = float_par * 2
    assert 'Attribute "Float_Parameter" does not exist.' in str(e)


def test_version_control(data_dir):
    proj = Project(
        "test_project",
        obs_names=["demo", "ztf"],
        version_control=True,
        catalog_kwargs={"default": "test"},
        analysis_kwargs={"num_injections": 3},
    )
    assert isinstance(proj.pars.git_hash, str)

    try:  # cleanup config files at the end
        output_filename = None

        # save the config file
        proj._save_config()
        output_filename = os.path.join(data_dir, proj.output_folder, "config.yaml")

        assert os.path.basename(proj.output_folder) == "TEST_PROJECT_" + proj.cfg_hash
        assert os.path.isfile(output_filename)

        # load a new Project with the output config file, and check that all
        # parameters on both lists are the same.
        list1 = proj.pars.get_pars_list(proj)

        proj2 = Project("test_project", cfg_file=output_filename)
        list2 = proj2.pars.get_pars_list(proj2)
        assert len(list1) == len(list2)

        for p1, p2 in zip(list1, list2):
            assert p1.compare(p2, ignore=["cfg_file"], verbose=True)
    finally:  # cleanup
        if output_filename is not None:
            if os.path.isfile(output_filename):
                os.remove(output_filename)
            folder = os.path.dirname(output_filename)
            if os.path.isdir(folder):
                os.rmdir(folder)
