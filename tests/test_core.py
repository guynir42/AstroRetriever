import os
import yaml
import time
import uuid
import pytest
from pprint import pprint

import numpy as np
import pandas as pd
from astropy.time import Time

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, InvalidRequestError

from src.utils import OnClose, UniqueList
from src.parameters import Parameters
from src.project import Project
from src.ztf import VirtualZTF

import src.database
from src.database import (
    Session,
    SmartSession,
    NoOpSession,
    NullQueryResults,
    safe_mkdir,
)
from src.source import Source, DEFAULT_PROJECT
from src.dataset import RawPhotometry, Lightcurve, PHOT_ZP, simplify, get_time_offset
from src.observatory import VirtualDemoObs
from src.catalog import Catalog
from src.detection import Detection
from src.properties import Properties
from src.histogram import Histogram
from src.utils import NamedList, UniqueList, CircularBufferList, random_string, legalize


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


def test_catalog(data_dir):
    filename = "test_catalog.csv"
    fullname = os.path.abspath(os.path.join(data_dir, "../catalogs", filename))

    try:
        Catalog.make_test_catalog(filename=filename, number=10)
        assert os.path.isfile(fullname)

        # set up a catalog with the default column definitions
        cat = Catalog(filename=filename, default="test")
        cat.load()
        assert cat.pars.filename == filename
        assert len(cat.data) == 10
        assert cat.data["ra"].dtype == np.float64
        assert cat.data["dec"].dtype == np.float64
        assert cat.pars.name_column in cat.data.columns

    finally:
        os.remove(fullname)
        assert not os.path.isfile(fullname)


def test_catalog_hdf5(data_dir):
    filename = "test_catalog.h5"
    fullname = os.path.abspath(os.path.join(data_dir, "../catalogs", filename))

    try:
        Catalog.make_test_catalog(filename=filename, number=10)
        assert os.path.isfile(fullname)

        # setup a catalog with the default column definitions
        cat = Catalog(filename=filename, default="test")
        cat.load()
        assert cat.pars.filename == filename
        assert len(cat.data) == 10
        assert cat.data["ra"].dtype == np.float64
        assert cat.data["dec"].dtype == np.float64
        assert cat.pars.name_column in cat.data.columns

    finally:
        os.remove(fullname)
        assert not os.path.isfile(fullname)


def test_catalog_wds():
    cat = Catalog(default="wds")
    cat.load()
    assert len(cat.data) > 0
    assert isinstance(cat.data["ra"][0], float)
    assert isinstance(cat.data["dec"][0], float)
    assert cat.pars.name_column in cat.get_columns()

    # more than a million sources
    assert len(cat) > 1000_000

    # catalog should be ordered by RA, so first objects are close to 360/0
    assert cat.data["ra"][0] > 359 or cat.data["ra"][0] < 1

    # average values are weighted by the galactic bulge
    assert abs(cat.data["ra"].mean() - 229) < 1
    assert abs(cat.data["dec"].mean() + 20) < 1

    # mean magnitude hovers around the limit of ~20
    assert abs(cat.data[cat.pars.mag_column].mean() - 20) < 0.1


def test_catalog_get_row():
    cat = Catalog(default="wds")
    cat.load()
    assert len(cat.data) > 0

    # get row based on index
    row = cat.get_row(0)  # first row
    assert isinstance(row, np.record)
    assert row["ra"] == cat.data["ra"][0]
    assert row["dec"] == cat.data["dec"][0]

    row = cat.get_row(-1)  # last row
    assert isinstance(row, np.record)
    assert row["ra"] == cat.data["ra"][-1]
    assert row["dec"] == cat.data["dec"][-1]

    idx = 7  # random choice
    row = cat.get_row(idx)
    assert isinstance(row, np.record)
    assert row["ra"] == cat.data["ra"][idx]
    assert row["dec"] == cat.data["dec"][idx]

    name = row[cat.pars.name_column]
    assert name == cat.data[cat.pars.name_column][idx]

    # get row based on name
    row = cat.get_row(name, index_type="name")
    assert isinstance(row, np.record)
    assert row["ra"] == cat.data["ra"][idx]
    assert row["dec"] == cat.data["dec"][idx]

    # get the row in the form of a dict
    row = cat.get_row(name, index_type="name", output="dict")
    assert isinstance(row, dict)
    assert row["name"] == str(name)
    assert row["ra"] == cat.data["ra"][idx]
    assert row["dec"] == cat.data["dec"][idx]

    # try to apply proper motion
    t = Time("2022-01-01")
    row = cat.get_row(name, index_type="name", output="dict", obstime=t)
    assert isinstance(row, dict)
    assert row["ra"] != cat.data["ra"][idx]
    assert abs(row["ra"] - cat.data["ra"][idx]) < 0.1

    # choose a preferred mag that is not Gaia_G
    row = cat.get_row(name, index_type="name", output="dict", preferred_mag="Gaia_BP")
    assert isinstance(row, dict)
    assert row["mag"] == cat.data["phot_bp_mean_mag"][idx]


def test_catalog_nearest_search():
    c = Catalog(default="test")
    c.load()

    idx = 2
    ra = c.data["ra"][idx]
    dec = c.data["dec"][idx]

    # search for the nearest object
    nearest = c.get_nearest_row(ra, dec, radius=2.0, output="dict")
    assert nearest["ra"] == ra
    assert nearest["dec"] == dec

    # try to nudge the coordinates a little bit
    nearest = c.get_nearest_row(
        ra + 0.3 / 3600, dec - 0.3 / 3600, radius=2.0, output="dict"
    )
    assert nearest["ra"] == ra
    assert nearest["dec"] == dec

    # make sure search works even if object is at RA=0
    c.data.loc[idx, "ra"] = 0

    nearest = c.get_nearest_row(0.1 / 3600, dec, radius=2.0, output="dict")
    assert nearest["ra"] == 0
    assert nearest["dec"] == dec

    nearest = c.get_nearest_row(-0.1 / 3600, dec, radius=2.0, output="dict")
    assert nearest["ra"] == 0
    assert nearest["dec"] == dec

    nearest = c.get_nearest_row(360 - 0.1 / 3600, dec, radius=2.0, output="dict")
    assert nearest["ra"] == 0
    assert nearest["dec"] == dec


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


def test_add_source_and_data(data_dir, test_hash):
    fullname = ""
    try:  # at end, delete the temp file

        with SmartSession() as session:
            # create a random source
            source_name = str(uuid.uuid4())

            # make sure source cannot be initialized with bad keyword
            with pytest.raises(ValueError) as e:
                _ = Source(name=source_name, foobar="test")
                assert "Unknown keyword argument" in str(e.value)

            new_source = Source(
                name=source_name,
                ra=np.random.uniform(0, 360),
                dec=np.random.uniform(-90, 90),
                test_hash=test_hash,
            )
            assert isinstance(new_source.raw_photometry, UniqueList)

            # add some data to the source
            num_points = 10
            filt = np.random.choice(["r", "g", "i", "z"], num_points)
            mjd = np.random.uniform(57000, 58000, num_points)
            mag = np.random.uniform(15, 20, num_points)
            mag_err = np.random.uniform(0.1, 0.5, num_points)
            test_data = dict(mjd=mjd, mag=mag, mag_err=mag_err, filter=filt)
            df = pd.DataFrame(test_data)

            # make sure data cannot be initialized with bad keyword
            with pytest.raises(ValueError) as e:
                _ = RawPhotometry(data=df, foobar="test")
                assert "Unknown keyword argument" in str(e.value)

            # add the data to a database mapped object
            new_data = RawPhotometry(
                data=df,
                source_name=source_name,
                observatory="demo",
                folder="data_temp",
                altdata=dict(foo="bar"),
                test_hash=test_hash,
            )

            # check the times make sense
            start_time = Time(min(df.mjd), format="mjd").datetime
            end_time = Time(max(df.mjd), format="mjd").datetime
            assert start_time == new_data.time_start
            assert end_time == new_data.time_end

            session.add(new_source)
            # must add this separately, as cascades are cancelled
            session.add(new_data)
            new_source.raw_photometry = [new_data]
            assert isinstance(new_source.raw_photometry, UniqueList)
            assert len(new_source.raw_photometry) == 1

            # check that re-appending this same data does not add another copy to list
            new_source.raw_photometry.append(new_data)
            assert len(new_source.raw_photometry) == 1

            # this should not work because
            # no filename was specified
            with pytest.raises(ValueError):
                session.commit()
            session.rollback()

            session.add(new_source)
            # must add this separately, as cascades are cancelled
            session.add(new_data)
            new_source.raw_photometry = [new_data]

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
            # must add this separately, as cascades are cancelled
            session.add(new_data)
            new_source.raw_photometry = [new_data]

            # filename should be auto-generated
            new_data.save()  # must save to allow RawPhotometry to be added to DB
            session.commit()  # this should now work fine
            assert new_source.id is not None
            assert new_source.id == new_data.source.id

            # try to recover the data
            filename = new_data.filename
            fullname = os.path.join(data_dir, "data_temp", filename)

            with pd.HDFStore(fullname, "r") as store:
                key = store.keys()[0]
                df_from_file = store.get(key)
                assert df_from_file.equals(df)
                altdata = store.get_storer(key).attrs.altdata
                assert altdata["foo"] == "bar"

        # check that the data is in the database
        with SmartSession() as session:
            source = session.scalars(
                sa.select(Source).where(Source.name == source_name)
            ).first()

            assert source is not None
            assert source.id == new_source.id
            source.get_data(
                obs="demo", data_type="photometry", level="raw", session=session
            )
            assert len(source.raw_photometry) == 1
            assert source.raw_photometry[0].filename == filename
            assert source.raw_photometry[0].filekey == new_data.filekey
            assert source.raw_photometry[0].source.id == new_source.id

            # this autoloads the data:
            assert source.raw_photometry[0].data.equals(df)

    finally:
        if os.path.isfile(fullname):
            os.remove(fullname)
            if os.path.isdir(os.path.dirname(fullname)):
                os.rmdir(os.path.dirname(fullname))
    with pytest.raises(FileNotFoundError):
        with open(fullname) as file:
            pass

    # make sure loading this data does not work without file
    with SmartSession() as session:
        source = session.scalars(
            sa.select(Source).where(Source.name == source_name)
        ).first()
        assert source is not None

        # get_data will load a RawPhotometry file without checking if it has data
        source.get_data(
            obs="demo",
            data_type="photometry",
            level="raw",
            session=session,
            check_data=False,
            delete_missing=False,
        )
        assert len(source.raw_photometry) == 1
        assert len(source.raw_photometry) == 1
        with pytest.raises(FileNotFoundError):
            source.raw_photometry[0].data.equals(df)

    # cleanup
    with SmartSession() as session:
        session.execute(sa.delete(Source).where(Source.name == source_name))
        session.commit()


def test_source_unique_constraint(test_hash):

    with SmartSession() as session:
        name1 = str(uuid.uuid4())
        source1 = Source(name=name1, ra=0, dec=0, test_hash=test_hash)
        assert source1.cfg_hash == ""  # the default has is an empty string
        session.add(source1)

        source2 = Source(name=name1, ra=0, dec=0, test_hash=test_hash)
        assert source1.cfg_hash == ""  # the default has is an empty string
        session.add(source2)

        # should fail as both sources have the same name
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

        name2 = str(uuid.uuid4())
        source2 = Source(name=name2, ra=0, dec=0, test_hash=test_hash)
        session.add(source1)
        session.add(source2)
        session.commit()

        # try to add source2 with the same name but different project
        assert source1.project == DEFAULT_PROJECT
        source2.project = "another test"
        source2.name = name1
        session.add(source2)
        session.commit()

        # try to add source2 with the same name but different cfg_hash
        source2.project = DEFAULT_PROJECT
        assert source2.project == source1.project
        assert source2.name == source1.name

        source2.cfg_hash = "another hash"
        session.add(source2)
        session.commit()


def test_raw_photometry_unique_constraint(raw_phot, raw_phot_no_exptime):

    with SmartSession() as session:
        name = str(uuid.uuid4())
        raw_phot.source_name = name
        raw_phot.filename = "unique_test1.h5"
        raw_phot.save()
        raw_phot_no_exptime.source_name = name
        raw_phot_no_exptime.filename = "unique_test2.h5"
        raw_phot_no_exptime.save()

        session.add(raw_phot)
        session.add(raw_phot_no_exptime)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

        # should work once the observatory name is different
        raw_phot_no_exptime.observatory = random_string(8)
        session.add(raw_phot)
        session.add(raw_phot_no_exptime)
        session.commit()

        # let's try to add the data with same obs
        # but different source name
        session.delete(raw_phot)
        raw_phot.observatory = raw_phot_no_exptime.observatory
        raw_phot.source_name = str(uuid.uuid4())
        session.add(raw_phot)
        session.commit()


def test_data_file_paths(raw_phot, data_dir):
    try:  # at end, delete the temp files
        raw_phot.save(overwrite=True)
        assert raw_phot.filename is not None
        assert "photometry" in raw_phot.filename
        assert raw_phot.filename.endswith(".h5")

    finally:
        raw_phot.delete_data_from_disk()
        assert not os.path.isfile(raw_phot.get_fullname())

    # just a filename does not affect folder
    # default folder is given as 'DATA'
    raw_phot.folder = None
    raw_phot.filename = "test.h5"
    assert raw_phot.folder is None
    assert raw_phot.filename == "test.h5"
    assert raw_phot.get_fullname() == os.path.join(data_dir, "DEMO/test.h5")

    # no folder is given, but has observatory name to use as default
    raw_phot.observatory = "ztf"
    assert raw_phot.get_fullname() == os.path.join(data_dir, "ZTF/test.h5")

    # give the folder explicitly, will override the default
    raw_phot.folder = "test"
    assert raw_phot.get_fullname() == os.path.join(data_dir, "test/test.h5")

    # adding a path to filename puts that path into "folder"
    raw_phot.folder = None
    raw_phot.filename = "path/to/test/test.h5"

    assert raw_phot.get_fullname() == os.path.join(data_dir, "ZTF/path/to/test/test.h5")

    # an absolute path in "folder" will ignore DATA_ROOT
    raw_phot.folder = "/path"
    raw_phot.filename = "to/test/test.h5"
    assert raw_phot.get_fullname() == "/path/to/test/test.h5"


def test_data_reduction(test_project, new_source, raw_phot_no_exptime):

    with SmartSession() as session:

        # add the data to a database mapped object
        new_source.project = test_project.name
        raw_phot_no_exptime.save(overwrite=True)
        new_source.raw_photometry.append(raw_phot_no_exptime)

        # reduce the data using the demo observatory
        assert len(test_project.observatories) == 1
        obs_key = list(test_project.observatories.keys())[0]
        assert obs_key == "DEMO"
        obs = test_project.observatories[obs_key]
        assert isinstance(obs, VirtualDemoObs)

        # cannot generate photometric data without an exposure time
        with pytest.raises(ValueError) as exc:
            obs.reduce(source=new_source, data_type="photometry")
        assert "No exposure time" in str(exc.value)

        # add exposure time to the dataframe:
        new_source.raw_photometry[0].data["exp_time"] = 30.0
        lightcurves = obs.reduce(source=new_source, data_type="photometry")

        session.add(new_source)
        session.add(raw_phot_no_exptime)
        session.add_all(lightcurves)
        with pytest.raises(ValueError) as exc:
            session.commit()
        assert "No filename" in str(exc.value)
        session.rollback()

        # must save dataset before adding it to DB
        [lc.save(overwrite=True) for lc in lightcurves]
        filenames = [lc.get_fullname() for lc in lightcurves]

        session.add(new_source)
        session.add(raw_phot_no_exptime)
        session.add_all(lightcurves)
        session.commit()

        # check that the data has been reduced as expected
        for lc in lightcurves:
            filt = lc.filter
            dff = raw_phot_no_exptime.data[raw_phot_no_exptime.data["filter"] == filt]
            dff = dff.sort_values(by="mjd", inplace=False)
            dff.reset_index(drop=True, inplace=True)

            # make sure it picks out the right points
            assert dff["mjd"].equals(lc.data["mjd"])
            assert dff["mag"].equals(lc.data["mag"])
            assert dff["mag_err"].equals(lc.data["magerr"])

            # make sure the number of points are correct
            assert lc.number == len(dff)
            # need -1 to remove the one column for MJD we add
            assert lc.shape == (len(dff), len(lc.colmap) - 1)

            # make sure the frame rate and exposure time are correct
            assert lc.exp_time == 30.0
            assert np.isclose(
                1.0 / lc.frame_rate, dff["mjd"].diff().median() * 24 * 3600
            )
            assert not lc.is_uniformly_sampled

            # make sure the average flux is correct
            flux = 10 ** (-0.4 * (dff["mag"].values - PHOT_ZP))
            assert np.isclose(lc.flux_mean, np.nanmean(flux))

            # make sure flux min/max are correct
            assert np.isclose(lc.flux_min, np.min(flux))
            assert np.isclose(lc.flux_max, np.max(flux))

            # make sure superfluous columns are dropped
            assert "oid" not in lc.data.columns

            # make sure the start/end times are correct
            assert np.isclose(Time(lc.time_start).mjd, dff["mjd"].min())
            assert np.isclose(Time(lc.time_end).mjd, dff["mjd"].max())

        session.delete(new_source)
        session.delete(raw_phot_no_exptime)
        [session.delete(lc) for lc in lightcurves]
        session.commit()

        data = session.scalars(
            sa.select(RawPhotometry).where(
                RawPhotometry.filekey == raw_phot_no_exptime.filekey
            )
        ).first()
        assert data is None
        data = session.scalars(
            sa.select(Lightcurve).where(Lightcurve.source_name == new_source.name)
        ).all()
        assert len(data) == 0
        assert not any([os.path.isfile(f) for f in filenames])

        # check some of the shorthands and indexing methods
        assert new_source.raw_photometry["demo"] == raw_phot_no_exptime
        assert new_source.raw_photometry[0] == new_source.rp["demo"]

        assert new_source.rl == new_source.reduced_lightcurves
        assert new_source.rl["demo"] == lightcurves
        assert new_source.rl["demo", 0] == lightcurves[0]


def test_reduced_data_file_keys(test_project, new_source, raw_phot):

    obs = test_project.observatories["demo"]
    new_source.raw_photometry.append(raw_phot)
    raw_phot.source = new_source

    try:  # at end, delete the temp file
        raw_phot.save(overwrite=True)
        basename = os.path.splitext(raw_phot.filename)[0]

        lcs = obs.reduce(source=new_source, data_type="photometry")

        for lc in lcs:
            lc.save(overwrite=True)
            assert basename in lc.filename

        # make sure all filenames are the same
        assert lcs[0].filename == list({lc.filename for lc in lcs})[0]

        # check all the data exists in the file
        with pd.HDFStore(lcs[0].get_fullname()) as store:
            for lc in lcs:
                assert os.path.join("/", lc.filekey) in store.keys()
                assert len(store[lc.filekey]) == len(lc.data)

    finally:
        raw_phot.delete_data_from_disk()
        filename = lcs[0].get_fullname()

        for lc in lcs:
            lc.delete_data_from_disk()

    assert not os.path.isfile(raw_phot.get_fullname())
    assert not os.path.isfile(filename)


@pytest.mark.flaky(max_runs=3)
def test_reducer_with_outliers(test_project, new_source, test_hash):
    num_points = 30
    outlier_indices = [5, 8, 12]
    flagged_indices = [5, 10, 15]
    new_data = None
    lightcurves = None

    with SmartSession() as session:
        try:  # at end, delete the temp file
            filt = "R"
            mjd = np.linspace(57000, 58000, num_points)
            mag_err = np.random.uniform(0.09, 0.11, num_points)
            mag = np.random.normal(18.5, 0.1, num_points)
            mag[outlier_indices] = np.random.normal(10, 0.1, len(outlier_indices))
            # also improve the relative error for the bright outlier:
            mag_err[8] = 0.01
            # turn the second bright outlier into a faint outlier:
            mag[12] = np.random.normal(20, 0.1, 1)
            flag = np.zeros(num_points, dtype=bool)
            flag[flagged_indices] = True
            test_data = dict(mjd=mjd, mag=mag, mag_err=mag_err, filter=filt, flag=flag)
            df = pd.DataFrame(test_data)

            # add the data to a database mapped object
            new_source.project = test_project.name
            new_data = RawPhotometry(
                data=df,
                source_name=new_source.name,
                observatory="demo",
                folder="data_temp",
                altdata=dict(exptime="25.0"),
                test_hash=test_hash,
            )
            new_data.source = new_source
            new_source.raw_photometry.append(new_data)
            new_data.save()

            # reduce the data use the demo observatory
            assert len(test_project.observatories) == 1
            obs = test_project.observatories["demo"]
            assert isinstance(obs, VirtualDemoObs)

            obs.pars.reduce_kwargs["drop_flagged"] = False
            lightcurves = obs.reduce(source=new_source, data_type="photometry")
            new_source.lightcurves = lightcurves

            assert len(lightcurves) == 1
            lc = lightcurves[0]

            session.add(new_source)
            session.add(new_data)
            lc.save()
            session.add(lc)
            session.commit()

            # check the data has been reduced as expected
            df2 = df[~df["flag"]]
            drop_idx = list(set(outlier_indices + flagged_indices))
            df3 = df.drop(drop_idx, axis=0)
            assert np.isclose(lc.mag_brightest, df2["mag"].min())
            assert np.isclose(lc.mag_faintest, df2["mag"].max())
            assert lc.num_good == num_points - len(flagged_indices)

            # print(f'flux_mean= {lc.flux_mean} | flux_mean_robust= {lc.flux_mean_robust}')
            # print(f'flux rms= {lc.flux_rms} | flux rms robust= {lc.flux_rms_robust}')
            # print(f'mag mean= {lc.mag_mean} | mag mean robust= {lc.mag_mean_robust}')
            # print(f'mag rms= {lc.mag_rms} | mag rms robust= {lc.mag_rms_robust}')

            # check the robust statistics are representative of the data without outliers
            assert abs(np.nanmean(df3["mag"]) - lc.mag_mean_robust) < 0.1
            assert abs(np.nanstd(df3["mag"]) - lc.mag_rms_robust) < 0.1

            # checks for snr, dsnr, and dmag and their extrema:
            df4 = df.copy()
            df4.loc[flagged_indices, :] = np.nan  # without the flagged points
            assert np.argmax(df4["mag"]) == 12
            assert np.argmin(df4["mag"]) == 8

            # print(f'snr: {lc.data["snr"].values}')
            # print(f'dsnr: {lc.data["dsnr"].values}')
            # print(f'dmag: {lc.data["dmag"].values}')

            # test the S/N
            assert abs(np.nanmedian(lc.data["snr"].values) - 10) < 2  # noise is 0.1
            assert lc.data["snr"][8] > 20  # bright outlier has high S/N
            assert lc.data["snr"][12] < 5  # faint outlier has low S/N

            # test the delta S/N
            dsnr = lc.data["dsnr"].values
            dsnr[outlier_indices] = np.nan  # remove the outliers
            # should be close to zero if noise estimate is correct
            assert abs(np.nanmean(dsnr)) < 0.3
            assert abs(np.nanstd(dsnr) - 1) < 0.3

            # test the delta mag
            dmag = lc.data["dmag"].values
            assert abs(dmag[5] - 8.5) < 0.5  # about 8.5 mag difference
            assert abs(dmag[8] - 8.5) < 0.5  # about 8.5 mag difference
            assert abs(dmag[12] + 1.5) < 0.5  # about 8.5 mag difference

            dmag[outlier_indices] = np.nan  # remove the outliers
            assert (
                abs(np.nanmean(dmag[dmag > 0]) - 0.1) < 0.3
            )  # close to 0.1 mag difference
            assert (
                abs(np.nanmean(dmag[dmag < 0]) + 0.1) < 0.3
            )  # close to -0.1 mag difference
            assert abs(np.nanmean(dmag)) < 0.1

            # also check that the data is uniformly sampled
            assert lc.is_uniformly_sampled

            # check the data is persisted
            loaded_raw_data = session.scalars(
                sa.select(RawPhotometry).where(
                    RawPhotometry.source_name == new_source.name
                )
            ).all()
            assert len(loaded_raw_data) == 1

            loaded_lcs = session.scalars(
                sa.select(Lightcurve).where(Lightcurve.source_name == new_source.name)
            ).all()
            assert len(loaded_lcs) == len(lightcurves)

        finally:
            if new_data:
                filename = new_data.filename
                new_data.delete_data_from_disk()
                assert not os.path.isfile(filename)

            if lightcurves:
                for lc in lightcurves:
                    session.delete(lc)
                session.commit()


def test_reducer_magnitude_conversions(test_project, new_source):
    pass
    # TODO: make sure all conversions of flux to magnitude are correct
    #  use explicit values and check them online with a magnitude calculator
    #  make sure the statistical errors are correct using a large number of points
    #  make sure the flux_min/max are correct


def test_filter_mapping(raw_phot, test_hash):

    # make a demo observatory with a string filtmap:
    obs = VirtualDemoObs(project="test", filtmap="<observatory>-<filter>")
    obs.test_hash = test_hash
    obs.pars.save_reduced = False  # do not save automatically

    # check parameter is propagated correctly
    assert obs.pars.filtmap is not None

    N1 = len(raw_phot.data) // 2
    N2 = len(raw_phot.data)

    raw_phot.data.loc[0:N1, "filter"] = "g"
    raw_phot.data.loc[N1:N2, "filter"] = "r"
    raw_phot.observatory = obs.name

    lcs = obs.reduce(raw_phot)
    assert len(lcs) == 2  # two filters

    lc_g = [lc for lc in lcs if lc.filter == "demo-g"][0]
    assert all(filt == "demo-g" for filt in lc_g.data["filter"])

    lc_r = [lc for lc in lcs if lc.filter == "demo-r"][0]
    assert all(filt == "demo-r" for filt in lc_r.data["filter"])

    # now use a dictionary filtmap
    obs.pars.filtmap = dict(r="Demo/R", g="Demo/G")

    lcs = obs.reduce(raw_phot)
    assert len(lcs) == 2  # two filters

    lc_g = [lc for lc in lcs if lc.filter == "Demo/G"][0]
    assert all(filt == "Demo/G" for filt in lc_g.data["filter"])

    lc_r = [lc for lc in lcs if lc.filter == "Demo/R"][0]
    assert all(filt == "Demo/R" for filt in lc_r.data["filter"])


def test_lightcurve_file_is_auto_deleted(saved_phot, lightcurve_factory):
    lc1 = lightcurve_factory()
    lc1.source = saved_phot.source
    lc1.raw_data = saved_phot

    with SmartSession() as session:
        lc1.save()
        session.add(lc1)
        session.add(lc1.source)
        session.commit()

    # with session closed, check file is there
    assert os.path.isfile(lc1.get_fullname())

    with SmartSession() as session:
        session.delete(lc1)
        session.commit()

    # with session closed, check file is gone
    assert not os.path.isfile(lc1.get_fullname())


def test_lightcurve_copy_constructor(saved_phot, lightcurve_factory):
    lc1 = lightcurve_factory()
    lc1.source = saved_phot.source
    lc1.raw_data = saved_phot

    lc1.altdata = {"exptime": 30.0}
    lc2 = Lightcurve(lc1)

    # data should be different but equal dataframes
    assert lc1.data is not lc2.data
    assert lc1.data.equals(lc2.data)

    # same for times and mjds
    assert lc1.times is not lc2.times
    assert np.all(lc1.times == lc2.times)
    assert lc1.mjds is not lc2.mjds
    assert np.all(lc1.mjds == lc2.mjds)

    # check some other attributes
    assert lc1.exp_time == lc2.exp_time
    assert lc1.filter == lc2.filter
    assert lc1.flux_max == lc2.flux_max

    # check the dictionaries are not related:
    assert lc1.altdata is not lc2.altdata
    assert lc1.altdata["exptime"] == lc2.altdata["exptime"]
    lc1.altdata["exptime"] = 100
    assert lc1.altdata["exptime"] != lc2.altdata["exptime"]
    assert lc1.was_processed == lc2.was_processed

    # make sure DB related attributes are not copied
    with SmartSession() as session:
        try:  # cleanup at the end
            lc1.save()
            session.add(lc1)
            session.add(lc1.source)
            session.commit()
            lc3 = Lightcurve(lc1)
            assert lc3.id is None
            assert lc3.filename is None
            assert lc3.filekey is None
            assert lc1.was_processed == lc3.was_processed
        except Exception:
            session.rollback()
            raise
        finally:  # remove lightcurves from DB and disk
            if lc1 in session:
                session.delete(lc1)
                session.commit()


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


@pytest.mark.flaky(max_runs=3)
def test_histogram():

    h = Histogram()
    # make sure the test does not
    # change if we modify the defaults
    h.pars.dtype = "uint32"
    h.pars.score_coords = {
        "snr": (-10, 10, 0.1),
        "dmag": (-3, 3, 0.1),
    }
    h.pars.source_coords = {
        "mag": (15, 21, 0.5),
    }
    h.pars.obs_coords = {
        "exptime": (30.0, 0.3),
        "filt": (),
    }
    h.initialize()

    num_snr = len(np.arange(-10, 10 + 0.1, 0.1))
    num_dmag = len(np.arange(-3, 3 + 0.1, 0.1))
    num_mag = len(np.arange(15, 21 + 0.5, 0.5))
    num_dynamic = 3  # guess the number of values for dynamic axes
    num_bytes = 4  # uint32

    assert h.get_size() == 0
    estimate_bytes = (num_snr + num_dmag) * num_mag * num_dynamic**2 * num_bytes
    assert h.get_size_estimate("bytes") == estimate_bytes

    # add some data with uniform filter
    num_points1 = 10
    df = pd.DataFrame(
        dict(
            mjd=np.linspace(57000, 58000, num_points1),
            snr=np.random.normal(0, 3, num_points1),
            dmag=0,
            exptime=30.0,
            filt="R",
        )
    )

    # make sure data has well defined dmag values
    df.loc[0:4, "dmag"] = 1.3

    # this will fail because the df doesn't have "mag"
    with pytest.raises(ValueError) as err:
        h.add_data(df)

    assert "Could not find data for axis mag" in str(err.value)

    # throwaway class to make a test source
    class FakeSource:
        pass

    source = FakeSource()
    source.id = np.random.randint(0, 1000)
    source.mag = 18
    h.add_data(df, source)
    assert h.data.coords["filt"] == ["R"]
    assert h.data.coords["exptime"] == [30]
    assert h.get_size("bytes") == (num_snr + num_dmag) * num_mag * num_bytes

    assert np.sum(h.data.snr_counts.values) == num_points1
    assert np.sum(h.data.dmag_counts.values) == num_points1

    # check the dmag values we used get summed correctly
    assert (
        h.data.dmag_counts.sel(dmag=0, method="nearest").sum().values == num_points1 - 5
    )
    assert h.data.dmag_counts.sel(dmag=1.3, method="nearest").sum().values == 5

    # add some data with varying filter
    num_points2 = 10
    df = pd.DataFrame(
        dict(
            mjd=np.linspace(57000, 58000, num_points2),
            snr=np.random.normal(0, 3, num_points2),
            dmag=0,
            exptime=30.0,
            filt=np.random.choice(["V", "I"], num_points2),
        )
    )

    h.add_data(df, source)

    assert set(h.data.coords["filt"].values) == {"R", "V", "I"}
    assert h.data.coords["exptime"] == [30]
    assert h.get_size("bytes") == (num_snr + num_dmag) * num_mag * 3 * num_bytes
    assert (
        h.data.dmag_counts.sel(dmag=0, method="nearest").sum().values == num_points2 + 5
    )
    assert (
        h.data.dmag_counts.sel(dmag=0, method="nearest").sum().values == num_points2 + 5
    )

    # check the filters have the correct counts
    assert h.data.dmag_counts.sel(filt="R").sum().values == num_points1
    assert (
        h.data.dmag_counts.sel(filt="V").sum().values == df[df["filt"] == "V"].shape[0]
    )
    assert (
        h.data.dmag_counts.sel(filt="I").sum().values == df[df["filt"] == "I"].shape[0]
    )
    assert h.data.snr_counts.sel(filt="R").sum().values == num_points1
    assert (
        h.data.snr_counts.sel(filt="V").sum().values == df[df["filt"] == "V"].shape[0]
    )
    assert (
        h.data.snr_counts.sel(filt="I").sum().values == df[df["filt"] == "I"].shape[0]
    )

    num_points3 = 100
    df = pd.DataFrame(
        dict(
            mjd=np.linspace(57000, 58000, num_points3),
            snr=np.random.normal(0, 3, num_points3),
            dmag=0,
            exptime=30.0,
            filt=np.random.choice(["R", "V", "I"], num_points3),
        )
    )
    df.loc[0:4, "exptime"] = 20.3
    df.loc[5:, "exptime"] = 39.8

    h.add_data(df, source)
    assert len(h.data.coords["exptime"]) == len(np.arange(20, 39.8, 0.3))
    assert h.data.sel(exptime=20.3, method="nearest").dmag_counts.sum().values == 5
    assert (
        h.data.sel(exptime=39.8, method="nearest").dmag_counts.sum().values
        == num_points3 - 5
    )
    assert (
        h.data.sel(exptime=30.0, method="nearest").dmag_counts.sum().values
        == num_points1 + num_points2
    )

    # check most of the S/N values are in the middle of the distribution
    snr = h.data.snr
    high_snr = h.data.snr_counts.sel(snr=snr[snr > 5]).sum().values
    low_snr = h.data.snr_counts.sel(snr=snr[snr < -5]).sum().values
    mid_snr = h.data.snr_counts.sel(snr=snr[(-5 <= snr) & (snr <= 5)]).sum().values
    assert mid_snr > low_snr * 5
    assert mid_snr > high_snr * 5

    # add some very high and very low values:
    df.loc[3, "snr"] = 100
    df.loc[4, "snr"] = -100
    df.loc[10:19, "dmag"] = 100

    h.add_data(df, source)
    assert h.data.snr.attrs["overflow"] == 1
    assert h.data.snr.attrs["underflow"] == 1
    assert h.data.dmag.attrs["overflow"] == 10
    assert h.data.dmag.attrs["underflow"] == 0

    # a new source with magnitude above range
    source.mag = 25.3
    h.add_data(df, source)
    assert h.data.mag.attrs["overflow"] == num_points3


@pytest.mark.flaky(max_runs=8)
def test_finder(simple_finder, new_source, lightcurve_factory):

    # this lightcurve has no outliers:
    lc = lightcurve_factory()
    new_source.reduced_lightcurves.append(lc)
    simple_finder.process([lc], new_source)
    det = simple_finder.detect([lc], new_source)
    assert len(det) == 0

    # this lightcurve has outliers:
    lc = lightcurve_factory()
    new_source.reduced_lightcurves[0] = lc
    n_sigma = 8
    mean_flux = lc.data.flux.mean()
    std_flux = lc.data.flux.std()
    flare_flux = mean_flux + std_flux * n_sigma
    lc.data.loc[4, "flux"] = flare_flux
    lc = Lightcurve(lc)
    new_source.processed_lightcurves = [lc]
    simple_finder.process([lc], new_source)
    det = simple_finder.detect([lc], new_source)

    assert len(det) == 1
    assert det[0].source_name == lc.source_name
    assert abs(det[0].snr - n_sigma) < 1.0  # more or less n sigma
    assert det[0].peak_time == Time(lc.data.mjd.iloc[4], format="mjd").datetime

    simple_finder.pars.max_det_per_lc = 2

    # check for negative detections:
    lc.data.loc[96, "flux"] = mean_flux - std_flux * n_sigma
    simple_finder.process([lc], new_source)
    det = simple_finder.detect([lc], new_source)

    assert len(det) == 2
    assert det[1].source_name == lc.source_name
    assert abs(det[1].snr + n_sigma) < 1.0  # more or less n sigma
    assert det[1].peak_time == Time(lc.data.mjd.iloc[96], format="mjd").datetime

    # now do not look for negative detections:
    lc.data["detected"] = False  # clear the previous detections
    simple_finder.pars.abs_snr = False
    simple_finder.process([lc], new_source)
    det = simple_finder.detect([lc], new_source)

    assert len(det) == 1
    assert det[0].peak_time == Time(lc.data.mjd.iloc[4], format="mjd").datetime

    # this lightcurve has bad data:
    lc = lightcurve_factory()
    lc.data.loc[4, "flux"] = np.nan
    lc.data.loc[np.arange(10, 20, 1), "flux"] = 5000
    lc.data.loc[np.arange(10, 20, 1), "flag"] = True
    lc.data.loc[50, "flux"] = flare_flux

    new_source.reduced_lightcurves[0] = lc
    lc = Lightcurve(lc)
    new_source.processed_lightcurves = [lc]
    simple_finder.process([lc], new_source)
    simple_finder.process([lc], new_source)
    det = simple_finder.detect([lc], new_source)

    assert len(det) == 1
    assert det[0].source_name == lc.source_name
    assert abs(det[0].snr - n_sigma) < 1.0  # more or less n sigma
    assert det[0].peak_time == Time(lc.data.mjd.iloc[50], format="mjd").datetime

    # this lightcurve has an outlier with five epochs
    lc = lightcurve_factory()
    lc.data.loc[10:14, "flux"] = flare_flux
    new_source.reduced_lightcurves[0] = lc
    lc = Lightcurve(lc)
    new_source.processed_lightcurves = [lc]
    simple_finder.process([lc], new_source)
    simple_finder.process([lc], new_source)
    det = simple_finder.detect([lc], new_source)

    assert len(det) == 1
    assert det[0].source_name == lc.source_name
    assert abs(det[0].snr - n_sigma) < 1.0  # more or less n sigma
    assert lc.data.mjd.iloc[10] < Time(det[0].peak_time).mjd < lc.data.mjd.iloc[15]
    assert np.isclose(Time(det[0].time_start).mjd, lc.data.mjd.iloc[10])
    assert np.isclose(Time(det[0].time_end).mjd, lc.data.mjd.iloc[14])


@pytest.mark.flaky(max_runs=8)
def test_analysis(analysis, new_source, raw_phot, test_hash):
    obs = VirtualDemoObs(project=analysis.pars.project, save_reduced=False)
    obs.test_hash = test_hash
    analysis.pars.save_anything = False
    new_source.raw_photometry.append(raw_phot)

    # there shouldn't be any detections:
    obs.reduce(new_source, "photometry")
    analysis.analyze_sources(new_source)
    assert new_source.properties is not None
    assert len(new_source.reduced_lightcurves) == 3
    assert len(analysis.detections) == 0

    # add a "flare" to the lightcurve:
    lc = new_source.reduced_lightcurves[0]
    n_sigma = 10
    std_flux = lc.data.flux.std()
    flare_flux = std_flux * n_sigma
    lc.data.loc[4, "flux"] += flare_flux

    new_source.reset_analysis()  # get rid of existing results
    analysis.analyze_sources(new_source)

    assert len(analysis.detections) == 1
    det = analysis.detections[0]
    assert det.source_name == lc.source_name
    assert det.snr - n_sigma < 2.0  # no more than the S/N we put in
    assert det.peak_time == Time(lc.data.mjd.iloc[4], format="mjd").datetime
    assert len(new_source.reduced_lightcurves) == 3  # should be 3 filters in raw_phot
    assert len(new_source.processed_lightcurves) == 3
    assert len(new_source.detections) == 1
    assert new_source.properties is not None

    # check that nothing was saved
    with SmartSession() as session:
        lcs = session.scalars(
            sa.select(Lightcurve).where(Lightcurve.source_name == lc.source_name)
        ).all()
        assert len(lcs) == 0
        detections = session.scalars(
            sa.select(Detection).where(Detection.source_name == new_source.name)
        ).all()
        assert len(detections) == 0
        properties = session.scalars(
            sa.select(Properties).where(Properties.source_name == new_source.name)
        ).all()
        assert len(properties) == 0

    try:  # now save everything

        with SmartSession() as session:
            analysis.pars.save_anything = True
            analysis.reset_histograms()
            new_source.reset_analysis()
            assert len(new_source.detections) == 0

            analysis.analyze_sources(new_source)
            assert len(new_source.detections) == 1

            assert new_source.properties is not None
            assert len(new_source.reduced_lightcurves) == 3
            assert len(new_source.processed_lightcurves) == 3

            # check lightcurves
            lcs = session.scalars(
                sa.select(Lightcurve).where(
                    Lightcurve.source_name == new_source.name,
                    Lightcurve.was_processed.is_(True),
                )
            ).all()
            assert len(lcs) == 3

            # check detections
            detections = session.scalars(
                sa.select(Detection).where(Detection.source_name == new_source.name)
            ).all()
            assert len(detections) == 1
            assert detections[0].snr - n_sigma < 2.0  # no more than the S/N we put in

            # lcs = detections[0].processed_photometry
            # assert lcs[0].time_start < lcs[1].time_start < lcs[2].time_start

            # check properties
            properties = session.scalars(
                sa.select(Properties).where(Properties.source_name == new_source.name)
            ).all()
            assert len(properties) == 1
            # # manually set the first lightcurve time_start to be after the others
            # detections[0].processed_photometry[
            #     0
            # ].time_start = datetime.datetime.utcnow()

            session.add(detections[0])
            session.commit()
            # now close the session and start a new one

        # with SmartSession() as session:
        #     detections = session.scalars(
        #         sa.select(Detection).where(Detection.source_name == new_source.name)
        #     ).all()
        #     lcs = detections[0].processed_photometry
        #
        #     assert len(lcs) == 3  # still three
        #     # order should be different (loaded sorted by time_start)
        #     assert lcs[0].time_start < lcs[1].time_start < lcs[2].time_start
        #     assert lcs[1].id > lcs[0].id > lcs[2].id  # last became first

        # check the number of values added to the histogram matches
        num_snr_values = analysis.all_scores.get_sum_scores()
        assert len(new_source.raw_photometry[0].data) == num_snr_values

        num_offset_values = analysis.quality_values.get_sum_scores()
        assert len(new_source.raw_photometry[0].data) == num_offset_values

    finally:  # remove all generated lightcurves and detections etc.
        analysis.remove_all_histogram_files(remove_backup=True)

        try:
            with SmartSession() as session:
                session.merge(new_source)
                session.commit()
                for lc in new_source.reduced_lightcurves:
                    lc.delete_data_from_disk()
                    if lc in session:
                        try:
                            session.delete(lc)
                        except Exception as e:
                            print(f"could not delete lc: {str(e)}")
                for lc in new_source.processed_lightcurves:
                    lc.delete_data_from_disk()
                    # session.add(lc)
                    if lc in session:
                        try:
                            session.delete(lc)
                        except Exception as e:
                            print(f"could not delete lc: {str(e)}")

                session.commit()
        except Exception as e:
            # print(str(e))
            raise e


@pytest.mark.flaky(max_runs=8)
def test_quality_checks(analysis, new_source, raw_phot, test_hash):
    analysis.pars.save_anything = False
    obs = VirtualDemoObs(project=analysis.pars.project, save_reduced=False)
    obs.test_hash = test_hash
    new_source.raw_photometry.append(raw_phot)
    obs.reduce(new_source, "photometry")

    # add a "flare" to the lightcurve:
    assert len(new_source.reduced_lightcurves) == 3
    lc = new_source.reduced_lightcurves[0]
    std_flux = lc.data.flux.std()
    lc.data.loc[8, "flux"] += std_flux * 12
    lc.data["flag"] = False
    lc.colmap["flag"] = "flag"
    lc.data.loc[8, "flag"] = True

    # look for the events, removing bad quality data
    analysis.finder.pars.remove_failed = True
    analysis.analyze_sources(new_source)
    assert len(analysis.detections) == 0

    # now replace the flag with a big offset
    lc.data.loc[8, "flag"] = False
    lc.data.dec = 0.0
    mean_ra = lc.data.ra.mean()
    std_ra = np.nanstd(np.abs(lc.data.ra - mean_ra))
    lc.data.loc[8, "ra"] = mean_ra + 10 * std_ra
    new_source.reset_analysis()
    analysis.analyze_sources(new_source)
    assert len(analysis.detections) == 0

    # now lets keep and flag bad events
    analysis.finder.pars.remove_failed = False
    new_source.reset_analysis()
    analysis.analyze_sources(new_source)
    assert len(analysis.detections) == 1
    det = analysis.detections[0]
    assert det.source_name == lc.source_name
    assert det.snr - 12 < 2.0  # no more than the S/N we put in
    assert det.peak_time == Time(lc.data.mjd.iloc[8], format="mjd").datetime
    assert det.quality_flag == 1
    assert abs(det.quality_values["offset"] - 10) < 3  # approximately 10 sigma offset

    # what happens if the peak has two measurements?
    lc.data["flag"] = False
    lc.data.loc[7, "flux"] += std_flux * 8  # add width to the flare
    lc.data.loc[9, "flux"] += std_flux * 8  # add width to the flare
    lc.data.loc[9, "ra"] = lc.data.loc[8, "ra"]  # the edge of the flare now has offset
    lc.data.loc[8, "ra"] = mean_ra  # the peak of the flare now has no offset

    analysis.detections = []
    new_source.reset_analysis()
    analysis.analyze_sources(new_source)

    assert len(analysis.detections) == 1
    det = analysis.detections[0]
    assert det.snr - 12 < 2.0  # no more than the S/N we put in
    assert det.peak_time == Time(lc.data.mjd.iloc[8], format="mjd").datetime
    assert (
        det.processed_photometry[0].data.loc[9, "qflag"] == 1
    )  # flagged because of offset
    assert (
        det.processed_photometry[0].data.loc[9, "offset"] > 2
    )  # this one has an offset
    assert det.processed_photometry[0].data.loc[8, "qflag"] == 0  # unflagged
    assert det.processed_photometry[0].data.loc[8, "offset"] < 2  # no offset

    assert det.quality_flag == 1  # still flagged, even though the peak is not
    assert abs(det.quality_values["offset"] - 10) < 2  # approximately 10 sigma offset


# TODO: add test for simulation events


def test_column_names_simplify_and_offset():
    assert simplify("Times") == "time"
    assert simplify("M-JD") == "mjd"
    assert simplify("BJD - 2457000, days") == "bjd"

    assert get_time_offset("BJD - 2457000, days") == -2457000


def test_export_skyportal_photometry(test_project, new_source, raw_phot):
    assert isinstance(test_project.demo, VirtualDemoObs)
    new_source.raw_photometry = [raw_phot]
    lightcurves = test_project.demo.reduce(source=new_source, data_type="photometry")

    lc = lightcurves[0]

    filename = "test_skyportal_photometry.h5"
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


def test_on_close_utility():
    a = []
    b = []

    def append_to_list(a, b, clear_a_at_end=False):
        if clear_a_at_end:
            _ = OnClose(lambda: a.clear())
        a.append(1)
        b.append(a[0])

    append_to_list(a, b)
    assert a == [1]
    assert b == [1]

    append_to_list(a, b, clear_a_at_end=True)
    assert a == []
    assert b == [1, 1]


def test_named_list():
    class TempObject:
        pass

    obj1 = TempObject()
    obj1.name = "One"

    obj2 = TempObject()
    obj2.name = "Two"

    nl = NamedList()
    nl.append(obj1)
    nl.append(obj2)

    assert len(nl) == 2
    assert nl[0] == obj1
    assert nl[1] == obj2

    assert nl["One"] == obj1
    assert nl["Two"] == obj2
    assert nl.keys() == ["One", "Two"]

    with pytest.raises(ValueError):
        nl["Three"]

    with pytest.raises(ValueError):
        nl["one"]

    with pytest.raises(IndexError):
        nl[2]

    with pytest.raises(TypeError):
        nl[1.0]

    # now a list that ignores case
    nl = NamedList(ignorecase=True)
    nl.append(obj1)
    nl.append(obj2)

    assert len(nl) == 2
    assert nl[0] == obj1
    assert nl[1] == obj2

    assert nl["one"] == obj1
    assert nl["two"] == obj2
    assert nl.keys() == ["One", "Two"]

    with pytest.raises(ValueError):
        nl["Three"]


def test_unique_list():
    class TempObject:
        pass

    obj1 = TempObject()
    obj1.name = "object one"
    obj1.foo = "foo1"
    obj1.bar = "common bar"

    obj2 = TempObject()
    obj2.name = "object two"
    obj2.foo = "foo2"
    obj2.bar = "common bar"

    # same attributes as obj1, but different object
    obj3 = TempObject()
    obj3.name = "object one"
    obj3.foo = "foo1"
    obj3.bar = "common bar"

    # the default is to use the name attribute
    ul = UniqueList()
    ul.append(obj1)
    ul.append(obj2)
    assert len(ul) == 2
    assert ul[0] == obj1
    assert ul[1] == obj2

    # appending obj3 will remove obj1
    ul.append(obj3)
    assert len(ul) == 2
    assert ul[0] == obj2
    assert ul[1] == obj3

    # check string indexing
    assert ul["object one"] == obj3
    assert ul["object two"] == obj2

    # now try with a different attribute
    ul = UniqueList(comparison_attributes=["foo", "bar"])
    ul.append(obj1)
    ul.append(obj2)
    assert len(ul) == 2
    assert ul[0] == obj1
    assert ul[1] == obj2

    # string indexing in this case returns a list
    assert ul["foo1"] == [obj1]
    assert ul["foo2"] == [obj2]

    # try indexing with a list or tuple
    assert ul[["foo1", "common bar"]] == obj1
    assert ul[["foo2", "common bar"]] == obj2

    # should work without brackets
    assert ul["foo1", "common bar"] == obj1
    assert ul["foo2", "common bar"] == obj2

    # appending obj3 will remove obj1
    ul.append(obj3)
    assert len(ul) == 2
    assert ul[0] == obj2
    assert ul[1] == obj3

    # try a list with three comparison_attributes
    ul = UniqueList(comparison_attributes=["name", "foo", "bar"])
    ul.append(obj1)
    ul.append(obj2)
    assert len(ul) == 2

    # check that array indexing works with two out of three attributes
    assert ul[["object one", "foo1"]] == [obj1]
    assert ul[["object two", "foo2"]] == [obj2]

    # check that we can ignore case
    obj4 = TempObject()
    obj4.name = "Foo"

    obj5 = TempObject()
    obj5.name = "fOO"

    ul = UniqueList(comparison_attributes=["name"], ignorecase=True)
    ul.append(obj4)
    ul.append(obj5)
    assert len(ul) == 1
    assert ul["foo"] == obj5
    assert ul["FOO"] == obj5


def test_circular_buffer_list():
    cbl = CircularBufferList(3)
    cbl.append(1)
    cbl.append(2)
    cbl.append(3)
    assert cbl == [1, 2, 3]
    assert cbl.total == 3
    cbl.append(4)
    assert cbl == [2, 3, 4]
    assert cbl.total == 4
    cbl.extend([5, 6])
    assert cbl == [4, 5, 6]
    assert cbl.total == 6


def test_safe_mkdir():
    # can make a folder inside the data folder
    new_path = os.path.join(src.database.DATA_ROOT, uuid.uuid4().hex)
    assert not os.path.isdir(new_path)

    safe_mkdir(new_path)
    assert os.path.isdir(new_path)

    os.rmdir(new_path)

    # can make a folder under the code root's results folder
    new_path = os.path.join(src.database.CODE_ROOT, "results", uuid.uuid4().hex)
    assert not os.path.isdir(new_path)

    safe_mkdir(new_path)
    assert os.path.isdir(new_path)

    os.rmdir(new_path)

    # can make a folder under the temporary data folder
    new_path = os.path.join(src.database.DATA_TEMP, uuid.uuid4().hex)
    assert not os.path.isdir(new_path)

    safe_mkdir(new_path)
    assert os.path.isdir(new_path)

    os.rmdir(new_path)

    # this does not work anywhere else:
    new_path = os.path.join(src.database.CODE_ROOT, uuid.uuid4().hex)
    assert not os.path.isdir(new_path)
    with pytest.raises(ValueError) as e:
        safe_mkdir(new_path)
    assert "Cannot make a new folder not inside the following folders" in str(e.value)

    # try a relative path
    new_path = os.path.join(src.database.CODE_ROOT, "results", "..", uuid.uuid4().hex)
    assert not os.path.isdir(new_path)
    with pytest.raises(ValueError) as e:
        safe_mkdir(new_path)
    assert "Cannot make a new folder not inside the following folders" in str(e.value)

    new_path = os.path.join(src.database.CODE_ROOT, "result", uuid.uuid4().hex)
    assert not os.path.isdir(new_path)
    with pytest.raises(ValueError) as e:
        safe_mkdir(new_path)
    assert "Cannot make a new folder not inside the following folders" in str(e.value)


def test_smart_session(new_source):

    try:  # make sure to re-state autobegin=True at the end
        # note that with regular sessions you'd need to call .begin()
        with Session() as session:
            # set this just to test when the sessions are closed:
            session.autobegin = False
            # now we need to add this at start of each session:
            session.begin()

            session.add(new_source)
            session.commit()

        assert new_source.id is not None

        assert session._transaction is None
        # this session has been closed, so this should fail
        with pytest.raises(InvalidRequestError):
            session.scalars(sa.select(Source).where(Source.id == new_source.id)).all()

        # try using a SmartSession, which should also begin the session:
        with SmartSession() as session:
            session.begin()
            # this should work
            sources = session.scalars(
                sa.select(Source).where(Source.id == new_source.id)
            ).all()
            assert any([s.id == new_source.id for s in sources])

        assert session._transaction is None
        # this session has been closed, so this should fail
        with pytest.raises(InvalidRequestError):
            session.scalars(sa.select(Source).where(Source.id == new_source.id)).all()

        # try using a SmartSession without a context manager inside a function
        def try_smart_session(session=None):
            with SmartSession(session) as session:
                if session._transaction is None:
                    session.begin()
                sources = session.scalars(
                    sa.select(Source).where(Source.id == new_source.id)
                ).all()
                assert len(sources) > 0

        try_smart_session()  # the function is like a context manager

        assert session._transaction is None
        # this session has been closed, so this should fail
        with pytest.raises(InvalidRequestError):
            session.scalars(sa.select(Source).where(Source.id == new_source.id)).all()

        # try calling the function again, but surrounded by a context manager
        with SmartSession() as session:
            try_smart_session(session)

            # session should still work even though function has finished
            sources = session.scalars(
                sa.select(Source).where(Source.id == new_source.id)
            ).all()
            assert len(sources) > 0

        assert session._transaction is None
        # this session has been closed, so this should fail
        with pytest.raises(InvalidRequestError):
            session.scalars(sa.select(Source).where(Source.id == new_source.id)).all()

        # with an explicit False this should be a no-op session
        with SmartSession(False) as session:
            assert isinstance(session, NoOpSession)

            query = session.scalars(sa.select(Source).where(Source.id == new_source.id))
            assert isinstance(query, NullQueryResults)
            sources = query.all()
            assert sources == []

            query = session.scalars(sa.select(Source).where(Source.id == new_source.id))
            assert isinstance(query, NullQueryResults)
            source = query.first()
            assert source is None

        # try opening a session inside an open session:
        with SmartSession() as session:
            session.begin()
            with SmartSession(session) as session2:
                assert session2 is session
                sources = session2.scalars(
                    sa.select(Source).where(Source.id == new_source.id)
                ).all()
                assert len(sources) > 0

            # this still works because internal session doesn't auto-close
            sources = session2.scalars(
                sa.select(Source).where(Source.id == new_source.id)
            ).all()
            assert len(sources) > 0

        assert session._transaction is None
        # this should fail because the external session is closed
        with pytest.raises(InvalidRequestError):
            session.scalars(sa.select(Source).where(Source.id == new_source.id)).all()

        # now change the global scope
        import src.database

        try:  # make sure we don't leave the global scope changed
            src.database.NO_DB_SESSION = True

            with SmartSession() as session:
                assert isinstance(session, NoOpSession)
                query = session.scalars(
                    sa.select(Source).where(Source.id == new_source.id)
                )
                assert isinstance(query, NullQueryResults)
                sources = query.all()
                assert sources == []

                query = session.scalars(
                    sa.select(Source).where(Source.id == new_source.id)
                )
                assert isinstance(query, NullQueryResults)
                source = query.first()
                assert source is None

        finally:
            src.database.NO_DB_SESSION = False

    finally:
        # make sure to re-state autobegin=True at the end
        with Session() as session:
            session.autobegin = True
