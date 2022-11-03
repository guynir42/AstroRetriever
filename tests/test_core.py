import os
import yaml
import time
import uuid
import pytest
import datetime
from pprint import pprint

import numpy as np
import pandas as pd
from astropy.time import Time

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from src.parameters import Parameters
from src.project import Project
from src.observatory import VirtualDemoObs
from src.ztf import VirtualZTF

from src.database import Session
from src.source import Source, DEFAULT_PROJECT
from src.dataset import RawPhotometry, Lightcurve, PHOT_ZP
from src.observatory import VirtualDemoObs
from src.catalog import Catalog
from src.detection import Detection
from src.properties import Properties
from src.histogram import Histogram


def test_load_save_parameters(data_dir):

    filename = "parameters_test.yaml"
    filename = os.path.abspath(os.path.join(data_dir, filename))

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
    assert "demo" in [obs.name for obs in proj.observatories]
    assert isinstance(proj.observatories[0], VirtualDemoObs)


def test_project_user_inputs():

    proj = Project(
        name="default_test",
        obs_names=["demo", "ZTF"],
        analysis_kwargs={"num_injections": 3},
        obs_kwargs={
            "reducer": {"reducer_key": "reducer_value"},
            "ZTF": {"credentials": {"username": "guy", "password": "12345"}},
            "DEMO": {"reducer": {"reducer_key": "reducer_value2"}},
        },
        catalog_kwargs={"default": "test"},
    )

    # check the project parameters are loaded correctly
    assert set(proj.pars.obs_names) == {"DEMO", "ZTF"}
    assert proj.catalog.pars.filename == "test.csv"

    # check the observatory was loaded correctly
    assert "ztf" in [obs.name for obs in proj.observatories]
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
    assert proj.observatories["ztf"]._credentials["username"] == "guy"
    assert proj.observatories["ztf"]._credentials["password"] == "12345"

    # check the reducer was overriden in the demo observatory
    assert proj.observatories["ztf"].pars.reducer["reducer_key"] == "reducer_value"
    assert proj.observatories["demo"].pars.reducer["reducer_key"] == "reducer_value2"


def test_project_config_file(data_dir):
    project_str1 = str(uuid.uuid4())
    project_str2 = str(uuid.uuid4())

    data = {
        "project": {  # project wide definitions
            "description": project_str1,  # random string
            "obs_names": ["demo", "ztf"],  # list of observatory names
        },
        "observatories": {  # general instructions to pass to observatories
            "reducer": {  # should be overriden by observatory reducer
                "reducer_key": "project_reduction",
            },
            "demo": {  # demo observatory specific definitions
                "demo_boolean": False,
                "demo_string": "test-string",
            },
            "ztf": {
                "credentials": {
                    "filename": os.path.abspath(
                        os.path.join(data_dir, "passwords_test.yaml")
                    ),
                },
                "reducer": {
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
    configs_folder = os.path.abspath(os.path.join(data_dir, "../configs"))

    if not os.path.isdir(configs_folder):
        os.mkdir(configs_folder)
    filename = os.path.join(configs_folder, "default_test.yaml")
    with open(filename, "w") as file:
        yaml.dump(data, file, sort_keys=False)
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
        assert proj.observatories["demo"].pars.demo_boolean is False
        # new parameter is successfully added
        assert proj.observatories["demo"].pars.demo_string == "test-string"
        # general project-wide reducer is used by demo observatory:
        assert (
            proj.observatories["demo"].pars.reducer["reducer_key"]
            == "project_reduction"
        )

        # check the ZTF calibration/analysis got their own parameters loaded
        assert "ztf" in proj.observatories
        assert isinstance(proj.observatories["ztf"], VirtualZTF)
        assert proj.observatories["ztf"].pars.reducer == {
            "reducer_key": "ztf_reduction"
        }

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
        assert proj.observatories["demo"].pars.demo_string == "new-test-string"

    finally:
        os.remove(filename)
        os.remove(data["observatories"]["ztf"]["credentials"]["filename"])


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
        proj.save_config()
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

    # magnitude hovers around the limit of ~20
    assert abs(cat.data[cat.pars.mag_column].mean() - 20) < 0.1


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
    source = obs.check_and_fetch_source(cat_row, save=False)
    data = source.raw_photometry[0]
    data.invent_filename(ra_deg=cat_row["ra"])

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
    source = obs.check_and_fetch_source(cat_row, save=False)
    data = source.raw_photometry[0]
    data.invent_filename(ra_deg=cat_row["ra"])

    assert (
        data.filename
        == f'RA{int(cat_row["ra"]):02d}/DEMO_photometry_{cat_row["name"]}.h5'
    )

    # test the key conventions:
    data.invent_filekey(source_name=name)
    assert data.filekey == f"{data.type}_{name}"


def test_add_source_and_data(data_dir):
    fullname = ""
    try:  # at end, delete the temp file

        with Session() as session:
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
            )

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
            )

            # check the times make sense
            start_time = Time(min(df.mjd), format="mjd").datetime
            end_time = Time(max(df.mjd), format="mjd").datetime
            assert start_time == new_data.time_start
            assert end_time == new_data.time_end

            new_source.raw_photometry.append(new_data)
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
            new_data.save()  # must save to allow RawPhotometry to be added to DB
            session.commit()  # this should now work fine
            assert new_source.id is not None
            assert new_source.id == new_data.sources[0].id

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
        with Session() as session:
            sources = session.scalars(
                sa.select(Source).where(Source.name == source_name)
            ).first()

            assert sources is not None
            assert len(sources.raw_photometry) == 1
            assert sources.raw_photometry[0].filename == filename
            assert sources.raw_photometry[0].filekey == new_data.filekey
            assert sources.raw_photometry[0].sources[0].id == new_source.id
            # this autoloads the data:
            assert sources.raw_photometry[0].data.equals(df)

    finally:
        if os.path.isfile(fullname):
            os.remove(fullname)
    with pytest.raises(FileNotFoundError):
        with open(fullname) as file:
            pass

    # make sure loading this data does not work without file
    with Session() as session:
        source = session.scalars(
            sa.select(Source).where(Source.name == source_name)
        ).first()
        assert source is not None
        assert len(source.raw_photometry) == 1
        with pytest.raises(FileNotFoundError):
            source.raw_photometry[0].data.equals(df)

    # make sure deleting the source also cleans up the data
    with Session() as session:
        session.execute(sa.delete(Source).where(Source.name == source_name))
        session.commit()
        data = session.scalars(
            sa.select(RawPhotometry).where(RawPhotometry.filekey == new_data.filekey)
        ).first()
        assert not any([s.name == source_name for s in data.sources])


def test_source_unique_constraint():

    with Session() as session:
        name1 = str(uuid.uuid4())
        source1 = Source(name=name1, ra=0, dec=0)
        assert source1.cfg_hash == ""  # the default has is an empty string
        session.add(source1)

        source2 = Source(name=name1, ra=0, dec=0)
        assert source1.cfg_hash == ""  # the default has is an empty string
        session.add(source2)

        # should fail as both sources have the same name
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

        name2 = str(uuid.uuid4())
        source2 = Source(name=name2, ra=0, dec=0)
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

    with Session() as session:
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
        raw_phot_no_exptime.observatory = str(uuid.uuid4())
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


def test_raw_photometry_relationships(new_source, new_source2, raw_phot):

    with Session() as session:
        new_source.raw_photometry = [raw_phot]
        new_source2.raw_photometry = [raw_phot]
        session.add(raw_phot)
        raw_phot.save()
        session.commit()

        ids = [new_source.id, new_source2.id]
        names = [new_source.name, new_source2.name]

        # check the linking is ok
        assert new_source.id is not None
        assert new_source2.id is not None

        # check all sources are linked to raw_phot
        assert all([s.id in ids for s in raw_phot.sources])
        assert all([s.name in names for s in raw_phot.sources])

        # TODO: make some reduced lightcurves:

        # TODO: test processed lightcurves show up as well


def test_data_reduction(test_project, new_source, raw_phot_no_exptime):

    with Session() as session:

        # add the data to a database mapped object
        source_id = new_source.id
        new_source.project = test_project.name
        raw_phot_no_exptime.save(overwrite=True)
        new_source.raw_photometry.append(raw_phot_no_exptime)

        # reduce the data using the demo observatory
        assert len(test_project.observatories) == 1
        obs_key = list(test_project.observatories.keys())[0]
        assert obs_key == "demo"
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
        with pytest.raises(ValueError) as exc:
            session.commit()
        assert "No filename" in str(exc.value)
        session.rollback()

        # must save dataset before adding it to DB
        [lc.save(overwrite=True) for lc in lightcurves]
        filenames = [lc.get_fullname() for lc in lightcurves]

        session.add(new_source)
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
            assert dff["mag_err"].equals(lc.data["mag_err"])

            # make sure the number of points are correct
            assert lc.number == len(dff)
            assert lc.shape == (len(dff), len(lc.colmap))

            # make sure the frame rate and exposure time are correct
            assert lc.exp_time == 30.0
            assert np.isclose(
                1.0 / lc.frame_rate, dff["mjd"].diff().median() * 24 * 3600
            )
            assert not lc.is_uniformly_sampled

            # make sure the average flux is correct
            flux = 10 ** (-0.4 * (dff["mag"].values - PHOT_ZP))
            assert np.isclose(lc.flux_mean, np.mean(flux))

            # make sure flux min/max are correct
            assert np.isclose(lc.flux_min, np.min(flux))
            assert np.isclose(lc.flux_max, np.max(flux))

            # make sure superfluous columns are dropped
            assert "oid" not in lc.data.columns

            # make sure the start/end times are correct
            assert np.isclose(Time(lc.time_start).mjd, dff["mjd"].min())
            assert np.isclose(Time(lc.time_end).mjd, dff["mjd"].max())

            # make sure relationships are correct
            assert lc.source_id == new_source.id
            assert lc.raw_data_id == raw_phot_no_exptime.id

        # make sure deleting the source also cleans up the data
        # session.execute(sa.delete(Source).where(Source.name == source_id))
        session.delete(new_source)
        session.commit()

        data = session.scalars(
            sa.select(RawPhotometry).where(
                RawPhotometry.filekey == raw_phot_no_exptime.filekey
            )
        ).first()
        assert data is None
        data = session.scalars(
            sa.select(Lightcurve).where(Lightcurve.source_id == source_id)
        ).all()
        assert len(data) == 0
        assert not any([os.path.isfile(f) for f in filenames])


def test_filter_mapping(raw_phot):

    # make a demo observatory with a string filtmap:
    obs = VirtualDemoObs(project="test", filtmap="<observatory>-<filter>")

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
    [print(lc.data._is_view) for lc in lcs]
    [print(lc.data._is_copy) for lc in lcs]
    assert len(lcs) == 2  # two filters

    lc_g = [lc for lc in lcs if lc.filter == "Demo/G"][0]
    assert all(filt == "Demo/G" for filt in lc_g.data["filter"])

    lc_r = [lc for lc in lcs if lc.filter == "Demo/R"][0]
    assert all(filt == "Demo/R" for filt in lc_r.data["filter"])


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


def test_reduced_data_file_keys(test_project, new_source, raw_phot):

    obs = test_project.observatories["demo"]
    raw_phot.altdata["exptime"] = 30.0
    new_source.raw_photometry.append(raw_phot)
    lcs = obs.reduce(source=new_source, data_type="photometry")

    try:  # at end, delete the temp file
        raw_phot.save(overwrite=True)
        basename = os.path.splitext(raw_phot.filename)[0]

        lcs = obs.reduce(source=new_source, data_type="photometry")

        for lc in lcs:
            lc.save(overwrite=True)
            assert basename in lc.filename

        # make sure all filenames are the same
        assert lcs[0].filename == list({lc.filename for lc in lcs})[0]

        # check the all the data exists in the file
        with pd.HDFStore(lcs[0].get_fullname()) as store:
            for lc in lcs:
                assert os.path.join("/", lc.filekey) in store.keys()
                assert len(store[lc.filekey]) == len(lc.data)

    finally:
        raw_phot.delete_data_from_disk()
        assert not os.path.isfile(raw_phot.get_fullname())


def test_reducer_with_outliers(test_project, new_source):
    num_points = 20
    outlier_indices = [5, 8, 12]
    flagged_indices = [5, 10, 15]
    new_data = None
    lightcurves = None

    with Session() as session:
        try:  # at end, delete the temp file
            filt = "R"
            mjd = np.linspace(57000, 58000, num_points)
            mag_err = np.random.uniform(0.09, 0.11, num_points)
            mag = np.random.normal(18, 0.1, num_points)
            mag[outlier_indices] = np.random.normal(10, 0.1, len(outlier_indices))
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
            )
            new_data.save()
            new_source.raw_photometry.append(new_data)

            # reduce the data use the demo observatory
            assert len(test_project.observatories) == 1
            obs_key = list(test_project.observatories.keys())[0]
            obs = test_project.observatories[obs_key]  # key should be "demo"
            assert isinstance(obs, VirtualDemoObs)

            obs.pars.reducer["drop_flagged"] = False
            lightcurves = obs.reduce(source=new_source, data_type="photometry")
            new_source.lightcurves = lightcurves

            assert len(lightcurves) == 1
            lc = lightcurves[0]
            lc.save()

            session.add(new_source)
            session.commit()

            # check the data has been reduced as expected
            df2 = df[~df["flag"]]
            drop_idx = list(set(outlier_indices + flagged_indices))
            df3 = df.drop(drop_idx, axis=0)
            assert np.isclose(lc.mag_min, df2["mag"].min())
            assert np.isclose(lc.mag_max, df2["mag"].max())
            assert lc.num_good == num_points - len(flagged_indices)
            assert abs(np.mean(df3["mag"]) - lc.mag_mean_robust) < 0.1
            assert abs(np.std(df2["mag"]) - lc.mag_rms) < 0.5
            assert abs(np.std(df3["mag"]) - lc.mag_rms_robust) < 0.1

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


def test_lightcurve_file_is_auto_deleted(saved_phot, lightcurve_factory):
    lc1 = lightcurve_factory()
    lc1.source = saved_phot.sources[0]
    lc1.raw_data = saved_phot

    with Session() as session:
        lc1.save()
        session.add(lc1)
        session.flush()
        session.commit()

    # with session closed, check file is there
    assert os.path.isfile(lc1.get_fullname())

    with Session() as session:
        session.delete(lc1)
        session.commit()

    # with session closed, check file is gone
    assert not os.path.isfile(lc1.get_fullname())


def test_lightcurve_copy_constructor(saved_phot, lightcurve_factory):
    lc1 = lightcurve_factory()
    lc1.source = saved_phot.sources[0]
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
    with Session() as session:
        try:  # cleanup at the end
            lc1.save()
            session.add(lc1)
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


@pytest.mark.flaky(reruns=3)
def test_demo_observatory_download_time(test_project):
    test_project.catalog.make_test_catalog()
    test_project.catalog.load()
    obs = test_project.observatories["demo"]

    t0 = time.time()
    obs.pars.num_threads_download = 0  # no multithreading
    obs.download_all_sources(0, 10, save=False, fetch_args={"wait_time": 1})
    assert len(obs.sources) == 10
    assert len(obs.datasets) == 10
    single_tread_time = time.time() - t0
    assert abs(single_tread_time - 10) < 2  # should take about 10s

    t0 = time.time()
    obs.sources = []
    obs.datasets = []
    obs.pars.num_threads_download = 5  # five multithreading cores
    obs.download_all_sources(0, 10, save=False, fetch_args={"wait_time": 5})
    assert len(obs.sources) == 10
    assert len(obs.datasets) == 10
    multitread_time = time.time() - t0
    assert abs(multitread_time - 10) < 2  # should take about 10s


def test_demo_observatory_save_downloaded(test_project):
    # make random sources unique to this test
    test_project.catalog.make_test_catalog()
    test_project.catalog.load()
    obs = test_project.observatories["demo"]
    try:
        obs.download_all_sources(0, 10, save=True, fetch_args={"wait_time": 0})

        # reloading these sources should be quick (no call to fetch should be sent)
        t0 = time.time()
        obs.download_all_sources(0, 10, fetch_args={"wait_time": 10})
        reload_time = time.time() - t0
        assert reload_time < 1  # should take less than 1s

    finally:
        pass
        # for d in obs.datasets:
        #     d.delete_data_from_disk()
        #
        # assert not os.path.isfile(obs.datasets[0].get_fullname())
        #
        # with Session() as session:
        #     for d in obs.datasets:
        #         session.delete(d)
        #     session.commit()


@pytest.mark.flaky(reruns=3)
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
    assert (
        h.get_size_estimate("bytes")
        == (num_snr + num_dmag) * num_mag * num_dynamic**2 * num_bytes
    )

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

    # add some data with well defined dmag values
    df.loc[0:4, "dmag"] = 1.3

    with pytest.raises(ValueError) as err:
        h.add_data(df)

    assert "Could not find data for axis mag" in str(err.value)

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

    # a new source with above range magnitude
    source.mag = 25.3
    h.add_data(df, source)
    assert h.data.mag.attrs["overflow"] == num_points3


@pytest.mark.flaky(reruns=5)
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
    assert det[0].source_id == lc.source_id
    assert abs(det[0].snr - n_sigma) < 1.0  # more or less n sigma
    assert det[0].peak_time == Time(lc.data.mjd.iloc[4], format="mjd").datetime

    simple_finder.pars.max_det_per_lc = 2

    # check for negative detections:
    lc.data.loc[96, "flux"] = mean_flux - std_flux * n_sigma
    simple_finder.process([lc], new_source)
    det = simple_finder.detect([lc], new_source)

    assert len(det) == 2
    assert det[1].source_id == lc.source_id
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
    assert det[0].source_id == lc.source_id
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
    assert det[0].source_id == lc.source_id
    assert abs(det[0].snr - n_sigma) < 1.0  # more or less n sigma
    assert lc.data.mjd.iloc[10] < Time(det[0].peak_time).mjd < lc.data.mjd.iloc[15]
    assert np.isclose(Time(det[0].time_start).mjd, lc.data.mjd.iloc[10])
    assert np.isclose(Time(det[0].time_end).mjd, lc.data.mjd.iloc[14])


@pytest.mark.flaky(reruns=3)
def test_analysis(analysis, new_source, raw_phot):
    analysis.pars.save_anything = False
    obs = VirtualDemoObs(project=analysis.pars.project)
    new_source.raw_photometry.append(raw_phot)

    # there shouldn't be any detections:
    obs.reduce(new_source, "photometry")
    analysis.analyze_sources(new_source)
    assert new_source.properties is not None
    assert len(analysis.detections) == 0

    # add a "flare" to the lightcurve:
    assert len(new_source.reduced_lightcurves) == 3
    lc = new_source.reduced_lightcurves[0]
    new_source.reset_analysis()
    n_sigma = 10
    std_flux = lc.data.flux.std()
    flare_flux = std_flux * n_sigma
    lc.data.loc[4, "flux"] += flare_flux
    analysis.analyze_sources(new_source)

    assert len(analysis.detections) == 1
    assert analysis.detections[0].source_name == lc.source_name
    assert analysis.detections[0].snr - n_sigma < 2.0  # no more than the S/N we put in
    assert (
        analysis.detections[0].peak_time
        == Time(lc.data.mjd.iloc[4], format="mjd").datetime
    )
    assert len(new_source.reduced_lightcurves) == 3  # should be 3 filters in raw_phot
    assert len(new_source.processed_lightcurves) == 0  # should be empty if not saving
    assert len(new_source.detections) == 0  # should be empty if not saving

    # check that nothing was saved
    with Session() as session:
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
        analysis.pars.save_anything = True
        new_source.reset_analysis()
        # make sure to save the raw/reduced data first
        raw_phot.save()
        for lc in new_source.reduced_lightcurves:
            lc.save()

        analysis.analyze_sources(new_source)
        assert len(new_source.detections) == 1
        assert new_source.properties is not None
        assert len(new_source.reduced_lightcurves) == 3
        assert len(new_source.processed_lightcurves) == 3

        with Session() as session:
            # check lightcurves
            lcs = session.scalars(
                sa.select(Lightcurve).where(
                    Lightcurve.source_name == new_source.name,
                    Lightcurve.was_processed.is_(False),
                )
            ).all()
            assert len(lcs) == 3
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

            lcs = detections[0].reduced_photometry
            assert lcs[0].time_start < lcs[1].time_start < lcs[2].time_start

            # check properties
            properties = session.scalars(
                sa.select(Properties).where(Properties.source_name == new_source.name)
            ).all()
            assert len(properties) == 1

            # manually set the first lightcurve time_start to be after the others
            detections[0].reduced_photometry[0].time_start = datetime.datetime.utcnow()

            session.add(detections[0])
            session.commit()
            # now close the session and start a new one

        with Session() as session:
            detections = session.scalars(
                sa.select(Detection).where(Detection.source_name == new_source.name)
            ).all()
            lcs = detections[0].reduced_photometry

            assert len(lcs) == 3  # still three
            # order should be different (loaded sorted by time_start)
            assert lcs[0].time_start < lcs[1].time_start < lcs[2].time_start
            assert lcs[1].id > lcs[0].id > lcs[2].id  # last became first

    finally:  # remove all generated lightcurves and detections etc.
        with Session() as session:
            for lc in new_source.reduced_lightcurves:
                lc.delete_data_from_disk()
                session.add(lc)
                session.delete(lc)
            for lc in new_source.processed_lightcurves:
                lc.delete_data_from_disk()
                session.add(lc)
                session.delete(lc)

            session.commit()


@pytest.mark.flaky(reruns=3)
def test_quality_checks(analysis, new_source, raw_phot):
    analysis.pars.save_anything = False
    obs = VirtualDemoObs(project=analysis.pars.project)
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
    std_ra = np.std(np.abs(lc.data.ra - mean_ra))
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
    assert abs(det.quality_values["offset"] - 10) < 2  # approximately 10 sigma offset

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
        det.reduced_photometry[0].data.loc[9, "qflag"] == 1
    )  # flagged because of offset
    assert det.reduced_photometry[0].data.loc[9, "offset"] > 2  # this one has an offset
    assert det.reduced_photometry[0].data.loc[8, "qflag"] == 0  # unflagged
    assert det.reduced_photometry[0].data.loc[8, "offset"] < 2  # no offset

    assert det.quality_flag == 1  # still flagged, even though the peak is not
    assert abs(det.quality_values["offset"] - 10) < 2  # approximately 10 sigma offset


# TODO: add test for simulation events
