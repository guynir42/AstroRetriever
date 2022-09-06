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
from src.dataset import RawData, Lightcurve, PHOT_ZP
from src.observatory import VirtualDemoObs
from src.catalog import Catalog
from src.histogram import Histogram

basepath = os.path.abspath(os.path.dirname(__file__))
src.dataset.DATA_ROOT = basepath


def test_load_save_parameters():

    filename = "parameters_test.yaml"
    filename = os.path.abspath(os.path.join(basepath, filename))

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
            "_default_keys",
            "verbose",
        } == set(new_data.keys())
        assert new_data["username"] == "guy"
        assert new_data["password"] == "12345"
        assert new_data["extra_parameter"] == "test"

    finally:
        # cleanup the test file
        os.remove(filename)


def test_default_project():
    proj = Project("default_test")
    assert proj.pars.obs_names == {"demo"}
    assert "demo" in [obs.name for obs in proj.observatories]
    assert isinstance(proj.observatories[0], VirtualDemoObs)


def test_project_user_inputs():

    project_str = str(uuid.uuid4())
    proj = Project(
        name="default_test",
        project_string=project_str,
        obs_names=["ZTF"],
        reducer={"reducer_key": "reducer_value"},
        analysis_kwargs={"analysis_key": "analysis_value"},
        obs_kwargs={},
        ZTF={"credentials": {"username": "guy", "password": "12345"}},
    )

    # check the project parameters are loaded correctly
    assert proj.pars.project_string == project_str
    assert proj.pars.obs_names == {"ZTF"}
    assert proj.catalog.pars.filename == "test.csv"

    # check the observatory was loaded correctly
    assert "ztf" in [obs.name for obs in proj.observatories]
    assert isinstance(proj.observatories[0], VirtualZTF)
    # check that observatories can be referenced using (case-insensitive) strings
    assert isinstance(proj.observatories["ztf"], VirtualZTF)
    assert isinstance(proj.observatories["ZTF"], VirtualZTF)
    assert isinstance(proj.observatories[0]._credentials, dict)
    assert proj.observatories[0]._credentials["username"] == "guy"
    assert proj.observatories[0]._credentials["password"] == "12345"


def test_project_config_file():
    project_str1 = str(uuid.uuid4())
    project_str2 = str(uuid.uuid4())

    data = {
        "project": {  # project wide definitions
            "project_string": project_str1,  # random string
            "obs_kwargs": {  # general instructions to pass to observatories
                "reducer": {  # should be overriden by observatory reducer
                    "reducer_key": "project_reduction",
                },
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
            "reducer": {
                "reducer_key": "ztf_reduction",
            },
        },
        "analysis": {
            "analysis_key": "project_analysis",
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
        proj = Project("default_test", cfg_file=None)
        assert not hasattr(proj.pars, "project_string")

        # load the default config file at configs/default_test.yaml
        proj = Project(
            "default_test",
            obs_names=["DemoObs", "ZTF"],
        )
        assert hasattr(proj.pars, "project_string")
        assert proj.pars.project_string == project_str1
        assert proj.analysis.pars.analysis_key == "project_analysis"

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
            project_string=project_str2,
            demo={
                "demo_string": "new-test-string"
            },  # directly override demo parameters
        )
        assert proj.pars.project_string == project_str2
        assert proj.observatories["demo"].pars.demo_string == "new-test-string"

    finally:
        os.remove(filename)
        os.remove(data["ztf"]["credentials"]["filename"])


def test_catalog():
    filename = "test_catalog.csv"
    fullname = os.path.abspath(os.path.join(basepath, "../catalogs", filename))

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


def test_catalog_hdf5():
    filename = "test_catalog.h5"
    fullname = os.path.abspath(os.path.join(basepath, "../catalogs", filename))

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
    assert cat.data["ra"].dtype == np.dtype(">f8")
    assert cat.data["dec"].dtype == np.dtype(">f8")
    assert cat.pars.name_column in cat.data.dtype.names

    # more than a million sources
    assert len(cat.data) > 1000_000

    # catalog should be ordered by RA, so first objects are close to 360/0
    assert cat.data["ra"][0] > 359 or cat.data["ra"][0] < 1

    # average values are weighted by the galactic bulge
    assert abs(cat.data["ra"].mean() - 229) < 1
    assert abs(cat.data["dec"].mean() + 20) < 1

    # magnitude hovers around the limit of ~20
    assert abs(cat.data[cat.pars.mag_column].mean() - 20) < 0.1


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


def test_data_reduction(test_project, new_source, raw_photometry_no_exptime):

    lightcurves = None

    with Session() as session:
        try:  # at end, delete the temp file
            # add the data to a database mapped object
            source_id = new_source.id
            new_source.project = test_project.name
            raw_photometry_no_exptime.save()
            new_source.raw_data.append(raw_photometry_no_exptime)

            # reduce the data use the demo observatory
            assert len(test_project.observatories) == 1
            obs_key = list(test_project.observatories.keys())[0]
            obs = test_project.observatories[obs_key]  # key should be "demo"
            assert isinstance(obs, VirtualDemoObs)

            # cannot generate photometric data without an exposure time
            with pytest.raises(ValueError) as exc:
                obs.reduce(raw_photometry_no_exptime, to="lcs", source=new_source)
            assert "No exposure time" in str(exc.value)

            # add exposure time to the dataframe:
            new_source.raw_data[0].data["exp_time"] = 30.0
            lightcurves = obs.reduce(
                raw_photometry_no_exptime, to="lcs", source=new_source
            )
            new_source.lightcurves = lightcurves
            session.add(new_source)

            with pytest.raises(ValueError) as exc:
                session.commit()
            assert "No filename" in str(exc.value)
            session.rollback()

            # must save dataset before adding it to DB
            [lc.save() for lc in lightcurves]
            session.commit()

            # check that the data has been reduced as expected
            for lc in lightcurves:
                filt = lc.filter
                dff = raw_photometry_no_exptime.data[
                    raw_photometry_no_exptime.data["filter"] == filt
                ]
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
                assert lc.raw_data_id == raw_photometry_no_exptime.id

        finally:
            filename = raw_photometry_no_exptime.filename
            raw_photometry_no_exptime.delete_data_from_disk()
            assert not os.path.isfile(filename)

            if lightcurves:
                filenames = [lc.filename for lc in lightcurves]
                [lc.delete_data_from_disk() for lc in lightcurves]
                assert not any([os.path.isfile(f) for f in filenames])

        # make sure deleting the source also cleans up the data
        session.execute(sa.delete(Source).where(Source.name == source_id))
        session.commit()
        data = session.scalars(
            sa.select(RawData).where(RawData.key == raw_photometry_no_exptime.key)
        ).first()
        assert data is None
        data = session.scalars(
            sa.select(Lightcurve).where(Lightcurve.source_id == source_id)
        ).all()
        assert len(data) == 0


def test_filter_mapping(raw_photometry):

    # make a demo observatory with a string filtmap:
    obs = VirtualDemoObs(project="test", filtmap="<observatory>-<filter>")

    # check parameter is propagated correctly
    assert obs.pars.filtmap is not None

    N1 = len(raw_photometry.data) // 2
    N2 = len(raw_photometry.data)

    raw_photometry.data.loc[0:N1, "filter"] = "g"
    raw_photometry.data.loc[N1:N2, "filter"] = "r"
    raw_photometry.observatory = obs.name

    lcs = obs.reduce(raw_photometry, to="lcs")
    assert len(lcs) == 2  # two filters

    lc_g = [lc for lc in lcs if lc.filter == "demo-g"][0]
    assert all(filt == "demo-g" for filt in lc_g.data["filter"])

    lc_r = [lc for lc in lcs if lc.filter == "demo-r"][0]
    assert all(filt == "demo-r" for filt in lc_r.data["filter"])

    # now use a dictionary filtmap
    obs.pars.filtmap = dict(r="Demo/R", g="Demo/G")

    lcs = obs.reduce(raw_photometry, to="lcs")
    assert len(lcs) == 2  # two filters

    lc_g = [lc for lc in lcs if lc.filter == "Demo/G"][0]
    assert all(filt == "Demo/G" for filt in lc_g.data["filter"])

    lc_r = [lc for lc in lcs if lc.filter == "Demo/R"][0]
    assert all(filt == "Demo/R" for filt in lc_r.data["filter"])


def test_data_filenames(raw_photometry):
    try:  # at end, delete the temp files
        raw_photometry.save()
        assert raw_photometry.filename is not None
        assert "photometry" in raw_photometry.filename
        assert raw_photometry.filename.endswith(".h5")

    finally:
        raw_photometry.delete_data_from_disk()
        assert not os.path.isfile(raw_photometry.get_fullname())

    # just a filename does not affect folder
    # default folder is given as 'DATA'
    raw_photometry.folder = None
    raw_photometry.filename = "test.h5"
    assert raw_photometry.folder is None
    assert raw_photometry.filename == "test.h5"
    assert raw_photometry.get_fullname() == os.path.join(basepath, "DATA/test.h5")

    # no folder is given, but has observatory name to use as default
    raw_photometry.observatory = "ztf"
    assert raw_photometry.get_fullname() == os.path.join(basepath, "ZTF/test.h5")

    # give the folder explicitly, will override the default
    raw_photometry.folder = "test"
    assert raw_photometry.get_fullname() == os.path.join(basepath, "test/test.h5")

    # adding a path to filename puts that path into "folder"
    raw_photometry.folder = None
    raw_photometry.filename = "path/to/test/test.h5"
    assert raw_photometry.folder == "path/to/test"
    assert raw_photometry.get_fullname() == os.path.join(
        basepath, "path/to/test/test.h5"
    )

    # an absolute path in "folder" will ignore DATA_ROOT
    raw_photometry.folder = None
    raw_photometry.filename = "/path/to/test/test.h5"
    assert raw_photometry.folder == "/path/to/test"
    assert raw_photometry.get_fullname() == "/path/to/test/test.h5"


def test_reduced_data_file_keys(test_project, new_source, raw_photometry):

    obs = test_project.observatories["demo"]
    raw_photometry.altdata["exptime"] = 30.0
    lcs = obs.reduce(raw_photometry, to="lcs", source=new_source)

    try:  # at end, delete the temp file
        raw_photometry.save()
        basename = os.path.splitext(raw_photometry.filename)[0]

        lcs = obs.reduce(raw_photometry, to="lcs", source=new_source)

        for lc in lcs:
            lc.save()
            assert basename in lc.filename

        # make sure all filenames are the same
        assert lcs[0].filename == list({lc.filename for lc in lcs})[0]

        # check the all the data exists in the file
        with pd.HDFStore(lcs[0].get_fullname()) as store:
            for lc in lcs:
                assert os.path.join("/", lc.key) in store.keys()
                assert len(store[lc.key]) == len(lc.data)

    finally:
        raw_photometry.delete_data_from_disk()
        assert not os.path.isfile(raw_photometry.get_fullname())
        for lc in lcs:
            lc.delete_data_from_disk()
        assert not os.path.isfile(lcs[0].get_fullname())


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
            source_id = new_source.id
            new_source.project = test_project.name
            new_data = RawData(
                data=df, folder="data_temp", altdata=dict(exptime="25.0")
            )
            new_data.save()
            new_source.raw_data.append(new_data)

            # reduce the data use the demo observatory
            assert len(test_project.observatories) == 1
            obs_key = list(test_project.observatories.keys())[0]
            obs = test_project.observatories[obs_key]  # key should be "demo"
            assert isinstance(obs, VirtualDemoObs)

            obs.pars.reducer["drop_flagged"] = False
            lightcurves = obs.reduce(new_data, to="lcs", source=new_source)
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

        finally:
            if new_data:
                filename = new_data.filename
                new_data.delete_data_from_disk()
                assert not os.path.isfile(filename)

            if lightcurves:
                filenames = [lc.filename for lc in lightcurves]
                [lc.delete_data_from_disk() for lc in lightcurves]
                assert not any([os.path.isfile(f) for f in filenames])


def test_reducer_magnitude_conversions(test_project, new_source):
    pass
    # TODO: make sure all conversions of flux to magnitude are correct
    #  use explicit values and check them online with a magnitude calculator
    #  make sure the statistical errors are correct using a large number of points
    #  make sure the flux_min/max are correct


@pytest.mark.flaky(reruns=2)
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


def test_finder():
    pass
