import os
import uuid
import pytest

import numpy as np
import pandas as pd
from astropy.time import Time

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from src.database import SmartSession
from src.dataset import RawPhotometry, Lightcurve, PHOT_ZP
from src.observatory import VirtualDemoObs
from src.utils import random_string


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
