import pytest

import numpy as np
from astropy.time import Time

import sqlalchemy as sa

from src.database import SmartSession

from src.dataset import Lightcurve
from src.observatory import VirtualDemoObs
from src.detection import Detection
from src.properties import Properties


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
