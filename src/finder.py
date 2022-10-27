import numpy as np
import pandas as pd

from src.parameters import Parameters
from src.source import Source
from src.dataset import RawPhotometry, Lightcurve
from src.detection import Detection


class ParsFinder(Parameters):
    def __init__(self, **kwargs):
        super().__init__()  # initialize base Parameters without passing arguments

        self.snr_threshold = self.add_par(
            "snr_threshold", 5, (float, int), "S/N threshold for detection"
        )
        self.snr_threshold_sidebands = self.add_par(
            "snr_threshold_sidebands",
            -2,
            (float, int),
            "S/N threshold for event region",
        )
        self.max_det_per_lc = self.add_par(
            "max_det_per_lc", 1, int, "Maximum number of detections per lightcurve"
        )
        self.abs_snr = self.add_par(
            "abs_snr",
            True,
            bool,
            "Use absolute S/N for detection (i.e., include negative)",
        )

        self.remove_failed = self.add_par(
            "remove_failed",
            False,
            bool,
            "Remove detections that did not pass quality cuts. ",
        )

        self._enforce_type_checks = True
        self._enforce_no_new_attrs = True

        self.load_then_update(kwargs)

    @classmethod
    def get_default_cfg_key(cls):
        """
        Get the default key to use when loading a config file.
        """
        return "finder"


class Finder:
    """
    A basic finder implementation, looks for
    high S/N times in the input lightcurves
    and produces detections.

    Also acts as a base class for more complicated
    finder objects. Accepts various data
    instances and produces Detection objects.

    If accepting a Lightcurve object, it adds
    some scores (like "snr" or "dmag") to the dataframe columns,
    if they don't exist already from the reduction stage.
    If any (or all) of the scores go over the threshold
    it should save a Detection object with the details
    of that detection (the Lightcurve ID, the time, S/N and so on).

    If accepting images, TBD...

    It is important that the Finder state
    does not change when applied to new data,
    since we expect to re-run the same data
    multiple times, before/after injecting simulations.
    The sim parameter is used when the data
    ingested has an injected event in it.
    If sim=None, assume real data,
    if sim=dict, assume simulated data
    (or injected data) where the "truth values"
    are passed in using the dictionary.

    """

    def __init__(self, **kwargs):
        self.pars = ParsFinder(**kwargs)

    def process(self, lightcurves, source, sim=None):
        """
        Process the input lightcurves and add columns
        to the dataframe containing various scores like
        "snr" or "score" that can later be used to make
        detections.


        Parameters
        ----------
        lightcurves: a list of Lightcurve objects
            The lightcurves to process. Must be able
            to either add new columns, or overwrite
            data in existing columns (e.g., when
            re-calculating scores after injecting
            simulations).
        source: Source object
            The source to get lightcurves from.
            May also use properties of the source
            to determine scores (e.g., magnitude).
        sim: dict, optional
            If sim=None, assume real data,
            if sim=dict, assume simulated data
            where the "truth values" are passed
            in using the dictionary.

        """
        for lc in lightcurves:
            # Add some scores to the lightcurve
            noise = self.estimate_flux_noise(lc, source)
            lc.data["snr"] = (lc.data["flux"] - lc.flux_mean_robust) / noise
            lc.data["dmag"] = lc.data["mag"] - lc.data["mag"].median()

            # a column to mark indices in the lightcurve
            # where an event was detected
            if "detected" not in lc.data.columns:
                lc.data["detected"] = False

    def detect(self, lightcurves, source, sim=None):
        """
        Ingest a list of lightcurves and produce
        detections for them.

        Parameters
        ----------
        lightcurves: list of Lightcurve objects
            The lightcurves to process. Must have already
            had been processed (e.g., should have a column
            in the dataframe called "snr").
        source : Source object
            The source from which the lightcurves were taken.
            May also use properties of the source
            to determine detections (e.g., magnitude).
        sim : dict or None
            The truth values for the injected event
            if this is a simulated lightcurve.
            If None, assume real data.

        Returns
        -------
        detections : list of Detection objects
            The detections produced by this finder.

        """

        detections = []  # List of detections to return

        for lc in lightcurves:
            # Find the detections (just look for high S/N)
            for i in range(self.pars.max_det_per_lc):
                snr = lc.data["snr"].values
                if self.pars.abs_snr:
                    snr = np.abs(snr)
                mask = lc.data["detected"].values
                mask |= np.isnan(snr)
                if "flag" in lc.data.columns:
                    mask |= lc.data[lc.colmap["flag"]].values > 0
                if "qflag" in lc.data.columns:
                    mask |= lc.data["qflag"].values > 0

                snr[mask] = 0

                idx = np.argmax(snr)
                mx = snr[idx]
                if mx > self.pars.snr_threshold:
                    # Create a detection object
                    detections.append(self.make_detection(idx, lc, source, sim))
                else:
                    break

        return detections

    def estimate_flux_noise(self, lightcurve, source=None):
        """
        Estimate the flux noise in the lightcurve.
        The base class behavior is to take the maximum
        between the robust RMS of the entire lightcurve,
        and the specific flux error for each measurement.

        Parameters
        ----------
        lightcurve: Lightcurve object
            The lightcurve to estimate the flux noise for.
        source: Source object, optional
            The source this lightcurve is associated with.
            Derived classes may use this to estimate the
            noise differently for different sources.

        Returns
        -------
        noise: array of floats
            The estimated flux noise for each measurement.
        """
        return np.maximum(lightcurve.data["fluxerr"], lightcurve.flux_rms_robust)

    def get_event_indices(self, lightcurve):
        """
        Get an estimate for the time range of the event.
        This is generally returned as the time range
        indices where the lightcurve surpasses the threshold.
        If using snr_threshold_sidebands then the threshold
        used to find the event region can be lower than that
        of the peak.
        If snr_threshold_sidebands is negative, it is applied
        as relative to the snr_threshold value.

        For more complicated events, this will be a range
        of indices where the lightcurve is above the threshold.


        Parameters
        ----------
        lightcurve: Lightcurve object
            The lightcurve where the event is detected.

        Returns
        -------
        event_indices: array of ints
            The indices of the lightcurve where the event is detected.

        """
        if "snr_threshold_sidebands" in self.pars:
            if self.pars.snr_threshold_sidebands < 0:
                # relative to snr_threshold
                thresh = self.pars.snr_threshold + self.pars.snr_threshold_sidebands
            else:  # use value as is
                thresh = self.pars.snr_threshold_sidebands
        else:
            thresh = self.pars.snr_threshold

        return lightcurve.data["snr"].values > thresh

    def make_detection(self, peak_idx, lightcurve, source, sim=None):
        """
        Make a Detection object from a lightcurve.

        Parameters
        ----------
        peak_idx: int
            The index of the peak of the event in the lightcurve.
        lightcurve: Lightcurve object
            The lightcurve where the event is detected.
        source: Source object
            The source this lightcurve is associated with.
        sim: dict or None
            The truth values for the injected event
            if this is a simulated lightcurve.
            If None, assume real data.

        Returns
        -------
        det: Detection object
            The detection object for this event.
        """
        det = Detection()
        det.method = "peak finding"
        det.data_types = self.pars.data_types
        det.source = source
        det.project = self.pars.project
        det.cfg_hash = source.cfg_hash

        det.raw_photometry = source.raw_photometry
        if sim is None:  # real data
            det.reduced_photometry = source.processed_photometry
        else:  # simulated data
            det.reduced_photometry = source.simulated_photometry

        det.raw_photometry.sort(key=lambda x: x.time_start)
        det.reduced_photometry.sort(key=lambda x: x.time_start)
        # mark the location of this detection:
        det.time_indices = self.get_event_indices(lightcurve)
        lightcurve.data.loc[det.time_indices, "detected"] = True

        # in this case time_start and peak start are the same
        det.time_start = lightcurve.times[np.where(det.time_indices)[0][0]]
        det.time_end = lightcurve.times[np.where(det.time_indices)[0][-1]]

        # save simulation values
        det.simulated = sim is not None
        det.sim_pars = sim

        # is this a test run?
        det.test_only = source.test_only

        # time of peak, snr and so on
        det.snr = lightcurve.data.loc[peak_idx, "snr"]
        # can add score and additional_scores if needed
        det.peak_time = lightcurve.times[peak_idx]
        det.peak_start = lightcurve.times[np.where(det.time_indices)[0][0]]
        det.peak_end = lightcurve.times[np.where(det.time_indices)[0][-1]]

        # TODO: add the quality cut values and quality_flag

        det.peak_mag = lightcurve.data.loc[peak_idx, lightcurve.colmap["mag"]]
        det.peak_mag_diff = (
            lightcurve.mag_mean_robust
            - lightcurve.data.loc[peak_idx, lightcurve.colmap["mag"]]
        )

        # can add matched filter here

        return det
