import numpy as np
import pandas as pd

from src.parameters import Parameters
from src.source import Source
from src.dataset import RawData, Lightcurve
from src.detection import DetectionInTime


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
    of that detection (the Lightcurve ID, the time,
    S/N and so on).

    If accepting images, TBD...

    It is important that the Finder state
    does not change when applied to new data,
    since we expect to re-run the same data
    multiple times, before and after injecting
    simulations.
    The sim parameter is used when the data
    ingested has an injected event in it.
    If sim=None, assume real data,
    if sim=dict, assume simulated data
    (or injected data) where the "truth values"
    are passed in using the dictionary.

    """

    def __init__(self, **kwargs):
        self.pars = Parameters.from_dict(kwargs, "finder")
        self.pars.default_values(
            snr_threshold=5,  # S/N threshold for detection
            snr_threshold_sidebands=-2,  # S/N threshold for event region
            max_det_per_lc=1,  # Maximum number of detections per lightcurve
            abs_snr=True,  # Use absolute S/N for detection (i.e., include negative)
        )

    def ingest_lightcurves(self, lightcurves, source=None, sim=None):
        """
        Ingest a list of lightcurves and produce
        detections for them.
        Also, will add some scores to the lightcurve.
        If sim=None, assume real data, and modify
        the lightcurve dataframe to add the scores.
        If sim=dict, assume simulated data
        so make a copy of the lightcurve dataframes
        before modifying them.

        Parameters
        ----------
        lightcurves : list of Lightcurve objects
            The lightcurves to ingest and produce
            detections for.

        source : Source object, optional
            The source this lightcurve is associated with.
            Derived classes may use properties of the source
            to determine detections.

        sim : dict or None
            The truth values for the injected event
            if this is a simulated lightcurve.

        Returns
        -------
        detections : list of Detection objects
            The detections produced by this finder.

        """

        detections = []  # List of detections to return
        if isinstance(lightcurves, Lightcurve):
            lightcurves = [lightcurves]

        for lc in lightcurves:
            # Add some scores to the lightcurve
            lc.data["snr"] = (
                lc.data["flux"] - lc.data["flux"].mean()
            ) / self.estimate_flux_noise(lc, source)
            lc.data["dmag"] = lc.data["mag"] - lc.data["mag"].median()

            # mark indices in the lightcurve where an event was detected
            if "detected" not in lc.data.columns:
                lc.data["detected"] = False

            # Find the detections (just look for high S/N)
            for i in range(self.pars.max_det_per_lc):
                snr = lc.data["snr"].values
                if self.pars.abs_snr:
                    snr = np.abs(snr)
                mx = np.max(snr, where=lc.data["detected"] == 0, initial=-np.inf)
                if mx > self.pars.snr_threshold:
                    # Create a detection object
                    detections.append(self.make_detection(lc, source, sim))
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

    def make_detection(self, lightcurve, source, sim=None):
        """
        Make a detection object from a lightcurve.

        Parameters
        ----------
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
        det = DetectionInTime()
        det.source = source
        det.raw_data = lightcurve.raw_data
        det.lightcurve = lightcurve
        peak = np.argmax(lightcurve.data["snr"])
        det.time_peak = lightcurve.times[peak]

        det.snr = lightcurve.data.loc[peak, "snr"]

        # mark the location of this detection:
        det.time_indices = self.get_event_indices(lightcurve)
        lightcurve.data.loc[det.time_indices, "detected"] = True
        det.time_start = lightcurve.times[det.time_indices[0]]
        det.time_end = lightcurve.times[det.time_indices[-1]]

        # save simulation values
        det.simulated = sim is not None
        det.sim_pars = sim

        return det
