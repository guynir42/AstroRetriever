import numpy as np
import pandas as pd

from sqlalchemy.orm.session import make_transient

from src.parameters import Parameters
from src.histogram import Histogram
from src.dataset import Lightcurve
from src.database import Session
from src.source import Source
from src.properties import Properties


class ParsAnalysis(Parameters):
    def __init__(self, **kwargs):
        super().__init__()  # initialize base Parameters without passing arguments

        self.num_injections = self.add_par(
            "num_injections",
            1,
            (int, float),
            "Number of fake events to inject per source (can be fractional)",
        )
        self.quality_module = self.add_par(
            "quality_module",
            "src.quality",
            str,
            "Module where the Quality class is defined",
        )
        self.quality_class = self.add_par(
            "quality_class", "Quality", str, "Name of the Quality class"
        )
        # quality_kwargs = {},  # parameters for the Quality class
        self.finder_module = self.add_par(
            "finder_module",
            "src.finder",
            str,
            "Module where the Finder class is defined",
        )
        self.finder_class = self.add_par(
            "finder_class", "Finder", str, "Name of the Finder class"
        )
        # finder_kwargs = {},  # parameters for the Finder class
        self.simulator_module = self.add_par(
            "simulation_module",
            "src.simulator",
            str,
            "Module where the Simulator class is defined",
        )
        self.simulator_class = self.add_par(
            "simulator_class", "Simulator", str, "Name of the Simulator class"
        )
        # simulator_kwargs = {},  # parameters for the Simulator class
        self.update_histograms = self.add_par(
            "update_histograms", True, bool, "Update the histograms on file"
        )
        # self.save_reduced_lightcurves = self.add_par(
        #     "save_reduced_lightcurves",
        #     True,
        #     bool,
        #     "Save reduced lightcurves to file and database",
        # )
        self.commit_processed = self.add_par(
            "commit_processed",
            True,
            bool,
            "Save and commit processed lightcurves "
            "after finder and quality cuts with a new filename",
        )
        self.commit_detections = self.add_par(
            "commit_detections", True, bool, "Save detections to database"
        )

        self._enforce_type_checks = True
        self._enforce_no_new_attrs = True

        self.load_then_update(kwargs)

    @classmethod
    def get_default_cfg_key(cls):
        """
        Get the default key to use when loading a config file.
        """
        return "analysis"

    def need_to_commit(self):
        """
        Check if any of the parameters require
        that we save to disk and/or commit anything to database.
        """

        ret = False
        ret |= self.commit_detections
        ret |= self.save_reduced_lightcurves
        ret |= self.save_processed_lightcurves
        ret |= self.update_histograms

        return ret


class Analysis:
    """
    This is a pipeline object that accepts a Source object
    (e.g., with some lightcurves) and performs analysis on it.
    Use pars.data_types to choose which kind of analysis is to be done.
    The default is to use photometry.
    Any sources with all empty raw data will be skipped.
    Other sources are assumed to have reduced datasets,
    and that they already contain the data required.
    For example, a source with a lightcurve that is missing
    its data (either in RAM or from disk) will raise an exception.

    The outputs of the analysis are:
    1) Properties object summarizing the results for each object (e.g., best S/N).
    2) Processed data (e.g., lightcurves) with quality flags and S/N, etc.
    3) Detection objects (if we find anything interesting)
    4) Update to the histogram (counts the number of measurements
       with a specific score like S/N, specific magnitude, etc.).
    5) Simulated datasets and detections (if using injections).

    The Quality object is used to scan the data and assign
    scores on various parameters, that can be used to disqualify
    parts of the data.
    For example, if the individual lightcurve RA/Dec values are
    too far from the source RA/Dec, then we can disqualify
    some data points, which will also flag the Detection objects
    that overlap those measurements.

    The Finder object then proceeds to add S/N (and other scores)
    to the processed lightcurve. These can be saved to the database
    and to disk, or they can be removed to save space.
    Use the pars.commit_processed to toggle this behavior.

    A Histogram object can be used to keep track of the number
    of measurements that are lost to each score,
    so we can fine tune the cut thresholds on the Quality object.

    The Simulator is used to inject fake sources into the data,
    which are recovered as Detection objects.
    The injection data is calculated after posting the new
    counts into the histogram.


    """

    def __init__(self, **kwargs):
        finder_kwargs = kwargs.pop("finder_kwargs", {})
        quality_kwargs = kwargs.pop("quality_kwargs", {})
        simulator_kwargs = kwargs.pop("simulator_kwargs", {})

        self.pars = ParsAnalysis(**kwargs)
        self.all_scores = Histogram()
        self.good_scores = Histogram()
        # self.extra_scores = []  # an optional list of extra Histogram objects
        self.quality_values = Histogram()
        self.finder = self.pars.get_class("finder", **finder_kwargs)
        self.checker = self.pars.get_class("quality", **quality_kwargs)
        self.sim = self.pars.get_class("simulator", **simulator_kwargs)
        # self.threshold = None  # Threshold object
        # self.extra_thresholds = []  # list of Threshold objects

        self.detections = []  # a list of Detection objects
        # TODO: should we limit the length of this list in memory?

    def analyze_sources(self, sources):
        """

        Parameters
        ----------
        sources: Source or list of Source objects
            A list of sources to run the analysis on.
            After this batch is done will update
            the histogram on disk.
        """
        if isinstance(sources, Source):
            sources = [sources]

        # TODO: load histograms from file
        batch_detections = []
        for source in sources:
            # check how many raw datasets are not empty
            non_empty = 0
            for dt in self.pars.data_types:
                for data in getattr(source, f"raw_{dt}"):
                    if not data.is_empty():
                        non_empty += 1
            if non_empty == 0:
                source.properties = Properties(has_data=False)
                continue  # skip sources without data

            # what data types go into the analysis?
            analysis_name = "analyze_" + "_and_".join(self.pars.data_types)
            analysis_func = getattr(self, analysis_name, None)
            if analysis_func is None or not callable(analysis_func):
                raise ValueError(
                    f'No analysis function named "{analysis_name}" was found. '
                )

            batch_detections += analysis_func(source)

        if self.pars.commit_processed:
            pass  # TODO: save lcs after adding S/N and quality flags

            # TODO: update the histogram with this source's data

        # TODO: update the histograms file

        # save the detections to the database
        if self.pars.commit_detections:
            with Session() as session:
                session.add_all(batch_detections)
                session.commit()

        self.detections += batch_detections

    def analyze_photometry(self, source):
        """
        Run the analysis on a list of processed Lightcurve
        objects associated with the given source.

        The processing is done using the Quality and Finder
        objects, that generate processed versions of the
        input data (e.g., lightcurves with S/N and quality flags).

        Then the Finder's detection code is used to find detection
        objects and return those.

        For each source, injection simulations are added using
        the Simulator object. The injected data (e.g., lightcurves)
        are re-processed using the same Quality and Finder objects,
        that re-apply the same cuts and scores to the injected data.
        Then the Finder looks for detections in those data, and
        outputs them along with the regular detection objects.

        Parameters
        ----------
        source: Source object
            The lightcurves for this source are scanned
            for detections. Additional info can be taken
            from the source metadata (e.g., the catalog row).

        Returns
        -------
        det: list of Detection objects
            The detections found in the data.
        """

        source.processed_lightcurves = [
            Lightcurve(lc) for lc in source.reduced_lightcurves
        ]
        self.check_lightcurves(source)

        new_det = self.detect_in_lightcurves(source)
        self.update_histograms(source.processed_lightcurves)

        sim_det = []
        for i in range(self.get_num_injections()):
            # add simulated events into the lightcurves
            sim_pars = self.inject_to_lightcurves(source)

            # re-run quality and finder on the simulated data
            self.process_lightcurves(source, sim_pars)

            # find detections in the simulated data
            sim_det += self.detect_in_lightcurves(
                source.simulated_lightcurves, sim_pars
            )

        det = new_det + sim_det

        return det

    def check_lightcurves(self, source, sim=None):
        """
        Apply the Quality object to the lightcurves,
        and add the results in a column in the lightcurve
        dataframe.
        Data that has any quality scores above / below
        some threshold could be disqualified, and any
        detections overlapping with such times are rejected.

        This function must be able to re-process lightcurves
        that already have the quality scores added
        (e.g., for running it on simulated data).

        Parameters
        ----------
        source: Source object
            The source to process the lightcurves for.
            By default, the lightcurves are processed
            in place using the source's processed_lightcurves.
            If sim is not None, then the lightcurves
            used will be the source's simulated_lightcurves.

        """
        if sim is None:
            lightcurves = source.processed_lightcurves
        else:
            lightcurves = source.simulated_lightcurves

        for lc in lightcurves:
            self.checker.check(lc)

    def detect_in_lightcurves(self, source, sim_pars=None):
        """
        Apply the Finder object(s) associated with this
        Analysis, to produce Detection objects based
        on the data in the list of Lightcurves.

        The lightcurve's data may be appended additional
        columns like "snr", and if such columns exists
        they will be overwritten.

        Parameters
        ----------
        source: Source object
            The lightcurves for this source are scanned.
        sim_pars: dict or None
            If None, assume the data is real.
            If a dict, the data is either simulated,
            or real data with injected fake objects.
            The dict should contain the parameters
            of the fake objects.

        Returns
        -------
        detections: list of Detection objects
            The detections found in the data.

        """
        pass

    def update_histograms(self, lightcurves):
        """
        Go over the histograms and update them.
        There are a few histograms that need to be updated.
        The first is the "all" histogram, which keeps track
        of the scores for all measurements, regardless of
        the quality flags.
        The second is a list of "pass" histograms,
        each associated with a specific Threshold object.
        Measurements that do not pass the respective thresholds
        are not included in the count for these histograms.
        There should be one or more Threshold objects,
        allowing the user to try different configurations
        simultaneously and measure the amount of data lost
        in each case.
        The last one is the "quality" histogram,
        which keeps track of the values (scores) of
        the various quality flags, so we can later
        adjust the thresholds we want to use.


        Parameters
        ----------
        lightcurves

        Returns
        -------

        """
        pass

    def inject_to_lightcurves(self, lightcurves):
        """
        Inject a fake source/event into the data.
        The fake source is added to the lightcurves,
        and the parameters of the fake source is
        returned in a dictionary.

        Parameters
        ----------
        lightcurves: list of Lightcurve objects
            Data that needs to be scanned for detections.

        Returns
        -------
        lightcurves: list of Lightcurve objects
            Data that now contains a fake source.
        sim_pars: dict
            Parameters of the fake source that was injected.

        """
        pass

    def get_num_injections(self):
        """
        Get a number of injections that should
        be made into each source.
        This number is not constant,
        instead assume a Poisson distribution
        where the mean is self.pars.num_injections.
        """

        return np.random.poisson(self.pars.num_injections)

    def save_histograms(self, path_to_save, project):
        """
        Save the histograms to a file.

        Parameters
        ----------
        path_to_save: str
            Path to the directory where the histograms
            should be saved.
        project: str
            Name of the project, used to create the filename.

        """
        pass

    def commit_detections(self):
        """
        Save the detections to the database.
        """
        pass


# I think we can get rid of this:
class Threshold:
    def __init__(self, name, **kwargs):
        self.name = name
        self.abs = kwargs.get("abs", False)
        self.type = kwargs.get("type", "float")
        if self.type == "float":
            self.threshold = kwargs.get("thresh", 5.0)
            self.max = kwargs.get("max", 10)
            self.min = kwargs.get("min", -10)
            self.step = kwargs.get("step", 0.1)
        elif self.type == "int":
            self.threshold = kwargs.get("thresh", 5)
            self.max = kwargs.get("max", 10)
            self.min = kwargs.get("min", -10)
            self.step = kwargs.get("step", 1)
        elif self.type == "bool":
            self.threshold = kwargs.get("thresh", 1)
            self.max = kwargs.get("max", 1)
            self.min = kwargs.get("min", 0)
            self.step = kwargs.get("step", 1)
        else:
            raise ValueError(
                f"Unknown threshold type: {self.type}. "
                f"Use 'float', 'int', or 'bool'."
            )
