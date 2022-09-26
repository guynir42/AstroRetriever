import numpy as np
import pandas as pd

from src.parameters import Parameters
from src.histogram import Histogram
from src.dataset import Lightcurve
from src.database import Session


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
        self.save_lightcurves = self.add_par(
            "save_lightcurves",
            True,
            bool,
            "Save processed lightcurves after finder and quality cuts",
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


class Analysis:
    """
    This is a pipeline object that accepts some data
    (e.g., lightcurves) and performs some analysis.
    The outputs of the analysis of a dataset are:
    1) Detection objects (if we find anything interesting)
    2) Update to the histogram (counts the number of measurements
       with a specific score like S/N, specific magnitude, etc.)

    The Quality object is used to scan the data and assign
    scores on various parameters, that can be used to disqualify
    some of the data.
    For example, if the individual lightcurve RA/Dec values are
    too far from the source RA/Dec, then we can disqualify
    some data points, which will also flag the Detection objects
    that overlap those measurements.
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

    def initialize(self):
        """
        Initialize the Analysis object.
        This is called before the first run of the pipeline.
        It initializes all the objects that are needed
        for the analysis, like the Quality object,
        using the parameters from the config file or
        from the user.

        """

        # TODO: do we really need a threshold object?
        # self.threshold = Threshold(self.pars.trigger_threshold_dict)

        # TODO: do we really need extra thresholds?
        # for cut_dict in self.pars.extra_cut_thresholds:
        #     self.extra_thresholds.append(self.pars.get_class("threshold", cut_dict))
        # for cut_dict in self.pars.extra_cut_thresholds:
        #     self.extra_scores.append(Histogram())

    def run(self, source):
        """

        Parameters
        ----------
        source

        Returns
        -------

        """
        # TODO: load histograms from file

        if self.pars.data_type == "lightcurves":
            data = source.lightcurves
            if not isinstance(data, list) or not all(
                isinstance(x, Lightcurve) for x in data
            ):
                raise ValueError("The data should be a list of Lightcurve objects.")
            det = self.run_lightcurves(source)
        # add more options here

        if self.pars.save_lightcurves:
            pass  # TODO: save lcs into subfolder

        # TODO: update the histograms

        if self.pars.commit_detections:
            with Session() as session:
                session.add_all(det)
                session.commit()

        self.detections += det

    def run_lightcurves(self, source):
        """
        Run the analysis on a list of Lightcurve objects
        associated with the given source.

        Parameters
        ----------
        source: Source object
            The source to run the analysis on.
            The main data taken from the source
            is the lightcurves list.
            Additional info can be taken from the source
            metadata (e.g., the catalog row).

        Returns
        -------
        det: list of Detection objects
            The detections found in the data.
        """

        self.check_lightcurves(source.lightcurves)
        new_det = self.detect_in_lightcurves(source.lightcurves)
        self.update_histograms(source.lightcurves)

        sim_det = []
        for i in range(self.get_num_injections()):
            sim_lcs, sim_pars = self.inject_to_lightcurves(source.lightcurves)
            sim_det += self.detect_in_lightcurves(sim_lcs, sim_pars)

        return new_det + sim_det

    def check_lightcurves(self, lightcurves):
        """
        Apply the Quality object to the lightcurvs,
        and add the results in a column in the lightcurve
        dataframe.
        Data that has any quality scores above / below
        some threshold could be disqualified, and any
        detections overlapping with such times are rejected.

        Parameters
        ----------
        lightcurves: list of Lightcurve objects
            Data that needs to be scanned for detections.

        """
        for lc in lightcurves:
            self.checker.check(lc)

    def detect_in_lightcurves(self, lightcurves, sim_pars=None):
        """
        Apply the Finder object(s) associated with this
        Analysis, to produce Detection objects based
        on the data in the list of Lightcurves.

        Parameters
        ----------
        lightcurves: list of Lightcurve objects
            Data that needs to be scanned for detections.
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

    def save_detections(self):
        """
        Save the detections to the database.
        """
        pass


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
