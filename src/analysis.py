import numpy as np
import pandas as pd

from src.parameters import Parameters
from src.histogram import Histogram
from src.dataset import Lightcurve
from src.database import Session
from src.source import Source


class ParsAnalysis(Parameters):
    def __init__(self, **kwargs):
        super().__init__()  # initialize base Parameters without passing arguments

        self.data_type = self.add_par(
            "data_type",
            "lightcurves",
            str,
            "Type of data to use for analysis (lightcurves, images, etc.)",
        )

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
        self.save_reduced_lightcurves = self.add_par(
            "save_reduced_lightcurves",
            True,
            bool,
            "Save reduced lightcurves to file and database",
        )
        self.save_processed_lightcurves = self.add_par(
            "save_processed_lightcurves",
            True,
            bool,
            "Save processed lightcurves after finder and quality cuts with a new filename",
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
        self.observatories = None
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

    def analyze_sources_batch(self, sources):
        """

        Parameters
        ----------
        sources: list of Source objects
            A list of sources to run the analysis on.
            After this batch is done, will commit the
            detections to the database and update
            the histogram file.

        """
        if isinstance(sources, Source):
            sources = [sources]

        # TODO: load histograms from file
        batch_detections = []
        for source in sources:
            if self.pars.data_type == "lightcurves":
                # for each raw data, need to make sure lightcurves exist
                for data in source.raw_data:
                    if not data.is_empty():
                        pass
                if len(source.lightcurves) == 0 or all(
                    [lc.check_data_exists for lc in source.lightcurves]
                ):
                    self.reduce_lightcurves(source)
                if len(source.processed_lightcurves) == 0 or all(
                    [lc.check_data_exists for lc in source.processed_lightcurves]
                ):
                    self.process_lightcurves(source)

                det = self.analyze_lightcurves(source)

            # add more options here besides lightcurves
            else:
                raise ValueError(f'Unknown data type: "{self.pars.data_type}"')

            batch_detections += det

        if self.pars.save_processed_lightcurves:
            pass  # TODO: save lcs after adding S/N and quality flags

            # TODO: update the histogram with this source's data

        # TODO: update the histograms file

        # save the detections to the database
        if self.pars.commit_detections:
            with Session() as session:
                session.add_all(batch_detections)
                session.commit()

        self.detections += batch_detections

    def reduce_lightcurves(self, source):
        """
        Reduce the lightcurves for this source.
        Each raw_data on the source will call
        the relevant observatory to do this.
        """

        if self.observatories is None:
            raise ValueError("The observatories are not set.")

        for raw_data in source.raw_data:
            obs = self.observatories[raw_data.observatory]
            source.lightcurves += obs.reduce(raw_data, to="lc", source=source)

        # TODO: move this to the end of each batch?
        if self.pars.save_reduced_lightcurves:
            with Session() as session:
                for lc in source.lightcurves:
                    # the filename should be the same as the raw_data + "_reduced"
                    lc.save()
                session.add(source)
                session.commit()

    def process_lightcurves(self, source, observatory=None):
        """
        Run quality checks and calculate S/N etc.
        on the source's lightcurves.
        The processed lightcurves are appended to the
        source object (These are saved to disk and database
        if using save_processed_lightcurves=True)

        Parameters
        ----------
        source: Source object
            The source to process the lightcurves for.
        observatory: Observatory object (optional)
            If given, only process the lightcurves from this observatory.

        Returns
        -------
        list of Lightcurve objects with data that
        has been processed, e.g., has S/N and quality flags.
        The Lightcurve object will also have has_processed=True.
        """

        pass

    def analyze_lightcurves(self, source):
        """
        Run the analysis on a list of processed Lightcurve
        objects associated with the given source.

        Then detection code is used to find detection
        objects and return those.

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

        new_lcs = self.process_lightcurves(source.lightcurves)
        new_det = self.detect_in_lightcurves(source.lightcurves)
        self.update_histograms(source.lightcurves)

        sim_det = []
        for i in range(self.get_num_injections()):
            sim_lcs, sim_pars = self.inject_to_lightcurves(source.lightcurves)
            sim_det += self.detect_in_lightcurves(sim_lcs, sim_pars)

        det = new_det + sim_det

        return

    def check_lightcurves(self, lightcurves):
        """
        Apply the Quality object to the lightcurves,
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

    def detect_in_lightcurves(self, source, sim_pars=None):
        """
        Apply the Finder object(s) associated with this
        Analysis, to produce Detection objects based
        on the data in the list of Lightcurves.

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
