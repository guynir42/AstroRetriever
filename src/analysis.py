import os

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
            "quality_class", "QualityChecker", str, "Name of the quality checker class"
        )

        self.finder_module = self.add_par(
            "finder_module",
            "src.finder",
            str,
            "Module where the Finder class is defined",
        )
        self.finder_class = self.add_par(
            "finder_class", "Finder", str, "Name of the Finder class"
        )

        self.simulator_module = self.add_par(
            "simulation_module",
            "src.simulator",
            str,
            "Module where the Simulator class is defined",
        )
        self.simulator_class = self.add_par(
            "simulator_class", "Simulator", str, "Name of the Simulator class"
        )

        self.save_anything = self.add_par(
            "save_anything",
            True,
            bool,
            "Don't save anything to disk or database. "
            "Override any of the other save parameters. ",
        )

        self.save_processed = self.add_par(
            "commit_processed",
            True,
            bool,
            "Save and commit processed lightcurves "
            "after finder and quality cuts with a new filename",
        )
        self.save_simulated = self.add_par(
            "commit_simulated",
            True,
            bool,
            "Save and commit simulated lightcurves "
            "after finder and quality cuts with a new filename",
        )
        self.save_detections = self.add_par(
            "commit_detections", True, bool, "Save detections to database"
        )

        self.save_histograms = self.add_par(
            "save_histograms", True, bool, "Update the histograms on file"
        )

        self.histogram_filename = self.add_par(
            "histogram_filename",
            "histograms.nc",
            str,
            "Filename for the histograms file",
        )  # TODO: should this be appended a cfg_hash?

        self._enforce_type_checks = True
        self._enforce_no_new_attrs = True

        self.load_then_update(kwargs)

    @classmethod
    def get_default_cfg_key(cls):
        """
        Get the default key to use when loading a config file.
        """
        return "analysis"

    def need_to_save(self):
        """
        Check if any of the parameters require
        that we save to disk and/or commit anything to database.
        """

        ret = False
        ret |= self.save_processed
        ret |= self.save_simulated
        ret |= self.save_detections
        ret |= self.save_histograms

        return ret


class Analysis:
    """
    This is a pipeline object that accepts a Source object
    (e.g., with some lightcurves) and performs analysis on it.
    Use pars.data_types to choose which kind of analysis
    is to be done. The default is to use photometry.
    Any sources with all-empty raw data will be skipped.
    Other sources are assumed to have reduced datasets,
    and that they already contain the data required.
    For example, a source with a lightcurve that is missing
    its data (either in RAM or from disk) will raise an exception.

    The outputs of the analysis are:
    1) Properties object, summarizing the results for each object (e.g., best S/N).
    2) Processed data (e.g., lightcurves) with quality flags and S/N, etc.
    3) Detection objects (if we find anything interesting)
    4) Updates to the histogram (counts the number of measurements
       with a specific score like S/N, specific magnitude, etc.).
    5) Simulated datasets and detections (if using injections).

    The QualityChecker object is used to scan the data and assign
    scores on various parameters, that can be used to disqualify
    parts of the data.
    For example, if the individual lightcurve RA/Dec values are
    too far from the source RA/Dec, then we can disqualify
    some data points, which will also flag the Detection objects
    that overlap those measurements.
    This uses, e.g., the "offset" quality cut.

    The Finder object then proceeds to add S/N (and other scores)
    to the processed lightcurve.
    The processed lightcurves can be saved to the database
    and to disk, or they can be removed to save space.
    Use the pars.commit_processed to toggle this behavior.

    quality_values is a histogram object is used to keep track
    of the number of measurements that are lost to each quality cut,
    so we can fine tune the cut thresholds on the QualityChecker object.
    all_scores and good_scores are used to track the number of measurements
    that end up with each score. The good_scores includes only data that
    passed all quality cuts.

    The Simulator is used to inject fake sources into the data,
    which are recovered as Detection objects.
    The injection data is calculated after posting the new
    counts into the histogram.

    """

    def __init__(self, **kwargs):
        quality_kwargs = kwargs.pop("quality_kwargs", {})
        finder_kwargs = kwargs.pop("finder_kwargs", {})
        simulator_kwargs = kwargs.pop("simulator_kwargs", {})
        self.output_folder = kwargs.pop("output_folder", os.getcwd())

        # kwargs to pass into the different histograms
        histogram_kwargs = kwargs.pop("histogram_kwargs", {})
        histogram_kwargs["initialize"] = False  # just making sure
        histogram_kwargs["output_folder"] = self.output_folder

        # ingest the rest of the kwargs:
        self.pars = ParsAnalysis(**kwargs)

        # quality check cuts and the values histograms
        self.checker = self.pars.get_class_instance("quality", **quality_kwargs)

        self.pars.add_defaults_to_dict(histogram_kwargs)  # project name etc
        self.quality_values = Histogram(**histogram_kwargs, name="quality_values")

        # update the kwargs with the right scores:
        self.quality_values.pick_out_coords(self.checker.pars.cut_names, "score")
        self.quality_values.initialize()

        # the finder and simulator
        self.finder = self.pars.get_class_instance("finder", **finder_kwargs)
        self.finder.checker = self.checker  # link to this here, too
        self.sim = self.pars.get_class_instance("simulator", **simulator_kwargs)

        self.all_scores = Histogram(**histogram_kwargs, name="all_scores")
        self.all_scores.pick_out_coords(self.finder.pars.score_names, "score")
        self.all_scores.initialize()
        self.good_scores = Histogram(**histogram_kwargs, name="good_scores")
        self.good_scores.pick_out_coords(self.finder.pars.score_names, "score")
        self.good_scores.initialize()
        # self.extra_scores = []  # an optional list of extra Histogram objects

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

        self.load_histograms()

        batch_detections = []
        for source in sources:
            # check how many raw datasets are not empty
            non_empty = 0
            for dt in self.pars.data_types:
                for data in getattr(source, f"raw_{dt}"):
                    if not data.is_empty():
                        non_empty += 1
            if non_empty == 0:
                source.properties = Properties(
                    has_data=False, project=self.pars.project
                )
                continue  # skip sources without data

            # what data types go into the analysis?
            analysis_name = "analyze_" + "_and_".join(self.pars.data_types)
            analysis_func = getattr(self, analysis_name, None)
            if analysis_func is None or not callable(analysis_func):
                raise ValueError(
                    f'No analysis function named "{analysis_name}" was found. '
                )

            batch_detections += analysis_func(source)

            # get rid of data we don't want saved to the source
            for dt in self.pars.data_types:
                # do we also need to clear these from the raw data relationship?
                if not self.pars.save_anything or not self.pars.save_processed:
                    setattr(source, f"processed_{dt}", [])  # clear list
                if not self.pars.save_anything or not self.pars.save_simulated:
                    setattr(source, f"simulated_{dt}", [])  # clear list

            # TODO: what happens if more than one data type on each det?
            for det in batch_detections:
                det.processed_data = None

            # get rid of these new detections if we don't
            # want to save them to DB (e.g., for debugging)
            if not self.pars.save_anything or not self.pars.save_detections:
                source.detections = []

        # this gets appended but never committed
        self.detections += batch_detections

        # save the detections to the database
        if self.pars.save_anything and self.pars.need_to_save():

            with Session() as session:
                try:  # if anything fails, must rollback all
                    if self.pars.save_histograms:
                        self.save_histograms(temp=True)

                    for source in sources:
                        session.add(source)
                        for dt in self.pars.data_types:
                            for data in getattr(source, f"raw_{dt}"):
                                if data.filename is None:
                                    raise ValueError(
                                        f"raw_{dt} (from {data.observatory}) "
                                        f"on Source {source.id} has no filename. "
                                        "Did you forget to save it?"
                                    )
                            for i, data in enumerate(getattr(source, f"reduced_{dt}")):
                                if data.filename is None:
                                    raise ValueError(
                                        f"reduced_{dt} (number {i}) on Source {source.id} "
                                        "has no filename. Did you forget to save it?"
                                    )
                            [data.save() for data in getattr(source, f"processed_{dt}")]
                            [data.save() for data in getattr(source, f"simulated_{dt}")]

                    session.commit()

                    # if all the file saving and DB interactions work,
                    # then we can commit the histograms (rename temp files)
                    self.commit_histograms()
                except Exception:
                    self.rollback_histograms()
                    session.rollback()
                    for source in sources:
                        for dt in self.pars.data_types:
                            [
                                lc.delete_data_from_disk()
                                for lc in getattr(source, f"processed_{dt}")
                            ]
                            [
                                lc.delete_data_from_disk()
                                for lc in getattr(source, f"simulated_{dt}")
                            ]

                    raise  # finally re-raise the exception

    def analyze_photometry(self, source):
        """
        Run the analysis on a list of processed Lightcurve
        objects associated with the given source.

        The processing is done using the QualityChecker and Finder
        objects, that generate processed versions of the
        input data (e.g., lightcurves with S/N and quality flags).

        Then the Finder's detection code is used to find detection
        objects and return those.

        For each source, injection simulations are added using
        the Simulator object. The injected data (e.g., lightcurves)
        are re-processed using the same QualityChecker and Finder objects,
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
        lcs = [Lightcurve(lc) for lc in source.reduced_lightcurves]
        # remove existing lightcurves and make copies of the
        # "reduced_lightcurves" to use as "processed_lightcurves"
        source.processed_lightcurves = lcs
        self.check_lightcurves(lcs, source)
        self.process_lightcurves(lcs, source)
        new_det = self.detect_in_lightcurves(lcs, source)
        # make sure to mark these as processed
        [setattr(lc, "was_processed", True) for lc in lcs]

        self.calc_props_from_lightcurves(lcs, source)

        self.update_histograms(lcs, source)

        sim_det = []
        source.simulated_lightcurves = []  # get rid of the old ones
        for i in range(self.get_num_injections()):
            # add simulated events into the lightcurves
            sim_lcs, sim_pars = self.inject_to_lightcurves(lcs, source, index=i)
            [setattr(lc, "was_simulated", True) for lc in sim_lcs]

            # re-run quality and finder on the simulated data
            self.check_lightcurves(sim_lcs, source, sim_pars)
            self.process_lightcurves(sim_lcs, source, sim_pars)

            # find detections in the simulated data
            sim_det += self.detect_in_lightcurves(sim_lcs, source, sim_pars)

        det = new_det + sim_det
        source.detections = det

        return det

    def check_lightcurves(self, lightcurves, source, sim=None):
        """
        Apply the QualityChecker object to the lightcurves,
        and add the results into columns in the lightcurve
        dataframe (or update existing columns).
        Data that has any quality scores above / below
        some threshold could be disqualified, and any
        detections overlapping with such times are rejected.

        This function must be able to re-process lightcurves
        that already have the quality scores added
        (e.g., for running it on simulated data).

        Parameters
        ----------
        lightcurves: list of Lightcurve objects
            The lightcurves to check. For regular usage,
            this would be the "processed_lightcurves".
            If using simulations (i.e., sim is not None),
            then this would be the "simulated_lightcurves".
        source: Source object
            The source to process the lightcurves for.
            By default, the lightcurves are processed
            in place using the source's processed_lightcurves.
            If sim is not None, then the lightcurves
            used will be the source's simulated_lightcurves.

        """
        self.checker.check(lightcurves, source, sim)

    def process_lightcurves(self, lightcurves, source, sim=None):
        """
        Apply the Finder object to the lightcurves,
        and add the results into columns in the lightcurve
        dataframe (or update existing columns).
        The resulting statistics (e.g., snr) are used
        to find detections in the lightcurves.

        This function must be able to re-process lightcurves
        that already have the quality scores added
        (e.g., for running it on simulated data).

        Parameters
        ----------
        lightcurves: list of Lightcurve objects
            The lightcurves to check. For regular usage,
            this would be the "processed_lightcurves".
            If using simulations (i.e., sim is not None),
            then this would be the "simulated_lightcurves".
        source: Source object
            The source to process the lightcurves for.
            By default, the lightcurves are processed
            in place using the source's processed_lightcurves.
            If sim is not None, then the lightcurves
            used will be the source's simulated_lightcurves.

        """
        self.finder.process(lightcurves, source, sim)

    def detect_in_lightcurves(self, lightcurves, source, sim=None):
        """
        Apply the Finder object(s) associated with this
        Analysis, to produce Detection objects based
        on the data in the list of Lightcurves.

        The lightcurve's data may be appended additional
        columns like "snr", and if such columns exists
        they will be overwritten.

        Parameters
        ----------
        lightcurves: list of Lightcurve objects
            The lightcurves to check. For regular usage,
            this would be the "processed_lightcurves".
            If using simulations (i.e., sim is not None),
            then this would be the "simulated_lightcurves".
        source: Source object
            The lightcurves for this source are scanned.
        sim: dict or None
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
        return self.finder.detect(lightcurves, source, sim)

    def calc_props_from_lightcurves(self, lightcurves, source):
        """
        Calculate some Properties on this source
        based on the lightcurves given.
        Will generally use processed_lightcurves,
        because this function is not used for
        simulated data.
        The output would be a Properties object
        that is appended to the source object.

        Parameters
        ----------
        lightcurves: list of Lightcurve objects
            The lightcurves to check. For regular usage,
            this would be the "processed_lightcurves".
        source: Source object
            The lightcurves for this source are scanned.
        """
        # TODO: calculate best S/N and so on...

        source.properties = Properties(has_data=True, project=self.pars.project)

    def update_histograms(self, lightcurves, source):
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
        lightcurves: list of Lightcurve objects
            The lightcurves to check. These are always
            going to be the "processed_lightcurves".
        source: Source object
            The lightcurves for this source are scanned.
            Adds the name of the source to the list of source
            names that were included in the histograms.

        Returns
        -------

        """
        pass  # TODO: implement this

    def inject_to_lightcurves(self, lightcurves, source, index=0):
        """
        Inject a fake source/event into the data.
        The fake source is added to the lightcurves,
        and the parameters of the fake source is
        returned in a dictionary.

        Parameters
        ----------
        lightcurves: list of Lightcurve objects
            Data that needs to be scanned for detections.
        source: Source object
            The source to inject the fake source into.
            Can use the properties of the source
            (e.g., magnitude) to determine the exact details
            of the simulated injected event.
        index: int
            Index of the fake source to inject.
            This is used to keep track of the serial number
            of the set of lightcurves, in case we are saving
            lightcurves to file and have multiple, different
            simulated injections on the same data.
        Returns
        -------
        lightcurves: list of Lightcurve objects
            Data that now contains a fake source.
        sim_pars: dict
            Parameters of the fake source that was injected.
        """
        # TODO: implement this
        sim_lcs = []
        sim_pars = {}

        return sim_lcs, sim_pars

    def get_num_injections(self):
        """
        Get a number of injections that should
        be made into each source.
        This number is not constant,
        instead assume a Poisson distribution
        where the mean is self.pars.num_injections.
        """

        return np.random.poisson(self.pars.num_injections)

    def get_all_histograms(self):
        """
        Get a list of all histograms that are
        associated with this Analysis object.
        """
        return [obj for obj in self.__dict__.values() if isinstance(obj, Histogram)]

    def load_histograms(self):
        """
        Load the histograms from file.
        This is used to continue a previous run,
        or to use the histograms to make plots.
        """
        for hist in self.get_all_histograms():
            hist.load()

    def save_histograms(self, temp=False):
        """
        Save the histograms to a (temporary) file.
        If using temp=False, will simply save all the
        histograms to their respective files.
        If using temp=True, will save the histograms
        into temporary files (appended with ".temp");
        To make sure the temp file replaces the old file,
        must also call commit_histograms().
        """
        suffix = "temp" if temp else None

        for hist in self.get_all_histograms():
            hist.save(suffix=suffix)

    def rollback_histograms(self):
        """
        Roll back the histograms saved to the temporary files.
        This is called in case there was a problem
        saving or committing any data.
        """
        for hist in self.get_all_histograms():
            hist.remove_data_from_file(suffix="temp")

    def commit_histograms(self):
        """
        Commit the histograms saved to the temporary files.
        This is called after the histograms have been saved
        to the temporary files, and we are sure that
        there were no problems.
        Will also create backup files for the histograms.
        """
        for hist in self.get_all_histograms():
            fullname = os.path.join(hist.output_folder, f"histograms_{hist.name}.nc")
            if os.path.exists(fullname):
                os.rename(fullname, fullname + ".backup")

            os.rename(fullname + ".temp", fullname)
