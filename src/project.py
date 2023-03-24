"""
The project is used to combine a catalog,
some observatories, and analysis objects.
"""
import os
import json
import yaml
import hashlib
import git
import traceback
import importlib

import numpy as np

import sqlalchemy as sa
from sqlalchemy.orm import joinedload

import src.database
from src.database import Base, SmartSession, safe_mkdir
from src.parameters import Parameters, get_class_from_data_type
from src.catalog import Catalog
from src.observatory import ParsObservatory
from src.source import Source
from src.dataset import RawPhotometry, Lightcurve
from src.analysis import Analysis
from src.properties import Properties
from src.detection import Detection
from src.utils import (
    help_with_class,
    help_with_object,
    NamedList,
    CircularBufferList,
    legalize,
)


class ParsProject(Parameters):
    def __init__(self, **kwargs):
        super().__init__()  # initialize base Parameters without passing arguments

        self.obs_names = self.add_par(
            "obs_names",
            ("demo"),
            (str, list),
            "List of observatory names or one name",
        )

        self.description = self.add_par("description", "", str, "Project description")

        self.source_buffer_size = self.add_par(
            "source_buffer_size",
            100,
            int,
            "Number of sources to keep inside the `sources` attibute of the project. ",
        )

        self.source_batch_size = self.add_par(
            "source_batch_size",
            10,
            int,
            "Number of sources to process in one batch, after which we commit the sources and save the histograms.",
        )

        self.max_num_total_exceptions = self.add_par(
            "max_num_total_exceptions",
            100,
            int,
            "Maximum number of exceptions (total) to raise before stopping the analysis.",
        )

        self.max_num_sequence_exceptions = self.add_par(
            "max_num_sequence_exceptions",
            10,
            int,
            "Maximum number of exceptions (in a row) to raise before stopping the analysis.",
        )

        self.version_control = self.add_par(
            "version_control", False, bool, "Whether to use version control"
        )

        self.ignore_missing_raw_data = self.add_par(
            "ignore_missing_raw_data",
            False,
            bool,
            "Whether to ignore (or else raise) missing raw data on disk.",
        )

        self.analysis_module = self.add_par(
            "analysis_module",
            "src.analysis",
            str,
            "Module to use for analysis.",
        )

        self.analysis_class = self.add_par(
            "analysis_class",
            "Analysis",
            str,
            "Class to use for analysis.",
        )

        # these are hashes to be automatically filled
        # and added to the output config file
        self.git_hash = self.add_par(
            "git_hash",
            None,
            (None, str),
            "Git hash of the current commit, if using version control",
        )

        self.cat_hash = self.add_par(
            "cat_hash",
            None,
            (None, str),
            "Hash of the names in the catalog for this project.",
        )

        self._enforce_no_new_attrs = True

        self.load_then_update(kwargs)

    @property
    def name(self):
        return self.project

    def _verify_observatory_names(self, names):
        """
        Check that the observatory names are a unique set of strings.

        Parameters
        ----------
        names : list, tuple, set or str
            List of observatory names.
            Can be a set, tuple or list.
            Can also be a single string.
            All inputs will be converted to upper case
            and saved as a tuple of unique strings.
        """
        # if only one observatory name is given, make it a list:
        if isinstance(names, str):
            names = [names]

        names = list(names)

        upper_obs = []
        for obs in names:
            if not isinstance(obs, str):
                raise TypeError("observatories must be a list of strings")
            upper_obs.append(obs.upper())

        # cast the list into a set to make each name unique
        names = list(set(upper_obs))

        super().__setattr__("obs_names", names)

    def __setattr__(self, key, value):
        if key in ("vc", "version_control"):
            key = "version_control"
        if key == "obs_names":
            self._verify_observatory_names(value)
            return

        if key == "name":
            key = "project"

        super().__setattr__(key, value)

    @classmethod
    def _get_default_cfg_key(cls):
        """
        Get the default key to use when loading a config file.
        """
        return "project"

    @staticmethod
    def get_pars_list(project_obj, verbose=False):
        """
        Get a list of Parameters objects for all
        the objects in the given project.

        Parameters
        ----------
        project_obj: ParsProject
            The project object to get the list of parameters for.
        verbose: bool
            Whether to print the names of the objects as they are found.

        Returns
        -------
        pars_list: list
            List of Parameters objects.
        """
        pars_list = []
        ParsProject._update_pars_list(project_obj, pars_list, verbose=verbose)
        ParsProject._order_pars_list(pars_list, verbose=verbose)

        return pars_list

    @staticmethod
    def _update_pars_list(obj, pars_list, object_list=None, verbose=False):
        """
        Recursively get the parameter objects from
        all sub-objects of the given object.
        Will not re-add the same parameter object
        if it is already in the list.

        Parameters
        ----------
        obj: any type
            object to search for parameters
        pars_list: list of Parameter objects
            list of parameter objects that can
            be turned into an output YAML file.
        object_list: list
            list of objects that have already been
            searched for parameters. This is used
            to avoid infinite recursion.
        verbose: bool
            print out the objects that are being
            searched for parameters. Default is False.
        """

        if (
            not hasattr(obj, "__dict__")
            or callable(obj)
            or type(obj).__module__ == "builtins"
            or hasattr(obj, "__iter__")
            or isinstance(obj, Base)
        ):
            return

        if object_list is None:
            object_list = []

        if obj in object_list:
            return
        else:
            object_list.append(obj)

        if verbose:
            print(f"scanning a {type(obj)} object...")
        if isinstance(obj, Parameters) and obj not in pars_list:
            pars_list.append(obj)
            return

        if verbose:
            print("go over items...")
        # if object itself is not a Parameters object,
        # maybe one of its attributes is
        for k, v in obj.__dict__.items():
            if (
                hasattr(v, "__iter__")
                and not isinstance(v, str)
                and not isinstance(v, np.ndarray)
            ):
                if verbose:
                    print(f"loading an iterable: {k}")
                for item in v:
                    ParsProject._update_pars_list(item, pars_list, object_list)
            elif isinstance(v, dict):
                if verbose:
                    print(f"loading a dict: {k}")
                for item in v.values():
                    ParsProject._update_pars_list(item, pars_list, object_list)
            else:
                if verbose:
                    print(f"loading a single object: {k}")
                ParsProject._update_pars_list(v, pars_list, object_list)

    @staticmethod
    def _order_pars_list(pars_list, verbose=False):
        """
        Reorder a list of Parameter objects such that
        ParsProject is first, then all observatories
        (in alphabetical order) then all other objects
        in alphabetical order of their class name.
        """
        pars_list.sort(key=lambda x: x.__class__.__name__)
        pars_list.sort(key=lambda x: isinstance(x, ParsObservatory), reverse=True)
        pars_list.sort(
            key=lambda x: x.__class__.__name__ == "ParsProject", reverse=True
        )

        if verbose:
            names_list = [p.__class__.__name__ for p in pars_list]
            print(names_list)


class Project:
    """
    Combine a catalog, observatories, and analysis objects.

    The project allows loading of parameters,
    and saving of results all under one object.
    Each project should be used for one science case,
    with a set of sources given by the catalog,
    a set of observatories given by the obs_names list,
    and the reduction and analysis to be done
    on the data from each observatory.
    """

    def __init__(self, name, **kwargs):
        """
        Create a new Project object.

        Parameters
        ----------
        name: str
            Name of the project.
            Will be used as default config filename,
            as default output filenames, and so on.
        Additional arguments are passed into the Parameters object.
        """
        self.name = name
        kwargs["project"] = name  # propagate this to sub-objects

        # make sure kwargs for contained objects are not passed to Parameters
        obs_kwargs = kwargs.pop("obs_kwargs", {})
        catalog_kwargs = kwargs.pop("catalog_kwargs", {})
        analysis_kwargs = kwargs.pop("analysis_kwargs", {})

        # this loads parameters from file, then from kwargs:
        self.pars = self._make_pars_object(kwargs)

        # add some default keys like "project" and "verbose" to kwargs
        self.pars.add_defaults_to_dict(obs_kwargs)
        self.pars.add_defaults_to_dict(catalog_kwargs)
        self.pars.add_defaults_to_dict(analysis_kwargs)

        # filled by _setup_output_folder at runtime:
        self.output_folder = None
        self.cfg_hash = None  # hash of the config file (for version control)
        self._test_hash = None

        # version control:
        if self.pars.version_control:
            try:
                repo = git.Repo(search_parent_directories=True)
                git_hash = repo.head.object.hexsha
            except git.exc.InvalidGitRepositoryError:
                # for deployed code (e.g., on a cloud computer)
                # might not have a git repo, so make sure to
                # deploy with current hash in environmental variable
                if os.getenv("RETRIEVER_GIT_HASH"):
                    git_hash = os.getenv("RETRIEVER_GIT_HASH")
                else:
                    print("No git repository found, cannot use version control.")
                    git_hash = None

            self.pars.git_hash = git_hash

        # make a catalog object based on the parameters:
        self.catalog = Catalog(**catalog_kwargs)
        self.catalog.load()
        self.pars.cat_hash = self.catalog.cat_hash

        self.observatories = NamedList(ignorecase=True)
        for obs in self.pars.obs_names:
            self.observatories.append(
                self._make_observatory(
                    name=obs,
                    inputs=obs_kwargs,
                )
            )

        # use the pars object's analysis_module and analysis_class
        # to initialize a custom analysis object
        self.analysis = self.pars.get_class_instance("analysis", **analysis_kwargs)
        self.analysis.observatories = self.observatories

        self.sources = None  # list to be filled by analysis
        self.failures_list = None  # list of failed sources
        self.num_sources_scanned = None

    def __setattr__(self, key, value):
        if key == "name" and value is not None:
            value = legalize(value)
        if key == "_test_hash":
            if hasattr(self, "analysis"):
                self.analysis._test_hash = value
            if hasattr(self, "observatories"):
                for obs in self.observatories:
                    obs._test_hash = value

        super().__setattr__(key, value)

    @staticmethod
    def _make_pars_object(kwargs):
        """
        Make the ParsProject object.
        When writing a subclass of this class
        that has its own subclassed Parameters,
        this function will allow the constructor
        of the superclass to instantiate the correct
        subclass Parameters object.
        """
        return ParsProject(**kwargs)

    @staticmethod
    def _get_observatory_classes(name):
        """
        Translate the name of the observatory
        into the module name, the class name
        and the class of the parameters object
        that goes in it.

        Parameters
        ----------
        name: str
            Name of the observatory.

        Returns
        -------
        module: module
            Module containing the observatory class.
        obs_class: class
            Class of the observatory object.
        pars_class: class
            Class of the parameters object.
        """
        if name.lower() in ["demo", "demoobs"]:  # this is built in to observatory.py
            module = "observatory"
            class_name = "VirtualDemoObs"
            pars_name = "ParsDemoObs"
        else:
            module = name.lower()
            class_name = f"Virtual{name.upper()}"
            pars_name = f"ParsObs{name.upper()}"
        module = importlib.import_module("." + module, package="src")
        obs_class = getattr(module, class_name)
        pars_class = getattr(module, pars_name)
        return module, obs_class, pars_class

    def _make_observatory(self, name, inputs):
        """
        Produce an Observatory object,
        use the name parameter to figure out
        what kind of subclass to use.
        Load configurations and apply them
        to the analysis objects for this observatory.

        Parameters
        ----------
        name: str
            Name of the observatory.
            The name is used to figure out
            the module and class name to use
            when creating the observatory.
            The module is interpreted as the
            lower case version of the name,
            while the class is built up of
            "Virtual"+name. For example,
            if name is "ZTF", the module
            is "ztf" and the class is "VirtualZTF".
            In special cases like "DemoObs",
            the class is loaded from the
            "observatory" module.
        inputs: dict
            Dictionary of keyword arguments
            to pass to the observatory class.
            Can contain keys matching the name
            of the observatory (e.g., "ztf", case insensitive)
            and these will be passed to the observatory
            and override any other arguments that
            are given to all observatories generally.

        """

        module, obs_class, _ = self._get_observatory_classes(name)

        if not isinstance(inputs, dict):
            raise TypeError(f'"inputs" must be a dictionary, not {type(inputs)}')

        new_obs = obs_class(**inputs)

        # TODO: separate reducer and use pars.get_class_instance to load it
        # parse parameters for reduction methods for this observatory
        # reducer_dict = {}
        # reducer_dict.update(self.pars.reduce_kwargsr)  # project pars
        # reducer_dict.update(new_obs.pars.reduce_kwargsr)  # observatory specific pars
        # new_obs.pars.reduce_kwargsr = reducer_dict

        # the catalog is just referenced from the project
        new_obs.catalog = self.catalog
        new_obs._test_hash = self._test_hash

        shorthand = legalize(name, to_lower=True)
        if not hasattr(self, shorthand):
            setattr(self, shorthand, new_obs)

        return new_obs

    def select_sources(self):
        """
        Generate a "select" statement that includes
        all sources associated with this project.
        This also filters only sources associated
        with the cfg_hash, if it defined for this project.
        """
        hash = self.cfg_hash if self.cfg_hash else ""
        stmt = sa.select(Source).where(
            Source.project == self.name, Source.cfg_hash == hash
        )
        return stmt

    def get_all_sources(self, session=None):
        """
        Get all sources associated with this project
        that have a corresponding row in the database.
        """
        with SmartSession(session) as session:

            hash = self.cfg_hash if self.cfg_hash else ""
            output = session.execute(
                sa.select(Source, Properties).where(
                    Source.project == self.name,
                    Source.cfg_hash == hash,
                    Properties.source_name == Source.name,
                )
            ).all()
            sources = []
            for out in output:
                source = out[0]
                source.properties = out[1]
                sources.append(source)

            return sources

    def select_detections(self):
        stmt = sa.select(Detection).where(
            Detection.project == self.name,
            Detection.cfg_hash == self.cfg_hash,
        )
        return stmt

    def get_detections(self, session=None):
        """
        Get all detections associated with this project.
        """
        with SmartSession(session) as session:
            return session.scalars(self.select_detections()).all()

    def delete_all_sources(
        self,
        remove_associated_data=False,
        remove_raw_data=False,
        remove_folder=True,
        remove_detections=False,
        session=None,
    ):
        """
        Delete all sources associated with this project.
        This includes all properties associated with each source.

        Parameters
        ----------
        remove_associated_data: bool
            If True, remove all data associated with this project.
            This includes reduced, processed and simulated data,
            of all types (e.g., photometry, spectra, images).
            This also removes data from disk, not just the database.
            Raw data is not removed from disk, as it is harder to recover,
            and is often shared between projects. Default is False.
        remove_raw_data: bool
            If True, remove all raw data associated with this project.
            This can include large amounts of data on disk, that can be
            shared with other projects, so it should be used with caution!
            Will only remove raw data associated with sources from this project.
            Default is False.
        remove_folder: bool
            If True, remove the folder(s) associated with this project.
            This includes the project output folder, and the raw data folder.
            Only removes folders that are empty.
            Default is True.
        remove_detections: bool
            If True, remove all detections associated with this project.
            This includes only detections associated with sources that are
            currently being removed from the project. Default is False.
        session: sqlalchemy.orm.session.Session
            Session to use for database queries.
            If None, a new session is created and closed at the end of the function.
        """
        with SmartSession(session) as session:

            sources = self.get_all_sources(session=session)

            raw_folders = set()
            obs_names = [obs.name for obs in self.observatories]
            hash = self.cfg_hash if self.cfg_hash else ""

            for source in sources:
                if remove_associated_data:  # lightcurves, images, etc.
                    for dt in self.pars.data_types:
                        # the reduced level class is the same for processed/simulated
                        DataClass = get_class_from_data_type(dt, level="reduced")
                        data = session.scalars(
                            sa.select(DataClass).where(
                                DataClass.source_name == source.name,
                                DataClass.project == self.name,
                                DataClass.cfg_hash == hash,
                                DataClass.observatory.in_(obs_names),
                            )
                        ).all()
                        for d in data:
                            d.delete_data_from_disk()
                            session.delete(d)

                        # remove folder if empty
                        if remove_folder:
                            if not os.listdir(self.output_folder):
                                os.rmdir(self.output_folder)

                        # check if we also need to remove raw data
                        if remove_raw_data:
                            DataClass = get_class_from_data_type(dt, level="raw")
                            data = session.scalars(
                                sa.select(DataClass).where(
                                    DataClass.source_name == source.name,
                                    DataClass.observatory.in_(obs_names),
                                )
                            )
                            for d in data:
                                raw_folders.add(d.get_path())
                                d.delete_data_from_disk()
                                session.delete(d)

                if remove_detections:
                    session.execute(
                        sa.delete(Detection).where(
                            Detection.source_name == source.name,
                            Detection.project == self.name,
                            Detection.cfg_hash == hash,
                        )
                    )

                session.delete(source)

            session.commit()

            self.sources = []
            for obs in self.observatories:
                obs.sources = []
                obs.raw_data = []

            if remove_associated_data and remove_raw_data and remove_folder:
                # remove any empty raw-data folders
                for folder in list(raw_folders):
                    if os.path.isdir(folder) and not os.listdir(folder):
                        os.rmdir(folder)

    def delete_project_files(self, remove_folder=True):
        """
        Delete all outputs associated with this project.

        This includes the output config file and histograms.
        The output folder is only deleted if it is empty.
        Usually, it would be empty only if also called
        `delete_all_sources(remove_associated_data=True)`.

        """
        cfg_file = os.path.join(self.output_folder, "config.yaml")
        if os.path.exists(cfg_file):
            os.remove(cfg_file)

        # delete all histograms
        for h in self.analysis.get_all_histograms():
            h.remove_data_from_file(remove_backup=True)
            h.initialize()

        if remove_folder:
            # remove the output folder if it is empty
            if not os.listdir(self.output_folder):
                os.rmdir(self.output_folder)

    def delete_everything(
        self, remove_raw_data=False, remove_folder=True, session=None
    ):
        """
        Delete all data associated with this project.
        This includes sources, detections, data products (e.g., lightcurves),
        histogram files, output config files, and the output folder (optional).
        Will remove all reduced/processed/simulated lightcurves, even if the
        sources they are associated with are already deleted.

        The only exception is that raw data is only removed if the source
        associated with it is still in the project when this is called.
        And even that, only if `remove_raw_data=True`.

        Parameters
        ----------
        remove_raw_data: bool
            If True, remove all raw data associated with this project.
            Only removes raw data associated with sources that are still
            in the project. Default is False.
        remove_folder: bool
            If True, remove the folder(s) associated with this project.
            Only removes empty folders. Default is True.
        session: sqlalchemy.orm.session.Session
            Session to use for database queries.
            If None, a new session is created and closed at the end of the function.

        """
        with SmartSession(session) as session:

            self.delete_all_sources(
                remove_associated_data=True,
                remove_raw_data=remove_raw_data,
                remove_folder=remove_folder,
                remove_detections=True,
                session=session,
            )

            session.execute(
                sa.delete(Lightcurve).where(
                    Lightcurve.project == self.name,
                    Lightcurve.cfg_hash == self.cfg_hash,
                )
            )
            # TODO: add additional data types here

            # remove histogram files and config file
            self.delete_project_files(remove_folder=remove_folder)

    def reset(self):
        self.sources = CircularBufferList(self.pars.source_buffer_size)
        self.failures_list = []
        self.num_sources_scanned = 0

    def run(self, start=0, finish=None, source_names=None, source_ids=None):
        """
        Run the full pipeline on each source in the catalog.

        For each source, will try to find a DB row with that name.
        If it doesn't exist, it will create one. If it already exists
        and has a Properties object associated with it,
        then the source has already been analyzed and the pipeline
        will skip it.
        Assuming a photometric analysis (it is the same for images, etc.):
        For each source that has not yet been analyzed,
        will look for RawPhotometry objects (for each observatory and source name).
        If there is a RawPhotometry row but the data is missing on disk,
        it could raise a RuntimeError or simply skip that source if
        pars.ignore_missing_raw_data is True.
        If no RawPhotometry exists, it will download the raw data
        using the observatory.
        For each RawPhotometry that is found or downloaded, will look
        for reduced datasets (Lightcurves) matching that raw data.
        If the reduced data exists on DB and on disk it will be used.
        If it is missing on either DB or disk, it will be re-reduced
        by the observatory object.
        After all RawPhotometry objects from all observatories are reduced
        the data is transferred to the Analysis object.
        This will (re-)create processed lighturves,
        which are used to make detections and properties objects,
        both of which are saved to the DB.
        The processed lightcurves are optionally given to the
        Simulator objects, which adds simulated lightcurves a
        and simulated detections, which are saved to DB.
        If the data for this source has not been added to the
        histograms, they will be updated as well.

        Parameters
        ----------
        start: int
            Index of the first source to analyze.
        finish: int
            Index of the last source, which is not included.
            So start=0, finish=10 will analyze sources 0-9.
        source_names: list of str
            List of source names to analyze.
            Can also give a single source name.
            If there's no overlap with the other
            arguments, nothing will be analyzed.
        source_ids: list of int
            List of source IDs to analyze.
            Can also give a single source ID.
            If there's no overlap with the other
            arguments, nothing will be analyzed.
        """
        import warnings
        from astroquery.exceptions import NoResultsWarning

        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", NoResultsWarning)
        self._save_config()

        if finish is None:
            finish = len(self.catalog.names)

        if source_names is not None:
            if isinstance(source_names, str):
                source_names = [source_names]

            if not isinstance(source_names, list):
                raise TypeError(
                    f"source_names must be a list of strings. Got {type(source_names)}"
                )

        if source_ids is not None:
            if isinstance(source_ids, int):
                source_ids = [source_ids]

            if not isinstance(source_ids, list):
                raise TypeError(
                    f"source_ids must be a list of integers. Got {type(source_ids)}"
                )

        types = self.pars.data_types
        if isinstance(types, str):
            types = [types]

        self.reset()

        for obs in self.observatories:
            obs.reset()

        hash = self.cfg_hash if self.cfg_hash else ""

        with SmartSession() as session:

            num_exceptions = 0
            num_exceptions_in_a_row = 0
            source_batch = []
            try:  # not matter what happens, save the batch at end
                for i, name in enumerate(self.catalog.names):
                    # only run sources in range
                    if i < start or i >= finish:
                        continue

                    # only run sources in source_names
                    if source_names is not None and name not in source_names:
                        continue

                    # only run sources in source_ids
                    if source_ids is not None and i not in source_ids:
                        continue

                    try:  # log any exceptions instead of stopping

                        # look for the source in the database
                        source = session.scalars(
                            sa.select(Source)
                            .where(
                                Source.name == name,
                                Source.project == self.name,
                                Source.cfg_hash == hash,
                            )
                            .options(joinedload(Source.properties))
                        ).first()

                        if source is None or source.properties is None:
                            cat_row = self.catalog.get_row(name, "name", "dict")

                            need_skip = False
                            # check the source has the needed data
                            for obs in self.observatories:
                                for dt in types:
                                    if need_skip:
                                        continue

                                    # this fetches all the data types
                                    source = obs.fetch_source(
                                        cat_row,
                                        source=source,
                                        save=True,
                                        session=session,
                                    )
                                    data = source.get_data(
                                        obs.name,
                                        data_type=dt,
                                        level="raw",
                                        session=session,
                                        check_data=False,
                                        append=True,
                                    )
                                    if len(data) == 1:
                                        data = data[0]
                                    else:
                                        raise RuntimeError(
                                            f'Found multiple raw {dt} for obs "{obs.name}" on source "{name}".'
                                        )

                                    # check dataset has data on disk/in memory
                                    # (if not, skip entire source or raise RuntimeError)
                                    if (
                                        data is not None
                                        and not data.check_data_exists()
                                    ):
                                        if self.pars.ignore_missing_raw_data:
                                            need_skip = True
                                            continue
                                        else:
                                            raise RuntimeError(
                                                f"RawData for source {source.name} and observatory {obs.name} "
                                                "is missing from disk. "
                                                "Set pars.ignore_missing_raw_data=True to ignore this."
                                            )

                                    # look for reduced data and reproduce if missing
                                    lcs = source.get_data(
                                        obs=obs.name,
                                        data_type=dt,
                                        level="reduced",
                                        session=session,
                                        check_data=False,
                                        append=True,
                                    )
                                    if len(lcs) == 0:  # reduce data
                                        obs.reduce(data, to="lcs")

                                # TODO: should we change this to "raw_photometry" or something?
                                obs.raw_data.append(data)

                            if source is None:
                                raise ValueError(f"Source {name} could not be found!")

                            # finished looping on observatories and data types
                            source_batch.append(source)
                            # reset count upon successful load/analysis
                            num_exceptions_in_a_row = 0

                        # make sure to append this source
                        self.sources.append(source)

                    except Exception as e:
                        self.pars.vprint(f"Error processing source {name}: {e}")
                        self.failures_list.append(
                            dict(index=i, error=traceback.format_exc(), cat_row=cat_row)
                        )
                        num_exceptions += 1
                        num_exceptions_in_a_row += 1

                        if (
                            num_exceptions_in_a_row
                            >= self.pars.max_num_sequence_exceptions
                        ):
                            print(
                                f"Too many exceptions in a row. Stopping after {num_exceptions_in_a_row} exceptions."
                            )
                            raise e

                        if num_exceptions >= self.pars.max_num_total_exceptions:
                            print(
                                f"Too many exceptions. Stopping after {num_exceptions} exceptions."
                            )
                            raise e

                    # count the sources processed and not
                    self.num_sources_scanned += 1
                    obs.sources.append(source)

                    if len(source_batch) >= self.pars.source_batch_size:
                        # send source with all data to analysis object for
                        # finding detections / adding properties
                        self.analysis.analyze_sources(source_batch)
                        session.add_all(source_batch)
                        session.commit()

                        source_batch = []

                # end of sources loop

            finally:  # save the batch
                self.analysis.analyze_sources(source_batch)
                session.add_all(source_batch)
                session.commit()

    def _save_config(self):
        """
        Save the configuration file to disk.
        This is the point where the project
        gets a dedicated output folder.

        Also calculate a cfg_hash if using
        version control.
        If not, will just set cfg_hash=""
        """

        # pick up all the config keys from all the objects
        pars_list = self.pars.get_pars_list(self)
        cfg_dict = {}

        for pars in pars_list:
            if pars._cfg_key is None:
                continue
            if pars._cfg_sub_key is None:
                cfg_dict[pars._cfg_key] = pars.to_dict(hidden=False)
            else:
                if pars._cfg_key not in cfg_dict:
                    cfg_dict[pars._cfg_key] = {}
                cfg_dict[pars._cfg_key][pars._cfg_sub_key] = pars.to_dict(hidden=False)

        cfg_json = json.dumps(cfg_dict)

        if self.pars.version_control:
            self.cfg_hash = hashlib.sha256(
                "".join(cfg_json).encode("utf-8")
            ).hexdigest()
        else:
            self.cfg_hash = ""

        for obs in self.observatories:
            obs.cfg_hash = self.cfg_hash

        self.catalog.cfg_hash = self.cfg_hash

        self._setup_output_folder()

        # write the config file to disk
        with open(os.path.join(self.output_folder, "config.yaml"), "w") as f:
            yaml.dump(cfg_dict, f, sort_keys=False)

    def _setup_output_folder(self):
        """
        Create a folder for the output of this project.
        This will include all the reduced and analyzed data,
        the histograms, and so on.

        It also includes a copy of the configuration file,
        with the final adjustments made by e.g., the user
        upon initialization or just before running the pipeline.

        If version control is enabled, the cfg hash is calculated
        for the full OUTPUT config file, and the hash is used to tag
        all DB objects and is appended to the output folder name.

        """
        self.output_folder = self.name.upper()

        # version control is enabled
        if self.pars.version_control:
            self.output_folder += f"_{self.cfg_hash}"

        self.output_folder = os.path.join(src.database.DATA_ROOT, self.output_folder)

        # create the output folder
        safe_mkdir(self.output_folder)

        self.analysis.output_folder = self.output_folder

    def help(self=None, owner_pars=None):
        """
        Print the help for this object and objects contained in it.
        """

        if isinstance(self, Project):
            help_with_object(self, owner_pars)
        elif self is None or self == Project:
            cls = Project
            subclasses = [Catalog, Analysis]

            for obs_name in ParsObservatory.allowed_obs_names:
                _, class_name, pars_name = cls._get_observatory_classes(obs_name)
                subclasses.append(class_name)

            help_with_class(
                cls,
                ParsProject,
                subclasses,
            )


if __name__ == "__main__":
    # proj = Project(name="tess_wds")
    # proj.run()
    from pprint import pprint
    from src.utils import sanitize_attributes
    from src.database import Session

    session = Session()
    rp = session.scalars(
        sa.select(RawPhotometry).where(RawPhotometry.number > 0)
    ).first()
    s = session.scalars(sa.select(Source).where(Source.name == rp.source_name)).first()
