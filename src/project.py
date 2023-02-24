"""
The project is used to combine a catalog,
some observatories, and analysis objects.
"""

import importlib
import os
import json
import yaml
import hashlib
import git

import numpy as np

import sqlalchemy as sa

import src.database
from src.database import Session, CloseSession
from src.parameters import Parameters, get_class_from_data_type
from src.catalog import Catalog
from src.observatory import ParsObservatory
from src.source import Source
from src.dataset import RawPhotometry, Lightcurve
from src.analysis import Analysis
from src.properties import Properties
from src.utils import help_with_class, help_with_object


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
        self.pars = ParsProject(**kwargs)

        # add some default keys like "project" and "verbose" to kwargs
        self.pars.add_defaults_to_dict(obs_kwargs)
        self.pars.add_defaults_to_dict(catalog_kwargs)
        self.pars.add_defaults_to_dict(analysis_kwargs)

        # filled by _setup_output_folder at runtime:
        self.output_folder = None
        self.cfg_hash = None  # hash of the config file (for version control)

        # version control:
        if self.pars.version_control:
            try:
                repo = git.Repo(search_parent_directories=True)
                git_hash = repo.head.object.hexsha
            except git.exc.InvalidGitRepositoryError:
                # for deployed code (e.g., on a cloud computer)
                # might not have a git repo, so make sure to
                # deploy with current hash in environmental variable
                if os.getenv("VO_GIT_HASH"):
                    git_hash = os.getenv("VO_GIT_HASH")
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
        # reducer_dict.update(self.pars.reducer)  # project pars
        # reducer_dict.update(new_obs.pars.reducer)  # observatory specific pars
        # new_obs.pars.reducer = reducer_dict

        # the catalog is just referenced from the project
        new_obs.catalog = self.catalog

        return new_obs

    def get_all_sources(self, session=None):
        """
        Get all sources associated with this project
        that have a corresponding row in the database.
        """
        # TODO: add cfg_hash to this
        stmt = sa.select(Source).where(Source.project == self.name)

        if session is None:
            session = Session()
            # make sure this session gets closed at end of function
            _ = CloseSession(session)

        sources = session.scalars(stmt).all()

        return sources

    def delete_all_sources(
        self, remove_all_data=False, remove_raw_data=False, session=None
    ):
        """
        Delete all sources associated with this project.

        Parameters
        ----------
        remove_all_data: bool
            If True, remove all data associated with this project.
            This includes reduced, processed and simulated data,
            of all types (e.g., photometry, spectra, images).
            This also removes data from disk, not just the database.
            Raw data is not removed from disk, as it is harder to recover,
            and is often shared between projects.
        remove_raw_data: bool
            If True, remove all raw data associated with this project.
            This can include large amounts of data on disk, that can be
            shared with other projects, so it should be used with caution!
            Will only remove raw data associated with sources from this project.
        """
        if session is None:
            session = Session()
            # make sure this session gets closed at end of function
            _ = CloseSession(session)

        # TODO: add cfg_hash to this
        sources = session.scalars(
            sa.select(Source).where(Source.project == self.name)
        ).all()

        if remove_all_data:
            for source in sources:
                for dt in self.pars.data_types:
                    # the reduced level class is the same for processed/simulated
                    DataClass = get_class_from_data_type(dt, level="reduced")
                    data = session.scalars(
                        sa.select(DataClass).where(DataClass.source_id == source.id)
                    ).all()
                    for d in data:
                        d.delete_data_from_disk()
                        session.delete(d)
                    if remove_raw_data:
                        # only remove data in a relationship with source
                        for d in getattr(source, f"raw_{dt}"):
                            d.delete_data_from_disk()
                            session.delete(d)
                session.delete(source)

        session.commit()

    def run(self, **kwargs):
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
        kwargs: additional arguments to use such as...

        """

        self._save_config()

        source_names = self.catalog.names
        types = self.pars.data_types
        if isinstance(types, str):
            types = [types]

        with Session() as session:
            # TODO: add batching of sources
            for name in source_names:

                source = session.scalars(
                    sa.select(Source).where(
                        Source.name == name,
                        Source.project == self.name,
                        Source.cfg_hash == self.cfg_hash,
                    )
                ).first()

                # # need to generate a new source
                # if source is None:
                #     cat_row = self.catalog.get_row(name, "name", "dict")
                #     source = Source(
                #         project=self.name,
                #         cfg_hash=self.cfg_hash,
                #         **cat_row,
                #     )

                # check if source has already been processed (has properties)
                if source is not None and source.properties is not None:
                    continue

                # no source found, need to make it (and download data maybe?)
                cat_row = self.catalog.get_row(name, "name", "dict")

                need_skip = False
                for obs in self.observatories:
                    if need_skip:
                        continue

                    # try to download the data for this observatory
                    source = obs.check_and_fetch_source(
                        cat_row, save=False, session=session
                    )

                    for dt in types:
                        if need_skip:
                            continue

                        if dt == "photometry":
                            # # check if raw data is attached to this source
                            # data = [
                            #     rd
                            #     for rd in source.raw_photometry
                            #     if rd.observatory == obs.name
                            # ]
                            #
                            # # if not, look around the DB for any raw data
                            # # with a matching name and observatory
                            # # (e.g., from another project or version)
                            # if len(data) == 0:
                            #     loaded_data = session.scalars(
                            #         sa.select(RawPhotometry).where(
                            #             RawPhotometry.source_name == name,
                            #             RawPhotometry.observatory == obs.name,
                            #         )
                            #     ).all()
                            #     source.raw_photometry += loaded_data

                            # did we find any raw photometry?
                            data = [
                                rp
                                for rp in source.raw_photometry
                                if rp.observatory == obs.name
                            ]

                            if len(data) == 0:
                                raise ValueError(
                                    f"Could not find any raw photometry on source {name}"
                                )

                            if len(data) > 1:
                                raise ValueError(
                                    "Each source should have only one "
                                    "RawPhotometry associated with each observatory. "
                                    f"For source {name} and observatory {obs.name}, "
                                    f"found {len(data)} RawPhotometry objects."
                                )
                            data = data[0]

                            # check dataset has data on disk/in memory
                            # (if not, skip entire source or raise RuntimeError)
                            if not data.check_data_exists():
                                if self.pars.ignore_missing_raw_data:
                                    need_skip = True
                                    continue
                                else:
                                    raise RuntimeError(
                                        f"RawData for source {source.name} and observatory {obs.name} "
                                        "is missing from disk. "
                                        "Set pars.ignore_missing_raw_data to True to ignore this."
                                    )

                            # look for reduced data and reproduce if missing
                            lcs = [
                                lc
                                for lc in source.reduced_lightcurves
                                if lc.observatory == obs.name
                            ]
                            if len(lcs) == 0:  # try to load from DB
                                lcs = session.scalars(
                                    sa.select(Lightcurve).where(
                                        Lightcurve.source_name == name,
                                        Lightcurve.observatory == obs.name,
                                        Lightcurve.project == self.name,
                                        Lightcurve.cfg_hash == self.cfg_hash,
                                        Lightcurve.was_processed.is_(False),
                                    )
                                ).all()
                            if len(lcs) == 0:  # reduce data
                                lcs = obs.reduce(data, to="lcs")

                            source.reduced_lightcurves += lcs

                        # add additional elif for other types...
                        else:
                            raise ValueError(f"Unknown data type: {dt}")

                # finished looping on observatories and data types

                # send source with all data to analysis object for
                # finding detections / adding properties
                self.analysis.analyze_sources(source)

                session.add(source)
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
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

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


class NamedList(list):
    """
    A list of objects, each of which has
    a "name" attribute.
    This list can be indexed by name,
    and also using numerical indices.
    """

    def __init__(self, ignorecase=False):
        self.ignorecase = ignorecase
        super().__init__()

    def convert_name(self, name):
        if self.ignorecase:
            return name.lower()
        else:
            return name

    def __getitem__(self, index):
        if isinstance(index, str):
            index_l = self.convert_name(index)
            num_idx = [self.convert_name(obs.name) for obs in self].index(index_l)
            return super().__getitem__(num_idx)
        elif isinstance(index, int):
            return super().__getitem__(index)
        else:
            raise TypeError(f"index must be a string or integer, not {type(index)}")

    def __contains__(self, name):
        return self.convert_name(name) in [self.convert_name(obs.name) for obs in self]

    def keys(self):
        return [obs.name for obs in self]


if __name__ == "__main__":

    src.database.DATA_ROOT = "/home/guyn/Dropbox/DATA"

    # proj = Project(
    #     name="default_test",  # must give a name to each project
    #     description="my project description",  # optional parameter example
    #     version_control=False,  # whether to use version control on products
    #     obs_names=["ZTF"],  # must give at least one observatory name
    #     # parameters to pass to the Analysis object:
    #     analysis_kwargs={
    #         "num_injections": 3,
    #         "finder_kwargs": {  # pass through Analysis into Finder
    #             "snr_threshold": 7.5,
    #         },
    #         "finder_module": "src.finder",  # can choose different module
    #         "finder_class": "Finder",  # can choose different class (e.g., MyFinder)
    #     },
    #     analysis_module="src.analysis",  # replace this with your code path
    #     analysis_class="Analysis",  # replace this with you class name (e.g., MyAnalysis)
    #     catalog_kwargs={"default": "WD"},  # load the default WD catalog
    #     # parameters to be passed into each observatory class
    #     obs_kwargs={
    #         "reducer": {
    #             "radius": 3,
    #             "gap": 40,
    #         },
    #         "ZTF": {  # specific instructions for the ZTF observatory only
    #             "credentials": {
    #                 "username": "guy",
    #                 "password": "12345",
    #             },
    #         },
    #     },
    #     verbose=True,
    # )
    proj = Project(
        name="default_test",
        obs_names=["demo"],
        analysis_kwargs={"num_injections": 3},
        obs_kwargs={},
        catalog_kwargs={"default": "test"},
        verbose=0,
    )
    # download all data for all sources in the catalog
    # and reduce the data (skipping raw and reduced data already on file)
    # and store the results as detection objects in the DB, along with
    # detection stats in the form of histogram arrays.
    proj.run()

    # Project.help()
    # proj.help()

    # proj.delete_all_sources()
    # proj.catalog = proj.catalog.make_smaller_catalog(range(20))
    # proj.run()

    # proj.observatories["ztf"].populate_sources(num_files=1, num_sources=3)
    # sources = proj.get_all_sources()
    # print(
    #     f'Database contains {len(sources)} sources associated with project "{proj.name}"'
    # )
    # for source in sources:
    #     for lc in source.lightcurves:
    #         lc.delete_data_from_disk()
