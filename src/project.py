import importlib
import os

import git

import sqlalchemy as sa

from src.database import Session
from src.parameters import Parameters
from src.catalog import Catalog
from src.source import Source
from src.analysis import Analysis

from src.database import DATA_ROOT


class Project:
    def __init__(self, name="default", **kwargs):
        """
        Create a new Project object.
        The project is used to combine a catalog,
        some observatories, and analysis objects.
        It allows loading of parameters saving of results
        all under one object.
        Each project should be used for one science case,
        with a set of sources given by the catalog,
        a set of observations given by the obs_names list,
        and the reduction and analysis to be done
        on the data from each observatory.

        Parameters
        ----------
        name: str
            Name of the project.
            Will be used as default config filename,
            as default output filenames, and so on.
        Additional arguments are passed into the Parameters object.
        """
        self.name = name
        # this loads parameters from file, then from kwargs:
        kwargs["project"] = name  # propagate this to sub-objects
        self.pars = Parameters.from_dict(kwargs, "project")

        # default values in case no config file is loaded:
        self.pars.default_values(
            obs_names=["demo"],
            obs_kwargs={},  # parameters for all observatories
            catalog_kwargs={"default": "test"},
            version_control=False,
        )

        # filled by setup_output_folder at runtime:
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

        # if only one observatory name is given, make it a set:
        if isinstance(self.pars.obs_names, str):
            self.pars.obs_names = {self.pars.obs_names}

        # verify that the list of observatory names is castable to a set
        if not isinstance(self.pars.obs_names, (set, list, tuple)):
            raise TypeError(
                f"obs_names must be a set, list, or tuple, not {type(self.pars.obs_names)}"
            )

        # cast the list into a set
        self.pars.obs_names = set(self.pars.obs_names)

        if not all([isinstance(obs, str) for obs in self.pars.obs_names]):
            raise TypeError("observatories must be a set of strings")

        self.pars.verify()  # make sure all parameters are set

        # make a catalog object based on the parameters:
        self.catalog = Catalog(**self.pars.catalog_kwargs, verbose=self.pars.verbose)
        self.catalog.load()

        general_kwargs = self.pars.obs_kwargs
        self.pars.add_defaults_to_dict(general_kwargs)

        self.observatories = NamedList(ignorecase=True)
        for obs in self.pars.obs_names:
            specific_kwargs = getattr(self.pars, obs, {})
            specific_kwargs.update({"project": self.name})

            self.observatories.append(
                self.make_observatory(
                    name=obs,
                    specific=specific_kwargs,
                    general=general_kwargs,
                )
            )

        self.analysis = self.pars.get_class("analysis")

    def make_observatory(self, name, specific, general):
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
        specific: dict
            Parameters that are specific to this observatory.
            The parameters passed into the constructor override
            those from file, both of which override the pars
            in "general".
        general: dict
            General parameters that are true for all observatories.
            Any of these pars are overriden by observatory-specific
            parameters, either loaded from file or given in "specific".
        """

        if name in ["demo", "DemoObs"]:  # this is built in to observatory.py
            module_name = "observatory"
            class_name = "VirtualDemoObs"
        else:
            module_name = name.lower()
            class_name = f"Virtual{name}"

        if not isinstance(specific, dict):
            raise TypeError(f'"specific" must be a dictionary, not {type(specific)}')

        if not isinstance(general, dict):
            raise TypeError(f'"general" must be a dictionary, not {type(general)}')

        module = importlib.import_module("." + module_name, package="src")
        obs_class = getattr(module, class_name)
        new_obs = obs_class(**specific)

        # only override any parameters not set
        # from the config file or the "specific" dict:
        new_obs.pars.replace_unset(**general)

        # TODO: separate reducer and use pars.get_class to load it
        # parse parameters for reduction methods for this observatory
        # reducer_dict = {}
        # reducer_dict.update(self.pars.reducer)  # project pars
        # reducer_dict.update(new_obs.pars.reducer)  # observatory specific pars
        # new_obs.pars.reducer = reducer_dict

        # the catalog is just referenced from the project
        new_obs.catalog = self.catalog

        new_obs.initialize()

        return new_obs

    def get_all_sources(self):
        """
        Get all sources from all observatories.
        """
        stmt = sa.select(Source).where(Source.project == self.name)
        with Session() as session:
            sources = session.scalars(stmt).all()
        return sources

    def delete_all_sources(self):
        """
        Delete all sources associated with this project.
        """
        stmt = sa.delete(Source).where(Source.project == self.name)
        with Session() as session:
            session.execute(stmt)
            session.commit()

    def setup_output_folder(self):
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

        # TODO: collect all parameter objects from sub-objects
        # and produce a massive config dictionary
        # translate that into a yaml file in memory
        # get that file's hash

        # version control is enabled
        if self.pars.version_control:
            self.output_folder += f"_{self.cfg_hash}"

        self.output_folder = os.path.join(DATA_ROOT, self.output_folder)

        # create the output folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # TODO: write the config file to the output folder

    # def download(self, **kwargs):
    #     pass
    #
    # def reduce(self, **kwargs):
    #     pass
    #
    # def analyze(self, **kwargs):
    #     pass

    def run(self, **kwargs):
        pass


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
    print("Starting a new project")
    proj = Project(
        name="WD",
        obs_names="ZTF",  # a single observatory named ZTF
        catalog={"default": "WD"},  # load the default WD catalog
        verbose=1,  # print out some info
        obs_kwargs={},  # general parameters to pass to all observatories
        ZTF={  # instructions for ZTF specifically
            "data_glob": "lightcurves_WD*.h5",  # filename format
            "catalog_matching": "number",  # match by catalog row number
            "dataset_identifier": "key",  # use key (HDF5 group name) as identifier
        },
    )

    # proj.delete_all_sources()
    # proj.observatories["ztf"].populate_sources(num_files=1, num_sources=3)
    # sources = proj.get_all_sources()
    # print(
    #     f'Database contains {len(sources)} sources associated with project "{proj.name}"'
    # )
    # for source in sources:
    #     for lc in source.lightcurves:
    #         lc.delete_data_from_disk()
