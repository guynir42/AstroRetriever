import importlib
import os

import sqlalchemy as sa

from src.database import Session
from src.parameters import Parameters
from src.catalog import Catalog
from src.source import Source
from src.analysis import Analysis


class Project:
    def __init__(self, name="default", params=None, obs_params=None, config=True):
        """
        Create a new Project object.
        The project is used to combine a catalog,
        some observatories, and analysis objects.
        It allows loading of parameters saving of results
        all under one object.
        Each project should be used for one science case,
        with a set of sources given by the catalog,
        a set of observations given by the observatories,
        and the calibration and analysis to be done
        on the data from each observatory.

        Parameters
        ----------
        name: str
            Name of the project.
            Will be used as default config filename,
            as default output filenames, and so on.
        params: dict
            Dictionary of parameters for the project.
            These should override any parameters
            in the config file.
            To configure, e.g., the calibration,
            use params={'calibration': {'cal_parameter1': 'value1', ...}}.
            To choose a list of observatories,
            use params={'observatories': ['ZTF', 'TESS', ...]}.
        obs_params: dict
            Dictionary of parameters for the observatories.
            Each key should be the name of an observatory,
            and any parameters given would override those
            given in the config file (if any).
        config: str or bool
            Name of the file to load.
            If False or None, no config file will be loaded.
            If True but not a string, will
            default to "configs/<project-name>.yaml"
            If a non-empty string is given,
            it will be used as the config filename.
        """
        self.name = name
        self.pars = Parameters(
            required_pars=[
                "observatories",
                "reducer",
                "analysis",
                "catalog",
            ]
        )
        # default values in case no config file is loaded:
        self.pars.observatories = set()  # set of observatory names
        self.pars.catalog = {}  # parameters for the catalog
        self.pars.reducer = {}  # global reducer pars for all observatories
        self.pars.analysis = {}  # global analysis parameters for all observatories

        # load the parameters from the config file:
        if config:  # note that empty string is also False!
            if isinstance(config, str):
                filepath = config
            else:
                basepath = os.path.dirname(__file__)
                filename = os.path.join("../configs", f"{self.name}.yaml")
                filepath = os.path.abspath(os.path.join(basepath, filename))
            self.pars.load(filepath, "project")

        # add any additional user inputs IN ADDITION to the config file:
        if params is not None:
            if not isinstance(params, dict):
                raise TypeError(f"params must be a dictionary, not {type(params)}")
            self.pars.update(params)

        # if only one observatory is given, make it a set:
        if isinstance(self.pars.observatories, str):
            self.pars.observatories = {self.pars.observatories}

        # verify that the list of observatory names is castable to a set
        if not isinstance(self.pars.observatories, (set, list, tuple)):
            raise TypeError(
                f"observatories must be a set, list, or tuple, not {type(self.pars.observatories)}"
            )

        # cast the list into a set
        self.pars.observatories = set(self.pars.observatories)

        # if no observatories are given, use the demo
        if len(self.pars.observatories) == 0:
            self.pars.observatories = {"DemoObs"}

        if not all([isinstance(obs, str) for obs in self.pars.observatories]):
            raise TypeError("observatory_names must be a set of strings")

        self.pars.verify()  # make sure all parameters are set

        # make a catalog object based on the parameters:
        if len(self.pars.catalog) == 0:
            self.pars.catalog = {"default": "test"}
        self.catalog = Catalog(**self.pars.catalog, verbose=self.pars.verbose)
        self.catalog.load()

        # make observatories:
        if obs_params is None:
            obs_params = {}

        self.observatories = [
            self.make_observatory(
                name=obs, params=obs_params.get(obs, {}), config=config
            )
            for obs in self.pars.observatories
        ]
        self.observatories = {obs.name: obs for obs in self.observatories}

    def make_observatory(self, name, params={}, config=None):
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
        params: dict
            Dictionary of parameters for the observatory.
            These would override any parameters
            loaded from the config file.
            If none, no parameters will be loaded
            after the config file (if any).
        config: str or bool
            Name of the file to load.
            If False or None, no config file will be loaded.
            If True but not a string, will
            default to "configs/<project-name>.yaml"
            If a non-empty string is given,
            it will be used as the config filename.

        """

        if name == "DemoObs":  # this is built in to observatory.py
            module_name = "observatory"
        else:
            module_name = name.lower()

        if not isinstance(params, dict):
            raise TypeError(f"params must be a dictionary, not {type(params)}")

        module = importlib.import_module("." + module_name, package="src")
        obs_class = getattr(module, f"Virtual{name}")
        new_obs = obs_class(project=self.name, config=config, **params)

        # parse parameters for reduction methods for this observatory
        reducer_dict = {}
        reducer_dict.update(self.pars.reducer)  # project pars
        reducer_dict.update(new_obs.pars.reducer)  # observatory specific pars
        new_obs.pars.reducer = reducer_dict

        # don't think we need analysis inside observatory anymore
        # # parse parameters for analysis of this observatory
        # new_obs.analysis.pars.update(self.pars.analysis)  # project pars
        # new_obs.analysis.pars.update(new_obs.pars.analysis)  # observatory pars
        # new_obs.analysis.pars.verify()
        # new_obs.analysis.initialize()

        new_obs.pars.verbose = self.pars.verbose

        # the catalog is just referenced from the project
        new_obs.catalog = self.catalog

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


if __name__ == "__main__":
    print("Starting a new project")
    proj = Project(
        name="WD",
        params={
            "observatories": "ZTF",  # a single observatory named ZTF
            "catalog": {"default": "WD"},  # load the default WD catalog
            "verbose": 1,  # print out some info
        },
        obs_params={
            "ZTF": {  # instructions for ZTF specifically
                "data_glob": "lightcurves_WD*.h5",  # filename format
                "catalog_matching": "number",  # match by catalog row number
                "dataset_identifier": "key",  # use key (HDF5 group name) as identifier
            }
        },
        config=False,
    )

    proj.delete_all_sources()
    proj.observatories["ztf"].populate_sources(num_files=1, num_sources=3)
    sources = proj.get_all_sources()
    print(
        f'Database contains {len(sources)} sources associated with project "{proj.name}"'
    )
    # for source in sources:
    #     for lc in source.lightcurves:
    #         lc.delete_data_from_disk()
