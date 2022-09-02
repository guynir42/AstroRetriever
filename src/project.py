import importlib
import os

import sqlalchemy as sa

from src.database import Session
from src import parameters
from src.catalog import Catalog
from src.source import Source
from src.analysis import Analysis


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
        self.pars = parameters.from_dict(kwargs, "project")
        self.pars.project = self.name  # propagate this to sub-objects

        # default values in case no config file is loaded:
        self.pars.default_values(
            obs_names=["demo"],
            obs_kwargs={},  # parameters for all observatories
            catalog_kwargs={"default": "test"},
        )

        # if only one observatory is given, make it a set:
        if isinstance(self.pars.obs_names, str):
            self.pars.observatories = {self.pars.observatories}

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

        self.observatories = []
        for obs in self.pars.obs_names:
            self.observatories.append(
                self.make_observatory(
                    name=obs,
                    specific=getattr(self.pars, obs, {}),
                    general=self.pars.obs_kwargs,
                )
            )

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
        else:
            module_name = name.lower()

        if not isinstance(specific, dict):
            raise TypeError(f'"specific" must be a dictionary, not {type(specific)}')

        if not isinstance(general, dict):
            raise TypeError(f'"general" must be a dictionary, not {type(general)}')

        module = importlib.import_module("." + module_name, package="src")
        obs_class = getattr(module, f"Virtual{name}")
        new_obs = obs_class(**specific)

        # only override any parameters not set
        # from the config file or the "specific" dict:
        new_obs.pars.replace_unset(**general)

        # TODO: separate reducer and use pars.get_class to load it
        # parse parameters for reduction methods for this observatory
        reducer_dict = {}
        reducer_dict.update(self.pars.reducer)  # project pars
        reducer_dict.update(new_obs.pars.reducer)  # observatory specific pars
        new_obs.pars.reducer = reducer_dict

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

    def download(self, **kwargs):
        pass

    def reduce(self, **kwargs):
        pass

    def analyze(self, **kwargs):
        pass


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
