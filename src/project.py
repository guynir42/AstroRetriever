import importlib
import os

from src.parameters import Parameters
from src.analysis import Analysis
from src.catalog import Catalog
from src.calibration import Calibration


class Project:
    def __init__(self, name="default", params=None, config=True):
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
                "analysis",
                "calibration",
                "catalog",
            ]
        )
        # default values in case no config file is loaded:
        self.pars.observatories = set()  # set of observatory names
        self.pars.catalog = {}  # parameters for the catalog
        self.pars.calibration = (
            {}
        )  # global calibration parameters for all observatories
        self.pars.analysis = {}  # global analysis parameters for all observatories

        # load the parameters from the config file:
        if config:  # note that empty string is also false!
            if isinstance(config, str):
                filename = config
            else:
                filename = os.path.join("configs", f"{self.name}.yaml")
            self.pars.load(filename, "project")

        # add any additional user inputs IN ADDITION to the config file:
        if params is not None:
            if not isinstance(params, dict):
                raise TypeError(f"params must be a dictionary, not {type(params)}")
            self.pars.update(params)

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
        self.catalog = Catalog(**self.pars.catalog)

        # make observatories:
        self.observatories = [
            self.make_observatory(obs, config) for obs in self.pars.observatories
        ]
        self.observatories = {obs.name: obs for obs in self.observatories}

    def make_observatory(self, name, config=None):
        """
        Produce an Observatory object,
        use the name parameter to figure out
        what kind of subclass to use.
        Load configurations and apply them
        to the calibration and analysis objects
        for this observatory.

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

        module = importlib.import_module("." + module_name, package="src")
        obs_class = getattr(module, f"Virtual{name}")
        new_obs = obs_class(project_name=self.name, config=config)
        # new_obs should contain all sub-objects
        # like the calibration and analysis,
        # but they are not initialized yet.
        # first, load the parameters, then initialize them

        # parse parameters for calibration of this observatory
        new_obs.calibration.pars.read(self.pars.calibration)  # project pars
        new_obs.calibration.pars.read(new_obs.pars.calibration)  # observatory pars
        new_obs.calibration.pars.verify()
        new_obs.calibration.initialize()

        # parse parameters for analysis of this observatory
        new_obs.analysis.pars.read(self.pars.analysis)  # project pars
        new_obs.analysis.pars.read(new_obs.pars.analysis)  # observatory pars
        new_obs.analysis.pars.verify()
        new_obs.analysis.initialize()

        # the catalog is just referenced
        # from the project object
        new_obs.catalog = self.catalog

        # verify parameter values and setup observatory
        new_obs.initialize()

        return new_obs