import os
import yaml
import validators
import numpy as np

from astropy.io import fits

from src.parameters import Parameters
from src.catalog import Catalog
from src.calibration import Calibration
from src.histograms import Histograms
from src.analysis import Analysis


class VirtualObservatory:
    """
    Base class for other virtual observatories.
    This class allows the user to load a catalog,
    and for each object in it, download data from
    some real observatory, run analysis on it,
    save the results for later, and so on.
    """

    def __init__(self, obs_name=None, project_name=None, config=True, keyname=None):
        """
        Create a new VirtualObservatory object,
        which is a base class for other observatories,
        not to be initialized directly.

        Parameters
        ----------
        obs_name: str
            Name of the observatory.
            This is used to find the key inside
            the config files for this observatory.
            This is also used to name the output files, etc.
        project_name: str
            Name of the project working with this observatory.
            This is used to find the config file,
            to produce the output files, etc.
        config: str or bool
            Name of the file to load.
            If False or None, no config file will be loaded.
            If True but not a string, will
            default to "configs/<project-name>.yaml"
            If a non-empty string is given,
            it will be used as the config filename.
        keyname: str
            Key inside file which is relevant to
            this observatory.
            Defaults to the observatory name
            (turned to lower case).
        """
        self.name = obs_name
        self.project_name = project_name
        self.pars = None  # parameters for the analysis
        self.catalog = None  # a Catalog object
        self.calibration = Calibration()  # a Calibration object
        self.analysis = Analysis()  # an Analysis object
        self.histograms = Histograms()  # a Histograms object
        self.database = None  # connection to database with objects
        self._credentials = None  # dictionary with usernames/passwords
        self._config = config  # True/False or config filename
        self._keyname = keyname  # key inside config file
        self.pars = Parameters(
            required_pars=[
                "calibration",
                "analysis",
                "data_folder",
                "data_glob",
            ]
        )
        self.pars.calibration = {}
        self.pars.analysis = {}

    def initialize(self):
        """
        Run this initialization code after loading the parameters.
        Verifies that all required parameters are set,
        and that the values are the right type, in range, etc.
        Calls any other initialization code needed to setup.

        """
        if not isinstance(self.name, str):
            raise TypeError("Observatory name was not set")

        if not isinstance(self.project_name, str):
            raise TypeError("project_name not set")

        if not isinstance(self.pars, Parameters):
            raise TypeError("No Parameters object has been loaded.")

        if not isinstance(self.catalog, Catalog):
            raise TypeError("No Catalog object has been loaded.")

        if not isinstance(self.calibration, Calibration):
            raise TypeError("No Calibration object has been loaded.")

        if not isinstance(self.analysis, Analysis):
            raise TypeError("No Analysis object has been loaded.")

        # if self.database is None:
        #     raise ValueError('No database object loaded')

        if not isinstance(self.histograms, Histograms):
            raise TypeError("No Histograms object has been loaded.")

        if hasattr(self.pars, "credentials"):
            self.load_passwords(**self.pars.credentials)

        self.pars.verify()

    def load_passwords(self, filename=None, key=None):
        """
        Load a YAML file with usernames, passwords, etc.

        Parameters
        ----------
        filename: str
            Name of the file to load.
            Defaults to "credentials.yaml"
        key: str
            Key inside file which is relevant to
            this observatory.
            Defaults to the observatory name
            (turned to lower case).
        """
        if filename is None:
            filename = "credentials.yaml"

        if key is None:
            key = self.name.lower()

        if os.path.isabs(filename):
            filepath = filename
        else:
            basepath = os.path.dirname(__file__)
            filepath = os.path.abspath(os.path.join(basepath, "..", filename))

        with open(filepath) as file:
            self._credentials = yaml.safe_load(file).get(key)

    def load_parameters(self):
        """
        Load a YAML file with parameters for getting/processing
        data from this observatory.
        After loading the parameters,
        additional parameters can be set using other
        config files, or by input arguments, etc.
        After finishing loading parameters,
        use self.pars.verify() to make sure
        all required parameters are set.
        """

        if self._keyname is None:
            keyname = self.name.lower()
        else:
            keyname = self._keyname

        if self._config:
            if isinstance(self._config, str):
                filename = self._config
            else:
                filename = os.path.join("configs", self.project_name + ".yaml")

            self.pars.load(filename, keyname)

        # after loading parameters from all files,
        # must verify that all required parameters are present
        # by calling self.pars.verify()

    def reset_histograms(self):
        pass

    def load_histograms(self):
        pass

    def save_histograms(self):
        pass

    def run_analysis(self):
        """
        Perform the the calibration and analysis
        on each object in the catalog.



        """
        self.analysis.load_simulator()
        self.reset_histograms()

        for row in self.catalog:
            source = self.get_source(row)
            if source is not None:
                self.calibration.apply_calibration(source)
                self.analysis.run(source, histograms=self.histograms)

    def get_source(self, row):
        """
        Load a Source object from the database
        based on a row in the catalog.

        Parameters
        ----------
        row: dict
            A row in the catalog.
            Must contain at least an object ID
            or RA/Dec coordinates.
        """
        return None  # TODO: implement this

    def download(self):
        raise NotImplementedError("download() must be implemented in subclass")

    def populate_sources(self):
        raise NotImplementedError("populate_sources() must be implemented in subclass")

    def save_source(self, ra, dec, data={}):
        pass


class VirtualDemoObs(VirtualObservatory):
    def __init__(self, project_name, config=None, keyname=None):
        """
        Generate an instance of a VirtualDemoObs object.
        This object can be used to test basic operations
        of this package, like pretending to download data,
        running analysis, etc.

        Parameters
        ----------
        Are the same as the VirtualObservatory class.
        The only difference is that the obs_name is set to "demo".

        """

        super().__init__("demo", project_name, config, keyname)
        self.pars.required_pars += ["demo_url", "demo_boolean"]
        # define any default parameters at the
        # source code level here
        # These could be overridden by the config file.
        self.pars.demo_url = "http://www.example.com"
        self.pars.demo_boolean = True
        self.pars.data_folder = "demo_data"
        self.pars.data_glob = project_name + "_Demo_*.h5"

        self.load_parameters()
        # after loading the parameters
        # the user may override them in external code
        # and then call initialize() to verify
        # that all required parameters are set.

    def initialize(self):
        """
        Verify inputs to the observatory.
        """
        super().initialize()  # check all required parameters are set

        # verify parameters have the correct type, etc.
        if self.pars.demo_url is None:
            raise ValueError("demo_url must be set to a valid URL")
        validators.url(self.pars.demo_url)

        if self.pars.demo_boolean is None or not isinstance(
            self.pars.demo_boolean, bool
        ):
            raise ValueError("demo_boolean must be set to a boolean")

        if not isinstance(self.pars.data_folder, str):
            raise ValueError("data_folder must be set to a string.")

        if not isinstance(self.pars.data_glob, str):
            raise ValueError("data_glob must be set to a string.")

        # additional setup code would go here...

    def download(self):
        """
        Generates a fake download of data.
        """
        pass
        # TODO: make a simple lightcurve simulator

    def populate_sources(self):
        """
        Match the catalog to the data,
        by creating sources on the database
        that connect each row in the catalog
        to a datafile on disk.
        """
        pass
        # TODO: figure out how this will work!
