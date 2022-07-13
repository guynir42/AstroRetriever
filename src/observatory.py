import os
import glob
import re
import yaml
import validators
import numpy as np
import pandas as pd
import sqlalchemy as sa

from astropy.io import fits

from src.database import Session
from src.source import Source, get_source_identifiers
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
                "dataset_identifier",
                "catalog_matching",
            ]
        )
        self.pars.calibration = {}
        self.pars.analysis = {}
        self.pars.dataset_attribute = "source_name"
        self.pars.dataset_identifier = "key"
        self.pars.catalog_matching = "name"

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
            # if any info (username, password, etc.)
            # is given by the config/user inputs
            # prefer to use that instead of the file content
            self._credentials.update(self.pars.credentials)

        self.pars.verify()

    def load_passwords(self, filename=None, key=None, **_):
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

        # if file doesn't exist, just return with an empty dict
        if os.path.exists(filepath):
            with open(filepath) as file:
                self._credentials = yaml.safe_load(file).get(key)
        else:
            self._credentials = {}

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

    def populate_sources(self, number=None):
        # raise NotImplementedError("populate_sources() must be implemented in subclass")
        """
        Read the list of files with data,
        and match them up with the catalog,
        so that each catalog row that has
        data associated with it is also
        instantiated in the database.

        Parameters
        ----------
        number: int
            The maximum number of files to read.
            Default is None, which means all files
            found in the directory.

        """
        if self.pars.catalog_matching == "number":
            column = "cat_index"
        elif self.pars.catalog_matching == "name":
            column = "cat_id"
        else:
            raise ValueError("catalog_matching must be either 'number' or 'name'")

        # get a list of existing sources and their ID
        source_ids = get_source_identifiers(self.project_name, column)

        dir = self.pars.get_data_path()
        if self.pars.verbose:
            print(f"Reading from data folder: {dir}")

        with Session() as session:
            for i, filename in enumerate(
                glob.glob(os.path.join(dir, self.pars.data_glob))
            ):
                if number is not None and i >= number:
                    break

                if self.pars.verbose:
                    print(f"Reading filename: {filename}")
                # TODO: add if-else for different file types
                with pd.HDFStore(filename) as store:
                    keys = store.keys()
                    for j, k in enumerate(keys):
                        data = store[k]
                        cat_id = self.find_dataset_identifier(data, k)
                        self.save_source(data, cat_id, source_ids, session)

        if self.pars.verbose:
            print("Done populating sources.")

    def find_dataset_identifier(self, data, key):
        """
        Find the identifier that connects the data
        loaded from file with the catalog row and
        eventually to the source in the database.

        Parameters
        ----------
        data:
            Data loaded from file.
            Can be a dataframe or other data types.
        key:
            Key of the data loaded from file.
            For HDF5 files, this would be
            the name of the group.

        Returns
        -------
        str or int
            The identifier that connects the data
            to the catalog row.
            If self.pars.catalog_matching is 'name',
            this would be a string (the name of the source).
            If it is 'number', this would be an integer
            (the index of the source in the catalog).
        """
        if self.pars.dataset_identifier == "attribute":
            if not hasattr(self.pars, "dataset_attribute"):
                raise ValueError(
                    "When using dataset_identifier='attribute', "
                    "you must specify the dataset_attribute, "
                    "that is the name of the attribute "
                    "that contains the identifier."
                )
            value = getattr(data, self.pars.dataset_attribute)
        elif self.pars.dataset_identifier == "key":
            if self.pars.catalog_matching == "number":
                value = int(re.search(r"\d+", key).group())
            elif self.pars.catalog_matching == "name":
                value = key
        else:
            raise ValueError("dataset_identifier must be either 'attribute' or 'key'")

        if self.pars.catalog_matching == "number":
            value = int(value)

        return value

    def save_source(self, data, cat_id, source_ids, session):
        """
        Save a source to the database,
        using the dataset loaded from file,
        and matching it to the catalog.
        If the source already exists in the database,
        nothing happens.
        If the source does not exist in the database,
        it is created and it's id is added to the source_ids.

        Parameters
        ----------
        data: dataframe or other data types
            Data loaded from file.
            For HDF5 files this is a dataframe.
        cat_id: str or int
            The identifier that connects the data
            to the catalog row. Can be a string
            (the name of the source) or an integer
            (the index of the source in the catalog).
        source_ids: set of str or int
            Set of identifiers for sources that already
            exist in the database.
            Any new data with the same identifier is skipped,
            and any new data not in the set is added.
        session: sqlalchemy.orm.session.Session
            The current session to which we add
            newly created sources.

        """
        if self.pars.verbose > 1:
            print(
                f"Loaded data for source {cat_id} | "
                f"len(data): {len(data)} | "
                f"id in source_ids: {cat_id in source_ids}"
            )

        if len(data) <= 0:
            return  # no data

        if cat_id in source_ids:
            return  # source already exists

        row = self.catalog.get_row(cat_id, self.pars.catalog_matching)

        (
            index,
            name,
            ra,
            dec,
            mag,
            mag_err,
            filter_name,
            alias,
        ) = self.catalog.extract_from_row(row)
        new_source = Source(
            name=name,
            ra=ra,
            dec=dec,
            project=self.project_name,
            alias=[alias] if alias else None,
            mag=mag,
            mag_err=mag_err,
            mag_filter=filter_name,
        )
        new_source.cat_id = name
        new_source.cat_index = index
        new_source.cat_name = self.catalog.pars.catalog_name
        new_source.datasets = self.parse_datasets(data)

        # TODO: is here the point where we also do analysis?

        session.add(new_source)
        session.commit()
        source_ids.add(cat_id)

    def parse_datasets(self, data):
        """

        Parameters
        ----------
        data

        Returns
        -------

        """

        # TODO: finish this
        return []


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
