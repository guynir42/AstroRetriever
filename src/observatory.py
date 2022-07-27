import os
import glob
import re
import yaml
import validators
import numpy as np
import pandas as pd

from src.database import Session
from src.source import Source, get_source_identifiers
from src.dataset import DatasetMixin, RawData, PhotometricData
from src.parameters import Parameters
from src.catalog import Catalog
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
        self.analysis = Analysis()  # an Analysis object
        self.histograms = Histograms()  # a Histograms object
        self.database = None  # connection to database with objects
        self._credentials = None  # dictionary with usernames/passwords
        self._config = config  # True/False or config filename
        self._keyname = keyname  # key inside config file
        self.pars = Parameters(
            required_pars=[
                "reducer",
                "analysis",
                "data_folder",
                "data_glob",
                "dataset_identifier",
                "catalog_matching",
            ]
        )
        self.pars.reducer = {}
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
        Perform analysis on each object in the catalog.

        """
        self.analysis.load_simulator()
        self.reset_histograms()

        for row in self.catalog:
            source = self.get_source(row)
            if source is not None:
                for d in source.datasets:
                    d.load()
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
                        # if j > 3:
                        #     break
                        data = store[k]
                        cat_id = self.find_dataset_identifier(data, k)
                        self.save_source(data, cat_id, source_ids, filename, k, session)

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

    def save_source(self, data, cat_id, source_ids, filename, key, session):
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
        filename: str
            Full path to the file from which the data was loaded.
        key: str
            Key inside the file if multiple datasets are in each file.
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

        new_data = RawData(
            data=data,
            observatory=self.name,
            filename=filename,
            key=key,
        )
        new_source.raw_data = [new_data]
        new_source.lightcurves = self.reduce(
            new_data, to="lightcurves", source=new_source
        )
        for lc in new_source.lightcurves:
            lc.save()

        session.add(new_source)
        session.commit()
        source_ids.add(cat_id)

        # TODO: is here the point where we also do analysis?

    def reduce(self, dataset, to="lcs", source=None, **kwargs):
        """
        Reduce raw data into more useful,
        second level data product.
        Input raw data could be
        raw photometry, images, cutouts, spectra.
        The type of reduction to use is inferred
        from the dataset's "type" attribute.

        The output that should be produced from
        the raw data can be lightcurves (i.e.,
        processed photometry ready for analysis)
        or SED (i.e., Spectral Energy Distribution,
        which is just a reduced spectra ready for
        analysis) or even just calibrating an image.
        Possible values for the "to" input are:
        "lcs", "sed", "img", "thumb".

        Parameters
        ----------
        dataset: a src.dataset.RawData object (or list of such objects)
            The raw data to reduce.
        to: str
            The type of output to produce.
            Possible values are:
            "lcs", "sed", "img", "thumb".
            If the input type is photometric data,
            the "to" will be replaced by "lcs".
            If the input type is a spectrum,
            the "to" will be replaced by "sed".
            Imaging data can be reduced into
            "img" (a calibrated image), "thumb" (a thumbnail),
            or "lcs" (a lightcurve of extracted sources).
        source: src.source.Source object
            The source to which the dataset belongs.
            If None, the reduction will not use any
            data of the source, such as the expected
            magnitude, the position, etc.
        kwargs: dict
            Additional arguments to pass to the reduction function.

        Returns
        -------
        an object of a subclass of src.dataset.Dataset
            The reduced dataset,
            can be, e.g., a PhotometricData object.
        """
        if isinstance(dataset, list):
            datasets = dataset
        else:
            datasets = [dataset]

        for i, d in enumerate(datasets):
            if not isinstance(d, RawData):
                raise ValueError(
                    f"Expected RawData object, but dataset {i} was a {type(d)}"
                )

        # parameters for the reduction
        # are taken from the config first,
        # then from the user inputs
        if hasattr(self.pars, "reducer") and isinstance(self.pars.reducer, dict):
            parameters = {}
            parameters.update(self.pars.reducer)
            parameters.update(kwargs)
            kwargs = parameters

        # arguments to be passed into the new dataset constructors
        init_kwargs = {}
        for att in DatasetMixin.default_copy_attributes:
            values = list({getattr(d, att) for d in datasets})
            # only copy values if they're the same
            # for all source (raw) datasets
            if len(values) == 1:
                init_kwargs[att] = values[0]

        # if all data comes from a single raw dataset
        # we should link that back from the new datasets
        raw_data_ids = list({d.id for d in datasets})
        if len(raw_data_ids) == 1:
            init_kwargs["raw_data_id"] = raw_data_ids[0]

        for att in DatasetMixin.default_update_attributes:
            new_dict = {}
            for d in datasets:
                new_value = getattr(d, att)
                if isinstance(new_value, dict):
                    new_dict.update(new_value)
            if len(new_dict) > 0:
                init_kwargs[att] = new_dict

        if to.lower() in ("lc", "lcs", "lightcurves", "photometry"):
            new_datasets = self.reduce_to_lightcurves(
                datasets, source, init_kwargs, **kwargs
            )
        elif to.lower() in ("sed", "seds", "spectra", "spectrum"):
            new_datasets = self.reduce_to_sed(datasets, source, init_kwargs, **kwargs)
        elif to.lower() == "img":
            new_datasets = self.reduce_to_image(datasets, source, init_kwargs, **kwargs)
        elif to.lower() == "thumb":
            new_datasets = self.reduce_to_thumbnail(
                datasets, source, init_kwargs, **kwargs
            )
        else:
            raise ValueError(f'Unknown value for "to" input: {to}')

        new_datasets = sorted(new_datasets, key=lambda d: d.time_start)

        return new_datasets

    def reduce_to_lightcurves(self, datasets, source=None, init_kwargs={}, **kwargs):
        raise NotImplementedError("Photometric reduction not implemented in this class")

    def reduce_to_sed(self, datasets, source=None, init_kwargs={}, **kwargs):
        raise NotImplementedError("SED reduction not implemented in this class")

    def reduce_to_image(self, datasets, source=None, init_kwargs={}, **kwargs):
        raise NotImplementedError("Image reduction not implemented in this class")

    def reduce_to_thumbnail(self, datasets, source=None, init_kwargs={}, **kwargs):
        raise NotImplementedError("Thumbnail reduction not implemented in this class")


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

    def reduce_to_lightcurves(
        self, datasets, source=None, init_kwargs={}, mag_range=None, drop_bad=False, **_
    ):
        """
        Reduce the datasets to lightcurves.

        Parameters
        ----------
        datasets: a list of src.dataset.RawData objects
            The raw data to reduce.
        source: src.source.Source object
            The source to which the dataset belongs.
            If None, the reduction will not use any
            data of the source, such as the expected
            magnitude, the position, etc.
        init_kwargs: dict
            A dictionary of keyword arguments to be
            passed to the constructor of the new dataset.
        mag_range: float or None
            If not None, and if the source is also given,
            this value will be used to remove datasets
            where the median magnitude is outside of this range,
            relative to the source's magnitude.
        drop_bad: bool
            If True, any points in the lightcurves will be
            dropped if their flag is non-zero
            or if their magnitude is NaN.
            This reduces the output size but will
            also not let bad data be transferred
            down the pipeline for further review.

        Returns
        -------
        a list of src.dataset.PhotometricData objects
            The reduced datasets, after minimal processing.
            The reduced datasets will have uniform filter,
            each dataset will be sorted by time,
            and some initial processing will be done,
            using the "reducer" parameter (or function inputs).
        """
        allowed_types = "photometry"
        allowed_dataclasses = pd.DataFrame
        for i, d in enumerate(datasets):
            # check the raw input types make sense
            if d.type is None or d.type not in allowed_types:
                raise ValueError(
                    f"Expected RawData to contain {str(allowed_types)}, "
                    f"but dataset {i} was a {d.type} dataset."
                )
            if not isinstance(d.data, allowed_dataclasses):
                raise ValueError(
                    f"Expected RawData to contain {str(allowed_dataclasses)}, "
                    f"but data in dataset {i} was a {type(d.data)} object."
                )

        # check the source magnitude is within the range
        if source and source.mag is not None and mag_range:
            # need to make a copy of the list so we don't
            # delete what we are iterating over!
            for d in list(datasets):
                mag = d.data[d.mag_col]

                if not (
                    source.mag - mag_range < np.nanmedian(mag) < source.mag + mag_range
                ):
                    datasets.remove(d)

        # split the data by filters
        # (assume all datasets have the same data class
        # and that the internal column structure is the same
        # which is a reasonable assumption as they all come
        # from the same observatory)
        if isinstance(datasets[0].data, pd.DataFrame):
            # make sure there is some photometric data available

            frames = [d.data for d in datasets]
            all_dfs = pd.concat(frames)

            filt_col = datasets[0].colmap["filter"]
            flag_col = (
                datasets[0].colmap["flag"] if "flag" in datasets[0].colmap else None
            )
            filters = all_dfs[filt_col].unique()
            dfs = []
            for f in filters:
                # new dataframe for each filter
                # each one with a new index
                df_new = all_dfs[all_dfs[filt_col] == f].reset_index(drop=True)
                # df_new = df_new.sort([datasets[0].time_col], inplace=False)
                # df_new.reset_index(drop=True, inplace=True)
                if drop_bad and flag_col is not None:
                    df_new = df_new[df_new[flag_col] == 0]

                dfs.append(df_new)
                # TODO: what happens if filter is in altdata, not in dataframe?

            new_datasets = []
            for df in dfs:
                new_datasets.append(PhotometricData(data=df, **init_kwargs))

        return new_datasets
