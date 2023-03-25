import os
import re
import warnings
from tables import NaturalNameWarning
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
import threading

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from astropy.time import Time
import h5py

import sqlalchemy as sa
from sqlalchemy import orm, event
from sqlalchemy.schema import UniqueConstraint

from sqlalchemy.dialects.postgresql import JSONB

import src.database
from src.database import engine, Base, SmartSession, safe_mkdir
from src.source import Source
from src.utils import random_string, legalize, luptitudes

lock = threading.Lock()


PHOT_ZP = 23.9
LOG_BASE = np.log(10) / 2.5

AUTOLOAD = True
AUTOSAVE = False
OVERWRITE = False

# 1% difference in time is considered uniform
UNIFORMITY_THRESHOLD = 0.01


def simplify(key):
    """
    Cleans up (AND bumps to lower case)
    a string to compare it to a list of expected
    keywords such as "jd" or "timestamps".

    Will remove spaces, underscores and dashes.
    Will remove any numbers, any values after a comma,
    and finally will remove trailing "s".
    """
    key = key.lower().replace(" ", "").replace("_", "").replace("-", "")

    key = re.sub(r"\d+", "", key)

    key = key.split(",")[0]

    if key[-1] == "s":
        key = key[:-1]

    return key


def get_time_offset(time_str):
    """
    When a timestamp column is given as a string
    with a numeric offset (e.g., "MJD-50000"),
    this function picks up that offset and returns it.
    This can be used to modify the times using
    the "to datetime" and "to mjd" lambdas.
    """
    val = re.search(r"[+-]\s+\d+", time_str)
    if val is None:
        return 0
    else:
        return float(val.group(0).replace(" ", ""))


def commit_and_save(datasets, session=None, save_kwargs={}):
    """
    Commit all datasets to the database, and then
    save all datasets to disk.
    If anything fails, will rollback the session
    and delete the data from disk.
    """
    with SmartSession(session) as session:

        if not isinstance(datasets, list):
            datasets = [datasets]

        if any([not isinstance(d, DatasetMixin) for d in datasets]):
            raise ValueError("All datasets must be of type DatasetMixin")

        for dataset in datasets:
            try:
                session.add(dataset)
                if not dataset.check_file_exists():
                    lock.acquire()  # thread blocks at this line until it can obtain lock
                    try:
                        dataset.save(**save_kwargs)
                    finally:
                        lock.release()

                session.commit()
            except Exception:
                session.rollback()
                dataset.delete_data_from_disk()
                raise


class DatasetMixin:
    """
    A Dataset object maps a location on disk
    with raw data in memory.
    Each Dataset is associated with one source
    (via the source_id foreign key).
    If there are multiple datasets in a file,
    use the "filekey" parameter to identify which
    one is associated with this Dataset.

    Parameters will be applied from kwargs
    if they match any existing attributes.
    If "data" is given as an input,
    it will be set first, allowing the object
    to calculate some attributes
    (like type and start/end times).
    After that any other properties will be
    assigned and can override the values
    calculated from the data.

    Subclasses of this mixin will contain
    specific type of data. For example
    the RawPhotometry class will contain raw
    photometric data dwonloaded as-is from the observatory.
    This can be reduced into a Lightcurve
    object using a reducer function
    from the correct observatory object.
    from the correct observatory object.

    """

    if True:  # put all column definitions in a single block
        source_name = sa.Column(
            sa.String,
            nullable=False,
            index=True,
            doc="Name of the source for which this observation was taken",
        )

        observatory = sa.Column(
            sa.String,
            nullable=False,
            index=True,
            doc="Observatory this dataset is associated with",
        )

        # original series of images used to make this dataset
        run_identifier = sa.Column(
            sa.String,
            nullable=True,
            index=True,
            doc="Identifier of the observing run this dataset is associated with, "
            "e.g., the set of images the include this source but others as well",
        )
        run_object = sa.Column(
            sa.String,
            nullable=True,
            index=True,
            doc="Name of object relative to the observing run this dataset is associated with",
        )

        # saving the data to file
        filename = sa.Column(
            sa.String,
            nullable=False,
            index=True,
            doc="Filename of the dataset, including relative path, "
            "that may be appended to the folder attribute. ",
        )

        folder = sa.Column(
            sa.String,
            nullable=True,
            doc="Folder where this dataset is stored. "
            "If relative path, it is relative to DATA_ROOT. "
            "If given as absolute path that overlaps with DATA_ROOT, "
            "it will be converted to a relative path, so it is portable. ",
        )

        filekey = sa.Column(
            sa.String,
            nullable=True,
            doc="Key of the dataset (e.g., in the HDF5 file it would be the group name)",
        )

        format = sa.Column(
            sa.String, nullable=False, default="hdf5", doc="Format of the dataset"
        )

        # TODO: add a method to use this dictionary
        load_instructions = sa.Column(
            JSONB,
            nullable=True,
            doc="Instructions dictionary for loading the data from disk",
        )

        colmap = sa.Column(
            JSONB,
            nullable=False,
            default={},
            doc="Dictionary mapping column names to column indices",
        )

        time_info = sa.Column(
            JSONB,
            nullable=False,
            default={},
            doc="Dictionary containing information about the time column "
            "(e.g., how to parse the time, with format and offset info).",
        )

        altdata = sa.Column(
            JSONB,
            nullable=False,
            default={},
            doc="Additional meta-data associated with this dataset",
        )

        # timing data and number / size of images
        time_start = sa.Column(
            sa.DateTime,
            nullable=True,
            doc="Start time of the dataset (e.g., time of first observation)",
        )
        time_end = sa.Column(
            sa.DateTime,
            nullable=True,
            doc="End time of the dataset (e.g., time of last observation)",
        )

        # data size and shape
        shape = sa.Column(
            sa.ARRAY(sa.Integer), nullable=False, doc="Shape of the dataset"
        )
        number = sa.Column(
            sa.Integer,
            nullable=False,
            doc="Number of observations/epochs in the dataset",
        )
        size = sa.Column(
            sa.Integer,
            nullable=False,
            doc="Size of the dataset in bytes",
        )

        public = sa.Column(
            sa.Boolean,
            nullable=False,
            default=False,
            doc="Whether this dataset is publicly available",
        )

        autoload = sa.Column(
            sa.Boolean,
            nullable=False,
            default=AUTOLOAD,
            doc="Whether this dataset should be automatically loaded (lazy loaded)",
        )

        # TODO: should this be persisted or defined
        #  each time the dataset is made/loaded?
        autosave = sa.Column(
            sa.Boolean,
            nullable=False,
            default=AUTOSAVE,
            doc="Whether this dataset should be automatically saved",
        )

        overwrite = sa.Column(
            sa.Boolean,
            nullable=False,
            default=OVERWRITE,
            doc="Whether the data on disk should be overwritten if it already exists "
            "(if False, an error will be raised)",
        )

    def __init__(self, **kwargs):
        # this is commented out because sometimes you want
        # to create an empty raw data and give it a filename
        # and have it loaded into this object from there
        # if "data" not in kwargs:
        #     raise ValueError("Must provide data to initialize a Dataset object")

        self._data = None
        self._times = None
        self._mjds = None
        self.source = None

        # these are only set when generating a new
        # object, not when loading from database
        self.autoload = AUTOLOAD
        self.autosave = AUTOSAVE
        self.overwrite = OVERWRITE

        # if given any info on the column mapping or time parsing:
        self.colmap = kwargs.pop("colmap", {}).copy()
        self.time_info = kwargs.pop("time_info", {}).copy()

        # first input data to allow
        # the object to calculate some attributes
        if "data" in kwargs:
            data = kwargs.pop("data")

            # first figure out the data columns and time conversions
            self._update_colmap(data)
            self.data = data  # also calculate times and other stats

        # override any existing attributes
        for k, v in list(kwargs.items()):
            if hasattr(self, k):
                setattr(self, k, kwargs.pop(k))

        # defined on the base class
        # pop out any kwargs that are attributes of self.
        self.keywords_to_columns(kwargs)

        additional_keywords = []
        if any([k not in additional_keywords for k in kwargs.items()]):
            raise ValueError(f"Unknown keyword arguments: {kwargs}")

        if not isinstance(self.filename, (str, type(None))):
            raise ValueError(f"Filename must be a string, not {type(self.filename)}")

        if not isinstance(self.filekey, (str, int, type(None))):
            raise ValueError(
                f"Key must be a string, int, or None, not {type(self.filekey)}"
            )

        # guess some attributes that were not given
        if self.format is None:
            self.format = self._guess_format()

        self.loaded_status = "new"
        # TODO: figure out the series identifier and object

    def __setattr__(self, key, value):
        if key == "filename":
            if isinstance(value, str) and os.path.isabs(value):
                raise ValueError("Filename must be a relative path, not absolute")
        if key == "folder":
            # if given as absolute path that overlaps with DATA_ROOT,
            # convert to relative path, so it is portable
            if isinstance(value, str) and os.path.isabs(value):
                if value.startswith(src.database.DATA_ROOT):
                    value = value[len(src.database.DATA_ROOT) :]
                    if value.startswith("/"):
                        value = value[1:]
        if key == "source" and value is not None:
            if not isinstance(value, Source):
                raise ValueError(f"Source must be a Source object, not {type(value)}")
            self.source_name = value.name
        if key == "project" and value is not None:
            value = legalize(value)
        if key == "observatory" and value is not None:
            value = legalize(value)
        super().__setattr__(key, value)

    @orm.reconstructor
    def _init_on_load(self):
        """
        This is called when the object
        is loaded from the database.
        ref: https://docs.sqlalchemy.org/en/14/orm/constructors.html
        """
        self._data = None
        self._times = None
        self._mjds = None
        self.source = None
        self.loaded_status = "database"

    def _guess_format(self):
        """
        Guess the format of the data file
        using its extension.

        Returns
        -------
        str
            Format of the data file.
        """
        if self.filename is not None:
            ext = os.path.splitext(self.filename)[1]
            if ext == ".h5" or ext == ".hdf5":
                return "hdf5"
            elif ext == ".fits":
                return "fits"
            elif ext == ".csv":
                return "csv"
            elif ext == ".json":
                return "json"
            elif ext == ".nc":
                return "netcdf"
            else:
                raise ValueError(f'Unknown data format "{ext}"')
        elif self.data is not None:
            if isinstance(self.data, np.ndarray):
                return "fits"
            elif isinstance(self.data, pd.DataFrame):
                return "hdf5"
            elif isinstance(self.data, xr.Dataset):
                return "netcdf"
        else:
            return None

    def _guess_extension(self):
        """
        Guess the extension of the data file
        based on the format.
        """

        if self.format == "hdf5":
            return ".h5"
        elif self.format == "fits":
            return ".fits"
        elif self.format == "csv":
            return ".csv"
        elif self.format == "json":
            return ".json"
        elif self.format == "netcdf":
            return ".nc"
        else:
            raise ValueError(f"Unknown format {self.format}")

    def calc_size(self):
        """
        Calculate the size of the data file.
        This does not include the metadata,
        only the dataframe itself.
        """
        # ref: https://stackoverflow.com/questions/18089667/how-to-estimate-how-much-memory-a-pandas-dataframe-will-need
        if self.data is None or len(self.data) == 0:
            return 0
        else:
            return int(self.data.memory_usage(index=True).sum())

    def get_path(self):
        """
        Get the name of the folder inside
        the DATA_ROOT folder where this dataset is stored.
        """

        if self.folder is not None:
            f = self.folder
        elif hasattr(self, "project") and self.project is not None:
            f = self.project
        elif self.observatory is not None:
            f = self.observatory.upper()
        else:
            f = "DATA"

        if os.path.isabs(f):
            return f
        else:
            return os.path.join(src.database.DATA_ROOT, f)

    def get_fullname(self):
        """
        Get the full path to the data file.
        """
        if self.get_path() is not None and self.filename is not None:
            return os.path.join(self.get_path(), self.filename)
        else:
            return None

    def check_file_exists(self):
        """
        Check if the file exists on disk.
        """
        if self.filename:
            if os.path.exists(self.get_fullname()):
                return True

        return False

    def check_data_exists(self):
        """
        Check if the data is loaded into memory,
        and if it is not, check that it can be loaded
        from disk.
        If the data is missing, returns a False but
        does not raise an exception (which is what
        would happen if trying to access "data" attribute
        when a file is missing).

        """
        if self._data is not None:
            return True
        else:
            if self.filename:
                if os.path.exists(self.get_fullname()):
                    if self.filekey is None:
                        return True  # has file, no key, good enough
                    else:
                        # has file, has key, check if it's in the file
                        return self.filekey in self.get_file_keys()

        return False

    def get_file_keys(self):
        if self.format == "hdf5":
            with h5py.File(self.get_fullname(), "r") as f:
                return list(f.keys())
        else:
            raise ValueError(f"Cannot get keys for format {self.format}")

    def is_empty(self):
        if self.number == 0:
            return True

    def load(self):
        """
        Loads the data from disk.
        Also attempts to load any
        additional attributes into "altdata".

        If file does not exist,
        or if loading fails,
        will raise an error.

        """

        if not self.check_file_exists():
            raise FileNotFoundError(f"File {self.get_fullname()} does not exist")
        if self.format is None:
            self.format = self._guess_format()

        if self.format == "hdf5":
            data, altdata = self._load_hdf5()
        elif self.format == "fits":
            data, altdata = self._load_fits()
        elif self.format == "csv":
            data, altdata = self._load_csv()
        elif self.format == "json":
            data, altdata = self._load_json()
        elif self.format == "netcdf":
            data, altdata = self._load_netcdf()
        else:
            raise ValueError(f"Unknown format {self.format}")

        self._update_colmap(data)
        self.data = data
        self.altdata = altdata

    def _load_hdf5(self):
        """
        Load the data from a HDF5 file.
        """
        with pd.HDFStore(self.get_fullname(), mode="r") as store:
            key = self.filekey
            if key is None:
                if len(store.keys()) == 1:
                    key = store.keys()[0]
                else:
                    raise ValueError("No key specified and multiple keys found in file")

            # load the data
            data = store.get(key)
            if data is None:
                raise ValueError(f"Key {key} not found in file {self.get_fullname()}")

            # load altdata
            altdata = {}
            if store.get_storer(key).attrs and "altdata" in store.get_storer(key).attrs:
                altdata = store.get_storer(key).attrs["altdata"]
            elif (
                store.get_storer(key).attrs
                and "altdata_keys" in store.get_storer(key).attrs
            ):
                keys = store.get_storer(key).attrs["altdata_keys"]
                for k in keys:
                    altdata[k] = store.get_storer(key).attrs[k]

            return data, altdata

    def _load_fits(self):
        pass

    def _load_csv(self):
        data = pd.read_csv(self.get_fullname())

        return data, {}

    def _load_json(self):
        pass

    def _load_netcdf(self):
        pass

    def _invent_filename(
        self, source_name=None, ra_deg=None, ra_minute=None, ra_second=None
    ):

        """
        Generate a filename and sub-folder with some pre-defined
        format that is consistent enough to have multiple sources
        saved to the same folder in a logical way that is
        easy to figure out even when file data is orphaned
        from the database objects.

        The filename will be <Observatory>_<data_type>_<source_name>.<ext>
        where <observatory> is the upper-case observatory name,
        the <data_type> is e.g., "photometry",
        the <source_name> is given by the input catalog
        and <.ext> is the extension (usually .h5).
        Inside the file, the HDF5 key will be <data_type>_<source_name>.
        If the dataset is reduced or processed, the key will be numbered
        as "reduction_1_of_3", or "processed_1_of_4", etc.
        The same is for simulated datasets, that can have multiple
        different simulated versions for each processed dataset:
        e.g., "processed_1_of_3_simulated_1_of_2".

        The default way to decide which source goes into which
        sub-folder is using RA (right ascension).
        This is a closed interval (0-360) and for most
        all-sky surveys the sources are spread out (mostly)
        evenly, although the galactic center may be more dense
        with sources, causing larger folders at RA~270.

        If given only ra_deg, will split the sources into
        360 folders, one for each integer degree bin.

        If ra_minute is given, will split the sources into 360x60 sub-folders,
        one for each integer minute bin. That means the path will be:
        RA<ra_deg>/<ra_minute>/
        If adding seconds, this will be:
        RA<ra_deg>/<ra_minute>/<ra_second>/
        These modes are useful if the survey you are working with
        has very dense coverage in a small area, and you want to
        split the sources into smaller folders.

        If not given a source right ascension at all,
        the ra range will be replaced with a random string.

        For subclasses that are reductions of the data
        found in another file (and have a "raw_data_filename" attribute),
        will just use that filename, appending the string "_reduced"
        or "_processed" or "_simulated" before adding the extension.

        Parameters
        ----------
        source_name : str, optional
            Name of the source to use in the filename.
            If not given, will try to get the name from
            the "source" attribute, or the first source
            in the "sources" attribute.
            If still not given, will use a random string.
        ra_deg : int, optional
            The integer degree of the right ascension of the source.
            If given as float, will just use floor(ra_deg).
            If given as None, folder will be a random string instead.
            The default is None, but it is highly recommended to give
            the RA of the source when saving.
        ra_minute : int, optional
            The integer minute of the right ascension of the source.
            If given as float, will just use floor(ra_minute).
            if given as None, will only split sources into integer
            degree sub-folders (default).
        ra_second : int, optional
            The integer second of the right ascension of the source.
            If given as float, will just use floor(ra_second).
            if given as None, will only split sources into integer
            degree and possibly minute sub-folders (default).

        """
        if hasattr(self, "raw_data_filename") and self.raw_data_filename is not None:
            self.filename, _ = os.path.splitext(self.raw_data_filename)
        else:
            if source_name is None:
                source_name = self.source_name
            if source_name is None:
                source_name = random_string()
            # need to make up a file name in a consistent way
            if ra_second is not None and (ra_deg is None or ra_minute is None):
                raise ValueError(
                    "If ra_second is given, ra_deg and ra_minute must also be given"
                )
            if ra_minute is not None and ra_deg is None:
                raise ValueError("If ra_minute is given, ra_deg must also be given")

            if ra_deg is not None:
                ra = int(ra_deg)
                binning = f"RA{ra:03d}"

                if ra_minute is not None:
                    ra = int(ra_minute)
                    binning += f"d{ra:02d}m"

                    if ra_second is not None:
                        ra = int(ra_second)
                        binning += f"{ra:02d}s"

            else:
                binning = random_string(15)

            # add prefix using the type of data and observatory
            obs = self.observatory.upper() if self.observatory else "UNKNOWN_OBS"
            data_type = self.type if self.type is not None else "data"
            self.filename = os.path.join(binning, f"{obs}_{data_type}_{source_name}")

        # check if need to add reduced/processed
        if isinstance(self, Lightcurve):
            if hasattr(self, "was_processed") and self.was_processed:
                self.filename += "_processed"
            else:
                self.filename += "_reduced"

        # add extension
        self.filename += self._guess_extension()

    def _invent_filekey(self, source_name=None, prefix=None, suffix=None):
        """
        Make an in-file key string to save the data into.
        This is used for e.g., HDF5 group names for each
        individual source's data.
        If source name is given as string / bytes, will use that
        as the key. If no source is given,
        will use a random 8 character string.

        The prefix and suffix can be used to add additional
        strings to the start or end of the key.

        Parameters
        ----------
        source_name: str or int, optional
            Name or number of the source.
        prefix: str, optional
            Add this string before the source name key.
        suffix: str, optional
            Add this string after the source name key.

        Returns
        -------
        str
            The key to use for the data in the file.
        """
        if source_name is None:
            source_name = self.source_name

        if source_name is not None:
            if isinstance(source_name, bytes):
                source_name = source_name.decode("utf-8")

            if isinstance(source_name, str):
                self.filekey = source_name
            else:
                raise TypeError("source name must be a string")
        else:
            self.filekey = random_string(8)

        # add the type of data
        self.filekey = f"{self.type}_{self.filekey}"

        if prefix:
            if prefix.endswith("_"):  # remove trailing underscore
                prefix = prefix[:-1]
            self.filekey = f"{prefix}_{self.filekey}"
        if suffix:
            if suffix.startswith("_"):  # remove leading underscore
                suffix = suffix[1:]
            self.filekey = f"{self.filekey}_{suffix}"

    def save(
        self,
        overwrite=None,
        source_name=None,
        ra_deg=None,
        ra_minute=None,
        ra_second=None,
        key_prefix=None,
        key_suffix=None,
    ):
        """
        Save the data to disk.

        Parameters
        ----------
        overwrite: bool
            If True, overwrite the file if it already exists.
            If False, raise an error if the file already exists.
            If None (default), use the "overwrite" attribute of the object.
        source_name: str, optional
            Name of the source to save the data for.
            If not given, will use the source attached to
            this object to get the name (if it exists).
            If not found, will use a random string for
            the file key.
        ra_deg: int, optional
            The integer degree of the right ascension of the source.
            Used to determine the filename such that multiple sources
            are grouped into a single file.
            If given as float, will just use floor(ra_deg).
            If given as None, filename will have a random string instead.
            (that means each source will have its own file).
            The default is None, but it is highly recommended to give
            the RA of the source when saving.
        ra_minute: int, optional
            The integer minute of the right ascension of the source.
            If given as float, will just use floor(ra_minute).
            if given as None, will only split sources into integer
            degree filenames (default).
        ra_second: int, optional
            The integer second of the right ascension of the source.
            If given as float, will just use floor(ra_second).
            if given as None, will only split sources into integer
            degree and possibly minute filenames (default).
        key_prefix: str, optional
            Add this string before the internal file key
            (which is the source name or a random string).
        key_suffix: str, optional
            Add this string after the internal file key
            (which is the source name or a random string).

        """
        if self._data is None:
            raise ValueError("No data to save!")

        if ra_deg is None:
            if self.source is not None:
                ra_deg = self.source.ra

        # if no filename/key are given, make them up
        if self.filename is None:
            self._invent_filename(
                source_name=source_name,
                ra_deg=ra_deg,
                ra_minute=ra_minute,
                ra_second=ra_second,
            )

        # for any of the formats where we need an in-file key:
        if self.filekey is None and self.format in ("hdf5",):
            self._invent_filekey(
                source_name=source_name, prefix=key_prefix, suffix=key_suffix
            )

        if overwrite is None:
            overwrite = self.overwrite

        # make a path if missing
        path = os.path.dirname(self.get_fullname())
        safe_mkdir(path)

        # specific format save functions
        if self.format == "hdf5":
            self._save_hdf5(overwrite)
        elif self.format == "fits":
            self._save_fits(overwrite)
        elif self.format == "csv":
            self._save_csv(overwrite)
        elif self.format == "json":
            self._save_json(overwrite)
        elif self.format == "netcdf":
            self._save_netcdf(overwrite)
        else:
            raise ValueError(f"Unknown format {self.format}")

    def _save_hdf5(self, overwrite):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NaturalNameWarning)
            if isinstance(self._data, xr.Dataset):
                # TODO: check if key already exists!
                self.data.to_hdf(self.get_fullname(), key=self.filekey)
            elif isinstance(self._data, pd.DataFrame):
                with pd.HDFStore(self.get_fullname()) as store:
                    if self.filekey in store:
                        if overwrite:
                            store.remove(self.filekey)
                        else:
                            raise ValueError(
                                f"Key {self.filekey} already exists in file {self.get_fullname()}"
                            )
                    # only store a key for non-empty dataframes
                    if len(self.data) > 0:
                        store.put(self.filekey, self.data)
                        if self.altdata:
                            altdata_to_write = self.altdata
                        else:
                            altdata_to_write = {}
                        keys = list(altdata_to_write.keys())
                        store.get_storer(self.filekey).attrs["altdata_keys"] = keys
                        for key in keys:
                            store.get_storer(self.filekey).attrs[
                                key
                            ] = altdata_to_write[key]

            elif isinstance(self._data, np.ndarray):
                with h5py.File(self.get_fullname(), "w") as f:
                    f.create_dataset(self.filekey, data=self.data)
                    if self.altdata:
                        for k, v in self.altdata.items():
                            f[self.filekey].attrs[k] = v
            else:
                raise ValueError(f"Unknown data type {type(self._data)}")

    def _save_fits(self, overwrite):
        pass

    def _save_csv(self, overwrite):
        pass

    def _save_json(self, overwrite):
        pass

    def _save_netcdf(self, overwrite):
        pass

    def delete_data_from_disk(self, remove_folders=True):
        """
        Delete the data from disk, if it exists.
        If the format is hdf5, will delete the key from the file.
        If there are no more keys in the file, will delete the file.

        Parameters
        ----------
        remove_folders: bool
            If True, will remove any folders that were created
            by this object, if they are empty.
        """
        if self.check_file_exists():
            need_to_delete = False
            if self.format == "hdf5":
                with pd.HDFStore(self.get_fullname()) as store:
                    if self.filekey in store:
                        store.remove(self.filekey)
                    if len(store.keys()) == 0:
                        need_to_delete = True

            elif self.format in ("csv", "json"):
                need_to_delete = True
            else:
                raise ValueError(f"Unknown format {self.format}")

            if need_to_delete:
                os.remove(self.get_fullname())

                # delete the folder if empty
                if remove_folders:
                    path = os.path.dirname(self.get_fullname())
                    if len(os.listdir(path)) == 0:
                        os.rmdir(path)

    def _update_colmap(self, data):
        """
        Calculate the column map for the data.
        Locates column names in the data that
        correspond to standardized column names.
        This mapping can be used to access data
        columns with weird, observatory specific
        naming conventions, using a uniform dict.
        E.g., ZTF uses "filtercode", but it would
        be accessed using colmap['filter'].
        """
        if isinstance(data, pd.DataFrame):
            columns = data.columns
        # other datatypes will call this differently...
        # TODO: get the columns for other data types

        for c in columns:  # timestamps
            example = data[c].iloc[0]
            if simplify(c) in ("jd", "juliandate"):
                self.time_info["format"] = "jd"
                self.time_info["offset"] = get_time_offset(c)  # e.g., JD-12345000
                self.colmap["time"] = c
                break
            elif simplify(c) in ("mjd",):
                self.time_info["format"] = "mjd"
                self.time_info["offset"] = get_time_offset(c)  # e.g., MJD-12345000
                self.colmap["time"] = c
                break
            elif simplify(c) in ("bjd",):
                self.time_info["format"] = "jd"
                self.time_info["offset"] = get_time_offset(c)  # e.g., BJD-12345000
                # TODO: must add some conversion between JD and BJD
                #  e.g., https://mail.python.org/pipermail/astropy/2014-April/002844.html
                self.colmap["time"] = c
                break
            elif simplify(c) in ("time", "datetime") and isinstance(
                example, (str, bytes)
            ):
                if "T" in example:
                    self.time_info["format"] = "isot"
                else:
                    self.time_info["format"] = "iso"
                self.time_info["offset"] = 0
                self.colmap["time"] = c
                break
            elif simplify(c) == ("time", "unix", "timestamp"):
                self.time_info["format"] = "unix"
                offset = get_time_offset(c)  # e.g., Unix-12345000
                self.time_info["offset"] = offset
                self.colmap["time"] = c
                break

        for c in columns:  # exposure time
            if simplify(c) in ("exptime", "exposuretime"):
                self.colmap["exptime"] = c
                break

        for c in columns:  # right ascension
            if simplify(c) in ("ra", "rightascension"):
                self.colmap["ra"] = c
                break

        for c in columns:  # declination
            if simplify(c) in ("dec", "declination"):
                self.colmap["dec"] = c
                break

        for c in columns:  # magnitude
            if simplify(c) in ("mag", "magnitude"):
                self.colmap["mag"] = c
                break

        for c in columns:  # magnitude error
            if simplify(c) in ("magerr", "magerr", "magerror"):
                self.colmap["magerr"] = c
                break

        for c in columns:  # fluxes
            if simplify(c) in ("flux", "fluxe", "count"):
                self.colmap["flux"] = c
                break

        for c in columns:  # flux errors
            if simplify(c) in ("fluxerr", "fluxerror", "counterr", "counterror"):
                self.colmap["fluxerr"] = c
                break

        for c in columns:  # filter
            if simplify(c) in ("filt", "filter", "filtername", "filtercode"):
                self.colmap["filter"] = c
                break

        for c in columns:  # bad data flags
            if simplify(c) in ("flag", "catflag", "baddata"):
                self.colmap["flag"] = c
                break

            if simplify(c) == simplify("QUALITY"):  # Kepler/TESS
                self.colmap["flag"] = c
                break

    def _calc_times(self):
        """
        Calculate datetimes and MJDs for each epoch,
        based on the conversions found in self.time_info.
        These values are calculated once when the data
        is loaded from disk or given as input,
        but are not saved in the DB or on disk.
        """
        if len(self._data) == 0:
            return
        if len(self.time_info) == 0:
            raise ValueError(
                "No time_info was found. Make sure to run update_colmap() first..."
            )

        def time_parser(t):
            return Time(
                t + self.time_info["offset"],
                format=self.time_info["format"],
                scale="utc",
            )

        t = time_parser(self._data[self.colmap["time"]])
        self.times = t.datetime
        self.time_start = min(self.times)
        self.time_end = max(self.times)

        self.mjds = t.mjd

    def plot(self, ax=None, **kwargs):
        """
        Plot the data, depending on its type
        """
        if self.type == "photometry":
            return self.plot_photometry(ax=ax, **kwargs)
        elif self.type == "spectrum":
            return self.plot_spectrum(ax=ax, **kwargs)
        elif self.type == "image":
            return self.plot_image(ax=ax, **kwargs)
        elif self.type == "cutouts":
            return self.plot_cutouts(ax=ax, **kwargs)
        else:
            pass  # should this be an error?

    def plot_photometry(
        self, ttype="mjd", ftype="mag", use_phot_zp=False, ax=None, **kwargs
    ):
        """
        Plot the photometry data.

        Parameters
        ----------
        ttype : str
            Type of time to plot. Options are:
            - "mjd": Modified Julian Date
            - "times": dates in YYYY-MM-DD format
        ftype: str
            Type of flux to plot. Options are:
            - "mag": Magnitude
            - "flux": Flux
        use_phot_zp : bool
            If True, use the photometric zero point (PHOT_ZP) to show fluxes.
            If false, use the fluxes as-is. If fluxes are not given,
            will default to using the PHOT_ZP.
            Only used when ftype="flux".
        ax: matplotlib.axes.Axes
            Axis to plot on. If None, a new figure is created.
        kwargs: dict
            Any additional keyword arguments are passed to matplotlib.
        """

        if ax is None:
            fig, ax = plt.subplots()
        if ttype == "times":
            t = self.times
        elif ttype == "mjd":
            t = self.mjds
        else:
            raise ValueError('ttype must be either "times" or "mjd"')

        if ftype == "mag":
            if "mag" in self.colmap:
                m = self.data[self.colmap["mag"]]
            else:  # short circuit if no data
                return ax
        elif ftype == "flux":
            if not use_phot_zp and "flux" in self.colmap:
                m = self.data[self.colmap["flux"]] if "flux" in self.colmap else None
            elif "mag" in self.colmap:
                m = 10 ** (-0.4 * (self.data[self.colmap["mag"]] - PHOT_ZP))
            else:  # short circuit if no data
                return ax
        else:
            raise ValueError('ftype must be either "mag" or "flux"')

        ax.plot(t, m, ".k", **kwargs, zorder=2)

        bad_idx = self.data[self.colmap["flag"]] > 0
        ax.plot(t[bad_idx], m[bad_idx], "xk", **kwargs, zorder=2)

        # add labels like "MJD" and "mag" to axes
        self._axis_labels(ax, ttype=ttype, ftype=ftype)  # TODO: add font_size

        return ax

    @staticmethod
    def _axis_labels(ax, ttype, ftype, font_size=12):
        if ttype == "times":
            # ax.set_xlabel("Time", fontsize=font_size) # don't need label on dates?
            formatter = mdates.DateFormatter("%Y-%m-%d")
            ax.xaxis.set_major_formatter(formatter)
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            for tick in ax.get_xticklabels():
                tick.set_rotation(-45)
        elif ttype == "mjd":
            ax.set_xlabel("MJD", fontsize=font_size)
        plt.xticks(fontsize=font_size - 4)

        # labels for flux/mag axis
        if ftype == "mag":
            ax.set_ylabel("Magnitude", fontsize=font_size)
        elif ftype == "flux":
            ax.set_ylabel("Flux", fontsize=font_size)
        plt.yticks(fontsize=font_size - 4)

    def plot_spectrum(self, ax=None, **kwargs):
        pass

    def plot_image(self, ax=None, **kwargs):
        pass

    def plot_cutouts(self, ax=None, **kwargs):
        pass

    @property
    def is_data_loaded(self):
        return self._data is not None

    @property
    def data(self):
        if self._data is None and self.autoload and self.filename is not None:
            self.load()
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, (np.ndarray, pd.DataFrame, xr.Dataset)):
            raise ValueError(
                "Data must be a numpy array, "
                "pandas DataFrame, or xarray Dataset, "
                f"not {type(data)}"
            )

        self._data = data

        # remove rows with nan timestamps
        if len(self._data.index) > 0 and "time" in self.colmap:
            self._data = data[~np.isnan(data[self.colmap["time"]])]

        self.shape = self._data.shape
        self.number = len(self._data)  # for imaging data this would be different?
        self.size = self.calc_size()
        self.format = self._guess_format()
        self._calc_times()

    @property
    def times(self):
        if self._data is None and self.autoload and self.filename is not None:
            self.load()
        if self._times is None:
            self._calc_times()
        return self._times

    @times.setter
    def times(self, value):
        self._times = value

    @property
    def mjds(self):
        if self._data is None and self.autoload and self.filename is not None:
            self.load()
        if self._mjds is None:
            self._calc_times()
        return self._mjds

    @mjds.setter
    def mjds(self, value):
        self._mjds = value

    @classmethod
    def backref_name(cls):
        if cls.__name__ == "RawPhotometry":
            return "raw_photometry"
        elif cls.__name__ == "LightCurve":
            return "lightcurves"

    # automatically copy these values
    # when reducing one dataset into another
    # (only copy if all parent datasets have
    # the same value)
    default_copy_attributes = [
        "series_identifier",
        "series_object",
        "autoload",
        "autosave",
        "overwrite",
        "public",
        "observatory",
        "cfg_hash",
        "test_hash",
        "folder",
        "colmap",
        "time_info",
    ]

    # automatically update the dictionaries
    # from all parent datasets into a new
    # dictionary in the child dataset(s)
    default_update_attributes = ["altdata"]


# TODO: Do we want to split this off into a separate file?
class RawPhotometry(DatasetMixin, Base):

    __tablename__ = "raw_photometry"

    source_name = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc="Name of the source for which this observation was taken",
    )

    __table_args__ = (
        UniqueConstraint("source_name", "observatory", name="_source_name_obs_name_uc"),
    )

    def __init__(self, **kwargs):
        """
        This class is used to store raw photometric data,
        that should probably need to be reduced into more
        manageable forms (saved using other subclasses).

        Parameters are the same as in as the __init__ of the Dataset class.

        """
        DatasetMixin.__init__(self, **kwargs)

    def __repr__(self):
        string = f"{self.__class__.__name__}(type={self.type}"

        if self.source_name is not None:
            string += f", source={self.source_name}"

        if self.observatory:
            string += f" ({self.observatory.upper()})"

        string += f", epochs={self.number}"
        string += f", file: {self.filename}"

        if self.filekey:
            string += f" (key: {self.filekey})"

        string += ")"

        return string

    @property
    def type(self):
        return "photometry"

    @staticmethod
    def make_random_photometry(
        number=100,
        mag_min=15,
        mag_max=20,
        magerr_min=0.05,
        magerr_max=0.2,
        mjd_min=57000,
        mjd_max=58000,
        oid_min=0,
        oid_max=5,
        filters=["g", "r", "i"],
        exptime=30,
        ra=None,
        ra_scatter=0.001,
        dec=None,
        dec_scatter=0.001,
    ):
        """
        Make a random photometry dataset,
        with randomly generated mjds, mags, magerrs, filters, and object ids.

        Parameters
        ----------
        number: int
            Number of observations to generate
        mag_min: float
            Minimum (brightest) magnitude to generate
        mag_max: float
            Maximum (faintest) magnitude to generate
        magerr_min: float
            Minimum magnitude error to generate
        magerr_max: float
            Maximum magnitude error to generate
        mjd_min: float
            Minimum MJD to generate
        mjd_max: float
            Maximum MJD to generate
        oid_min: int
            Minimum object id to generate
        oid_max: int
            Maximum object id to generate
        filters: list of str
            List of filters to generate
        exptime: float, optional
            Exposure time to generate
            If not given, the photometry points
            will not have an exposure time column.
        ra: float, optional
            Right ascension to generate the photometry around.
            If None, will randomly choose a point in the sky.
            Should be given in degrees!
        ra_scatter: float
            Scatter in right ascension to generate the photometry around.
            Should be given in degrees!
        dec: float, optional
            Declination to generate the photometry around.
            If None, will randomly choose a point in the sky.
            Should be given in degrees!
        dec_scatter: float
            Scatter in declination to generate the photometry around.
            Should be given in degrees!

        Returns
        -------

        """

        if not isinstance(filters, list):
            raise ValueError("filters must be a list of strings")

        mean_mag = np.random.uniform(mag_min, mag_max)
        filt = np.random.choice(filters, number)
        mjd = np.random.uniform(mjd_min, mjd_max, number)
        mag_err = np.random.uniform(magerr_min, magerr_max, number)
        mag = np.array([np.random.normal(mean_mag, err) for err in mag_err])
        oid = np.random.randint(oid_min, oid_max, number)
        test_data = dict(mjd=mjd, mag=mag, mag_err=mag_err, filter=filt, oid=oid)
        df = pd.DataFrame(test_data)
        if ra is None:
            ra = np.random.uniform(0, 360)
        if dec is None:
            dec = np.random.uniform(-90, 90)

        df["ra"] = ra + np.random.normal(0, ra_scatter, number)
        df["dec"] = dec + np.random.normal(0, dec_scatter, number)

        if exptime:
            df["exptime"] = exptime

        return df


class Lightcurve(DatasetMixin, Base):

    __tablename__ = "lightcurves"

    __table_args__ = (
        UniqueConstraint(
            "source_name",
            "observatory",
            "series_number",
            "simulation_number",
            "was_processed",
            "cfg_hash",
            name="_lightcurve_uc",
        ),
    )

    if True:  # put all the column definitions in one block
        # source_id = sa.Column(
        #     sa.Integer,
        #     sa.ForeignKey("sources.id", ondelete="CASCADE"),
        #     nullable=False,
        #     index=True,
        #     doc="ID of the source this dataset is associated with",
        # )
        #
        # source = orm.relationship(
        #     "Source",
        #     doc="Source associated with this lightcurve dataset",
        #     cascade="all",
        #     foreign_keys="Lightcurve.source_id",
        # )
        #
        # source_name = association_proxy("source", "name")

        project = sa.Column(
            sa.String,
            nullable=False,
            index=True,
            doc="Project this lightcurve is associated with",
        )

        cfg_hash = sa.Column(
            sa.String,
            nullable=False,
            index=True,
            default="",
            doc="Hash of the configuration used to generate this object."
            "(leave empty if not using version control)",
        )

        raw_data_id = sa.Column(
            sa.Integer,
            sa.ForeignKey("raw_photometry.id", ondelete="CASCADE"),
            index=True,
            doc="ID of the raw dataset that was used "
            "to produce this reduced dataset.",
        )

        raw_data_filename = sa.Column(
            sa.String,
            nullable=True,
            index=True,
            doc="Filename of the raw dataset that "
            "was used to produce this reduced dataset.",
        )

        series_number = sa.Column(
            sa.Integer,
            nullable=False,
            default=1,
            index=True,
            doc="Serial number for this reduced dataset, "
            "numbering it out of all reduced datasets "
            "produced from the same raw data.",
        )

        series_total = sa.Column(
            sa.Integer,
            nullable=False,
            default=1,
            index=True,
            doc="Total number of reduced datasets, produced from the same raw data.",
        )

        simulation_number = sa.Column(
            sa.Integer,
            nullable=True,
            index=False,
            doc="Serial number for this reduced and simulated lightcurve. "
            "For each reduced lightcurve can have multiple simulated ones. "
            "This keeps track of the injection number for each reduced lightcurve.",
        )

        simulation_total = sa.Column(
            sa.Integer,
            nullable=True,
            index=False,
            doc="Total number of simulated lightcurves made "
            "for each reduced lightcurve. ",
        )

        num_good = sa.Column(
            sa.Integer,
            nullable=False,
            index=True,
            doc="Number of good points in the lightcurve.",
        )

        flux_mean = sa.Column(
            sa.Float,
            nullable=True,
            index=True,
            doc="Mean flux of the dataset",
        )

        @property
        def mag_mean(self):
            if self.flux_mean is None:
                return None
            return -2.5 * np.log10(self.flux_mean) + PHOT_ZP

        flux_rms = sa.Column(
            sa.Float,
            nullable=True,
            index=True,
            doc="Mean flux scatter of the dataset",
        )

        @property
        def mag_rms(self):
            if self.flux_mean_robust and self.flux_rms is not None:
                return self.flux_rms / self.flux_mean_robust / LOG_BASE
            else:
                return None

        flux_mean_robust = sa.Column(
            sa.Float,
            nullable=True,
            index=True,
            doc="Robust mean flux of the dataset" "calculated using sigma clipping",
        )

        @property
        def mag_mean_robust(self):
            if self.flux_mean_robust is None:
                return None
            return -2.5 * np.log10(self.flux_mean_robust) + PHOT_ZP

        flux_rms_robust = sa.Column(
            sa.Float,
            nullable=True,
            index=True,
            doc="Robust mean flux scatter of the dataset"
            "calculated using sigma clipping",
        )

        @property
        def mag_rms_robust(self):
            if self.flux_mean_robust and self.flux_rms_robust is not None:
                return self.flux_rms_robust / self.flux_mean_robust / LOG_BASE
            else:
                return None

        flux_max = sa.Column(
            sa.Float,
            nullable=True,
            index=True,
            doc="Maximum flux of the dataset",
        )

        @property
        def mag_brightest(self):
            if self.flux_max is None:
                return None
            return -2.5 * np.log10(self.flux_max) + PHOT_ZP

        flux_min = sa.Column(
            sa.Float,
            nullable=True,
            index=True,
            doc="Minimum flux of the dataset",
        )

        @property
        def mag_faintest(self):
            if self.flux_min is None:
                return None
            return -2.5 * np.log10(self.flux_min) + PHOT_ZP

        snr_max = sa.Column(
            sa.Float,
            nullable=True,
            index=True,
            doc="Maximum S/N of the dataset",
        )

        snr_min = sa.Column(
            sa.Float,
            nullable=True,
            index=True,
            doc="Minimum S/N of the dataset",
        )

        snr_median = sa.Column(
            sa.Float,
            nullable=True,
            index=True,
            doc="Mean S/N of the dataset",
        )

        dsnr_max = sa.Column(
            sa.Float,
            nullable=True,
            index=True,
            doc="Maximum delta S/N of the dataset",
        )

        dsnr_min = sa.Column(
            sa.Float,
            nullable=True,
            index=True,
            doc="Minimum delta S/N of the dataset",
        )

        dmag_brightest = sa.Column(
            sa.Float,
            nullable=True,
            index=True,
            doc="Maximum delta mag (brightest magnitude) of the change of flux",
        )

        dmag_faintest = sa.Column(
            sa.Float,
            nullable=True,
            index=True,
            doc="Minimum delta mag (faintest magnitude) of the change of flux",
        )

        filter = sa.Column(
            sa.String,
            nullable=False,
            index=True,
            doc="Filter used to acquire this dataset",
        )

        exp_time = sa.Column(
            sa.Float,
            nullable=False,
            doc="Median exposure time of each frame, in seconds.",
        )
        frame_rate = sa.Column(
            sa.Float,
            nullable=True,
            doc="Median frame rate (frequency) of exposures in Hz.",
        )
        is_uniformly_sampled = sa.Column(
            sa.Boolean,
            nullable=False,
            default=False,
            doc="Is the dataset sampled uniformly in time?",
        )

        was_processed = sa.Column(
            sa.Boolean,
            nullable=False,
            default=False,
            index=True,
            doc="True when lightcurve has been processed/analyzed "
            "and has quality cuts and S/N applied.",
        )

        is_simulated = sa.Column(
            sa.Boolean,
            nullable=False,
            default=False,
            index=True,
            doc="True when lightcurve is simulated "
            "(or when injected with simulated events).",
        )

    def __init__(self, data=None, **kwargs):
        """
        This class keeps a set of photometric measurements
        for a source, after performing some data reduction.
        The data can be stored in memory as a pandas DataFrame,
        an xarray Dataset, or a numpy array.
        On disk, the reduced data can be stored in a CSV file,
        a NetCDF file, or an HDF5 file.

        Each raw dataset can be associated with zero or more
        reduced datasets, and the data itself can be loaded
        from file when needed.

        Parameters are the same as the __init__ of the Dataset class.

        """
        if isinstance(data, (pd.DataFrame, xr.Dataset, np.ndarray)):
            kwargs["data"] = data
        if isinstance(data, Lightcurve):
            other = data
            for k, v in other.__dict__.items():
                if k in [
                    "_sa_instance_state",
                    "data",
                    "_data",
                    "id",
                    "_filename",
                    "filename",
                    "_filekey",
                    "filekey",
                ]:
                    continue

                if hasattr(v, "_sa_instance_state"):
                    # don't deep copy a SA managed object!
                    setattr(self, k, v)
                    continue

                setattr(self, k, deepcopy(v))

            # make sure to grab the data as well
            self.data = other.data.copy(deep=True)
            return

        if "data" not in kwargs:
            raise ValueError("Lightcurve must be initialized with data")

        self.filtmap = None  # get this as possible argument

        DatasetMixin.__init__(self, **kwargs)

        # if given a colmap, make sure to remove columns
        # that are not in the data anymore
        for k, v in self.colmap.copy().items():
            if v not in data.columns:
                self.colmap.pop(k)

        fcol = self.colmap.get("filter")  # shorthand

        if fcol is None:  # no filter, use observatory name instead
            self.filter = self.observatory.upper()
        else:  # use the filtmap to convert to a standard filter name
            # replace the filter name with a more specific one
            if "filtmap" in kwargs and kwargs["filtmap"] is not None:
                if isinstance(kwargs["filtmap"], dict):

                    def filter_mapping(filt):
                        return kwargs["filtmap"].get(filt, filt)

                elif isinstance(kwargs["filtmap"], str):

                    def filter_mapping(filt):
                        new_filt = kwargs["filtmap"]
                        if self.observatory:
                            new_filt = new_filt.replace(
                                "<observatory>", self.observatory.lower()
                            )
                        return new_filt.replace("<filter>", filt)

                self.data.loc[:, fcol] = self.data.loc[:, fcol].map(filter_mapping)

            filters = self.data[fcol].unique()
            if len(filters) > 1:
                raise ValueError("All filters must be the same for a Lightcurve")
            self.filter = filters[0]

        # sort the data by time it was recorded
        if isinstance(self.data, pd.DataFrame):
            self.data = self.data.sort_values([self.colmap["time"]], inplace=False)
            self.data.reset_index(drop=True, inplace=True)

        # get flux from mag or vice-versa
        self._calc_mag_flux()

        # make sure keys in altdata are standardized
        self._translate_altdata()

        # find exposure time, frame rate, uniformity
        self._find_cadence()

        # get averages and standard deviations
        self._calc_stats()

        # get the signal-to-noise ratio
        self._calc_snr()

        # get the peak flux and S/N
        self._calc_best()

        # remove columns we don't use
        self._drop_and_rename_columns()

    def __repr__(self):
        string = []
        string.append(
            f"{self.__class__.__name__}("
            f"id={self.id}, "
            f"source={self.source_name}, "
            f"epochs={self.number}"
        )
        if self.was_processed:
            string.append("processed")
            if self.is_simulated:
                string[-1] += " (sim)"
        else:
            string.append("reduced")

        if self.observatory:
            string.append(self.observatory.upper())
        if self.mag_mean_robust is not None and self.mag_rms_robust is not None:
            string.append(
                f"mag[{self.filter}]={self.mag_mean_robust:.2f}\u00B1{self.mag_rms_robust:.2f}"
            )
        string.append(f"file: {self.filename}")

        if self.filekey:
            string.append(f"key: {self.filekey}")
        string = ", ".join(string)
        string += ")"

        return string

    @property
    def type(self):
        return "photometry"

    # overload the DatasetMixin method
    def _invent_filekey(self, source_name=None, prefix=None, suffix=None):
        DatasetMixin._invent_filekey(self, source_name, prefix, suffix)

        number = self.series_number if self.series_number else 0
        total = self.series_total if self.series_total else 0

        if not self.was_processed:  # reduced (unprocessed) data
            self.filekey += f"_reduction_{number:02d}_of_{total:02d}"
        else:  # processed lightcurve
            self.filekey += f"_processed_{number:02d}_of_{total:02d}"

        if self.is_simulated:
            self.filekey += f"_simulated_{self.simulation_number:02d}_of_{self.simulation_total:02d}"

    def _translate_altdata(self):
        """
        Change the column names given in altdata
        to conform to internal naming convention.
        E.g., change an entry for "exposure_time"
        to one named "exptime".
        """

        if self.altdata is None:
            return

        for key, value in self.altdata.copy().items():
            if simplify(key) == "exposuretime":
                self.altdata["exptime"] = value
                del self.altdata[key]

    def _calc_mag_flux(self):
        """
        Calculate the flux from the magnitude,
        or the magnitude from the flux
        (if both are given, do nothing).
        This also updates the colmap.

        """
        # make sure there is some photometric data available
        if "mag" not in self.colmap and "flux" not in self.colmap:
            raise ValueError("No magnitude or flux column found in data.")

        # calculate the fluxes from the magnitudes
        if "mag" in self.colmap and "flux" not in self.colmap:
            mags = self.data[self.colmap["mag"]]
            fluxes = 10 ** ((-mags + PHOT_ZP) / 2.5)
            self.data["flux"] = fluxes
            self.colmap["flux"] = "flux"

            # what about the errors?
            if "magerr" in self.colmap:
                magerr = self.data[self.colmap["magerr"]]
                self.data["fluxerr"] = fluxes * magerr * LOG_BASE
                self.colmap["fluxerr"] = "fluxerr"

        # calculate the magnitudes from the fluxes
        if "mag" not in self.colmap and "flux" in self.colmap:
            fluxes = self.data[self.colmap["flux"]]
            # calculate the magnitudes from the fluxes
            good_points = np.logical_and(np.invert(np.isnan(fluxes)), fluxes > 0)
            mags = -2.5 * np.log10(fluxes, where=good_points) + PHOT_ZP
            mags[np.invert(good_points)] = np.nan
            self.data["mag"] = mags
            self.colmap["mag"] = "mag"

            # what about the errors?
            if "fluxerr" in self.colmap:
                fluxerr = self.data[self.colmap["fluxerr"]]
                magerr = fluxerr / fluxes / LOG_BASE
                magerr[np.invert(good_points)] = np.nan
                self.data["magerr"] = magerr
                self.colmap["magerr"] = "magerr"

        # TODO: should there be another option for when both are given?

    def _find_cadence(self):
        """
        Find the exposure time and frame rate of the data.
        """
        if "exptime" in self.colmap:
            self.exp_time = float(np.nanmedian(self.data[self.colmap["exptime"]]))
        elif self.altdata:

            keys = ["exp_time", "exptime", "exposure_time", "exposuretime"]
            for key in keys:
                if key in self.altdata:
                    self.exp_time = float(self.altdata[key])
                    break

        if self.exp_time is None:
            raise ValueError("No exposure time found in data or altdata.")

        if len(self.times) > 1:
            dt = np.diff(self.times.astype(np.datetime64))
            dt = dt.astype(np.int64) / 1e6  # convert microseconds to seconds
            self.frame_rate = float(1 / np.nanmedian(dt))

            # check the relative amplitude of the time difference between measurements.
            dt_amp = np.quantile(dt, 0.95) - np.quantile(dt, 0.05)
            dt_amp *= self.frame_rate  # divide by median(dt)
            self.is_uniformly_sampled = dt_amp < UNIFORMITY_THRESHOLD

    def _calc_stats(self):
        """
        Calculate summary statistics on this lightcurve.
        """
        fluxes = self.data[self.colmap["flux"]]

        if "flag" in self.colmap:
            flags = self.data[self.colmap["flag"]].values.astype(bool)
            fluxes = fluxes[np.invert(flags)]

        self.flux_mean = float(np.nanmean(fluxes)) if len(fluxes) else None
        self.flux_rms = float(np.nanstd(fluxes)) if len(fluxes) else None

        # robust statistics
        self.flux_mean_robust, self.flux_rms_robust = self._sigma_clipping(fluxes)

        # only count the good points
        self.num_good = len(fluxes)
        # additional statistics like first/last detected?

    @staticmethod
    def _sigma_clipping(input_values, iterations=3, sigma=3.0):
        """
        Calculate a robust estimate of the mean and scatter
        of the values given to it, using a few iterations
        of finding the median and standard deviation from it,
        and removing any outliers more than "sigma" times
        from that median.
        If the number of samples is less than 5,
        the function returns the nanmedian and nanstd of
        those values without removing outliers.

        Parameters
        ----------
        input_values: one dimensional array of floats
            The input values, either magnitudes or fluxes.
        iterations: int scalar
            Maximum number of iterations to use to remove
            outliers. If no outliers are found, the loop
            is cut short. Default is 3.
        sigma: float scalar
            How many times the standard deviation should
            a measurement fall from the median value,
            to be considered an outlier.
            Default is 3.0.

        Returns
        -------
        2-tuple of floats
            get the median and scatter (RMS) of the distribution,
            without including outliers.
        """

        if len(input_values) == 0:
            return None, None

        if len(input_values) < 5:
            return np.nanmedian(input_values), np.nanstd(input_values)

        values = input_values.copy()

        mean_value = np.nanmedian(values)
        scatter = np.nanstd(values)
        num_values_prev = np.sum(np.isnan(values) == 0)

        for i in range(iterations):
            # remove outliers
            values[abs(values - mean_value) / scatter > sigma] = np.nan

            num_values = np.sum(np.isnan(values) == 0)

            # check if there are no new outliers removed this iteration
            # OR don't proceed with too few data points
            if num_values_prev == num_values or num_values < 5:
                break

            num_values_prev = num_values
            mean_value = np.nanmean(values)
            scatter = np.nanstd(values)

        return float(mean_value), float(scatter)

    def _calc_snr(self):
        fluxes = self.data[self.colmap["flux"]]
        fluxerrs = self.data[self.colmap["fluxerr"]]
        if self.flux_rms_robust:
            worst_err = np.maximum(self.flux_rms_robust, fluxerrs)
        else:
            worst_err = fluxerrs

        # signal to noise ratio of flux (using the biggest error of the two)
        self.data["snr"] = fluxes / worst_err

        # signal to noise ratio of the flux residuals after removing the mean
        # TODO: replace this with the s/n of the "detrend flux"?
        self.data["dsnr"] = (fluxes - self.flux_mean_robust) / worst_err

        # the amount of magnification of the flux relative to the mean, in units of magnitudes (delta-mag)
        lup_flux = luptitudes(fluxes, self.flux_rms_robust)
        lup_mean = luptitudes(self.flux_mean_robust, self.flux_rms_robust)
        self.data["dmag"] = lup_mean - lup_flux  # positive dmag means brighter!

        self.colmap["snr"] = "snr"
        self.colmap["dsnr"] = "dsnr"
        self.colmap["dmag"] = "dmag"

    def _calc_best(self):
        """
        Find some minimal/maximal S/N values
        and other similar properties on the data.
        """

        snr = self.data[self.colmap["snr"]]
        dsnr = self.data[self.colmap["dsnr"]]
        dmag = self.data[self.colmap["dmag"]]
        flux = self.data[self.colmap["flux"]]

        if "flag" in self.colmap:
            flags = self.data[self.colmap["flag"]].values.astype(bool)
            flux = flux[np.invert(flags)]
            snr = snr[np.invert(flags)]
            dsnr = dsnr[np.invert(flags)]
            dmag = dmag[np.invert(flags)]

        if len(snr) > 0:
            self.flux_max = float(np.nanmax(flux))
            self.flux_min = float(np.nanmin(flux))

            self.snr_max = float(np.nanmax(snr))
            self.snr_min = float(np.nanmin(snr))

            self.dsnr_max = float(np.nanmax(dsnr))
            self.dsnr_min = float(np.nanmin(dsnr))

            self.dmag_brightest = float(np.nanmax(dmag))
            self.dmag_faintest = float(np.nanmin(dmag))

    def _drop_and_rename_columns(self):
        """
        Remove from the underying data the
        unused columns (those not defined in
        the colmap) and then rename them to match
        the names in the colmap (e.g., filtercode-> filter).

        This also changes the colmap so it has all
        the new column names (it becomes an identity map),
        to enable backward compatibility with methods that
        need to access the data in its original form.

        It also adds a column named "mjd" for convenience.
        """

        inv_colmap = {v: k for k, v in self.colmap.items()}
        new_cols = list(inv_colmap.keys())
        if isinstance(self.data, pd.DataFrame):
            self.data = self.data[new_cols]
            self.data.rename(columns=inv_colmap, inplace=True)

            # just in case we still use the colmap, it should
            # also point back to the new column names
            self.colmap = {k: k for k in self.colmap.keys()}

            # add a column with the MJD
            self.data["mjd"] = self.mjds
            self.colmap["mjd"] = "mjd"

        # what about other data types, e.g., xarrays?
        elif isinstance(self.data, np.ndarray):
            raise ValueError("Not implemented yet.")
        elif isinstance(self.data, xr.Dataset):
            raise ValueError("Not implemented yet.")
        else:
            raise ValueError(f"Unknown data type: {type(self.data)}")

    def copy(self):
        for k, v in self.__dict__.items():
            if k != "_sa_instance_state":
                pass
        # TODO: we need to finish this!

    def _get_filter_plot_color(self):
        """
        Get a color for plotting the lightcurve.
        """
        colors = {
            "g": "green",
            "zg": "green",
            "r": "red",
            "zr": "red",
            "i": "#660000",
            "zi": "#660000",
            "b": "blue",
        }
        default_color = "#ff00ff"

        return colors.get(self.filter.lower(), default_color)

    def plot(
        self, ttype="mjd", ftype="mag", threshold=5.0, font_size=16, ax=None, **kwargs
    ):
        """
        Plot the lightcurve.

        Parameters
        ----------
        ttype : str
            The type of the x-axis. Can be "mjd" or "times".
        ftype : str
            The type of the y-axis. Can be "mag" or "flux".
        threshold : float
            The threshold for the S/N to mark outliers.
            Default is 5.0.
        font_size : int
            The font size for the plot. Default is 16.
        ax : matplotlib.axes.Axes
            The axes to plot on. If None, a new figure is created.
        kwargs: dict
            Additional keyword arguments to pass to the matplotlib plot function.

        """

        # parameter validation
        if ax is None:
            fig, ax = plt.subplots()

        if ttype == "times":
            t = self.times
        elif ttype == "mjd":
            t = self.mjds
        else:
            raise ValueError('ttype must be either "times" or "mjd"')

        if ftype == "mag":
            m = self.data[self.colmap["mag"]] if "mag" in self.colmap else None
            e = self.data[self.colmap["magerr"]] if "magerr" in self.colmap else None
        elif ftype == "flux":
            m = self.data[self.colmap["flux"]] if "flux" in self.colmap else None
            e = self.data[self.colmap["fluxerr"]] if "fluxerr" in self.colmap else None
        else:
            raise ValueError('ftype must be either "mag" or "flux"')

        if m is None:
            return ax  # short circuit if no data

        # color options, etc
        options = dict(fmt="o", color=self._get_filter_plot_color(), zorder=1)
        options.update(dict(label=f"{self.filter} {ftype} values"))
        options.update(kwargs)

        # actual plot function (with or without errors)
        if e is not None:
            ax.errorbar(t, m, e, **options)
        else:
            ax.plot(t, m, **options)

        # add labels like "MJD" and "mag" to axes
        self._axis_labels(ax, ttype=ttype, ftype=ftype, font_size=font_size)

        # add area scatter
        if ftype == "mag":
            mean_value = np.ones(len(t)) * self.mag_mean_robust
            scatter = np.ones(len(t)) * self.mag_rms_robust
        elif ftype == "flux":
            mean_value = np.ones(len(t)) * self.flux_mean_robust
            scatter = np.ones(len(t)) * self.flux_rms_robust
        ax.fill_between(
            t,
            mean_value - 3 * scatter,
            mean_value + 3 * scatter,
            color=self._get_filter_plot_color(),
            zorder=0,
            alpha=0.2,
            label=f"{self.filter} 3-\u03C3 scatter",
        )

        # add annotations for points with S/N above 5 sigma
        det_idx = np.where(abs(self.data["dsnr"]) > threshold)[0]
        for i in det_idx:
            if self.data[self.colmap["flag"]][i] == 0:
                ax.annotate(
                    text=f' {self.data["dsnr"][i]:.2f}',
                    xy=(t[i], m[i]),
                )

        # setup the axis position for the legend
        pos = ax.get_position()
        pos.y0 = 0.2
        pos.y1 = 0.98
        pos.x0 = 0.1
        pos.x1 = 0.7
        ax.set_position(pos)

        # handle the legend
        # remove repeated labes: https://stackoverflow.com/a/56253636/18256949
        handles, labels = ax.get_legend_handles_labels()
        unique = [
            (h, l)
            for i, (h, l) in enumerate(zip(handles, labels))
            if l not in labels[:i]
        ]
        ax.legend(
            *zip(*unique),
            loc="upper left",
            bbox_to_anchor=(1.02, 1.02),
            fontsize=font_size - 2,
        )

        return ax

    def _get_sncosmo_filter(self, filter):
        if self.observatory.lower() == "ztf":
            d = dict(zg="ztfg", zr="ztfi", zi="ztfz")
        elif self.observatory.lower() == "tess":
            d = dict(tess="tess::red")
        else:
            d = {}

        return d.get(filter.lower(), filter)

    def export_to_skyportal(self, filename="lightcurve.h5", overwrite=False):
        """
        Create an HDF5 file that can be uploaded to SkyPortal
        as a photometric series.

        Parameters
        ----------
        filename: str
            The name of the HDF5 file to create. Default is 'lightcurve.h5'.
            If no extension is given, '.h5' is appended.
        overwrite: bool
            If True, overwrite the file if it already exists.
            Default is False, so if file exists, will raise
            a FileExistsError.
        """
        if len(filename.split(".")) == 1:
            filename += ".h5"

        if not overwrite and os.path.isfile(filename):
            raise FileExistsError(
                f"File {filename} already exists. Set overwrite=True to overwrite."
            )

        df = self.data.copy()
        df.rename(columns={"exptime": "exp_time"}, inplace=True)
        df["mjd"] = self.mjds  # make sure this is not some BJD-offset nonsense

        metadata = dict(
            exp_time=self.exp_time,
            filter=self._get_sncosmo_filter(self.filter),
            ra=self.altdata["ra"],
            dec=self.altdata["dec"],
            series_name=self.altdata["series_name"],
            series_obj_id=self.altdata["object_id"],
        )

        if "ra_err" in self.altdata:
            metadata["ra_unc"] = self.altdata["ra_err"]
        if "dec_err" in self.altdata:
            metadata["dec_unc"] = self.altdata["dec_err"]
        if "time_stamp_alignment" in self.altdata:
            metadata["time_stamp_alignment"] = self.altdata["time_stamp_alignment"]

        for k, v in metadata.items():
            if k is None:
                raise ValueError(f"metadata key {k} is None")

        with pd.HDFStore(filename, mode="w") as store:
            store.put(
                "photometry",
                df,
                format="table",
                index=None,
                track_times=False,
            )
            if metadata is not None:
                store.get_storer("photometry").attrs.metadata = metadata


# make sure all the tables exist
RawPhotometry.metadata.create_all(engine)
Lightcurve.metadata.create_all(engine)

# add relationships between sources and data

# this maintains a many-to-many relationship between
# raw data and sources, because multiple sources
# from different projects/git hashes can access
# the same raw data
# ref: https://docs.sqlalchemy.org/en/14/orm/basic_relationships.html#many-to-many
# source_raw_photometry_association = sa.Table(
#     "source_raw_photometry_association",
#     Base.metadata,
#     sa.Column(
#         "source_id",
#         sa.Integer,
#         sa.ForeignKey("sources.id", ondelete="CASCADE"),
#         primary_key=True,
#     ),
#     sa.Column(
#         "raw_photometry_id",
#         sa.Integer,
#         sa.ForeignKey("raw_photometry.id", ondelete="CASCADE"),
#         primary_key=True,
#     ),
# )
#
# Source.raw_photometry = orm.relationship(
#     "RawPhotometry",
#     secondary=source_raw_photometry_association,
#     back_populates="sources",
#     lazy="selectin",
#     cascade="",
#     order_by="RawPhotometry.time_start",
#     doc="Raw photometry associated with this source",
# )
#
#
# RawPhotometry.sources = orm.relationship(
#     "Source",
#     secondary=source_raw_photometry_association,
#     back_populates="raw_photometry",
#     lazy="selectin",
#     cascade="",
#     doc="Sources associated with this raw photometry",
# )


# Source._reduced_photometry_from_db = orm.relationship(
#     "Lightcurve",
#     primaryjoin="and_(Lightcurve.source_id==Source.id, "
#     "Lightcurve.was_processed==False, "
#     "Lightcurve.is_simulated==False)",
#     back_populates="source",
#     overlaps="_processed_photometry_from_db, _simulated_photometry_from_db",
#     cascade="save-update, merge, refresh-expire, expunge",
#     lazy="selectin",
#     single_parent=True,
#     # passive_deletes=True,
#     order_by="Lightcurve.time_start",
#     doc="Reduced photometric datasets associated with this source",
# )

# Source.redu_lcs = add_alias("reduced_photometry")


# Source._processed_photometry_from_db = orm.relationship(
#     "Lightcurve",
#     primaryjoin="and_(Lightcurve.source_id==Source.id, "
#     "Lightcurve.was_processed==True, "
#     "Lightcurve.is_simulated==False)",
#     back_populates="source",
#     overlaps="_reduced_photometry_from_db, _simulated_photometry_from_db",
#     cascade="save-update, merge, refresh-expire, expunge",
#     lazy="selectin",
#     single_parent=True,
#     # passive_deletes=True,
#     order_by="Lightcurve.time_start",
#     doc="Reduced and processed photometric datasets associated with this source",
# )

# Source.proc_lcs = add_alias("processed_photometry")

# Source._simulated_photometry_from_db = orm.relationship(
#     "Lightcurve",
#     primaryjoin="and_(Lightcurve.source_id==Source.id, "
#     "Lightcurve.was_processed==True, "
#     "Lightcurve.is_simulated==True)",
#     back_populates="source",
#     overlaps="_reduced_photometry_from_db, _processed_photometry_from_db",
#     cascade="save-update, merge, refresh-expire, expunge",
#     lazy="selectin",
#     single_parent=True,
#     # passive_deletes=True,
#     order_by="Lightcurve.time_start",
#     doc="Reduced and simulated photometric datasets associated with this source",
# )

# Source.sim_lcs = add_alias("simulated_photometry")


# RawPhotometry does NOT link back to ALL associated lightcurves
# If we delete a RawPhotometry, it will cascade to delete all
# the associated Lightcurves using the foreign key's ondelete="CASCADE"
# (which means postres will delete it for us)
# but the ORM doesn't know that each RawPhotometry is associated with
# a lightcurve, and doesn't add or delete them when we do suff with
# the RawPhotometry object
# RawPhotometry.lightcurves = orm.relationship(
#     "Lightcurve",
#     back_populates="raw_data",
#     cascade="delete",  # do not automatically add reduced/processed lightcurves
#     doc="Lightcurves derived from this raw dataset.",
# )
#
#
# Lightcurve.raw_data = orm.relationship(
#     "RawPhotometry",
#     # back_populates="lightcurves",
#     cascade="",
#     doc="The raw dataset that was used to produce this reduced dataset.",
# )


@event.listens_for(RawPhotometry, "before_insert")
@event.listens_for(Lightcurve, "before_insert")
def insert_new_dataset(mapper, connection, target):
    """
    This function is called before a new dataset is inserted into the database.
    It checks that a file is associated with this object
    and if it doesn't exist, it creates it, if autosave is True,
    otherwise it raises an error.
    """
    if not target.check_file_exists():
        if target.autosave:
            target.save()
        else:
            if target.filename is None:
                raise ValueError(
                    f"No filename specified for {target}. "
                    "Save the dataset to disk to generate a filename. "
                )
            raise ValueError(
                f"File {target.get_fullname()}"
                "does not exist and autosave is disabled. "
                "Please create the file manually."
            )


@event.listens_for(Lightcurve, "after_delete")
def delete_dataset(mapper, connection, target):
    """
    This function is called after a dataset is deleted from the database.
    It checks that a file is associated with this object
    and if it exists, it deletes it.
    """
    # TODO: maybe add an autodelete attribute?
    #  have it False by default for raw data
    #  and True for reduced/processed data?
    if target.check_file_exists():
        target.delete_data_from_disk()


if __name__ == "__main__":
    import matplotlib
    from src.source import Source

    matplotlib.use("qt5agg")
    with SmartSession() as session:

        sources = session.scalars(sa.select(Source).where(Source.project == "WD")).all()
        if len(sources):
            ax = sources[0].plot_photometry(ttype="mjd", ftype="flux")
