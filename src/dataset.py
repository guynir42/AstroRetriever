import os
import uuid
import string
import warnings
from tables import NaturalNameWarning
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from astropy.time import Time
import h5py

import sqlalchemy as sa
from sqlalchemy import orm, event
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.ext.associationproxy import association_proxy

from sqlalchemy.dialects.postgresql import JSONB

from src.database import Base, Session, engine
from src.source import Source

# root folder is either defined via an environment variable
# or is the in the main repository, under subfolder "data"

from src.database import DATA_ROOT

PHOT_ZP = 23.9
LOG_BASES = np.log(10) / 2.5

AUTOLOAD = True
AUTOSAVE = False
OVERWRITE = False

# 1% difference in time is considered uniform
UNIFORMITY_THRESHOLD = 0.01


def simplify(key):
    return key.lower().replace(" ", "").replace("_", "").replace("-", "")


def add_alias(att):
    return property(
        fget=lambda self: getattr(self, att),
        fset=lambda self, value: setattr(self, att, value),
        doc=f'Alias for "{att}"',
    )


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

    """

    observatory = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc="Observatory this dataset is associated with",
    )

    # original series of images used to make this dataset
    series_identifier = sa.Column(
        sa.String,
        nullable=True,
        index=True,
        doc="Identifier of the series this dataset is associated with, "
        "e.g., the set of images the include this source but others as well",
    )
    series_object = sa.Column(
        sa.String,
        nullable=True,
        index=True,
        doc="Name of object relative to the series this dataset is associated with",
    )

    # saving the data to file
    _filename = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc="Filename of the dataset, including path, relative to the DATA_ROOT",
    )

    _folder = sa.Column(
        sa.String,
        nullable=True,
        doc="Folder where this dataset is stored. "
        "If relative path, it is relative to DATA_ROOT. ",
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

    altdata = sa.Column(
        JSONB,
        nullable=True,
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
    shape = sa.Column(sa.ARRAY(sa.Integer), nullable=False, doc="Shape of the dataset")
    number = sa.Column(
        sa.Integer, nullable=False, doc="Number of observations/epochs in the dataset"
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
        self._data = None
        self.colmap = {}
        self._times = None
        self._mjds = None
        self.time_info = {}

        # these are only set when generating a new
        # object, not when loading from database
        self.autoload = AUTOLOAD
        self.autosave = AUTOSAVE
        self.overwrite = OVERWRITE

        # first input data to allow
        # the object to calculate some attributes
        if "data" in kwargs:
            self.data = kwargs.pop("data")

        # override any existing attributes
        for k, v in list(kwargs.items()):
            if hasattr(self, k):
                setattr(self, k, kwargs.pop(k))

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
            self.format = self.guess_format()

        # TODO: figure out the series identifier and object

    @orm.reconstructor
    def init_on_load(self):
        """
        This is called when the object
        is loaded from the database.
        ref: https://docs.sqlalchemy.org/en/14/orm/constructors.html
        """
        self._data = None
        self.colmap = {}
        self._times = None
        self._mjds = None
        self.time_info = {}

    def guess_format(self):
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

    def guess_extension(self):
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
        """
        # TODO: implement this
        # ref: https://stackoverflow.com/questions/18089667/how-to-estimate-how-much-memory-a-pandas-dataframe-will-need
        return 0

    def get_path(self):
        """
        Get the name of the folder inside
        the DATA_ROOT folder where this dataset is stored.
        """

        if self.folder is not None:
            f = self.folder
        elif hasattr(self, "project") and self.project is not None:
            f = self.project.upper()
        elif self.observatory is not None:
            f = self.observatory.upper()
        else:
            f = "DATA"

        if os.path.isabs(f):
            return f
        else:
            return os.path.join(DATA_ROOT, f)

    def get_fullname(self):
        """
        Get the full path to the data file.
        """
        if self.filename:
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
            self.format = self.guess_format()

        if self.format == "hdf5":
            self.load_hdf5()
        elif self.format == "fits":
            self.load_fits()
        elif self.format == "csv":
            self.load_csv()
        elif self.format == "json":
            self.load_json()
        elif self.format == "netcdf":
            self.load_netcdf()
        else:
            raise ValueError(f"Unknown format {self.format}")

    def load_hdf5(self):
        """
        Load the data from a HDF5 file.
        """
        with pd.HDFStore(self.get_fullname()) as store:
            key = self.filekey
            if key is None:
                if len(store.keys()) == 1:
                    key = store.keys()[0]
                else:
                    raise ValueError("No key specified and multiple keys found in file")

            # load the data
            self.data = store.get(key)
            if self.data is None:
                raise ValueError(f"Key {key} not found in file {self.get_fullname()}")
            # load metadata
            if store.get_storer(key).attrs:
                self.altdata = store.get_storer(key).attrs["altdata"]

    def load_fits(self):
        pass

    def load_csv(self):
        self.data = pd.read_csv(self.get_fullname())

    def load_json(self):
        pass

    def load_netcdf(self):
        pass

    @staticmethod
    def random_string(length=16):
        letters = list(string.ascii_lowercase)
        return "".join(np.random.choice(letters, length))

    def invent_filename(self, ra_deg=None, ra_minute=None, ra_second=None):

        """
        Generate a filename with some pre-defined format
        that is consistent enough to have multiple sources
        saved to the same file in a logical way that is
        easy to figure out even when file data is orphaned
        from the database objects.

        The default way to decide which source goes into which
        file is using RA (right ascension).
        This is a closed interval (0-360) and for most
        all sky surveys the sources are spread out (mostly)
        evenly, although the galactic center may be more dense
        with sources, causing larger files with RA~270.

        If given only ra_deg, will split the sources into
        360 files, one for each integer degree bin.
        The filename will be <Observatory>_<data_type>_RA<ra_deg>.ext
        where .ext is the extension and ra_deg will be a 3-digit integer
        value (zero padded) containing all sources with RA in the range
        [ra_deg, ra_deg+1).

        If ra_minute is given, will split the sources into 360x60 files,
        one for each integer minute bin. That means the filename will be:
        <Observatory>_<data_type>_RA<ra_deg>_<ra_minute>.ext
        If adding seconds, this will be:
        <Observatory>_<data_type>_RA<ra_deg>_<ra_minute>_<ra_second>.ext
        These modes are useful if the survey you are working with
        has very dense coverage in a small area, and you want to
        split the sources into smaller files.

        If not given a source right ascension at all,
        the ra range will be replaced with a random string.
        <observatory>_<data_type>_<random 16 char string>.ext

        For subclasses that are reductions of the data
        found in another file (and have a "raw_data_filename" attribute),
        will just use that filename, appending the string "_reduced"
        or "_processed" or "_simulated" before adding the extension.

        Parameters
        ----------
        ra_deg : int, optional
            The integer degree of the right ascension of the source.
            If given as float, will just use floor(ra_deg).
            If given as None, filename will have a random string instead.
            (that means each source will have its own file).
            The default is None, but it is highly recommended to give
            the RA of the source when saving.
        ra_minute : int, optional
            The integer minute of the right ascension of the source.
            If given as float, will just use floor(ra_minute).
            if given as None, will only split sources into integer
            degree filenames (default).
        ra_second : int, optional
            The integer second of the right ascension of the source.
            If given as float, will just use floor(ra_second).
            if given as None, will only split sources into integer
            degree and possibly minute filenames (default).

        """
        if hasattr(self, "raw_data_filename") and self.raw_data_filename is not None:
            basename = os.path.splitext(self.raw_data_filename)[0]
            if hasattr(self, "was_processed") and self.was_processed:
                self.filename = basename + "_processed"
            else:
                self.filename = basename + "_reduced"
        else:
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
                    binning += f"_{ra:02d}"

                    if ra_second is not None:
                        ra = int(ra_second)
                        binning += f"_{ra:02d}"

            else:
                binning = self.random_string(15)

            # add prefix using the type of data and observatory
            obs = self.observatory.upper() if self.observatory else "UNKNOWN_OBS"
            data_type = self.type if self.type is not None else "Data_"
            self.filename = f"{obs}_{data_type}_{binning}"

        # add extension
        self.filename += self.guess_extension()

    def invent_filekey(self, source_name=None, prefix=None, suffix=None):
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
            if hasattr(self, "source") and self.source is not None:
                source_name = self.source.name
            elif hasattr(self, "sources") and len(self.sources) > 0:
                source_name = self.sources[0].name

        if source_name is not None:
            if isinstance(source_name, bytes):
                source_name = source_name.decode("utf-8")

            if isinstance(source_name, str):
                self.filekey = source_name
            else:
                raise TypeError("source must be a string")
        else:
            self.filekey = self.random_string(8)

        # add the type of data
        self.filekey = f"{self.type}_{self.filekey}"

        if prefix is not None:
            if prefix.endswith("_"):  # remove trailing underscore
                prefix = prefix[:-1]
            self.filekey = f"{prefix}_{self.filekey}"
        if suffix is not None:
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
            if hasattr(self, "source") and self.source is not None:
                ra_deg = self.source.ra
            elif hasattr(self, "sources") and len(self.sources) > 0:
                ra_deg = self.sources[0].ra

        # if no filename/key are given, make them up
        if self.filename is None:
            self.invent_filename(
                ra_deg=ra_deg, ra_minute=ra_minute, ra_second=ra_second
            )

        # for any of the formats where we need an in-file key:
        if self.filekey is None and self.format in ("hdf5",):
            self.invent_filekey(
                source_name=source_name, prefix=key_prefix, suffix=key_suffix
            )

        if overwrite is None:
            overwrite = self.overwrite

        # make a path if missing
        path = os.path.dirname(self.get_fullname())
        if not os.path.isdir(path):
            os.makedirs(path)

        # specific format save functions
        if self.format == "hdf5":
            self.save_hdf5(overwrite)
        elif self.format == "fits":
            self.save_fits(overwrite)
        elif self.format == "csv":
            self.save_csv(overwrite)
        elif self.format == "json":
            self.save_json(overwrite)
        elif self.format == "netcdf":
            self.save_netcdf(overwrite)
        else:
            raise ValueError(f"Unknown format {self.format}")

    def save_hdf5(self, overwrite):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NaturalNameWarning)
            if isinstance(self._data, xr.Dataset):
                # TODO: check if key already exists!
                self.data.to_hdf(
                    self.get_fullname(), key=self.filekey
                )  # this actually works??
            elif isinstance(self._data, pd.DataFrame):
                with pd.HDFStore(self.get_fullname()) as store:
                    if self.filekey in store:
                        if overwrite:
                            store.remove(self.filekey)
                        else:
                            raise ValueError(
                                f"Key {self.filekey} already exists in file {self.get_fullname()}"
                            )

                    store.put(self.filekey, self.data)
                    if self.altdata:
                        store.get_storer(self.filekey).attrs["altdata"] = self.altdata
            elif isinstance(self._data, np.ndarray):
                with h5py.File(self.get_fullname(), "w") as f:
                    f.create_dataset(self.filekey, data=self.data)
                    if self.altdata:
                        for k, v in self.altdata.items():
                            f[self.filekey].attrs[k] = v
            else:
                raise ValueError(f"Unknown data type {type(self._data)}")

    def save_fits(self, overwrite):
        pass

    def save_csv(self, overwrite):
        pass

    def save_json(self, overwrite):
        pass

    def save_netcdf(self, overwrite):
        pass

    def delete_data_from_disk(self):
        """
        Delete the data from disk, if it exists.
        If the format is hdf5, will delete the key from the file.
        If there are no more keys in the file, will delete the file.
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

                # TODO: delete the folder if empty?
                #  maybe add a parameter to do that?

    def find_colmap(self, data):
        """
        Calculate the column map for the data.
        This locates columns in the data that
        correspond to known types of data and
        assigns them.
        """

        if isinstance(data, pd.DataFrame):
            columns = data.columns
        # other datatypes will call this differently...
        # TODO: get the columns for other data types

        for c in columns:  # timestamps
            if simplify(c) in ("jd", "jds", "juliandate", "juliandates"):
                self.time_info["format"] = "mjd"
                self.time_info["to datetime"] = lambda t: Time(
                    t, format="jd", scale="utc"
                ).datetime
                self.time_info["to mjd"] = lambda t: Time(
                    t, format="jd", scale="utc"
                ).mjd
                self.colmap["time"] = c
                break
            elif simplify(c) in ("mjd", "mjds"):
                self.time_info["format"] = "mjd"
                self.time_info["to datetime"] = lambda t: Time(
                    t, format="mjd", scale="utc"
                ).datetime
                self.time_info["to mjd"] = lambda t: t
                self.colmap["time"] = c
                break
            elif simplify(c) in ("time", "times", "datetime", "datetimes"):
                if isinstance(data[c][0], (str, bytes)):
                    if "T" in data[c][0]:
                        self.time_info["format"] = "isot"
                        self.time_info["to datetime"] = lambda t: Time(
                            t, format="isot", scale="utc"
                        ).datetime
                        self.time_info["to mjd"] = lambda t: Time(
                            t, format="isot", scale="utc"
                        ).mjd
                    else:
                        self.time_info["format"] = "iso"
                        self.time_info["to datetime"] = lambda t: Time(
                            t, format="iso", scale="utc"
                        ).datetime
                        self.time_info["to mjd"] = lambda t: Time(
                            t, format="iso", scale="utc"
                        ).mjd
                self.colmap["time"] = c
                break
            elif simplify(c) == "timestamps":
                self.time_info["format"] = "unix"
                self.time_info["to datetime"] = lambda t: Time(
                    t, format="unix", scale="utc"
                ).datetime
                self.time_info["to mjd"] = lambda t: Time(
                    t, format="unix", scale="utc"
                ).mjd
                self.colmap["time"] = c
                break

        for c in columns:  # exposure time
            if simplify(c) in ("exptime", "exptimes", "exposuretime", "exposuretimes"):
                self.colmap["exptime"] = c
                break

        for c in columns:  # right ascension
            if simplify(c) in ("ra", "ras", "rightascension", "rightascensions"):
                self.colmap["ra"] = c
                break

        for c in columns:  # declination
            if simplify(c) in ("dec", "decs", "declination", "declinations"):
                self.colmap["dec"] = c
                break

        for c in columns:  # magnitude
            if simplify(c) in ("mag", "mags", "magnitude", "magnitudes"):
                self.colmap["mag"] = c
                break

        for c in columns:  # magnitude error
            if simplify(c) in ("magerr", "magerr", "magerror"):
                self.colmap["magerr"] = c
                break

        for c in columns:  # fluxes
            if simplify(c) in ("flux", "fluxes", "count", "counts"):
                self.colmap["flux"] = c
                break

        for c in columns:  # flux errors
            if simplify(c) in (
                "fluxerr",
                "fluxerr",
                "fluxerror",
                "counterr",
                "counterror",
            ):
                self.colmap["fluxerr"] = c
                break

        for c in columns:  # filter
            if simplify(c) in ("filt", "filter", "filtername", "filtercode"):
                self.colmap["filter"] = c
                break

        for c in columns:  # bad data flags
            if simplify(c) in ("flag", "catflag", "catflags", "baddata"):
                self.colmap["flag"] = c
                break

    def calc_times(self, data):
        """
        Calculate datetimes and MJDs for each epoch,
        based on the conversions found in self.time_info.
        These values are calculated once when the data
        is loaded from disk or given as input,
        but are not saved in the DB or on disk.
        """
        if len(data) == 0:
            return
        self.times = self.time_info["to datetime"](data[self.colmap["time"]])
        self.time_start = min(self.times)
        self.time_end = max(self.times)

        self.mjds = self.time_info["to mjd"](data[self.colmap["time"]])

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
        self.axis_labels(ax, ttype=ttype, ftype=ftype)  # TODO: add font_size

        return ax

    @staticmethod
    def axis_labels(ax, ttype, ftype, font_size=12):
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
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        """
        The filename ALWAYS refers to the file without path.
        The path should be determined by "folder" and "DATA_ROOT".
        If given a full path to a file, the path is removed.
        If that path is DATA_ROOT/<some_folder> it will assign
        <some_folder> to self.folder.
        If it doesn't fit that but has an absolute path,
        that gets saved in self.folder, which overrides
        the value of DATA_ROOT
        (this is not great, as the data could move around
        and leave the "folder" property fixed in the DB).
        """
        if value is None:
            self._filename = None
            return

        (path, filename) = os.path.split(value)
        self._filename = filename
        if path:
            self.folder = path

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, value):
        """
        Use the folder to find the file under DATA_ROOT.
        If given as a relative path, or an absolute path
        that starts with DATA_ROOT, it will be saved
        relative to DATA_ROOT.
        If it is an absolute path different than
        DATA_ROOT, it will be saved as is.
        This is not great, as the data could move around
        and leave the "folder" property fixed in the DB.
        If you want to deliberately make the folder an
        absolute path with the current value of DATA_ROOT,
        set src.dataset.DATA_ROOT to something else,
        assign the absolute path to "folder" and then
        change back DATA_ROOT to the original value.
        """
        if value is None:
            self._folder = None
            return
        if value.startswith(DATA_ROOT):  # relative to root
            self._folder = value[len(DATA_ROOT) + 1 :]
        else:
            self._folder = value

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

        self.shape = data.shape
        self.number = len(data)  # for imaging data this would be different?
        self.size = self.calc_size()
        self.format = self.guess_format()
        self.find_colmap(data)
        self.calc_times(data)

    @property
    def times(self):
        if self._data is None and self.autoload and self.filename is not None:
            self.load()
        if self._times is None:
            self.calc_times()
        return self._times

    @times.setter
    def times(self, value):
        self._times = value

    @property
    def mjds(self):
        if self._data is None and self.autoload and self.filename is not None:
            self.load()
        if self._mjds is None:
            self.calc_times()
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
        "folder",
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

        if len(self.sources) > 0:
            f", source={self.sources[0].name}"

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

    def make_random_photometry(
        self,
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

        self.data = df


class Lightcurve(DatasetMixin, Base):

    __tablename__ = "lightcurves"

    source_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of the source this dataset is associated with",
    )

    source = orm.relationship(
        "Source",
        doc="Source associated with this lightcurve dataset",
        cascade="all",
        foreign_keys="Lightcurve.source_id",
    )

    source_name = association_proxy("source", "name")

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
        doc="ID of the raw dataset that was used " "to produce this reduced dataset.",
    )

    raw_data_filename = sa.Column(
        sa.String,
        nullable=True,
        index=True,
        doc="Filename of the raw dataset that "
        "was used to produce this reduced dataset.",
    )

    reduction_number = sa.Column(
        sa.Integer,
        nullable=False,
        default=1,
        index=True,
        doc="Serial number for this reduced dataset, "
        "numbering it out of all reduced datasets "
        "producted from the same raw data.",
    )

    reduction_total = sa.Column(
        sa.Integer,
        nullable=False,
        default=1,
        index=True,
        doc="Total number of reduced datasets, " "producted from the same raw data.",
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
        if self.flux_mean and self.flux_rms is not None:
            return self.flux_rms / self.flux_mean / LOG_BASES
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
        doc="Robust mean flux scatter of the dataset" "calculated using sigma clipping",
    )

    @property
    def mag_rms_robust(self):
        if self.flux_mean_robust and self.flux_rms_robust is not None:
            return self.flux_rms_robust / self.flux_mean_robust / LOG_BASES
        else:
            return None

    flux_max = sa.Column(
        sa.Float,
        nullable=True,
        index=True,
        doc="Maximum flux of the dataset",
    )

    @property
    def mag_min(self):
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
    def mag_max(self):
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

    filter = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc="Filter used to acquire this dataset",
    )

    exp_time = sa.Column(
        sa.Float, nullable=False, doc="Median exposure time of each frame, in seconds."
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

            return

        if "data" not in kwargs:
            raise ValueError("Lightcurve must be initialized with data")

        self.filtmap = None  # get this as possible argument
        DatasetMixin.__init__(self, **kwargs)

        try:
            fcol = self.colmap["filter"]  # shorthand

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
                                "<observatory>", self.observatory
                            )
                        return new_filt.replace("<filter>", filt)

                self.data.loc[:, fcol] = self.data.loc[:, fcol].map(filter_mapping)

            filters = self.data[fcol].values
            if not all([f == filters[0] for f in filters]):
                raise ValueError("All filters must be the same for a Lightcurve")
            self.filter = filters[0]

            # sort the data by time it was recorded
            if isinstance(self.data, pd.DataFrame):
                self.data = self.data.sort_values([self.colmap["time"]], inplace=False)
                self.data.reset_index(drop=True, inplace=True)

            # get flux from mag or vice-versa
            self.calc_mag_flux()

            # make sure keys in altdata are standardized
            self.translate_altdata()

            # find exposure time, frame rate, uniformity
            self.find_cadence()

            # get averages and standard deviations
            self.calc_stats()

            # get the signal-to-noise ratio
            self.calc_snr()

            # get the peak flux and S/N
            self.calc_best()

            # remove columns we don't use
            self.drop_columns()

        except Exception:
            # if construction fails, don't want to leave
            # this object attached to the raw data object
            self.raw_data = None
            self.raw_data_id = None
            raise

    def __repr__(self):
        string = (
            f"{self.__class__.__name__}("
            f"id={self.id}, "
            f"source={self.source_name}, "
            f"epochs={self.number}"
        )
        if self.observatory:
            string += f" ({self.observatory.upper()})"
        if self.mag_mean_robust is not None and self.mag_rms_robust is not None:
            string += f", mag[{self.filter}]={self.mag_mean_robust:.2f}\u00B1{self.mag_rms_robust:.2f})"
        string += f", file: {self.filename}"

        if self.filekey:
            string += f" (key: {self.filekey})"

        string += ")"

        return string

    @property
    def type(self):
        return "photometry"

    # overload the DatasetMixin method
    def invent_filekey(self, source_name=None, prefix=None, suffix=None):
        DatasetMixin.invent_filekey(self, source_name, prefix, suffix)

        number = self.reduction_number if self.reduction_number else 1
        total = self.reduction_total if self.reduction_total else 1

        if not self.was_processed:  # reduced (unprocessed) data
            self.filekey += f"_reduction_{number:02d}_of_{total:02d}"
        else:  # processed lightcurve
            self.filekey += f"_processed_{number:02d}_of_{total:02d}"

        if self.is_simulated:
            self.filekey += f"_simulated_{self.simulation_number:02d}_of_{self.simulation_total:02d}"

    def translate_altdata(self):
        """
        Change the column names given in altdata
        to conform to internal naming convention.
        E.g., change an entry for "exposure_time"
        to one named "exptime".
        """

        if self.altdata is None:
            return

        for key, value in self.altdata.items():
            if simplify(key) in ("exposure_time"):
                self.altdata["exptime"] = value
                del self.altdata[key]

    def calc_mag_flux(self):
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
                self.data["fluxerr"] = fluxes * magerr * LOG_BASES
                self.colmap["fluxerr"] = "fluxerr"

        # calculate the magnitudes from the fluxes
        if "mag" not in self.colmap and "flux" in self.colmap:
            fluxes = self.data[self.colmap["flux"]]
            # calculate the magnitudes from the fluxes
            good_points = np.logical_and(np.invert(np.isnan(fluxes)), fluxes > 0)
            mags = -2.5 * np.log10(fluxes, where=good_points) + PHOT_ZP
            mags[np.invert(good_points)] = np.nan
            self.data[self.colmap["mag"]] = mags
            self.colmap["mag"] = "mag"

            # what about the errors?
            if "fluxerr" in self.colmap:
                fluxerr = self.data[self.colmap["fluxerr"]]
                magerr = fluxerr / fluxes / LOG_BASES
                magerr[np.invert(good_points)] = np.nan
                self.data["magerr"] = magerr
                self.colmap["magerr"] = "magerr"

    def find_cadence(self):
        """
        Find the exposure time and frame rate of the data.
        """
        if "exptime" in self.colmap:
            self.exp_time = np.median(self.data[self.colmap["exptime"]])
        elif self.altdata:

            keys = ["exp_time", "exptime", "exposure_time", "exposuretime"]
            for key in keys:
                if key in self.altdata:
                    self.exp_time = self.altdata[key]
                    break

        if self.exp_time is None:
            raise ValueError("No exposure time found in data or altdata.")

        if len(self.times) > 1:
            dt = np.diff(self.times.astype(np.datetime64))
            dt = dt.astype(np.int64) / 1e6  # convert microseconds to seconds
            self.frame_rate = 1 / np.nanmedian(dt)

            # check the relative amplitude of the time difference
            # between measurements.
            dt_amp = np.quantile(dt, 0.95) - np.quantile(dt, 0.05)
            dt_amp *= self.frame_rate  # divide by median(dt)
            self.is_uniformly_sampled = dt_amp < UNIFORMITY_THRESHOLD

    def calc_stats(self):
        """
        Calculate summary statistics on this lightcurve.
        """
        fluxes = self.data[self.colmap["flux"]]

        if "flag" in self.colmap:
            flags = self.data[self.colmap["flag"]].values.astype(bool)
            fluxes = fluxes[np.invert(flags)]

        self.flux_mean = np.nanmean(fluxes) if len(fluxes) else None
        self.flux_rms = np.nanstd(fluxes) if len(fluxes) else None

        # robust statistics
        self.flux_mean_robust, self.flux_rms_robust = self.sigma_clipping(fluxes)

        # only count the good points
        self.num_good = len(fluxes)
        # additional statistics like first/last detected?

    @staticmethod
    def sigma_clipping(input_values, iterations=3, sigma=3.0):
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
            mean_value = np.nanmedian(values)
            scatter = np.nanstd(values)

        return mean_value, scatter

    def calc_snr(self):
        fluxes = self.data[self.colmap["flux"]]
        fluxerrs = self.data[self.colmap["fluxerr"]]
        if self.flux_rms_robust:
            worst_err = np.maximum(self.flux_rms_robust, fluxerrs)
        else:
            worst_err = fluxerrs

        self.data["snr"] = (fluxes - self.flux_mean_robust) / worst_err
        self.colmap["snr"] = "snr"

    def calc_best(self):
        """
        Find some minimal/maximal S/N values
        and other similar properties on the data.
        """

        snr = self.data[self.colmap["snr"]]
        flux = self.data[self.colmap["flux"]]

        if "flag" in self.colmap:
            flags = self.data[self.colmap["flag"]].values.astype(bool)
            snr = snr[np.invert(flags)]
            flux = flux[np.invert(flags)]

        if len(snr) > 0:
            self.snr_max = np.nanmax(snr)
            self.snr_min = np.nanmin(snr)
            self.flux_max = np.nanmax(flux)
            self.flux_min = np.nanmin(flux)

    def drop_columns(self):

        cols = [self.colmap[c] for c in self.colmap.keys()]

        self.data = self.data[cols]
        # what about other data types, e.g., xarrays?

    def copy(self):
        for k, v in self.__dict__.items():
            if k != "_sa_instance_state":
                pass

    def get_filter_plot_color(self):
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

    def plot(self, ttype="mjd", ftype="mag", font_size=16, ax=None, **kwargs):
        """
        Plot the lightcurve.

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
        options = dict(fmt="o", color=self.get_filter_plot_color(), zorder=1)
        options.update(dict(label=f"{self.filter} {ftype} values"))
        options.update(kwargs)

        # actual plot function (with or without errors)
        if e is not None:
            ax.errorbar(t, m, e, **options)
        else:
            ax.plot(t, m, **options)

        # add labels like "MJD" and "mag" to axes
        self.axis_labels(ax, ttype=ttype, ftype=ftype, font_size=font_size)

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
            color=self.get_filter_plot_color(),
            zorder=0,
            alpha=0.2,
            label=f"{self.filter} 3-\u03C3 scatter",
        )

        # add annotations for points with S/N above 5 sigma
        det_idx = np.where(abs(self.data["snr"]) > 5.0)[0]
        for i in det_idx:
            if self.data[self.colmap["flag"]][i] == 0:
                ax.annotate(
                    text=f' {self.data["snr"][i]:.2f}',
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


# make sure all the tables exist
RawPhotometry.metadata.create_all(engine)
Lightcurve.metadata.create_all(engine)

# add relationships between sources and data

# this maintains a many-to-many relationship between
# raw data and sources, because multiple sources
# from different projects/git hashes can access
# the same raw data
# ref: https://docs.sqlalchemy.org/en/14/orm/basic_relationships.html#many-to-many
source_raw_photometry_association = sa.Table(
    "source_raw_photometry_association",
    Base.metadata,
    sa.Column(
        "source_id",
        sa.Integer,
        sa.ForeignKey("sources.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column(
        "raw_photometry_id",
        sa.Integer,
        sa.ForeignKey("raw_photometry.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)

Source.raw_photometry = orm.relationship(
    "RawPhotometry",
    secondary=source_raw_photometry_association,
    back_populates="sources",
    lazy="selectin",
    cascade="all",
    doc="Raw photometry associated with this source",
)


RawPhotometry.sources = orm.relationship(
    "Source",
    secondary=source_raw_photometry_association,
    back_populates="raw_photometry",
    lazy="selectin",
    cascade="all",
    doc="Sources associated with this raw photometry",
)


Source.reduced_lightcurves = orm.relationship(
    "Lightcurve",
    primaryjoin="and_(Lightcurve.source_id==Source.id, "
    "Lightcurve.was_processed==False, "
    "Lightcurve.is_simulated==False)",
    back_populates="source",
    overlaps="processed_lightcurves, simulated_lightcurves",
    cascade="save-update, merge, refresh-expire, expunge, delete, delete-orphan",
    lazy="selectin",
    single_parent=True,
    passive_deletes=True,
    doc="Reduced photometric datasets associated with this source",
)

Source.reduced_photometry = add_alias("reduced_lightcurves")
Source.redu_lcs = add_alias("reduced_lightcurves")


Source.processed_lightcurves = orm.relationship(
    "Lightcurve",
    primaryjoin="and_(Lightcurve.source_id==Source.id, "
    "Lightcurve.was_processed==True, "
    "Lightcurve.is_simulated==False)",
    back_populates="source",
    overlaps="reduced_lightcurves, simulated_lightcurves",
    cascade="save-update, merge, refresh-expire, expunge, delete, delete-orphan",
    lazy="selectin",
    single_parent=True,
    passive_deletes=True,
    doc="Reduced and processed photometric datasets associated with this source",
)

Source.processed_photometry = add_alias("processed_lightcurves")
Source.proc_lcs = add_alias("processed_lightcurves")

Source.simulated_lightcurves = orm.relationship(
    "Lightcurve",
    primaryjoin="and_(Lightcurve.source_id==Source.id, "
    "Lightcurve.was_processed==True, "
    "Lightcurve.is_simulated==True)",
    back_populates="source",
    overlaps="reduced_lightcurves, processed_lightcurves",
    cascade="save-update, merge, refresh-expire, expunge, delete, delete-orphan",
    lazy="selectin",
    single_parent=True,
    passive_deletes=True,
    doc="Reduced and simulated photometric datasets associated with this source",
)

Source.processed_photometry = add_alias("processed_lightcurves")
Source.proc_lcs = add_alias("processed_lightcurves")


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


Lightcurve.raw_data = orm.relationship(
    "RawPhotometry",
    # back_populates="lightcurves",
    cascade="all",
    doc="The raw dataset that was used to produce this reduced dataset.",
)


@event.listens_for(RawPhotometry, "before_insert")
@event.listens_for(Lightcurve, "before_insert")
def insert_new_dataset(mapper, connection, target):
    """
    This function is called before a new dataset is inserted into the database.
    It checks that a file is associated with this object
    and if it doesn't exist, it creates it, if autosave is True,
    otherwise it raises an error.
    """

    if target.filename is None:
        raise ValueError(
            "No filename specified for this dataset. "
            "Save the dataset to disk to generate a (random) filename. "
        )

    if not target.check_file_exists():
        if target.autosave:
            target.save()
        else:
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
    with Session() as session:

        sources = session.scalars(sa.select(Source).where(Source.project == "WD")).all()
        if len(sources):
            ax = sources[0].plot_photometry(ttype="mjd", ftype="flux")
