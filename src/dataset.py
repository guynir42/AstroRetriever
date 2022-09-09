import os
import uuid
import string

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
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.dialects.postgresql import JSONB

from src.database import Base, Session, engine


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


class DatasetMixin:
    def __init__(self, **kwargs):
        """
        Produce a Dataset object,
        which maps a location on disk
        with raw data in memory.
        Each Dataset is associated with one source
        (via the source_id foreign key).
        If there are multiple datasets in a file,
        use the "key" parameter to identify which
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
        the RawData class will contain raw data
        directly from the observatory.
        This can be turned into e.g.,
        a Lightcurve using a reducer function
        from the correct observatory object.

        """

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
            self.data = kwargs["data"]
            del kwargs["data"]

        # if no project is given, assume testing
        # and make DB objects that can be deleted
        # override this with kwargs['project'] later
        self.project = "test"

        # override any existing attributes
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if not isinstance(self.filename, (str, type(None))):
            raise ValueError(f"Filename must be a string, not {type(self.filename)}")

        if not isinstance(self.key, (str, int, type(None))):
            raise ValueError(
                f"Key must be a string, int, or None, not {type(self.key)}"
            )

        # guess some attributes that were not given
        if self.format is None:
            self.format = self.guess_format()

        # TODO: figure out the series identifier and object

    def __repr__(self):
        string = (
            f"{self.__class__.__name__}(type={self.type}, "
            f"source={self.source_id}, "
            f"epochs={self.number}"
        )
        if self.observatory:
            string += f" ({self.observatory.upper()})"

        string += f", file: {self.filename}"

        if self.key:
            string += f" (key: {self.key})"

        string += ")"

        return string

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
        using its extention.

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

    def guess_type(self):
        """
        Guess the type of data,
        based on the shape and type of data.

        Returns
        -------
        str
            Can be either photometry, spectrum, image.
        """
        # TODO: make this a little bit smarter...
        return "photometry"

    def calc_size(self):
        """
        Calculate the size of the data file.
        """
        # TODO: implement this
        return 0

    def get_path(self):
        """
        Get the name of the folder inside
        the DATA_ROOT folder where this dataset is stored.
        """

        if self.folder is not None:
            f = self.folder
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
            return os.path.exists(self.get_fullname())
        else:
            return False

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
            key = self.key
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
            self.altdata = store.get_storer(key).attrs

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

    def invent_filename(self, batch=1000, digits=7):
        """
        Generate a random filename.
        If the source id is given,
        will put the data in a file named
        <type>_<source_id_lower>_<source_id_higher>
        (with the appropriate extension).
        The lower and higher would be the previous
        and next batch of sources (given by batch),
        and zero padded to the number given by digits.
        """
        if hasattr(self, "raw_data_filename") and self.raw_data_filename is not None:
            basename = os.path.splitext(self.raw_data_filename)[0]
            self.filename = basename + "_reduced"
        else:
            if self.source_id is not None:
                lower = self.source_id // batch * batch
                higher = (self.source_id // batch + 1) * batch
                self.filename = f"{lower:0{digits}d}_{higher:0{digits}d}"
            else:
                self.filename = self.random_string(15)
            # add prefix using the type of data
            if self.type:
                self.filename = self.type + "_" + self.filename
            else:
                self.filename = "data_" + self.filename

        # add extension
        self.filename += self.guess_extension()

    def save(self, overwrite=None):
        """
        Save the data to disk.

        Parameters
        ----------
        overwrite: bool
            If True, overwrite the file if it already exists.
            If False, raise an error if the file already exists.
            If None, use the "overwrite" attribute of the object

        """
        # if no filename/key are given, make them up
        if self.filename is None:
            self.invent_filename()

        if self.key is None and self.format in ("hdf5"):
            self.key = self.random_string(8)

        if overwrite is None:
            overwrite = self.overwrite
        # if not overwrite and self.check_file_exists():
        #     raise ValueError(f"File {self.get_fullname()} already exists")

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
        if isinstance(self.data, xr.Dataset):
            # TODO: check if key already exists!
            self.data.to_hdf(self.get_fullname(), key=self.key)  # this actually works??
        elif isinstance(self.data, pd.DataFrame):
            with pd.HDFStore(self.get_fullname()) as store:
                if self.key in store:
                    if overwrite:
                        store.remove(self.key)
                    else:
                        raise ValueError(
                            f"Key {self.key} already exists in file {self.get_fullname()}"
                        )

                store.put(self.key, self.data)
                if self.altdata:
                    for k, v in self.altdata.items():
                        setattr(store.get_storer(self.key).attrs, k, v)
        elif isinstance(self.data, np.ndarray):
            with h5py.File(self.get_fullname(), "w") as f:
                f.create_dataset(self.key, data=self.data)
                if self.altdata:
                    for k, v in self.altdata.items():
                        f[self.key].attrs[k] = v
        else:
            raise ValueError(f"Unknown data type {type(self.data)}")

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
        Delete the data file from disk,
        if it exists.
        """
        if self.check_file_exists():
            need_to_delete = False
            if self.format == "hdf5":
                with pd.HDFStore(self.get_fullname()) as store:
                    if self.key in store:
                        store.remove(self.key)
                    if len(store.keys()) == 0:
                        need_to_delete = True

            elif self.format in ("csv", "json"):
                need_to_delete = True
            else:
                raise ValueError(f"Unknown format {self.format}")

            if need_to_delete:
                os.remove(self.get_fullname())

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
        if self.type is None:
            self.type = self.guess_type()
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

    @declared_attr
    def source_id(cls):
        return sa.Column(
            sa.Integer,
            sa.ForeignKey("sources.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
            doc="ID of the source this dataset is associated with",
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

    key = sa.Column(
        sa.String,
        nullable=True,
        doc="Key of the dataset (e.g., in the HDF5 file it would be the group name)",
    )

    type = sa.Column(
        sa.String, nullable=False, default="photometry", doc="Type of the dataset"
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
        "project",
        "observatory",
        "cfg_hash",
        "folder",
    ]

    # automatically update the dictionaries
    # from all parent datasets into a new
    # dictionary in the child dataset(s)
    default_update_attributes = ["altdata"]


class RawData(DatasetMixin, Base):

    __tablename__ = "raw_data"

    def __init__(self, **kwargs):
        """
        This class is used to store raw data from a survey,
        that should probably need to be reduced into more
        manageable forms (saved using other subclasses).

        Parameters are the same as in as the __init__ of the Dataset class.

        """
        DatasetMixin.__init__(self, **kwargs)
        Base.__init__(self)

    def make_random_photometry(
        self,
        number=30,
        mag_min=15,
        mag_max=20,
        magerr_min=0.1,
        magerr_max=0.5,
        mjd_min=57000,
        mjd_max=58000,
        oid_min=0,
        oid_max=5,
        filters=["g", "r", "i"],
        exptime=30,
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

        Returns
        -------

        """

        if not isinstance(filters, list):
            raise ValueError("filters must be a list of strings")

        filt = np.random.choice(filters, number)
        mjd = np.random.uniform(mjd_min, mjd_max, number)
        mag = np.random.uniform(mag_min, mag_max, number)
        mag_err = np.random.uniform(magerr_min, magerr_max, number)
        oid = np.random.randint(oid_min, oid_max, number)
        test_data = dict(mjd=mjd, mag=mag, mag_err=mag_err, filter=filt, oid=oid)
        df = pd.DataFrame(test_data)

        if exptime:
            df["exptime"] = exptime

        self.data = df

    detections_in_time = orm.relationship(
        "DetectionInTime",
        back_populates="raw_data",
        cascade="all, delete-orphan",
    )


class Lightcurve(DatasetMixin, Base):

    __tablename__ = "lightcurves"

    def __init__(self, **kwargs):
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
        if "data" not in kwargs:
            raise ValueError("Lightcurve must be initialized with data")

        DatasetMixin.__init__(self, **kwargs)
        Base.__init__(self)
        self.type = "lightcurves"

        if "raw_data_id" in kwargs:
            self.raw_data_id = kwargs["raw_data_id"]

        fcol = self.colmap["filter"]  # shorthand

        # replace the filter name with a more specific one
        if "filtmap" in kwargs:
            if isinstance(kwargs["filtmap"], dict):

                def filter_mapping(filt):
                    return kwargs["filtmap"].get(filt, filt)

            elif isinstance(kwargs["filtmap"], str):

                def filter_mapping(filt):
                    new_filt = kwargs["filtmap"]
                    if self.observatory:
                        new_filt = new_filt.replace("<observatory>", self.observatory)
                    return new_filt.replace("<filter>", filt)

            self.data[fcol] = self.data[fcol].map(filter_mapping)

        filters = self.data[fcol]
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

        # get the signal to noise ratio
        self.calc_snr()

        # get the peak flux and S/N
        self.calc_best()

        # remove columns we don't use
        self.drop_columns()

    def __repr__(self):
        string = (
            f"{self.__class__.__name__}("
            f"source={self.source_id}, "
            f"epochs={self.number}"
        )
        if self.observatory:
            string += f" ({self.observatory.upper()})"
        string += f", mag[{self.filter}]={self.mag_mean_robust:.2f}\u00B1{self.mag_rms_robust:.2f})"
        string += f", file: {self.filename}"

        if self.key:
            string += f" (key: {self.key})"

        string += ")"

        return string

    # maybe this should happen at the base class?
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

    raw_data_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("raw_data.id"),
        nullable=True,
        index=True,
        doc="ID of the raw dataset that was used to produce this reduced dataset.",
    )

    raw_data = orm.relationship(
        "RawData",
        backref="lightcurves",
        cascade="all",
        doc="The raw dataset that was used to produce this reduced dataset.",
    )

    raw_data_filename = sa.Column(
        sa.String,
        nullable=True,
        index=True,
        doc="Filename of the raw dataset that was used to produce this reduced dataset.",
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

    detections_in_time = orm.relationship(
        "DetectionInTime",
        back_populates="lightcurve",
        cascade="all, delete-orphan",
    )


# make sure all the tables exist
RawData.metadata.create_all(engine)
Lightcurve.metadata.create_all(engine)


@event.listens_for(RawData, "before_insert")
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
            "Save the dataset to disk to generate a uuid4 filename. "
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


if __name__ == "__main__":
    import matplotlib
    from src.source import Source

    matplotlib.use("qt5agg")
    with Session() as session:

        sources = session.scalars(sa.select(Source).where(Source.project == "WD")).all()
        ax = sources[0].plot_photometry(ttype="mjd", ftype="flux")
