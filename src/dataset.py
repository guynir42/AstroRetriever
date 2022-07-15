import os
import uuid

import numpy as np
import pandas as pd
import xarray as xr

from astropy.time import Time
import h5py

import sqlalchemy as sa
from sqlalchemy import func, event
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.dialects.postgresql import JSONB

from src.database import Base, Session, engine

# root folder is either defined via an environment variable
# or is the in the main repository, under subfolder "data"
DATA_ROOT = os.getenv("VO_DATA")
if DATA_ROOT is None:
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

utcnow = func.timezone("UTC", func.current_timestamp())


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
        a PhotometricData using a reducer function
        from the correct observatory object.

        """
        if "data" in kwargs:
            self.data = kwargs["data"]
            del kwargs["data"]

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # verify some inputs are the right type
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

    def calc_size(self):
        """
        Calculate the size of the data file.
        """
        # TODO: implement this
        return 0

    def get_path(self, full=False):
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

        f = os.path.join(DATA_ROOT, f)

        if full:
            append_folder = os.path.dirname(self.filename)
            if append_folder:
                f = os.path.join(f, append_folder)

        return f

    def get_fullname(self):
        """
        Get the full path to the data file.
        """

        return os.path.join(self.get_path(), self.filename)

    def check_file_exists(self):
        """
        Check if the file exists on disk.
        """
        return os.path.exists(self.get_fullname())

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
        pass

    def load_json(self):
        pass

    def load_netcdf(self):
        pass

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
            self.filename = str(uuid.uuid4()) + self.guess_extension()

        if self.key is None and self.format in ("hdf5"):
            self.key = str(uuid.uuid4())

        if overwrite is None:
            overwrite = self.overwrite
        if not overwrite and self.check_file_exists():
            raise ValueError(f"File {self.get_fullname()} already exists")

        # make a path if missing
        if not os.path.isdir(self.get_path(full=True)):
            os.makedirs(self.get_path(full=True))

        # specific format save functions
        if self.format == "hdf5":
            self.save_hdf5()
        elif self.format == "fits":
            self.save_fits()
        elif self.format == "csv":
            self.save_csv()
        elif self.format == "json":
            self.save_json()
        elif self.format == "netcdf":
            self.save_netcdf()
        else:
            raise ValueError(f"Unknown format {self.format}")

    def save_hdf5(self):
        if isinstance(self.data, xr.Dataset):
            self.data.to_hdf(self.get_fullname(), key=self.key)  # this actually works??
        elif isinstance(self.data, pd.DataFrame):
            with pd.HDFStore(self.get_fullname(), "w") as store:
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

    def save_fits(self):
        pass

    def save_csv(self):
        pass

    def save_json(self):
        pass

    def save_netcdf(self):
        pass

    def delete_data_from_disk(self):
        """
        Delete the data file from disk.
        """
        os.remove(self.get_fullname())

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

        if isinstance(data, pd.DataFrame):
            columns = data.columns
        # other datatypes will call this differently...
        # TODO: get the columns for other data types

        t = None  # datetimes for each epoch
        for c in columns:
            if c.lower() in ("jd", "jds", "juliandate", "juliandates"):
                t = data[c].values
                t = Time(t, format="jd", scale="utc").datetime
                break
            elif c.lower() in ("mjd", "mjds"):
                t = data[c].values
                t = Time(t, format="mjd", scale="utc").datetime
                break
            elif c.lower() in ("time", "times", "datetime", "datetimes"):
                t = data[c].values
                if isinstance(t[0], (str, bytes)):
                    if "T" in t[0]:
                        t = Time(t, format="isot", scale="utc").datetime
                    else:
                        t = Time(t, format="iso", scale="utc").datetime
                break
            elif c.lower() == "timestamps":
                t = data[c].values
                t = Time(t, format="unix", scale="utc").datetime
                break

        if t is not None:
            self.time_start = min(t)
            self.time_end = max(t)

    id = sa.Column(
        sa.Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
        doc="Unique identifier for this dataset",
    )

    created_at = sa.Column(
        sa.DateTime,
        nullable=False,
        default=utcnow,
        index=True,
        doc="UTC time of insertion of object's row into the database.",
    )

    modified = sa.Column(
        sa.DateTime,
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
        doc="UTC time the object's row was last modified in the database.",
    )

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
    filename = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc="Filename of the dataset, including path, relative to the DATA_ROOT",
    )

    folder = sa.Column(
        sa.String,
        nullable=True,
        doc="Folder inside the DATA_ROOT folder where this dataset is stored",
    )

    key = sa.Column(
        sa.String,
        nullable=True,
        doc="Key of the dataset (e.g., in the HDF5 " "file it would be the group name)",
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

    # how this was observed
    observatory = sa.Column(
        sa.String,
        nullable=True,
        index=True,
        doc="Name of the observatory this dataset is associated with",
    )

    public = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc="Whether this dataset is publicly available",
    )

    autosave = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc="Whether this dataset should be automatically saved",
    )

    autoload = sa.Column(
        sa.Boolean,
        nullable=False,
        default=True,
        doc="Whether this dataset should be automatically loaded (lazy loaded)",
    )

    overwrite = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc="Whether the data on disk should be overwritten if it already exists "
        "(if False, an error will be raised)",
    )


class RawData(DatasetMixin, Base):
    def __init__(self, **kwargs):
        """
        This class is used to store raw data from a survey,
        that should probably need to be reduced into more
        manageable forms (saved using other subclasses).

        Parameters are the same as in as the __init__ of the Dataset class.

        """
        DatasetMixin.__init__(self, **kwargs)
        Base.__init__(self)

    __tablename__ = "raw_data"


class PhotometricData(DatasetMixin, Base):
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
        DatasetMixin.__init__(self, **kwargs)
        Base.__init__(self)
        self.type = "photometry"

        # TODO: figure out frame rate, exp time, uniformity, from data
        # TODO: get more statistics on the lightcurve (summary data)

    __tablename__ = "photometric_data"

    raw_data_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("raw_data.id"),
        nullable=False,
        index=True,
        doc="ID of the raw dataset that was used to produce this reduced dataset.",
    )

    mean_mag = sa.Column(
        sa.Float,
        nullable=False,
        index=True,
        doc="Mean magnitude of the dataset",
    )
    mag_rms = sa.Column(
        sa.Float,
        nullable=False,
        index=True,
        doc="Mean magnitude error of the dataset",
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
        nullable=False,
        doc="Median frame rate (frequency) of exposures in Hz.",
    )
    is_uniformly_sampled = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc="Is the dataset sampled uniformly in time?",
    )


# make sure the table exists
RawData.metadata.create_all(engine)
# make sure the table exists
PhotometricData.metadata.create_all(engine)


@event.listens_for(RawData, "before_insert")
# @event.listens_for(PhotometricData, 'before_insert')
def insert_new_dataset(mapper, connection, target):
    """
    This function is called before a new dataset is inserted into the database.
    It checks that a file is associated with this object
    and if it doesn't exist, it creates it (if autoload is True)
    otherwise it raises an error.
    """
    if target.filename is None:
        raise ValueError("No filename specified for this dataset")

    if not target.check_file_exists():
        if target.autosave:
            target.save()
        else:
            raise ValueError(
                f"File {target.get_fullname()}"
                "does not exist and autosave is disabled. "
                "Please create the file manually."
            )
