import os
import numpy as np
import pandas as pd
import xarray as xr
import h5py
import sqlalchemy as sa
from sqlalchemy import func
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.dialects.postgresql import JSONB

from src.database import Base

DATA_ROOT = os.getenv("VO_DATA") or ""

utcnow = func.timezone("UTC", func.current_timestamp())


class Dataset:
    def __init__(self, data, filename, key=None, observatory=None):
        """
        Produce a Dataset object,
        which maps a location on disk
        with raw data in memory.
        Each Dataset is associated with one source
        (via the source_id foreign key).
        If there are multiple datasets in a file,
        use the key to identify which one is associated
        with this Dataset.

        A Dataset should be calibrated to produce
        useful inputs for an analysis.


        Parameters
        ----------
        data: xr.Dataset or pd.Dataframe or np.array
            Raw data as it was loaded from file.
        filename: str
            Full path to the file on disk.
        key: str or int or None
            Key to identify this dataset in the file.
            Leave None if there's only one Dataset per file.
        observatory: str
            Name of the observatory from which
            this dataset was taken.
        """
        # self.source_id = source_id
        if not isinstance(data, (np.ndarray, pd.DataFrame, xr.Dataset)):
            raise ValueError(
                "Data must be a numpy array, "
                "pandas DataFrame, or xarray Dataset, "
                f"not {type(data)}"
            )
        self.data = data
        if not isinstance(filename, str):
            raise ValueError(f"Filename must be a string, not {type(filename)}")
        self.filename = filename
        if not isinstance(key, (str, int, type(None))):
            raise ValueError(f"Key must be a string, int, or None, not {type(key)}")
        # optional parameters are None by default
        self.key = key
        self.observatory = observatory

        self.shape = data.shape
        self.number = len(data)  # for imaging data this would be different?
        self.size = self.calc_size()
        self.format = self.guess_format()

        # TODO: figure out start/end date from data
        # TODO: figure out the frame rate, exp time, uniformity, from data
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

    def calc_size(self):
        """
        Calculate the size of the data file.
        """
        # TODO: implement this
        return 0

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

    def check_file_exists(self):
        """
        Check if the file exists on disk.
        """
        return os.path.exists(self.filename)

    def load(self):
        pass

    def save(self, overwrite=False):
        if not overwrite and self.check_file_exists():
            raise ValueError(f"File {self.filename} already exists")
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
            self.data.to_hdf(self.filename, key=self.key)  # this actually works??
        elif isinstance(self.data, pd.DataFrame):
            with pd.HDFStore(self.filename, "w") as store:
                store.put(self.key, self.data)
                if self.altdata:
                    for k, v in self.altdata.items():
                        setattr(store.get_storer(self.key).attrs, k, v)
        elif isinstance(self.data, np.ndarray):
            with h5py.File(self.filename, "w") as f:
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
        os.remove(self.filename)


class RawData(Dataset, Base):
    def __init__(self, *args, **kwargs):
        """
        This class is used to store raw data from a survey,
        that should probably need to be reduced into more
        manageable forms (saved using other subclasses).

        Parameters are the same as in as the __init__ of the Dataset class.

        """
        Dataset.__init__(self, *args, **kwargs)
        Base.__init__(self)

    __tablename__ = "raw_data"


class PhotometricData(Dataset, Base):
    def __init__(self, *args, **kwargs):
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
        Dataset.__init__(self, *args, **kwargs)
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
