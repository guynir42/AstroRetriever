import os
import numpy as np
import pandas as pd
import xarray as xr
import sqlalchemy as sa
from sqlalchemy.orm import relationship

from database import Base

DATA_ROOT = os.getenv("VO_DATA") or ""


class Dataset(Base):

    __tablename__ = "datasets"

    id = sa.Column(
        sa.Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
        doc="Unique identifier for this dataset",
    )
    source_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("sources.id"),
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
    format = sa.Column(
        sa.String, nullable=False, default="hdf5", doc="Format of the dataset"
    )
    type = sa.Column(
        sa.String, nullable=False, default="photometry", doc="Type of the dataset"
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
    exp_time = sa.Column(
        sa.Float, nullable=False, doc="Median exposure time of each frame, in seconds."
    )
    frame_rate = sa.Column(
        sa.Float,
        nullable=False,
        doc="Median frame rate (frequency) of exposures in Hz.",
    )
    shape = sa.Column(sa.ARRAY(sa.Integer), nullable=True, doc="Shape of the dataset")
    number = sa.Column(
        sa.Integer, nullable=True, doc="Number of observations/epochs in the dataset"
    )

    # how this was observed
    observatory = sa.Column(
        sa.String,
        nullable=True,
        index=True,
        doc="Name of the observatory this dataset is associated with",
    )
    instrument = sa.Column(
        sa.String,
        nullable=True,
        index=True,
        doc="Name of the instrument this dataset is associated with",
    )
    filter = sa.Column(
        sa.String, nullable=True, index=True, doc="Filter used to measure the dataset"
    )

    public = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc="Whether this dataset is publicly available",
    )

    def load_data(self):
        pass

    def save_data(self):
        pass

    def delete_data(self):
        pass
