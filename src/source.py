import sqlalchemy as sa
from sqlalchemy.orm import relationship

import conesearch_alchemy
import healpix_alchemy as ha
from astropy import units as u

from database import Base
from src.dataset import Dataset
from src.detection import Detection


class Source(Base, conesearch_alchemy.Point):

    __tablename__ = "sources"

    id = sa.Column(
        sa.Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
        doc="Unique identifier for this source",
    )
    name = sa.Column(
        sa.String, nullable=False, index=True, unique=True, doc="Name of the source"
    )

    # ra and dec are included from the Point class

    classification = sa.Column(
        sa.String, nullable=True, doc="Classification of the source"
    )

    alias = sa.Column(
        sa.ARRAY(sa.String),
        nullable=False,
        default=[],
        doc="A list of additional names for this source",
    )

    mag = sa.Column(sa.Float, nullable=True, index=True, doc="Magnitude of the source")
    mag_err = sa.Column(
        sa.Float, nullable=True, doc="Error in the magnitude of the source"
    )
    mag_filter = sa.Column(
        sa.String,
        nullable=True,
        doc="Filter used to measure the magnitude of the source",
    )

    datasets = relationship(
        "Dataset",
        backref="sources",
        cascade="save-update, merge, refresh-expire, expunge, delete",
        lazy="selectin",
        single_parent=True,
        passive_deletes=True,
        doc="Datasets associated with this source",
    )
    detections = relationship(
        "Detection",
        backref="sources",
        cascade="save-update, merge, refresh-expire, expunge, delete",
        lazy="selectin",
        single_parent=True,
        passive_deletes=True,
        doc="Detections associated with this source",
    )

    def __init__(
        self, name, ra, dec, alias=None, mag=None, mag_err=None, mag_filter=None
    ):

        self.name = name
        self.ra = ra
        self.dec = dec

        if alias is not None:
            self.alias = alias

        if mag is not None:
            self.mag = mag

        if mag_err is not None:
            self.mag_err = mag_err

        if mag_filter is not None:
            self.mag_filter = mag_filter

        # assign this coordinate a healpix ID
        self.healpix = ha.constants.HPX.lonlat_to_healpix(ra * u.deg, dec * u.deg)
