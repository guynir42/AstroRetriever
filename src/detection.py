import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy import orm, func
from sqlalchemy.ext.declarative import declared_attr

from src.database import Base, engine
from src.source import Source
from src.dataset import RawPhotometry, Lightcurve

utcnow = func.timezone("UTC", func.current_timestamp())


class DetectionMixin:

    project = sa.Column(
        sa.String, nullable=False, index=True, doc="Project this detection belongs to."
    )

    cfg_hash = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        default="",
        doc="Hash of the configuration used to generate this object."
        "(leave empty if not using version control)",
    )

    time_start = sa.Column(
        sa.DateTime,
        nullable=True,
        doc="Beginning of time interval relevant to this event (UTC)",
    )

    time_end = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc="End of time interval relevant to this event (UTC)",
    )

    @classmethod
    def backref_name(cls):
        if cls.__name__ == "DetectionInTime":
            return "detections_in_time"
        if cls.__name__ == "DetectionInPeriod":
            return "detections_in_period"
        if cls.__name__ == "DetectionInImage":
            return "detections_in_images"
        if cls.__name__ == "DetectionInSpectrum":
            return "detections_in_spectra"

    @declared_attr
    def source_id(cls):
        return sa.Column(
            sa.ForeignKey("sources.id"),
            nullable=False,
            index=True,
            doc="ID of the source this detection is associated with",
        )

    @declared_attr
    def source(cls):
        return orm.relationship(
            "Source",
            back_populates=cls.backref_name(),
            cascade="all",
            foreign_keys=f"{cls.__name__}.source_id",
        )

    source_name = association_proxy("source", "name")

    simulated = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        index=True,
        doc="Is this detection simulated (True) or real (False)",
    )

    sim_pars = sa.Column(
        JSONB,
        nullable=True,
        doc="Parameters used to simulate this detection",
    )

    snr = sa.Column(
        sa.Float,
        nullable=True,
        index=True,
        doc="Signal-to-noise ratio of this detection",
    )

    score = sa.Column(
        sa.Float,
        nullable=True,
        index=True,
        doc="Statistical score of the detection",
    )

    extra_scores = sa.Column(
        JSONB,
        default={},
        doc="Extra scores associated with this detection",
        index=True,
    )

    quality_cuts = sa.Column(
        JSONB,
        default={},
        doc="Quality cuts values associated with this detection",
    )


class DetectionInTime(Base, DetectionMixin):

    __tablename__ = "detections_in_time"

    raw_data_id = sa.Column(
        sa.ForeignKey("raw_photometry.id"),
        nullable=True,
        index=True,
        doc="ID of the raw dataset this detection is associated with",
    )

    raw_data = orm.relationship(
        "RawPhotometry",
        back_populates="detections",
        cascade="all",
        foreign_keys=f"DetectionInTime.raw_data_id",
    )

    time_peak = sa.Column(
        sa.DateTime,
        nullable=False,
        index=True,
        doc="Time of the detection (UTC)",
    )

    lightcurve_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("lightcurves.id"),
        nullable=True,
        index=True,
        doc="ID of the lightcurve this detection is associated with",
    )
    lightcurve = orm.relationship(
        "Lightcurve",
        back_populates="detections_in_time",
        cascade="all",
        doc="reduced photometric data in which this detection was found",
    )

    __table_args__ = (
        sa.UniqueConstraint("time_peak", "source_id", name="_detection_in_time_uc"),
    )


# make sure all the tables exist
DetectionInTime.metadata.create_all(engine)

Source.detections_in_time = orm.relationship(
    "DetectionInTime",
    back_populates="source",
    cascade="save-update, merge, refresh-expire, expunge, delete, delete-orphan",
    lazy="selectin",
    single_parent=True,
    passive_deletes=True,
    doc="Detections associated with lightcurves from this source",
)


RawPhotometry.detections = orm.relationship(
    "DetectionInTime",
    back_populates="raw_data",
    cascade="all, delete-orphan",
)


Lightcurve.detections_in_time = orm.relationship(
    "DetectionInTime",
    back_populates="lightcurve",
    cascade="all, delete-orphan",
)


# TODO: Add a DetectionInPeriod class
# TODO: Add a DetectionInImages class
# TODO: Add a DetectionInSpectrum class
