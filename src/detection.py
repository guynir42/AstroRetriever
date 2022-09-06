import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from src.database import Base, engine
from sqlalchemy import orm, func
from sqlalchemy.ext.declarative import declared_attr

utcnow = func.timezone("UTC", func.current_timestamp())


class DetectionMixin:

    id = sa.Column(
        sa.Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
        doc="Unique identifier for this detection",
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

    time_start = sa.Column(
        sa.Float,
        nullable=True,
        doc="Beginning of time interval relevant to this event (UTC)",
    )

    time_end = sa.Column(
        sa.Float,
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

    @declared_attr
    def raw_data_id(cls):
        return sa.Column(
            sa.ForeignKey("raw_data.id"),
            nullable=True,
            index=True,
            doc="ID of the raw dataset this detection is associated with",
        )

    @declared_attr
    def raw_data(cls):
        return orm.relationship(
            "RawData",
            back_populates=cls.backref_name(),
            cascade="all",
            foreign_keys=f"{cls.__name__}.raw_data_id",
        )

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

    time_peak = sa.Column(
        sa.Float,
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

# TODO: Add a DetectionInPeriod class
# TODO: Add a DetectionInImages class
# TODO: Add a DetectionInSpectrum class
