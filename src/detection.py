import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from src.database import Base
from sqlalchemy import orm, func

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

    source_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("sources.id"),
        nullable=False,
        index=True,
        doc="ID of the source this detection is associated with",
    )
    source = orm.relationship(
        "Source", back_populates="detections", cascade="all, delete-orphan"
    )

    dataset_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("raw_data.id"),
        nullable=True,
        index=True,
        doc="ID of the dataset this detection is associated with",
    )
    dataset = orm.relationship(
        "RawData", back_populates="detections", cascade="all, delete-orphan"
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

    time = sa.Column(
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
        "LightCurve", back_populates="detections", cascade="all, delete-orphan"
    )

    __table_args__ = (
        sa.UniqueConstraint("time", "source_id", name="_detection_in_time_uc"),
    )


# TODO: Add a DetectionInPeriod class
# TODO: Add a DetectionInImages class
# TODO: Add a DetectionInSpectrum class
