import sqlalchemy as sa

from src.database import Base


class Detection(Base):

    __tablename__ = "detections"

    id = sa.Column(
        sa.Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
        doc="Unique identifier for this detection",
    )
    source_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("sources.id"),
        nullable=False,
        index=True,
        doc="ID of the source this detection is associated with",
    )
    dataset_id = sa.Column(
        sa.Integer,
        sa.ForeignKey("datasets.id"),
        nullable=True,
        index=True,
        doc="ID of the dataset this detection is associated with",
    )

    simulated = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        index=True,
        doc="Is this detection simulated (True) or real (False)",
    )

    score = sa.Column(
        sa.Float,
        nullable=True,
        doc="Statistical score of the detection (in units of signal to noise)",
    )
    time = sa.Column(sa.DateTime, nullable=True, doc="Time of the detection ")
    period = sa.Column(
        sa.Float,
        nullable=True,
        doc="Period of the detection (only relevant for periodic detections)",
    )
