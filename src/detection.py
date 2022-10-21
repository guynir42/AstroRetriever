import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy import orm, func
from sqlalchemy.ext.declarative import declared_attr

from src.database import Base, engine
from src.source import Source
from src.dataset import RawPhotometry, Lightcurve

utcnow = func.timezone("UTC", func.current_timestamp())


class Detection(Base):

    __tablename__ = "detections"

    # TODO: can we just define separate detection types?

    method = sa.Column(
        sa.String,
        nullable=False,
        default="peak finding",
        doc="detection method used to find this. ",
    )

    data_types = sa.Column(
        ARRAY(sa.String),
        nullable=False,
        default=["photometry"],
        doc="The data types the detection is based on.",
    )

    source_id = sa.Column(
        sa.ForeignKey("sources.id"),
        index=True,
        doc="ID of the source this detection is associated with",
    )

    source = orm.relationship(
        "Source",
        back_populates="detections",
        cascade="all",
        foreign_keys="Detection.source_id",
    )

    # prefer to have this as column so we can still
    # attach a detection that was removed from a source
    # source_name = association_proxy("source", "name")
    source_name = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc="Name of the source this detection is associated with",
    )

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
        index=True,
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

    failed_flag = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        index=True,
        doc="Will be true if any of the quality cuts failed to pass. ",
    )

    quality_values = sa.Column(
        JSONB,
        default={},
        doc="Quality cuts values associated with this detection",
    )

    def __setattr__(self, key, value):
        if key == "source":
            self.source_name = value.name
        super().__setattr__(key, value)

    # detections made on lightcurves / photometric data:
    peak_time = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc="Time of the detection (UTC)",
    )

    peak_start = sa.Column(
        sa.DateTime,
        nullable=True,
        index=False,
        doc="Start of the peak (UTC)",
    )

    peak_end = sa.Column(
        sa.DateTime,
        nullable=True,
        index=False,
        doc="End of the peak (UTC)",
    )

    peak_mag = sa.Column(
        sa.Float, nullable=True, index=True, doc="Magnitude of the detection."
    )

    peak_dmag = sa.Column(
        sa.Float,
        nullable=True,
        index=True,
        doc="Magnitude difference of the detection.",
    )

    raw_photometry_peak_number = sa.Column(
        sa.Integer,
        nullable=True,
        index=False,
        doc="Index of the raw photometry where the brightest peak was detected. ",
    )

    raw_photometry_data_ranges = sa.Column(
        JSONB,
        nullable=True,
        index=False,
        doc="For each raw photometry data that is relevant to this detection"
        "put the data indices that are part of the detection. "
        "This is a dict keyed on integers (raw photometry index in the"
        "list attached to this object), with values that are a list of "
        "integers on the dataframe for each raw photometry. ",
    )

    reduced_photometry_peak_number = sa.Column(
        sa.Integer,
        nullable=True,
        index=False,
        doc="Index of the reduced photometry where the brightest peak was detected. ",
    )

    reduced_photometry_data_ranges = sa.Column(
        JSONB,
        nullable=True,
        index=False,
        doc="For each raw photometry data that is relevant to this detection"
        "put the data indices that are part of the detection. "
        "This is a dict keyed on integers (raw photometry index in the"
        "list attached to this object), with values that are a list of "
        "integers on the dataframe for each raw photometry. ",
    )

    # using a matched filter could use saving some parameters:
    matched_filter_kernel_index = sa.Column(
        sa.Integer,
        nullable=True,
        index=False,
        doc="Index of the matched filter kernel that was used to detect this. ",
    )
    match_filter_bank_name = sa.Column(
        sa.String,
        nullable=True,
        index=False,
        doc="Name of the matched filter bank that was used to detect this. ",
    )
    matched_filter_kernel_props = sa.Column(
        JSONB,
        nullable=True,
        index=False,
        doc="Properties of the matched filter kernel that was used to detect this. ",
    )

    # TODO: maybe a unique constraint with simulated too?
    #  what about non-peak-time detections?
    # __table_args__ = (
    #     sa.UniqueConstraint("time_peak", "source_id", name="_detection_in_time_uc"),
    # )


# add relationships!
Source.detections = orm.relationship(
    "Detection",
    back_populates="source",
    cascade="save-update, merge, refresh-expire, expunge, delete, delete-orphan",
    lazy="selectin",
    single_parent=True,
    passive_deletes=True,
    doc="Detections associated with this source",
)

# relationships to photometric datasets
detection_raw_photometry_association = sa.Table(
    "detection_raw_photometry_association",
    Base.metadata,
    sa.Column(
        "detection_id",
        sa.Integer,
        sa.ForeignKey("detections.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column(
        "raw_photometry_id",
        sa.Integer,
        sa.ForeignKey("raw_photometry.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)

Detection.raw_photometry = orm.relationship(
    "RawPhotometry",
    cascade="all",
    secondary=detection_raw_photometry_association,
    doc="raw photometric data in which this detection was found",
)

detection_reduced_photometry_association = sa.Table(
    "detection_reduced_photometry_association",
    Base.metadata,
    sa.Column(
        "detection_id",
        sa.Integer,
        sa.ForeignKey("detections.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column(
        "reduced_photometry_id",
        sa.Integer,
        sa.ForeignKey("lightcurves.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)

Detection.reduced_photometry = orm.relationship(
    "Lightcurve",
    cascade="all",
    secondary=detection_reduced_photometry_association,
    doc="reduced or processed or simulated "
    "photometric data in which this detection was found",
)


# make sure all the tables exist
Detection.metadata.create_all(engine)


# TODO: add relationships to spectrum datasets
# TODO: add relationships to periodograms
# TODO: add relationships to image datasets
