import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.associationproxy import association_proxy

from src.database import Base, engine
from src.source import Source
from sqlalchemy import orm, func

utcnow = func.timezone("UTC", func.current_timestamp())


class Properties(Base):

    __tablename__ = "properties"

    source_id = sa.Column(
        sa.ForeignKey("sources.id"),
        nullable=False,
        index=True,
        doc="ID of the source these properties are associated with",
    )

    source = orm.relationship(
        "Source",
        back_populates="_properties_from_db",
        cascade="all",
        foreign_keys="Properties.source_id",
    )

    source_name = association_proxy("source", "name")

    project = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc="Project these properties are associated with",
    )

    cfg_hash = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        default="",
        doc="Hash of the configuration used to generate this object."
        "(leave empty if not using version control)",
    )

    props = sa.Column(
        JSONB,
        nullable=False,
        default={},
        doc="Properties of the source",
    )

    has_data = sa.Column(
        sa.Boolean,
        nullable=False,
        index=True,
        default=True,
        doc="A source with empty datasets would have no "
        "usable properties but still needs to have a "
        "Properties object so we don't re-analyze it.",
    )


Source._properties_from_db = orm.relationship(
    "Properties",
    back_populates="source",
    cascade="save-update, merge, refresh-expire, expunge, delete, delete-orphan",
    lazy="selectin",
    single_parent=True,
    uselist=False,
    passive_deletes=True,
    doc="Properties associated with this source",
)


Properties.metadata.create_all(engine)
