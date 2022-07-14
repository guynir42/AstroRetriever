import warnings
import sqlalchemy as sa
from sqlalchemy import func
from sqlalchemy.orm import relationship
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

import conesearch_alchemy
import healpix_alchemy as ha
from astropy import units as u

from src.database import Base, Session
from src.dataset import Dataset
from src.detection import Detection

# ref: https://github.com/skyportal/conesearch-alchemy

DEFAULT_PROJECT = "test_project"

# get rid of annoying cosd/sind warnings regarding conesearch_alchemy:
# ref: https://github.com/tiangolo/sqlmodel/issues/189#issuecomment-1018014753
warnings.filterwarnings(
    "ignore", ".*Class .* will not make use of SQL compilation caching.*"
)


utcnow = func.timezone("UTC", func.current_timestamp())


def get_source_identifiers(project_name, column="id"):
    """
    Get all source identifiers from a given project.

    Parameters
    ----------
    project_name: str
        Name of the project.
    column: str
        Name of the column to get identifiers from.
        Default is "id".

    Returns
    -------
    list
        Set of identifiers.
    """

    with Session() as session:
        stmt = sa.select([getattr(Source, column)])
        stmt = stmt.where(Source.project == project_name)
        source_ids = session.execute(stmt).all()

        return {s[0] for s in source_ids}


def cone_search(ra, dec, sep=2 / 3600):
    """
    Find all sources at a radius "sep" from a given position.
    Returns a select statement to be executed with a session:
    >> session.scalars(cone_search(...)).all()


    Parameters
    ----------
    ra: float
        Center of cone's right ascension (in degrees).
    dec: float
        Center of cone's declination (in degrees).
    sep: float, optional
        Radius of cone (in degrees).

    Returns
    -------
    sources: select statement
        A select statement that matches sources within the cone.
        Can be further filtered or executed using a session.
    """
    p = conesearch_alchemy.Point(ra=ra, dec=dec)
    stmt = sa.select(Source).where(Source.within(p, sep))
    return stmt


class Source(Base, conesearch_alchemy.Point):

    __tablename__ = "sources"

    id = sa.Column(
        sa.Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
        doc="Unique identifier for this source",
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

    name = sa.Column(sa.String, nullable=False, index=True, doc="Name of the source")

    # ra and dec are included from the Point class

    project = sa.Column(
        sa.String,
        nullable=False,
        default=DEFAULT_PROJECT,
        index=True,
        doc="Project name to which this source is associated with",
    )

    origin = sa.Column(
        sa.String,
        nullable=True,
        index=True,
        doc="Where this source came from in a general sense",
    )

    classification = sa.Column(
        sa.String, nullable=True, doc="Classification of the source"
    )

    alias = sa.Column(
        sa.ARRAY(sa.String),
        nullable=False,
        default=[],
        doc="A list of additional names for this source",
    )

    # magnitude of the source
    mag = sa.Column(sa.Float, nullable=True, index=True, doc="Magnitude of the source")
    mag_err = sa.Column(
        sa.Float, nullable=True, doc="Error in the magnitude of the source"
    )
    mag_filter = sa.Column(
        sa.String,
        nullable=True,
        doc="Filter used to measure the magnitude of the source",
    )

    # catalog related stuff
    cat_index = sa.Column(
        sa.Integer,
        nullable=True,
        index=True,
        doc="Index of the source in the catalog",
    )
    cat_id = sa.Column(
        sa.String, nullable=True, index=True, doc="ID of the source in the catalog"
    )
    cat_name = sa.Column(
        sa.String, nullable=True, doc="Name of the catalog to which this source belongs"
    )
    cat_row = sa.Column(
        JSONB, nullable=True, doc="Row from the catalog used to create this source"
    )

    raw_data = relationship(
        "RawData",
        backref="sources",
        cascade="save-update, merge, refresh-expire, expunge, delete",
        lazy="selectin",
        single_parent=True,
        passive_deletes=True,
        doc="Raw Datasets associated with this source",
    )

    lightcurves = relationship(
        "PhotometricData",
        backref="sources",
        cascade="save-update, merge, refresh-expire, expunge, delete",
        lazy="selectin",
        single_parent=True,
        passive_deletes=True,
        doc="Photometric Datasets associated with this source",
    )

    # detections = relationship(
    #     "Detection",
    #     backref="sources",
    #     cascade="save-update, merge, refresh-expire, expunge, delete",
    #     lazy="selectin",
    #     single_parent=True,
    #     passive_deletes=True,
    #     doc="Detections associated with this source",
    # )

    __table_args__ = (
        UniqueConstraint("name", "project", name="_source_name_in_project_uc"),
    )

    def __init__(
        self,
        name,
        ra,
        dec,
        project=None,
        alias=None,
        mag=None,
        mag_err=None,
        mag_filter=None,
    ):

        self.name = name
        self.ra = ra
        self.dec = dec

        if project is not None:
            self.project = project
        else:
            self.project = DEFAULT_PROJECT

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

    def check_duplicates(self, project=None, sep=2 / 3600, session=None):
        """
        Check if this source is a duplicate of another source,
        by using a cone search on other sources from the same project.

        Parameters
        ----------
        project: str
            Project name to search for duplicates in.
            If not given, will default to DEFAULT_PROJECT
        sep: float
            Separation in degrees to search for duplicates.
            Default is 2 arcseconds.
        session: sqlalchemy.orm.session.Session
            Session to use for the cone search.
            If not given, will use a new session.

        Returns
        -------
        boolean
            True if this source is a duplicate, False otherwise.
        """
        if project is None:
            project = DEFAULT_PROJECT

        stmt = cone_search(ra=self.ra, dec=self.dec, sep=sep)
        stmt = stmt.where(Source.project == project)

        if session is None:
            session = Session()

        sources = session.scalars(stmt).first()
        return sources is not None

    def __repr__(self):
        string = (
            f'Source(name="{self.name}", ra={self.ra}, dec={self.dec}, mag= {self.mag}, '
            f'project="{self.project}", datasets= {len(self.raw_data)})'
        )
        return string


if __name__ == "__main__":
    print(get_source_identifiers(DEFAULT_PROJECT, "id"))
