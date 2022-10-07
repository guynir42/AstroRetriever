import warnings
import sqlalchemy as sa

import matplotlib
import matplotlib.pyplot as plt

from sqlalchemy import func
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

# ref: https://github.com/skyportal/conesearch-alchemy
import conesearch_alchemy
import healpix_alchemy as ha
from astropy import units as u

from src.database import Base, Session, engine
from src.catalog import Catalog

# matplotlib.use("qt5agg")


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
    # TODO: add option to filter on cfg_hash too
    Parameters
    ----------
    project_name: str
        Name of the project.
    column: str
        Name of the column to get identifiers from.
        Default is "id". Also useful is "name".

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


def angle_diff(a1, a2):
    """
    Find the distance between two angles (in degrees).

    Ref: https://gamedev.stackexchange.com/a/4472
    """
    return 180 - abs(abs(a1 - a2) - 180)


class Source(Base, conesearch_alchemy.Point):

    __tablename__ = "sources"

    __table_args__ = (
        UniqueConstraint(
            "name", "project", "cfg_hash", name="_source_name_in_project_uc"
        ),
    )

    name = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc="Name of the source",
    )

    # ra and dec are included from the Point class

    project = sa.Column(
        sa.String,
        nullable=False,
        default=DEFAULT_PROJECT,
        index=True,
        doc="Project name to which this source is associated with",
    )

    cfg_hash = sa.Column(
        sa.String,
        nullable=False,
        default="",
        index=True,
        doc="Hash of the configuration used to create this source "
        "(leave empty if not using version control)",
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

    aliases = sa.Column(
        sa.ARRAY(sa.String),
        nullable=False,
        default=[],
        doc="A list of additional names for this source",
    )

    # magnitude of the source
    mag = sa.Column(
        sa.Float,
        nullable=True,
        index=True,
        doc="Magnitude of the source",
    )
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

    # NOTE: all the source relationships are defined
    # where the data is defined, e.g., dataset.py and detection.py

    def __init__(self, **kwargs):

        self.name = kwargs.pop("name", None)
        if self.name is None:
            raise ValueError("Source must have a name.")

        self.ra = kwargs.pop("ra", None)
        self.dec = kwargs.pop("dec", None)
        self.project = kwargs.pop("project", DEFAULT_PROJECT)
        self.mag = kwargs.pop("mag", None)
        self.mag_err = kwargs.pop("mag_err", None)
        self.mag_filter = kwargs.pop("mag_filter", None)
        self.alias = kwargs.pop("alias", None)
        self.cat_index = kwargs.pop("cat_index", None)
        self.cfg_hash = ""

        # assign this coordinate a healpix ID
        if self.ra is not None and self.dec is not None:
            self.healpix = ha.constants.HPX.lonlat_to_healpix(
                self.ra * u.deg, self.dec * u.deg
            )

        additional_keywords = []
        if any([k not in additional_keywords for k in kwargs.items()]):
            raise ValueError(f"Unknown keyword arguments: {kwargs}")

    def __repr__(self):
        mag = f"{self.mag:.2f}" if self.mag is not None else "None"
        string = (
            f'Source(name="{self.name}", '
            f"ra={Catalog.ra2sex(self.ra)}, "
            f"dec={Catalog.dec2sex(self.dec)}, "
            f'mag= {mag}, project="{self.project}", '
            f"datasets= {len(self.raw_photometry)})"  # TODO: what about other kinds of data?
        )
        return string

    def plot_photometry(self, ax=None, ftype="mag", ttype="times", **kwargs):
        """
        Plot this source on a given axis.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            Axis to plot on.
            If not given, will create a new axis.
        **kwargs:
            Keyword arguments to pass to `matplotlib.pyplot.plot`.
        """
        if ax is None:
            ax = plt.gca()

        for rp in self.raw_photometry:
            if rp.type == "photometry":
                rp.plot(ax=ax, ftype=ftype, ttype=ttype, use_phot_zp=True, **kwargs)
        for lc in self.lightcurves:
            lc.plot(ax=ax, ftype=ftype, ttype=ttype, **kwargs)

        return ax

    def check_duplicates(self, project=None, sep=2 / 3600, session=None):
        """
        Check if this source is a duplicate of another source,
        by using a cone search on other sources from the same project.
        # TODO: should also be able to limit to sources with the same cfg_hash

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


# make sure the table exists
Source.metadata.create_all(engine)


if __name__ == "__main__":
    import src.dataset

    src.dataset.DATA_ROOT = "/home/guyn/data"

    session = Session()
    sources = session.scalars(sa.select(Source).where(Source.project == "WD")).all()

    sources[0].plot_photometry()
