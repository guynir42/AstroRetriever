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

from src.database import Base, Session, CloseSession, engine

from src.parameters import (
    allowed_data_types,
    convert_data_type,
    get_class_from_data_type,
)
from src.utils import ra2sex, dec2sex

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

        self.keywords_to_columns(kwargs)

        # extra info from the cat_row that is not saved to the Source:
        additional_keywords = ["pmra", "pmdec"]
        if any([k not in additional_keywords for k in kwargs.keys()]):
            raise ValueError(f"Unknown keyword arguments: {kwargs}")

    def __repr__(self):
        mag = f"{self.mag:.2f}" if self.mag is not None else "None"
        string = f'Source(name="{self.name}"'
        if self.ra is not None:
            string += f", ra={ra2sex(self.ra)}"
        if self.dec is not None:
            string += f", dec={dec2sex(self.dec)}"
        string += f', mag= {mag}, project="{self.project}"'
        string += f", datasets= {len(self.raw_photometry)})"  # TODO: what about other kinds of data?

        return string

    def get_raw_data(self, obs, data_type=None, session=None):
        """
        Get the raw data object associated with this source,
        the observatory named by "obs" and the data type
        requested by "data_type".
        If given a session object, will also look for the
        raw data in the database even if it wasn't associated
        with this source.

        Parameters
        ----------
        obs: str
            Name of the observatory that produced
            the requested raw data.
        data_type: str (optional)
            Type of data, could be "photometry", "spectroscopy",
            "images", etc.
            If not given, will try to infer the data type
            by looking at the different types of data
            associated with this source.
            If there is more than one type of data,
            will raise an error.
        session: sqlalchemy session object (optional)
            If given, will also search the DB for matching
            raw data objects, if none are found attached
            to the source.

        Returns
        -------
        raw_data: a RawPhotometry or similar object
            The object containing the data that matches
            this source and the required observatory
            and data type.
            The object does not necessarily have data
            loaded and does not necessarily have that
            data saved on disk (these should be tested
            separately).
            If no matching data is found, returns None.
        """
        if data_type is not None:  # data type is given explicitly
            data_type = convert_data_type(data_type)
        else:
            # for each data type, check if there is that sort of data
            # on this source, if it is not an empty list, add it
            types = {t for t in allowed_data_types if getattr(self, f"raw_{t}", [])}
            if len(types) == 0:
                raise ValueError("No data types found for this source.")
            elif len(types) > 1:
                raise ValueError("More than one data type found for this source.")
            else:
                data_type = list(types)[0]

        data_class = get_class_from_data_type(data_type)
        # if source existed in DB it should have raw data objects
        # if it doesn't that means the data needs to be downloaded
        raw_data = [  # e.g., go over all raw_photometry...
            data
            for data in getattr(self, f"raw_{data_type}")
            if data.observatory == obs
        ]
        if len(raw_data) == 0:
            raw_data = None
        elif len(raw_data) == 1:
            raw_data = raw_data[0]
        else:
            raise RuntimeError(
                f"Source {self.name} has more than one "
                f"{data_class.__name__} object from this observatory."
            )

        # try to recover the raw data from the DB directly
        # this may find data that was associated with the
        # same source but in a different project or cfg_hash (version)
        if raw_data is None and session is not None:
            raw_data = session.scalars(
                sa.select(data_class).where(
                    data_class.source_name == self.name,
                    data_class.observatory == obs,
                )
            ).first()

        return raw_data

    def remove_raw_data(self, obs, data_type=None, session=None):
        """
        Remove from this source the raw data object associated
        with the observatory named by "obs" and the data type
        requested by "data_type".
        If given a session object, will also delete the raw data
        in the database, if it finds it.
        Will return a boolean notifying if the data was
        found and deleted.

        Parameters
        ----------
        obs: str
            Name of the observatory that produced
            the requested raw data.
        data_type: str (optional)
            Type of data, could be "photometry", "spectroscopy",
            "images", etc.
            If not given, will try to infer the data type
            by looking at the different types of data
            associated with this source.
            If there is more than one type of data,
            will raise an error.
        session: sqlalchemy session object (optional)
            If given, will also search the DB for a matching
            raw data object, and delete it if found.

        Returns
        -------
        deleted: bool
            True if the data was found and deleted, False otherwise.
        """
        if data_type is not None:  # data type is given explicitly
            data_type = convert_data_type(data_type)
        else:
            # for each data type, check if there is that sort of data
            # on this source, if it is not an empty list, add it
            types = {t for t in allowed_data_types if getattr(self, f"raw_{t}", [])}
            if len(types) == 0:
                raise ValueError("No data types found for this source.")
            elif len(types) > 1:
                raise ValueError("More than one data type found for this source.")
            else:
                data_type = list(types)[0]

        data_class = get_class_from_data_type(data_type)
        # if source existed in DB it should have raw data objects
        # if it doesn't that means the data needs to be downloaded
        raw_data = [  # e.g., go over all raw_photometry...
            data
            for data in getattr(self, f"raw_{data_type}")
            if data.observatory == obs
        ]
        if len(raw_data) == 0:
            return False
        elif len(raw_data) == 1:
            raw_data = raw_data[0]
            getattr(self, f"raw_{data_type}").remove(raw_data)
            if session is not None:
                session.delete(raw_data)
            return True
        else:
            raise RuntimeError(
                f"Source {self.name} has more than one "
                f"{data_class.__name__} object from this observatory."
            )

    def reset_analysis(self, session=None):
        """
        Remove all analysis results from this object,
        including detections, properties, and processed
        lightcurves.

        This does not affect the histogram results!

        Parameters
        ----------
        session: sqlalchemy session object (optional)
            If given, will also remove the analysis results
            from the database.
        """
        self.properties = None
        self.detections = []
        for dt in allowed_data_types:
            setattr(self, f"processed_{dt}", [])
            setattr(self, f"simulated_{dt}", [])

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

    def remove_from_database_and_disk(self, session=None, remove_raw_data=False):
        """
        Remove this source from the database and from disk.
        """
        if session is None:
            session = Session()
            # make sure this session gets closed at end of function
            _ = CloseSession(session)
        if remove_raw_data:
            for rp in self.raw_photometry:
                # maybe add an autodelete for RawPhotometry?
                rp.remove_from_disk()
                session.delete(rp)
        for lc in self.lightcurves:
            lc.remove_from_disk()
            session.delete(lc)
        session.delete(self)
        session.commit()


# make sure the table exists
Source.metadata.create_all(engine)


if __name__ == "__main__":
    import src.dataset

    src.dataset.DATA_ROOT = "/home/guyn/data"

    session = Session()
    sources = session.scalars(sa.select(Source).where(Source.project == "WD")).all()

    sources[0].plot_photometry()
