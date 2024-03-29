import warnings
import sqlalchemy as sa
from sqlalchemy import inspect

import matplotlib.pyplot as plt

from sqlalchemy import func, orm
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

# ref: https://github.com/skyportal/conesearch-alchemy
import conesearch_alchemy
import healpix_alchemy as ha
from astropy import units as u

from src.database import engine, Base, SmartSession

from src.parameters import (
    allowed_data_types,
    convert_data_type,
    get_class_from_data_type,
)
from src.utils import ra2sex, dec2sex, add_alias, UniqueList, legalize


DEFAULT_PROJECT = "TEST_PROJECT"

# get rid of annoying cosd/sind warnings regarding conesearch_alchemy:
# ref: https://github.com/tiangolo/sqlmodel/issues/189#issuecomment-1018014753
warnings.filterwarnings(
    "ignore", ".*Class .* will not make use of SQL compilation caching.*"
)


utcnow = func.timezone("UTC", func.current_timestamp())


def get_source_identifiers(project_name, cfg_hash=None, column="id"):
    """
    Get all source identifiers from a given project.

    Parameters
    ----------
    project_name: str
        Name of the project.
    cfg_hash: str, optional
        Hash of the configuration file.
    column: str
        Name of the column to get identifiers from.
        Default is "id". Also useful is "name".

    Returns
    -------
    list
        Set of identifiers.
    """

    hash = cfg_hash if cfg_hash is not None else ""
    with SmartSession() as session:
        stmt = sa.select([getattr(Source, column)])
        stmt = stmt.where(
            Source.project == legalize(project_name), Source.cfg_hash == hash
        )
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

    if True:  # put all column definitions on one block
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

        local_names = sa.Column(
            JSONB,
            nullable=False,
            default={},
            doc="A dictionary of local names for this source, "
            "where the key is the observatory name (upper case)"
            "and the value is the name of the source in each observatories"
            "internal naming convention. ",
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
            sa.String,
            nullable=True,
            doc="Name of the catalog to which this source belongs",
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
        self.aliases = kwargs.pop("alias", [])
        self.local_names = kwargs.pop("local_names", {})
        self.cat_index = kwargs.pop("cat_index", None)
        self.cfg_hash = ""

        self.raw_photometry = []
        self.reduced_photometry = []
        self.processed_photometry = []
        self.simulated_photometry = []
        self.detections = None
        self.properties = None
        self.loaded_status = "new"

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

    @orm.reconstructor
    def _init_on_load(self):
        """
        This is called when the object
        is loaded from the database.
        ref: https://docs.sqlalchemy.org/en/14/orm/constructors.html
        """
        self.raw_photometry = []
        self.reduced_photometry = []
        self.processed_photometry = []
        self.simulated_photometry = []
        self.detections = None
        self.loaded_status = "database"

    def __setattr__(self, key, value):
        if key == "raw_photometry":
            if not isinstance(value, list):
                raise ValueError("raw_photometry must be a list")
            new_value = UniqueList(["observatory"], ignorecase=True)
            for item in value:
                item.source = self
                new_value.append(item)
            value = new_value
        if key in [
            "reduced_photometry",
            "processed_photometry",
            "simulated_photometry",
        ]:
            if not isinstance(value, list):
                raise ValueError(f"{key} must be a list")
            new_value = UniqueList(["observatory", "series_number"], ignorecase=True)
            for item in value:
                item.source = self
                new_value.append(item)
            value = new_value
        if key == "properties" and value is not None:
            value.source_name = self.name
            value.project = self.project
            value.cfg_hash = self.cfg_hash

        if key == "project" and value is not None:
            value = legalize(value)

        super().__setattr__(key, value)

    rp = add_alias("raw_photometry")

    reduced_lightcurves = add_alias("reduced_photometry")
    rl = add_alias("reduced_photometry")

    def save_reduced_photometry(self, session=None):
        """
        Save the reduced photometry to the database.
        """
        with SmartSession(session) as session:

            try:
                for lc in self.reduced_photometry:
                    lc.sanitize()
                    lc.save()
                    session.add(lc)
                session.commit()
            except Exception as e:
                session.rollback()
                raise e

    processed_lightcurves = add_alias("processed_photometry")
    pl = add_alias("processed_photometry")

    def save_processed_photometry(self, session=None):
        """
        Save the processed photometry to the database.
        """
        with SmartSession(session) as session:
            try:
                for lc in self.processed_photometry:
                    lc.sanitize()
                    lc.save(overwrite=True)
                    session.add(lc)
                session.commit()
            except Exception as e:
                session.rollback()
                raise e

    simulated_lightcurves = add_alias("simulated_photometry")
    sl = add_alias("simulated_photometry")

    def save_simulated_photometry(self, session=None):
        """
        Save the simulated photometry to the database.
        """
        with SmartSession(session) as session:
            try:
                for lc in self.simulated_photometry:
                    lc.sanitize()
                    lc.save(overwrite=True)
                    session.add(lc)
                session.commit()
            except Exception as e:
                session.rollback()
                raise e

    def save_detections(self, session=None):
        """
        Save the detections to the database.
        """
        with SmartSession(session) as session:
            if self.detections is not None:
                for det in self.detections:
                    det.sanitize()
                    session.add(det)

            session.commit()

    def save(self, session=None):
        """Save the source to the database"""
        with SmartSession(session) as session:
            self.sanitize()
            if self.properties is not None:
                self.properties.sanitize()
            session.add(self)

            session.commit()

    def get_data(
        self,
        obs,
        data_type=None,
        level="raw",
        session=None,
        check_data=True,
        search_orphans=True,
        delete_missing=True,
        append=True,
    ):
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
        level: str
            Processing level of the data: "raw", "reduced",
            "processed" or "simulated". Default is "raw".
        session: sqlalchemy session object (optional)
            Will open a new session if not given (or None),
            and close it at the end of the function.
            If given as a session object, will leave the session
            open for use in the external scope.
            To disable database interaction, provide session=False.
        check_data: bool
            If True, will check that the files exist on disk
            and that they contain the correct key (e.g., for HDF5 files).
            If no data is found, the function will delete any DB objects
            that don't have corresponding data. Default is True.
        search_orphans: bool
            If True, will search the data directory for files
            that match the source name and the observatory name,
            and in the case of reduced/process/simulated data,
            will search the data folder of the project. Default is True.
            This allows capturing data that was downloaded/processed
            but got detached from the corresponding DB objects.
            THIS IS NOT YET IMPLEMENTED...
        delete_missing: bool
            If True, will delete any DB objects that don't have
            corresponding data. Default is True.
            This will allow the calling code to re-create new
            objects that are associated with newly downloaded data.
        append: bool
            If True, will append the new data to the existing source.
            Default is True.

        Returns
        -------
        data: a list of RawPhotometry or similar objects
            The objects containing the data that matches
            this source and the required observatory
            and data type.
            The objects do not necessarily have data
            loaded and do not necessarily have that
            data saved on disk (these should be tested
            separately).
            If no matching data is found, returns [].
        """

        if data_type is not None:  # data type is given explicitly
            data_type = convert_data_type(data_type)
        else:
            # for each data type, check if there is that sort of data
            # on this source, if it is not an empty list, add it
            types = {t for t in allowed_data_types if getattr(self, f"{level}_{t}", [])}
            if len(types) == 0:
                raise ValueError("No data types found for this source.")
            elif len(types) > 1:
                raise ValueError(
                    "More than one data type found for this source. "
                    "Please provide a data type explicitly"
                )
            else:
                data_type = list(types)[0]

        # Class of the data requested, e.g., RawPhotometry
        DataClass = get_class_from_data_type(data_type, level=level)
        # if source existed in DB it should have raw data objects
        # if it doesn't that means the data needs to be downloaded
        found_data = [  # e.g., go over all raw_photometry...
            data
            for data in getattr(self, f"{level}_{data_type}")
            if data.observatory == legalize(obs)
        ]

        if level == "raw" and len(found_data) > 1:
            raise RuntimeError(
                f"Source {self.name} has more than one "
                f"{DataClass.__name__} object from this observatory."
            )

        # what to do if we didn't find any data attached to the source?
        if len(found_data) == 0:
            with SmartSession(session) as session:
                # try to recover the data from the DB directly
                if len(found_data) == 0 and session is not None:
                    if level == "raw":
                        found_data = session.scalars(
                            sa.select(DataClass).where(
                                DataClass.source_name == self.name,
                                DataClass.observatory == legalize(obs),
                            )
                        ).all()
                    else:
                        # search for data with the correct level,
                        # the same project name and cfg_hash,
                        # and only look for un-processed data
                        found_data = session.scalars(
                            sa.select(DataClass).where(
                                DataClass.source_name == self.name,
                                DataClass.observatory == legalize(obs),
                                DataClass.project == legalize(self.project),
                                DataClass.cfg_hash == self.cfg_hash,
                                DataClass.was_processed == False,
                            )
                        ).all()

        # check if the loaded objects have
        # corresponding files with data in them
        if check_data:
            for data in found_data:
                # data must exist either in memory (in .data) or on disk
                if not data.is_data_loaded:
                    if not data.check_file_exists():
                        if delete_missing:
                            if session is not None:
                                try:
                                    session.delete(data)
                                    session.commit()
                                except Exception:
                                    session.rollback()
                            found_data.remove(data)
                    # TODO: maybe need to remove this part if we are saving files
                    #  without any data (not even an empty dataframe)
                    else:
                        try:  # verify the data is really there
                            data.load()
                        except KeyError as e:
                            if "No object named" in str(e):
                                # This does not exist in the file
                                if delete_missing:
                                    try:
                                        session.delete(data)
                                        session.commit()
                                    except Exception:
                                        session.rollback()
                                    found_data.remove(data)

                        except Exception as e:
                            raise e

            # if no data was found, try to search for orphans
            if len(found_data) == 0 and search_orphans:
                pass  # TODO: need to figure out how to search for files
                #      that are not associated with DB objects

            for d in found_data:
                d.source = self

        if append:  # append this data on the source (should check for repeats)
            getattr(self, f"{level}_{data_type}").extend(found_data)

        return found_data

    @staticmethod
    def find_source_with_raw_data(
        source_list,
        obs,
        data_type="photometry",
        session=None,
        check_data=True,
        search_orphans=True,
        delete_missing=False,
    ):
        """
        Go over a list of sources, and for each source
        check if raw photometry exists that matches the
        observatory name given by obs.

        For each source will call get_data() with the associated
        parameters check_data, search_orphans and delete_missing
        (note that delete_missing is set to False by default).

        If no session is given, will open one and close it.

        Parameters
        ----------
        source_list: list of Source objects
            The list of sources to check.
            The list should be sorted and filtered before calling this function.
        obs: str
            The name of the observatory to search data from.
        data_type: str
            The type of data to search for. Default is 'photometry'.
        session: sqlalchemy session
            The session to use to query the DB.
            If None, will open a new session and close it.
        check_data: bool
            If True, will check if the data is actually there.
            Default is True.
        search_orphans:
            If True, will search the data directory for files
            that match the source name and the observatory name,
            and get them from file if they exist.
            Default is True.
        delete_missing: bool
            If True, will delete any DB objects that don't have
            corresponding data. Default is False.

        Returns
        -------
        source: Source object
            The first source in the list that has raw data.
            If no source is found, returns None.
        """
        with SmartSession(session) as session:

            for s in source_list:
                data = s.get_data(
                    obs=obs,
                    data_type=data_type,
                    level="raw",
                    session=session,
                    check_data=check_data,
                    search_orphans=search_orphans,
                    delete_missing=delete_missing,
                )
                if len(data) > 0:
                    return s

    # TODO: why is this even needed? should we expand to cover reduced/processed data too?
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
            If not given (or None), will open a session
            and close it at the end of the function.
            If given a session, will use it to delete
            the data from the database and leave it open.
            To avoid database interactions, pass session=False.

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

        data_class = get_class_from_data_type(data_type, level="raw")
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
            with SmartSession(session) as session:
                session.delete(raw_data)
                session.commit()  # should this just be a flush?
            return True
        else:
            raise RuntimeError(
                f"Source {self.name} has more than one "
                f"{data_class.__name__} object from this observatory."
            )

    def reset_analysis(self, session=False):
        """
        Remove all analysis results from this object,
        including detections, properties, and processed
        lightcurves.

        This does not affect the histogram results!

        Parameters
        ----------
        session: sqlalchemy session object (optional)
            If False (default), will not affect the database.
            If given a session, will use it to remove analysis results
            from the database as well. The session will remain open.
            If given None, will open a session and close it at the end of the function.
        """

        # TODO: need to check if we actually need this session stuff
        with SmartSession(session) as session:
            session.merge(self)

            if inspect(self.properties).persistent:
                session.delete(self.properties)

            for d in self.detections:
                if inspect(d).persistent:
                    session.delete(d)

            for dt in allowed_data_types:
                for d in getattr(self, f"processed_{dt}", []):
                    if inspect(d).persistent:
                        session.delete(d)

                for d in getattr(self, f"simulated_{dt}", []):
                    if inspect(d).persistent:
                        session.delete(d)

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

        lcs = []
        if self.processed_photometry:
            lcs = self.processed_photometry
        elif self.reduced_photometry:
            lcs = self.reduced_photometry

        for lc in lcs:
            lc.plot(ax=ax, ftype=ftype, ttype=ttype, **kwargs)

        return ax

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
        with SmartSession(session) as session:
            stmt = cone_search(ra=self.ra, dec=self.dec, sep=sep)
            stmt = stmt.where(
                Source.project == project, Source.cfg_hash == self.cfg_hash
            )
            sources = session.scalars(stmt).first()

            return sources is not None

    def remove_from_database_and_disk(self, session=None, remove_raw_data=False):
        """
        Remove this source from the database and from disk.
        """
        with SmartSession(session) as session:
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
    import src.database

    src.database.DATA_ROOT = "/home/guyn/data"

    session = src.database.Session()
    session.begin()

    sources = session.scalars(sa.select(Source).where(Source.project == "WD")).all()

    sources[0].plot_photometry()
