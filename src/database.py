# This file is used to control database interaction.

# The following lines are probably not needed anymore:
# Make sure to define a AstroRetriever database by
# going to /etc/postgresql/14/main/pg_hba.conf and adding:
# host AstroRetriever AstroRetriever 127.0.0.1/32 trust
# You may also need to check the port number in
# /etc/postgresql/14/main/postgresql.conf
# (it is usually 5432)
# Finally, you may need to do "sudo service postgresql restart"

# create DB using: psql -U postgres -d postgres -c "CREATE DATABASE AstroRetriever"
# or follow this example: https://stackoverflow.com/a/30971098/18256949

# To drop the entire database (in case things get very messed up):
# Use sudo -u postgres psql -c "DROP DATABASE AstroRetriever"
# This will only work if no connections are active.

import os

from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy import func
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from sqlalchemy.orm.session import make_transient

DATA_ROOT = os.getenv("RETRIEVER_DATA")
if DATA_ROOT is None:  # TODO: should also check if folder exists?
    DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

# set this to True if you want to disable all database interactions
NO_DB_SESSION = False

url = "postgresql://postgres:postgres@localhost:5432/astroretriever"

utcnow = func.timezone("UTC", func.current_timestamp())

engine = sa.create_engine(url, future=True)
if not database_exists(engine.url):
    create_database(engine.url)

# print(f"Is database found: {database_exists(engine.url)}")

Session = scoped_session(
    sessionmaker(bind=engine, expire_on_commit=False, autobegin=False)
)


class NullQueryResults:
    @staticmethod
    def all():
        return []

    @staticmethod
    def first():
        return None

    @staticmethod
    def last():
        return None

    @staticmethod
    def one():
        return None

    @staticmethod
    def one_or_none():
        return None


class NoOpSession:
    @staticmethod
    def delete(*_, **__):
        pass

    @staticmethod
    def add(*_, **__):
        pass

    @staticmethod
    def add_all(*_, **__):
        pass

    @staticmethod
    def commit(*_, **__):
        pass

    @staticmethod
    def rollback(*_, **__):
        pass

    @staticmethod
    def close():
        pass

    @staticmethod
    def execute(*_, **__):
        return NullQueryResults()

    @staticmethod
    def scalars(*_, **__):
        return NullQueryResults()

    @staticmethod
    def get(*_, **__):
        return None

    @staticmethod
    def merge(*_, **__):
        return None


@contextmanager
def SmartSession(input_session=None):
    """
    Retrun a Session() instance that may or may not
    be inside a context manager.

    If the input is already a session, just return that.
    If the input is None, create a session that would
    close at the end of the life of the calling scope.
    If the input is False, create a no-op session.
    If the global switch NO_DB_SESSION is set to True,
    will always return a no-op session.

    """
    # this session will never do anything
    if NO_DB_SESSION:
        yield NoOpSession()

    # open a new session and close it when outer scope is done
    elif input_session is None:
        with Session() as session:
            session.begin()
            yield session

    # explicitly ask for a no-op session
    elif input_session is False:
        yield NoOpSession()

    # return the input session with the same scope as given
    elif isinstance(input_session, NoOpSession) or isinstance(
        input_session, sa.orm.session.Session
    ):
        yield input_session

    # wrong input type
    else:
        raise TypeError(
            "input_session must be a sqlalchemy session or a NoOpSession or None"
        )


class CloseSession:
    def __init__(self, session=None):
        self.session = session

    def __del__(self):
        if self.session is not None:
            self.session.close()


def clear_tables():
    from src.source import Source
    from src.dataset import RawPhotometry, Lightcurve
    from src.detection import Detection
    from src.properties import Properties

    try:
        Properties.metadata.drop_all(engine)
    except:
        pass
    try:
        Detection.metadata.drop_all(engine)
    except:
        pass
    try:
        Lightcurve.metadata.drop_all(engine)
    except:
        pass
    try:
        RawPhotometry.metadata.drop_all(engine)
    except:
        pass
    try:
        Source.metadata.drop_all(engine)
    except:
        pass


def clear_test_objects():
    from src.source import Source
    from src.dataset import RawPhotometry, Lightcurve
    from src.detection import Detection
    from src.properties import Properties

    with Session() as session:
        session.execute(sa.delete(Properties).where(Source.test_hash.is_not(None)))
        session.execute(sa.delete(Detection).where(Source.test_hash.is_not(None)))
        session.execute(sa.delete(Lightcurve).where(Source.test_hash.is_not(None)))
        session.execute(sa.delete(RawPhotometry).where(Source.test_hash.is_not(None)))
        session.execute(sa.delete(Source).where(Source.test_hash.is_not(None)))


class RetrieverBase:
    """Base class for all AstroRetriever classes."""

    id = sa.Column(
        sa.Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
        doc="Unique identifier for this dataset",
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

    test_hash = sa.Column(
        sa.String,
        nullable=True,
        default=False,
        doc="Apply this to any test objects, "
        "either in the testing suite or when "
        "just debugging code interactively. "
        "Remove such objects using clear_test_objects() "
        "which removes any rows with non-null test_hash."
        "To clean up a specific test, write a unique string"
        "to this column and then clear only those rows.",
    )

    def keywords_to_columns(self, input_dict):
        """
        Read off any keywords that exist on this
        object and apply them to self, while also
        removing these keyword/values from the input dict.
        """
        for k in list(input_dict.keys()):
            if hasattr(self, k):
                setattr(self, k, input_dict.pop(k))


Base = declarative_base(cls=RetrieverBase)


if __name__ == "__main__":
    import numpy as np
    from src.source import Source

    # Source.metadata.create_all(engine)

    # with Session() as session:
    #     new_source = Source(
    #         name=str(uuid.uuid4()),
    #         ra=np.random.uniform(0, 360),
    #         dec=np.random.uniform(-90, 90),
    #     )
    #     if not new_source.check_duplicates(session=session, sep=2 / 3600):
    #         session.add(new_source)
    #         session.commit()
    #     else:
    #         print(
    #             f'Duplicate source found within {2}" of ra= {new_source.ra:.3f} / dec= {new_source.dec:.3f}'
    #         )
