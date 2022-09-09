# This file is used to control database interaction.

# The following lines are probably not needed anymore:
# Make sure to define a virtualobserver database by
# going to /etc/postgresql/14/main/pg_hba.conf and adding:
# host virtualobserver virtualobserver 127.0.0.1/32 trust
# You may also need to check the port number in
# /etc/postgresql/14/main/postgresql.conf
# (it is usually 5432)
# Finally, you may need to do "sudo service postgresql restart"

# create DB using: psql -U postgres -d postgres -c "CREATE DATABASE virtualobserver"
# or follow this example: https://stackoverflow.com/a/30971098/18256949

import os
import sqlalchemy as sa
from sqlalchemy import func
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.orm import sessionmaker, declarative_base

DATA_ROOT = os.getenv("VO_DATA")
if DATA_ROOT is None:
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

url = "postgresql://postgres:postgres@localhost:5432/virtualobserver"

utcnow = func.timezone("UTC", func.current_timestamp())

engine = sa.create_engine(url, future=True)
if not database_exists(engine.url):
    create_database(engine.url)

# print(f"Is database found: {database_exists(engine.url)}")

Session = sessionmaker(bind=engine)


def clear_tables():
    from src.source import Source
    from src.dataset import RawData, Lightcurve
    from src.detection import DetectionInTime

    Source.metadata.drop_all(engine)
    RawData.metadata.drop_all(engine)
    Lightcurve.metadata.drop_all(engine)
    DetectionInTime.metadata.drop_all(engine)


def clear_test_sources():
    from src.source import Source

    with Session() as session:
        session.query(Source).filter(Source.project == "test_project").delete()


class VO_Base:
    """Base class for all VO classes."""

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

    project = sa.Column(
        sa.String, nullable=False, index=True, doc="Project this object belongs to."
    )

    observatory = sa.Column(
        sa.String,
        nullable=True,
        index=True,
        doc="Name of the observatory this data is associated with.",
    )

    cfg_hash = sa.Column(
        sa.String,
        nullable=True,
        index=True,
        doc="Hash of the configuration used to generate this object.",
    )


Base = declarative_base(cls=VO_Base)


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
