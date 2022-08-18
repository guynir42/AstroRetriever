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

import uuid
import sqlalchemy as sa
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.orm import sessionmaker, declarative_base


url = "postgresql://postgres:postgres@localhost:5432/virtualobserver"

engine = sa.create_engine(url, future=True)
if not database_exists(engine.url):
    create_database(engine.url)

# print(f"Is database found: {database_exists(engine.url)}")

Session = sessionmaker(bind=engine)

Base = declarative_base()


def clear_tables():
    from src.source import Source
    from src.dataset import RawData, Lightcurve
    from src.detection import Detection

    Source.metadata.drop_all(engine)
    RawData.metadata.drop_all(engine)
    Lightcurve.metadata.drop_all(engine)
    # Detection.metadata.drop_all(engine)


def clear_test_sources():
    from src.source import Source

    with Session() as session:
        session.query(Source).filter(Source.project == "test_project").delete()


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
