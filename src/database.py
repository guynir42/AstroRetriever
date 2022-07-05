# This file is used to control database interaction.
# Make sure to define a virtualobserver database by
# going to /etc/postgresql/14/main/pg_hba.conf and adding:
# host virtualobserver virtualobserver 127.0.0.1/32 trust
# You may also need to check the port number in
# /etc/postgresql/14/main/postgresql.conf
# (it is usually 5432)
# Finally, you may need to do "sudo service postgresql restart"

# create DB using: psql -U postgres -d postgres -c "CREATE DATABASE virtualobserver"
# or follow this example: https://stackoverflow.com/a/30971098/18256949
import sqlalchemy as sa
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.orm import sessionmaker, declarative_base

url = "postgresql://postgres@localhost:5432/virtualobserver"

engine = sa.create_engine(url, future=True)
if not database_exists(engine.url):
    create_database(engine.url)

print(f"Is database found: {database_exists(engine.url)}")

Session = sessionmaker(bind=engine)

Base = declarative_base()

if __name__ == "__main__":
    import numpy as np
    from src.source import Source

    # from src.dataset import Dataset
    # from src.detection import Detection

    engine.echo = True
    Source.metadata.create_all(engine)

    with Session() as session:
        new_source = Source(
            name="test_source",
            ra=np.random.uniform(0, 360),
            dec=np.random.uniform(-90, 90),
        )
        session.add(new_source)
        session.commit()
