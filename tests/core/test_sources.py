import os
import uuid
import pytest

import numpy as np
import pandas as pd
from astropy.time import Time

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError


from src.database import SmartSession
from src.source import Source, DEFAULT_PROJECT
from src.properties import Properties
from src.dataset import RawPhotometry
from src.utils import UniqueList


def test_add_source_and_data(data_dir, test_hash):
    fullname = ""
    try:  # at end, delete the temp file

        with SmartSession() as session:
            # create a random source
            source_name = str(uuid.uuid4())

            # make sure source cannot be initialized with bad keyword
            with pytest.raises(ValueError) as e:
                _ = Source(name=source_name, foobar="test")
                assert "Unknown keyword argument" in str(e.value)

            new_source = Source(
                name=source_name,
                ra=np.random.uniform(0, 360),
                dec=np.random.uniform(-90, 90),
                test_hash=test_hash,
            )
            assert isinstance(new_source.raw_photometry, UniqueList)

            # add some data to the source
            num_points = 10
            filt = np.random.choice(["r", "g", "i", "z"], num_points)
            mjd = np.random.uniform(57000, 58000, num_points)
            mag = np.random.uniform(15, 20, num_points)
            mag_err = np.random.uniform(0.1, 0.5, num_points)
            test_data = dict(mjd=mjd, mag=mag, mag_err=mag_err, filter=filt)
            df = pd.DataFrame(test_data)

            # make sure data cannot be initialized with bad keyword
            with pytest.raises(ValueError) as e:
                _ = RawPhotometry(data=df, foobar="test")
                assert "Unknown keyword argument" in str(e.value)

            # add the data to a database mapped object
            new_data = RawPhotometry(
                data=df,
                source_name=source_name,
                observatory="demo",
                folder="data_temp",
                altdata=dict(foo="bar"),
                test_hash=test_hash,
            )

            # check the times make sense
            start_time = Time(min(df.mjd), format="mjd").datetime
            end_time = Time(max(df.mjd), format="mjd").datetime
            assert start_time == new_data.time_start
            assert end_time == new_data.time_end

            session.add(new_source)
            # must add this separately, as cascades are cancelled
            session.add(new_data)
            new_source.raw_photometry = [new_data]
            assert isinstance(new_source.raw_photometry, UniqueList)
            assert len(new_source.raw_photometry) == 1

            # check that re-appending this same data does not add another copy to list
            new_source.raw_photometry.append(new_data)
            assert len(new_source.raw_photometry) == 1

            # this should not work because
            # no filename was specified
            with pytest.raises(ValueError):
                session.commit()
            session.rollback()

            session.add(new_source)
            # must add this separately, as cascades are cancelled
            session.add(new_data)
            new_source.raw_photometry = [new_data]

            new_data.filename = "test_data.h5"
            # this should not work because the file
            # does not yet exist and autosave is False
            assert new_data.autosave is False
            assert new_data.check_file_exists() is False
            with pytest.raises(ValueError):
                session.commit()
            session.rollback()
            new_data.filename = None  # reset the filename

            session.add(new_source)
            # must add this separately, as cascades are cancelled
            session.add(new_data)
            new_source.raw_photometry = [new_data]

            # filename should be auto-generated
            new_data.save()  # must save to allow RawPhotometry to be added to DB
            session.commit()  # this should now work fine
            assert new_source.id is not None
            assert new_source.id == new_data.source.id

            # try to recover the data
            filename = new_data.filename
            fullname = os.path.join(data_dir, "data_temp", filename)

            with pd.HDFStore(fullname, "r") as store:
                key = store.keys()[0]
                df_from_file = store.get(key)
                assert df_from_file.equals(df)
                altdata = store.get_storer(key).attrs.altdata
                assert altdata["foo"] == "bar"

        # check that the data is in the database
        with SmartSession() as session:
            source = session.scalars(
                sa.select(Source).where(Source.name == source_name)
            ).first()

            assert source is not None
            assert source.id == new_source.id
            source.get_data(
                obs="demo", data_type="photometry", level="raw", session=session
            )
            assert len(source.raw_photometry) == 1
            assert source.raw_photometry[0].filename == filename
            assert source.raw_photometry[0].filekey == new_data.filekey
            assert source.raw_photometry[0].source.id == new_source.id

            # this autoloads the data:
            assert source.raw_photometry[0].data.equals(df)

    finally:
        if os.path.isfile(fullname):
            os.remove(fullname)
            if os.path.isdir(os.path.dirname(fullname)):
                os.rmdir(os.path.dirname(fullname))
    with pytest.raises(FileNotFoundError):
        with open(fullname) as file:
            pass

    # make sure loading this data does not work without file
    with SmartSession() as session:
        source = session.scalars(
            sa.select(Source).where(Source.name == source_name)
        ).first()
        assert source is not None

        # get_data will load a RawPhotometry file without checking if it has data
        source.get_data(
            obs="demo",
            data_type="photometry",
            level="raw",
            session=session,
            check_data=False,
            delete_missing=False,
        )
        assert len(source.raw_photometry) == 1
        assert len(source.raw_photometry) == 1
        with pytest.raises(FileNotFoundError):
            source.raw_photometry[0].data.equals(df)

    # cleanup
    with SmartSession() as session:
        session.execute(sa.delete(Source).where(Source.name == source_name))
        session.commit()


def test_source_unique_constraint(test_hash):

    with SmartSession() as session:
        name1 = str(uuid.uuid4())
        source1 = Source(name=name1, ra=0, dec=0, test_hash=test_hash)
        assert source1.cfg_hash == ""  # the default has is an empty string
        session.add(source1)

        source2 = Source(name=name1, ra=0, dec=0, test_hash=test_hash)
        assert source1.cfg_hash == ""  # the default has is an empty string
        session.add(source2)

        # should fail as both sources have the same name
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

        name2 = str(uuid.uuid4())
        source2 = Source(name=name2, ra=0, dec=0, test_hash=test_hash)
        session.add(source1)
        session.add(source2)
        session.commit()

        # try to add source2 with the same name but different project
        assert source1.project == DEFAULT_PROJECT
        source2.project = "another test"
        source2.name = name1
        session.add(source2)
        session.commit()

        # try to add source2 with the same name but different cfg_hash
        source2.project = DEFAULT_PROJECT
        assert source2.project == source1.project
        assert source2.name == source1.name

        source2.cfg_hash = "another hash"
        session.add(source2)
        session.commit()


def test_source_properties(test_hash):
    with SmartSession() as session:
        name1 = str(uuid.uuid4())
        source = Source(name=name1, ra=0, dec=0, test_hash=test_hash)
        source.properties = Properties()
        session.add(source)
        session.commit()
        source_id = source.id
        assert source_id is not None
        prop_id = source.properties.id
        assert prop_id is not None

    # now make sure they are persisted
    with SmartSession() as session:
        source = session.scalars(
            sa.select(Source).where(Source.id == source_id)
        ).first()
        assert source is not None
        assert source.id == source_id
        assert source.properties.id == prop_id

        prop = session.scalars(
            sa.select(Properties).where(Properties.id == prop_id)
        ).first()
        assert prop is not None
        assert prop.id == prop_id
        assert prop.source.id == source_id
        assert prop.source_name == name1

        # what happens if props are removed from source?
        source.properties = None
        session.commit()

    with SmartSession() as session:
        source = session.scalars(
            sa.select(Source).where(Source.id == source_id)
        ).first()
        assert source is not None
        assert source.id == source_id

        prop = session.scalars(
            sa.select(Properties).where(Properties.id == prop_id)
        ).first()
        assert prop is None

        # delete the source as well
        session.delete(source)
        session.commit()

    with SmartSession() as session:
        name2 = str(uuid.uuid4())
        source = Source(name=name2, ra=0, dec=0, test_hash=test_hash)
        p = Properties()
        p.source = source
        session.add(p)
        session.commit()
        source_id = source.id
        assert source_id is not None
        prop_id = source.properties.id
        assert prop_id is not None

    # now make sure they are persisted
    with SmartSession() as session:
        source = session.scalars(
            sa.select(Source).where(Source.id == source_id)
        ).first()
        assert source is not None
        assert source.id == source_id
        assert source.properties.id == prop_id

        prop = session.scalars(
            sa.select(Properties).where(Properties.id == prop_id)
        ).first()
        assert prop is not None
        assert prop.id == prop_id
        assert prop.source.id == source_id
        assert prop.source_name == name2
