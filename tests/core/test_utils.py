import os
import uuid
import pytest
import numpy as np

import sqlalchemy as sa
from sqlalchemy.exc import InvalidRequestError

from src.utils import OnClose

import src.database
from src.database import (
    Session,
    SmartSession,
    NoOpSession,
    NullQueryResults,
    safe_mkdir,
)
from src.source import Source
from src.utils import (
    NamedList,
    UniqueList,
    CircularBufferList,
    find_file_ignore_case,
    luptitudes,
    sanitize_attributes,
)


def test_luptitudes():
    # check that well behaved fluxes have equal magnitudes and luptitudes
    number = 300
    mean = 1000
    rms = 10
    fluxes = np.random.normal(mean, rms, number)
    lups = luptitudes(fluxes, rms) - luptitudes(mean, rms)
    mags = -2.5 * np.log10(fluxes / mean)

    assert all(np.isclose(lups, mags, rtol=0.01))

    # low fluxes are not so well-behaved
    number = 300
    mean = 10
    rms = 10
    fluxes = np.random.normal(mean, rms, number)
    fluxes[fluxes < 0.01] = np.nan  # avoid numbers that are bad for magnitudes
    lups = luptitudes(fluxes, rms) - luptitudes(mean, rms)
    mags = -2.5 * np.log10(fluxes / mean)

    # mostly they are not close
    assert sum(np.isclose(lups, mags, rtol=0.01) == 0) > 0.9 * number


def test_sanitize_attributes():
    class Obj:
        pass

    a1 = dict(
        i=8,
        np_i=np.int64(10),
        f=8.0,
        np_f=np.float64(10.0),
        s="hello",
        c=Obj(),
        n=None,
        nan=np.nan,
        l=[1, 2, np.int32(3)],
        a=np.array([1, np.float64(2), 3]),
        d=dict(a=1, b={"c": np.float32(2.0)}),
        b=True,
        np_b=np.bool_(False),
    )

    a2 = sanitize_attributes(a1)

    assert a1["i"] == 8
    assert isinstance(a1["i"], int)
    assert a2["i"] == 8
    assert isinstance(a2["i"], int)

    assert a1["np_i"] == 10
    assert isinstance(a1["np_i"], np.int64)
    assert not isinstance(a1["np_i"], int)
    assert a2["np_i"] == 10
    assert isinstance(a2["np_i"], int)

    assert a1["f"] == 8.0
    assert isinstance(a1["f"], float)
    assert a2["f"] == 8.0
    assert isinstance(a2["f"], float)

    assert a1["np_f"] == 10.0
    assert isinstance(a1["np_f"], np.float64)
    assert a2["np_f"] == 10.0
    assert isinstance(a2["np_f"], float)
    assert not isinstance(a2["np_f"], np.float64)

    assert a1["s"] == "hello"
    assert isinstance(a1["s"], str)
    assert a2["s"] == "hello"
    assert isinstance(a2["s"], str)

    assert a1["c"] == a2["c"]
    assert isinstance(a1["c"], Obj)
    assert isinstance(a2["c"], Obj)

    assert a1["n"] is None
    assert a2["n"] is None

    assert a1["l"] == [1, 2, 3]
    assert isinstance(a1["l"], list)
    assert a2["l"] == [1, 2, 3]
    assert isinstance(a2["l"], list)

    assert a2["l"][2] == 3
    assert isinstance(a2["l"][2], int)
    assert not isinstance(a2["l"][2], np.int32)

    assert isinstance(a1["a"], np.ndarray)
    assert all(a1["a"] == [1, 2, 3])
    assert isinstance(a2["a"], list)
    assert a2["a"] == [1, 2, 3]
    assert a1["a"][1] == 2
    assert isinstance(a1["a"][1], np.float64)
    assert a2["a"][1] == 2
    assert isinstance(a2["a"][1], float)

    assert a1["d"] == dict(a=1, b={"c": 2.0})
    assert isinstance(a1["d"], dict)
    assert a1["d"]["b"]["c"] == 2.0
    assert isinstance(a1["d"]["b"]["c"], np.float32)

    assert a2["d"] == dict(a=1, b={"c": 2.0})
    assert isinstance(a2["d"], dict)
    assert a2["d"]["b"]["c"] == 2.0
    print(type(a2["d"]["b"]["c"]))
    assert isinstance(a2["d"]["b"]["c"], float)

    assert a1["nan"] is np.nan
    assert a2["nan"] is None

    assert a1["b"] == 1
    assert isinstance(a1["b"], bool)
    print(type(a2["b"]))
    assert a2["b"] == 1
    assert isinstance(a2["b"], bool)

    assert a1["np_b"] == 0
    assert isinstance(a1["np_b"], np.bool_)
    assert not isinstance(a1["np_b"], bool)
    assert a2["np_b"] == 0
    assert isinstance(a2["np_b"], bool)
    assert not isinstance(a2["np_b"], np.bool_)


def test_on_close_utility():
    a = []
    b = []

    def append_to_list(a, b, clear_a_at_end=False):
        if clear_a_at_end:
            _ = OnClose(lambda: a.clear())
        a.append(1)
        b.append(a[0])

    append_to_list(a, b)
    assert a == [1]
    assert b == [1]

    append_to_list(a, b, clear_a_at_end=True)
    assert a == []
    assert b == [1, 1]


def test_named_list():
    class TempObject:
        pass

    obj1 = TempObject()
    obj1.name = "One"

    obj2 = TempObject()
    obj2.name = "Two"

    nl = NamedList()
    nl.append(obj1)
    nl.append(obj2)

    assert len(nl) == 2
    assert nl[0] == obj1
    assert nl[1] == obj2

    assert nl["One"] == obj1
    assert nl["Two"] == obj2
    assert nl.keys() == ["One", "Two"]

    with pytest.raises(ValueError):
        nl["Three"]

    with pytest.raises(ValueError):
        nl["one"]

    with pytest.raises(IndexError):
        nl[2]

    with pytest.raises(TypeError):
        nl[1.0]

    # now a list that ignores case
    nl = NamedList(ignorecase=True)
    nl.append(obj1)
    nl.append(obj2)

    assert len(nl) == 2
    assert nl[0] == obj1
    assert nl[1] == obj2

    assert nl["one"] == obj1
    assert nl["two"] == obj2
    assert nl.keys() == ["One", "Two"]

    with pytest.raises(ValueError):
        nl["Three"]


def test_unique_list():
    class TempObject:
        pass

    obj1 = TempObject()
    obj1.name = "object one"
    obj1.foo = "foo1"
    obj1.bar = "common bar"

    obj2 = TempObject()
    obj2.name = "object two"
    obj2.foo = "foo2"
    obj2.bar = "common bar"

    # same attributes as obj1, but different object
    obj3 = TempObject()
    obj3.name = "object one"
    obj3.foo = "foo1"
    obj3.bar = "common bar"

    # the default is to use the name attribute
    ul = UniqueList()
    ul.append(obj1)
    ul.append(obj2)
    assert len(ul) == 2
    assert ul[0] == obj1
    assert ul[1] == obj2

    # appending obj3 will remove obj1
    ul.append(obj3)
    assert len(ul) == 2
    assert ul[0] == obj2
    assert ul[1] == obj3

    # check string indexing
    assert ul["object one"] == obj3
    assert ul["object two"] == obj2

    # now try with a different attribute
    ul = UniqueList(comparison_attributes=["foo", "bar"])
    ul.append(obj1)
    ul.append(obj2)
    assert len(ul) == 2
    assert ul[0] == obj1
    assert ul[1] == obj2

    # string indexing in this case returns a list
    assert ul["foo1"] == [obj1]
    assert ul["foo2"] == [obj2]

    # try indexing with a list or tuple
    assert ul[["foo1", "common bar"]] == obj1
    assert ul[["foo2", "common bar"]] == obj2

    # should work without brackets
    assert ul["foo1", "common bar"] == obj1
    assert ul["foo2", "common bar"] == obj2

    # appending obj3 will remove obj1
    ul.append(obj3)
    assert len(ul) == 2
    assert ul[0] == obj2
    assert ul[1] == obj3

    # try a list with three comparison_attributes
    ul = UniqueList(comparison_attributes=["name", "foo", "bar"])
    ul.append(obj1)
    ul.append(obj2)
    assert len(ul) == 2

    # check that array indexing works with two out of three attributes
    assert ul[["object one", "foo1"]] == [obj1]
    assert ul[["object two", "foo2"]] == [obj2]

    # check that we can ignore case
    obj4 = TempObject()
    obj4.name = "Foo"

    obj5 = TempObject()
    obj5.name = "fOO"

    ul = UniqueList(comparison_attributes=["name"], ignorecase=True)
    ul.append(obj4)
    ul.append(obj5)
    assert len(ul) == 1
    assert ul["foo"] == obj5
    assert ul["FOO"] == obj5


def test_circular_buffer_list():
    cbl = CircularBufferList(3)
    cbl.append(1)
    cbl.append(2)
    cbl.append(3)
    assert cbl == [1, 2, 3]
    assert cbl.total == 3
    cbl.append(4)
    assert cbl == [2, 3, 4]
    assert cbl.total == 4
    cbl.extend([5, 6])
    assert cbl == [4, 5, 6]
    assert cbl.total == 6


def test_safe_mkdir():
    # can make a folder inside the data folder
    new_path = os.path.join(src.database.DATA_ROOT, uuid.uuid4().hex)
    assert not os.path.isdir(new_path)

    safe_mkdir(new_path)
    assert os.path.isdir(new_path)

    os.rmdir(new_path)

    # can make a folder under the code root's results folder
    new_path = os.path.join(src.database.CODE_ROOT, "results", uuid.uuid4().hex)
    assert not os.path.isdir(new_path)

    safe_mkdir(new_path)
    assert os.path.isdir(new_path)

    os.rmdir(new_path)

    # can make a folder under the temporary data folder
    new_path = os.path.join(src.database.DATA_TEMP, uuid.uuid4().hex)
    assert not os.path.isdir(new_path)

    safe_mkdir(new_path)
    assert os.path.isdir(new_path)

    os.rmdir(new_path)

    # this does not work anywhere else:
    new_path = os.path.join(src.database.CODE_ROOT, uuid.uuid4().hex)
    assert not os.path.isdir(new_path)
    with pytest.raises(ValueError) as e:
        safe_mkdir(new_path)
    assert "Cannot make a new folder not inside the following folders" in str(e.value)

    # try a relative path
    new_path = os.path.join(src.database.CODE_ROOT, "results", "..", uuid.uuid4().hex)
    assert not os.path.isdir(new_path)
    with pytest.raises(ValueError) as e:
        safe_mkdir(new_path)
    assert "Cannot make a new folder not inside the following folders" in str(e.value)

    new_path = os.path.join(src.database.CODE_ROOT, "result", uuid.uuid4().hex)
    assert not os.path.isdir(new_path)
    with pytest.raises(ValueError) as e:
        safe_mkdir(new_path)
    assert "Cannot make a new folder not inside the following folders" in str(e.value)


def test_find_file_ignore_case(data_dir):
    current_dir = os.getcwd()
    try:
        os.chdir(data_dir)

        filename1 = "foo.txt"
        filename2 = "Foo.txt"
        filename3 = "FOO.txt"
        subfolder = "subfolder"

        open(filename1, "w").close()
        open(filename2, "w").close()

        os.mkdir(subfolder)
        open(os.path.join(subfolder, filename3), "w").close()

        # lower case returns lower case
        assert find_file_ignore_case(filename1) == os.path.join(data_dir, filename1)

        # watch out! the lower case will be grabbed before the real match!
        assert find_file_ignore_case(filename2) == os.path.join(data_dir, filename1)

        # search for this file in the subfolder specifically
        assert find_file_ignore_case(filename3, subfolder) == os.path.join(
            data_dir, subfolder, filename3
        )

        # search in the subfolder using an absolute path
        assert find_file_ignore_case(
            filename3, os.path.join(data_dir, subfolder)
        ) == os.path.join(data_dir, subfolder, filename3)

        # if we don't specify the subfolder, it will get the lower case file, not the one in the subfolder
        assert find_file_ignore_case(filename3) == os.path.join(data_dir, filename1)

        # if we specify the folders, in this order, it finds the first filename
        assert find_file_ignore_case(filename3, [".", subfolder]) == os.path.join(
            data_dir, filename1
        )

        # if we change the order of folders to search it returns the correct file:
        assert find_file_ignore_case(filename3, [subfolder, "."]) == os.path.join(
            data_dir, subfolder, filename3
        )

    finally:
        os.chdir(current_dir)


def test_smart_session(new_source):

    try:  # make sure to re-state autobegin=True at the end
        # note that with regular sessions you'd need to call .begin()
        with Session() as session:
            # set this just to test when the sessions are closed:
            session.autobegin = False
            # now we need to add this at start of each session:
            session.begin()

            session.add(new_source)
            session.commit()

        assert new_source.id is not None

        assert session._transaction is None
        # this session has been closed, so this should fail
        with pytest.raises(InvalidRequestError):
            session.scalars(sa.select(Source).where(Source.id == new_source.id)).all()

        # try using a SmartSession, which should also begin the session:
        with SmartSession() as session:
            session.begin()
            # this should work
            sources = session.scalars(
                sa.select(Source).where(Source.id == new_source.id)
            ).all()
            assert any([s.id == new_source.id for s in sources])

        assert session._transaction is None
        # this session has been closed, so this should fail
        with pytest.raises(InvalidRequestError):
            session.scalars(sa.select(Source).where(Source.id == new_source.id)).all()

        # try using a SmartSession without a context manager inside a function
        def try_smart_session(session=None):
            with SmartSession(session) as session:
                if session._transaction is None:
                    session.begin()
                sources = session.scalars(
                    sa.select(Source).where(Source.id == new_source.id)
                ).all()
                assert len(sources) > 0

        try_smart_session()  # the function is like a context manager

        assert session._transaction is None
        # this session has been closed, so this should fail
        with pytest.raises(InvalidRequestError):
            session.scalars(sa.select(Source).where(Source.id == new_source.id)).all()

        # try calling the function again, but surrounded by a context manager
        with SmartSession() as session:
            try_smart_session(session)

            # session should still work even though function has finished
            sources = session.scalars(
                sa.select(Source).where(Source.id == new_source.id)
            ).all()
            assert len(sources) > 0

        assert session._transaction is None
        # this session has been closed, so this should fail
        with pytest.raises(InvalidRequestError):
            session.scalars(sa.select(Source).where(Source.id == new_source.id)).all()

        # with an explicit False this should be a no-op session
        with SmartSession(False) as session:
            assert isinstance(session, NoOpSession)

            query = session.scalars(sa.select(Source).where(Source.id == new_source.id))
            assert isinstance(query, NullQueryResults)
            sources = query.all()
            assert sources == []

            query = session.scalars(sa.select(Source).where(Source.id == new_source.id))
            assert isinstance(query, NullQueryResults)
            source = query.first()
            assert source is None

        # try opening a session inside an open session:
        with SmartSession() as session:
            session.begin()
            with SmartSession(session) as session2:
                assert session2 is session
                sources = session2.scalars(
                    sa.select(Source).where(Source.id == new_source.id)
                ).all()
                assert len(sources) > 0

            # this still works because internal session doesn't auto-close
            sources = session2.scalars(
                sa.select(Source).where(Source.id == new_source.id)
            ).all()
            assert len(sources) > 0

        assert session._transaction is None
        # this should fail because the external session is closed
        with pytest.raises(InvalidRequestError):
            session.scalars(sa.select(Source).where(Source.id == new_source.id)).all()

        # now change the global scope
        import src.database

        try:  # make sure we don't leave the global scope changed
            src.database.NO_DB_SESSION = True

            with SmartSession() as session:
                assert isinstance(session, NoOpSession)
                query = session.scalars(
                    sa.select(Source).where(Source.id == new_source.id)
                )
                assert isinstance(query, NullQueryResults)
                sources = query.all()
                assert sources == []

                query = session.scalars(
                    sa.select(Source).where(Source.id == new_source.id)
                )
                assert isinstance(query, NullQueryResults)
                source = query.first()
                assert source is None

        finally:
            src.database.NO_DB_SESSION = False

    finally:
        # make sure to re-state autobegin=True at the end
        with Session() as session:
            session.autobegin = True
