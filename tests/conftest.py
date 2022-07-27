import uuid
import numpy as np

import pytest

from src.source import Source
from src.project import Project


@pytest.fixture
def new_source():
    source = Source(
        name=str(uuid.uuid4()),
        ra=np.random.uniform(0, 360),
        dec=np.random.uniform(-90, 90),
    )
    return source


@pytest.fixture
def test_project():
    project = Project(name="test_project", config=False)
    return project


@pytest.fixture
def ztf_project():
    project = Project(
        name="test_ZTF",
        params={
            "observatories": "ZTF",  # a single observatory named ZTF
        },
        config=False,
    )
    return project
