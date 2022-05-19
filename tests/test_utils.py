import os
import yaml

from src import utils


def test_utils():
    filename = "passwords_test.yml"
    with open(filename, "w") as file:
        data = {"test": {"username": "guy", "password": "12345"}}
        yaml.dump(data, file, sort_keys=False)
    credentials = utils.get_username_password("test", filename)

    assert credentials[0] == "guy"
    assert credentials[1] == "12345"

    os.remove(filename)
