import numpy as np
import yaml

from src import utils


def test_utils():
    with open("passwords.yml", "w") as file:
        data = {"test": {"username": "guy", "password": "12345"}}
        yaml.dump(data, file, sort_keys=False)
    credentials = utils.get_username_password("test")

    assert credentials[0] == "guy"
    assert credentials[1] == "12345"
