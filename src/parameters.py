import os
import yaml


DATA_ROOT = os.getenv("DATA") or ""


class Parameters:
    """
    Keep track of parameters for any of the other classes.
    The list of required parameters has to be defined either
    by hard-coded values or by a YAML file.
    Use the load() method to load the parameters from a YAML file.
    Use the read() method to load the parameters from a dictionary.
    Use verify() to make sure all parameters are set to some
    value (including None!).
    Use save() to save the parameters to a YAML file
    to keep track of what parameters were used.

    """

    def __init__(self, required_pars=[]):
        """
        Setup a Parameters object.
        After setting up, the parameters can be set
        either by hard-coded values or by a YAML file,
        using the load() method,
        or by a dictionary, using the read() method.

        Parameters
        ----------
        required_pars: list of str
            A list of strings with the names of the required parameters.
            If any of these parameters are not set after loading,
            a ValueError is raised.
        """
        self.required_pars = required_pars

    def verify(self):
        """
        Make sure all required parameters were
        set by external code or by reading a
        file or dictionary.
        If not, raises a ValueError.
        """
        for p in self.required_pars:
            if not hasattr(self, p):
                raise ValueError(f"Parameter {p} is not set.")

    def load(self, filename, key=None):
        """
        Read parameters from a YAML file.
        If any parameters were already defined,
        they will be overridden by the values in the file.

        Parameters
        ----------
        filename: str
            Full or relative path and name to the YAML file..
        key: str
            Read only a specific key from the YAML file,
            and use only the keys under that to populate
            the parameters.
        """
        try:

            if os.path.isabs(filename):
                filepath = filename
            else:
                basepath = os.path.dirname(__file__)
                filepath = os.path.abspath(os.path.join(basepath, "..", filename))

            with open(filepath) as file:
                config = yaml.safe_load(file)

            if key is None:
                self.read(config)
            else:
                self.read(config[key])

        except FileNotFoundError:
            raise

    def read(self, dictionary):
        """
        Read parameters from a dictionary.
        If any parameters were already defined,
        they will be overridden by the values in the dictionary.

        Parameters
        ----------
        dictionary: dict
            A dictionary with the parameters.
        """
        for k, v in dictionary.items():
            setattr(self, k, v)

    def update(self, dictionary):
        """
        Update parameters from a dictionary.
        Any dict or set parameters already defined
        will be updated by the values in the dictionary,
        otherwise values are replaced by the input values.

        Parameters
        ----------
        dictionary: dict
            A dictionary with the parameters.
        """

        for k, v in dictionary.items():
            if hasattr(self, k):  # need to update
                if isinstance(getattr(self, k), (dict, set)):
                    getattr(self, k).update(v)
                else:
                    setattr(self, k, v)
            else:  # just add this parameter
                setattr(self, k, v)

    def save(self, filename):
        """
        Save parameters to a YAML file.
        This is used to keep a record of the actual
        parameters used in the analysis.

        Parameters
        ----------
        filename: str
            Full or relative path and name to the YAML file.
        """
        # TODO: what about combining parameters from multiple objects?
        with open(filename, "w") as file:
            yaml.dump(self.__dict__, file, default_flow_style=False)

    def get_data_path(self):
        """
        Get the path to the data directory.
        """
        if hasattr(self, "data_folder"):
            return os.path.join(DATA_ROOT or "", self.data_path)
        else:
            return DATA_ROOT or ""
