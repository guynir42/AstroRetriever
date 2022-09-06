import os
import yaml
from pprint import pprint

from src.database import DATA_ROOT

# A cached dictionary of dictionaries
# loaded form YAML files.
# Each time load() is called with the same
# filename but different key, it will not
# need to re-read the file from disk.
LOADED_FILES = {}


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
        self.verbose = 0  # level of verbosity (0=quiet)

        # these have not been set by file or kwargs input
        self._default_keys = []

    def __contains__(self, key):
        return hasattr(self, key)

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

    def default_values(self, **kwargs):
        """
        Add the input values as attributes of this object,
        but for each attribute only add it if it has not been
        defined already.
        This is useful for hard-coding default values that may or
        may not have been loaded in a previous call to e.g., load().
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)
                self._default_keys.append(k)

    def replace_unset(self, **kwargs):
        """
        Replace any unset parameters with the input values.
        If key does not exist on pars, or if it was
        set by default_values() then it will be set
        by the values given in kwargs.
        This method can only be called after default_values(),
        and before manually changing any attributes.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k) or k in self._default_keys:
                setattr(self, k, v)

    @staticmethod
    def get_file_from_disk(filename):
        if filename not in LOADED_FILES:
            with open(filename) as file:
                LOADED_FILES[filename] = yaml.safe_load(file)

        return LOADED_FILES[filename]

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

            config = self.get_file_from_disk(filepath)

            if key is None:
                self.read(config)
            else:
                self.read(config.get(key, {}))

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
                if isinstance(getattr(self, k), set) and isinstance(v, (set, list)):
                    getattr(self, k).update(v)
                elif isinstance(getattr(self, k), dict) and isinstance(v, dict):
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
            return os.path.join(DATA_ROOT or "", self.data_folder)
        else:
            return DATA_ROOT or ""

    def get_class(self, name, **kwargs):
        """
        Get a class from a string.
        To load a class, there must be (at least) two
        definitions in the parameter object:
        - <name>_module: the import path to the file containing
          the class definition. E.g., src.my_simulator
        - <name>_class: the name of the class. E.g., MySimulator.
        - <name>_kwargs: a dictionary with initialization arguments.
          The kwargs given to this function override those loaded from file.

        Parameters
        ----------
        name: str
            The name of the class.
            If one of the default (core) classes is requested,
            the class can be loaded even without specifying the
            module and class names.
            E.g., if name="Analysis", then the Analysis class
            will be loaded from the "src.analysis" module.
        kwargs: dict
            Additional keyword arguments to pass to the class.

        Returns
        -------
        class
            The class object.
        """

        # default module and class_name for core classes:
        if name.lower() == "analysis":
            module = "src.analysis"
            class_name = "Analysis"
        elif name.lower() == "simulator":
            module = "src.simulator"
            class_name = "Simulator"
        elif name.lower() == "catalog":
            module = "src.catalog"
            class_name = "Catalog"
        elif name.lower() == "finder":
            module = "src.finder"
            class_name = "Finder"
        elif name.lower() == "quality":
            module = "src.quality"
            class_name = "Quality"
        else:
            module = None
            class_name = None

        module = getattr(self, f"{name}_module", module)
        class_name = getattr(self, f"{name}_class", class_name)
        class_kwargs = getattr(self, f"{name}_kwargs", {})

        if module is None or class_name is None:
            raise ValueError(
                f"Cannot find module {name}_module "
                f"or class {name}_class in parameters."
            )

        # if pars of calling class had any of these parameters
        # pass them to the pars of the class being loaded
        # unless they're already defined in the class_kwargs
        self.add_defaults_to_dict(class_kwargs)

        # any other arguments passed in from caller override
        class_kwargs.update(kwargs)

        return getattr(__import__(module, fromlist=[class_name]), class_name)(
            **class_kwargs
        )

    def add_defaults_to_dict(self, inputs):
        """
        Add some default keywords to the inputs dictionary.
        If these keys already exist, don't update them.
        This is useful to automatically propagate
        parameter values that need to be shared by
        sub-objects.
        """
        keys = ["project", "cfg_file", "verbose"]
        for k in keys:
            if hasattr(self, k) and k not in inputs:
                inputs[k] = getattr(self, k)

    def print(self):
        """
        Print the parameters.
        """
        pprint(self.__dict__)

    @staticmethod
    def from_dict(inputs, default_key=None):
        """
        Create a Parameters object from a dictionary.
        Will try to load a YAML file if given a project name,
        ("project" key in the dictionary) or if given a "cfg_file" key.
        If no

        Parameters
        ----------
        inputs: dict
            A dictionary with the parameters.

        default_key: str, optional
            The key to use when searching for a sub-dictionary
            in the config file. If not given, will load the entire
            YAML file.
            Will be overriden if user specifies a different cfg_key.
        """
        pars = Parameters()
        project = inputs.get("project", None)
        if project is not None:
            default_filename = os.path.join(
                os.path.dirname(__file__), "../configs", f"{project}.yaml"
            )
        else:
            default_filename = None
        filename = inputs.get("cfg_file", default_filename)
        if filename is not None:
            if not os.path.isabs(filename):
                basepath = os.path.dirname(__file__)
                filename = os.path.abspath(
                    os.path.join(basepath, "../configs", filename)
                )

            if os.path.isfile(filename):
                key = inputs.get("cfg_key", default_key)
                # print(f'Loading parameters from {filename} key "{key}"')
                pars.load(filename, key=key)
            elif "cfg_file" in inputs:
                # only raise if user explicitly specified a file
                raise FileNotFoundError(f"Could not find config file {filename}")

        # the user inputs override the config file
        pars.read(inputs)
        return pars
