import os
import yaml

from src.database import DATA_ROOT

# A cached dictionary of dictionaries
# loaded form YAML files.
# Each time load() is called with the same
# filename but different key, it will not
# need to re-read the file from disk.
LOADED_FILES = {}


# TODO: use typing module to specify types and Annotatted for descriptions
# ref: https://stackoverflow.com/a/8820636/18256949


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

    def __init__(self):
        """
        Setup a Parameters object.
        After setting up, the parameters can be set
        either by hard-coded values or by a YAML file,
        using the load() method,
        or by a dictionary, using the read() method.

        When adding parameters, use the add_par() method,
        that accepts the variable name, the type(s),
        and a docstring describing the parameter.
        This allows for type checking and documentation.

        Subclasses of this class should add their own
        parameters then override the allow_adding_new_attributes()
        method to return False, to prevent the user from
        adding new parameters not defined in the subclass.
        """

        self.__typecheck__ = {}
        self.__defaultpars__ = {}
        self.__docstrings__ = {}
        self.verbose = self.add_par("verbose", 0, int, "Level of verbosity (0=quiet).")

        self._enforce_type_checks = self.add_par(
            "_enforce_type_checks",
            False,
            bool,
            "Choose if input values should be checked "
            "against the type defined in add_par().",
        )

        self._enforce_no_new_attrs = self.add_par(
            "_enforce_no_new_attrs",
            False,
            bool,
            "Choose if new attributes should be allowed "
            "to be added to the Parameters object. "
            "Set to True to lock the object from further changes. ",
        )

        self._default_cfg_key = self.add_par(
            "_default_cfg_key",
            None,
            str,
            "The default key to use when loading a YAML file.",
        )

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __setattr__(self, key, value):
        """
        Set an attribute of this object.
        There are some limitations on what can be set:
        1) if this class has allow_adding_new_attributes=False,
           no new attributes can be added by the user
           (to prevent setting parameters with typoes, etc).
        2) if self._enforce_type_checks=True, then the type of the
           value must match the types allowed by the add_par() method.

        """
        new_attrs_check = (
            hasattr(self, "_enforce_no_new_attrs") and self._enforce_no_new_attrs
        )

        if new_attrs_check and key not in self.__dict__:
            raise AttributeError(f'Attribute "{key}" does not exist.')

        type_checks = (
            hasattr(self, "_enforce_type_checks") and self._enforce_type_checks
        )
        if (
            type_checks
            and key in self.__typecheck__
            and not isinstance(value, self.__typecheck__[key])
        ):
            raise TypeError(
                f'Parameter "{key}" must be of type {self.__typecheck__[key]}'
            )
        super().__setattr__(key, value)

    def add_par(self, name, default, par_types, docstring):
        """
        Add a parameter to the list of allowed parameters.
        To add a value in one line (in the __init__ method):
        self.new_var = self.add_par('new_var', (bool, NoneType), False, "Description of new_var.")

        Parameters
        ----------
        name: str
            Name of the parameter. Must match the name of the variable
            that the return value is assigned to, e.g., self.new_var.
        default: any
            The default value of the parameter. Must match the given par_types.
        par_types: type or tuple of types
            The type(s) of the parameter. Will be enforced using isinstance(),
            when _enforce_type_checks=True.
        docstring: str
            A description of the parameter. Will be used in the docstring of the class,
            and when using the print() method or get_par_string() method.

        Returns
        -------

        """
        if name in self.__typecheck__:
            raise ValueError(f"Parameter {name} already exists.")
        self.__typecheck__[name] = par_types
        self.__docstrings__[name] = docstring
        self.__defaultpars__[name] = default
        self[name] = default  # this should fail if wrong type?
        return default

    def get_par_string(self, name):
        """
        Get the value, docstring and default of a parameter.
        """

        desc = default = types = ""

        if name in self.__docstrings__:
            desc = self.__docstrings__[name].strip()
            if desc.endswith("."):
                desc = desc[:-1]
        if name in self.__defaultpars__:
            default = f"default= {self.__defaultpars__[name]}"
        if name in self.__typecheck__:
            types = f"type= {self.__typecheck__[name]}"
        joined = " | ".join([desc, default, types])
        s = f"= {self[name]} [{joined}]"
        return s

    def default_values(self, **kwargs):
        """
        Add the input values as attributes of this object,
        but for each attribute only add it if it has not been
        defined already.
        This is useful for hard-coding default values that may or
        may not have been loaded in a previous call to e.g., load().
        """
        for k, v in kwargs.items():
            if k not in self:
                self[k] = v
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
            if k not in self or k in self._default_keys:
                self[k] = v

    def load_then_update(self, inputs):
        """
        Check for filename and load it if found.
        Then update the parameters with the input values.

        Parameters
        ----------
        inputs: dict
            The input values to update the parameters with.

        Returns
        -------
        dict
            The combined config dictionary using the values
            from file (if loaded) and the input values
            that override any keys from file.
        """

        # check if need to load from disk
        (cfg_file, cfg_key) = self.extract_cfg_file_and_key(inputs)
        config = self.load(cfg_file, cfg_key, raise_if_missing="cfg_file" in inputs)

        # apply the input kwargs (override config file)
        config.update(inputs)

        for k, v in config.items():
            setattr(self, k, v)

        return config

    @staticmethod
    def get_file_from_disk(filename):
        if filename not in LOADED_FILES:
            with open(filename) as file:
                LOADED_FILES[filename] = yaml.safe_load(file)

        return LOADED_FILES[filename]

    @staticmethod
    def extract_cfg_file_and_key(inputs):
        """
        Scan a dictionary "inputs" for parameters
        that may point to a config filename.
        If "project" key is found, will use that name
        as the config filename.
        If "cfg_file" is found, will override "project"
        and use that instead.
        To prevent loading a config file, use cfg_file=False.

        Will also look for a "cfg_key" keyword to use to
        locate the key inside the config file.

        Parameters
        ----------
        inputs: dict
            Keyword arguments to scan for a filename to use.

        Returns
        -------
        filename: str
            The filename to use, or None if no filename was found.
        cfg_key: str
            The key to use inside the config file, or None if no key was found.
        """
        filename = inputs.get("cfg_file", inputs.get("project", None))
        cfg_key = inputs.get("cfg_key", None)
        return filename, cfg_key

    def load(self, filename, key=None, raise_if_missing=True):
        """
        Read parameters from a YAML file.
        If any parameters were already defined,
        they will be overridden by the values in the file.

        Parameters
        ----------
        filename: str or bool
            Full or relative path and name to the YAML file.
            If given as False, will not load anything.
        key: str or None
            Read only a specific key from the YAML file,
            and use only the keys under that to populate
            the parameters.
            If None, will try to load the default key
            using the subclass get_default_cfg_key() method.
        raise_if_missing: bool
            If True, will raise an error if the key is not found (default).
            If False, will quietly return an empty dictionary.

        Returns
        -------
        dict
            The dictionary that was loaded from the file.
        """
        if filename is False:
            return {}  # asked explicitly to not load anything
        try:

            if os.path.isabs(filename):
                filepath = filename
            else:
                basepath = os.path.dirname(__file__)
                filepath = os.path.abspath(
                    os.path.join(basepath, "../configs", filename)
                )

            config = self.get_file_from_disk(filepath)

            # subclasses may define a default key
            if key is None:
                key = self._default_cfg_key

            if key is not None:
                config = config.get(key, {})

            return config

        except FileNotFoundError:
            if raise_if_missing:
                raise
            else:
                return {}
        # ignore all other loading exceptions?

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
            self[k] = v

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
            if k in self:  # need to update
                if isinstance(self[k], set) and isinstance(v, (set, list)):
                    self[k].update(v)
                elif isinstance(self[k], dict) and isinstance(v, dict):
                    self[k].update(v)
                else:
                    self[k] = v
            else:  # just add this parameter
                self[k] = v

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
            outputs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
            yaml.dump(outputs, file, default_flow_style=False)

    def get_data_path(self):
        """
        Get the path to the data directory.
        """
        if "data_folder" in self:
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
            if k in self and k not in inputs:
                inputs[k] = self[k]

    def print(self):
        """
        Print the parameters.
        """
        names = []
        desc = []
        for name in self.__dict__:
            if name.startswith("_"):
                continue
            desc.append(self.get_par_string(name))
            names.append(name)

        max_length = max(len(n) for n in names)
        for n, d in zip(names, desc):
            print(f"{n:>{max_length}}{d}")

    @classmethod
    def from_dict(cls, inputs, default_key=None):
        """
        TO BE DEPRECATED
        Create a Parameters object from a dictionary.
        Will try to load a YAML file if given a project name,
        ("project" key in the dictionary) or if given a "cfg_file" key.
        If no config file is found, can either silently continue with
        only the given inputs, or raise an error
        if cfg_file is given explicitly.

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
        if default_key is None:
            default_key = cls.get_default_cfg_key()

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


if __name__ == "__main__":
    p = Parameters()
