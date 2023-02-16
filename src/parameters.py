import os
import copy
import yaml

from src.database import DATA_ROOT

# A cached dictionary of dictionaries
# loaded form YAML files.
# Each time load() is called with the same
# filename but different key, it will not
# need to re-read the file from disk.
LOADED_FILES = {}


# TODO: use typing module to specify types and Annotated for descriptions
# ref: https://stackoverflow.com/a/8820636/18256949


# parameters that are propagated from one Parameters object
# to the next when creating subclasses.
# If the child Parameters doesn't have any one of these
# then that key is just skipped
propagated_keys = ["data_types", "project", "cfg_file", "verbose"]

# possible values for the data_types parameter
allowed_data_types = ["photometry", "spectra", "images"]


def convert_data_type(data_type):
    """
    Accept a string and return a string that is a valid data type.
    Convert aliases of data types to the canonical name.

    Parameters
    ----------
    data_type: str
        The data type to convert.
        Could be, e.g., "lightcurves" instead of "photometry".

    Returns
    -------
    data_type: str
        The canonical name of the data type.
        Will be one of the allowed_data_types.
    """
    if data_type.lower() in [
        "photometry",
        "phot",
        "lightcurve",
        "lightcurves",
        "lc",
        "lcs",
    ]:
        out_type = "photometry"
    elif data_type.lower() in ["spectra", "spec", "spectrum", "spectra", "sed", "seds"]:
        out_type = "spectra"
    elif data_type.lower() in ["images", "image", "im", "img", "imgs"]:
        out_type = "images"
    elif data_type.lower() in ["cutout", "cutouts", "thumbnail", "thumbnails"]:
        out_type = "cutouts"
    else:
        raise ValueError(
            f'Data type given "{data_type}" ' f"is not one of {allowed_data_types}"
        )
    return out_type


def normalize_data_types(data_types):
    """
    Go over a single string or list of strings
    and check that they conform to one of the
    allowed data types (or their aliases).
    Raises a ValueError if any of the data
    types is not allowed.

    Parameters
    ----------
    data_types: str or list of str
        The data types to check.
    """
    if isinstance(data_types, str):
        return [convert_data_type(data_types)]
    return sorted(convert_data_type(dt) for dt in data_types)


def get_class_from_data_type(data_type, level="raw"):
    from src.dataset import RawPhotometry, Lightcurve

    if data_type == "photometry":
        if level == "raw":
            return RawPhotometry
        elif level in ["reduced", "processed", "simulated"]:
            return Lightcurve
        else:
            raise ValueError(
                f"Unknown level {level}. "
                "Use 'raw', 'reduced', 'processed' or 'simulated'."
            )
    # elif data_type == "spectra":
    #     return RawSpectra
    # add more data types here
    else:
        raise ValueError(
            f'Data type given "{data_type}" is not one of {allowed_data_types}'
        )


class Parameters:
    """
    Keep track of parameters for any of the other classes.

    The list of required parameters has to be defined either
    by hard-coded values or by a YAML file.
    You can access the parameters as attributes, but also
    as dictionary items (using "in" and "pars[key]").

    Use load_then_update() to let the object try to read
    a config file, and after that apply any additional inputs
    (given as a dictionary from e.g., a kwargs).
    Note that the parameters given by the user (the kwargs or
    the "inputs" given to this function) will override those
    from the file.
    In subclasses, where the parameters are already hard-coded
    into the object, expect this to be called at the end of the
    __init__ method, where the kwargs is used as "inputs".
    So an object with a parameters sub-class will immediately
    load from file and then update with the kwargs.
    To avoid any attempt to load from file, use cfg_file=False.
    Using cfg_file=None will search for the default config file
    in the project folder.

    Methods
    -------
    - add_par() to add a new parameter (mostly in the __init__).
    - load() the parameters from a YAML file.
    - read() the parameters from a dictionary.
    - update() takes parameters from a dictionary and updates (instead of overriding dict/set parameters).
    - save() the parameters to a YAML file.
    - to_dict() converts the non-private parameters to a dictionary.
    - copy() makes a deep copy of this object.
    - compare() checks if the parameters of two objects are the same.
    - load_then_update() loads from file and then updates with a dictionary.
    - get_data_path() returns the path to the data folder.
    - get_class_instance() create a contained object using the keywords in this object.
    - add_defaults_to_dict() adds some attributes that should be shared to all pars.
    - print() shows the parameters and descriptions.

    Adding new parameters
    ---------------------
    To add new parameters directly on the object,
    or in the __init__ of a subclass, use the add_par() method.
    signature: add_par(par_name, default, types, description).
    This allows adding the type, the default, and the description
    of the parameter, which are very useful when printing out
    the details of the configuration.
    The parameters must be added this way to be saved to a config file.
    Any parameters that start with "_" are not saved to the config file,
    and are generally used only internally, not as user-facing parameters.

    If the values are hard-coded, e.g., in a subclass,
    then use the self._enforce_no_new_attrs = True at
    the end of the constructor. This will raise an error
    if the user tries to feed the wrong attribute to the
    pars object, either through a YAML file, through
    initialization kwargs, or by setting attributes
    directly.

    The types of the parameters are enforced, so you
    cannot set a parameter of the wrong type,
    unless you specify self._enforce_type_checks = False.

    Config Files
    ------------
    When reading a YAML file, the object will look for
    a key defined is _cfg_key and will also look for a sub-key,
    which is a specific keyword inside the main key.
    E.g., if _cfg_key="observatories" and _cfg_sub_key="ZTF",
    the object will load the parameters generally defined under
    the top-level key "observatories" and then will look for
    the sub-key "ZTF" inside the "observatories", and will
    apply the parameters there next (overwriting the top-level).

    The cfg (sub) key can be input manually, or as kwargs to
    subclasses of this object (see below). But if not given,
    they could get default values using the class method
    _get_default_cfg_key(). For sub-classes it is a good idea
    to override this function to give the usual key expected
    in the config file (e.g., "observatories").

    Sub classes
    -----------
    When sub-classing, make sure to call the super().__init__(),
    and then add all the specific parameters using add_par().
    Then finish by calling self._enforce_no_new_attrs = True.
    If you are sub-sub-classing, then make sure to set
    self._enforce_no_new_attrs = False before adding new
    parameters, and locking it back to True at the end.
    Make sure to override the _get_default_cfg_key()
    and optionally override the __setattr__() method
    if you need additional input verification/formatting.


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
        self.project = self.add_par("project", None, str, "Name of the project")
        self.cfg_file = self.add_par(
            "cfg_file",
            None,
            (None, str, bool),
            "Path to the YAML file with the parameters. "
            "If None, the default file named as the project will be used. "
            "If False, no file will be loaded.",
        )
        self.data_types = self.add_par(
            "data_types",
            "photometry",
            list,
            "Types of data to use (e.g., photometry, spectroscopy)",
        )

        self.verbose = self.add_par("verbose", 0, int, "Level of verbosity (0=quiet).")

        self._enforce_type_checks = self.add_par(
            "_enforce_type_checks",
            True,
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

        self._cfg_key = self.add_par(
            "_cfg_key",
            None,
            (None, str),
            "The key to use when loading the parameters from a YAML file. "
            "This is also the key that will be used when writing the parameters "
            "to the output config file. ",
        )
        self._cfg_sub_key = self.add_par(
            "_cfg_sub_key",
            None,
            (None, str),
            "The sub-key to use when loading the parameters from a YAML file. "
            "E.g., the observatory name under observatories. ",
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

        if new_attrs_check and key not in self.__dict__ and key not in propagated_keys:
            raise AttributeError(f'Attribute "{key}" does not exist.')

        if key == "data_types":
            value = normalize_data_types(value)

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
            and when using the print() method or _get_par_string() method.

        Returns
        -------

        """
        if name in self.__typecheck__:
            raise ValueError(f"Parameter {name} already exists.")
        if not isinstance(par_types, tuple):
            par_types = (par_types,)
        par_types = tuple(type(pt) if pt is None else pt for pt in par_types)
        if float in par_types:
            par_types += (int,)
        self.__typecheck__[name] = par_types
        self.__docstrings__[name] = docstring
        self.__defaultpars__[name] = default
        self[name] = default  # this should fail if wrong type?
        return default

    def load_then_update(self, inputs):
        """
        Check filename (if given) and load it if found.
        Then, update the parameters with the inputs dict.

        Parameters
        ----------
        inputs: dict
            The input values to update the parameters with.

        Returns
        -------
        dict
            The combined config dictionary using the values
            from file (if loaded) and the input values
            that override any keys from the file.
        """

        # check if need to load from disk
        (cfg_file, cfg_key, explicit) = self._extract_cfg_file_and_key(inputs)

        config = self.load(cfg_file, cfg_key, raise_if_missing=explicit)
        self._cfg_key = cfg_key
        # apply the input kwargs (override config file)
        if "demo_boolean" in config:
            print(f'config["demo_boolean"] = {config["demo_boolean"]}')
        config.update(inputs)

        # if there's a way to set up a default configuration
        if "default" in config and hasattr(self, "setup_from_defaults"):
            self.setup_from_defaults(config["default"])

        # only after loading the default configuration, update with data from file/user
        for k, v in config.items():
            setattr(self, k, v)

        return config

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
        raise_if_missing: bool
            If True, will raise an error if the key is not found (default).
            If False, will quietly return an empty dictionary.

        Returns
        -------
        dict
            The dictionary that was loaded from the file.
        """
        if filename is None or filename is False:
            return {}  # asked explicitly to not load anything
        try:

            if os.path.isabs(filename):
                filepath = filename
            else:
                basepath = os.path.dirname(__file__)
                filepath = os.path.abspath(
                    os.path.join(basepath, "../configs", filename)
                )

            if not filepath.lower().endswith(("yml", "yaml", "cfg")):
                filepath += ".yaml"

            # print(f'Loading config from "{filepath}" with key "{key}"')
            config = self._get_file_from_disk(filepath)

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

    def to_dict(self, hidden=False):
        """
        Convert parameters to a dictionary.
        Only get the parameters that were defined
        using the add_par method.

        Parameters
        ----------
        hidden: bool
            If True, include hidden parameters.
            By default, does not include hidden parameters.

        Returns
        -------
        output: dict
            A dictionary with the parameters.
        """
        output = {}
        for k in self.__defaultpars__.keys():
            if hidden or not k.startswith("_"):
                output[k] = self[k]

        return output

    def copy(self):
        """
        Create a copy of the parameters.
        """
        return copy.deepcopy(self)

    def get_data_path(self):
        """
        Get the path to the data directory.
        """
        if "data_folder" in self:
            return os.path.join(DATA_ROOT or "", self.data_folder)
        else:
            return DATA_ROOT or ""

    def get_class_instance(self, name, **kwargs):
        """
        Get a class from a string.
        To load a class, there must be (at least) two
        definitions in the parameter object:
        - <name>_module: the import path to the file containing
          the class definition. E.g., src.my_simulator
        - <name>_class: the name of the class. E.g., MySimulator.
        - <name>_kwargs: a dictionary with initialization arguments.
          The kwargs given as inputs to this function
          override those loaded from file.

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
            Additional keyword arguments to pass to the class,
            overriding those in <name>_kwargs.

        Returns
        -------
        class
            The class object.
        """

        # default module and class_name for core classes:
        name = name.lower()
        if name == "analysis":
            module = "src.analysis"
            class_name = "Analysis"
        elif name == "simulator":
            module = "src.simulator"
            class_name = "Simulator"
        elif name == "catalog":
            module = "src.catalog"
            class_name = "Catalog"
        elif name == "finder":
            module = "src.finder"
            class_name = "Finder"
        elif name == "quality":
            module = "src.quality"
            class_name = "Quality"
        elif name == "histogram":
            module = "src.histogram"
            class_name = "Histogram"
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
        sub-objects (e.g., project name).
        """
        keys = propagated_keys
        for k in keys:
            if k in self and k not in inputs:
                inputs[k] = self[k]

    def print(self, owner_pars=None):
        """
        Print the parameters.

        If given an owner_pars input,
        will not print any of the default
        parameters if their values are the same
        in the owner_pars object.
        """
        if owner_pars is not None and not isinstance(owner_pars, Parameters):
            raise ValueError("owner_pars must be a Parameters object.")

        names = []
        desc = []
        defaults = []
        for name in self.__dict__:
            if name.startswith("_"):
                continue
            if owner_pars is not None:
                if name in propagated_keys and self[name] == owner_pars[name]:
                    defaults.append(name)
                    continue

            desc.append(self._get_par_string(name))
            names.append(name)

        if len(defaults) > 0:
            print(f" Propagated pars: {', '.join(defaults)}")
        if len(names) > 0:
            max_length = max(len(n) for n in names)
            for n, d in zip(names, desc):
                print(f" {n:>{max_length}}{d}")

    def compare(self, other, hidden=False, ignore=None, verbose=False):
        """
        Check that all parameters are the same between
        two Parameter objects. Will only check those parameters
        that were added using the add_par() method.
        By default, ignores hidden parameters even if they were
        added using add_par().

        Parameters
        ----------
        other: Parameters object
            The other Parameters object to compare to.
        hidden: bool
            If True, include hidden parameters.
            By default, does not include hidden parameters.
        verbose: bool
            If True, print the differences between the two
            Parameter objects.

        Returns
        -------
        same: bool
            True if all parameters are the same.

        """
        if ignore is None:
            ignore = []

        same = True
        for k in self.__defaultpars__.keys():
            if k in ignore:
                continue
            if hidden or not k.startswith("_") and self[k] != other[k]:
                same = False
                if not verbose:
                    break
                print(f'Par "{k}" is different: {self[k]} vs {other[k]}')

        return same

    @classmethod
    def _extract_cfg_file_and_key(cls, inputs):
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
        If that is not found, will try the key in
        the static _get_default_cfg_key() method.

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
            If None, will try to load the default key
            using the subclass get_default_cfg_key() method.
        explicit: bool
            True if the filename was explicitly given in the inputs
            as "cfg_file", not inferred from the project name.
        """
        filename = inputs.get("cfg_file", None)
        explicit = filename is not None

        if filename is None:
            filename = inputs.get("project", None)

        cfg_key = inputs.get("cfg_key", None)

        # subclasses may define a default key
        if cfg_key is None:
            cfg_key = cls._get_default_cfg_key()

        return filename, cfg_key, explicit

    def _get_par_string(self, name):
        """
        Get the value, docstring and default of a parameter.
        """

        desc = default = types = ""
        value = self[name]

        if name in self.__docstrings__:
            desc = self.__docstrings__[name].strip()
            if desc.endswith("."):
                desc = desc[:-1]
        if name in self.__defaultpars__:
            def_value = self.__defaultpars__[name]
            if def_value == value:
                default = "default"
            else:
                default = f"default= {def_value}"
        if name in self.__typecheck__:
            types = self.__typecheck__[name]
            if not isinstance(types, tuple):
                types = (types,)
            types = (t.__name__ for t in types)
            types = f'types= {", ".join(types)}'
        extra = ", ".join([s for s in (default, types) if s])
        if extra:
            extra = f" [{extra}]"

        if isinstance(value, str):
            value = f'"{value}"'
        s = f"= {value} % {desc}{extra}"

        return s

    @staticmethod
    def _get_file_from_disk(filename):
        """
        Lazy load the file from disk. If already in the LOADED_FILES dict, will get that instead.
        """
        if filename not in LOADED_FILES:
            with open(filename) as file:
                LOADED_FILES[filename] = yaml.safe_load(file)

        return LOADED_FILES[filename]

    @classmethod
    def _get_default_cfg_key(cls):
        """
        Get the default key to use when loading a config file.
        """
        return None


if __name__ == "__main__":
    p = Parameters()
