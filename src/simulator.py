from src.parameters import Parameters
from src.utils import help_with_class, help_with_object


class ParsSimulator(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        # TODO: add more parameters here:

        self._enforce_no_new_attrs = True

        self.load_then_update(kwargs)

    @classmethod
    def get_default_cfg_key(cls):
        """
        Get the default key to use when loading a config file.
        """
        return "simulator"


class Simulator:
    def help(self=None, owner_pars=None):
        """
        Print the help for this object and objects contained in it.
        """
        if isinstance(self, Simulator):
            help_with_object(self, owner_pars)
        elif self is None or self == Simulator:
            help_with_class(Simulator, ParsSimulator)

    def __init__(self, **kwargs):
        self.pars = ParsSimulator(**kwargs)
