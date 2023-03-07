from src.parameters import Parameters
from src.utils import help_with_class, help_with_object


class ParsSimulator(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        # TODO: add more parameters here:

        self._enforce_no_new_attrs = True

        self.load_then_update(kwargs)

    @classmethod
    def _get_default_cfg_key(cls):
        """
        Get the default key to use when loading a config file.
        """
        return "simulator"


class Simulator:
    """
    Produce fake events and inject them into real data.

    This is a basic simulator that adds some single-frame flares
    with variable brightness into otherwise real light curves.

    Other, more sophisticated simulators should be added as subclasses.

    """

    def __init__(self, **kwargs):
        self.pars = self._make_pars_object(kwargs)

    @staticmethod
    def _make_pars_object(kwargs):
        """
        Make the ParsSimulator object.
        When writing a subclass of this class
        that has its own subclassed Parameters,
        this function will allow the constructor
        of the superclass to instantiate the correct
        subclass Parameters object.
        """
        return ParsSimulator(**kwargs)

    def help(self=None, owner_pars=None):
        """
        Print the help for this object and objects contained in it.
        """
        if isinstance(self, Simulator):
            help_with_object(self, owner_pars)
        elif self is None or self == Simulator:
            help_with_class(Simulator, ParsSimulator)
