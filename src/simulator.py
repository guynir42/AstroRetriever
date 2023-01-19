from src.parameters import Parameters
from src.utils import help_with_class


class ParsSimulator(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        # TODO: add more parameters here:

        self._enforce_no_new_attrs = True

        self.load_then_update(kwargs)


class Simulator:
    @classmethod
    def help(cls):
        """
        Print the help for this object and objects contained in it.
        """
        help_with_class(cls, ParsSimulator)

    def __init__(self, **kwargs):
        pass
