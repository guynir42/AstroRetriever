import yaml


class Parameters:
    """
    Keep track of parameters for any of the other classes.
    The list of required parameters has to be defined either
    by hard-coded values or by a YAML file.
    Use verify() to make sure all parameters are set to some
    value (including None!).

    """

    def __init__(self):
        self.required_parameters = []

    def verify(self):
        for p in self.required_parameters:
            if not hasattr(self, p):
                raise ValueError(f"Parameter {p} is not set.")

    def load(self, filename):
        config = yaml.safe_load(filename)
        for k, v in config.items():
            setattr(self, k, v)

    def save(self, filename):
        with open(filename, "w") as file:
            yaml.dump(self.__dict__, file, default_flow_style=False)
