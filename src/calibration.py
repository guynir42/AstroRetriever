import numpy as np

from src.parameters import Parameters


class Calibration:
    def __init__(self):
        self.pars = Parameters(required_pars=["type"])
        self.pars.type = "photometry"

    def initialize(self):
        pass
