import xarray as xr

from src.parameters import Parameters


class Histograms:
    def __init__(self):
        self.pars = Parameters(required_pars=[])
