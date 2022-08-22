import xarray as xr

from src.parameters import Parameters


class Histogram:
    """
    A wrapper around an xarray.Dataset to store histograms.
    This object can also load and save the underlying data into a netCDF file.

    The reason to keep track of this data is to be able to set
    thresholds for detection, based on the background distribution.
    Another reason is to know the amount of time or number of observations
    that did not have a detection, which can be used to calculate
    the upper limit (in case of no detections) or to calculate the
    physical rates (in case of detections).

    Histograms can also be added with each other (using the + operator).
    This simply adds together the counts in each bin,
    and expands the bins if they have different limits.
    """

    def __init__(self):
        self.pars = Parameters(
            required_pars=[
                "dtype",
            ]
        )
        self.data = None

    def verify_parameters(self):
        self.pars.verify()
        if self.pars.dtype not in ("uint16", "uint32"):
            raise ValueError(
                f"Unsupported dtype: {self.pars.dtype}, " f"must be uint16 or uint32."
            )
