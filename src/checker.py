import numpy as np
import pandas as pd

from src.parameters import Parameters


class ParsChecker(Parameters):
    def __init__(self, **kwargs):
        super().__init__()  # initialize base Parameters without passing arguments

        self.offset_threshold = self.add_par(
            "offset_threshold",
            3.5,
            (float, int),
            "Flag measurements with offsets (RA+Dec combined) "
            "that are larger than this many times the offset noise.",
        )

        self._enforce_type_checks = True
        self._enforce_no_new_attrs = True

        self.load_then_update(kwargs)

    @classmethod
    def get_default_cfg_key(cls):
        """
        Get the default key to use when loading a config file.
        """
        return "checker"


class Checker:
    def __init__(self, **kwargs):
        self.pars = ParsChecker(**kwargs)

    def check(self, lightcurves, source, sim=None):
        """
        TODO: finish this docstring and function!

        Parameters
        ----------
        lightcurves: list of Lightcurve objects
            The lightcurves to check.
            Each lightcurve will be added some
            new columns (unless already there)
            that contain the values of data quality
            checks and a flag for all measurements
            that is False if they all pass the thresholds
            and True if any of them fail the check.
        source: Source object
            The source to check. Can use some source
            properties to calculate the quality checks,
            e.g., the source magnitude might affect
            some checks.
        sim: dict or None
            If not None, the dictionary should contain
            the simulation parameters used to generate
            the lightcurves.
            If None, the lightcurves are assumed to be
            real data.
        """
        for lc in lightcurves:
            lc.data["qflag"] = False  # assume all measurements are good

            flag = lc.data[lc.colmap["flag"]]
            lc.data["qflag"] |= flag != 0  # flag bad measurements
            ra = lc.data[lc.colmap["ra"]].values
            dec = lc.data[lc.colmap["dec"]].values
            ra -= np.median(ra)
            # correct the RA for high declinations
            x = ra * np.cos(dec * np.pi / 180)

            dec -= np.median(dec)
            y = dec

            offset = np.sqrt(x**2 + y**2)
            offset_norm = (offset - np.mean(offset)) / np.std(offset)
            lc.data["offset"] = offset_norm
            lc.data["qflag"] |= np.abs(offset_norm) > self.pars.offset_threshold
