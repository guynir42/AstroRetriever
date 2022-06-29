import os
import yaml
import numpy as np

from astropy.io import fits

from catalog import Catalog
from calibration import Calibration
from histograms import Histograms
from analysis import Analysis


class VirtualObservatory:
    """
    Base class for other virtual observatories.
    This class allows the user to load a catalog,
    and for each object in it, download data from
    some real observatory, run analysis on it,
    save the results for later, and so on.
    """

    def __init__(self):
        # project name should be different for each science case
        # it will determine the name of all saved files
        # and in general objects should not be mixed between projects
        self.project_name = "default"
        self.params = {}  # parameters for the analysis
        self.catalog = None  # a Catalog object
        self.calibration = None  # a Calibration object
        self.analysis = None  # an Analysis object
        self.histograms = None  # a Histograms object
        self.database = None  # connection to database with objects
        self._passwords = None  # dictionary with passwords

    def load_passwords(self, filename=None):
        if filename is None:
            filename = "passwords.yaml"
        self._passwords = yaml.safe_load(filename)

    def load_catalog(self, filename=None):
        if filename is None:
            # TODO: cite paper and explain how to download this file
            if self.project_name == "WD":
                filename = "GaiaEDR3_WD_main.fits"
            else:
                filename = self.project_name + "_catalog.fits"

        path = "catalogs"
        self.catalog = Catalog(os.join(path, filename))

        # with fits.open(os.path.join(path, filename)) as hdul:
        #     # read the table headers
        #     names = {}
        #     units = {}
        #     comments = {}
        #     # TODO: get this number from the header
        #     for i in range(161):
        #         names[i] = hdul[1].header.get(f'TTYPE{i + 1}')
        #         units[i] = hdul[1].header.get(f'TUNIT{i + 1}')
        #         comments[i] = hdul[1].header.get(f'TCOMM{i + 1}')
        #
        #     self.catalog = np.array(hdul[1].data)

    def load_analysis(self, filename=None):
        if filename is None:
            filename = self.project_name + "_analysis.yaml"

        path = "analysis_configs"

        self.analysis = Analysis.load(os.join(path, filename))

    def save_analysis(self):
        pass

    def load_histograms(self):
        pass

    def reset_histograms(self):
        pass

    def save_histograms(self):
        pass

    def run_analysis(self):

        if self.calibration is None:
            raise ValueError("No calibration object loaded")
        if not isinstance(self.calibration, Calibration):
            raise ValueError("calibration object is not an instance of Calibration")

        if self.analysis is None:
            raise ValueError("No analysis object loaded")
        if not isinstance(self.analysis, Analysis):
            raise ValueError("analysis object is not an instance of Analysis")

        if self.histograms is None:
            raise ValueError("No histograms object loaded")
        if not isinstance(self.analysis, Analysis):
            raise ValueError("histograms object is not an instance of Histograms")

        self.analysis.load_simulator()
        self.reset_histograms()

        for source in self.catalog:
            self.calibration.apply_calibration(source)
            self.analysis.run(source, histograms=self.histograms)

    def download(self):
        raise NotImplementedError("download() must be implemented in subclass")

    def save_source(self, ra, dec, data={}):
        pass


class VirtualDemoObseratory(VirtualObservatory):
    pass
