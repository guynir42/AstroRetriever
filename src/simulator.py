import time
import numpy as np
import xarray as xr
import scipy

from src.parameters import Parameters
from src.utils import help_with_class, help_with_object


class ParsSimulator(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.dimensionality = self.add_par(
            "dimensionality",
            "single",
            str,
            'Dimensionality of the simulated flares. Can be "single" or "multi". ',
        )

        self.selection_method = self.add_par(
            "selection_method",
            "overlap",
            str,
            'Method of selecting the simulated flares. Can be "overlap" or "grid". ',
        )

        self.overlap_required = self.add_par(
            "overlap_required",
            0.95,
            float,
            "How much overlap is needed between kernels. ",
        )

        self.width_grid = self.add_par(
            "width_grid",
            [0.1, 0.2, 0.3],
            list,
            "Grid of widths to use for the simulated flares. ",
        )

        self.min_width = self.add_par(
            "min_width",
            0.01,
            float,
            "Minimum width of the simulated flares. ",
        )

        self.max_width = self.add_par(
            "max_width",
            30,
            float,
            "Maximum width of the simulated flares. ",
        )

        self.time_units = self.add_par(
            "time_units",
            "days",
            str,
            "Units of the width of the simulated flares. ",
        )

        self.cadence = self.add_par(
            "cadence",
            20 / 3600 / 24,
            float,
            "Cadence of observations to match the templates to. " "The default is the TESS fast cadence. ",
        )

        self.template_shape = self.add_par(
            "template_shape",
            "gaussian",
            str,
            "Shape of the simulated flares, when using single-dimensional templates. " 'Can be "gaussian" or "box". ',
        )

        # TODO: add more parameters here:
        #  e.g., pars to make a multi-dimensional template bank

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

        self.bank = None  # can be an xarray dataset with templates

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

    def make_template_bank(self):
        """
        Make a template bank of flares/occultations.
        """
        t0 = time.time()

        taxis = self._make_time_axis()

        if self.pars.dimensionality == "single":
            templates = []
            widths = []
            norms = []
            if self.pars.selection_method == "overlap":
                arbitrary_upper_limit = 10000
                log_base = 1.01  # use 1% logarithmic increments
                width = self.pars.min_width

                # this loop counts the number of successful templates added
                for i in range(0, arbitrary_upper_limit):
                    if width > self.pars.max_width:
                        break  # don't make any more templates!

                    for j in range(0, arbitrary_upper_limit):
                        width = width * log_base**j  # logarithmic jumps
                        if width > self.pars.max_width:  # don't make any more templates!
                            break

                        test_template = self._make_single_template(taxis, width)
                        test_normalization = np.sqrt(np.sum(test_template**2))
                        # check this new template only against the last one, as they are ordered
                        if len(templates) == 0:
                            overlap = 0
                        else:
                            overlap = self._check_template_overlap(test_template, templates[-1])

                        if overlap < self.pars.overlap_required:  # need to add this template!
                            templates.append(test_template)
                            widths.append(width)
                            norms.append(test_normalization)

                            self.pars.vprint(
                                f"Added template {i} with "
                                f"width {width:.3f} {self.pars.time_units}. "
                                f"Used {j} trials. Total runtime: {time.time() - t0:.2f}s. "
                            )

                            break

            elif self.pars.selection_method == "grid":
                widths = self.pars.width_grid
                for width in widths:
                    templates.append(self._make_single_template(taxis, width))
                    norms.append(np.sqrt(np.sum(templates[-1] ** 2)))

                    self.pars.vprint(
                        f"Template with width {width:.3f} {self.pars.time_units} added. "
                        f"Total runtime: {time.time() - t0:.2f}s. "
                    )

            else:
                raise ValueError(f'Unknown selection method: {self.pars.selection_method}. Use "overlap" or "grid".')

        elif self.pars.dimensionality == "multi":
            pass  # TODO: implement this
        else:
            raise ValueError(f'Unknown dimensionality: {self.pars.dimensionality}. Use "single" or "multi".')

        # TODO: what about multi-dimensional templates?
        self.bank = xr.Dataset(
            data_vars={
                "templates": (("width", "time"), np.array(templates)),
                "normalization": (("width",), np.array(norms)),
            },
            coords={
                "time": taxis,
                "width": np.array(widths),
            },
        )

        self.bank.time.attrs["units"] = self.pars.time_units
        self.bank.time.attrs["description"] = "Time axis for the templates"
        self.bank.time.attrs["long_name"] = "Time"
        self.bank.width.attrs["units"] = self.pars.time_units
        self.bank.width.attrs["description"] = "Width of the templates"
        self.bank.width.attrs["long_name"] = "Full Width at Half Maximum"

        self.pars.vprint(f"Done making {len(self.bank.width)} templates. Total runtime: {time.time() - t0:.2f}s.")

    def _make_time_axis(self):
        """
        Make the time axis for the template bank.
        """
        time_range = self.pars.max_width * 2
        taxis = np.arange(-time_range, time_range, self.pars.cadence)
        return taxis

    def _make_single_template(self, taxis, width):
        """
        Make a single template.
        """
        if self.pars.template_shape == "gaussian":
            width /= np.sqrt(2 * np.log(2))  # convert from FWHM to sigma
            template = np.exp(-(taxis**2) / (2 * width**2)) / np.sqrt(2 * np.pi * width**2)
        elif self.pars.template_shape == "box":
            template = np.zeros_like(taxis)
            template[np.abs(taxis) < width / 2] = 1
        else:
            raise ValueError(f'Unknown template shape: {self.pars.template_shape}. Use "gaussian" or "box".')
        return template

    def help(self=None, owner_pars=None):
        """
        Print the help for this object and objects contained in it.
        """
        if isinstance(self, Simulator):
            help_with_object(self, owner_pars)
        elif self is None or self == Simulator:
            help_with_class(Simulator, ParsSimulator)

    @staticmethod
    def _check_template_overlap(t1, t2, assume_centered=True):
        """
        Check the overlap between two templates.

        Parameters
        ----------
        t1 : np.ndarray
            The first template.
        t2 : np.ndarray
            The second template.
        assume_centered : bool
            If True, assume that the templates are centered on the time axis.
            If False, calculate the time shift that maximizes the overlap
            (using convolution). Default is True.
        """

        norm = np.sqrt(np.sum(t1**2) * np.sum(t2**2))
        if assume_centered:
            return np.sum(t1 * t2) / norm
        else:
            return np.max(scipy.signal.correlate(t1, t2, mode="valid")) / norm
            # scipy.signal.correlate (using FFT) on 500,000 point template
            # and a TESS lightcurve with 100,000 points takes 0.14s,
            # compared to 95s (!) using direct convolution.


if __name__ == "__main__":
    import sys

    sys.path.append("/home/guyn/Dropbox/python")  # need a better way to get this code!

    import sqlalchemy as sa

    from src.database import Session
    from src.dataset import Lightcurve

    s = Simulator(verbose=True)
    s.make_template_bank()

    with Session() as session:
        lcs = session.scalars(
            sa.select(Lightcurve).where(Lightcurve.number > 0).order_by(sa.desc(Lightcurve.created_at)).limit(10)
        ).all()
