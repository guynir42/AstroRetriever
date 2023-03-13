import os
import json

import numpy as np
import pandas as pd
import xarray as xr

from src.database import safe_mkdir
from src.parameters import Parameters
from src.source import Source
from src.utils import help_with_class, help_with_object, unit_convert_bytes, is_scalar


# TODO: should this be saved to the database?


class ParsHistogram(Parameters):
    """
    A histogram object's parameters are saved in this class.
    Most of the parameters of the histograms have to do with
    defining the coordinates for the DataArrays.
    These are given as coordinate specs:
    either a 3-tuple for a statis axis or
    an empty tuple or 2-tuple for a dynamic axis.
    A static axis is defined as (start, stop, step),
    using np.arange(start, stop + step, step).
    A dynamic axis is defined as (start, step) or ()
    in case of a string based coordinate.
    If given a start and step, it will add bins with the
    correct width (step) until it can accommodate new data.
    An empty tuple will create an axis without a defined step.
    This only works if the coordinate is string-based
    (e.g., filter names).

    The three types of coordinates are score, source, and obs.
    The score coordinates are special in that they define
    one DataArray per coordinate (per score).
    The source and obs are common for all DataArrays.

    In addition, the dtype parameter controls the underlying
    data type of the histogram arrays.
    Since this is a histogram, the data type
    must be unsigned integers. Choose uint16 for smaller
    data sets (in RAM and on disk) when expecting the data
    to be sparse, or uint32 for larger data sets but without
    the risk of overflow.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.dtype = self.add_par(
            "dtype",
            "uint32",
            str,
            "Data type of underlying array (must be uint16 or uint32)",
        )
        default_score_coords = {
            "snr": (-20, 20, 0.1),
            "dmag": (-3, 3, 0.1),
            "offset": (-5, 5, 0.1),
        }  # other options: any of the quality cuts
        self.score_coords = self.add_par(
            "score_coords",
            default_score_coords,
            dict,
            "Coordinate specs for the score axes",
        )
        self.score_names = self.add_par(
            "score_names",
            None,
            (list, None),
            "Use only these scores (if None, use all)",
        )

        default_source_coords = {
            "mag": (15, 21, 0.5),
        }  # other options: color, ecl lat, mass, radius
        self.source_coords = self.add_par(
            "source_coords",
            default_source_coords,
            dict,
            "Coordinate specs for the source axes",
        )
        # TODO: we could consider adding a source_names parameter

        default_obs_coords = {
            "exptime": (30, 1),
            "filter": (),
        }  # other options: airmass, zp, magerr
        self.obs_coords = self.add_par(
            "obs_coords",
            default_obs_coords,
            dict,
            "Coordinate specs for the observation axes",
        )
        # TODO: we could consider adding an obs_names parameter

        self.raise_on_repeat_source = self.add_par(
            "raise_on_repeat_source",
            False,
            bool,
            "Raise an error if a source name already "
            "exists in the histogram source_names set.",
        )

        self._enforce_no_new_attrs = True

        self.load_then_update(kwargs)

    def __setattr__(self, key, value):
        if key == "dtype" and value not in ("uint16", "uint32"):
            raise ValueError(
                f"Unsupported dtype: {value}, " f"must be uint16 or uint32."
            )
        if key in ["score_coords", "source_coords", "obs_coords"]:
            value = {k: list(v) for k, v in value.items()}

        super().__setattr__(key, value)

    @classmethod
    def _get_default_cfg_key(cls):
        """
        Get the default key to use when loading a config file.
        """
        return "histograms"


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

    Coordinates: the values are saved as an xarray Dataset,
    where the dataset values are always the counts of how many
    epochs/data points fall into each bin.
    The bins are along coordinates in either
    score, source, or observation (obs) space.
    The score coordinates are for the measured values,
    like signal-to-noise ratio (snr), or delta-magnitude (dmag).
    The source coordinates depend on the source,
    and are given by the catalog row for that source (e.g., magnitude).
    The observation coordinates are the values given by
    the specific observation (e.g., exposure time, filter).

    Choose coordinates with names that match the catalog values
    (for source coordinates), or the columns of the lightcurve dataframe
    for the observation coordinates.

    Data quality cuts: The score coordinates can be used also
    to track the results for various quality cuts on the data.
    Make sure the names of the score_coords are the same as the
    names of the columns added by the quality cuts.

    DataArrays: the datasets should include one DataArray for each
    of the score coordinates, sharing all the source- and obs-coordinates.

    """

    def __init__(self, **kwargs):
        """
        Constructor for a histogram object.
        Can contain multiple DataArrays for different score coordinates.
        If the default parameters are ok, or if giving updated parameters
        through the kwargs to this function, then provide initialiaze=True.
        Otherwise, the initialize function must be called explicitly,
        after adjusting the parameters.

        Parameters
        ----------
        name: str
            Name of the histogram object, e.g., all_score, quality_cuts.
            This is used to name the output netCDF file (histograms_<name>.nc).
        output_folder: str
            The path (relative or absolute) to the folder where the output
            netCDF file will be saved. Should be the same place that the
            project saves all files.
        initialize: bool
            If True, then initialize the histogram right after passing
            the kwargs to the parameters object. Otherwise, the
            initialize function must be called explicitly.
        kwargs: additional arguments are passed into the parameters object.
        """

        can_initialize = kwargs.pop("initialize", False)
        self.name = kwargs.pop("name", None)
        self.output_folder = kwargs.pop("output_folder", None)

        self.pars = self._make_pars_object(kwargs)
        self.data = None
        self.source_names = set()

        if can_initialize:
            self.initialize()

    def initialize(self):
        """
        Build the histogram data structure,
        including coordinates and DataArrays.
        This can potentially take up a lot of RAM
        and could be time consuming.
        Therefore, it is not called automatically when
        making this object, only when the initialize=True
        flag is given to the constructor.
        Otherwise, the user gets an opportunity to adjust
        the parameters, check the array size, and only
        then explicitly call this function.
        """

        if self.pars.dtype not in ("uint16", "uint32"):
            raise ValueError(
                f"Unsupported dtype: {self.pars.dtype}, must be uint16 or uint32."
            )

        # create the coordinates
        coords = {}
        for input_ in ("score", "source", "obs"):
            for k, v in getattr(self.pars, f"{input_}_coords").items():
                # if we have a names override, only use those names
                names = getattr(self.pars, f"{input_}_names", None)
                if names is not None and k not in names:
                    continue

                coords[k] = self._create_coordinate(k, v)
                coords[k].attrs["input"] = input_

        data_shape = tuple(
            len(v) for v in coords.values() if v.attrs["input"] in ("source", "obs")
        )

        # these coordinates are shared by all the DataArrays
        common_coord_names = [
            k for k, v in coords.items() if v.attrs["input"] in ("source", "obs")
        ]

        data_vars = {}
        for k, v in coords.items():
            if v.attrs["input"] == "score":
                data_vars[k + "_counts"] = (
                    common_coord_names + [k],
                    np.zeros(data_shape + (len(v),), dtype=self.pars.dtype),
                )

        self.data = xr.Dataset(data_vars, coords=coords)
        if self.name is not None:
            self.data.attrs["name"] = self.name

        self.data.attrs["source_names"] = []
        self.source_names = set()

    @staticmethod
    def _make_pars_object(kwargs):
        """
        Make the ParsHistogram object.
        When writing a subclass of this class
        that has its own subclassed Parameters,
        this function will allow the constructor
        of the superclass to instantiate the correct
        subclass Parameters object.
        """
        return ParsHistogram(**kwargs)

    def _create_coordinate(self, name, specs):
        """
        Create a coordinate axis with a preset range
        or make a dynamic axis.
        This axis will be used in defining the xarray
        for the histogram.

        Parameters
        ----------
        name: str
            The name of the coordinate.
        specs:
            A tuple/list of (start, stop, step) for a fixed range,
            or an empty tuple/list or a 2-tuple/list (start, step) for a dynamic range.
            An extra last element can be given as a string
            to specify the units of the coordinate.
            In that case the tuple/list will be (start, stop, step, units)
            or (start, step, units) or (units).

        Returns
        -------
        xarray.DataArray
            The coordinate axis.
        """

        if not isinstance(specs, (tuple, list)):
            raise ValueError(f"Coordinate specs must be a list or tuple: {specs}")

        if len(specs) and isinstance(specs[-1], str):
            units = specs[-1]
            specs = specs[:-1]
        else:
            units = ""

        if len(specs) == 0:
            # dynamic range
            ax = xr.DataArray([], dims=[name])
            ax.attrs["type"] = "dynamic"
        elif len(specs) == 2:
            # dynamic range with a fixed step
            ax = xr.DataArray(np.array([specs[0]]), dims=[name])
            ax.attrs["step"] = specs[-1]
            ax.attrs["type"] = "dynamic"
        elif len(specs) == 3:
            # fixed range
            start, stop, step = specs
            ax = xr.DataArray(
                np.arange(start, stop + step, step),
                dims=[name],
            )
            ax.attrs["step"] = specs[-1]
            ax.attrs["type"] = "fixed"
            ax.attrs["overflow"] = 0
            ax.attrs["underflow"] = 0
        else:
            raise ValueError(
                f"Coordinate specs must be a tuple of length 0 or 3: {specs}"
            )

        ax.attrs["long_name"] = self._get_coordinate_name(name)
        ax.attrs["units"] = units

        return ax

    @staticmethod
    def _get_coordinate_name(name):
        """
        Get the long name of a coordinate.
        If the name is not recognized (hard coded),
        return the name itself.

        Parameters
        ----------
        name:
            Short name given to the coordinate,
            which also matches the column name in the catalog
            or in the lightcurve data.

        Returns
        -------
        str
            Long name of the coordinate
        """

        return {
            "mag": "Magnitude",
            "dmag": "Delta Magnitude",
            "snr": "Signal to Noise Ratio",
            "exptime": "Exposure Time",
            "filt": "Filter",
        }.get(name, name)

    def get_size(self, units="mb"):
        """
        Get the size of the histogram in memory.
        If some of the axes are empty (e.g., dynamic axes
        where data has not yet been added), the size would
        be zero. In that case, use get_size_estimate() instead.

        Parameters
        ----------
        units: str
            Can be 'bytes', 'kb', 'mb', or 'gb'.
            Default is "mb".

        Returns
        -------
        float
            The size of the histogram in memory,
            in whatever units were requested.
        """
        total_size = 0
        for d in self.data.data_vars.values():
            total_size += d.size * d.dtype.itemsize

        return total_size / unit_convert_bytes(units)

    def get_size_estimate(self, units="mb", dyn_coord_size=3, dyn_score_size=100):
        """
        Get an estimate for the size of the histogram in memory.
        If any of the (dynamic) axis are not yet filled, will use
        estimates for the final size of these axes when estimating
        memory footprint.
        The assumption is that dynamic coordinate axes (like filter)
        will have a small number of unique values (like 3), and that the
        dynamic score axes (like SNR) will have a large number of
        unique values (like 100).

        Parameters
        ----------
        units: str
            Can be 'bytes', 'kb', 'mb', or 'gb'.
            Default is "mb".
        dyn_coord_size: int
            The number of unique values to assume for dynamic coordinate axes.
            Default is 3, which is appropriate for having a few values.
        dyn_score_size: int
            The number of unique values to assume for dynamic score axes.
            Default is 100, which is appropriate for having a lot of values,
            spanning the dynamic range of possible scores.

        Returns
        -------
        float
            The size of the histogram in memory,
            in whatever units were requested.
        """

        common_size = 1
        for coord in self.data.coords.values():
            # only count non-score coordinates (shared by all DataArrays)
            if coord.attrs["input"] in ("source", "obs"):
                if len(coord) <= 1:
                    common_size *= dyn_coord_size
                else:
                    common_size *= len(coord)

        score_size = 0
        for coord in self.data.coords.values():
            # only count score coordinates
            if coord.attrs["input"] == "score":
                if len(coord) <= 1:
                    score_size += dyn_score_size
                else:
                    score_size += len(coord)

        total_size = common_size * score_size

        # get the number of bytes in one of the arrays
        array_names = list(self.data.keys())
        total_size *= self.data[array_names[0]].dtype.itemsize

        return total_size / unit_convert_bytes(units)

    def get_sum_scores(self):
        """
        Get the sum of all the scores in the histogram.

        Returns
        -------
        dict
            A dictionary with the sum of each score.
        """
        score = self.pars.score_names[0]
        sums = int(self.data[f"{score}_counts"].sum().values)
        for coord in self.data.coords:
            sums += int(self.data[coord].attrs.get("overflow", 0))
            sums += int(self.data[coord].attrs.get("underflow", 0))
            # TODO: what about a NaNs count?

        return sums

    def add_data(self, *args, **kwargs):
        """
        Input a dataframe, and other possible objects
        that contain information of the data,
        and store the data in the histogram.

        If any objects have scalar properties,
        or if the dataframe has a single row,
        or if some of the columns contain unique values,
        those scalar values will be used to slice into
        the histogram, and speed up the binning process.
        All the other values will be binned using
        np.histogramdd, with the assumption that each
        data point goes into the bin based on being
        closest to the center of the bin
        (i.e., we keep track of bin centers, not edges).

        Parameters
        ----------
        args:
            Dataframe or other objects containing data.
            The other objects can be, e.g., a Source object
            (that contains info on the magnitude and color)
            or a catalog row, altdata, etc.

        kwargs:
            Additional arguments to pass to the binning function.
            Optional keywords are:
            - source_name: add the source name to the list of sources
              that have already been added to the histogram. This is
              a shortcut for using add_source_name().
              Sources that have already been added would either be ignored
              in subsequent calls to add_data(), or would raise an exception
              if pars.raise_on_repeat_source is True.
        """
        # input data is a dictionary with a key for each axis,
        # and each value is a scalar or an array of values.
        # this gets ingested by the histogram DataArray
        # after it is filled from all objects given in args.
        input_data = {}

        # check if source is already in the histogram
        for arg in args:
            if isinstance(arg, Source):
                if arg.name in self.source_names:
                    if self.pars.raise_on_repeat_source:
                        raise ValueError(
                            f"Source {arg.name} already exists in histogram"
                        )
                    else:
                        return  # quietly exit without adding the data
                # this doesn't work because we often add multiple LCs from the same source:
                # else:
                #     self.source_names.add(arg.name)

        # go over args and see if any objects
        # have attributes that match one of the axes.
        for axis in self.data.dims:
            input_data[axis] = None
            for obj in args:
                values = None
                if hasattr(obj, "__contains__") and axis in obj:
                    values = obj[axis]
                elif hasattr(obj, axis) and not isinstance(obj, pd.DataFrame):
                    # only get this attribute if it isn't a DataFrame,
                    # Since some attributes on the DataFrame (e.g., "filter")
                    # are methods and not actual column names, in some cases
                    values = getattr(obj, axis)

                if values is not None:
                    if not is_scalar(values):
                        # an array, but need to check if all are the same
                        if len(np.unique(values)) == 1:
                            input_data[axis] = values[0]
                        else:
                            input_data[axis] = np.array(values)
                    else:  # a scalar value / string
                        input_data[axis] = values

                    break  # take the first thing that matches

        # check the length of the timeseries that is given
        sample_len = None
        for obj in args:
            if isinstance(obj, pd.DataFrame):
                sample_len = len(obj)
                break
        if sample_len is None:
            sample_len = 1

        # how many different scores are we tracking? (snr, dmag, etc)
        num_scores = len(self.data.data_vars)

        # check all data axes have a coordinate value or values array
        for axis in self.data.dims:
            if input_data[axis] is None:
                raise ValueError(f"Could not find data for axis {axis}")

        # check if any scalar values, or min/max of
        # the arrays of values, are outside the
        # range of any of the dynamic axes.
        # If so, expand the axes to include the new values.
        for axis in self.data.dims:
            if self.data.coords[axis].attrs["type"] == "fixed":
                continue  # no need to expand fixed axes

            # scalar string
            if isinstance(input_data[axis], str):
                if (
                    self.data.coords[axis].size == 0
                    or input_data[axis] not in self.data.coords[axis].values
                ):
                    self._expand_axis(axis, input_data[axis])

            # list of strings
            elif hasattr(input_data[axis], "__len__") and all(
                isinstance(x, str) for x in input_data[axis]
            ):
                if not set(input_data[axis]).issubset(self.data.coords[axis].values):
                    self._expand_axis(axis, input_data[axis])
            else:  # array of numbers
                mx = max(self.data[axis] + self.data[axis].attrs["step"] / 2)
                mn = min(self.data[axis] - self.data[axis].attrs["step"] / 2)
                if hasattr(input_data[axis], "__len__"):
                    new_mx = max(input_data[axis])
                    new_mn = min(input_data[axis])
                else:
                    new_mx = input_data[axis]
                    new_mn = input_data[axis]
                if new_mx > mx or new_mn < mn:
                    self._expand_axis(axis, input_data[axis])

        # here is where we actually increase the bin counts
        for name, da in self.data.data_vars.items():
            # each da is a DataArray for a different score

            # get a slice of the full array that matches any scalar values
            indices = {}
            array_values = {}
            for ax in da.dims:
                if is_scalar(input_data[ax]):
                    indices[ax] = self._get_index(ax, input_data[ax])
                else:
                    array_values[ax] = input_data[ax]

            # make sure all array values have the same length
            for v in array_values.values():
                if len(v) != sample_len:
                    raise ValueError("Array values must all have the same length!")

            # for all static axes, keep track of the
            # overflow/underflow counts
            in_range = True  # if any scalars are out of range, will be false
            for ax in da.dims:
                if da.coords[ax].attrs["type"] == "fixed":
                    if da.coords[ax].attrs["input"] == "score":
                        num_values_to_add = sample_len
                    else:
                        # because the common axes are shared by all scores
                        # we will end up adding the sample_len multiple times
                        num_values_to_add = sample_len / num_scores

                    if ax in indices and indices[ax] < 0:
                        da.coords[ax].attrs["underflow"] += num_values_to_add
                        in_range = False
                    elif ax in indices and indices[ax] >= len(da.coords[ax]):
                        da.coords[ax].attrs["overflow"] += num_values_to_add
                        in_range = False
                    # else: do nothing, there's no overflow/underflow

            if in_range:
                # if all the arrays have unique values, just add the number of measurements:
                if not array_values:  # empty dict
                    # TODO: check that this actually works...
                    self.data[name][indices] += np.array(sample_len).astype(da.dtype)
                else:
                    # bin the non-unique dataframe columns into the appropriate axes
                    da_slice = self.data[name].isel(**indices)
                    if set(da_slice.dims) != set(array_values.keys()):
                        raise ValueError("Slice into data array has wrong dimensions!")

                    # setup the bin edges and values
                    # make sure they're ordered by the da_slice dims
                    bins = []  # array of bin edges
                    values = []  # array of new values to count
                    for dim in da_slice.dims:
                        centers = da_slice.coords[dim].values

                        # convert strings to numbers:
                        if centers.dtype.kind in ("S", "U"):
                            # lookup table shows the order of strings in the coordinate
                            lookup = {val: ind for ind, val in enumerate(centers)}

                            # convert to numbers according
                            # to alphabetical order
                            uniq, rev_ind = np.unique(
                                array_values[dim], return_inverse=True
                            )

                            # convert alphabetical order to the order
                            # of values in the lookup table
                            # ref: https://stackoverflow.com/a/16993364/18256949
                            ordered_array = np.array([lookup[x] for x in uniq])[rev_ind]

                            # ordered_array should be a list of numbers,
                            # each representing a string, according to the
                            # order of strings in the axis.
                            values.append(ordered_array)
                            edges = np.arange(-0.5, len(centers) + 0.5, 1)
                        else:  # get bin edges from center values
                            step = da_slice.coords[dim].attrs["step"]
                            edges = np.concatenate(
                                [[centers[0] - step / 2], centers + step / 2]
                            )

                            values.append(array_values[dim])
                        bins.append(edges)

                    counts, _ = np.histogramdd(values, bins)
                    da_slice += counts.astype(da.dtype)  # add to existing counts

                    # add the over/underflow:
                    for ax in array_values:
                        if da_slice.coords[ax].attrs["type"] == "fixed":
                            if da_slice.coords[ax].attrs["input"] == "score":
                                correction = 1
                            else:
                                # because the common axes are shared by all scores
                                # we will end up adding the sample_len multiple times
                                correction = 1 / num_scores
                            mx = max(da_slice.coords[ax].values)
                            mn = min(da_slice.coords[ax].values)
                            step = da_slice.coords[ax].attrs["step"]
                            num_values_to_add = np.sum(array_values[ax] > mx + step / 2)

                            self.data[name][ax].attrs["overflow"] += (
                                num_values_to_add * correction
                            )
                            num_values_to_add = np.sum(array_values[ax] < mn - step / 2)
                            self.data[name][ax].attrs["underflow"] += (
                                num_values_to_add * correction
                            )

        if "source_name" in kwargs:
            self.add_source_name(kwargs["source_name"])

    def add_source_name(self, source_name):
        """
        Add a source name to the list of source names.

        Parameters
        ----------
        source_name : str
            Name of the source to add to the list of source names.
        """
        if source_name not in self.source_names:
            self.source_names.add(source_name)

    def _expand_axis(self, axis, new_values):
        """
        Expand the axis to include the new values.
        This only works for dynamic axes.
        String based axes are expanded simply by
        adding any values that are not included.
        Numeric axes are expanded by adding as many
        bins as needed (with the predefined step)
        so that the min/max of the new values
        are included in the new axis.

        Parameters
        ----------
        axis: str
            The name of the axis to expand.
            Will load the data coordinate DataArray.
        new_values: array-like
            The new values to include in the axis.
            This can be a scalar string or number,
            or a numeric array or a string array.

        """
        if isinstance(new_values, str):
            if self.data.coords[axis].size == 0:
                new_coord = np.array([new_values])
            elif new_values not in self.data.coords[axis].values:
                new_coord = np.concatenate(
                    [self.data.coords[axis].values, [new_values]]
                )
            # else: do nothing, the value exists in the axes
        else:  # scalar or array
            # str array/list
            if hasattr(new_values, "__len__") and all(
                isinstance(v, str) for v in new_values
            ):
                new_values = list(set(new_values) - set(self.data.coords[axis].values))

                if self.data.coords[axis].size > 0:
                    new_coord = np.concatenate(
                        (self.data.coords[axis].values, new_values)
                    )
                else:
                    new_coord = new_values
            else:  # scalar or array of numbers
                if not hasattr(new_values, "__len__"):
                    new_values = [new_values]
                step = self.data[axis].attrs["step"]
                mx = max(max(new_values), max(self.data[axis].values))
                mx = round(mx / step) * step  # round to nearest step
                mn = min(min(new_values), min(self.data[axis].values))
                mn = round(mn / step) * step  # round to nearest step
                # the new values up to the original axis
                lower = np.arange(mn, min(self.data[axis]), step)
                new_coord = self.data[axis]
                if len(lower) > 0:
                    if np.isclose(lower[-1], min(self.data[axis])):
                        lower = lower[:-1]

                    new_coord = np.concatenate((lower, new_coord))

                # the new values after the original axis
                upper = np.arange(max(new_coord) + step, mx + step, step)
                if len(upper) > 0:
                    new_coord = np.concatenate((new_coord, upper))

        # make a new array with all the same coords,
        # except replace the one axis with the new coord
        new_data = {}
        for da in self.data.values():
            coords = {}
            sizes = {}
            for ax in da.dims:
                if ax == axis:
                    coords[ax] = xr.DataArray(
                        new_coord,
                        coords={ax: new_coord},
                        dims=ax,
                        attrs=self.data.coords[ax].attrs,
                    )
                    sizes[ax] = len(new_coord)
                else:
                    coords[ax] = da.coords[ax]
                    sizes[ax] = len(da.coords[ax])

            sizes = [sizes[ax] for ax in da.dims]

            new_data[da.name] = xr.DataArray(
                data=np.zeros(sizes, dtype=da.dtype), coords=coords, dims=da.dims
            )

            # if existing data array is not empty, must add it to the new array
            if self.data[da.name].size > 0:
                with xr.set_options(arithmetic_join="outer"):
                    new_data[da.name] = (
                        (new_data[da.name] + self.data[da.name])
                        .fillna(0)
                        .astype(da.dtype)
                    )

        new_dataset = xr.Dataset(new_data)
        new_dataset.attrs = self.data.attrs.copy()
        self.data = new_dataset

    def _get_index(self, axis, value):
        """
        Find the index of the closest value in a coordinate
        named "axis", to the value given.

        Parameters
        ----------
        axis: str
            Name of coordinate to look up.
        value: scalar str or float
            Which value to try to match.
            If float, will find the closest match.
            If string, will find an exact match
            (or raise a ValueError if not found).

        Returns
        -------
        int
            The index of the closest value in the coordinate.
            If the value coordinate is outside the range
            (more than 1/2 step away from the min/max
            of the given coordinate axis) the index will
            be -1 (underflow) or len(coord) (overflow).
        """
        if isinstance(value, str):
            if value not in self.data.coords[axis].values:
                raise ValueError(f"Value {value} not in axis {axis}")
            return np.where(self.data.coords[axis].values == value)[0][0]
        else:
            step = self.data.coords[axis].attrs["step"]
            if value > max(self.data.coords[axis].values) + step / 2:
                return self.data.coords[axis].size
            elif value < min(self.data.coords[axis].values) - step / 2:
                return -1
            else:
                return np.argmin(np.abs(self.data.coords[axis].values - value))

    @staticmethod
    def from_netcdf(filename):
        """
        Load the data from a netcdf file.
        """
        if not os.path.exists(filename):
            raise ValueError(f"File {filename} does not exist.")

        with xr.open_dataset(filename) as ds:
            data = ds.load()

        folder, filename = os.path.split(filename)
        h = Histogram()
        if "name" in data.attrs:
            h.name = data.attrs["name"]
        if "pars" in data.attrs:
            parameters = json.loads(data.attrs["pars"])
            h.pars = ParsHistogram(**parameters)
        if "source_names" in data.attrs:
            if isinstance(data.attrs["source_names"], list):
                names = data.attrs["source_names"]
            else:
                names = [data.attrs["source_names"]]
            h.source_names = set(names)

        h.output_folder = folder
        h.data = data

        return h

    def get_fullname(self, suffix=None):
        """
        Get the name (path included) of the file associated with this histogram.
        """

        filename = "histograms"
        if self.name is not None:
            filename += f"_{self.name}"
        filename += ".nc"

        if suffix is not None:
            if not suffix.startswith("."):
                suffix = "." + suffix
            filename += suffix

        fullname = os.path.join(self.output_folder, filename)

        return fullname

    def load(self, suffix=None):
        """
        Load the data from the file.
        If suffix is given, will add that to the
        filename, after the extension
        (e.g., "histograms_all_score.nc.temp")
        """
        fullname = self.get_fullname(suffix=suffix)
        if os.path.exists(fullname):
            with xr.open_dataset(fullname) as ds:
                self.data = ds.load()

            if "name" in self.data.attrs:
                self.name = self.data.attrs["name"]

            if "pars" in self.data.attrs:
                parameters = json.loads(self.data.attrs["pars"])
                self.pars = ParsHistogram(**parameters)

            if "source_names" in self.data.attrs:
                if isinstance(self.data.attrs["source_names"], list):
                    names = self.data.attrs["source_names"]
                else:
                    names = [self.data.attrs["source_names"]]
                self.source_names = set(names)

    def save(self, suffix=None):
        """
        Save the data to the file.
        If suffix is given, will add that to the
        filename, after the extension
        (e.g., "histograms_all_score.nc.temp")
        """
        # make sure folder is there
        if not os.path.isdir(self.output_folder):
            safe_mkdir(self.output_folder)

        fullname = self.get_fullname(suffix=suffix)
        # netCDF files can't store dicts, must convert to string
        self.data.attrs["pars"] = json.dumps(self.pars.to_dict())
        self.data.attrs["source_names"] = list(self.source_names)
        self.data.to_netcdf(fullname, mode="w")

    def remove_data_from_file(self, suffix=None, remove_backup=False):
        """
        Save the data to the file.
        If suffix is given, will add that to the
        filename, after the extension
        (e.g., "histograms_all_score.nc.temp")
        """
        fullname = self.get_fullname(suffix=suffix)
        if os.path.exists(fullname):
            os.remove(fullname)
        if remove_backup:
            backup = fullname + ".backup"
            if os.path.exists(backup):
                os.remove(backup)

    def help(self=None, owner_pars=None):
        """
        Print the help for this object and objects contained in it.
        """
        if isinstance(self, Histogram):
            help_with_object(self, owner_pars)
        elif self is None or self == Histogram:
            help_with_class(Histogram, ParsHistogram)


if __name__ == "__main__":
    import pandas as pd
    import sqlalchemy as sa

    from src.database import SmartSession
    from src.source import Source
    from src.dataset import Lightcurve

    h = Histogram()
    h.initialize()

    # with SmartSession() as session:
    #     source = session.scalars(sa.select(Source).where(Source.project=='WD')).first()
    #     lc = source.lightcurves[0]
    #     df = lc.data
