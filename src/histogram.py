import numpy as np
import pandas as pd
import xarray as xr

from src.parameters import Parameters


# TODO: should this be saved to the database?


class ParsHistogram(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.dtype = self.add_par(
            "dtype",
            "uint32",
            str,
            "Data type of underlying array (must be uint16 or uint32)",
        )
        default_score_coords = {
            "snr": (-10, 10, 0.1),
            "dmag": (-3, 3, 0.1),
        }  # other options: any of the quality cuts
        self.score_coords = self.add_par(
            "score_coords", default_score_coords, dict, "Coordinates for the score axes"
        )

        default_source_coords = {
            "mag": (15, 21, 0.5),
        }  # other options: color, ecl lat, mass, radius
        self.source_coords = self.add_par(
            "source_coords",
            default_source_coords,
            dict,
            "Coordinates for the source axes",
        )

        default_obs_coords = {
            "exptime": (30, 1),
            "filt": (),
        }  # other options: airmass, zp, magerr
        self.obs_coords = self.add_par(
            "obs_coords",
            default_obs_coords,
            dict,
            "Coordinates for the observation axes",
        )

        def __setattr__(self, key, value):
            if key == "dtype" and value not in ("uint16", "uint32"):
                raise ValueError(
                    f"Unsupported dtype: {value}, " f"must be uint16 or uint32."
                )

            super().__setattr__(key, value)


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
    like Signal to Noise Ratio (snr), or Delta Magnitude (dmag).
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

        can_initialize = kwargs.get("initialize", False)

        self.pars = ParsHistogram(**kwargs)
        self.data = None

        if can_initialize:
            self.initialize()

    def initialize(self):

        # create the coordinates
        coords = {}
        for input_ in ("score", "source", "obs"):
            for k, v in getattr(self.pars, f"{input_}_coords").items():
                coords[k] = self.create_coordinate(k, v)
                coords[k].attrs["input"] = input_

        data_shape = tuple(
            len(v) for v in coords.values() if v.attrs["input"] != "score"
        )

        # these coordinates are shared by all the DataArrays
        common_coord_names = [
            k for k, v in coords.items() if v.attrs["input"] != "score"
        ]

        data_vars = {}
        for k, v in coords.items():
            if v.attrs["input"] == "score":
                data_vars[k + "_counts"] = (
                    common_coord_names + [k],
                    np.zeros(data_shape + (len(v),), dtype=self.pars.dtype),
                )

        self.data = xr.Dataset(data_vars, coords=coords)

    def create_coordinate(self, name, specs):
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
            A tuple of (start, stop, step) for a fixed range,
            or an empty tuple for a dynamic range.

        Returns
        -------
        xarray.DataArray
            The coordinate axis.
        """

        if not isinstance(specs, tuple):
            raise ValueError(f"Coordinate specs must be a tuple: {specs}")

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

        ax.attrs["long_name"] = self.get_coordinate_name(name)
        ax.attrs["units"] = units

        return ax

    @staticmethod
    def get_coordinate_name(name):
        """
        Get the long name of a coordinate.
        If the name is not recognized,
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
            Can be 'kb', 'mb', or 'gb'.
            Otherwise, assume 'bytes' are returned.

        Returns
        -------
        float
            The size of the histogram in memory,
            in whatever units were requested.
        """
        total_size = 0
        for d in self.data.data_vars.values():
            total_size += d.size * d.dtype.itemsize

        return total_size / self.unit_convert_bytes(units)

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
            Can be 'kb', 'mb', or 'gb'.
            Otherwise, assume 'bytes' are returned.
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
        for d in self.data.coords.values():
            # only count non-source coordinates (shared by all DataArrays)
            if d.attrs["input"] != "score":
                if len(d) <= 1:
                    common_size *= dyn_coord_size
                else:
                    common_size *= len(d)

        score_size = 0
        for d in self.data.coords.values():
            # only count non-source coordinates (shared by all DataArrays)
            if d.attrs["input"] == "score":
                if len(d) <= 1:
                    score_size += dyn_score_size
                else:
                    score_size += len(d)

        total_size = common_size * score_size

        array_names = list(self.data.keys())
        total_size *= self.data[array_names[0]].dtype.itemsize

        return total_size / self.unit_convert_bytes(units)

    @staticmethod
    def unit_convert_bytes(units):
        if units.endswith("s"):
            units = units[:-1]

        return {
            "kb": 1024,
            "mb": 1024**2,
            "gb": 1024**3,
        }.get(units.lower(), 1)

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
        All the other values will be binned using ...

        Parameters
        ----------
        args:
            Dataframe or other objects containing data.
            The other objects can be, e.g., a Source object
            (that contains info on the magnitude and color)
            or a catalog row, etc.

        kwargs:
            Additional arguments to pass to the binning function.
            Currently there are no additional arguments.

        Returns
        -------

        """
        # go over args and see if any objects
        # have attributes that match one of the axes.
        input_data = {}
        for axis in self.data.dims:
            input_data[axis] = None
            for obj in args:
                if hasattr(obj, axis):
                    values = getattr(obj, axis)
                    if not isinstance(values, str) and hasattr(values, "__len__"):
                        # an array, but need to check if all are the same
                        if len(np.unique(values)) == 1:
                            input_data[axis] = values[0]
                        else:
                            input_data[axis] = np.array(values)
                    else:  # assume it is a scalar value / string
                        input_data[axis] = values
                    break  # take the first thing that matches

        sample_len = None
        for obj in args:
            if isinstance(obj, pd.DataFrame):
                sample_len = len(obj)
                break
        if sample_len is None:
            sample_len = 1

        num_scores = len(self.data.data_vars)

        # check all data axes have a coordinate value
        for axis in self.data.dims:
            if input_data[axis] is None:
                raise ValueError(f"Could not find data for axis {axis}")

        # check if any scalar values, or min/max of
        # the arrays of values, are outside the
        # range of any of the dynamic axes.
        # If so, expand the axes to include the new values.
        for axis in self.data.dims:
            if self.data.coords[axis].attrs["type"] == "fixed":
                continue

            # scalar string
            if isinstance(input_data[axis], str):
                if (
                    self.data.coords[axis].size == 0
                    or input_data[axis] not in self.data.coords[axis].values
                ):
                    self.expand_axis(axis, input_data[axis])
            # list of strings
            elif hasattr(input_data[axis], "__len__") and all(
                isinstance(x, str) for x in input_data[axis]
            ):
                if not set(input_data[axis]).issubset(self.data.coords[axis].values):
                    self.expand_axis(axis, input_data[axis])
            else:
                mx = max(self.data[axis] + self.data[axis].attrs["step"] / 2)
                mn = min(self.data[axis] - self.data[axis].attrs["step"] / 2)
                if hasattr(input_data[axis], "__len__"):
                    new_mx = max(input_data[axis])
                    new_mn = min(input_data[axis])
                else:
                    new_mx = input_data[axis]
                    new_mn = input_data[axis]
                if new_mx > mx or new_mn < mn:
                    self.expand_axis(axis, input_data[axis])

        # here is where we actually increase the bin counts
        for name, da in self.data.data_vars.items():
            # each da is for a different score

            # get a slice of the full array that matches
            # any scalar values
            indices = {}
            array_values = {}
            for ax in da.dims:
                if isinstance(input_data[ax], str) or not hasattr(
                    input_data[ax], "__len__"
                ):
                    indices[ax] = self.get_index(ax, input_data[ax])
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
                        num_values_to_add = sample_len / num_scores
                    if ax in indices and indices[ax] < 0:
                        da.coords[ax].attrs["underflow"] += num_values_to_add
                        in_range = False
                    elif ax in indices and indices[ax] >= len(da.coords[ax]):
                        da.coords[ax].attrs["overflow"] += num_values_to_add
                        in_range = False

            if in_range:
                # if all the arrays have unique values, just add the number of measurements:
                if not array_values:
                    self.data[name][indices] += np.array(sample_len).astype(da.dtype)
                else:
                    # bin the remaining dataframe columns into
                    # the appropriate axes
                    da_slice = self.data[name].isel(**indices)
                    if set(da_slice.dims) != set(array_values.keys()):
                        raise ValueError("Slice into data array has wrong dimensions!")

                    # setup the bin edges and values
                    # make sure they're ordered by the da_slice dims
                    bins = []
                    values = []
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
                    da_slice += counts.astype(da.dtype)

                    # add the over/underflow:
                    for ax in array_values:
                        if da_slice.coords[ax].attrs["type"] == "fixed":
                            if da_slice.coords[ax].attrs["input"] == "score":
                                correction = 1
                            else:
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

    def expand_axis(self, axis, new_values):
        if isinstance(new_values, str):
            if (
                self.data.coords[axis].size == 0
                or new_values not in self.data.coords[axis].values
            ):
                if self.data.coords[axis].size > 0:
                    new_coord = self.data.coords[axis].values + [new_values]
                else:
                    new_coord = [new_values]
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
                if np.isclose(lower[-1], min(self.data[axis])):
                    lower = lower[:-1]

                new_coord = np.concatenate(
                    (
                        lower,
                        self.data[axis],
                    )
                )

                # the new values after the original axis
                upper = np.arange(max(new_coord) + step, mx + step, step)

                new_coord = np.concatenate(
                    (
                        new_coord,
                        upper,
                    )
                )

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

        self.data = new_dataset

    def get_index(self, axis, value):
        """
        Find the index of the closest value in a coordinate
        named "axis", to the value given.

        Parameters
        ----------
        axis: str
            Name of coordinate to look up.
        value: str or float
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
            if (
                value
                > max(self.data.coords[axis].values)
                + self.data.coords[axis].attrs["step"] / 2
            ):
                return self.data.coords[axis].size
            elif (
                value
                < min(self.data.coords[axis].values)
                - self.data.coords[axis].attrs["step"] / 2
            ):
                return -1
            else:
                return np.argmin(np.abs(self.data.coords[axis].values - value))


if __name__ == "__main__":
    import pandas as pd
    import sqlalchemy as sa

    from src.database import Session
    from src.source import Source
    from src.dataset import Lightcurve

    h = Histogram()
    h.initialize()

    # with Session() as session:
    #     source = session.scalars(sa.select(Source).where(Source.project=='WD')).first()
    #     lc = source.lightcurves[0]
    #     df = lc.data
