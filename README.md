# virtualobserver

A package used for downloading and processing images from various astronomical surveys

## Architecture

The goal of this project is to download, process, and store data from various astronomical surveys.
This is accomplished using a few classes:

- `Catalog`: stores a list of objects, each with a name, RA, Dec, and magnitude etc.
- `VirtualObservatory`: contains methods to download and store images, lightcurves, or other datasets.
- `Analysis`: runs some custom pipeline to reduce the data to smaller summary statistics, find interesting events, etc.
- `Source`: an entry for a single astronomical object, includes a row from the catalog,
  associated lightcurves, images, etc., and any analysis results.
  Each `Source` can be applied an `Analysis` object, which appends some summary statistics to that object,
  and may find some `Detection` objects.
- `Dataset`: a base class for various types of data, including images, lightcurves, and other data.
  A `Dataset` must have the ability to dump data to a file and later retrieve it.
- `Calibration`: an object to run some custom calibration on a dataset.
  Before using a dataset, a calibration must be applied to each dataset object.
  Datasets are saved to disk as they were downloaded, so calibration should be applied upon loading from disk.
- `PhotometricDataset`: a set of time vs. flux measurements for a single object. Inherits from `Dataset`.
  There are zero or more `PhotometricDataset` objects for each `Source`.
  After these are calibrated, they can be combined into a single "lightcurve" for that `Source`.
  An `Analysis` applied to a `Source` can, for example, use the lighturve as the input data.
- `Detection`: a specific point in time (or period) for an object, that may have astrophysical significance.
  These should contain or link back to the relevant data needed to decide if they are real or an artifact.
  A `Source` can contain zero or more `Detections`s.
- `Simulator`: takes the data that should be input to an `Analysis` object and injects a simulated event into it.
  Each `Analysis` must be able to run in "simulation mode" that produces `Detection` objects marked as "simulated".
-

## Goals and requirements

The first thing we need to be able to do is download data for a set of sources.
The sources should be given as a list, in the form of a `Catalog`.
This could be manually entered, or queried using some cuts based on e.g., Gaia.

The catalog can be given to `VirtualObservatory` objects,
that can then download the data and store it locally,
linking the data files to `Dataset` objects in memory.
There should be an easy way to parallelize this process.

`Sources`s should be persisted in memory (in a database)
and should be merged when multiple `Sources` are found
with different surveys (i.e., different `VirtualObservatory` objects).

Each `Source` can have multiple `Dataset` objects,
e.g., `PhotometricDataset` objects for time series flux measurements.
These need to be calibrated using a predefined `Calibration` object.
After calibration they can be used as a single "lightcurve" for that `Source`.
It should be relatively easy to display a calibrated lightcurve for a specific `Source`.

For each `Source` we can use `Analysis` objects to reduce the data to a summary statistic,
that should be persisted and be queryable for all sources.
The products of the `Analysis` should include:

- `Detection` objects, which can be used to find real or simulated events.
- Summary statistics on each source, for e.g., prioritizing followup.
- Some histograms including an estimate of the amount of data of various quality
  that was scanned across all sources in the catalog.
  This should live on the `Analysis` object and should be
  additive when joining results from parallel sessions using
  different `Analysis` objects.

A `Simulator` object can be used to inject simulated events into the data,
which produces simulated `Detection` objects. This allows testing of the
sensitivity of the analysis. These objects should be saved along with the
real events but should be marked as simulated in some hidden way.

The outputs from running the entire analysis should include:

- A list of `Detection` objects, some real, some simulated.
- A histogram of the coverage as function of data quality.
- `Source` objects with their associated summary statistics.
  Each of these data products must be saved to disk and should be
  easy to retreive for a given analysis.
