# virtualobserver

A package used for downloading and processing images from various astronomical surveys.

## Architecture

The goal of this project is to download, process,
and store data (and retreive it!)
originating in various astronomical surveys.

### Workflow

The workflow is as follows:

1. Choose a list of targets based on some criteria from a catalog (e.g. Gaia).
2. Download images or photometry from one or more surveys.
3. Reduce the raw data into a format where data is uniform enough to be used in analysis.
4. Run analysis on the data and save the results per source, and some statistics on the data as a whole.

Each of these steps uses parameters that are stored in a configuration file.
Each of these steps produces data products that are saved in a way that is easy
to retrieve from disk and to continue the workflow without having to redo previous steps.
We use several modes of persistence, and objects that lazy load the content from disk when needed.

### Persistence of data

Examples for storing and retrieving data products:

- A `Project` object can load a configuration file and store the parameters in memory.
- A `Catalog` can read FITS files with a table of sources.
- Instances of `Source` and `RawData` and `Lightcurve` are persisted in a postgres database.
- Data objects like `RawData` keep track of a filename on disk (in HDF5 or FITS format),
  and load the data from disk when needed.
- A `Histograms` object is associated with a netCDF file on disk,
  which is loaded into a multidiemensional `xarray` when needed.

### Classes:

The module contains several classes:

- `Project`: A top level object that handles the configuration parameters for downloading, saving and analyzing data.
  Each project would have a list of observatories, a catalog, and a unique project name that associates all database objects
  with a particular project. After initializing a project, with a given configuration, the object can then download
  any files not yet saved to disk, load the all files, reduce the data and run analysis.
- `Parameters`:
- `Catalog`: stores a list of objects, each with a name, RA, Dec, and magnitude etc.
- `VirtualObservatory`: contains methods to download and store images, lightcurves, or other datasets.
  Subclasses of the `VirtualObservatory` class are used to download data from specific surveys.
  An example subclass would be `VirtualZTF` to download and reduce data from the Zwicky Transient Facility
  (<https://www.ztf.caltech.edu/>).
- `Analysis`: runs some custom pipeline to reduce the data to smaller summary statistics, find interesting events, etc.
- `Source`: an entry for a single astronomical object, includes a row from the catalog,
  associated lightcurves, images, etc., and any analysis results.
  Each `Source` can be applied an `Analysis` object, which appends some summary statistics to that object,
  and may find some `Detection` objects.
- `DatasetMixin`: a base class for various types of data, including images, lightcurves, and other data.
  A `DatasetMixin` has the ability to dump data to a file and later retrieve it.
  All astronomical data classes have `times` (a vector of datetime objects) and `mjds`
  (a vector of modified julian dates) as attributes. They all have a filename and if required,
  will also keep an in-file key if each file contains multiple entries (e.g., HDF5 files).
- `RawData` is used for all sort of unprocessed data, including images, lightcurves, etc. Inherits from `DatasetMixin`.
  This class is mostly used to store filenames to allow data to be easily saved/loaded for future analysis.
  There are zero or more `RawData` objects associated with each `Source`, usually one per survey.
- `Lightcurve`: a set of time vs. flux measurements for a single object. Inherits from `DatasetMixin`.
  There are zero or more `Lightcurve` objects for each `Source`.
  Each `Lightcurve` is for a single filter in a single survey, usually for a single observing season/run.
  An `Analysis` applied to a `Source` can, for example, use the lightcurves as the input data.
- `Detection`: a specific point in time (or period) for an object, that may have astrophysical significance.
  These should contain or link back to the relevant data needed to decide if they are real or an artifact.
  A `Source` can contain zero or more `Detections`s.
- `Simulator`: takes the data that should be input to an `Analysis` object and injects a simulated event into it.
  Each `Analysis` must be able to run in "simulation mode" that produces `Detection` objects marked as "simulated".
- `Histograms`: To be added...

### Files and directories

The directory structure is as follows:

- `src` is the source code directory, containing all the python files.
- `tests` contains unit tests for the code.
- `catalogs` contains files with lists of astronomical objects.
- `configs` contains configuration files for various projects.
- `results` contains analysis results.

Important files in the `src` folder are:

- Each of the following modules contains a single class with a similar name:
  `project.py`, `parameters.py`, `catalog.py`, `analysis.py`, `source.py`, `detection.py`, `simulator.py`, `histograms.py`.
- `virtualobservatory.py` contains the `VirtualObservatory` class, which is a base class for all survey-specific classes.
  It also contains the `VirtualDemoObs` which is an example observatory that can be used for testing data reduction.
- `database.py` does not contain any classes, but is is used to setup the database connection using SQLAlchemy.
- `ztf.py` contains the `VirtualZTF` class, which is a subclass of `VirtualObservatory`
  and is used to download data from ZTF, and reduce the data into usable products.

## Installation

Download the code from github:

```commandline
git clone https://github.com/guynir42/virtualobserver.git
```

To develope and contribute to this repo,
first [make a fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
on the github page, then clone your fork to your local machine
(i.e., replace `guynir42` with your github username).

Generate a virtual environment and install the required python packages:

```commandline
cd virtualobserver/
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### additional requirements

PostgreSQL version 14 to run the database server.

```
sudo apt install postgresql
```

#### Data folder

Raw data folder should be set up with the environment variable `VO_DATA`.
This folder should be on a drive with enough space to contain the raw data.
Internally, the `DATA_ROOT` variable in `src/dataset.py` can be modified
to temporarily save data to other places (e.g., for testing).

#### Additional folders:

You may want to generate folders for `catalogs` and `configs` in the root directory.

## Using the data

To be expanded...

### Working with the database

To be expanded...

### Intermediate data products

To be expanded...

### Analysis results

For each `Source` we use `Analysis` objects to reduce the data to a summary statistic,
that is persisted queryable for all sources.
The products of the `Analysis` include:

- `Detection` objects, which can be used to find real or simulated events.
- Summary statistics on each source, for e.g., prioritizing followup.
- Some histograms including an estimate of the amount of data of various quality
  that was scanned across all sources in the catalog.
  This is represented by the `Histograms` class.
  These are additive to allow joining results from parallel sessions
  each outputting different `Histograms` objects.

A `Simulator` object can be used to inject simulated events into the data,
which produces simulated `Detection` objects. This allows testing of the
sensitivity of the analysis. These objects should be saved along with the
real events but is marked as simulated using a hidden attribute.
