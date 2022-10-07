# virtualobserver

A package used for downloading and processing images from various astronomical surveys.

## Installation

#### Prerequisites

PostgreSQL version 14 to run the database server.

```
sudo apt install postgresql libpq-dev
```

#### Install from github

Download the code from github:

```commandline
git clone https://github.com/guynir42/virtualobserver.git
```

To develop and contribute to this repo,
first [make a fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
on the GitHub page, then clone your fork to your local machine
(i.e., replace `guynir42` with your GitHub username).

Generate a virtual environment and install the required python packages:

```commandline
cd virtualobserver/
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Virtual environments can be deactivated using `deactivate`.
To re-activate the environment, use `source venv/bin/activate`,
inside the `virtualobserver` directory.
This makes sure that the code uses the specific versions of the packages
given by the `requirements.txt` file.

#### Testing

To make sure the pacakge is working,
either after installation or after making changes,
run the unit tests:

```commandline
pytest
```

#### Data folder

Raw data folder should be set up with the environment variable `VO_DATA`.
This folder should be on a drive with enough space to contain the raw data.
Internally, the `DATA_ROOT` variable in `src/dataset.py` can be modified
to temporarily save data to other places (e.g., for testing):

```python
from src import dataset
dataset.DATA_ROOT = "/path/to/data/folder"
```

#### Additional folders:

You may want to generate folders for `catalogs` and `configs` in the root directory.
These folder may be generated automatically.

## Architecture

The goal of this project is to download, process,
and store data (and retrieve it!)
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
- A `Catalog` can read FITS or CSV files with a table of sources.
- Instances of `Source` and `RawPhotometry` and `Lightcurve` are persisted in a postgres database.
- Data objects like `RawPhotometry` keep track of a filename on disk (in HDF5 or FITS format),
  and load the data from disk when needed.
- A `Histograms` object is associated with a netCDF file on disk,
  which is loaded into a multidimensional `xarray` when needed.

Data that is saved to disk is automatically retrieved
by `virtualobserver` so that each call to a method like
`run()` on the project object or `download_all_sources()`
on the observatory object will continue from where it left off.

### Data folders

Raw data is downloaded from separate surveys,
and is saved under the `DATA_ROOT` folder,
in a separate folder for each observatory
(observatory names are pushed to upper case
when used in folder names).
For example, saving data from ZTF will put the raw
data files into the `DATA_ROOT/ZTF` folder.

Reduced data, as well as analysis results,
are saved in a separate folder for each project.
The project name is used (in upper case), as in
`DATA_ROOT/PROJECT_NAME`.
If using version control, that folder would
be appended the hash of the final config file,
to make sure that the data is not overwritten
when the config file is changed.
E.g, `DATA_ROOT/PROJECT_NAME_<md5 hash of config file>`
The project folder will also contain the final config file
used in the analysis, saved with the name
`00_config_<md5 hash of config file>.yaml`.

### Raw file names and keys

Raw data files are saved in the `DATA_ROOT/OBSERVATORY` folder,
with a filename that helps find the data easily even without the
associated database objects.

The default setting, is to store HDF5 files with data for each
object saved as a different group key in the file.
By default, datasets for all sources in an integer right ascension (RA) degree bin,
are saved in a single file.
That means data for two sources with RA 123.1 degrees and 123.7 degrees
will both be saved into a file named:
`DATA_ROOT/ZTF/ZTF_photometry_RA123.h5`,
where it is assumed the observatory is ZTF and the data is photometry.
Inside this file, the key for each source will be the source name in the catalog.
That means that data downloaded for different projects,
using different catalogs, may end up saving the same source (with different names)
into the same file, with redundancy.
If the name of the source is the same across both catalogs,
the data will not be saved twice.
The reason to combine sources by their RA is to make sure
the source data is spread out across a finite number of files,
without making any of the files too large to handle.
This allows separate threads to download data for different sources
and still keep multiple sources organized into a single file.

### Reduced files and other products

Once some raw data is on disk,
data products can be generated using the observatory object
or the analysis object.
The observatory object can be used to reduce the data,
making it easier to work with and analyze.
For example, raw photometry data can be reduced to lightcurves:

```python
lcs = obs.reduce(raw_data, to='lcs')
```

will produce some `Lightcurve` objects,
possible more than one lightcurve per raw dataset.
This is because the raw data may contain data from multiple filters,
or from different observing seasons/runs.
The details of how a `RawPhotometry` object is reduced to a `Lightcurve` object
are determined by the specific observatory object and reduction parameters.

When saving a `Lightcurve` object,
it would be placed, by default, into a folder named after the project (in upper case),
with a filename identical to the raw data file,
with an appended `_reduced` suffix before the file extension.
E.g., `DATA_ROOT/TEST_PROJECT/DEMO_photometry_RA123_reduced.h5`.
This is because the reduced data depends on the details of the project,
and is not shared across projects.
If using version control, the folder would be appended the hash of the final config file,
as explained below.

The file key for the lightcurve object would be the source name in the catalog,
appended with `_reduction_XX_of_YY` where `XX` is the number of the reduction,
out of a total of `YY` reductions.
The reduction number is just the order the lightcurves were created by the analysis code,
and do not necessarily reflect the time of observations.

### Version control

**This is not yet implemented.**

The way version control works is that
when running a project (using the `run` command),
the parameters for all different classes are collected
into a new config file, including changes applied by the user
through interactive or script commands.
The _final config file_ will also contain
a hash of the git commit that was used,
so that even when running the same project
with the same parameters but with different code,
the final config hash will still be different.
It is saved in the project folder,
for future reference, and its hash
is used to tag the data products folder,
containing the reduced data and the analysis results.

To turn on version control, specify a parameter
`version_control: True` in the project config
or directly to the parameters object
`project.pars.version_control = True`.
You can also use `vc` as short for `version_control`.

When disabled, the output folder will just be named
by the project name, and the content in it could be
outdated if the code/parameters were changed.
This is useful for exploratory analysis.
Note that `virtualobserver` will happily re-use
existing data products even if changes were made
to the code or parameters, so if version control is
disabled, the user must be responsible for clearing
the old database rows and data folders when making
substantial changes.

An important caveat is that raw data,
downloaded directly from each survey,
is not managed by version control,
and is shared between projects.
This is because downloading raw data usually takes
a long time, and is meant to be unaffected by
the code or parameters chosen for different projects.

### Classes:

The module contains several classes:

- `Project`: A top level object that handles the configuration parameters for downloading, saving and analyzing data.
  Each project would have a list of observatories, a catalog, and a unique project name that associates all database objects
  with a particular project. After initializing a project, with a given configuration, the object can then download
  any files not yet saved to disk, load files from disk, reduce the data and run analysis.
- `Parameters`: Keep track of parameters from the user input or from a config file.
  These objects are attributes of `Project`, `VirtualObservatory` and `Analysis` objects,
  so all the tunable parameters can be maintained for each parent object.
- `Catalog`: stores a list of objects, each with a name, RA, Dec, and magnitude etc.
- `VirtualObservatory`: contains methods to download and store images, lightcurves, or other datasets.
  Subclasses of the `VirtualObservatory` class are used to download data from specific surveys.
  An example subclass would be `VirtualZTF` to download and reduce data from the Zwicky Transient Facility
  (<https://www.ztf.caltech.edu/>).
  The observatory objects are also used to reduce data (e.g., from raw photometry to usable lightcurves).
- `Analysis`: runs some custom pipeline to reduce the data to smaller summary statistics, find interesting events, etc.
- `Source`: an entry for a single astronomical object, includes a row from the catalog,
  associated lightcurves, images, etc., and any analysis results.
  Each `Source` can be applied an `Analysis` object, which appends some summary statistics to that object,
  and may find some `Detection` objects.
- `DatasetMixin`: a base class for various types of data, including images, lightcurves, and other data.
  A `DatasetMixin` has the ability to save data to a file and later retrieve it.
  All astronomical data classes have `times` (a vector of datetime objects) and `mjds`
  (a vector of modified julian dates) as attributes. They all have a filename and if required,
  will also keep an in-file key for files containing multiple entries (e.g., HDF5 files).
- `RawPhotometry` is used to store unprocessed photometric data. Inherits from `DatasetMixin`.
  This class is mostly used to store filenames to allow data to be easily saved/loaded for future analysis.
  There are zero or more `RawPhotometry` objects associated with each `Source`, upt to one per observatory.
  Raw data is saved once per observatory, and can be reused in different projects (if they share sources).
- `Lightcurve`: a set of time vs. flux measurements for a single object. Inherits from `DatasetMixin`.
  There are zero or more `Lightcurve` objects for each `Source`.
  Each `Lightcurve` is for a single filter in a single survey, usually for a single observing season/run.
  An `Analysis` applied to a `Source` can, for example, use the lightcurves as the input data.
  There are one or more `Lightcurve` objects associated with each `RawPhotometry` object.
- `DetectionMixin`: A base class for all sorts of detection objects. The simplest example is given below,
  which is the `DetectionInTime` object that stores information about a time-local event like a transient.
  Other subclasses that can be useful are `DetectionInPeriod` or `DetectionInSpectrum`.
- `DetectionInTime`: a specific point in time for an object, that may have astrophysical significance.
  These refer back to the relevant data used to produce them.
  A `Source` can contain zero or more `DetectionsInTime`s.
- `Simulator`: takes the data that should be input to an `Analysis` object and injects a simulated event into it.
  Each `Analysis` must be able to run in "simulation mode" that produces `Detection` objects marked as "simulated".
- `Histogram`: A multidimensional array (in `xarray` format) that saves statistics on the analysis.
  This can be important to maintain a record of, e.g., how many epochs each source was observed,
  so any detections (or null detections) can be translated into rates or upper limits on rates.
  The multiple dimensions of these histograms are used to bin the data along different values
  of the input data like the source magnitude (i.e., having many epochs on faint sources may be less
  useful than having a few epochs on a bright source).

### Files and directories

The directory structure is as follows:

- `src` is the source code directory, containing all the python files.
- `tests` contains unit tests for the code.
- `catalogs` contains files with lists of astronomical objects.
- `configs` contains configuration files for various projects.
- `results` contains analysis results.

Important files in the `src` folder are:

- Each of the following modules contains a single class with a similar name:
  `analysis.py`, `catalog.py`, `finder.py`, `histograms.py`, `parameters.py`, `project.py`, `quality.py`,
  `simulator.py`, and `source.py`.
- `observatory.py` contains the `VirtualObservatory` class, which is a base class for all survey-specific classes.
  It also contains the `VirtualDemoObs` which is an example observatory that can be used for testing data reduction.
- `database.py` does not contain any classes, but is used to set up the database connection using SQLAlchemy.
- `dataset.py` contains the `DatasetMixin` class, which is a base class for all data types.
  It also contains the `RawPhotometry` class, which is used to store filenames for data that has not yet been reduced,
  and the `Lightcurve` class, which is used to store reduced photometry (lightcurves).
- `detection.py` contains the `DetectionMixin` class, which is a base class for all detection objects.
  It also contains the `DetectionInTime` class, which is a specific example of a detection class.
- `ztf.py` contains the `VirtualZTF` class, which is a subclass of `VirtualObservatory`
  and is used to download data from ZTF, and reduce the data into usable products.

## Usage examples

### Full analysis pipeline

Define a project and run the full pipeline:

```python
from src.project import Project
proj = Project(
    name="default_test",  # must give a name to each project
    project_string="my project",  # random parameter example
    version_control=False, # whether to use version control on products
    obs_names=["ZTF"],  # must give at least one observatory name
    # parameters to pass to the Analysis object:
    analysis_kwargs={"analysis_key": "analysis_value"},
    catalog_kwargs={"default": "WD"},  # load the default WD catalog
    # parameters to be passed into each observatory class
    obs_kwargs={'reducer': {'radius': 3, 'gap': 40}},
    # specific instructions to be used by the ZTF observatory only
    ZTF={"credentials": {"username": "guy", "password": "12345"}}
)

# download all data for all sources in the catalog
# and reduce the data (skipping raw and reduced data already on file)
# and store the results as detection objects in the DB, along with
# detection stats in the form of histograms array.
proj.run()

```

This project will download a Gaia-based FITS file
with a list of white dwarfs,
and set up a ZTF observatory with the given credentials.
The `run()` method will then backfill data products
that are missing from the data folder.
For example, the project will not download raw data that has already
been downloaded, it will not reduce data that has already been reduced,
and will not produce analysis results that have already been produced.

### Config files and constructor arguments

Input parameters can be saved in a configuration file
instead of specified in the constructor.
To call that file, use the special keywords
`cfg_file` to specify the file name,
and the `cfg_key` to specify the key in the file
to load for into that specific object (a project in this example).
For example, the above code can be written as:

```python
proj = Project(name='project_name', cfg_file="project_name.cfg", cfg_key="project")
```

The `cfg_file` must be an absolute path, otherwise it should be
specified without a relative path, and is assumed to be in the `configs` folder.
In addition, even if the `cfg_file` and `cfg_key` are not specified,
the `Parameters` object of the project will try to find a file
named `configs/<project name>.yaml` and look for a `project` key in it.
Only if `cfg_file` is specified, the code will raise an error if no such file exists.
To explicitly avoid loading a config file, use `cfg_file=None`.

If mixing config file arguments and constructor arguments,
the constructor arguments override those in the config file.
If the user changes any of the attributes of the `Parameters` object,
the new values will be used when calling `run()` on the project.

### Using custom code and arguments

To customize your analysis, two major inputs are used:
(1) Choosing arguments (parameters) for each object in the pipeline,
(2) Adding subclasses with custom code.

As an example, initialize a project with an Analysis object:

```python
proj = Project(
    name="default_test",  # must give a name to each project
    analysis_kwargs={"analysis_key": "analysis_value"},
)
```

This will create the default `Analysis` object from `analysis.py`,
and pass it the `analysis_kwargs` dictionary when it is initialized.
The argument `analysis_key` is stored in the `Analysis` object's `pars` object,
which is a `Parameters` object (or subclass of it).

This, however, limits the analysis to code already in the `Analysis` class.
To subclass it, simple add a new file and class inside of it (that inherits from `Analysis`).
For example, create a new file `my_analysis.py` with the following contents:

```python

# extensions/my_analysis.py
from src.analysis import Analysis

class NewAnalysis(Analysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pars = Parameters.from_dict(kwargs, "analysis")
        self.pars.default_values(
            new_analysis_key="new_analysis_value"
        )
    # override this part of the analysis
    def run_lightcurves(self, source):
        ...
```

The file in this example is put into the `extensions` folder,
but it can be saved anywhere (including the `src` folder).
Initialize the project with the new class:

```python
proj = Project(
    name="default_test",  # must give a name to each project
    analysis_kwargs={"new_analysis_key": "new_analysis_value"},
    analysis_module="extensions.my_analysis",
    analysis_class="NewAnalysis",
)
```

### Downloading data

Use only one of the observatories to download the data:

```python
obs = proj.observatories["ZTF"]  # can also use observatories[0]
obs.pars.num_threads_download = 0  # can use higher values for multi-threading
obs.download_all_sources(0, 1000, save=True)  # grab first 1000 sources in catalog
# use save=False to only download the data (for debugging)

len(obs.sources)  # should only contain 100 latest Source objects
len(obs.datasets)  # should only contain 100 latest RawPhotometry objects
```

This code will download all the data for the first 1000 sources in the catalog.
If any of the raw data has already been downloaded, it will not be downloaded again.

Internally, the `VirtualObservatory` superclass will
take care of database interaction and file creation,
possibly running multiple threads
(controlled by the `pars.num_threads_download` parameter).
Inside this code is a function `fetch_data_from_observatory()`
That is only implemented in observatory subclasses.
This function gets a `cat_row` dictionary with some
info on the specific source (name, RA/Dec, magnitude, etc.),
and additional keyword arguments,
and produces raw data (generally a dataframe) and an `altdata` dictionary
with additional information about the data (metadata).
This function calls whatever specific web API is used by the
specific observatory to download the data.

### Loading a catalog

Use the default white dwarf (WD) catalog from
<https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.3877G/abstract>.

```python
from src.catalog import Catalog
cat = Catalog(default="WD")
cat.load()
```

The `load()` method will download the catalog file if it is not already present,
and read it from memory if it is not already loaded.

### Reducing lightcurves

To reduce lightcurves, first define an observatory
with optional `reducer` parameters or use the inputs directly.

```python
from src.ztf import VirtualZTF

obs = VirtualZTF(
    name="ZTF",
    credentials={"username": "guy", "password": "12345"},
    reducer={"gap": 60},
)
```

Load the raw data and split it into lightcurves for different filters:

```python
from src.dataset import RawPhotometry
data = RawPhotometry(filename="my_raw_data.h5")
data.load()
obs.pars.reducer = {'gap': 60}
lcs = obs.reduce(data, to='lcs')
```

This should produce a list of `Lightcurve` objects, one for each filter/season.
In this case the `gap` parameter is used to split the lightcurve into multiple
"seasons" if there is a gap of more than 60 days between epochs.

## Using the data

Below are some examples for how to load and use the data
stored in various ways across database and filesystem.

### Working with the database

Database objects can be retrieved using SQLAlchemy queries.
Each database interaction is started using a `session` object.

```python
import sqlalchemy as sa
from src.database import Session
session = Session()
```

To get full objects (rather than tuples with specific columns)
use the `session.scalars()`.
Inside the `scalars` block, use the `sa.select(Class)` method
to select from one of the tables of mapped objects
(mapped classes include `Source`, `RawPhotometry`, `Lightcurve`, `DetectionInTime`, etc).
Use the `all()` or `first()` methods to get all or the first object.
To filter the results use the `where()` method on the select statement object.

```python
sources = session.scalars(sa.select(Source)).all()
source = session.scalars(
  sa.select(Source).where(
    Source.name == 'J123.1-32.13'
  )
).first()
```

To add a new object to the database,
or to delete an object from the database use:

```python
session.add(new_object)
session.delete(object_to_delete)
```

Make sure that the data on disk is deleted first, e.g.,

```python
data = session.scalars(
  sa.select(RawPhotometry).where(
    RawPhotometry.source_id == source.id
  )
).first()

data.delete_data_from_disk()
session.delete(data)
```

Finally, each transaction should end with a `session.commit()`,
to transfer the changes to the database.
If there are any errors, use `session.rollback()` to undo the changes.

When using the session interactively,
one should remember to call `session.close()` when done.
When using the session in a script,
it is best to use the `with` statement to ensure that the session is closed.

```python
with Session() as session:
  source = session.scalars(
    sa.select(Source).where(
      Source.name == 'J123.1-32.13'
    )
  ).first()
  data = source.raw_data[0]
  lcs = obs.reduce(data, to='lcs')
  for lc in lcs:
    lc.save()
    session.add(lc)

  session.commit()
```

### loading data from disk

Raw data and reduced data should always be associated
with a database object, either a `RawPhotometry` or `Lightcurve` object.
Each of them has a `get_fullname()` method that returns the full path
to the file on disk, and if relevant, also has a `filekey` attribute
that keeps track of the in-file key for this dataset (e.g., in HDF5 files).

To load the data, simple use the `load()` method of the object,
or just refer to the `data` attribute of the object (it will lazy load the data).
To save new data to disk use the `save()` method.
If the data is associated with a source, the filename
will be generated using the source's right ascension,
and the file key will be the source name.

```python
raw_data.save()
raw_data.filename  # ZTF_photometry_RA123.h5
raw_data.get_fullname()  # /path/to/dataroot/ZTF/ZTF_photometry_RA123.h5
raw_data.filekey  # J123.1-32.13

lc.save()
lc.filename  #  ZTF_photometry_RA123_reduced.h5
lc.get_fullname()  # /path/to/dataroot/PROJ_NAME/ZTF_photometry_RA123_reduced.h5
lc.filekey  # J123.1-32.13_reduction_01_of_03
```

As long as the datasets are associated with a database object,
there is no need to know the file name, path or key.
Simply load the objects from database and use the `data`.
The dataset objects also have some plotting tools that can
be used to quickly visualize the data.
These will lazy load the `data` attribute from disk.

```python
with Session() as session:
  data = session.scalars(
    sa.select(RawPhotometry).where(
      RawPhotometry.source_name == "J123.1-32.13"
    )
  ).first()

  mag_array = data.data['mag']
  data.plot()
```

### Existing data

When running down the list of sources in a catalog,
each one is first checked against the database to see
if the data already exists.
If the database object exists but the data is missing on disk,
it will be re-downloaded or re-reduced and attached to the object.
If the database object does not exist,
the data is downloaded/reduced and a new object is added to the database.

The raw data is only downloaded once for all projects,
assuming the different projects use the same name for the source.
Other intermediate products are saved for each project separately.

In some cases, raw data remains on disk while the database is cleared.
If this happens, use the observatory `populate_sources` method to
scan all the data files in `DATA_ROOT` and associate a source from the catalog
with each record on disk.
This may require some tweaking of the observatory parameters,
so the file key matches the correct column in the catalog.

### Analysis results

For each `Source` we use `Analysis` objects to reduce the data to a summary statistic,
that is persisted and queryable for all sources.
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
real events but are marked as simulated using a hidden attribute.
