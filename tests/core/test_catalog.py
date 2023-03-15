import os
import numpy as np
from astropy.time import Time

from src.catalog import Catalog


def test_demo_catalog(data_dir):
    filename = "test_catalog.csv"
    fullname = os.path.abspath(os.path.join(data_dir, "../catalogs", filename))

    try:
        Catalog.make_test_catalog(filename=filename, number=10)
        assert os.path.isfile(fullname)

        # set up a catalog with the default column definitions
        cat = Catalog(filename=filename, default="test")
        cat.load()
        assert cat.pars.filename == filename
        assert len(cat.data) == 10
        assert cat.data["ra"].dtype == np.float64
        assert cat.data["dec"].dtype == np.float64
        assert cat.pars.name_column in cat.data.columns

    finally:
        os.remove(fullname)
        assert not os.path.isfile(fullname)


def test_catalog_hdf5(data_dir):
    filename = "test_catalog.h5"
    fullname = os.path.abspath(os.path.join(data_dir, "../catalogs", filename))

    try:
        Catalog.make_test_catalog(filename=filename, number=10)
        assert os.path.isfile(fullname)

        # setup a catalog with the default column definitions
        cat = Catalog(filename=filename, default="test")
        cat.load()
        assert cat.pars.filename == filename
        assert len(cat.data) == 10
        assert cat.data["ra"].dtype == np.float64
        assert cat.data["dec"].dtype == np.float64
        assert cat.pars.name_column in cat.data.columns

    finally:
        os.remove(fullname)
        assert not os.path.isfile(fullname)


def test_catalog_wds():
    cat = Catalog(default="wds")
    cat.load()
    assert len(cat.data) > 0
    assert isinstance(cat.data["ra"][0], float)
    assert isinstance(cat.data["dec"][0], float)
    assert cat.pars.name_column in cat.get_columns()

    # more than a million sources
    assert len(cat) > 1000_000

    # catalog should be ordered by RA, so first objects are close to 360/0
    assert cat.data["ra"][0] > 359 or cat.data["ra"][0] < 1

    # average values are weighted by the galactic bulge
    assert abs(cat.data["ra"].mean() - 229) < 1
    assert abs(cat.data["dec"].mean() + 20) < 1

    # mean magnitude hovers around the limit of ~20
    assert abs(cat.data[cat.pars.mag_column].mean() - 20) < 0.1


def test_catalog_get_row():
    cat = Catalog(default="wds")
    cat.load()
    assert len(cat.data) > 0

    # get row based on index
    row = cat.get_row(0)  # first row
    assert isinstance(row, np.record)
    assert row["ra"] == cat.data["ra"][0]
    assert row["dec"] == cat.data["dec"][0]

    row = cat.get_row(-1)  # last row
    assert isinstance(row, np.record)
    assert row["ra"] == cat.data["ra"][-1]
    assert row["dec"] == cat.data["dec"][-1]

    idx = 7  # random choice
    row = cat.get_row(idx)
    assert isinstance(row, np.record)
    assert row["ra"] == cat.data["ra"][idx]
    assert row["dec"] == cat.data["dec"][idx]

    name = row[cat.pars.name_column]
    assert name == cat.data[cat.pars.name_column][idx]

    # get row based on name
    row = cat.get_row(name, index_type="name")
    assert isinstance(row, np.record)
    assert row["ra"] == cat.data["ra"][idx]
    assert row["dec"] == cat.data["dec"][idx]

    # get the row in the form of a dict
    row = cat.get_row(name, index_type="name", output="dict")
    assert isinstance(row, dict)
    assert row["name"] == str(name)
    assert row["ra"] == cat.data["ra"][idx]
    assert row["dec"] == cat.data["dec"][idx]

    # try to apply proper motion
    t = Time("2022-01-01")
    row = cat.get_row(name, index_type="name", output="dict", obstime=t)
    assert isinstance(row, dict)
    assert row["ra"] != cat.data["ra"][idx]
    assert abs(row["ra"] - cat.data["ra"][idx]) < 0.1

    # choose a preferred mag that is not Gaia_G
    row = cat.get_row(name, index_type="name", output="dict", preferred_mag="Gaia_BP")
    assert isinstance(row, dict)
    assert row["mag"] == cat.data["phot_bp_mean_mag"][idx]


def test_catalog_nearest_search():
    c = Catalog(default="test")
    c.load()

    idx = 2
    ra = c.data["ra"][idx]
    dec = c.data["dec"][idx]

    # search for the nearest object
    nearest = c.get_nearest_row(ra, dec, radius=2.0, output="dict")
    assert nearest["ra"] == ra
    assert nearest["dec"] == dec

    # try to nudge the coordinates a little bit
    nearest = c.get_nearest_row(
        ra + 0.3 / 3600, dec - 0.3 / 3600, radius=2.0, output="dict"
    )
    assert nearest["ra"] == ra
    assert nearest["dec"] == dec

    # make sure search works even if object is at RA=0
    c.data.loc[idx, "ra"] = 0

    nearest = c.get_nearest_row(0.1 / 3600, dec, radius=2.0, output="dict")
    assert nearest["ra"] == 0
    assert nearest["dec"] == dec

    nearest = c.get_nearest_row(-0.1 / 3600, dec, radius=2.0, output="dict")
    assert nearest["ra"] == 0
    assert nearest["dec"] == dec

    nearest = c.get_nearest_row(360 - 0.1 / 3600, dec, radius=2.0, output="dict")
    assert nearest["ra"] == 0
    assert nearest["dec"] == dec
