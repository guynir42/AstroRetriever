import os
import pandas as pd

from src.observatory import VirtualDemoObs


def test_export_skyportal_photometry(test_project, new_source, raw_phot):
    assert isinstance(test_project.demo, VirtualDemoObs)
    new_source.raw_photometry = [raw_phot]
    lightcurves = test_project.demo.reduce(source=new_source, data_type="photometry")

    lc = lightcurves[0]

    filename = "test_skyportal_photometry.h5"
    try:  # make sure to remove file at the end
        lc.export_to_skyportal(filename)

        with pd.HDFStore(filename) as store:
            keys = store.keys()
            assert len(keys) == 1
            key = keys[0]
            df = store[key]
            for name in ["mjd", "flux", "fluxerr"]:
                assert name in df.columns

            metadata = store.get_storer(key).attrs["metadata"]

            for name in [
                "series_name",
                "series_obj_id",
                "exp_time",
                "ra",
                "dec",
                "filter",
                "time_stamp_alignment",
            ]:
                assert name in metadata

    finally:
        if os.path.isfile(filename):
            os.remove(filename)
