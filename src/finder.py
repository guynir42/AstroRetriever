import numpy as np
import pandas as pd


class FinderBase:
    """
    A base class for finder objects.
    This pipeline object accepts various data
    instances and produces Detection objects.

    If accepting a Lightcurve object, it could
    add some scores (like "snr" or "dmag") to the dataframe
    columns, if they don't exist already from the reduction stage.
    If any (or all) of the scores go over the threshold
    it should save a Detection object with the details
    of that detection (the Lightcurve ID, the time,
    S/N and so on).

    If accepting images, TBD...

    It is important that the Finder object
    does not change when applied to new data,
    since we expect to re-run the same data
    multiple times, before and after injecting
    simulations.
    The sim parameter is also used when the data
    ingested has an injected event in it.
    If sim=None, assume real data,
    if sim=dict, assume simulated data
    (or injected data) where the "truth values"
    are passed in using the dictionary.




    """
