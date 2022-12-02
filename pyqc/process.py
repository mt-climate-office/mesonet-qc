import pandas as pd
import pyqc.checks as ck
from .columns import Columns

from typing import Callable

def check_observations(
    dat: pd.DataFrame, thresholds:pd.DataFrame, columns: Columns, *checks: Callable, variance_df: pd.DataFrame=None
) -> pd.DataFrame:
    """Given a long-formatted dataframe of observations and a dataframe of qa/qc check thresholds for 
    each element in the observations, perform all appropriate qa/qc checks.

    Args:
        dat (pd.DataFrame): A long-formatted dataframe of wether observations with the following columns:
        `station`, `datetime`, `element`, `value`.
        thresholds (pd.DataFrame): A dataframe of thresholds for different QA/QC tests. Should have
        columns outlined in the `Columns` class. 
        columns (Columns): A Class mapping the column names of `dat` and `thresholds` to those used 
        in the QA/QC checking functions. 
        *checks (Callable): QA/QC functions from `check.py` that will be used to check the observations.
        variance_df (pd.DataFrame): A DataFrame with columns `element` and `sd` and `date` that gives the 
        daily standard deviation of daily observations of a given element. If provided it will be used
        for the persistence delta QA/QC check. If left blank, it is assumed that sub-hourly values are provided
        for an entire day in order to calculate the sd. 

    Returns:
        pd.DataFrame: A new observations dataframe additional QA coluns
    """
    
    dat = dat.merge(thresholds, on = ["station", "element"], how="left")
    kwargs = {'variance_df': variance_df} if variance_df else {}
    
    for check in checks:
        dat = check(dat, columns=columns, **kwargs)
    dat = dat[['station', 'datetime', 'element', 'value', 'qa_range', 'qa_step', 'qa_delta']]
    return dat

# thresholds = pd.read_csv("../test/elements.csv")
# thresholds = thresholds[['station', 'element', 'range_min', 'range_max', 'step_size', 'persistence_delta', 'spatial_sd']]
# dat = pd.read_csv("https://mesonet.climate.umt.edu/api/v2/observations?stations=aceabsar&start_time=2022-10-01&end_time=2022-10-03&wide=False&type=csv&level=0")
# dat = dat[['station', 'datetime', 'element', 'value']]
# columns = Columns()

# check_observations(dat, thresholds, columns, ck.check_range_pd, ck.check_step_pd, ck.check_variance_pd)