from typing import Callable, List

import pandas as pd

import pyqc.checks as ck
from .columns import Columns


def check_observations(
    dat: pd.DataFrame,
    thresholds: pd.DataFrame,
    columns: Columns,
    *checks: Callable,
    variance_df: pd.DataFrame = None,
    keep_columns: List[str] = None
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
        keep_columns (List[str]): A list of columns (or a pattern to match to columns) in the original dataframe that
        should be kept and returned in the final dataframe. If left as None, 'station', 'datetime', 'element', 'value', and 'units'
        columns will be kept. 

    Returns:
        pd.DataFrame: A new observations dataframe additional QA coluns
    """

    # Make sure the like_element check is the final check that is done. 
    check_names = [x.__name__ for x in checks]
    if ("check_like_elements" in check_names) and (check_names[-1] != "check_like_elements"):
        func = checks.pop(check_names.index("check_like_elements"))
        checks.append(func)


    if keep_columns is None:
        keep_columns = ["station", "datetime", "^element$", "value", "units", "qa_"]
        
    if "qa_" not in keep_columns:
        keep_columns.append("qa_")

    dat = dat.merge(thresholds, on=["station", "element"], how="left")
    kwargs = {"variance_df": variance_df} if variance_df else {}
    for check in checks:
        dat = check(dat, columns=columns, **kwargs)

    cols = dat.columns.to_series().str.contains("|".join(keep_columns))

    dat = dat[dat.columns[cols]]
    return dat

# import numpy as np

# thresholds = pd.read_csv("../test/elements.csv")
# thresholds = thresholds[thresholds["date_end"].isna()]
# dat = pd.read_csv("../test/observations.csv")
# dat = dat[['station', 'datetime', 'element', 'value']]

# columns = Columns()
# checks = [ck.check_range_pd, ck.check_step_pd, ck.check_like_elements]
# dat = check_observations(dat, thresholds, columns, *checks)


# dat = dat.merge(thresholds, on=["station", "element"], how="left")a