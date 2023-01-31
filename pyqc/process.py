import datetime as dt
from typing import Callable, List

import pandas as pd

from .columns import Columns


def match_elements_by_date(dat: pd.DataFrame, elements: pd.DataFrame) -> pd.DataFrame:
    """Filter elements to match sensor deployment dates.
    Args:
        dat (pd.DataFrame): DataFrame of all observations that will be QA/QC'd
        elements (pd.DataFrame): DataFrame of all sensor deployments at a given station.
    Returns:
        pd.DataFrame: A filtered version of `elements` where each element is filtered to match
        the date of the observations.
    """
    elements = elements.copy(deep=True)
    elements = elements.assign(date_end=elements["date_end"].fillna(dt.date.today()))
    tmp = dat.assign(date=dat["datetime"].dt.date)[
        ["element", "date"]
    ].drop_duplicates()

    elements = elements.merge(tmp, on="element", how="left")
    elements = elements[
        (elements["date_start"] <= elements["date"])
        & (elements["date_end"] > elements["date"])
    ]
    elements = elements.drop(columns=["date"]).drop_duplicates(ignore_index=True)
    return elements


def check_observations(
    dat: pd.DataFrame,
    elements: pd.DataFrame,
    columns: Columns,
    *checks: Callable,
    keep_columns: List[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Given a long-formatted dataframe of observations and a dataframe of qa/qc check elements for
    each element in the observations, perform all appropriate qa/qc checks.

    Args:
        dat (pd.DataFrame): A long-formatted dataframe of wether observations with the following columns:
        `station`, `datetime`, `element`, `value`.
        elements (pd.DataFrame): A dataframe of elements for different QA/QC tests. Should have
        columns outlined in the `Columns` class.
        columns (Columns): A Class mapping the column names of `dat` and `elements` to those used
        in the QA/QC checking functions.
        *checks (Callable): QA/QC functions from `check.py` that will be used to check the observations.
        keep_columns (List[str]): A list of columns (or a pattern to match to columns) in the original dataframe that
        should be kept and returned in the final dataframe. If left as None, 'station', 'datetime', 'element', 'value', and 'units'
        columns will be kept.
        **kwargs: Values to be passed to the check functions, these include `variance_df` and
        `filter_first`

    Returns:
        pd.DataFrame: A new observations dataframe additional QA coluns
    """

    # Make sure the like_element check is the final check that is done.
    check_names = [x.__name__ for x in checks]
    if ("check_like_elements" in check_names) and (
        check_names[-1] != "check_like_elements"
    ):
        func = checks.pop(check_names.index("check_like_elements"))
        checks.append(func)

    if keep_columns is None:
        keep_columns = ["station", "datetime", "^element$", "value", "units", "qa_"]

    if "qa_" not in keep_columns:
        keep_columns.append("qa_")

    elements = match_elements_by_date(dat, elements)
    if any(elements["element"].duplicated()):
        raise IndexError(
            """There is a duplicated element in the elements table! 
            Please make sure all elements in this table are unique."""
        )

    dat = dat.merge(elements, on=["station", "element"], how="left")

    for check in checks:
        dat = check(dat, columns=columns, **kwargs)

    cols = dat.columns.to_series().str.contains("|".join(keep_columns))
    dat = dat[dat.columns[cols]]

    if dat.shape != dat.drop_duplicates().shape:
        qa_cols = dat.columns[dat.columns.to_series().str.contains("qa_")]
        dat[qa_cols] = -1

    return dat


# import numpy as np
# elements = pd.read_csv("../test/elements.csv")
# elements = elements[elements["date_end"].isna()]
# dat = pd.read_csv("../test/observations.csv")
# dat = dat[['station', 'datetime', 'element', 'value']]

# # elements = pd.read_csv("~/Desktop/elements.csv")
# # dat = pd.read_csv("~/Desktop/dat.csv")
# # columns = Columns()
# # checks = [check_range_pd, check_step_pd, check_like_elements]
# dat = check_observations(dat, elements, columns, *checks, filter_first=True)
# dat = dat.merge(elements, on=["station", "element"], how="left")
