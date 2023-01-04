from typing import List

import numpy as np
import pandas as pd

from .columns import Columns

Numeric = float | int | np.number


def check_range(x: Numeric, low: Numeric, high: Numeric) -> int:
    """Check that a value falls within an accepted range.

    Args:
        x (Numeric): The value to check.
        low (Numeric): The minimum bound on the range check.
        high (Numeric): The maximum bound on the range check.

    Returns:
        int: Returns 0 if the check passes, 1 if the check fails.
    """
    if x > low and x < high:
        return 0
    return 1


def check_range_pd(dat: pd.DataFrame, columns: Columns, **kwargs) -> pd.DataFrame:
    """Check that all observations fall within an accepted range of values in a Pandas DataFrame.

    Args:
        dat (pd.DataFrame): A DataFrame that has has both observations and criteria needed to run the range check.
        columns (Columns): A Columns object that provides column mappings for everything needed to run the test.

    Returns:
        pd.DataFrame: Updated DataFrame that now has a `qa_range` column with associated QA/QC flag values.
    """
    dat = dat.assign(
        qa_range=dat[columns.compare_col].between(
            dat[columns.min_col], dat[columns.max_col], inclusive="both"
        )
    )
    dat = dat.assign(qa_range=(-dat["qa_range"]).astype(int))

    for i, row in dat.iterrows():
        val = row['qa_range']

        if np.isnan(row['range_min']) or np.isnan(row['range_max']):
            val = np.nan
        dat.at[i, "qa_range"] = val
    return dat


def check_step(x: Numeric, prev: Numeric, threshold: Numeric) -> int:
    """Check that the difference between two consecutive values are below a given threshold.

    Args:
        x (Numeric): The value to check.
        prev (Numeric): The previous value to compare with `x`.
        threshold (Numeric): The threshold that the difference between `x` and `prev` must be smaller than.

    Returns:
        int:  Returns 0 if the check passes, 1 if the check fails.
    """
    if abs(prev - x) < threshold:
        return 0
    return 1


def check_step_pd(dat: pd.DataFrame, columns: Columns, **kwargs) -> pd.DataFrame:
    """Check step size criteria and assign QA flag for all observations in a DataFrame

    Args:
        dat (pd.DataFrame): A DataFrame that has has both observations and criteria needed to run the step check.
        columns (Columns): A Columns object that provides column mappings for everything needed to run the test.

    Returns:
        pd.DataFrame:  Updated DataFrame that now has a `qa_step` column with associated QA/QC flag values.
    """
    dat = dat.sort_values([columns.elem_col, columns.dt_col], ascending=False, ignore_index=True)

    dat = dat.set_index([columns.dt_col])

    diffs = dat.groupby(columns.elem_col)[columns.compare_col].rolling(2).apply(
            lambda x: x.iloc[1] - x.iloc[0]
        ).shift(-1).reset_index().rename(columns={"value": "diff"})

    dat = dat.reset_index()
    dat = dat.merge(diffs, how="left")

    dat = dat[~dat["diff"].isna()].reset_index(drop=True)
    dat = dat.assign(qa_step=(~(dat["diff"] < dat[columns.step_col])).astype(int))
    dat = dat.drop(columns=["diff"])

    for i, row in dat.iterrows():
        val = row['qa_step']

        if np.isnan(row['step_size']):
            val = np.nan
        dat.at[i, "qa_step"] = val

    return dat


def check_variance(x: np.ndarray | List[Numeric], threshold: Numeric) -> int:
    """Check that the standard deviation of all values within a day is above a given threshold.

    Args:
        x (np.ndarray | List[Numeric]): Array-like object of all observations over the course of a day.
        threshold (Numeric): The standard deviation value that must be exceeded for the test to pass.

    Returns:
        int: Returns 0 if the check passes, 1 if the check fails.
    """
    if np.std(x) > threshold:
        return 0
    return 1


def _calc_daily_variance(dat: pd.DataFrame, columns: Columns) -> pd.DataFrame:

    dat = dat.assign(datetime=pd.to_datetime(dat[columns.dt_col]))

    sd = (
        dat.groupby([pd.Grouper(key=columns.dt_col, freq="1D"), columns.elem_col])[
            columns.compare_col
        ]
        .std()
        .reset_index()
    )
    sd = sd.assign(date=sd[columns.dt_col].dt.date)
    dat = dat.assign(date=dat[columns.dt_col].dt.date)
    sd = sd[["date", columns.elem_col, columns.compare_col]]
    sd.columns = ["date", columns.elem_col, "sd"]

    dat = dat.merge(sd, on=[columns.elem_col, "date"], how="left")
    dat = dat[["date", columns.elem_col, "sd"]]
    dat = dat.drop_duplicates(ignore_index=True)
    return dat


def check_variance_pd(
    dat: pd.DataFrame, columns: Columns, **kwargs: pd.DataFrame
) -> pd.DataFrame:
    """Check that the standard deviation of observations over the course of a day are above a
    specified threshold.

    Args:
        dat (pd.DataFrame): A DataFrame of observations and threshold values for the test.
        columns (Columns): A mapping of columns to use in the calculation.
        **kwargs (pd.DataFrame): Optional - A DataFrame of daily variance for each element.

    Returns:
        pd.DataFrame: _description_
    """

    if "variance_df" in kwargs:
        dat = dat.assign(date=pd.to_datetime(dat[columns.dt_col]).dt.date)
        dat = dat.merge(
            kwargs["variance_df"], on=[columns.elem_col, "date"], how="left"
        )
    else:
        var = _calc_daily_variance(dat, columns)
        dat = dat.assign(date=pd.to_datetime(dat[columns.dt_col]).dt.date)
        dat = dat.merge(var, on=[columns.elem_col, "date"], how="left")

    dat = dat.assign(qa_delta=(~(dat["sd"] > dat[columns.delta_col])).astype(int))

    dat = dat.drop(columns=["sd"])
    for i, row in dat.iterrows():
        val = row['qa_delta']

        if np.isnan(row['persistence_delta']):
            val = np.nan
        dat.at[i, "qa_delta"] = val
    return dat
