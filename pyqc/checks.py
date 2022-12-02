import numpy as np
import pandas as pd
from typing import List

from columns import Columns

Numeric = float | int | np.number

def check_range(x: Numeric, low: Numeric, high: Numeric) -> int:
    """_summary_

    Args:
        x (Numeric): _description_
        low (Numeric): _description_
        high (Numeric): _description_

    Returns:
        int: _description_
    """
    if x > low and x < high:
        return 0
    return 1


def check_range_pd(dat: pd.DataFrame, columns: Columns, **kwargs) -> pd.DataFrame:
    """_summary_

    Args:
        dat (pd.DataFrame): _description_
        columns (Columns): _description_

    Returns:
        pd.DataFrame: _description_
    """
    dat = dat.assign(
        qa_range = dat[columns.compare_col].between(dat[columns.min_col], dat[columns.max_col], inclusive="both")
    )
    dat = dat.assign(
        qa_range = (-dat['qa_range']).astype(int)
    )
    return dat


def check_step(x: Numeric, prev: Numeric, threshold: Numeric) -> int:
    """_summary_

    Args:
        x (Numeric): _description_
        prev (Numeric): _description_
        threshold (Numeric): _description_

    Returns:
        int: _description_
    """
    if abs(prev - x) < threshold:
        return 0
    return 1


def check_step_pd(dat: pd.DataFrame, columns: Columns, **kwargs) -> pd.DataFrame:
    """_summary_

    Args:
        dat (pd.DataFrame): _description_
        columns (Columns): _description_

    Returns:
        pd.DataFrame: _description_
    """
    dat = dat.assign(
        diff = abs(dat[columns.compare_col].rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0]))
    )
    dat = dat[~dat['diff'].isna()]
    dat = dat.assign(
        qa_step = (~(dat['diff'] < dat[columns.step_col])).astype(int)
    )
    dat = dat.drop(columns=["diff"])
    return dat



def check_variance(x: np.ndarray | List[Numeric], threshold: Numeric) -> int:
    """_summary_

    Args:
        x (np.ndarray | List[Numeric]): _description_
        threshold (Numeric): _description_

    Returns:
        int: _description_
    """
    if np.std(x) > threshold:
        return 0
    return 1


def check_variance_pd(dat: pd.DataFrame, columns: Columns, **kwargs: pd.DataFrame) -> pd.DataFrame:
    """Check that the standard deviation of observations over the course of a day are above a
    specified threshold.

    Args:
        dat (pd.DataFrame): A DataFrame of observations and threshold values for the test.
        columns (Columns): A mapping of columns to use in the calculation. 
        **kwargs (pd.DataFrame): Optional - A DataFrame of daily variance. 

    Returns:
        pd.DataFrame: _description_
    """
    if "variance_df" in kwargs:
        dat = dat.merge(sd, on=[columns.elem_col, "date"], how="left")
    else: 
        dat = dat.assign(
            datetime = pd.to_datetime(dat[columns.dt_col])
        )
        
        sd = dat.groupby([pd.Grouper(key=columns.dt_col, freq="1D"), columns.elem_col])[columns.compare_col].std().reset_index()
        sd = sd.assign(
            date = sd[columns.dt_col].dt.date
        )
        dat = dat.assign(
            date = dat[columns.dt_col].dt.date
        )
        sd = sd[["date", columns.elem_col, columns.compare_col]]
        sd.columns = ["date", columns.elem_col, "sd"]

        dat = dat.merge(sd, on=[columns.elem_col, "date"], how="left")

    dat = dat.assign(
        qa_delta = (~(dat['sd'] > dat[columns.delta_col])).astype(int)
    )

    dat = dat.drop(columns=["sd"])
    return dat
