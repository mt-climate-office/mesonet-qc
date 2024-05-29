from typing import Callable, List

import pandas as pd

from .columns import Columns


def merge_elements_by_date(
    dat: pd.DataFrame, elements: pd.DataFrame, columns: Columns
) -> pd.DataFrame:
    """Make sure elements are aligned with proper observation dates.
    Args:
        dat (pd.DataFrame): DataFrame of all observations that will be QA/QC'd
        elements (pd.DataFrame): DataFrame of all sensor deployments at a given station.
        columns (Columns): An instance of the `Columns` class.
    Returns:
        pd.DataFrame: A filtered version of `elements` where each element is filtered to match
        the date of the observations.
    """
    obs_dt = dat["datetime"].dt.tz

    elements[columns.start_col] = (
        pd.to_datetime(elements[columns.start_col])
        .dt.tz_localize("UTC")
        .dt.tz_convert(obs_dt)
    )
    elements[columns.end_col] = (
        pd.to_datetime(elements[columns.end_col])
        .dt.tz_localize("UTC")
        .dt.tz_convert(obs_dt)
    )

    elements = elements.assign(
        date_end=elements[columns.end_col].fillna(
            pd.Timestamp.now(dat["datetime"].dt.tz)
        )
    )

    elements = elements.assign(
        date_end=elements["date_end"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    )

    dat = dat.drop_duplicates()
    pre_shp = dat.shape

    if "id" in dat.columns:
        elements.loc[~elements['element'].str.contains('soil'), 'sdi12_address'] = None
        dat = dat.merge(
            elements,
            left_on=["station", "element", "id"],
            right_on=["station", "element", "sdi12_address"],
        )
    else:
        dat = dat.merge(elements, on=["element", "station"], how="left")
    no_element = dat[~dat["datetime"].between(dat["date_start"], dat["date_end"])]
    dat = dat[dat["datetime"].between(dat["date_start"], dat["date_end"])]
    dat = dat.sort_values("date_end", ascending=False)
    if "id" in dat.columns:
        duped = dat[["station", "datetime", "element", "id"]].duplicated()
    else:
        duped = dat[["station", "datetime", "element"]].duplicated()
    dat = dat[~duped]

    # Rejoin if an elements metadata is missing. This will fill flags with -1 value.
    if pre_shp[0] > dat.shape[0]:
        no_element = no_element[
            (no_element["date_start"].isna()) & (no_element["date_end"].isna())
        ]
        dat = pd.concat([dat, no_element], ignore_index=True)

    # Recheck to see if this is still the case
    if pre_shp[0] > dat.shape[0]:
        raise IndexError(
            """An element being reported is missing in the elements table!
            Please check the elements table!"""
        )

    if pre_shp[0] < dat.shape[0]:
        raise IndexError(
            """There is a duplicated element in the elements table! 
            Please make sure all elements in this table are unique."""
        )

    return dat


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
    checks = list(checks)
    check_names = [x.__name__ for x in checks]
    if ("check_like_elements" in check_names) and (
        check_names[-1] != "check_like_elements"
    ):
        func = checks.pop(check_names.index("check_like_elements"))
        checks.append(func)

    if keep_columns is None:
        keep_columns = [
            "station",
            "datetime",
            "^element$",
            "value",
            "units",
            "qa_",
            "^id$",
        ]

    if "qa_" not in keep_columns:
        keep_columns.append("qa_")

    dat = merge_elements_by_date(dat, elements, columns)

    for check in checks:
        dat = check(dat, columns=columns, **kwargs)

    cols = dat.columns.to_series().str.contains("|".join(keep_columns))
    dat = dat[dat.columns[cols]]

    if dat.shape != dat.drop_duplicates().shape:
        qa_cols = dat.columns[dat.columns.to_series().str.contains("qa_")]
        dat[qa_cols] = -1

    return dat
