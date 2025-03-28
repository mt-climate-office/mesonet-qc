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

    if (columns.flag_min_col in dat.columns) and (columns.flag_max_col in dat.columns):
        dat = dat.assign(
            flag_range=dat[columns.compare_col].between(
                dat[columns.flag_min_col], dat[columns.flag_max_col], inclusive="both"
            )
        )
        dat = dat.assign(
            flag_range=np.where(
                dat[columns.flag_min_col].isna(), True, dat["flag_range"]
            )
        )
        dat = dat.assign(qa_range=(~dat.qa_range | ~dat.flag_range).astype(int))
    else:
        dat = dat.assign(qa_range=(~dat["qa_range"]).astype(int))

    dat = dat.assign(qa_range=np.where(dat["range_min"].isna(), -1, dat["qa_range"]))

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


def check_step_pd(
    dat: pd.DataFrame, columns: Columns, filter_first: bool = True, **kwargs
) -> pd.DataFrame:
    """Check step size criteria and assign QA flag for all observations in a DataFrame

    Args:
        dat (pd.DataFrame): A DataFrame that has has both observations and criteria needed to run the step check.
        columns (Columns): A Columns object that provides column mappings for everything needed to run the test.
        filter_first (bool): When doing the step check, the first value in a timeseries cannot be QA'd because
        there is no previous value to compare it to. If `filter_first` is True, this first value is simply filtered
        out. If it is set to False, the observation is kept and the `qa_step` column is assigned a fill value of -1.
    Returns:
        pd.DataFrame:  Updated DataFrame that now has a `qa_step` column with associated QA/QC flag values.
    """
    sort_cols = [columns.elem_col, columns.dt_col]

    if "id" in dat.columns:
        sort_cols.append("id")
    dat = dat.sort_values(sort_cols, ascending=False, ignore_index=True)

    dat = dat.set_index([columns.dt_col])

    grp_cols = ["id", columns.elem_col] if "id" in dat.columns else [columns.elem_col]

    diffs = (
        dat.groupby(grp_cols, dropna=False)[columns.compare_col]
        .rolling(2)
        .apply(lambda x: abs(x.iloc[1] - x.iloc[0]))
        .shift(-1)
        .reset_index()
        .rename(columns={"value": "diff"})
    )

    if all(diffs["diff"].isna()):
        dat = dat.assign(qa_step=-1)
        dat = dat.reset_index()
        return dat

    dat = dat.reset_index()
    dat = dat.merge(diffs, how="left")

    if filter_first:
        dat = dat[~dat["diff"].isna()].reset_index(drop=True)

    dat = dat.assign(qa_step=(~(dat["diff"] < dat[columns.step_col])).astype(int))
    if not filter_first:
        dat = dat.assign(qa_step=np.where(dat["diff"].isna(), -1, dat["qa_step"]))

    dat = dat.drop(columns=["diff"])
    dat = dat.assign(qa_step=np.where(dat[columns.step_col].isna(), -1, dat["qa_step"]))

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

    dat = dat.assign(qa_delta=(~(dat["sd"] >= dat[columns.delta_col])).astype(int))

    dat = dat.drop(columns=["sd", "date"])

    dat = dat.assign(
        qa_delta=np.where(dat[columns.delta_col].isna(), -1, dat["qa_delta"])
    )

    return dat


def check_like_elements(
    dat: pd.DataFrame, columns: Columns, **kwargs: pd.DataFrame
) -> pd.DataFrame:
    qa_cols = dat.columns.to_series().str.contains("qa_")
    split = {k: v for k, v in dat.groupby("element")}
    out = []

    for element, tmp in split.items():
        try:
            elev = int(element.split("_")[-1])
        except ValueError:
            elev = None

        try:
            like_elems = tmp[columns.shared_col].values[0].split(",")
            if elev:
                new_like_elems = [x + f"_{elev:04}" for x in like_elems]
                like_elems += new_like_elems
            filt = dat[dat["element"].isin(like_elems)]

            # Reversed because this order will be used to assign QA bit flags.
            # Binary number goes from R -> L so we need to reverse the elements to match.
            like_elems = list(reversed(like_elems))
            like_elems.insert(0, "datetime")
            like_elems = dict(zip(like_elems, range(len(like_elems)), strict=True))

            filt = filt[["datetime", "element"]].assign(
                # Sum the QA columns
                qa_sum=filt[filt.columns[qa_cols]].apply(
                    lambda x: -1 if -1 in x else sum(x), axis=1
                )
            )
            filt = (
                filt.assign(qa_sum=filt["qa_sum"].apply(lambda x: min(1, x)))
                .reset_index(drop=True)
                .pivot(index="datetime", columns="element", values="qa_sum")
                .rename_axis(None, axis=1)
                .reset_index()
                .fillna(-1)
            )

            # Join the columns to create a binary string
            filt = filt.assign(
                qa_shared=filt[
                    # Reorder the columns so they are in the same order as specified in the QA column
                    sorted(filt.columns, key=lambda x: like_elems[x])
                ]
                .iloc[:, 1:]
                .map(lambda x: str(int(x)).replace("-1", "0"))
                .apply("".join, axis=1)
            )[["datetime", "qa_shared"]]

            # Convert binary string to integer.
            filt = filt.assign(
                qa_shared=filt["qa_shared"].apply(
                    lambda x: -1 if "-1" in x else int(x, 2)
                )
            )

            tmp = tmp.merge(filt, how="left", on="datetime")
        except AttributeError:
            tmp = tmp.assign(qa_shared=0)
        except ValueError:
            # This means there is an airtable error.
            tmp = tmp.assign(qa_shared=-1)

        out.append(tmp)

    out = pd.concat(out)
    out = out.assign(qa_shared=out["qa_shared"].fillna(0))

    return out


def apply_outage_check(row):
    if not row["outage_ranges"]:
        return 0

    time = row["datetime"]
    out = 0
    for outage in row["outage_ranges"]:
        outage = pd.to_datetime(outage).tz_localize(time.tz)
        if len(outage) == 2:
            tmp = int(outage[0] <= time <= outage[1])
        else:
            tmp = int(time > outage[0])
        out = max(out, tmp)

    return out


def check_outages(
    dat: pd.DataFrame, columns: Columns, **kwargs: pd.DataFrame
) -> pd.DataFrame:
    dat = dat.assign(qa_outage=dat.apply(apply_outage_check, axis=1))
    return dat
