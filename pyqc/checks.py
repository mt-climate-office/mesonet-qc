import logging
from typing import List

import numpy as np
import pandas as pd

from .columns import Columns

logger = logging.getLogger(__name__)

Numeric = float | int | np.number


def check_range(x: Numeric, low: Numeric, high: Numeric) -> int:
    """Check that a value falls within an accepted range.

    Args:
        x (Numeric): The value to check.
        low (Numeric): The minimum bound on the range check (inclusive).
        high (Numeric): The maximum bound on the range check (inclusive).

    Returns:
        int: Returns 0 if the check passes, 1 if the check fails, -1 if any value is NaN.
    
    Examples:
        >>> import numpy as np
        >>> 
        >>> # Temperature check: valid range 32-120°F
        >>> check_range(75, 32, 120)  # Normal temperature
        0
        >>> check_range(150, 32, 120)  # Too hot - sensor error
        1
        >>> check_range(32, 32, 120)  # Boundary value - acceptable
        0
        >>> check_range(np.nan, 32, 120)  # Missing data
        -1
    """
    if pd.isna(x) or pd.isna(low) or pd.isna(high):
        return -1
    
    if low <= x <= high:
        return 0
    return 1


def check_range_pd(dat: pd.DataFrame, columns: Columns, **kwargs) -> pd.DataFrame:
    """Check that all observations fall within an accepted range of values in a Pandas DataFrame.

    This function checks if values in ``columns.compare_col`` are within the range specified by
    ``columns.min_col`` and ``columns.max_col``. It also supports an optional "flag" range check
    using ``columns.flag_min_col`` and ``columns.flag_max_col``.

    Args:
        dat (pd.DataFrame): A DataFrame containing observations and range criteria columns.
        columns (Columns): A Columns object mapping logical names to DataFrame column names.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        pd.DataFrame: The input DataFrame with a new ``qa_range`` column.
            - 0: Pass (value within range)
            - 1: Fail (value outside range)
            - -1: Not applicable (missing range criteria or value)

    Examples:
        >>> import pandas as pd
        >>> from pyqc.columns import Columns
        >>> cols = Columns(compare_col='val', min_col='min', max_col='max')
        >>> df = pd.DataFrame({
        ...     'val': [10, 50, 100, pd.NA],
        ...     'min': [20, 20, 20, 20],
        ...     'max': [80, 80, 80, 80]
        ... })
        >>> check_range_pd(df, cols)[['val', 'qa_range']]
           val  qa_range
        0   10         1
        1   50         0
        2  100         1
        3  <NA>       -1
    """
    # Calculate base range check
    qa_range = dat[columns.compare_col].between(
        dat[columns.min_col], dat[columns.max_col], inclusive="both"
    )

    # Check for flag range columns and apply if present
    if (columns.flag_min_col in dat.columns) and (columns.flag_max_col in dat.columns):
        try:
            flag_range = dat[columns.compare_col].between(
                dat[columns.flag_min_col], dat[columns.flag_max_col], inclusive="both"
            )
            # Set flag_range to True where flag_min is NaN (no flagging required)
            flag_range = np.where(dat[columns.flag_min_col].isna(), True, flag_range)
            # Combine both checks: fail if either range or flag check fails
            qa_range = (~qa_range | ~flag_range).astype(int)
        except (TypeError, KeyError) as e:
            logger.warning(f"Flag range check skipped due to: {e}")
            qa_range = (~qa_range).astype(int)
    else:
        qa_range = (~qa_range).astype(int)

    # Mark as -1 (not applicable) where range_min is missing
    qa_range = np.where(dat[columns.min_col].isna(), -1, qa_range)

    return dat.assign(qa_range=qa_range)


def check_step(x: Numeric, prev: Numeric, threshold: Numeric) -> int:
    """Check that the difference between two consecutive values are below a given threshold.

    Args:
        x (Numeric): The value to check.
        prev (Numeric): The previous value to compare with `x`.
        threshold (Numeric): The threshold that the difference between `x` and `prev` must be smaller than.

    Returns:
        int:  Returns 0 if the check passes, 1 if the check fails, -1 if any value is NaN.
    
    Examples:
        >>> import numpy as np
        >>> 
        >>> # Temperature readings 5 minutes apart, max step 10°F
        >>> check_step(75.5, 75.0, 10)  # Normal gradual change (0.5°F)
        0
        >>> check_step(90.0, 75.0, 10)  # Sudden jump (15°F) - sensor error
        1
        >>> check_step(np.nan, 75.0, 10)  # Missing current reading
        -1
    """
    if pd.isna(x) or pd.isna(prev) or pd.isna(threshold):
        return -1
    
    if abs(prev - x) < threshold:
        return 0
    return 1


def check_step_pd(
    dat: pd.DataFrame,
    columns: Columns,
    filter_first: bool = True,
    lookback: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """Check step size criteria and assign QA flag for all observations in a DataFrame.

    Calculates the absolute difference between the current value and a previous value
    (determined by ``lookback``) within each group defined by element (and optionally id).
    Compares this difference to ``columns.step_col``.

    Args:
        dat (pd.DataFrame): DataFrame containing observations and step criteria.
        columns (Columns): Column mappings.
        filter_first (bool): If True, removes the first ``lookback`` rows of each group
            (where step cannot be calculated). If False, keeps them with flag -1.
        lookback (int): Number of periods to look back for comparison. Default is 1.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        pd.DataFrame: Input DataFrame with ``qa_step`` column.
            - 0: Pass (step < threshold)
            - 1: Fail (step >= threshold)
            - -1: Not applicable (missing threshold or first value)

    Examples:
        >>> import pandas as pd
        >>> from pyqc.columns import Columns
        >>> cols = Columns(compare_col='val', step_col='step', dt_col='dt', elem_col='elem')
        >>> df = pd.DataFrame({
        ...     'dt': pd.to_datetime(['2021-01-01 12:00', '2021-01-01 12:05', '2021-01-01 12:10']),
        ...     'elem': ['temp', 'temp', 'temp'],
        ...     'val': [10, 12, 50],
        ...     'step': [5, 5, 5]
        ... })
        >>> check_step_pd(df, cols, filter_first=False)[['val', 'qa_step']]
           val  qa_step
        0   10       -1
        1   12        0
        2   50        1
    """
    sort_cols = [columns.elem_col, columns.dt_col]

    if "id" in dat.columns:
        sort_cols.append("id")
    
    # Sort by element and datetime (ascending) to ensure diff works correctly
    dat = dat.sort_values(sort_cols, ascending=True, ignore_index=True)

    grp_cols = ["id", columns.elem_col] if "id" in dat.columns else [columns.elem_col]

    # Calculate the absolute difference between current and the value 'lookback' steps ago
    dat["diff"] = dat.groupby(grp_cols)[columns.compare_col].diff(periods=lookback).abs()

    # If filter_first is True, remove rows where 'diff' is NaN (first values in group)
    if filter_first:
        dat = dat.dropna(subset=["diff"]).reset_index(drop=True)

    # Assign qa_step based on whether 'diff' is less than 'step_col'
    # 0 if diff < step_col (pass), 1 otherwise (fail)
    dat = dat.assign(qa_step=(~(dat["diff"] < dat[columns.step_col])).astype(int))

    # If not filter_first, set qa_step to -1 where 'diff' is NaN (first value in group)
    if not filter_first:
        dat = dat.assign(qa_step=np.where(dat["diff"].isna(), -1, dat["qa_step"]))

    # Set qa_step to -1 where step_col is NaN (no threshold to compare against)
    dat = dat.assign(
        qa_step=np.where(dat[columns.step_col].isna(), -1, dat["qa_step"])
    )

    # Drop the temporary 'diff' column
    dat = dat.drop(columns=["diff"])
    return dat


def check_variance(x: np.ndarray | List[Numeric], threshold: Numeric) -> int:
    """Check that the standard deviation of all values within a day is above a given threshold.

    Args:
        x (np.ndarray | List[Numeric]): Array-like object of all observations over the course of a day.
        threshold (Numeric): The standard deviation value that must be exceeded for the test to pass.

    Returns:
        int: Returns 0 if the check passes, 1 if the check fails, -1 if inputs are invalid.
    
    Examples:
        >>> import numpy as np
        >>> 
        >>> # Temperature over a day should vary by at least 2°F std dev
        >>> temps_normal = [70, 72, 75, 78, 76, 73, 71, 70]  # Natural variation
        >>> check_variance(temps_normal, 2.0)
        0
        >>> 
        >>> # Stuck sensor - no variation
        >>> temps_stuck = [70, 70, 70, 70, 70, 70, 70, 70]
        >>> check_variance(temps_stuck, 2.0)
        1
        >>> 
        >>> # Handle missing data
        >>> temps_with_nan = [70, np.nan, 75, 78, np.nan, 73]
        >>> check_variance(temps_with_nan, 2.0)  # Uses only valid values
        0
    """
    # Convert to numpy array and filter out NaN values
    x_clean = np.array(x)
    x_clean = x_clean[~np.isnan(x_clean)]
    
    # Return -1 if no valid data or threshold is NaN
    if len(x_clean) == 0 or pd.isna(threshold):
        return -1
    
    if np.std(x_clean) > threshold:
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
    """Check that the daily standard deviation of observations is above a specified threshold.

    This check is designed to detect stuck sensors that report constant or near-constant values
    over an entire day. It groups data by day and element, calculates the standard deviation,
    and compares it to ``columns.delta_col``.

    Args:
        dat (pd.DataFrame): DataFrame containing observations and variance threshold columns.
        columns (Columns): Column mappings.
        **kwargs:
            variance_df (pd.DataFrame, optional): Pre-calculated daily variance DataFrame.
            If provided, it avoids recalculating variance. Must contain ``columns.elem_col``,
            'date', and 'sd'.

    Returns:
        pd.DataFrame: Input DataFrame with ``qa_delta`` column.
            - 0: Pass (std dev > threshold)
            - 1: Fail (std dev <= threshold)
            - -1: Not applicable (missing threshold)

    Examples:
        >>> import pandas as pd
        >>> from pyqc.columns import Columns
        >>> cols = Columns(compare_col='val', delta_col='min_std', dt_col='dt', elem_col='elem')
        >>> df = pd.DataFrame({
        ...     'dt': pd.to_datetime(['2021-01-01 00:00', '2021-01-01 12:00']),
        ...     'elem': ['temp', 'temp'],
        ...     'val': [10, 10],  # Constant value -> std dev is 0
        ...     'min_std': [0.1, 0.1]
        ... })
        >>> check_variance_pd(df, cols)[['val', 'qa_delta']]
           val  qa_delta
        0   10         1
        1   10         1
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


def check_persistence_window(
    dat: pd.DataFrame,
    columns: Columns,
    max_identical_readings: int = 6,
    **kwargs
) -> pd.DataFrame:
    """Check for consecutive identical readings indicating a stuck sensor.
    
    This check complements the daily variance check by detecting short-term
    persistence that might not trigger the daily standard deviation threshold.
    
    Args:
        dat (pd.DataFrame): A DataFrame of observations with columns for datetime,
            element, and values.
        columns (Columns): A mapping of columns to use in the calculation.
        max_identical_readings (int): Maximum allowed consecutive identical values.
            Default is 6, which would flag 6+ identical readings in a row.
        **kwargs: Additional keyword arguments (unused, for compatibility).
    
    Returns:
        pd.DataFrame: Original DataFrame with added 'qa_persistence' column containing:
            - 0: Check passed (no excessive persistence detected)
            - 1: Check failed (sensor appears stuck with too many identical readings)
    
    Examples:
        >>> import pandas as pd
        >>> from pyqc.columns import Columns
        >>> 
        >>> # Sample data with stuck sensor
        >>> dat = pd.DataFrame({
        ...     'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
        ...     'element': ['air_temp'] * 10,
        ...     'value': [50.0, 50.1, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 51.0, 51.5]
        ... })
        >>> 
        >>> result = check_persistence_window(dat, Columns(), max_identical_readings=6)
        >>> # Rows with 6+ consecutive 50.0 values will be flagged
    
    Notes:
        - Values are compared for exact equality
        - Check is applied per element (and per id if present)
        - First occurrence of a value in a sequence is not flagged
        - Useful for real-time monitoring where daily variance check cannot run
    """
    grp_cols = [columns.elem_col]
    if "id" in dat.columns:
        grp_cols.append("id")
    
    dat = dat.sort_values([*grp_cols, columns.dt_col], ignore_index=True)
    
    # Identify when value changes from previous reading
    value_changed = (
        dat.groupby(grp_cols, dropna=False)[columns.compare_col]
        .diff()
        .ne(0)
    )
    
    # Count consecutive identical values using cumsum on value changes
    # This creates a group ID that increments each time the value changes
    group_id = value_changed.astype(int).cumsum()
    
    # Count size of each consecutive group
    dat['_consecutive_count'] = (
        dat.groupby([*grp_cols, group_id], dropna=False)
        .cumcount() + 1
    )
    
    # Flag if consecutive count reaches or exceeds threshold
    qa_persistence = (dat['_consecutive_count'] >= max_identical_readings).astype(int)
    
    # Clean up temporary column and return
    dat = dat.drop(columns=['_consecutive_count'])
    
    return dat.assign(qa_persistence=qa_persistence)
