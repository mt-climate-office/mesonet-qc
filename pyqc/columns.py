from dataclasses import dataclass


@dataclass
class Columns:
    """A class to keep track of the columns used to QA/QC check data.
    Args:
        compare_col (str): The column with data values to be QA/QC checked.
        start_col (str): The column giving the start date of an element.
        end_date (str): The column giing the end date of an element.
        min_col (str): The column specifying the minimum valid range of an observation.
        max_col (str): The column specifying the maximum valid range of an observation.
        flag_min_col (str): Some variables should raise a flag if its observations are out of a specified range,
        even if the values are technically within the range of what the sensor can measure. The `flag_min_col` and
        `flag_max_col` columns give the min and max range of this criteria, respectively.
        flag_max_col (str): As above but the max range and observation can have before a flag is raised.
        step_col (str): The column specifying the largest allowable step size between consecutive observations.
        dt_col (str): The column giving the datetime of the observations.
        elem_col (str): The column specifying the variable that each observation is.
        delta_col (str): The column speficying the minimum allowable standard deviation across a day of observations.
        like_col (str): The column specifying which other elements a given observation should be compared to.
        shared_sensor(str): The column specifying which other elements originate from the same instrument on a station.
    """

    compare_col: str = "value"
    start_col: str = "date_start"
    end_col: str = "date_end"
    min_col: str = "range_min"
    max_col: str = "range_max"
    flag_min_col: str = "flag_min"
    flag_max_col: str = "flag_max"
    step_col: str = "step_size"
    dt_col: str = "datetime"
    elem_col: str = "element"
    delta_col: str = "persistence_delta"
    like_col: str = "like_element"
    shared_col: str = "shared_sensor"
