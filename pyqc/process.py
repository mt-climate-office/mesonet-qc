import pandas as pd


def check_observations(dat: pd.DataFrame, thresholds:pd.DataFrame) -> pd.DataFrame:
    """Given a long-formatted dataframe of observations and a dataframe of qa/qc check thresholds for 
    each element in the observations, perform all appropriate qa/qc checks.

    Args:
        dat (pd.DataFrame): A long-formatted dataframe of wether observations with the following columns:
        `station`, `datetime`, `element`, `value`, `units`.
        thresholds (pd.DataFrame): A dataframe of thresholds with the following columns:
        `element`, `range_min`, `range_max`, `step_size`, `persistence_delta`

    Returns:
        pd.DataFrame: A new observations dataframe with an additional QA column. 
    """
    pass


