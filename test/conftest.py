import pandas as pd
import pytest
from pyqc.columns import Columns
from pyqc.checks import _calc_daily_variance


@pytest.fixture(scope='session')
def observations() -> pd.DataFrame:
    dat = pd.read_csv("./test/observations.csv")
    return dat


@pytest.fixture(scope='session')
def elements() -> pd.DataFrame:
    dat = pd.read_csv("./test/elements.csv")
    return dat


@pytest.fixture(scope='session')
def variance_df(observations) -> pd.DataFrame:
    columns = Columns()
    dat = _calc_daily_variance(observations, columns)
    return dat
