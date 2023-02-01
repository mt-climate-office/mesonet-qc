import pandas as pd
import pytest

from pyqc.checks import _calc_daily_variance
from pyqc.columns import Columns


@pytest.fixture(scope="session")
def observations() -> pd.DataFrame:
    dat = pd.read_csv("./test/observations.csv")
    dat['datetime'] = pd.to_datetime(dat['datetime'])
    return dat


@pytest.fixture(scope="session")
def elements() -> pd.DataFrame:
    dat = pd.read_csv("./test/elements.csv")
    dat['date_start'] = pd.to_datetime(dat['date_start']).dt.date
    dat['date_end'] = pd.to_datetime(dat['date_end']).dt.date
    return dat


@pytest.fixture(scope="session")
def variance_df(observations) -> pd.DataFrame:
    columns = Columns()
    dat = _calc_daily_variance(observations, columns)
    return dat
