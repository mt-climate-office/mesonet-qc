import pandas as pd
import pytest

from pyqc.checks import _calc_daily_variance
from pyqc.columns import Columns


@pytest.fixture(scope="session")
def observations() -> pd.DataFrame:
    dat = pd.read_csv("./test/observations.csv")
    return dat


@pytest.fixture(scope="session")
def elements() -> pd.DataFrame:
    dat = pd.read_csv("./test/elements.csv")
    dat = dat[dat["date_end"].isna()]
    return dat


@pytest.fixture(scope="session")
def variance_df(observations) -> pd.DataFrame:
    columns = Columns()
    dat = _calc_daily_variance(observations, columns)
    return dat
