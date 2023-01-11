import pandas as pd

import pyqc.checks as ck
from pyqc.columns import Columns


def test_check_range():
    assert ck.check_range(5, 0, 6) == 0, "Value in valid range fails test."
    assert ck.check_range(-1, 0, 6) == 1, "Value outside of valid range fails test."


def test_check_step():
    assert ck.check_step(4, 5, 2) == 0, "Values within valid step size fail test."
    assert ck.check_step(4, 8, 2) == 1, "Values outside of valid step size fail test."


def test_check_variance():
    assert (
        ck.check_variance([1, 2, 1, 2, 1, 2], 0.25) == 0
    ), "Std dev check fails when it should pass."
    assert (
        ck.check_variance([1, 2, 1, 2, 1, 2], 1) == 1
    ), "Std dev check passes when it should fail."


def test_check_range_pd(observations, elements):
    columns = Columns()
    dat = observations.merge(elements, on=["station", "element"], how="left")
    dat = ck.check_range_pd(dat, columns)
    assert "qa_range" in dat.columns, "Range QA flag not properly added to DataFrame."


def test_check_step_pd(observations, elements):
    columns = Columns()
    dat = observations.merge(elements, on=["station", "element"], how="left")
    dat = ck.check_step_pd(dat, columns)
    assert "qa_step" in dat.columns, "Step QA flag not properly added to DataFrame."


def test_check_step_pd_with_filter_first_false(observations, elements):
    columns = Columns()
    dat = observations.merge(elements, on=["station", "element"], how="left")
    new = ck.check_step_pd(dat, columns, filter_first=False)
    assert -1 in new['qa_step'].values, "The initial value was not kept properly."
    assert dat.shape[0] == new.shape[0], "Rows were removed in the step check when they should not have."


def test_check_variance_pd(observations, elements):
    columns = Columns()
    dat = observations.merge(elements, on=["station", "element"], how="left")
    dat = ck.check_variance_pd(dat, columns)
    assert "qa_delta" in dat.columns, "Delta QA flag not properly added to DataFrame."


def test_check_variance_with_premade_deltas(observations, elements):
    columns = Columns()
    premade = ck._calc_daily_variance(observations, columns)
    dat = observations.merge(elements, on=["station", "element"], how="left")
    dat = ck.check_variance_pd(dat, columns, variance_df=premade)
    assert "qa_delta" in dat.columns, "Delta QA flag not properly added to DataFrame."


def test_check_like_elements(observations, elements):
    columns = Columns()
    dat = observations.merge(elements, on=["station", "element"], how="left")
    dat = ck.check_range_pd(dat, columns)
    dat = ck.check_like_elements(dat, columns)
    assert "qa_shared" in dat.columns, "Step QA flag not properly added to DataFrame."
