import pyqc.checks as ck
from pyqc.columns import Columns
from pyqc.process import check_observations


def test_check_observations(observations, elements):
    columns = Columns()
    dat = check_observations(
        observations,
        elements,
        columns,
        ck.check_range_pd,
        ck.check_step_pd,
        ck.check_variance_pd,
    )
    assert "qa_range" in dat.columns, "Range QA flag not properly added to DataFrame."
    assert -1 not in dat["qa_range"].to_list(), "QA range test yielded incorrect values"
    assert "qa_step" in dat.columns, "Step QA flag not properly added to DataFrame."
    assert -1 not in dat["qa_step"].to_list(), "QA step test yielded incorrect values"
    assert "qa_delta" in dat.columns, "Delta QA flag not properly added to DataFrame."
    assert -1 not in dat["qa_delta"].to_list(), "QA delta test yielded incorrect values"
