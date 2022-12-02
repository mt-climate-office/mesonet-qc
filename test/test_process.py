import pyqc.checks as ck
from pyqc.process import check_observations
from pyqc.columns import Columns

def test_check_observations(observations, elements):
    columns = Columns()
    dat = check_observations(
        observations, elements, columns, 
        ck.check_range_pd, ck.check_step_pd, ck.check_variance_pd
    )
    assert "qa_range" in dat.columns, "Range QA flag not properly added to DataFrame."
    assert "qa_step" in dat.columns, "Step QA flag not properly added to DataFrame."
    assert "qa_delta" in dat.columns, "Delta QA flag not properly added to DataFrame."
