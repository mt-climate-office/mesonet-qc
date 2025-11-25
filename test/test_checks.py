import pyqc.checks as ck
from pyqc.columns import Columns


def test_check_range():
    assert ck.check_range(5, 0, 6) == 0, "Value in valid range fails test."
    assert ck.check_range(-1, 0, 6) == 1, "Value outside of valid range fails test."


def test_check_step():
    assert ck.check_step(4, 5, 2) == 0, "Values within valid step size fail test."
    assert ck.check_step(4, 8, 2) == 1, "Values outside of valid step size fail test."


def test_check_variance():
    assert ck.check_variance([1, 2, 1, 2, 1, 2], 0.25) == 0, (
        "Std dev check fails when it should pass."
    )
    assert ck.check_variance([1, 2, 1, 2, 1, 2], 1) == 1, (
        "Std dev check passes when it should fail."
    )


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
    assert -1 in new["qa_step"].values, "The initial value was not kept properly."
    assert dat.shape[0] <= new.shape[0], (
        "Rows were removed in the step check when they should not have."
    )


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


# Edge case tests for boundary values
def test_check_range_boundary_values():
    """Test that boundary values are handled correctly."""
    assert ck.check_range(0, 0, 6) == 0, "Lower boundary should pass"
    assert ck.check_range(6, 0, 6) == 0, "Upper boundary should pass"
    assert ck.check_range(3, 3, 3) == 0, "Equal boundaries should pass"
    assert ck.check_range(-0.001, 0, 6) == 1, "Just below lower boundary should fail"
    assert ck.check_range(6.001, 0, 6) == 1, "Just above upper boundary should fail"


# NaN handling tests
def test_check_range_with_nan():
    """Test NaN handling in range check."""
    import numpy as np
    
    assert ck.check_range(np.nan, 0, 6) == -1, "NaN value should return -1"
    assert ck.check_range(5, np.nan, 6) == -1, "NaN lower bound should return -1"
    assert ck.check_range(5, 0, np.nan) == -1, "NaN upper bound should return -1"


def test_check_step_with_nan():
    """Test NaN handling in step check."""
    import numpy as np
    
    assert ck.check_step(np.nan, 5, 2) == -1, "NaN current value should return -1"
    assert ck.check_step(4, np.nan, 2) == -1, "NaN previous value should return -1"
    assert ck.check_step(4, 5, np.nan) == -1, "NaN threshold should return -1"


def test_check_variance_with_all_nan():
    """Test variance check with all NaN values."""
    import numpy as np
    
    assert ck.check_variance([np.nan, np.nan, np.nan], 0.25) == -1, (
        "All NaN should return -1"
    )


def test_check_variance_with_constant_values():
    """Test variance when std is exactly zero."""
    assert ck.check_variance([5, 5, 5, 5], 0.1) == 1, (
        "Zero variance should fail when threshold > 0"
    )
    # When both std and threshold are 0, std > threshold is False, so check fails
    assert ck.check_variance([5, 5, 5, 5], 0.0) == 1, (
        "Zero variance should fail when threshold is 0 (0 is not > 0)"
    )


def test_check_variance_with_partial_nan():
    """Test variance check with some NaN values."""
    import numpy as np
    
    result = ck.check_variance([1, np.nan, 2, 1, 2, np.nan, 1], 0.25)
    assert result in [0, 1], "Should handle partial NaN gracefully and return valid result"
    
    # Test with mostly NaN values
    result = ck.check_variance([np.nan, np.nan, 1, np.nan], 0.1)
    assert result == 1, "Single value (zero std dev) should fail variance check"


def test_check_variance_with_nan_threshold():
    """Test variance check with NaN threshold."""
    import numpy as np
    
    assert ck.check_variance([1, 2, 3], np.nan) == -1, "NaN threshold should return -1"


def test_check_range_negative_values():
    """Test range check with negative values."""
    assert ck.check_range(-5, -10, 0) == 0, "Negative value in negative range should pass"
    assert ck.check_range(-15, -10, 0) == 1, "Value below negative range should fail"


def test_check_persistence_window_basic(observations, elements):
    """Test basic persistence window check functionality."""
    import pandas as pd
    
    columns = Columns()
    
    # Create data with stuck sensor - 8 consecutive identical values
    stuck_data = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
        'station': ['test'] * 10,
        'element': ['air_temp'] * 10,
        'value': [50.0, 50.1, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 51.0, 51.5]
    })
    
    result = ck.check_persistence_window(stuck_data, columns, max_identical_readings=6)
    
    assert 'qa_persistence' in result.columns, "Persistence QA flag not added"
    assert 1 in result['qa_persistence'].values, "Should flag stuck sensor"
    assert 0 in result['qa_persistence'].values, "Should have some passing values"


def test_check_persistence_window_no_stuck_sensor():
    """Test that varying data doesn't trigger persistence check."""
    import pandas as pd
    
    columns = Columns()
    
    # Data with natural variation
    varying_data = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
        'station': ['test'] * 10,
        'element': ['air_temp'] * 10,
        'value': [50.0, 50.1, 50.2, 50.3, 50.2, 50.1, 50.0, 49.9, 50.0, 50.1]
    })
    
    result = ck.check_persistence_window(varying_data, columns, max_identical_readings=6)
    
    assert all(result['qa_persistence'] == 0), "No values should be flagged for varying data"


def test_check_persistence_window_threshold_sensitivity():
    """Test that threshold parameter affects detection."""
    import pandas as pd
    
    columns = Columns()
    
    # Data with 5 consecutive identical values
    data = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=8, freq='5min'),
        'station': ['test'] * 8,
        'element': ['air_temp'] * 8,
        'value': [50.0, 50.0, 50.0, 50.0, 50.0, 51.0, 51.0, 51.0]
    })
    
    # With threshold of 6, should not flag
    result_high = ck.check_persistence_window(data, columns, max_identical_readings=6)
    assert all(result_high['qa_persistence'] == 0), "Should not flag with threshold=6"
    
    # With threshold of 4, should flag
    result_low = ck.check_persistence_window(data, columns, max_identical_readings=4)
    assert 1 in result_low['qa_persistence'].values, "Should flag with threshold=4"


# Integration tests
def test_full_qa_pipeline(observations, elements):
    """Test complete QA pipeline with all checks."""
    from pyqc.process import check_observations
    
    columns = Columns()
    checks = [
        ck.check_range_pd,
        ck.check_step_pd,
        ck.check_variance_pd,
        ck.check_like_elements
    ]
    
    result = check_observations(observations, elements, columns, *checks)
    
    # Verify all QA columns exist
    expected_columns = ['qa_range', 'qa_step', 'qa_delta', 'qa_shared']
    assert all(col in result.columns for col in expected_columns), (
        f"Missing QA columns. Expected {expected_columns}, got {result.columns.tolist()}"
    )
    
    # Verify no data loss (allowing for filter_first in step check)
    assert len(result) > 0, "Result should not be empty"
    
    # Verify flag values are valid (qa_shared can be binary encoded, so allow integers >= -1)
    qa_cols = [col for col in result.columns if col.startswith('qa_')]
    for col in qa_cols:
        if col == 'qa_shared':
            # qa_shared uses binary encoding, so it can be any non-negative integer or -1
            assert result[col].min() >= -1, f"Column {col} has values < -1"
        else:
            # Other QA columns should only be -1, 0, or 1
            assert result[col].isin([-1, 0, 1]).all(), (
                f"Column {col} has invalid values: {result[col].unique()}"
            )


def test_qa_with_timezone_handling(observations, elements):
    """Test that QA checks handle timezones correctly."""
    import pandas as pd
    
    columns = Columns()
    
    # Make a copy and ensure it's timezone-naive first
    obs_tz = observations.copy()
    obs_tz['datetime'] = pd.to_datetime(obs_tz['datetime'])
    
    # Remove timezone if it exists, then localize to America/Denver
    if obs_tz['datetime'].dt.tz is not None:
        obs_tz['datetime'] = obs_tz['datetime'].dt.tz_localize(None)
    obs_tz['datetime'] = obs_tz['datetime'].dt.tz_localize('America/Denver')
    
    dat = obs_tz.merge(elements, on=["station", "element"], how="left")
    
    # Should not raise errors
    result = ck.check_range_pd(dat, columns)
    
    assert 'qa_range' in result.columns
    assert result['datetime'].dt.tz is not None, "Timezone should be preserved"


def test_qa_flag_combinations():
    """Test that multiple QA checks can be combined correctly."""
    import pandas as pd
    
    columns = Columns()
    
    # Create data that will fail multiple checks
    data = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=5, freq='5min'),
        'station': ['test'] * 5,
        'element': ['air_temp'] * 5,
        'value': [50.0, 150.0, 150.0, 150.0, 50.0],  # Out of range and sudden jumps
        'range_min': [32.0] * 5,
        'range_max': [120.0] * 5,
        'step_size': [10.0] * 5
    })
    
    # Apply multiple checks
    result = ck.check_range_pd(data, columns)
    result = ck.check_step_pd(result, columns)
    
    # Verify both QA columns exist
    assert 'qa_range' in result.columns
    assert 'qa_step' in result.columns
    
    # The 150.0 values should fail range check
    range_failures = result[result['value'] == 150.0]['qa_range'].values
    assert all(range_failures == 1), "Out of range values should be flagged"
    
    # Large jumps should fail step check
    assert 1 in result['qa_step'].values, "Large steps should be flagged"


