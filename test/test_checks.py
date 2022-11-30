from pyqc.checks import check_range, check_step, check_variance


def test_check_range():
    assert check_range(5, 0, 6) == 0, "Value in valid range fails test."
    assert check_range(-1, 0, 6) == 1, "Value outside of valid range fails test."


def test_check_step():
    assert check_step(4, 5, 2) == 0, "Values within valid step size fail test."
    assert check_step(4, 8, 2) == 1, "Values outside of valid step size fail test."


def test_check_variance():
    assert check_variance([1, 2, 1, 2, 1, 2], 0.25) == 0, "Std dev check fails when it should pass."
    assert check_variance([1, 2, 1, 2, 1, 2], 1) == 1, "Std dev check passes when it should fail."