import numpy as np
from typing import List

Numeric = float | int | np.number

def check_range(x: Numeric, low: Numeric, high: Numeric) -> int:
    if x > low and x < high:
        return 0
    return 1


def check_step(x: Numeric, prev: Numeric, threshold: Numeric) -> int:
    if abs(prev - x) < threshold:
        return 0
    return 1


def check_variance(x: np.ndarray | List[Numeric], threshold: Numeric) -> int:
    if np.std(x) > threshold:
        return 0
    return 1