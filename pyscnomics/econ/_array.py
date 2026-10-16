"""Small typed helpers for NumPy float arrays."""

from typing import SupportsFloat, cast

import numpy as np
import numpy.typing as npt


def sum_float64(values: npt.NDArray[np.float64]) -> np.float64:
    """Sum float64 values with NumPy's reduction and scalar semantics."""
    # NumPy's stub returns Any here; subclasses may return a zero-dimensional array.
    return np.float64(cast(SupportsFloat, np.sum(values, dtype=np.float64)))


def last_float64(values: npt.NDArray[np.float64]) -> np.float64:
    """Return the final C-order element, preserving the empty-take error."""
    if values.size == 0:
        raise IndexError("cannot do a non-empty take from an empty axes.")
    return np.float64(values.item(-1))
