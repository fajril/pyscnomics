from collections.abc import Callable
from typing import Literal

import numpy as np

class OptimizeResult(dict[str, object]):
    fun: float
    x: float

def minimize_scalar(
    fun: Callable[[float], float | np.float64],
    *,
    bounds: tuple[float | np.float64, float | np.float64],
    method: Literal["bounded"],
) -> OptimizeResult: ...
