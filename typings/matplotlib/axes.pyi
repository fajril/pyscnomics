from typing import Literal

import numpy as np
from numpy.typing import NDArray

from matplotlib.ticker import Locator

class Axis:
    def set_major_locator(self, locator: Locator) -> None: ...

class Axes:
    xaxis: Axis

    def set_title(self, label: str) -> object: ...
    def plot(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        drawstyle: Literal["steps-post"],
    ) -> list[object]: ...
    def set_xlabel(self, xlabel: str, *, fontsize: int, labelpad: int) -> object: ...
    def set_ylabel(self, ylabel: str, *, fontsize: int, labelpad: int) -> object: ...
    def grid(
        self,
        *,
        which: Literal["major", "minor"],
        linewidth: int | float,
    ) -> None: ...
    def minorticks_on(self) -> None: ...
