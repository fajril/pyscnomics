from typing import Protocol

from matplotlib.axes import Axes
from matplotlib.figure import Figure

class _AxesGrid(Protocol):
    def __getitem__(self, index: tuple[int, int], /) -> Axes: ...

def subplots(
    nrows: int,
    ncols: int,
    *,
    figsize: tuple[int, int],
) -> tuple[Figure, _AxesGrid]: ...
def tight_layout(*, pad: int | float) -> None: ...
def savefig(fname: str) -> None: ...
