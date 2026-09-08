import numpy as np
import numpy.typing as npt

class _UniformDistribution:
    def rvs(
        self, *, loc: float, scale: float, size: int
    ) -> npt.NDArray[np.float64]: ...

class _TriangularDistribution:
    def rvs(
        self, *, c: float, loc: float, scale: float, size: int
    ) -> npt.NDArray[np.float64]: ...

class _TruncatedNormalDistribution:
    def rvs(
        self, *, a: float, b: float, loc: float, scale: float, size: int
    ) -> npt.NDArray[np.float64]: ...

uniform: _UniformDistribution
triang: _TriangularDistribution
truncnorm: _TruncatedNormalDistribution
