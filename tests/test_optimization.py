from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pytest

from pyscnomics.contracts.run_arguments import GrossSplitRunArguments
from pyscnomics.dataset.sample import load_data
from pyscnomics.econ.selection import (
    GrossSplitRegime,
    OptimizationParameter,
    OptimizationTarget,
)
from pyscnomics.optimize.optimization import (
    OptimizationException,
    adjust_useful_life_years,
    optimize_psc,
)


class OptimizationConfiguration(TypedDict):
    parameter: list[OptimizationParameter]
    min: npt.NDArray[np.float64]
    max: npt.NDArray[np.float64]


def _gross_split_arguments(project_duration: int) -> GrossSplitRunArguments:
    return {
        "effective_tax_rate": 0.22,
        "vat_rate": np.full(project_duration, 0.11, dtype=np.float64),
        "inflation_rate": 0.0,
        "inflation_rate_applied_to": None,
        "regime": GrossSplitRegime.PERMEN_ESDM_12_2020,
    }


def test_optimize_psc_vat_rate_preserves_base_array_and_reaches_target() -> None:
    contract = load_data("case3", "gross_split")
    assert contract is not None
    arguments = _gross_split_arguments(contract.project_duration)
    contract.run(**arguments)
    base_target = float(contract.get_summary()["ctr_irr"])
    configuration: OptimizationConfiguration = {
        "parameter": [OptimizationParameter.VAT_RATE],
        "min": np.array([0.05], dtype=np.float64),
        "max": np.array([0.15], dtype=np.float64),
    }

    names, values, result, executed = optimize_psc(
        configuration,
        load_data("case3", "gross_split"),
        arguments,
        base_target,
        {},
        OptimizationTarget.IRR,
    )

    assert names == ["VAT Rate"]
    assert len(values) == 1
    assert isinstance(values[0], list)
    assert len(values[0]) == contract.project_duration
    np.testing.assert_allclose(values[0], 0.11)
    assert result == base_target
    assert len(executed) == 1


def test_optimize_psc_vat_rate_numpy_target_is_bounded_and_close() -> None:
    contract = load_data("case3", "gross_split")
    assert contract is not None
    arguments = _gross_split_arguments(contract.project_duration)
    contract.run(**arguments)
    target = np.float64(float(contract.get_summary()["ctr_irr"]) + 0.01)
    configuration: OptimizationConfiguration = {
        "parameter": [OptimizationParameter.VAT_RATE],
        "min": np.array([0.05], dtype=np.float64),
        "max": np.array([0.15], dtype=np.float64),
    }

    names, values, result, executed = optimize_psc(
        configuration,
        load_data("case3", "gross_split"),
        arguments,
        target,
        {},
        OptimizationTarget.IRR,
    )

    assert names == ["VAT Rate"]
    assert isinstance(values[0], list)
    assert len(values[0]) == contract.project_duration
    assert all(0.05 <= value <= 0.15 for value in values[0])
    assert result is not None
    assert abs(float(result) - float(target)) <= 1e-4
    assert len(executed) == 1


@pytest.mark.parametrize("array_dtype", [np.int64, np.float64])
@pytest.mark.parametrize("adjustment", [0.25, np.float64(0.25)])
def test_adjust_useful_life_years_preserves_shape_and_returns_integer_years(
    array_dtype: type[np.int64] | type[np.float64],
    adjustment: float,
) -> None:
    if array_dtype is np.int64:
        useful_life = np.array([2, 3, 5, 12], dtype=np.int64)
    else:
        useful_life = np.array([2, 3, 5, 12], dtype=np.float64)

    adjusted = adjust_useful_life_years(adjustment, useful_life)

    np.testing.assert_array_equal(adjusted, [2, 3, 4, 9])
    assert adjusted.dtype == np.dtype(np.int64)
    assert adjusted.shape == (4,)


@pytest.mark.parametrize("array_dtype", [np.int64, np.float64])
def test_adjust_useful_life_years_rejects_values_below_two(
    array_dtype: type[np.int64] | type[np.float64],
) -> None:
    with pytest.raises(
        OptimizationException,
        match=r"Useful life contains values below 2 years: \[1(?:\.\d*)?\]",
    ):
        if array_dtype is np.int64:
            _ = adjust_useful_life_years(0.25, np.array([1, 2], dtype=np.int64))
        else:
            _ = adjust_useful_life_years(0.25, np.array([1, 2], dtype=np.float64))
