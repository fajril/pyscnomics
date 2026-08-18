from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from pyscnomics.contracts.project import BaseProject
from pyscnomics.dataset.sample import read_fluid_type
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import FluidType
from pyscnomics.api.converter import SensitivityBM
from pyscnomics.optimize.sensitivity import (
    SensitivityException,
    _adjust_contract,
    _get_multipliers,
    sensitivity_psc,
)
from pyscnomics.optimize.uncertainty import min_mean_max_retriever


def test_generic_lifting_adjustment_scales_every_producer_type():
    lifting = tuple(
        Lifting(
            start_year=2023,
            end_year=2024,
            prod_year=np.array([2023, 2024]),
            lifting_rate=np.array(values),
            price=np.ones(2),
            fluid_type=fluid_type,
        )
        for fluid_type, values in (
            (FluidType.OIL, [100, 100]),
            (FluidType.GAS, [20, 20]),
            (FluidType.SULFUR, [5, 5]),
        )
    )
    contract = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
        approval_year=2023,
        lifting=lifting,
    )

    adjusted = _adjust_contract(
        contract=contract,
        contract_arguments={},
        element="LIFTING",
        adjustment_value=2.0,
    )

    np.testing.assert_allclose(adjusted.lifting[0].lifting_rate, [200, 200])
    np.testing.assert_allclose(adjusted.lifting[1].lifting_rate, [40, 40])
    np.testing.assert_allclose(adjusted.lifting[2].lifting_rate, [10, 10])


def test_sensitivity_result_includes_generic_lifting():
    class Summary:
        def get_summary(self, **_kwargs):
            return {indicator: 1.0 for indicator in (
                "ctr_npv",
                "ctr_irr",
                "ctr_pi",
                "ctr_pot",
                "gov_take",
                "ctr_net_share",
            )}

    with patch(
        "pyscnomics.optimize.sensitivity._adjust_contract",
        return_value=Summary(),
    ):
        result = sensitivity_psc(
            contract=object(),
            contract_arguments={},
            summary_arguments={},
            min_deviation=0.1,
            max_deviation=0.1,
            dataframe_output=False,
            step=2,
        )

    assert "LIFTING" in result["ctr_npv"][1.0]


def test_lpg_lifting_uses_backend_gas_aggregate():
    assert read_fluid_type("LPG propane") is FluidType.GAS
    assert read_fluid_type("LPG butane") is FluidType.GAS


def test_increment_generates_uniform_multipliers_across_asymmetric_range():
    multipliers = _get_multipliers(
        min_deviation=0.3,
        max_deviation=0.8,
        base_value=1.0,
        increment=0.01,
    )

    assert len(multipliers) == 111
    np.testing.assert_allclose(multipliers[0], 0.70)
    np.testing.assert_allclose(multipliers[-1], 1.80)
    assert 1.0 in multipliers
    np.testing.assert_allclose(np.diff(multipliers), 0.01)


def test_increment_always_includes_base_value():
    multipliers = _get_multipliers(
        min_deviation=0.333,
        max_deviation=0.777,
        base_value=1.0,
        increment=0.01,
    )

    assert 1.0 in multipliers
    assert len(multipliers) == round(0.333 / 0.01) + round(0.777 / 0.01) + 1


def test_step_path_unchanged_when_increment_absent():
    multipliers = _get_multipliers(
        min_deviation=0.2,
        max_deviation=0.2,
        base_value=1.0,
        step=10,
    )

    assert len(multipliers) == 21
    np.testing.assert_allclose(np.diff(multipliers), 0.02)


def test_increment_must_be_positive():
    with pytest.raises(SensitivityException):
        _get_multipliers(min_deviation=0.1, max_deviation=0.1, increment=0)


def test_sensitivity_result_with_increment_includes_every_multiplier():
    class Summary:
        def get_summary(self, **_kwargs):
            return {indicator: 1.0 for indicator in (
                "ctr_npv",
                "ctr_irr",
                "ctr_pi",
                "ctr_pot",
                "gov_take",
                "ctr_net_share",
            )}

    with patch(
        "pyscnomics.optimize.sensitivity._adjust_contract",
        return_value=Summary(),
    ):
        result = sensitivity_psc(
            contract=object(),
            contract_arguments={},
            summary_arguments={},
            min_deviation=0.3,
            max_deviation=0.8,
            base_value=1.0,
            increment=0.01,
            dataframe_output=False,
        )

    assert len(result["ctr_npv"]) == 111
    assert 1.0 in result["ctr_npv"]
    assert "LIFTING" in result["ctr_npv"][1.0]


def test_sensitivity_bm_accepts_increment_without_step():
    args = SensitivityBM(min_deviation=0.3, max_deviation=0.8, increment=0.01)

    dumped = args.model_dump()
    assert dumped["increment"] == 0.01
    assert dumped["step"] == 10


def test_sensitivity_bm_accepts_step_without_increment():
    args = SensitivityBM(min_deviation=0.2, max_deviation=0.2, step=10)

    assert args.model_dump()["increment"] is None


def test_uncertainty_lifting_statistics_include_every_producer_type():
    oil = SimpleNamespace(lifting_rate=np.array([100.0]), price=np.array([10.0]))
    gas = SimpleNamespace(lifting_rate=np.array([20.0]), price=np.array([1.0]))
    sulfur = SimpleNamespace(lifting_rate=np.array([5.0]), price=np.array([0.1]))
    contract = SimpleNamespace(
        capital_cost_total=SimpleNamespace(cost=np.array([1.0])),
        intangible_cost_total=SimpleNamespace(cost=np.array([1.0])),
        opex_total=SimpleNamespace(cost=np.array([1.0])),
        asr_cost_total=SimpleNamespace(cost=np.array([1.0])),
        lbt_cost_total=SimpleNamespace(cost=np.array([1.0])),
        cost_of_sales_total=SimpleNamespace(cost=np.array([1.0])),
        lifting=(oil, gas, sulfur),
        _oil_lifting=oil,
        _gas_lifting=gas,
    )

    result = min_mean_max_retriever(contract)

    assert result["min_lifting"] == 5.0
    assert result["mean_lifting"] == 125 / 3
    assert result["max_lifting"] == 100.0
