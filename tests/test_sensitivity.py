from datetime import date
from typing import Literal, TypeGuard, TypeVar
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from pyscnomics.api.converter import SensitivityBM
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.run_arguments import (
    GrossSplitRunArguments,
    validate_cost_recovery_run_arguments,
    validate_gross_split_run_arguments,
)
from pyscnomics.contracts.transition import Transition
from pyscnomics.dataset.sample import load_data, read_fluid_type
from pyscnomics.econ.costs import ASR, CapitalCost, CostOfSales, Intangible, LBT, OPEX
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import (
    DeprMethod,
    DiscountingMode,
    FTPTaxRegime,
    FluidType,
    GrossSplitRegime,
    InflationAppliedTo,
    NPVSelection,
    OtherRevenue,
    TaxRegime,
)
from pyscnomics.optimize.adjuster import (
    AdjustData,
    AdjustDataInput,
    ArgumentsState,
    ContractArguments,
    FiscalConfigData,
    GeneralConfigData,
    GrossSplitConfig,
    PSCKey,
    get_intangible_adjustment,
    get_opex_adjustment,
    get_price_and_rate_adjustment,
    get_tangible_adjustment,
)
from pyscnomics.optimize.sensitivity import (
    SensitivityException,
    adjust_sensitivity_contract,
    get_sensitivity_multipliers,
    sensitivity_psc,
)
from pyscnomics.optimize.uncertainty import (
    get_multipliers_montecarlo,
    get_summary_dict,
    min_mean_max_retriever,
    uncertainty_psc,
)
from pyscnomics.tools.summary import get_summary


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

    adjusted = adjust_sensitivity_contract(
        contract=contract,
        contract_arguments={},
        element="LIFTING",
        adjustment_value=2.0,
    )

    assert adjusted.lifting is not None
    np.testing.assert_allclose(adjusted.lifting[0].lifting_rate, [200, 200])
    np.testing.assert_allclose(adjusted.lifting[1].lifting_rate, [40, 40])
    np.testing.assert_allclose(adjusted.lifting[2].lifting_rate, [10, 10])


def _make_opex() -> OPEX:
    return OPEX(
        start_year=2023,
        end_year=2024,
        expense_year=np.array([2023, 2024]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
        cost_type=[None, None],
        description=["fixed", "variable"],
        tax_portion=np.array([0.5, 0.75]),
        tax_discount=np.array([0.1, 0.2]),
        fixed_cost=np.array([100.0, 200.0]),
        prod_rate=np.array([10.0, 20.0]),
        cost_per_volume=np.array([1.0, 2.0]),
    )


def test_opex_adjustment_preserves_current_dataclass_fields_for_single_contract():
    original = _make_opex()

    adjusted = get_opex_adjustment(
        contract_type="PSC Cost Recovery (CR)",
        opex_aggregate=(original,),
        opex_multiplier=2.0,
    )

    assert isinstance(adjusted, tuple)
    result = adjusted[0]
    assert result.fixed_cost is not None
    assert result.cost_per_volume is not None
    assert result.expense_year is not None
    assert original.expense_year is not None
    assert result.tax_portion is not None
    assert original.tax_portion is not None
    assert result.tax_discount is not None
    assert original.tax_discount is not None
    np.testing.assert_allclose(result.fixed_cost, [200.0, 400.0])
    np.testing.assert_allclose(result.cost_per_volume, [2.0, 4.0])
    np.testing.assert_array_equal(result.expense_year, original.expense_year)
    assert result.cost_allocation == original.cost_allocation
    assert result.cost_type == original.cost_type
    assert result.description == original.description
    np.testing.assert_array_equal(result.tax_portion, original.tax_portion)
    np.testing.assert_array_equal(result.tax_discount, original.tax_discount)


def test_opex_adjustment_preserves_current_dataclass_fields_for_transition():
    original = _make_opex()
    aggregate: dict[Literal["PSC 1", "PSC 2"], tuple[OPEX]] = {
        "PSC 1": (original,),
        "PSC 2": (_make_opex(),),
    }

    adjusted = get_opex_adjustment(
        contract_type="Transition CR - GS",
        opex_aggregate=aggregate,
        opex_multiplier=2.0,
    )

    assert isinstance(adjusted, dict)
    assert set(adjusted) == {"PSC 1", "PSC 2"}
    for psc in ("PSC 1", "PSC 2"):
        result = adjusted[psc][0]
        assert result.fixed_cost is not None
        assert result.cost_per_volume is not None
        assert result.expense_year is not None
        assert result.tax_portion is not None
        assert result.tax_discount is not None
        np.testing.assert_allclose(result.fixed_cost, [200.0, 400.0])
        np.testing.assert_allclose(result.cost_per_volume, [2.0, 4.0])
        np.testing.assert_array_equal(result.expense_year, [2023, 2024])
        assert result.cost_allocation == [FluidType.OIL, FluidType.OIL]
        assert result.cost_type == [None, None]
        assert result.description == ["fixed", "variable"]
        np.testing.assert_array_equal(result.tax_portion, [0.5, 0.75])
        np.testing.assert_array_equal(result.tax_discount, [0.1, 0.2])


def test_price_adjustment_preserves_lifting_baseline_and_input_arrays():
    original = Lifting(
        start_year=2023,
        end_year=2024,
        prod_year=np.array([2023, 2024]),
        lifting_rate=np.array([10.0, 20.0]),
        price=np.array([3.0, 4.0]),
        fluid_type=FluidType.OIL,
        prod_rate=np.array([40.0, 70.0]),
        prod_rate_baseline=np.array([1.0, 2.0]),
    )

    adjusted = get_price_and_rate_adjustment(
        contract_type="PSC Cost Recovery (CR)",
        lifting_aggregate=(original,),
        price_multiplier=2.0,
        rate_multiplier=3.0,
    )

    assert isinstance(adjusted, tuple)
    result = adjusted[0]
    assert result.prod_rate is not None
    assert result.prod_rate_baseline is not None
    assert original.prod_rate is not None
    assert original.prod_rate_baseline is not None
    np.testing.assert_allclose(result.price, [6.0, 8.0])
    np.testing.assert_allclose(result.lifting_rate, [30.0, 60.0])
    np.testing.assert_array_equal(result.prod_rate, original.prod_rate)
    np.testing.assert_array_equal(
        result.prod_rate_baseline, original.prod_rate_baseline
    )
    np.testing.assert_array_equal(original.price, [3.0, 4.0])
    np.testing.assert_array_equal(original.lifting_rate, [10.0, 20.0])


def test_cost_adjustments_preserve_current_capital_and_intangible_fields():
    capital = CapitalCost(
        start_year=2023,
        end_year=2024,
        expense_year=np.array([2023]),
        cost=np.array([100.0]),
        cost_allocation=[FluidType.OIL],
        cost_type=[None],
        description=["capital"],
        tax_portion=np.array([0.5]),
        tax_discount=np.array([0.2]),
        pis_year=np.array([2023]),
        salvage_value=np.array([5.0]),
        useful_life=np.array([4.0]),
        depreciation_factor=np.array([0.6]),
        is_ic_applied=[True],
    )
    intangible = Intangible(
        start_year=2023,
        end_year=2024,
        expense_year=np.array([2023]),
        cost=np.array([40.0]),
        cost_allocation=[FluidType.GAS],
        cost_type=[None],
        description=["intangible"],
        tax_portion=np.array([0.25]),
        tax_discount=np.array([0.1]),
    )

    adjusted_capital_aggregate = get_tangible_adjustment(
        "PSC Cost Recovery (CR)", (capital,), 2.0
    )
    adjusted_intangible_aggregate = get_intangible_adjustment(
        "PSC Cost Recovery (CR)", (intangible,), 2.0
    )
    assert isinstance(adjusted_capital_aggregate, tuple)
    assert isinstance(adjusted_intangible_aggregate, tuple)
    adjusted_capital = adjusted_capital_aggregate[0]
    adjusted_intangible = adjusted_intangible_aggregate[0]
    assert adjusted_capital.cost is not None
    assert adjusted_capital.salvage_value is not None
    assert capital.cost_type is not None
    assert adjusted_capital.tax_portion is not None
    assert capital.tax_portion is not None
    assert adjusted_capital.tax_discount is not None
    assert capital.tax_discount is not None
    assert adjusted_capital.pis_year is not None
    assert capital.pis_year is not None
    assert adjusted_capital.useful_life is not None
    assert capital.useful_life is not None
    assert adjusted_capital.depreciation_factor is not None
    assert capital.depreciation_factor is not None
    assert adjusted_intangible.cost is not None
    assert adjusted_intangible.tax_portion is not None
    assert intangible.tax_portion is not None
    assert adjusted_intangible.tax_discount is not None
    assert intangible.tax_discount is not None

    np.testing.assert_allclose(adjusted_capital.cost, [200.0])
    np.testing.assert_allclose(adjusted_capital.salvage_value, [5.0])
    assert adjusted_capital.cost_type == capital.cost_type
    assert adjusted_capital.description == capital.description
    np.testing.assert_array_equal(adjusted_capital.tax_portion, capital.tax_portion)
    np.testing.assert_array_equal(adjusted_capital.tax_discount, capital.tax_discount)
    np.testing.assert_array_equal(adjusted_capital.pis_year, capital.pis_year)
    np.testing.assert_array_equal(adjusted_capital.useful_life, capital.useful_life)
    np.testing.assert_array_equal(
        adjusted_capital.depreciation_factor, capital.depreciation_factor
    )
    assert adjusted_capital.is_ic_applied == capital.is_ic_applied
    np.testing.assert_allclose(adjusted_intangible.cost, [80.0])
    assert adjusted_intangible.cost_type == intangible.cost_type
    assert adjusted_intangible.description == intangible.description
    np.testing.assert_array_equal(
        adjusted_intangible.tax_portion, intangible.tax_portion
    )
    np.testing.assert_array_equal(
        adjusted_intangible.tax_discount, intangible.tax_discount
    )


def _make_adjuster_lifting(
    fluid_type: FluidType,
    start_year: int = 2023,
    end_year: int = 2024,
) -> Lifting:
    return Lifting(
        start_year=start_year,
        end_year=end_year,
        prod_year=np.array([start_year, end_year]),
        lifting_rate=np.ones(2),
        price=np.ones(2),
        fluid_type=fluid_type,
        prod_rate=np.ones(2),
        prod_rate_baseline=np.ones(2),
    )


_Value = TypeVar("_Value")
_PSC_KEYS: tuple[PSCKey, PSCKey] = ("PSC 1", "PSC 2")


def _psc_mapping(value: _Value) -> dict[PSCKey, _Value]:
    return {"PSC 1": value, "PSC 2": value}


def _is_single_arguments(
    value: ArgumentsState | None,
) -> TypeGuard[ContractArguments]:
    return isinstance(value, dict) and "decline_factor" in value


def _is_transition_arguments(
    value: ArgumentsState | None,
) -> TypeGuard[ContractArguments]:
    return isinstance(value, dict) and "unrec_portion" in value


def _make_adjuster_input(transition: bool) -> AdjustDataInput:
    fluid_types = (
        FluidType.OIL,
        FluidType.GAS,
        FluidType.SULFUR,
        FluidType.ELECTRICITY,
        FluidType.CO2,
    )
    lifting_aggregates: list[tuple[Lifting, ...] | dict[PSCKey, tuple[Lifting, ...]]]
    if transition:
        lifting_by_psc: dict[PSCKey, tuple[Lifting, ...]] = {
            "PSC 1": tuple(
                _make_adjuster_lifting(fluid, start_year=2023, end_year=2024)
                for fluid in fluid_types
            ),
            "PSC 2": tuple(
                _make_adjuster_lifting(fluid, start_year=2025, end_year=2026)
                for fluid in fluid_types
            ),
        }
        lifting_aggregates = [
            {psc: (lifting_by_psc[psc][index],) for psc in _PSC_KEYS}
            for index in range(5)
        ]
    else:
        lifting = tuple(_make_adjuster_lifting(fluid) for fluid in fluid_types)
        lifting_aggregates = [(item,) for item in lifting]

    def zero_costs(
        start_year: int, end_year: int
    ) -> tuple[
        tuple[CapitalCost, ...],
        tuple[Intangible, ...],
        tuple[OPEX, ...],
        tuple[ASR, ...],
    ]:
        expense_year = np.array([start_year])
        return (
            (
                CapitalCost(
                    start_year=start_year,
                    end_year=end_year,
                    expense_year=expense_year,
                    cost=np.array([0.0]),
                    cost_allocation=[FluidType.OIL],
                    pis_year=expense_year,
                ),
            ),
            (
                Intangible(
                    start_year=start_year,
                    end_year=end_year,
                    expense_year=expense_year,
                    cost=np.array([0.0]),
                    cost_allocation=[FluidType.OIL],
                ),
            ),
            (
                OPEX(
                    start_year=start_year,
                    end_year=end_year,
                    expense_year=expense_year,
                    cost_allocation=[FluidType.OIL],
                    fixed_cost=np.array([0.0]),
                    prod_rate=np.array([0.0]),
                    cost_per_volume=np.array([0.0]),
                ),
            ),
            (
                ASR(
                    start_year=start_year,
                    end_year=end_year,
                    expense_year=expense_year,
                    cost=np.array([0.0]),
                    cost_allocation=[FluidType.OIL],
                ),
            ),
        )

    first_costs = zero_costs(2023, 2024)
    second_costs = zero_costs(2025, 2026)
    capital_cost_aggregate: (
        tuple[CapitalCost, ...] | dict[PSCKey, tuple[CapitalCost, ...]]
    )
    intangible_cost_aggregate: (
        tuple[Intangible, ...] | dict[PSCKey, tuple[Intangible, ...]]
    )
    opex_aggregate: tuple[OPEX, ...] | dict[PSCKey, tuple[OPEX, ...]]
    asr_cost_aggregate: tuple[ASR, ...] | dict[PSCKey, tuple[ASR, ...]]
    if transition:
        capital_cost_aggregate = {
            "PSC 1": first_costs[0],
            "PSC 2": second_costs[0],
        }
        intangible_cost_aggregate = {
            "PSC 1": first_costs[1],
            "PSC 2": second_costs[1],
        }
        opex_aggregate = {"PSC 1": first_costs[2], "PSC 2": second_costs[2]}
        asr_cost_aggregate = {"PSC 1": first_costs[3], "PSC 2": second_costs[3]}
    else:
        capital_cost_aggregate = first_costs[0]
        intangible_cost_aggregate = first_costs[1]
        opex_aggregate = first_costs[2]
        asr_cost_aggregate = first_costs[3]

    if transition:
        fiscal_config = FiscalConfigData(
            npv_mode=NPVSelection.NPV_SKK_REAL_TERMS,
            discounting_mode=DiscountingMode.END_YEAR,
            inflation_rate=_psc_mapping(0.03),
            tax_mode=_psc_mapping(TaxRegime.NAILED_DOWN),
            tax_rate=_psc_mapping(0.22),
            tax_payment_config=_psc_mapping(FTPTaxRegime.PDJP_20_2017),
            asr_future_rate=_psc_mapping(0.0),
            depreciation_method=_psc_mapping(DeprMethod.SL),
            decline_factor={"PSC 1": 0.72, "PSC 2": 0.81},
            vat_rate=_psc_mapping(0.0),
            lbt_rate=_psc_mapping(0.0),
            sulfur_revenue_config=_psc_mapping(OtherRevenue.ADDITION_TO_GAS_REVENUE),
            electricity_revenue_config=_psc_mapping(
                OtherRevenue.ADDITION_TO_OIL_REVENUE
            ),
            co2_revenue_config=_psc_mapping(OtherRevenue.ADDITION_TO_GAS_REVENUE),
            transferred_unrec_cost=0.0,
            sunk_cost_reference_year=2023,
        )
    else:
        fiscal_config = FiscalConfigData(
            npv_mode=NPVSelection.NPV_SKK_REAL_TERMS,
            discounting_mode=DiscountingMode.END_YEAR,
            inflation_rate=0.03,
            tax_mode=TaxRegime.NAILED_DOWN,
            tax_rate=0.22,
            tax_payment_config=FTPTaxRegime.PDJP_20_2017,
            asr_future_rate=0.0,
            depreciation_method=DeprMethod.SL,
            decline_factor=0.72,
            vat_rate=0.0,
            lbt_rate=0.0,
            sulfur_revenue_config=OtherRevenue.ADDITION_TO_GAS_REVENUE,
            electricity_revenue_config=OtherRevenue.ADDITION_TO_OIL_REVENUE,
            co2_revenue_config=OtherRevenue.ADDITION_TO_GAS_REVENUE,
            transferred_unrec_cost=0.0,
            sunk_cost_reference_year=2023,
        )
    gross_split_config = GrossSplitConfig(
        dmo_is_weighted=False,
        oil_dmo_volume=0.25,
        oil_dmo_fee=1.0,
        oil_dmo_period=60,
        gas_dmo_volume=1.0,
        gas_dmo_fee=1.0,
        gas_dmo_period=60,
        field_status="No POD",
        field_location="Onshore",
        reservoir_depth="<=2500",
        infrastructure_availability="Well Developed",
        reservoir_type="Conventional",
        oil_api="<25",
        domestic_content_use="50<=x<70",
        production_stage="Secondary",
        co2_content="<5",
        h2s_content="<100",
        ministry_discretion_split=0.0,
    )
    contract_type = "Transition GS - GS" if transition else "PSC Gross Split (GS)"
    general_config = GeneralConfigData(
        type_of_contract=contract_type,
        discount_rate_start_year=2023,
        discount_rate=0.1,
        inflation_rate_applied_to=None,
        start_date_project=date(2023, 1, 1),
        end_date_project=date(2024, 12, 31),
        oil_onstream_date=None,
        gas_onstream_date=None,
        start_date_project_second=date(2025, 1, 1),
        end_date_project_second=date(2026, 12, 31),
        oil_onstream_date_second=None,
        gas_onstream_date_second=None,
    )
    return AdjustDataInput(
        general_config_data=general_config,
        fiscal_config_data=fiscal_config,
        oil_lifting_aggregate_total=lifting_aggregates[0],
        gas_lifting_aggregate_total=lifting_aggregates[1],
        sulfur_lifting_aggregate=lifting_aggregates[2],
        electricity_lifting_aggregate=lifting_aggregates[3],
        co2_lifting_aggregate=lifting_aggregates[4],
        capital_cost_aggregate=capital_cost_aggregate,
        intangible_cost_aggregate=intangible_cost_aggregate,
        opex_aggregate=opex_aggregate,
        asr_cost_aggregate=asr_cost_aggregate,
        psc_cr_data=None,
        psc_gs_data=gross_split_config,
        psc_transition_cr_to_cr=None,
        psc_transition_cr_to_gs=None,
        psc_transition_gs_to_gs={
            "Gross Split Config": gross_split_config,
            "Gross Split Config (2)": gross_split_config,
        },
        psc_transition_gs_to_cr=None,
    )


def test_gross_split_decline_factor_keeps_single_and_transition_mappings_distinct():
    single = AdjustData(
        multipliers=np.ones(5),
        data=_make_adjuster_input(transition=False),
    )
    _, single_arguments, _ = single.activate()

    assert _is_single_arguments(single_arguments)
    assert single_arguments["decline_factor"] == 0.72

    transition = AdjustData(
        multipliers=np.ones(5),
        data=_make_adjuster_input(transition=True),
    )
    transition_result = transition.activate()
    assert transition_result[0] is transition.transition

    assert isinstance(transition.psc_arguments, dict)
    assert isinstance(transition.psc_arguments["PSC 1"], dict)
    assert isinstance(transition.psc_arguments["PSC 2"], dict)
    assert transition.psc_arguments["PSC 1"]["decline_factor"] == 0.72
    assert transition.psc_arguments["PSC 2"]["decline_factor"] == 0.81


def test_adjust_data_arguments_execute_current_gross_split_contracts():
    single = AdjustData(
        multipliers=np.ones(5), data=_make_adjuster_input(transition=False)
    )
    single_contract, single_arguments, _ = single.activate()
    assert isinstance(single_contract, GrossSplit)
    assert _is_single_arguments(single_arguments)
    single_contract.run(**validate_gross_split_run_arguments(single_arguments))

    transition = AdjustData(
        multipliers=np.ones(5), data=_make_adjuster_input(transition=True)
    )
    transition_contract, transition_arguments, _ = transition.activate()
    assert isinstance(transition_contract, Transition)
    assert _is_transition_arguments(transition_arguments)
    unrec_portion = transition_arguments["unrec_portion"]
    assert isinstance(unrec_portion, (float, int))
    transition_contract.run(unrec_portion=unrec_portion)


def test_sensitivity_result_includes_generic_lifting():
    contract = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
        approval_year=2023,
    )
    summary = {
        indicator: 1.0
        for indicator in (
            "ctr_npv",
            "ctr_irr",
            "ctr_pi",
            "ctr_pot",
            "gov_take",
            "ctr_net_share",
        )
    }

    with (
        patch(
            "pyscnomics.optimize.sensitivity.adjust_sensitivity_contract",
            return_value=contract,
        ),
        patch("pyscnomics.optimize.sensitivity._get_summary", return_value=summary),
    ):
        result = sensitivity_psc(
            contract=contract,
            contract_arguments={},
            summary_arguments={},
            min_deviation=0.1,
            max_deviation=0.1,
            dataframe_output=False,
            step=2,
        )

    base_result = next(
        values for key, values in result["ctr_npv"].items() if key == 1.0
    )
    assert "LIFTING" in base_result


def test_lpg_lifting_uses_backend_gas_aggregate():
    assert read_fluid_type("LPG propane") is FluidType.GAS
    assert read_fluid_type("LPG butane") is FluidType.GAS


def test_increment_generates_uniform_multipliers_across_asymmetric_range():
    multipliers = get_sensitivity_multipliers(
        min_deviation=0.3,
        max_deviation=0.8,
        base_value=1.0,
        increment=0.01,
    )

    assert len(multipliers) == 111
    np.testing.assert_allclose(multipliers.flat[0], 0.70)
    np.testing.assert_allclose(multipliers.flat[-1], 1.80)
    assert 1.0 in multipliers
    np.testing.assert_allclose(np.diff(multipliers), 0.01)


def test_increment_always_includes_base_value():
    multipliers = get_sensitivity_multipliers(
        min_deviation=0.333,
        max_deviation=0.777,
        base_value=1.0,
        increment=0.01,
    )

    assert 1.0 in multipliers
    assert len(multipliers) == round(0.333 / 0.01) + round(0.777 / 0.01) + 1


def test_step_path_unchanged_when_increment_absent():
    multipliers = get_sensitivity_multipliers(
        min_deviation=0.2,
        max_deviation=0.2,
        base_value=1.0,
        step=10,
    )

    assert len(multipliers) == 21
    np.testing.assert_allclose(np.diff(multipliers), 0.02)


def test_increment_must_be_positive():
    with pytest.raises(SensitivityException):
        _ = get_sensitivity_multipliers(
            min_deviation=0.1, max_deviation=0.1, increment=0
        )


def test_sensitivity_result_with_increment_includes_every_multiplier():
    contract = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
        approval_year=2023,
    )
    summary = {
        indicator: 1.0
        for indicator in (
            "ctr_npv",
            "ctr_irr",
            "ctr_pi",
            "ctr_pot",
            "gov_take",
            "ctr_net_share",
        )
    }

    with (
        patch(
            "pyscnomics.optimize.sensitivity.adjust_sensitivity_contract",
            return_value=contract,
        ),
        patch("pyscnomics.optimize.sensitivity._get_summary", return_value=summary),
    ):
        result = sensitivity_psc(
            contract=contract,
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
    base_result = next(
        values for key, values in result["ctr_npv"].items() if key == 1.0
    )
    assert "LIFTING" in base_result


def test_sensitivity_bm_accepts_increment_without_step():
    args = SensitivityBM(min_deviation=0.3, max_deviation=0.8, increment=0.01)

    dumped = args.model_dump()
    assert dumped["increment"] == 0.01
    assert dumped["step"] == 10


def test_sensitivity_bm_accepts_step_without_increment():
    args = SensitivityBM(min_deviation=0.2, max_deviation=0.2, step=10)

    assert args.model_dump()["increment"] is None


def test_sensitivity_bm_rejects_fractional_step():
    with pytest.raises(ValidationError):
        _ = SensitivityBM.model_validate(
            {"min_deviation": 0.2, "max_deviation": 0.2, "step": 10.5}
        )


def test_api_sensitivity_rejects_fractional_step():
    """The adapter rejects a non-integral multiplier-count before execution."""
    import pyscnomics.api.adapter as adapter

    contract = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
        approval_year=2023,
    )

    with patch.object(
        adapter,
        "get_baseproject",
        return_value=(
            {},
            contract,
            {},
            {
                "reference_year": 2022,
                "discount_rate_start_year": None,
                "inflation_rate": 0.03,
                "discount_rate": 0.1,
                "npv_mode": NPVSelection.NPV_SKK_REAL_TERMS,
                "discounting_mode": DiscountingMode.END_YEAR,
                "profitability_discounted": False,
            },
        ),
    ):
        with pytest.raises(
            TypeError, match="sensitivity_arguments.step must be an integer"
        ):
            _ = adapter.get_sensitivity(
                {
                    "sensitivity_arguments": {
                        "min_deviation": 0.1,
                        "max_deviation": 0.1,
                        "base_value": 1.0,
                        "step": 10.5,
                    }
                },
                "Base Project",
            )


def test_sensitivity_psc_accepts_integer_inflation_rate():
    contract = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
        approval_year=2023,
    )
    contract.run()

    with patch(
        "pyscnomics.optimize.sensitivity.adjust_sensitivity_contract",
        return_value=contract,
    ):
        result = sensitivity_psc(
            contract=contract,
            contract_arguments={},
            summary_arguments={"inflation_rate": 0},
            min_deviation=0.0,
            max_deviation=0.0,
            step=1,
            dataframe_output=False,
        )

    expected_indicators = {
        "ctr_npv",
        "ctr_irr",
        "ctr_pi",
        "ctr_pot",
        "gov_take",
        "ctr_net_share",
    }
    assert set(result) == expected_indicators
    assert all(1.0 in values for values in result.values())


def test_sensitivity_psc_rejects_bool_inflation_rate():
    contract = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
        approval_year=2023,
    )
    contract.run()

    with (
        patch(
            "pyscnomics.optimize.sensitivity.adjust_sensitivity_contract",
            return_value=contract,
        ),
        pytest.raises(SensitivityException, match="inflation_rate must be a float"),
    ):
        _ = sensitivity_psc(
            contract=contract,
            contract_arguments={},
            summary_arguments={"inflation_rate": True},
            min_deviation=0.0,
            max_deviation=0.0,
            step=1,
            dataframe_output=False,
        )


@pytest.mark.parametrize("argument_name", ["inflation_rate", "vat_rate"])
def test_sensitivity_psc_accepts_integer_zero_runtime_rates(
    argument_name: str,
) -> None:
    contract = load_data("case3", "gross_split")
    run_arguments: GrossSplitRunArguments = {
        "effective_tax_rate": 0.22,
        "vat_rate": 0.11,
        "inflation_rate": 0.0,
        "inflation_rate_applied_to": None,
        "regime": GrossSplitRegime.PERMEN_ESDM_12_2020,
    }
    contract.run(**run_arguments)

    zero_rate_arguments: dict[str, object] = {**run_arguments, argument_name: 0}
    float_rate_arguments: dict[str, object] = {
        **run_arguments,
        argument_name: 0.0,
    }
    integer_result = sensitivity_psc(
        contract=contract,
        contract_arguments=zero_rate_arguments,
        summary_arguments={},
        min_deviation=0,
        max_deviation=0,
        step=1,
        dataframe_output=False,
    )
    float_result = sensitivity_psc(
        contract=contract,
        contract_arguments=float_rate_arguments,
        summary_arguments={},
        min_deviation=0,
        max_deviation=0,
        step=1,
        dataframe_output=False,
    )

    assert set(integer_result) == {
        "ctr_npv",
        "ctr_irr",
        "ctr_pi",
        "ctr_pot",
        "gov_take",
        "ctr_net_share",
    }
    assert set(integer_result) == set(float_result)
    for indicator in integer_result:
        assert set(integer_result[indicator]) == {1.0}
        assert set(integer_result[indicator]) == set(float_result[indicator])
        assert integer_result[indicator] == float_result[indicator]


def test_run_argument_rate_normalization_preserves_arrays_and_rejects_invalid_types() -> None:
    rates = np.array([0.0, 0.1], dtype=np.float64)
    for validator in (
        validate_cost_recovery_run_arguments,
        validate_gross_split_run_arguments,
    ):
        normalized_integer = validator({"inflation_rate": 0}).get("inflation_rate")
        assert isinstance(normalized_integer, float)
        assert normalized_integer == 0.0
        assert validator({"inflation_rate": rates}).get("inflation_rate") is rates
        for invalid in (True, "0"):
            with pytest.raises(TypeError, match="inflation_rate must be a float"):
                _ = validator({"inflation_rate": invalid})


def test_gross_split_run_arguments_preserve_inflation_scope() -> None:
    normalized = validate_gross_split_run_arguments(
        {"inflation_rate_applied_to": InflationAppliedTo.OPEX}
    )

    assert normalized.get("inflation_rate_applied_to") is InflationAppliedTo.OPEX

    normalized_without_inflation = validate_gross_split_run_arguments(
        {"inflation_rate_applied_to": None}
    )
    assert "inflation_rate_applied_to" in normalized_without_inflation
    assert normalized_without_inflation["inflation_rate_applied_to"] is None

    with pytest.raises(TypeError, match="inflation_rate_applied_to has an invalid value"):
        _ = validate_gross_split_run_arguments(
            {"inflation_rate_applied_to": "OPEX"}
        )


def test_sensitivity_preserves_gross_split_inflation_scope() -> None:
    results = {}
    for target in (None, InflationAppliedTo.OPEX):
        contract = load_data("case3", "gross_split")
        assert isinstance(contract, GrossSplit)
        arguments: GrossSplitRunArguments = {
            "effective_tax_rate": 0.22,
            "vat_rate": 0.11,
            "inflation_rate": 0.05,
            "inflation_rate_applied_to": target,
            "regime": GrossSplitRegime.PERMEN_ESDM_12_2020,
        }
        contract_arguments: dict[str, object] = dict(arguments)
        sensitivity = sensitivity_psc(
            contract=contract,
            contract_arguments=contract_arguments,
            summary_arguments={},
            min_deviation=0.0,
            max_deviation=0.0,
            step=1,
            dataframe_output=False,
        )

        expected = load_data("case3", "gross_split")
        assert isinstance(expected, GrossSplit)
        expected.run(**arguments)
        expected_summary = get_summary(expected)
        base_multiplier = np.float64(1.0)
        assert (
            sensitivity["ctr_npv"][base_multiplier]["OPEX"]
            == expected_summary["ctr_npv"]
        )
        results[target] = sensitivity["ctr_npv"][base_multiplier]["OPEX"]

    assert results[InflationAppliedTo.OPEX] < results[None]


def test_uncertainty_summary_defaults_to_zero_inflation():
    arguments = get_summary_dict({"summary_arguments": {}})

    assert arguments["inflation_rate"] == 0.0


def test_uncertainty_summary_rejects_non_scalar_inflation():
    with pytest.raises(TypeError, match="inflation_rate must be a number"):
        _ = get_summary_dict({"summary_arguments": {"inflation_rate": [0.02, 0.03]}})


def test_uncertainty_psc_accepts_integer_bounds_for_gross_split():
    contract = GrossSplit(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
        approval_year=2023,
    )

    def run_with_bounds(bound: int | float) -> dict[str, object]:
        return uncertainty_psc(
            contract=contract,
            contract_arguments={},
            summary_arguments={},
            run_number=1,
            min_oil_price=bound,
            mean_oil_price=bound,
            max_oil_price=bound,
            min_gas_price=bound,
            mean_gas_price=bound,
            max_gas_price=bound,
            min_opex=bound,
            mean_opex=bound,
            max_opex=bound,
            min_capex=bound,
            mean_capex=bound,
            max_capex=bound,
            min_lifting=bound,
            mean_lifting=bound,
            max_lifting=bound,
            oil_price_stddev=bound,
            gas_price_stddev=bound,
            opex_stddev=bound,
            capex_stddev=bound,
            lifting_stddev=bound,
        )

    np.random.seed(1729)
    integer_result = run_with_bounds(1)
    np.random.seed(1729)
    float_result = run_with_bounds(1.0)

    assert integer_result == float_result
    assert isinstance(integer_result, dict)


def test_uncertainty_psc_rejects_bool_bounds():
    contract = GrossSplit(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
        approval_year=2023,
    )

    with pytest.raises(
        TypeError, match="uncertainty parameters must have numeric bounds"
    ):
        _ = uncertainty_psc(
            contract=contract,
            contract_arguments={},
            summary_arguments={},
            run_number=1,
            min_oil_price=True,
            mean_oil_price=1.0,
            max_oil_price=1.0,
        )


def test_uncertainty_lifting_statistics_include_every_producer_type():
    lifting = tuple(
        Lifting(
            start_year=2023,
            end_year=2023,
            prod_year=np.array([2023]),
            lifting_rate=np.array([rate]),
            price=np.array([price]),
            fluid_type=fluid_type,
        )
        for fluid_type, rate, price in (
            (FluidType.OIL, 100.0, 10.0),
            (FluidType.GAS, 20.0, 1.0),
            (FluidType.SULFUR, 5.0, 0.1),
        )
    )
    expense_year = np.array([2023])
    cost = np.array([1.0])
    allocation = [FluidType.OIL]
    contract = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        approval_year=2023,
        lifting=lifting,
        capital_cost=(
            CapitalCost(
                2023, 2023, expense_year, cost, allocation, pis_year=expense_year
            ),
        ),
        intangible_cost=(Intangible(2023, 2023, expense_year, cost, allocation),),
        opex=(
            OPEX(
                2023,
                2023,
                expense_year,
                allocation,
                fixed_cost=cost,
                prod_rate=np.zeros(1),
                cost_per_volume=np.zeros(1),
            ),
        ),
        asr_cost=(ASR(2023, 2023, expense_year, cost, allocation),),
        lbt_cost=(LBT(2023, 2023, expense_year, cost, allocation),),
        cost_of_sales=(CostOfSales(2023, 2023, expense_year, cost, allocation),),
    )

    result = min_mean_max_retriever(contract)

    assert result["min_lifting"] == 5.0
    assert result["mean_lifting"] == 125 / 3
    assert result["max_lifting"] == 100.0


def test_uncertainty_statistics_support_transition_contracts():
    adjusted = AdjustData(
        multipliers=np.ones(5), data=_make_adjuster_input(transition=True)
    )
    contract, _, _ = adjusted.activate()
    assert isinstance(contract, Transition)

    result = min_mean_max_retriever(contract)

    assert result == {
        "min_capex": 0.0,
        "mean_capex": 0.0,
        "max_capex": 0.0,
        "min_opex": 0.0,
        "mean_opex": 0.0,
        "max_opex": 0.0,
        "min_oil_price": 1.0,
        "mean_oil_price": 1.0,
        "max_oil_price": 1.0,
        "min_gas_price": 1.0,
        "mean_gas_price": 1.0,
        "max_gas_price": 1.0,
        "min_lifting": 1.0,
        "mean_lifting": 1.0,
        "max_lifting": 1.0,
    }


@pytest.mark.parametrize(
    ("distribution", "minimum", "mean", "maximum", "standard_deviation"),
    [
        ("uniform", 80.0, 100.0, 120.0, 5.0),
        ("triangular", 80.0, 100.0, 120.0, 5.0),
        ("normal", 80.0, 100.0, 120.0, 5.0),
        ("lognormal", 80.0, 100.0, 120.0, 5.0),
    ],
)
def test_uncertainty_multipliers_keep_distribution_bounds_and_dtype(
    distribution: str,
    minimum: float,
    mean: float,
    maximum: float,
    standard_deviation: float,
):
    np.random.seed(1729)
    multipliers = get_multipliers_montecarlo(
        run_number=64,
        distribution=distribution,
        min_value=minimum,
        mean_value=mean,
        max_value=maximum,
        std_dev=standard_deviation,
    )

    assert multipliers.dtype == np.dtype(np.float64)
    assert multipliers.shape == (64,)
    assert np.all(multipliers >= minimum / mean)
    assert np.all(multipliers <= maximum / mean)
