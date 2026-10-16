from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.transition import (
    Transition,
    TransitionException,
    adjusting_contract_arguments,
)
from pyscnomics.dataset.sample import load_data
from pyscnomics.econ.costs import ASR, CapitalCost, CostOfSales, Intangible, LBT, OPEX
from pyscnomics.econ.selection import (
    DiscountingMode,
    FTPTaxRegime,
    GrossSplitRegime,
    NPVSelection,
)
from pyscnomics.tools.table import get_table
from pyscnomics.tools.summary import Summary, get_summary


Scheme = Literal["cr", "gs"]
TableArray = npt.NDArray[np.float64] | npt.NDArray[np.int64]


class _ArraySubclass(np.ndarray[tuple[int, ...], np.dtype[np.float64]]):
    pass


CR_RUN_ARGUMENTS: dict[str, object] = {
    "effective_tax_rate": 0.405,
    "ftp_tax_regime": FTPTaxRegime.DIRECT_MODE,
    "post_uu_22_year2001": True,
}
GS_RUN_ARGUMENTS: dict[str, object] = {
    "effective_tax_rate": 0.22,
    "vat_rate": 0.11,
    "inflation_rate": 0.0,
    "inflation_rate_applied_to": None,
    "regime": GrossSplitRegime.PERMEN_ESDM_12_2020,
}


def _load_child(scheme: Scheme) -> CostRecovery | GrossSplit:
    if scheme == "cr":
        return load_data("case1", "cost_recovery")
    return load_data("case3", "gross_split")


def _make_transition(first: Scheme, second: Scheme) -> Transition:
    first_contract = _load_child(first)
    second_contract = _load_child(second)
    first_arguments = (
        CR_RUN_ARGUMENTS.copy() if first == "cr" else GS_RUN_ARGUMENTS.copy()
    )
    second_arguments = (
        CR_RUN_ARGUMENTS.copy() if second == "cr" else GS_RUN_ARGUMENTS.copy()
    )
    return Transition(
        contract1=first_contract,
        contract2=second_contract,
        argument_contract1=first_arguments,
        argument_contract2=second_arguments,
    )


def _shift_cost_years(
    cost: CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales,
    offset: int,
) -> None:
    cost.start_year += offset
    cost.end_year += offset
    if cost.expense_year is not None:
        cost.expense_year += offset
    if cost.project_years is not None:
        cost.project_years += offset
    if isinstance(cost, CapitalCost) and cost.pis_year is not None:
        cost.pis_year = cost.pis_year.astype(np.int64) + offset
    if isinstance(cost, (ASR, LBT)) and cost.final_year is not None:
        cost.final_year = cost.final_year.astype(np.int64) + offset


def _shift_gross_split_years(contract: GrossSplit, offset: int) -> None:
    contract.start_date = contract.start_date.replace(
        year=contract.start_date.year + offset
    )
    contract.end_date = contract.end_date.replace(year=contract.end_date.year + offset)
    contract.project_years = contract.project_years.astype(np.int64) + offset
    contract.approval_year += offset
    if contract.oil_onstream_date is not None:
        contract.oil_onstream_date = contract.oil_onstream_date.replace(
            year=contract.oil_onstream_date.year + offset
        )
    if contract.gas_onstream_date is not None:
        contract.gas_onstream_date = contract.gas_onstream_date.replace(
            year=contract.gas_onstream_date.year + offset
        )
    if contract.lifting is not None:
        for lifting in contract.lifting:
            lifting.start_year += offset
            lifting.end_year += offset
            lifting.prod_year += offset
            lifting.project_years += offset
    for costs in (
        contract.capital_cost,
        contract.intangible_cost,
        contract.opex,
        contract.asr_cost,
        contract.lbt_cost,
        contract.cost_of_sales,
    ):
        if costs is not None:
            for cost in costs:
                _shift_cost_years(cost, offset)


def _make_sequential_transition() -> Transition:
    first_contract = _load_child("cr")
    second_contract = _load_child("gs")
    assert isinstance(second_contract, GrossSplit)
    _shift_gross_split_years(second_contract, 26)
    return Transition(
        contract1=first_contract,
        contract2=second_contract,
        argument_contract1=CR_RUN_ARGUMENTS.copy(),
        argument_contract2=GS_RUN_ARGUMENTS.copy(),
    )


def _consolidated_column(
    contract: CostRecovery | GrossSplit,
    *,
    cost_recovery_key: str,
    gross_split_key: str,
) -> TableArray:
    key = cost_recovery_key if isinstance(contract, CostRecovery) else gross_split_key
    return contract.get_consolidated_table_columns()[key]


@pytest.fixture
def transition_cr_to_gs() -> Transition:
    return _make_transition("cr", "gs")


@pytest.mark.parametrize(
    "first,second",
    [
        ("cr", "gs"),
        ("gs", "cr"),
        ("gs", "gs"),
    ],
)
def test_transition_table_returns_child_tables(
    first: Scheme,
    second: Scheme,
) -> None:
    transition = _make_transition(first, second)
    transition.run()
    first_child, second_child = transition.get_transitioned_contracts()

    first_cashflow = _consolidated_column(
        first_child,
        cost_recovery_key="cashflow",
        gross_split_key="ctr_cashflow",
    )
    second_cashflow = _consolidated_column(
        second_child,
        cost_recovery_key="cashflow",
        gross_split_key="ctr_cashflow",
    )

    table_result = get_table(transition)
    assert table_result is not None
    oil, gas, consolidated = table_result
    assert len(oil) == len(gas) == len(consolidated) == 2
    assert all(
        isinstance(frame, pd.DataFrame)
        for pair in (oil, gas, consolidated)
        for frame in pair
    )

    for frame in (oil[0], gas[0], consolidated[0]):
        np.testing.assert_array_equal(frame["years"], first_child.project_years)
    for frame in (oil[1], gas[1], consolidated[1]):
        np.testing.assert_array_equal(frame["years"], second_child.project_years)

    if isinstance(first_child, CostRecovery):
        np.testing.assert_array_equal(consolidated[0]["cashflow"], first_cashflow)
    else:
        np.testing.assert_array_equal(consolidated[0]["ctr_cashflow"], first_cashflow)
    if isinstance(second_child, CostRecovery):
        np.testing.assert_array_equal(consolidated[1]["cashflow"], second_cashflow)
    else:
        np.testing.assert_array_equal(consolidated[1]["ctr_cashflow"], second_cashflow)


def test_transition_table_requires_run(transition_cr_to_gs: Transition) -> None:
    with pytest.raises(
        TransitionException,
        match=r"Transition\.run\(\) must be called before table generation",
    ):
        _ = get_table(transition_cr_to_gs)


def test_transition_get_summary_uses_transition_profile() -> None:
    transition = _make_sequential_transition()
    assert transition.contract1.end_date.year + 1 == transition.contract2.start_date.year
    np.testing.assert_array_equal(
        transition.contract2.project_years,
        np.arange(
            transition.contract2.start_date.year,
            transition.contract2.end_date.year + 1,
            dtype=np.int64,
        ),
    )
    transition.run()
    reference_year = transition.contract1.start_date.year
    expected_summary = Summary(
        contract=transition,
        reference_year=reference_year,
        inflation_rate=0.03,
        discount_rate=0.08,
        npv_mode=NPVSelection.NPV_REAL_TERMS,
        discounting_mode=DiscountingMode.MID_YEAR,
        profitability_discounted=False,
    )
    expected_summary.run()

    actual = get_summary(
        transition,
        reference_year=reference_year,
        inflation_rate=0.03,
        discount_rate=0.08,
        npv_mode=NPVSelection.NPV_REAL_TERMS,
        discounting_mode=DiscountingMode.MID_YEAR,
        profitability_discounted=False,
    )

    assert actual == expected_summary.get_summary_dict()
    profile = transition.get_post_run_summary_profile()
    oil_sunk_cost = profile.oil_sunk_cost
    gas_sunk_cost = profile.gas_sunk_cost
    assert oil_sunk_cost is not None
    assert gas_sunk_cost is not None
    assert actual["sunk_cost"] == float(np.sum(oil_sunk_cost + gas_sunk_cost))


def test_adjusting_contract_arguments_preserves_integer_scalar() -> None:
    arguments: dict[str, object] = {"cum_production_split_offset": 0}
    prior_rows = np.array([1.0, 2.0], dtype=np.float64)
    post_rows = np.array([3.0, 4.0], dtype=np.float64)

    adjusted = adjusting_contract_arguments(
        arguments_dict=arguments,
        first_contract=True,
        prior_rows=prior_rows,
        post_rows=post_rows,
    )

    assert adjusted["cum_production_split_offset"] == 0


def test_adjusting_contract_arguments_concatenates_ndarray_subclass() -> None:
    subclass_values = np.array([5.0, 6.0], dtype=np.float64).view(_ArraySubclass)
    arguments: dict[str, object] = {"cum_production_split_offset": subclass_values}
    prior_rows = np.array([1.0, 2.0], dtype=np.float64)
    post_rows = np.array([3.0, 4.0], dtype=np.float64)

    adjusted = adjusting_contract_arguments(
        arguments_dict=arguments,
        first_contract=True,
        prior_rows=prior_rows,
        post_rows=post_rows,
    )

    result = adjusted["cum_production_split_offset"]
    assert isinstance(result, np.ndarray)
    assert result.shape == (4,)
    assert result[0] == 5.0
    assert result[1] == 6.0
    assert result[2] == 3.0
    assert result[3] == 4.0


def test_transitioned_contracts_requires_run(
    transition_cr_to_gs: Transition,
) -> None:
    with pytest.raises(
        TransitionException,
        match=r"Transition\.run\(\) must be called before table generation",
    ):
        _ = transition_cr_to_gs.get_transitioned_contracts()


@pytest.mark.parametrize(
    "first,second",
    [
        ("cr", "gs"),
        ("gs", "cr"),
        ("gs", "gs"),
    ],
)
def test_transition_run_populates_project_years(
    first: Scheme,
    second: Scheme,
) -> None:
    transition = _make_transition(first, second)
    transition.run()

    assert transition.project_years is not None
    assert len(transition.project_years) >= 1


@pytest.mark.parametrize(
    "first,second",
    [
        ("cr", "gs"),
        ("gs", "cr"),
        ("gs", "gs"),
    ],
)
def test_transitioned_contracts_are_new_aligned_children(
    first: Scheme,
    second: Scheme,
) -> None:
    transition = _make_transition(first, second)
    transition.run()

    first_child, second_child = transition.get_transitioned_contracts()
    project_years = transition.project_years
    assert project_years is not None
    assert first_child is not transition.contract1
    assert second_child is not transition.contract2
    np.testing.assert_array_equal(first_child.project_years, project_years)
    np.testing.assert_array_equal(second_child.project_years, project_years)
    assert first_child.start_date == second_child.start_date
    assert first_child.end_date == second_child.end_date


@pytest.mark.parametrize(
    "first,second",
    [
        ("cr", "gs"),
        ("gs", "cr"),
        ("gs", "gs"),
    ],
)
def test_transition_consolidated_aggregates_from_mutated_child_after_run(
    first: Scheme,
    second: Scheme,
) -> None:
    transition = _make_transition(first, second)
    transition.run()
    first_child, second_child = transition.get_transitioned_contracts()

    first_tax_payment = _consolidated_column(
        first_child,
        cost_recovery_key="tax_payment",
        gross_split_key="tax",
    )
    second_tax_payment = _consolidated_column(
        second_child,
        cost_recovery_key="tax_payment",
        gross_split_key="tax",
    )
    first_cashflow = _consolidated_column(
        first_child,
        cost_recovery_key="cashflow",
        gross_split_key="ctr_cashflow",
    )
    second_cashflow = _consolidated_column(
        second_child,
        cost_recovery_key="cashflow",
        gross_split_key="ctr_cashflow",
    )
    first_government_take = _consolidated_column(
        first_child,
        cost_recovery_key="government_take",
        gross_split_key="government_take",
    )
    second_government_take = _consolidated_column(
        second_child,
        cost_recovery_key="government_take",
        gross_split_key="government_take",
    )
    transition_profiles = transition.get_fiscal_profiles()
    np.testing.assert_array_equal(
        transition_profiles.tax_payment,
        first_tax_payment + second_tax_payment,
    )
    np.testing.assert_array_equal(
        transition_profiles.cashflow,
        first_cashflow + second_cashflow,
    )
    np.testing.assert_array_equal(
        transition_profiles.government_take,
        first_government_take + second_government_take,
    )


@pytest.mark.parametrize(
    "first,second",
    [
        ("cr", "gs"),
        ("gs", "cr"),
        ("gs", "gs"),
    ],
)
def test_transition_undepreciated_scalars_from_child_totals(
    first: Scheme,
    second: Scheme,
) -> None:
    transition = _make_transition(first, second)
    transition.run()
    first_child, second_child = transition.get_transitioned_contracts()

    first_oil, first_gas, _ = first_child.get_transition_fiscal_inputs()
    second_oil, second_gas, _ = second_child.get_transition_fiscal_inputs()
    transition_profiles = transition.get_fiscal_profiles()
    assert transition_profiles.oil_undepreciated_asset == first_oil + second_oil
    assert transition_profiles.gas_undepreciated_asset == first_gas + second_gas
    assert transition_profiles.consolidated_undepreciated_asset == (
        transition_profiles.oil_undepreciated_asset
        + transition_profiles.gas_undepreciated_asset
    )


@pytest.mark.parametrize(
    "first,second",
    [
        ("cr", "gs"),
        ("gs", "cr"),
        ("gs", "gs"),
    ],
)
def test_transition_gov_ets_zero_vectors_have_project_year_shape(
    first: Scheme,
    second: Scheme,
) -> None:
    transition = _make_transition(first, second)
    transition.run()

    project_years = transition.project_years
    assert project_years is not None
    transition_profiles = transition.get_fiscal_profiles()
    assert transition_profiles.oil_government_ets.shape == project_years.shape
    assert transition_profiles.gas_government_ets.shape == project_years.shape


def test_transition_gov_ftp_position_gs_to_cr() -> None:
    transition = _make_transition("gs", "cr")
    transition.run()

    _, second_child = transition.get_transitioned_contracts()
    assert isinstance(second_child, CostRecovery)
    gov_ftp = transition.get_fiscal_profiles().government_ftp
    second_gov_ftp = second_child.get_consolidated_table_columns()["ftp_gov"]
    np.testing.assert_array_equal(gov_ftp, second_gov_ftp)


def test_transition_gov_ftp_position_gs_to_gs() -> None:
    transition = _make_transition("gs", "gs")
    transition.run()

    _, second_child = transition.get_transitioned_contracts()
    assert isinstance(second_child, GrossSplit)
    gov_ftp = transition.get_fiscal_profiles().government_ftp
    second_gov_share = second_child.get_consolidated_table_columns()["government_share"]
    np.testing.assert_array_equal(gov_ftp, second_gov_share)


def test_transition_gov_ftp_position_cr_to_gs() -> None:
    transition = _make_transition("cr", "gs")
    transition.run()

    first_child, second_child = transition.get_transitioned_contracts()
    assert isinstance(first_child, CostRecovery)
    assert isinstance(second_child, GrossSplit)
    gov_ftp = transition.get_fiscal_profiles().government_ftp
    first_gov_ftp = first_child.get_consolidated_table_columns()["ftp_gov"]
    second_gov_share = second_child.get_consolidated_table_columns()["government_share"]
    np.testing.assert_array_equal(gov_ftp, first_gov_ftp + second_gov_share)
