from datetime import date

import numpy as np
import pytest

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.transition import Transition
from pyscnomics.dataset.sample import load_data
from pyscnomics.econ.cashflow import Cashflow, CashflowException
from pyscnomics.econ.selection import FTPTaxRegime, GrossSplitRegime


def _completed_transition() -> Transition:
    first = load_data("case1", "cost_recovery")
    second = load_data("case3", "gross_split")
    assert isinstance(first, CostRecovery)
    assert isinstance(second, GrossSplit)
    transition = Transition(
        contract1=first,
        contract2=second,
        argument_contract1={
            "effective_tax_rate": 0.405,
            "ftp_tax_regime": FTPTaxRegime.DIRECT_MODE,
            "post_uu_22_year2001": True,
        },
        argument_contract2={
            "effective_tax_rate": 0.22,
            "vat_rate": 0.11,
            "inflation_rate": 0.0,
            "inflation_rate_applied_to": None,
            "regime": GrossSplitRegime.PERMEN_ESDM_12_2020,
        },
    )
    transition.run()
    return transition


def test_multiple_cashflow_aligns_contract_years() -> None:
    transition = _completed_transition()
    profile = transition.get_post_run_summary_profile()
    transition_cashflow = profile.consolidated_cashflow
    assert transition_cashflow is not None

    cashflow = Cashflow(contract=(transition, transition))
    cashflow.get_cashflow_all()

    np.testing.assert_array_equal(cashflow.project_years, profile.project_years[:-1])
    assert cashflow.cashflow is not None
    np.testing.assert_array_equal(cashflow.cashflow, 2.0 * transition_cashflow[:-1])


def test_cashflow_rejects_an_unrun_base_project() -> None:
    contract = BaseProject(
        start_date=date(2020, 1, 1),
        end_date=date(2022, 12, 31),
        approval_year=2020,
    )

    with pytest.raises(
        CashflowException, match="only receive contract that has been run"
    ):
        _ = Cashflow(contract=contract)


def test_cashflow_uses_completed_base_project_public_profile() -> None:
    contract = BaseProject(
        start_date=date(2020, 1, 1),
        end_date=date(2022, 12, 31),
        approval_year=2020,
    )
    contract.run()

    cashflow = Cashflow(contract=contract)
    cashflow.get_cashflow_all()

    profile = contract.get_post_run_summary_profile()
    profile_cashflow = profile.consolidated_cashflow
    computed_cashflow = cashflow.cashflow
    assert profile_cashflow is not None
    assert computed_cashflow is not None
    np.testing.assert_array_equal(cashflow.project_years, profile.project_years)
    np.testing.assert_array_equal(computed_cashflow, profile_cashflow)


def test_transition_cashflow_uses_completed_transition_years() -> None:
    transition = _completed_transition()

    cashflow = Cashflow(contract=transition)
    cashflow.get_cashflow_all()

    profile = transition.get_post_run_summary_profile()
    transition_years = profile.project_years
    transition_cashflow = profile.consolidated_cashflow
    computed_cashflow = cashflow.cashflow
    assert transition_years is not None
    assert transition_cashflow is not None
    assert computed_cashflow is not None
    assert cashflow.start_year == transition_years[0]
    assert cashflow.end_year == transition_years[-1]
    np.testing.assert_array_equal(cashflow.project_years, transition_years)
    np.testing.assert_array_equal(computed_cashflow, transition_cashflow)


def test_multiple_cashflow_uses_completed_transition_year_bounds() -> None:
    transition = _completed_transition()

    cashflow = Cashflow(contract=(transition,))

    transition_years = transition.get_post_run_summary_profile().project_years
    assert cashflow.start_year == transition_years[0]
    assert cashflow.end_year == transition_years[-1]
