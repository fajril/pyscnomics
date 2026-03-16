"""
A collection of testing units for ASR class.
"""

import pytest
import numpy as np
from pyscnomics.econ.selection import FluidType, CostType
from pyscnomics.econ.costs import ASR, ASRException


# Parameters for example
expense_year_1 = np.array([2023, 2024, 2025, 2026,])
cost_1 = np.array([200, 150, 100, 50])
cost_allocation_1 = [FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL]
cost_type_1 = [
    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
]
tax_portion_1 = np.array([1, 1, 1, 1])


def test_asr_incorrect_year_input():
    """A unit testing for incorrect data input: start year is after end year"""

    with pytest.raises(ASRException):

        ASR(
            start_year=2033,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2029, 2030]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )


def test_asr_unequal_length_of_data_input():
    """A unit testing for incorrect data input: unequal length of data input"""

    with pytest.raises(ASRException):

        ASR(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2029]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )


def test_asr_incorrect_expense_year_input():
    """A unit testing for incorrect data input: incorrect expense year"""

    with pytest.raises(ASRException):
        ASR(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2029, 2031]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )

    with pytest.raises(ASRException):
        ASR(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2020, 2025]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )


def test_comparison():
    """A unit testing for comparison in ASR class"""

    asr_mangga = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    asr_apel = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    asr_nanas = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([25, 25]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    assert asr_mangga == asr_apel
    assert asr_apel != asr_nanas
    assert asr_nanas <= asr_mangga
    assert asr_apel >= asr_nanas


def test_asr_dunder_add():

    asr_mangga = ASR(
        start_year=2023,
        end_year=2030,
        expense_year=expense_year_1,
        cost=cost_1,
        cost_allocation=cost_allocation_1,
        cost_type=cost_type_1,
        future_rate=0.2,
    )

    asr_apel = ASR(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2029, 2029]),
        cost=np.array([500, 500]),
        cost_allocation=[FluidType.GAS, FluidType.GAS],
        cost_type=[CostType.SUNK_COST, CostType.SUNK_COST],
        future_rate=0.01,
    )

    asr_total = asr_mangga + asr_apel

    expected = {
        "expense_year": np.array([2023, 2024, 2025, 2026, 2029, 2029]),
        "cost": np.array([200, 150, 100, 50, 500, 500]),
        "cost_allocation": [
            FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
            FluidType.GAS, FluidType.GAS
        ],
        "cost_type": [
            CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
            CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
            CostType.SUNK_COST, CostType.SUNK_COST,
        ],
        "future_rate": np.array([0.2, 0.2, 0.2, 0.2, 0.01, 0.01]),
    }

    # Execute testings
    np.testing.assert_allclose(asr_total.expense_year, expected["expense_year"])
    np.testing.assert_allclose(asr_total.cost, expected["cost"])
    assert asr_total.cost_allocation == expected["cost_allocation"]
    assert asr_total.cost_type == expected["cost_type"]
    np.testing.assert_allclose(asr_total.future_rate, expected["future_rate"])


def test_asr_expenditures():
    """A unit testing for expenditures method in ASR class"""

    # Expected result
    case1 = [0, 0, 112.6162419, 110.4080803, 54.121608, 53.0604, 0, 0]
    case2 = [0, 0, 118.247054, 115.9284843, 56.8276884, 55.71342, 0, 0]

    # Calculated result
    asr_mangga = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 50, 50]),
        future_rate=np.array([0.02, 0.02, 0.02, 0.02]),
        expense_year=np.array([2025, 2026, 2027, 2028]),
        tax_portion=np.array([1, 1, 1, 1]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    case1_calc = asr_mangga.expenditures_post_tax()
    case2_calc = asr_mangga.expenditures_post_tax(tax_rate=0.05)

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(case1, case1_calc)
    np.testing.assert_allclose(case2, case2_calc)
