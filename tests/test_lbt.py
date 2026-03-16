"""
A collection of testing units for LBT class.
"""

import pytest
import numpy as np
from pyscnomics.econ.selection import FluidType, CostType
from pyscnomics.econ.costs import LBT, LBTException


# Parameters for example
expense_year_1 = np.array([2023, 2024, 2025, 2026])
cost_1 = np.array([200, 150, 100, 50])
cost_allocation_1 = [FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL]
cost_type_1 = [
    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
]
cost_type_2 = [
    CostType.SUNK_COST, CostType.SUNK_COST, CostType.SUNK_COST, CostType.SUNK_COST
]
tax_portion_1 = np.array([1, 1, 1, 1])


def test_lbt_incorrect_start_year():
    with pytest.raises(LBTException):
        LBT(
            start_year=2033,
            end_year=2030,
            expense_year=np.array([2026, 2027]),
            cost=np.array([100, 50]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
            cost_type=[CostType.SUNK_COST, CostType.SUNK_COST],
        )


def test_lbt_incorrect_unequal_arrays():
    with pytest.raises(LBTException):
        LBT(
            start_year=2033,
            end_year=2030,
            expense_year=np.array([2026, 2027]),
            cost=np.array([100, 50, 25, 10]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
            cost_type=[CostType.SUNK_COST, CostType.SUNK_COST],
        )


def test_lbt_incorrect_expense_year():
    with pytest.raises(LBTException):
        LBT(
            start_year=2033,
            end_year=2030,
            expense_year=np.array([2026, 2033]),
            cost=np.array([100, 50]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
            cost_type=[CostType.SUNK_COST, CostType.SUNK_COST],
        )

    with pytest.raises(LBTException):
        LBT(
            start_year=2033,
            end_year=2030,
            expense_year=np.array([2018, 2027]),
            cost=np.array([100, 50]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
            cost_type=[CostType.SUNK_COST, CostType.SUNK_COST],
        )


def test_lbt_dunder_comparison():

    lbt_mangga = LBT(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2026, 2027]),
        cost=np.array([100, 100]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
        cost_type=[CostType.SUNK_COST, CostType.SUNK_COST],
    )

    lbt_apel = LBT(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2026, 2027]),
        cost=np.array([100, 50]),
        cost_allocation=[FluidType.GAS, FluidType.GAS],
        cost_type=[CostType.SUNK_COST, CostType.SUNK_COST],
    )

    lbt_nanas = LBT(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2026, 2027]),
        cost=np.array([50, 50]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
        cost_type=[CostType.SUNK_COST, CostType.SUNK_COST],
    )

    lbt_jeruk = LBT(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2026, 2027]),
        cost=np.array([100, 100]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
        cost_type=[CostType.SUNK_COST, CostType.SUNK_COST],
    )

    assert lbt_mangga == lbt_jeruk
    assert lbt_apel != lbt_jeruk
    assert lbt_nanas < lbt_mangga
    assert lbt_jeruk > lbt_apel


def test_lbt_dunder_add():

    lbt_mangga = LBT(
        start_year=2023,
        end_year=2030,
        expense_year=expense_year_1,
        cost=cost_1,
        cost_allocation=cost_allocation_1,
        cost_type=cost_type_1,
    )

    lbt_apel = LBT(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2029, 2029]),
        cost=np.array([400, 400]),
        cost_allocation=[FluidType.GAS, FluidType.GAS],
        cost_type=[CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST],
    )

    lbt_total = lbt_mangga + lbt_apel

    expected = {
        "expense_year": np.array([2023, 2024, 2025, 2026, 2029, 2029]),
        "cost": np.array([200, 150, 100, 50, 400, 400]),
        "cost_allocation": [
            FluidType.OIL, FluidType.OIL, FluidType.OIL,
            FluidType.OIL, FluidType.GAS, FluidType.GAS,
        ],
        "cost_type": [
            CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
            CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
            CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
        ],
    }

    # Execute testings
    np.testing.assert_allclose(lbt_total.expense_year, expected["expense_year"])
    np.testing.assert_allclose(lbt_total.cost, expected["cost"])
    assert lbt_total.cost_allocation == expected["cost_allocation"]
    assert lbt_total.cost_type == expected["cost_type"]
