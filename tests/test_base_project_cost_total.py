"""
A collection of unit testigs to validate calculations
involving total cost categories
"""

from datetime import date
import numpy as np

from pyscnomics.econ.selection import OtherRevenue, FluidType, CostType
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales
from pyscnomics.contracts.project import BaseProject


kwargs = {
    "start_date": date(year=2023, month=1, day=1),
    "end_date": date(year=2034, month=12, day=31),
    "oil_onstream_date": date(year=2030, month=1, day=1),
    "gas_onstream_date": date(year=2029, month=1, day=1),
    "approval_year": 2027,
}


def test_capital_cost_total():

    # Instance #1
    capital_mangga = CapitalCost(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        cost=np.array(
            [
                200, 200, 200, 200, 150, 100,
                100, 75, 50, 50, 50, 50
            ]
        ),
        cost_allocation=(
            [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
            ]
        ),
    )

    # Instance #2
    capital_apel = CapitalCost(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        cost=np.array(
            [
                200, 200, 200, 200, 150, 100,
                100, 75, 50, 50, 50, 50
            ]
        ),
        cost_allocation=(
            [
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
            ]
        ),
    )

    capital_total = tuple([capital_mangga, capital_apel])

    # Calculated results
    pr = BaseProject(**kwargs, capital_cost=capital_total)
    calculated = {
        "expense_year": pr.capital_cost_total.expense_year,
        "cost": pr.capital_cost_total.cost,
    }

    # Expected results
    expected = {
        "expense_year": np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034,
                2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        "cost": np.array(
            [
                200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50,
                200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50,
            ]
        ),
    }

    # Execute testings
    np.testing.assert_allclose(calculated["expense_year"], expected["expense_year"])
    np.testing.assert_allclose(calculated["cost"], expected["cost"])


def test_intangible_cost_total():

    # Instance #1
    intangible_mangga = Intangible(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        cost=np.array(
            [
                200, 200, 200, 200, 150, 100,
                100, 75, 50, 50, 50, 50
            ]
        ),
        cost_allocation=(
            [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
            ]
        ),
    )

    # Instance #2
    intangible_apel = Intangible(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        cost=np.array(
            [
                200, 200, 200, 200, 150, 100,
                100, 75, 50, 50, 50, 50
            ]
        ),
        cost_allocation=(
            [
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
            ]
        ),
    )

    intangible_total = tuple([intangible_mangga, intangible_apel])

    # Calculated results
    pr = BaseProject(**kwargs, intangible_cost=intangible_total)

    calculated = {
        "expense_year": pr.intangible_cost_total.expense_year,
        "cost": pr.intangible_cost_total.cost,
    }

    # Expected results
    expected = {
        "expense_year": np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034,
                2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        "cost": np.array(
            [
                200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50,
                200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50,
            ]
        ),
    }

    # Execute testings
    np.testing.assert_allclose(calculated["expense_year"], expected["expense_year"])
    np.testing.assert_allclose(calculated["cost"], expected["cost"])


def test_lbt_cost_total():

    # Instance #1
    lbt_mangga = LBT(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        cost=np.array(
            [
                200, 200, 200, 200, 150, 100,
                100, 75, 50, 50, 50, 50
            ]
        ),
        cost_allocation=(
            [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
            ]
        ),
    )

    # Intance #2
    lbt_apel = LBT(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        cost=np.array(
            [
                200, 200, 200, 200, 150, 100,
                100, 75, 50, 50, 50, 50
            ]
        ),
        cost_allocation=(
            [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
            ]
        ),
    )

    lbt_total = tuple([lbt_mangga, lbt_apel])

    # Calculated results
    pr = BaseProject(**kwargs, lbt_cost=lbt_total)

    calculated = {
        "expense_year": pr.lbt_cost_total.expense_year,
        "cost": pr.lbt_cost_total.cost,
        "length": len(pr.lbt_cost_total)
    }

    expected = {
        "expense_year": np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034,
                2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        "cost": np.array(
            [
                200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50,
                200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50,
            ]
        ),
        "length": 24,
    }

    # Execute testings
    np.testing.assert_allclose(calculated["expense_year"], expected["expense_year"])
    np.testing.assert_allclose(calculated["cost"], expected["cost"])
    assert calculated["length"] == expected["length"]



