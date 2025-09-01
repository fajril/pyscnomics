"""
A collection of unit testings to validate cost classification by fluid
in module grosssplit.py
"""

import numpy as np
from datetime import date

from pyscnomics.econ.selection import OtherRevenue, FluidType, CostType
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales
from pyscnomics.contracts.grossplit import GrossSplit


kwargs = {
    "start_date": date(year=2023, month=1, day=1),
    "end_date": date(year=2034, month=12, day=31),
    "oil_onstream_date": date(year=2030, month=1, day=1),
    "gas_onstream_date": date(year=2029, month=1, day=1),
    "approval_year": None,
    "base_split_ctr_oil": 0.43,
    "base_split_ctr_gas": 0.48,
    "split_ministry_disc": 0.08,
    "oil_dmo_volume_portion": 0.25,
    "oil_dmo_fee_portion": 1.0,
    "gas_dmo_volume_portion": 1.0,
    "gas_dmo_fee_portion": 1.0,
    "oil_dmo_holiday_duration": 60,
    "gas_dmo_holiday_duration": 60,
    "oil_carry_forward_depreciation": 10,
    "gas_carry_forward_depreciation": 10,
}


def test_classify_capital_cost_1():

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

    # Instance 2
    capital_apel = CapitalCost(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        cost=np.array(
            [
                20, 20, 20, 15, 10,
                10, 7.5, 5, 5, 5, 5,
            ]
        ),
        cost_allocation=(
            [
                FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
            ]
        ),
    )

    capital_total = tuple([capital_mangga, capital_apel])

    # Calculated results
    gs = GrossSplit(**kwargs, capital_cost=capital_total)
    calc = gs._classify_costs_by_fluid(classifier=gs._classify_capital_cost_by_fluid)

    calculated = {
        "oil_length": len(calc["oil"]),
        "oil_expense_year": calc["oil"].expense_year,
        "oil_cost": calc["oil"].cost,
        "gas_length": len(calc["gas"]),
        "gas_expense_year": calc["gas"].expense_year,
        "gas_cost": calc["gas"].cost,
    }

    # Expected results
    expected = {
        "oil_length": 12,
        "oil_expense_year": np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        "oil_cost": np.array(
            [
                200, 200, 200, 200, 150, 100,
                100, 75, 50, 50, 50, 50,
            ]
        ),
        "gas_length": 11,
        "gas_expense_year": np.array(
            [
                2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        "gas_cost": np.array(
            [
                20, 20, 20, 15, 10,
                10, 7.5, 5, 5, 5, 5,
            ]
        ),
    }

    # Execute testings
    assert calculated["oil_length"] == expected["oil_length"]
    assert calculated["gas_length"] == expected["gas_length"]
    np.testing.assert_allclose(calculated["oil_expense_year"], expected["oil_expense_year"])
    np.testing.assert_allclose(calculated["gas_expense_year"], expected["gas_expense_year"])
    np.testing.assert_allclose(calculated["oil_cost"], expected["oil_cost"])
    np.testing.assert_allclose(calculated["gas_cost"], expected["gas_cost"])


def test_classify_capital_cost_2():

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
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
            ]
        ),
    )

    # Instance #2
    capital_nanas = CapitalCost(
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
                20, 20, 20, 20, 15, 10,
                10, 7.5, 5, 5, 5, 5,
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

    capital_total = tuple([capital_mangga, capital_nanas])

    # Calculated results
    gs = GrossSplit(**kwargs, capital_cost=capital_total)
    calc = gs._classify_costs_by_fluid(classifier=gs._classify_capital_cost_by_fluid)

    calculated = {
        "oil_length": len(calc["oil"]),
        "oil_expense_year": calc["oil"].expense_year,
        "oil_cost": calc["oil"].cost,
        "oil_cost_type": calc["oil"].cost_type,
        "gas_length": len(calc["gas"]),
        "gas_expense_year": calc["gas"].expense_year,
        "gas_cost": calc["gas"].cost,
        "gas_cost_type": calc["gas"].cost_type,
    }

    # Expected results
    expected = {
        "oil_length": 1,
        "oil_expense_year": np.array([2023]),
        "oil_cost": np.array([0]),
        "oil_cost_type": [CostType.PRE_ONSTREAM_COST],
        "gas_length": 24,
        "gas_expense_year": np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034,
                2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        "gas_cost": np.array(
            [
                200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50,
                20, 20, 20, 20, 15, 10, 10, 7.5, 5, 5, 5, 5,
            ]
        ),
        "gas_cost_type": (
            [
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            ]
        ),
    }

    # Execute testings
    assert calculated["oil_length"] == expected["oil_length"]
    assert calculated["gas_length"] == expected["gas_length"]
    assert calculated["oil_cost_type"] == expected["oil_cost_type"]
    assert calculated["gas_cost_type"] == expected["gas_cost_type"]
    np.testing.assert_allclose(calculated["oil_expense_year"], expected["oil_expense_year"])
    np.testing.assert_allclose(calculated["gas_expense_year"], expected["gas_expense_year"])
    np.testing.assert_allclose(calculated["oil_cost"], expected["oil_cost"])
    np.testing.assert_allclose(calculated["gas_cost"], expected["gas_cost"])


def test_classify_cost_of_sales_1():

    # Instance #1
    cos_mangga = CostOfSales(
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
    cos_apel = CostOfSales(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        cost=np.array(
            [
                20, 20, 20, 15, 10,
                10, 7.5, 5, 5, 5, 5,
            ]
        ),
        cost_allocation=(
            [
                FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
            ]
        ),
    )

    cos_total = tuple([cos_mangga, cos_apel])

    # Calculated results
    gs = GrossSplit(**kwargs, cost_of_sales=cos_total)
    calc = gs._classify_costs_by_fluid(classifier=gs._classify_cost_of_sales_by_fluid)

    calculated = {
        "oil_length": len(calc["oil"]),
        "oil_expense_year": calc["oil"].expense_year,
        "oil_cost": calc["oil"].cost,
        "gas_length": len(calc["gas"]),
        "gas_expense_year": calc["gas"].expense_year,
        "gas_cost": calc["gas"].cost,
    }

    # Expected results
    expected = {
        "oil_length": 12,
        "oil_expense_year": np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        "oil_cost": np.array(
            [
                200, 200, 200, 200, 150, 100,
                100, 75, 50, 50, 50, 50,
            ]
        ),
        "gas_length": 11,
        "gas_expense_year": np.array(
            [2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034]
        ),
        "gas_cost": np.array(
            [20, 20, 20, 15, 10, 10, 7.5, 5, 5, 5, 5]
        ),
    }

    # Execute testings
    assert calculated["oil_length"] == expected["oil_length"]
    assert calculated["gas_length"] == expected["gas_length"]
    np.testing.assert_allclose(calculated["oil_expense_year"], expected["oil_expense_year"])
    np.testing.assert_allclose(calculated["gas_expense_year"], expected["gas_expense_year"])
    np.testing.assert_allclose(calculated["oil_cost"], expected["oil_cost"])
    np.testing.assert_allclose(calculated["gas_cost"], expected["gas_cost"])


def test_classify_cost_of_sales_2():

    # Instance #1
    cos_mangga = CostOfSales(
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

    # Instance #2
    cos_nanas = CostOfSales(
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
                20, 20, 20, 20, 15, 10,
                10, 7.5, 5, 5, 5, 5,
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

    cos_total = tuple([cos_mangga, cos_nanas])

    # Calculated results
    gs = GrossSplit(**kwargs, cost_of_sales=cos_total)
    calc = gs._classify_costs_by_fluid(classifier=gs._classify_cost_of_sales_by_fluid)

    calculated = {
        "oil_length": len(calc["oil"]),
        "oil_expense_year": calc["oil"].expense_year,
        "oil_cost": calc["oil"].cost,
        "oil_cost_type": calc["oil"].cost_type,
    }

    # Expected results
    expected = {
        "oil_length": 1,
        "oil_expense_year": np.array([2023]),
        "oil_cost": np.array([0]),
        "oil_cost_type": [CostType.PRE_ONSTREAM_COST],
    }

    # Execute testings
    assert calculated["oil_length"] == expected["oil_length"]
    assert calculated["oil_cost_type"] == expected["oil_cost_type"]
    np.testing.assert_allclose(calculated["oil_expense_year"], expected["oil_expense_year"])
    np.testing.assert_allclose(calculated["oil_cost"], expected["oil_cost"])
