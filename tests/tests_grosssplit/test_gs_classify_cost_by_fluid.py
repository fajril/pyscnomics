"""
A collection of unit testings to validate cost classification by fluid
in module grosssplit.py
"""

import numpy as np
from datetime import date
from typing import TypedDict

from pyscnomics.econ.selection import CostType, FluidType
from pyscnomics.econ.costs import CapitalCost, CostOfSales
from pyscnomics.contracts.gross_split import GrossSplit


class GrossSplitKwargs(TypedDict):
    start_date: date
    end_date: date
    oil_onstream_date: date
    gas_onstream_date: date
    approval_year: int
    split_ministry_disc: float
    oil_dmo_volume_portion: float
    oil_dmo_fee_portion: float
    gas_dmo_volume_portion: float
    gas_dmo_fee_portion: float
    oil_dmo_holiday_duration: int
    gas_dmo_holiday_duration: int
    oil_carry_forward_depreciation: int | float
    gas_carry_forward_depreciation: int | float


class CapitalClassification(TypedDict):
    oil_length: int
    oil_expense_year: np.ndarray[tuple[int], np.dtype[np.int64]]
    oil_cost: np.ndarray[tuple[int], np.dtype[np.float64]]
    gas_length: int
    gas_expense_year: np.ndarray[tuple[int], np.dtype[np.int64]]
    gas_cost: np.ndarray[tuple[int], np.dtype[np.float64]]


class CapitalClassificationWithTypes(CapitalClassification):
    oil_cost_type: list[CostType | None]
    gas_cost_type: list[CostType | None]


class CostOfSalesClassification(TypedDict):
    oil_length: int
    oil_expense_year: np.ndarray[tuple[int], np.dtype[np.int64]]
    oil_cost: np.ndarray[tuple[int], np.dtype[np.float64]]
    gas_length: int
    gas_expense_year: np.ndarray[tuple[int], np.dtype[np.int64]]
    gas_cost: np.ndarray[tuple[int], np.dtype[np.float64]]


class CostOfSalesClassificationWithTypes(CostOfSalesClassification):
    oil_cost_type: list[CostType | None]


class OilCostOfSalesClassificationWithTypes(TypedDict):
    oil_length: int
    oil_expense_year: np.ndarray[tuple[int], np.dtype[np.int64]]
    oil_cost: np.ndarray[tuple[int], np.dtype[np.float64]]
    oil_cost_type: list[CostType | None]


kwargs: GrossSplitKwargs = {
    "start_date": date(year=2023, month=1, day=1),
    "end_date": date(year=2034, month=12, day=31),
    "oil_onstream_date": date(year=2030, month=1, day=1),
    "gas_onstream_date": date(year=2029, month=1, day=1),
    "approval_year": 2023,
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
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
            ]
        ),
        cost=np.array([200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50]),
        cost_allocation=(
            [
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
            ]
        ),
    )

    # Instance 2
    capital_apel = CapitalCost(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
            ]
        ),
        cost=np.array(
            [
                20,
                20,
                20,
                15,
                10,
                10,
                7.5,
                5,
                5,
                5,
                5,
            ]
        ),
        cost_allocation=(
            [
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
            ]
        ),
    )

    capital_total = tuple([capital_mangga, capital_apel])

    # Calculated results
    gs = GrossSplit(**kwargs, capital_cost=capital_total)
    calc = {
        "oil": gs.classify_capital_cost_by_fluid(FluidType.OIL),
        "gas": gs.classify_capital_cost_by_fluid(FluidType.GAS),
    }
    assert calc["oil"].expense_year is not None
    assert calc["oil"].cost is not None
    assert calc["gas"].expense_year is not None
    assert calc["gas"].cost is not None

    calculated: CapitalClassification = {
        "oil_length": len(calc["oil"]),
        "oil_expense_year": calc["oil"].expense_year,
        "oil_cost": calc["oil"].cost,
        "gas_length": len(calc["gas"]),
        "gas_expense_year": calc["gas"].expense_year,
        "gas_cost": calc["gas"].cost,
    }

    # Expected results
    expected: CapitalClassification = {
        "oil_length": 12,
        "oil_expense_year": np.array(
            [
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
            ]
        ),
        "oil_cost": np.array(
            [
                200,
                200,
                200,
                200,
                150,
                100,
                100,
                75,
                50,
                50,
                50,
                50,
            ]
        ),
        "gas_length": 11,
        "gas_expense_year": np.array(
            [
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
            ]
        ),
        "gas_cost": np.array(
            [
                20,
                20,
                20,
                15,
                10,
                10,
                7.5,
                5,
                5,
                5,
                5,
            ]
        ),
    }

    # Execute testings
    assert calculated["oil_length"] == expected["oil_length"]
    assert calculated["gas_length"] == expected["gas_length"]
    np.testing.assert_allclose(
        calculated["oil_expense_year"], expected["oil_expense_year"]
    )
    np.testing.assert_allclose(
        calculated["gas_expense_year"], expected["gas_expense_year"]
    )
    np.testing.assert_allclose(calculated["oil_cost"], expected["oil_cost"])
    np.testing.assert_allclose(calculated["gas_cost"], expected["gas_cost"])


def test_classify_capital_cost_2():

    # Instance #1
    capital_mangga = CapitalCost(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
            ]
        ),
        cost=np.array([200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50]),
        cost_allocation=(
            [
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
            ]
        ),
    )

    # Instance #2
    capital_nanas = CapitalCost(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
            ]
        ),
        cost=np.array(
            [
                20,
                20,
                20,
                20,
                15,
                10,
                10,
                7.5,
                5,
                5,
                5,
                5,
            ]
        ),
        cost_allocation=(
            [
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
            ]
        ),
    )

    capital_total = tuple([capital_mangga, capital_nanas])

    # Calculated results
    gs = GrossSplit(**kwargs, capital_cost=capital_total)
    calc = {
        "oil": gs.classify_capital_cost_by_fluid(FluidType.OIL),
        "gas": gs.classify_capital_cost_by_fluid(FluidType.GAS),
    }
    assert calc["oil"].expense_year is not None
    assert calc["oil"].cost is not None
    assert calc["oil"].cost_type is not None
    assert calc["gas"].expense_year is not None
    assert calc["gas"].cost is not None
    assert calc["gas"].cost_type is not None

    calculated: CapitalClassificationWithTypes = {
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
    expected: CapitalClassificationWithTypes = {
        "oil_length": 1,
        "oil_expense_year": np.array([2023]),
        "oil_cost": np.array([0]),
        "oil_cost_type": [None],
        "gas_length": 24,
        "gas_expense_year": np.array(
            [
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
            ]
        ),
        "gas_cost": np.array(
            [
                200,
                200,
                200,
                200,
                150,
                100,
                100,
                75,
                50,
                50,
                50,
                50,
                20,
                20,
                20,
                20,
                15,
                10,
                10,
                7.5,
                5,
                5,
                5,
                5,
            ]
        ),
        "gas_cost_type": [None] * 24,
    }

    # Execute testings
    assert calculated["oil_length"] == expected["oil_length"]
    assert calculated["gas_length"] == expected["gas_length"]
    assert calculated["oil_cost_type"] == expected["oil_cost_type"]
    assert calculated["gas_cost_type"] == expected["gas_cost_type"]
    np.testing.assert_allclose(
        calculated["oil_expense_year"], expected["oil_expense_year"]
    )
    np.testing.assert_allclose(
        calculated["gas_expense_year"], expected["gas_expense_year"]
    )
    np.testing.assert_allclose(calculated["oil_cost"], expected["oil_cost"])
    np.testing.assert_allclose(calculated["gas_cost"], expected["gas_cost"])


def test_classify_cost_of_sales_1():

    # Instance #1
    cos_mangga = CostOfSales(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
            ]
        ),
        cost=np.array([200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50]),
        cost_allocation=(
            [
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
            ]
        ),
    )

    # Instance #2
    cos_apel = CostOfSales(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
            ]
        ),
        cost=np.array(
            [
                20,
                20,
                20,
                15,
                10,
                10,
                7.5,
                5,
                5,
                5,
                5,
            ]
        ),
        cost_allocation=(
            [
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
            ]
        ),
    )

    cos_total = tuple([cos_mangga, cos_apel])

    # Calculated results
    gs = GrossSplit(**kwargs, cost_of_sales=cos_total)
    calc = {
        "oil": gs.classify_cost_of_sales_by_fluid(FluidType.OIL),
        "gas": gs.classify_cost_of_sales_by_fluid(FluidType.GAS),
    }
    assert calc["oil"].expense_year is not None
    assert calc["oil"].cost is not None
    assert calc["gas"].expense_year is not None
    assert calc["gas"].cost is not None

    calculated: CostOfSalesClassification = {
        "oil_length": len(calc["oil"]),
        "oil_expense_year": calc["oil"].expense_year,
        "oil_cost": calc["oil"].cost,
        "gas_length": len(calc["gas"]),
        "gas_expense_year": calc["gas"].expense_year,
        "gas_cost": calc["gas"].cost,
    }

    # Expected results
    expected: CostOfSalesClassification = {
        "oil_length": 12,
        "oil_expense_year": np.array(
            [
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
            ]
        ),
        "oil_cost": np.array(
            [
                200,
                200,
                200,
                200,
                150,
                100,
                100,
                75,
                50,
                50,
                50,
                50,
            ]
        ),
        "gas_length": 11,
        "gas_expense_year": np.array(
            [2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034]
        ),
        "gas_cost": np.array([20, 20, 20, 15, 10, 10, 7.5, 5, 5, 5, 5]),
    }

    # Execute testings
    assert calculated["oil_length"] == expected["oil_length"]
    assert calculated["gas_length"] == expected["gas_length"]
    np.testing.assert_allclose(
        calculated["oil_expense_year"], expected["oil_expense_year"]
    )
    np.testing.assert_allclose(
        calculated["gas_expense_year"], expected["gas_expense_year"]
    )
    np.testing.assert_allclose(calculated["oil_cost"], expected["oil_cost"])
    np.testing.assert_allclose(calculated["gas_cost"], expected["gas_cost"])


def test_classify_cost_of_sales_2():

    # Instance #1
    cos_mangga = CostOfSales(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
            ]
        ),
        cost=np.array([200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50]),
        cost_allocation=(
            [
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
            ]
        ),
    )

    # Instance #2
    cos_nanas = CostOfSales(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
            ]
        ),
        cost=np.array(
            [
                20,
                20,
                20,
                20,
                15,
                10,
                10,
                7.5,
                5,
                5,
                5,
                5,
            ]
        ),
        cost_allocation=(
            [
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
            ]
        ),
    )

    cos_total = tuple([cos_mangga, cos_nanas])

    # Calculated results
    gs = GrossSplit(**kwargs, cost_of_sales=cos_total)
    calc = {
        "oil": gs.classify_cost_of_sales_by_fluid(FluidType.OIL),
        "gas": gs.classify_cost_of_sales_by_fluid(FluidType.GAS),
    }
    assert calc["oil"].expense_year is not None
    assert calc["oil"].cost is not None
    assert calc["oil"].cost_type is not None

    calculated: OilCostOfSalesClassificationWithTypes = {
        "oil_length": len(calc["oil"]),
        "oil_expense_year": calc["oil"].expense_year,
        "oil_cost": calc["oil"].cost,
        "oil_cost_type": calc["oil"].cost_type,
    }

    # Expected results
    expected: OilCostOfSalesClassificationWithTypes = {
        "oil_length": 1,
        "oil_expense_year": np.array([2023]),
        "oil_cost": np.array([0]),
        "oil_cost_type": [None],
    }

    # Execute testings
    assert calculated["oil_length"] == expected["oil_length"]
    assert calculated["oil_cost_type"] == expected["oil_cost_type"]
    np.testing.assert_allclose(
        calculated["oil_expense_year"], expected["oil_expense_year"]
    )
    np.testing.assert_allclose(calculated["oil_cost"], expected["oil_cost"])
