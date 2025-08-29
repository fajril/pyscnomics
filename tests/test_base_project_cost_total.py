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
    pass



def test_classify_capital_cost_by_fluid_1():

    # Synthetic Capital Cost Data
    oil_capital = CapitalCost(
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
        cost_type=(
            [
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            ]
        ),
        tax_portion=np.array(
            [
                1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
            ]
        ),
    )

    gas_capital = CapitalCost(
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
                10, 7.5, 5, 5, 5, 5
            ]
        ),
        cost_allocation=(
            [
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
            ]
        ),
        cost_type=(
            [
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            ]
        ),
        tax_portion=np.array(
            [
                1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
            ]
        ),
    )

    total_capital = tuple([oil_capital, gas_capital])

    pr = BaseProject(
        start_date=date(year=2023, month=1, day=1),
        end_date=date(year=2034, month=12, day=31),
        oil_onstream_date=date(year=2030, month=1, day=1),
        gas_onstream_date=date(year=2029, month=1, day=1),
        approval_year=2027,
        capital_cost=total_capital,
    )

    calculated_oil = {
        "oil": pr._classify_capital_cost_by_fluid(fluid_type=FluidType.OIL).cost,
        "gas": pr._classify_capital_cost_by_fluid(fluid_type=FluidType.GAS).cost,
    }

    expected_cost = {
        "oil": np.array([200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50]),
        "gas": np.array([20, 20, 20, 20, 15, 10, 10, 7.5, 5, 5, 5, 5]),
    }

    np.testing.assert_allclose(calculated_oil["oil"], expected_cost["oil"])
    np.testing.assert_allclose(calculated_oil["gas"], expected_cost["gas"])


def test_classify_capital_cost_by_fluid_2():

    # Synthetic Capital Cost Data
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
        cost_type=(
            [
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            ]
        ),
        tax_portion=np.array(
            [
                1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
            ]
        ),
    )

    capital_total = tuple([capital_mangga])

    pr = BaseProject(
        start_date=date(year=2023, month=1, day=1),
        end_date=date(year=2034, month=12, day=31),
        oil_onstream_date=date(year=2030, month=1, day=1),
        gas_onstream_date=date(year=2029, month=1, day=1),
        approval_year=2027,
        capital_cost=capital_total,
    )

    calc_oil = pr._classify_capital_cost_by_fluid(fluid_type=FluidType.OIL)
    calc_gas = pr._classify_capital_cost_by_fluid(fluid_type=FluidType.GAS)

    calculated = {
        "oil_cost": calc_oil.cost,
        "gas_cost": calc_gas.cost,
        "oil_expense_year": calc_oil.expense_year,
        "gas_expense_year": calc_gas.expense_year,
        "oil_cost_type": calc_oil.cost_type,
        "gas_cost_type": calc_gas.cost_type,
    }

    expected = {
        "oil_cost": np.array([0]),
        "gas_cost": np.array([200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50]),
        "oil_expense_year": np.array([2023]),
        "gas_expense_year": np.array(
            [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034]
        ),
        "oil_cost_type": [CostType.PRE_ONSTREAM_COST],
        "gas_cost_type": [
            CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
        ],
    }

    np.testing.assert_allclose(calculated["oil_cost"], expected["oil_cost"])
    np.testing.assert_allclose(calculated["gas_cost"], expected["gas_cost"])
    np.testing.assert_allclose(calculated["oil_expense_year"], expected["oil_expense_year"])
    np.testing.assert_allclose(calculated["gas_expense_year"], expected["gas_expense_year"])


def test_classify_intangible_cost_by_fluid_1():

    # Synthetic Intangible Data
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
                200, 200, 200, 200,
                150, 100, 100, 75,
                50, 50, 50, 50
            ]
        ),
        cost_allocation=(
            [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
            ]
        ),
        cost_type=(
            [
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            ]
        ),
        tax_portion=np.array(
            [
                1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
            ]
        ),
    )

    intangible_total = tuple([intangible_mangga])

    pr = BaseProject(
        start_date=date(year=2023, month=1, day=1),
        end_date=date(year=2034, month=12, day=31),
        oil_onstream_date=date(year=2030, month=1, day=1),
        gas_onstream_date=date(year=2029, month=1, day=1),
        approval_year=2027,
        intangible_cost=intangible_total,
    )

    calculated_cost = {
        "oil": pr._classify_intangible_cost_by_fluid(fluid_type=FluidType.OIL).cost,
        "gas": pr._classify_intangible_cost_by_fluid(fluid_type=FluidType.GAS).cost,
    }

    expected_cost = {
        "oil": np.array([200, 200, 200, 200, 50, 50, 50, 50]),
        "gas": np.array([150, 100, 100, 75]),
    }

    np.testing.assert_allclose(calculated_cost["oil"], expected_cost["oil"])
    np.testing.assert_allclose(calculated_cost["gas"], expected_cost["gas"])


def test_classify_intangible_cost_by_fluid_2():

    # Synthetic Intangible Data
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
                200, 200, 200, 200,
                150, 100, 100, 75,
                50, 50, 50, 50
            ]
        ),
        cost_allocation=(
            [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
            ]
        ),
        cost_type=(
            [
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            ]
        ),
        tax_portion=np.array(
            [
                1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
            ]
        ),
    )

    intangible_total = tuple([intangible_mangga])

    pr = BaseProject(
        start_date=date(year=2023, month=1, day=1),
        end_date=date(year=2034, month=12, day=31),
        oil_onstream_date=date(year=2030, month=1, day=1),
        gas_onstream_date=date(year=2029, month=1, day=1),
        approval_year=2027,
        intangible_cost=intangible_total,
    )

    calc_oil = pr._classify_intangible_cost_by_fluid(fluid_type=FluidType.OIL)
    calc_gas = pr._classify_intangible_cost_by_fluid(fluid_type=FluidType.GAS)

    calculated = {
        "oil_cost": calc_oil.cost,
        "gas_cost": calc_gas.cost,
        "oil_expense_year": calc_oil.expense_year,
        "gas_expense_year": calc_gas.expense_year,
    }

    expected = {
        "oil_cost": np.array([200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50]),
        "gas_cost": np.array([0]),
        "oil_expense_year": np.array(
            [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034]
        ),
        "gas_expense_year": np.array([2023]),
    }

    np.testing.assert_allclose(calculated["oil_cost"], expected["oil_cost"])
    np.testing.assert_allclose(calculated["gas_cost"], expected["gas_cost"])
    np.testing.assert_allclose(calculated["oil_expense_year"], expected["oil_expense_year"])
    np.testing.assert_allclose(calculated["gas_expense_year"], expected["gas_expense_year"])


def test_classify_opex_by_fluid():

    # Specify synthetic OPEX data
    opex_mangga = OPEX(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        fixed_cost=np.array(
            [
                200, 200, 200, 200,
                150, 100, 100, 75,
                50, 50, 50, 50
            ]
        ),
        cost_allocation=(
            [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
            ]
        ),
        cost_type=(
            [
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            ]
        ),
        tax_portion=np.array(
            [
                1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
            ]
        ),
    )

    opex_apel = OPEX(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        fixed_cost=np.array(
            [
                200, 200, 200, 200,
                150, 100, 100, 75,
                50, 50, 50, 50
            ]
        ),
        cost_allocation=(
            [
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
            ]
        ),
        cost_type=(
            [
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            ]
        ),
        tax_portion=np.array(
            [
                1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
            ]
        ),
    )

    # Create OPEX instances
    pr1 = BaseProject(
        start_date=date(year=2023, month=1, day=1),
        end_date=date(year=2034, month=12, day=31),
        oil_onstream_date=date(year=2030, month=1, day=1),
        gas_onstream_date=date(year=2029, month=1, day=1),
        approval_year=2027,
        opex=tuple([opex_mangga]),
    )

    pr2 = BaseProject(
        start_date=date(year=2023, month=1, day=1),
        end_date=date(year=2034, month=12, day=31),
        oil_onstream_date=date(year=2030, month=1, day=1),
        gas_onstream_date=date(year=2029, month=1, day=1),
        approval_year=2027,
        opex=tuple([opex_apel]),
    )

    # Define calculated and expected results
    calc_oil_1 = pr1._classify_opex_by_fluid(fluid_type=FluidType.OIL)
    calc_gas_1 = pr1._classify_opex_by_fluid(fluid_type=FluidType.GAS)
    calc_oil_2 = pr2._classify_opex_by_fluid(fluid_type=FluidType.OIL)
    calc_gas_2 = pr2._classify_opex_by_fluid(fluid_type=FluidType.GAS)

    calculated = {
        "oil_1_cost": calc_oil_1.fixed_cost,
        "gas_1_cost": calc_gas_1.fixed_cost,
        "oil_2_cost": calc_oil_2.fixed_cost,
        "gas_2_cost": calc_gas_2.fixed_cost,
    }

    expected = {
        "oil_1_cost": np.array([200, 200, 200, 200, 50, 50, 50, 50]),
        "gas_1_cost": np.array([150, 100, 100, 75]),
        "oil_2_cost": np.array([0]),
        "gas_2_cost": np.array([200, 200, 200, 200, 150, 100, 100, 75, 50, 50, 50, 50]),
    }

    # Execute testings
    np.testing.assert_allclose(calculated["oil_1_cost"], expected["oil_1_cost"])
    np.testing.assert_allclose(calculated["gas_1_cost"], expected["gas_1_cost"])
    np.testing.assert_allclose(calculated["oil_2_cost"], expected["oil_2_cost"])
    np.testing.assert_allclose(calculated["gas_2_cost"], expected["gas_2_cost"])
