"""
A collection of unit testings to validate operations occurred
in class BaseProject: sunk cost
"""

from datetime import date
import numpy as np

from pyscnomics.econ.selection import OtherRevenue, FluidType, CostType
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales
from pyscnomics.contracts.project import BaseProject


# Synthetic
params_base = {
    "sulfur_revenue": OtherRevenue.REDUCTION_TO_OIL_OPEX,
    "electricity_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
    "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
    "tax_rate": 0.0,
    "year_inflation": None,
    "inflation_rate": 0.0,
    "inflation_rate_applied_to": None,
}


def test_base_project_get_oil_lifting():

    # Synthetic Data Lifting
    oil_lifting = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2030, 2031, 2032, 2033, 2034]),
        lifting_rate=np.array([100, 100, 100, 100, 100]),
        price=np.array([10, 10, 10, 10, 10]),
        fluid_type=FluidType.OIL,
    )

    gas_lifting = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2030, 2031, 2032, 2033, 2029]),
        lifting_rate=np.array([10, 10, 10, 10, 10]),
        price=np.array([1, 1, 1, 1, 1]),
        fluid_type=FluidType.GAS,
    )

    total_lifting = tuple([oil_lifting, gas_lifting])

    pr = BaseProject(
        start_date=date(year=2023, month=1, day=1),
        end_date=date(year=2034, month=12, day=31),
        oil_onstream_date=date(year=2030, month=1, day=1),
        gas_onstream_date=date(year=2029, month=1, day=1),
        approval_year=2027,
        lifting=total_lifting,
    )

    calculated = {
        "lifting_rate": pr._get_oil_lifting().lifting_rate,
        "price": pr._get_oil_lifting().price,
    }

    expected = {
        "lifting_rate": np.array([100, 100, 100, 100, 100]),
        "price": np.array([10, 10, 10, 10, 10]),
    }

    np.testing.assert_allclose(calculated["lifting_rate"], expected["lifting_rate"])
    np.testing.assert_allclose(calculated["price"], expected["price"])


def test_base_project_get_gas_lifting():

    # Synthetic Data Lifting
    oil_lifting = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2030, 2031, 2032, 2033, 2034]),
        lifting_rate=np.array([100, 100, 100, 100, 100]),
        price=np.array([10, 10, 10, 10, 10]),
        fluid_type=FluidType.OIL,
    )

    gas_lifting = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2030, 2031, 2032, 2033, 2029]),
        lifting_rate=np.array([10, 10, 10, 10, 10]),
        price=np.array([1, 1, 1, 1, 1]),
        fluid_type=FluidType.GAS,
    )

    total_lifting = tuple([oil_lifting, gas_lifting])

    pr = BaseProject(
        start_date=date(year=2023, month=1, day=1),
        end_date=date(year=2034, month=12, day=31),
        oil_onstream_date=date(year=2030, month=1, day=1),
        gas_onstream_date=date(year=2029, month=1, day=1),
        approval_year=2027,
        lifting=total_lifting,
    )

    calculated = {
        "lifting_rate": pr._get_gas_lifting().lifting_rate,
        "price": pr._get_gas_lifting().price,
    }
    expected = {
        "lifting_rate": np.array([10, 10, 10, 10, 10]),
        "price": np.array([1, 1, 1, 1, 1]),
    }

    np.testing.assert_allclose(calculated["lifting_rate"], expected["lifting_rate"])
    np.testing.assert_allclose(calculated["price"], expected["price"])


def test_base_project_get_sulfur_lifting():

    # Synthetic Data Lifting
    oil_lifting = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2030, 2031, 2032, 2033, 2034]),
        lifting_rate=np.array([100, 100, 100, 100, 100]),
        price=np.array([10, 10, 10, 10, 10]),
        fluid_type=FluidType.OIL,
    )

    gas_lifting = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2030, 2031, 2032, 2033, 2029]),
        lifting_rate=np.array([10, 10, 10, 10, 10]),
        price=np.array([1, 1, 1, 1, 1]),
        fluid_type=FluidType.GAS,
    )

    sulfur_lifting = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2030, 2031, 2032, 2033, 2029]),
        lifting_rate=np.array([5, 5, 5, 5, 5]),
        price=np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        fluid_type=FluidType.SULFUR,
    )

    total_lifting = tuple([oil_lifting, gas_lifting, sulfur_lifting])

    pr = BaseProject(
        start_date=date(year=2023, month=1, day=1),
        end_date=date(year=2034, month=12, day=31),
        oil_onstream_date=date(year=2030, month=1, day=1),
        gas_onstream_date=date(year=2029, month=1, day=1),
        approval_year=2027,
        lifting=total_lifting,
    )

    calculated = {
        "lifting_rate": pr._get_sulfur_lifting().lifting_rate,
        "price": pr._get_sulfur_lifting().price,
    }
    expected = {
        "lifting_rate": np.array([5, 5, 5, 5, 5]),
        "price": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    }

    np.testing.assert_allclose(calculated["lifting_rate"], expected["lifting_rate"])
    np.testing.assert_allclose(calculated["price"], expected["price"])


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
