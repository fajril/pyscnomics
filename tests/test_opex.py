"""
A collection of unit testings for OPEX class.
"""

import pytest
import numpy as np
from pyscnomics.econ.selection import FluidType, CostType
from pyscnomics.econ.costs import OPEX, OPEXException, Intangible


# Parameters for example
expense_year_1 = np.array([2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])
cost_1 = np.array([200, 200, 200, 150, 100, 75, 25, 25])
cost_2 = np.array([200, 200, 200, 150])
cost_allocation_1 = [
    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
]
cost_type_1 = [
    np.nan, np.nan, CostType.SUNK_COST, CostType.SUNK_COST,
    CostType.SUNK_COST, CostType.SUNK_COST, None, None,
]
cost_type_2 = [
    "CostType.SUNK_COST", CostType.SUNK_COST, CostType.SUNK_COST,
    CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
    CostType.POST_ONSTREAM_COST,
]
cost_type_3 = [
    CostType.SUNK_COST, CostType.SUNK_COST, CostType.SUNK_COST,
    CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
    CostType.POST_ONSTREAM_COST,
]
tax_portion_1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])


def test_opex_incorrect_year_input():
    """A unit testing for incorrect data input: start year is after end year"""

    with pytest.raises(OPEXException):

        OPEX(
            start_year=2033,
            end_year=2030,
            fixed_cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )


def test_opex_unequal_length_input_1():
    with pytest.raises(OPEXException):
        OPEX(
            start_year=2023,
            end_year=2030,
            expense_year=expense_year_1,
            fixed_cost=cost_2,
            cost_allocation=cost_allocation_1,
            cost_type=cost_type_3,
        )


def test_opex_unequal_length_input_2():
    with pytest.raises(OPEXException):
        OPEX(
            start_year=2023,
            end_year=2030,
            fixed_cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
            prod_rate=np.array([100, 50]),
            cost_per_volume=np.array([0.1, 0.1, 0.1]),
        )


def test_opex_incorrect_expense_year_input():
    """A unit testing for incorrect data input: expense year is after the end year of the project"""

    with pytest.raises(OPEXException):

        OPEX(
            start_year=2023,
            end_year=2030,
            fixed_cost=np.array([100, 50]),
            expense_year=np.array([2028, 2033]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )


def test_opex_comparison_error():
    """A unit testing for misuse of OPEX: comparing different instances/objects"""

    opex_mangga = OPEX(
        start_year=2023,
        end_year=2030,
        expense_year=expense_year_1,
        fixed_cost=cost_1,
        cost_allocation=cost_allocation_1,
        cost_type=cost_type_1,
    )

    opex_apel = OPEX(
        start_year=2023,
        end_year=2030,
        expense_year=expense_year_1,
        fixed_cost=cost_1,
        cost_allocation=cost_allocation_1,
        cost_type=cost_type_1,
    )

    check_equality = (opex_mangga == opex_apel)

    assert check_equality is True


def test_opex_comparison():
    """A unit testing for comparison between OPEX instances"""

    opex_mangga = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    opex_apel = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    opex_jeruk = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([50, 25]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    opex_nanas = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
        prod_rate=np.array([100, 100]),
        cost_per_volume=np.array([0.1, 0.1]),
    )

    # Execute comparison testing
    assert opex_mangga == opex_apel
    assert opex_mangga != opex_jeruk
    assert opex_jeruk < opex_apel
    assert opex_mangga <= opex_nanas
    assert opex_apel > opex_jeruk
    assert opex_apel >= opex_jeruk


def test_opex_arithmetics_incorrect():
    """A unit testing for misuse of arithmetic operations upon an instance of OPEX"""

    mangga_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    mangga_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    with pytest.raises(OPEXException):
        assert mangga_opex + 100

    with pytest.raises(OPEXException):
        assert mangga_opex - 50

    with pytest.raises(OPEXException):
        assert mangga_opex * mangga_intangible

    with pytest.raises(OPEXException):
        assert mangga_opex / 0


def test_opex_dunder_add():

    opex_mangga = OPEX(
        start_year=2023,
        end_year=2030,
        expense_year=expense_year_1,
        fixed_cost=cost_1,
        cost_allocation=cost_allocation_1,
        cost_type=cost_type_1,
    )

    opex_apel = OPEX(
        start_year=2023,
        end_year=2030,
        expense_year=expense_year_1,
        fixed_cost=cost_1,
        cost_allocation=cost_allocation_1,
        cost_type=cost_type_3,
    )

    opex_add = opex_mangga + opex_apel

    expected = {
        "expense_year": np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030,
                2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030,
            ]
        ),
        "fixed_cost": np.array(
            [
                200, 200, 200, 150, 100, 75, 25, 25,
                200, 200, 200, 150, 100, 75, 25, 25,
            ]
        ),
        "cost_allocation": (
            [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
            ]
        ),
        "cost_type": (
            [
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.SUNK_COST, CostType.SUNK_COST, CostType.SUNK_COST,
                CostType.SUNK_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST,
                CostType.SUNK_COST, CostType.SUNK_COST, CostType.SUNK_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST,
            ]
        ),
    }

    np.testing.assert_allclose(opex_add.expense_year, expected["expense_year"])
    np.testing.assert_allclose(opex_add.fixed_cost, expected["fixed_cost"])
    assert opex_add.cost_allocation == expected["cost_allocation"]
    assert opex_add.cost_type == expected["cost_type"]


def test_opex_arithmetics():

    # Expected results
    add1 = [110, 110, 170, 170, 60, 60, 0, 0]
    add2 = [60, 60, 0, 0, 0, 0, 210, 210]
    mul1 = [0, 0, 0, 0, 0, 0, 420, 420]
    div1 = [30, 30, 0, 0, 0, 0, 0, 0]

    # Calculated results
    mangga_opex = OPEX(
        start_year=2023,
        end_year=2026,
        fixed_cost=np.array([100, 100, 100, 100]),
        expense_year=np.array([2023, 2024, 2025, 2026]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
        prod_rate=np.array([100, 100, 100, 100]),
        cost_per_volume=np.array([0.1, 0.1, 0.1, 0.1]),
    )

    apel_opex = OPEX(
        start_year=2025,
        end_year=2030,
        fixed_cost=np.array([50, 50, 50, 50]),
        expense_year=np.array([2025, 2026, 2027, 2028]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
        prod_rate=np.array([100, 100, 100, 100]),
        cost_per_volume=np.array([0.1, 0.1, 0.1, 0.1]),
    )

    nanas_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([200, 200]),
        expense_year=np.array([2030, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
        prod_rate=np.array([100, 100]),
        cost_per_volume=np.array([0.1, 0.1]),
    )

    jeruk_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([50, 50]),
        expense_year=np.array([2024, 2023]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
        prod_rate=np.array([100, 100]),
        cost_per_volume=np.array([0.1, 0.1]),
    )

    calc_add1 = (mangga_opex + apel_opex).expenditures_post_tax()
    calc_add2 = (nanas_opex + jeruk_opex).expenditures_post_tax()
    calc_mul1 = (nanas_opex * 2).expenditures_post_tax()
    calc_div1 = (jeruk_opex / 2).expenditures_post_tax()

    # Execute testing
    np.testing.assert_allclose(add1, calc_add1)
    np.testing.assert_allclose(add2, calc_add2)
    np.testing.assert_allclose(mul1, calc_mul1)
    np.testing.assert_allclose(div1, calc_div1)


def test_opex_expenditures():
    """A unit testing for expenditures method in OPEX class"""

    # Expected results
    expense1 = [0, 0, 0, 0, 0, 100, 50, 0]
    expense2 = [0, 0, 0, 0, 0, 110, 60, 0]

    # Calculated results
    mangga_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    apel_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
        prod_rate=np.array([100, 100]),
        cost_per_volume=np.array([0.1, 0.1]),
    )

    calc_expense1 = mangga_opex.expenditures_post_tax()
    calc_expense2 = apel_opex.expenditures_post_tax()

    # Execute testing
    np.testing.assert_allclose(expense1, calc_expense1)
    np.testing.assert_allclose(expense2, calc_expense2)


def test_opex_expenditures_with_tax_and_inflation():
    """ A unit testing for expenditures method in OPEX class """

    # Expected results
    case1 = np.array([0, 105, 110.25, 115.7625, 121.550625, 0, 0, 0])
    case2 = np.array([0, 102, 103, 104, 105, 0, 0, 0])
    case3 = np.array([0, 107.18504, 113.69756, 119.46108, 124.31724, 0, 0, 0])

    # Calculated results
    opex_mangga = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([100, 100, 100, 50]),
        expense_year=np.array([2024, 2025, 2026, 2027]),
        tax_portion=np.array([1, 1, 1, 1]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    opex_jeruk = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([50]),
        tax_portion=np.array([1]),
        expense_year=np.array([2027]),
        cost_allocation=[FluidType.OIL],
    )

    opex_apel = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([100, 100, 100, 100]),
        expense_year=np.array([2024, 2025, 2026, 2027]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
        tax_portion=np.array([0.36, 0.36, 0.36, 0.36]),
        tax_discount=0.743,
    )

    opex_add = opex_mangga + opex_jeruk

    case1_calc = opex_add.expenditures_post_tax(inflation_rate=0.05)
    case2_calc = opex_add.expenditures_post_tax(
        tax_rate=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
    )
    case3_calc = opex_apel.expenditures_post_tax(
        tax_rate=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
        inflation_rate=np.array([0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]),
    )

    # Execute testing
    np.testing.assert_allclose(case1, case1_calc)
    np.testing.assert_allclose(case2, case2_calc)
    np.testing.assert_allclose(case3, case3_calc)
