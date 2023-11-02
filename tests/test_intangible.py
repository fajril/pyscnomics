"""
A collection of unit testing for Intangible class.
"""

import pytest
import numpy as np
from pyscnomics.econ.selection import FluidType, TaxType
from pyscnomics.econ.costs import Tangible, Intangible, IntangibleException


def test_intangible_incorrect_year_input():
    """ A unit testing for incorrect data input: start year is after end year """

    with pytest.raises(IntangibleException):
        Intangible(
            start_year=2033,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )


def test_intangible_unequal_length_of_data_input():
    """ A unit testing for incorrect data input: unequal length of data input """

    with pytest.raises(IntangibleException):
        Intangible(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2028]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )


def test_intangible_incorrect_expense_year_input():
    """ A unit testing for incorrect data input: incorrect expense year """

    with pytest.raises(IntangibleException):
        Intangible(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2035, 2036]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )

    with pytest.raises(IntangibleException):
        Intangible(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2021, 2025]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )


def test_intangible_comparison_error():
    """ A unit testing for misuse of Intangible: comparing different instances/objects """

    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        useful_life=np.array([5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    mangga_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    # Execute testing
    check_equality = mangga_intangible == mangga_tangible

    assert check_equality is False


def test_intangible_comparison():
    """ A unit testing for comparison between Intangible instances """

    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([50, 50]),
        expense_year=np.array([2025, 2029]),
        useful_life=np.array([5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    mangga_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    apel_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    jeruk_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    # Execute comparison testing
    assert mangga_intangible == apel_intangible
    assert mangga_intangible != jeruk_intangible
    assert mangga_intangible < jeruk_intangible
    assert mangga_tangible <= mangga_intangible
    assert mangga_intangible >= 100


def test_intangible_arithmetics_incorrect():
    """ A unit testing for misuse of arithmetic operations upon an instance of Intangible """

    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2029]),
        useful_life=np.array([5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    mangga_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    with pytest.raises(IntangibleException):
        assert mangga_intangible + 100

    with pytest.raises(IntangibleException):
        assert mangga_intangible - 50

    with pytest.raises(IntangibleException):
        assert mangga_intangible - mangga_tangible

    with pytest.raises(IntangibleException):
        assert mangga_intangible * mangga_intangible

    with pytest.raises(IntangibleException):
        assert mangga_intangible / 0


def test_intangible_arithmetics():
    """ A unit testing for arithmetic operations upon an instance of Intangible """

    # Expected results
    add = [0, 0, 0, 0, 0, 200, 100, 0]
    sub = [-100, 0, 0, 0, 0, 100, 50, -50]
    mul = [0, 0, 0, 0, 0, -200, -100, 0]
    div = [0, 0, 0, 0, 0, 50, 25, 0]

    # Calculated results
    mangga_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    apel_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    jeruk_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2023, 2030]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    calc_add = mangga_intangible + apel_intangible
    calc_sub = apel_intangible - jeruk_intangible
    calc_mul = -2 * mangga_intangible
    calc_div = mangga_intangible / 2

    # Execute testing
    np.testing.assert_allclose(add, calc_add.expenditures())
    np.testing.assert_allclose(sub, calc_sub.expenditures())
    np.testing.assert_allclose(mul, calc_mul.expenditures())
    np.testing.assert_allclose(div, calc_div.expenditures())


def test_intangible_expenditures():
    """ A unit testing for expenditures method in Intangible class """

    # Expected result
    expense1 = [0, 0, 0, 0, 0, 100, 50, 0]
    expense2 = [50, 50, 0, 0, 0, 100, 150, 100]

    # Calculated result
    intangible_mangga = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    intangible_apel = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([50, 50, 100, 100]),
        expense_year=np.array([2023, 2024, 2029, 2030]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    expense1_calc = intangible_mangga.expenditures()
    expense2_calc = (intangible_mangga + intangible_apel).expenditures()

    # Execute testing
    np.testing.assert_allclose(expense1, expense1_calc)
    np.testing.assert_allclose(expense2, expense2_calc)


def test_intangible_expenditures_with_tax_and_inflation():
    """ A unit testing for expenditures method in Intangible class with inflation scheme """

    # Expected result
    case1 = np.array([0, 102, 105.06, 109.2624, 114.72552, 0, 0, 0])
    case2 = np.array([
        0,
        105.57,
        108.7371,
        113.086584,
        118.7409132,
        0,
        0,
        0,
    ])

    # Calculated result
    intangible_mangga = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([0, 100, 100, 100, 100, 0, 0, 0]),
        expense_year=np.array([2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]),
        cost_allocation=[
            FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
            FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS
        ],
    )

    intangible_apel = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([0, 100, 100, 100, 100, 0, 0, 0]),
        expense_year=np.array([2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]),
        cost_allocation=[
            FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
            FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS
        ],
        lbt_discount=np.array([0, 0.3, 0.3, 0.3, 0.3, 0, 0, 0]),
    )

    case1_calc = intangible_mangga.expenditures(
        inflation_rate=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
    )

    case2_calc = intangible_apel.expenditures(
        tax_type=TaxType.LBT,
        inflation_rate=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
        lbt_rate=0.05,
    )

    # Execute testing
    np.testing.assert_allclose(case1, case1_calc)
    np.testing.assert_allclose(case2, case2_calc)
