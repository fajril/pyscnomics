"""
A collection of unit testing for Tangible class
"""

import pytest
import numpy as np
from pyscnomics.econ.selection import DeprMethod, FluidType, TaxType
from pyscnomics.econ.costs import CapitalException, CapitalCost


def test_capital_incorrect_year_input():
    """A unit testing for incorrect data input: start year is after end year"""

    with pytest.raises(CapitalException):
        CapitalCost(
            start_year=2030,
            end_year=2023,
            cost=np.array([200, 200]),
            expense_year=np.array([2023, 2024]),
            salvage_value=np.array([25, 25]),
            useful_life=np.array([5, 5]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )


def test_capital_unequal_length_of_data_input():
    """A unit testing for incorrect data input: unequal length of data input"""

    with pytest.raises(CapitalException):
        CapitalCost(
            start_year=2023,
            end_year=2032,
            cost=np.array([200, 200]),
            expense_year=np.array([2023, 2024, 2025]),
            salvage_value=np.array([25, 25]),
            useful_life=np.array([5, 5]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )


def test_capital_incorrect_expense_year_input():
    """A unit testing for incorrect data input: expense year is after end year of the project"""

    with pytest.raises(CapitalException):
        CapitalCost(
            start_year=2023,
            end_year=2032,
            cost=np.array([200, 200]),
            expense_year=np.array([2023, 2040]),
            salvage_value=np.array([25, 25]),
            useful_life=np.array([5, 5]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )

    with pytest.raises(CapitalException):
        CapitalCost(
            start_year=2023,
            end_year=2032,
            cost=np.array([200, 200]),
            expense_year=np.array([2020, 2023]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )


def test_capital_comparison():
    """Units testing for comparison operation"""

    mangga_capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([200, 200]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25, 25]),
        useful_life=np.array([5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    jeruk_capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([200, 200]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25, 25]),
        useful_life=np.array([5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    hiu_capital = CapitalCost(
        start_year=2025,
        end_year=2030,
        cost=np.array([100, 100]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=[FluidType.GAS, FluidType.GAS],
    )

    # Execute comparison operations
    assert mangga_capital == jeruk_capital
    assert mangga_capital != hiu_capital
    assert mangga_capital == 400
    assert mangga_capital >= hiu_capital
    assert hiu_capital <= jeruk_capital


def test_capital_arithmetics_misuse():
    """A unit testing for incorrect arithmetic operations"""

    mangga_capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100]),
        expense_year=np.array([2023, 2024]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    with pytest.raises(CapitalException):
        assert mangga_capital + 500

    with pytest.raises(CapitalException):
        assert mangga_capital - 100

    with pytest.raises(CapitalException):
        assert mangga_capital / 0


def test_capital_arithmetics():
    """A unit testing for mathematical operations (associated with an instance of Capital class)"""

    mangga_capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100]),
        expense_year=np.array([2024, 2023]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    apel_capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([50, 50]),
        expense_year=np.array([2026, 2027]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    jeruk_capital = CapitalCost(
        start_year=2023,
        end_year=2032,
        cost=np.array([200]),
        expense_year=np.array([2023]),
        cost_allocation=[FluidType.OIL],
    )

    # Expected results
    add1 = np.array([100, 100, 50, 50])
    add2 = np.array([100, 100, 50, 50, 200])
    mul1 = np.array([200, 200])
    mul2 = np.array([-25, -25])
    div1 = np.array([10])
    div2 = 0.5

    # Calculated results
    calc_add1 = mangga_capital + apel_capital
    calc_add2 = mangga_capital + apel_capital + jeruk_capital
    calc_mul1 = mangga_capital * 2
    calc_mul2 = -0.5 * apel_capital
    calc_div1 = jeruk_capital / 20
    calc_div2 = apel_capital / mangga_capital

    # Execute tests
    np.testing.assert_allclose(add1, calc_add1.cost)
    np.testing.assert_allclose(add2, calc_add2.cost)
    np.testing.assert_allclose(mul1, calc_mul1.cost)
    np.testing.assert_allclose(mul2, calc_mul2.cost)
    np.testing.assert_allclose(div1, calc_div1.cost)
    assert div2 == calc_div2


def test_capital_expenditures():
    """A unit testing for expenditures method in Capital class"""

    # Expected result
    expenses = [100, 100, 150, 0, 0, 0, 0, 0, 0, 0]

    # Calculated result
    mangga_capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 100]),
        expense_year=np.array([2023, 2024, 2025]),
        useful_life=np.array([5, 5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    jeruk_capital = CapitalCost(
        start_year=2025,
        end_year=2032,
        cost=np.array([50]),
        expense_year=np.array([2025]),
        useful_life=np.array([5]),
        cost_allocation=[FluidType.OIL],
    )

    total_capital = mangga_capital + jeruk_capital
    calc_expenses = total_capital.expenditures_post_tax()

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(expenses, calc_expenses)


def test_capital_expenditures_with_tax_and_inflation():
    """
    A unit testing for expenditures method in Capital class,
    taking into account tax and inflation schemes.
    """

    # Expected results
    case1 = np.array([0, 102, 104.04, 106.1208, 108.243216, 0, 0, 0])
    case2 = np.array([0, 102, 105.06, 109.2624, 114.72552, 0, 0, 0])
    case3 = np.array([0, 112, 112, 112, 112, 0, 0, 0])

    # Calculated results
    mangga_capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 100, 50]),
        expense_year=np.array([2024, 2025, 2026, 2027]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
        tax_portion=np.array([1, 1, 1, 1]),
    )

    jeruk_capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([50]),
        expense_year=np.array([2027]),
        cost_allocation=[FluidType.OIL],
        tax_portion=np.array([1,]),
    )

    capital_add = mangga_capital + jeruk_capital

    case1_calc = capital_add.expenditures_post_tax(
        inflation_rate=0.02
    )
    case2_calc = capital_add.expenditures_post_tax(
        inflation_rate=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
    )
    case3_calc = capital_add.expenditures_post_tax(
        tax_rate=0.12,
    )

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(case1, case1_calc)
    np.testing.assert_allclose(case2, case2_calc)
    np.testing.assert_allclose(case3, case3_calc)


def test_capital_depreciation():
    """A unit testing for total depreciation rate of Capital object"""

    # Expected results: depreciation_charge, undepreciated asset
    depreSL = [0, 0, 20, 30, 30, 30, 30, 10]
    depreDB = [0, 0, 20, 26, 20.8, 16.64, 46.08, 20.48]
    depreDDB = [0, 0, 40, 24, 14.4, 28.64, 24.96, 7.2]
    deprePSC = [0, 0, 50, 25, 12.5, 6.25, 31.25, 12.5]

    undepreSL = 0
    undepreDB = 0
    undepreDDB = 10.8
    undeprePSC = 12.5

    # Calculated results: depreciation_charge, undepreciated asset
    mangga_capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
        useful_life=np.array([5, 5]),
    )

    apel_capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2028]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
        useful_life=np.array([5, 5]),
    )

    nanas_capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
        useful_life=np.array([5, 5]),
    )

    calc_depreSL, calc_undepreSL = mangga_capital.total_depreciation_rate(
        depr_method=DeprMethod.SL
    )

    calc_depreDB, calc_undepreDB = mangga_capital.total_depreciation_rate(
        depr_method=DeprMethod.DB, decline_factor=1
    )

    calc_depreDDB, calc_undepreDDB = apel_capital.total_depreciation_rate(
        depr_method=DeprMethod.DB, decline_factor=2
    )

    calc_deprePSC, calc_undeprePSC = nanas_capital.total_depreciation_rate(
        depr_method=DeprMethod.PSC_DB
    )

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(depreSL, calc_depreSL)
    np.testing.assert_allclose(depreDB, calc_depreDB)
    np.testing.assert_allclose(depreDDB, calc_depreDDB)
    np.testing.assert_allclose(deprePSC, calc_deprePSC)

    np.testing.assert_allclose(np.array([undepreSL]), np.array([calc_undepreSL]))
    np.testing.assert_allclose(np.array([undepreDB]), np.array([calc_undepreDB]))
    np.testing.assert_allclose(np.array([undepreDDB]), np.array([calc_undepreDDB]))
    np.testing.assert_allclose(np.array([undeprePSC]), np.array([calc_undeprePSC]))


def test_capital_depreciation_with_tax_and_inflation():
    """A unit testing for total depreciation rate of Capital object"""

    # Expected results: depreciation_charge, undepreciated asset
    depreCase1 = [
        50,
        77.5,
        93.875,
        75.878125,
        71.45171875,
        35.88210938,
        18.10511719,
        7.416035156,
    ]

    depreCase2 = [
        50,
        76,
        90.53,
        72.5806,
        68.09668,
        34.11084,
        17.151045,
        6.9996225,
    ]

    undepreCase1 = 3.798457031
    undepreCase2 = 3.58517249999

    # Calculated results
    capital_mangga = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 100, 50, 50, 0, 0, 0]),
        expense_year=np.array([2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]),
        cost_allocation=[
            FluidType.OIL,
            FluidType.OIL,
            FluidType.OIL,
            FluidType.OIL,
            FluidType.OIL,
            FluidType.OIL,
            FluidType.OIL,
            FluidType.OIL,
        ],
    )

    depreCase1_calc, undepreCase1_calc = capital_mangga.total_depreciation_rate(
        inflation_rate=0.05
    )

    depreCase2_calc, undepreCase2_calc = capital_mangga.total_depreciation_rate(
        tax_rate=np.array([0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]),
        inflation_rate=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
    )

    # Execute testing
    np.testing.assert_allclose(depreCase1, depreCase1_calc)
    np.testing.assert_allclose(undepreCase1, undepreCase1_calc)
    np.testing.assert_allclose(depreCase2, depreCase2_calc)
    np.testing.assert_allclose(undepreCase2, undepreCase2_calc)


def test_capital_book_value():
    """A unit testing for total depreciation book value of Capital object"""

    # Expected results: depreciation book value
    bookSL = [0, 0, 80, 100, 70, 40, 10, 0]
    bookDB = [0, 0, 80, 104, 83.2, 66.56, 20.48, 0]
    bookDDB = [0, 0, 60, 36, 21.6, 42.96, 18, 10.8]
    bookPSC = [0, 0, 50, 25, 12.5, 6.25, 25, 12.5]

    # Calculated results: depreciation book value
    mangga_capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        useful_life=np.array([5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    apel_capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2028]),
        useful_life=np.array([5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    nanas_capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2029]),
        useful_life=np.array([5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    calc_bookSL = mangga_capital.total_depreciation_book_value(
        depr_method=DeprMethod.SL
    )

    calc_bookDB = mangga_capital.total_depreciation_book_value(
        depr_method=DeprMethod.DB, decline_factor=1
    )

    calc_bookDDB = apel_capital.total_depreciation_book_value(
        depr_method=DeprMethod.DB, decline_factor=2
    )

    calc_bookPSC = nanas_capital.total_depreciation_book_value(
        depr_method=DeprMethod.PSC_DB
    )

    # Execute testing
    np.testing.assert_allclose(bookSL, calc_bookSL)
    np.testing.assert_allclose(bookDB, calc_bookDB)
    np.testing.assert_allclose(bookDDB, calc_bookDDB)
    np.testing.assert_allclose(bookPSC, calc_bookPSC)
