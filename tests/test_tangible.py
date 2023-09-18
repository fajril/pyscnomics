"""
A collection of unit testing for Tangible class
"""

import pytest
import numpy as np
from pyscnomics.econ.selection import DeprMethod, FluidType
from pyscnomics.econ.costs import TangibleException, Tangible


def test_tangible_incorrect_year_input():
    """A unit testing for incorrect data input: start year is after end year"""

    with pytest.raises(TangibleException):

        Tangible(
            start_year=2030,
            end_year=2023,
            cost=np.array([200, 200]),
            expense_year=np.array([2023, 2024]),
            salvage_value=np.array([25, 25]),
            useful_life=np.array([5, 5]),
            cost_allocation=FluidType.OIL,
        )


def test_tangible_unequal_length_of_data_input():
    """A unit testing for incorrect data input: unequal length of data input"""

    with pytest.raises(TangibleException):

        Tangible(
            start_year=2023,
            end_year=2032,
            cost=np.array([200, 200]),
            expense_year=np.array([2023, 2024, 2025]),
            salvage_value=np.array([25, 25]),
            useful_life=np.array([5, 5]),
            cost_allocation=FluidType.OIL,
        )


def test_tangible_incorrect_expense_year_input():
    """A unit testing for incorrect data input: expense year is after end year of the project"""

    with pytest.raises(TangibleException):

        Tangible(
            start_year=2023,
            end_year=2032,
            cost=np.array([200, 200]),
            expense_year=np.array([2023, 2040]),
            salvage_value=np.array([25, 25]),
            useful_life=np.array([5, 5]),
            cost_allocation=FluidType.OIL,
        )

    with pytest.raises(TangibleException):

        Tangible(
            start_year=2023,
            end_year=2032,
            cost=np.array([200, 200]),
            expense_year=np.array([2020, 2023]),
            cost_allocation=FluidType.OIL,
        )


def test_tangible_expenditures():
    """A unit testing for expenditures method in Tangible class"""

    # Expected result
    expenses = [100, 100, 150, 0, 0, 0, 0, 0, 0, 0]

    # Calculated result
    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 100]),
        expense_year=np.array([2023, 2024, 2025]),
        useful_life=np.array([5, 5, 5]),
        cost_allocation=FluidType.OIL,
    )

    jeruk_tangible = Tangible(
        start_year=2025,
        end_year=2032,
        cost=np.array([50]),
        expense_year=np.array([2025]),
        useful_life=np.array([5]),
        cost_allocation=FluidType.OIL,
    )

    total_tangible = mangga_tangible + jeruk_tangible
    calc_expenses = total_tangible.expenditures()

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(expenses, calc_expenses)


def test_tangible_comparison():
    """Units testing for comparison operation"""

    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([200, 200]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25, 25]),
        useful_life=np.array([5, 5]),
        cost_allocation=FluidType.OIL,
    )

    jeruk_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([200, 200]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25, 25]),
        useful_life=np.array([5, 5]),
        cost_allocation=FluidType.OIL,
    )

    hiu_tangible = Tangible(
        start_year=2025,
        end_year=2030,
        cost=np.array([100, 100]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=FluidType.GAS,
    )

    # Execute comparison operations
    assert mangga_tangible == jeruk_tangible
    assert mangga_tangible != hiu_tangible
    assert mangga_tangible == 400
    assert mangga_tangible >= hiu_tangible
    assert hiu_tangible <= jeruk_tangible


def test_tangible_arithmetics_incorrect_fluidtype():
    """A unit testing for incorrect addition operation: addition between two different fluid types"""

    with pytest.raises(TangibleException):

        mangga_tangible_oil = Tangible(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 100]),
            expense_year=np.array([2023, 2024]),
            cost_allocation=FluidType.OIL,
        )

        mangga_tangible_gas = Tangible(
            start_year=2023,
            end_year=2030,
            cost=np.array([50, 50]),
            expense_year=np.array([2026, 2027]),
            cost_allocation=FluidType.GAS,
        )

        assert mangga_tangible_oil + mangga_tangible_gas


def test_tangible_arithmetics_misuse():
    """A unit testing for incorrect arithmetic operations"""

    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100]),
        expense_year=np.array([2023, 2024]),
        cost_allocation=FluidType.OIL,
    )

    with pytest.raises(TangibleException):
        assert mangga_tangible + 500

    with pytest.raises(TangibleException):
        assert mangga_tangible - 100

    with pytest.raises(TangibleException):
        assert mangga_tangible / 0


def test_tangible_arithmetics():
    """A unit testing for mathematical operations (associated with an instance of Tangible class)"""

    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100]),
        expense_year=np.array([2024, 2023]),
        cost_allocation=FluidType.OIL,
    )

    apel_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([50, 50]),
        expense_year=np.array([2026, 2027]),
        cost_allocation=FluidType.OIL,
    )

    jeruk_tangible = Tangible(
        start_year=2023,
        end_year=2032,
        cost=np.array([200]),
        expense_year=np.array([2023]),
        cost_allocation=FluidType.OIL,
    )

    # Expected results
    add1 = np.array([100, 100, 50, 50])
    add2 = np.array([100, 100, 50, 50, 200])
    mul1 = np.array([200, 200])
    mul2 = np.array([-25, -25])
    div1 = np.array([10])
    div2 = 0.5

    # Calculated results
    calc_add1 = mangga_tangible + apel_tangible
    calc_add2 = mangga_tangible + apel_tangible + jeruk_tangible
    calc_mul1 = mangga_tangible * 2
    calc_mul2 = -0.5 * apel_tangible
    calc_div1 = jeruk_tangible / 20
    calc_div2 = apel_tangible / mangga_tangible

    # Execute tests
    np.testing.assert_allclose(add1, calc_add1.cost)
    np.testing.assert_allclose(add2, calc_add2.cost)
    np.testing.assert_allclose(mul1, calc_mul1.cost)
    np.testing.assert_allclose(mul2, calc_mul2.cost)
    np.testing.assert_allclose(div1, calc_div1.cost)
    assert div2 == calc_div2


def test_tangible_total_depreciation_rate():
    """A unit testing for total depreciation rate of Tangible object"""

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
    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=FluidType.OIL,
        useful_life=np.array([5, 5]),
    )

    apel_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2028]),
        cost_allocation=FluidType.OIL,
        useful_life=np.array([5, 5]),
    )

    nanas_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2029]),
        cost_allocation=FluidType.OIL,
        useful_life=np.array([5, 5]),
    )

    calc_depreSL, calc_undepreSL = mangga_tangible.total_depreciation_rate(
        depr_method=DeprMethod.SL
    )

    calc_depreDB, calc_undepreDB = mangga_tangible.total_depreciation_rate(
        depr_method=DeprMethod.DB, decline_factor=1
    )

    calc_depreDDB, calc_undepreDDB = apel_tangible.total_depreciation_rate(
        depr_method=DeprMethod.DB, decline_factor=2
    )

    calc_deprePSC, calc_undeprePSC = nanas_tangible.total_depreciation_rate(
        depr_method=DeprMethod.PSC
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


def test_tangible_total_depreciation_book_value():
    """A unit testing for total depreciation book value of Tangible object"""

    # Expected results: depreciation book value
    bookSL = [0, 0, 80, 100, 70, 40, 10, 0]
    bookDB = [0, 0, 80, 104, 83.2, 66.56, 20.48, 0]
    bookDDB = [0, 0, 60, 36, 21.6, 42.96, 18, 10.8]
    bookPSC = [0, 0, 50, 25, 12.5, 6.25, 25, 12.5]

    # Calculated results: depreciation book value
    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=FluidType.OIL,
        useful_life=np.array([5, 5]),
    )

    apel_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2028]),
        cost_allocation=FluidType.OIL,
        useful_life=np.array([5, 5]),
    )

    nanas_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2029]),
        cost_allocation=FluidType.OIL,
        useful_life=np.array([5, 5]),
    )

    calc_bookSL = mangga_tangible.total_depreciation_book_value(
        depr_method=DeprMethod.SL
    )

    calc_bookDB = mangga_tangible.total_depreciation_book_value(
        depr_method=DeprMethod.DB, decline_factor=1
    )

    calc_bookDDB = apel_tangible.total_depreciation_book_value(
        depr_method=DeprMethod.DB, decline_factor=2
    )

    calc_bookPSC = nanas_tangible.total_depreciation_book_value(
        depr_method=DeprMethod.PSC
    )

    # Execute testing
    np.testing.assert_allclose(bookSL, calc_bookSL)
    np.testing.assert_allclose(bookDB, calc_bookDB)
    np.testing.assert_allclose(bookDDB, calc_bookDDB)
    np.testing.assert_allclose(bookPSC, calc_bookPSC)
