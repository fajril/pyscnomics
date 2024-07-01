"""
A collection of unit testing for Tangible class
"""

import pytest
import numpy as np
from pyscnomics.econ.selection import DeprMethod, FluidType, TaxType
from pyscnomics.econ.costs import CapitalException, CapitalCost


def test_tangible_incorrect_year_input():
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


def test_tangible_unequal_length_of_data_input():
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


def test_tangible_incorrect_expense_year_input():
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


def test_tangible_comparison():
    """Units testing for comparison operation"""

    mangga_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([200, 200]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25, 25]),
        useful_life=np.array([5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    jeruk_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([200, 200]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25, 25]),
        useful_life=np.array([5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    hiu_tangible = CapitalCost(
        start_year=2025,
        end_year=2030,
        cost=np.array([100, 100]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=[FluidType.GAS, FluidType.GAS],
    )

    # Execute comparison operations
    assert mangga_tangible == jeruk_tangible
    assert mangga_tangible != hiu_tangible
    assert mangga_tangible == 400
    assert mangga_tangible >= hiu_tangible
    assert hiu_tangible <= jeruk_tangible


def test_tangible_arithmetics_misuse():
    """A unit testing for incorrect arithmetic operations"""

    mangga_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100]),
        expense_year=np.array([2023, 2024]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    with pytest.raises(CapitalException):
        assert mangga_tangible + 500

    with pytest.raises(CapitalException):
        assert mangga_tangible - 100

    with pytest.raises(CapitalException):
        assert mangga_tangible / 0


def test_tangible_arithmetics():
    """A unit testing for mathematical operations (associated with an instance of Tangible class)"""

    mangga_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100]),
        expense_year=np.array([2024, 2023]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    apel_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([50, 50]),
        expense_year=np.array([2026, 2027]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    jeruk_tangible = CapitalCost(
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


def test_tangible_expenditures():
    """A unit testing for expenditures method in Tangible class"""

    # Expected result
    expenses = [100, 100, 150, 0, 0, 0, 0, 0, 0, 0]

    # Calculated result
    mangga_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 100]),
        expense_year=np.array([2023, 2024, 2025]),
        useful_life=np.array([5, 5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    jeruk_tangible = CapitalCost(
        start_year=2025,
        end_year=2032,
        cost=np.array([50]),
        expense_year=np.array([2025]),
        useful_life=np.array([5]),
        cost_allocation=[FluidType.OIL],
    )

    total_tangible = mangga_tangible + jeruk_tangible
    calc_expenses = total_tangible.expenditures()

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(expenses, calc_expenses)


def test_tangible_expenditures_with_tax_and_inflation():
    """
    A unit testing for expenditures method in Tangible class,
    taking into account tax and inflation schemes.
    """

    # Expected results
    case1 = np.array([0, 102, 104.04, 106.1208, 108.243216, 0, 0, 0])
    case2 = np.array([0, 102, 105.06, 109.2624, 114.72552, 0, 0, 0])
    case3 = np.array([0, 105, 105, 105, 105, 0, 0, 0])
    case4 = np.array([0, 102.24, 101.92, 101.6, 107.688, 0, 0, 0])

    # Calculated results
    tangible_mangga = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 100, 50]),
        expense_year=np.array([2024, 2025, 2026, 2027]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    tangible_jeruk = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([50]),
        expense_year=np.array([2027]),
        cost_allocation=[FluidType.OIL],
    )

    tangible_apel = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 100, 100]),
        expense_year=np.array([2024, 2025, 2026, 2027]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
        lbt_portion=np.array([0.8, 0.8, 0.8, 0.8]),
        lbt_discount=np.array([0.6, 0.6, 0.6, 0.2]),

    )

    tangible_add = tangible_mangga + tangible_jeruk

    case1_calc = tangible_add.expenditures(
        year_ref=2023,
        inflation_rate=0.02
    )
    case2_calc = tangible_add.expenditures(
        year_ref=2023,
        inflation_rate=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
    )
    case3_calc = tangible_add.expenditures(
        year_ref=2023,
        tax_type=TaxType.VAT,
        vat_rate=0.05,
    )
    case4_calc = tangible_apel.expenditures(
        year_ref=2026,
        tax_type=TaxType.LBT,
        lbt_rate=np.array([0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]),
        inflation_rate=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
    )

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(case1, case1_calc)
    np.testing.assert_allclose(case2, case2_calc)
    np.testing.assert_allclose(case3, case3_calc)
    np.testing.assert_allclose(case4, case4_calc)


def test_tangible_depreciation():
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
    mangga_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
        useful_life=np.array([5, 5]),
    )

    apel_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2028]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
        useful_life=np.array([5, 5]),
    )

    nanas_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
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


def test_tangible_depreciation_with_tax_and_inflation():
    """A unit testing for total depreciation rate of Tangible object"""

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
        54,
        81.57,
        96.4668,
        76.91478,
        71.6610252,
        35.8661376,
        18.0025563,
        7.3137519,
    ]

    undepreCase1 = 3.798457031
    undepreCase2 = 3.7285794

    # Calculated results
    tangible_mangga = CapitalCost(
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

    depreCase1_calc, undepreCase1_calc = tangible_mangga.total_depreciation_rate(
        inflation_rate=0.05
    )

    depreCase2_calc, undepreCase2_calc = tangible_mangga.total_depreciation_rate(
        tax_type=TaxType.VAT,
        vat_rate=np.array([0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]),
        inflation_rate=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
    )

    # Execute testing
    np.testing.assert_allclose(depreCase1, depreCase1_calc)
    np.testing.assert_allclose(undepreCase1, undepreCase1_calc)
    np.testing.assert_allclose(depreCase2, depreCase2_calc)
    np.testing.assert_allclose(undepreCase2, undepreCase2_calc)


def test_tangible_book_value():
    """A unit testing for total depreciation book value of Tangible object"""

    # Expected results: depreciation book value
    bookSL = [0, 0, 80, 100, 70, 40, 10, 0]
    bookDB = [0, 0, 80, 104, 83.2, 66.56, 20.48, 0]
    bookDDB = [0, 0, 60, 36, 21.6, 42.96, 18, 10.8]
    bookPSC = [0, 0, 50, 25, 12.5, 6.25, 25, 12.5]

    # Calculated results: depreciation book value
    mangga_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        useful_life=np.array([5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    apel_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2028]),
        useful_life=np.array([5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    nanas_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2029]),
        useful_life=np.array([5, 5]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
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
        depr_method=DeprMethod.PSC_DB
    )

    # Execute testing
    np.testing.assert_allclose(bookSL, calc_bookSL)
    np.testing.assert_allclose(bookDB, calc_bookDB)
    np.testing.assert_allclose(bookDDB, calc_bookDDB)
    np.testing.assert_allclose(bookPSC, calc_bookPSC)
