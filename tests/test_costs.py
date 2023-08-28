"""
A unit testing for costs module.
"""

import pytest
import numpy as np
from pyscnomics.econ.selection import DeprMethod, FluidType
from pyscnomics.econ.costs import (
    TangibleException,
    IntangibleException,
    OPEXException,
    ASRException,
    Tangible,
    Intangible,
    OPEX,
    ASR,
)


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


def test_tangible_expenditures():
    """A unit testing for expenditures method in Tangible class"""

    # Expected result
    expenditures = [100, 100, 150, 0, 0, 0, 0, 0, 0, 0]

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
    calc_expenditures = total_tangible.expenditures()

    # Execute testing
    np.testing.assert_allclose(expenditures, calc_expenditures)


def test_tangible_comparison():
    """Units testing for comparison operation"""

    # Create first instance of Tangible class
    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2033,
        cost=np.array([200, 200]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25, 25]),
        useful_life=np.array([5, 5]),
        cost_allocation=FluidType.OIL,
    )

    # Create second instance of Tangible class
    jeruk_tangible = Tangible(
        start_year=2023,
        end_year=2033,
        cost=np.array([200, 200]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25, 25]),
        useful_life=np.array([5, 5]),
        cost_allocation=FluidType.OIL,
    )

    # Create third instance of Tangible class
    hiu_tangible = Tangible(
        start_year=2023,
        end_year=2033,
        cost=np.array([100, 100]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25, 25]),
        useful_life=np.array([5, 5]),
        cost_allocation=FluidType.GAS,
    )

    # Execute comparison operations
    assert mangga_tangible == jeruk_tangible
    assert mangga_tangible != hiu_tangible
    assert mangga_tangible == 400
    assert mangga_tangible != hiu_tangible
    assert mangga_tangible > hiu_tangible
    assert hiu_tangible <= jeruk_tangible


def test_tangible_arithmetics_incorrect_fluidtype():
    """A unit testing for incorrect addition operation: addition between two different fluid types"""

    with pytest.raises(TangibleException):

        mangga_tangible_oil = Tangible(
            start_year=2023,
            end_year=2032,
            cost=np.array([100, 100]),
            expense_year=np.array([2023, 2024]),
            salvage_value=np.array([25, 25]),
            useful_life=np.array([5, 5]),
            cost_allocation=FluidType.OIL,
        )

        mangga_tangible_gas = Tangible(
            start_year=2023,
            end_year=2032,
            cost=np.array([50, 50]),
            expense_year=np.array([2026, 2027]),
            salvage_value=np.array([25, 25]),
            useful_life=np.array([5, 5]),
            cost_allocation=FluidType.GAS,
        )

        mangga_tangible_oil + mangga_tangible_gas


def test_tangible_arithmetics_misuse():
    """Units testing for incorrect arithmetic operations"""

    with pytest.raises(TangibleException):

        mangga_tangible = Tangible(
            start_year=2023,
            end_year=2032,
            cost=np.array([100, 100]),
            expense_year=np.array([2023, 2024]),
            cost_allocation=FluidType.OIL,
        )

        mangga_tangible + 500
        # mangga_tangible * -2
        # mangga_tangible / 0
        # mangga_tangible / -5


def test_tangible_arithmetics():
    """Units testing for mathematical operations (associated with an instance of Tangible class)"""

    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2032,
        cost=np.array([100, 100]),
        expense_year=np.array([2023, 2024]),
        useful_life=np.array([5, 5]),
        cost_allocation=FluidType.OIL,
    )

    apel_tangible = Tangible(
        start_year=2023,
        end_year=2032,
        cost=np.array([50, 50]),
        expense_year=np.array([2026, 2027]),
        useful_life=np.array([5, 5]),
        cost_allocation=FluidType.OIL,
    )

    jeruk_tangible = Tangible(
        start_year=2023,
        end_year=2032,
        cost=np.array([50]),
        expense_year=np.array([2031]),
        useful_life=np.array([5]),
        cost_allocation=FluidType.OIL,
    )

    # Expected results
    add_tangible1 = np.array([100, 100, 50, 50])
    add_tangible2 = np.array([50, 100, 100])
    mult_tangible1 = np.array([200, 200])
    mult_tangible2 = np.array([500, 500])
    div_tangible1 = np.array([25, 25])
    div_tangible2 = 0.5
    div_tangible3 = 2.0

    # Calculated results
    calc_add_tangible1 = mangga_tangible + apel_tangible
    calc_add_tangible2 = jeruk_tangible + mangga_tangible
    calc_mult_tangible1 = mangga_tangible * 2
    calc_mult_tangible2 = 5 * mangga_tangible
    calc_div_tangible1 = apel_tangible / 2
    calc_div_tangible2 = apel_tangible / mangga_tangible
    calc_div_tangible3 = mangga_tangible / apel_tangible

    # Execute tests
    np.testing.assert_allclose(add_tangible1, calc_add_tangible1.cost)
    np.testing.assert_allclose(add_tangible2, calc_add_tangible2.cost)
    np.testing.assert_allclose(mult_tangible1, calc_mult_tangible1.cost)
    np.testing.assert_allclose(mult_tangible2, calc_mult_tangible2.cost)
    np.testing.assert_allclose(div_tangible1, calc_div_tangible1.cost)
    assert div_tangible2 == calc_div_tangible2
    assert div_tangible3 == calc_div_tangible3


def test_tangible_total_depreciation_rate():
    """A unit testing for total depreciation rate of Tangible object"""

    # Expected results: depreciation_charge
    manggaDepreSL = [0, 0, 20, 30, 30, 30, 30, 10]
    manggaDepreDB = [0, 0, 20, 26, 20.8, 16.64, 13.312, 4.096]
    manggaDepreDoubleDB = [0, 0, 40, 44, 26.4, 15.84, 9.504, 2.592]

    # Calculated results: depreciation_charge
    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=FluidType.OIL,
        useful_life=np.array([5, 5]),
    )

    calcManggaDepreSL = mangga_tangible.total_depreciation_rate(
        depr_method=DeprMethod.SL
    )
    calcManggaDepreDB = mangga_tangible.total_depreciation_rate(
        depr_method=DeprMethod.DB, decline_factor=1
    )
    calcManggaDepreDoubleDB = mangga_tangible.total_depreciation_rate(
        depr_method=DeprMethod.DB, decline_factor=2
    )

    # Execute testing
    np.testing.assert_allclose(manggaDepreSL, calcManggaDepreSL)
    np.testing.assert_allclose(manggaDepreDB, calcManggaDepreDB)
    np.testing.assert_allclose(manggaDepreDoubleDB, calcManggaDepreDoubleDB)


def test_tangible_total_depreciation_book_value():
    """A unit testing for total depreciation book value of Tangible object"""

    # Expected results: depreciation book value
    manggaBookSL = [0, 0, 80, 100, 70, 40, 10, 0]
    manggaBookDB = [0, 0, 80, 104, 83.2, 66.56, 53.248, 49.152]
    manggaBookDoubleDB = [0, 0, 60, 66, 39.6, 23.76, 14.256, 11.664]

    # Calculated results: depreciation book value
    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=FluidType.OIL,
        useful_life=np.array([5, 5]),
    )

    calcManggaBookSL = mangga_tangible.total_depreciation_book_value(
        depr_method=DeprMethod.SL
    )
    calcManggaBookDB = mangga_tangible.total_depreciation_book_value(
        depr_method=DeprMethod.DB, decline_factor=1
    )
    calcManggaBookDoubleDB = mangga_tangible.total_depreciation_book_value(
        depr_method=DeprMethod.DB, decline_factor=2
    )

    # Execute testing
    np.testing.assert_allclose(manggaBookSL, calcManggaBookSL)
    np.testing.assert_allclose(manggaBookDB, calcManggaBookDB)
    np.testing.assert_allclose(manggaBookDoubleDB, calcManggaBookDoubleDB)


def test_intangible_incorrect_year_input():
    """A unit testing for incorrect data input: start year is after end year"""

    with pytest.raises(IntangibleException):

        Intangible(
            start_year=2033,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
        )


def test_intangible_unequal_length_of_data_input():
    """A unit testing for incorrect data input: unequal length of data input"""

    with pytest.raises(IntangibleException):

        Intangible(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2028]),
            cost_allocation=FluidType.OIL,
        )


def test_intangible_incorrect_expense_year_input():
    """A unit testing for incorrect data input: expense year is after end year of the project"""

    with pytest.raises(IntangibleException):

        Intangible(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2035, 2036]),
            cost_allocation=FluidType.OIL,
        )


def test_intangible_expenditures():
    """A unit testing for expenditures method in Intangible class"""

    # Expected result
    expense1 = [0, 0, 0, 0, 0, 100, 50, 0]
    expense2 = [50, 50, 0, 0, 0, 0, 100, 100]

    # Calculated result
    mangga_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=FluidType.OIL,
    )

    apel_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([50, 50, 100, 100]),
        expense_year=np.array([2023, 2024, 2029, 2030]),
        cost_allocation=FluidType.GAS,
    )

    calc_expense1 = mangga_intangible.expenditures()
    calc_expense2 = apel_intangible.expenditures()

    # Execute testing
    np.testing.assert_allclose(expense1, calc_expense1)
    np.testing.assert_allclose(expense2, calc_expense2)


def test_intangible_comparison_error():
    """A unit testing for misuse of Intangible: comparing different instances/objects"""

    with pytest.raises(IntangibleException):

        mangga_tangible = Tangible(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            useful_life=np.array([5, 5]),
            cost_allocation=FluidType.OIL,
        )

        mangga_intangible = Intangible(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
        )

        assert mangga_intangible == mangga_tangible


def test_intangible_comparison():
    """A unit testing for comparison between Intangible instances"""

    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([50, 50]),
        expense_year=np.array([2025, 2029]),
        cost_allocation=FluidType.OIL,
        useful_life=np.array([5, 5]),
    )

    mangga_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=FluidType.OIL,
    )

    apel_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=FluidType.OIL,
    )

    jeruk_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=FluidType.OIL,
    )

    # Execute comparison testing
    assert mangga_intangible == apel_intangible
    assert mangga_intangible != jeruk_intangible
    assert mangga_intangible < jeruk_intangible
    assert mangga_tangible <= mangga_intangible
    assert mangga_intangible >= 100


def test_intangible_arithmetics_incorrect():
    """A unit testing for misuse of arithmetic operations upon an instance of Intangible"""

    with pytest.raises(IntangibleException):

        mangga_tangible = Tangible(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2025, 2029]),
            cost_allocation=FluidType.OIL,
            useful_life=np.array([5, 5]),
        )

        mangga_intangible = Intangible(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
        )

        assert mangga_intangible + 100
        assert mangga_intangible - 50
        assert mangga_tangible - mangga_intangible
        assert mangga_intangible * -2
        assert 0 * mangga_intangible
        assert mangga_intangible * mangga_intangible
        assert mangga_intangible / 0
        assert mangga_intangible / -2


def test_intangible_arithmetics():
    """A unit testing for arithmetic operations upon an instance of Intangible"""

    # Expected results
    add1 = [0, 0, 0, 0, 0, 200, 100, 0]
    add2 = [0, 0, 100, 0, 0, 100, 100, 0]
    sub1 = [0, 0, 0, 0, 0, 0, 0, 0]
    sub2 = [100, 0, 0, 0, 0, -100, -50, 50]
    sub3 = [-100, 0, 0, 0, 0, 100, 50, -50]
    mul1 = [0, 0, 0, 0, 0, 200, 100, 0]
    mul2 = [0, 0, 0, 0, 0, 200, 100, 0]
    div1 = 2
    div2 = [0, 0, 0, 0, 0, 50, 25, 0]

    # Calculated results
    mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2029]),
        cost_allocation=FluidType.OIL,
        useful_life=np.array([5, 5]),
    )

    mangga_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=FluidType.OIL,
    )

    apel_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=FluidType.OIL,
    )

    jeruk_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2023, 2030]),
        cost_allocation=FluidType.GAS,
    )

    nanas_intangible = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([50, 25]),
        expense_year=np.array([2023, 2024]),
        cost_allocation=FluidType.GAS,
    )

    calc_add1 = mangga_intangible + apel_intangible
    calc_add2 = mangga_tangible + mangga_intangible
    calc_sub1 = apel_intangible - mangga_intangible
    calc_sub2 = jeruk_intangible - apel_intangible
    calc_sub3 = apel_intangible - jeruk_intangible
    calc_mul1 = mangga_intangible * 2
    calc_mul2 = 2 * mangga_intangible
    calc_div1 = mangga_intangible / nanas_intangible
    calc_div2 = mangga_intangible / 2

    # Execute testing
    np.testing.assert_allclose(add1, calc_add1)
    np.testing.assert_allclose(add2, calc_add2)
    np.testing.assert_allclose(sub1, calc_sub1)
    np.testing.assert_allclose(sub2, calc_sub2)
    np.testing.assert_allclose(sub3, calc_sub3)
    np.testing.assert_allclose(mul1, calc_mul1.expenditures())
    np.testing.assert_allclose(mul2, calc_mul2.expenditures())
    assert div1 == calc_div1
    np.testing.assert_allclose(div2, calc_div2.expenditures())


def test_opex_incorrect_year_input():
    """A unit testing for incorrect data input: start year is after end year"""

    with pytest.raises(OPEXException):

        OPEX(
            start_year=2033,
            end_year=2030,
            fixed_cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
        )


def test_opex_unequal_length_of_data_input():
    """A unit testing for incorrect data input: unequal length of data input"""

    # Case 1
    with pytest.raises(OPEXException):

        OPEX(
            start_year=2023,
            end_year=2030,
            fixed_cost=np.array([100, 50, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
            prod_rate=np.array([100, 50]),
            cost_per_volume=np.array([0.1, 0.1])
        )

    # Case 2
    with pytest.raises(OPEXException):

        OPEX(
            start_year=2023,
            end_year=2030,
            fixed_cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
            prod_rate=np.array([100, 100])
        )

    # Case 3
    with pytest.raises(OPEXException):

        OPEX(
            start_year=2023,
            end_year=2030,
            fixed_cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
            cost_per_volume=np.array([0.1, 0.1])
        )

    # Case 4
    with pytest.raises(OPEXException):

        OPEX(
            start_year=2023,
            end_year=2030,
            fixed_cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
            prod_rate=np.array([100, 50]),
            cost_per_volume=np.array([0.1, 0.1, 0.1])
        )


def test_opex_incorrect_expense_year_input():
    """A unit testing for incorrect data input: expense year is after the end year of the project"""

    with pytest.raises(OPEXException):

        OPEX(
            start_year=2023,
            end_year=2030,
            fixed_cost=np.array([100, 50]),
            expense_year=np.array([2028, 2033]),
            cost_allocation=FluidType.OIL,
        )


def test_opex_enpenditures():
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
        cost_allocation=FluidType.OIL,
    )

    apel_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=FluidType.OIL,
        prod_rate=np.array([100, 100]),
        cost_per_volume=np.array([0.1, 0.1])
    )

    calc_expense1 = mangga_opex.expenditures()
    calc_expense2 = apel_opex.expenditures()

    # Execute testing
    np.testing.assert_allclose(expense1, calc_expense1)
    np.testing.assert_allclose(expense2, calc_expense2)


def test_opex_comparison_error():
    """A unit testing for misuse of OPEX: comparing different instances/objects"""

    with pytest.raises(OPEXException):

        mangga_opex = OPEX(
            start_year=2023,
            end_year=2030,
            fixed_cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
        )

        mangga_intangible = Intangible(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
        )

        assert mangga_opex == mangga_intangible


def test_opex_comparison():
    """A unit testing for comparison between OPEX instances"""

    mangga_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=FluidType.OIL,
    )

    apel_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=FluidType.OIL,
    )

    jeruk_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([50, 25]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=FluidType.OIL,
    )

    nanas_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=FluidType.OIL,
        prod_rate=np.array([100, 100]),
        cost_per_volume=np.array([0.1, 0.1])
    )

    # Execute comparison testing
    assert mangga_opex == apel_opex
    assert mangga_opex != jeruk_opex
    assert jeruk_opex < apel_opex
    assert mangga_opex <= nanas_opex
    assert nanas_opex > mangga_opex
    assert apel_opex >= jeruk_opex


def test_opex_arithmetics_incorrect():
    """A unit testing for misuse of arithmetic operations upon an instance of OPEX"""

    with pytest.raises(OPEXException):

        mangga_opex = OPEX(
            start_year=2023,
            end_year=2030,
            fixed_cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
        )

        mangga_intangible = Intangible(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
        )

        assert mangga_opex + 100
        assert mangga_opex - 50
        assert mangga_opex - mangga_intangible
        assert mangga_opex * -2
        assert 0 * mangga_opex
        assert mangga_opex * mangga_intangible
        assert mangga_opex / -2
        assert mangga_opex / 0


def test_opex_arithmetics():
    """A unit testing for arithmetic operations upon an instance of OPEX"""

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
        cost_allocation=FluidType.OIL,
        prod_rate=np.array([100, 100, 100, 100]),
        cost_per_volume=np.array([0.1, 0.1, 0.1, 0.1])
    )

    apel_opex = OPEX(
        start_year=2025,
        end_year=2030,
        fixed_cost=np.array([50, 50, 50, 50]),
        expense_year=np.array([2025, 2026, 2027, 2028]),
        cost_allocation=FluidType.OIL,
        prod_rate=np.array([100, 100, 100, 100]),
        cost_per_volume=np.array([0.1, 0.1, 0.1, 0.1])
    )

    nanas_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([200, 200]),
        expense_year=np.array([2030, 2029]),
        cost_allocation=FluidType.OIL,
        prod_rate=np.array([100, 100]),
        cost_per_volume=np.array([0.1, 0.1])
    )

    jeruk_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([50, 50]),
        expense_year=np.array([2024, 2023]),
        cost_allocation=FluidType.OIL,
        prod_rate=np.array([100, 100]),
        cost_per_volume=np.array([0.1, 0.1])
    )

    calc_add1 = mangga_opex + apel_opex
    calc_add2 = nanas_opex + jeruk_opex
    calc_mul1 = nanas_opex * 2
    calc_div1 = jeruk_opex / 2

    # Execute testing
    np.testing.assert_allclose(add1, calc_add1.expenditures())
    np.testing.assert_allclose(add2, calc_add2.expenditures())
    np.testing.assert_allclose(mul1, calc_mul1.expenditures())
    np.testing.assert_allclose(div1, calc_div1.expenditures())


def test_asr_incorrect_year_input():
    """A unit testing for incorrect data input: start year is after end year"""

    with pytest.raises(ASRException):

        ASR(
            start_year=2033,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2029, 2030]),
            cost_allocation=FluidType.OIL,
            rate=0.02,
        )


def test_asr_unequal_length_of_data_input():
    """A unit testing for incorrect data input: unequal length of data input"""

    with pytest.raises(ASRException):

        ASR(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2029]),
            cost_allocation=FluidType.OIL,
            rate=0.02,
        )


def test_asr_incorrect_expense_year_input():
    """A unit testing for incorrect data input: expense year is after end year of the project"""

    with pytest.raises(ASRException):

        ASR(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2029, 2031]),
            cost_allocation=FluidType.OIL,
            rate=0.02,
        )


def test_asr_expenditures():
    """A unit testing for expenditures method in ASR class"""

    # Expected result
    expense1 = [0, 0, 0, 0, 0, 0, 51, 101]

    # Calculated result
    mangga_asr = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2029, 2030]),
        cost_allocation=FluidType.OIL,
        rate=0.02,
    )

    calc_expense1 = mangga_asr.expenditures()

    # Execute testing
    np.testing.assert_allclose(expense1, calc_expense1)
