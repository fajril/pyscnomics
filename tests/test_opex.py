"""
A collection of unit testings for OPEX class.
"""

import pytest
import numpy as np
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.costs import OPEX, OPEXException, Intangible


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

    with pytest.raises(OPEXException):

        OPEX(
            start_year=2023,
            end_year=2030,
            fixed_cost=np.array([100, 50, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
            prod_rate=np.array([100, 50]),
            cost_per_volume=np.array([0.1, 0.1]),
        )

    with pytest.raises(OPEXException):

        OPEX(
            start_year=2023,
            end_year=2030,
            fixed_cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
            prod_rate=np.array([100, 100]),
        )

    with pytest.raises(OPEXException):

        OPEX(
            start_year=2023,
            end_year=2030,
            fixed_cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
            cost_per_volume=np.array([0.1, 0.1]),
        )

    with pytest.raises(OPEXException):

        OPEX(
            start_year=2023,
            end_year=2030,
            fixed_cost=np.array([100, 50]),
            expense_year=np.array([2028, 2029]),
            cost_allocation=FluidType.OIL,
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
            cost_allocation=FluidType.OIL,
        )


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
        cost_allocation=FluidType.OIL,
    )

    apel_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([100, 50]),
        expense_year=np.array([2028, 2029]),
        cost_allocation=FluidType.OIL,
        prod_rate=np.array([100, 100]),
        cost_per_volume=np.array([0.1, 0.1]),
    )

    calc_expense1 = mangga_opex.expenditures()
    calc_expense2 = apel_opex.expenditures()

    # Execute testing
    np.testing.assert_allclose(expense1, calc_expense1)
    np.testing.assert_allclose(expense2, calc_expense2)


def test_opex_comparison_error():
    """A unit testing for misuse of OPEX: comparing different instances/objects"""

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

    # Execute testing
    check_equality = mangga_opex == mangga_intangible

    assert check_equality is False


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
        cost_per_volume=np.array([0.1, 0.1]),
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

    with pytest.raises(OPEXException):
        assert mangga_opex + 100

    with pytest.raises(OPEXException):
        assert mangga_opex - 50

    with pytest.raises(OPEXException):
        assert mangga_opex * mangga_intangible

    with pytest.raises(OPEXException):
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
        cost_per_volume=np.array([0.1, 0.1, 0.1, 0.1]),
    )

    apel_opex = OPEX(
        start_year=2025,
        end_year=2030,
        fixed_cost=np.array([50, 50, 50, 50]),
        expense_year=np.array([2025, 2026, 2027, 2028]),
        cost_allocation=FluidType.OIL,
        prod_rate=np.array([100, 100, 100, 100]),
        cost_per_volume=np.array([0.1, 0.1, 0.1, 0.1]),
    )

    nanas_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([200, 200]),
        expense_year=np.array([2030, 2029]),
        cost_allocation=FluidType.OIL,
        prod_rate=np.array([100, 100]),
        cost_per_volume=np.array([0.1, 0.1]),
    )

    jeruk_opex = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([50, 50]),
        expense_year=np.array([2024, 2023]),
        cost_allocation=FluidType.OIL,
        prod_rate=np.array([100, 100]),
        cost_per_volume=np.array([0.1, 0.1]),
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
