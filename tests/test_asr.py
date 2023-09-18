"""
A collection of testing units for ASR class.
"""

import pytest
import numpy as np
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.costs import ASR, ASRException


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
    """A unit testing for incorrect data input: incorrect expense year"""

    with pytest.raises(ASRException):
        ASR(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2029, 2031]),
            cost_allocation=FluidType.OIL,
            rate=0.02,
        )

    with pytest.raises(ASRException):
        ASR(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2020, 2025]),
            cost_allocation=FluidType.OIL,
            rate=0.02,
        )


def test_asr_expenditures():
    """A unit testing for expenditures method in ASR class"""

    # Expected result
    expense = [
        0,
        0,
        18.40134672,
        29.22566832,
        29.22566832,
        29.22566832,
        29.22566832,
        29.22566832,
    ]

    # Calculated result
    mangga_asr = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=FluidType.OIL,
        rate=0.02,
    )

    calc_expense = mangga_asr.expenditures()

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(expense, calc_expense)


def test_comparison():
    """A unit testing for comparison in ASR class"""

    mangga_asr = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=FluidType.OIL,
        rate=0.02,
    )

    apel_asr = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=FluidType.OIL,
        rate=0.02,
    )

    nanas_asr = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([25, 25]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=FluidType.OIL,
        rate=0.05,
    )

    assert mangga_asr == apel_asr
    assert apel_asr != nanas_asr
    assert nanas_asr <= mangga_asr
    assert apel_asr >= nanas_asr


def test_arithmetics():
    """A unit testing for arithmetics operations"""

    # Expected result
    add = [
        0,
        0,
        36.80269344,
        58.45133664,
        58.45133664,
        58.45133664,
        58.45133664,
        58.45133664,
    ]
    sub = [0, 0, 0, 0, 0, 0, 0, 0]
    mul = [1000, 500]
    div = [10, 5]

    # Calculated result
    mangga_asr = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=FluidType.OIL,
        rate=0.02,
    )

    apel_asr = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=FluidType.OIL,
        rate=0.02,
    )

    calc_add = mangga_asr + apel_asr
    calc_sub = mangga_asr - apel_asr
    calc_mul = 10 * mangga_asr
    calc_div = mangga_asr / 10

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(add, calc_add)
    np.testing.assert_allclose(sub, calc_sub)
    np.testing.assert_allclose(mul, calc_mul.cost)
    np.testing.assert_allclose(div, calc_div.cost)
