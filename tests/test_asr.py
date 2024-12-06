"""
A collection of testing units for ASR class.
"""

import pytest
import numpy as np
from pyscnomics.econ.selection import FluidType, TaxType
from pyscnomics.econ.costs import ASR, ASRException


def test_asr_incorrect_year_input():
    """A unit testing for incorrect data input: start year is after end year"""

    with pytest.raises(ASRException):

        ASR(
            start_year=2033,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2029, 2030]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )


def test_asr_unequal_length_of_data_input():
    """A unit testing for incorrect data input: unequal length of data input"""

    with pytest.raises(ASRException):

        ASR(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2029]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )


def test_asr_incorrect_expense_year_input():
    """A unit testing for incorrect data input: incorrect expense year"""

    with pytest.raises(ASRException):
        ASR(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2029, 2031]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )

    with pytest.raises(ASRException):
        ASR(
            start_year=2023,
            end_year=2030,
            cost=np.array([100, 50]),
            expense_year=np.array([2020, 2025]),
            cost_allocation=[FluidType.OIL, FluidType.OIL],
        )


def test_comparison():
    """A unit testing for comparison in ASR class"""

    mangga_asr = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    apel_asr = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 50]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    nanas_asr = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([25, 25]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=[FluidType.OIL, FluidType.OIL],
    )

    assert mangga_asr == apel_asr
    assert apel_asr != nanas_asr
    assert nanas_asr <= mangga_asr
    assert apel_asr >= nanas_asr


def test_asr_expenditures():
    """A unit testing for expenditures method in ASR class"""

    # Expected result
    case1 = [0, 0, 112.6162419, 110.4080803, 54.121608, 53.0604, 0, 0]
    case2 = [0, 0, 118.247054, 115.9284843, 56.8276884, 55.71342, 0, 0]

    # Calculated result
    asr_mangga = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 50, 50]),
        future_rate=np.array([0.02, 0.02, 0.02, 0.02]),
        expense_year=np.array([2025, 2026, 2027, 2028]),
        tax_portion=np.array([1, 1, 1, 1]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    case1_calc = asr_mangga.expenditures_post_tax()
    case2_calc = asr_mangga.expenditures_post_tax(tax_rate=0.05)

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(case1, case1_calc)
    np.testing.assert_allclose(case2, case2_calc)
