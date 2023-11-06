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
    case3 = [0, 0, 149.2378749, 146.4220659, 72.51378503, 72.50050412, 0, 0]

    # Calculated result
    asr_mangga = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 50, 50]),
        expense_year=np.array([2025, 2026, 2027, 2028]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    case1_calc = asr_mangga.expenditures()
    case2_calc = asr_mangga.expenditures(tax_type=TaxType.VAT, vat_rate=0.05)
    case3_calc = asr_mangga.expenditures(
        tax_type=TaxType.LBT,
        lbt_rate=np.array([0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]),
        inflation_rate=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
        future_rate=0.05,
    )

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(case1, case1_calc)
    np.testing.assert_allclose(case2, case2_calc)
    np.testing.assert_allclose(case3, case3_calc)


def test_asr_proportion():
    """ A unit testing for proportion method in ASR class """

    # Expected result
    case1 = [0, 0, 18.76937365, 40.85098972, 54.38139172, 72.06819172, 72.06819172, 72.06819172]
    case2 = [0, 0, 20.69323445, 46.25546525, 62.70175345, 85.27509019, 85.27509019, 85.27509019]
    case3 = [0, 0, 23.9156041, 52.2517601, 69.90607776, 93.59426836, 93.59426836, 93.59426836]

    # Calculated result
    asr_mangga = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 50, 50]),
        expense_year=np.array([2025, 2026, 2027, 2028]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    asr_apel = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 50, 50]),
        expense_year=np.array([2025, 2026, 2027, 2028]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
        lbt_discount=0.68,
    )

    case1_calc = asr_mangga.proportion()
    case2_calc = asr_mangga.proportion(tax_type=TaxType.VAT, inflation_rate=0.05)
    case3_calc = asr_apel.proportion(
        tax_type=TaxType.LBT,
        inflation_rate=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
        lbt_rate=np.array([0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]),
        future_rate=0.05,
    )

    # Execute testing
    np.testing.assert_allclose(case1, case1_calc)
    np.testing.assert_allclose(case2, case2_calc)
    np.testing.assert_allclose(case3, case3_calc)


def test_arithmetics():
    """A unit testing for arithmetics operations"""

    # Expected result
    add = [0, 0, 37.53874731, 59.62036337, 59.62036337, 59.62036337, 59.62036337, 59.62036337]
    sub = [0, 0, 0, 0, 0, 0, 0, 0]
    mul = [1000, 500]
    div = [10, 5]

    # Calculated result
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

    add_calc = (mangga_asr + apel_asr).proportion()
    sub_calc = (mangga_asr - apel_asr).proportion()
    mul_calc = 10 * mangga_asr
    div_calc = mangga_asr / 10

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(add, add_calc)
    np.testing.assert_allclose(sub, sub_calc, atol=1E-12)
    np.testing.assert_allclose(mul, mul_calc.cost)
    np.testing.assert_allclose(div, div_calc.cost)
