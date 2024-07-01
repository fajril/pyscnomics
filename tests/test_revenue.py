"""
A unit testing for revenue module.
"""

import numpy as np
import pytest

from pyscnomics.econ.revenue import LiftingException, Lifting
from pyscnomics.econ.costs import FluidType


def test_lifting_incorrect_data_input():
    """
    Initial check on inappropriate data input.

    The test should raise a ValueError when the project duration is
    less than the length of lifting data.
    """

    with pytest.raises(LiftingException):
        Lifting(
            start_year=2023,
            end_year=2027,
            lifting_rate=np.array([50, 50, 50, 50, 50, 50, 50, 50]),
            price=np.array([10, 10, 10, 10, 10, 10, 10, 10]),
            fluid_type=FluidType.OIL,
            prod_year=np.array([2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]),
        )

    with pytest.raises(LiftingException):
        Lifting(
            start_year=2023,
            end_year=2030,
            lifting_rate=np.array([50, 50, 50, 50, 50, 50, 50, 50]),
            price=np.array([10, 10, 10, 10, 10, 10, 10, 10]),
            fluid_type=FluidType.OIL,
            prod_year=np.array([2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028]),
        )

    with pytest.raises(LiftingException):
        Lifting(
            start_year=2023,
            end_year=2030,
            lifting_rate=np.array([50, 50, 50, 50, 50, 50, 50, 50]),
            price=np.array([10, 10, 10, 10, 10, 10, 10, 10]),
            fluid_type=FluidType.OIL,
            prod_year=np.array([2023, 2024, 2025, 2026, 2027, 2028, 2029, 2031]),
        )


def test_lifting_comparison():
    """
    A unit test for comparison involving an instance of Lifting.
    """

    mangga_lifting = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2025, 2026, 2027]),
        fluid_type=FluidType.OIL,
        prod_rate=np.array([150, 150, 150]),
        prod_rate_baseline=np.array([50, 50, 50]),
    )

    jeruk_lifting = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2025, 2026, 2027]),
        fluid_type=FluidType.OIL,
        prod_rate=np.array([150, 150, 150]),
        prod_rate_baseline=np.array([50, 50, 50]),
    )

    apel_lifting = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2025, 2026, 2027]),
        fluid_type=FluidType.GAS,
        prod_rate=np.array([150, 150, 150]),
        prod_rate_baseline=np.array([50, 50, 50]),
    )

    nanas_lifting = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([50, 50, 50]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2025, 2026, 2027]),
        fluid_type=FluidType.OIL,
    )

    # Execute testing conditions
    assert mangga_lifting == jeruk_lifting
    assert mangga_lifting != apel_lifting
    assert nanas_lifting < mangga_lifting
    assert nanas_lifting <= jeruk_lifting
    assert mangga_lifting > nanas_lifting
    assert jeruk_lifting >= nanas_lifting


def test_lifting_arithmetics():
    """
    Test arithmetic operations involving an instance of Lifting.
    """

    # Specify the expected result
    add1 = np.array([0, 0, 0, 0, 1000, 1000, 1000, 0])
    add2 = np.array([500, 500, 0, 0, 1000, 1000, 1500, 500])

    mul1 = np.array([0, 0, 0, 0, 2000, 2000, 2000, 0])
    mul2 = np.array([0.0, 0.0, 0.0, 0.0, -1000.0, -1000.0, -1000.0, 0.0])

    div1 = np.array([250, 250, 0, 0, 0, 0, 250, 250])
    div2 = np.array([-50, -50, 0, 0, 0, 0, -50, -50])

    # Create an instance for OIL revenue
    mangga_lifting = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2027, 2028, 2029]),
        fluid_type=FluidType.OIL,
        prod_rate=np.array([200, 200, 200]),
    )

    nanas_lifting = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([50, 50, 50, 50]),
        price=np.array([10, 10, 10, 10]),
        prod_year=np.array([2029, 2030, 2024, 2023]),
        fluid_type=FluidType.OIL,
        prod_rate=np.array([200, 200, 200, 200]),
    )

    total_lifting = mangga_lifting + nanas_lifting

    calc_add1 = mangga_lifting.revenue()
    calc_add2 = total_lifting.revenue()
    calc_mul1 = mangga_lifting * 2
    calc_mul2 = -1 * mangga_lifting
    calc_div1 = nanas_lifting / 2
    calc_div2 = nanas_lifting / -10

    # Test whether the instance is equal to the expected result
    np.testing.assert_allclose(add1, calc_add1)
    np.testing.assert_allclose(add2, calc_add2)

    np.testing.assert_allclose(mul1, calc_mul1.revenue())
    np.testing.assert_allclose(mul2, calc_mul2.revenue())

    np.testing.assert_allclose(div1, calc_div1.revenue())
    np.testing.assert_allclose(div2, calc_div2.revenue())


def test_lifting_lifting_rate():

    # Raise an exception for an inappropriate data
    with pytest.raises(LiftingException):
        Lifting(
            start_year=2023,
            end_year=2030,
            lifting_rate=100,
            price=np.array([10, 10, 10]),
            prod_year=np.array([2024, 2025, 2026]),
            fluid_type=FluidType.OIL,
            ghv=np.array([1, 1, 1]),
            prod_rate=np.array([200, 200, 200]),
            prod_rate_baseline=np.array([50, 50, 50]),
        )

    # Correct result
    lifting_rate_mangga = np.array([100., 100., 100.])

    lifting_mangga = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2024, 2025, 2026]),
        fluid_type=FluidType.OIL,
        ghv=np.array([1, 1, 1]),
        prod_rate=np.array([200, 200, 200]),
        prod_rate_baseline=np.array([50, 50, 50]),
    )

    lifting_rate_mangga_calc = lifting_mangga.lifting_rate

    # Test whether expected == calculated
    np.testing.assert_allclose(lifting_rate_mangga, lifting_rate_mangga_calc)


