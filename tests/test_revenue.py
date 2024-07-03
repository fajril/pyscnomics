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


def test_prepare_lifting_rate():
    """
    Test the preparation of lifting_rate attribute in the Lifting class.

    This function tests two scenarios:
    1.  Ensures that an inappropriate data input raises a `LiftingException`.
    2.  Verifies that the calculated lifting rate matches the expected lifting rate
        for correct input data.

    Raises
    ------
    LiftingException
        If the input data is inappropriate.

    Notes
    -----
    The function performs the following tests:
    1.  Attempts to create a `Lifting` instance with inappropriate data
        and checks if a `LiftingException` is raised.
    2.  Creates a `Lifting` instance with valid data and verifies that the calculated lifting rate
        matches the expected lifting rate using `np.testing.assert_allclose`.
    """
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

    # Expected result
    lifting_rate_mangga = np.array([100., 100., 100.])

    # Calculated result
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


def test_prepare_price():
    """
    Test the preparation of price attribute in the Lifting class.

    This function tests two scenarios:
    1. Ensures that an inappropriate data input raises a `LiftingException`.
    2. Verifies that the calculated price matches the expected price for correct input data.

    Raises
    ------
    LiftingException
        If the input data is inappropriate.

    Notes
    -----
    The function performs the following tests:
    1.  Attempts to create a `Lifting` instance with inappropriate data (non-array price)
        and checks if a `LiftingException` is raised.
    2.  Creates a `Lifting` instance with valid data and verifies that the calculated
        price matches the expected price using `np.testing.assert_allclose`.
    """
    # Raise an exception for an inappropriate data
    with pytest.raises(LiftingException):
        Lifting(
            start_year=2023,
            end_year=2030,
            lifting_rate=np.array([100, 100, 100]),
            price=10,
            prod_year=np.array([2024, 2025, 2026]),
            fluid_type=FluidType.GAS,
            ghv=np.array([1, 1, 1]),
            prod_rate=np.array([200, 200, 200]),
            prod_rate_baseline=np.array([50, 50, 50]),
        )

    # Expected result
    price_apel_expected = np.array([10., 10., 10.])

    # Calculated result
    lifting_apel = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2024, 2025, 2026]),
        fluid_type=FluidType.GAS,
        ghv=np.array([1, 1, 1]),
        prod_rate=np.array([200, 200, 200]),
        prod_rate_baseline=np.array([50, 50, 50]),
    )

    price_apel_calculated = lifting_apel.price

    # Test whether expected == calculated
    np.testing.assert_allclose(price_apel_expected, price_apel_calculated)


def test_prepare_prod_year():
    """
    Test the preparation of attribute prod_year in the Lifting class.

    This function tests two scenarios:
    1.  Ensures that an inappropriate data input raises a `LiftingException`.
    2.  Verifies that the calculated production year matches the expected production year
        for correct input data.

    Raises
    ------
    LiftingException
        If the input data is inappropriate.

    Notes
    -----
    The function performs the following tests:
    1.  Attempts to create a `Lifting` instance with inappropriate data (non-array production year)
        and checks if a `LiftingException` is raised.
    2.  Creates a `Lifting` instance with valid data and verifies that the calculated production year
        matches the expected production year using `np.testing.assert_allclose`.
    """
    # Raise an exception for an inappropriate data
    with pytest.raises(LiftingException):
        Lifting(
            start_year=2023,
            end_year=2030,
            lifting_rate=np.array([100, 100, 100]),
            price=np.array([10, 10, 10]),
            prod_year=[2024, 2025, 2026],
            fluid_type=FluidType.GAS,
            ghv=np.array([1, 1, 1]),
            prod_rate=np.array([200, 200, 200]),
            prod_rate_baseline=np.array([50, 50, 50]),
        )

    # Expected result
    prod_year_apel_expected = np.array([2024, 2025, 2026])

    # Calculated result
    lifting_apel = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2024, 2025, 2026]),
        fluid_type=FluidType.GAS,
        ghv=np.array([1, 1, 1]),
        prod_rate=np.array([200, 200, 200]),
        prod_rate_baseline=np.array([50, 50, 50]),
    )

    prod_year_apel_calculated = lifting_apel.prod_year

    # Test whether expected == calculated
    np.testing.assert_allclose(prod_year_apel_expected, prod_year_apel_calculated)


def test_prepare_ghv():
    """
    Test the preparation of attribute ghv in the Lifting class.

    This function tests two scenarios:
    1. Ensures that an inappropriate data input raises a `LiftingException`.
    2. Verifies that the calculated GHV matches the expected GHV for correct input data.

    Raises
    ------
    LiftingException
        If the input data is inappropriate.

    Notes
    -----
    The function performs the following tests:
    1.  Attempts to create a `Lifting` instance with inappropriate data (non-array GHV)
        and checks if a `LiftingException` is raised.
    2.  Creates two `Lifting` instances with valid data and verifies that the calculated GHV
        matches the expected GHV using `np.testing.assert_allclose`.
    """
    # Raise an exception for an inappropriate data
    with pytest.raises(LiftingException):
        Lifting(
            start_year=2023,
            end_year=2030,
            lifting_rate=np.array([100, 100, 100]),
            price=np.array([10, 10, 10]),
            prod_year=[2024, 2025, 2026],
            fluid_type=FluidType.GAS,
            ghv=1,
            prod_rate=np.array([200, 200, 200]),
            prod_rate_baseline=np.array([50, 50, 50]),
        )

    # Expected results
    ghv_expected1 = np.array([1., 1., 1.])
    ghv_expected2 = np.array([0.5, 0.5, 0.5])

    # Calculated results
    lifting_apel1 = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2024, 2025, 2026]),
        fluid_type=FluidType.GAS,
        prod_rate=np.array([200, 200, 200]),
        prod_rate_baseline=np.array([50, 50, 50]),
    )

    lifting_apel2 = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2024, 2025, 2026]),
        ghv=np.array([0.5, 0.5, 0.5]),
        fluid_type=FluidType.GAS,
        prod_rate=np.array([200, 200, 200]),
        prod_rate_baseline=np.array([50, 50, 50]),
    )

    ghv_calculated1 = lifting_apel1.ghv
    ghv_calculated2 = lifting_apel2.ghv

    # Test whether expected == calculated
    np.testing.assert_allclose(ghv_expected1, ghv_calculated1)
    np.testing.assert_allclose(ghv_expected2, ghv_calculated2)


def test_prepare_prod_rate():
    """
    Test the preparation of attribute prod_rate in the Lifting class.

    This function tests three scenarios:
    1.  Ensures that an inappropriate data input raises a `LiftingException`.
    2.  Verifies that the calculated production rate matches the expected production rate
        for correct input data.

    Raises
    ------
    LiftingException
        If the input data is inappropriate.

    Notes
    -----
    The function performs the following tests:
    1.  Attempts to create a `Lifting` instance with inappropriate data (non-array production rate)
        and checks if a `LiftingException` is raised.
    2.  Attempts to create a `Lifting` instance with mismatched production rate and
        baseline production rate arrays and checks if a `LiftingException` is raised.
    3.  Creates two `Lifting` instances with valid data and verifies that the calculated
        production rate matches the expected production rate using `np.testing.assert_allclose`.
    """
    # Raise an exception for an inappropriate data
    with pytest.raises(LiftingException):
        Lifting(
            start_year=2023,
            end_year=2030,
            lifting_rate=np.array([100, 100, 100]),
            price=np.array([10, 10, 10]),
            prod_year=[2024, 2025, 2026],
            fluid_type=FluidType.GAS,
            ghv=np.array([1, 1, 1]),
            prod_rate=200,
            prod_rate_baseline=np.array([50, 50, 50]),
        )

    # Raise an exception for an inappropriate data
    with pytest.raises(LiftingException):
        Lifting(
            start_year=2023,
            end_year=2030,
            lifting_rate=np.array([100, 100, 100]),
            price=np.array([10, 10, 10]),
            prod_year=[2024, 2025, 2026],
            fluid_type=FluidType.GAS,
            ghv=np.array([1, 1, 1]),
            prod_rate=np.array([50, 50, 50]),
            prod_rate_baseline=np.array([50, 50, 50]),
        )

    # Expected results
    prod_rate_expected1 = np.array([100., 100., 100.])
    prod_rate_expected2 = np.array([150., 150., 150.])

    # Calculated results
    lifting_apel1 = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2024, 2025, 2026]),
        fluid_type=FluidType.GAS,
        prod_rate_baseline=np.array([50, 50, 50]),
    )

    lifting_apel2 = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2024, 2025, 2026]),
        fluid_type=FluidType.GAS,
        prod_rate=np.array([150, 150, 150]),
        prod_rate_baseline=np.array([50, 50, 50]),
    )

    prod_rate_calculated1 = lifting_apel1.prod_rate
    prod_rate_calculated2 = lifting_apel2.prod_rate

    # Test whether expected == calculated
    np.testing.assert_allclose(prod_rate_expected1, prod_rate_calculated1)
    np.testing.assert_allclose(prod_rate_expected2, prod_rate_calculated2)


def test_prepare_prod_rate_baseline():
    """
    Test the preparation of attribute prod_rate_baseline in the Lifting class.

    This function tests two scenarios:
    1.  Ensures that an inappropriate data input raises a `LiftingException`.
    2.  Verifies that the calculated baseline production rate matches the expected baseline
        production rate for correct input data.

    Raises
    ------
    LiftingException
        If the input data is inappropriate.

    Notes
    -----
    The function performs the following tests:
    1.  Attempts to create a `Lifting` instance with inappropriate data
        (non-array baseline production rate) and checks if a `LiftingException` is raised.
    2.  Creates two `Lifting` instances with valid data and verifies that the calculated
        baseline production rate matches the expected baseline production rate using
        `np.testing.assert_allclose`.
    """
    # Raise an exception for an inappropriate data
    with pytest.raises(LiftingException):
        Lifting(
            start_year=2023,
            end_year=2030,
            lifting_rate=np.array([100, 100, 100]),
            price=np.array([10, 10, 10]),
            prod_year=[2024, 2025, 2026],
            fluid_type=FluidType.GAS,
            ghv=np.array([1, 1, 1]),
            prod_rate=np.array([150, 150, 150]),
            prod_rate_baseline=50,
        )

    # Expected results
    prod_rate_baseline_expected1 = np.array([0., 0., 0.])
    prod_rate_baseline_expected2 = np.array([50, 50, 50])

    # Calculated results
    lifting_apel1 = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2024, 2025, 2026]),
        ghv=np.array([1, 1, 1]),
        fluid_type=FluidType.GAS,
        prod_rate=np.array([150, 150, 150]),
    )

    lifting_apel2 = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2024, 2025, 2026]),
        ghv=np.array([1, 1, 1]),
        fluid_type=FluidType.GAS,
        prod_rate=np.array([150, 150, 150]),
        prod_rate_baseline=np.array([50, 50, 50]),
    )

    prod_rate_baseline_calculated1 = lifting_apel1.prod_rate_baseline
    prod_rate_baseline_calculated2 = lifting_apel2.prod_rate_baseline

    # Test whether expected == calculated
    np.testing.assert_allclose(prod_rate_baseline_expected1, prod_rate_baseline_calculated1)
    np.testing.assert_allclose(prod_rate_baseline_expected2, prod_rate_baseline_calculated2)


def test_prepare_prod_rate_total():
    """
    Test the preparation of attribute prod_rate_total in the Lifting class.

    This function verifies that the calculated total production rate matches the
    expected total production rate for given input data.

    Notes
    -----
    The function performs the following test:
    Creates a `Lifting` instance with valid data and verifies that the calculated total
    production rate matches the expected total production rate using `np.testing.assert_allclose`.
    """
    # Expected result
    prod_rate_total_expected = np.array([200., 200., 200.])

    # Calculated result
    lifting_apel = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2024, 2025, 2026]),
        ghv=np.array([1, 1, 1]),
        fluid_type=FluidType.GAS,
        prod_rate=np.array([150, 150, 150]),
        prod_rate_baseline=np.array([50, 50, 50]),
    )

    prod_rate_total_calculated = lifting_apel.prod_rate_total

    # Test whether expected == calculated
    np.testing.assert_allclose(prod_rate_total_expected, prod_rate_total_calculated)


def test_prepare_get_prod_rate_total_arr():
    """
    Test the `get_prod_rate_total_arr` method in the Lifting class.

    This function verifies that the calculated production rate total array matches
    the expected production rate total array for given input data.

    Notes
    -----
    The function performs the following test:
    Creates a `Lifting` instance with valid data and verifies that the calculated production rate
    total array matches the expected production rate total array using `np.testing.assert_allclose`.
    """
    # Expected result
    prod_rate_total_arr_expected = np.array([0., 200., 200., 0., 0., 0., 200., 200.])

    # Calculated result
    lifting_apel = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100, 100]),
        price=np.array([10, 10, 10, 10]),
        prod_year=np.array([2024, 2025, 2029, 2030]),
        ghv=np.array([1, 1, 1, 1]),
        fluid_type=FluidType.GAS,
        prod_rate=np.array([150, 150, 150, 150]),
        prod_rate_baseline=np.array([50, 50, 50, 50]),
    )

    prod_rate_total_arr_calculated = lifting_apel.get_prod_rate_total_arr()

    # Test whether expected == calculated
    np.testing.assert_allclose(prod_rate_total_arr_expected, prod_rate_total_arr_calculated)
