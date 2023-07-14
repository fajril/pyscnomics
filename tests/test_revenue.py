import numpy as np
import pytest
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import FluidType


def test_oil_revenue_for_exception_value_error():

    """
    Initial check on inappropriate data input.

    The test should raise a ValueError when the project duration is
    less than the length of lifting data.

    Returns
    -------
    ValueError
    """

    with pytest.raises(ValueError):

        # Create an instance of revenue with project duration < the length of lifting data
        oil_rev_calc = Lifting(
            start_year=2023,
            end_year=2027,
            lifting_rate=np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]),
            price=np.array([10 for _ in range(11)]),
            fluid_type=FluidType.OIL
        ).revenue()


def test_oil_revenue_normal_condition():

    """
    Check revenue calculation for OIL under normal condition (project_duration = length of lifting data)

    Returns
    -------
    revenue OIL: np.ndarray
        The revenue of OIL where the number of elements is equal to the length of OIL production data.
    """

    # Specify the expected result
    oil_rev = np.array(
        [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10_000]
    )

    # Create an instance for OIL revenue
    oil_rev_calc = Lifting(
        start_year=2023,
        end_year=2032,
        lifting_rate=np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
        price=np.array([10 for _ in range(10)]),
        fluid_type=FluidType.OIL
    ).revenue()

    # Test whether the instance is equal to the expected result
    np.testing.assert_array_almost_equal(oil_rev, oil_rev_calc)


def test_gas_revenue_normal_condition():

    """
    Check revenue calculation for GAS under normal condition (project_duration = length of lifting data)

    Returns
    -------
    revenue GAS: np.ndarray
        The revenue of GAS where the number of elements is equal to the length of GAS production data.
    """

    # Specify the expected result
    gas_rev = np.array(
        [5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 1000, 500]
    )

    # Create an instance for GAS revenue
    gas_rev_calc = Lifting(
        start_year=2023,
        end_year=2032,
        lifting_rate=np.linspace(100, 10, 10),
        price=np.array([10 for i in range(10)]),
        fluid_type=FluidType.GAS,
        ghv=np.array([5 for _ in range(10)])
    ).revenue()

    # Test whether the instance is equal to the expected result
    np.testing.assert_array_almost_equal(gas_rev, gas_rev_calc)


def test_oil_revenue_project_duration_longer_than_production_data():

    """
    Check revenue calculation for OIL when project duration is longer than the length of
    production data.

    Returns
    -------
    revenue OIL: np.ndarray
        The revenue of OIL in which the remaining elements of revenue associated with
        additional years after the end of the project duration are set to zero.
    """

    # Specify the expected result
    oil_rev = np.array(
        [5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 1000, 500, 0, 0, 0, 0, 0]
    )

    # Create an instance of OIL revenue
    oil_rev_calc = Lifting(
        start_year=2023,
        end_year=2037,
        lifting_rate=np.array([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]),
        price=np.array([5 for _ in range(10)]),
        fluid_type=FluidType.OIL
    ).revenue()

    # Test whether the instance is equal to the expected result
    np.testing.assert_array_almost_equal(oil_rev, oil_rev_calc)


def test_gas_revenue_project_duration_longer_than_production_data():

    """
    Check revenue calculation for GAS when project duration is longer than the length of
    production data.

    Returns
    -------
    revenue GAS: np.ndarray
        The revenue of GAS in which the remaining elements of revenue associated with
        additional years after the end of the project duration are set to zero.
    """

    # Specify the expected result
    gas_rev = np.array(
        [5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 1000, 500, 0, 0, 0, 0, 0, 0]
    )

    # Create an instance of GAS revenue
    gas_rev_calc = Lifting(
        start_year=2023,
        end_year=2038,
        lifting_rate=np.linspace(100, 10, 10),
        price=np.array([10 for i in range(10)]),
        fluid_type=FluidType.GAS,
        ghv=np.array([5 for _ in range(10)])
    ).revenue()

    # Test whether the instance is equal to the expected result
    np.testing.assert_array_almost_equal(gas_rev, gas_rev_calc)


def test_lifting_comparison():

    """
    Test comparison between two instances of lifting.

    Returns
    -------
    equality
    unequality
    less_than
    less_than_equal_to
    greater_than
    greater_than_equal_to
    """

    # Create the first instance of OIL revenue
    mangga_lifting = Lifting(
        start_year=2023,
        end_year=2032,
        lifting_rate=np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10]),
        price=np.array([10 for _ in range(10)]),
        fluid_type=FluidType.OIL
    )

    # Create the second instance of OIL revenue
    jeruk_lifting = Lifting(
        start_year=2023,
        end_year=2037,
        lifting_rate=np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 5]),
        price=np.array([10 for _ in range(10)]),
        fluid_type=FluidType.OIL
    )

    # Create the third instance of OIL revenue
    hiu_lifting = Lifting(
        start_year=2023,
        end_year=2032,
        lifting_rate=np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 5]),
        price=np.array([10 for _ in range(10)]),
        fluid_type=FluidType.OIL
    )

    # Create the fourth instance of OIL revenue
    anggur_lifting = Lifting(
        start_year=2023,
        end_year=2032,
        lifting_rate=np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10]),
        price=np.array([10 for _ in range(10)]),
        fluid_type=FluidType.OIL
    )

    # Execute testing
    assert mangga_lifting == anggur_lifting
    assert mangga_lifting != jeruk_lifting
    assert hiu_lifting < mangga_lifting
    assert hiu_lifting <= jeruk_lifting
    assert mangga_lifting > jeruk_lifting
    assert mangga_lifting >= hiu_lifting


def test_arithmetic():

    total_rev = np.array(
        [50, 90, 70, 50, 30, 10]
    )

    total_rev_mult_by_two = np.array(
        [100, 80, 60, 40, 20]
    )

    total_rev_sub = np.zeros(5)

    # Create the first instance of OIL revenue
    mangga_lifting = Lifting(
        start_year=2024,
        end_year=2028,
        lifting_rate=np.array([50, 40, 30, 20, 10]),
        price=np.ones(5),
        fluid_type=FluidType.OIL
    )

    # Create the second instance of OIL revenue
    jeruk_lifting = Lifting(
        start_year=2023,
        end_year=2027,
        lifting_rate=np.array([50, 40, 30, 20, 10]),
        price=np.ones(5),
        fluid_type=FluidType.OIL
    )

    calc_total_rev = mangga_lifting + jeruk_lifting
    calc_mult_rev_by_two = mangga_lifting * -2
    calc_sub = mangga_lifting - mangga_lifting

    np.testing.assert_array_almost_equal(total_rev, calc_total_rev)
    np.testing.assert_array_almost_equal(total_rev_mult_by_two, calc_mult_rev_by_two)
    np.testing.assert_array_almost_equal(total_rev_sub, calc_sub)




