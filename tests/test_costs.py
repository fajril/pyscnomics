"""
A unit testing for costs module.
"""

import pytest
import numpy as np
from pyscnomics.econ.costs import TangibleException, Tangible, FluidType


def test_tangible_incorrect_year_input():
    """
    Check for incorrect data input: start year is after end year.

    Returns
    -------
    TangibleException
        Error message.
    """

    with pytest.raises(TangibleException):

        Tangible(
            start_year=2030,
            end_year=2023,
            cost=np.array([2_000, 2_000]),
            expense_year=np.array([2023, 2024]),
            salvage_value=np.array([25_000, 25_000]),
            useful_life=np.array([5, 5]),
            cost_allocation=FluidType.OIL,
        )


def test_tangible_unequal_length_of_data_input():
    """
    Check of incorrect data input: unequal length of data input.

    Returns
    -------
    TangibleException
        Error message.
    """

    with pytest.raises(TangibleException):

        Tangible(
            start_year=2023,
            end_year=2032,
            cost=np.array([2_000, 2_000]),
            expense_year=np.array([2023, 2024, 2025]),
            salvage_value=np.array([25_000, 25_000]),
            useful_life=np.array([5, 5]),
            cost_allocation=FluidType.OIL,
        )


def test_tangible_incorrect_expense_year_input():
    """
    Check for incorrect data input: expense year is after end year of the project.

    Returns
    -------
    TangibleException
        Error message.
    """

    with pytest.raises(TangibleException):

        Tangible(
            start_year=2023,
            end_year=2032,
            cost=np.array([2_000, 2_000]),
            expense_year=np.array([2023, 2040]),
            salvage_value=np.array([25_000, 25_000]),
            useful_life=np.array([5, 5]),
            cost_allocation=FluidType.OIL,
        )


def test_tangible_comparison():
    """
    Test comparison operation.

    Returns
    -------
    bool: True or False
    """

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
    """
    Test for incorrect addition operation: addition between two different fluid types.

    Returns
    -------
    TangibleException
        Error message.
    """

    with pytest.raises(TangibleException):

        mangga_tangible = Tangible(
            start_year=2023,
            end_year=2032,
            cost=np.array([100, 100]),
            expense_year=np.array([2023, 2024]),
            salvage_value=np.array([25, 25]),
            useful_life=np.array([5, 5]),
            cost_allocation=FluidType.OIL,
        )

        jeruk_tangible = Tangible(
            start_year=2023,
            end_year=2032,
            cost=np.array([50, 50]),
            expense_year=np.array([2026, 2027]),
            salvage_value=np.array([25, 25]),
            useful_life=np.array([5, 5]),
            cost_allocation=FluidType.GAS,
        )

        mangga_tangible + jeruk_tangible


def test_tangible_arithmetics_misuse():
    """
    Test for incorrect arithmetic operations.

    Returns
    -------
    TangibleException
        Error message.
    """

    with pytest.raises(TangibleException):

        mangga_tangible = Tangible(
            start_year=2023,
            end_year=2032,
            cost=np.array([100, 100]),
            expense_year=np.array([2023, 2024]),
            cost_allocation=FluidType.OIL,
        )

        mangga_tangible + 500
        mangga_tangible * -2
        mangga_tangible / 0
        mangga_tangible / -5


def test_tangible_arithmetics():
    """
    Tests for mathematical operations between two Tangible instances.

    Returns
    -------
    bool: True
    """

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
    div_tangible2 = np.array([0.5, 0.5])
    div_tangible3 = np.array([2, 2])

    # Calculated results
    calc_add_tangible1 = mangga_tangible + apel_tangible
    calc_add_tangible2 = jeruk_tangible + mangga_tangible
    calc_mult_tangible1 = mangga_tangible * 2
    calc_mult_tangible2 = 10 * mangga_tangible
    calc_div_tangible1 = apel_tangible / 2
    calc_div_tangible2 = apel_tangible / mangga_tangible
    calc_div_tangible3 = mangga_tangible / apel_tangible

    # Execute tests
    np.allclose(add_tangible1, calc_add_tangible1.cost)
    np.allclose(add_tangible2, calc_add_tangible2.cost)
    np.allclose(mult_tangible1, calc_mult_tangible1.cost)
    np.allclose(mult_tangible2, calc_mult_tangible2.cost)
    np.allclose(div_tangible1, calc_div_tangible1.cost)
    np.allclose(div_tangible2, calc_div_tangible2)
    np.allclose(div_tangible3, calc_div_tangible3)


def test_tangible_expenditures():
    raise NotImplemented


def test_tangible():

    """Test Tangible class"""
    depreciation_charge = [
        80000.0,
        128000.0,
        76800.0,
        46080.0,
        18200.0,
        920,
        0,
        0,
        0,
        0,
        0,
    ]

    mangga_tangible = psc.Tangible(
        start_year=2023,
        end_year=2033,
        cost=np.array([200_000, 200_000]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25_000, 25_000]),
        useful_life=np.array([5, 5]),
        cost_allocation=[psc.FluidType.OIL, psc.FluidType.OIL],
    )
    depreciation_charge_calc = mangga_tangible.total_depreciation_rate(
        depr_method=psc.DeprMethod.DB
    )
    np.testing.assert_array_almost_equal(depreciation_charge, depreciation_charge_calc)
