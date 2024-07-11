"""
A collection of unit testing for CostOfSales class.
"""

import pytest
import numpy as np
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.costs import CostOfSales, CostOfSalesException


def test_cos_incorrect_start_year_end_year():
    """ A unit testing for incorrect data input: start year is after end year """

    with pytest.raises(CostOfSalesException):
        CostOfSales(
            start_year=2031,
            end_year=2030,
        )


def test_cos_incorrect_expense_year():
    """
    A unit testing for incorrect expense year data input:
    expense_year data is not given as a numpy.ndarray datatype.
    """
    with pytest.raises(CostOfSalesException):
        CostOfSales(
            start_year=2023,
            end_year=2030,
            expense_year=[2023, 2024, 2029],
        )


def test_cos_incorrect_cost():
    """
    A unit testing for incorrect cost data input:
    cost data is not given as a numpy.ndarray datatype.
    """
    with pytest.raises(CostOfSalesException):
        CostOfSales(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2026]),
            cost=[50, 50, 50, 50],
        )


def test_cos_incorrect_cost_allocation():
    """
    A unit testing for incorrect cost_allocation data input:
    cost_allocation is not given as a list datatype.
    """
    with pytest.raises(CostOfSalesException):
        CostOfSales(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2029]),
            cost=np.array([150, 150, 100]),
            cost_allocation=FluidType.GAS,
        )


def test_cos_incorrect_unequal_length_array():
    """ A unit testing for unequal length of arrays """

    with pytest.raises(CostOfSalesException):
        CostOfSales(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2029]),
            cost=np.array([50, 50, 50, 50]),
            cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS],
        )


def test_cos_incorrect_year_input():
    """
    A unit testing for incorrect expense year data input:
    (1) Expense year is before the start year of the project,
    (2) Expense year is after the end year of the project
    """
    with pytest.raises(CostOfSalesException):
        CostOfSales(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2019, 2024, 2025]),
        )

    with pytest.raises(CostOfSalesException):
        CostOfSales(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2024, 2025, 2032]),
        )


def test_cos_prepare_expense_year():
    """
    Test the preparation of `expense_year` attribute in the `CostOfSales` class.

    This function tests whether the `expense_year` attribute is correctly initialized
    in the `CostOfSales` class. Two test cases are checked:

    1. When `expense_year` is not provided, it should be initialized to the range
       from `start_year` to `end_year`.
    2. When `expense_year` is provided, it should be set correctly in the instance.

    Expected Results
    ----------------
    expense_year_exp1 : np.ndarray
        The expected `expense_year` array when it is not provided.
    expense_year_exp2 : np.ndarray
        The expected `expense_year` array when it is provided.

    Calculated Results
    ------------------
    cos1 : CostOfSales
        An instance of `CostOfSales` with default `expense_year`.
    cos2 : CostOfSales
        An instance of `CostOfSales` with a specified `expense_year`.
    expense_year_calc1 : np.ndarray
        The calculated `expense_year` array for `cos1`.
    expense_year_calc2 : np.ndarray
        The calculated `expense_year` array for `cos2`.

    Assertions
    ----------
    np.testing.assert_allclose(expense_year_exp1, expense_year_calc1)
        Check if the expected `expense_year` matches the calculated value for `cos1`.
    np.testing.assert_allclose(expense_year_exp2, expense_year_calc2)
        Check if the expected `expense_year` matches the calculated value for `cos2`.
    """

    # Expected results
    expense_year_exp1 = np.array([2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])
    expense_year_exp2 = np.array([2023, 2024, 2029])

    # Calculated results
    cos1 = CostOfSales(
        start_year=2023,
        end_year=2030,
    )

    cos2 = CostOfSales(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2023, 2024, 2029]),
        cost=np.array([150, 150, 100]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    expense_year_calc1 = cos1.expense_year
    expense_year_calc2 = cos2.expense_year

    # Check whether expected == calculated
    np.testing.assert_allclose(expense_year_exp1, expense_year_calc1)
    np.testing.assert_allclose(expense_year_exp2, expense_year_calc2)


def test_cos_prepare_cost():
    """
    Test the preparation of the `cost` attribute in the `CostOfSales` class.

    This function tests whether the `cost` attribute is correctly initialized in the
    `CostOfSales` class. Two test cases are checked:

    1. When `cost` is not provided, it should be initialized to an array of zeros
       with the same length as the project duration.
    2. When `cost` is provided, it should be set correctly in the instance.

    Expected Results
    ----------------
    cost_exp1 : np.ndarray
        The expected `cost` array when it is not provided.
    cost_exp2 : np.ndarray
        The expected `cost` array when it is provided.

    Calculated Results
    ------------------
    cos1 : CostOfSales
        An instance of `CostOfSales` with default `cost`.
    cos2 : CostOfSales
        An instance of `CostOfSales` with a specified `cost`.
    cost_calc1 : np.ndarray
        The calculated `cost` array for `cos1`.
    cost_calc2 : np.ndarray
        The calculated `cost` array for `cos2`.

    Assertions
    ----------
    np.testing.assert_allclose(cost_exp1, cost_calc1)
        Check if the expected `cost` matches the calculated value for `cos1`.
    np.testing.assert_allclose(cost_exp2, cost_calc2)
        Check if the expected `cost` matches the calculated value for `cos2`.
    """

    # Expected results
    cost_exp1 = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    cost_exp2 = np.array([150, 150, 100])

    # Calculated results
    cos1 = CostOfSales(
        start_year=2023,
        end_year=2030,
    )

    cos2 = CostOfSales(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2024, 2025, 2029]),
        cost=np.array([150, 150, 100]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    cost_calc1 = cos1.cost
    cost_calc2 = cos2.cost

    # Check whether expected == calculated
    np.testing.assert_allclose(cost_exp1, cost_calc1)
    np.testing.assert_allclose(cost_exp2, cost_calc2)


def test_cos_prepare_cost_allocation():
    """
    Test the preparation of the `cost_allocation` attribute in the `CostOfSales` class.

    This function tests whether the `cost_allocation` attribute is correctly initialized
    in the `CostOfSales` class. Two test cases are checked:

    1. When `cost_allocation` is not provided, it should be initialized to a list of
       `FluidType.OIL` with the same length as the project duration.
    2. When `cost_allocation` is provided, it should be set correctly in the instance.

    Expected Results
    ----------------
    cost_allocation_exp1 : list of FluidType
        The expected `cost_allocation` list when it is not provided.
    cost_allocation_exp2 : list of FluidType
        The expected `cost_allocation` list when it is provided.

    Calculated Results
    ------------------
    cos1 : CostOfSales
        An instance of `CostOfSales` with default `cost_allocation`.
    cos2 : CostOfSales
        An instance of `CostOfSales` with a specified `cost_allocation`.
    cost_allocation_calc1 : list of FluidType
        The calculated `cost_allocation` list for `cos1`.
    cost_allocation_calc2 : list of FluidType
        The calculated `cost_allocation` list for `cos2`.

    Assertions
    ----------
    assert cost_allocation_exp1 == cost_allocation_calc1
        Check if the expected `cost_allocation` matches the calculated value for `cos1`.
    assert cost_allocation_exp2 == cost_allocation_calc2
        Check if the expected `cost_allocation` matches the calculated value for `cos2`.
    """
    # Expected results
    cost_allocation_exp1 = [
        FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
        FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
    ]

    cost_allocation_exp2 = [
        FluidType.OIL, FluidType.OIL, FluidType.GAS, FluidType.GAS,
    ]

    # Calculated results
    cos1 = CostOfSales(
        start_year=2023,
        end_year=2030,
    )

    cos2 = CostOfSales(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2023, 2024, 2025, 2026]),
        cost=np.array([50, 50, 50, 50]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.GAS, FluidType.GAS]
    )

    cost_allocation_calc1 = cos1.cost_allocation
    cost_allocation_calc2 = cos2.cost_allocation

    # Check whether expected == calculated
    assert cost_allocation_exp1 == cost_allocation_calc1
    assert cost_allocation_exp2 == cost_allocation_calc2
