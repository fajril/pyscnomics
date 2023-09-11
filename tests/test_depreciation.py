"""
A unit testing for depreciation module.
"""

import numpy as np
from pyscnomics.econ.depreciation import (
    straight_line_depreciation_rate,
    declining_balance_depreciation_rate,
    psc_declining_balance_depreciation_rate,
    straight_line_book_value,
    declining_balance_book_value,
    psc_declining_balance_book_value,
)


def test_straight_line_depreciation():
    """A unit testing for straight line depreciation"""

    # Expected results
    depreciationCharge1 = [35.0, 35.0, 35.0, 35.0, 35.0]
    depreciationCharge2 = [35.0, 35.0, 35.0, 35.0, 35.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    depreciationCharge3 = [33.33333333, 33.33333333, 33.33333333]

    bookValue1 = [165.0, 130.0, 95.0, 60.0, 25.0]
    bookValue2 = [165.0, 130.0, 95.0, 60.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0]
    bookValue3 = [66.66666666, 33.33333333, 0.0]

    # Calculated results
    calcDepreciationCharge1 = straight_line_depreciation_rate(
        cost=200, salvage_value=25, useful_life=5
    )

    calcDepreciationCharge2 = straight_line_depreciation_rate(
        cost=200, salvage_value=25, useful_life=5, depreciation_len=10
    )

    calcDepreciationCharge3 = straight_line_depreciation_rate(
        cost=100, salvage_value=0, useful_life=3
    )

    calcBookValue1 = straight_line_book_value(cost=200, salvage_value=25, useful_life=5)

    calcBookValue2 = straight_line_book_value(
        cost=200, salvage_value=25, useful_life=5, depreciation_len=10
    )

    calcBookValue3 = straight_line_book_value(cost=100, salvage_value=0, useful_life=3)

    # Execute testing (expected = calculated)
    np.testing.assert_allclose(depreciationCharge1, calcDepreciationCharge1)
    np.testing.assert_allclose(depreciationCharge2, calcDepreciationCharge2)
    np.testing.assert_allclose(depreciationCharge3, calcDepreciationCharge3)
    np.testing.assert_allclose(bookValue1, calcBookValue1)
    np.testing.assert_allclose(bookValue2, calcBookValue2)
    np.testing.assert_allclose(bookValue3, calcBookValue3)


def test_decline_balance_depreciation():
    """
    A unit testing for declining balance depreciation.

    Notes
    -----
    Case 1: salvage value is 0; useful life is 5; project duration is 10 years.
    Case 2: salvage value is 50; useful life is 5;; project duration is 10 years.

    In both cases, the value of depreciation charge at the t = useful_life has to
    account for the remaining cost, namely depreciation charge at t = useful life
    has to be equal to (cost - salvage_value - sum(depreciation_charge)).

    As a result, in the final array of depreciation charge, the following condition has to apply:
    (cost - salvage_value - sum(depreciation_charge)) = 0.
    """

    cost = 100.0
    salvage1 = 0.0
    salvage2 = 50.0
    tolerance = 1e-12

    # Expected results
    depre1 = [20, 16, 12.8, 10.24, 40.96, 0, 0, 0, 0, 0]
    book1 = [80, 64, 51.2, 40.96, 0, 0, 0, 0, 0, 0]

    depre2 = [20.0, 16.0, 12.8, 1.2, 0, 0, 0, 0, 0, 0]
    book2 = [80.0, 64.0, 51.2, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]

    # Calculated results (expected == calculated)
    calc_depre1 = declining_balance_depreciation_rate(
        cost=cost,
        salvage_value=salvage1,
        useful_life=5,
        decline_factor=1,
        depreciation_len=10,
    )

    calc_book1 = declining_balance_book_value(
        cost=cost,
        salvage_value=salvage1,
        useful_life=5,
        decline_factor=1,
        depreciation_len=10,
    )

    calc_depre2 = declining_balance_depreciation_rate(
        cost=cost,
        salvage_value=salvage2,
        useful_life=5,
        decline_factor=1,
        depreciation_len=10,
    )

    calc_book2 = declining_balance_book_value(
        cost=cost,
        salvage_value=salvage2,
        useful_life=5,
        decline_factor=1,
        depreciation_len=10,
    )

    # Execute testing
    np.testing.assert_allclose(depre1, calc_depre1)
    np.testing.assert_allclose(
        cost - salvage1 - np.sum(calc_depre1, keepdims=True),
        np.array([0.0]),
        atol=tolerance,
    )
    np.testing.assert_allclose(book1, calc_book1)

    np.testing.assert_allclose(depre2, calc_depre2)
    np.testing.assert_allclose(
        cost - salvage2 - np.sum(calc_depre2, keepdims=True),
        np.array([0.0]),
        atol=tolerance,
    )
    np.testing.assert_allclose(book2, calc_book2)


def test_double_decline_balance_depreciation():
    """
    A unit testing for double declining balance depreciation.

    Notes
    -----
    Case 1: salvage value is 0; useful life is 5; project duration is 10 years.
    Case 2: salvage value is 50; useful life is 5;; project duration is 10 years.

    In both cases, the value of depreciation charge at the t = useful_life has to
    account for the remaining cost, namely depreciation charge at t = useful life
    has to be equal to (cost - salvage_value - sum(depreciation_charge)).

    As a result, in the final array of depreciation charge, the following condition has to apply:
    (cost - salvage_value - sum(depreciation_charge)) = 0.
    """

    cost = 100.0
    salvage1 = 0.0
    salvage2 = 50.0
    tolerance = 1e-12

    # Expected results
    depre1 = [40.0, 24.0, 14.4, 8.64, 12.96, 0.0, 0.0, 0.0, 0.0, 0.0]
    book1 = [60.0, 36.0, 21.6, 12.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    depre2 = [40.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    book2 = [60.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]

    # Calculated results
    calc_depre1 = declining_balance_depreciation_rate(
        cost=cost,
        salvage_value=salvage1,
        useful_life=5,
        decline_factor=2,
        depreciation_len=10,
    )

    calc_book1 = declining_balance_book_value(
        cost=cost,
        salvage_value=salvage1,
        useful_life=5,
        decline_factor=2,
        depreciation_len=10,
    )

    calc_depre2 = declining_balance_depreciation_rate(
        cost=cost,
        salvage_value=salvage2,
        useful_life=5,
        decline_factor=2,
        depreciation_len=10,
    )

    calc_book2 = declining_balance_book_value(
        cost=cost,
        salvage_value=salvage2,
        useful_life=5,
        decline_factor=2,
        depreciation_len=10,
    )

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(depre1, calc_depre1)
    np.testing.assert_allclose(
        cost - salvage1 - np.sum(calc_depre1, keepdims=True),
        np.array([0.0]),
        atol=tolerance,
    )
    np.testing.assert_allclose(book1, calc_book1)

    np.testing.assert_allclose(depre2, calc_depre2)
    np.testing.assert_allclose(
        cost - salvage2 - np.sum(calc_depre2, keepdims=True),
        np.array([0.0]),
        atol=tolerance,
    )
    np.testing.assert_allclose(book2, calc_book2)


def test_psc_decline_balance_depreciation():
    """
    A unit testing for psc declining balance depreciation.

    Notes
    -----
    Case 1: useful_life = 5; project_duration = 10
    Case 2: useful_life = 3; project_duration = 10
    """

    cost = 100
    useful_life1 = 5
    useful_life2 = 3
    tolerance = 1e-12

    # Expected results
    depre1 = [50, 25, 12.5, 6.25, 6.25, 0, 0, 0, 0, 0]
    book1 = [50, 25, 12.5, 6.25, 0, 0, 0, 0, 0, 0]

    depre2 = [50, 25, 25, 0, 0, 0, 0, 0, 0, 0]
    book2 = [50, 25, 0, 0, 0, 0, 0, 0, 0, 0]

    # Calculated results
    calc_depre1 = psc_declining_balance_depreciation_rate(
        cost=cost, useful_life=useful_life1, depreciation_factor=0.5, depreciation_len=10
    )

    calc_book1 = psc_declining_balance_book_value(
        cost=cost,
        useful_life=useful_life1,
        depreciation_factor=0.5,
        depreciation_len=10
    )

    calc_depre2 = psc_declining_balance_depreciation_rate(
        cost=cost,
        useful_life=useful_life2,
        depreciation_factor=0.5,
        depreciation_len=10
    )

    calc_book2 = psc_declining_balance_book_value(
        cost=cost,
        useful_life=useful_life2,
        depreciation_factor=0.5,
        depreciation_len=10
    )

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(depre1, calc_depre1)
    np.testing.assert_allclose(cost - np.sum(calc_depre1), np.array([0]), atol=tolerance)
    np.testing.assert_allclose(book1, calc_book1)

    np.testing.assert_allclose(depre2, calc_depre2)
    np.testing.assert_allclose(cost - np.sum(calc_depre2), np.array([0]), atol=tolerance)
    np.testing.assert_allclose(book2, calc_book2)
