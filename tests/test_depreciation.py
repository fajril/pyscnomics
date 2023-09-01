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
    psc_declining_balance_book_value
)


def test_straight_line_depreciation():
    """A unit testing for straight line depreciation"""

    # Expected results
    depreciationCharge1 = [35.0, 35.0, 35.0, 35.0, 35.0]
    depreciationCharge2 = [35.0, 35.0, 35.0, 35.0, 35.0, 0, 0, 0, 0, 0]
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
    """A unit testing for declining balance depreciation"""

    # Expected results
    depre1 = [20, 16, 12.8, 10.24, 40.96, 0, 0, 0]
    book1 = [80, 64, 51.2, 40.96, 0, 0, 0, 0]

    # Calculated results
    calc_depre1 = declining_balance_depreciation_rate(
        cost=100, salvage_value=0, useful_life=5, decline_factor=1, depreciation_len=8
    )
    calc_book1 = declining_balance_book_value(
        cost=100, salvage_value=0, useful_life=5, decline_factor=1, depreciation_len=8
    )

    # Execute testing
    np.testing.assert_allclose(depre1, calc_depre1)
    np.testing.assert_allclose(book1, calc_book1)


def test_double_decline_balance_depreciation():
    """A unit testing for double declining balance depreciation"""

    # Expected results
    depreciationCharge1 = [40.0, 24.0, 14.4, 1.6, 0]
    depreciationCharge2 = [40.0, 24.0, 14.4, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    bookValue1 = [60.0, 36.0, 21.6, 20.0, 20.0]
    bookValue2 = [60.0, 36.0, 21.6, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]

    # Calculated results
    calcDepreciationCharge1 = declining_balance_depreciation_rate(
        cost=100, salvage_value=20, useful_life=5, decline_factor=2
    )

    calcDepreciationCharge2 = declining_balance_depreciation_rate(
        cost=100, salvage_value=20, useful_life=5, decline_factor=2, depreciation_len=10
    )

    calcBookValue1 = declining_balance_book_value(
        cost=100, salvage_value=20, useful_life=5, decline_factor=2
    )

    calcBookValue2 = declining_balance_book_value(
        cost=100, salvage_value=20, useful_life=5, decline_factor=2, depreciation_len=10
    )

    # Execute testing
    np.testing.assert_allclose(depreciationCharge1, calcDepreciationCharge1)
    np.testing.assert_allclose(depreciationCharge2, calcDepreciationCharge2)
    np.testing.assert_allclose(bookValue1, calcBookValue1)
    np.testing.assert_allclose(bookValue2, calcBookValue2)


def test_psc_decline_balance_depreciation():
    """ A unit testing for psc declining balance depreciation """

    # Expected results
    depre1 = [50, 25, 12.5, 6.25, 6.25, 0, 0, 0]
    depre2 = [50, 25, 12.5, 6.25, 3.125, 1.5625, 0.78125, 0.78125, 0, 0]
    book1 = [50, 25, 12.5, 6.25, 0, 0, 0, 0]
    book2 = [50, 25, 12.5, 6.25, 3.125, 1.5625, 0.78125, 0, 0, 0]

    # Calculated results
    calc_depre1 = psc_declining_balance_depreciation_rate(
        cost=100,
        useful_life=5,
        depreciation_factor=0.5,
        depreciation_len=8
    )

    calc_depre2 = psc_declining_balance_depreciation_rate(
        cost=100,
        useful_life=8,
        depreciation_factor=0.5,
        depreciation_len=10
    )

    calc_book1 = psc_declining_balance_book_value(
        cost=100,
        useful_life=5,
        depreciation_factor=0.5,
        depreciation_len=8
    )

    calc_book2 = psc_declining_balance_book_value(
        cost=100,
        useful_life=8,
        depreciation_factor=0.5,
        depreciation_len=10
    )

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(depre1, calc_depre1)
    np.testing.assert_allclose(depre2, calc_depre2)
    np.testing.assert_allclose(book1, calc_book1)
    np.testing.assert_allclose(book2, calc_book2)

