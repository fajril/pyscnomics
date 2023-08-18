"""
A unit testing for depreciation.
"""

import numpy as np
from pyscnomics.econ.depreciation import (
    straight_line_depreciation_rate,
    declining_balance_depreciation_rate,
    straight_line_book_value,
    declining_balance_book_value,
)


def test_straight_line_depreciation():
    """
    Test for straight line depreciation.
    """

    # Specify the expected results
    depreciation_charge1 = [35.0, 35.0, 35.0, 35.0, 35.0]
    depreciation_charge2 = [35.0, 35.0, 35.0, 35.0, 35.0, 0, 0, 0, 0, 0]
    depreciation_charge3 = [33.33333333, 33.33333333, 33.33333333]

    book_value1 = [165.0, 130.0, 95.0, 60.0, 25.0]
    book_value2 = [165.0, 130.0, 95.0, 60.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0]
    book_value3 = [66.66666666, 33.33333333, 0.0]

    # Execute calculations
    calc_depreciation_charge1 = straight_line_depreciation_rate(
        cost=200, salvage_value=25, useful_life=5
    )
    calc_depreciation_charge2 = straight_line_depreciation_rate(
        cost=200, salvage_value=25, useful_life=5, depreciation_len=10
    )
    calc_depreciation_charge3 = straight_line_depreciation_rate(
        cost=100, salvage_value=0, useful_life=3
    )
    calc_book_value1 = straight_line_book_value(
        cost=200, salvage_value=25, useful_life=5
    )
    calc_book_value2 = straight_line_book_value(
        cost=200, salvage_value=25, useful_life=5, depreciation_len=10
    )
    calc_book_value3 = straight_line_book_value(
        cost=100, salvage_value=0, useful_life=3
    )

    # Execute testing (expected = calculated)
    np.testing.assert_allclose(depreciation_charge1, calc_depreciation_charge1)
    np.testing.assert_allclose(depreciation_charge2, calc_depreciation_charge2)
    np.testing.assert_allclose(depreciation_charge3, calc_depreciation_charge3)
    np.testing.assert_allclose(book_value1, calc_book_value1)
    np.testing.assert_allclose(book_value2, calc_book_value2)
    np.testing.assert_allclose(book_value3, calc_book_value3)


def test_decline_balance_depreciation():
    """
    Test for declining balance depreciation.
    """

    item_cost = 200_000
    salvage_value = 25_000
    useful_life = 5

    depreciation_charge = [80000.0, 48000.0, 28800.0, 17280.0, 920.0]
    book_value = [120000.0, 72000.0, 43200.0, 25920.0, 25000.0]

    depreciation_charge_calc = declining_balance_depreciation_rate(
        cost=item_cost,
        salvage_value=salvage_value,
        useful_life=useful_life,
        decline_factor=2,
    )

    book_value_calc = declining_balance_book_value(
        cost=item_cost,
        salvage_value=salvage_value,
        useful_life=useful_life,
        decline_factor=2,
    )

    np.testing.assert_array_almost_equal(depreciation_charge, depreciation_charge_calc)
    np.testing.assert_array_almost_equal(book_value, book_value_calc)
