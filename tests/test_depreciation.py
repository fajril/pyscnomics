import numpy as np
import pyscnomics as psc


def test_straight_line_depreciation():
    """Test for Straight Line Depreciation Method"""
    item_cost = 200_000
    salvage_value = 25_000
    useful_life = 5

    depreciation_charge = [35000.0, 35000.0, 35000.0, 35000.0, 35000.0]
    book_value = [165000.0, 130000.0, 95000.0, 60000.0, 25000.0]

    depreciation_charge_calc = psc.straight_line_depreciation_rate(
        cost=item_cost, salvage_value=salvage_value, useful_life=useful_life
    )

    book_value_calc = psc.straight_line_book_value(
        cost=item_cost, salvage_value=salvage_value, useful_life=useful_life
    )

    np.testing.assert_array_almost_equal(depreciation_charge, depreciation_charge_calc)
    np.testing.assert_array_almost_equal(book_value, book_value_calc)


def test_decline_balance_depreciation():
    """Test for Decline balance Depreciation.
    In this test we use decline factor = 2
    """

    item_cost = 200_000
    salvage_value = 25_000
    useful_life = 5

    depreciation_charge = [80000.0, 48000.0, 28800.0, 17280.0, 920.0]
    book_value = [120000.0, 72000.0, 43200.0, 25920.0, 25000.0]

    depreciation_charge_calc = psc.declining_balance_depreciation_rate(
        cost=item_cost,
        salvage_value=salvage_value,
        useful_life=useful_life,
        decline_factor=2,
    )

    book_value_calc = psc.declining_balance_book_value(
        cost=item_cost,
        salvage_value=salvage_value,
        useful_life=useful_life,
        decline_factor=2,
    )

    np.testing.assert_array_almost_equal(depreciation_charge, depreciation_charge_calc)
    np.testing.assert_array_almost_equal(book_value, book_value_calc)
