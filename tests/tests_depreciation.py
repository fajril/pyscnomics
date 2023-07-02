import numpy as np
import pyscnomics as psc

def test_straight_line_depreciation_rate():
    """ Test for Straight Line Depreciation Method
    """
    item_cost = 200_000
    salvage_value = 25_000
    useful_life = 5

    depreciation_charge = [35000., 35000., 35000., 35000., 35000.]
    book_value = [165000., 130000.,  95000.,  60000.,  25000.]

    depreciation_charge_calc = psc.straight_line_depreciation_rate(
        cost=item_cost,
        salvage_value=salvage_value,
        useful_life=useful_life
    )

    book_value_calc = psc.straight_line_book_value(
        cost=item_cost,
        salvage_value=salvage_value,
        useful_life=useful_life
    )

    np.testing.assert_array_almost_equal(depreciation_charge, depreciation_charge_calc)
    np.testing.assert_array_almost_equal(book_value, book_value_calc)

def test_decline_balance_depreciation():
    """ Test for Decline balance Depreciation.
    In this test we use decline factor = 2
    """

    item_cost = 200_000
    salvage_value = 25_000
    useful_life = 5

    depreciation_charge = []
    book_value = []

    depreciation_charge_calc = psc.decline_balance_depreciation_rate(
        cost=item_cost,
        salvage_value=salvage_value,
        useful_life=useful_life,
    )

    book_value_calc = psc.decline_balance_book_value(
        cost=item_cost,
        salvage_value=salvage_value,
        useful_life=useful_life
    )

    np.testing.assert_array_almost_equal(depreciation_charge, depreciation_charge_calc)
    np.testing.assert_array_almost_equal(book_value, book_value_calc)