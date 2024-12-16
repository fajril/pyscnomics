"""
This module contains unit tests for the econ_limit function from the pyscnomics.econ module.
The tests focus on different cashflow scenarios to ensure the correct index of maximum NPV 
is returned based on various limit methods.
"""
import numpy as np
import pytest
from pyscnomics.econ import econ_limit
from pyscnomics.econ.selection import LimitMethod


def test_max_npv():
    """Test the MAX_NPV function for correct index of maximum NPV."""
    cashflow_data = np.array([100, 200, 300, 50, -75, -150])
    expected_index = 3  # Assuming npv(cashflow[:4]) is the maximum
    result = econ_limit(cashflow_data, method=LimitMethod.MAX_NPV)
    assert (
        result == expected_index
    ), f"Expected index {expected_index}, but got {result}."


def test_max_npv_empty_cashflow():
    """Test MAX_NPV with an empty cashflow array."""
    empty_cashflow = np.array([])
    with pytest.raises(ValueError):
        econ_limit(empty_cashflow, method=LimitMethod.MAX_NPV)


def test_max_npv_single_cashflow():
    """Test MAX_NPV with a single cashflow entry."""
    single_cashflow = np.array([100])
    result = econ_limit(single_cashflow, method=LimitMethod.MAX_NPV)
    assert result == 0, "Expected index 0 for single cashflow."


def test_max_npv_negative_cashflow():
    """Test MAX_NPV with all negative cashflows."""
    negative_cashflow = np.array([-100, -200, -300])
    result = econ_limit(negative_cashflow, method=LimitMethod.MAX_NPV)
    assert result == 0, "Expected index 0 for all negative cashflows."


def test_max_npv_mixed_cashflow():
    """Test MAX_NPV with mixed cashflows."""
    mixed_cashflow = np.array([100, -50, 200, -100, 100])
    result = econ_limit(mixed_cashflow, method=LimitMethod.MAX_NPV)
    expected_index = 2  # Assuming npv(cashflow[:3]) is the maximum
    assert (
        result == expected_index
    ), f"Expected index {expected_index}, but got {result}."


def test_max_cum_cashflow():
    """Test the MAX_NPV function for correct index of maximum NPV."""
    cashflow_data = np.array([100, 200, 300, 50, -75, -150])
    expected_index = 3  # Assuming npv(cashflow[:4]) is the maximum
    result = econ_limit(cashflow_data, method=LimitMethod.MAX_CUM_CASHFLOW)
    assert (
        result == expected_index
    ), f"Expected index {expected_index}, but got {result}."


def test_max_cum_cf_empty_cashflow():
    """Test MAX_NPV with an empty cashflow array."""
    empty_cashflow = np.array([])
    with pytest.raises(ValueError):
        econ_limit(empty_cashflow, method=LimitMethod.MAX_CUM_CASHFLOW)


def test_max_cum_cf_single_cashflow():
    """Test MAX_NPV with a single cashflow entry."""
    single_cashflow = np.array([100])
    result = econ_limit(single_cashflow, method=LimitMethod.MAX_CUM_CASHFLOW)
    assert result == 0, "Expected index 0 for single cashflow."


def test_max_cum_cf_negative_cashflow():
    """Test MAX_NPV with all negative cashflows."""
    negative_cashflow = np.array([-100, -200, -300])
    result = econ_limit(negative_cashflow, method=LimitMethod.MAX_CUM_CASHFLOW)
    assert result == 0, "Expected index 0 for all negative cashflows."


def test_max_cum_cf_mixed_cashflow():
    """Test MAX_NPV with mixed cashflows."""
    mixed_cashflow = np.array([100, -50, 200, -100, 100])
    result = econ_limit(mixed_cashflow, method=LimitMethod.MAX_CUM_CASHFLOW)
    expected_index = 2  # Assuming npv(cashflow[:3]) is the maximum
    assert (
        result == expected_index
    ), f"Expected index {expected_index}, but got {result}."


def test_negative_cashflow():
    """Test the MAX_NPV function for correct index of maximum NPV."""
    cashflow_data = np.array([100, 200, 300, 50, -75, -150])
    expected_index = 3  # Assuming npv(cashflow[:4]) is the maximum
    result = econ_limit(cashflow_data, method=LimitMethod.NEGATIVE_CASHFLOW)
    assert (
        result == expected_index
    ), f"Expected index {expected_index}, but got {result}."


def test_neg_cf_empty_cashflow():
    """Test MAX_NPV with an empty cashflow array."""
    empty_cashflow = np.array([])
    with pytest.raises(ValueError):
        econ_limit(empty_cashflow, method=LimitMethod.NEGATIVE_CASHFLOW)


def test_neg_cf_single_cashflow():
    """Test MAX_NPV with a single cashflow entry."""
    single_cashflow = np.array([100])
    result = econ_limit(single_cashflow, method=LimitMethod.NEGATIVE_CASHFLOW)
    assert result == 0, "Expected index 0 for single cashflow."


def test_neg_cf_negative_cashflow():
    """Test MAX_NPV with all negative cashflows."""
    negative_cashflow = np.array([-100, -200, -300])
    result = econ_limit(negative_cashflow, method=LimitMethod.NEGATIVE_CASHFLOW)
    assert result == 0, "Expected index 0 for all negative cashflows."


def test_neg_cf_mixed_cashflow():
    """Test MAX_NPV with mixed cashflows."""
    mixed_cashflow = np.array([100, 50, -200, -100, 100])
    result = econ_limit(mixed_cashflow, method=LimitMethod.NEGATIVE_CASHFLOW)
    expected_index = 1
    assert (
        result == expected_index
    ), f"Expected index {expected_index}, but got {result}."

def test_mixed_cf():
    """Test Negative Cashflow method with mixed cashflows."""
    # Defining the cashflow
    cf_0 = np.array([-1000, -200, 30, 133, 1000, 2000, -100, 2222, -10])
    cf_1 = np.array([-150, -100, -200, 1, -20, -10, 100, 40, -40])
    cf_2 = np.array([150, -100, -200, 1, 20, -10, 100, 40, -40])
    cf_3 = np.array([150, 100, 200, 1, 20, 10, 100, 40, 40])
    cf_4 = np.array([-150, -100, -200, -1, -20, -10, -100, -40, -40])
    cf_5 = np.array([150, 100, 200, 1, 20, 10, 100, 40, -40])
    cf_6 = np.array([-10, 10])
    cf_7 = np.array([10, -11])
    cf_8 = np.array([-10, 10, -10])
    cf_9 = np.array([-10, 10, -10, 10])
    cf_10 = np.array([-10, 10, -10, -10])
    cf_11 = np.array([-10])
    cf_12 = np.array([10])

    # Defining the economic limit index
    idx_cf_0 = 5
    idx_cf_1 = 3
    idx_cf_2 = 0
    idx_cf_3 = 8
    idx_cf_4 = 0
    idx_cf_5 = 7
    idx_cf_6 = 1
    idx_cf_7 = 0
    idx_cf_8 = 1
    idx_cf_9 = 1
    idx_cf_10 = 1
    idx_cf_11 = 0
    idx_cf_12 = 0

    list_cf = [cf_0, cf_1, cf_2, cf_3, cf_4, cf_5, cf_6, cf_7, cf_8, cf_9, cf_10, cf_11, cf_12]
    list_idx = [idx_cf_0, idx_cf_1, idx_cf_2, idx_cf_3, idx_cf_4, idx_cf_5, idx_cf_6, idx_cf_7, idx_cf_8, idx_cf_9, idx_cf_10, idx_cf_11, idx_cf_12]


    for idx, cf in enumerate(list_cf):
        expected_index = list_idx[idx]
        idx_econ_limit = econ_limit(cf, method=LimitMethod.NEGATIVE_CASHFLOW)
        assert (
                idx_econ_limit == expected_index
        ), f"Expected index {expected_index}, but got {idx_econ_limit} at cashflow {idx}."

