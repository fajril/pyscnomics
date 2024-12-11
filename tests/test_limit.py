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
