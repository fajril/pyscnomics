"""
A collection of unit testings for CASE 04
"""

# import pytest
import numpy as np
import pandas as pd
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.econ.selection import ContractType
from pyscnomics.dataset.case_04 import Case04
from pyscnomics.validation.preparation import execute_contract
from pyscnomics.tools.table import get_table


# Specify cost types
cost_types = [
    "capital",
    "intangible",
    "opex",
    "asr",
    "lbt",
]

# Specify arguments to run function "execute_contract()"
kwargs_execute = {
    "cls": Case04,
    "contract_type": ContractType.GROSS_SPLIT,
    "run_as_dict": True,
}

# Run the contract using function "execute_contract()"
ctr = execute_contract(**kwargs_execute)

# Results of run in terms of "data", "contract", "summary", and "cashflow"
data: dict = ctr["data"]
contract: GrossSplit = ctr["contract"]
summary: dict = ctr["summary"]
cashflow_table_oil: pd.DataFrame = get_table(contract=contract)[0]


def _calc_attr(attr: str) -> np.ndarray:
    """
    Return the cashflow array for a given attribute.

    Parameters
    ----------
    attr : str
        Key of the cashflow attribute to extract.

    Returns
    -------
    ndarray
        Array representation of the selected cashflow column.
    """
    return cashflow_table_oil[attr].to_numpy()


def test_project_years():
    """
    Verify that the computed project years match the expected array.

    Asserts
    -------
    All elements of the calculated and expected year arrays are equal.
    """

    project_yrs = {
        # Calculated result
        "calculated": _calc_attr(attr="years"),

        # Expected result
        "expected": np.array(
            [
                2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032,
                2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043,
                2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054,
                2055, 2056, 2057, 2058, 2059
            ]
        ),
    }

    # Execute testing
    np.testing.assert_allclose(project_yrs["calculated"], project_yrs["expected"])


def test_lifting():
    """
    Validate that the calculated lifting values match the expected array.

    Asserts
    -------
    The computed lifting array is element-wise equal to the expected values.
    """

    lft_oil = {
        # Calculated result
        "calculated": _calc_attr(attr="lifting"),

        # Expected result
        "expected": np.array(
            [
                0, 0, 30.75, 91.37, 105.21, 97.14, 90.53, 85.58, 82.18, 80.09, 79.88,
                79.45, 77.57, 71.64, 65.13, 58.91, 53.70, 49.67, 46.70, 43.82, 41.47,
                37.00, 34.91, 35.27, 33.81, 31.97, 29.97, 28.03, 26.49, 21.99, 18.37,
                15.51, 14.65, 14.20, 13.85, 13.43, 13.07, 12.78
            ]
        ),
    }

    # Execute testing
    np.testing.assert_allclose(lft_oil["calculated"], lft_oil["expected"], atol=1e-2)


def test_price():
    """
    Validate that the calculated price array matches the expected constant values.

    Asserts
    -------
    The computed price array is element-wise equal to the expected values.
    """

    price_oil = {
        # Calculated result
        "calculated": _calc_attr(attr="price"),

        # Expected result
        "expected": np.array(
            [
                0, 0, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                60, 60, 60, 60
            ]
        ),
    }

    # Execute testing
    np.testing.assert_allclose(price_oil["calculated"], price_oil["expected"])


def test_revenue():
    """
    Validate that the calculated revenue array matches expected values.

    Asserts
    -------
    The computed revenue array is element-wise equal to the expected array
    within a tolerance of 1e-2.
    """

    rev = {
        # Calculated result
        "calculated": _calc_attr(attr="revenue"),

        # Expected result
        "expected": np.array(
            [
                0.00, 0.00, 1844.98, 5482.15, 6312.71, 5828.36, 5432.02, 5134.75, 4930.61,
                4805.30, 4792.76, 4766.79, 4654.10, 4298.34, 3907.77, 3534.57, 3221.95,
                2979.90, 2801.78, 2629.40, 2488.32, 2220.10, 2094.50, 2116.40, 2028.38,
                1918.20, 1797.98, 1681.84, 1589.35, 1319.64, 1101.94, 930.78, 878.87,
                851.85, 831.23, 806.09, 784.23, 767.03
            ]
        ),
    }

    # Execute testing
    np.testing.assert_allclose(rev["calculated"], rev["expected"], atol=1e-2)


def test_sulfur():
    """
    Validate sulfur-related arrays for lifting, price, and revenue.

    Asserts
    -------
    Each calculated sulfur array (lifting, price, revenue) matches the
    corresponding expected zero-valued array.
    """

    sulfur = {
        # Calculated results
        "calculated": {
            "lifting": _calc_attr(attr="lifting_sulfur"),
            "price": _calc_attr(attr="price_sulfur"),
            "revenue": _calc_attr(attr="revenue_sulfur"),
        },

        # Expected results
        "expected": {
            "lifting": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "price": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "revenue": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        },
    }

    calc = sulfur["calculated"]
    expected = sulfur["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["lifting"], expected["lifting"])
    np.testing.assert_allclose(calc["price"], expected["price"])
    np.testing.assert_allclose(calc["revenue"], expected["revenue"])


def test_electricity():
    """
    Validate electricity-related lifting, price, and revenue arrays.

    Asserts
    -------
    Each calculated electricity array matches the corresponding expected
    zero-valued array.
    """

    electricity = {
        # Calculated results
        "calculated": {
            "lifting": _calc_attr(attr="lifting_electricity"),
            "price": _calc_attr(attr="price_electricity"),
            "revenue": _calc_attr(attr="revenue_electricity"),
        },

        # Expected results
        "expected": {
            "lifting": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "price": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "revenue": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }
    }

    calc = electricity["calculated"]
    expected = electricity["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["lifting"], expected["lifting"])
    np.testing.assert_allclose(calc["price"], expected["price"])
    np.testing.assert_allclose(calc["revenue"], expected["revenue"])


def test_co2():
    """
    Validate CO₂-related lifting, price, and revenue arrays.

    Asserts
    -------
    Each calculated CO₂ array matches the corresponding expected
    zero-valued array.
    """

    co2 = {
        # Calculated results
        "calculated": {
            "lifting": _calc_attr(attr="lifting_co2"),
            "price": _calc_attr(attr="price_co2"),
            "revenue": _calc_attr(attr="revenue_co2"),
        },

        # Expected results
        "expected": {
            "lifting": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "price": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "revenue": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }
    }

    calc = co2["calculated"]
    expected = co2["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["lifting"], expected["lifting"])
    np.testing.assert_allclose(calc["price"], expected["price"])
    np.testing.assert_allclose(calc["revenue"], expected["revenue"])


def test_sunk_cost():
    """
    Test sunk-cost arrays.

    Confirms
    --------
    calculated : np.ndarray
        Computed sunk-cost components.
    expected : np.ndarray
        Reference zero arrays.

    Ensures
    -------
    All calculated sunk-cost arrays match expected values (within tolerance).
    """

    sc = {
        # Calculated results
        "calculated": {
            "depreciable": _calc_attr(attr="sunk_cost_depreciable"),
            "non_depreciable": _calc_attr(attr="sunk_cost_non_depreciable"),
            "total": _calc_attr(attr="sunk_cost"),
        },

        # Expected results
        "expected": {
            "depreciable": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "non_depreciable": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "total": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        },
    }

    calc = sc["calculated"]
    expected = sc["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["depreciable"], expected["depreciable"])
    np.testing.assert_allclose(calc["non_depreciable"], expected["non_depreciable"])
    np.testing.assert_allclose(calc["total"], expected["total"])


def test_preonstream_cost():
    """
    Test pre-onstream cost arrays.

    Confirms
    --------
    calculated : dict of np.ndarray
        Computed depreciable, non-depreciable, and total pre-onstream costs.
    expected : dict of np.ndarray
        Reference zero arrays for each component.

    Ensures
    -------
    All calculated pre-onstream cost arrays match expected values within tolerance.
    """

    preos = {
        # Calculated results
        "calculated": {
            "depreciable": _calc_attr(attr="preonstream_depreciable"),
            "non_depreciable": _calc_attr(attr="preonstream_non_depreciable"),
            "total": _calc_attr(attr="preonstream"),
        },

        # Expected results
        "expected": {
            "depreciable": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "non_depreciable": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "total": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        },
    }

    calc = preos["calculated"]
    expected = preos["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["depreciable"], expected["depreciable"])
    np.testing.assert_allclose(calc["non_depreciable"], expected["non_depreciable"])
    np.testing.assert_allclose(calc["total"], expected["total"])


def test_postonstream_cost():
    """
    Test post-onstream cost components.

    Verifies
    --------
    depreciable, non_depreciable, total : np.ndarray
        Calculated arrays match expected values to 3 decimals.
    """

    postos = {
        # Calculated results
        "calculated": {
            "depreciable": _calc_attr(attr="postonstream_depreciable"),
            "non_depreciable": _calc_attr(attr="postonstream_non_depreciable"),
            "total": _calc_attr(attr="postonstream"),
        },

        # Expected results
        "expected": {
            "depreciable": np.array(
                [
                    0, 0, 859.2911103, 1485.898703, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "non_depreciable": np.array(
                [
                    0, 0, 3906.762992, 8746.718243, 1245.062232, 1151.694406, 1075.293857,
                    1017.989124, 978.635938, 954.4802572, 952.0640202, 947.0570758,
                    925.3340805, 856.7541669, 781.4643751, 709.5242494, 649.2597842,
                    602.6011005, 568.2642533, 535.0350084, 507.8381318, 456.1352544,
                    431.9216428, 436.143989, 419.1762795, 397.9371541, 374.7635268,
                    352.3743514, 334.544582, 282.5541868, 240.5881638, 207.5933747,
                    197.5872441, 192.3773228, 188.4030789, 183.5573557, 179.343225,
                    176.0269398
                ]
            ),
            "total": np.array(
                [
                    0, 0, 4766.054103, 10232.61695, 1245.062232, 1151.694406, 1075.293857,
                    1017.989124, 978.635938, 954.4802572, 952.0640202, 947.0570758,
                    925.3340805, 856.7541669, 781.4643751, 709.5242494, 649.2597842,
                    602.6011005, 568.2642533, 535.0350084, 507.8381318, 456.1352544,
                    431.9216428, 436.143989, 419.1762795, 397.9371541, 374.7635268,
                    352.3743514, 334.544582, 282.5541868, 240.5881638, 207.5933747,
                    197.5872441, 192.3773228, 188.4030789, 183.5573557, 179.343225,
                    176.0269398
                ]
            ),
        },
    }

    # Execute testing
    def _validate(target: str):
        return np.testing.assert_array_almost_equal(
            postos["calculated"][target], postos["expected"][target], decimal=3
        )

    _validate(target="depreciable")
    _validate(target="non_depreciable")
    _validate(target="total")



