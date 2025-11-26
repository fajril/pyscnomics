"""
A collection of unit testings for CASE_02
"""

# import pytest
import numpy as np
import pandas as pd
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.econ.selection import ContractType
from pyscnomics.dataset.case_02 import Case02
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
    "cls": Case02,
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


def calc_attr(attr: str) -> np.ndarray:
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


def run_test_arr(calculated: np.ndarray, expected: np.ndarray, is_strict: bool = False):
    """
    Assert numerical agreement between two arrays.

    Parameters
    ----------
    calculated : np.ndarray
        Computed array to test.
    expected : np.ndarray
        Reference array.
    is_strict : bool, default=False
        If True, require full precision; otherwise compare to 2 decimals.

    Returns
    -------
    None
        Raises an assertion error if the arrays differ.
    """

    if is_strict:
        return np.testing.assert_array_almost_equal(calculated, expected)

    return np.testing.assert_array_almost_equal(calculated, expected, decimal=3)


def test_project_years():
    """
    Verify that the computed project years match the expected array.

    Asserts
    -------
    All elements of the calculated and expected year arrays are equal.
    """

    project_yrs = {
        # Calculated result
        "calculated": calc_attr(attr="years"),

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
    run_test_arr(**project_yrs)


def test_lifting():
    """
    Validate that the calculated lifting values match the expected array.

    Asserts
    -------
    The computed lifting array is element-wise equal to the expected values.
    """

    lft_oil = {
        # Calculated result
        "calculated": calc_attr(attr="lifting"),

        # Expected result
        "expected": np.array(
            [
                0, 0, 30.74964895, 91.36912501, 105.2117737, 97.13926262, 90.53372936,
                85.57920581, 82.17675934, 80.08827758, 79.87937156, 79.44647497,
                77.56832135, 71.63895425, 65.1294562, 58.90956773, 53.69914798,
                49.66507373, 46.69633607, 43.82336087, 41.47193963, 37.00174824,
                34.90825783, 35.27331866, 33.80630342, 31.96998482, 29.96641066,
                28.0306596, 26.48911132, 21.99406136, 18.36571097, 15.51300664,
                14.64788421, 14.19743838, 13.85382829, 13.43487075, 13.07052021,
                12.78379673
            ]
        ),
    }

    # Execute testing
    run_test_arr(**lft_oil, is_strict=True)


def test_price():
    """
    Validate that the calculated price array matches the expected constant values.

    Asserts
    -------
    The computed price array is element-wise equal to the expected values.
    """

    price_oil = {
        # Calculated result
        "calculated": calc_attr(attr="price"),

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
    run_test_arr(**price_oil, is_strict=True)


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
        "calculated": calc_attr(attr="revenue"),

        # Expected result
        "expected": np.array(
            [
                0, 0, 1844.978937, 5482.147501, 6312.706424, 5828.355757, 5432.023762,
                5134.752348, 4930.605561, 4805.296655, 4792.762294, 4766.788498,
                4654.099281, 4298.337255, 3907.767372, 3534.574064, 3221.948879,
                2979.904424, 2801.780164, 2629.401652, 2488.316378, 2220.104895,
                2094.49547, 2116.399119, 2028.378205, 1918.199089, 1797.984639,
                1681.839576, 1589.346679, 1319.643681, 1101.942658, 930.7803986,
                878.8730525, 851.8463029, 831.2296971, 806.0922447, 784.2312128,
                767.0278037
            ]
        ),
    }

    # Execute testing
    run_test_arr(**rev, is_strict=True)


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
            "lifting": calc_attr(attr="lifting_sulfur"),
            "price": calc_attr(attr="price_sulfur"),
            "revenue": calc_attr(attr="revenue_sulfur"),
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
    run_test_arr(calc["lifting"], expected["lifting"], is_strict=True)
    run_test_arr(calc["price"], expected["price"], is_strict=True)
    run_test_arr(calc["revenue"], expected["revenue"], is_strict=True)


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
            "lifting": calc_attr(attr="lifting_electricity"),
            "price": calc_attr(attr="price_electricity"),
            "revenue": calc_attr(attr="revenue_electricity"),
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
    run_test_arr(calc["lifting"], expected["lifting"], is_strict=True)
    run_test_arr(calc["price"], expected["price"], is_strict=True)
    run_test_arr(calc["revenue"], expected["revenue"], is_strict=True)


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
            "lifting": calc_attr(attr="lifting_co2"),
            "price": calc_attr(attr="price_co2"),
            "revenue": calc_attr(attr="revenue_co2"),
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
    run_test_arr(calc["lifting"], expected["lifting"], is_strict=True)
    run_test_arr(calc["price"], expected["price"], is_strict=True)
    run_test_arr(calc["revenue"], expected["revenue"], is_strict=True)


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
            "depreciable": calc_attr(attr="sunk_cost_depreciable"),
            "non_depreciable": calc_attr(attr="sunk_cost_non_depreciable"),
            "total": calc_attr(attr="sunk_cost"),
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
    run_test_arr(calc["depreciable"], expected["depreciable"], is_strict=True)
    run_test_arr(calc["non_depreciable"], expected["non_depreciable"], is_strict=True)
    run_test_arr(calc["total"], expected["total"], is_strict=True)


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
            "depreciable": calc_attr(attr="preonstream_depreciable"),
            "non_depreciable": calc_attr(attr="preonstream_non_depreciable"),
            "total": calc_attr(attr="preonstream"),
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
    run_test_arr(calc["depreciable"], expected["depreciable"], is_strict=True)
    run_test_arr(calc["non_depreciable"], expected["non_depreciable"], is_strict=True)
    run_test_arr(calc["total"], expected["total"], is_strict=True)


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
            "depreciable": calc_attr(attr="postonstream_depreciable"),
            "non_depreciable": calc_attr(attr="postonstream_non_depreciable"),
            "total": calc_attr(attr="postonstream"),
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

    calc = postos["calculated"]
    expected = postos["expected"]

    # Execute testings
    run_test_arr(calc["depreciable"], expected["depreciable"])
    run_test_arr(calc["non_depreciable"], expected["non_depreciable"])
    run_test_arr(calc["total"], expected["total"])


def test_expenditures_pre_tax():
    pass


def test_indirect_tax():
    pass


def test_expenditures_post_tax():
    pass


def test_investments():
    pass


def test_sandbox():
    t1 = summary
    print('\t')
    print(f'Filetype: {type(t1)}')
    print(f'Length: {len(t1)}')
    print('t1 = \n', t1)

