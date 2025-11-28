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
categories = [
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


def test_sandbox():
    t1 = cashflow_table_oil
    print('\t')
    print(f'Filetype: {type(t1)}')
    print(f'Keys: {t1.keys()}')
    print(f'Length: {len(t1)}')
    print('t1 = \n', t1)


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

    t1 = project_yrs["calculated"]
    print('\t')
    print(f'Filetype: {type(t1)}')
    print(f'Length: {len(t1)}')
    print('t1 = \n', t1)

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
    """
    Test pre-tax expenditure calculations.

    Verifies that computed pre-tax expenditures for each cost category
    (capital, intangible, opex, and ASR) match expected NumPy arrays.
    The test compares model-generated results obtained via
    ``calc_attr('<category>_expenditures_pre_tax')`` against fixed
    reference values.
    """

    pre_tax = {
        # Calculated results
        "calculated": {
            cat: calc_attr(attr=f"{cat}_expenditures_pre_tax") for cat in categories
        },

        # Expected results
        "expected": {
            "capital": np.array(
                [
                    0, 0, 774.1361354, 1326.695271, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "intangible": np.array(
                [
                    0, 0, 3176.383, 6605.103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "opex": np.array(
                [
                    0, 0, 284.4711524, 1081.030049, 973.3351612, 898.6547464, 837.5456371,
                    791.7103488, 760.233636, 740.9126735, 738.9800422, 734.9752292,
                    717.6000544, 662.7462936, 602.5256252, 544.984193, 496.7815578,
                    459.46153, 431.9971443, 405.4186761, 383.6652079, 342.3105733,
                    322.9432748, 326.3205256, 312.7488742, 295.7607236, 277.2252583,
                    259.3172381, 245.0560666, 203.4714604, 169.9048654, 143.5139271,
                    135.5105064, 131.343342, 128.1645362, 124.2886762, 120.9179966,
                    118.2654603
                ]
            ),
            "asr": np.array(
                [
                    0, 0, 28.16770602, 28.16770602, 28.16770602, 28.16770602, 28.16770602,
                    28.16770602, 28.16770602, 28.16770602, 28.16770602, 28.16770602,
                    28.16770602, 28.16770602, 28.16770602, 28.16770602, 28.16770602,
                    28.16770602, 28.16770602, 28.16770602, 28.16770602, 28.16770602,
                    28.16770602, 28.16770602, 28.16770602, 28.16770602, 28.16770602,
                    28.16770602, 28.16770602, 28.16770602, 28.16770602, 28.16770602,
                    28.16770602, 28.16770602, 28.16770602, 28.16770602, 28.16770602,
                    28.16770602
                ]
            ),
            "lbt": np.array(
                [
                    0, 0, 37.04717706, 110.0815218, 126.759145, 117.0333836, 109.0750371,
                    103.1058272, 99.00655966, 96.49035683, 96.23866686, 95.71711304,
                    93.45431356, 86.31061208, 78.46796883, 70.9742472, 64.69673348,
                    59.83648082, 56.2597457, 52.79838518, 49.96539287, 44.57970628,
                    42.05746903, 42.49729432, 40.72983436, 38.51743771, 36.10353156,
                    33.77133869, 31.91408132, 26.49844512, 22.12700858, 18.6900704,
                    17.64777089, 17.10507376, 16.69109232, 16.18633227, 15.74736275,
                    15.4019183
                ]
            ),
        },
    }

    calc = pre_tax["calculated"]
    expected = pre_tax["expected"]

    # Execute testings
    run_test_arr(calc["capital"], expected["capital"], is_strict=True)
    run_test_arr(calc["intangible"], expected["intangible"], is_strict=True)
    run_test_arr(calc["opex"], expected["opex"], is_strict=True)
    run_test_arr(calc["asr"], expected["asr"], is_strict=True)
    run_test_arr(calc["lbt"], expected["lbt"], is_strict=True)


def test_indirect_tax():
    """
    Test indirect tax calculations.

    Ensures that model-computed indirect taxes for all cost categories
    (capital, intangible, opex, ASR, and LBT) match expected NumPy arrays.
    Computed values are obtained via ``calc_attr('<category>_indirect_tax')``.
    """

    indirect_tax = {
        # Calculated results
        "calculated": {cat: calc_attr(attr=f"{cat}_indirect_tax") for cat in categories},

        # Expected results
        "expected": {
            "capital": np.array(
                [
                    0, 0, 85.15497489, 159.2034325, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "intangible": np.array(
                [
                    0, 0, 349.40213, 792.61236, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "opex": np.array(
                [
                    0, 0, 31.29182676, 129.7236059, 116.8002193, 107.8385696, 100.5054764,
                    95.00524185, 91.22803633, 88.90952082, 88.67760507, 88.19702751,
                    86.11200653, 79.52955523, 72.30307503, 65.39810316, 59.61378693,
                    55.13538361, 51.83965731, 48.65024113, 46.03982495, 41.0772688,
                    38.75319298, 39.15846307, 37.52986491, 35.49128683, 33.26703099,
                    31.11806858, 29.406728, 24.41657525, 20.38858384, 17.22167125,
                    16.26126077, 15.76120104, 15.37974435, 14.91464115, 14.51015959,
                    14.19185524
                ]
            ),
            "asr": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "lbt": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        },
    }

    calc = indirect_tax["calculated"]
    expected = indirect_tax["expected"]

    # Execute testings
    run_test_arr(calc["capital"], expected["capital"], is_strict=True)
    run_test_arr(calc["intangible"], expected["intangible"], is_strict=True)
    run_test_arr(calc["opex"], expected["opex"], is_strict=True)
    run_test_arr(calc["asr"], expected["asr"], is_strict=True)
    run_test_arr(calc["lbt"], expected["lbt"], is_strict=True)


def test_expenditures_post_tax():
    """
    Test post-tax expenditure calculations.

    Validates that computed post-tax expenditures for each cost category
    (capital, intangible, opex, and ASR) match expected NumPy arrays.
    Computed values are obtained via ``calc_attr('<category>_postonstream')``.
    """

    post_tax = {
        # Calculated results
        "calculated": {cat: calc_attr(attr=f"{cat}_postonstream") for cat in categories},

        # Expected results
        "expected": {
            "capital": np.array(
                [
                    0, 0, 859.2911103, 1485.898703, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "intangible": np.array(
                [
                    0, 0, 3525.78513, 7397.71536, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "opex": np.array(
                [
                    0, 0, 315.7629792, 1210.753655, 1090.135381, 1006.493316, 938.0511135,
                    886.7155906, 851.4616724, 829.8221944, 827.6576473, 823.1722567,
                    803.712061, 742.2758488, 674.8287003, 610.3822962, 556.3953447,
                    514.5969137, 483.8368016, 454.0689172, 429.7050329, 383.3878421,
                    361.6964678, 365.4789886, 350.2787391, 331.2520104, 310.4922892,
                    290.4353067, 274.4627946, 227.8880357, 190.2934492, 160.7355983,
                    151.7717672, 147.104543, 143.5442806, 139.2033174, 135.4281562,
                    132.4573155
                ]
            ),
            "asr": np.array(
                [
                    0, 0, 28.16770602, 28.16770602, 28.16770602, 28.16770602, 28.16770602,
                    28.16770602, 28.16770602, 28.16770602, 28.16770602, 28.16770602,
                    28.16770602, 28.16770602, 28.16770602, 28.16770602, 28.16770602,
                    28.16770602, 28.16770602, 28.16770602, 28.16770602, 28.16770602,
                    28.16770602, 28.16770602, 28.16770602, 28.16770602, 28.16770602,
                    28.16770602, 28.16770602, 28.16770602, 28.16770602, 28.16770602,
                    28.16770602, 28.16770602, 28.16770602, 28.16770602, 28.16770602,
                    28.16770602
                ]
            ),
            "lbt": np.array(
                [
                    0, 0, 37.04717706, 110.0815218, 126.759145, 117.0333836, 109.0750371,
                    103.1058272, 99.00655966, 96.49035683, 96.23866686, 95.71711304,
                    93.45431356, 86.31061208, 78.46796883, 70.9742472, 64.69673348,
                    59.83648082, 56.2597457, 52.79838518, 49.96539287, 44.57970628,
                    42.05746903, 42.49729432, 40.72983436, 38.51743771, 36.10353156,
                    33.77133869, 31.91408132, 26.49844512, 22.12700858, 18.6900704,
                    17.64777089, 17.10507376, 16.69109232, 16.18633227, 15.74736275,
                    15.4019183
                ]
            ),
        },
    }

    calc = post_tax["calculated"]
    expected = post_tax["expected"]

    # Execute testings
    run_test_arr(calc["capital"], expected["capital"], is_strict=True)
    run_test_arr(calc["intangible"], expected["intangible"], is_strict=True)
    run_test_arr(calc["opex"], expected["opex"], is_strict=True)
    run_test_arr(calc["asr"], expected["asr"], is_strict=True)
    run_test_arr(calc["lbt"], expected["lbt"], is_strict=True)


def test_investments():

    investments = {
        # Calculated results
        "calculated": {
            "capital": calc_attr(attr="expenses_capital"),
            "non_capital": calc_attr(attr="expenses_non_capital"),
            "total": None,
        },

        # Expected results
        "expected": {
            "capital": np.array(
                [
                    0, 0, 859.2911103, 1485.898703, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            "non_capital": np.array(
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
            "total": None,
        },
    }

    calc = investments["calculated"]
    expected = investments["expected"]

    # Execute testings
    run_test_arr(calc["capital"], expected["capital"], is_strict=True)
    run_test_arr(calc["non_capital"], expected["non_capital"], is_strict=True)





