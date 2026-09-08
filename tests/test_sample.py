from datetime import date

import numpy as np
import pytest

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.project import BaseProject
from pyscnomics.dataset.sample import (
    assign_cost,
    assign_lifting,
    assign_onstream_date,
    load_data,
    load_testing,
    read_json_file,
    read_cost_type,
    read_fluid_type,
)
from pyscnomics.econ.costs import LBT, CostOfSales
from pyscnomics.econ.revenue import LiftingException
from pyscnomics.econ.selection import CostType, FluidType


def test_read_fluid_type_scalar_and_list() -> None:
    assert read_fluid_type("Oil") is FluidType.OIL
    assert read_fluid_type("unknown") is None
    assert read_fluid_type(["Oil", "unknown", "Gas"]) == [
        FluidType.OIL,
        FluidType.GAS,
    ]


def test_read_cost_type_scalar_list_and_invalid_inputs() -> None:
    assert read_cost_type("sunk_cost") is CostType.SUNK_COST
    assert read_cost_type(["sunk_cost", None, "postonstream_cost"]) == [
        CostType.SUNK_COST,
        None,
        CostType.POST_ONSTREAM_COST,
    ]
    assert isinstance(read_cost_type(None), TypeError)
    with pytest.raises(KeyError):
        _ = read_cost_type("invalid")
    with pytest.raises(TypeError):
        _ = read_cost_type(["invalid"])


def test_assign_onstream_date_supports_date_and_none() -> None:
    assert assign_onstream_date("21/08/2026") == date(2026, 8, 21)
    assert assign_onstream_date(None) is None


def test_load_data_dispatches_supported_contracts_and_returns_none_for_unknown() -> (
    None
):
    assert isinstance(load_data("case1", "project"), BaseProject)
    assert isinstance(load_data("case1", "cost_recovery"), CostRecovery)
    assert isinstance(load_data("case3", "gross_split"), GrossSplit)
    assert load_data("case1", "unsupported") is None


def test_load_testing_without_testing_returns_value_error() -> None:
    result = load_testing("cost_recovery", "missing")

    assert isinstance(result, ValueError)
    assert str(result) == "The testing data for this dataset is still in development."


def test_assign_lifting_missing_prod_rate_preserves_key_error() -> None:
    data = {
        "lifting": {
            "Oil": {
                "start_year": 2020,
                "end_year": 2020,
                "lifting_rate": [1.0],
                "price": [1.0],
                "prod_year": [2020],
                "fluid_type": "Oil",
                "prod_rate_baseline": [1.0],
            }
        }
    }

    with pytest.raises(KeyError, match="'prod_rate'"):
        _ = assign_lifting(data)


def test_assign_lifting_missing_prod_rate_baseline_preserves_key_error() -> None:
    data = {
        "lifting": {
            "Oil": {
                "start_year": 2020,
                "end_year": 2020,
                "lifting_rate": [1.0],
                "price": [1.0],
                "prod_year": [2020],
                "fluid_type": "Oil",
                "prod_rate": [1.0],
            }
        }
    }

    with pytest.raises(KeyError, match="'prod_rate_baseline'"):
        _ = assign_lifting(data)


def test_assign_lifting_rejects_list_fluid_type() -> None:
    data = {
        "lifting": {
            "Oil": {
                "start_year": 2020,
                "end_year": 2020,
                "lifting_rate": [1.0],
                "price": [1.0],
                "prod_year": [2020],
                "fluid_type": ["Oil"],
                "prod_rate": None,
                "prod_rate_baseline": None,
            }
        }
    }

    with pytest.raises(LiftingException):
        _ = assign_lifting(data)


def test_assign_cost_reads_lbt_and_cost_of_sales_sections() -> None:
    data = {
        "capital_cost": None,
        "intangible": None,
        "opex": None,
        "asr": {
            "ASR": {
                "start_year": 2020,
                "end_year": 2020,
                "cost": [1.0],
                "expense_year": [2020],
                "cost_allocation": ["Oil"],
                "description": ["asr"],
                "final_year": [2020],
                "future_rate": [0.0],
            }
        },
        "lbt": {
            "LBT": {
                "start_year": 2020,
                "end_year": 2020,
                "cost": [2.0],
                "expense_year": [2020],
                "cost_allocation": ["Gas"],
                "description": ["lbt"],
                "tax_portion": [0.0],
                "tax_discount": [0.0],
                "final_year": [2020],
                "utilized_land_area": [1.0],
                "utilized_building_area": [1.0],
                "njop_land": [1.0],
                "njop_building": [1.0],
                "gross_revenue": [1.0],
            }
        },
        "cost_of_sales": {
            "COS": {
                "start_year": 2020,
                "end_year": 2020,
                "cost": [3.0],
                "expense_year": [2020],
                "cost_allocation": ["Sulfur"],
                "description": ["cos"],
                "tax_portion": [0.0],
                "tax_discount": [0.0],
            }
        },
    }

    _, _, _, _, lbt_list, cos_list = assign_cost(data)

    assert lbt_list is not None
    assert cos_list is not None
    assert isinstance(lbt_list[0], LBT)
    assert isinstance(cos_list[0], CostOfSales)
    lbt_cost = lbt_list[0].cost
    cos_cost = cos_list[0].cost
    assert lbt_cost is not None
    assert cos_cost is not None
    np.testing.assert_array_equal(lbt_cost, [2.0])
    np.testing.assert_array_equal(cos_cost, [3.0])


def test_loaded_arrays_preserve_numpy_dtype_inference() -> None:
    data = read_json_file("case1")

    lifting = assign_lifting(data)
    assert lifting is not None
    lifting_rate = lifting[0].lifting_rate
    prod_year = lifting[0].prod_year
    assert lifting_rate is not None
    assert prod_year is not None
    assert lifting_rate.dtype == np.dtype("float64")
    assert prod_year.dtype == np.dtype("int64")

    capital, _, _, _, _, _ = assign_cost(data)
    assert capital is not None
    useful_life = capital[0].useful_life
    assert useful_life is not None
    assert useful_life.dtype == np.dtype("float64")
