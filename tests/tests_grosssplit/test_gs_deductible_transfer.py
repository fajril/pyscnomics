"""Gross Split deductible summary must include cross-fluid transfer coverage."""

from array import array
import copy
import math
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import TypeAdapter, ValidationError

from pyscnomics.api.adapter import get_grosssplit
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.econ.indicator import irr, npv_point_forward
from pyscnomics.econ.selection import DiscountingMode

FloatArray = npt.NDArray[np.float64]
_STRING_OBJECT_MAP = TypeAdapter(dict[str, object])
ResultName = Literal[
    "_oil_deductible_cost",
    "_gas_deductible_cost",
    "_transfer_to_oil",
    "_transfer_to_gas",
    "_oil_cost_tobe_deducted",
    "_gas_cost_tobe_deducted",
    "_consolidated_carward_cost_aftertf",
    "_oil_ctr_share_before_transfer",
    "_gas_ctr_share_before_transfer",
    "_oil_ddmo",
    "_gas_ddmo",
    "_consolidated_tax_payment",
    "_consolidated_cashflow",
    "_oil_total_indirect_tax",
    "_gas_total_indirect_tax",
]


class _GrossSplitResults:
    """Runtime-validated access to arrays populated by ``GrossSplit.run``."""

    _contract: GrossSplit

    def __init__(self, contract: GrossSplit) -> None:
        self._contract = contract

    def array(self, name: ResultName) -> FloatArray:
        value = vars(self._contract).get(name)
        if (
            not isinstance(value, np.ndarray)
            or value.ndim != 1
            or value.dtype != np.dtype(np.float64)
        ):
            raise TypeError(f"Expected a float64 array result for {name}")
        return np.asarray(value, dtype=np.float64)


def _section(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    try:
        section = _STRING_OBJECT_MAP.validate_python(value, strict=True)
    except ValidationError as exc:
        raise TypeError(f"Expected {name} to contain a string-keyed mapping") from exc
    payload[name] = section
    return section


def _summary_float(summary: dict[str, object], name: str) -> int | float:
    value = summary.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected {name} to contain a numeric result")
    return value


def _array_sum(values: FloatArray) -> float:
    return math.fsum(array("d", values))


def _float_values(values: FloatArray) -> list[float]:
    return [float(value) for value in array("d", values)]


def _payload() -> dict[str, object]:
    empty_cost = {
        "start_year": 2025,
        "end_year": 2026,
        "expense_year": [2025],
        "cost_allocation": ["Oil"],
        "cost": [0.0],
        "cost_type": [None],
        "tax_portion": [0.0],
        "tax_discount": [0.0],
        "description": ["-"],
    }

    return {
        "setup": {
            "start_date": "01/01/2025",
            "end_date": "31/12/2026",
            "oil_onstream_date": "01/01/2025",
            "gas_onstream_date": "01/01/2025",
            "approval_year": 2025,
            "is_pod_1": False,
            "is_strict": True,
        },
        "summary_arguments": {
            "discount_rate_start_year": 2025,
            "inflation_rate": 0.0,
            "discount_rate": 0.0,
            "npv_mode": "SKK Full Cycle Nominal Terms",
            "discounting_mode": "End Year",
            "profitability_discounted": False,
        },
        "grosssplit": {
            "field_status": "No POD",
            "field_loc": "Onshore",
            "res_depth": "<=2500",
            "infra_avail": "Well Developed",
            "res_type": "Conventional",
            "api_oil": ">=25",
            "domestic_use": "70<=x<100",
            "prod_stage": "Primary",
            "co2_content": "<5",
            "h2s_content": "<100",
            "base_split_ctr_oil": 0.9,
            "base_split_ctr_gas": 0.9,
            "split_ministry_disc": 0.0,
            "oil_dmo_volume_portion": 0.0,
            "oil_dmo_fee_portion": 1.0,
            "oil_dmo_holiday_duration": 0,
            "gas_dmo_volume_portion": 0.0,
            "gas_dmo_fee_portion": 1.0,
            "gas_dmo_holiday_duration": 0,
            "oil_carry_forward_depreciation": 0.0,
            "gas_carry_forward_depreciation": 0.0,
        },
        "contract_arguments": {
            "sulfur_revenue": "Addition to Oil Revenue",
            "electricity_revenue": "Addition to Gas Revenue",
            "co2_revenue": "Reduction to Oil OPEX",
            "tax_regime": "nailed down",
            "effective_tax_rate": 0.0,
            "depr_method": "PSC Declining Balance",
            "decline_factor": 2.0,
            "vat_rate": 0.0,
            "inflation_rate": 0.0,
            "inflation_rate_applied_to": "CAPEX",
            "cum_production_split_offset": 0.0,
            "amortization": False,
            "regime": "PERMEN_ESDM_12_2020",
            "sum_undepreciated_cost": True,
            "oil_cost_of_sales_applied": False,
            "gas_cost_of_sales_applied": False,
            "reservoir_type_permen_2024": "conventional",
        },
        "lifting": {
            "Oil Source": {
                "start_year": 2025,
                "end_year": 2026,
                "prod_rate_baseline": [100.0, 100.0],
                "prod_rate": None,
                "lifting_rate": [100.0, 100.0],
                "price": [100.0, 100.0],
                "prod_year": [2025, 2026],
                "fluid_type": "Oil",
            },
            "Gas Source": {
                "start_year": 2025,
                "end_year": 2026,
                "prod_rate_baseline": [0.0, 0.0],
                "prod_rate": [0.0, 0.0],
                "lifting_rate": [0.0, 0.0],
                "price": [1.0, 1.0],
                "prod_year": [2025, 2026],
                "ghv": [1.0, 1.0],
                "fluid_type": "Gas",
            },
        },
        "capital": {
            "empty": {
                **empty_cost,
                "pis_year": [None],
                "useful_life": [None],
                "depreciation_factor": [None],
                "is_ic_applied": [False],
                "salvage_value": [None],
            }
        },
        "intangible": {"empty": dict(empty_cost)},
        "opex": {
            "gas_opex": {
                "start_year": 2025,
                "end_year": 2026,
                "expense_year": [2025],
                "cost_allocation": ["Gas"],
                "fixed_cost": [5000.0],
                "cost_type": ["postonstream_cost"],
                "prod_rate": [0.0],
                "cost_per_volume": [0.0],
                "tax_portion": [0.0],
                "tax_discount": [0.0],
                "description": ["gas cost"],
            }
        },
        "asr": {"empty": {**empty_cost, "final_year": [2025], "future_rate": [0.0]}},
        "lbt": {
            "empty": {
                **empty_cost,
                "final_year": [2025],
                "utilized_land_area": [0.0],
                "utilized_building_area": [0.0],
                "njop_land": [0.0],
                "njop_building": [0.0],
                "gross_revenue": [0.0],
            }
        },
        "cost_of_sales": {"empty": dict(empty_cost)},
    }


def test_deductible_summary_includes_transfer_coverage():
    """Gas costs covered by an oil surplus are still recovered deductible costs."""

    summary, contract, _, _ = get_grosssplit(data=copy.deepcopy(_payload()))
    results = _GrossSplitResults(contract)

    pre_transfer_deductible = float(
        np.nansum(
            results.array("_oil_deductible_cost")
            + results.array("_gas_deductible_cost")
        )
    )
    transfers = float(
        np.nansum(results.array("_transfer_to_oil") + results.array("_transfer_to_gas"))
    )
    cost_to_be_deducted = float(
        np.nansum(
            results.array("_oil_cost_tobe_deducted")
            + results.array("_gas_cost_tobe_deducted")
        )
    )
    carry_after_transfer = _float_values(
        results.array("_consolidated_carward_cost_aftertf")
    )[-1]
    deductible_cost = _summary_float(summary, "cost_recovery/deductible_cost")
    unrec_cost = _summary_float(summary, "unrec_cost")

    assert transfers > 0
    assert pre_transfer_deductible == 0
    assert deductible_cost == pre_transfer_deductible + transfers
    assert deductible_cost == cost_to_be_deducted - carry_after_transfer
    assert unrec_cost == carry_after_transfer


def test_tax_uses_combined_pre_transfer_income():
    """Tax must not be deferred by the oil/gas transfer."""

    payload = _payload()
    _section(payload, "contract_arguments")["effective_tax_rate"] = 0.376

    summary, contract, _, _ = get_grosssplit(data=copy.deepcopy(payload))
    results = _GrossSplitResults(contract)

    combined_taxable = (
        results.array("_oil_ctr_share_before_transfer")
        + results.array("_gas_ctr_share_before_transfer")
        - results.array("_oil_deductible_cost")
        - results.array("_gas_deductible_cost")
        - results.array("_oil_ddmo")
        - results.array("_gas_ddmo")
    )

    np.testing.assert_allclose(
        results.array("_consolidated_tax_payment"),
        combined_taxable * 0.376,
        atol=1e-6,
    )
    assert _summary_float(summary, "gov_tax_income") > 0


def test_point_forward_net_cashflow_remains_full_lifecycle():
    """Point-forward NPV must not change the reported project cash flow."""

    payload = _payload()
    reference_year = 2026
    summary_arguments = _section(payload, "summary_arguments")
    summary_arguments["discount_rate_start_year"] = reference_year
    summary_arguments["npv_mode"] = "Point Forward"

    summary, contract, _, _ = get_grosssplit(data=copy.deepcopy(payload))
    results = _GrossSplitResults(contract)

    point_forward = contract.project_years >= reference_year
    cashflow = results.array("_consolidated_cashflow")
    assert _summary_float(summary, "ctr_net_cashflow") == _array_sum(cashflow)
    assert _summary_float(summary, "ctr_irr") == irr(cashflow[point_forward])
    assert _summary_float(summary, "ctr_npv") == npv_point_forward(
        cashflow=cashflow,
        cashflow_years=contract.project_years,
        discount_rate=0.1,
        reference_year=reference_year,
        discounting_mode=DiscountingMode.END_YEAR,
    )


def test_total_indirect_taxes_includes_lbt():
    """Total indirect taxes include VAT and LBT."""

    payload = _payload()
    payload["lbt"] = {
        "gas_lbt": {
            "start_year": 2025,
            "end_year": 2026,
            "expense_year": [2025],
            "cost_allocation": ["Gas"],
            "cost": [100.0],
            "cost_type": ["postonstream_cost"],
            "final_year": [2025],
            "utilized_land_area": [0.0],
            "utilized_building_area": [0.0],
            "njop_land": [0.0],
            "njop_building": [0.0],
            "gross_revenue": [0.0],
            "tax_portion": [0.0],
            "tax_discount": [0.0],
            "description": ["gas lbt"],
        }
    }

    summary, contract, _, _ = get_grosssplit(data=copy.deepcopy(payload))
    results = _GrossSplitResults(contract)

    expected_vat = _array_sum(results.array("_oil_total_indirect_tax")) + _array_sum(
        results.array("_gas_total_indirect_tax")
    )
    total_indirect_taxes = _summary_float(summary, "total_indirect_taxes")
    lbt = _summary_float(summary, "lbt")
    assert total_indirect_taxes == expected_vat + lbt
    assert total_indirect_taxes > 0
