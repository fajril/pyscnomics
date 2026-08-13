"""Gross Split deductible summary must include cross-fluid transfer coverage."""

import copy

import numpy as np

from pyscnomics.api.adapter import get_grosssplit


def _payload() -> dict:
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

    cost_sum = float(
        np.nansum(contract._oil_cost_tobe_deducted + contract._gas_cost_tobe_deducted)
    )
    pre_transfer_deductible = float(
        np.nansum(contract._oil_deductible_cost + contract._gas_deductible_cost)
    )
    transfers = float(np.nansum(contract._transfer_to_oil + contract._transfer_to_gas))
    carry_after_transfer = float(contract._consolidated_carward_cost_aftertf[-1])

    assert transfers > 0
    assert pre_transfer_deductible == 0
    assert summary["cost_recovery/deductible_cost"] == cost_sum - carry_after_transfer
    assert summary["unrec_cost"] == carry_after_transfer


def test_tax_uses_combined_pre_transfer_income():
    """Tax must not be deferred by the oil/gas transfer."""

    payload = _payload()
    payload["contract_arguments"]["effective_tax_rate"] = 0.376

    summary, contract, _, _ = get_grosssplit(data=copy.deepcopy(payload))

    combined_taxable = (
        contract._oil_ctr_share_before_transfer
        + contract._gas_ctr_share_before_transfer
        - contract._oil_deductible_cost
        - contract._gas_deductible_cost
        - contract._oil_ddmo
        - contract._gas_ddmo
    )

    np.testing.assert_allclose(
        contract._consolidated_tax_payment,
        combined_taxable * 0.376,
        atol=1e-6,
    )
    assert summary["gov_tax_income"] > 0


def test_point_forward_cashflow_and_irr_match_reference_year():
    """Point-forward summary uses cash flow from the reference year onward."""

    payload = _payload()
    payload["summary_arguments"]["discount_rate_start_year"] = 2026
    payload["summary_arguments"]["npv_mode"] = "Point Forward"

    summary, contract, _, _ = get_grosssplit(data=copy.deepcopy(payload))

    mask = contract.project_years >= 2026
    point_forward_cf = contract._consolidated_cashflow[mask].sum(dtype=float)

    assert summary["ctr_net_cashflow"] == point_forward_cf
    assert summary["ctr_net_cashflow"] != contract._consolidated_cashflow.sum(
        dtype=float
    )
