"""Real serializer payload fixtures shared by the Task 5 preflight tests.

The payloads are built from the bundled repository samples (the same real
sources the canonical API tests use) so the tests exercise actual accepted
input shapes rather than invented ones. Nothing here rewrites the original QA
reports or example files.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

from pyscnomics.dataset import sample

_FIXTURES = Path(__file__).parent / "fixtures" / "input_validation"


def corrected_payload() -> dict[str, object]:
    """A real Gross Split POD I serializer payload (strict off)."""
    raw = cast(
        "dict[str, object]",
        json.loads((_FIXTURES / "corrected.json").read_text()),
    )
    return cast("dict[str, object]", copy.deepcopy(raw["payload"]))


def transition_child_payload(scheme: str) -> dict[str, object]:
    """Build one real transition child block for a CR or GS scheme."""
    project = sample.read_json_file("case1" if scheme == "cr" else "case3")
    config = sample.read_json_file("cost_recovery" if scheme == "cr" else "gross_split")
    start_year = int(str(project["start_date"])[-4:])
    end_year = int(str(project["end_date"])[-4:])
    data: dict[str, object] = {
        "setup": {
            "start_date": project["start_date"],
            "end_date": project["end_date"],
            "oil_onstream_date": project["oil_onstream_date"],
            "gas_onstream_date": project["gas_onstream_date"],
            "approval_year": start_year,
        },
        "lifting": project["lifting"],
        "capital": project["capital_cost"],
        "intangible": project["intangible"],
        "opex": project["opex"],
        "asr": project["asr"],
        "lbt": (
            project["lbt"]
            if project["lbt"] is not None
            else _zero_lbt(start_year, end_year)
        ),
        "cost_of_sales": (
            project["cost_of_sales"]
            if project["cost_of_sales"] is not None
            else _zero_cos(start_year, end_year)
        ),
        "costrecovery" if scheme == "cr" else "grosssplit": config,
        "grosssplit" if scheme == "cr" else "costrecovery": None,
        "contract_arguments": {
            "sulfur_revenue": "Addition to Oil Revenue",
            "electricity_revenue": "Addition to Oil Revenue",
            "co2_revenue": "Addition to Oil Revenue",
            "vat_rate": 0.0,
            "inflation_rate": 0.0,
            "inflation_rate_applied_to": "CAPEX",
            "depr_method": "PSC Declining Balance",
            "decline_factor": 2,
            "sum_undepreciated_cost": True,
            "tax_regime": "nailed down",
            "effective_tax_rate": 0.405,
            "ftp_tax_regime": "Direct Mode",
            "post_uu_22_year2001": True,
            "is_dmo_end_weighted": False,
            "regime": "PERMEN_ESDM_12_2020",
            "reservoir_type_permen_2024": "conventional",
            "amortization": False,
            "cum_production_split_offset": 0,
        },
    }
    if scheme == "cr":
        data["costrecovery"] = {
            **config,
            "tax_split_type": "Conventional",
            "condition_dict": {},
            "indicator_rc_icp_sliding": [],
        }
    return data


def single_contract_payload(scheme: str) -> dict[str, object]:
    """A real standalone single-contract payload for a CR or GS scheme.

    Mirrors the canonical ``tests/test_api.py`` builders: one scheme child block
    from the bundled repository samples plus the summary arguments. Used as the
    real input for the CR/GS corruption matrix rather than an invented shape.
    """
    payload = transition_child_payload(scheme)
    payload["summary_arguments"] = transition_summary_arguments()
    return payload


def baseproject_payload() -> dict[str, object]:
    """A real standalone Base Project payload (no CR/GS contract map)."""
    payload = transition_child_payload("gs")
    payload["grosssplit"] = None
    payload["costrecovery"] = None
    payload["summary_arguments"] = transition_summary_arguments()
    return payload


def transition_summary_arguments() -> dict[str, object]:
    return {
        "reference_year": 2023,
        "discount_rate_start_year": 2023,
        "inflation_rate": 0.0,
        "discount_rate": 0.1,
        "npv_mode": "SKK Full Cycle Nominal Terms",
        "discounting_mode": "End Year",
        "profitability_discounted": False,
    }


def transition_payload_for(scheme_1: str, scheme_2: str) -> dict[str, object]:
    """One valid Transition payload for a CR/GS child combination."""
    return {
        "contract_1": transition_child_payload(scheme_1),
        "contract_2": transition_child_payload(scheme_2),
        "contract_arguments": {"unrec_portion": 0.0},
        "summary_arguments": transition_summary_arguments(),
    }


def transition_combinations() -> list[tuple[str, str, str]]:
    """The four real CR/GS transition combinations with a stable label."""
    return [
        ("cr-cr", "cr", "cr"),
        ("cr-gs", "cr", "gs"),
        ("gs-cr", "gs", "cr"),
        ("gs-gs", "gs", "gs"),
    ]


def gs_pod1_two_block_child() -> dict[str, object]:
    """One strict GS POD I child with the two reviewed issue blocks.

    Produces three gas intangible pre-onstream rows and four gas OPEX
    pre-onstream rows, matching the corrected example's two displayed groups.
    """
    return {
        "setup": {
            "start_date": "01/01/2019",
            "end_date": "31/12/2050",
            "oil_onstream_date": "01/01/2029",
            "gas_onstream_date": "01/01/2029",
            "approval_year": 2024,
            "is_pod_1": True,
            "is_strict": True,
        },
        "contract_arguments": {},
        "lifting": {},
        "capital": {},
        "intangible": {
            "base": {
                "start_year": 2019,
                "end_year": 2050,
                "expense_year": [2026, 2027, 2028],
                "cost": [76951.27737, 183682.6991, 268439.1445],
                "cost_allocation": ["Gas", "Gas", "Gas"],
                "cost_type": ["preonstream_cost"] * 3,
            }
        },
        "opex": {
            "base": {
                "start_year": 2019,
                "end_year": 2050,
                "expense_year": [2025, 2026, 2027, 2028],
                "fixed_cost": [25521.58125, 60382.44525, 19108.18638, 18900.21773],
                "cost_allocation": ["Gas", "Gas", "Gas", "Gas"],
                "cost_type": ["preonstream_cost"] * 4,
            }
        },
        "asr": {},
        "lbt": {},
        "grosssplit": {},
        "costrecovery": None,
    }


def strict_transition_two_block_payload() -> dict[str, object]:
    """A Transition payload whose two children each carry the two issue blocks."""
    child = gs_pod1_two_block_child()
    return {
        "contract_1": copy.deepcopy(child),
        "contract_2": copy.deepcopy(child),
        "contract_arguments": {"unrec_portion": 0.0},
        "summary_arguments": {},
    }


def combine_request(
    cases: list[dict[str, object]], reference_year: int = 2023
) -> dict[str, object]:
    return {
        "cases": cases,
        "options": {"reference_year": reference_year},
    }


def _zero_lbt(start_year: int, end_year: int) -> dict[str, object]:
    return {
        "Zero LBT": {
            "start_year": start_year,
            "end_year": end_year,
            "expense_year": [start_year],
            "cost": [0.0],
            "cost_allocation": ["Oil"],
            "description": ["Zero LBT"],
            "final_year": [start_year],
            "utilized_land_area": [0.0],
            "utilized_building_area": [0.0],
            "njop_land": [0.0],
            "njop_building": [0.0],
            "gross_revenue": [0.0],
            "tax_portion": [0.0],
            "tax_discount": [0.0],
        }
    }


def _zero_cos(start_year: int, end_year: int) -> dict[str, object]:
    return {
        "Zero COS": {
            "start_year": start_year,
            "end_year": end_year,
            "expense_year": [start_year],
            "cost": [0.0],
            "cost_allocation": ["Oil"],
            "description": ["Zero COS"],
            "tax_portion": [0.0],
            "tax_discount": [0.0],
        }
    }
