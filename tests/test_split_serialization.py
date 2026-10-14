from typing import Protocol

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from pyscnomics.api.main import app


class _Response(Protocol):
    status_code: int

    def json(self) -> object: ...


class _Client(Protocol):
    def post(self, url: str, *, json: object) -> _Response: ...


client: _Client = TestClient(app, raise_server_exceptions=False)
record = TypeAdapter(dict[str, object])
year_lists = TypeAdapter(dict[str, list[int]])


@pytest.mark.parametrize(
    ("ministerial_split", "expected_years"),
    [(0.0, []), (1.0, [2024, 2025, 2026])],
)
def test_split_response_serializes_maximum_years(
    ministerial_split: float, expected_years: list[int]
) -> None:
    payload = {
        "setup": {
            "start_date": "01/01/2024",
            "end_date": "31/12/2026",
            "gas_onstream_date": "01/01/2024",
            "approval_year": 2024,
        },
        "summary_arguments": {},
        "contract_arguments": {
            "regime": "PERMEN_ESDM_13_2024",
            "reservoir_type_permen_2024": "conventional",
            "cum_production_split_offset": 0,
            "vat_rate": 0,
            "inflation_rate": 0,
            "inflation_rate_applied_to": "None",
            "depr_method": "PSC Declining Balance",
            "decline_factor": 2,
            "sum_undepreciated_cost": True,
            "tax_regime": "nailed down",
            "effective_tax_rate": 0.376,
            "is_dmo_end_weighted": True,
            "amortization": False,
            "sulfur_revenue": "Reduction to Oil OPEX",
            "electricity_revenue": "Reduction to Oil OPEX",
            "co2_revenue": "Reduction to Gas OPEX",
        },
        "grosssplit": {
            "split_ministry_disc": ministerial_split,
            "base_split_ctr_oil": 0.47,
            "base_split_ctr_gas": 0.49,
            "res_type": "conventional",
            "field_loc_2024": "ultradeep_offshore",
            "infra_avail_2024": "not_available",
            "field_reserves_2024": "high",
        },
        "lifting": {
            "Gas": {
                "start_year": 2024,
                "end_year": 2026,
                "prod_year": [2024, 2025, 2026],
                "fluid_type": "Gas",
                "lifting_rate": [100, 100, 100],
                "price": [50, 50, 50],
                "ghv": [1, 1, 1],
            }
        },
    }
    response = client.post("/api/grosssplit/split", json=payload)
    assert response.status_code == 200
    body = record.validate_python(response.json(), strict=True)
    years = year_lists.validate_python(body["years_of_maximum_split"], strict=True)
    assert years["gas"] == expected_years
