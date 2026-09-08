"""
API smoke tests — verify core endpoints and import chains work after refactoring.
Run: uv run pytest tests/test_api.py -v
"""

import asyncio
from collections.abc import Coroutine, Mapping
from datetime import date
import math
from typing import Protocol, TypeGuard, final

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from pyscnomics.dataset import sample

from pyscnomics.api.converter import (
    CasesRequestBM,
    Data,
    DataTransition,
    convert_dict_to_capital,
    convert_to_float,
)
from pyscnomics.api.main import app


_STRING_RECORD = TypeAdapter(dict[str, object])
_RECORD_LIST = TypeAdapter(list[dict[str, object]])


@pytest.mark.parametrize("value", [np.int32(7), np.int64(7), np.uint64(7)])
def test_convert_to_float_accepts_numpy_integers(value: object) -> None:
    assert convert_to_float(value) == 7.0


class _IndexOnly:
    def __index__(self) -> int:
        return 9


@pytest.mark.parametrize(
    ("value", "expected"),
    [(bytearray(b"7.25"), 7.25), (memoryview(b"8.5"), 8.5), (_IndexOnly(), 9.0)],
)
def test_convert_to_float_accepts_float_protocol_inputs(
    value: object, expected: float
) -> None:
    assert convert_to_float(value) == expected


def test_cost_input_mapping_rejects_malformed_nested_record() -> None:
    with pytest.raises(ValidationError):
        _ = convert_dict_to_capital(
            {"capital": {"start_year": 2023, "end_year": "2024"}}
        )


class _Response(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def json(self) -> object: ...


class _ApiClient(Protocol):
    def get(self, url: str) -> _Response: ...

    def options(self, url: str, *, headers: Mapping[str, str]) -> _Response: ...

    def post(self, url: str, *, json: object) -> _Response: ...


class _MonkeyPatch(Protocol):
    def setattr(self, target: object, name: str, value: object) -> None: ...


class _WarningRoute(Protocol):
    def __call__(
        self, data: Data
    ) -> Coroutine[object, object, list[dict[str, object]]]: ...


class _TableRoute(Protocol):
    def __call__(self, data: Data) -> Coroutine[object, object, dict[str, object]]: ...


def _is_record(value: object) -> TypeGuard[dict[str, object]]:
    try:
        _ = _STRING_RECORD.validate_python(value, strict=True)
    except ValidationError:
        return False
    return True


def _is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_record_list(value: object) -> TypeGuard[list[dict[str, object]]]:
    try:
        _ = _RECORD_LIST.validate_python(value, strict=True)
    except ValidationError:
        return False
    return True


def _response_body(response: _Response) -> dict[str, object]:
    body = response.json()
    assert _is_record(body)
    return body


def _data_request() -> Data:
    return Data.model_validate(
        {
            "setup": {"start_date": "01/01/2023", "end_date": "31/12/2027"},
            "summary_arguments": {},
            "contract_arguments": {},
        }
    )


def _cases_request() -> CasesRequestBM:
    return CasesRequestBM.model_validate(
        {
            "cases": [{"cId": 1, "ctr": 1, "data": {}}],
            "options": {"reference_year": 2023},
        }
    )


client: _ApiClient = TestClient(app)


def _transition_child_payload(scheme: str) -> dict[str, object]:
    """Build one scheme payload from bundled repository JSON evidence."""
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
        "lbt": project["lbt"]
        if project["lbt"] is not None
        else {
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
        },
        "cost_of_sales": (
            project["cost_of_sales"]
            if project["cost_of_sales"] is not None
            else {
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


def _transition_payload() -> dict[str, object]:
    return {
        "contract_1": _transition_child_payload("cr"),
        "contract_2": _transition_child_payload("gs"),
        "contract_arguments": {"unrec_portion": 0.0},
        "summary_arguments": {
            "reference_year": 2023,
            "discount_rate_start_year": 2023,
            "inflation_rate": 0.0,
            "discount_rate": 0.1,
            "npv_mode": "SKK Full Cycle Nominal Terms",
            "discounting_mode": "End Year",
            "profitability_discounted": False,
        },
    }


# ── GET endpoints ──────────────────────────────────────────────────


class TestGetEndpoints:
    def test_root_returns_version(self):
        resp = client.get("/api/")
        assert resp.status_code == 200
        body = _response_body(resp)
        assert "Pyscnomics" in body

    def test_health_endpoint(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert _response_body(resp) == {"status": "ok"}

    def test_version_endpoint(self):
        from pyscnomics import __version__

        resp = client.get("/api/version")
        assert resp.status_code == 200
        assert _response_body(resp) == {"version": __version__}

    def test_cors_headers(self):
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"

    def test_docs_available(self):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_available(self):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = _response_body(resp)
        assert "paths" in schema
        paths = schema["paths"]
        assert _is_record(paths)
        assert "/api/" in paths
        assert "/api/costrecovery" in paths
        assert "/api/grosssplit" in paths


# ── Library import chain tests ─────────────────────────────────────


class TestImportChains:
    """Verify core modules import cleanly — catches broken imports after refactoring."""

    def test_import_contracts(self):
        from pyscnomics.contracts.costrecovery import CostRecovery
        from pyscnomics.contracts.gross_split import GrossSplit
        from pyscnomics.contracts.transition import Transition
        from pyscnomics.contracts.project import BaseProject

        _ = (CostRecovery, GrossSplit, Transition, BaseProject)

    def test_import_optimize(self):
        from pyscnomics.optimize.optimization import optimize_psc
        from pyscnomics.optimize.sensitivity import sensitivity_psc
        from pyscnomics.optimize.uncertainty import uncertainty_psc

        _ = (optimize_psc, sensitivity_psc, uncertainty_psc)

    def test_import_tools(self):
        from pyscnomics.tools.table import get_table
        from pyscnomics.tools.summary import get_summary

        _ = (get_table, get_summary)

    def test_import_io(self):
        from pyscnomics.io import getattr as getattr_module

        assert getattr_module.__name__ == "pyscnomics.io.getattr"

    def test_import_api(self):
        from pyscnomics.api.main import app
        from pyscnomics.api.router import router
        from pyscnomics.api.adapter import get_grosssplit
        from pyscnomics.api.converter import Data

        _ = (app, router, get_grosssplit, Data)


# ── Sample data tests ──────────────────────────────────────────────


class TestSampleData:
    """Verify sample dataset loading and basic contract construction."""

    def test_load_testing_works(self):
        """Verify load_testing returns expected reference data."""
        from pyscnomics.dataset.sample import load_testing

        oil_revenue = load_testing(dataset_type="case1", key="oil_revenue")
        assert oil_revenue is not None
        assert not isinstance(oil_revenue, ValueError)
        assert len(oil_revenue) > 0

    def test_grosssplit_construct(self):
        """GrossSplit construct with required args."""
        from pyscnomics.contracts.gross_split import GrossSplit

        psc = GrossSplit(
            start_date=date(year=2023, month=1, day=1),
            end_date=date(year=2034, month=12, day=31),
            approval_year=2023,
        )
        assert psc is not None
        assert psc.start_date == date(2023, 1, 1)

    def test_costrecovery_construct(self):
        """CostRecovery with minimal required args."""
        from pyscnomics.contracts.costrecovery import CostRecovery

        psc = CostRecovery(
            start_date=date(year=2023, month=1, day=1),
            end_date=date(year=2034, month=12, day=31),
            approval_year=2023,
        )
        assert psc is not None


# ── Setup defaults ───────────────────────────────────────────────


class TestSetupDefaults:
    """Missing setup fields should default instead of raising KeyError."""

    def test_missing_approval_year_defaults_to_start_year(self):
        from pyscnomics.api.adapter import get_setup_dict

        result = get_setup_dict(
            data={
                "setup": {
                    "start_date": "01/01/2023",
                    "end_date": "31/12/2027",
                    "oil_onstream_date": None,
                    "gas_onstream_date": None,
                },
            }
        )
        assert result[2] == 2023

    def test_null_approval_year_defaults_to_start_year(self):
        from pyscnomics.api.adapter import get_setup_dict

        result = get_setup_dict(
            data={
                "setup": {
                    "start_date": "01/01/2023",
                    "end_date": "31/12/2027",
                    "approval_year": None,
                },
            }
        )
        assert result[2] == 2023


# ── Contract argument defaults ───────────────────────────────────


class TestContractArgumentDefaults:
    def test_dump_includes_reservoir_type_permen_2024(self):
        from pyscnomics.api.converter import ContractArgumentsBM

        assert (
            ContractArgumentsBM().model_dump()["reservoir_type_permen_2024"]
            == "conventional"
        )
        assert (
            ContractArgumentsBM(
                reservoir_type_permen_2024="unconventional"
            ).model_dump()["reservoir_type_permen_2024"]
            == "unconventional"
        )


# ── Gross Split schema ────────────────────────────────────────────


class TestGrossSplitSchema:
    def test_dump_preserves_permen_2024_fields(self):
        from pyscnomics.api.converter import GrossSplitBM

        dumped = GrossSplitBM(
            field_loc_2024="shallow_offshore",
            infra_avail_2024="available",
            field_reserves_2024="low",
        ).model_dump()

        assert dumped["field_loc_2024"] == "shallow_offshore"
        assert dumped["infra_avail_2024"] == "available"
        assert dumped["field_reserves_2024"] == "low"

    def test_frontend_shaped_table_payload_preserves_nullable_contract_fields(
        self, monkeypatch: _MonkeyPatch
    ) -> None:
        """Frontend null selectors and inactive CR block survive route validation."""
        import pyscnomics.api.router as router

        calls: dict[str, object] = {}

        def fake_get_contract_table(
            data: dict[str, object], contract_type: str
        ) -> tuple[dict[str, object], list[dict[str, object]]]:
            calls["data"] = data
            calls["contract_type"] = contract_type
            return {"consolidated": {}}, []

        monkeypatch.setattr(router, "get_contract_table", fake_get_contract_table)

        payload = {
            "setup": {
                "start_date": "01/01/2023",
                "end_date": "31/12/2027",
            },
            "summary_arguments": {"discount_rate_start_year": 2024},
            "contract_arguments": {"reservoir_type_permen_2024": "unconventional"},
            "costrecovery": None,
            "grosssplit": {
                "field_status": None,
                "field_loc": None,
                "res_depth": None,
                "infra_avail": None,
                "res_type": None,
                "api_oil": None,
                "domestic_use": None,
                "prod_stage": None,
                "co2_content": None,
                "h2s_content": None,
                "field_reserves": None,
            },
        }

        resp = client.post("/api/grosssplit/table", json=payload)

        assert resp.status_code == 200
        assert _response_body(resp) == {"consolidated": {}}
        assert calls["contract_type"] == "Gross Split"
        call_data = calls["data"]
        assert _is_record(call_data)
        assert call_data["costrecovery"] is None
        grosssplit = call_data["grosssplit"]
        assert _is_record(grosssplit)
        assert grosssplit["field_reserves"] is None
        contract_arguments = call_data["contract_arguments"]
        assert _is_record(contract_arguments)
        assert contract_arguments["reservoir_type_permen_2024"] == "unconventional"
        summary_arguments = call_data["summary_arguments"]
        assert _is_record(summary_arguments)
        assert summary_arguments["discount_rate_start_year"] == 2024


# ── Cost type schema ─────────────────────────────────────────────


class TestCostTypeSchema:
    def test_capital_null_pis_year_falls_back_to_expense_year(self):
        """A null PIS entry retains the engine's expense-year fallback."""
        from pyscnomics.api.converter import convert_dict_to_capital

        capital = convert_dict_to_capital(
            {
                "well": {
                    "start_year": 2023,
                    "end_year": 2027,
                    "expense_year": [2024],
                    "cost": [100.0],
                    "pis_year": [None],
                }
            }
        )

        assert capital is not None
        pis_year = capital[0].pis_year
        assert pis_year is not None
        assert pis_year.tolist() == [2024]

    def test_data_dump_preserves_cost_type(self):
        """HTTP cost classification must survive Pydantic model_dump."""
        from pyscnomics.api.converter import Data

        payload: dict[str, object] = {
            "setup": {
                "start_date": "01/01/2023",
                "end_date": "31/12/2027",
            },
            "summary_arguments": {},
            "contract_arguments": {},
            "capital": {
                "well": {
                    "start_year": 2023,
                    "end_year": 2027,
                    "expense_year": [2023],
                    "cost": [100.0],
                    "cost_allocation": ["Oil"],
                    "cost_type": ["sunk_cost"],
                }
            },
        }

        data = Data.model_validate(payload)

        assert data.capital is not None
        assert data.capital["well"].cost_type == ["sunk_cost"]

    def test_grosssplit_table_preserves_cost_type(
        self, monkeypatch: _MonkeyPatch
    ) -> None:
        """The real route hands the adapter the original cost_type array."""
        import pyscnomics.api.router as router

        calls: dict[str, object] = {}

        def fake_get_contract_table(
            data: dict[str, object], contract_type: str
        ) -> tuple[dict[str, object], list[dict[str, object]]]:
            calls["data"] = data
            calls["contract_type"] = contract_type
            return {"consolidated": {}}, []

        monkeypatch.setattr(router, "get_contract_table", fake_get_contract_table)

        payload = {
            "setup": {
                "start_date": "01/01/2023",
                "end_date": "31/12/2027",
            },
            "summary_arguments": {},
            "contract_arguments": {},
            "capital": {
                "well": {
                    "start_year": 2023,
                    "end_year": 2027,
                    "expense_year": [2023],
                    "cost": [100.0],
                    "cost_allocation": ["Oil"],
                    "cost_type": ["sunk_cost"],
                }
            },
        }

        resp = client.post("/api/grosssplit/table", json=payload)

        assert resp.status_code == 200
        assert calls["contract_type"] == "Gross Split"
        call_data = calls["data"]
        assert _is_record(call_data)
        capital = call_data["capital"]
        assert _is_record(capital)
        well = capital["well"]
        assert _is_record(well)
        assert well["cost_type"] == ["sunk_cost"]


# ── Setup schema ─────────────────────────────────────────────────


class TestSetupSchema:
    def test_data_dump_preserves_setup_fields(self):
        """Core setup flags must survive Pydantic model_dump."""
        from pyscnomics.api.converter import Data

        payload: dict[str, object] = {
            "setup": {
                "start_date": "01/01/2025",
                "end_date": "31/12/2030",
                "approval_year": 2026,
                "is_pod_1": False,
                "is_strict": True,
            },
            "summary_arguments": {},
            "contract_arguments": {},
        }

        setup = Data.model_validate(payload).setup

        assert setup.approval_year == 2026
        assert setup.is_pod_1 is False
        assert setup.is_strict is True


# ── Summary arguments schema ─────────────────────────────────────


class TestSummaryArgumentsSchema:
    def test_models_preserve_untyped_list_element_shapes(self):
        """Parameterizing former bare lists must not narrow validation."""
        from pyscnomics.api.converter import ContractArgumentsBM, SummaryArgumentsBM

        inflation_curve: list[object] = [{"year": 2025, "rate": 0.03}]
        summary = SummaryArgumentsBM(inflation_rate=inflation_curve).model_dump()
        contract = ContractArgumentsBM(
            effective_tax_rate=inflation_curve,
            vat_rate=inflation_curve,
            lbt_rate=inflation_curve,
            inflation_rate=inflation_curve,
            cum_production_split_offset=inflation_curve,
        ).model_dump()

        assert summary["inflation_rate"] == inflation_curve
        for field in (
            "effective_tax_rate",
            "vat_rate",
            "lbt_rate",
            "inflation_rate",
            "cum_production_split_offset",
        ):
            assert contract[field] == inflation_curve

    def test_data_dump_preserves_discount_rate_start_year(self):
        """The frontend's discount reference year must survive model_dump."""
        from pyscnomics.api.converter import Data

        payload: dict[str, object] = {
            "setup": {
                "start_date": "01/01/2025",
                "end_date": "31/12/2030",
            },
            "summary_arguments": {
                "discount_rate_start_year": 2024,
                "inflation_rate": 0.03,
                "discount_rate": 0.12,
                "npv_mode": "SKK Full Cycle Nominal Terms",
                "discounting_mode": "Mid Year",
                "profitability_discounted": False,
            },
            "contract_arguments": {},
        }

        summary_arguments = Data.model_validate(payload).summary_arguments

        assert summary_arguments.discount_rate_start_year == 2024


# ── Optimization schema ───────────────────────────────────────────


class TestOptimizationSchema:
    def test_optimization_dict_rejects_empty_selections(self):
        import pytest
        from pydantic import ValidationError

        from pyscnomics.api.converter import OptimizationDictBM

        with pytest.raises(ValidationError) as excinfo:
            _ = OptimizationDictBM(parameter=[], min=[], max=[])

        empty_locs = {
            error["loc"][0] for error in excinfo.value.errors() if error["loc"]
        }
        assert empty_locs == {"parameter", "min", "max"}

    def test_optimization_dict_accepts_nonempty_selections(self):
        from pyscnomics.api.converter import OptimizationDictBM

        dumped = OptimizationDictBM(
            parameter=["Oil Contractor Pre Tax"],
            min=[0.1],
            max=[0.2],
        ).model_dump()

        assert dumped == {
            "parameter": ["Oil Contractor Pre Tax"],
            "min": [0.1],
            "max": [0.2],
        }


# ── Transition uncertainty regression ─────────────────────────────


class TestTransitionUncertainty:
    """Regression: transition uncertainty must use DataTransition + "Transition"."""

    def test_datatransition_has_uncertainty_arguments(self):
        """DataTransition truthfully exposes its optional uncertainty section."""
        transition = DataTransition.model_validate(_transition_payload())

        assert transition.uncertainty_arguments is None
        assert _data_request().uncertainty_arguments is None

    def test_transition_uncertainty_delegates_transition(
        self, monkeypatch: _MonkeyPatch
    ) -> None:
        """Route dumps payload and delegates with contract_type="Transition"."""
        import pyscnomics.api.router as router
        from pyscnomics.api.converter import DataTransition
        from pyscnomics.api.router import calculate_transition_uncertainty

        calls: dict[str, object] = {}

        def fake_get_uncertainty(
            data: dict[str, object], contract_type: str
        ) -> dict[str, object]:
            calls["data"] = data
            calls["contract_type"] = contract_type
            return {"ok": True}

        monkeypatch.setattr(router, "get_uncertainty", fake_get_uncertainty)

        request = DataTransition.model_validate(_transition_payload())
        result = asyncio.run(calculate_transition_uncertainty(request))
        assert result == {"ok": True}
        assert calls["data"] == request.model_dump()
        assert calls["contract_type"] == "Transition"

    def test_transition_summary_adapts_discount_start_year(
        self, monkeypatch: _MonkeyPatch
    ) -> None:
        """Shared summary helper receives its canonical reference_year name."""
        import pyscnomics.api.adapter as adapter

        calls: dict[str, object] = {}

        @final
        class FakeTransition:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def run(self, **_kwargs: object) -> None:
                pass

        def fake_costrecovery(
            data: dict[str, object], summary_result: bool = True
        ) -> tuple[None, object, dict[str, object], None]:
            assert "costrecovery" in data or "grosssplit" in data
            assert summary_result is False
            return None, object(), {}, None

        def fake_summary(**kwargs: object) -> dict[str, object]:
            calls.update(kwargs)
            return {}

        def identity_summary(dict_object: Mapping[str, object]) -> dict[str, object]:
            return dict(dict_object)

        monkeypatch.setattr(adapter, "Transition", FakeTransition)
        monkeypatch.setattr(adapter, "get_costrecovery", fake_costrecovery)
        monkeypatch.setattr(adapter, "get_summary", fake_summary)
        monkeypatch.setattr(adapter, "convert_summary_to_dict", identity_summary)

        payload: dict[str, object] = {
            "contract_1": {"costrecovery": {}, "grosssplit": None},
            "contract_2": {"costrecovery": {}, "grosssplit": None},
            "contract_arguments": {"unrec_portion": 0.5},
            "summary_arguments": {
                "discount_rate_start_year": 2024,
                "inflation_rate": 0.03,
                "discount_rate": 0.1,
                "npv_mode": "SKK Full Cycle Nominal Terms",
                "discounting_mode": "End Year",
                "profitability_discounted": False,
            },
        }

        _ = adapter.get_transition(payload)

        assert calls["reference_year"] == 2024
        assert "discount_rate_start_year" not in calls

    @pytest.mark.parametrize(
        ("reference_year", "expected_reference_year"),
        [(2026, 2026), (None, 2022)],
    )
    def test_transition_summary_falls_back_to_reference_year(
        self,
        monkeypatch: _MonkeyPatch,
        reference_year: int | None,
        expected_reference_year: int,
    ) -> None:
        """A null discount start year falls back to the supplied reference year."""
        import pyscnomics.api.adapter as adapter

        calls: dict[str, object] = {}

        @final
        class FakeTransition:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def run(self, **_kwargs: object) -> None:
                pass

        def fake_costrecovery(
            data: dict[str, object], summary_result: bool = True
        ) -> tuple[None, object, dict[str, object], None]:
            assert "costrecovery" in data or "grosssplit" in data
            assert summary_result is False
            return None, object(), {}, None

        def fake_summary(**kwargs: object) -> dict[str, object]:
            calls.update(kwargs)
            return {}

        def identity_summary(dict_object: Mapping[str, object]) -> dict[str, object]:
            return dict(dict_object)

        monkeypatch.setattr(adapter, "Transition", FakeTransition)
        monkeypatch.setattr(adapter, "get_costrecovery", fake_costrecovery)
        monkeypatch.setattr(adapter, "get_summary", fake_summary)
        monkeypatch.setattr(adapter, "convert_summary_to_dict", identity_summary)

        summary_arguments: dict[str, object] = {
            "discount_rate_start_year": None,
            "inflation_rate": 0.03,
            "discount_rate": 0.1,
            "npv_mode": "SKK Full Cycle Nominal Terms",
            "discounting_mode": "End Year",
            "profitability_discounted": False,
        }
        if reference_year is not None:
            summary_arguments["reference_year"] = reference_year

        payload: dict[str, object] = {
            "contract_1": {"costrecovery": {}, "grosssplit": None},
            "contract_2": {"costrecovery": {}, "grosssplit": None},
            "contract_arguments": {"unrec_portion": 0.5},
            "summary_arguments": summary_arguments,
        }

        _, _, _, actual_summary_arguments = adapter.get_transition(payload)

        assert calls["reference_year"] == expected_reference_year
        assert actual_summary_arguments["reference_year"] == expected_reference_year
        assert actual_summary_arguments["discount_rate_start_year"] is None

    def test_transition_detailed_summary_maps_external_helper_arguments(
        self, monkeypatch: _MonkeyPatch
    ) -> None:
        """Detailed-summary delegation maps API years to the legacy helper signature."""
        import pyscnomics.api.adapter as adapter

        calls: dict[str, object] = {}
        transition = object()
        summary_arguments: dict[str, object] = {
            "reference_year": 2027,
            "discount_rate_start_year": None,
            "inflation_rate": 0.03,
            "discount_rate": 0.1,
            "npv_mode": "SKK Full Cycle Nominal Terms",
            "discounting_mode": "End Year",
            "profitability_discounted": False,
        }

        def fake_get_transition(
            data: dict[str, object], summary_result: bool = True
        ) -> tuple[dict[str, object], object, dict[str, object], dict[str, object]]:
            assert data == {}
            assert summary_result is True
            return {}, transition, {}, summary_arguments

        monkeypatch.setattr(adapter, "get_transition", fake_get_transition)

        def fake_summary(**kwargs: object) -> dict[str, float]:
            calls.update(kwargs)
            return {"ctr_npv": 123.0, "gov_take": 45.0}

        monkeypatch.setattr(adapter, "get_summary", fake_summary)

        result = adapter.get_detailed_summary({}, "Transition")

        assert result == {"ctr_npv": 123.0, "gov_take": 45.0}
        assert calls["contract"] is transition
        assert calls["reference_year"] == 2027
        assert "discount_rate_start_year" not in calls


class TestAdapterExecutionSemantics:
    def test_costrecovery_optimization_builds_each_baseline_input(
        self, monkeypatch: _MonkeyPatch
    ) -> None:
        """Optimization preserves the established three adapter executions."""
        import pyscnomics.api.adapter as adapter

        calls = 0
        optimizer_calls: dict[str, object] = {}

        @final
        class FakeCapital:
            useful_life: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([5.0])

        @final
        class FakeContract:
            capital_cost_total: FakeCapital = FakeCapital()

        contract = FakeContract()

        def fake_costrecovery(
            data: dict[str, object], summary_result: bool = True
        ) -> tuple[
            dict[str, object], FakeContract, dict[str, object], dict[str, object]
        ]:
            nonlocal calls
            assert "optimization_arguments" in data
            assert summary_result is True
            calls += 1
            return (
                {},
                contract,
                {"run": calls},
                {
                    "reference_year": 2022,
                    "discount_rate_start_year": None,
                    "inflation_rate": 0.03,
                    "discount_rate": calls,
                    "npv_mode": "mode",
                    "discounting_mode": "discounting",
                    "profitability_discounted": False,
                },
            )

        def fake_optimize(
            **kwargs: object,
        ) -> tuple[list[str], list[float], float, list[FakeContract]]:
            optimizer_calls.update(kwargs)
            return ["Oil Contractor Pre Tax"], [0.2], 1.0, [contract]

        monkeypatch.setattr(adapter, "get_costrecovery", fake_costrecovery)
        monkeypatch.setattr(adapter, "optimize_psc", fake_optimize)

        _ = adapter.get_contract_optimization(
            {
                "optimization_arguments": {
                    "dict_optimization": {
                        "parameter": ["Oil Contractor Pre Tax"],
                        "min": [0.1],
                        "max": [0.2],
                    },
                    "target_optimization": 1.0,
                    "target_parameter": "NPV",
                }
            },
            "Cost Recovery",
        )

        assert calls == 3
        assert optimizer_calls["contract_arguments"] == {"run": 2}
        summary_arguments = optimizer_calls["summary_arguments"]
        assert _is_record(summary_arguments)
        assert summary_arguments["discount_rate"] == 3
        assert "reference_year" not in summary_arguments


# ── Warnings endpoints ────────────────────────────────────────────


class TestWarningsEndpoints:
    """Canonical warnings endpoints reuse adapters and expose contract warnings."""

    def test_warnings_route_returns_empty_without_messages(
        self, monkeypatch: _MonkeyPatch
    ) -> None:
        """No warning_messages attribute/None -> []."""
        import pyscnomics.api.router as router

        def fake_adapter(
            data: dict[str, object], summary_result: bool = True
        ) -> tuple[None, object, dict[str, object], None]:
            assert "setup" in data
            assert summary_result is True
            return None, object(), {}, None

        monkeypatch.setattr(router, "get_baseproject", fake_adapter)

        assert asyncio.run(router.get_baseproject_warnings(_data_request())) == []

    def test_warnings_route_returns_records(self, monkeypatch: _MonkeyPatch) -> None:
        """warning_messages.to_dict(orient="records") is returned verbatim."""
        import pyscnomics.api.router as router

        @final
        class Messages:
            def to_dict(self, *, orient: str) -> list[dict[str, object]]:
                assert orient == "records"
                return [{"No.": 1, "Category": "BASE_PROJECT", "Message": "warn"}]

        @final
        class FakeContract:
            warning_messages: Messages = Messages()

        def fake_adapter(
            data: dict[str, object], summary_result: bool = True
        ) -> tuple[None, FakeContract, dict[str, object], None]:
            assert "setup" in data
            assert summary_result is True
            return None, FakeContract(), {}, None

        monkeypatch.setattr(router, "get_baseproject", fake_adapter)
        assert asyncio.run(router.get_baseproject_warnings(_data_request())) == [
            {"No.": 1, "Category": "BASE_PROJECT", "Message": "warn"}
        ]

    def test_warnings_route_returns_list_verbatim(
        self, monkeypatch: _MonkeyPatch
    ) -> None:
        """List-backed dict warning messages are returned unchanged."""
        import pyscnomics.api.router as router

        messages: list[dict[str, object]] = [
            {"No.": 1, "Category": "BASE_PROJECT", "Message": "warn"}
        ]

        @final
        class FakeContract:
            warning_messages: list[dict[str, object]] = messages

        def fake_adapter(
            data: dict[str, object], summary_result: bool = True
        ) -> tuple[None, FakeContract, dict[str, object], None]:
            assert "setup" in data
            assert summary_result is True
            return None, FakeContract(), {}, None

        monkeypatch.setattr(router, "get_baseproject", fake_adapter)
        assert asyncio.run(router.get_baseproject_warnings(_data_request())) == messages

    def test_warnings_route_normalizes_tuple_messages(
        self, monkeypatch: _MonkeyPatch
    ) -> None:
        """Engine tuple warnings become Message dicts for the frontend."""
        import pyscnomics.api.router as router

        @final
        class FakeContract:
            warning_messages: list[tuple[str, str]] = [
                ("Gross Split", "First warning"),
                ("Gross Split", "Second warning"),
            ]

        def fake_adapter(
            data: dict[str, object], summary_result: bool = True
        ) -> tuple[None, FakeContract, dict[str, object], None]:
            assert "setup" in data
            assert summary_result is True
            return None, FakeContract(), {}, None

        monkeypatch.setattr(router, "get_baseproject", fake_adapter)
        assert asyncio.run(router.get_baseproject_warnings(_data_request())) == [
            {"Message": "First warning"},
            {"Message": "Second warning"},
        ]

    def test_warnings_routes_delegate_to_adapters(
        self, monkeypatch: _MonkeyPatch
    ) -> None:
        """Each warnings route calls its adapter with the dumped payload and returns records."""
        import pyscnomics.api.router as router

        @final
        class Messages:
            def to_dict(self, *, orient: str) -> list[dict[str, object]]:
                assert orient == "records"
                return [{"ok": True}]

        @final
        class FakeContract:
            warning_messages: Messages = Messages()

        calls: dict[str, object] = {}

        def fake_adapter(
            data: dict[str, object], _summary_result: bool = True
        ) -> tuple[None, FakeContract, dict[str, object], None]:
            calls["data"] = data
            return None, FakeContract(), {}, None

        def assert_route(
            route: _WarningRoute,
        ) -> None:
            result = asyncio.run(route(_data_request()))
            assert result == [{"ok": True}]
            assert calls["data"] == _data_request().model_dump()

        for target_name, route in (
            ("get_baseproject", router.get_baseproject_warnings),
            ("get_costrecovery", router.get_costrecovery_warnings),
            ("get_grosssplit", router.get_grosssplit_warnings),
        ):
            monkeypatch.setattr(router, target_name, fake_adapter)
            assert_route(route)

        def fake_transition_adapter(
            data: dict[str, object], _summary_result: bool = True
        ) -> tuple[None, FakeContract, dict[str, object], None]:
            calls["data"] = data
            return None, FakeContract(), {}, None

        monkeypatch.setattr(router, "get_transition", fake_transition_adapter)
        transition_result = asyncio.run(
            router.get_transition_warnings(
                DataTransition.model_validate(_transition_payload())
            )
        )
        assert transition_result == [{"ok": True}]
        assert (
            calls["data"]
            == DataTransition.model_validate(_transition_payload()).model_dump()
        )

    def test_warnings_paths_in_openapi(self):
        """All four warnings paths are registered."""
        schema = _response_body(client.get("/openapi.json"))
        paths = schema["paths"]
        assert _is_record(paths)
        for path in [
            "/api/baseproject/warnings",
            "/api/costrecovery/warnings",
            "/api/grosssplit/warnings",
            "/api/transition/warnings",
        ]:
            assert path in paths


# ── Canonical table routes ────────────────────────────────────────


class TestTableRoutes:
    """Canonical table routes return only the table dict, not warnings."""

    def test_table_routes_return_table_dict(self, monkeypatch: _MonkeyPatch) -> None:
        """Each table route returns table dict and delegates with its contract type."""
        import pyscnomics.api.router as router

        table_dict: dict[str, object] = {"consolidated": {}}
        calls: dict[str, object] = {}

        def fake_get_contract_table(
            data: dict[str, object], contract_type: str
        ) -> tuple[dict[str, object], list[object]]:
            calls["data"] = data
            calls["contract_type"] = contract_type
            return table_dict, []

        monkeypatch.setattr(router, "get_contract_table", fake_get_contract_table)

        for route, contract_type in (
            (router.get_costrecovery_table, "Cost Recovery"),
            (router.get_grosssplit_table, "Gross Split"),
            (router.get_baseproject_table, "Base Project"),
        ):
            table_route: _TableRoute = route
            result = asyncio.run(table_route(_data_request()))
            assert result == table_dict
            assert calls["data"] == _data_request().model_dump()
            assert calls["contract_type"] == contract_type

        transition_result = asyncio.run(
            router.get_transition_table(
                DataTransition.model_validate(_transition_payload())
            )
        )
        assert transition_result == table_dict
        assert (
            calls["data"]
            == DataTransition.model_validate(_transition_payload()).model_dump()
        )
        assert calls["contract_type"] == "Transition"

    def test_grosssplit_table_accepts_nullable_capital_arrays(
        self, monkeypatch: _MonkeyPatch
    ) -> None:
        """CapitalCost arrays with JSON null elements pass request validation."""
        import pyscnomics.api.router as router

        calls: dict[str, object] = {}

        def fake_get_contract_table(
            data: dict[str, object], contract_type: str
        ) -> tuple[dict[str, object], list[object]]:
            calls["data"] = data
            calls["contract_type"] = contract_type
            return {"consolidated": {}}, []

        monkeypatch.setattr(router, "get_contract_table", fake_get_contract_table)

        payload: dict[str, object] = {
            "setup": {
                "start_date": "01/01/2023",
                "end_date": "31/12/2027",
            },
            "summary_arguments": {},
            "contract_arguments": {},
            "capital": {
                "well": {
                    "start_year": 2023,
                    "end_year": 2027,
                    "expense_year": [2023, 2024, 2025, 2026, 2027],
                    "cost": [1000, 1000, 1000, 1000, 1000],
                    "pis_year": [2023, 2024, None, 2026, 2027],
                    "salvage_value": [0, 0, None, 0, 0],
                    "useful_life": [5, 5, None, 5, 5],
                    "depreciation_factor": [0.5, 0.5, None, 0.5, 0.5],
                    "is_ic_applied": [True, False, None, False, True],
                }
            },
        }

        resp = client.post("/api/grosssplit/table", json=payload)
        assert resp.status_code == 200
        assert _response_body(resp) == {"consolidated": {}}
        assert calls["contract_type"] == "Gross Split"


# ── Combine / incremental endpoints ──────────────────────────────


class TestCombineIncrement:
    """Canonical /api/combine and /api/increment reuse core Summary behavior."""

    def test_request_model_shape(self):
        """Direct frontend JSON maps to cases + options with string enums."""
        from pyscnomics.api.converter import CasesRequestBM

        request = CasesRequestBM.model_validate(
            {
                "cases": [
                    {"cId": 11, "ctr": 1, "data": {"setup": {}}},
                    {"cId": 12, "ctr": 2, "data": {"setup": {}}},
                ],
                "options": {
                    "reference_year": 2023,
                    "inflation_rate": 0.05,
                    "discount_rate": 0.1,
                    "npv_mode": "SKK Full Cycle Nominal Terms",
                    "discounting_mode": "End Year",
                    "profitability_discounted": True,
                },
            }
        )
        assert [case.cId for case in request.cases] == [11, 12]
        assert [case.ctr for case in request.cases] == [1, 2]
        assert request.options.npv_mode == "SKK Full Cycle Nominal Terms"

    def test_routes_delegate_to_adapters(self, monkeypatch: _MonkeyPatch) -> None:
        """Routes dump the validated payload and call the combine/increment adapters."""
        import pyscnomics.api.router as router

        calls: dict[str, object] = {}

        def fake_combine(payload: dict[str, object]) -> dict[str, object]:
            calls["combine"] = payload
            return {"combine": []}

        def fake_increment(payload: dict[str, object]) -> dict[str, object]:
            calls["increment"] = payload
            return {"increment": []}

        monkeypatch.setattr(router, "get_combine_summary", fake_combine)
        monkeypatch.setattr(router, "get_increment_summary", fake_increment)

        request = _cases_request()
        assert asyncio.run(router.calculate_combine(request)) == {"combine": []}
        assert asyncio.run(router.calculate_increment(request)) == {"increment": []}
        assert calls["combine"] == request.model_dump()
        assert calls["increment"] == request.model_dump()

    def test_merged_result_keeps_case_order_and_appends_derived_row(
        self, monkeypatch: _MonkeyPatch
    ) -> None:
        """Per-case rows keep input order; cId=0 derived row is appended last."""
        import pyscnomics.api.adapter as adapter

        contracts: list[tuple[dict[str, object], int]] = []
        summaries: dict[str, object] = {"a": 1, "b": 2}

        def fake_run(data: dict[str, object], ctr: int) -> object:
            contracts.append((data, ctr))
            return object()

        def fake_case_summary(data: dict[str, object], ctr: int) -> dict[str, object]:
            assert data == {"setup": {}}
            return {"case": ctr}

        def fake_table(
            contract: object,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            assert contract is not None
            frame = pd.DataFrame({"years": [2023, 2024], "cashflow": [10.0, 20.0]})
            return frame, frame, frame

        @final
        class FakeSummary:
            kwargs: dict[str, object]

            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def case_combine(
                self,
            ) -> dict[str, np.ndarray[tuple[int], np.dtype[np.int64]]]:
                return {"years": np.array([2023, 2024]), "cashflow": np.array([1, 2])}

            def get_executive_summary(self) -> dict[str, object]:
                return summaries

        monkeypatch.setattr(adapter, "_run_case_contract", fake_run)
        monkeypatch.setattr(adapter, "_case_summary", fake_case_summary)
        monkeypatch.setattr(adapter, "get_table", fake_table)
        monkeypatch.setattr(adapter, "Summary", FakeSummary)

        payload: dict[str, object] = {
            "cases": [
                {"cId": 5, "ctr": 1, "data": {"setup": {}}},
                {"cId": 9, "ctr": 2, "data": {"setup": {}}},
            ],
            "options": {
                "reference_year": 2023,
                "inflation_rate": 0.0,
                "discount_rate": 0.1,
                "npv_mode": "SKK Full Cycle Nominal Terms",
                "discounting_mode": "End Year",
                "profitability_discounted": True,
            },
        }
        combined = adapter.get_combine_summary(payload)["combine"]
        assert _is_record_list(combined)
        result = combined

        assert [row["cId"] for row in result] == [5, 9, 0]
        assert [row["ctr"] for row in result] == [1, 2, -1]
        assert result[0]["summary"] == {"case": 1}
        table_consolidated = result[-1]["table_consolidated"]
        assert isinstance(table_consolidated, dict)
        assert table_consolidated["years"] == {2023: 2023, 2024: 2024}
        assert table_consolidated["cashflow"] == {2023: 1, 2024: 2}
        assert result[-1]["summary"] == summaries

    def test_increment_uses_case_incremental(self, monkeypatch: _MonkeyPatch) -> None:
        """Increment mode calls Summary.case_incremental, not case_combine."""
        import pyscnomics.api.adapter as adapter

        calls: dict[str, object] = {}

        def fake_run(data: dict[str, object], ctr: int) -> object:
            assert data == {"setup": {}}
            assert ctr == 1
            return object()

        def fake_table(
            contract: object,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            assert contract is not None
            frame = pd.DataFrame({"years": [2023], "cashflow": [10.0]})
            return frame, frame, frame

        @final
        class FakeSummary:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def case_incremental(
                self,
            ) -> dict[str, np.ndarray[tuple[int], np.dtype[np.int64]]]:
                calls["mode"] = "incremental"
                return {"years": np.array([2023]), "cashflow": np.array([5])}

            def get_executive_summary(self) -> dict[str, object]:
                return {}

        def fake_case_summary(data: dict[str, object], ctr: int) -> dict[str, object]:
            assert data == {"setup": {}}
            assert ctr == 1
            return {}

        monkeypatch.setattr(adapter, "_run_case_contract", fake_run)
        monkeypatch.setattr(adapter, "_case_summary", fake_case_summary)
        monkeypatch.setattr(adapter, "get_table", fake_table)
        monkeypatch.setattr(adapter, "Summary", FakeSummary)

        payload: dict[str, object] = {
            "cases": [{"cId": 1, "ctr": 1, "data": {"setup": {}}}],
            "options": {
                "reference_year": 2023,
                "inflation_rate": 0.0,
                "discount_rate": 0.1,
                "npv_mode": "SKK Full Cycle Nominal Terms",
                "discounting_mode": "End Year",
                "profitability_discounted": True,
            },
        }
        increment = adapter.get_increment_summary(payload)["increment"]
        assert _is_record_list(increment)
        result = increment
        assert calls["mode"] == "incremental"
        assert result[-1]["cId"] == 0

    def test_transition_case_uses_second_consolidated_frame(
        self, monkeypatch: _MonkeyPatch
    ) -> None:
        """Transition get_table returns a TransitionTable; use second consolidated."""
        import pyscnomics.api.adapter as adapter
        from pyscnomics.contracts.transition import Transition
        from pyscnomics.dataset.sample import load_data

        def fake_run(data: dict[str, object], ctr: int) -> Transition:
            assert data == {"setup": {}}
            assert ctr == 3
            return Transition(
                contract1=load_data("case1", "cost_recovery"),
                contract2=load_data("case3", "gross_split"),
                argument_contract1={},
                argument_contract2={},
            )

        def fake_table(
            contract: object,
        ) -> tuple[
            tuple[pd.DataFrame, pd.DataFrame],
            tuple[pd.DataFrame, pd.DataFrame],
            tuple[pd.DataFrame, pd.DataFrame],
        ]:
            assert contract is not None
            frame = pd.DataFrame({"years": [2023], "cashflow": [1.0]})
            # current transition shape: (oil pair, gas pair, consolidated pair)
            return (frame, frame), (frame, frame), (frame, frame)

        @final
        class FakeSummary:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def case_combine(
                self,
            ) -> dict[str, np.ndarray[tuple[int], np.dtype[np.int64]]]:
                return {"years": np.array([2023]), "cashflow": np.array([1])}

            def get_executive_summary(self) -> dict[str, object]:
                return {}

        def fake_case_summary(data: dict[str, object], ctr: int) -> dict[str, object]:
            assert data == {"setup": {}}
            assert ctr == 3
            return {}

        monkeypatch.setattr(adapter, "_run_case_contract", fake_run)
        monkeypatch.setattr(adapter, "_case_summary", fake_case_summary)
        monkeypatch.setattr(adapter, "get_table", fake_table)
        monkeypatch.setattr(adapter, "Summary", FakeSummary)

        payload: dict[str, object] = {
            "cases": [{"cId": 3, "ctr": 3, "data": {"setup": {}}}],
            "options": {
                "reference_year": 2023,
                "inflation_rate": 0.0,
                "discount_rate": 0.1,
                "npv_mode": "SKK Full Cycle Nominal Terms",
                "discounting_mode": "End Year",
                "profitability_discounted": True,
            },
        }
        combined = adapter.get_combine_summary(payload)["combine"]
        assert _is_record_list(combined)
        result = combined
        table_consolidated = result[0]["table_consolidated"]
        assert isinstance(table_consolidated, dict)
        assert table_consolidated["cashflow"] == {2023: 1.0}


class TestTransitionTableRealEndpoint:
    """Real Transition table path through adapter and router (no monkeypatch)."""

    def test_get_contract_table_transition_real_payload(self) -> None:
        import pyscnomics.api.adapter as adapter

        payload = _transition_payload()
        table_all, warnings = adapter.get_contract_table(payload, "Transition")
        assert set(table_all) >= {"contract_1", "contract_2", "execution_info"}
        contract_1 = table_all["contract_1"]
        contract_2 = table_all["contract_2"]
        assert _is_record(contract_1)
        assert _is_record(contract_2)
        assert set(contract_1) == {"oil", "gas", "consolidated"}
        assert set(contract_2) == {"oil", "gas", "consolidated"}
        for side in ("contract_1", "contract_2"):
            for kind in ("oil", "gas", "consolidated"):
                side_value = table_all[side]
                assert _is_record(side_value)
                kind_value = side_value[kind]
                assert _is_record(kind_value)
                assert len(kind_value) >= 1
        # first child is Cost Recovery, second is Gross Split
        first_consolidated = contract_1["consolidated"]
        second_consolidated = contract_2["consolidated"]
        assert _is_record(first_consolidated)
        assert _is_record(second_consolidated)
        assert "cashflow" in first_consolidated
        assert "ctr_cashflow" in second_consolidated
        assert isinstance(warnings, list)

    def test_router_transition_table_real_payload(self) -> None:
        import pyscnomics.api.adapter as adapter

        resp = client.post("/api/transition/table", json=_transition_payload())
        assert resp.status_code == 200
        body = _response_body(resp)
        assert set(body) >= {"contract_1", "contract_2", "execution_info"}

        adapter_table_raw, _ = adapter.get_contract_table(
            _transition_payload(), "Transition"
        )
        adapter_contract_1 = adapter_table_raw["contract_1"]
        adapter_contract_2 = adapter_table_raw["contract_2"]
        assert _is_mapping(adapter_contract_1)
        assert _is_mapping(adapter_contract_2)
        adapter_table: dict[str, Mapping[object, object]] = {
            "contract_1": adapter_contract_1,
            "contract_2": adapter_contract_2,
        }
        for side in ("contract_1", "contract_2"):
            side_body = body[side]
            assert _is_record(side_body)
            assert set(side_body) == {"oil", "gas", "consolidated"}
            side_kinds = side_body
            for kind in ("oil", "gas", "consolidated"):
                kind_body_value = side_kinds[kind]
                assert _is_record(kind_body_value)
                kind_body = kind_body_value
                adapter_kind_value = adapter_table[side][kind]
                assert _is_mapping(adapter_kind_value)
                adapter_kind = adapter_kind_value
                assert set(kind_body) == set(adapter_kind)
                for column in adapter_kind:
                    assert isinstance(column, str)
                    adapter_values_value = adapter_kind[column]
                    assert _is_mapping(adapter_values_value)
                    body_keys = kind_body[column]
                    assert _is_record(body_keys)
                    assert set(body_keys) == {str(key) for key in adapter_values_value}
            consolidated_body_value = side_kinds["consolidated"]
            adapter_consolidated_value = adapter_table[side]["consolidated"]
            assert _is_record(consolidated_body_value)
            assert _is_mapping(adapter_consolidated_value)
            consolidated_body = consolidated_body_value
            adapter_consolidated = adapter_consolidated_value
            representative_key = "cashflow" if side == "contract_1" else "ctr_cashflow"
            representative_value = consolidated_body[representative_key]
            direct_representative_value = adapter_consolidated[representative_key]
            assert _is_record(representative_value)
            assert _is_mapping(direct_representative_value)
            representative = representative_value
            direct_representative = direct_representative_value
            for year, expected in direct_representative.items():
                actual = representative[str(year)]
                if isinstance(expected, float) and math.isnan(expected):
                    assert isinstance(actual, float)
                    assert math.isnan(actual)
                else:
                    actual_value = actual
                    assert actual_value == expected
