"""
API smoke tests — verify core endpoints and import chains work after refactoring.
Run: uv run pytest tests/test_api.py -v
"""

from datetime import date
from fastapi.testclient import TestClient
import pytest

from pyscnomics.api.main import app


client = TestClient(app)


# ── GET endpoints ──────────────────────────────────────────────────

class TestGetEndpoints:
    def test_root_returns_version(self):
        resp = client.get("/api/")
        assert resp.status_code == 200
        body = resp.json()
        assert "Pyscnomics" in body

    def test_health_endpoint(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_version_endpoint(self):
        from pyscnomics import __version__
        resp = client.get("/api/version")
        assert resp.status_code == 200
        assert resp.json() == {"version": __version__}

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
        schema = resp.json()
        assert "paths" in schema
        assert "/api/" in schema["paths"]
        assert "/api/costrecovery" in schema["paths"]
        assert "/api/grosssplit" in schema["paths"]


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

        _ = getattr_module

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
        oil_revenue = load_testing(dataset_type='case1', key='oil_revenue')
        assert oil_revenue is not None
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
        self, monkeypatch
    ):
        """Frontend null selectors and inactive CR block survive route validation."""
        import pyscnomics.api.router as router

        calls = {}

        def fake_get_contract_table(data=None, contract_type=None):
            calls["data"] = data
            calls["contract_type"] = contract_type
            return {"consolidated": {}}, object()

        monkeypatch.setattr(router, "get_contract_table", fake_get_contract_table)

        payload = {
            "setup": {
                "start_date": "01/01/2023",
                "end_date": "31/12/2027",
            },
            "summary_arguments": {"discount_rate_start_year": 2024},
            "contract_arguments": {
                "reservoir_type_permen_2024": "unconventional"
            },
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
        assert resp.json() == {"consolidated": {}}
        assert calls["contract_type"] == "Gross Split"
        assert calls["data"]["costrecovery"] is None
        assert calls["data"]["grosssplit"]["field_reserves"] is None
        assert (
            calls["data"]["contract_arguments"]["reservoir_type_permen_2024"]
            == "unconventional"
        )
        assert calls["data"]["summary_arguments"]["discount_rate_start_year"] == 2024


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
        assert capital[0].pis_year.tolist() == [2024]

    def test_data_dump_preserves_cost_type(self):
        """HTTP cost classification must survive Pydantic model_dump."""
        from typing import Any

        from pyscnomics.api.converter import Data

        payload: dict[str, Any] = {
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

        dumped = Data(**payload).model_dump()

        assert dumped["capital"]["well"]["cost_type"] == ["sunk_cost"]

    def test_grosssplit_table_preserves_cost_type(self, monkeypatch):
        """The real route hands the adapter the original cost_type array."""
        import pyscnomics.api.router as router

        calls = {}

        def fake_get_contract_table(data=None, contract_type=None):
            calls["data"] = data
            calls["contract_type"] = contract_type
            return {"consolidated": {}}, object()

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
        assert calls["data"]["capital"]["well"]["cost_type"] == ["sunk_cost"]


# ── Setup schema ─────────────────────────────────────────────────

class TestSetupSchema:
    def test_data_dump_preserves_setup_fields(self):
        """Core setup flags must survive Pydantic model_dump."""
        from typing import Any

        from pyscnomics.api.converter import Data

        payload: dict[str, Any] = {
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

        dumped = Data(**payload).model_dump()["setup"]

        assert dumped["approval_year"] == 2026
        assert dumped["is_pod_1"] is False
        assert dumped["is_strict"] is True


# ── Summary arguments schema ─────────────────────────────────────

class TestSummaryArgumentsSchema:
    def test_models_preserve_untyped_list_element_shapes(self):
        """Parameterizing former bare lists must not narrow validation."""
        from pyscnomics.api.converter import ContractArgumentsBM, SummaryArgumentsBM

        inflation_curve = [{"year": 2025, "rate": 0.03}]
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
        from typing import Any

        from pyscnomics.api.converter import Data

        payload: dict[str, Any] = {
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

        dumped = Data(**payload).model_dump()["summary_arguments"]

        assert dumped["discount_rate_start_year"] == 2024


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
        from pydantic import TypeAdapter

        from pyscnomics.api.converter import Data, DataTransition, UncertaintyBM

        assert "uncertainty_arguments" in DataTransition.model_fields
        field = DataTransition.model_fields["uncertainty_arguments"]
        assert field.annotation == UncertaintyBM | None
        assert field.default is None
        assert TypeAdapter(field.annotation).validate_python(None) is None
        assert "uncertainty_arguments" in Data.model_fields

    def test_transition_uncertainty_route_signature(self):
        """Route accepts DataTransition, not Data."""
        import inspect

        from pyscnomics.api.converter import DataTransition
        from pyscnomics.api.router import calculate_transition_uncertainty

        sig = inspect.signature(calculate_transition_uncertainty)
        assert sig.parameters["data"].annotation is DataTransition

    def test_transition_uncertainty_delegates_transition(self, monkeypatch):
        """Route dumps payload and delegates with contract_type="Transition"."""
        import asyncio
        from typing import Any, cast

        import pyscnomics.api.router as router
        from pyscnomics.api.router import calculate_transition_uncertainty

        calls = {}

        def fake_get_uncertainty(data=None, contract_type=None):
            calls["data"] = data
            calls["contract_type"] = contract_type
            return {"ok": True}

        monkeypatch.setattr(router, "get_uncertainty", fake_get_uncertainty)

        class Fake:
            def model_dump(self):
                return {"dumped": True}

        result = asyncio.run(calculate_transition_uncertainty(cast(Any, Fake())))
        assert result == {"ok": True}
        assert calls["data"] == {"dumped": True}
        assert calls["contract_type"] == "Transition"

    def test_transition_summary_adapts_discount_start_year(self, monkeypatch):
        """Shared summary helper receives its canonical reference_year name."""
        import pyscnomics.api.adapter as adapter

        calls = {}

        class FakeTransition:
            def __init__(self, **_kwargs):
                pass

            def run(self, **_kwargs):
                pass

        def fake_costrecovery(data=None, summary_result=True):
            return None, object(), {}, None

        def fake_summary(**kwargs):
            calls.update(kwargs)
            return {}

        monkeypatch.setattr(adapter, "Transition", FakeTransition)
        monkeypatch.setattr(adapter, "get_costrecovery", fake_costrecovery)
        monkeypatch.setattr(adapter, "get_summary", fake_summary)
        monkeypatch.setattr(
            adapter, "convert_summary_to_dict", lambda dict_object: dict_object
        )

        payload = {
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

        adapter.get_transition(payload)

        assert calls["reference_year"] == 2024
        assert "discount_rate_start_year" not in calls

    @pytest.mark.parametrize(
        ("reference_year", "expected_reference_year"),
        [(2026, 2026), (None, 2022)],
    )
    def test_transition_summary_falls_back_to_reference_year(
        self, monkeypatch, reference_year, expected_reference_year
    ):
        """A null discount start year falls back to the supplied reference year."""
        import pyscnomics.api.adapter as adapter

        calls = {}

        class FakeTransition:
            def __init__(self, **_kwargs):
                pass

            def run(self, **_kwargs):
                pass

        def fake_costrecovery(data=None, summary_result=True):
            return None, object(), {}, None

        def fake_summary(**kwargs):
            calls.update(kwargs)
            return {}

        monkeypatch.setattr(adapter, "Transition", FakeTransition)
        monkeypatch.setattr(adapter, "get_costrecovery", fake_costrecovery)
        monkeypatch.setattr(adapter, "get_summary", fake_summary)
        monkeypatch.setattr(
            adapter, "convert_summary_to_dict", lambda dict_object: dict_object
        )

        summary_arguments = {
            "discount_rate_start_year": None,
            "inflation_rate": 0.03,
            "discount_rate": 0.1,
            "npv_mode": "SKK Full Cycle Nominal Terms",
            "discounting_mode": "End Year",
            "profitability_discounted": False,
        }
        if reference_year is not None:
            summary_arguments["reference_year"] = reference_year

        payload = {
            "contract_1": {"costrecovery": {}, "grosssplit": None},
            "contract_2": {"costrecovery": {}, "grosssplit": None},
            "contract_arguments": {"unrec_portion": 0.5},
            "summary_arguments": summary_arguments,
        }

        _, _, _, summary_arguments = adapter.get_transition(payload)

        assert calls["reference_year"] == expected_reference_year
        assert summary_arguments["reference_year"] == expected_reference_year
        assert summary_arguments["discount_rate_start_year"] is None

    def test_transition_detailed_summary_maps_external_helper_arguments(
        self, monkeypatch
    ):
        """Detailed-summary delegation maps API years to the legacy helper signature."""
        import pyscnomics.api.adapter as adapter

        calls = {}
        transition = object()
        summary_arguments = {
            "reference_year": 2027,
            "discount_rate_start_year": None,
            "inflation_rate": 0.03,
            "discount_rate": 0.1,
            "npv_mode": "SKK Full Cycle Nominal Terms",
            "discounting_mode": "End Year",
            "profitability_discounted": False,
        }

        monkeypatch.setattr(
            adapter,
            "get_transition",
            lambda data: ({}, transition, {}, summary_arguments),
        )

        def fake_summary(**kwargs):
            calls.update(kwargs)
            return {"ctr_npv": 123.0, "gov_take": 45.0}

        monkeypatch.setattr(adapter, "get_summary", fake_summary)

        result = adapter.get_detailed_summary({}, "Transition")

        assert result == {"ctr_npv": 123.0, "gov_take": 45.0}
        assert calls["contract"] is transition
        assert calls["reference_year"] == 2027
        assert "discount_rate_start_year" not in calls


class TestAdapterExecutionSemantics:
    def test_costrecovery_optimization_builds_each_baseline_input(self, monkeypatch):
        """Optimization preserves the established three adapter executions."""
        from types import SimpleNamespace

        import numpy as np

        import pyscnomics.api.adapter as adapter

        calls = 0
        optimizer_calls = {}
        capital = SimpleNamespace(useful_life=np.array([5.0]))
        contract = SimpleNamespace(capital_cost_total=capital)

        def fake_costrecovery(data):
            nonlocal calls
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

        def fake_optimize(**kwargs):
            optimizer_calls.update(kwargs)
            return ["Oil Contractor Pre Tax"], [0.2], 1.0, [contract]

        monkeypatch.setattr(adapter, "get_costrecovery", fake_costrecovery)
        monkeypatch.setattr(adapter, "optimize_psc", fake_optimize)

        adapter.get_contract_optimization(
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
        assert optimizer_calls["summary_arguments"]["discount_rate"] == 3
        assert "reference_year" not in optimizer_calls["summary_arguments"]


# ── Warnings endpoints ────────────────────────────────────────────

class TestWarningsEndpoints:
    """Canonical warnings endpoints reuse adapters and expose contract warnings."""

    def test_extract_warnings_absent_returns_empty(self):
        """No warning_messages attribute/None -> []."""
        from pyscnomics.api.router import _extract_warnings

        assert _extract_warnings((None, object(), {}, None)) == []

        class NoWarnings:
            warning_messages = None

        assert _extract_warnings((None, NoWarnings(), {}, None)) == []

    def test_extract_warnings_returns_records(self):
        """warning_messages.to_dict(orient="records") is returned verbatim."""
        from pyscnomics.api.router import _extract_warnings

        class Messages:
            def to_dict(self, orient):
                assert orient == "records"
                return [{"No.": 1, "Category": "BASE_PROJECT", "Message": "warn"}]

        class FakeContract:
            warning_messages = Messages()

        assert _extract_warnings((None, FakeContract(), {}, None)) == [
            {"No.": 1, "Category": "BASE_PROJECT", "Message": "warn"}
        ]

    def test_extract_warnings_returns_list_verbatim(self):
        """List-backed dict warning messages are returned unchanged."""
        from pyscnomics.api.router import _extract_warnings

        messages = [{"No.": 1, "Category": "BASE_PROJECT", "Message": "warn"}]

        class FakeContract:
            warning_messages = messages

        assert _extract_warnings((None, FakeContract(), {}, None)) == messages

    def test_extract_warnings_normalizes_tuple_messages(self):
        """Engine tuple warnings become Message dicts for the frontend."""
        from pyscnomics.api.router import _extract_warnings

        class FakeContract:
            warning_messages = [
                ("Gross Split", "First warning"),
                ("Gross Split", "Second warning"),
            ]

        assert _extract_warnings((None, FakeContract(), {}, None)) == [
            {"Message": "First warning"},
            {"Message": "Second warning"},
        ]

    def test_warnings_routes_delegate_to_adapters(self, monkeypatch):
        """Each warnings route calls its adapter with the dumped payload and returns records."""
        import asyncio
        from typing import Any, cast

        import pyscnomics.api.router as router

        class Messages:
            def to_dict(self, orient):
                return [{"ok": True}]

        class FakeContract:
            warning_messages = Messages()

        calls = {}

        def run_case(route_name: str, adapter_name: str):
            def fake_adapter(data=None, summary_result=True):
                calls[adapter_name] = data
                return None, FakeContract(), {}, None

            monkeypatch.setattr(router, adapter_name, fake_adapter)

            class Payload:
                def model_dump(self):
                    return {"dumped": True}

            route = getattr(router, route_name)
            result = asyncio.run(route(cast(Any, Payload())))
            assert result == [{"ok": True}]
            assert calls[adapter_name] == {"dumped": True}

        for route_name, adapter_name in [
            ("get_baseproject_warnings", "get_baseproject"),
            ("get_costrecovery_warnings", "get_costrecovery"),
            ("get_grosssplit_warnings", "get_grosssplit"),
            ("get_transition_warnings", "get_transition"),
        ]:
            run_case(route_name, adapter_name)

    def test_transition_warnings_route_signature(self):
        """Transition warnings route accepts DataTransition, not Data."""
        import inspect

        from pyscnomics.api.converter import DataTransition
        from pyscnomics.api.router import get_transition_warnings

        sig = inspect.signature(get_transition_warnings)
        assert sig.parameters["data"].annotation is DataTransition

    def test_warnings_paths_in_openapi(self):
        """All four warnings paths are registered."""
        schema = client.get("/openapi.json").json()
        for path in [
            "/api/baseproject/warnings",
            "/api/costrecovery/warnings",
            "/api/grosssplit/warnings",
            "/api/transition/warnings",
        ]:
            assert path in schema["paths"]


# ── Canonical table routes ────────────────────────────────────────

class TestTableRoutes:
    """Canonical table routes return only the table dict, not warnings."""

    def test_table_routes_return_table_dict(self, monkeypatch):
        """Each table route returns table dict and delegates with its contract type."""
        import asyncio
        from typing import Any, cast

        import pyscnomics.api.router as router

        table_dict = {"consolidated": {}}
        sentinel = object()
        calls = {}

        def fake_get_contract_table(data=None, contract_type=None):
            calls["data"] = data
            calls["contract_type"] = contract_type
            return table_dict, sentinel

        monkeypatch.setattr(router, "get_contract_table", fake_get_contract_table)

        class Payload:
            def model_dump(self):
                return {"dumped": True}

        for route_name, contract_type in [
            ("get_costrecovery_table", "Cost Recovery"),
            ("get_grosssplit_table", "Gross Split"),
            ("get_transition_table", "Transition"),
            ("get_baseproject_table", "Base Project"),
        ]:
            route = getattr(router, route_name)
            result = asyncio.run(route(cast(Any, Payload())))
            assert result == table_dict
            assert calls["data"] == {"dumped": True}
            assert calls["contract_type"] == contract_type

    def test_grosssplit_table_accepts_nullable_capital_arrays(self, monkeypatch):
        """CapitalCost arrays with JSON null elements pass request validation."""
        import pyscnomics.api.router as router

        calls = {}

        def fake_get_contract_table(data=None, contract_type=None):
            calls["data"] = data
            calls["contract_type"] = contract_type
            return {"consolidated": {}}, object()

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
        assert resp.json() == {"consolidated": {}}
        assert calls["contract_type"] == "Gross Split"


# ── Combine / incremental endpoints ──────────────────────────────

class TestCombineIncrement:
    """Canonical /api/combine and /api/increment reuse core Summary behavior."""

    def test_request_model_shape(self):
        """Direct frontend JSON maps to cases + options with string enums."""
        from pyscnomics.api.converter import CasesRequestBM

        payload = CasesRequestBM.model_validate(
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
        ).model_dump()
        assert [c["cId"] for c in payload["cases"]] == [11, 12]
        assert [c["ctr"] for c in payload["cases"]] == [1, 2]
        assert payload["options"]["npv_mode"] == "SKK Full Cycle Nominal Terms"

    def test_routes_delegate_to_adapters(self, monkeypatch):
        """Routes dump the validated payload and call the combine/increment adapters."""
        import asyncio
        from typing import Any, cast

        import pyscnomics.api.router as router

        calls = {}

        def fake_combine(payload=None):
            calls["combine"] = payload
            return {"combine": []}

        def fake_increment(payload=None):
            calls["increment"] = payload
            return {"increment": []}

        monkeypatch.setattr(router, "get_combine_summary", fake_combine)
        monkeypatch.setattr(router, "get_increment_summary", fake_increment)

        class Fake:
            def model_dump(self):
                return {"cases": [], "options": {}}

        assert asyncio.run(router.calculate_combine(cast(Any, Fake()))) == {
            "combine": []
        }
        assert asyncio.run(router.calculate_increment(cast(Any, Fake()))) == {
            "increment": []
        }
        assert calls["combine"] == {"cases": [], "options": {}}
        assert calls["increment"] == {"cases": [], "options": {}}

    def test_merged_result_keeps_case_order_and_appends_derived_row(
        self, monkeypatch
    ):
        """Per-case rows keep input order; cId=0 derived row is appended last."""
        import numpy as np

        import pyscnomics.api.adapter as adapter

        contracts = []
        summaries = {"a": 1, "b": 2}

        def fake_run(data, ctr):
            contracts.append((data, ctr))
            return object()

        def fake_case_summary(data, ctr):
            return {"case": ctr}

        def fake_table(contract):
            frame = __import__("pandas").DataFrame(
                {"years": [2023, 2024], "cashflow": [10.0, 20.0]}
            )
            return frame, frame, frame

        class FakeSummary:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def case_combine(self):
                return {"years": np.array([2023, 2024]), "cashflow": np.array([1, 2])}

            def get_executive_summary(self):
                return summaries

        monkeypatch.setattr(adapter, "_run_case_contract", fake_run)
        monkeypatch.setattr(adapter, "_case_summary", fake_case_summary)
        monkeypatch.setattr(adapter, "get_table", fake_table)
        monkeypatch.setattr(adapter, "Summary", FakeSummary)

        payload = {
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
        result = adapter.get_combine_summary(payload)["combine"]

        assert [row["cId"] for row in result] == [5, 9, 0]
        assert [row["ctr"] for row in result] == [1, 2, -1]
        assert result[0]["summary"] == {"case": 1}
        assert result[-1]["table_consolidated"]["years"] == {2023: 2023, 2024: 2024}
        assert result[-1]["table_consolidated"]["cashflow"] == {2023: 1, 2024: 2}
        assert result[-1]["summary"] == summaries

    def test_increment_uses_case_incremental(self, monkeypatch):
        """Increment mode calls Summary.case_incremental, not case_combine."""
        import numpy as np

        import pyscnomics.api.adapter as adapter

        calls = {}

        def fake_run(data, ctr):
            return object()

        def fake_table(contract):
            frame = __import__("pandas").DataFrame(
                {"years": [2023], "cashflow": [10.0]}
            )
            return frame, frame, frame

        class FakeSummary:
            def __init__(self, **kwargs):
                pass

            def case_incremental(self):
                calls["mode"] = "incremental"
                return {"years": np.array([2023]), "cashflow": np.array([5])}

            def get_executive_summary(self):
                return {}

        monkeypatch.setattr(adapter, "_run_case_contract", fake_run)
        monkeypatch.setattr(adapter, "_case_summary", lambda data, ctr: {})
        monkeypatch.setattr(adapter, "get_table", fake_table)
        monkeypatch.setattr(adapter, "Summary", FakeSummary)

        payload = {
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
        result = adapter.get_increment_summary(payload)["increment"]
        assert calls["mode"] == "incremental"
        assert result[-1]["cId"] == 0

    def test_transition_case_uses_second_consolidated_frame(self, monkeypatch):
        """Transition get_table returns a tuple; use the second consolidated frame."""
        import numpy as np

        import pyscnomics.api.adapter as adapter

        def fake_run(data, ctr):
            return object()

        def fake_table(contract):
            frame = __import__("pandas").DataFrame(
                {"years": [2023], "cashflow": [1.0]}
            )
            # legacy transition shape: (oil tuple, gas tuple, consolidated tuple)
            return (frame,), (frame,), (frame, frame)

        class FakeSummary:
            def __init__(self, **kwargs):
                pass

            def case_combine(self):
                return {"years": np.array([2023]), "cashflow": np.array([1])}

            def get_executive_summary(self):
                return {}

        monkeypatch.setattr(adapter, "_run_case_contract", fake_run)
        monkeypatch.setattr(adapter, "_case_summary", lambda data, ctr: {})
        monkeypatch.setattr(adapter, "get_table", fake_table)
        monkeypatch.setattr(adapter, "Summary", FakeSummary)

        payload = {
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
        result = adapter.get_combine_summary(payload)["combine"]
        assert result[0]["table_consolidated"]["cashflow"] == {2023: 1.0}
