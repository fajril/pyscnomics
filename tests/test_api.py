"""
API smoke tests — verify core endpoints and import chains work after refactoring.
Run: uv run pytest tests/test_api.py -v
"""

import pytest
from datetime import date
from fastapi.testclient import TestClient

from pyscnomics.api.main import app


client = TestClient(app)


# ── GET endpoints ──────────────────────────────────────────────────

class TestGetEndpoints:
    def test_root_returns_version(self):
        resp = client.get("/api/")
        assert resp.status_code == 200
        body = resp.json()
        assert "Pyscnomics" in body

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

    def test_import_optimize(self):
        from pyscnomics.optimize.optimization import optimize_psc
        from pyscnomics.optimize.sensitivity import sensitivity_psc
        from pyscnomics.optimize.uncertainty import uncertainty_psc

    def test_import_tools(self):
        from pyscnomics.tools.table import get_table
        from pyscnomics.tools.summary import get_summary

    def test_import_io(self):
        from pyscnomics.io import getattr as _

    def test_import_api(self):
        from pyscnomics.api.main import app
        from pyscnomics.api.router import router
        from pyscnomics.api.adapter import get_grosssplit
        from pyscnomics.api.converter import Data


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
