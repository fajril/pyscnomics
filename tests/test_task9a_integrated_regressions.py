"""Task 9A integrated regressions: the cross-family HTTP corruption matrix.

The per-family unit suites already cover each contract in isolation. This module
closes the combinations they left open by driving the *real* routes with the
*real* serialized fixtures (the corrected example plus the bundled CR/GS sample
blocks) for every contract family, nested combine/increment, raw non-finite
bodies, request-ID/legacy-detail compatibility and the strict-on group/row
counts.

Every assertion goes through the real FastAPI ``TestClient`` and the real
request models. No original example is edited; strict is enabled only in test
copies. These tests are source/integration evidence only and do not substitute
for the browser acceptance that Sol owns.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Protocol, cast

import pytest
from fastapi.testclient import TestClient

from pyscnomics.api.main import app
from tests._preflight_fixtures import (
    baseproject_payload,
    combine_request,
    corrected_payload,
    single_contract_payload,
    transition_combinations,
    transition_payload_for,
)
from tests._validation_api_support import (
    as_path,
    as_records,
    client,
    response_body,
)

# ── Raw (non-JSON-safe) client for literal NaN/Infinity bodies ───────────────


class _RawResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def json(self) -> object: ...


class _RawClient(Protocol):
    def post(
        self,
        url: str,
        *,
        content: str,
        headers: Mapping[str, str] | None = ...,
    ) -> _RawResponse: ...


_raw_client: _RawClient = TestClient(app, raise_server_exceptions=False)


# ── Contract families with their real validate/calculation routes ────────────


def _family_payloads() -> list[tuple[str, str, dict[str, object]]]:
    return [
        ("grosssplit", "/api/grosssplit", corrected_payload()),
        ("costrecovery", "/api/costrecovery", single_contract_payload("cr")),
        ("baseproject", "/api/baseproject", baseproject_payload()),
        ("transition", "/api/transition", transition_payload_for("cr", "gs")),
    ]


def _scope(payload: dict[str, object]) -> dict[str, object]:
    """The block that carries setup/cost tables (transition uses contract_1)."""
    if "setup" in payload:
        return payload
    return cast("dict[str, object]", payload["contract_1"])


def _first_cost_row(
    payload: dict[str, object], section: str = "opex"
) -> dict[str, object]:
    tables = cast("dict[str, object]", _scope(payload)[section])
    key = next(iter(tables))
    return cast("dict[str, object]", tables[key])


def _setup(payload: dict[str, object]) -> dict[str, object]:
    return cast("dict[str, object]", _scope(payload)["setup"])


def _amount_key(row: dict[str, object]) -> str:
    return "fixed_cost" if "fixed_cost" in row else "cost"


def _mutate(payload: dict[str, object], mutation: str) -> dict[str, object]:
    """Apply one real structural corruption to a copied family payload."""
    mutated = copy.deepcopy(payload)
    setup = _setup(mutated)
    if mutation == "bad_date":
        setup["start_date"] = "not-a-date"
    elif mutation == "reversed":
        setup["end_date"] = "01/01/2018"
    elif mutation == "year_out":
        row = _first_cost_row(mutated)
        row["expense_year"] = [2090, *cast("list[object]", row["expense_year"])[1:]]
    elif mutation == "array_mismatch":
        row = _first_cost_row(mutated)
        key = _amount_key(row)
        row[key] = cast("list[object]", row[key])[:1]
    elif mutation == "bad_number":
        row = _first_cost_row(mutated)
        key = _amount_key(row)
        row[key] = ["abc", *cast("list[object]", row[key])[1:]]
    else:  # pragma: no cover - guard against typos in the parametrization
        raise AssertionError(f"unknown mutation {mutation}")
    return mutated


_STRUCTURAL_MUTATIONS = ("bad_date", "reversed", "year_out", "array_mismatch")
_ALL_MUTATIONS = (*_STRUCTURAL_MUTATIONS, "bad_number")
_FAMILIES = ("grosssplit", "costrecovery", "baseproject", "transition")


def _family(payload_name: str) -> tuple[str, dict[str, object]]:
    for name, route, payload in _family_payloads():
        if name == payload_name:
            return route, payload
    raise AssertionError(f"unknown family {payload_name}")  # pragma: no cover


# ── Matrix: no false-valid preflight, no opaque 500 ──────────────────────────


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("mutation", _ALL_MUTATIONS)
def test_structural_matrix_never_false_valid_and_never_opaque_500(
    family: str, mutation: str
) -> None:
    route, payload = _family(family)
    mutated = _mutate(payload, mutation)

    preflight = client.post(f"{route}/validate", json=mutated)
    direct = client.post(route, json=mutated)

    # Never an opaque internal failure on either route.
    assert preflight.status_code in {200, 422}, (
        family,
        mutation,
        preflight.status_code,
    )
    assert direct.status_code == 422, (family, mutation, direct.status_code)

    if preflight.status_code == 200:
        body = response_body(preflight)
        # A structurally broken payload must never be certified calculable.
        assert body["can_calculate"] is False, (family, mutation)
        assert as_records(body["errors"]), (family, mutation)

    direct_body = response_body(direct)
    if direct_body.get("kind") == "input_validation":
        # Domain/recoverable errors carry the shared envelope and a request id.
        assert direct.headers.get("X-Request-ID") == direct_body["request_id"]
    else:
        # Otherwise it is a real FastAPI structural error: standard detail array.
        assert isinstance(direct_body.get("detail"), list), (family, mutation)


@pytest.mark.parametrize("family", _FAMILIES)
def test_mismatched_arrays_are_reported_not_ignored(family: str) -> None:
    """The ignored-array behaviour must never be re-introduced (R12)."""
    route, payload = _family(family)
    mutated = _mutate(payload, "array_mismatch")
    preflight = client.post(f"{route}/validate", json=mutated)
    assert preflight.status_code == 200
    body = response_body(preflight)
    assert body["can_calculate"] is False
    # At least one reported issue is attributable to the truncated cost row.
    assert as_records(body["errors"])


@pytest.mark.parametrize("family", _FAMILIES)
def test_bad_number_and_raw_nan_are_user_input_422(family: str) -> None:
    route, payload = _family(family)
    bad_number = _mutate(payload, "bad_number")
    for target in (f"{route}/validate", route):
        resp = client.post(target, json=bad_number)
        assert resp.status_code == 422, (family, target)
        assert isinstance(response_body(resp).get("detail"), list)

    # A literal non-finite number in a JSON body must be a safe 422, not a 500.
    finite = copy.deepcopy(payload)
    row = _first_cost_row(finite)
    key = _amount_key(row)
    row[key] = [1.0, *cast("list[object]", row[key])[1:]]
    raw = json.dumps(finite, allow_nan=True).replace("[1.0,", "[NaN,", 1)
    for target in (f"{route}/validate", route):
        resp = _raw_client.post(
            target, content=raw, headers={"content-type": "application/json"}
        )
        assert resp.status_code == 422, (family, target)
        body = response_body(resp)
        assert body.get("kind") == "input_validation", (family, target)


# ── Nested combine/increment retain the corruption ───────────────────────────

_NESTED = (
    ("costrecovery", 1),
    ("grosssplit", 2),
    ("baseproject", 0),
)


@pytest.mark.parametrize(("payload_name", "ctr"), _NESTED)
@pytest.mark.parametrize("route", ["/api/combine", "/api/increment"])
def test_nested_bad_case_survives_combine_increment(
    payload_name: str, ctr: int, route: str
) -> None:
    _, payload = _family(payload_name)
    good = copy.deepcopy(payload)
    bad = _mutate(payload, "array_mismatch")

    request = combine_request(
        [
            {"cId": 1, "ctr": ctr, "data": good},
            {"cId": 2, "ctr": ctr, "data": bad},
        ]
    )

    preflight = client.post(f"{route}/validate", json=request)
    assert preflight.status_code == 200, (payload_name, route)
    body = response_body(preflight)
    assert body["can_calculate"] is False

    paths = [
        as_path(loc["path"])
        for error in as_records(body["errors"])
        for loc in as_records(error["locations"])
    ]
    assert paths, (payload_name, route)
    # The bad case is the second one and keeps its case/period prefix.
    assert any(path[:3] == ["cases", 1, "data"] for path in paths), (
        payload_name,
        route,
    )

    direct = client.post(route, json=request)
    assert direct.status_code == 422, (payload_name, route)
    assert response_body(direct).get("kind") == "input_validation"


@pytest.mark.parametrize(("label", "scheme_1", "scheme_2"), transition_combinations())
def test_nested_transition_bad_block_survives_combine_increment(
    label: str, scheme_1: str, scheme_2: str
) -> None:
    data = transition_payload_for(scheme_1, scheme_2)
    child = cast("dict[str, object]", data["contract_2"])
    cast("dict[str, object]", child["setup"])["end_date"] = "01/01/2018"
    request = combine_request([{"cId": 1, "ctr": 4, "data": data}])

    for route in ("/api/combine", "/api/increment"):
        preflight = client.post(f"{route}/validate", json=request)
        assert preflight.status_code == 200, (label, route)
        body = response_body(preflight)
        assert body["can_calculate"] is False, (label, route)
        paths = [
            as_path(loc["path"])
            for error in as_records(body["errors"])
            for loc in as_records(error["locations"])
        ]
        assert paths
        assert all(path[:3] == ["cases", 0, "data"] for path in paths), (label, route)
        assert all(path[3] in {"contract_1", "contract_2"} for path in paths)

        direct = client.post(route, json=request)
        assert direct.status_code == 422, (label, route)


# ── Valid nullable defaults and all four transition combinations ─────────────


@pytest.mark.parametrize(("label", "scheme_1", "scheme_2"), transition_combinations())
def test_four_transition_combos_and_nullable_defaults_preserved(
    label: str, scheme_1: str, scheme_2: str
) -> None:
    payload = transition_payload_for(scheme_1, scheme_2)

    # Nullable onstream dates accepted through the engine default must not be
    # turned into a structural error, and the direct calculation still succeeds.
    _setup(payload)["gas_onstream_date"] = None
    direct = client.post("/api/transition", json=payload)
    assert direct.status_code == 200, label

    preflight = client.post("/api/transition/validate", json=payload)
    assert preflight.status_code == 200, label
    body = response_body(preflight)
    # No fabricated error for the nullable default; incomplete checks disclosed.
    assert body["errors"] == [], label


# ── Server-generated request IDs and actionable legacy detail ────────────────


def test_domain_422_has_server_request_id_and_actionable_legacy_detail() -> None:
    payload = _mutate(corrected_payload(), "year_out")
    resp = client.post("/api/grosssplit", json=payload)
    assert resp.status_code == 422
    body = response_body(resp)
    request_id = resp.headers.get("X-Request-ID")
    assert request_id
    assert request_id == body["request_id"]

    detail = body.get("detail")
    assert isinstance(detail, str)
    assert detail.startswith("PyscErr:")
    assert len(detail) > len("PyscErr:")


def test_spoofed_request_id_is_not_trusted() -> None:
    payload = _mutate(corrected_payload(), "year_out")
    resp = client.post(
        "/api/grosssplit",
        json=payload,
        headers={"X-Request-ID": "spoofed-client-value"},
    )
    assert resp.status_code == 422
    body = response_body(resp)
    assert body["request_id"] != "spoofed-client-value"
    assert resp.headers["X-Request-ID"] == body["request_id"]


# ── Strict-on group/row counts (integrated HTTP form) ────────────────────────


def test_strict_on_corrected_two_groups_seven_unique_rows_over_http() -> None:
    payload = corrected_payload()
    cast("dict[str, object]", payload["setup"])["is_strict"] = True
    resp = client.post("/api/grosssplit/validate", json=payload)
    assert resp.status_code == 200
    body = response_body(resp)
    assert body["can_calculate"] is False

    errors = as_records(body["errors"])
    assert len(errors) == 2

    paths = [
        as_path(loc["path"])
        for error in errors
        for loc in as_records(error["locations"])
    ]
    # Two issue groups, seven unique affected submitted rows (3 + 4).
    assert len(paths) == 7
    assert len({tuple(path[:4]) for path in paths}) == 7

    # The direct calculation refuses the same strict-on input; it does not 500.
    direct = client.post("/api/grosssplit", json=payload)
    assert direct.status_code == 422
    assert response_body(direct)["kind"] == "input_validation"
