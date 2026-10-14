"""T5 regressions: transition preflight validates applicable root arguments and
requires exactly one child scheme with the execution predicate (F3)."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import cast

import pytest

from tests._preflight_fixtures import transition_payload_for
from tests._validation_api_support import as_path, as_records, client, is_record, response_body


def _errors(body: Mapping[str, object]) -> list[dict[str, object]]:
    return as_records(body["errors"])


def _error_paths(body: Mapping[str, object]) -> list[list[str | int]]:
    paths: list[list[str | int]] = []
    for issue in _errors(body):
        locations = issue["locations"]
        assert isinstance(locations, list)
        for loc in cast("list[object]", locations):
            if is_record(loc):
                paths.append(as_path(loc["path"]))
    return paths


@pytest.mark.parametrize("mode_field", ["npv_mode", "discounting_mode"])
def test_transition_root_mode_invalid_before_work(mode_field: str) -> None:
    payload = transition_payload_for("cr", "gs")
    summary = payload["summary_arguments"]
    assert is_record(summary)
    summary[mode_field] = "bad"

    preflight = client.post("/api/transition/validate", json=payload)
    assert preflight.status_code == 200
    report = response_body(preflight)
    assert report["validation_complete"] is True
    assert report["can_calculate"] is False
    assert ["summary_arguments", mode_field] in _error_paths(report)

    direct = client.post("/api/transition", json=payload)
    assert direct.status_code == 422
    body = response_body(direct)
    assert body["kind"] == "input_validation"
    assert _errors(body)


def test_transition_root_mode_invalid_nested_batch_prefix() -> None:
    payload = transition_payload_for("cr", "gs")
    summary = payload["summary_arguments"]
    assert is_record(summary)
    summary["npv_mode"] = "bad"
    request: dict[str, object] = {
        "cases": [{"cId": 1, "ctr": 3, "data": payload}],
        "options": {"reference_year": 2025},
    }

    preflight = client.post("/api/combine/validate", json=request)
    assert preflight.status_code == 200
    report = response_body(preflight)
    assert report["can_calculate"] is False
    assert ["cases", 0, "data", "summary_arguments", "npv_mode"] in _error_paths(report)


@pytest.mark.parametrize("period_key", ["contract_1", "contract_2"])
def test_transition_requires_exactly_one_child_scheme(period_key: str) -> None:
    # Both non-null maps.
    both = transition_payload_for("cr", "gs")
    child = both[period_key]
    assert is_record(child)
    child["costrecovery"] = {"something": 1}
    child["grosssplit"] = {"something": 1}

    response = client.post("/api/transition", json=both)
    assert response.status_code == 422
    body = response_body(response)
    assert any(
        issue["code"] == "TRANSITION_SCHEME_UNRECOGNIZED" for issue in _errors(body)
    )
    paths = _error_paths(body)
    assert [period_key, "costrecovery"] in paths
    assert [period_key, "grosssplit"] in paths

    # Neither selected.
    neither = transition_payload_for("cr", "gs")
    child = neither[period_key]
    assert is_record(child)
    child["costrecovery"] = None
    child["grosssplit"] = None

    response = client.post("/api/transition", json=neither)
    assert response.status_code == 422
    body = response_body(response)
    assert any(
        issue["code"] == "TRANSITION_SCHEME_UNRECOGNIZED" for issue in _errors(body)
    )


def test_valid_transition_combinations_remain_calculable() -> None:
    for scheme_1, scheme_2 in (("cr", "cr"), ("cr", "gs"), ("gs", "cr"), ("gs", "gs")):
        payload = copy.deepcopy(transition_payload_for(scheme_1, scheme_2))
        preflight = client.post("/api/transition/validate", json=payload)
        assert preflight.status_code == 200, (scheme_1, scheme_2)
        report = response_body(preflight)
        # Enum-only root checks must not reject an accepted combination.
        assert not any(
            "SUMMARY_" in str(issue["code"]) for issue in _errors(report)
        ), (scheme_1, scheme_2)


def test_transition_root_sensitivity_step_is_located() -> None:
    payload = transition_payload_for("cr", "gs")
    payload["sensitivity_arguments"] = {
        "min_deviation": 0.0,
        "max_deviation": 0.0,
        "step": 0,
        "base_value": 1.0,
    }

    preflight = client.post("/api/transition/validate", json=payload)
    assert preflight.status_code == 200
    report = response_body(preflight)
    assert report["can_calculate"] is False
    assert ["sensitivity_arguments", "step"] in _error_paths(report)

    direct = client.post("/api/transition/sensitivity", json=payload)
    assert direct.status_code == 422
    body = response_body(direct)
    assert body["kind"] == "input_validation"
    issue = next(
        item for item in _errors(body) if item["code"] == "SENSITIVITY_STEP_NOT_POSITIVE"
    )
    assert "Step is 0" in str(issue["message"])
    assert any(path == ["sensitivity_arguments", "step"] for path in _error_paths(body))


def test_transition_root_uncertainty_and_optimization_are_located() -> None:
    payload = transition_payload_for("cr", "gs")
    payload["uncertainty_arguments"] = {
        "run_number": 0,
        "oil_price_distribution": "Normal",
        "gas_price_distribution": "Normal",
        "opex_distribution": "Normal",
        "capex_distribution": "Normal",
        "lifting_distribution": "Normal",
    }
    report = response_body(client.post("/api/transition/validate", json=payload))
    assert report["can_calculate"] is False
    assert ["uncertainty_arguments", "run_number"] in _error_paths(report)

    payload = transition_payload_for("cr", "gs")
    payload["optimization_arguments"] = {
        "dict_optimization": {
            "parameter": ["not_a_parameter"],
            "min": [0.0],
            "max": [1.0],
        },
        "target_optimization": 1.0,
        "target_parameter": "not_a_parameter",
    }
    report = response_body(client.post("/api/transition/validate", json=payload))
    assert report.get("can_calculate") is False
    assert (
        ["optimization_arguments", "target_parameter"] in _error_paths(report)
        or ["optimization_arguments", "dict_optimization", "parameter", 0]
        in _error_paths(report)
    )


def test_transition_discount_start_year_uses_combined_bounds() -> None:
    payload = transition_payload_for("cr", "gs")
    summary = payload["summary_arguments"]
    assert is_record(summary)
    summary["discount_rate_start_year"] = 9999

    preflight = client.post("/api/transition/validate", json=payload)
    assert preflight.status_code == 200
    report = response_body(preflight)
    assert report["can_calculate"] is False
    assert ["summary_arguments", "discount_rate_start_year"] in _error_paths(report)

    direct = client.post("/api/transition", json=payload)
    assert direct.status_code == 422
    body = response_body(direct)
    assert _errors(body)

    # The separate exploratory ``reference_year`` field is not newly rejected.
    accepted = transition_payload_for("cr", "gs")
    accepted_summary = accepted["summary_arguments"]
    assert is_record(accepted_summary)
    accepted_summary["reference_year"] = 9999
    accepted_preflight = client.post("/api/transition/validate", json=accepted)
    assert response_body(accepted_preflight)["can_calculate"] is True


def test_transition_split_ignores_unused_root_modes() -> None:
    for field_name in ("npv_mode", "discounting_mode"):
        payload = transition_payload_for("cr", "gs")
        summary = payload["summary_arguments"]
        assert is_record(summary)
        summary[field_name] = "bad"

        split = client.post("/api/transition/split", json=payload)
        assert split.status_code == 200, field_name

        summary_response = client.post("/api/transition", json=payload)
        assert summary_response.status_code == 422, field_name
        assert _errors(response_body(summary_response))

        preflight = response_body(client.post("/api/transition/validate", json=payload))
        assert preflight["can_calculate"] is False, field_name
