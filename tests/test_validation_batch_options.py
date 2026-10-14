"""T6 regressions: invalid combine/increment options are located input errors
before numerical work, and are never certified or attributed to a case (F2)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

from tests._preflight_fixtures import combine_request, corrected_payload
from tests._validation_api_support import as_path, as_records, client, is_record, response_body


def _cases() -> list[dict[str, object]]:
    return [{"cId": 1, "ctr": 2, "data": corrected_payload()}]


def _request_with_bad_option(field: str) -> dict[str, object]:
    request = combine_request(_cases(), reference_year=2025)
    options = request["options"]
    assert is_record(options)
    options[field] = "bad"
    return request


def _without_execution_datetime(value: object) -> object:
    """Normalize the volatile execution timestamp only (as the example parity does)."""
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        return {
            key: _without_execution_datetime(item)
            for key, item in mapping.items()
            if key != "execution_datetime"
        }
    if isinstance(value, list):
        return [
            _without_execution_datetime(item)
            for item in cast("list[object]", value)
        ]
    return value


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


@pytest.mark.parametrize("operation", ["combine", "increment"])
@pytest.mark.parametrize("mode_field", ["npv_mode", "discounting_mode"])
def test_batch_bad_modes_are_located_before_work(operation: str, mode_field: str) -> None:
    payload = _request_with_bad_option(mode_field)

    preflight = client.post(f"/api/{operation}/validate", json=payload)
    assert preflight.status_code == 200
    report = response_body(preflight)
    assert report["validation_complete"] is True
    assert report["can_calculate"] is False
    assert ["options", mode_field] in _error_paths(report)

    direct = client.post(f"/api/{operation}", json=payload)
    assert direct.status_code == 422
    body = response_body(direct)
    assert body["kind"] == "input_validation"
    assert _errors(body)


@pytest.mark.parametrize("operation", ["combine", "increment"])
@pytest.mark.parametrize("mode_field", ["npv_mode", "discounting_mode"])
def test_batch_global_option_guidance_targets_the_options_control(
    operation: str, mode_field: str
) -> None:
    """A batch-global option issue must direct the user to Options, not the
    contract summary settings, and keep the rejected value."""
    request = _request_with_bad_option(mode_field)
    preflight = client.post(f"/api/{operation}/validate", json=request)
    assert preflight.status_code == 200
    issue = next(
        item
        for item in _errors(response_body(preflight))
        if any(
            is_record(loc) and as_path(loc["path"]) == ["options", mode_field]
            for loc in cast("list[object]", item["locations"])
        )
    )
    guidance = str(issue["guidance"])
    assert "Options" in guidance
    assert "summary settings" not in guidance
    assert "bad" in str(issue["message"])


@pytest.mark.parametrize("operation", ["combine", "increment"])
def test_batch_ignores_unconsumed_transition_analysis(operation: str) -> None:
    """An analysis block no batch worker reads cannot block a calculation."""
    import json

    from pyscnomics.api.adapter import get_combine_summary, get_increment_summary
    from pyscnomics.api.converter import CasesRequestBM
    from tests._preflight_fixtures import transition_payload_for

    data = transition_payload_for("cr", "gs")
    data["sensitivity_arguments"] = {
        "min_deviation": 0,
        "max_deviation": 0,
        "step": 0,
        "base_value": 1,
    }
    request = combine_request([{"cId": 1, "ctr": 3, "data": data}], reference_year=2025)

    direct = client.post(f"/api/{operation}", json=request)
    assert direct.status_code == 200, response_body(direct)
    dump = CasesRequestBM.model_validate(request).model_dump()
    worker = get_combine_summary if operation == "combine" else get_increment_summary
    expected = cast("object", json.loads(json.dumps(worker(payload=dump))))
    got = cast("object", json.loads(json.dumps(response_body(direct))))
    assert _without_execution_datetime(got) == _without_execution_datetime(expected)

    # The validation-only route still reports the present invalid block.
    preflight = client.post(f"/api/{operation}/validate", json=request)
    assert preflight.status_code == 200
    report = response_body(preflight)
    assert report["can_calculate"] is False
    assert ["cases", 0, "data", "sensitivity_arguments", "step"] in _error_paths(report)


@pytest.mark.parametrize("operation", ["combine", "increment"])
def test_batch_ignores_unconsumed_uncertainty_and_optimization(
    operation: str,
) -> None:
    """Uncertainty and optimization blocks are ignored by batch calculation."""
    import json

    from pyscnomics.api.adapter import get_combine_summary, get_increment_summary
    from pyscnomics.api.converter import CasesRequestBM
    from tests._preflight_fixtures import transition_payload_for

    data = transition_payload_for("cr", "gs")
    data["uncertainty_arguments"] = {
        "run_number": 0,
        "oil_price_distribution": "Normal",
        "gas_price_distribution": "Normal",
        "opex_distribution": "Normal",
        "capex_distribution": "Normal",
        "lifting_distribution": "Normal",
    }
    data["optimization_arguments"] = {
        "dict_optimization": {
            "parameter": ["not_a_parameter"],
            "min": [0.0],
            "max": [1.0],
        },
        "target_optimization": 1.0,
        "target_parameter": "not_a_parameter",
    }
    request = combine_request([{"cId": 1, "ctr": 3, "data": data}], reference_year=2025)

    direct = client.post(f"/api/{operation}", json=request)
    assert direct.status_code == 200, response_body(direct)
    dump = CasesRequestBM.model_validate(request).model_dump()
    worker = get_combine_summary if operation == "combine" else get_increment_summary
    expected = cast("object", json.loads(json.dumps(worker(payload=dump))))
    got = cast("object", json.loads(json.dumps(response_body(direct))))
    assert _without_execution_datetime(got) == _without_execution_datetime(expected)

    preflight = response_body(client.post(f"/api/{operation}/validate", json=request))
    assert preflight["can_calculate"] is False
    paths = _error_paths(preflight)
    assert ["cases", 0, "data", "uncertainty_arguments", "run_number"] in paths
    assert any(
        path[:3] == ["cases", 0, "data"]
        and path[3] == "optimization_arguments"
        for path in paths
    )


@pytest.mark.parametrize("operation", ["combine", "increment"])
def test_nested_consumed_summary_issue_keeps_case_prefix(operation: str) -> None:
    """A consumed nested summary issue keeps its ``cases/1/data`` prefix."""
    first = corrected_payload()
    second = corrected_payload()
    summary = second["summary_arguments"]
    assert is_record(summary)
    summary["npv_mode"] = "bad"
    request = combine_request(
        [
            {"cId": 1, "ctr": 2, "data": first},
            {"cId": 2, "ctr": 2, "data": second},
        ],
        reference_year=2025,
    )

    direct = client.post(f"/api/{operation}", json=request)
    assert direct.status_code == 422
    direct_paths = _error_paths(response_body(direct))
    assert ["cases", 1, "data", "summary_arguments", "npv_mode"] in direct_paths

    preflight = response_body(client.post(f"/api/{operation}/validate", json=request))
    assert preflight["can_calculate"] is False
    assert ["cases", 1, "data", "summary_arguments", "npv_mode"] in _error_paths(preflight)


@pytest.mark.parametrize("operation", ["combine", "increment"])
def test_batch_valid_options_stay_calculable(operation: str) -> None:
    payload = combine_request(_cases(), reference_year=2025)

    preflight = client.post(f"/api/{operation}/validate", json=payload)
    assert preflight.status_code == 200
    report = response_body(preflight)
    assert report["can_calculate"] is True
    assert report["errors"] == []


@pytest.mark.parametrize("operation", ["combine", "increment"])
def test_batch_options_and_case_errors_aggregate(operation: str) -> None:
    # A strict GS POD I case with the two reviewed blocks plus a bad global mode.
    case = corrected_payload()
    setup = case.get("setup")
    assert is_record(setup)
    setup["is_strict"] = True
    payload = combine_request([{"cId": 1, "ctr": 2, "data": case}], reference_year=2025)
    options = payload["options"]
    assert is_record(options)
    options["npv_mode"] = "bad"

    preflight = client.post(f"/api/{operation}/validate", json=payload)
    assert preflight.status_code == 200
    report = response_body(preflight)
    assert report["can_calculate"] is False

    paths = _error_paths(report)
    # The global option issue stays global (options prefix), never case 0.
    assert ["options", "npv_mode"] in paths
    # Nested classification issues retain the case/data prefix.
    assert any(path[:3] == ["cases", 0, "data"] for path in paths)
    # The global option issue is never attributed to the first case.
    assert not any(path[:4] == ["cases", 0, "data", "options"] for path in paths)
