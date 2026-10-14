"""Aggregation tests for multi-error input validation (Task 2)."""

from __future__ import annotations

from pyscnomics.api.validation import (
    validate_cases_input,
    validate_contract_input,
)
from pyscnomics.input_validation import count_affected_rows, count_issue_groups
from tests.test_validation_rules import gs_payload


def test_corrected_fixture_two_groups_seven_rows() -> None:
    report = validate_contract_input(
        gs_payload(), "Gross Split", operation="grosssplit"
    )
    assert len(report.errors) == 2
    assert count_issue_groups(report.errors) == 2
    assert count_affected_rows(report.errors) == 7
    assert report.can_calculate is False


def test_sample_fixture_two_groups_eight_rows() -> None:
    report = validate_contract_input(
        gs_payload(opex_years=[2025, 2026, 2027, 2028, 2029]),
        "Gross Split",
        operation="grosssplit",
    )
    assert count_issue_groups(report.errors) == 2
    assert count_affected_rows(report.errors) == 8


def test_loose_mode_returns_warnings_and_can_calculate() -> None:
    report = validate_contract_input(
        gs_payload(strict=False), "Gross Split", operation="grosssplit"
    )
    assert report.errors == ()
    assert len(report.warnings) == 2
    assert report.can_calculate is True


def test_transition_prefixes_locations() -> None:
    child: dict[str, object] = {
        "setup": gs_payload()["setup"],
        "contract_arguments": {},
        "lifting": {},
        "capital": {},
        "intangible": gs_payload()["intangible"],
        "opex": gs_payload()["opex"],
        "asr": {},
        "lbt": {},
        "grosssplit": {},
    }
    payload: dict[str, object] = {
        "contract_1": child,
        "contract_2": child,
        "contract_arguments": {"unrec_portion": 0.0},
        "summary_arguments": {},
    }
    report = validate_contract_input(payload, "Transition", operation="transition")
    assert report.errors
    prefixes = {
        loc.path[0] for issue in report.errors for loc in issue.locations
    }
    assert prefixes <= {"contract_1", "contract_2"}
    assert prefixes  # at least one prefix present


def test_combine_retains_case_prefix() -> None:
    nested: dict[str, object] = {
        **gs_payload(),
        "summary_arguments": {},
        "contract_arguments": {},
    }
    payload: dict[str, object] = {
        "cases": [
            {"cId": 1, "ctr": 2, "data": nested},
            {"cId": 2, "ctr": 2, "data": nested},
        ],
        "options": {"reference_year": 2024},
    }
    report = validate_cases_input(payload, "combine")
    assert report.errors
    prefixes = {
        (loc.path[0], loc.path[1]) for issue in report.errors for loc in issue.locations
    }
    assert ("cases", 0) in prefixes or ("cases", 1) in prefixes
