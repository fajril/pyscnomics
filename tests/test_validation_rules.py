"""Tests for the shared engine input-validation predicates (Task 2)."""

from __future__ import annotations

from pyscnomics.input_validation import ValidationCollector
from pyscnomics.validation_rules import (
    collect_contract_issues,
    collect_grosssplit_pod1_noncapital_issues,
    collect_malformed_numeric_issues,
    collect_preonstream_window_issues,
)


def _setup(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "start_date": "01/01/2019",
        "end_date": "31/12/2050",
        "oil_onstream_date": "01/01/2029",
        "gas_onstream_date": "01/01/2029",
        "approval_year": 2024,
        "is_pod_1": True,
        "is_strict": True,
    }
    base.update(overrides)
    return base


def gs_payload(
    *,
    intangible_years: list[int] | None = None,
    opex_years: list[int] | None = None,
    fluid: str = "Gas",
    strict: bool = True,
    pod1: bool = True,
) -> dict[str, object]:
    intangible_years = intangible_years or [2026, 2027, 2028]
    opex_years = opex_years or [2025, 2026, 2027, 2028]
    return {
        "setup": _setup(is_strict=strict, is_pod_1=pod1),
        "grosssplit": {},
        "intangible": {
            "base": {
                "start_year": 2019,
                "end_year": 2050,
                "expense_year": intangible_years,
                "cost": [100.0 + i for i in range(len(intangible_years))],
                "cost_allocation": [fluid] * len(intangible_years),
                "cost_type": ["preonstream_cost"] * len(intangible_years),
            }
        },
        "opex": {
            "base": {
                "start_year": 2019,
                "end_year": 2050,
                "expense_year": opex_years,
                "fixed_cost": [200.0 + i for i in range(len(opex_years))],
                "cost_allocation": [fluid] * len(opex_years),
                "cost_type": ["preonstream_cost"] * len(opex_years),
            }
        },
    }


def test_collects_intangible_and_opex_preonstream_issues() -> None:
    issues = collect_grosssplit_pod1_noncapital_issues(gs_payload())
    by_code = {issue.code: issue for issue in issues}
    assert len(by_code["GS_POD1_NONCAPITAL_INTANGIBLE"].locations) == 3
    assert len(by_code["GS_POD1_NONCAPITAL_OPEX"].locations) == 4
    assert all(issue.severity == "error" for issue in issues)


def test_sample_fixture_includes_2029_opex_entry() -> None:
    issues = collect_grosssplit_pod1_noncapital_issues(
        gs_payload(opex_years=[2025, 2026, 2027, 2028, 2029])
    )
    by_code = {issue.code: issue for issue in issues}
    assert len(by_code["GS_POD1_NONCAPITAL_INTANGIBLE"].locations) == 3
    assert len(by_code["GS_POD1_NONCAPITAL_OPEX"].locations) == 5
    opex_paths = [loc.path for loc in by_code["GS_POD1_NONCAPITAL_OPEX"].locations]
    assert any(path[-1] == 4 for path in opex_paths)


def test_loose_severity_matches_existing_engine() -> None:
    issues = collect_grosssplit_pod1_noncapital_issues(gs_payload(strict=False))
    assert issues
    assert all(issue.severity == "warning" for issue in issues)


def test_non_pod1_skips_classification_rule() -> None:
    assert collect_grosssplit_pod1_noncapital_issues(gs_payload(pod1=False)) == ()


def test_pod1_noncapital_message_explains_why_the_classification_conflicts() -> None:
    issue = next(
        item
        for item in collect_grosssplit_pod1_noncapital_issues(gs_payload())
        if item.code == "GS_POD1_NONCAPITAL_INTANGIBLE"
    )
    message = issue.message
    guidance = issue.guidance
    # Names the actual fluid, section and entered classification.
    assert "Gas" in message
    assert "Intangible Costs" in message
    assert "Preonstream Cost" in message
    # Explains the model conflict: capital-cost treatment with amortization,
    # versus a direct charge without amortization.
    assert "capital-cost treatment" in message
    assert "amortized" in message
    assert "without amortization" in message
    # Both strictness modes are described honestly.
    assert "Strict mode stops" in message
    assert "loose mode warns" in message
    # The next action reviews the intended treatment, classification and POD I.
    assert "cost treatment" in guidance
    assert "Cost Type" in guidance
    assert "POD I" in guidance


def test_pod1_noncapital_message_reasoning_covers_sunk_and_loose_mode() -> None:
    payload: dict[str, object] = {
        "setup": _setup(is_strict=False, is_pod_1=True),
        "grosssplit": {},
        "opex": {
            "base": {
                "start_year": 2019,
                "end_year": 2050,
                "expense_year": [2026],
                "fixed_cost": [10.0],
                "cost_allocation": ["Oil"],
                "cost_type": ["sunk_cost"],
            }
        },
    }
    issues = collect_grosssplit_pod1_noncapital_issues(payload)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.severity == "warning"
    assert issue.code == "GS_POD1_NONCAPITAL_OPEX"
    assert "Sunk Cost" in issue.message
    assert "capital-cost treatment" in issue.message
    assert "loose mode warns" in issue.message
    assert "POD I" in issue.guidance


def test_collects_both_fluids() -> None:
    payload = gs_payload()
    _ = payload.pop("opex")
    payload["intangible"] = {
        "oil": {
            "start_year": 2019,
            "end_year": 2050,
            "expense_year": [2026],
            "cost": [10.0],
            "cost_allocation": ["Oil"],
            "cost_type": ["preonstream_cost"],
        },
        "gas": {
            "start_year": 2019,
            "end_year": 2050,
            "expense_year": [2026],
            "cost": [20.0],
            "cost_allocation": ["Gas"],
            "cost_type": ["preonstream_cost"],
        },
    }
    issues = collect_grosssplit_pod1_noncapital_issues(payload)
    assert len(issues) == 2
    fluids = {str(issue.context["fluid"]) for issue in issues}
    assert fluids == {"Oil", "Gas"}


def test_invalid_dates_skip_only_dependent_checks() -> None:
    payload = gs_payload()
    payload["setup"] = _setup(gas_onstream_date=None)
    collector = ValidationCollector()
    collect_contract_issues(payload, "Gross Split", collector)
    report = collector.report()
    # Classification checks are independent of the dates and still run.
    assert any(
        issue.code == "GS_POD1_NONCAPITAL_INTANGIBLE" for issue in report.errors
    )
    # The date-dependent window check is disclosed as skipped, not fabricated.
    assert any(check.scope == "preonstream_window" for check in report.skipped_checks)
    assert report.validation_complete is False


def test_mismatched_arrays_block_and_skip_dependent_checks() -> None:
    payload = gs_payload()
    payload["intangible"] = {
        "base": {
            "start_year": 2019,
            "end_year": 2050,
            "expense_year": [2026],
            "cost": [10.0, 20.0, 30.0],
            "cost_allocation": ["Gas"],
            "cost_type": ["preonstream_cost", "preonstream_cost", "preonstream_cost"],
        }
    }
    collector = ValidationCollector()
    collect_contract_issues(payload, "Gross Split", collector)
    report = collector.report()
    # A mismatched array is a blocking structural error, never silently ignored.
    assert any(issue.code == "ARRAY_LENGTH_MISMATCH" for issue in report.errors)
    # Dependent row checks for that table are disclosed as skipped.
    assert any("row_checks" in check.scope for check in report.skipped_checks)
    assert report.can_calculate is False


def test_explicit_invalid_numeric_value_is_not_zero() -> None:
    payload = gs_payload()
    payload["intangible"] = {
        "base": {
            "start_year": 2019,
            "end_year": 2050,
            "expense_year": [2026],
            "cost": ["not-a-number"],
            "cost_allocation": ["Gas"],
            "cost_type": ["preonstream_cost"],
        }
    }
    issues = collect_malformed_numeric_issues(payload)
    assert len(issues) == 1
    assert issues[0].code == "NUMERIC_FIELD_CONTAINS_TEXT"
    assert issues[0].severity == "error"


def test_malformed_numeric_message_uses_a_human_field_label() -> None:
    payload = gs_payload()
    payload["opex"] = {
        "base": {
            "start_year": 2019,
            "end_year": 2050,
            "expense_year": [2026],
            "fixed_cost": ["abc"],
            "cost_allocation": ["Gas"],
            "cost_type": ["preonstream_cost"],
        }
    }
    issues = collect_malformed_numeric_issues(payload)
    assert len(issues) == 1
    message = issues[0].message
    assert "Fixed cost" in message
    assert "fixed_cost" not in message


def test_preonstream_window_rule_flags_year_after_onstream() -> None:
    payload = gs_payload(intangible_years=[2030])
    payload["setup"] = _setup(is_pod_1=False)
    issues = collect_preonstream_window_issues(payload)
    assert any(issue.code == "PREONSTREAM_YEAR_OUTSIDE_PROJECT" for issue in issues)
