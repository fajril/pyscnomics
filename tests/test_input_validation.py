"""Tests for the shared input-validation contract types."""

from __future__ import annotations

import json
import math
from typing import cast

import numpy as np
import pytest

from pyscnomics.input_validation import (
    InputValidationError,
    IssueLocation,
    JsonValue,
    NonFiniteJsonError,
    Severity,
    SkippedCheck,
    ValidationCollector,
    ValidationEnvelope,
    ValidationIssue,
    ValidationReport,
    count_affected_rows,
    count_issue_groups,
    validation_message,
)
from pyscnomics.operation_validation import collect_forecast_prerequisite_issues


def _issue(
    code: str = "X",
    *,
    severity: Severity = "error",
    paths: tuple[tuple[str | int, ...], ...] = (("intangible", "t", "cost_type", 0),),
    context: dict[str, JsonValue] | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=severity,
        message="msg",
        guidance="guide",
        locations=tuple(IssueLocation(p) for p in paths),
        context={"amount": 1.5} if context is None else context,
    )


def test_envelope_serializes_exact_property_names() -> None:
    env = ValidationEnvelope(
        schema_version=1,
        kind="input_validation",
        message="Calculation stopped: 1 input issue needs attention.",
        request_id="abc",
        validation_complete=True,
        errors=(_issue(),),
        can_calculate=False,
    )
    payload = env.to_json()
    assert set(payload) >= {
        "schema_version",
        "kind",
        "message",
        "request_id",
        "validation_complete",
        "errors",
        "warnings",
        "skipped_checks",
        "can_calculate",
    }
    assert payload["schema_version"] == 1
    assert json.dumps(payload, allow_nan=False)


def test_can_calculate_omitted_for_calculation_errors() -> None:
    env = ValidationEnvelope(
        schema_version=1,
        kind="input_validation",
        message="m",
        request_id="r",
        validation_complete=True,
        errors=(_issue(),),
    )
    assert "can_calculate" not in env.to_json()


def test_server_error_envelope_cannot_certify() -> None:
    env = ValidationEnvelope(
        schema_version=1,
        kind="server_error",
        message="m",
        request_id="r",
        validation_complete=False,
    )
    assert env.errors == ()
    assert env.to_json()["validation_complete"] is False


def test_issue_location_rejects_invalid_segments() -> None:
    bad_path = cast("tuple[str | int, ...]", (1.5,))
    with pytest.raises(ValueError):
        _ = IssueLocation(bad_path).to_json()


def test_nonfinite_context_is_rejected() -> None:
    issue = ValidationIssue(
        code="X",
        severity="error",
        message="m",
        guidance="g",
        context={"value": float("inf")},
    )
    with pytest.raises(NonFiniteJsonError):
        _ = issue.to_json()


def test_numpy_context_values_are_serialized_as_native() -> None:
    issue = ValidationIssue(
        code="X",
        severity="error",
        message="m",
        guidance="g",
        context=cast("dict[str, JsonValue]", {"year": np.int64(2026), "amount": np.float64(1.25)}),
    )
    payload = issue.to_json()
    assert payload["context"] == {"year": 2026, "amount": 1.25}


def test_group_and_row_counts_are_separate() -> None:
    issues = (
        _issue("A", paths=(("intangible", "t", "cost_type", 0), ("intangible", "t", "cost_type", 1))),
        _issue("A", paths=(("intangible", "t", "cost_type", 1), ("intangible", "t", "cost_type", 2))),
        _issue("B", paths=(("opex", "t", "cost_type", 0),)),
    )
    assert count_issue_groups(issues) == 2
    assert count_affected_rows(issues) == 4


def test_validation_message_singular_and_plural() -> None:
    assert validation_message((_issue("A"),)).startswith("Calculation stopped: 1 input issue needs")
    assert validation_message((_issue("A"), _issue("B"))).startswith(
        "Calculation stopped: 2 input issues need"
    )


def test_collector_separates_errors_and_warnings_and_tracks_skip() -> None:
    collector = ValidationCollector()
    collector.add(_issue("E", severity="error"))
    collector.add(_issue("W", severity="warning"))
    collector.skip(SkippedCheck("scope", "reason", (IssueLocation(("setup",)),)))
    report = collector.report()
    assert len(report.errors) == 1
    assert len(report.warnings) == 1
    assert report.validation_complete is False
    assert report.can_calculate is False


def test_report_to_error_synthesizes_incomplete_issue() -> None:
    report = ValidationReport(
        skipped_checks=(SkippedCheck("dates", "missing"),),
        validation_complete=False,
    )
    error = report.to_error()
    assert isinstance(error, InputValidationError)
    assert error.errors[0].code == "VALIDATION_INCOMPLETE"


def test_specialized_input_error_retains_domain_superclass() -> None:
    from pyscnomics.contracts.gross_split import GrossSplitException

    class GrossSplitInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
        GrossSplitException, InputValidationError
    ):
        def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
            InputValidationError.__init__(self, issues)

    exc = GrossSplitInputError((_issue(),))
    assert isinstance(exc, GrossSplitException)
    assert isinstance(exc, InputValidationError)
    assert str(exc) == "msg"


def test_float_boundary_values_remain_finite() -> None:
    issue = ValidationIssue(
        code="X",
        severity="error",
        message="m",
        guidance="g",
        context={"amount": 76951.27737, "zero": 0.0},
    )
    context = issue.to_json()["context"]
    assert isinstance(context, dict)
    amount = context["amount"]
    assert isinstance(amount, float)
    assert math.isfinite(amount)


def test_groups_distinguish_section_fluid_and_classification() -> None:
    intangible_oil = _issue(
        "SAME",
        paths=(("intangible", "t", "cost_type", 0),),
        context={
            "cost_section": "intangible",
            "fluid": "Oil",
            "cost_category": "preonstream_cost",
        },
    )
    opex_gas = _issue(
        "SAME",
        paths=(("opex", "t", "cost_type", 0),),
        context={
            "cost_section": "opex",
            "fluid": "Gas",
            "cost_category": "preonstream_cost",
        },
    )
    assert count_issue_groups((intangible_oil, opex_gas)) == 2


def test_groups_distinguish_case_path_prefix() -> None:
    case_zero = _issue(
        "SAME",
        paths=(("cases", 0, "data", "intangible", "t", "cost_type", 0),),
        context={"cost_section": "intangible", "fluid": "Oil"},
    )
    case_one = _issue(
        "SAME",
        paths=(("cases", 1, "data", "intangible", "t", "cost_type", 0),),
        context={"cost_section": "intangible", "fluid": "Oil"},
    )
    assert count_issue_groups((case_zero, case_one)) == 2


def test_groups_do_not_split_on_table_name() -> None:
    first_table = _issue(
        "SAME",
        paths=(("intangible", "2026", "cost_type", 0),),
        context={
            "cost_section": "intangible",
            "fluid": "Oil",
            "cost_category": "preonstream_cost",
        },
    )
    second_table = _issue(
        "SAME",
        paths=(("intangible", "2027", "cost_type", 0),),
        context={
            "cost_section": "intangible",
            "fluid": "Oil",
            "cost_category": "preonstream_cost",
        },
    )
    assert count_issue_groups((first_table, second_table)) == 1


def test_groups_distinguish_transition_prefix() -> None:
    contract_one = _issue(
        "SAME",
        paths=(("contract_1", "intangible", "t", "cost_type", 0),),
        context={"cost_section": "intangible", "fluid": "Oil"},
    )
    contract_two = _issue(
        "SAME",
        paths=(("contract_2", "intangible", "t", "cost_type", 0),),
        context={"cost_section": "intangible", "fluid": "Oil"},
    )
    assert count_issue_groups((contract_one, contract_two)) == 2


def test_groups_preserve_case_prefix_for_configuration() -> None:
    case_zero = _issue(
        "INPUT_FORMAT_INVALID",
        paths=(("cases", 0, "data", "setup", "start_date"),),
    )
    case_one = _issue(
        "INPUT_FORMAT_INVALID",
        paths=(("cases", 1, "data", "setup", "start_date"),),
    )
    assert count_issue_groups((case_zero, case_one)) == 2


_ALL_SECTIONS = (
    "setup",
    "summary_arguments",
    "contract_arguments",
    "lifting",
    "capital",
    "intangible",
    "opex",
    "asr",
    "lbt",
    "cost_of_sales",
)


def _section_path(
    section: str,
    *,
    case: int | None = None,
    period: str | None = None,
) -> tuple[str | int, ...]:
    path: tuple[str | int, ...] = ()
    if case is not None:
        path = path + ("cases", case, "data")
    if period is not None:
        path = path + (period,)
    return path + (section, "field")


@pytest.mark.parametrize("section", _ALL_SECTIONS)
def test_groups_distinguish_top_level_sections(section: str) -> None:
    first = _issue("SAME", paths=((section, "field"),))
    second = _issue("SAME", paths=(("__other_section__", "field"),))
    assert count_issue_groups((first, second)) == 2


@pytest.mark.parametrize("section", _ALL_SECTIONS)
@pytest.mark.parametrize("case", [None, 0, 2])
def test_groups_distinguish_same_section_across_cases(
    section: str, case: int | None
) -> None:
    first = _issue("SAME", paths=(_section_path(section, case=case, period="contract_1"),))
    second = _issue(
        "SAME", paths=(_section_path(section, case=case, period="contract_1"),)
    )
    other_case = 1 if case != 1 else 3
    third = _issue(
        "SAME", paths=(_section_path(section, case=other_case, period="contract_1"),)
    )
    assert count_issue_groups((first, second)) == 1
    assert count_issue_groups((first, third)) == 2


@pytest.mark.parametrize("section", _ALL_SECTIONS)
@pytest.mark.parametrize("case", [None, 0])
def test_groups_distinguish_nested_transition_periods(
    section: str, case: int | None
) -> None:
    contract_one = _issue(
        "SAME", paths=(_section_path(section, case=case, period="contract_1"),)
    )
    contract_two = _issue(
        "SAME", paths=(_section_path(section, case=case, period="contract_2"),)
    )
    assert count_issue_groups((contract_one, contract_two)) == 2


def test_groups_distinguish_nested_case_transition_repro() -> None:
    contract_one = _issue(
        "SAME", paths=(("cases", 0, "data", "contract_1", "setup", "start_date"),)
    )
    contract_two = _issue(
        "SAME", paths=(("cases", 0, "data", "contract_2", "setup", "start_date"),)
    )
    assert count_issue_groups((contract_one, contract_two)) == 2


def test_group_section_uses_top_level_not_nested_cost_word() -> None:
    nested_capital = _issue(
        "SAME", paths=(("contract_arguments", "capital", "foo", 0),)
    )
    vat_rate = _issue("SAME", paths=(("contract_arguments", "vat_rate", "foo", 0),))
    assert count_issue_groups((nested_capital, vat_rate)) == 1


def test_group_section_ignores_cost_section_named_table() -> None:
    reserved_table = _issue("SAME", paths=(("intangible", "opex", "cost_type", 0),))
    normal_table = _issue("SAME", paths=(("intangible", "base", "cost_type", 0),))
    assert count_issue_groups((reserved_table, normal_table)) == 1


def test_group_classification_ignores_numeric_actual() -> None:
    first = _issue(
        "SAME",
        paths=(("intangible", "t", "cost_type", 0),),
        context={"cost_section": "intangible", "fluid": "Oil", "actual": 1},
    )
    second = _issue(
        "SAME",
        paths=(("intangible", "t", "cost_type", 0),),
        context={"cost_section": "intangible", "fluid": "Oil", "actual": 2},
    )
    assert count_issue_groups((first, second)) == 1


def test_configuration_issue_not_counted_as_cost_row() -> None:
    configuration = _issue("CFG", paths=(("setup", "is_pod_1"),))
    row = _issue("ROW", paths=(("intangible", "t", "cost_type", 0),))
    assert count_affected_rows((configuration,)) == 0
    assert count_affected_rows((configuration, row)) == 1


def test_configuration_arrays_are_not_cost_rows() -> None:
    fiscal = _issue("F", paths=(("contract_arguments", "effective_tax_rate", 0),))
    summary = _issue("S", paths=(("summary_arguments", "inflation_rate", 0),))
    assert count_affected_rows((fiscal,)) == 0
    assert count_affected_rows((summary,)) == 0


def test_row_identity_rejects_extra_nesting_after_cost_word() -> None:
    nested = _issue("X", paths=(("contract_arguments", "capital", "foo", "field", 0),))
    assert count_affected_rows((nested,)) == 0


def test_row_identity_rejects_nested_case_config_array() -> None:
    config = _issue(
        "X", paths=(("cases", 0, "data", "contract_arguments", "vat_rate", 0),)
    )
    assert count_affected_rows((config,)) == 0


def test_row_identity_accepts_reserved_table_name_without_changing_section() -> None:
    cost_type = _issue("X", paths=(("intangible", "opex", "cost_type", 0),))
    expense_year = _issue("X", paths=(("intangible", "opex", "expense_year", 0),))
    assert count_affected_rows((cost_type, expense_year)) == 1


def test_row_identity_retains_nested_case_and_transition_prefix() -> None:
    direct = _issue("X", paths=(("intangible", "base", "cost_type", 0),))
    nested = _issue(
        "X", paths=(("cases", 0, "data", "contract_1", "intangible", "base", "cost_type", 0),)
    )
    assert count_affected_rows((direct, nested)) == 2


def test_row_identity_ignores_field_segment() -> None:
    cost_type = _issue("A", paths=(("intangible", "t", "cost_type", 0),))
    expense_year = _issue("A", paths=(("intangible", "t", "expense_year", 0),))
    assert count_affected_rows((cost_type, expense_year)) == 1


def test_row_identity_retains_table_and_prefix() -> None:
    base = _issue("A", paths=(("intangible", "base", "cost_type", 0),))
    other_table = _issue("A", paths=(("intangible", "other", "cost_type", 0),))
    prefixed = _issue(
        "A", paths=(("contract_1", "intangible", "base", "cost_type", 0),)
    )
    assert count_affected_rows((base, other_table, prefixed)) == 3


def test_row_identity_includes_nested_case_cost_row() -> None:
    cost_type = _issue(
        "A", paths=(("cases", 0, "data", "intangible", "base", "cost_type", 0),)
    )
    expense_year = _issue(
        "A", paths=(("cases", 0, "data", "intangible", "base", "expense_year", 0),)
    )
    assert count_affected_rows((cost_type, expense_year)) == 1


def test_can_calculate_requires_complete_and_no_skips() -> None:
    contradictory = ValidationReport(
        validation_complete=True,
        skipped_checks=(SkippedCheck("dates", "missing"),),
    )
    assert contradictory.can_calculate is False
    clean = ValidationReport(validation_complete=True)
    assert clean.can_calculate is True


# ── Supported NumPy scalar forecast prerequisites (F4 / A3) ──────────────


def test_forecast_prerequisite_accepts_supported_numpy_zero_scalars() -> None:
    """Supported NumPy zero scalars receive the same located prerequisite."""
    for zero in (
        np.int64(0),
        np.float32(0),
        np.array(0.0),
        np.longdouble(0),
        np.array(np.longdouble(0)),
    ):
        ltp = collect_forecast_prerequisite_issues({"volume": zero}, "ltp")
        assert [issue.code for issue in ltp] == ["LTP_VOLUME_ZERO_PREREQUISITE"]
        assert ltp[0].locations[0].path == ("volume",)
        assert ltp[0].context["actual"] == 0.0

        # The RPD predicate short-circuits on a zero volume, so it reports only
        # that prerequisite; the other zeros are validated separately below.
        rpd_volume = collect_forecast_prerequisite_issues(
            {
                "volume": zero,
                "drate": 0.08,
                "q_plateau_ratio": 0.1,
                "q_min_ratio": 0.05,
            },
            "rpd",
        )
        assert [issue.code for issue in rpd_volume] == [
            "RPD_VOLUME_ZERO_PREREQUISITE"
        ]

        # A NumPy-scalar zero on another RPD prerequisite is located too, once
        # the volume itself is a supported nonzero value.
        rpd_ratio = collect_forecast_prerequisite_issues(
            {
                "volume": np.float32(100),
                "drate": np.array(0.08),
                "q_plateau_ratio": 0.1,
                "q_min_ratio": np.int64(0),
            },
            "rpd",
        )
        assert [issue.code for issue in rpd_ratio] == [
            "RPD_MIN_RATIO_ZERO_PREREQUISITE"
        ]


def test_forecast_prerequisite_keeps_numpy_nonzero_accepted() -> None:
    """Supported NumPy nonzeros keep producing no prerequisite issue."""
    for nonzero in (
        np.int64(100),
        np.float32(100),
        np.array(100.0),
        np.longdouble(100),
        np.array(np.longdouble(100)),
    ):
        assert collect_forecast_prerequisite_issues({"volume": nonzero}, "ltp") == ()
        assert (
            collect_forecast_prerequisite_issues(
                {
                    "volume": nonzero,
                    "drate": np.float32(0.08),
                    "q_plateau_ratio": 0.1,
                    "q_min_ratio": 0.05,
                },
                "rpd",
            )
            == ()
        )


def test_forecast_prerequisite_rejects_bool_and_nonscalar_arrays() -> None:
    """bool/NumPy bool and nonscalar arrays are not scalar representations."""
    # A zero-ish bool is not a supported numeric representation, so no issue is
    # fabricated for it (mirrors the existing Python-bool rejection).
    assert collect_forecast_prerequisite_issues({"volume": False}, "ltp") == ()
    assert collect_forecast_prerequisite_issues({"volume": np.bool_(False)}, "ltp") == ()
    # A one-element (nonscalar) array is rejected rather than coerced.
    assert collect_forecast_prerequisite_issues({"volume": np.array([0.0])}, "ltp") == ()
