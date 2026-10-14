"""Shared structured input-validation contract for the PSCnomics engine and API.

This module owns the single schema (``schema_version: 1``) used by new API error
and validation responses. Property names here are mirrored by the frontend
TypeScript interfaces; keep them stable.

The types are intentionally dependency-light so that contract helpers can raise
structured input errors without importing the API layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

import numpy as np

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

Severity: TypeAlias = Literal["error", "warning"]
EnvelopeKind: TypeAlias = Literal["input_validation", "server_error"]

# Cost-table sections that can carry addressable submitted rows. Kept local so
# the contract module does not import the rule module that depends on it.
COST_SECTIONS: tuple[str, ...] = (
    "capital",
    "intangible",
    "opex",
    "asr",
    "lbt",
    "cost_of_sales",
)


class NonFiniteJsonError(ValueError):
    """Raised when an issue context contains a non-finite JSON number."""


def _json_value(value: object) -> JsonValue:
    """Convert an engine value into a JSON-safe value.

    NumPy scalar/array values are converted to native Python values. Non-finite
    floats are rejected rather than emitted as invalid JSON.
    """
    if isinstance(value, np.generic):
        value = cast(object, value.item())
    elif isinstance(value, np.ndarray):
        value = cast(object, value.tolist())

    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonFiniteJsonError("Non-finite number is not valid JSON")
        return value
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        return [_json_value(item) for item in items]
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        result: dict[str, JsonValue] = {}
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise NonFiniteJsonError("JSON object keys must be strings")
            result[key] = _json_value(item)
        return result

    # Fallback for enums / date-like values: use their string form.
    return str(value)


def _json_path(path: tuple[str | int, ...]) -> list[JsonValue]:
    result: list[JsonValue] = []
    raw_path = cast("tuple[object, ...]", path)
    for segment in raw_path:
        if not isinstance(segment, (str, int)) or isinstance(segment, bool):
            raise ValueError(f"Invalid issue path segment: {segment!r}")
        result.append(segment)
    return result


@dataclass(frozen=True)
class IssueLocation:
    """A path into the submitted financial payload.

    The path is relative to the request body and never includes the FastAPI
    ``body`` prefix. It is not a URL or executable navigation instruction.
    """

    path: tuple[str | int, ...]

    def to_json(self) -> dict[str, JsonValue]:
        return {"path": _json_path(self.path)}

    @classmethod
    def from_parts(cls, *parts: str | int) -> "IssueLocation":
        return cls(tuple(parts))


@dataclass(frozen=True)
class SkippedCheck:
    """A dependent check that could not run because a prerequisite was invalid."""

    scope: str
    reason: str
    blocking_paths: tuple[IssueLocation, ...] = ()

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "scope": self.scope,
            "reason": self.reason,
            "blocking_paths": [loc.to_json() for loc in self.blocking_paths],
        }


@dataclass(frozen=True)
class ValidationIssue:
    """One user-input issue or warning in the shared contract."""

    code: str
    severity: Severity
    message: str
    guidance: str
    locations: tuple[IssueLocation, ...] = ()
    related_locations: tuple[IssueLocation, ...] = ()
    context: dict[str, JsonValue] = field(default_factory=dict)

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "guidance": self.guidance,
            "locations": [loc.to_json() for loc in self.locations],
            "related_locations": [loc.to_json() for loc in self.related_locations],
            "context": _json_value(self.context),
        }


@dataclass(frozen=True)
class ValidationEnvelope:
    """The shared API envelope for input-validation and server-error responses."""

    schema_version: Literal[1]
    kind: EnvelopeKind
    message: str
    request_id: str
    validation_complete: bool
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()
    skipped_checks: tuple[SkippedCheck, ...] = ()
    can_calculate: bool | None = None

    def to_json(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "message": self.message,
            "request_id": self.request_id,
            "validation_complete": self.validation_complete,
            "errors": [issue.to_json() for issue in self.errors],
            "warnings": [issue.to_json() for issue in self.warnings],
            "skipped_checks": [check.to_json() for check in self.skipped_checks],
        }
        if self.can_calculate is not None:
            payload["can_calculate"] = self.can_calculate
        return payload


class InputValidationError(Exception):
    """Carries a structured validation report as a catchable exception.

    Specialized input errors subclass both this type and the existing domain
    exception (for example ``GrossSplitException``) so that Python library
    callers keep catching the documented superclass.
    """

    issues: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...]
    validation_complete: bool
    skipped_checks: tuple[SkippedCheck, ...]

    def __init__(
        self,
        issues: tuple[ValidationIssue, ...],
        *,
        validation_complete: bool = True,
        skipped_checks: tuple[SkippedCheck, ...] = (),
        warnings: tuple[ValidationIssue, ...] = (),
    ) -> None:
        self.issues = issues
        self.warnings = warnings
        self.validation_complete = validation_complete
        self.skipped_checks = skipped_checks
        super().__init__(self._summary())

    def _summary(self) -> str:
        if self.issues:
            return self.issues[0].message
        return "Input validation failed."

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    def to_report(self) -> "ValidationReport":
        return ValidationReport(
            errors=self.errors,
            warnings=self.warnings
            or tuple(issue for issue in self.issues if issue.severity == "warning"),
            validation_complete=self.validation_complete,
            skipped_checks=self.skipped_checks,
        )


def engine_input_issue(*, code: str, message: str) -> ValidationIssue:
    """Build a minimal structured issue for a verified engine input failure."""
    return ValidationIssue(
        code=code,
        severity="error",
        message=message,
        guidance="",
    )


@dataclass(frozen=True)
class ValidationReport:
    """Aggregated result of running the shared input checks."""
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()
    validation_complete: bool = True
    skipped_checks: tuple[SkippedCheck, ...] = ()

    @property
    def can_calculate(self) -> bool:
        """A report may certify calculation only when nothing is pending.

        A contradictory report (``validation_complete=True`` alongside skipped
        checks) must not be trusted, so skipped checks are also required to be
        empty.
        """
        return not self.errors and self.validation_complete and not self.skipped_checks

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        return self.errors + self.warnings

    def to_error(self) -> InputValidationError:
        errors = self.errors
        if not errors and self.skipped_checks:
            first = self.skipped_checks[0]
            errors = (
                ValidationIssue(
                    code="VALIDATION_INCOMPLETE",
                    severity="error",
                    message=(
                        f"Some checks could not run because {first.scope} needs "
                        "correction. Fix the listed issue, then select Check again."
                    ),
                    guidance=first.reason,
                    locations=first.blocking_paths,
                    context={"scope": first.scope},
                ),
            )
        return InputValidationError(
            errors,
            validation_complete=self.validation_complete,
            skipped_checks=self.skipped_checks,
            warnings=self.warnings,
        )


class ValidationCollector:
    """Accumulates independent issues and skipped checks before reporting."""

    def __init__(self) -> None:
        self._errors: list[ValidationIssue] = []
        self._warnings: list[ValidationIssue] = []
        self._skipped: list[SkippedCheck] = []

    def add(self, issue: ValidationIssue) -> None:
        if issue.severity == "error":
            self._errors.append(issue)
        else:
            self._warnings.append(issue)

    def extend(self, issues: tuple[ValidationIssue, ...]) -> None:
        for issue in issues:
            self.add(issue)

    def skip(self, check: SkippedCheck) -> None:
        self._skipped.append(check)

    def report(self, *, validation_complete: bool | None = None) -> ValidationReport:
        complete = not self._skipped if validation_complete is None else validation_complete
        return ValidationReport(
            errors=tuple(self._errors),
            warnings=tuple(self._warnings),
            validation_complete=complete,
            skipped_checks=tuple(self._skipped),
        )


def count_issue_groups(issues: tuple[ValidationIssue, ...]) -> int:
    """Count displayed issue groups.

    A group is distinguished by its code together with the period/case path
    prefix, cost section, fluid, and classification. Counting by code alone
    would collapse separate presented issues (for example a date-window issue
    for Oil OPEX and one for Gas Intangible) into a single banner group.
    """
    groups: set[tuple[object, ...]] = set()
    for issue in issues:
        context = issue.context
        section = context.get("cost_section") or context.get("section")
        fluid = context.get("fluid")
        classification = context.get("cost_category")
        prefix: tuple[str | int, ...] = ()
        section_from_path: str | None = None
        if issue.locations:
            first_path = issue.locations[0].path
            prefix = _case_prefix(first_path)
            section_from_path = _path_section(first_path)
        if section is None:
            section = section_from_path
        groups.add(
            (
                issue.code,
                prefix,
                _hashable(section),
                _hashable(fluid),
                _hashable(classification),
            )
        )
    return len(groups)


def count_affected_rows(issues: tuple[ValidationIssue, ...]) -> int:
    """Count distinct affected submitted rows across issues.

    Only cost-table rows participate. Configuration locations (which do not end
    in a row index) are excluded, and rows are identified without their field
    segment so ``cost_type`` and ``expense_year`` for the same row count once.
    """
    seen: set[tuple[str | int, ...]] = set()
    for issue in issues:
        for loc in issue.locations:
            identity = _row_identity(loc.path)
            if identity is not None:
                seen.add(identity)
    return len(seen)


def _row_identity(path: tuple[str | int, ...]) -> tuple[str | int, ...] | None:
    """Return a stable row identity for a cost-table location, else ``None``.

    Uses one positional parser: after the optional ``cases/<index>/data`` and
    ``contract_1``/``contract_2`` prefixes, the next segment must be an actual
    top-level cost section, followed by exactly ``table/field/row index``.
    Arbitrary configuration paths that merely contain a cost-section word (for
    example ``contract_arguments/capital/foo/field/0``) are rejected. The field
    segment is dropped while the table name, row index, and the full case and
    transition prefixes are retained.
    """
    prefix, section, index = _split_path(path)
    if section is None or section not in COST_SECTIONS:
        return None
    remainder = path[index + 1 :]
    if len(remainder) != 3:
        return None
    table, field, row = remainder
    if not (isinstance(table, str) and isinstance(field, str)):
        return None
    if isinstance(row, bool) or not isinstance(row, int):
        return None
    return prefix + (section, table, row)


def _split_path(
    path: tuple[str | int, ...],
) -> tuple[tuple[str | int, ...], str | None, int]:
    """Split a location path into (leading prefix, top-level section, index).

    The leading prefix is an optional ``cases/<int>/data`` followed by an
    optional ``contract_1``/``contract_2``. The section is the actual top-level
    payload section immediately after those prefixes; it is never derived from
    an arbitrary search for a cost-section word in a user table name. ``index``
    is the position of the section segment (or ``len(path)`` when absent).
    """
    prefix_parts: list[str | int] = []
    index = 0
    if (
        len(path) >= 3
        and path[0] == "cases"
        and isinstance(path[1], int)
        and not isinstance(path[1], bool)
        and path[2] == "data"
    ):
        prefix_parts = [path[0], path[1], path[2]]
        index = 3
    if index < len(path):
        candidate = path[index]
        if isinstance(candidate, str) and candidate in ("contract_1", "contract_2"):
            prefix_parts.append(candidate)
            index += 1
    section: str | None = None
    if index < len(path):
        candidate = path[index]
        if isinstance(candidate, str):
            section = candidate
    return tuple(prefix_parts), section, index


def _path_section(path: tuple[str | int, ...]) -> str | None:
    _, section, _ = _split_path(path)
    return section


def _case_prefix(path: tuple[str | int, ...]) -> tuple[str | int, ...]:
    """Return the full case/transition prefix that leads any payload section."""
    prefix, _, _ = _split_path(path)
    return prefix


def _hashable(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def validation_message(errors: tuple[ValidationIssue, ...]) -> str:
    """Build the banner summary from the displayed issue groups."""
    groups = count_issue_groups(errors)
    if groups == 0:
        return "Input checks completed. No blocking issues were found."
    noun = "input issue needs" if groups == 1 else "input issues need"
    return f"Calculation stopped: {groups} {noun} attention."
