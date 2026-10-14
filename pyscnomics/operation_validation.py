"""Pure standalone-operation input predicates (LTP, RPD, econlimit, ASR, LBT).

These translate the existing period/series prerequisites of the standalone
workers into located structured issues *before* any numerical work. Only rules
the existing workers already enforce are mirrored; a true runtime failure stays a
server error. The predicates are pure so both the API guard and the corresponding
library boundary checks can reuse them without an import cycle.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Literal, TypeAlias, cast

import numpy as np

from pyscnomics.input_validation import (
    IssueLocation,
    SkippedCheck,
    ValidationIssue,
)
from pyscnomics.validation_rules import (
    collect_project_date_issues,
    period_is_reversed,
    unequal_lengths,
    year_above,
    year_below,
)

StandaloneOperation: TypeAlias = Literal[
    "ltp", "rpd", "econlimit", "asr_expenditures", "lbt_expenditures"
]

_Path = tuple[str | int, ...]
_PathMap = Callable[[_Path], _Path]


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_float(value: object) -> float | None:
    """Return a finite float for entered numeric values, else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    return _numpy_scalar_float(value)


def _numpy_scalar_float(value: object) -> float | None:
    """Return a float for a supported NumPy scalar/0-d numeric array, else None.

    The forecast workers accept NumPy integer/floating scalars and
    zero-dimensional numeric arrays for their volumes and parameters (the same
    values with a nonzero magnitude are accepted). A NumPy bool or a nonscalar
    array is not a scalar representation and is left uninspected rather than
    coerced, so no prerequisite is fabricated for it.
    """
    if isinstance(value, np.ndarray):
        if value.ndim != 0 or value.dtype.kind not in ("i", "u", "f"):
            return None
        return float(cast("float", cast("object", value)))
    if isinstance(value, (np.integer, np.floating)):
        return float(cast("float", cast("object", value)))
    return None


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return None


def _series_length(value: object) -> int:
    if isinstance(value, (list, tuple)):
        return len(cast("list[object] | tuple[object, ...]", value))
    return 0


# The LTP worker only models the Oil and Gas fluid enums. ``read_fluid_type``
# maps the LPG lifting categories onto the gas enum, so they are accepted by
# the worker and must be accepted here too; every other category (for example
# the recognized Sulfur/Electricity/CO2 lifting fluids) fails the worker.
_LTP_FLUIDS: tuple[str, ...] = ("Oil", "Gas", "LPG propane", "LPG butane")


def _ltp_fluid_issues(payload: Mapping[str, object]) -> tuple[ValidationIssue, ...]:
    """Locate an unsupported LTP fluid before the converter rejects it.

    ``get_ltp`` only models the Oil and Gas fluid enums, but the converter maps
    the LPG lifting categories (propane/butane) onto the gas enum. Those inputs
    are therefore accepted by the worker and are accepted here; the remaining
    recognized-but-unsupported fluids (Sulfur/Electricity/CO2) still fail the
    worker. This mirrors that existing rule instead of inventing a new one.
    """
    fluid = payload.get("fluid_type")
    if isinstance(fluid, str) and fluid in _LTP_FLUIDS:
        return ()
    return (
        ValidationIssue(
            code="LTP_FLUID_UNRECOGNIZED",
            severity="error",
            message=(
                f"Fluid Type is '{fluid}'. Choose Oil or Gas for the LTP model."
            ),
            guidance=(
                "Review Fluid Type, choose Oil or Gas, then try again."
            ),
            locations=(IssueLocation(("fluid_type",)),),
            context={
                "field": "Fluid Type",
                "actual": str(fluid),
                "allowed": list(_LTP_FLUIDS),
            },
        ),
    )


def collect_forecast_prerequisite_issues(
    payload: Mapping[str, object],
    operation: Literal["ltp", "rpd"],
) -> tuple[ValidationIssue, ...]:
    """The existing zero prerequisites of the forecast arithmetic, located.

    The RPD model derives ``plateau rate = volume × ratio`` and then divides by
    the plateau rate (``log(minimum / plateau)``) and by the decline rate; the
    LTP model derives every rate from the volume. A zero volume, ratio or
    decline rate therefore reaches an undefined division/logarithm before any
    forecast exists. Only these demonstrated prerequisites are mirrored; a
    negative value stays accepted exactly as the existing worker accepts it.
    """
    if operation == "ltp":
        volume = _as_float(payload.get("volume"))
        if volume != 0:
            return ()
        return (
            ValidationIssue(
                code="LTP_VOLUME_ZERO_PREREQUISITE",
                severity="error",
                message=(
                    "Volume is 0. The LTP forecast derives every production rate "
                    "from the resource volume, so a zero volume cannot produce a "
                    "forecast."
                ),
                guidance=(
                    "Review Volume and enter the value that matches your intended "
                    "forecast, then try again."
                ),
                locations=(IssueLocation(("volume",)),),
                context={"field": "Volume", "actual": 0.0, "operation": "ltp"},
            ),
        )

    issues: list[ValidationIssue] = []
    volume = _as_float(payload.get("volume"))
    plateau_ratio = _as_float(payload.get("q_plateau_ratio"))
    minimum_ratio = _as_float(payload.get("q_min_ratio"))
    decline_rate = _as_float(payload.get("drate"))

    if volume == 0:
        issues.append(
            ValidationIssue(
                code="RPD_VOLUME_ZERO_PREREQUISITE",
                severity="error",
                message=(
                    "Volume is 0. The RPD forecast derives the plateau rate from "
                    "the volume and the decline period divides by that plateau "
                    "rate, so a zero volume cannot produce a forecast."
                ),
                guidance=(
                    "Review Volume and the RPD parameters, correct the value that "
                    "does not match your intended forecast, then try again."
                ),
                locations=(IssueLocation(("volume",)),),
                related_locations=(IssueLocation(("q_plateau_ratio",)),),
                context={
                    "field": "Volume",
                    "actual": 0.0,
                    "operation": "rpd",
                },
            )
        )
        return tuple(issues)

    if plateau_ratio == 0:
        issues.append(
            ValidationIssue(
                code="RPD_PLATEAU_RATIO_ZERO_PREREQUISITE",
                severity="error",
                message=(
                    "Ratio of plateau rate and volume is 0. The RPD forecast "
                    "computes the plateau rate as Volume × Ratio of plateau rate "
                    "and volume and then divides by that plateau rate, so a zero "
                    "ratio cannot produce a forecast."
                ),
                guidance=(
                    "Review Ratio of plateau rate and volume and the RPD "
                    "parameters, correct the value that does not match your "
                    "intended forecast, then try again."
                ),
                locations=(IssueLocation(("q_plateau_ratio",)),),
                related_locations=(IssueLocation(("volume",)),),
                context={
                    "field": "Ratio of plateau rate and volume",
                    "actual": 0.0,
                    "operation": "rpd",
                },
            )
        )

    if minimum_ratio == 0:
        issues.append(
            ValidationIssue(
                code="RPD_MIN_RATIO_ZERO_PREREQUISITE",
                severity="error",
                message=(
                    "Ratio of minimum rate is 0. The RPD decline period is "
                    "calculated from the logarithm of the minimum rate divided by "
                    "the plateau rate, and the minimum rate is Volume × Ratio of "
                    "minimum rate, so a zero ratio cannot produce a forecast."
                ),
                guidance=(
                    "Review Ratio of minimum rate and the RPD parameters, correct "
                    "the value that does not match your intended forecast, then "
                    "try again."
                ),
                locations=(IssueLocation(("q_min_ratio",)),),
                related_locations=(IssueLocation(("volume",)),),
                context={
                    "field": "Ratio of minimum rate",
                    "actual": 0.0,
                    "operation": "rpd",
                },
            )
        )

    if decline_rate == 0:
        issues.append(
            ValidationIssue(
                code="RPD_DECLINE_RATE_ZERO_PREREQUISITE",
                severity="error",
                message=(
                    "Decline rate is 0. The RPD decline period divides by the "
                    "decline rate, so a zero decline rate cannot produce a "
                    "forecast."
                ),
                guidance=(
                    "Review Decline rate and the RPD parameters, correct the value "
                    "that does not match your intended forecast, then try again."
                ),
                locations=(IssueLocation(("drate",)),),
                context={
                    "field": "Decline rate",
                    "actual": 0.0,
                    "operation": "rpd",
                },
            )
        )

    return tuple(issues)


def collect_forecast_period_issues(
    *,
    start_year: int,
    end_year: int,
    forecast_length: int,
    operation: Literal["ltp", "rpd"],
) -> tuple[ValidationIssue, ...]:
    """The existing forecast-length feasibility rule, as located issues.

    The LTP/RPD workers predict a profile of ``forecast_length`` years and then
    require the requested period to cover it: the last modelled year is
    ``start_year + forecast_length - 1``. A shorter End Year produces the same
    failure the worker already raises, reported with the actual and required end
    year instead of an opaque server error.
    """
    if forecast_length <= 0:
        return ()
    required_end = start_year + forecast_length - 1
    if end_year >= required_end:
        return ()
    label = "LTP" if operation == "ltp" else "RPD"
    if operation == "rpd":
        message = (
            f"The production period ends in {end_year}, but these RPD parameters "
            f"require a period ending in {required_end} or later."
        )
        guidance = "Review End Year and the RPD parameters, then try again."
    else:
        message = (
            f"The production period ends in {end_year}, but this LTP model "
            f"requires a period ending in {required_end} or later."
        )
        guidance = "Review End Year and the LTP parameters, then try again."
    return (
        ValidationIssue(
            code=f"{label}_FORECAST_PERIOD_TOO_SHORT",
            severity="error",
            message=message,
            guidance=guidance,
            locations=(IssueLocation(("end_year",)),),
            context={
                "field": "End Year",
                "actual_end_year": end_year,
                "required_end_year": required_end,
                "start_year": start_year,
                "forecast_length": forecast_length,
            },
        ),
    )


# ── Public ASR/LBT expenditure predicates ────────────────────────────────

_EXPENDITURE_TABLES: dict[str, tuple[str, str]] = {
    "asr_expenditures": ("asr", "ASR"),
    "lbt_expenditures": ("lbt", "LBT"),
}

_EXPENDITURE_ARRAY_FIELDS: tuple[str, ...] = (
    "expense_year",
    "cost",
    "cost_allocation",
    "cost_type",
    "description",
    "final_year",
    "future_rate",
    "tax_portion",
    "tax_discount",
    "utilized_land_area",
    "utilized_building_area",
    "njop_land",
    "njop_building",
    "gross_revenue",
)

_EXPENDITURE_FIELD_LABELS: dict[str, str] = {
    "expense_year": "Year",
    "cost": "Cost",
    "cost_allocation": "Cost Allocation",
    "cost_type": "Cost Type",
    "description": "Description",
    "final_year": "Final Year",
    "future_rate": "Future Rate",
    "tax_portion": "Tax Portion",
    "tax_discount": "Tax Discount",
    "utilized_land_area": "Utilized Land Area",
    "utilized_building_area": "Utilized Building Area",
    "njop_land": "NJOP Land",
    "njop_building": "NJOP Building",
    "gross_revenue": "Gross Revenue",
}

_EXPENDITURE_SECTION_LABELS: dict[str, str] = {
    "asr": "ASR Costs",
    "lbt": "LBT Costs",
}


def normalize_expenditure_input(
    data: Mapping[str, object],
    operation: Literal["asr_expenditures", "lbt_expenditures"],
) -> dict[str, object]:
    """Adapt the public flat expenditure request to the worker's input shape.

    The public ``ASRExpendituresBM``/``LBTExpendituresBM`` models accept a flat
    ``start_date``/``end_date``/cost-map request, while the existing
    ``get_asr_expenditures``/``get_lbt_expenditures`` workers consume a
    pseudo-project with a ``setup`` block, an unchanged ASR/LBT cost map and no
    unrelated costs. A request that already carries the internal ``setup`` shape
    is recognized and returned unchanged, preserving accepted library input.
    """
    setup = data.get("setup")
    if isinstance(setup, Mapping):
        return dict(data)

    normalized: dict[str, object] = {
        "setup": {
            "start_date": data.get("start_date", "01/01/2010"),
            "end_date": data.get("end_date", "01/01/2010"),
            "oil_onstream_date": None,
            "gas_onstream_date": None,
            "approval_year": None,
            "is_pod_1": False,
        },
        "lifting": None,
        "capital": None,
        "intangible": None,
        "opex": None,
        "asr": None,
        "lbt": None,
        "cost_of_sales": None,
    }
    if operation == "asr_expenditures":
        # The worker consumes a map of named cost tables; wrap the unchanged
        # public flat block under one table name.
        normalized["asr"] = {"ASR": data.get("asr")}
    else:
        normalized["lbt"] = {"LBT": data.get("lbt")}
    return normalized


def _sequence_length(value: object) -> int | None:
    """Return a list/tuple length, or ``None`` when the value is not one."""
    if isinstance(value, (list, tuple)):
        return len(cast("list[object] | tuple[object, ...]", value))
    return None


def _expenditure_block(
    view: Mapping[str, object], operation: str
) -> Mapping[str, object] | None:
    """Return the validated expenditure block from the validation view."""
    section, table = _EXPENDITURE_TABLES[operation]
    section_value = _as_mapping(view.get(section))
    if section_value is None:
        return None
    return _as_mapping(section_value.get(table))


def _expenditure_date_issues(
    view: Mapping[str, object], map_path: _PathMap
) -> tuple[ValidationIssue, ...]:
    """The existing project-date predicates with expenditure-aware copy."""
    _, date_issues = collect_project_date_issues(view)
    results: list[ValidationIssue] = []
    for issue in date_issues:
        if issue.code == "DATE_FORMAT_INVALID":
            field = "Start" if issue.locations[0].path[-1] == "start_date" else "End"
            raw = issue.context.get("entered_value")
            results.append(
                replace(
                    issue,
                    message=(
                        f"Project {field} Date contains '{raw}'. Enter a valid date "
                        "in the accepted format, for example 01/01/2010."
                    ),
                    guidance=(
                        "Review the project dates, then select Check again."
                    ),
                )
            )
        else:
            results.append(issue)
    return tuple(_remap_issue(issue, map_path) for issue in results)


def _expenditure_block_issues(
    block: Mapping[str, object],
    *,
    section: str,
    table: str,
    map_path: _PathMap,
) -> tuple[ValidationIssue | SkippedCheck, ...]:
    """Mirror the existing ASR/LBT row predicates on one submitted block.

    Only rules the workers already enforce are mirrored: a reversed component
    period, unequal row-array lengths and expense years outside the block
    period. A blocked shape reports its dependent-checks skip explicitly.
    """
    section_label = _EXPENDITURE_SECTION_LABELS.get(section, section)
    table_path = map_path((section, table))
    skip_scope = f"{section}:{table}:row_checks"
    results: list[ValidationIssue | SkippedCheck] = []

    if period_is_reversed(block.get("start_year"), block.get("end_year")):
        results.append(
            ValidationIssue(
                code="COMPONENT_PERIOD_REVERSED",
                severity="error",
                message=(
                    f"{section_label} start year is after its end year. Correct the "
                    "start and end years, then select Check again."
                ),
                guidance=(
                    f"Review the start and end years in {section_label}, then select "
                    "Check again."
                ),
                locations=(IssueLocation(table_path),),
                context={
                    "section": section,
                    "start_year": _as_int(block.get("start_year")),
                    "end_year": _as_int(block.get("end_year")),
                },
            )
        )
        results.append(
            SkippedCheck(
                scope=skip_scope,
                reason=(
                    f"Some checks could not run because {section_label} start or end "
                    "year needs correction. Fix the listed issue, then select "
                    "Check again."
                ),
                blocking_paths=(IssueLocation(table_path),),
            )
        )
        return tuple(results)

    lengths: dict[str, int] = {}
    for field in _EXPENDITURE_ARRAY_FIELDS:
        count = _sequence_length(block.get(field))
        if count is not None:
            lengths[field] = count
    mismatch = unequal_lengths(lengths)
    if mismatch is not None:
        ordered = sorted(
            mismatch.items(),
            key=lambda item: (
                item[1],
                _EXPENDITURE_ARRAY_FIELDS.index(item[0]),
            ),
        )
        named = "; ".join(
            f"{_EXPENDITURE_FIELD_LABELS.get(field, field)} has {count} "
            + f"entr{'y' if count == 1 else 'ies'}"
            for field, count in ordered
        )
        results.append(
            ValidationIssue(
                code="ARRAY_LENGTH_MISMATCH",
                severity="error",
                message=f"{named}. Each cost row needs its matching year.",
                guidance=(
                    f"Open {section_label} and correct the entered rows so every cost "
                    "entry has one matching year, then select Check again."
                ),
                locations=tuple(
                    IssueLocation(map_path((section, table, field)))
                    for field, _ in ordered
                ),
                context={"section": section, "lengths": dict(mismatch)},
            )
        )
        results.append(
            SkippedCheck(
                scope=skip_scope,
                reason=(
                    f"Some checks could not run because {section_label} rows do not "
                    "have matching values. Fix the listed issue, then select "
                    "Check again."
                ),
                blocking_paths=(IssueLocation(table_path),),
            )
        )
        return tuple(results)

    expense_years = block.get("expense_year")
    start = _as_int(block.get("start_year"))
    end = _as_int(block.get("end_year"))
    if isinstance(expense_years, list) and start is not None and end is not None:
        values = cast("list[object]", expense_years)
        low = year_below(values, start)
        high = year_above(values, end)
        if low is not None or high is not None:
            actual = low if low is not None else high
            index = 0
            for position, value in enumerate(values):
                if _as_int(value) == actual:
                    index = position
                    break
            results.append(
                ValidationIssue(
                    code="EXPENSE_YEAR_OUTSIDE_PROJECT",
                    severity="error",
                    message=(
                        f"Year is {actual}; the project runs from {start} to {end}. "
                        "Correct Year if this cost belongs within that period, or "
                        "review the project dates in General Configuration."
                    ),
                    guidance=(
                        f"Review the marked row in {section_label} and the project "
                        "dates in General Configuration, then select Check again."
                    ),
                    locations=(
                        IssueLocation(
                            map_path((section, table, "expense_year", index))
                        ),
                    ),
                    related_locations=(
                        IssueLocation(map_path((section, table, "start_year"))),
                        IssueLocation(map_path((section, table, "end_year"))),
                    ),
                    context={
                        "field": "Year",
                        "actual_year": actual,
                        "start_year": start,
                        "end_year": end,
                    },
                )
            )
    return tuple(results)


def _remap_issue(issue: ValidationIssue, map_path: _PathMap) -> ValidationIssue:
    """Rewrite an issue's locations through the submitted-path mapping."""
    return replace(
        issue,
        locations=tuple(IssueLocation(map_path(loc.path)) for loc in issue.locations),
        related_locations=tuple(
            IssueLocation(map_path(loc.path)) for loc in issue.related_locations
        ),
    )


def _collect_expenditure_issues(
    payload: Mapping[str, object],
    operation: Literal["asr_expenditures", "lbt_expenditures"],
) -> tuple[ValidationIssue | SkippedCheck, ...]:
    """Locate the demonstrated public ASR/LBT input inconsistencies.

    The public flat request is converted into a read-only validation view that
    mirrors the worker's internal shape, then the shared predicates run against
    it. Locations are mapped back to the submitted public paths; an already
    internal (nested ``setup``) request keeps its nested paths.
    """
    section = _EXPENDITURE_TABLES[operation][0]
    internal = isinstance(payload.get("setup"), Mapping)
    view = dict(payload) if internal else normalize_expenditure_input(payload, operation)

    def map_path(path: _Path) -> _Path:
        if internal:
            return path
        if path == ("setup", "start_date"):
            return ("start_date",)
        if path == ("setup", "end_date"):
            return ("end_date",)
        if len(path) >= 2 and path[0] == section:
            return (section, *path[2:])
        return path

    issues = _expenditure_date_issues(view, map_path)
    results: list[ValidationIssue | SkippedCheck] = list(issues)
    if issues:
        results.append(
            SkippedCheck(
                scope=f"{section}:{_EXPENDITURE_TABLES[operation][1]}:row_checks",
                reason=(
                    "Some checks could not run because the project dates need "
                    "correction. Fix the listed issue, then select Check again."
                ),
                blocking_paths=tuple(
                    IssueLocation(map_path(loc.path))
                    for issue in issues
                    for loc in issue.locations
                ),
            )
        )
        return tuple(results)

    block = _expenditure_block(view, operation)
    if block is None:
        return tuple(results)
    results.extend(
        _expenditure_block_issues(
            block,
            section=section,
            table=_EXPENDITURE_TABLES[operation][1],
            map_path=map_path,
        )
    )
    return tuple(results)


def collect_operation_issues(
    payload: Mapping[str, object],
    operation: StandaloneOperation,
) -> tuple[ValidationIssue | SkippedCheck, ...]:
    """Return the located input issues for one standalone operation."""
    if operation in ("asr_expenditures", "lbt_expenditures"):
        return _collect_expenditure_issues(payload, operation)

    if operation in ("ltp", "rpd"):
        issues: list[ValidationIssue] = []
        if operation == "ltp":
            issues.extend(_ltp_fluid_issues(payload))
        issues.extend(collect_forecast_prerequisite_issues(payload, operation))

        start = _as_int(payload.get("start_year"))
        end = _as_int(payload.get("end_year"))
        if start is not None and end is not None and start > end:
            code = "LTP_PERIOD_REVERSED" if operation == "ltp" else "RPD_PERIOD_REVERSED"
            issues.append(
                ValidationIssue(
                    code=code,
                    severity="error",
                    message=(
                        f"Start Year is {start}; End Year is {end}. End Year must be "
                        "the same as or later than Start Year."
                    ),
                    guidance=(
                        "Correct the year that does not match your intended "
                        "production period, then try again."
                    ),
                    locations=(
                        IssueLocation(("start_year",)),
                        IssueLocation(("end_year",)),
                    ),
                    context={
                        "field": "Start Year / End Year",
                        "start_year": start,
                        "end_year": end,
                    },
                )
            )
        return tuple(issues)

    if operation == "econlimit":
        years_length = _series_length(payload.get("years"))
        cash_flow_length = _series_length(payload.get("cash_flow"))
        if years_length == 0 and cash_flow_length == 0:
            return (
                ValidationIssue(
                    code="ECONLIMIT_SERIES_EMPTY",
                    severity="error",
                    message=(
                        "No year and cash-flow entries were provided. Enter the "
                        "cash-flow series with its matching years, then try again."
                    ),
                    guidance=(
                        "Enter the matching year and cash-flow entries, then try again."
                    ),
                    locations=(
                        IssueLocation(("years",)),
                        IssueLocation(("cash_flow",)),
                    ),
                    context={"field": "Years / Cash Flow"},
                ),
            )
        if years_length != cash_flow_length:
            return (
                ValidationIssue(
                    code="ECONLIMIT_SERIES_LENGTH_MISMATCH",
                    severity="error",
                    message=(
                        f"There are {years_length} year entries and {cash_flow_length} "
                        "cash-flow entries. Each cash-flow entry needs one matching year."
                    ),
                    guidance=(
                        "Correct the series so the years and cash-flow entries match, "
                        "then try again."
                    ),
                    locations=(
                        IssueLocation(("years",)),
                        IssueLocation(("cash_flow",)),
                    ),
                    context={
                        "field": "Years / Cash Flow",
                        "year_count": years_length,
                        "cash_flow_count": cash_flow_length,
                    },
                ),
            )
        return ()
