"""Shared engine input-validation predicates.

These predicates consume the normalized values of the existing engine checks
and construct structured :class:`ValidationIssue` objects with source
locations. They do not add new economic constraints; each predicate mirrors an
existing engine rule and its accepted boundary semantics.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Literal, cast

from pyscnomics.input_validation import (
    IssueLocation,
    JsonValue,
    Severity,
    SkippedCheck,
    ValidationCollector,
    ValidationIssue,
)

ContractType = Literal["Base Project", "Cost Recovery", "Gross Split", "Transition"]

# Sections that the Gross Split POD I rule classifies as non-capital costs.
NON_CAPITAL_SECTIONS: tuple[str, ...] = (
    "intangible",
    "opex",
    "asr",
    "lbt",
    "cost_of_sales",
)

# All cost sections that may carry a ``cost_type`` classification.
COST_SECTIONS: tuple[str, ...] = ("capital",) + NON_CAPITAL_SECTIONS

_SECTION_LABELS: dict[str, str] = {
    "capital": "Capital Costs",
    "intangible": "Intangible Costs",
    "opex": "OPEX Costs",
    "asr": "ASR Costs",
    "lbt": "LBT Costs",
    "cost_of_sales": "Cost of Sales",
}

_CLASSIFICATION_LABELS: dict[str, str] = {
    "sunk_cost": "Sunk Cost",
    "preonstream_cost": "Preonstream Cost",
    "postonstream_cost": "Postonstream Cost",
}

_POD1_RULE_CODE: dict[str, str] = {
    "intangible": "GS_POD1_NONCAPITAL_INTANGIBLE",
    "opex": "GS_POD1_NONCAPITAL_OPEX",
    "asr": "GS_POD1_NONCAPITAL_ASR",
    "lbt": "GS_POD1_NONCAPITAL_LBT",
    "cost_of_sales": "GS_POD1_NONCAPITAL_COS",
}

_NONCAPITAL_COST_TYPES = {"sunk_cost", "preonstream_cost"}


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return None


def _as_sequence(value: object) -> list[object] | None:
    if isinstance(value, (list, tuple)):
        return list(cast("list[object] | tuple[object, ...]", value))
    return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _normalized_text(value: object) -> str | None:
    if isinstance(value, str):
        return value.strip().lower()
    return None


# ── Shared pure structural predicates ───────────────────────────────────
#
# These are the dependency-light decision helpers that BOTH the real engine
# construction/check sites and the API preflight call. They never import the
# engine modules, so the engine can depend on them without a cycle, and the
# preflight and the engine cannot drift apart. Each returns a plain value (or
# ``None`` when there is no violation) so the caller owns the message and the
# exception/issue type.


def parse_engine_date(value: object) -> date | None:
    """Parse a date using the engine adapter's exact accepted formats.

    Strings must be ``%d/%m/%Y`` and integers are unix timestamps, mirroring
    ``api.converter.convert_str_to_date``. Anything else returns ``None``.
    This deliberately does not search arbitrary text for a four-digit year, so
    a malformed date is reported as malformed instead of silently interpreted.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        try:
            return datetime.fromtimestamp(value).date()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%d/%m/%Y").date()
        except ValueError:
            return None
    return None


def date_year(value: object) -> int | None:
    """Return the year of an engine-parsable date, else ``None``."""
    parsed = parse_engine_date(value)
    return parsed.year if parsed is not None else None


def period_is_reversed(start_year: object, end_year: object) -> bool:
    """True when a project or component period starts after it ends."""
    start = _as_int(start_year)
    end = _as_int(end_year)
    if start is None or end is None:
        return False
    return start > end


def dates_reversed(start: date | None, end: date | None) -> bool:
    """True when a start date is after its end date (full date comparison)."""
    if start is None or end is None:
        return False
    return start > end


def unequal_lengths(named_lengths: Mapping[str, int]) -> dict[str, int] | None:
    """Return the length map when entries disagree, else ``None``.

    ``None`` means every named array shares one length (the engine accepts it).
    A returned mapping carries the exact per-field lengths for reporting.
    """
    lengths = dict(named_lengths)
    if not lengths:
        return None
    first = next(iter(lengths.values()))
    if all(length == first for length in lengths.values()):
        return None
    return lengths


def year_below(years: Iterable[object], minimum: int) -> int | None:
    """Return the smallest integer year below ``minimum``, else ``None``."""
    numeric = [value for value in (_as_int(year) for year in years) if value is not None]
    if not numeric:
        return None
    low = min(numeric)
    return low if low < minimum else None


def year_above(years: Iterable[object], maximum: int) -> int | None:
    """Return the largest integer year above ``maximum``, else ``None``."""
    numeric = [value for value in (_as_int(year) for year in years) if value is not None]
    if not numeric:
        return None
    high = max(numeric)
    return high if high > maximum else None


def missing_value_indices(values: Iterable[object]) -> list[int]:
    """Return indices whose value is ``None`` or ``NaN``."""
    missing: list[int] = []
    for index, value in enumerate(values):
        if value is None:
            missing.append(index)
        elif isinstance(value, float) and value != value:
            missing.append(index)
    return missing


def out_of_range_indices(
    values: Iterable[object], low: float, high: float
) -> list[int]:
    """Return indices whose numeric value falls outside ``[low, high]``."""
    outside: list[int] = []
    for index, value in enumerate(values):
        numeric = _as_float(value)
        if numeric is None:
            continue
        if numeric < low or numeric > high:
            outside.append(index)
    return outside


def any_below(values: Iterable[object], reference: Iterable[object]) -> bool:
    """True when any paired ``value`` is strictly below its ``reference``."""
    for value, limit in zip(values, reference):
        numeric = _as_float(value)
        limit_numeric = _as_float(limit)
        if numeric is None or limit_numeric is None:
            continue
        if numeric < limit_numeric:
            return True
    return False


def magnitude_exceeds(values: Iterable[object], reference: Iterable[object]) -> bool:
    """True when any ``abs(value)`` exceeds the paired ``abs(reference)``.

    Mirrors the capital ``salvage_value`` magnitude rule.
    """
    for value, limit in zip(values, reference):
        numeric = _as_float(value)
        limit_numeric = _as_float(limit)
        if numeric is None or limit_numeric is None:
            continue
        if abs(numeric) > abs(limit_numeric):
            return True
    return False


def lifting_exceeds_production(
    lifting: Iterable[object], production: Iterable[object]
) -> bool:
    """True when any lifting value is greater than its production value.

    Mirrors ``Lifting.__post_init__`` (``prod_rate < lifting_rate``).
    """
    for lift, prod in zip(lifting, production):
        lift_value = _as_float(lift)
        prod_value = _as_float(prod)
        if lift_value is None or prod_value is None:
            continue
        if prod_value < lift_value:
            return True
    return False


def inconsistent_component_years(
    project_year: int, component_years: Iterable[object]
) -> list[int]:
    """Return component years that do not equal the project's boundary year."""
    return [
        value
        for value in (_as_int(year) for year in component_years)
        if value is not None and value != project_year
    ]


# ── Shared classification / approval / PIS predicates (Task 2B) ─────────
#
# These mirror the existing engine rules for cost-type periods, boundary years,
# approval-year bounds, sunk-cost cutoff, the post-onstream window, capital PIS
# years and the CR/GS attribute ranges. The engine check sites and the API
# preflight call the same helpers so the two cannot drift apart.

CLASSIFICATION_SUNK = "sunk_cost"
CLASSIFICATION_PREONSTREAM = "preonstream_cost"
CLASSIFICATION_POSTONSTREAM = "postonstream_cost"
CLASSIFICATION_ORDER: tuple[str, str, str] = (
    CLASSIFICATION_SUNK,
    CLASSIFICATION_PREONSTREAM,
    CLASSIFICATION_POSTONSTREAM,
)

# Classifications the engine forbids inside each strict-inequality period.
_INVALID_CLASSIFICATIONS_BY_PERIOD: dict[str, tuple[str, ...]] = {
    CLASSIFICATION_SUNK: (CLASSIFICATION_POSTONSTREAM,),
    CLASSIFICATION_PREONSTREAM: (CLASSIFICATION_POSTONSTREAM,),
    CLASSIFICATION_POSTONSTREAM: (CLASSIFICATION_SUNK, CLASSIFICATION_PREONSTREAM),
}


def default_cost_type_for_year(
    expense_year: int, *, approval_year: int, onstream_year: int, is_pod_1: bool
) -> str:
    """Return the engine's default classification for an unset cost type.

    Mirrors ``BaseProject._assign_default_cost_types``: before approval a POD I
    row defaults to sunk and a non-POD I row to pre-onstream; between approval
    and onstream it defaults to pre-onstream; from onstream onward it defaults
    to post-onstream.
    """
    if expense_year < approval_year:
        return CLASSIFICATION_SUNK if is_pod_1 else CLASSIFICATION_PREONSTREAM
    if expense_year < onstream_year:
        return CLASSIFICATION_PREONSTREAM
    return CLASSIFICATION_POSTONSTREAM


def effective_classification(
    raw: object,
    *,
    expense_year: int | None,
    approval_year: int,
    onstream_year: int,
    is_pod_1: bool,
) -> str | None:
    """Return the submitted classification, or the engine default when unset.

    An explicit submitted value is always preserved; the default is only used
    for a genuinely unset (``None``/blank) classification with a known year.
    """
    explicit = _normalized_text(raw)
    if explicit is not None:
        return explicit
    if expense_year is None:
        return None
    return default_cost_type_for_year(
        expense_year,
        approval_year=approval_year,
        onstream_year=onstream_year,
        is_pod_1=is_pod_1,
    )


def classification_period(
    expense_year: int, *, approval_year: int, onstream_year: int
) -> str | None:
    """Return the engine's consistency period, or ``None`` at a boundary year.

    Mirrors ``_validate_cost_type_consistency``: the comparison is strict on
    both ends, so the approval year and the onstream year belong to no period
    and are governed by the boundary-year rule instead.
    """
    if expense_year < approval_year:
        return CLASSIFICATION_SUNK
    if approval_year < expense_year < onstream_year:
        return CLASSIFICATION_PREONSTREAM
    if expense_year > onstream_year:
        return CLASSIFICATION_POSTONSTREAM
    return None


def classification_allowed_in_period(period: str, classification: str) -> bool:
    """True when the classification is permitted in the inferred period."""
    return classification not in _INVALID_CLASSIFICATIONS_BY_PERIOD.get(period, ())


def boundary_year_classification_violation(
    expense_year: int,
    classification: str,
    *,
    approval_year: int,
    onstream_year: int,
) -> str | None:
    """Return the violated boundary-year rule, or ``None`` when valid.

    Mirrors ``_validate_boundary_year_cost_types``: post-onstream is not
    permitted at the approval year and sunk is not permitted at the onstream
    year.
    """
    if expense_year == approval_year and classification == CLASSIFICATION_POSTONSTREAM:
        return "approval_year"
    if expense_year == onstream_year and classification == CLASSIFICATION_SUNK:
        return "onstream_year"
    return None


def approval_year_violation(
    approval_year: int, *, start_year: int, end_year: int, onstream_year: int
) -> str | None:
    """Return the violated approval-year rule, or ``None`` when valid.

    Mirrors ``BaseProject._validate_approval_year``: the approval year must lie
    within the project period and not fall after the earliest onstream year.
    """
    if approval_year < start_year:
        return "before_project"
    if approval_year > end_year:
        return "after_project"
    if approval_year > onstream_year:
        return "after_onstream"
    return None


def sunk_cost_cutoff_year(*, approval_year: object, onstream_year: object) -> int:
    """The engine's sunk-cost cutoff: the earlier of approval and onstream."""
    approval = _as_int(approval_year)
    onstream = _as_int(onstream_year)
    if approval is None or onstream is None:
        return 0
    return min(approval, onstream)


def sunk_cost_year_exceeds_cutoff(expense_year: object, cutoff_year: object) -> bool:
    """True when a sunk-cost year is after the cutoff (strictly greater)."""
    year = _as_int(expense_year)
    cutoff = _as_int(cutoff_year)
    if year is None or cutoff is None:
        return False
    return year > cutoff


def postonstream_year_before_onstream(expense_year: object, onstream_year: object) -> bool:
    """True when a post-onstream year precedes the effective onstream year."""
    year = _as_int(expense_year)
    onstream = _as_int(onstream_year)
    if year is None or onstream is None:
        return False
    return year < onstream


def nonzero_indices(costs: Iterable[object]) -> list[int]:
    """Return indices whose numeric cost is non-zero (engine ``cost != 0``)."""
    indices: list[int] = []
    for index, value in enumerate(costs):
        numeric = _as_float(value)
        if numeric is None:
            continue
        if numeric != 0.0:
            indices.append(index)
    return indices


def pis_before_onstream_indices(
    pis_years: Iterable[object], costs: Iterable[object], onstream_year: int
) -> list[int]:
    """Indices whose PIS year precedes onstream and whose cost is non-zero."""
    indices: list[int] = []
    for index, (raw_pis, raw_cost) in enumerate(zip(pis_years, costs)):
        pis = _as_int(raw_pis)
        if pis is None or pis >= onstream_year:
            continue
        cost = _as_float(raw_cost)
        if cost is None or abs(cost) == 0.0:
            continue
        indices.append(index)
    return indices


def pis_after_onstream_indices(
    pis_years: Iterable[object], onstream_year: int
) -> list[int]:
    """Indices whose PIS year is strictly after the onstream year."""
    indices: list[int] = []
    for index, raw_pis in enumerate(pis_years):
        pis = _as_int(raw_pis)
        if pis is not None and pis > onstream_year:
            indices.append(index)
    return indices


def fraction_value_out_of_range(value: object, low: float, high: float) -> bool:
    """True when a numeric fraction falls outside the inclusive range."""
    numeric = _as_float(value)
    if numeric is None:
        return False
    return numeric < low or numeric > high


def length_mismatch(length: int, expected: int) -> bool:
    """True when an array length does not equal the expected project length."""
    return length != expected


def length_exceeds_project(length: int, project_duration: int) -> bool:
    """True when an array is longer than the project duration."""
    return length > project_duration


def contains_negative(values: Iterable[object]) -> bool:
    """True when any numeric value is strictly negative."""
    for value in values:
        numeric = _as_float(value)
        if numeric is not None and numeric < 0:
            return True
    return False


def condition_dict_is_mixed_form(raw: object) -> bool:
    """True when a condition dict mixes normalized bands with column arrays."""
    mapping = _as_mapping(raw)
    if mapping is None or not mapping:
        return False
    first = next(iter(mapping.values()))
    if not isinstance(first, Mapping):
        return False
    return any(not isinstance(value, Mapping) for value in mapping.values())


def cum_split_offset_length_mismatch(length: int, project_duration: int) -> bool:
    """True when a cumulative-split offset array is not one value per year."""
    return length != project_duration


# ── Shared Gross Split variable-split completeness (Task 2B follow-up) ──
#
# The engine's variable-split calculation (``_get_var_split_08_2017`` /
# ``_get_var_split_52_2017`` / ``_get_var_split_13_2024``) builds one coefficient
# per active-regime component and raises when any component is ``None``
# (``_require_variable_split_value``). These helpers express exactly which
# components the active regime requires so the engine and the preflight agree.

# Components required by PERMEN ESDM No. 8/2017, 52/2017, 20/2019 and 12/2020.
GS_VARIABLE_SPLIT_COMPONENTS_2017: tuple[str, ...] = (
    "field_status",
    "field_loc",
    "res_depth",
    "infra_avail",
    "res_type",
    "api_oil",
    "domestic_use",
    "prod_stage",
    "co2_content",
    "h2s_content",
)
# Components required by PERMEN ESDM No. 13/2024 for a conventional reservoir.
GS_VARIABLE_SPLIT_COMPONENTS_2024: tuple[str, ...] = (
    "field_loc_2024",
    "infra_avail_2024",
    "field_reserves_2024",
)
# Short regime codes as they appear in the API payload / enum names.
GS_REGIME_2017_CODES: frozenset[str] = frozenset(
    {
        "PERMEN_ESDM_8_2017",
        "PERMEN_ESDM_52_2017",
        "PERMEN_ESDM_20_2019",
        "PERMEN_ESDM_12_2020",
    }
)
# Regimes whose variable-split completeness the engine and the preflight enforce.
# The current regulation (PERMEN ESDM No. 13/2024) is enforced; the legacy
# 2017 family can be added here once its legacy null-selector payloads are
# migrated. ``variable_split_required_components`` above stays regime-generic.
GS_ENFORCED_REGIME_CODES: frozenset[str] = frozenset({"PERMEN_ESDM_13_2024"})


def _normalized_regime_code(value: object) -> str | None:
    """Return the short regime code for a string or ``GrossSplitRegime`` enum."""
    if isinstance(value, Enum):
        return value.name.upper()
    if isinstance(value, str):
        return value.strip().upper()
    return None


def _normalized_reservoir_type(value: object) -> str | None:
    """Return ``conventional`` / ``unconventional`` for a string or enum."""
    if isinstance(value, Enum):
        raw = cast("object", value.value)
        if isinstance(raw, str):
            return raw.strip().lower()
        return value.name.strip().lower()
    if isinstance(value, str):
        return value.strip().lower()
    return None


def variable_split_required_components(
    regime: object, reservoir_type: object
) -> tuple[str, ...]:
    """Return the variable-split components the active regime requires.

    Mirrors the engine's ``_wrapper_variable_split`` dispatch: the 2017-family
    regimes require the ten field/reservoir components, and PERMEN ESDM 13/2024
    requires the three 2024 components for a conventional reservoir (an
    unconventional reservoir uses a fixed factor and needs none). An
    unrecognized regime returns no requirement; the engine/converter owns that
    dispatch.
    """
    code = _normalized_regime_code(regime)
    if code == "PERMEN_ESDM_13_2024":
        if _normalized_reservoir_type(reservoir_type) == "unconventional":
            return ()
        return GS_VARIABLE_SPLIT_COMPONENTS_2024
    if code in GS_REGIME_2017_CODES:
        return GS_VARIABLE_SPLIT_COMPONENTS_2017
    return ()


def variable_split_enforced(regime: object) -> bool:
    """Whether the completeness rule is enforced for the active regime."""
    return _normalized_regime_code(regime) in GS_ENFORCED_REGIME_CODES


def _component_is_missing(value: object) -> bool:
    """True when a component is unset (``None``) or an empty string."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def missing_variable_split_components(
    components: Mapping[str, object],
) -> list[str]:
    """Return the required component names that are missing, in order."""
    return [name for name, value in components.items() if _component_is_missing(value)]


@dataclass(frozen=True)
class EffectiveOnstream:
    """Result of the shared effective-onstream resolver.

    ``source`` is ``"revenue"`` (inferred from the earliest positive revenue
    year), ``"provided"`` (a supplied date was validated), or ``"project_end"``
    (no positive revenue exists, so the engine falls back to the project end).
    """

    year: int
    inferred: bool
    source: str


@dataclass(frozen=True)
class OnstreamInputs:
    """The API-prepared inputs the shared onstream resolver needs.

    ``revenue`` is aligned to the project years starting at the project start
    year (the engine's ``Lifting.revenue()`` alignment) and ``lifting_prod_years``
    carries the actual lifting production years.
    """

    revenue: tuple[float, ...]
    lifting_prod_years: tuple[int, ...]


class OnstreamResolutionError(ValueError):
    """Raised when a supplied onstream year violates the engine rule.

    ``code`` is a stable rule identifier the preflight maps to an issue code.
    """

    code: str
    message: str
    context: dict[str, JsonValue]

    def __init__(
        self,
        code: str,
        message: str,
        *,
        context: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = dict(context or {})


def resolve_onstream_year(
    *,
    fluid: str,
    provided_year: int | None,
    start_year: int,
    end_year: int,
    revenue: Iterable[object],
    lifting_prod_years: Iterable[object],
) -> EffectiveOnstream:
    """Resolve the effective onstream year exactly as ``BaseProject`` does.

    ``revenue`` is aligned to the project years starting at ``start_year`` (the
    engine's ``Lifting.revenue()`` alignment). Rules:

    * positive revenue and no supplied date → earliest positive-revenue year;
    * positive revenue and a supplied date → the year must be within the project
      period and equal to either the earliest lifting year or the earliest
      revenue year;
    * no positive revenue → the project end year, regardless of any supplied
      date.
    """
    positive = [
        index for index, value in enumerate(revenue) if (_as_float(value) or 0.0) > 0.0
    ]
    if positive:
        earliest_revenue_year = start_year + positive[0]
        if provided_year is None:
            return EffectiveOnstream(earliest_revenue_year, True, "revenue")

        if not (start_year <= provided_year <= end_year):
            raise OnstreamResolutionError(
                "ONSTREAM_YEAR_OUTSIDE_PROJECT",
                f"{fluid.capitalize()} onstream year ({provided_year}) must be within "
                + f"project period ({start_year}-{end_year}).",
                context={"provided_year": provided_year},
            )

        lifting_years = [
            value
            for value in (_as_int(year) for year in lifting_prod_years)
            if value is not None
        ]
        earliest_lifting_year = min(lifting_years) if lifting_years else provided_year
        allowed_years = sorted({earliest_lifting_year, earliest_revenue_year})
        if provided_year not in allowed_years:
            raise OnstreamResolutionError(
                "ONSTREAM_YEAR_NOT_ALLOWED",
                f"Invalid {fluid} onstream year ({provided_year}). "
                + f"Allowed values: {allowed_years}.",
                context={
                    "provided_year": provided_year,
                    "allowed_years": cast(JsonValue, allowed_years),
                },
            )
        return EffectiveOnstream(provided_year, False, "provided")

    return EffectiveOnstream(end_year, False, "project_end")


class TransitionSchemeResolutionError(ValueError):
    """Raised when a transition child block does not select exactly one scheme.

    ``selected_schemes`` names every scheme that was submitted (both, or none).
    """

    code: str = "TRANSITION_SCHEME_UNRECOGNIZED"
    selected_schemes: tuple[str, ...]

    def __init__(
        self, selected_schemes: tuple[str, ...], *, message: str | None = None
    ) -> None:
        super().__init__(message or "Contract type is not recognized")
        self.selected_schemes = selected_schemes


def resolve_transition_scheme(
    block: Mapping[str, object],
) -> Literal["Cost Recovery", "Gross Split"]:
    """Resolve one transition child's scheme with the execution predicate.

    Execution accepts exactly one non-null scheme block; both non-null and
    neither are rejected. This never prefers Gross Split merely because its block
    is present.
    """
    has_cr = block.get("costrecovery") is not None
    has_gs = block.get("grosssplit") is not None
    selected = tuple(
        name
        for name, present in (("Cost Recovery", has_cr), ("Gross Split", has_gs))
        if present
    )
    if len(selected) != 1:
        raise TransitionSchemeResolutionError(selected)
    return cast('Literal["Cost Recovery", "Gross Split"]', selected[0])


def _row_cost(section: str, row: Mapping[str, object], index: int) -> float:
    """Return the engine-equivalent per-row cost for a submitted cost table."""
    if section == "opex":
        fixed = _as_float(_item(row.get("fixed_cost"), index)) or 0.0
        prod_rate = _as_float(_item(row.get("prod_rate"), index)) or 0.0
        per_volume = _as_float(_item(row.get("cost_per_volume"), index)) or 0.0
        return fixed + prod_rate * per_volume
    return _as_float(_item(row.get("cost"), index)) or 0.0


def _item(value: object, index: int) -> object:
    seq = _as_sequence(value)
    if seq is None or index >= len(seq):
        return None
    return seq[index]


def _fluid_label(value: object) -> str:
    text = _normalized_text(value)
    if text == "gas":
        return "Gas"
    if text == "oil":
        return "Oil"
    return "Oil"


def _path(prefix: tuple[str | int, ...], *rest: str | int) -> tuple[str | int, ...]:
    return prefix + tuple(rest)


def collect_grosssplit_pod1_noncapital_issues(
    payload: Mapping[str, object],
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Collect Gross Split POD I non-capital sunk/preonstream classification issues.

    Mirrors ``GrossSplit._check_invalid_non_capital_costs``: in a POD I Gross
    Split contract, nonzero non-capital costs classified as ``sunk_cost`` or
    ``preonstream_cost`` are not permitted. Both cost categories and both
    fluids are aggregated here instead of stopping at the first violation.
    """
    setup = _as_mapping(payload.get("setup"))
    if setup is None:
        return ()
    is_pod_1 = setup.get("is_pod_1")
    if is_pod_1 is not True:
        return ()

    # Missing settings and new cases default to strict mode; loose mode returns
    # the same applicable violations as warnings.
    severity: Severity = "warning" if setup.get("is_strict") is False else "error"

    groups: dict[tuple[str, str, str], list[tuple[str, int, int, float]]] = {}

    for section in NON_CAPITAL_SECTIONS:
        tables = _as_mapping(payload.get(section))
        if tables is None:
            continue
        for table_name, row_value in tables.items():
            row = _as_mapping(row_value)
            if row is None:
                continue
            cost_types = _as_sequence(row.get("cost_type"))
            allocations = _as_sequence(row.get("cost_allocation"))
            expense_years = _as_sequence(row.get("expense_year"))
            if cost_types is None:
                continue
            for index, raw_type in enumerate(cost_types):
                classification = _normalized_text(raw_type)
                if classification not in _NONCAPITAL_COST_TYPES:
                    continue
                if _row_cost(section, row, index) == 0.0:
                    continue
                fluid = _fluid_label(_item(allocations, index) if allocations else None)
                year = _as_int(_item(expense_years, index) if expense_years else None)
                cost = _row_cost(section, row, index)
                groups.setdefault(
                    (fluid, section, classification), []
                ).append((table_name, index, year or 0, cost))

    issues: list[ValidationIssue] = []
    for (fluid, section, classification), rows in sorted(groups.items()):
        section_label = _SECTION_LABELS.get(section, section)
        classification_label = _CLASSIFICATION_LABELS.get(classification, classification)
        locations = tuple(
            IssueLocation(
                _path(path_prefix, section, table_name, "cost_type", index)
            )
            for table_name, index, _year, _cost in rows
        )
        years = sorted({year for _t, _i, year, _c in rows if year})
        first_cost = rows[0][3]
        message = (
            f"{fluid} {section_label} have Cost Type set to {classification_label}. "
            "These costs are expected to use capital-cost treatment and be amortized in "
            "a Gross Split POD I contract; non-capital treatment charges them directly "
            "without amortization. Strict mode stops this combination and loose mode "
            "warns."
        )
        guidance = (
            f"Open {section_label} and review Cost Type for the marked rows, the "
            "intended cost treatment and the POD I setting in Contract Configuration, "
            "then select Check again."
        )
        context: dict[str, JsonValue] = {
            "fluid": fluid,
            "cost_section": section,
            "cost_category": classification,
            "actual": classification,
            "expense_years": cast(JsonValue, years),
            "cost_amount": first_cost,
        }
        issues.append(
            ValidationIssue(
                code=_POD1_RULE_CODE.get(section, "GS_POD1_NONCAPITAL"),
                severity=severity,
                message=message,
                guidance=guidance,
                locations=locations,
                related_locations=(IssueLocation(_path(path_prefix, "setup", "is_pod_1")),),
                context=context,
            )
        )
    return tuple(issues)


def _onstream_years(
    setup: Mapping[str, object],
) -> tuple[int, int] | None:
    oil = setup.get("oil_onstream_date")
    gas = setup.get("gas_onstream_date")
    oil_year = date_year(oil)
    gas_year = date_year(gas)
    if oil_year is None or gas_year is None:
        return None
    return oil_year, gas_year


def collect_preonstream_window_issues(
    payload: Mapping[str, object],
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Collect pre-onstream year-window violations.

    Mirrors ``BaseProject._validate_preonstream`` for both POD I and non-POD I
    contracts. Missing onstream dates are reported by the caller as a skipped
    check; this predicate assumes the dates are present.
    """
    setup = _as_mapping(payload.get("setup"))
    if setup is None:
        return ()
    onstream = _onstream_years(setup)
    if onstream is None:
        return ()
    onstream_year = min(onstream)
    approval_year = _as_int(setup.get("approval_year"))
    is_pod_1 = setup.get("is_pod_1") is True
    severity: Severity = "warning" if setup.get("is_strict") is False else "error"

    groups: dict[tuple[str, str], list[tuple[str, int, int, float]]] = {}
    for section in COST_SECTIONS:
        tables = _as_mapping(payload.get(section))
        if tables is None:
            continue
        for table_name, row_value in tables.items():
            row = _as_mapping(row_value)
            if row is None:
                continue
            cost_types = _as_sequence(row.get("cost_type"))
            allocations = _as_sequence(row.get("cost_allocation"))
            expense_years = _as_sequence(row.get("expense_year"))
            if cost_types is None or expense_years is None:
                continue
            for index, raw_type in enumerate(cost_types):
                if _normalized_text(raw_type) != "preonstream_cost":
                    continue
                if _row_cost(section, row, index) == 0.0:
                    continue
                year = _as_int(_item(expense_years, index))
                if year is None:
                    continue
                violation = year > onstream_year or (
                    is_pod_1 and approval_year is not None and year < approval_year
                )
                if not violation:
                    continue
                fluid = _fluid_label(_item(allocations, index) if allocations else None)
                groups.setdefault((fluid, section), []).append(
                    (table_name, index, year, _row_cost(section, row, index))
                )

    issues: list[ValidationIssue] = []
    for (fluid, section), rows in sorted(groups.items()):
        locations = tuple(
            IssueLocation(
                _path(path_prefix, section, table_name, "expense_year", index)
            )
            for table_name, index, _year, _cost in rows
        )
        years = sorted({year for _t, _i, year, _c in rows})
        actual_year = years[0]
        message = (
            f"Year is {actual_year}; the project runs from "
            f"{approval_year if approval_year is not None else 'the project start'} "
            f"to {onstream_year}."
        )
        guidance = (
            "Correct Year if this cost belongs within that period, or review the "
            "project dates in General Configuration."
        )
        context: dict[str, JsonValue] = {
            "fluid": fluid,
            "cost_section": section,
            "actual_year": actual_year,
            "affected_years": cast(JsonValue, years),
            "onstream_year": onstream_year,
        }
        issues.append(
            ValidationIssue(
                code="PREONSTREAM_YEAR_OUTSIDE_PROJECT",
                severity=severity,
                message=message,
                guidance=guidance,
                locations=locations,
                related_locations=(
                    IssueLocation(_path(path_prefix, "setup", "oil_onstream_date")),
                    IssueLocation(_path(path_prefix, "setup", "gas_onstream_date")),
                ),
                context=context,
            )
        )
    return tuple(issues)


# ── Task 2B classification / approval / PIS preflight collectors ────────

_CR_FRACTION_ATTRIBUTES: tuple[str, ...] = (
    "oil_ftp_portion",
    "gas_ftp_portion",
    "oil_ctr_pretax_share",
    "gas_ctr_pretax_share",
    "oil_ic_rate",
    "gas_ic_rate",
    "oil_cr_cap_rate",
    "gas_cr_cap_rate",
    "oil_dmo_volume_portion",
    "oil_dmo_fee_portion",
    "gas_dmo_volume_portion",
    "gas_dmo_fee_portion",
)
_CR_DISCRETE_ATTRIBUTES: tuple[str, ...] = (
    "oil_dmo_holiday_duration",
    "gas_dmo_holiday_duration",
)
_CR_CARRY_FORWARD_ATTRIBUTES: tuple[str, ...] = (
    "oil_carry_forward_depreciation",
    "gas_carry_forward_depreciation",
)
_GS_FRACTION_ATTRIBUTES: tuple[str, ...] = (
    "split_ministry_disc",
    "oil_dmo_volume_portion",
    "oil_dmo_fee_portion",
    "gas_dmo_volume_portion",
    "gas_dmo_fee_portion",
)
_GS_DISCRETE_ATTRIBUTES: tuple[str, ...] = (
    "oil_dmo_holiday_duration",
    "gas_dmo_holiday_duration",
)
_GS_CARRY_FORWARD_ATTRIBUTES: tuple[str, ...] = (
    "oil_carry_forward_depreciation",
    "gas_carry_forward_depreciation",
)


def _strict_severity(setup: Mapping[str, object]) -> Severity:
    """The engine's severity: loose mode reports the same rule as a warning."""
    return "warning" if setup.get("is_strict") is False else "error"


def _effective_onstream_year(setup: Mapping[str, object]) -> int | None:
    """The earliest of the supplied oil/gas onstream years, else ``None``."""
    onstream = _onstream_years(setup)
    if onstream is None:
        return None
    return min(onstream)


def _approval_year_for(setup: Mapping[str, object], bounds: ProjectBounds) -> int:
    """The engine's approval year: the supplied value or the project start."""
    value = _as_int(setup.get("approval_year"))
    return bounds.start_year if value is None else value


def _array_length(value: object) -> int | None:
    sequence = _as_sequence(value)
    return len(sequence) if sequence is not None else None


@dataclass(frozen=True)
class _ClassifiedRow:
    """One submitted cost row with its engine-effective classification."""

    section: str
    table: str
    index: int
    year: int | None
    classification: str | None
    fluid: str
    cost: float


def _iter_classified_rows(
    payload: Mapping[str, object],
    *,
    approval_year: int,
    onstream_year: int,
    is_pod_1: bool,
) -> Iterator[_ClassifiedRow]:
    """Yield every submitted cost row with its effective cost type.

    An explicit submitted classification is preserved; only an unset
    classification receives the engine's default for its year, so the preflight
    never invents a classification that the engine would not assign.
    """
    for section in COST_SECTIONS:
        tables = _as_mapping(payload.get(section))
        if tables is None:
            continue
        for table_name, row_value in tables.items():
            row = _as_mapping(row_value)
            if row is None:
                continue
            cost_types = _as_sequence(row.get("cost_type"))
            if cost_types is None:
                continue
            expense_years = _as_sequence(row.get("expense_year"))
            allocations = _as_sequence(row.get("cost_allocation"))
            for index in range(len(cost_types)):
                year = (
                    _as_int(_item(expense_years, index))
                    if expense_years is not None
                    else None
                )
                classification = effective_classification(
                    _item(cost_types, index),
                    expense_year=year,
                    approval_year=approval_year,
                    onstream_year=onstream_year,
                    is_pod_1=is_pod_1,
                )
                fluid = _fluid_label(
                    _item(allocations, index) if allocations is not None else None
                )
                yield _ClassifiedRow(
                    section,
                    table_name,
                    index,
                    year,
                    classification,
                    fluid,
                    _row_cost(section, row, index),
                )


def collect_approval_year_issues(
    payload: Mapping[str, object],
    bounds: ProjectBounds,
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Validate the approval year against the project and onstream bounds.

    Mirrors ``BaseProject._validate_approval_year``. Requires the supplied
    onstream dates; the caller discloses the skipped scope when they are
    missing.
    """
    setup = _as_mapping(payload.get("setup"))
    if setup is None:
        return ()
    onstream_year = _effective_onstream_year(setup)
    if onstream_year is None:
        return ()
    approval = _approval_year_for(setup, bounds)
    violation = approval_year_violation(
        approval,
        start_year=bounds.start_year,
        end_year=bounds.end_year,
        onstream_year=onstream_year,
    )
    if violation is None:
        return ()
    if violation == "after_onstream":
        code = "APPROVAL_YEAR_AFTER_ONSTREAM"
        message = (
            f"Approval Year is {approval}; it must not be later than the effective "
            f"onstream year {onstream_year}."
        )
    else:
        code = "APPROVAL_YEAR_OUTSIDE_PROJECT"
        message = (
            f"Approval Year is {approval}; the project runs from "
            f"{bounds.start_year} to {bounds.end_year}."
        )
    return (
        ValidationIssue(
            code=code,
            severity="error",
            message=message,
            guidance=(
                "Review Approval Year in General Configuration and the onstream "
                "setting in Contract Configuration, then select Check again."
            ),
            locations=(IssueLocation(_path(path_prefix, "setup", "approval_year")),),
            related_locations=(
                IssueLocation(_path(path_prefix, "setup", "oil_onstream_date")),
                IssueLocation(_path(path_prefix, "setup", "gas_onstream_date")),
            ),
            context={
                "actual_year": approval,
                "start_year": bounds.start_year,
                "end_year": bounds.end_year,
                "onstream_year": onstream_year,
            },
        ),
    )


def collect_cost_classification_issues(
    payload: Mapping[str, object],
    bounds: ProjectBounds,
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Validate effective cost-type classifications against project timing.

    Mirrors ``_validate_cost_type_consistency`` and
    ``_validate_boundary_year_cost_types`` for every fluid, section and table,
    aggregating all independent violations instead of stopping at the first.
    """
    setup = _as_mapping(payload.get("setup"))
    if setup is None:
        return ()
    onstream_year = _effective_onstream_year(setup)
    if onstream_year is None:
        return ()
    approval = _approval_year_for(setup, bounds)
    is_pod_1 = setup.get("is_pod_1") is True
    severity = _strict_severity(setup)

    groups: dict[tuple[str, str, str, str, str], list[_ClassifiedRow]] = {}
    for row in _iter_classified_rows(
        payload,
        approval_year=approval,
        onstream_year=onstream_year,
        is_pod_1=is_pod_1,
    ):
        if row.year is None or row.classification is None:
            continue
        if row.classification not in CLASSIFICATION_ORDER:
            continue
        period = classification_period(
            row.year, approval_year=approval, onstream_year=onstream_year
        )
        if period is not None and not classification_allowed_in_period(
            period, row.classification
        ):
            groups.setdefault(
                ("COST_TYPE_PERIOD_MISMATCH", row.fluid, row.section, row.classification, period),
                [],
            ).append(row)
            continue
        boundary = boundary_year_classification_violation(
            row.year,
            row.classification,
            approval_year=approval,
            onstream_year=onstream_year,
        )
        if boundary == "approval_year":
            groups.setdefault(
                (
                    "COST_TYPE_POSTONSTREAM_AT_APPROVAL",
                    row.fluid,
                    row.section,
                    row.classification,
                    "approval_year",
                ),
                [],
            ).append(row)
        elif boundary == "onstream_year":
            groups.setdefault(
                (
                    "COST_TYPE_SUNK_AT_ONSTREAM",
                    row.fluid,
                    row.section,
                    row.classification,
                    "onstream_year",
                ),
                [],
            ).append(row)

    issues: list[ValidationIssue] = []
    for (code, fluid, section, classification, period), rows in sorted(groups.items()):
        section_label = _SECTION_LABELS.get(section, section)
        classification_label = _CLASSIFICATION_LABELS.get(classification, classification)
        locations = tuple(
            IssueLocation(
                _path(path_prefix, section, table_name, "cost_type", index)
            )
            for table_name, index in ((row.table, row.index) for row in rows)
        )
        years = sorted({row.year for row in rows if row.year is not None})
        if code == "COST_TYPE_PERIOD_MISMATCH":
            actual_year = years[0] if years else None
            message = (
                f"{fluid} {section_label} have Cost Type set to {classification_label} "
                f"for Year {actual_year}, which falls in the {period.replace('_', ' ')} "
                "period. This classification is not allowed in that period."
            )
        elif code == "COST_TYPE_POSTONSTREAM_AT_APPROVAL":
            message = (
                f"{fluid} {section_label} have Cost Type set to {classification_label} "
                f"at the Approval Year ({approval}). Postonstream Cost is not allowed "
                "at the approval year."
            )
        else:
            message = (
                f"{fluid} {section_label} have Cost Type set to {classification_label} "
                f"at the Onstream Year ({onstream_year}). Sunk Cost is not allowed at "
                "the onstream year."
            )
        issues.append(
            ValidationIssue(
                code=code,
                severity=severity,
                message=message,
                guidance=(
                    f"Open {section_label} and review Cost Type in the marked rows. "
                    "Correct the entry that does not match your intended scenario, "
                    "then select Check again."
                ),
                locations=locations,
                related_locations=(
                    IssueLocation(_path(path_prefix, "setup", "approval_year")),
                ),
                context={
                    "fluid": fluid,
                    "cost_section": section,
                    "cost_category": classification,
                    "actual": classification,
                    "affected_years": cast(JsonValue, years),
                    "approval_year": approval,
                    "onstream_year": onstream_year,
                },
            )
        )
    return tuple(issues)


def collect_sunk_cutoff_issues(
    payload: Mapping[str, object],
    bounds: ProjectBounds,
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Validate sunk-cost years against the existing cutoff year.

    Mirrors ``BaseProject._validate_sunkcost``: the cutoff is the earlier of the
    approval year and the earliest onstream year; strict mode raises and loose
    mode records the same rule as a warning.
    """
    setup = _as_mapping(payload.get("setup"))
    if setup is None:
        return ()
    onstream_year = _effective_onstream_year(setup)
    if onstream_year is None:
        return ()
    approval = _approval_year_for(setup, bounds)
    is_pod_1 = setup.get("is_pod_1") is True
    severity = _strict_severity(setup)
    cutoff = sunk_cost_cutoff_year(approval_year=approval, onstream_year=onstream_year)

    groups: dict[tuple[str, str], list[_ClassifiedRow]] = {}
    for row in _iter_classified_rows(
        payload,
        approval_year=approval,
        onstream_year=onstream_year,
        is_pod_1=is_pod_1,
    ):
        if row.classification != CLASSIFICATION_SUNK or row.year is None:
            continue
        if not sunk_cost_year_exceeds_cutoff(row.year, cutoff):
            continue
        groups.setdefault((row.fluid, row.section), []).append(row)

    issues: list[ValidationIssue] = []
    for (fluid, section), rows in sorted(groups.items()):
        section_label = _SECTION_LABELS.get(section, section)
        years = sorted({row.year for row in rows if row.year is not None})
        actual_year = years[0] if years else None
        issues.append(
            ValidationIssue(
                code="SUNK_COST_AFTER_CUTOFF",
                severity=severity,
                message=(
                    f"Year is {actual_year}; sunk costs must not occur after the "
                    f"cutoff year {cutoff} (the earlier of Approval Year {approval} "
                    f"and Onstream Year {onstream_year})."
                ),
                guidance=(
                    f"Open {section_label} and review Year and Cost Type in the marked "
                    "rows, or review the governing approval and onstream settings, "
                    "then select Check again."
                ),
                locations=tuple(
                    IssueLocation(
                        _path(path_prefix, section, row.table, "expense_year", row.index)
                    )
                    for row in rows
                ),
                related_locations=(
                    IssueLocation(_path(path_prefix, "setup", "approval_year")),
                ),
                context={
                    "fluid": fluid,
                    "cost_section": section,
                    "actual_year": actual_year,
                    "affected_years": cast(JsonValue, years),
                    "cutoff_year": cutoff,
                    "approval_year": approval,
                    "onstream_year": onstream_year,
                },
            )
        )
    return tuple(issues)


def collect_postonstream_window_issues(
    payload: Mapping[str, object],
    bounds: ProjectBounds,
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Validate post-onstream years against the effective onstream year.

    Mirrors ``BaseProject._validate_postonstream``; the engine always raises for
    this rule (it has no strict branch), so the preflight reports an error.
    """
    setup = _as_mapping(payload.get("setup"))
    if setup is None:
        return ()
    onstream_year = _effective_onstream_year(setup)
    if onstream_year is None:
        return ()
    approval = _approval_year_for(setup, bounds)
    is_pod_1 = setup.get("is_pod_1") is True

    groups: dict[tuple[str, str], list[_ClassifiedRow]] = {}
    for row in _iter_classified_rows(
        payload,
        approval_year=approval,
        onstream_year=onstream_year,
        is_pod_1=is_pod_1,
    ):
        if row.classification != CLASSIFICATION_POSTONSTREAM or row.year is None:
            continue
        if not postonstream_year_before_onstream(row.year, onstream_year):
            continue
        groups.setdefault((row.fluid, row.section), []).append(row)

    issues: list[ValidationIssue] = []
    for (fluid, section), rows in sorted(groups.items()):
        section_label = _SECTION_LABELS.get(section, section)
        years = sorted({row.year for row in rows if row.year is not None})
        actual_year = years[0] if years else None
        issues.append(
            ValidationIssue(
                code="POSTONSTREAM_YEAR_BEFORE_ONSTREAM",
                severity="error",
                message=(
                    f"Year is {actual_year}; post-onstream costs must not occur "
                    f"before the effective onstream year {onstream_year}."
                ),
                guidance=(
                    f"Open {section_label} and review Year and Cost Type in the marked "
                    "rows, or review the lifting onstream setting, then select Check "
                    "again."
                ),
                locations=tuple(
                    IssueLocation(
                        _path(path_prefix, section, row.table, "expense_year", row.index)
                    )
                    for row in rows
                ),
                related_locations=(
                    IssueLocation(_path(path_prefix, "setup", "oil_onstream_date")),
                    IssueLocation(_path(path_prefix, "setup", "gas_onstream_date")),
                ),
                context={
                    "fluid": fluid,
                    "cost_section": section,
                    "actual_year": actual_year,
                    "affected_years": cast(JsonValue, years),
                    "onstream_year": onstream_year,
                },
            )
        )
    return tuple(issues)


def _pis_directions(
    contract_type: ContractType,
    classification: str,
    *,
    is_pod_1: bool,
    is_strict: bool,
) -> set[str]:
    """Which capital PIS rules the engine applies for this contract/row."""
    directions: set[str] = set()
    if contract_type not in ("Cost Recovery", "Gross Split"):
        return directions
    if classification == CLASSIFICATION_PREONSTREAM:
        directions.add("before")
        if contract_type == "Gross Split" and is_pod_1:
            directions.add("after")
    elif classification == CLASSIFICATION_SUNK:
        if contract_type == "Gross Split" and is_pod_1:
            directions.update({"before", "after"})
        elif not is_strict:
            directions.add("before")
    return directions


def collect_capital_pis_issues(
    payload: Mapping[str, object],
    bounds: ProjectBounds,
    contract_type: ContractType,
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Validate capital PIS years against the effective onstream year.

    Mirrors ``_check_capital_pis_years_before_onstream`` and
    ``_check_capital_pis_years_after_onstream`` per existing contract class and
    strict severity.
    """
    if contract_type not in ("Cost Recovery", "Gross Split"):
        return ()
    setup = _as_mapping(payload.get("setup"))
    if setup is None:
        return ()
    onstream_year = _effective_onstream_year(setup)
    if onstream_year is None:
        return ()
    approval = _approval_year_for(setup, bounds)
    is_pod_1 = setup.get("is_pod_1") is True
    is_strict = setup.get("is_strict") is not False
    severity = _strict_severity(setup)

    tables = _as_mapping(payload.get("capital"))
    if tables is None:
        return ()
    groups: dict[tuple[str, str, str, str], list[tuple[str, int, int]]] = {}
    for table_name, row_value in tables.items():
        row = _as_mapping(row_value)
        if row is None:
            continue
        cost_types = _as_sequence(row.get("cost_type"))
        if cost_types is None:
            continue
        expense_years = _as_sequence(row.get("expense_year"))
        pis_years = _as_sequence(row.get("pis_year"))
        costs = _as_sequence(row.get("cost"))
        allocations = _as_sequence(row.get("cost_allocation"))
        if pis_years is None or costs is None:
            continue
        for index in range(len(cost_types)):
            year = (
                _as_int(_item(expense_years, index))
                if expense_years is not None
                else None
            )
            classification = effective_classification(
                _item(cost_types, index),
                expense_year=year,
                approval_year=approval,
                onstream_year=onstream_year,
                is_pod_1=is_pod_1,
            )
            if classification is None:
                continue
            directions = _pis_directions(
                contract_type, classification, is_pod_1=is_pod_1, is_strict=is_strict
            )
            if not directions:
                continue
            pis = _as_int(_item(pis_years, index))
            fluid = _fluid_label(
                _item(allocations, index) if allocations is not None else None
            )
            if pis is None:
                continue
            if "before" in directions and pis_before_onstream_indices(
                [pis], [_item(costs, index)], onstream_year
            ):
                groups.setdefault(
                    ("PIS_YEAR_BEFORE_ONSTREAM", fluid, "capital", classification), []
                ).append((table_name, index, pis))
            if "after" in directions and pis_after_onstream_indices(
                [pis], onstream_year
            ):
                groups.setdefault(
                    ("PIS_YEAR_AFTER_ONSTREAM", fluid, "capital", classification), []
                ).append((table_name, index, pis))

    issues: list[ValidationIssue] = []
    for (code, fluid, section, classification), rows in sorted(groups.items()):
        years = sorted({pis for _t, _i, pis in rows})
        actual_year = years[0] if years else None
        if code == "PIS_YEAR_BEFORE_ONSTREAM":
            message = (
                f"PIS Year (placed-in-service year) is {actual_year}; it is before the "
                f"effective onstream year {onstream_year}."
            )
        else:
            message = (
                f"PIS Year (placed-in-service year) is {actual_year}; it is after the "
                f"effective onstream year {onstream_year}."
            )
        issues.append(
            ValidationIssue(
                code=code,
                severity=severity,
                message=message,
                guidance=(
                    "Open Capital Costs and review PIS Year (placed-in-service year) "
                    "in the marked rows, or review the effective onstream date, then "
                    "select Check again."
                ),
                locations=tuple(
                    IssueLocation(_path(path_prefix, section, table_name, "pis_year", index))
                    for table_name, index, _pis in rows
                ),
                related_locations=(
                    IssueLocation(_path(path_prefix, "setup", "oil_onstream_date")),
                    IssueLocation(_path(path_prefix, "setup", "gas_onstream_date")),
                ),
                context={
                    "fluid": fluid,
                    "cost_section": section,
                    "cost_category": classification,
                    "actual_year": actual_year,
                    "affected_years": cast(JsonValue, years),
                    "onstream_year": onstream_year,
                },
            )
        )
    return tuple(issues)


def collect_costrecovery_capital_sunk_issues(
    payload: Mapping[str, object],
    bounds: ProjectBounds,
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Flag non-zero capital sunk costs in a Cost Recovery contract.

    Mirrors ``CostRecovery._check_invalid_capital_sunk_costs``: strict mode
    raises and loose mode records the same rule as a warning.
    """
    setup = _as_mapping(payload.get("setup"))
    if setup is None:
        return ()
    onstream_year = _effective_onstream_year(setup)
    if onstream_year is None:
        return ()
    approval = _approval_year_for(setup, bounds)
    is_pod_1 = setup.get("is_pod_1") is True
    severity = _strict_severity(setup)

    groups: dict[tuple[str, str], list[_ClassifiedRow]] = {}
    for row in _iter_classified_rows(
        payload,
        approval_year=approval,
        onstream_year=onstream_year,
        is_pod_1=is_pod_1,
    ):
        if row.section != "capital" or row.classification != CLASSIFICATION_SUNK:
            continue
        if row.cost == 0.0:
            continue
        groups.setdefault((row.fluid, row.section), []).append(row)

    issues: list[ValidationIssue] = []
    for (fluid, section), rows in sorted(groups.items()):
        years = sorted({row.year for row in rows if row.year is not None})
        actual_year = years[0] if years else None
        issues.append(
            ValidationIssue(
                code="CR_CAPITAL_SUNK_COST",
                severity=severity,
                message=(
                    f"{fluid} Capital Costs have a Sunk Cost classification in a Cost "
                    f"Recovery contract (Year {actual_year}). Cost Recovery does not "
                    "allow capital sunk costs."
                ),
                guidance=(
                    "Open Capital Costs and review Cost Type in the marked rows, or "
                    "review the contract type, then select Check again."
                ),
                locations=tuple(
                    IssueLocation(
                        _path(path_prefix, section, row.table, "cost_type", row.index)
                    )
                    for row in rows
                ),
                context={
                    "fluid": fluid,
                    "cost_section": section,
                    "cost_category": CLASSIFICATION_SUNK,
                    "affected_years": cast(JsonValue, years),
                },
            )
        )
    return tuple(issues)


def collect_grosssplit_nonpod1_sunk_issues(
    payload: Mapping[str, object],
    bounds: ProjectBounds,
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Flag non-zero sunk costs in a non-POD I Gross Split contract.

    Mirrors ``GrossSplit._check_no_sunk_costs`` for both capital and non-capital
    sections; strict mode raises and loose mode records a warning.
    """
    setup = _as_mapping(payload.get("setup"))
    if setup is None:
        return ()
    if setup.get("is_pod_1") is True:
        return ()
    onstream_year = _effective_onstream_year(setup)
    if onstream_year is None:
        return ()
    approval = _approval_year_for(setup, bounds)
    severity = _strict_severity(setup)

    groups: dict[tuple[str, str], list[_ClassifiedRow]] = {}
    for row in _iter_classified_rows(
        payload,
        approval_year=approval,
        onstream_year=onstream_year,
        is_pod_1=False,
    ):
        if row.classification != CLASSIFICATION_SUNK:
            continue
        if row.cost == 0.0:
            continue
        groups.setdefault((row.fluid, row.section), []).append(row)

    issues: list[ValidationIssue] = []
    for (fluid, section), rows in sorted(groups.items()):
        section_label = _SECTION_LABELS.get(section, section)
        years = sorted({row.year for row in rows if row.year is not None})
        issues.append(
            ValidationIssue(
                code="GS_NONPOD1_SUNK_COST",
                severity=severity,
                message=(
                    f"{fluid} {section_label} have a Sunk Cost classification in a "
                    "non-POD I Gross Split contract, which does not allow sunk costs."
                ),
                guidance=(
                    f"Open {section_label} and review Cost Type in the marked rows, or "
                    "review the POD setting in Contract Configuration, then select "
                    "Check again."
                ),
                locations=tuple(
                    IssueLocation(
                        _path(path_prefix, section, row.table, "cost_type", row.index)
                    )
                    for row in rows
                ),
                related_locations=(
                    IssueLocation(_path(path_prefix, "setup", "is_pod_1")),
                ),
                context={
                    "fluid": fluid,
                    "cost_section": section,
                    "cost_category": CLASSIFICATION_SUNK,
                    "affected_years": cast(JsonValue, years),
                },
            )
        )
    return tuple(issues)


def _attribute_issue(
    *,
    code: str,
    section: str,
    attribute: str,
    message: str,
    path_prefix: tuple[str | int, ...],
) -> ValidationIssue:
    label = _field_label(attribute)
    return ValidationIssue(
        code=code,
        severity="error",
        message=message,
        guidance=(
            f"Review {label} in {section}, correct the value, then select Check again."
        ),
        locations=(IssueLocation(_path(path_prefix, "costrecovery" if section == "Cost Recovery" else "grosssplit", attribute)),),
        context={"attribute": attribute, "field": label},
    )


def _collect_attribute_issues(
    block: Mapping[str, object],
    *,
    section: str,
    code_prefix: str,
    fraction_attributes: tuple[str, ...],
    discrete_attributes: tuple[str, ...],
    carry_forward_attributes: tuple[str, ...],
    project_duration: int,
    path_prefix: tuple[str | int, ...],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for attribute in fraction_attributes:
        value = block.get(attribute)
        if value is None:
            continue
        length = _array_length(value)
        if length is not None:
            if length_mismatch(length, project_duration):
                issues.append(
                    _attribute_issue(
                        code=f"{code_prefix}_ATTRIBUTE_LENGTH_MISMATCH",
                        section=section,
                        attribute=attribute,
                        message=(
                            f"The {_field_label(attribute)} length ({length}) must equal "
                            f"the project duration ({project_duration})."
                        ),
                        path_prefix=path_prefix,
                    )
                )
        elif fraction_value_out_of_range(value, 0.0, 1.0):
            issues.append(
                _attribute_issue(
                    code=f"{code_prefix}_ATTRIBUTE_OUT_OF_RANGE",
                    section=section,
                    attribute=attribute,
                    message=(
                        f"The {_field_label(attribute)} value ({value}) is outside the "
                        "allowed range 0.0 - 1.0."
                    ),
                    path_prefix=path_prefix,
                )
            )
    for attribute in discrete_attributes:
        value = block.get(attribute)
        if value is None:
            continue
        if contains_negative([value]):
            issues.append(
                _attribute_issue(
                    code=f"{code_prefix}_ATTRIBUTE_NEGATIVE",
                    section=section,
                    attribute=attribute,
                    message=(
                        f"The {_field_label(attribute)} value ({value}) is below 0."
                    ),
                    path_prefix=path_prefix,
                )
            )
    for attribute in carry_forward_attributes:
        value = block.get(attribute)
        if value is None:
            continue
        sequence = _as_sequence(value)
        length = len(sequence) if sequence is not None else None
        if sequence is not None and length is not None:
            if contains_negative(sequence):
                issues.append(
                    _attribute_issue(
                        code=f"{code_prefix}_CARRY_FORWARD_NEGATIVE",
                        section=section,
                        attribute=attribute,
                        message=(
                            f"The {_field_label(attribute)} contains negative values."
                        ),
                        path_prefix=path_prefix,
                    )
                )
            elif length_exceeds_project(length, project_duration):
                issues.append(
                    _attribute_issue(
                        code=f"{code_prefix}_CARRY_FORWARD_TOO_LONG",
                        section=section,
                        attribute=attribute,
                        message=(
                            f"The {_field_label(attribute)} length ({length}) exceeds "
                            f"the project duration ({project_duration})."
                        ),
                        path_prefix=path_prefix,
                    )
                )
        elif contains_negative([value]):
            issues.append(
                _attribute_issue(
                    code=f"{code_prefix}_CARRY_FORWARD_NEGATIVE",
                    section=section,
                    attribute=attribute,
                    message=(
                        f"The {_field_label(attribute)} contains negative values."
                    ),
                    path_prefix=path_prefix,
                )
            )
    return issues


def collect_costrecovery_attribute_issues(
    payload: Mapping[str, object],
    bounds: ProjectBounds,
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Validate Cost Recovery fraction, DMO, carry-forward and condition inputs.

    Mirrors ``CostRecovery._check_attributes`` and ``_normalize_condition_dict``.
    """
    block = _as_mapping(payload.get("costrecovery"))
    if block is None:
        return ()
    duration = bounds.end_year - bounds.start_year + 1
    issues = _collect_attribute_issues(
        block,
        section="Cost Recovery",
        code_prefix="CR",
        fraction_attributes=_CR_FRACTION_ATTRIBUTES,
        discrete_attributes=_CR_DISCRETE_ATTRIBUTES,
        carry_forward_attributes=_CR_CARRY_FORWARD_ATTRIBUTES,
        project_duration=duration,
        path_prefix=path_prefix,
    )
    if condition_dict_is_mixed_form(block.get("condition_dict")):
        issues.append(
            ValidationIssue(
                code="CR_CONDITION_DICT_MIXED_FORM",
                severity="error",
                message=(
                    "The condition table mixes normalized and column forms. Use one "
                    "consistent form."
                ),
                guidance=(
                    "Review the condition table in Cost Recovery, correct its form, "
                    "then select Check again."
                ),
                locations=(
                    IssueLocation(_path(path_prefix, "costrecovery", "condition_dict")),
                ),
                context={"attribute": "condition_dict"},
            )
        )
    return tuple(issues)


def collect_grosssplit_attribute_issues(
    payload: Mapping[str, object],
    bounds: ProjectBounds,
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Validate Gross Split fraction, DMO, carry-forward and split-offset inputs.

    Mirrors ``GrossSplit._check_attributes`` and
    ``_check_cum_production_split_offset``.
    """
    block = _as_mapping(payload.get("grosssplit"))
    duration = bounds.end_year - bounds.start_year + 1
    issues: list[ValidationIssue] = []
    if block is not None:
        issues.extend(
            _collect_attribute_issues(
                block,
                section="Gross Split",
                code_prefix="GS",
                fraction_attributes=_GS_FRACTION_ATTRIBUTES,
                discrete_attributes=_GS_DISCRETE_ATTRIBUTES,
                carry_forward_attributes=_GS_CARRY_FORWARD_ATTRIBUTES,
                project_duration=duration,
                path_prefix=path_prefix,
            )
        )
    arguments = _as_mapping(payload.get("contract_arguments"))
    if arguments is not None:
        offset = arguments.get("cum_production_split_offset")
        length = _array_length(offset)
        if length is not None and cum_split_offset_length_mismatch(length, duration):
            issues.append(
                ValidationIssue(
                    code="GS_CUM_PRODUCTION_SPLIT_OFFSET_LENGTH",
                    severity="error",
                    message=(
                        f"The cumulative production split offset length ({length}) must "
                        f"equal the project duration ({duration})."
                    ),
                    guidance=(
                        "Review the cumulative production split offset, correct its "
                        "length, then select Check again."
                    ),
                    locations=(
                        IssueLocation(
                            _path(
                                path_prefix,
                                "contract_arguments",
                                "cum_production_split_offset",
                            )
                        ),
                    ),
                    context={"length": length, "project_duration": duration},
                )
            )
    return tuple(issues)


_GS_COMPONENT_LABELS: dict[str, str] = {
    "field_status": "Field Status",
    "field_loc": "Field Location",
    "res_depth": "Reservoir Depth",
    "infra_avail": "Infrastructure Availability",
    "res_type": "Reservoir Type",
    "api_oil": "Oil API",
    "domestic_use": "Domestic Use",
    "prod_stage": "Production Stage",
    "co2_content": "CO2 Content",
    "h2s_content": "H2S Content",
    "field_loc_2024": "Field Location (2024)",
    "infra_avail_2024": "Infrastructure Availability (2024)",
    "field_reserves_2024": "Field Reserves (2024)",
}

_GS_REGIME_LABELS: dict[str, str] = {
    "PERMEN_ESDM_8_2017": "PERMEN ESDM No. 8/2017",
    "PERMEN_ESDM_52_2017": "PERMEN ESDM No. 52/2017",
    "PERMEN_ESDM_20_2019": "PERMEN ESDM No. 20/2019",
    "PERMEN_ESDM_12_2020": "PERMEN ESDM No. 12/2020",
    "PERMEN_ESDM_13_2024": "PERMEN ESDM No. 13/2024",
}


def collect_grosssplit_variable_split_issues(
    payload: Mapping[str, object],
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Report the active Gross Split regime's missing variable-split components.

    Mirrors the engine's ``_require_variable_split_value`` completeness rule for
    the active regime (from ``contract_arguments.regime`` and
    ``reservoir_type_permen_2024``). An incomplete split block would otherwise be
    certified as valid and then fail during calculation, so it is reported here
    as a blocking user-input issue.
    """
    block = _as_mapping(payload.get("grosssplit"))
    if block is None:
        return ()
    arguments = _as_mapping(payload.get("contract_arguments"))
    regime_raw = arguments.get("regime") if arguments is not None else None
    reservoir_raw = (
        arguments.get("reservoir_type_permen_2024")
        if arguments is not None
        else None
    )
    if not variable_split_enforced(regime_raw):
        return ()
    regime_code = _normalized_regime_code(regime_raw)
    required = variable_split_required_components(regime_raw, reservoir_raw)
    if not required:
        return ()
    present = {name: block.get(name) for name in required}
    missing = missing_variable_split_components(present)
    if not missing:
        return ()
    regime_label = _GS_REGIME_LABELS.get(regime_code or "", "the selected regime")
    issues: list[ValidationIssue] = []
    for name in missing:
        label = _GS_COMPONENT_LABELS.get(name, _field_label(name))
        issues.append(
            ValidationIssue(
                code="GS_VARIABLE_SPLIT_COMPONENT_MISSING",
                severity="error",
                message=(
                    f"{label} is required to compute the Gross Split variable split "
                    f"for {regime_label}. Enter a value for this component, then "
                    "select Check again."
                ),
                guidance=(
                    "Open Contract Configuration and set the Gross Split variable "
                    "split components for the selected regime, then select Check again."
                ),
                locations=(IssueLocation(_path(path_prefix, "grosssplit", name)),),
                related_locations=(
                    IssueLocation(
                        _path(path_prefix, "contract_arguments", "regime")
                    ),
                ),
                context={
                    "field": name,
                    "component": name,
                    "regime": regime_code,
                    "reservoir_type": _normalized_reservoir_type(reservoir_raw),
                },
            )
        )
    return tuple(issues)


def _sections_present(payload: Mapping[str, object]) -> set[str]:
    present: set[str] = set()
    for section in COST_SECTIONS:
        if _as_mapping(payload.get(section)):
            present.add(section)
    return present


_NUMERIC_FIELDS: tuple[str, ...] = (
    "cost",
    "fixed_cost",
    "prod_rate",
    "cost_per_volume",
    "tax_portion",
    "tax_discount",
    "pis_year",
    "useful_life",
    "final_year",
)


def _is_text_value(value: object) -> bool:
    if isinstance(value, str):
        # Allow an empty string to be treated as an unset optional value.
        return value.strip() != ""
    return False


def collect_malformed_numeric_issues(
    payload: Mapping[str, object],
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Flag entered text in a numeric cost field (a mechanical correction)."""
    issues: list[ValidationIssue] = []
    for section in COST_SECTIONS:
        tables = _as_mapping(payload.get(section))
        if tables is None:
            continue
        for table_name, row_value in tables.items():
            row = _as_mapping(row_value)
            if row is None:
                continue
            for field_name in _NUMERIC_FIELDS:
                values = _as_sequence(row.get(field_name))
                if values is None:
                    continue
                for index, raw in enumerate(values):
                    if not _is_text_value(raw):
                        continue
                    section_label = _SECTION_LABELS.get(section, section)
                    issues.append(
                        ValidationIssue(
                            code="NUMERIC_FIELD_CONTAINS_TEXT",
                            severity="error",
                            message=(
                                f"{_field_label(field_name)} contains '{raw}'. "
                                "Enter a number."
                            ),
                            guidance=(
                                f"Open {section_label} and correct the marked row, "
                                "then select Check again."
                            ),
                            locations=(
                                IssueLocation(
                                    _path(
                                        path_prefix, section, table_name, field_name, index
                                    )
                                ),
                            ),
                            context={
                                "field": field_name,
                                "entered_value": str(raw),
                                "section": section,
                            },
                        )
                    )
    return tuple(issues)


# ── Structural preflight collectors ─────────────────────────────────────

_FINAL_YEAR_SECTIONS: tuple[str, ...] = ("asr", "lbt")
_PIS_SECTIONS: tuple[str, ...] = ("capital",)
_NONFINITE_FIELDS: tuple[str, ...] = (
    "cost",
    "fixed_cost",
    "prod_rate",
    "cost_per_volume",
    "tax_portion",
    "tax_discount",
    "pis_year",
    "useful_life",
    "final_year",
    "salvage_value",
    "depreciation_factor",
    "future_rate",
    "utilized_land_area",
    "utilized_building_area",
    "njop_land",
    "njop_building",
    "gross_revenue",
)
_LIFTING_NUMERIC_FIELDS: tuple[str, ...] = (
    "lifting_rate",
    "price",
    "prod_rate",
    "prod_rate_baseline",
    "ghv",
)
_LIFTING_REQUIRED_FIELDS: tuple[str, ...] = (
    "start_year",
    "end_year",
    "lifting_rate",
    "price",
    "prod_year",
    "fluid_type",
)
_LIFTING_ARRAY_FIELDS: tuple[str, ...] = (
    "prod_year",
    "lifting_rate",
    "price",
    "prod_rate",
    "prod_rate_baseline",
    "ghv",
)


@dataclass(frozen=True)
class ProjectBounds:
    """The effective project year span used by structural year checks."""

    start_year: int
    end_year: int


def _nonfinite_indices(values: Iterable[object]) -> list[int]:
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or value is None:
            continue
        numeric = _as_float(value)
        if numeric is not None and (numeric != numeric or numeric in (float("inf"), float("-inf"))):
            result.append(index)
    return result


def collect_nonfinite_numeric_issues(
    payload: Mapping[str, object],
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Flag non-finite (NaN/Infinity) numbers in entered numeric arrays.

    This mirrors the engine's numeric contract: an entered numeric value must be
    finite. ``None`` placeholders (nullable optional fields) are not flagged.
    """
    issues: list[ValidationIssue] = []
    for section in COST_SECTIONS:
        tables = _as_mapping(payload.get(section))
        if tables is None:
            continue
        for table_name, row_value in tables.items():
            row = _as_mapping(row_value)
            if row is None:
                continue
            for field_name in _NONFINITE_FIELDS:
                values = _as_sequence(row.get(field_name))
                if values is None:
                    continue
                for index in _nonfinite_indices(values):
                    section_label = _SECTION_LABELS.get(section, section)
                    issues.append(
                        ValidationIssue(
                            code="NUMERIC_FIELD_NOT_FINITE",
                            severity="error",
                            message=(
                                f"{_field_label(field_name)} must be a finite number."
                            ),
                            guidance=(
                                f"Open {section_label} and correct the marked row, "
                                "then select Check again."
                            ),
                            locations=(
                                IssueLocation(
                                    _path(
                                        path_prefix, section, table_name, field_name, index
                                    )
                                ),
                            ),
                            context={"field": field_name, "section": section},
                        )
                    )
    lifting = _as_mapping(payload.get("lifting"))
    if lifting is not None:
        for name, entry in lifting.items():
            row = _as_mapping(entry)
            if row is None:
                continue
            for field_name in _LIFTING_NUMERIC_FIELDS:
                values = _as_sequence(row.get(field_name))
                if values is None:
                    continue
                for index in _nonfinite_indices(values):
                    issues.append(
                        ValidationIssue(
                            code="NUMERIC_FIELD_NOT_FINITE",
                            severity="error",
                            message=(
                                f"{_field_label(field_name)} must be a finite number."
                            ),
                            guidance=(
                                "Open Lifting and correct the marked row, "
                                "then select Check again."
                            ),
                            locations=(
                                IssueLocation(
                                    _path(
                                        path_prefix, "lifting", name, field_name, index
                                    )
                                ),
                            ),
                            context={"field": field_name, "section": "lifting"},
                        )
                    )
    return tuple(issues)


def _field_label(name: str) -> str:
    words = name.replace("_", " ").strip()
    if not words:
        return name
    return words[0].upper() + words[1:]


def collect_project_date_issues(
    payload: Mapping[str, object],
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ProjectBounds | None, tuple[ValidationIssue, ...]]:
    """Parse and validate the project period using the engine date rules.

    Returns the effective project bounds (or ``None`` when unavailable) and any
    issues found. A malformed date or a reversed period yields ``None`` so the
    caller skips dependent year checks and discloses the blocked scope.
    """
    setup = _as_mapping(payload.get("setup"))
    if setup is None:
        return None, ()
    start_raw = setup.get("start_date")
    end_raw = setup.get("end_date")
    if start_raw is None or end_raw is None:
        return None, ()

    issues: list[ValidationIssue] = []
    start = parse_engine_date(start_raw)
    end = parse_engine_date(end_raw)
    if start is None:
        issues.append(
            ValidationIssue(
                code="DATE_FORMAT_INVALID",
                severity="error",
                message=(
                    f"Project Start contains '{start_raw}'. "
                    "Enter a date in DD/MM/YYYY format."
                ),
                guidance=(
                    "Review Project Start in General Configuration, "
                    "then select Check again."
                ),
                locations=(IssueLocation(_path(path_prefix, "setup", "start_date")),),
                context={"field": "Project Start", "entered_value": str(start_raw)},
            )
        )
    if end is None:
        issues.append(
            ValidationIssue(
                code="DATE_FORMAT_INVALID",
                severity="error",
                message=(
                    f"Project End contains '{end_raw}'. "
                    "Enter a date in DD/MM/YYYY format."
                ),
                guidance=(
                    "Review Project End in General Configuration, "
                    "then select Check again."
                ),
                locations=(IssueLocation(_path(path_prefix, "setup", "end_date")),),
                context={"field": "Project End", "entered_value": str(end_raw)},
            )
        )
    if start is None or end is None:
        return None, tuple(issues)

    if dates_reversed(start, end):
        issues.append(
            ValidationIssue(
                code="PROJECT_PERIOD_REVERSED",
                severity="error",
                message=(
                    "Project Start is after Project End. Review both dates in "
                    "General Configuration and correct the date that does not "
                    "match the intended project period."
                ),
                guidance=(
                    "Review Project Start and Project End in General Configuration, "
                    "then select Check again."
                ),
                locations=(IssueLocation(_path(path_prefix, "setup", "start_date")),),
                related_locations=(
                    IssueLocation(_path(path_prefix, "setup", "end_date")),
                ),
                context={
                    "start_date": str(start_raw),
                    "end_date": str(end_raw),
                },
            )
        )
        return None, tuple(issues)

    return ProjectBounds(start.year, end.year), tuple(issues)


def _year_bounds_issues(
    *,
    section: str,
    table_name: str,
    field_name: str,
    field_label: str,
    code: str,
    years: Iterable[object],
    bounds: ProjectBounds,
    path_prefix: tuple[str | int, ...],
) -> list[ValidationIssue]:
    low = year_below(years, bounds.start_year)
    high = year_above(years, bounds.end_year)
    if low is None and high is None:
        return []
    actual_year = low if low is not None else high
    indices = [
        index
        for index, value in enumerate(years)
        if _as_int(value) in {low, high} and _as_int(value) is not None
    ]
    if not indices:
        indices = [0]
    section_label = _SECTION_LABELS.get(section, section)
    return [
        ValidationIssue(
            code=code,
            severity="error",
            message=(
                f"{field_label} is {actual_year}; the project runs from "
                f"{bounds.start_year} to {bounds.end_year}. Correct {field_label} "
                "if the entry belongs within that period, or review the project "
                "dates in General Configuration."
            ),
            guidance=(
                f"Review the marked row in {section_label} and the project dates in "
                "General Configuration, then select Check again."
            ),
            locations=(
                IssueLocation(
                    _path(path_prefix, section, table_name, field_name, indices[0])
                ),
            ),
            context={
                "actual_year": cast(JsonValue, actual_year),
                "start_year": bounds.start_year,
                "end_year": bounds.end_year,
            },
        )
    ]


def _component_project_mismatch(
    row: Mapping[str, object], bounds: ProjectBounds
) -> list[tuple[str, int]]:
    mismatches: list[tuple[str, int]] = []
    start = _as_int(row.get("start_year"))
    end = _as_int(row.get("end_year"))
    if start is not None and start != bounds.start_year:
        mismatches.append(("start_year", start))
    if end is not None and end != bounds.end_year:
        mismatches.append(("end_year", end))
    return mismatches


def collect_cost_structural_issues(
    payload: Mapping[str, object],
    bounds: ProjectBounds,
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue | SkippedCheck, ...]:
    """Collect structural/date/array issues for every submitted cost table.

    Independent tables and fluids are aggregated. A table with an invalid shape
    (reversed period, component/project mismatch, or unequal arrays) reports the
    blocking issue and a skipped-check record, then its dependent row checks are
    not attempted.
    """
    results: list[ValidationIssue | SkippedCheck] = []
    for section in COST_SECTIONS:
        tables = _as_mapping(payload.get(section))
        if tables is None:
            continue
        for table_name, row_value in tables.items():
            row = _as_mapping(row_value)
            if row is None:
                continue
            section_label = _SECTION_LABELS.get(section, section)
            table_path = _path(path_prefix, section, table_name)

            if period_is_reversed(row.get("start_year"), row.get("end_year")):
                results.append(
                    ValidationIssue(
                        code="COMPONENT_PERIOD_REVERSED",
                        severity="error",
                        message=(
                            f"{section_label} period start year is after its end year. "
                            "Correct the start and end years, then select Check again."
                        ),
                        guidance=(
                            f"Review the start and end years in {section_label}, "
                            "then select Check again."
                        ),
                        locations=(IssueLocation(table_path),),
                        context={
                            "section": section,
                            "start_year": cast(JsonValue, _as_int(row.get("start_year"))),
                            "end_year": cast(JsonValue, _as_int(row.get("end_year"))),
                        },
                    )
                )
                results.append(
                    SkippedCheck(
                        scope=f"{section}:{table_name}:row_checks",
                        reason=(
                            f"Some checks could not run because {section_label} "
                            "start or end year needs correction. Fix the listed "
                            "issue, then select Check again."
                        ),
                        blocking_paths=(IssueLocation(table_path),),
                    )
                )
                continue

            mismatch = _component_project_mismatch(row, bounds)
            if mismatch:
                fields = ", ".join(_field_label(name) for name, _ in mismatch)
                results.append(
                    ValidationIssue(
                        code="COMPONENT_PERIOD_MISMATCH",
                        severity="error",
                        message=(
                            f"{section_label} {fields} do not match the project period "
                            f"({bounds.start_year}-{bounds.end_year}). Review the "
                            "component period and the project dates in General "
                            "Configuration."
                        ),
                        guidance=(
                            f"Review {section_label} and the project dates in General "
                            "Configuration, then select Check again."
                        ),
                        locations=(IssueLocation(table_path),),
                        context={
                            "section": section,
                            "project_start_year": bounds.start_year,
                            "project_end_year": bounds.end_year,
                        },
                    )
                )
                results.append(
                    SkippedCheck(
                        scope=f"{section}:{table_name}:row_checks",
                        reason=(
                            f"Some checks could not run because {section_label} period "
                            "does not match the project period. Fix the listed issue, "
                            "then select Check again."
                        ),
                        blocking_paths=(IssueLocation(table_path),),
                    )
                )
                continue

            lengths: dict[str, int] = {}
            for field_name, value in row.items():
                sequence = _as_sequence(value)
                if sequence is not None:
                    lengths[field_name] = len(sequence)
            mismatch_lengths = unequal_lengths(lengths)
            if mismatch_lengths is not None:
                visible = ", ".join(_field_label(name) for name in mismatch_lengths)
                results.append(
                    ValidationIssue(
                        code="ARRAY_LENGTH_MISMATCH",
                        severity="error",
                        message=(
                            "The input rows do not have matching values for "
                            f"{visible}. Each entered row needs a value for the "
                            "required fields. Review the affected input table, then "
                            "select Check again."
                        ),
                        guidance=(
                            f"Review {section_label} and make each entered row have a "
                            "value for the required fields, then select Check again."
                        ),
                        locations=(IssueLocation(table_path),),
                        context={
                            "section": section,
                            "lengths": cast(JsonValue, mismatch_lengths),
                        },
                    )
                )
                results.append(
                    SkippedCheck(
                        scope=f"{section}:{table_name}:row_checks",
                        reason=(
                            f"Some checks could not run because {section_label} rows "
                            "do not have matching values. Fix the listed issue, then "
                            "select Check again."
                        ),
                        blocking_paths=(IssueLocation(table_path),),
                    )
                )
                continue

            expense_years = _as_sequence(row.get("expense_year"))
            if expense_years is not None:
                results.extend(
                    _year_bounds_issues(
                        section=section,
                        table_name=table_name,
                        field_name="expense_year",
                        field_label="Year",
                        code="EXPENSE_YEAR_OUTSIDE_PROJECT",
                        years=expense_years,
                        bounds=bounds,
                        path_prefix=path_prefix,
                    )
                )
            if section in _FINAL_YEAR_SECTIONS:
                final_years = _as_sequence(row.get("final_year"))
                if final_years is not None:
                    results.extend(
                        _year_bounds_issues(
                            section=section,
                            table_name=table_name,
                            field_name="final_year",
                            field_label="Final Year",
                            code="FINAL_YEAR_OUTSIDE_PROJECT",
                            years=final_years,
                            bounds=bounds,
                            path_prefix=path_prefix,
                        )
                    )
            if section in _PIS_SECTIONS:
                pis_years = _as_sequence(row.get("pis_year"))
                if pis_years is not None:
                    results.extend(
                        _year_bounds_issues(
                            section=section,
                            table_name=table_name,
                            field_name="pis_year",
                            field_label="PIS Year (placed-in-service year)",
                            code="PIS_YEAR_OUTSIDE_PROJECT",
                            years=pis_years,
                            bounds=bounds,
                            path_prefix=path_prefix,
                        )
                    )
    return tuple(results)


def collect_lifting_structural_issues(
    payload: Mapping[str, object],
    bounds: ProjectBounds,
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Collect structural/date/array issues for every submitted lifting entry."""
    lifting = _as_mapping(payload.get("lifting"))
    if lifting is None:
        return ()
    issues: list[ValidationIssue] = []
    for name, entry in lifting.items():
        row = _as_mapping(entry)
        if row is None:
            continue
        entry_path = _path(path_prefix, "lifting", name)
        missing = [
            field for field in _LIFTING_REQUIRED_FIELDS if row.get(field) is None
        ]
        if missing:
            labels = ", ".join(_field_label(field) for field in missing)
            issues.append(
                ValidationIssue(
                    code="LIFTING_REQUIRED_FIELD_MISSING",
                    severity="error",
                    message=(
                        f"Lifting entry '{name}' is missing {labels}. Each entered "
                        "row needs a value for the required fields."
                    ),
                    guidance=(
                        "Review the affected Lifting row, then select Check again."
                    ),
                    locations=(IssueLocation(entry_path),),
                    context={"missing_fields": cast(JsonValue, missing)},
                )
            )
            continue

        if period_is_reversed(row.get("start_year"), row.get("end_year")):
            issues.append(
                ValidationIssue(
                    code="COMPONENT_PERIOD_REVERSED",
                    severity="error",
                    message=(
                        f"Lifting entry '{name}' period start year is after its end "
                        "year. Correct the start and end years, then select Check again."
                    ),
                    guidance="Review the affected Lifting row, then select Check again.",
                    locations=(IssueLocation(entry_path),),
                    context={"section": "lifting"},
                )
            )
            continue

        mismatch = _component_project_mismatch(row, bounds)
        if mismatch:
            fields = ", ".join(_field_label(field) for field, _ in mismatch)
            issues.append(
                ValidationIssue(
                    code="COMPONENT_PERIOD_MISMATCH",
                    severity="error",
                    message=(
                        f"Lifting entry '{name}' {fields} do not match the project "
                        f"period ({bounds.start_year}-{bounds.end_year}). Review the "
                        "entry period and the project dates in General Configuration."
                    ),
                    guidance=(
                        "Review Lifting and the project dates in General Configuration, "
                        "then select Check again."
                    ),
                    locations=(IssueLocation(entry_path),),
                    context={
                        "section": "lifting",
                        "project_start_year": bounds.start_year,
                        "project_end_year": bounds.end_year,
                    },
                )
            )
            continue

        lengths = {
            field_name: len(sequence)
            for field_name in _LIFTING_ARRAY_FIELDS
            if (sequence := _as_sequence(row.get(field_name))) is not None
        }
        mismatch_lengths = unequal_lengths(lengths)
        if mismatch_lengths is not None:
            visible = ", ".join(_field_label(field) for field in mismatch_lengths)
            issues.append(
                ValidationIssue(
                    code="ARRAY_LENGTH_MISMATCH",
                    severity="error",
                    message=(
                        "The input rows do not have matching values for "
                        f"{visible}. Each entered row needs a value for the "
                        "required fields. Review the affected input table, then "
                        "select Check again."
                    ),
                    guidance=(
                        "Review the affected Lifting row, then select Check again."
                    ),
                    locations=(IssueLocation(entry_path),),
                    context={"section": "lifting", "lengths": cast(JsonValue, mismatch_lengths)},
                )
            )
            continue

        prod_years = _as_sequence(row.get("prod_year"))
        if prod_years is not None:
            issues.extend(
                _lifting_year_bounds_issues(
                    name=name,
                    field_name="prod_year",
                    field_label="Production Year",
                    code="PRODUCTION_YEAR_OUTSIDE_PROJECT",
                    years=prod_years,
                    bounds=bounds,
                    path_prefix=path_prefix,
                )
            )

        prod_rate = _as_sequence(row.get("prod_rate"))
        lifting_rate = _as_sequence(row.get("lifting_rate"))
        if (
            prod_rate is not None
            and lifting_rate is not None
            and lifting_exceeds_production(lifting_rate, prod_rate)
        ):
            issues.append(
                ValidationIssue(
                    code="LIFTING_RATE_EXCEEDS_PRODUCTION",
                    severity="error",
                    message=(
                        f"Lifting entry '{name}' has a lifting rate larger than its "
                        "production rate. Correct the lifting or production rate, "
                        "then select Check again."
                    ),
                    guidance=(
                        "Review the affected Lifting row, then select Check again."
                    ),
                    locations=(IssueLocation(entry_path),),
                    context={"section": "lifting"},
                )
            )
    return tuple(issues)


def _lifting_year_bounds_issues(
    *,
    name: str,
    field_name: str,
    field_label: str,
    code: str,
    years: Iterable[object],
    bounds: ProjectBounds,
    path_prefix: tuple[str | int, ...],
) -> list[ValidationIssue]:
    low = year_below(years, bounds.start_year)
    high = year_above(years, bounds.end_year)
    if low is None and high is None:
        return []
    actual_year = low if low is not None else high
    indices = [
        index
        for index, value in enumerate(years)
        if _as_int(value) in {low, high} and _as_int(value) is not None
    ]
    if not indices:
        indices = [0]
    return [
        ValidationIssue(
            code=code,
            severity="error",
            message=(
                f"{field_label} is {actual_year}; the project runs from "
                f"{bounds.start_year} to {bounds.end_year}. Correct {field_label} "
                "if the entry belongs within that period, or review the project "
                "dates in General Configuration."
            ),
            guidance="Review the affected Lifting row, then select Check again.",
            locations=(
                IssueLocation(
                    _path(path_prefix, "lifting", name, field_name, indices[0])
                ),
            ),
            context={
                "actual_year": cast(JsonValue, actual_year),
                "start_year": bounds.start_year,
                "end_year": bounds.end_year,
            },
        )
    ]


# ── Task 2C fiscal / summary / analysis predicates and preflight ────────
#
# These mirror the existing engine rules for summary and economic indicators,
# the cashflow reference year, tax/inflation arrays, depreciation, sensitivity,
# uncertainty and optimization. The engine check site and the API preflight call
# the same helper so the two cannot drift apart. They never import the engine
# modules, so the engine can depend on them without a cycle.

_NPV_MODE_LABELS: frozenset[str] = frozenset(
    {
        "SKK Full Cycle Real Terms",
        "SKK Full Cycle Nominal Terms",
        "Full Cycle Real Terms",
        "Full Cycle Nominal Terms",
        "Point Forward",
    }
)
_DISCOUNTING_MODE_LABELS: frozenset[str] = frozenset({"End Year", "Mid Year"})
_UNCERTAINTY_DISTRIBUTIONS: frozenset[str] = frozenset(
    {"uniform", "triangular", "normal", "lognormal"}
)
_OPTIMIZATION_TARGET_LABELS: frozenset[str] = frozenset({"IRR", "NPV", "PI"})
_OPTIMIZATION_PARAMETER_LABELS: frozenset[str] = frozenset(
    {
        "Oil Contractor Pre Tax",
        "Gas Contractor Pre Tax",
        "Oil FTP Portion",
        "Gas FTP Portion",
        "Oil IC",
        "Gas IC",
        "Oil DMO Fee",
        "Gas Dmo Fee",
        "VAT Rate",
        "Effective Tax Rate",
        "Ministerial Discretion",
        "VAT Discount",
        "LBT Discount",
        "Depreciation Acceleration",
    }
)


def npv_mode_is_valid(mode: object) -> bool:
    """True when the string is one of the engine's accepted NPV modes."""
    return isinstance(mode, str) and mode in _NPV_MODE_LABELS


def discounting_mode_is_valid(mode: object) -> bool:
    """True when the string is one of the engine's accepted discounting modes."""
    return isinstance(mode, str) and mode in _DISCOUNTING_MODE_LABELS


def reference_year_violation(
    reference_year: object, start_year: object, end_year: object
) -> str | None:
    """Return ``"before"``/``"after"`` when a reference year is out of range.

    Mirrors ``tools.summary.get_summary`` and ``Summary.__post_init__`` (and the
    ``Cashflow.check_reference_year`` rule): the reference year must lie within
    the project years.
    """
    reference = _as_int(reference_year)
    start = _as_int(start_year)
    end = _as_int(end_year)
    if reference is None or start is None or end is None:
        return None
    if reference < start:
        return "before"
    if reference > end:
        return "after"
    return None


def year_array_outside_project(
    years: Iterable[object], start_year: int, end_year: int
) -> str | None:
    """Return ``"before"``/``"after"`` when any year array value is out of range.

    Mirrors ``econ.costs_tools.get_cost_adjustment_by_inflation`` for the
    ``year_inflation`` array.
    """
    if year_below(years, start_year) is not None:
        return "before"
    if year_above(years, end_year) is not None:
        return "after"
    return None


def depreciation_factor_out_of_range(value: object) -> bool:
    """True when a depreciation factor falls outside the engine's ``[0, 1]``.

    Mirrors ``econ.depreciation.psc_declining_balance_depreciation_rate``.
    """
    numeric = _as_float(value)
    if numeric is None:
        return False
    return numeric < 0.0 or numeric > 1.0


def salvage_exceeds_cost(salvage_value: object, cost: object) -> bool:
    """True when the salvage value is strictly greater than its cost.

    Mirrors ``econ.depreciation.prepare_amortization``.
    """
    salvage = _as_float(salvage_value)
    base = _as_float(cost)
    if salvage is None or base is None:
        return False
    return salvage > base


def pis_year_appearance_count(years: Iterable[object], pis_year: object) -> int:
    """Count how many project years equal the placed-in-service year.

    Mirrors ``econ.depreciation.unit_of_production_rate`` (``match.size != 1``).
    """
    target = _as_int(pis_year)
    if target is None:
        return 0
    return sum(1 for year in years if _as_int(year) == target)


def deviation_is_negative(value: object) -> bool:
    """True when a sensitivity deviation is strictly negative."""
    numeric = _as_float(value)
    return numeric is not None and numeric < 0.0


def increment_not_positive(value: object) -> bool:
    """True when a sensitivity increment is zero or negative."""
    numeric = _as_float(value)
    return numeric is not None and numeric <= 0.0


def step_not_positive(value: object) -> bool:
    """True when a sensitivity step count is zero or negative."""
    numeric = _as_float(value)
    return numeric is not None and numeric <= 0.0


def run_number_not_positive(value: object) -> bool:
    """True when a Monte Carlo run count is zero or negative."""
    numeric = _as_int(value)
    return numeric is not None and numeric < 1


def uncertainty_distribution_is_valid(value: object) -> bool:
    """True when the distribution name matches the engine's accepted set."""
    return isinstance(value, str) and value.strip().lower() in _UNCERTAINTY_DISTRIBUTIONS


def uncertainty_parameters_violation(
    *,
    minimum: object,
    mean: object,
    maximum: object,
    std_dev: object,
    distribution: object,
) -> str | None:
    """Return the engine's first violation code for one Monte Carlo parameter set.

    Mirrors ``optimize.uncertainty.get_multipliers_montecarlo``: an all-equal
    triple is accepted (the engine returns ones), a zero mean, ``max <= min``,
    an unrecognized distribution, a zero standard deviation for a normal family,
    and non-positive inputs for a log-normal are each rejected in that order.
    """
    low = _as_float(minimum)
    mid = _as_float(mean)
    high = _as_float(maximum)
    std = _as_float(std_dev)
    if low is None or mid is None or high is None or std is None:
        return None
    if low == mid == high:
        return None
    if mid == 0.0:
        return "zero_mean"
    if high <= low:
        return "range"
    if not uncertainty_distribution_is_valid(distribution):
        return "distribution"
    name = cast(str, distribution).strip().lower()
    if name in ("normal", "lognormal") and std == 0.0:
        return "zero_std"
    if name == "lognormal" and (low <= 0.0 or mid <= 0.0 or high <= 0.0):
        return "lognormal_nonpositive"
    return None


def adjustment_out_of_unit_range(value: object) -> bool:
    """True when an optimization adjustment value falls outside ``[0, 1]``."""
    numeric = _as_float(value)
    if numeric is None:
        return False
    return numeric < 0.0 or numeric > 1.0


def useful_life_too_short(value: object) -> bool:
    """True when a useful-life value is below the engine's minimum of 2."""
    numeric = _as_int(value)
    return numeric is not None and numeric < 2


def _analysis_issue(
    *,
    code: str,
    message: str,
    guidance: str,
    path: tuple[str | int, ...],
    context: Mapping[str, JsonValue] | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity="error",
        message=message,
        guidance=guidance,
        locations=(IssueLocation(path),),
        context=dict(context or {}),
    )


def _summary_reference_year(arguments: Mapping[str, object]) -> object:
    """The reference year the adapter passes to a non-transition summary.

    Non-transition contracts receive ``discount_rate_start_year`` directly
    (``api.adapter._contract_summary``); when it is unset the engine defaults the
    reference year to the project start, so there is no violation. The
    ``reference_year`` field is only consumed by the transition/combine summary
    path, which this collector does not own.
    """
    return arguments.get("discount_rate_start_year")


def collect_summary_options_issues(
    arguments: Mapping[str, object],
    bounds: ProjectBounds | None,
    *,
    path_prefix: tuple[str | int, ...],
    check_reference_year: bool = True,
    settings_label: str = "the summary settings",
) -> tuple[ValidationIssue, ...]:
    """Validate the shared summary/option arguments the engine consumes.

    Mirrors ``api.adapter.get_summary_dict`` enum conversion and
    ``tools.summary.get_summary`` reference-year bounds. ``path_prefix`` is the
    location prefix of the arguments block (``summary_arguments`` for a
    contract, ``options`` for a batch). Only rules the engine already enforces
    are mirrored. ``check_reference_year`` is false for operations that never
    compute a summary or whose reference semantics differ from a single contract.
    ``settings_label`` names where the user corrects the value so the guidance
    points at the actual owner control (``Options`` for a batch, never the
    contract summary page).
    """
    issues: list[ValidationIssue] = []

    npv_mode = arguments.get("npv_mode")
    if npv_mode is not None and not npv_mode_is_valid(npv_mode):
        issues.append(
            _analysis_issue(
                code="SUMMARY_NPV_MODE_UNRECOGNIZED",
                message=(
                    f"NPV Mode is '{npv_mode}'. Choose one of: "
                    + ", ".join(sorted(_NPV_MODE_LABELS))
                    + "."
                ),
                guidance=(
                    f"Review NPV Mode in {settings_label}, correct it, then "
                    "select Check again."
                ),
                path=_path(path_prefix, "npv_mode"),
                context={
                    "field": "NPV Mode",
                    "actual": str(npv_mode),
                    "allowed": cast(JsonValue, sorted(_NPV_MODE_LABELS)),
                },
            )
        )

    discounting_mode = arguments.get("discounting_mode")
    if discounting_mode is not None and not discounting_mode_is_valid(discounting_mode):
        issues.append(
            _analysis_issue(
                code="SUMMARY_DISCOUNTING_MODE_UNRECOGNIZED",
                message=(
                    f"Discounting Mode is '{discounting_mode}'. Choose one of: "
                    + ", ".join(sorted(_DISCOUNTING_MODE_LABELS))
                    + "."
                ),
                guidance=(
                    f"Review Discounting Mode in {settings_label}, correct it, "
                    "then select Check again."
                ),
                path=_path(path_prefix, "discounting_mode"),
                context={
                    "field": "Discounting Mode",
                    "actual": str(discounting_mode),
                    "allowed": cast(JsonValue, sorted(_DISCOUNTING_MODE_LABELS)),
                },
            )
        )

    reference_year = _summary_reference_year(arguments)
    if check_reference_year and bounds is not None:
        violation = reference_year_violation(
            reference_year, bounds.start_year, bounds.end_year
        )
        if violation is not None:
            actual = _as_int(reference_year)
            issues.append(
                _analysis_issue(
                    code="SUMMARY_REFERENCE_YEAR_OUTSIDE_PROJECT",
                    message=(
                        f"Discounting Reference Year is {actual}; the project runs from "
                        f"{bounds.start_year} to {bounds.end_year}."
                    ),
                    guidance=(
                        f"Review the reference year in {settings_label} or the project "
                        "dates in General Configuration, then select Check again."
                    ),
                    path=_path(path_prefix, "discount_rate_start_year"),
                    context={
                        "field": "Discounting Reference Year",
                        "actual_year": cast(JsonValue, actual),
                        "start_year": bounds.start_year,
                        "end_year": bounds.end_year,
                    },
                )
            )
    return tuple(issues)


def collect_summary_argument_issues(
    payload: Mapping[str, object],
    bounds: ProjectBounds,
    *,
    path_prefix: tuple[str | int, ...] = (),
    check_reference_year: bool = True,
) -> tuple[ValidationIssue, ...]:
    """Validate the submitted ``summary_arguments`` block."""
    arguments = _as_mapping(payload.get("summary_arguments"))
    if arguments is None:
        return ()
    return collect_summary_options_issues(
        arguments,
        bounds,
        path_prefix=_path(path_prefix, "summary_arguments"),
        check_reference_year=check_reference_year,
    )


def collect_sensitivity_argument_issues(
    payload: Mapping[str, object],
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Validate the submitted sensitivity multipliers arguments.

    Mirrors ``optimize.sensitivity.get_sensitivity_multipliers``: deviations must
    be non-negative, and the increment and step must be greater than zero.
    """
    arguments = _as_mapping(payload.get("sensitivity_arguments"))
    if arguments is None:
        return ()
    issues: list[ValidationIssue] = []

    for field_name, label in (
        ("min_deviation", "Minimum Deviation"),
        ("max_deviation", "Maximum Deviation"),
    ):
        value = arguments.get(field_name)
        if value is not None and deviation_is_negative(value):
            issues.append(
                _analysis_issue(
                    code="SENSITIVITY_DEVIATION_NEGATIVE",
                    message=(
                        f"{label} is {value}. Deviations must be non negative."
                    ),
                    guidance=(
                        "Enter a deviation of 0% or more in the sensitivity settings, "
                        "then select Check again."
                    ),
                    path=_path(path_prefix, "sensitivity_arguments", field_name),
                    context={
                        "field": label,
                        "actual": cast(JsonValue, _as_float(value)),
                    },
                )
            )

    increment = arguments.get("increment")
    if increment is not None and increment_not_positive(increment):
        issues.append(
            _analysis_issue(
                code="SENSITIVITY_INCREMENT_NOT_POSITIVE",
                message=(
                    f"Increment is {increment}. Increment must be greater than zero."
                ),
                guidance=(
                    "Enter an increment greater than zero in the sensitivity "
                    "settings, or clear it, then select Check again."
                ),
                path=_path(path_prefix, "sensitivity_arguments", "increment"),
                context={
                    "field": "Increment",
                    "actual": cast(JsonValue, _as_float(increment)),
                },
            )
        )

    step = arguments.get("step")
    if step is not None and step_not_positive(step):
        issues.append(
            _analysis_issue(
                code="SENSITIVITY_STEP_NOT_POSITIVE",
                message=f"Step is {step}. Step must be greater than zero.",
                guidance=(
                    "Enter a step greater than zero in the sensitivity settings, "
                    "then select Check again."
                ),
                path=_path(path_prefix, "sensitivity_arguments", "step"),
                context={
                    "field": "Step",
                    "actual": cast(JsonValue, _as_float(step)),
                },
            )
        )
    return tuple(issues)


_UNCERTAINTY_PARAMETER_FIELDS: tuple[tuple[str, str], ...] = (
    ("oil_price", "Oil Price"),
    ("gas_price", "Gas Price"),
    ("opex", "OPEX"),
    ("capex", "CAPEX"),
    ("lifting", "Lifting"),
)

_UNCERTAINTY_VIOLATION_CODES: dict[str, str] = {
    "zero_mean": "UNCERTAINTY_ZERO_MEAN",
    "range": "UNCERTAINTY_RANGE_INVALID",
    "distribution": "UNCERTAINTY_DISTRIBUTION_UNRECOGNIZED",
    "zero_std": "UNCERTAINTY_ZERO_STD",
    "lognormal_nonpositive": "UNCERTAINTY_LOGNORMAL_NONPOSITIVE",
}


def collect_uncertainty_argument_issues(
    payload: Mapping[str, object],
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Validate the submitted Monte Carlo uncertainty arguments.

    Mirrors ``optimize.uncertainty.get_multipliers_montecarlo`` per parameter
    group and the positive run count the simulation requires.
    """
    arguments = _as_mapping(payload.get("uncertainty_arguments"))
    if arguments is None:
        return ()
    issues: list[ValidationIssue] = []

    run_number = arguments.get("run_number")
    if run_number is not None and run_number_not_positive(run_number):
        issues.append(
            _analysis_issue(
                code="UNCERTAINTY_RUN_NUMBER_INVALID",
                message=(
                    f"Number of Runs is {run_number}. The number of runs must be "
                    "at least 1."
                ),
                guidance=(
                    "Enter a number of runs of 1 or more in the uncertainty "
                    "settings, then select Check again."
                ),
                path=_path(path_prefix, "uncertainty_arguments", "run_number"),
                context={
                    "field": "Number of Runs",
                    "actual": cast(JsonValue, _as_int(run_number)),
                },
            )
        )

    for key, label in _UNCERTAINTY_PARAMETER_FIELDS:
        distribution_field = f"{key}_distribution"
        distribution = arguments.get(distribution_field)
        if distribution is not None and not uncertainty_distribution_is_valid(
            distribution
        ):
            issues.append(
                _analysis_issue(
                    code="UNCERTAINTY_DISTRIBUTION_UNRECOGNIZED",
                    message=(
                        f"{label} Distribution is '{distribution}'. Choose one of: "
                        + ", ".join(sorted(name.title() for name in _UNCERTAINTY_DISTRIBUTIONS))
                        + "."
                    ),
                    guidance=(
                        "Review the distribution in the uncertainty settings, "
                        "correct it, then select Check again."
                    ),
                    path=_path(path_prefix, "uncertainty_arguments", distribution_field),
                    context={
                        "field": f"{label} Distribution",
                        "actual": str(distribution),
                    },
                )
            )
            continue
        violation = uncertainty_parameters_violation(
            minimum=arguments.get(f"min_{key}"),
            mean=arguments.get(f"mean_{key}"),
            maximum=arguments.get(f"max_{key}"),
            std_dev=arguments.get(f"{key}_stddev"),
            distribution=distribution,
        )
        if violation is None:
            continue
        code = _UNCERTAINTY_VIOLATION_CODES.get(violation, "UNCERTAINTY_RANGE_INVALID")
        issues.append(
            _analysis_issue(
                code=code,
                message=_uncertainty_message(violation, label, arguments, key),
                guidance=(
                    "Review the minimum, mean, maximum and standard deviation in "
                    "the uncertainty settings, then select Check again."
                ),
                path=_path(path_prefix, "uncertainty_arguments", f"min_{key}"),
                context={
                    "field": label,
                    "min": cast(JsonValue, _as_float(arguments.get(f"min_{key}"))),
                    "mean": cast(JsonValue, _as_float(arguments.get(f"mean_{key}"))),
                    "max": cast(JsonValue, _as_float(arguments.get(f"max_{key}"))),
                    "stddev": cast(JsonValue, _as_float(arguments.get(f"{key}_stddev"))),
                },
            )
        )
    return tuple(issues)


def _uncertainty_message(
    violation: str, label: str, arguments: Mapping[str, object], key: str
) -> str:
    low = _as_float(arguments.get(f"min_{key}"))
    mid = _as_float(arguments.get(f"mean_{key}"))
    high = _as_float(arguments.get(f"max_{key}"))
    if violation == "zero_mean":
        return f"{label} mean is 0. The mean must not be zero."
    if violation == "range":
        return (
            f"{label} maximum ({high}) must be greater than its minimum ({low})."
        )
    if violation == "zero_std":
        return f"{label} standard deviation is 0. It must be greater than zero."
    if violation == "lognormal_nonpositive":
        return (
            f"{label} uses a LogNormal distribution, so its minimum ({low}), mean "
            f"({mid}) and maximum ({high}) must all be positive."
        )
    return f"{label} parameters are inconsistent (minimum {low}, mean {mid}, maximum {high})."


def collect_optimization_argument_issues(
    payload: Mapping[str, object],
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> tuple[ValidationIssue, ...]:
    """Validate the submitted optimization target and parameter names.

    Mirrors ``optimize.optimization.optimize_psc_core`` target/parameter
    dispatch.
    """
    arguments = _as_mapping(payload.get("optimization_arguments"))
    if arguments is None:
        return ()
    issues: list[ValidationIssue] = []

    target = arguments.get("target_parameter")
    if target is not None and not (
        isinstance(target, str) and target in _OPTIMIZATION_TARGET_LABELS
    ):
        issues.append(
            _analysis_issue(
                code="OPTIMIZATION_TARGET_UNRECOGNIZED",
                message=(
                    f"Optimization Target is '{target}'. Choose one of: "
                    + ", ".join(sorted(_OPTIMIZATION_TARGET_LABELS))
                    + "."
                ),
                guidance=(
                    "Review the optimization target, correct it, then select "
                    "Check again."
                ),
                path=_path(path_prefix, "optimization_arguments", "target_parameter"),
                context={"field": "Optimization Target", "actual": str(target)},
            )
        )

    dict_optimization = _as_mapping(arguments.get("dict_optimization"))
    if dict_optimization is not None:
        parameters = _as_sequence(dict_optimization.get("parameter"))
        if parameters is not None:
            for index, parameter in enumerate(parameters):
                if isinstance(parameter, str) and parameter in _OPTIMIZATION_PARAMETER_LABELS:
                    continue
                issues.append(
                    _analysis_issue(
                        code="OPTIMIZATION_PARAMETER_UNRECOGNIZED",
                        message=(
                            f"Optimization Parameter is '{parameter}'. Choose a "
                            "parameter from the optimization parameter list."
                        ),
                        guidance=(
                            "Review the optimization parameter, correct it, then "
                            "select Check again."
                        ),
                        path=_path(
                            path_prefix,
                            "optimization_arguments",
                            "dict_optimization",
                            "parameter",
                            index,
                        ),
                        context={
                            "field": "Optimization Parameter",
                            "actual": str(parameter),
                        },
                    )
                )
    return tuple(issues)


def _malformed_onstream_date_issues(
    setup: Mapping[str, object],
    path_prefix: tuple[str | int, ...],
) -> tuple[ValidationIssue, ...]:
    """Located issues for non-null onstream dates the engine cannot parse.

    ``None`` is the accepted nullable default and is never reported. Only a
    supplied value (string or int) that fails the engine date parser is a
    malformed date; the actual value and usable format are named so the user can
    correct the intended date instead of receiving an execution-time 500.
    """
    issues: list[ValidationIssue] = []
    for fluid in ("oil", "gas"):
        key = f"{fluid}_onstream_date"
        raw = setup.get(key)
        if raw is None or parse_engine_date(raw) is not None:
            continue
        label = f"{fluid.capitalize()} Onstream Date"
        issues.append(
            ValidationIssue(
                code="ONSTREAM_DATE_INVALID",
                severity="error",
                message=(
                    f"{label} contains '{raw}'. Enter a valid date, for example "
                    "01/01/2029, or leave it blank to use the calculated default."
                ),
                guidance=(
                    "Review the onstream date and the dependent onstream and cost "
                    "classification checks await its correction, then select Check "
                    "again."
                ),
                locations=(IssueLocation(_path(path_prefix, "setup", key)),),
                context={"field": label, "entered_value": str(raw)},
            )
        )
    return tuple(issues)


def _resolve_effective_onstream_setup(
    setup: Mapping[str, object],
    bounds: ProjectBounds,
    onstream_inputs: Mapping[str, OnstreamInputs] | None,
    collector: ValidationCollector,
    path_prefix: tuple[str | int, ...],
) -> Mapping[str, object] | None:
    """Return a setup view with effective onstream dates, or ``None``.

    ``None`` means the dates could not be resolved. That happens when a
    supplied value is malformed, when no prepared inputs exist for a nullable
    date, or when a supplied year violates the engine rule. Malformed and
    supplied-year violations are added to the collector as located issues; the
    caller then discloses the dependent skips.
    """
    # A non-null value that the engine cannot parse is malformed input, not an
    # accepted nullable default. Report it before resolving any fallback so a
    # bad date is never certified (the execution path would still parse it).
    malformed = _malformed_onstream_date_issues(setup, path_prefix)
    if malformed:
        for issue in malformed:
            collector.add(issue)
        return None

    if not onstream_inputs:
        # Without prepared revenue/production inputs the effective engine rule
        # cannot be evaluated. Keep the submitted years when they parse so an
        # accepted both-supplied payload is not newly blocked; a nullable date
        # stays an unresolved prerequisite for the caller to disclose.
        return setup if _onstream_years(setup) is not None else None

    resolved: dict[str, object] = dict(setup)
    for fluid in ("oil", "gas"):
        key = f"{fluid}_onstream_date"
        inputs = onstream_inputs.get(fluid)
        if inputs is None:
            return None
        try:
            effective = resolve_onstream_year(
                fluid=fluid,
                provided_year=date_year(setup.get(key)),
                start_year=bounds.start_year,
                end_year=bounds.end_year,
                revenue=inputs.revenue,
                lifting_prod_years=inputs.lifting_prod_years,
            )
        except OnstreamResolutionError as exc:
            collector.add(
                ValidationIssue(
                    code=exc.code,
                    severity="error",
                    message=exc.message,
                    guidance=(
                        "Correct the onstream year or the lifting and revenue inputs, "
                        "then select Check again."
                    ),
                    locations=(IssueLocation(_path(path_prefix, "setup", key)),),
                    context=dict(exc.context),
                )
            )
            return None
        # Use the engine's accepted ``%d/%m/%Y`` format so dependent checks that
        # read the effective view (``date_year``/``_onstream_years``) parse it
        # exactly like a submitted date.
        resolved[key] = f"01/01/{effective.year}"

    return resolved


@dataclass(frozen=True)
class OperationConsumption:
    """Which optional argument blocks the actual route worker consumes.

    The policy is derived from the workers, not the route names: a route that
    builds no summary (Gross Split split, Transition split/table) ignores the
    submitted summary options; only the dedicated ``*/sensitivity``,
    ``*/uncertainty`` and ``*/optimization`` workers read those blocks. Batch
    Combine/Incremental build every case summary and read the batch-global
    options, but never the per-case analysis blocks.
    """

    summary: bool
    sensitivity: bool
    uncertainty: bool
    optimization: bool


_VALIDATION_ONLY_CONSUMPTION = OperationConsumption(True, True, True, True)

_SUMMARYLESS_OPERATIONS: tuple[str, ...] = (
    "grosssplit/split",
    "transition/split",
    "transition/table",
)


def operation_consumption(operation: str | None) -> OperationConsumption:
    """Return the consumed optional blocks for one actual operation.

    ``None`` (internal callers without a route) and ``validate`` are
    validation-only: every present block is checked so a preflight can disclose
    an invalid block. A calculation operation checks only what its worker reads.
    """
    if operation is None or operation == "validate":
        return _VALIDATION_ONLY_CONSUMPTION
    return OperationConsumption(
        summary=operation not in _SUMMARYLESS_OPERATIONS,
        sensitivity=operation.endswith("/sensitivity"),
        uncertainty=operation.endswith("/uncertainty"),
        optimization=operation.endswith("/optimization"),
    )


def collect_contract_issues(
    payload: Mapping[str, object],
    contract_type: ContractType,
    collector: ValidationCollector,
    *,
    path_prefix: tuple[str | int, ...] = (),
    check_summary_reference: bool = True,
    onstream_inputs: Mapping[str, OnstreamInputs] | None = None,
    operation: str | None = None,
    consumption: OperationConsumption | None = None,
) -> None:
    """Run all shared predicates for one contract payload into a collector.

    When the submitted onstream dates are nullable, ``onstream_inputs`` lets the
    collector resolve the engine's effective onstream years so valid nullable
    inputs are checked with the same defaults execution uses. The submitted
    payload is never mutated: a shallow effective view is used for dependent
    checks only.

    ``operation`` selects the actual worker's consumption policy so an unused
    analysis block does not block a calculation. ``consumption`` is an internal
    override for nested transition children, which never run their own analysis
    workers.
    """
    consumed = consumption if consumption is not None else operation_consumption(operation)
    collector.extend(
        collect_malformed_numeric_issues(payload, path_prefix=path_prefix)
    )
    collector.extend(
        collect_nonfinite_numeric_issues(payload, path_prefix=path_prefix)
    )
    if consumed.sensitivity:
        collector.extend(
            collect_sensitivity_argument_issues(payload, path_prefix=path_prefix)
        )
    if consumed.uncertainty:
        collector.extend(
            collect_uncertainty_argument_issues(payload, path_prefix=path_prefix)
        )
    if consumed.optimization:
        collector.extend(
            collect_optimization_argument_issues(payload, path_prefix=path_prefix)
        )
    if contract_type == "Gross Split":
        variable_split_issues = collect_grosssplit_variable_split_issues(
            payload, path_prefix=path_prefix
        )
        collector.extend(variable_split_issues)
        if variable_split_issues:
            # The variable-split components are a prerequisite for the split
            # calculation, so disclose that the dependent check could not run.
            collector.skip(
                SkippedCheck(
                    scope="grosssplit_variable_split",
                    reason=(
                        "Some checks could not run because the Gross Split variable "
                        "split components need correction. Fix the listed issue, then "
                        "select Check again."
                    ),
                    blocking_paths=tuple(
                        location
                        for issue in variable_split_issues
                        for location in issue.locations
                    )[:10],
                )
            )
        collector.extend(
            collect_grosssplit_pod1_noncapital_issues(
                payload, path_prefix=path_prefix
            )
        )

    setup = _as_mapping(payload.get("setup"))
    if setup is None:
        collector.skip(
            SkippedCheck(
                scope="project_dates",
                reason="Project setup is missing, so date-dependent checks were skipped.",
                blocking_paths=(IssueLocation(_path(path_prefix, "setup")),),
            )
        )
        return

    bounds, date_issues = collect_project_date_issues(payload, path_prefix=path_prefix)
    collector.extend(date_issues)
    if bounds is None:
        if date_issues:
            collector.skip(
                SkippedCheck(
                    scope="project_period_checks",
                    reason=(
                        "Some checks could not run because Project Start or Project End "
                        "needs correction. Fix the listed issue, then select Check again."
                    ),
                    blocking_paths=(
                        IssueLocation(_path(path_prefix, "setup", "start_date")),
                        IssueLocation(_path(path_prefix, "setup", "end_date")),
                    ),
                )
            )
        return

    if consumed.summary:
        collector.extend(
            collect_summary_argument_issues(
                payload,
                bounds,
                path_prefix=path_prefix,
                check_reference_year=check_summary_reference,
            )
        )

    for check in collect_cost_structural_issues(
        payload, bounds, path_prefix=path_prefix
    ):
        if isinstance(check, SkippedCheck):
            collector.skip(check)
        else:
            collector.add(check)
    collector.extend(
        collect_lifting_structural_issues(
            payload, bounds, path_prefix=path_prefix
        )
    )

    # CR/GS attribute checks depend only on the project duration, not onstream.
    if contract_type == "Cost Recovery":
        collector.extend(
            collect_costrecovery_attribute_issues(
                payload, bounds, path_prefix=path_prefix
            )
        )
    elif contract_type == "Gross Split":
        collector.extend(
            collect_grosssplit_attribute_issues(
                payload, bounds, path_prefix=path_prefix
            )
        )

    effective_setup = _resolve_effective_onstream_setup(
        setup, bounds, onstream_inputs, collector, path_prefix
    )
    if effective_setup is None:
        if _sections_present(payload):
            collector.skip(
                SkippedCheck(
                    scope="preonstream_window",
                    reason=(
                        "Onstream dates are required to check pre-onstream cost years, "
                        "so that check was skipped."
                    ),
                    blocking_paths=(
                        IssueLocation(_path(path_prefix, "setup", "oil_onstream_date")),
                        IssueLocation(_path(path_prefix, "setup", "gas_onstream_date")),
                    ),
                )
            )
        collector.skip(
            SkippedCheck(
                scope="classification_checks",
                reason=(
                    "Onstream dates are required to check cost classifications, "
                    "so those checks were skipped."
                ),
                blocking_paths=(
                    IssueLocation(_path(path_prefix, "setup", "oil_onstream_date")),
                    IssueLocation(_path(path_prefix, "setup", "gas_onstream_date")),
                ),
            )
        )
        return

    if effective_setup is not setup:
        # An internal effective view for dependent checks only. The submitted
        # dates in the caller's mapping are never overwritten.
        payload = {**payload, "setup": effective_setup}

    collector.extend(
        collect_approval_year_issues(payload, bounds, path_prefix=path_prefix)
    )
    collector.extend(
        collect_cost_classification_issues(payload, bounds, path_prefix=path_prefix)
    )
    collector.extend(
        collect_sunk_cutoff_issues(payload, bounds, path_prefix=path_prefix)
    )
    collector.extend(
        collect_postonstream_window_issues(payload, bounds, path_prefix=path_prefix)
    )
    collector.extend(
        collect_preonstream_window_issues(payload, path_prefix=path_prefix)
    )
    if contract_type == "Cost Recovery":
        collector.extend(
            collect_costrecovery_capital_sunk_issues(
                payload, bounds, path_prefix=path_prefix
            )
        )
    elif contract_type == "Gross Split":
        collector.extend(
            collect_grosssplit_nonpod1_sunk_issues(
                payload, bounds, path_prefix=path_prefix
            )
        )
    collector.extend(
        collect_capital_pis_issues(
            payload, bounds, contract_type, path_prefix=path_prefix
        )
    )
