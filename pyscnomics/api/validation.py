"""Contract-level input validation for the API boundary.

``validate_contract_input`` runs the shared engine predicates without any
numerical work. ``validate_cases_input`` applies the same predicates to every
nested case of a combine/increment request and refuses to certify a case whose
``data`` is not a real contract payload. ``require_valid_input`` turns a
non-calculable report into the structured exception handled by
:mod:`pyscnomics.api.errors`.

Nested structural problems are raised as FastAPI's own
``RequestValidationError`` so the standard ``detail`` array (with the ``body``
prefix and pydantic diagnostics) is preserved alongside the shared envelope,
exactly like a malformed top-level request body.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

from pyscnomics.api.converter import Data, DataTransition
from pyscnomics.input_validation import (
    IssueLocation,
    SkippedCheck,
    ValidationCollector,
    ValidationIssue,
    ValidationReport,
)
from pyscnomics.api.validation_inputs import prepare_onstream_inputs
from pyscnomics.operation_validation import StandaloneOperation, collect_operation_issues
from pyscnomics.validation_rules import (
    ContractType,
    OnstreamInputs,
    OperationConsumption,
    ProjectBounds,
    TransitionSchemeResolutionError,
    collect_contract_issues,
    collect_optimization_argument_issues,
    collect_project_date_issues,
    collect_sensitivity_argument_issues,
    collect_summary_options_issues,
    collect_uncertainty_argument_issues,
    operation_consumption,
    resolve_transition_scheme,
)

_CTR_TO_CONTRACT: dict[int, ContractType] = {
    0: "Base Project",
    1: "Cost Recovery",
    2: "Gross Split",
    3: "Transition",
    4: "Transition",
    5: "Transition",
    6: "Transition",
}

# The contract model used to structurally validate one nested case ``data``.
# A single case is validated with ``Data``; every transition index (3..6) uses
# ``DataTransition`` so both ``contract_1`` and ``contract_2`` are checked.
_NESTED_CASE_MODELS: dict[int, type[BaseModel]] = {
    0: Data,
    1: Data,
    2: Data,
    3: DataTransition,
    4: DataTransition,
    5: DataTransition,
    6: DataTransition,
}


def _nested_case_entry(ctr: object) -> tuple[type[BaseModel], ContractType] | None:
    if isinstance(ctr, bool) or not isinstance(ctr, int):
        return None
    model = _NESTED_CASE_MODELS.get(ctr)
    contract_type = _CTR_TO_CONTRACT.get(ctr)
    if model is None or contract_type is None:
        return None
    return model, contract_type


def _prepared_onstream_inputs(
    data: Mapping[str, object],
    path_prefix: tuple[str | int, ...] = (),
) -> Mapping[str, OnstreamInputs] | None:
    """Prepare the effective onstream inputs when the project dates are safe."""
    bounds, _ = collect_project_date_issues(data, path_prefix=path_prefix)
    if bounds is None:
        return None
    prepared = prepare_onstream_inputs(data, bounds)
    return prepared or None


def validate_contract_input(
    data: Mapping[str, object],
    contract_type: ContractType,
    *,
    operation: str,
) -> ValidationReport:
    """Run shared input checks for one contract payload.

    The ``operation`` label selects whether summary-dependent checks apply: the
    split routes never compute a summary, so a default reference year outside the
    project must not block them. The label does not change the other predicates.
    """
    check_summary_reference = not operation.endswith("/split")
    collector = ValidationCollector()
    if contract_type == "Transition":
        _collect_transition_issues(
            data,
            collector,
            operation=operation,
            check_summary_reference=check_summary_reference,
        )
    else:
        collect_contract_issues(
            data,
            contract_type,
            collector,
            check_summary_reference=check_summary_reference,
            onstream_inputs=_prepared_onstream_inputs(data),
            operation=operation,
        )
    return collector.report()


def _collect_transition_root_issues(
    data: Mapping[str, object],
    collector: ValidationCollector,
    *,
    operation: str | None,
    path_prefix: tuple[str | int, ...] = (),
) -> None:
    """Validate the transition root arguments the engine actually consumes.

    A transition route only consumes the root arguments its worker reads. The
    split and table workers build no summary, so a bad root summary enum or
    reference year must not block them; the summary/analysis routes do consume
    those settings and are checked with the effective combined project bounds.
    The exploratory root ``reference_year`` remains accepted (only
    ``discount_rate_start_year`` is a summary reference). Root analysis blocks
    are validated for the operation that runs them, or for every block present
    on the ``validate`` preflight.
    """
    consumed = operation_consumption(operation)
    consumes_summary = consumed.summary
    consumes_sensitivity = consumed.sensitivity
    consumes_uncertainty = consumed.uncertainty
    consumes_optimization = consumed.optimization
    summary = data.get("summary_arguments")
    bounds = _transition_combined_bounds(data) if consumes_summary else None
    if isinstance(summary, Mapping) and consumes_summary:
        collector.extend(
            collect_summary_options_issues(
                cast("Mapping[str, object]", summary),
                bounds,
                path_prefix=(*path_prefix, "summary_arguments"),
                check_reference_year=True,
            )
        )

    if consumes_sensitivity:
        collector.extend(
            collect_sensitivity_argument_issues(data, path_prefix=path_prefix)
        )
    if consumes_uncertainty:
        collector.extend(
            collect_uncertainty_argument_issues(data, path_prefix=path_prefix)
        )
    if consumes_optimization:
        collector.extend(
            collect_optimization_argument_issues(data, path_prefix=path_prefix)
        )


# Whether each transition route consumes the root summary options and the root
# sensitivity/uncertainty/optimization blocks is owned by the shared
# ``validation_rules.operation_consumption`` policy, so the API and the core
# rules cannot disagree. ``validate`` checks every present block.
def _transition_child_consumption(operation: str | None) -> OperationConsumption:
    """A transition child never runs its own analysis workers.

    The worker builds each child with ``summary_result=False`` and never reads
    the child's analysis blocks; only the validation-only preflight checks every
    present block. ``summary`` therefore follows the same rule.
    """
    validation_only = operation is None or operation == "validate"
    return OperationConsumption(
        summary=validation_only,
        sensitivity=validation_only,
        uncertainty=validation_only,
        optimization=validation_only,
    )


def _transition_combined_bounds(
    data: Mapping[str, object],
) -> ProjectBounds | None:
    """The combined span of both transition children, or ``None``.

    The root summary reference year is valid when it falls inside the effective
    combined project period the transition contract spans.
    """
    starts: list[int] = []
    ends: list[int] = []
    for key in ("contract_1", "contract_2"):
        block = data.get(key)
        if not isinstance(block, Mapping):
            continue
        bounds, _ = collect_project_date_issues(cast("Mapping[str, object]", block))
        if bounds is None:
            continue
        starts.append(bounds.start_year)
        ends.append(bounds.end_year)
    if not starts:
        return None
    return ProjectBounds(min(starts), max(ends))


def _collect_transition_issues(
    data: Mapping[str, object],
    collector: ValidationCollector,
    *,
    operation: str | None = None,
    check_summary_reference: bool = True,
    path_prefix: tuple[str | int, ...] = (),
) -> None:
    """Recurse into both transition contract blocks with their real child scheme.

    Each child's scheme is resolved with the same exactly-one-scheme predicate as
    execution (never a preferential Gross Split choice). A child that selects both
    schemes or neither produces a located issue and discloses its dependent scope
    instead of being silently checked. Locations are prefixed with the caller's
    ``path_prefix`` followed by ``contract_1`` or ``contract_2``.
    """
    _collect_transition_root_issues(
        data, collector, operation=operation, path_prefix=path_prefix
    )

    found = False
    for key in ("contract_1", "contract_2"):
        block = data.get(key)
        child_path = (*path_prefix, key)
        if not isinstance(block, Mapping):
            continue
        block_mapping = cast("Mapping[str, object]", block)
        found = True
        period = key.split("_")[-1]
        try:
            scheme = resolve_transition_scheme(block_mapping)
        except TransitionSchemeResolutionError as exc:
            both = len(exc.selected_schemes) > 1
            if both:
                message = (
                    f"Contract {period} has both Cost Recovery and Gross Split "
                    "settings. Use the scheme intended for this contract period and "
                    "review the other settings, then select Check again."
                )
            else:
                message = (
                    f"Contract {period} has neither a Cost Recovery nor a Gross Split "
                    "setting. Choose the scheme intended for this contract period, then "
                    "select Check again."
                )
            collector.add(
                ValidationIssue(
                    code="TRANSITION_SCHEME_UNRECOGNIZED",
                    severity="error",
                    message=message,
                    guidance=(
                        "Review both scheme blocks for this contract period and keep "
                        "only the intended one, then select Check again."
                    ),
                    locations=(
                        IssueLocation((*child_path, "costrecovery")),
                        IssueLocation((*child_path, "grosssplit")),
                    ),
                    context={
                        "contract_period": period,
                        "selected_schemes": list(exc.selected_schemes),
                    },
                )
            )
            collector.skip(
                SkippedCheck(
                    scope=f"{key}:scheme",
                    reason=(
                        f"Contract {period} does not select exactly one scheme, so its "
                        "dependent checks were skipped."
                    ),
                    blocking_paths=(
                        IssueLocation((*child_path, "costrecovery")),
                        IssueLocation((*child_path, "grosssplit")),
                    ),
                )
            )
            continue
        collect_contract_issues(
            block_mapping,
            scheme,
            collector,
            path_prefix=child_path,
            check_summary_reference=check_summary_reference,
            onstream_inputs=_prepared_onstream_inputs(block_mapping, child_path),
            operation=operation,
            consumption=_transition_child_consumption(operation),
        )
    if not found:
        collector.skip(
            SkippedCheck(
                scope="transition_contracts",
                reason=(
                    "Transition contract blocks are missing, so their checks "
                    "were skipped."
                ),
                blocking_paths=(IssueLocation((*path_prefix, "contract_1")),),
            )
        )


def _validated_cases(
    payload: Mapping[str, object],
) -> list[tuple[int, dict[str, object], ContractType]]:
    """Structurally validate every nested case and return the accepted ones.

    A non-mapping case, a missing/non-mapping ``data``, an unknown ``ctr`` or a
    ``data`` that is not a real contract payload is never skipped silently: the
    pydantic diagnostics are collected and raised as a FastAPI
    ``RequestValidationError`` so the standard ``detail`` array and the shared
    envelope are both produced. Accepted cases are returned as their validated
    model dump so nested checks use the same effective defaults as execution.
    """
    cases = payload.get("cases")
    if not isinstance(cases, (list, tuple)):
        return []
    case_items = cast("list[object] | tuple[object, ...]", cases)
    errors: list[dict[str, object]] = []
    accepted: list[tuple[int, dict[str, object], ContractType]] = []
    for index, case in enumerate(case_items):
        base: tuple[str | int, ...] = ("body", "cases", index)
        if not isinstance(case, Mapping):
            errors.append(
                {
                    "type": "model_type",
                    "loc": base,
                    "msg": "Input should be a valid dictionary or instance of CombineCaseBM",
                    "input": case,
                }
            )
            continue
        case_mapping = cast("Mapping[str, object]", case)
        entry = _nested_case_entry(case_mapping.get("ctr"))
        if entry is None:
            errors.append(
                {
                    "type": "union_tag_invalid",
                    "loc": (*base, "ctr"),
                    "msg": (
                        "The case contract type is not a supported Base Project, "
                        "Cost Recovery, Gross Split or Transition index."
                    ),
                    "input": case_mapping.get("ctr"),
                }
            )
            continue
        model, contract_type = entry
        data = case_mapping.get("data")
        if not isinstance(data, Mapping):
            errors.append(
                {
                    "type": "model_type",
                    "loc": (*base, "data"),
                    "msg": "Input should be a valid dictionary or instance of the contract payload",
                    "input": data,
                }
            )
            continue
        data_mapping = cast("Mapping[str, object]", data)
        try:
            instance = model.model_validate(data_mapping)
        except ValidationError as exc:
            for error in exc.errors():
                errors.append(
                    {
                        **error,
                        "loc": (*base, "data", *tuple(error["loc"])),
                    }
                )
            continue
        accepted.append((index, instance.model_dump(), contract_type))
    if errors:
        raise RequestValidationError(errors, body=dict(payload))
    return accepted


def validate_cases_input(
    payload: Mapping[str, object],
    operation: str | None,
) -> ValidationReport:
    """Validate every case in a combine/increment request.

    Each nested ``data`` is first validated against the real contract model
    chosen by the case ``ctr`` (``Data`` for a single contract, ``DataTransition``
    for any transition index). Only structurally valid cases are checked, and
    nested contract locations are prefixed with ``cases/{index}/data`` so the
    frontend can attribute issues to the originating case.

    ``operation`` selects the actual worker's consumption policy: the batch
    calculation workers build each case summary but never run a case's own
    analysis workers, while ``validate`` (preflight) checks every present block.
    """
    consumed = operation_consumption(operation)
    collector = ValidationCollector()

    # Batch-global options are validated once, with ``options``-prefixed
    # locations, and never attributed to a case. Only the shared enum predicates
    # apply; a batch reference year is not required to fall inside every case.
    options = payload.get("options")
    if isinstance(options, Mapping) and consumed.summary:
        collector.extend(
            collect_summary_options_issues(
                cast("Mapping[str, object]", options),
                None,
                path_prefix=("options",),
                check_reference_year=False,
                settings_label="Options",
            )
        )

    for index, data, contract_type in _validated_cases(payload):
        path_prefix = ("cases", index, "data")
        if contract_type == "Transition":
            _collect_transition_issues(
                data, collector, operation=operation, path_prefix=path_prefix
            )
        else:
            collect_contract_issues(
                data,
                contract_type,
                collector,
                path_prefix=path_prefix,
                onstream_inputs=_prepared_onstream_inputs(data, path_prefix),
                operation=operation,
            )
    return collector.report()


def validate_operation_input(
    payload: Mapping[str, object],
    operation: StandaloneOperation,
) -> ValidationReport:
    """Validate one standalone operation's effective input without work."""
    collector = ValidationCollector()
    for item in collect_operation_issues(payload, operation):
        if isinstance(item, SkippedCheck):
            collector.skip(item)
        else:
            collector.add(item)
    return collector.report()


def require_valid_input(report: ValidationReport) -> None:
    """Raise the structured input error when calculation is not permitted.

    A report may certify a calculation only when it is complete, calculable and
    carries no errors. A genuinely incomplete report with no verified errors must
    not enter numerical work merely because ``errors`` is empty: ``to_error``
    turns its skipped prerequisites into the ``VALIDATION_INCOMPLETE`` issue.
    """
    if not report.can_calculate:
        raise report.to_error()
