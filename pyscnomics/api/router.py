"""
Module to handle API router.
"""

from typing import Literal, Protocol, TypeGuard, runtime_checkable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ConfigDict, TypeAdapter, ValidationError

from pyscnomics import __version__
from pyscnomics.api.errors import (
    get_request_id,
    validation_success_response,
)
from pyscnomics.api.validation import (
    require_valid_input,
    validate_cases_input,
    validate_contract_input,
    validate_operation_input,
)
from pyscnomics.api.validation_inputs import normalize_expenditure_request
from pyscnomics.validation_rules import ContractType
from pyscnomics.api.adapter import (
    get_baseproject,
    get_costrecovery,
    get_combine_summary,
    get_contract_table,
    get_contract_optimization,
    get_grosssplit,
    get_increment_summary,
    get_transition,
    get_detailed_summary,
    get_ltp_dict,
    get_rpd_dict,
    get_grosssplit_split,
    get_transition_split,
    get_economic_limit,
    get_asr_expenditures,
    get_lbt_expenditures,
    get_sensitivity,
    get_uncertainty,
)
from pyscnomics.api.converter import (
    Data,
    EconLimit,
    ASRExpendituresBM,
    LBTExpendituresBM,
    CasesRequestBM,
)
from pyscnomics.api.converter import DataTransition
from pyscnomics.api.converter import LtpBM, RpdBM

router = APIRouter(prefix="/api")


def _guard_contract(
    data: Data | DataTransition, contract_type: ContractType, operation: str
) -> None:
    """Validate a contract payload before any numerical work."""
    require_valid_input(
        validate_contract_input(
            data.model_dump(), contract_type, operation=operation
        )
    )


def _guard_cases(
    data: CasesRequestBM, operation: Literal["combine", "increment"]
) -> None:
    require_valid_input(validate_cases_input(data.model_dump(), operation))


def _validate_response(
    data: Data | DataTransition, contract_type: ContractType, request: Request
) -> JSONResponse:
    report = validate_contract_input(
        data.model_dump(), contract_type, operation="validate"
    )
    return validation_success_response(report, get_request_id(request))


@runtime_checkable
class _WarningSource(Protocol):
    warning_messages: object


@runtime_checkable
class _RecordsConvertible(Protocol):
    def to_dict(self, *, orient: Literal["records"]) -> list[dict[str, object]]: ...


_WARNING_RECORD_ADAPTER = TypeAdapter(dict[str, object], config=ConfigDict(strict=True))


@router.get("/")
async def read_root():
    """
    Route to get the current running PySCnomics version.
    """
    return {"Pyscnomics": "Version 1.4.0"}


@router.get("/health")
async def health_check():
    """Health-check endpoint."""
    return {"status": "ok"}


@router.get("/version")
async def api_version():
    """Return the installed pyscnomics version."""
    return {"version": __version__}


@router.post("/costrecovery")
async def calculate_costrecovery(data: Data) -> dict[str, object]:
    """
    ## Cost Recovery
    Route to calculate a contract using Cost Recovery Scheme and get its executive summary.

    ### Data Input Structure
    data:
    - setup
    - summary_arguments
    - costrecovery
    - contract_arguments
    - capital
    - tangible
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments
    """
    _guard_contract(data, "Cost Recovery", "costrecovery")
    result = get_costrecovery(data=data.model_dump())[0]
    return result


@router.post("/costrecovery/detailed_summary")
async def get_costrecovery_detailed(data: Data) -> dict[str, object]:
    """
    ## Cost Recovery Detailed Summary
    Route to get a contract detailed summary using Cost Recovery Scheme.

    ### Data Input Structure
    data:
    - setup
    - summary_arguments
    - costrecovery
    - contract_arguments
    - lifting
    - capital
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments
    """
    _guard_contract(data, "Cost Recovery", "costrecovery/detailed_summary")
    result = get_detailed_summary(data=data.model_dump(), contract_type="Cost Recovery")

    return result


@router.post("/costrecovery/table")
async def get_costrecovery_table(data: Data) -> dict[str, object]:
    """
    ## Cost Recovery Table
    Route to calculate a contract using Cost Recovery Scheme and get its cashflow table.

    ### Data Input Structure
    data:
    - setup
    - summary_arguments
    - costrecovery
    - contract_arguments
    - lifting
    - capital
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments
    """
    _guard_contract(data, "Cost Recovery", "costrecovery/table")
    return get_contract_table(data=data.model_dump(), contract_type="Cost Recovery")[0]


@router.post("/costrecovery/optimization")
async def calculate_costrecovery_optimization(data: Data) -> dict[str, object]:
    """
    ## Cost Recovery Optimization
    Route to calculate the optimization of a contract using Cost Recovery Scheme.

    ### Data Input Structure
    data:
    - setup
    - summary_arguments
    - costrecovery
    - contract_arguments
    - lifting
    - capital
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments
    """
    _guard_contract(data, "Cost Recovery", "costrecovery/optimization")
    return get_contract_optimization(
        data=data.model_dump(), contract_type="Cost Recovery"
    )


@router.post("/grosssplit")
async def calculate_grosssplit(data: Data) -> dict[str, object]:
    """
    ## Gross Split
    Route to calculate a contract using Gross Split Scheme and get its executive summary.

    ### Data Input Structure
    data:
    - setup
    - summary_arguments
    - grosssplit
    - contract_arguments
    - lifting
    - capital
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments
    """
    _guard_contract(data, "Gross Split", "grosssplit")
    result = get_grosssplit(data=data.model_dump())[0]
    return result


@router.post("/grosssplit/detailed_summary")
async def get_grosssplit_detailed(data: Data) -> dict[str, object]:
    """
    ## Gross Split Detailed Summary
    Route to get a contract detailed summary using Gross Split Scheme.

    ### Data Input Structure
    data:
    - setup
    - summary_arguments
    - costrecovery
    - contract_arguments
    - lifting
    - capital
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments
    """
    _guard_contract(data, "Gross Split", "grosssplit/detailed_summary")
    result = get_detailed_summary(data=data.model_dump(), contract_type="Gross Split")
    return result


@router.post("/grosssplit/table")
async def get_grosssplit_table(data: Data) -> dict[str, object]:
    """
    ## Gross Split Table
    Route to calculate a contract using Gross Split Scheme and get its cashflow table.

    ### Data Input Structure
    data:
    - setup
    - summary_arguments
    - grosssplit
    - contract_arguments
    - lifting
    - capital
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments
    """
    _guard_contract(data, "Gross Split", "grosssplit/table")
    return get_contract_table(data=data.model_dump(), contract_type="Gross Split")[0]


@router.post("/grosssplit/optimization")
async def calculate_grosssplit_optimization(data: Data) -> dict[str, object]:
    """
    ## Gross Split Optimization
    Route to calculate the optimization of a contract using Gross Split Scheme.

    ### Data Input Structure
    data:
    - setup
    - summary_arguments
    - grosssplit
    - contract_arguments
    - lifting
    - capital
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments
    """
    _guard_contract(data, "Gross Split", "grosssplit/optimization")
    return get_contract_optimization(
        data=data.model_dump(), contract_type="Gross Split"
    )


@router.post("/transition")
async def calculate_transition(data: DataTransition) -> dict[str, object]:
    """
    ## Transition
    Route to calculate a contract using Transition Scheme and get its executive summary.

    ### Data Input Structure
    data:
    - contract_1
        -- setup
        -- costrecovery or grosssplit
        -- contract_arguments
        -- lifting
        -- capital
        -- intangible
        -- opex
        -- asr
    - contract_2
        -- setup
        -- costrecovery or grosssplit
        -- contract_arguments
        -- lifting
        -- capital
        -- intangible
        -- opex
        -- asr
    - contract_arguments
    - summary_arguments
    """
    _guard_contract(data, "Transition", "transition")
    result = get_transition(data=data.model_dump())[0]

    return result


@router.post("/transition/detailed_summary")
async def get_transition_detailed(data: DataTransition) -> dict[str, object]:
    """
    ## Transition Detailed Summary
    Route to get a contract detailed summary using Transition Scheme.

    ### Data Input Structure
    data:
    - setup
    - summary_arguments
    - costrecovery
    - contract_arguments
    - lifting
    - capital
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments
    """
    _guard_contract(data, "Transition", "transition/detailed_summary")
    result = get_detailed_summary(data=data.model_dump(), contract_type="Transition")

    return result


@router.post("/transition/table")
async def get_transition_table(data: DataTransition) -> dict[str, object]:
    """
    ## Transition Table
    Route to calculate a contract using Transition Scheme and get its cashflow table.

    ### Data Input Structure
    data:
    - contract_1
        -- setup
        -- costrecovery or grosssplit
        -- contract_arguments
        -- lifting
        -- capital
        -- intangible
        -- opex
        -- asr
    - contract_2
        -- setup
        -- costrecovery or grosssplit
        -- contract_arguments
        -- lifting
        -- capital
        -- intangible
        -- opex
        -- asr
    - contract_arguments
    - summary_arguments
    """
    _guard_contract(data, "Transition", "transition/table")
    return get_contract_table(data=data.model_dump(), contract_type="Transition")[0]


@router.post("/transition/optimization")
async def calculate_transition_optimization(data: DataTransition) -> dict[str, object]:
    """
    ## Transition Optimization
    Route to calculate the optimization of a contract using Transition Scheme.

    ### Data Input Structure
    data:
    - setup
    - summary_arguments
    - grosssplit
    - contract_arguments
    - lifting
    - capital
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments
    """
    _guard_contract(data, "Transition", "transition/optimization")
    return get_contract_optimization(data=data.model_dump(), contract_type="Transition")


@router.post("/baseproject")
async def calculate_baseproject(data: Data) -> dict[str, object]:
    """
    ## Base Project
    Route to calculate a contract using Base Project Scheme and get its executive summary.

    ### Data Input Structure
    data:
    - setup
    - summary_arguments
    - contract_arguments
    - lifting
    - capital
    - intangible
    - opex
    - asr
    """
    _guard_contract(data, "Base Project", "baseproject")
    result = get_baseproject(data=data.model_dump())[0]
    return result


@router.post("/baseproject/table")
async def get_baseproject_table(data: Data) -> dict[str, object]:
    """
    ## Base Project Table
    Route to calculate a contract using Base Project Scheme and get its cashflow table.

    ### Data Input Structure
    data:
    - setup
    - summary_arguments
    - contract_arguments
    - lifting
    - capital
    - intangible
    - opex
    - asr
    """
    _guard_contract(data, "Base Project", "baseproject/table")
    return get_contract_table(data=data.model_dump(), contract_type="Base Project")[0]


@router.post("/baseproject/detailed_summary")
async def get_baseproject_detailed(data: Data) -> dict[str, object]:
    """
    ## Base Project Detailed Summary
    Route to get a contract detailed summary using Base Project Scheme.

    ### Data Input Structure
    data:
    - setup
    - summary_arguments
    - contract_arguments
    - lifting
    - capital
    - intangible
    - opex
    - asr
    """
    _guard_contract(data, "Base Project", "baseproject/detailed_summary")

    result = get_detailed_summary(data=data.model_dump(), contract_type="Base Project")

    return result


@router.post("/ltp")
async def calculate_ltp(data: LtpBM) -> dict[str, object]:
    """
    ## Calculate LTP model
    Route to calculate a ltp model.

    ### Data Input Structure
    volume: float | int
    start_year: int
    end_year: int
    fluid_type: str
    """
    require_valid_input(validate_operation_input(data.model_dump(), "ltp"))
    return get_ltp_dict(data=data.model_dump())


@router.post("/rpd")
async def calculate_rdp(data: RpdBM) -> dict[str, object]:
    """
    ## Calculate RPD model
    Route to calculate a rdp model.

    ### Data Input Structure
    year_rampup: int
    drate: float | int
    q_plateau_ratio: float | int
    q_min_ratio: float | int
    volume: float | int
    start_year: int
    end_year: int
    """
    require_valid_input(validate_operation_input(data.model_dump(), "rpd"))
    return get_rpd_dict(data=data.model_dump())


@router.post("/grosssplit/split")
async def get_grosssplit_split_information(data: Data) -> dict[str, object]:
    """
    ## The Split Information
    Route to retrieve the information of contractor split in Gross Split contract scheme.

    ### Data Input Structure
    data:
    - setup
    - summary_arguments
    - grosssplit
    - contract_arguments
    - lifting
    - capital
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments
    """
    _guard_contract(data, "Gross Split", "grosssplit/split")
    return get_grosssplit_split(data=data.model_dump())


@router.post("/transition/split")
async def get_transition_split_information(data: DataTransition) -> dict[str, object]:
    """
    ## The Split Information
    Route to retrieve the information of contractor split in Transition contract scheme
    if the transition contracts is consisting one or two gross split contracts.

    ### Data Input Structure
    data:
    - contract_1
        -- setup
        -- costrecovery or grosssplit
        -- contract_arguments
        -- lifting
        -- capital
        -- intangible
        -- opex
        -- asr
    - contract_2
        -- setup
        -- costrecovery or grosssplit
        -- contract_arguments
        -- lifting
        -- capital
        -- intangible
        -- opex
        -- asr
    - contract_arguments
    - summary_arguments
    """
    _guard_contract(data, "Transition", "transition/split")
    result = get_transition_split(data=data.model_dump())

    return result


@router.post("/econlimit")
async def calculate_economic_limit(data: EconLimit) -> int:
    """
    ## Retrieve The Economic Limit Years
    Route to get the years of the economic limit.

    ### Data Input Structure
    years: list[int]
    cash_flow: list[int] | list[float]
    method: str
    """
    require_valid_input(validate_operation_input(data.model_dump(), "econlimit"))
    return int(get_economic_limit(data=data.model_dump()))


@router.post("/asr_expenditures")
async def calculate_asr_expenditures(data: ASRExpendituresBM) -> dict[str, object]:
    """
    ## Retrieve The Expenditures of ASR Cost
    Route to get the expenditures of ASR Cost.
    """
    payload = data.model_dump()
    require_valid_input(validate_operation_input(payload, "asr_expenditures"))
    return get_asr_expenditures(data=normalize_expenditure_request(payload, "asr_expenditures"))


@router.post("/lbt_expenditures")
async def calculate_lbt_expenditures(data: LBTExpendituresBM) -> dict[str, object]:
    """
    ## Retrieve The Expenditures of LBT Cost
    Route to get the expenditures of LBT Cost.
    """
    payload = data.model_dump()
    require_valid_input(validate_operation_input(payload, "lbt_expenditures"))
    return get_lbt_expenditures(data=normalize_expenditure_request(payload, "lbt_expenditures"))


@router.post("/costrecovery/sensitivity")
async def calculate_costrecovery_sensitivity(data: Data) -> dict[str, object]:
    """
    ## Retrieve The Sensitivity of a cost recovery contract.
    Route to get the sensitivity of a cost recovery contract.
    """
    _guard_contract(data, "Cost Recovery", "costrecovery/sensitivity")
    return get_sensitivity(data=data.model_dump(), contract_type="Cost Recovery")


@router.post("/grosssplit/sensitivity")
async def calculate_grosssplit_sensitivity(data: Data) -> dict[str, object]:
    """
    ## Retrieve The Sensitivity of a gross split contract.
    Route to get the sensitivity of a contract.
    """
    _guard_contract(data, "Gross Split", "grosssplit/sensitivity")
    return get_sensitivity(data=data.model_dump(), contract_type="Gross Split")


@router.post("/transition/sensitivity")
async def calculate_transition_sensitivity(data: DataTransition) -> dict[str, object]:
    """
    ## Retrieve The Sensitivity of a gross split contract.
    Route to get the sensitivity of a contract.
    """
    _guard_contract(data, "Transition", "transition/sensitivity")
    return get_sensitivity(data=data.model_dump(), contract_type="Transition")


@router.post("/baseproject/sensitivity")
async def calculate_project_sensitivity(data: Data) -> dict[str, object]:
    """
    ## Retrieve The Sensitivity of a base project contract.
    Route to get the sensitivity of a contract.
    """
    _guard_contract(data, "Base Project", "baseproject/sensitivity")
    return get_sensitivity(data=data.model_dump(), contract_type="Base Project")


@router.post("/costrecovery/uncertainty")
async def calculate_costrecovery_uncertainty(data: Data) -> dict[str, object]:
    """
    ## Retrieve The Uncertainty of a cost recovery contract.
    Route to get the uncertainty of a cost recovery contract.
    """
    _guard_contract(data, "Cost Recovery", "costrecovery/uncertainty")
    return get_uncertainty(data=data.model_dump(), contract_type="Cost Recovery")


@router.post("/grosssplit/uncertainty")
async def calculate_grosssplit_uncertainty(data: Data) -> dict[str, object]:
    """
    ## Retrieve The Uncertainty of a gross split contract.
    Route to get the uncertainty of a gross split contract.
    """
    _guard_contract(data, "Gross Split", "grosssplit/uncertainty")
    return get_uncertainty(data=data.model_dump(), contract_type="Gross Split")


@router.post("/transition/uncertainty")
async def calculate_transition_uncertainty(data: DataTransition) -> dict[str, object]:
    """
    ## Retrieve The Uncertainty of a transition contract.
    Route to get the uncertainty of a transition contract.
    """
    _guard_contract(data, "Transition", "transition/uncertainty")
    return get_uncertainty(data=data.model_dump(), contract_type="Transition")


@router.post("/baseproject/uncertainty")
async def calculate_baseproject_uncertainty(data: Data) -> dict[str, object]:
    """
    ## Retrieve The Uncertainty of a base project contract.
    Route to get the uncertainty of a base projcet contract.
    """
    _guard_contract(data, "Base Project", "baseproject/uncertainty")
    return get_uncertainty(data=data.model_dump(), contract_type="Base Project")


def _extract_warnings(adapter_result: tuple[object, ...]) -> list[dict[str, object]]:
    """
    Extract warning messages from an adapter result tuple.

    The contract object is the second element of the adapter result tuple.
    Returns ``[]`` when the contract has no ``warning_messages``.
    """
    contract = adapter_result[1]
    if not isinstance(contract, _WarningSource):
        return []

    warning_messages = contract.warning_messages
    if _is_object_list(warning_messages):
        return [_warning_record(item) for item in warning_messages]
    if isinstance(warning_messages, _RecordsConvertible):
        return warning_messages.to_dict(orient="records")
    return []


def _is_string_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    """Return whether a value is a dictionary with string keys."""
    if not isinstance(value, dict):
        return False
    try:
        _ = _WARNING_RECORD_ADAPTER.validate_python(value)
    except ValidationError:
        return False
    return True


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    """Return whether a value is a list."""
    return isinstance(value, list)


def _warning_record(value: object) -> dict[str, object]:
    """Normalize one engine warning into the API response shape."""
    if _is_string_object_dict(value):
        return value
    if _is_object_tuple(value) and len(value) >= 2:
        return {"Message": value[1]}
    return {}


def _is_object_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    """Return whether a value is a tuple."""
    return isinstance(value, tuple)


@router.post("/baseproject/warnings")
async def get_baseproject_warnings(data: Data) -> list[dict[str, object]]:
    """
    ## Base Project Warnings
    Route to retrieve the warning messages of a Base Project contract.
    """
    _guard_contract(data, "Base Project", "baseproject/warnings")
    return _extract_warnings(get_baseproject(data=data.model_dump()))


@router.post("/costrecovery/warnings")
async def get_costrecovery_warnings(data: Data) -> list[dict[str, object]]:
    """
    ## Cost Recovery Warnings
    Route to retrieve the warning messages of a Cost Recovery contract.
    """
    _guard_contract(data, "Cost Recovery", "costrecovery/warnings")
    return _extract_warnings(get_costrecovery(data=data.model_dump()))


@router.post("/grosssplit/warnings")
async def get_grosssplit_warnings(data: Data) -> list[dict[str, object]]:
    """
    ## Gross Split Warnings
    Route to retrieve the warning messages of a Gross Split contract.
    """
    _guard_contract(data, "Gross Split", "grosssplit/warnings")
    return _extract_warnings(get_grosssplit(data=data.model_dump()))


@router.post("/transition/warnings")
async def get_transition_warnings(data: DataTransition) -> list[dict[str, object]]:
    """
    ## Transition Warnings
    Route to retrieve the warning messages of a Transition contract.
    """
    _guard_contract(data, "Transition", "transition/warnings")
    return _extract_warnings(get_transition(data=data.model_dump()))


@router.post("/combine")
async def calculate_combine(data: CasesRequestBM) -> dict[str, object]:
    """
    ## Combine Cases
    Route to combine multiple contract cases into one economic result.

    ### Data Input Structure
    data:
    - cases: list[{cId, ctr, data}]
    - options: {reference_year, inflation_rate, discount_rate, npv_mode,
      discounting_mode, profitability_discounted}
    """
    _guard_cases(data, "combine")
    return get_combine_summary(payload=data.model_dump())


@router.post("/increment")
async def calculate_increment(data: CasesRequestBM) -> dict[str, object]:
    """
    ## Incremental Cases
    Route to calculate the incremental result between contract cases.

    ### Data Input Structure
    data:
    - cases: list[{cId, ctr, data}]
    - options: {reference_year, inflation_rate, discount_rate, npv_mode,
      discounting_mode, profitability_discounted}
    """
    _guard_cases(data, "increment")
    return get_increment_summary(payload=data.model_dump())


# ---------------------------------------------------------------------------
# Validation-only endpoints. These return the shared envelope with
# ``can_calculate`` and never run numerical or analysis work.
# ---------------------------------------------------------------------------


@router.post("/baseproject/validate")
async def validate_baseproject(data: Data, request: Request) -> JSONResponse:
    """Validate a Base Project payload without calculating."""
    return _validate_response(data, "Base Project", request)


@router.post("/costrecovery/validate")
async def validate_costrecovery(data: Data, request: Request) -> JSONResponse:
    """Validate a Cost Recovery payload without calculating."""
    return _validate_response(data, "Cost Recovery", request)


@router.post("/grosssplit/validate")
async def validate_grosssplit(data: Data, request: Request) -> JSONResponse:
    """Validate a Gross Split payload without calculating."""
    return _validate_response(data, "Gross Split", request)


@router.post("/transition/validate")
async def validate_transition(data: DataTransition, request: Request) -> JSONResponse:
    """Validate a Transition payload without calculating."""
    return _validate_response(data, "Transition", request)


@router.post("/combine/validate")
async def validate_combine(data: CasesRequestBM, request: Request) -> JSONResponse:
    """Validate every case in a combine payload without calculating."""
    report = validate_cases_input(data.model_dump(), "validate")
    return validation_success_response(report, get_request_id(request))


@router.post("/increment/validate")
async def validate_increment(data: CasesRequestBM, request: Request) -> JSONResponse:
    """Validate every case in an incremental payload without calculating."""
    report = validate_cases_input(data.model_dump(), "validate")
    return validation_success_response(report, get_request_id(request))
