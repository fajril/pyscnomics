from fastapi import APIRouter
from pyscnomics.api.adapter import (get_baseproject,
                                    get_costrecovery,
                                    get_contract_table,
                                    get_contract_optimization,
                                    get_grosssplit,
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
                                    get_uncertainty)
from pyscnomics.api.converter import Data, EconLimit, ASRExpendituresBM, LBTExpendituresBM
from pyscnomics.api.converter import DataTransition
from pyscnomics.api.converter import LtpBM, RpdBM


router = APIRouter(prefix='/api')


@router.get("/")
async def read_root():
    """
    Route to get the current running PySCnomics version.
    """
    return {"Pyscnomics": "Version 1.0.0"}


@router.post("/costrecovery")
async def calculate_costrecovery(data: Data) -> dict:
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
    result = get_costrecovery(data=data.model_dump())[0]
    return result


@router.post("/costrecovery/detailed_summary")
async def get_costrecovery_detailed(data: Data) -> dict:
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
    result = get_detailed_summary(
        data=data.model_dump(),
        contract_type='Cost Recovery')
    return result


@router.post("/costrecovery/table")
async def get_costrecovery_table(data: Data) -> dict:
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
    return get_contract_table(data=data.model_dump(), contract_type='Cost Recovery')


@router.post("/costrecovery/optimization")
async def calculate_costrecovery_optimization(data: Data) -> dict:
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
    return get_contract_optimization(data=data.model_dump(), contract_type='Cost Recovery')


@router.post("/grosssplit")
async def calculate_grosssplit(data: Data) -> dict:
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
    result = get_grosssplit(data=data.model_dump())[0]
    return result


@router.post("/grosssplit/detailed_summary")
async def get_grosssplit_detailed(data: Data) -> dict:
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
    result = get_detailed_summary(
        data=data.model_dump(),
        contract_type='Gross Split')
    return result


@router.post("/grosssplit/table")
async def get_grosssplit_table(data: Data) -> dict:
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
    return get_contract_table(data=data.model_dump(), contract_type='Gross Split')


@router.post("/grosssplit/optimization")
async def calculate_grosssplit_optimization(data: Data) -> dict:
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
    return get_contract_optimization(data=data.model_dump(), contract_type='Gross Split')


@router.post("/transition")
async def calculate_transition(data: DataTransition) -> dict:
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
    result = get_transition(data=data.model_dump())[0]
    return result


@router.post("/transition/detailed_summary")
async def get_transition_detailed(data: DataTransition) -> dict:
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
    result = get_detailed_summary(
        data=data.model_dump(),
        contract_type='Transition')
    return result


@router.post("/transition/table")
async def get_transition_table(data: DataTransition) -> dict:
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
    return get_contract_table(data=data.model_dump(), contract_type='Transition')


@router.post("/transition/optimization")
async def calculate_transition_optimization(data: DataTransition) -> dict:
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
    return get_contract_optimization(data=data.model_dump(), contract_type='Transition')


@router.post("/baseproject")
async def calculate_baseproject(data: Data) -> dict:
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
    result = get_baseproject(data=data.model_dump())[0]
    return result


@router.post("/baseproject/table")
async def get_baseproject_table(data: Data) -> dict:
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
    return get_contract_table(data=data.model_dump(), contract_type='Base Project')


@router.post("/baseproject/detailed_summary")
async def get_baseproject_detailed(data: Data) -> dict:
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
    result = get_detailed_summary(
        data=data.model_dump(),
        contract_type='Base Project')
    return result


@router.post("/ltp")
async def calculate_ltp(data: LtpBM) -> dict:
    """
    ## Calculate LTP model
    Route to calculate a ltp model.

    ### Data Input Structure
    volume: float | int
    start_year: int
    end_year: int
    fluid_type: str

    """
    return get_ltp_dict(data=data.model_dump())


@router.post("/rpd")
async def calculate_rdp(data: RpdBM) -> dict:
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
    return get_rpd_dict(data=data.model_dump())


@router.post("/grosssplit/split")
async def get_grosssplit_split_information(data: Data) -> dict:
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
    return get_grosssplit_split(data=data.model_dump())


@router.post("/transition/split")
async def get_transition_split_information(data: DataTransition) -> dict:
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
    return get_economic_limit(data=data.model_dump())


@router.post("/asr_expenditures")
async def calculate_asr_expenditures(data: ASRExpendituresBM) -> dict:
    """
    ## Retrieve The Expenditures of ASR Cost
    Route to get the expenditures of ASR Cost.
    """
    return get_asr_expenditures(data=data.model_dump())


@router.post("/lbt_expenditures")
async def calculate_lbt_expenditures(data: LBTExpendituresBM) -> dict:
    """
    ## Retrieve The Expenditures of LBT Cost
    Route to get the expenditures of LBT Cost.
    """
    return get_lbt_expenditures(data=data.model_dump())


@router.post("/costrecovery/sensitivity")
async def calculate_costrecovery_sensitivity(data: Data) -> dict:
    """
    ## Retrieve The Sensitivity of a cost recovery contract.
    Route to get the sensitivity of a cost recovery contract.
    """
    return get_sensitivity(
        data=data.model_dump(),
        contract_type='Cost Recovery')

@router.post("/grosssplit/sensitivity")
async def calculate_grosssplit_sensitivity(data: Data) -> dict:
    """
    ## Retrieve The Sensitivity of a gross split contract.
    Route to get the sensitivity of a contract.
    """
    return get_sensitivity(
        data=data.model_dump(),
        contract_type='Gross Split')

@router.post("/transition/sensitivity")
async def calculate_transition_sensitivity(data: DataTransition) -> dict:
    """
    ## Retrieve The Sensitivity of a gross split contract.
    Route to get the sensitivity of a contract.
    """
    return get_sensitivity(
        data=data.model_dump(),
        contract_type='Transition')


@router.post("/costrecovery/uncertainty")
async def calculate_costrecovery_uncertainty(data: Data) -> dict:
    """
    ## Retrieve The Uncertainty of a cost recovery contract.
    Route to get the uncertainty of a cost recovery contract.
    """
    return get_uncertainty(
        data=data.model_dump(),
        contract_type='Cost Recovery')

