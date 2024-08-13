from fastapi import APIRouter
from pyscnomics.api.adapter import (get_baseproject,
                                    get_costrecovery,
                                    get_contract_table,
                                    get_contract_optimization,
                                    get_grosssplit,
                                    get_transition,
                                    get_detailed_summary,
                                    get_ltp_dict,
                                    get_rdp_dict)
from pyscnomics.api.converter import Data
from pyscnomics.api.converter import DataTransition
from pyscnomics.api.converter import LtpBM, RdpBM


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
    - lifting
    - tangible
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments

    """
    return get_costrecovery(data=data.dict())[0]


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
    - tangible
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments

    """

    return get_detailed_summary(data=data.dict(),
                                contract_type='Cost Recovery')


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
    - tangible
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments

    """
    return get_contract_table(data=data.dict(), contract_type='Cost Recovery')


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
    - tangible
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments

    """
    return get_contract_optimization(data=data.dict(), contract_type='Cost Recovery')


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
    - tangible
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments

    """
    return get_grosssplit(data=data.dict())[0]


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
    - tangible
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments

    """

    return get_detailed_summary(data=data.dict(),
                                contract_type='Gross Split')


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
    - tangible
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments

    """
    return get_contract_table(data=data.dict(), contract_type='Gross Split')


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
    - tangible
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments

    """
    return get_contract_optimization(data=data.dict(), contract_type='Gross Split')


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
        -- tangible
        -- intangible
        -- opex
        -- asr
    - contract_2
        -- setup
        -- costrecovery or grosssplit
        -- contract_arguments
        -- lifting
        -- tangible
        -- intangible
        -- opex
        -- asr
    - contract_arguments
    - summary_arguments


    """
    return get_transition(data=data.dict())[0]


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
    - tangible
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments

    """

    return get_detailed_summary(data=data.dict(),
                                contract_type='Transition')


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
        -- tangible
        -- intangible
        -- opex
        -- asr
    - contract_2
        -- setup
        -- costrecovery or grosssplit
        -- contract_arguments
        -- lifting
        -- tangible
        -- intangible
        -- opex
        -- asr
    - contract_arguments
    - summary_arguments

    """
    return get_contract_table(data=data.dict(), contract_type='Transition')


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
    - tangible
    - intangible
    - opex
    - asr
    - optimization_arguments
    - sensitivity_arguments

    """
    return get_contract_optimization(data=data.dict(), contract_type='Transition')


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
    - tangible
    - intangible
    - opex
    - asr

    """
    return get_baseproject(data=data.dict())[0]


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
    - tangible
    - intangible
    - opex
    - asr

    """
    return get_contract_table(data=data.dict(), contract_type='Base Project')


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
    - tangible
    - intangible
    - opex
    - asr

    """

    return get_detailed_summary(data=data.dict(),
                                contract_type='Base Project')


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
    return get_ltp_dict(data=data.dict())


@router.post("/rdp")
async def calculate_rdp(data: RdpBM) -> dict:
    """
    ## Calculate RDP model
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
    return get_rdp_dict(data=data.dict())

