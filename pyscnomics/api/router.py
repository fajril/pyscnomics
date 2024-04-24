from fastapi import APIRouter
from pyscnomics.api.adapter import (get_costrecovery,
                                    get_contract_table,
                                    get_contract_optimization,
                                    get_grosssplit,
                                    get_transition)
from pyscnomics.api.converter import Data
from pyscnomics.api.converter import DataTransition


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


@router.post("/transition/table")
async def get_transition_table(data: DataTransition) -> dict:
    """
    ## Gross Split Table
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

