from fastapi import APIRouter
from pyscnomics.api.adapter import (get_costrecovery,
                                    get_contract_table,
                                    get_contract_optimization,
                                    get_grosssplit,
                                    get_transition)
from pyscnomics.api.converter import Data


router = APIRouter(prefix='/api')


@router.get("/")
async def read_root():
    """
    Route to get the current running PySCnomics version.
    """
    return {"Pyscnomics": "Version 1.0.0"}


@router.post("/costrecovery")
async def calculate_costrecovery(data: Data) -> dict:
    return get_costrecovery(data=data.dict())[0]


@router.post("/costrecovery/table")
async def get_costrecovery_table(data: Data) -> dict:
    return get_contract_table(data=data.dict(), contract_type='Cost Recovery')


@router.post("/costrecovery/optimization")
async def calculate_costrecovery_optimization(data: Data) -> dict:
    return get_contract_optimization(data=data.dict(), contract_type='Cost Recovery')


@router.post("/grosssplit")
async def calculate_grosssplit(data: Data) -> dict:
    return get_grosssplit(data=data.dict())[0]


@router.post("/grosssplit/table")
async def get_grosssplit_table(data: Data) -> dict:
    return get_contract_table(data=data.dict(), contract_type='Gross Split')


@router.post("/grosssplit/optimization")
async def calculate_grosssplit_optimization(data: Data) -> dict:
    return get_contract_optimization(data=data.dict(), contract_type='Gross Split')


@router.post("/transition")
async def calculate_transition(data: Data) -> dict:
    return get_transition(data=data.dict())[0]


@router.post("/transition/table")
async def get_transition_table(data: Data) -> dict:
    return get_contract_table(data=data, contract_type='Transition')

