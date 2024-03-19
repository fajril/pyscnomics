"""
This file is utilized for routing the API.
"""
from fastapi import FastAPI

from pyscnomics.api.adapter import get_contract_costrecovery


app = FastAPI()


@app.get("/")
async def read_root():
    return {"Pyscnomics": "Version 1.0.0"}


@app.post("/baseproject")
async def calculate_baseproject():

    return {" "}


@app.post("/costrecovery")
async def calculate_costrecovery(data: dict) -> dict:
    result = get_contract_costrecovery(data=data)

    return result


@app.post("/costrecovery/sensitivity")
async def calculate_costrecovery_sensitivity():

    return {" "}


@app.post("/costrecovery/optimization")
async def calculate_costrecovery_optimization():

    return {" "}


@app.post("/costrecovery/uncertainty")
async def calculate_costrecovery_uncertainty():

    return {" "}


@app.post("/grosssplit")
async def calculate_grosssplit():

    return {" "}


@app.post("/grosssplit/sensitivity")
async def calculate_grosssplit_sensitivity():

    return {" "}


@app.post("/grosssplit/optimization")
async def calculate_grosssplit_optimization():

    return {" "}


@app.post("/grosssplit/uncertainty")
async def calculate_grosssplit_uncertainty():

    return {" "}


@app.post("/tools/table")
async def get_table():

    return {" "}


@app.post("/tools/")
async def get_table():

    return {" "}







