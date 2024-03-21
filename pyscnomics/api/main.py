"""
This file is utilized for routing the API.
"""
from fastapi import FastAPI
import json
from pyscnomics.api.adapter import get_costrecovery, get_contract_table, get_contract_optimization

description = """
## Description
We welcome you to our economic toolkit. PySCnomics 🚀.

This package contains tailored functionalities for assessing economic feasibility of oil and gas projects following the state-of-the-art Production Sharing Contract (PSC) schemes in Indonesia.
PySCnomics is the product of join research between Indonesia's Special Task Force for Upstream Oil and Gas Business Activities (SKK Migas) and the Department of Petroleum Engineering, Institut Teknologi Bandung (ITB).

"""

app = FastAPI(
    title="PySCnomics",
    description=description,
    version="1.0.0",)


@app.get("/")
async def read_root():
    return {"Pyscnomics": "Version 1.0.0"}


@app.post("/costrecovery")
async def calculate_costrecovery(data: dict) -> dict:
    return get_costrecovery(data=data)[0]


@app.post("/costrecovery/table")
async def get_costrecovery_table(data: dict) -> dict:
    return get_contract_table(data=data)


@app.post("/costrecovery/optimization")
async def calculate_costrecovery_optimization(data: dict) -> dict:
    return get_contract_optimization(data=data)


@app.post("/grosssplit")
async def calculate_grosssplit():

    return {" "}


@app.post("/grosssplit/optimization")
async def calculate_grosssplit_optimization():

    return {" "}


@app.post("/tools/table")
async def get_table():

    return {" "}



