"""
This file is utilized for routing the API.
"""
from fastapi import FastAPI, APIRouter
from pyscnomics.api.adapter import (get_costrecovery,
                                    get_contract_table,
                                    get_contract_optimization,
                                    get_grosssplit,
                                    get_transition)
from pyscnomics.api.converter import Data
from pyscnomics.api.router import router


description = """
## Description
We welcome you to our library, PySCnomics 🚀.

This package contains tailored functionalities for assessing economic feasibility of oil and gas projects following the state-of-the-art Production Sharing Contract (PSC) schemes in Indonesia.
PySCnomics is the product of join research between Indonesia's Special Task Force for Upstream Oil and Gas Business Activities (SKK Migas) and the Department of Petroleum Engineering, Institut Teknologi Bandung (ITB).

"""

app = FastAPI(
    title="PySCnomics",
    description=description,
    version="1.0.0")


app.include_router(router)


