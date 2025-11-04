"""
A collection of required data for CASE 1: DUMMY
"""

import numpy as np
from datetime import date
from dataclasses import dataclass, field

from pyscnomics.econ.selection import FluidType, CostType, PoolData
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import (
    CapitalCost,
    Intangible,
    OPEX,
    ASR,
    LBT,
    CostOfSales,
)


def get_kwargs(contract_type: str):

    # Base project
    kwargs_base_project = {
        # Base parameters
        "start_date": date(year=2023, month=1, day=1),
        "end_date": date(year=2032, month=12, day=31),
        "oil_onstream_date": date(year=2030, month=1, day=1),
        "gas_onstream_date": date(year=2029, month=1, day=1),
        "approval_year": 2026,
        "is_pod_1": False,
    }

    # Cost recovery
    kwargs_cost_recovery = {

    }

    # Gross split
    kwargs_gross_split = {
        None
    }

    kwargs_contract = {
        "base_project": None,
        "cost_recovery": None,
        "gross_split": None,
    }

    try:
        return kwargs_contract[contract_type]

    except KeyError:
        raise ValueError(f"Unrecognized contract type: {contract_type!r}")



case1_base_project = {
    "setup": {
        "start_date": date(year=2023, month=1, day=1),
        "end_date": date(year=2032, month=12, day=31),
        "oil_onstream_date": date(year=2030, month=1, day=1),
        "gas_onstream_date": date(year=2029, month=1, day=1),
        "approval_year": 2026,
        "is_pod_1": False,
    },
    "lifting": {
        "oil": Lifting(
            start_year=2023,
            end_year=2032,
            prod_year=np.array([2030, 2031, 2032]),
            lifting_rate=np.array([100, 100, 100]),
            price=np.array([120, 120, 120]),
            fluid_type=FluidType.OIL,
        ),
        "gas": Lifting(
            start_year=2023,
            end_year=2032,
            prod_year=np.array([2029, 2030, 2031]),
            lifting_rate=np.array([10, 10, 10]),
            price=np.array([1, 1, 1]),
            fluid_type=FluidType.GAS,
        )
    }
}