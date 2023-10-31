"""
A module to execute preliminary evaluations
"""

import numpy as np
from datetime import datetime
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit

from pyscnomics.contracts.transition import transition

# Defining Start Date and End Date
psc_1_start_date = datetime.strptime("01/01/2010", '%d/%m/%Y').date()
psc_1_end_date = datetime.strptime("20/6/2019", '%d/%m/%Y').date()
psc_1_oil_onstream_date = datetime.strptime("01/01/2010", '%d/%m/%Y').date()

psc_2_start_date = datetime.strptime("21/6/2019", '%d/%m/%Y').date()
psc_2_end_date = datetime.strptime("31/12/2028", '%d/%m/%Y').date()
psc_2_oil_onstream_date = datetime.strptime("21/6/2019", '%d/%m/%Y').date()


# Defining the Oil lifting data
psc1_oil_lifting = Lifting(
    start_year=2010,
    end_year=2019,
    lifting_rate=np.repeat(100, 10),
    price=np.repeat(60, 10),
    prod_year=np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]),
    fluid_type=FluidType.OIL)

psc2_oil_lifting = Lifting(
    start_year=2019,
    end_year=2028,
    lifting_rate=np.repeat(50, 10),
    price=np.repeat(60, 10),
    prod_year=np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028]),
    fluid_type=FluidType.OIL)


# Defining the Oil Tangible Data - Drilling Tangible
psc1_oil_tangible = Tangible(
    start_year=2010,
    end_year=2019,
    cost=np.array([200]),
    expense_year=np.array([2010]),
    pis_year=np.array([2010]),
    useful_life=np.array([5]),
    depreciation_factor=np.array([0.25]),
    cost_allocation=FluidType.OIL)

psc2_oil_tangible = Tangible(
    start_year=2019,
    end_year=2028,
    cost=np.array([200]),
    expense_year=np.array([2020]),
    pis_year=np.array([2020]),
    useful_life=np.array([5]),
    depreciation_factor=np.array([0.25]),
    cost_allocation=FluidType.OIL)


# Defining the Oil Intangible Data
psc1_oil_intang = Intangible(
    start_year=2010,
    end_year=2019,
    cost=np.array([50]),
    expense_year=np.array([2010]),
    cost_allocation=FluidType.OIL)

psc2_oil_intang = Intangible(
    start_year=2019,
    end_year=2028,
    cost=np.array([50]),
    expense_year=np.array([2020]),
    cost_allocation=FluidType.OIL)

# Defining the Oil OPEX Data
psc1_oil_opex_cost = OPEX(
    start_year=2010,
    end_year=2019,
    fixed_cost=np.repeat(5, 10),
    expense_year=np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]),
    cost_allocation=FluidType.OIL)

psc2_oil_opex_cost = OPEX(
    start_year=2019,
    end_year=2028,
    fixed_cost=np.repeat(5, 10),
    expense_year=np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028]),
    cost_allocation=FluidType.OIL)


# Defining the Oil ASR Data
psc1_oil_asr_cost = ASR(
    start_year=2010,
    end_year=2019,
    cost=np.array([0]),
    expense_year=np.array([2010]),
    cost_allocation=FluidType.OIL)

psc2_oil_asr_cost = ASR(
    start_year=2019,
    end_year=2028,
    cost=np.array([0]),
    expense_year=np.array([2020]),
    cost_allocation=FluidType.OIL)


psc1 = CostRecovery(
    start_date=psc_1_start_date,
    end_date=psc_1_end_date,
    oil_onstream_date=psc_1_oil_onstream_date,
    lifting=tuple([psc1_oil_lifting]),
    tangible_cost=tuple([psc1_oil_tangible]),
    intangible_cost=tuple([psc1_oil_intang]),
    opex=tuple([psc1_oil_opex_cost]),
    asr_cost=tuple([psc1_oil_asr_cost]))


psc2 = CostRecovery(
    start_date=psc_2_start_date,
    end_date=psc_2_end_date,
    oil_onstream_date=psc_2_oil_onstream_date,
    lifting=tuple([psc2_oil_lifting]),
    tangible_cost=tuple([psc2_oil_tangible]),
    intangible_cost=tuple([psc2_oil_intang]),
    opex=tuple([psc2_oil_opex_cost]),
    asr_cost=tuple([psc2_oil_asr_cost]))


if __name__ == "__main__":
    psc_trans = transition(contract1=psc1, contract2=psc2)



