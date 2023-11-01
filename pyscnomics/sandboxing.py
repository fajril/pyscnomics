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
psc_1_start_date = datetime.strptime("01/01/2019", '%d/%m/%Y').date()
psc_1_end_date = datetime.strptime("22/4/2020", '%d/%m/%Y').date()

psc_2_start_date = datetime.strptime("23/4/2020", '%d/%m/%Y').date()
psc_2_end_date = datetime.strptime("31/12/2027", '%d/%m/%Y').date()

# Defining the Gas lifting data
psc1_gas_lifting = Lifting(
    start_year=2019,
    end_year=2020,
    lifting_rate=np.array([4.3410078435945200, 2.7038619250834900]),
    price=np.array([6.620, 6.620]),
    prod_year=np.array([2019, 2020]),
    fluid_type=FluidType.GAS)

psc2_gas_lifting = Lifting(
    start_year=,
    end_year=,
    lifting_rate=,
    price=,
    prod_year=,
    fluid_type=FluidType.GAS)

# Defining the Gas Tangible Data - Drilling Tangible
psc1_gas_tangible = Tangible(
    start_year=2019,
    end_year=2020,
    cost=np.array([3.36367743703704000, 0.80273224043715800]),
    expense_year=np.array([2019, 2020]),
    pis_year=np.array([2019, 2020]),
    useful_life=np.array([5, 5]),
    depreciation_factor=np.array([0.25, 0.25]),
    cost_allocation=FluidType.GAS)

# psc2_gas_tangible = Tangible(
#     start_year=,
#     end_year=,
#     cost=np.array([]),
#     expense_year=np.array([]),
#     pis_year=np.array([]),
#     useful_life=np.array([]),
#     depreciation_factor=np.array([]),
#     cost_allocation=FluidType.GAS)


# Defining the Gas Intangible Data
psc1_gas_intang = Intangible(
    start_year=2019,
    end_year=2020,
    cost=np.array([ 9.532633600000]),
    expense_year=np.array([2019]),
    cost_allocation=FluidType.GAS)

# psc2_gas_intang = Intangible(
#     start_year=,
#     end_year=,
#     cost=np.array([]),
#     expense_year=np.array([]),
#     cost_allocation=FluidType.GAS)

# Defining the Gas OPEX Data
psc1_gas_opex_cost = OPEX(
    start_year=2019,
    end_year=2020,
    fixed_cost=np.array([2.076908222642980, 1.297582047244550]),
    expense_year=np.array([2019, 2020]),
    cost_allocation=FluidType.GAS)

# psc2_gas_opex_cost = OPEX(
#     start_year=,
#     end_year=,
#     fixed_cost=np.array([]),
#     expense_year=np.array([]),
#     cost_allocation=FluidType.GAS)

# Defining the Gas ASR Data
psc1_gas_asr_cost_opx = OPEX(
    start_year=2019,
    end_year=2020,
    fixed_cost=np.array([0.035515809523809900, 0.010965263596148900]),
    expense_year=np.array([2019, 2020]),
    cost_allocation=FluidType.GAS)

# psc2_gas_asr_cost_opx = OPEX(
#     start_year=,
#     end_year=,
#     fixed_cost=np.array([]),
#     expense_year=np.array([]),
#     cost_allocation=FluidType.GAS)

psc1_gas_asr_cost = ASR(
    start_year=2019,
    end_year=2020,
    cost=np.array([0]),
    expense_year=np.array([2019]),
    cost_allocation=FluidType.GAS)

# psc2_gas_asr_cost = ASR(
#     start_year=,
#     end_year=,
#     cost=np.array([0]),
#     expense_year=np.array([]),
#     cost_allocation=FluidType.GAS)

psc1 = CostRecovery(
    start_date=psc_1_start_date,
    end_date=psc_1_end_date,
    lifting=tuple([psc1_gas_lifting]),
    tangible_cost=tuple([psc1_gas_tangible]),
    intangible_cost=tuple([psc1_gas_intang]),
    opex=tuple([psc1_gas_opex_cost]),
    asr_cost=tuple([psc1_gas_asr_cost]),
    oil_ftp_is_available=True,
    oil_ftp_is_shared=True,
    oil_ftp_portion=0.2,
    gas_ftp_is_available=True,
    gas_ftp_is_shared=True,
    gas_ftp_portion=0.2,
    oil_ctr_pretax_share=0.3361,
    gas_ctr_pretax_share=0.5042,
    oil_dmo_volume_portion=0.25,
    oil_dmo_fee_portion=0.15,
    oil_dmo_holiday_duration=0,
    gas_dmo_volume_portion=0.25,
    gas_dmo_fee_portion=0.15,
    gas_dmo_holiday_duration=0)

# psc2 = GrossSplit(
#     start_date=psc_2_start_date,
#     end_date=psc_2_end_date,
#     oil_onstream_date=psc_2_oil_onstream_date,
#     lifting=tuple([psc2_gas_lifting]),
#     tangible_cost=tuple([psc2_gas_tangible]),
#     intangible_cost=tuple([psc2_gas_intang]),
#     opex=tuple([psc2_gas_opex_cost]),
#     asr_cost=tuple([psc2_gas_asr_cost]),
#     field_status=,
#     field_loc=,
#     res_depth=,
#     infra_avail=,
#     res_type=,
#     api_oil=,
#     domestic_use=,
#     prod_stage=,
#     co2_content=,
#     h2s_content=,
#     ctr_effective_tax_rate=,
#     base_split_ctr_oil=,
#     base_split_ctr_gas=,
#     split_ministry_disc=,
#     oil_dmo_volume_portion=,
#     oil_dmo_fee_portion=,
#     oil_dmo_holiday_duration=,
#     gas_dmo_volume_portion=,
#     gas_dmo_fee_portion=,
#     gas_dmo_holiday_duration=,)


if __name__ == "__main__":
    psc1.run()
    # psc_trans = transition(contract1=psc1, contract2=psc2)


