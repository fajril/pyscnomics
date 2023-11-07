import numpy as np
from datetime import date
import calendar
import itertools

from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.econ.selection import FluidType
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts import psc_tools


# Defining the gap between the prior contract and the new contract

# Get the proportion of the days of contract transition to a year

# Modifying the Lifting attributes into the current updates project length, fill it with zeros

# Modifying the Tangible, Intangible, OPEX, ASR attributes into the current updates project length, fill it with zeros

# Modify the latest row of Lifting, Tangible, Intangible, OPEX, ASR into the proportioned value

# Run each PSC

# Sum the generated cashflow


def parse_dataclass(contract):
    # Getting the attributes of the contract
    attr_contract = vars(contract)

    # Condition where only one fluid is produced
    produced_fluid = [i.fluid_type for i in attr_contract['lifting']]

    if FluidType.OIL not in produced_fluid:
        attr_contract['oil_onstream_date'] = None
    if FluidType.GAS not in produced_fluid:
        attr_contract['gas_onstream_date'] = None

    if isinstance(contract, CostRecovery):
        new_contract = CostRecovery(
            start_date=attr_contract['start_date'],
            end_date=attr_contract['end_date'],
            oil_onstream_date=attr_contract['oil_onstream_date'],
            gas_onstream_date=attr_contract['gas_onstream_date'],
            lifting=attr_contract['lifting'],
            tangible_cost=attr_contract['tangible_cost'],
            intangible_cost=attr_contract['intangible_cost'],
            opex=attr_contract['opex'],
            asr_cost=attr_contract['asr_cost'],
            oil_ftp_is_available=attr_contract['oil_ftp_is_available'],
            oil_ftp_is_shared=attr_contract['oil_ftp_is_shared'],
            oil_ftp_portion=attr_contract['oil_ftp_portion'],
            gas_ftp_is_available=attr_contract['gas_ftp_is_available'],
            gas_ftp_is_shared=attr_contract['gas_ftp_is_shared'],
            gas_ftp_portion=attr_contract['gas_ftp_portion'],
            tax_split_type=attr_contract['tax_split_type'],
            condition_dict=attr_contract['condition_dict'],
            indicator_rc_icp_sliding=attr_contract['indicator_rc_icp_sliding'],
            oil_ctr_pretax_share=attr_contract['oil_ctr_pretax_share'],
            gas_ctr_pretax_share=attr_contract['gas_ctr_pretax_share'],
            oil_ic_rate=attr_contract['oil_ic_rate'],
            gas_ic_rate=attr_contract['gas_ic_rate'],
            ic_is_available=attr_contract['ic_is_available'],
            oil_cr_cap_rate=attr_contract['oil_cr_cap_rate'],
            gas_cr_cap_rate=attr_contract['gas_cr_cap_rate'],
            oil_dmo_volume_portion=attr_contract['oil_dmo_volume_portion'],
            oil_dmo_fee_portion=attr_contract['oil_dmo_fee_portion'],
            oil_dmo_holiday_duration=attr_contract['oil_dmo_holiday_duration'],
            gas_dmo_volume_portion=attr_contract['gas_dmo_volume_portion'],
            gas_dmo_fee_portion=attr_contract['gas_dmo_fee_portion'],
            gas_dmo_holiday_duration=attr_contract['gas_dmo_holiday_duration'])
    else:
        new_contract = GrossSplit(
            start_date=attr_contract['start_date'],
            end_date=attr_contract['end_date'],
            oil_onstream_date=attr_contract['oil_onstream_date'],
            gas_onstream_date=attr_contract['gas_onstream_date'],
            lifting=attr_contract['lifting'],
            tangible_cost=attr_contract['tangible_cost'],
            intangible_cost=attr_contract['intangible_cost'],
            opex=attr_contract['opex'],
            asr_cost=attr_contract['asr_cost'],
            field_status=attr_contract['field_status'],
            field_loc=attr_contract['field_loc'],
            res_depth=attr_contract['res_depth'],
            infra_avail=attr_contract['infra_avail'],
            res_type=attr_contract['res_type'],
            api_oil=attr_contract['api_oil'],
            domestic_use=attr_contract['domestic_use'],
            prod_stage=attr_contract['prod_stage'],
            co2_content=attr_contract['co2_content'],
            h2s_content=attr_contract['h2s_content'],
            base_split_ctr_oil=attr_contract['base_split_ctr_oil'],
            base_split_ctr_gas=attr_contract['base_split_ctr_gas'],
            split_ministry_disc=attr_contract['split_ministry_disc'],
            oil_dmo_volume_portion=attr_contract['oil_dmo_volume_portion'],
            oil_dmo_fee_portion=attr_contract['oil_dmo_fee_portion'],
            oil_dmo_holiday_duration=attr_contract['oil_dmo_holiday_duration'],
            gas_dmo_volume_portion=attr_contract['gas_dmo_volume_portion'],
            gas_dmo_fee_portion=attr_contract['gas_dmo_fee_portion'],
            gas_dmo_holiday_duration=attr_contract['gas_dmo_holiday_duration'],
            conversion_bboe2bscf=attr_contract['conversion_bboe2bscf'])

    return new_contract


def transition(contract1: CostRecovery | GrossSplit,
               contract2: CostRecovery | GrossSplit,
               tax_rate: float = 0.4):
    # Defining the proportional of the day of the year
    days_in_year = 365 + calendar.isleap(int(contract2.start_date.year))
    dty = (contract2.start_date - date(contract2.start_date.year, 1, 1)).days / days_in_year

    # Defining the transition start date and end date
    start_date_trans = min([contract1.start_date, contract2.start_date])
    end_date_trans = max([contract1.end_date, contract2.end_date])

    # Condition where the first contract end on the 31 December and new contract start on  1 January
    condition = np.logical_and(contract1.end_date.day == 31, contract1.end_date.month == 12)
    if condition:
        additional_year = 1
    else:
        additional_year = 0

    project_years_trans = np.arange(start_date_trans.year, end_date_trans.year + 1 + additional_year)
    project_duration_trans = end_date_trans.year - start_date_trans.year + 1 + additional_year

    # Defining the row gap and description between the prior contract and the new contract
    zeros_to_new = np.zeros(contract2.project_duration - 1 + additional_year)
    zeros_to_prior = np.zeros(contract1.project_duration - 1 + additional_year)
    desc_to_new = ['-'] * (contract2.project_duration - 1 + additional_year)
    desc_to_prior = ['-'] * (contract1.project_duration - 1 + additional_year)

    # Defining the array of years between the prior contract to the new contract
    years_to_new = np.arange(contract1.end_date.year + 1, contract2.end_date.year + 1 + additional_year)
    years_to_prior = np.arange(contract1.start_date.year, contract2.start_date.year + additional_year)

    # Modifying the contract1
    # Changing the start_date and end_date of contract1
    contract1.start_date = start_date_trans
    contract1.end_date = end_date_trans
    contract1.project_duration = project_duration_trans
    contract1.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract1 lifting
    for lift in contract1.lifting:
        lift.start_year = start_date_trans.year
        lift.end_year = end_date_trans.year

        lift.lifting_rate = np.concatenate((lift.lifting_rate, zeros_to_new))
        lift.price = np.concatenate((lift.price, zeros_to_new))
        lift.ghv = np.concatenate((lift.ghv, zeros_to_new))
        lift.prod_rate = np.concatenate((lift.prod_rate, zeros_to_new))
        lift.prod_year = np.concatenate((lift.prod_year, years_to_new))

        lift.project_duration = project_duration_trans
        lift.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract1 tangible
    for tang in contract1.tangible_cost:
        tang.start_year = start_date_trans.year
        tang.end_year = end_date_trans.year

        tang.cost = np.concatenate((tang.cost, zeros_to_new))
        tang.pis_year = np.concatenate((tang.pis_year, zeros_to_new)).astype(int)
        tang.salvage_value = np.concatenate((tang.salvage_value, zeros_to_new))
        tang.useful_life = np.concatenate((tang.useful_life, zeros_to_new))
        tang.depreciation_factor = np.concatenate((tang.depreciation_factor, zeros_to_new))
        tang.description = tang.description + desc_to_new
        tang.expense_year = np.concatenate((tang.expense_year, years_to_new)).astype(int)

        tang.project_duration = project_duration_trans
        tang.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract1 intangible
    for intang in contract1.intangible_cost:
        intang.start_year = start_date_trans.year
        intang.end_year = end_date_trans.year

        intang.cost = np.concatenate((intang.cost, zeros_to_new))
        intang.description = intang.description + desc_to_new
        intang.expense_year = np.concatenate((intang.expense_year, years_to_new)).astype(int)

        intang.project_duration = project_duration_trans
        intang.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract1 opex
    for opx in contract1.opex:
        opx.start_year = start_date_trans.year
        opx.end_year = end_date_trans.year

        opx.fixed_cost = np.concatenate((opx.fixed_cost, zeros_to_new))
        opx.expense_year = np.concatenate((opx.expense_year, years_to_new)).astype(int)
        opx.prod_rate = np.concatenate((opx.prod_rate, zeros_to_new))
        opx.cost_per_volume = np.concatenate((opx.cost_per_volume, zeros_to_new))
        opx.description = opx.description + desc_to_new

        opx.variable_cost = np.concatenate((opx.variable_cost, zeros_to_new))
        opx.cost = np.concatenate((opx.cost, zeros_to_new))
        opx.project_duration = project_duration_trans
        opx.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract1 asr
    for asr in contract1.asr_cost:
        asr.start_year = start_date_trans.year
        asr.end_year = end_date_trans.year

        asr.cost = np.concatenate((asr.cost, zeros_to_new))
        asr.description = asr.description + desc_to_new
        asr.expense_year = np.concatenate((asr.expense_year, years_to_new)).astype(int)

        asr.project_duration = project_duration_trans
        asr.project_years = project_years_trans

    # Modifying the contract2
    # Changing the start_date and end_date of contract2
    contract2.start_date = start_date_trans
    contract2.end_date = end_date_trans
    contract2.project_duration = project_duration_trans
    contract2.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract2 lifting
    for lift in contract2.lifting:
        lift.start_year = start_date_trans.year
        lift.end_year = end_date_trans.year

        lift.lifting_rate = np.concatenate((zeros_to_prior, lift.lifting_rate))
        lift.price = np.concatenate((zeros_to_prior, lift.price))
        lift.ghv = np.concatenate((zeros_to_prior, lift.ghv))
        lift.prod_rate = np.concatenate((zeros_to_prior, lift.prod_rate))
        lift.prod_year = np.concatenate((years_to_prior, lift.prod_year)).astype(int)

        lift.project_duration = project_duration_trans
        lift.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract2 tangible
    for tang in contract2.tangible_cost:
        tang.start_year = start_date_trans.year
        tang.end_year = end_date_trans.year

        tang.cost = np.concatenate((zeros_to_prior, tang.cost))
        tang.pis_year = np.concatenate((zeros_to_prior, tang.pis_year)).astype(int)
        tang.salvage_value = np.concatenate((zeros_to_prior, tang.salvage_value))
        tang.useful_life = np.concatenate((zeros_to_prior, tang.useful_life))
        tang.depreciation_factor = np.concatenate((zeros_to_prior, tang.depreciation_factor))
        tang.description = desc_to_prior + tang.description
        tang.expense_year = np.concatenate((years_to_prior, tang.expense_year)).astype(int)

        tang.project_duration = project_duration_trans
        tang.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract2 intangible
    for intang in contract2.intangible_cost:
        intang.start_year = start_date_trans.year
        intang.end_year = end_date_trans.year

        intang.cost = np.concatenate((zeros_to_prior, intang.cost))
        intang.description = desc_to_prior + intang.description
        intang.expense_year = np.concatenate((years_to_prior, intang.expense_year)).astype(int)

        intang.project_duration = project_duration_trans
        intang.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract2 opex
    for opx in contract2.opex:
        opx.start_year = start_date_trans.year
        opx.end_year = end_date_trans.year

        opx.fixed_cost = np.concatenate((zeros_to_prior, opx.fixed_cost))
        opx.expense_year = np.concatenate((years_to_prior, opx.expense_year)).astype(int)
        opx.prod_rate = np.concatenate((zeros_to_prior, opx.prod_rate))
        opx.cost_per_volume = np.concatenate((zeros_to_prior, opx.cost_per_volume))
        opx.description = desc_to_prior + opx.description

        opx.variable_cost = np.concatenate((zeros_to_prior, opx.variable_cost))
        opx.cost = np.concatenate((zeros_to_prior, opx.cost))
        opx.project_duration = project_duration_trans
        opx.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract2 asr
    for asr in contract2.asr_cost:
        asr.start_year = start_date_trans.year
        asr.end_year = end_date_trans.year

        asr.cost = np.concatenate((zeros_to_prior, asr.cost))
        asr.description = desc_to_prior + asr.description
        asr.expense_year = np.concatenate((years_to_prior, asr.expense_year)).astype(int)

        asr.project_duration = project_duration_trans
        asr.project_years = project_years_trans

    # Parsing the attributes to the new object of contract
    contract1_new = parse_dataclass(contract=contract1)
    contract2_new = parse_dataclass(contract=contract2)

    # Executing the new contract
    contract1_new.run()
    contract2_new.run()

    # Defining the unrecoverable cost from the prior contract
    latest_unrec = contract1_new._consolidated_unrecovered_after_transfer[-1]
    unrec_cost_prior = np.zeros_like(contract2_new.project_years, dtype=float)
    unrec_cost_prior[len(years_to_prior)] = latest_unrec

    # Calculating the Unrecoverable cost in the transitioned cashflow
    unrec_trans = psc_tools.get_unrecovered_cost(revenue=contract2_new._consolidated_taxable_income,
                                                 depreciation=unrec_cost_prior,
                                                 non_capital=np.zeros_like(contract2_new.project_years, dtype=float),
                                                 ftp_ctr=np.zeros_like(contract2_new.project_years, dtype=float),
                                                 ftp_gov=np.zeros_like(contract2_new.project_years, dtype=float),
                                                 ic=np.zeros_like(contract2_new.project_years, dtype=float))

    cost_to_be_recovered_trans = psc_tools.get_cost_to_be_recovered(unrecovered_cost=unrec_trans)

    cost_recovery_trans = CostRecovery._get_cost_recovery(
        revenue=contract2_new._consolidated_taxable_income,
        ftp=np.zeros_like(contract2_new.project_years, dtype=float),
        ic=np.zeros_like(contract2_new.project_years, dtype=float),
        depreciation=unrec_cost_prior,
        non_capital=np.zeros_like(contract2_new.project_years, dtype=float),
        cost_to_be_recovered=cost_to_be_recovered_trans,
        cr_cap_rate=1.0)

    # Calculate taxable income
    taxable_income_trans = CostRecovery._get_ets_before_transfer(revenue=contract2_new._consolidated_taxable_income,
                                                                 ftp_ctr=np.zeros_like(contract2_new.project_years, dtype=float),
                                                                 ftp_gov=np.zeros_like(contract2_new.project_years, dtype=float),
                                                                 ic=np.zeros_like(contract2_new.project_years, dtype=float),
                                                                 cost_recovery=cost_recovery_trans)

    tax_payment_transition = taxable_income_trans * tax_rate


    return contract1_new, contract2_new





