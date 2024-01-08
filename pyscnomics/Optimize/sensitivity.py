import numpy as np
import pandas as pd

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.tools.summary import get_summary
from pyscnomics.tools import table
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.revenue import Lifting


def adjust_contract(contract: CostRecovery | GrossSplit,
                    adjusted_variable: str,
                    multiply_factor: float,
                    contract_arguments: dict,
                    summary_arguments: dict):
    # Some attributes that have to be adjusted, they are OPEX, CAPEX, Oil Price, Gas Price, Prod Factor

    # Modifying the OPEX if the chosen adjusted variable is OPEX
    new_opex = contract.opex
    if adjusted_variable == 'OPEX':
        list_opex = []
        for opx in contract.opex:
            opex = opx * multiply_factor
            list_opex.append(opex)
        new_opex = tuple(list_opex)

    # Modifying the CAPEX if the chosen adjusted variable is CAPEX
    new_capex_tangible = contract.tangible_cost
    new_capex_intangible = contract.intangible_cost

    # Modifying the CAPEX if the chosen adjusted variable is CAPEX
    if adjusted_variable == 'CAPEX':
        # Modifying the Tangible
        list_capex_tangible = []
        for cpx in contract.tangible_cost:
            capex = cpx * multiply_factor
            list_capex_tangible.append(capex)
        new_capex_tangible = tuple(list_capex_tangible)

        # Modifying the Intangible
        list_capex_intangible = []
        for cpx in contract.intangible_cost:
            capex = cpx * multiply_factor
            list_capex_intangible.append(capex)
        new_capex_intangible = tuple(list_capex_intangible)

    # Modifying the Price if the chosen adjusted variable is Oil Price or Gas Price
    # if adjusted_variable == 'Oil Price' or adjusted_variable == 'Gas Price':
    new_lifting = contract.lifting
    if adjusted_variable == 'Oil Price':
        list_lifting = []
        for lft in contract.lifting:
            if lft.fluid_type == FluidType.OIL:
                lifting_price = lft.price * multiply_factor
                lift = Lifting(start_year=lft.start_year,
                               end_year=lft.end_year,
                               lifting_rate=lft.lifting_rate,
                               price=lifting_price,
                               prod_year=lft.prod_year,
                               fluid_type=lft.fluid_type,
                               ghv=lft.ghv,
                               prod_rate=lft.prod_rate)
                list_lifting.append(lift)
            elif lft.fluid_type == FluidType.GAS:
                lift = Lifting(start_year=lft.start_year,
                               end_year=lft.end_year,
                               lifting_rate=lft.lifting_rate,
                               price=lft.price,
                               prod_year=lft.prod_year,
                               fluid_type=lft.fluid_type,
                               ghv=lft.ghv,
                               prod_rate=lft.prod_rate)
                list_lifting.append(lift)
        new_lifting = tuple(list_lifting)

    # Modifying the Price if the chosen adjusted variable is Oil Price or Gas Price
    # if adjusted_variable == 'Oil Price' or adjusted_variable == 'Gas Price':
    if adjusted_variable == 'Gas Price':
        list_lifting = []
        for lft in contract.lifting:
            if lft.fluid_type == FluidType.GAS:
                lifting_price = lft.price * multiply_factor
                lift = Lifting(start_year=lft.start_year,
                               end_year=lft.end_year,
                               lifting_rate=lft.lifting_rate,
                               price=lifting_price,
                               prod_year=lft.prod_year,
                               fluid_type=lft.fluid_type,
                               ghv=lft.ghv,
                               prod_rate=lft.prod_rate)
                list_lifting.append(lift)
            elif lft.fluid_type == FluidType.OIL:
                lift = Lifting(start_year=lft.start_year,
                               end_year=lft.end_year,
                               lifting_rate=lft.lifting_rate,
                               price=lft.price,
                               prod_year=lft.prod_year,
                               fluid_type=lft.fluid_type,
                               ghv=lft.ghv,
                               prod_rate=lft.prod_rate)
                list_lifting.append(lift)
        new_lifting = tuple(list_lifting)

    # Modifying the Lifting Rate if the chosen adjusted variable is Prod Factor
    if adjusted_variable == 'Prod Factor':
        list_lifting = []
        for lft in contract.lifting:
            lifting_rate_adjusted = lft.lifting_rate * multiply_factor
            lift = Lifting(start_year=lft.start_year,
                           end_year=lft.end_year,
                           lifting_rate=lifting_rate_adjusted,
                           price=lft.price,
                           prod_year=lft.prod_year,
                           fluid_type=lft.fluid_type,
                           ghv=lft.ghv,
                           prod_rate=lft.prod_rate)
            list_lifting.append(lift)

        new_lifting = tuple(list_lifting)

    # Making new contract object that filled with adjusted attributes
    # Getting the attributes of the contract
    attr_contract = vars(contract)

    # Condition where only one fluid is produced
    produced_fluid = [i.fluid_type for i in attr_contract['lifting']]

    if FluidType.OIL not in produced_fluid:
        attr_contract['oil_onstream_date'] = None
    if FluidType.GAS not in produced_fluid:
        attr_contract['gas_onstream_date'] = None

    new_contract = None
    if isinstance(contract, CostRecovery):
        new_contract = CostRecovery(
            start_date=attr_contract['start_date'],
            end_date=attr_contract['end_date'],
            oil_onstream_date=attr_contract['oil_onstream_date'],
            gas_onstream_date=attr_contract['gas_onstream_date'],
            lifting=new_lifting,
            tangible_cost=new_capex_tangible,
            intangible_cost=new_capex_intangible,
            opex=new_opex,
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

    # Running the new contract
    new_contract.run(**contract_arguments)
    summary_arguments['contract'] = new_contract
    contract_summary = get_summary(**summary_arguments)

    return contract_summary, new_contract


def sensitivity_psc(steps: int,
                    diff: float,
                    contract: CostRecovery | GrossSplit,
                    contract_arguments: dict,
                    summary_arguments: dict
                    ):
    # Defining the array named steps_arr that containing swings factor of the sens
    increment = diff / steps
    right_steps = np.arange(1 + increment, 1 + diff + increment, increment)
    left_steps = np.arange(1 - diff, 1, increment)
    steps_arr = np.concatenate((left_steps, np.array([1]), right_steps))

    sens_dict = {'OPEX': {},
                 'CAPEX': {},
                 'Oil Price': {},
                 'Gas Price': {},
                 'Prod Factor': {},
                 }

    list_of_psc = []

    # Defining the list to contain the result of the contract variance and running the contract along the steps_arr
    for key in sens_dict.keys():
        # Defining the array to contain the result
        result_npv = np.empty(len(steps_arr))
        result_irr = np.empty(len(steps_arr))
        result_pi = np.empty(len(steps_arr))
        result_gov_take = np.empty(len(steps_arr))
        result_ctr_take = np.empty(len(steps_arr))

        for index, mul_factor in enumerate(steps_arr):
            # Defining the contract for each loop
            psc = contract

            # Adjusting the contract
            result_psc, contract_new = adjust_contract(contract=psc,
                                                       adjusted_variable=key,
                                                       multiply_factor=mul_factor,
                                                       contract_arguments=contract_arguments,
                                                       summary_arguments=summary_arguments)

            # Filling the summary result to result list
            result_npv[index] = result_psc['ctr_npv']
            result_irr[index] = result_psc['ctr_irr']
            result_pi[index] = result_psc['ctr_pi']
            result_gov_take[index] = result_psc['gov_take']
            result_ctr_take[index] = result_psc['ctr_net_share']

            key_result = {'steps': steps_arr,
                          'ctr_npv': result_npv,
                          'ctr_irr': result_irr,
                          'ctr_pi': result_pi,
                          'gov_take': result_gov_take,
                          'ctr_net_share': result_ctr_take, }

            sens_dict[key] = key_result

            list_of_psc.append(contract_new)

    return sens_dict, list_of_psc
