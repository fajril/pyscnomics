import numpy as np
from datetime import date
import calendar

from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit


# Defining the gap between the prior contract and the new contract

# Modifying the Lifting attributes into the current updates project length, fill it with zeros

# Modifying the Tangible, Intangible, OPEX, ASR attributes into the current updates project length, fill it with zeros

# Get the proportion of the days of contract transition to a year

# Modify the latest row of Lifting, Tangible, Intangible, OPEX, ASR into the proportioned value

# Run each PSC

# Sum the generated cashflow


def transition(contract1: CostRecovery | GrossSplit,
               contract2: CostRecovery | GrossSplit):
    # Defining the proportional of the day of the year
    days_in_year = 365 + calendar.isleap(int(contract2.start_date.year))
    days_over_year = (contract2.start_date - date(contract2.start_date.year, 1, 1)).days / days_in_year

    # Defining the transition start date and end date
    start_date_trans = min([contract1.start_date, contract2.start_date])
    end_date_trans = max([contract1.end_date, contract2.end_date])
    project_years_trans = np.arange(start_date_trans.year, end_date_trans.year + 1)
    project_duration_trans = end_date_trans.year - start_date_trans.year + 1

    # Defining the row gap and description between the prior contract and the new contract
    zeros_to_new = np.zeros(contract2.project_duration - 1)
    zeros_to_prior = np.zeros(contract1.project_duration - 1)
    desc_to_new = ['-'] * (contract2.project_duration - 1)
    desc_to_prior = ['-'] * (contract1.project_duration - 1)

    # Defining the array of years between the prior contract to the new contract
    years_to_new = np.arange(contract1.end_date.year + 1, contract2.end_date.year + 1)
    years_to_prior = np.arange(contract1.start_date.year, contract2.start_date.year)

    # Modifying the contract1
    # Changing the start_date and end_date of contract1
    contract1.start_date = start_date_trans
    contract1.end_date = end_date_trans
    contract1.project_duration = project_duration_trans

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
        tang.pis_year = np.concatenate((tang.pis_year, zeros_to_new))
        tang.salvage_value = np.concatenate((tang.salvage_value, zeros_to_new))
        tang.useful_life = np.concatenate((tang.useful_life, zeros_to_new))
        tang.depreciation_factor = np.concatenate((tang.depreciation_factor, zeros_to_new))
        tang.description = tang.description + desc_to_new
        tang.expense_year = np.concatenate((tang.expense_year, years_to_new))

        tang.project_duration = project_duration_trans
        tang.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract1 intangible
    for intang in contract1.tangible_cost:
        intang.start_year = start_date_trans.year
        intang.end_year = end_date_trans.year

        intang.cost = np.concatenate((intang.cost, zeros_to_new))
        intang.description = intang.description + desc_to_new
        intang.expense_year = np.concatenate((intang.expense_year, years_to_new))

        intang.project_duration = project_duration_trans
        intang.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract1 opex
    for opx in contract1.opex:
        opx.start_year = start_date_trans.year
        opx.end_year = end_date_trans.year

        opx.cost = np.concatenate((opx.cost, zeros_to_new))
        opx.description = opx.description + desc_to_new
        opx.expense_year = np.concatenate((opx.expense_year, years_to_new))

        opx.project_duration = project_duration_trans
        opx.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract1 asr
    for asr in contract1.asr_cost:
        asr.start_year = start_date_trans.year
        asr.end_year = end_date_trans.year

        asr.cost = np.concatenate((asr.cost, zeros_to_new))
        asr.description = asr.description + desc_to_new
        asr.expense_year = np.concatenate((asr.expense_year, years_to_new))

        asr.project_duration = project_duration_trans
        asr.project_years = project_years_trans

    # Modifying the contract2
    # Changing the start_date and end_date of contract2
    contract2.start_date = start_date_trans
    contract2.end_date = end_date_trans
    contract1.project_duration = project_duration_trans

    # Concatenating the zeros_to_new and years_to_new to the contract1 lifting
    for lift in contract2.lifting:
        lift.start_year = start_date_trans.year
        lift.end_year = end_date_trans.year

        lift.lifting_rate = np.concatenate((zeros_to_prior, lift.lifting_rate))
        lift.price = np.concatenate((zeros_to_prior, lift.price))
        lift.ghv = np.concatenate((zeros_to_prior, lift.ghv))
        lift.prod_rate = np.concatenate((zeros_to_prior, lift.prod_rate))
        lift.prod_year = np.concatenate((years_to_prior, lift.prod_year))

        lift.project_duration = project_duration_trans
        lift.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract1 tangible
    for tang in contract2.tangible_cost:
        tang.start_year = start_date_trans.year
        tang.end_year = end_date_trans.year

        tang.cost = np.concatenate((zeros_to_prior, tang.cost))
        tang.pis_year = np.concatenate((zeros_to_prior, tang.pis_year))
        tang.salvage_value = np.concatenate((zeros_to_prior, tang.salvage_value))
        tang.useful_life = np.concatenate((zeros_to_prior, tang.useful_life))
        tang.depreciation_factor = np.concatenate((zeros_to_prior, tang.depreciation_factor))
        tang.description = desc_to_prior + tang.description
        tang.expense_year = np.concatenate((years_to_prior, tang.expense_year))

        tang.project_duration = project_duration_trans
        tang.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract1 intangible
    for intang in contract2.tangible_cost:
        intang.start_year = start_date_trans.year
        intang.end_year = end_date_trans.year

        intang.cost = np.concatenate((zeros_to_prior, intang.cost))
        intang.description = desc_to_prior + intang.description
        intang.expense_year = np.concatenate((years_to_prior, intang.expense_year))

        intang.project_duration = project_duration_trans
        intang.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract1 opex
    for opx in contract2.opex:
        opx.start_year = start_date_trans.year
        opx.end_year = end_date_trans.year

        opx.cost = np.concatenate((zeros_to_prior, opx.cost))
        opx.description = desc_to_prior + opx.description
        opx.expense_year = np.concatenate((years_to_prior, opx.expense_year))

        opx.project_duration = project_duration_trans
        opx.project_years = project_years_trans

    # Concatenating the zeros_to_new and years_to_new to the contract1 asr
    for asr in contract2.asr_cost:
        asr.start_year = start_date_trans.year
        asr.end_year = end_date_trans.year

        asr.cost = np.concatenate((zeros_to_prior, asr.cost))
        asr.description = desc_to_prior + asr.description
        asr.expense_year = np.concatenate((years_to_prior, asr.expense_year))

        asr.project_duration = project_duration_trans
        asr.project_years = project_years_trans

    # Adjusting the row where its on the transition year
    indices = np.argwhere()


