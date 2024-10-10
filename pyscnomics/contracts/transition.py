from dataclasses import dataclass, field
import numpy as np
from datetime import date

from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR
from pyscnomics.econ.selection import FluidType
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts import psc_tools


def adjust_rows(original_value: float | np.ndarray,
                first_contract: bool,
                prior_rows: np.ndarray,
                post_rows: np.ndarray, ):
    """
    Function to adjust an array based on the given condition, tailored for transition contract.
    Parameters
    ----------
    original_value: float | np.ndarray
        The original value of the data.
    first_contract: bool
        The condition whether the row is existed in first or second contract in Transition scheme.
    prior_rows: np.ndarray
        The array that will be placed before the given value.
    post_rows:
        The array that will be placed after the given value.
    Returns
    -------
    adjusted_values: np.ndarray
        The array containin the adjusted value.

    """
    if isinstance(original_value, float):
        adjusted_values = original_value

    elif isinstance(original_value, np.ndarray) and first_contract:
        adjusted_values = np.concatenate((original_value, post_rows))

    elif isinstance(original_value, np.ndarray) and not first_contract:
        adjusted_values = np.concatenate((prior_rows, original_value))

    else:
        adjusted_values = original_value

    return adjusted_values


def adjusting_contract_arguments(arguments_dict: dict,
                                 project_years: np.ndarray,
                                 first_contract: bool,
                                 prior_rows: np.ndarray,
                                 post_rows: np.ndarray,
                                 ):
    """
    The function used to adjust the contract arguments before parsed to the transition contract.
    Parameters
    ----------
    arguments_dict: dict
        The dictionary containing the arguments of a contract.
    project_years: np.ndarray
        The array of the project years.
    first_contract: bool
        The condition whether the row is existed in first or second contract in Transition scheme.
    prior_rows: np.ndarray
        The array that will be placed before the given value.
    post_rows:
        The array that will be placed after the given value.
    Returns
    -------
    arguments_dict: dict
        The dictionary containing adjusted arguments.
    """
    if 'tax_rate' in arguments_dict.keys():
        arguments_dict['tax_rate'] = adjust_rows(original_value=arguments_dict['tax_rate'],
                                                 first_contract=first_contract,
                                                 prior_rows=prior_rows,
                                                 post_rows=post_rows, )

    if 'vat_rate' in arguments_dict.keys():
        arguments_dict['vat_rate'] = adjust_rows(original_value=arguments_dict['vat_rate'],
                                                 first_contract=first_contract,
                                                 prior_rows=prior_rows,
                                                 post_rows=post_rows, )

    if 'lbt_rate' in arguments_dict.keys():
        arguments_dict['lbt_rate'] = adjust_rows(original_value=arguments_dict['lbt_rate'],
                                                 first_contract=first_contract,
                                                 prior_rows=prior_rows,
                                                 post_rows=post_rows, )

    if 'inflation_rate' in arguments_dict.keys():
        arguments_dict['inflation_rate'] = adjust_rows(original_value=arguments_dict['inflation_rate'],
                                                       first_contract=first_contract,
                                                       prior_rows=prior_rows,
                                                       post_rows=post_rows, )

    if 'cum_production_split_offset' in arguments_dict.keys():
        arguments_dict['cum_production_split_offset'] = adjust_rows(
            original_value=arguments_dict['cum_production_split_offset'],
            first_contract=first_contract,
            prior_rows=prior_rows,
            post_rows=post_rows, )

    return arguments_dict


@dataclass
class Transition:
    """
    Dataclass that represents Transition contract.

    Parameters
    ----------
    contract1: CostRecovery | GrossSplit
        The object of the first contract.
    contract2: CostRecovery | GrossSplit
        The object of the second contract.
    argument_contract1: dict
        The argument of the first contract.
    argument_contract2: dict
        The argument of the second contract.
    """
    contract1: CostRecovery | GrossSplit
    contract2: CostRecovery | GrossSplit
    argument_contract1: dict
    argument_contract2: dict

    # Attributes to define later
    _contract1_transitioned: CostRecovery | GrossSplit = field(default=None, init=False, repr=False)
    _contract2_transitioned: CostRecovery | GrossSplit = field(default=None, init=False, repr=False)

    project_years: np.ndarray = field(default=None, init=False, repr=False)

    _oil_lifting: Lifting = field(default=None, init=False, repr=False)
    _gas_lifting: Lifting = field(default=None, init=False, repr=False)
    _sulfur_lifting: Lifting = field(default=None, init=False, repr=False)
    _electricity_lifting: Lifting = field(default=None, init=False, repr=False)
    _co2_lifting: Lifting = field(default=None, init=False, repr=False)

    _oil_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _gas_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _sulfur_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _electricity_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _co2_revenue: np.ndarray = field(default=None, init=False, repr=False)

    _oil_capital_expenditures: np.ndarray = field(default=None, init=False, repr=False)
    _gas_capital_expenditures: np.ndarray = field(default=None, init=False, repr=False)
    _oil_intangible_expenditures: np.ndarray = field(default=None, init=False, repr=False)
    _gas_intangible_expenditures: np.ndarray = field(default=None, init=False, repr=False)
    _oil_opex_expenditures: np.ndarray = field(default=None, init=False, repr=False)
    _gas_opex_expenditures: np.ndarray = field(default=None, init=False, repr=False)
    _oil_asr_expenditures: np.ndarray = field(default=None, init=False, repr=False)
    _gas_asr_expenditures: np.ndarray = field(default=None, init=False, repr=False)

    _oil_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _gas_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _sulfur_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _electricity_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _co2_wap_price: np.ndarray = field(default=None, init=False, repr=False)

    _oil_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _oil_deductible_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_deductible_cost: np.ndarray = field(default=None, init=False, repr=False)
    _oil_unrec_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_unrec_cost: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ctr_ets: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_ets: np.ndarray = field(default=None, init=False, repr=False)
    _oil_gov_ets: np.ndarray = field(default=None, init=False, repr=False)
    _gas_gov_ets: np.ndarray = field(default=None, init=False, repr=False)
    _net_operating_profit: np.ndarray = field(default=None, init=False, repr=False)
    _ctr_net_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _ctr_ftp: np.ndarray = field(default=None, init=False, repr=False)
    _gov_ftp: np.ndarray = field(default=None, init=False, repr=False)
    _ddmo: np.ndarray = field(default=None, init=False, repr=False)
    _tax_payment: np.ndarray = field(default=None, init=False, repr=False)
    _government_take: np.ndarray = field(default=None, init=False, repr=False)
    _oil_undepreciated_asset: float = field(default=None, init=False, repr=False)
    _gas_undepreciated_asset: float = field(default=None, init=False, repr=False)

    # Consolidated Attributes
    _consolidated_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_undepreciated_asset: np.ndarray | float = field(default=None, init=False, repr=False)
    _consolidated_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ddmo: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_tax_payment: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_government_take: np.ndarray = field(default=None, init=False, repr=False)

    @staticmethod
    def _parse_dataclass(
            contract: CostRecovery | GrossSplit,
            start_date_trans: date,
            end_date_trans: date,
            lifting: tuple,
            capital: tuple,
            intangible: tuple,
            opex: tuple,
            asr: tuple,
    ):

        # Condition where only one fluid is produced
        produced_fluid = [i.fluid_type for i in contract.lifting]

        if FluidType.OIL not in produced_fluid:
            oil_onstream_date = None
        else:
            oil_onstream_date = contract.oil_onstream_date
        if FluidType.GAS not in produced_fluid:
            gas_onstream_date = None
        else:
            gas_onstream_date = contract.gas_onstream_date

        if isinstance(contract, CostRecovery):
            new_contract = CostRecovery(
                start_date=start_date_trans,
                end_date=end_date_trans,
                oil_onstream_date=oil_onstream_date,
                gas_onstream_date=gas_onstream_date,
                lifting=lifting,
                capital_cost=capital,
                intangible_cost=intangible,
                opex=opex,
                asr_cost=asr,
                oil_ftp_is_available=contract.oil_ftp_is_available,
                oil_ftp_is_shared=contract.oil_ftp_is_shared,
                oil_ftp_portion=contract.oil_ftp_portion,
                gas_ftp_is_available=contract.gas_ftp_is_available,
                gas_ftp_is_shared=contract.gas_ftp_is_shared,
                gas_ftp_portion=contract.gas_ftp_portion,
                tax_split_type=contract.tax_split_type,
                condition_dict=contract.condition_dict,
                indicator_rc_icp_sliding=contract.indicator_rc_icp_sliding,
                oil_ctr_pretax_share=contract.oil_ctr_pretax_share,
                gas_ctr_pretax_share=contract.gas_ctr_pretax_share,
                oil_ic_rate=contract.oil_ic_rate,
                gas_ic_rate=contract.gas_ic_rate,
                ic_is_available=contract.ic_is_available,
                oil_cr_cap_rate=contract.oil_cr_cap_rate,
                gas_cr_cap_rate=contract.gas_cr_cap_rate,
                oil_dmo_volume_portion=contract.oil_dmo_volume_portion,
                oil_dmo_fee_portion=contract.oil_dmo_fee_portion,
                oil_dmo_holiday_duration=contract.oil_dmo_holiday_duration,
                gas_dmo_volume_portion=contract.gas_dmo_volume_portion,
                gas_dmo_fee_portion=contract.gas_dmo_fee_portion,
                gas_dmo_holiday_duration=contract.gas_dmo_holiday_duration)
        else:
            new_contract = GrossSplit(
                start_date=start_date_trans,
                end_date=end_date_trans,
                oil_onstream_date=oil_onstream_date,
                gas_onstream_date=gas_onstream_date,
                lifting=lifting,
                capital_cost=capital,
                intangible_cost=intangible,
                opex=opex,
                asr_cost=asr,
                field_status=contract.field_status,
                field_loc=contract.field_loc,
                res_depth=contract.res_depth,
                infra_avail=contract.infra_avail,
                res_type=contract.res_type,
                api_oil=contract.api_oil,
                domestic_use=contract.domestic_use,
                prod_stage=contract.prod_stage,
                co2_content=contract.co2_content,
                h2s_content=contract.h2s_content,
                base_split_ctr_oil=contract.base_split_ctr_oil,
                base_split_ctr_gas=contract.base_split_ctr_gas,
                split_ministry_disc=contract.split_ministry_disc,
                oil_dmo_volume_portion=contract.oil_dmo_volume_portion,
                oil_dmo_fee_portion=contract.oil_dmo_fee_portion,
                oil_dmo_holiday_duration=contract.oil_dmo_holiday_duration,
                gas_dmo_volume_portion=contract.gas_dmo_volume_portion,
                gas_dmo_fee_portion=contract.gas_dmo_fee_portion,
                gas_dmo_holiday_duration=contract.gas_dmo_holiday_duration,
            )

        return new_contract

    def run(self, unrec_portion: float = 0.0):
        # Defining the transition start date and end date
        start_date_trans = min([self.contract1.start_date, self.contract2.start_date])
        end_date_trans = max([self.contract1.end_date, self.contract2.end_date])

        # Condition where the first contract end on the 31 December and new contract start on  1 January
        condition = np.logical_and(self.contract1.end_date.day == 31, self.contract1.end_date.month == 12)
        if condition:
            # additional_year = 1
            additional_year = 0
            end_date_condition = 1
        else:
            additional_year = 0
            end_date_condition = 0

        project_years_trans = np.arange(start_date_trans.year, end_date_trans.year + 1 + additional_year)
        project_duration_trans = end_date_trans.year - start_date_trans.year + 1 + additional_year

        # Defining the array of years between the prior contract to the new contract
        years_to_new = np.arange(self.contract1.end_date.year + 1,
                                 self.contract2.end_date.year + 1 + additional_year - end_date_condition)
        years_to_prior = np.arange(self.contract1.start_date.year,
                                   self.contract2.start_date.year + additional_year - end_date_condition)

        len_to_new = len(years_to_new)
        len_to_prior = len(years_to_prior)

        # Defining the row gap and description between the prior contract and the new contract
        zeros_to_new = np.zeros_like(years_to_new, dtype=float)
        zeros_to_prior = np.zeros_like(years_to_prior, dtype=float)
        desc_to_new = ['-'] * len_to_new
        desc_to_prior = ['-'] * len_to_prior
        fluid_to_new = [FluidType.OIL] * len_to_new
        fluid_to_prior = [FluidType.OIL] * len_to_prior
        ic_to_new = [False] * len_to_new
        ic_to_prior = [False] * len_to_prior

        # Modifying the contract1
        # Changing the start_date and end_date of contract1
        new_lifting_1st = [
            Lifting(start_year=start_date_trans.year,
                    end_year=end_date_trans.year,
                    lifting_rate=np.concatenate((lift.lifting_rate, zeros_to_new)),
                    price=np.concatenate((lift.price, zeros_to_new)),
                    prod_year=np.concatenate((lift.prod_year, years_to_new)),
                    fluid_type=lift.fluid_type,
                    ghv=np.concatenate((lift.ghv, zeros_to_new)),
                    prod_rate=np.concatenate((lift.prod_rate, zeros_to_new)),
                    prod_rate_baseline=np.concatenate((lift.prod_rate_baseline, zeros_to_new)))
            for lift in self.contract1.lifting
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract1 tangible
        new_tang_1st = [
            CapitalCost(start_year=start_date_trans.year,
                        end_year=end_date_trans.year,
                        cost=np.concatenate((tang.cost, zeros_to_new)),
                        expense_year=np.concatenate((tang.expense_year, years_to_new)).astype(int),
                        cost_allocation=tang.cost_allocation + fluid_to_new,
                        description=tang.description + desc_to_new,
                        vat_portion=np.concatenate((tang.vat_portion, zeros_to_new)),
                        vat_discount=np.concatenate((tang.vat_discount, zeros_to_new)),
                        lbt_portion=np.concatenate((tang.lbt_portion, zeros_to_new)),
                        lbt_discount=np.concatenate((tang.lbt_discount, zeros_to_new)),
                        pis_year=np.concatenate((tang.pis_year, zeros_to_new)).astype(int),
                        salvage_value=np.concatenate((tang.salvage_value, zeros_to_new)),
                        useful_life=np.concatenate((tang.useful_life, zeros_to_new)),
                        depreciation_factor=np.concatenate((tang.depreciation_factor, zeros_to_new)),
                        is_ic_applied=tang.is_ic_applied + ic_to_new,)
            for tang in self.contract1.capital_cost
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract1 intangible
        new_intang_1st = [
            Intangible(start_year=start_date_trans.year,
                       end_year=end_date_trans.year,
                       cost=np.concatenate((intang.cost, zeros_to_new)),
                       expense_year=np.concatenate((intang.expense_year, years_to_new)).astype(int),
                       cost_allocation=intang.cost_allocation + fluid_to_new,
                       description=intang.description + desc_to_new,
                       vat_portion=np.concatenate((intang.vat_portion, zeros_to_new)),
                       vat_discount=np.concatenate((intang.vat_discount, zeros_to_new)),
                       lbt_portion=np.concatenate((intang.lbt_portion, zeros_to_new)),
                       lbt_discount=np.concatenate((intang.lbt_discount, zeros_to_new)))
            for intang in self.contract1.intangible_cost
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract1 opex
        new_opex_1st = [
            OPEX(start_year=start_date_trans.year,
                 end_year=end_date_trans.year,
                 expense_year=np.concatenate((opx.expense_year, years_to_new)).astype(int),
                 cost_allocation=opx.cost_allocation + fluid_to_new,
                 description=opx.description + desc_to_new,
                 fixed_cost=np.concatenate((opx.fixed_cost, zeros_to_new)),
                 prod_rate=np.concatenate((opx.prod_rate, zeros_to_new)),
                 cost_per_volume=np.concatenate((opx.cost_per_volume, zeros_to_new)),
                 vat_portion=np.concatenate((opx.vat_portion, zeros_to_new)),
                 vat_discount=np.concatenate((opx.vat_discount, zeros_to_new)),
                 lbt_portion=np.concatenate((opx.lbt_portion, zeros_to_new)),
                 lbt_discount=np.concatenate((opx.lbt_discount, zeros_to_new)),)
            for opx in self.contract1.opex
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract1 asr
        new_asr_1st = [
            ASR(start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                cost=np.concatenate((asr.cost, zeros_to_new)),
                expense_year=np.concatenate((asr.expense_year, years_to_new)).astype(int),
                cost_allocation=asr.cost_allocation + fluid_to_new,
                description=asr.description + desc_to_new,
                vat_portion=np.concatenate((asr.vat_portion, zeros_to_new)),
                vat_discount=np.concatenate((asr.vat_discount, zeros_to_new)),
                lbt_portion=np.concatenate((asr.lbt_portion, zeros_to_new)),
                lbt_discount=np.concatenate((asr.lbt_discount, zeros_to_new)))
            for asr in self.contract1.asr_cost
        ]

        # Modifying the contract2
        # Changing the start_date and end_date of contract2

        # Concatenating the zeros_to_new and years_to_new to the contract2 lifting
        new_lifting_2nd = [
            Lifting(start_year=start_date_trans.year,
                    end_year=end_date_trans.year,
                    lifting_rate=np.concatenate((zeros_to_prior, lift.lifting_rate)),
                    price=np.concatenate((zeros_to_prior, lift.price)),
                    prod_year=np.concatenate((years_to_prior, lift.prod_year)).astype(int),
                    fluid_type=lift.fluid_type,
                    ghv=np.concatenate((zeros_to_prior, lift.ghv)),
                    prod_rate=np.concatenate((zeros_to_prior, lift.prod_rate)),
                    prod_rate_baseline=np.concatenate((zeros_to_prior, lift.prod_rate_baseline)))
            for lift in self.contract2.lifting
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract2 tangible
        new_tang_2nd = [
            CapitalCost(start_year=start_date_trans.year,
                        end_year=end_date_trans.year,
                        cost=np.concatenate((zeros_to_prior, tang.cost)),
                        expense_year=np.concatenate((years_to_prior, tang.expense_year)).astype(int),
                        cost_allocation=fluid_to_prior + tang.cost_allocation,
                        description=desc_to_prior + tang.description,
                        vat_portion=np.concatenate((zeros_to_prior, tang.vat_portion)),
                        vat_discount=np.concatenate((zeros_to_prior, tang.vat_discount)),
                        lbt_portion=np.concatenate((zeros_to_prior, tang.lbt_portion)),
                        lbt_discount=np.concatenate((zeros_to_prior, tang.lbt_discount)),
                        pis_year=np.concatenate((zeros_to_prior, tang.pis_year)).astype(int),
                        salvage_value=np.concatenate((zeros_to_prior, tang.salvage_value)),
                        useful_life=np.concatenate((zeros_to_prior, tang.useful_life)),
                        depreciation_factor=np.concatenate((zeros_to_prior, tang.depreciation_factor)),
                        is_ic_applied=ic_to_prior + tang.is_ic_applied)
            for tang in self.contract2.capital_cost
        ]
        # Concatenating the zeros_to_new and years_to_new to the contract2 intangible
        new_intang_2nd = [
            Intangible(start_year=start_date_trans.year,
                       end_year=end_date_trans.year,
                       cost=np.concatenate((zeros_to_prior, intang.cost)),
                       expense_year=np.concatenate((years_to_prior, intang.expense_year)).astype(int),
                       cost_allocation=fluid_to_prior + intang.cost_allocation,
                       description=desc_to_prior + intang.description,
                       vat_portion=np.concatenate((zeros_to_prior, intang.vat_portion)),
                       vat_discount=np.concatenate((zeros_to_prior, intang.vat_discount)),
                       lbt_portion=np.concatenate((zeros_to_prior, intang.lbt_portion)),
                       lbt_discount=np.concatenate((zeros_to_prior, intang.lbt_discount)))
            for intang in self.contract2.intangible_cost
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract2 opex
        new_opex_2nd = [
            OPEX(start_year=start_date_trans.year,
                 end_year=end_date_trans.year,
                 expense_year=np.concatenate((years_to_prior, opx.expense_year)).astype(int),
                 cost_allocation=fluid_to_prior + opx.cost_allocation,
                 description=desc_to_prior + opx.description,
                 fixed_cost=np.concatenate((zeros_to_prior, opx.fixed_cost)),
                 prod_rate=np.concatenate((zeros_to_prior, opx.prod_rate)),
                 cost_per_volume=np.concatenate((zeros_to_prior, opx.cost_per_volume)),
                 vat_portion=np.concatenate((zeros_to_prior, opx.vat_portion)),
                 vat_discount=np.concatenate((zeros_to_prior, opx.vat_discount)),
                 lbt_portion=np.concatenate((zeros_to_prior, opx.lbt_portion)),
                 lbt_discount=np.concatenate((zeros_to_prior, opx.lbt_discount)))
            for opx in self.contract2.opex
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract2 asr
        new_asr_2nd = [
            ASR(start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                cost=np.concatenate((zeros_to_prior, asr.cost)),
                expense_year=np.concatenate((years_to_prior, asr.expense_year)).astype(int),
                cost_allocation=fluid_to_prior + asr.cost_allocation,
                description=desc_to_prior + asr.description,
                vat_portion=np.concatenate((zeros_to_prior, asr.vat_portion)),
                vat_discount=np.concatenate((zeros_to_prior, asr.vat_discount)),
                lbt_portion=np.concatenate((zeros_to_prior, asr.lbt_portion)),
                lbt_discount=np.concatenate((zeros_to_prior, asr.lbt_discount)))
            for asr in self.contract2.asr_cost
        ]

        # Parsing the attributes to the new object of contract
        contract1_new = self._parse_dataclass(
            contract=self.contract1,
            start_date_trans=start_date_trans,
            end_date_trans=end_date_trans,
            lifting=tuple(new_lifting_1st),
            capital=tuple(new_tang_1st),
            intangible=tuple(new_intang_1st),
            opex=tuple(new_opex_1st),
            asr=tuple(new_asr_1st),
        )

        contract2_new = self._parse_dataclass(
            contract=self.contract2,
            start_date_trans=start_date_trans,
            end_date_trans=end_date_trans,
            lifting=tuple(new_lifting_2nd),
            capital=tuple(new_tang_2nd),
            intangible=tuple(new_intang_2nd),
            opex=tuple(new_opex_2nd),
            asr=tuple(new_asr_2nd),
        )

        # Adjusting the contract arguments
        new_argument_contract1 = adjusting_contract_arguments(arguments_dict=self.argument_contract1,
                                                              project_years=project_years_trans,
                                                              first_contract=True,
                                                              prior_rows=zeros_to_prior,
                                                              post_rows=zeros_to_new,)

        new_argument_contract2 = adjusting_contract_arguments(arguments_dict=self.argument_contract2,
                                                              project_years=project_years_trans,
                                                              first_contract=False,
                                                              prior_rows=zeros_to_prior,
                                                              post_rows=zeros_to_new,)

        # Executing the new contract
        # for i in self.argument_contract1.keys():
        #     print(i, ': ', self.argument_contract1[i])
        # input()

        contract1_new.run(**new_argument_contract1)
        contract2_new.run(**new_argument_contract2)

        # Defining the unrecoverable cost from the prior contract
        # Condition if the contract is Cost Recovery
        if isinstance(contract1_new, CostRecovery):
            latest_unrec = contract1_new._consolidated_unrecovered_after_transfer[-1]
        # Condition if the contract is Gross Split
        else:
            latest_unrec = contract1_new._consolidated_carward_cost_aftertf[-1]
        unrec_cost_prior = np.zeros_like(contract2_new.project_years, dtype=float)
        unrec_cost_prior[len(years_to_prior)] = latest_unrec

        # Calculating the Unrecoverable cost in the transitioned cashflow
        unrec_trans = psc_tools.get_unrecovered_cost(revenue=contract2_new._consolidated_taxable_income,
                                                     depreciation=unrec_cost_prior,
                                                     non_capital=np.zeros_like(contract2_new.project_years,
                                                                               dtype=float),
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

        # Calculate taxable income after considering transferred unrecoverable cost from contract1
        taxable_income_trans = CostRecovery._get_ets_before_transfer(revenue=contract2_new._consolidated_taxable_income,
                                                                     ftp_ctr=np.zeros_like(contract2_new.project_years,
                                                                                           dtype=float),
                                                                     ftp_gov=np.zeros_like(contract2_new.project_years,
                                                                                           dtype=float),
                                                                     ic=np.zeros_like(contract2_new.project_years,
                                                                                      dtype=float),
                                                                     cost_recovery=cost_recovery_trans)

        # Calculate tax payment of adjusted taxable income
        tax_payment_transition = taxable_income_trans * contract2_new._tax_rate_arr

        # Calculate new ctr net share
        contract2_new._consolidated_ctr_net_share = (contract2_new._consolidated_taxable_income -
                                                     (unrec_cost_prior * unrec_portion) -
                                                     tax_payment_transition)

        # Calculate new cashflow of contract2
        contract2_new._consolidated_cashflow = (contract2_new._consolidated_cashflow +
                                                contract2_new._consolidated_tax_payment -
                                                tax_payment_transition
                                                )

        # Calculate new government take
        contract2_new._consolidated_government_take = (contract2_new._consolidated_government_take -
                                                       contract2_new._consolidated_tax_payment +
                                                       tax_payment_transition)

        contract2_new._consolidated_tax_payment = tax_payment_transition

        # Parsing new contract to the attributes of Transition dataclass
        self._contract1_transitioned = contract1_new
        self._contract2_transitioned = contract2_new

        # Populating transition attributes
        # Lifting
        self._oil_lifting = (self._contract1_transitioned._oil_lifting +
                             self._contract2_transitioned._oil_lifting)

        self._gas_lifting = (self._contract1_transitioned._gas_lifting +
                             self._contract2_transitioned._gas_lifting)

        self._sulfur_lifting = (self._contract1_transitioned._sulfur_lifting +
                                self._contract2_transitioned._sulfur_lifting)

        self._electricity_lifting = (self._contract1_transitioned._electricity_lifting +
                                     self._contract2_transitioned._electricity_lifting)

        self._co2_lifting = (self._contract1_transitioned._co2_lifting +
                             self._contract2_transitioned._co2_lifting)

        # Revenue
        self._oil_revenue = (self._contract1_transitioned._oil_revenue +
                             self._contract2_transitioned._oil_revenue)

        self._gas_revenue = (self._contract1_transitioned._gas_revenue +
                             self._contract2_transitioned._gas_revenue)

        self._sulfur_revenue = (self._contract1_transitioned._sulfur_revenue +
                                self._contract2_transitioned._sulfur_revenue)

        self._electricity_revenue = (self._contract1_transitioned._electricity_revenue +
                                     self._contract2_transitioned._electricity_revenue)

        self._co2_revenue = (self._contract1_transitioned._co2_revenue +
                             self._contract2_transitioned._co2_revenue)

        # Tangible
        self._oil_capital_expenditures = (self._contract1_transitioned._oil_capital_expenditures +
                                          self._contract2_transitioned._oil_capital_expenditures)

        self._gas_capital_expenditures = (self._contract1_transitioned._gas_capital_expenditures +
                                          self._contract2_transitioned._gas_capital_expenditures)

        # Intangible
        self._oil_intangible_expenditures = (self._contract1_transitioned._oil_intangible_expenditures +
                                             self._contract2_transitioned._oil_intangible_expenditures)

        self._gas_intangible_expenditures = (self._contract1_transitioned._gas_intangible_expenditures +
                                             self._contract2_transitioned._gas_intangible_expenditures)

        # Opex
        self._oil_opex_expenditures = (self._contract1_transitioned._oil_opex_expenditures +
                                       self._contract2_transitioned._oil_opex_expenditures)

        self._gas_opex_expenditures = (self._contract1_transitioned._gas_opex_expenditures +
                                       self._contract2_transitioned._gas_opex_expenditures)

        # ASR
        self._oil_asr_expenditures = (self._contract1_transitioned._oil_asr_expenditures +
                                      self._contract2_transitioned._oil_asr_expenditures)

        self._gas_asr_expenditures = (self._contract1_transitioned._gas_asr_expenditures +
                                      self._contract2_transitioned._gas_asr_expenditures)

        # WAP Price
        self._oil_wap_price = self._contract2_transitioned._oil_wap_price
        self._gas_wap_price = self._contract2_transitioned._gas_wap_price
        self._sulfur_wap_price = self._contract2_transitioned._sulfur_wap_price
        self._electricity_wap_price = self._contract2_transitioned._electricity_wap_price
        self._co2_wap_price = self._contract2_transitioned._co2_wap_price

        # Undepreciated Asset
        self._oil_undepreciated_asset = (self.contract1._oil_undepreciated_asset +
                                         self.contract2._oil_undepreciated_asset)
        self._gas_undepreciated_asset = (self.contract1._gas_undepreciated_asset +
                                         self.contract2._gas_undepreciated_asset)
        self._consolidated_undepreciated_asset = self._oil_undepreciated_asset + self._gas_undepreciated_asset

        # Sunk Cost
        self._oil_sunk_cost = self._contract1_transitioned._oil_sunk_cost + self._contract2_transitioned._oil_sunk_cost
        self._gas_sunk_cost = self._contract1_transitioned._gas_sunk_cost + self._contract2_transitioned._gas_sunk_cost

        # Deductible Cost / Cost Recovery
        _oil_deductible_cost_1 = None
        _gas_deductible_cost_1 = None
        _oil_deductible_cost_2 = None
        _gas_deductible_cost_2 = None

        _oil_unrec_cost_1 = None
        _gas_unrec_cost_1 = None
        _oil_unrec_cost_2 = None
        _gas_unrec_cost_2 = None

        _oil_ctr_ets_1 = None
        _gas_ctr_ets_1 = None
        _oil_ctr_ets_2 = None
        _gas_ctr_ets_2 = None

        _oil_gov_ets_1 = None
        _gas_gov_ets_1 = None
        _oil_gov_ets_2 = None
        _gas_gov_ets_2 = None

        _ctr_ftp_1 = None
        _gov_ftp_1 = None
        _ctr_ftp_2 = None
        _gov_ftp_2 = None

        # Determining variables if the type of Contract1 is Cost Recovery
        if isinstance(self.contract1, CostRecovery):
            _oil_deductible_cost_1 = self._contract1_transitioned._oil_cost_recovery_after_tf
            _gas_deductible_cost_1 = self._contract1_transitioned._gas_cost_recovery_after_tf
            _oil_unrec_cost_1 = self._contract1_transitioned._oil_unrecovered_after_transfer
            _gas_unrec_cost_1 = self._contract1_transitioned._gas_unrecovered_after_transfer
            _oil_ctr_ets_1 = self._contract1_transitioned._oil_ets_after_transfer
            _gas_ctr_ets_1 = self._contract1_transitioned._gas_ets_after_transfer
            _oil_gov_ets_1 = self._contract1_transitioned._oil_government_share
            _gas_gov_ets_1 = self._contract1_transitioned._gas_government_share
            _ctr_ftp_1 = self._contract1_transitioned._consolidated_ftp_ctr
            _gov_ftp_1 = self._contract1_transitioned._consolidated_ftp_gov

        # Determining variables if the type of Contract1 is Gross Split
        elif isinstance(self.contract1, GrossSplit):
            _oil_deductible_cost_1 = self._contract1_transitioned._oil_deductible_cost
            _gas_deductible_cost_1 = self._contract1_transitioned._gas_deductible_cost
            _oil_unrec_cost_1 = self._contract1_transitioned._oil_carward_deduct_cost
            _gas_unrec_cost_1 = self._contract1_transitioned._gas_carward_deduct_cost
            _oil_ctr_ets_1 = self._contract1_transitioned._oil_ctr_share_after_transfer
            _gas_ctr_ets_1 = self._contract1_transitioned._gas_ctr_share_after_transfer
            _oil_gov_ets_1 = np.zeros_like(self._contract1_transitioned.project_years)
            _gas_gov_ets_1 = np.zeros_like(self._contract1_transitioned.project_years)
            _ctr_ftp_1 = np.zeros_like(self._contract1_transitioned.project_years, dtype=float)
            _gov_ftp_1 = self._contract1_transitioned._consolidated_gov_share_before_tf

        # Determining variables if the type of Contract2 is Cost Recovery
        if isinstance(self.contract2, CostRecovery):
            _oil_deductible_cost_2 = self._contract2_transitioned._oil_cost_recovery_after_tf
            _gas_deductible_cost_2 = self._contract2_transitioned._gas_cost_recovery_after_tf
            _oil_unrec_cost_2 = self._contract2_transitioned._oil_unrecovered_after_transfer
            _gas_unrec_cost_2 = self._contract2_transitioned._gas_unrecovered_after_transfer
            _oil_ctr_ets_2 = self._contract2_transitioned._oil_ets_after_transfer
            _gas_ctr_ets_2 = self._contract2_transitioned._gas_ets_after_transfer
            _oil_gov_ets_2 = self._contract2_transitioned._oil_government_share
            _gas_gov_ets_2 = self._contract2_transitioned._gas_government_share
            _ctr_ftp_2 = self._contract2_transitioned._consolidated_ftp_ctr
            _gov_ftp_2 = self._contract2_transitioned._consolidated_ftp_gov

        # Determining variables if the type of Contract2 is Gross Split
        elif isinstance(self.contract2, GrossSplit):
            _oil_deductible_cost_2 = self._contract2_transitioned._oil_deductible_cost
            _gas_deductible_cost_2 = self._contract2_transitioned._gas_deductible_cost
            _oil_unrec_cost_2 = self._contract2_transitioned._oil_carward_deduct_cost
            _gas_unrec_cost_2 = self._contract2_transitioned._gas_carward_deduct_cost
            _oil_ctr_ets_2 = self._contract2_transitioned._oil_ctr_share_after_transfer
            _gas_ctr_ets_2 = self._contract2_transitioned._gas_ctr_share_after_transfer
            _oil_gov_ets_2 = np.zeros_like(self._contract2_transitioned.project_years)
            _gas_gov_ets_2 = np.zeros_like(self._contract2_transitioned.project_years)
            _ctr_ftp_2 = np.zeros_like(self._contract2_transitioned.project_years, dtype=float)
            _gov_ftp_2 = self._contract2_transitioned._consolidated_gov_share_before_tf

        # Deductible Cost / Cost Recovery
        self._oil_deductible_cost = _oil_deductible_cost_1 + _oil_deductible_cost_2
        self._gas_deductible_cost = _gas_deductible_cost_1 + _gas_deductible_cost_2

        # Carry Forward Cost / Unrecoverable Cost
        self._oil_unrec_cost = _oil_unrec_cost_2
        self._gas_unrec_cost = _gas_unrec_cost_2

        # ETS
        self._oil_ctr_ets = _oil_ctr_ets_1 + _oil_ctr_ets_2
        self._gas_ctr_ets = _gas_ctr_ets_1 + _gas_ctr_ets_2

        self._oil_gov_ets = _oil_gov_ets_1 + _oil_gov_ets_2
        self._gas_gov_ets = _gas_gov_ets_1 + _gas_gov_ets_2

        # Contractor Net Operating Profit
        self._net_operating_profit = (self._contract1_transitioned._consolidated_ctr_net_share +
                                      self._contract2_transitioned._consolidated_ctr_net_share)

        # Contractor Net Cash Flow
        self._ctr_net_cashflow = (self._contract1_transitioned._consolidated_cashflow +
                                  self._contract2_transitioned._consolidated_cashflow)

        # FTP
        self._ctr_ftp = _ctr_ftp_1 + _ctr_ftp_2
        self._gov_ftp = _gov_ftp_1 + _gov_ftp_2

        # DDMO
        self._ddmo = (self._contract1_transitioned._consolidated_ddmo +
                      self._contract2_transitioned._consolidated_ddmo)

        # Tax
        self._tax_payment = (self._contract1_transitioned._consolidated_tax_payment +
                             self._contract2_transitioned._consolidated_tax_payment)

        # Government Take
        self._government_take = (self._contract1_transitioned._consolidated_government_take +
                                 self._contract2_transitioned._consolidated_government_take)

        # Consolidated Attributes
        # Consolidated Revenue
        self._consolidated_revenue = self._oil_revenue + self._gas_revenue

        # Consolidated Sunk Cost
        self._consolidated_sunk_cost = self._oil_sunk_cost + self._gas_sunk_cost

        # Consolidated Cash Flow
        self._consolidated_cashflow = (self._contract1_transitioned._consolidated_cashflow +
                                       self._contract2_transitioned._consolidated_cashflow)

        # Consolidated DDMO
        self._consolidated_ddmo = (self._contract1_transitioned._consolidated_ddmo +
                                   self._contract2_transitioned._consolidated_ddmo)

        # Consolidated Tax Payment
        self._consolidated_tax_payment = (self._contract1_transitioned._consolidated_tax_payment +
                                          self._contract2_transitioned._consolidated_tax_payment)

        # Consolidated Government Take
        self._consolidated_government_take = (self._contract1_transitioned._consolidated_government_take +
                                              self._contract2_transitioned._consolidated_government_take)

        # Project Years
        self.project_years = np.copy(self._contract1_transitioned.project_years)
