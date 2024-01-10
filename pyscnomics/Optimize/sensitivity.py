"""
Configuration to undertake sensitivity analysis.
"""
import numpy as np
from dataclasses import dataclass, field, InitVar

from pyscnomics.io.aggregator import Aggregate
from pyscnomics.econ.revenue import Lifting

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit


def get_multipliers(
    min_deviation: float,
    max_deviation: float,
    base_value: float = 1.0,
    step: int = 10,
    number_of_params: int = 5,
) -> np.ndarray:
    """
    Generate multipliers for different economic parameters within a specified range.

    Parameters
    ----------
    min_deviation: float
        The minimum deviation from the base value.
    max_deviation: float
        The maximum deviation from the base value.
    base_value: float, optional
        The base value for the multipliers. Default is 1.0.
    step: int, optional
        The number of steps to create multipliers. Default is 10.
    number_of_params: int, optional
        The number of parameters to vary in sensitivity analysis. Default is 5.

    Returns
    -------
    multipliers: np.ndarray
        A 3D NumPy array containing multipliers for different economic factors.
    """
    # Specify the minimum and maximum values
    min_val = base_value - min_deviation
    max_val = base_value + max_deviation

    min_multipliers = np.linspace(min_val, base_value, step + 1)
    max_multipliers = np.linspace(base_value, max_val, step + 1)
    tot_multipliers = np.concatenate((min_multipliers, max_multipliers[1:]))

    # Specify array multipliers
    multipliers = (
        np.ones(
            [number_of_params, len(tot_multipliers), number_of_params],
            dtype=np.float_,
        )
    )

    for i in range(number_of_params):
        multipliers[i, :, i] = tot_multipliers.copy()

    # Specify total number of run
    total_run = len(tot_multipliers) * number_of_params

    return multipliers, total_run


def get_price_and_rate_adjustment(
    contract_type: str,
    lifting_aggregate: tuple | dict,
    price_multiplier: float,
    rate_multiplier: float,
) -> tuple | dict:
    """
    Adjust price and lifting rate for sensitivity analysis.

    Parameters
    ----------
    contract_type: str
        The type of contract.
    lifting_aggregate: tuple or dict
        The aggregate of lifting data for a particular fluid type.
    price_multiplier: float
        Multiplier to adjust prices.
    rate_multiplier: float
        Multiplier to adjust lifting rates.

    Returns
    -------
    tuple or dict
        The adjusted lifting aggregate with updated price and lifting rate.
    """
    # For single contract
    if "Transition" not in contract_type:
        # Extract price and lifting_rate data from the original lifting_aggregate data
        lifting_aggregate_attrs = {
            "price": (
                [
                    lifting_aggregate[i].price
                    for i, val in enumerate(lifting_aggregate)
                ]
            ),
            "rate": (
                [
                    lifting_aggregate[i].lifting_rate
                    for i, val in enumerate(lifting_aggregate)
                ]
            ),
        }

        # Perform modification to price and lifting_rate data
        updated_attrs = {
            "price": [pr * price_multiplier for pr in lifting_aggregate_attrs["price"]],
            "rate": [rt * rate_multiplier for rt in lifting_aggregate_attrs["rate"]],
        }

        # Return new tuple of modified lifting_aggregate data
        return tuple(
            [
                Lifting(
                    start_year=lifting_aggregate[i].start_year,
                    end_year=lifting_aggregate[i].end_year,
                    lifting_rate=updated_attrs["rate"][i],
                    price=updated_attrs["price"][i],
                    prod_year=lifting_aggregate[i].prod_year,
                    fluid_type=lifting_aggregate[i].fluid_type,
                    ghv=lifting_aggregate[i].ghv,
                    prod_rate=lifting_aggregate[i].prod_rate,
                )
                for i, val in enumerate(lifting_aggregate)
            ]
        )

    # For transition contract
    else:
        # Extract price and lifting_rate data from the original lifting_aggregate data
        lifting_aggregate_attrs = {
            "price": {
                psc: (
                    [
                        lifting_aggregate[psc][i].price
                        for i, val in enumerate(lifting_aggregate[psc])
                    ]
                )
                for psc in ["PSC 1", "PSC 2"]
            },
            "rate": {
                psc: (
                    [
                        lifting_aggregate[psc][i].lifting_rate
                        for i, val in enumerate(lifting_aggregate[psc])
                    ]
                )
                for psc in ["PSC 1", "PSC 2"]
            },
        }

        # Perform modification to price and lifting_rate data
        updated_attrs = {
            "price": {
                psc: [pr * price_multiplier for pr in lifting_aggregate_attrs["price"][psc]]
                for psc in ["PSC 1", "PSC 2"]
            },
            "rate": {
                psc: [rt * rate_multiplier for rt in lifting_aggregate_attrs["rate"][psc]]
                for psc in ["PSC 1", "PSC 2"]
            },
        }

        # Return new dictionary of modified lifting_aggregate data
        return {
            psc: tuple(
                [
                    Lifting(
                        start_year=lifting_aggregate[psc][i].start_year,
                        end_year=lifting_aggregate[psc][i].end_year,
                        lifting_rate=updated_attrs["rate"][psc][i],
                        price=updated_attrs["price"][psc][i],
                        prod_year=lifting_aggregate[psc][i].prod_year,
                        fluid_type=lifting_aggregate[psc][i].fluid_type,
                        ghv=lifting_aggregate[psc][i].ghv,
                        prod_rate=lifting_aggregate[psc][i].prod_rate,
                    )
                    for i, val in enumerate(lifting_aggregate)
                ]
            )
            for psc in ["PSC 1", "PSC 2"]
        }


def get_rate_adjustment(
    contract_type: str,
    lifting_aggregate: tuple | dict,
    rate_multiplier: float,
) -> tuple | dict:
    """
    Adjust lifting rate for sensitivity analysis.

    Parameters
    ----------
    contract_type: str
        The type of contract.
    lifting_aggregate: tuple or dict
        The aggregate of lifting for a particular fluid type.
    rate_multiplier: float
        A scalar multiplier to adjust lifting rate.

    Returns
    -------
    tuple or dict
        The adjusted lifting aggregate with updated lifting rate.
    """
    # For single contract
    if "Transition" not in contract_type:
        lifting_aggregate_rate = (
            [
                lifting_aggregate[i].lifting_rate
                for i, val in enumerate(lifting_aggregate)
            ]
        )

        for i, val in enumerate(lifting_aggregate):
            lifting_aggregate[i].lifting_rate = (
                lifting_aggregate_rate[i] * rate_multiplier
            ).copy()

    # For transition contract
    else:
        lifting_aggregate_rate = {
            psc: (
                [
                    lifting_aggregate[psc][i].lifting_rate
                    for i, val in enumerate(lifting_aggregate[psc])
                ]
            )
            for psc in ["PSC 1", "PSC 2"]
        }

        for psc in ["PSC 1", "PSC 2"]:
            for i, val in enumerate(lifting_aggregate[psc]):
                lifting_aggregate[psc][i].lifting_rate = (
                    lifting_aggregate_rate[psc][i] * rate_multiplier
                ).copy()

    return lifting_aggregate


def get_cost_adjustment(
    contract_type: str,
    cost_aggregate: tuple | dict,
    cost_multiplier: float,
) -> tuple | dict:
    """
    Adjust costs for sensitivity analysis.

    Parameters
    ----------
    contract_type: str
        The type of contract.
    cost_aggregate: tuple or dict
        The aggregate of costs for a particular cost allocation.
    cost_multiplier: float
        A scalar multiplier to adjust costs.

    Returns
    -------
    tuple or dict
        The adjusted cost aggregate with updated cost.
    """
    # For single contract
    if "Transition" not in contract_type:
        cost_aggregate_attr = (
            [
                cost_aggregate[i].cost
                for i, val in enumerate(cost_aggregate)
            ]
        )

        for i, val in enumerate(cost_aggregate):
            cost_aggregate[i].cost = (cost_aggregate_attr[i] * cost_multiplier).copy()

    # For transition contract
    else:
        cost_aggregate_attr = {
            psc: (
                [
                    cost_aggregate[psc][i].cost
                    for i, val in enumerate(cost_aggregate[psc])
                ]
            )
            for psc in ["PSC 1", "PSC 2"]
        }

        for psc in ["PSC 1", "PSC 2"]:
            for i, val in enumerate(cost_aggregate[psc]):
                cost_aggregate[psc][i].cost = (cost_aggregate_attr[psc][i] * cost_multiplier).copy()

    return cost_aggregate


class SensitivityException(Exception):
    """ Exception to be raised for a misuse of SensitivityData class """

    pass


@dataclass
class SensitivityData:
    """ 123 """
    # Parameters
    multipliers: np.ndarray
    workbook_path: str = field(default=None)
    data: Aggregate = field(default=None, repr=False)

    # Attributes to be defined later
    contract_type: str = field(default=None, init=False, repr=False)
    summary_arguments: dict = field(default=None, init=False, repr=False)
    sensitivity_data: dict = field(default=None, init=False)
    psc: CostRecovery | GrossSplit | dict = field(default=None, init=False, repr=False)
    psc_arguments: dict = field(default=None, init=False, repr=False)

    def __post_init__(self):

        # Prepare attribute workbook_path
        if self.workbook_path is None:
            self.workbook_path = "Workbook.xlsb"

        # Prepare attribute summary_arguments
        self.summary_arguments = {
            "reference_year": self.data.general_config_data.discount_rate_start_year,
            "inflation_rate": self.data.fiscal_config_data.inflation_rate,
            "discount_rate": self.data.general_config_data.discount_rate,
            "npv_mode": self.data.fiscal_config_data.npv_mode,
            "discounting_mode": self.data.fiscal_config_data.discounting_mode,
        }

        # Prepare attribute contract_type
        self.contract_type = self.data.general_config_data.type_of_contract

        # Prepare parameter sensitivity data
        self.sensitivity_data = {
            # Adjust oil and condensate lifting data
            "oil_lifting_aggregate_total": (
                get_price_and_rate_adjustment(
                    contract_type=self.contract_type,
                    lifting_aggregate=self.data.oil_lifting_aggregate_total,
                    price_multiplier=self.multipliers[0],
                    rate_multiplier=self.multipliers[4],
                )
            ),
            # # Adjust gas, lpg proprane, and lpg butane lifting data
            # "gas_lifting_aggregate_total": (
            #     get_price_and_rate_adjustment(
            #         contract_type=self.contract_type,
            #         lifting_aggregate=self.data.gas_lifting_aggregate_total,
            #         price_multiplier=self.multipliers[1],
            #         rate_multiplier=self.multipliers[4],
            #     )
            # ),
            # # Adjust sulfur lifting data
            # "sulfur_lifting_aggregate": (
            #     get_rate_adjustment(
            #         contract_type=self.contract_type,
            #         lifting_aggregate=self.data.sulfur_lifting_aggregate,
            #         rate_multiplier=self.multipliers[4],
            #     )
            # ),
            # # Adjust electricity lifting data
            # "electricity_lifting_aggregate": (
            #     get_rate_adjustment(
            #         contract_type=self.contract_type,
            #         lifting_aggregate=self.data.electricity_lifting_aggregate,
            #         rate_multiplier=self.multipliers[4],
            #     )
            # ),
            # # Adjust CO2 lifting data
            # "co2_lifting_aggregate": (
            #     get_rate_adjustment(
            #         contract_type=self.contract_type,
            #         lifting_aggregate=self.data.co2_lifting_aggregate,
            #         rate_multiplier=self.multipliers[4],
            #     )
            # ),
            # # Adjust OPEX data
            # "opex_aggregate": (
            #     get_cost_adjustment(
            #         contract_type=self.contract_type,
            #         cost_aggregate=self.data.opex_aggregate,
            #         cost_multiplier=self.multipliers[2],
            #     )
            # ),
            # # Adjust CAPEX data: tangible
            # "tangible_cost_aggregate": (
            #     get_cost_adjustment(
            #         contract_type=self.contract_type,
            #         cost_aggregate=self.data.tangible_cost_aggregate,
            #         cost_multiplier=self.multipliers[3],
            #     )
            # ),
            # # Adjust CAPEX data: intangible
            # "intangible_cost_aggregate": (
            #     get_cost_adjustment(
            #         contract_type=self.contract_type,
            #         cost_aggregate=self.data.intangible_cost_aggregate,
            #         cost_multiplier=self.multipliers[3],
            #     )
            # )
        }

        print('\t')
        print(f'Filetype: {type(self.sensitivity_data)}')
        print('sensitivity_data = \n', self.sensitivity_data)

    def _get_single_contract_project(self):
        pass

    def _get_single_contract_cr(self):
        # Prepare total lifting data
        lifting_total = (
            self.sensitivity_data["oil_lifting_aggregate_total"]
            + self.sensitivity_data["gas_lifting_aggregate_total"]
            + self.sensitivity_data["sulfur_lifting_aggregate"]
            + self.sensitivity_data["electricity_lifting_aggregate"]
            + self.sensitivity_data["co2_lifting_aggregate"]
        )

        # Create an instance of PSC CR
        self.psc = CostRecovery(
            start_date=self.data.general_config_data.start_date_project,
            end_date=self.data.general_config_data.end_date_project,
            oil_onstream_date=self.data.general_config_data.oil_onstream_date,
            gas_onstream_date=self.data.general_config_data.gas_onstream_date,
            lifting=lifting_total,
            tangible_cost=self.sensitivity_data["tangible_cost_aggregate"],
            intangible_cost=self.sensitivity_data["intangible_cost_aggregate"],
            opex=self.sensitivity_data["opex_aggregate"],
            asr_cost=self.data.asr_cost_aggregate,
            oil_ftp_is_available=self.data.psc_cr_data.oil_ftp_availability,
            oil_ftp_is_shared=self.data.psc_cr_data.oil_ftp_is_shared,
            oil_ftp_portion=self.data.psc_cr_data.oil_ftp_portion,
            gas_ftp_is_available=self.data.psc_cr_data.gas_ftp_availability,
            gas_ftp_is_shared=self.data.psc_cr_data.gas_ftp_is_shared,
            gas_ftp_portion=self.data.psc_cr_data.gas_ftp_portion,
            tax_split_type=self.data.psc_cr_data.split_type,
            condition_dict=self.data.psc_cr_data.icp_sliding_scale,
            indicator_rc_icp_sliding=self.data.psc_cr_data.indicator_rc_split_sliding_scale,
            oil_ctr_pretax_share=self.data.psc_cr_data.oil_ctr_pretax,
            gas_ctr_pretax_share=self.data.psc_cr_data.gas_ctr_pretax,
            oil_ic_rate=self.data.psc_cr_data.ic_oil,
            gas_ic_rate=self.data.psc_cr_data.ic_gas,
            ic_is_available=self.data.psc_cr_data.ic_availability,
            oil_cr_cap_rate=self.data.psc_cr_data.oil_cr_cap_rate,
            gas_cr_cap_rate=self.data.psc_cr_data.gas_cr_cap_rate,
            oil_dmo_volume_portion=self.data.psc_cr_data.oil_dmo_volume,
            oil_dmo_fee_portion=self.data.psc_cr_data.oil_dmo_fee,
            oil_dmo_holiday_duration=self.data.psc_cr_data.oil_dmo_period,
            gas_dmo_volume_portion=self.data.psc_cr_data.gas_dmo_volume,
            gas_dmo_fee_portion=self.data.psc_cr_data.gas_dmo_fee,
            gas_dmo_holiday_duration=self.data.psc_cr_data.gas_dmo_period,
        )

        # Specify arguments for PSC CR
        self.psc_arguments = {
            "sulfur_revenue": self.data.fiscal_config_data.sulfur_revenue_config,
            "electricity_revenue": self.data.fiscal_config_data.electricity_revenue_config,
            "co2_revenue": self.data.fiscal_config_data.co2_revenue_config,
            "is_dmo_end_weighted": self.data.psc_cr_data.dmo_is_weighted,
            "tax_regime": self.data.fiscal_config_data.tax_mode,
            "tax_rate": self.data.fiscal_config_data.tax_rate,
            "ftp_tax_regime": self.data.fiscal_config_data.tax_payment_config,
            "sunk_cost_reference_year": self.data.fiscal_config_data.sunk_cost_reference_year,
            "depr_method": self.data.fiscal_config_data.depreciation_method,
            "decline_factor": self.data.fiscal_config_data.decline_factor,
            "vat_rate": self.data.fiscal_config_data.vat_rate,
            "lbt_rate": self.data.fiscal_config_data.lbt_rate,
            "inflation_rate": self.data.fiscal_config_data.inflation_rate,
            "future_rate": float(self.data.fiscal_config_data.asr_future_rate),
            "inflation_rate_applied_to": self.data.general_config_data.inflation_rate_applied_to,
            # "year_ref": None,
            # "tax_type": None,
        }

        # Fill the summary contract argument
        self.summary_arguments["contract"] = self.psc

        return self.psc, self.psc_arguments, self.summary_arguments

    def _get_single_contract_gs(self):
        pass

    def activate(self):
        """
        123
        """
        if self.contract_type == "Project":
            return self._get_single_contract_project()

        elif self.contract_type == "PSC Cost Recovery (CR)":
            return self._get_single_contract_cr()

        elif self.contract_type == "PSC Gross Split (GS)":
            return self._get_single_contract_gs()
