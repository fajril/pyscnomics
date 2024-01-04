"""
Configuration to undertake sensitivity analysis.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, InitVar

from pyscnomics.io.aggregator import Aggregate


def get_multipliers(
    min_deviation: float,
    max_deviation: float,
    base_value: float = 1.0,
    step: int = 2,
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

        for i, val in enumerate(lifting_aggregate):
            # Adjust price
            lifting_aggregate[i].price = (
                lifting_aggregate_attrs["price"][i] * price_multiplier
            ).copy()

            # Adjust lifting rate
            lifting_aggregate[i].lifting_rate = (
                lifting_aggregate_attrs["rate"][i] * rate_multiplier
            ).copy()

    # For transition contract
    else:
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

        for psc in ["PSC 1", "PSC 2"]:
            for i, val in enumerate(lifting_aggregate[psc]):
                # Adjust price
                lifting_aggregate[psc][i].price = (
                    lifting_aggregate_attrs["price"][psc][i] * price_multiplier
                ).copy()

                # Adjust lifting rate
                lifting_aggregate[psc][i].lifting_rate = (
                    lifting_aggregate_attrs["rate"][psc][i] * rate_multiplier
                ).copy()

    return lifting_aggregate


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
    contract_type: str
    oil_lifting_aggr_tot_init: InitVar[tuple | dict] = field(repr=False)
    gas_lifting_aggr_tot_init: InitVar[tuple | dict] = field(repr=False)
    # sulfur_lifting_aggregate: dict | tuple = field(repr=False)
    # electricity_lifting_aggregate: dict | tuple = field(repr=False)
    # co2_lifting_aggregate: dict | tuple = field(repr=False)
    # opex_aggregate: dict | tuple = field(repr=False)
    # tangible_cost_aggregate: dict | tuple = field(repr=False)
    # intangible_cost_aggregate: dict | tuple = field(repr=False)
    multipliers: np.ndarray

    # Attributes to be defined later
    oil_lifting_aggregate_total: tuple | dict = field(init=False, repr=False)
    gas_lifting_aggregate_total: tuple | dict = field(init=False, repr=False)

    sensitivity_data: dict = field(init=False)

    def __post_init__(
        self,
        oil_lifting_aggr_tot_init: tuple | dict,
        gas_lifting_aggr_tot_init: tuple | dict,
    ):
        # Adjust oil and condensate lifting data
        self.oil_lifting_aggregate_total = (
            get_price_and_rate_adjustment(
                contract_type=self.contract_type,
                lifting_aggregate=oil_lifting_aggr_tot_init,
                price_multiplier=self.multipliers[0],
                rate_multiplier=self.multipliers[4],
            )
        )

        # Adjust gas, lpg propane, and lpg_butane lifting data
        self.gas_lifting_aggregate_total = (
            get_price_and_rate_adjustment(
                contract_type=self.contract_type,
                lifting_aggregate=gas_lifting_aggr_tot_init,
                price_multiplier=self.multipliers[1],
                rate_multiplier=self.multipliers[4],
            )
        )

        print('\t')
        print(f'Filetype: {type(self.oil_lifting_aggregate_total)}')
        print(f'Length: {len(self.oil_lifting_aggregate_total)}')
        print('self.oil_lifting_aggregate_total = \n', self.oil_lifting_aggregate_total)

        print('\t')
        print(f'Filetype: {type(self.gas_lifting_aggregate_total)}')
        print(f'Length: {len(self.gas_lifting_aggregate_total)}')
        print('self.gas_lifting_aggregate_total = \n', self.gas_lifting_aggregate_total)

        # # Adjust oil price
        # oil_price_adjusted = get_oil_price_adjustment(
        #     contract_type=self.contract_type,
        #     oil_lifting_aggregate_total=self.oil_lifting_aggregate_total,
        #     oil_price_multiplier=self.multipliers[0],
        # )
        #
        # # Adjust gas price
        # gas_price_adjusted = get_gas_price_adjustment(
        #     contract_type=self.contract_type,
        #     gas_lifting_aggregate_total=self.gas_lifting_aggregate_total,
        #     gas_price_multiplier=self.multipliers[1],
        # )
        #
        # # Adjust opex
        # opex_adjusted = get_opex_adjustment(
        #     contract_type=self.contract_type,
        #     opex_aggregate=self.opex_aggregate,
        #     opex_multiplier=self.multipliers[2],
        # )
        #
        # # Adjust capex
        # capex_aggregates = [self.tangible_cost_aggregate, self.intangible_cost_aggregate]
        # capex_adjusted = {
        #     key: get_capex_adjustment(
        #         contract_type=self.contract_type,
        #         capex_aggregate=capex_aggregates[i],
        #         capex_multiplier=self.multipliers[3],
        #     )
        #     for i, key in enumerate(["tangible", "intangible"])
        # }
        #
        # # Adjust lifting rate
        # cum_prod_aggregates = [
        #     self.oil_lifting_aggregate_total,
        #     self.gas_lifting_aggregate_total,
        #     self.sulfur_lifting_aggregate,
        #     self.electricity_lifting_aggregate,
        #     self.co2_lifting_aggregate,
        # ]
        # cum_prod_adjusted = {
        #     key: get_lifting_adjustment(
        #         contract_type=self.contract_type,
        #         lifting_aggregate=cum_prod_aggregates[i],
        #         lifting_multiplier=self.multipliers[4],
        #     )
        #     for i, key in enumerate(
        #         [
        #             "oil_total",
        #             "gas_total",
        #             "sulfur",
        #             "electricity",
        #             "co2",
        #         ]
        #     )
        # }
        #
        # # Prepare sensitivity_data
        # self.sensitivity_data = {
        #     "oil_price": oil_price_adjusted,
        #     "gas_price": gas_price_adjusted,
        #     "opex": opex_adjusted,
        #     "capex": capex_adjusted,
        #     "cum_prod": cum_prod_adjusted,
        # }

# def get_sensitivity_data(
#     contract_type: str,
#     oil_lifting_aggregate_total: dict | tuple,
#     gas_lifting_aggregate_total: dict | tuple,
#     sulfur_lifting_aggregate: dict | tuple,
#     electricity_lifting_aggregate: dict | tuple,
#     co2_lifting_aggregate: dict | tuple,
#     opex_aggregate: dict | tuple,
#     tangible_cost_aggregate: dict | tuple,
#     intangible_cost_aggregate: dict | tuple,
#     multipliers: np.ndarray,
# ):
#     # Adjust oil price
#     oil_price_adjusted = get_oil_price_adjustment(
#         contract_type=contract_type,
#         oil_lifting_aggregate_total=oil_lifting_aggregate_total,
#         oil_price_multiplier=multipliers[0],
#     )
#
#     # Adjust gas price
#     gas_price_adjusted = get_gas_price_adjustment(
#         contract_type=contract_type,
#         gas_lifting_aggregate_total=gas_lifting_aggregate_total,
#         gas_price_multiplier=multipliers[1],
#     )
#
#     # Adjust opex
#     opex_adjusted = get_opex_adjustment(
#         contract_type=contract_type,
#         opex_aggregate=opex_aggregate,
#         opex_multiplier=multipliers[2],
#     )
#
#     # Adjust capex
#     capex_aggregates = [tangible_cost_aggregate, intangible_cost_aggregate]
#     capex_adjusted = {
#         key: get_capex_adjustment(
#             contract_type=contract_type,
#             capex_aggregate=capex_aggregates[i],
#             capex_multiplier=multipliers[3],
#         )
#         for i, key in enumerate(["tangible", "intangible"])
#     }
#
#     # Adjust lifting rate
#     cum_prod_aggregates = [
#         oil_lifting_aggregate_total,
#         gas_lifting_aggregate_total,
#         sulfur_lifting_aggregate,
#         electricity_lifting_aggregate,
#         co2_lifting_aggregate,
#     ]
#     cum_prod_adjusted = {
#         key: get_lifting_adjustment(
#             contract_type=contract_type,
#             lifting_aggregate=cum_prod_aggregates[i],
#             lifting_multiplier=multipliers[4],
#         )
#         for i, key in enumerate(
#             [
#                 "oil_total",
#                 "gas_total",
#                 "sulfur",
#                 "electricity",
#                 "co2",
#             ]
#         )
#     }
#
#     # Container to store the adjusted data
#     sensitivity_data = {
#         "oil_price": oil_price_adjusted,
#         "gas_price": gas_price_adjusted,
#         "opex": opex_adjusted,
#         "capex": capex_adjusted,
#         "cum_prod": cum_prod_adjusted,
#     }


