"""
Collection of functions to administer Monte Carlo simulation.
"""
import copy
import numpy as np
from scipy.stats import uniform, triang, truncnorm
import dask.bag as db
from dask.distributed import Client
from pyscnomics.api.adapter import (
            get_baseproject,
            get_costrecovery,
            get_grosssplit,
            get_transition,
        )
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.econ import FluidType
from pyscnomics.econ.selection import UncertaintyDistribution
from pyscnomics.io.getattr import get_contract_attributes

# import pandas as pd



class MonteCarloException(Exception):
    """ Exception to be raised for a misuse of MonteCarlo class """

    pass

################################################ Uncertainty Detached ################################################

def get_multipliers_montecarlo(
    run_number: int,
    distribution: str,
    min_value: float,
    mean_value: float,
    max_value: float,
    std_dev: float,
) -> np.ndarray:
    """
    Generate an array of multipliers for Monte Carlo simulation based on the specified distribution.

    Parameters
    ----------
    run_number: int
        Number of runs.
    distribution: str
        Type of distribution ("Uniform", "Triangular", or "Normal").
    min_value: float
        Minimum value for the distribution.
    mean_value: float
        Mean (or central value) for the distribution.
    max_value: float
        Maximum value for the distribution.
    std_dev: float
        Standard deviation for the normal distribution.

    Returns
    -------
    multipliers: np.ndarray
        Array of multipliers generated using Monte Carlo simulation.

    Notes
    -----
    - For "Uniform" distribution, the function uses a uniform random variable.
    - For "Triangular" distribution, the function uses a triangular random variable.
    - For "Normal" distribution, the function uses a truncated normal random variable.
    """
    # Uniform distribution
    if distribution == "Uniform":
        # Modify minimum and maximum values
        params = [min_value, max_value]
        attrs_updated = {
            key: float(params[i] / mean_value)
            for i, key in enumerate(["min_value", "max_value"])
        }

        # Determine multipliers
        multipliers = uniform.rvs(
            loc=attrs_updated["min_value"],
            scale=attrs_updated["max_value"] - attrs_updated["min_value"],
            size=run_number,
        )

    # Triangular distribution
    elif distribution == "Triangular":
        # Modify minimum and maximum values
        params = [min_value, mean_value, max_value]
        attrs_updated = {
            key: float(params[i] / mean_value)
            for i, key in enumerate(["min_value", "mean_value", "max_value"])
        }

        # Determine mode (central point)
        c = (
            (attrs_updated["mean_value"] - attrs_updated["min_value"]) /
            (attrs_updated["max_value"] - attrs_updated["min_value"])
        )

        # Determine multipliers
        multipliers = triang.rvs(
            c=c,
            loc=attrs_updated["min_value"],
            scale=attrs_updated["max_value"] - attrs_updated["min_value"],
            size=run_number,
        )

    # Normal distribution
    elif distribution == "Normal":
        # Modify minimum and maximum values
        params = [min_value, mean_value, max_value, std_dev]
        attrs_updated = {
            key: float(params[i] / mean_value)
            for i, key in enumerate(["min_value", "mean_value", "max_value", "std_dev"])
        }

        # Determine z-values
        zvalues = {
            key: float(
                (attrs_updated[key] - attrs_updated["mean_value"]) / attrs_updated["std_dev"]
            )
            for key in ["min_value", "mean_value", "max_value"]
        }

        # Determine multipliers
        multipliers_init = truncnorm.rvs(
            a=zvalues["min_value"],
            b=zvalues["max_value"],
            loc=zvalues["mean_value"],
            scale=1,
            size=run_number,
        )

        multipliers = (multipliers_init * attrs_updated["std_dev"]) + attrs_updated["mean_value"]

    else:
        raise MonteCarloException(
            f"The type of distribution is unavailable. "
            f"Please select type of distribution between: "
            f"(i) Uniform, (ii) Triangular, or (iii) Normal."
        )

    return multipliers

def min_mean_max_retriever(
        contract: BaseProject | CostRecovery | GrossSplit | Transition,
        verbose: bool = False
):
    """
    Function to get the value of min, max and stddev of each montecarlo elements.

    Parameters
    ----------
    contract : BaseProject | CostRecovery | GrossSplit | Transition
        The observed contract object
    verbose: bool
        The option to print the min, mean and max.

    Returns
    -------
    out: dict
    """
    # Retrieve Min, Mean, and Max for CAPEX
    min_capex = np.min(np.concatenate(
        (contract.capital_cost_total.cost, contract.intangible_cost_total.cost)))
    mean_capex = np.average(np.concatenate(
        (contract.capital_cost_total.cost, contract.intangible_cost_total.cost)))
    max_capex = np.max(np.concatenate(
        (contract.capital_cost_total.cost, contract.intangible_cost_total.cost)))

    # Retrieve Min, Mean, and Max for OPEX
    min_opex = np.min(np.concatenate(
        (contract.opex_total.cost, contract.asr_cost_total.cost,
         contract.lbt_cost_total.cost, contract.cost_of_sales_total.cost)))
    mean_opex = np.average(np.concatenate(
        (contract.opex_total.cost, contract.asr_cost_total.cost,
         contract.lbt_cost_total.cost, contract.cost_of_sales_total.cost)))
    max_opex = np.max(np.concatenate(
        (contract.opex_total.cost, contract.asr_cost_total.cost,
         contract.lbt_cost_total.cost, contract.cost_of_sales_total.cost)))

    # Retrieve Min, Mean, and Max for Oil Price
    min_oil_price = np.min(contract._oil_lifting.price)
    mean_oil_price = np.average(contract._oil_lifting.price)
    max_oil_price = np.max(contract._oil_lifting.price)

    # Retrieve Min, Mean, Max for Gas Price
    min_gas_price = np.min(contract._gas_lifting.price)
    mean_gas_price = np.average(contract._gas_lifting.price)
    max_gas_price = np.max(contract._gas_lifting.price)

    # Retrieve Min, Mean, Max for Oil Lifting
    min_lifting = np.min([
        np.min(contract._oil_lifting.lifting_rate),
        np.min(contract._gas_lifting.lifting_rate)
    ])
    mean_lifting = np.mean([
        np.mean(contract._oil_lifting.lifting_rate),
        np.mean(contract._gas_lifting.lifting_rate)
    ])
    max_lifting = np.max([
        np.max(contract._oil_lifting.lifting_rate),
        np.max(contract._gas_lifting.lifting_rate)
    ])

    result =  {
        'min_capex': min_capex,
        'mean_capex': mean_capex,
        'max_capex': max_capex,
        'min_opex': min_opex,
        'mean_opex': mean_opex,
        'max_opex': max_opex,
        'min_oil_price': min_oil_price,
        'mean_oil_price': mean_oil_price,
        'max_oil_price': max_oil_price,
        'min_gas_price': min_gas_price,
        'mean_gas_price': mean_gas_price,
        'max_gas_price': max_gas_price,
        'min_lifting': min_lifting,
        'mean_lifting': mean_lifting,
        'max_lifting': max_lifting,
    }

    if verbose is True:
        print('Parameter used:')
        for key, value in result.items():
            print(key,': ', value)
        print('')

    return result



class ProcessMonte:
    target = ["npv", "irr", "pi", "pot", "gov_take", "ctr_net_share"]

    def __init__(self, type, contract, numSim, params):
        self.type = type
        self.numSim = numSim
        self.baseContract = contract
        self.parameter = params
        self.hasGas = False
        for i in range(len(self.parameter)):
            if self.parameter[i]["id"] == 1:
                self.hasGas = True
                break

        # Get multipliers
        self.multipliers = np.ones([self.numSim, len(self.parameter)], dtype=np.float64)

        for i in range(len(self.parameter)):
            self.multipliers[:, i] = get_multipliers_montecarlo(
                run_number=self.numSim,
                distribution=(
                    self.parameter[i]["dist"].value
                ),
                min_value=self.parameter[i]["min"],
                mean_value=self.parameter[i]["base"],
                max_value=self.parameter[i]["max"],
                std_dev=self.parameter[i]["stddev"],
            )

    def Adjust_Data(self, multipliers: np.ndarray):
        Adj_Contract = copy.deepcopy(self.baseContract)

        def Adj_Partial_Data(
            contract_: dict, par: str, key: str, multiplier: float, datakeys: list = []
        ):
            for item_key in contract_[key].keys():
                item = contract_[key][item_key]
                if (
                    par == "Lifting"
                    and key == "lifting"
                    and item["fluid_type"] == "Gas"
                ):
                    continue
                if key == "lifting":
                    if (
                        (par == "Oil Price" and item["fluid_type"] == "Oil")
                        or (par == "Gas Price" and item["fluid_type"] == "Gas")
                        or par == "Lifting"
                    ):
                        lifting_key = (
                            ["price"]
                            if par == "Oil Price" or par == "Gas Price"
                            else ["lifting_rate", "prod_rate"]
                        )
                        for lift_keys in lifting_key:
                            item[lift_keys] = (
                                np.array(item[lift_keys]) * multiplier
                            ).tolist()
                else:
                    for data_key in datakeys:
                        item[data_key] = (
                            np.array(item[data_key]) * multiplier
                        ).tolist()

        # for iloop in range(2 if self.type >= 3 else 1):
        contract_ = (
            # Adj_Contract if self.type < 3 else Adj_Contract[f"contract_{iloop+1}"]
            Adj_Contract if self.type < 3 else Adj_Contract[f"contract_{2}"]
        )
        for i in range(len(self.parameter)):
            # OIl
            if self.parameter[i]["id"] == 0:
                Adj_Partial_Data(contract_, "Oil Price", "lifting", multipliers[i])
            elif self.parameter[i]["id"] == 1:
                Adj_Partial_Data(contract_, "Gas Price", "lifting", multipliers[i])
            elif self.parameter[i]["id"] == 2:
                Adj_Partial_Data(
                    contract_,
                    "OPEX",
                    "opex",
                    multipliers[i],
                    ["fixed_cost", "cost_per_volume"],
                )
            elif self.parameter[i]["id"] == 3:
                Adj_Partial_Data(
                    contract_, "CAPEX", "capital", multipliers[i], ["cost"]
                )
                Adj_Partial_Data(
                    contract_, "CAPEX", "intangible", multipliers[i], ["cost"]
                )
            elif self.parameter[i]["id"] == 4:
                Adj_Partial_Data(contract_, "Lifting", "lifting", multipliers[i])

        return Adj_Contract

    def calcContract(self, n: int):
        try:
            print(f"Monte Progress:{n}", flush=True)
            # time.sleep(100)

            dataAdj = self.Adjust_Data(self.multipliers[n, :])
            csummary = (
                get_costrecovery(data=dataAdj)[0]
                if self.type == 1
                else (
                    get_grosssplit(data=dataAdj)[0]
                    if self.type == 2
                    else (
                        get_transition(data=dataAdj)[0]
                        if self.type >= 3
                        else get_baseproject(data=dataAdj)[0]
                    )
                )
            )
            del dataAdj
            return {
                "n": n,
                "output": (
                    csummary["ctr_npv"],
                    csummary["ctr_irr"],
                    csummary["ctr_pi"],
                    csummary["ctr_pot"],
                    csummary["gov_take"],
                    csummary["ctr_net_share"],
                ),
            }
        except Exception as err:
            print(f"Error: {err}")
            return {
                "n": n,
                "output": (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            }

    def calculate(self):
        results = np.zeros(
            [self.numSim, len(self.target) + len(self.parameter)], dtype=np.float64
        )

        # # Execute MonteCarlo simulation
        # client = Client()
        # b = db.from_sequence(range(self.numSim), partition_size=100)
        # futures = b.map(self.calcContract).compute()
        # # print(futures)
        # for res in futures:
        #     # for res in outcalcmonte.get():
        #     results[res["n"], 0 : len(self.target)] = res["output"]
        #     results[res["n"], len(self.target) :] = [
        #         self.multipliers[res["n"], index] * item["base"]
        #         for index, item in enumerate(self.parameter)
        #     ]
        #
        # client.close()

        # Use ProcessPoolExecutor for parallel execution
        # import concurrent.futures
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     futures = [executor.submit(self.calcContract, n) for n in range(self.numSim)]
        #     for future in concurrent.futures.as_completed(futures):
        #         res = future.result()
        #         results[res["n"], 0: len(self.target)] = res["output"]
        #         results[res["n"], len(self.target):] = [
        #             self.multipliers[res["n"], index] * item["base"]
        #             for index, item in enumerate(self.parameter)
        #         ]


        # Execute MonteCarlo simulation using pathos multiprocessing
        from pathos.multiprocessing import ProcessingPool as Pool
        with Pool() as pool:
            futures = pool.map(self.calcContract, range(self.numSim))

        for res in futures:
            results[res["n"], 0: len(self.target)] = res["output"]
            results[res["n"], len(self.target):] = [
                self.multipliers[res["n"], index] * item["base"]
                for index, item in enumerate(self.parameter)
            ]

        # Sorted the results
        results_sorted = np.take_along_axis(
            arr=results,
            indices=np.argsort(results, axis=0),
            axis=0,
        )
        # Specify probability
        prob = np.arange(1, self.numSim + 1, dtype=np.float64) / self.numSim

        # Arrange the results
        results_arranged = np.concatenate((prob[:, np.newaxis], results_sorted), axis=1)

        # Calculate P10, P50, P90
        percentiles = np.percentile(
            a=results_arranged,
            q=[10, 50, 90],
            method="higher",
            axis=0,
        )

        # Determine indices of data
        indices = np.linspace(0, self.numSim, 101)[0:-1].astype(int)
        indices[0] = 1
        if indices[-1] != self.numSim - 1:
            indices = np.append(indices, int(self.numSim - 1))

        # Final outcomes
        outcomes = {
            "params": (
                ["Oil Price", "Gas Price", "Opex", "Capex", "Cum. prod."]
                if self.hasGas
                else ["Oil Price", "Opex", "Capex", "Cum. prod."]
            ),
            "results": results_arranged[indices, :].tolist(),
            "P10": percentiles[0, :].tolist(),
            "P50": percentiles[1, :].tolist(),
            "P90": percentiles[2, :].tolist(),
        }

        return outcomes

def uncertainty_psc(
        contract: BaseProject | CostRecovery | GrossSplit | Transition,
        contract_arguments: dict,
        summary_arguments: dict,
        run_number: int = 10,
        min_oil_price: float = None,
        mean_oil_price: float = None,
        max_oil_price: float = None,
        min_gas_price: float = None,
        mean_gas_price: float = None,
        max_gas_price: float = None,
        min_opex: float = None,
        mean_opex: float = None,
        max_opex: float = None,
        min_capex: float = None,
        mean_capex: float = None,
        max_capex: float = None,
        min_lifting: float = None,
        mean_lifting: float = None,
        max_lifting: float = None,
        oil_price_stddev: float = 1.25,
        gas_price_stddev: float = 1.25,
        opex_stddev: float = 1.25,
        capex_stddev: float = 1.25,
        lifting_stddev: float = 1.25,
        oil_price_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        gas_price_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        opex_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        capex_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        lifting_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        verbose: bool = False,
):
    # Translating the contract type before parsing into ProcessMonte class
    if isinstance(contract, CostRecovery):
        contract_type = 1
    elif isinstance(contract, GrossSplit):
        contract_type = 2
    elif isinstance(contract, Transition):
        contract_type = 3
    else:
        contract_type = 0

    # Retrieve the Min Max and Average Values
    min_max_mean_original = min_mean_max_retriever(contract=contract, verbose=verbose)

    min_max_mean_std = {
        'min_oil_price': min_oil_price,
        'mean_oil_price': mean_oil_price,
        'max_oil_price': max_oil_price,
        'min_gas_price': min_gas_price,
        'mean_gas_price': mean_gas_price,
        'max_gas_price': max_gas_price,
        'min_opex': min_opex,
        'mean_opex': mean_opex,
        'max_opex': max_opex,
        'min_capex': min_capex,
        'mean_capex': mean_capex,
        'max_capex': max_capex,
        'min_lifting': min_lifting,
        'mean_lifting': mean_lifting,
        'max_lifting': max_lifting,
    }

    # If the element is not being input from argument, fill with original value
    for element in min_max_mean_std.keys():
        if min_max_mean_std[element] is None:
            min_max_mean_std[element] = min_max_mean_original[element]

    # Iterate over the dictionary
    for key in list(min_max_mean_std.keys()):
        if key.startswith("min_"):
            # Base key (e.g., 'capex', 'opex', etc.)
            base = key[4:]
            min_key = f"min_{base}"
            mean_key = f"mean_{base}"
            max_key = f"max_{base}"

            # Check if min, mean, and max values are the same
            if (
                    min_key in min_max_mean_std and mean_key in min_max_mean_std and max_key in min_max_mean_std and
                    min_max_mean_std[min_key] == min_max_mean_std[mean_key] == min_max_mean_std[max_key]
            ):
                # Adjust min and max values
                min_max_mean_std[min_key] = 0.8 * min_max_mean_std[min_key]  # Set min to 0.8 of the min
                min_max_mean_std[max_key] = 1.2 * min_max_mean_std[max_key]  # Set max to 1.2 of the max

    parameter= [
        {"id": 0, "dist": oil_price_distribution, "min": min_max_mean_std['min_oil_price'], "base": min_max_mean_std['mean_oil_price'], "max": min_max_mean_std['max_oil_price'], "stddev": oil_price_stddev},
        {"id": 1, "dist": gas_price_distribution, "min": min_max_mean_std['min_gas_price'], "base": min_max_mean_std['mean_gas_price'], "max": min_max_mean_std['max_gas_price'],"stddev": gas_price_stddev},
        {"id": 2, "dist": opex_distribution, "min": min_max_mean_std['min_opex'], "base": min_max_mean_std['mean_opex'], "max": min_max_mean_std['max_opex'],"stddev": opex_stddev},
        {"id": 3, "dist": capex_distribution, "min": min_max_mean_std['min_capex'], "base": min_max_mean_std['mean_capex'], "max": min_max_mean_std['max_capex'],"stddev": capex_stddev},
        {"id": 4, "dist": lifting_distribution, "min": min_max_mean_std['min_lifting'], "base": min_max_mean_std['mean_lifting'], "max": min_max_mean_std['max_lifting'],"stddev": lifting_stddev}
    ]

    # Condition when there is no gas produced
    fluid_produced = [lift.fluid_type.value for lift in contract.lifting]
    if FluidType.GAS not in fluid_produced:
        del parameter[1]

    # Constructing the contract key
    contract_dict = get_contract_attributes(
        contract=contract,
        contract_arguments=contract_arguments,
        summary_arguments=summary_arguments
    )

    # Executing the montecarlo
    monte = ProcessMonte(
        contract_type,
        contract_dict,
        run_number,
        parameter,
    )

    return monte.calculate()

################################################ Uncertainty Detached ################################################
# """
# This Code is made to detached the uncertainty module in the previous version which depended with the excel
# into fully able to be run in python.
# """
#
# from pyscnomics.econ.selection import UncertaintyDistribution
# from pyscnomics.optimize.sensitivity import _adjust_element_single_contract
#
# def get_multipliers_monte(
#     run_number: int,
#     distribution: UncertaintyDistribution,
#     min_value: float,
#     mean_value: float,
#     max_value: float,
#     std_dev: float,
# ) -> np.ndarray:
#     """
#     Generate an array of multipliers for Monte Carlo simulation based on the specified distribution.
#
#     Parameters
#     ----------
#     run_number: int
#         Number of runs.
#     distribution: str
#         Type of distribution ("Uniform", "Triangular", or "Normal").
#     min_value: float
#         Minimum value for the distribution.
#     mean_value: float
#         Mean (or central value) for the distribution.
#     max_value: float
#         Maximum value for the distribution.
#     std_dev: float
#         Standard deviation for the normal distribution.
#
#     Returns
#     -------
#     multipliers: np.ndarray
#         Array of multipliers generated using Monte Carlo simulation.
#
#     Notes
#     -----
#     - For "Uniform" distribution, the function uses a uniform random variable.
#     - For "Triangular" distribution, the function uses a triangular random variable.
#     - For "Normal" distribution, the function uses a truncated normal random variable.
#     """
#     # Uniform distribution
#     if distribution == UncertaintyDistribution.UNIFORM:
#         # Modify minimum and maximum values
#         params = [min_value, max_value]
#         attrs_updated = {
#             key: float(params[i] / mean_value)
#             for i, key in enumerate(["min_value", "max_value"])
#         }
#
#         # Determine multipliers
#         multipliers = uniform.rvs(
#             loc=attrs_updated["min_value"],
#             scale=attrs_updated["max_value"] - attrs_updated["min_value"],
#             size=run_number,
#         )
#
#     # Triangular distribution
#     elif distribution == UncertaintyDistribution.TRIANGULAR:
#         # Modify minimum and maximum values
#         params = [min_value, mean_value, max_value]
#         attrs_updated = {
#             key: float(params[i] / mean_value)
#             for i, key in enumerate(["min_value", "mean_value", "max_value"])
#         }
#
#         # Determine mode (central point)
#         c = (
#             (attrs_updated["mean_value"] - attrs_updated["min_value"]) /
#             (attrs_updated["max_value"] - attrs_updated["min_value"])
#         )
#
#         # Determine multipliers
#         multipliers = triang.rvs(
#             c=c,
#             loc=attrs_updated["min_value"],
#             scale=attrs_updated["max_value"] - attrs_updated["min_value"],
#             size=run_number,
#         )
#
#     # Normal distribution
#     elif distribution == UncertaintyDistribution.NORMAL:
#         # Modify minimum and maximum values
#         params = [min_value, mean_value, max_value, std_dev]
#         attrs_updated = {
#             key: float(params[i] / mean_value)
#             for i, key in enumerate(["min_value", "mean_value", "max_value", "std_dev"])
#         }
#
#         # Determine z-values
#         zvalues = {
#             key: float(
#                 (attrs_updated[key] - attrs_updated["mean_value"]) / attrs_updated["std_dev"]
#             )
#             for key in ["min_value", "mean_value", "max_value"]
#         }
#
#         # Determine multipliers
#         multipliers_init = truncnorm.rvs(
#             a=zvalues["min_value"],
#             b=zvalues["max_value"],
#             loc=zvalues["mean_value"],
#             scale=1,
#             size=run_number,
#         )
#
#         multipliers = (multipliers_init * attrs_updated["std_dev"]) + attrs_updated["mean_value"]
#
#     else:
#         raise MonteCarloException(
#             f"The type of distribution is unavailable. "
#             f"Please select type of distribution between: "
#             f"(i) Uniform, (ii) Triangular, or (iii) Normal."
#         )
#
#     return multipliers
#
# def min_mean_max_retriever(
#         contract: BaseProject | CostRecovery | GrossSplit | Transition,
# ):
#     # Retrieve Min, Mean, and Max for CAPEX
#     min_capex = np.min(np.concatenate(
#         (contract.capital_cost_total.cost, contract.intangible_cost_total.cost)))
#     mean_capex = np.average(np.concatenate(
#         (contract.capital_cost_total.cost, contract.intangible_cost_total.cost)))
#     max_capex = np.max(np.concatenate(
#         (contract.capital_cost_total.cost, contract.intangible_cost_total.cost)))
#
#     # Retrieve Min, Mean, and Max for OPEX
#     min_opex = np.min(np.concatenate(
#         (contract.opex_total.cost, contract.asr_cost_total.cost,
#          contract.lbt_cost_total.cost, contract.cost_of_sales_total.cost)))
#     mean_opex = np.average(np.concatenate(
#         (contract.opex_total.cost, contract.asr_cost_total.cost,
#          contract.lbt_cost_total.cost, contract.cost_of_sales_total.cost)))
#     max_opex = np.max(np.concatenate(
#         (contract.opex_total.cost, contract.asr_cost_total.cost,
#          contract.lbt_cost_total.cost, contract.cost_of_sales_total.cost)))
#
#     # Retrieve Min, Mean, and Max for Oil Price
#     min_oil_price = np.min(contract._oil_lifting.price)
#     mean_oil_price = np.average(contract._oil_lifting.price)
#     max_oil_price = np.max(contract._oil_lifting.price)
#
#     # Retrieve Min, Mean, Max for Gas Price
#     min_gas_price = np.min(contract._gas_lifting.price)
#     mean_gas_price = np.average(contract._gas_lifting.price)
#     max_gas_price = np.max(contract._gas_lifting.price)
#
#     # Retrieve Min, Mean, Max for Oil Lifting
#     min_oil_lifting = np.min(contract._oil_lifting.lifting_rate)
#     mean_oil_lifting = np.average(contract._oil_lifting.lifting_rate)
#     max_oil_lifting = np.max(contract._oil_lifting.lifting_rate)
#
#     # Retrieve Min, Mean, Max for Gas Lifting
#     min_gas_lifting = np.min(contract._gas_lifting.lifting_rate)
#     mean_gas_lifting = np.average(contract._gas_lifting.lifting_rate)
#     max_gas_lifting = np.max(contract._gas_lifting.lifting_rate)
#
#     return {
#         'min_capex': min_capex,
#         'mean_capex': mean_capex,
#         'max_capex': max_capex,
#         'min_opex': min_opex,
#         'mean_opex': mean_opex,
#         'max_opex': max_opex,
#         'min_oil_price': min_oil_price,
#         'mean_oil_price': mean_oil_price,
#         'max_oil_price': max_oil_price,
#         'min_gas_price': min_gas_price,
#         'mean_gas_price': mean_gas_price,
#         'max_gas_price': max_gas_price,
#         'min_oil_lifting': min_oil_lifting,
#         'mean_oil_lifting': mean_oil_lifting,
#         'max_oil_lifting': max_oil_lifting,
#         'min_gas_lifting': min_gas_lifting,
#         'mean_gas_lifting': mean_gas_lifting,
#         'max_gas_lifting': max_gas_lifting,
#     }
#
#
# def uncertainty_psc(
#         contract: BaseProject | CostRecovery | GrossSplit | Transition,
#         contract_arguments: dict,
#         summary_arguments: dict,
#         run_number: int = 100,
#         min_capex: float = None,
#         mean_capex: float = None,
#         max_capex: float = None,
#         min_opex: float = None,
#         mean_opex: float = None,
#         max_opex: float = None,
#         min_oil_price: float = None,
#         mean_oil_price: float = None,
#         max_oil_price: float = None,
#         min_gas_price: float = None,
#         mean_gas_price: float = None,
#         max_gas_price: float = None,
#         min_oil_lifting: float = None,
#         mean_oil_lifting: float = None,
#         max_oil_lifting: float = None,
#         min_gas_lifting: float = None,
#         mean_gas_lifting: float = None,
#         max_gas_lifting: float = None,
#         capex_stddev: float = 1.25,
#         opex_stddev: float = 1.25,
#         oil_price_stddev: float = 1.25,
#         gas_price_stddev: float = 1.25,
#         oil_lifting_stddev: float = 1.25,
#         gas_lifting_stddev: float = 1.25,
#         capex_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
#         opex_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
#         oil_price_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
#         gas_price_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
#         oil_lifting_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
#         gas_lifting_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
#         dataframe_output: bool = True,
# ) -> tuple:
#     """
#     The function to get the uncertainty analysis (Monte Carlo) of a contract.
#
#     Parameters
#     ----------
#     contract: BaseProject | CostRecovery | GrossSplit | Transition
#         The contract that the uncertainty will be retrieved.
#     contract_arguments: dict
#         The contract arguments of the contract.
#     summary_arguments: dict
#         The summary arguments.
#     run_number: int = 100
#         The total run number for the monte carlo.
#     min_capex: float = None
#         Minimum value for the capital cost component.
#     mean_capex: float = None
#         Average value for the capital cost component.
#     max_capex: float = None
#         Maximum value for the capital cost component.
#     min_opex: float = None
#         Minimum value for the opex cost component.
#     mean_opex: float = None
#         Average value for the opex cost component.
#     max_opex: float = None
#         Maximum value for the opex cost component.
#     min_oil_price: float = None
#         Minimum value for the price of oil lifting.
#     mean_oil_price: float = None
#         Average value for the price of oil lifting.
#     max_oil_price: float = None
#         Maximum value for the price of oil lifting.
#     min_gas_price: float = None
#         Minimum value for the price of gas lifting.
#     mean_gas_price: float = None
#         Average value for the price of gas lifting.
#     max_gas_price: float = None
#         Maximum value for the price of gas lifting.
#     min_oil_lifting: float = None
#         Minimum value for the oil lifting.
#     mean_oil_lifting: float = None
#         Average value for the oil lifting.
#     max_oil_lifting: float = None
#         Maximum value for the oil lifting.
#     min_gas_lifting: float = None
#         Minimum value for the gas lifting.
#     mean_gas_lifting: float = None
#         Average value for the gas lifting.
#     max_gas_lifting: float = None
#         Maximum value for the gas lifting.
#     capex_stddev: float = 1.25
#         The standard deviation of the capital cost element which will be used in monte carlo
#     opex_stddev: float = 1.25
#         The standard deviation of the opex cost element which will be used in monte carlo
#     oil_price_stddev: float = 1.25
#         The standard deviation of the oil price element which will be used in monte carlo
#     gas_price_stddev: float = 1.25
#         The standard deviation of the gas price element which will be used in monte carlo
#     oil_lifting_stddev: float = 1.25
#         The standard deviation of the oil lifting element which will be used in monte carlo
#     gas_lifting_stddev: float = 1.25
#         The standard deviation of the gas lifting element which will be used in monte carlo
#     capex_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL
#         The distribution option which will be used in generating capex values.
#     opex_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL
#         The distribution option which will be used in generating opex values.
#     oil_price_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL
#         The distribution option which will be used in generating oil price values.
#     gas_price_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL
#         The distribution option which will be used in generating gas price values.
#     oil_lifting_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL
#         The distribution option which will be used in generating oil lifting values.
#     gas_lifting_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL
#         The distribution option which will be used in generating gas lifting values.
#     dataframe_output: bool = True
#         The option whether the output in a dataframe form or dictionary.
#     Returns
#     -------
#     out: tuple
#         Uncertainty result,
#         Group of the uncertainty result [P10, P50 AND P90]
#     """
#     # Retrieve the Min Max and Average Values
#     min_max_mean_original = min_mean_max_retriever(contract=contract)
#
#     min_max_mean_std = {
#         'min_capex': min_capex,
#         'mean_capex': mean_capex,
#         'max_capex': max_capex,
#         'min_opex': min_opex,
#         'mean_opex': mean_opex,
#         'max_opex': max_opex,
#         'min_oil_price': min_oil_price,
#         'mean_oil_price': mean_oil_price,
#         'max_oil_price': max_oil_price,
#         'min_gas_price': min_gas_price,
#         'mean_gas_price': mean_gas_price,
#         'max_gas_price': max_gas_price,
#         'min_oil_lifting': min_oil_lifting,
#         'mean_oil_lifting': mean_oil_lifting,
#         'max_oil_lifting': max_oil_lifting,
#         'min_gas_lifting': min_gas_lifting,
#         'mean_gas_lifting': mean_gas_lifting,
#         'max_gas_lifting': max_gas_lifting,
#     }
#
#     # If the element is not being input from argument, fill with original value
#     for element in min_max_mean_std.keys():
#         if min_max_mean_std[element] is None:
#             min_max_mean_std[element] = min_max_mean_original[element]
#
#     # Constructing the dictionary to contain the uncertainty data
#     uncertainty_dict = {
#         'CAPEX': {
#             'min': min_max_mean_std['min_capex'],
#             'mean': min_max_mean_std['mean_capex'],
#             'max': min_max_mean_std['max_capex'],
#             'std_dev': capex_stddev,
#             'distribution': capex_distribution,
#         },
#         'OPEX': {
#             'min': min_max_mean_std['min_opex'],
#             'mean': min_max_mean_std['mean_opex'],
#             'max': min_max_mean_std['max_opex'],
#             'std_dev': opex_stddev,
#             'distribution': opex_distribution,
#         },
#         'OILPRICE': {
#             'min': min_max_mean_std['min_oil_price'],
#             'mean': min_max_mean_std['mean_oil_price'],
#             'max': min_max_mean_std['max_oil_price'],
#             'std_dev': oil_price_stddev,
#             'distribution': oil_price_distribution,
#         },
#         'GASPRICE': {
#             'min': min_max_mean_std['min_gas_price'],
#             'mean': min_max_mean_std['mean_gas_price'],
#             'max': min_max_mean_std['max_gas_price'],
#             'std_dev': gas_price_stddev,
#             'distribution': gas_price_distribution,
#         },
#         'OILLIFTING': {
#             'min': min_max_mean_std['min_oil_lifting'],
#             'mean': min_max_mean_std['mean_oil_lifting'],
#             'max': min_max_mean_std['max_oil_lifting'],
#             'std_dev': oil_lifting_stddev,
#             'distribution': oil_lifting_distribution,
#         },
#         'GASLIFTING': {
#             'min': min_max_mean_std['min_gas_lifting'],
#             'mean': min_max_mean_std['mean_gas_lifting'],
#             'max': min_max_mean_std['max_gas_lifting'],
#             'std_dev': gas_lifting_stddev,
#             'distribution': gas_lifting_distribution,
#         },
#     }
#
#     # Removing the same value or zeros value of min max and mean element
#     uncertainty_dict = {
#         key: value for key, value in uncertainty_dict.items() if np.array([value['min'], value['mean'], value['max']]).std() != 0.0
#     }
#
#     # Get the multipliers based on the constructed uncertainty dictionary
#     for element in uncertainty_dict.keys():
#         uncertainty_dict[element]['multipliers'] = get_multipliers_monte(
#             run_number=run_number,
#             distribution=uncertainty_dict[element]['distribution'],
#             min_value=uncertainty_dict[element]['min'],
#             mean_value=uncertainty_dict[element]['mean'],
#             max_value=uncertainty_dict[element]['max'],
#             std_dev=uncertainty_dict[element]['std_dev'],
#         )
#
#     # Make a container for the adjusted contract
#     adjusted_contract_list = [contract] * run_number
#
#     # Adjust the element of each contract based on each multiplier simultaneously
#     for index in range(run_number):
#         for element in uncertainty_dict.keys():
#             adjusted_contract_list[index] = _adjust_element_single_contract(
#                 contract=adjusted_contract_list[index],
#                 contract_arguments=contract_arguments,
#                 element=element,
#                 adjustment_value=uncertainty_dict[element]['multipliers'][index],
#                 run_contract=False,
#             )
#
#     ## The alternative code using lambda
#     # for index, contract in enumerate(adjusted_contract_list):
#     #     adjusted_contract_list[index] = reduce(
#     #         lambda psc, element: _adjust_element_single_contract(
#     #             contract=psc,
#     #             contract_arguments=contract_arguments,
#     #             element=element,
#     #             adjustment_value=uncertainty_dict[element]['multipliers'][index],
#     #             run_contract=False,
#     #         ),
#     #         uncertainty_dict.keys(),
#     #         contract
#     #     )
#
#     # Running all contract in the adjusted_contract_list
#     for cntrct in adjusted_contract_list:
#         cntrct.run(**contract_arguments)
#
#     # Retrieving the summary of each contract in adjusted_contract_list
#     summary_list = [get_summary(
#                 **{**summary_arguments, 'contract': contrct}
#             ) for contrct in adjusted_contract_list]
#
#     # Todo: separate the routine of adjust and running the contract to accommodate Multi Processing
#     # Constructing a dictionary to contain the sorted result
#     target = ["ctr_npv", "ctr_irr", "ctr_pi", "ctr_pot", "gov_take", "ctr_net_share"]
#     filtered_result = {
#         key :np.array([
#             summary[key]
#             for summary in summary_list
#         ])
#         for key in target
#     }
#
#     # Sorting the result
#     for indicator in filtered_result.keys():
#         filtered_result[indicator] = np.take_along_axis(
#             arr=filtered_result[indicator],
#             indices=np.argsort(filtered_result[indicator], axis=0),
#             axis=0,
#         )
#
#     percentiles_dict = {
#         key: np.percentile(
#             a=filtered_result[key],
#             q=[10, 50, 90],
#             method="lower",
#             axis=0,
#         )
#         for key in filtered_result.keys()
#     }
#
#     # Defining the array of probability
#     filtered_result['prob'] = (
#             np.arange(1, run_number + 1, dtype=np.float_)
#             / run_number
#     )
#
#     if dataframe_output is True:
#         return (pd.DataFrame(filtered_result).set_index('prob'),
#                 pd.DataFrame(percentiles_dict).set_index(pd.Index(['P10', 'P50', 'P90'])))
#
#     else:
#         return (filtered_result,
#                 percentiles_dict)
