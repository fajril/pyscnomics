"""
A collection of procedures to generate cashflow of a contract in the form of DataFrame.
"""

from typing import TypeAlias, overload

import pandas as pd

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.transition import Transition

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", 200)

ContractTable: TypeAlias = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
TransitionTable: TypeAlias = tuple[
    tuple[pd.DataFrame, pd.DataFrame],
    tuple[pd.DataFrame, pd.DataFrame],
    tuple[pd.DataFrame, pd.DataFrame],
]


def get_table_costrecovery_oil(contract: CostRecovery) -> pd.DataFrame:
    """Build the oil cashflow table for a CostRecovery contract."""
    return pd.DataFrame(contract.get_oil_table_columns())


def get_table_costrecovery_gas(contract: CostRecovery) -> pd.DataFrame:
    """Build the gas cashflow table for a CostRecovery contract."""
    return pd.DataFrame(contract.get_gas_table_columns())


def get_table_costrecovery_consolidated(contract: CostRecovery) -> pd.DataFrame:
    """Build the consolidated cashflow table for a CostRecovery contract."""
    return pd.DataFrame(contract.get_consolidated_table_columns())


def get_table_grosssplit_oil(contract: GrossSplit) -> pd.DataFrame:
    """Build the oil cashflow table for a GrossSplit contract."""
    return pd.DataFrame(contract.get_oil_table_columns())


def get_table_grosssplit_gas(contract: GrossSplit) -> pd.DataFrame:
    """Build the gas cashflow table for a GrossSplit contract."""
    return pd.DataFrame(contract.get_gas_table_columns())


def get_table_grosssplit_consolidated(contract: GrossSplit) -> pd.DataFrame:
    """Build the consolidated cashflow table for a GrossSplit contract."""
    return pd.DataFrame(contract.get_consolidated_table_columns())


def get_table_baseproject_oil(contract: BaseProject) -> pd.DataFrame:
    """
    Construct the base project cashflow table for oil in a BaseProject contract.

    The function extracts key oil-related attributes from the contract, including
    lifting, revenue, expenditures, indirect taxes, sunk costs, and pre-/post-onstream
    components. It consolidates them into a structured pandas DataFrame for
    downstream economic analysis.

    Parameters
    ----------
    contract : BaseProject
        The contract instance containing all oil-related financial and production
        attributes.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the oil cashflow table with columns for project years,
        lifting, prices, revenues, sunk costs, pre-/post-onstream costs, expenditures
        (pre-tax, indirect tax, post-tax), expenses, and net cashflow.

    Notes
    -----
    - Post-onstream costs are split into depreciable and non-depreciable components.
    - Expenditure categories include: capital, intangible, opex, ASR, LBT, and
      cost of sales.
    - The resulting DataFrame serves as a base input for project-level
      economic evaluations.
    """

    return pd.DataFrame(contract.get_oil_table_columns())


def get_table_baseproject_gas(contract: BaseProject) -> pd.DataFrame:
    """
    Construct the base project cashflow table for gas in a BaseProject contract.

    This function extracts gas-related economic attributes from the contract,
    including lifting, price, revenue, sunk costs, pre-/post-onstream expenditures,
    indirect taxes, and total cashflow. The extracted data are compiled into a
    structured pandas DataFrame suitable for economic analysis and reporting.

    Parameters
    ----------
    contract : BaseProject
        The BaseProject contract instance containing gas production and
        financial attributes.

    Returns
    -------
    pandas.DataFrame
        A DataFrame representing the gas cashflow table with columns for
        project years, lifting, prices, revenues, sunk costs, pre-/post-onstream
        costs, expenditures (pre-tax, indirect tax, post-tax), expenses, and
        overall cashflow.

    Notes
    -----
    - Post-onstream costs are separated into depreciable and non-depreciable components.
    - Expenditure categories include: capital, intangible, opex, ASR, LBT,
      and cost of sales.
    - The resulting DataFrame provides the core structure for gas-related
      project economic evaluation.
    """

    return pd.DataFrame(contract.get_gas_table_columns())


def get_table_baseproject_consolidated(contract: BaseProject) -> pd.DataFrame:
    """
    Construct the consolidated base project cashflow table for a BaseProject contract.

    This function compiles consolidated financial and production data from the
    contract, including lifting, prices, revenues, sunk costs, pre-/post-onstream
    expenditures, indirect taxes, and total cashflow. The resulting data are
    structured into a pandas DataFrame for overall project-level economic analysis.

    Parameters
    ----------
    contract : BaseProject
        The BaseProject contract instance containing consolidated production
        and financial attributes.

    Returns
    -------
    pandas.DataFrame
        A DataFrame representing the consolidated cashflow table, including
        project years, lifting, prices, revenues, sunk costs, pre-/post-onstream
        costs, expenditures (pre-tax, indirect tax, post-tax), expenses, and
        total cashflow.

    Notes
    -----
    - Post-onstream costs are divided into depreciable and non-depreciable parts.
    - Expenditure categories include: capital, intangible, opex, ASR, LBT,
      and cost of sales.
    - The consolidated table integrates oil and gas components into a unified
      cashflow representation for the entire project.
    """

    return pd.DataFrame(contract.get_consolidated_table_columns())


def _get_single_contract_table(contract: object) -> ContractTable:
    if isinstance(contract, CostRecovery):
        return (
            get_table_costrecovery_oil(contract=contract),
            get_table_costrecovery_gas(contract=contract),
            get_table_costrecovery_consolidated(contract=contract),
        )
    if isinstance(contract, GrossSplit):
        return (
            get_table_grosssplit_oil(contract=contract),
            get_table_grosssplit_gas(contract=contract),
            get_table_grosssplit_consolidated(contract=contract),
        )
    if isinstance(contract, BaseProject):
        return (
            get_table_baseproject_oil(contract=contract),
            get_table_baseproject_gas(contract=contract),
            get_table_baseproject_consolidated(contract=contract),
        )
    raise TypeError(f"Unsupported contract type: {type(contract).__name__}")


def _get_transition_table(contract: Transition) -> TransitionTable:
    first, second = contract.get_transitioned_contracts()
    first_oil, first_gas, first_consolidated = _get_single_contract_table(first)
    second_oil, second_gas, second_consolidated = _get_single_contract_table(second)
    return (
        (first_oil, second_oil),
        (first_gas, second_gas),
        (first_consolidated, second_consolidated),
    )


@overload
def get_table(contract: Transition) -> TransitionTable: ...


@overload
def get_table(
    contract: CostRecovery | GrossSplit | BaseProject,
) -> ContractTable: ...


def get_table(
    contract: CostRecovery | GrossSplit | BaseProject | Transition,
) -> ContractTable | TransitionTable:
    """
    Generate OIL, GAS, and CONSOLIDATED cashflow tables for a PSC contract.

    Depending on the contract type (CostRecovery, GrossSplit, BaseProject, or
    Transition), this function dispatches to the appropriate table-construction
    routines and returns the three corresponding cashflow tables.

    Parameters
    ----------
    contract : CostRecovery or GrossSplit or BaseProject or Transition
        PSC contract instance for which cashflow tables are generated.

    Returns
    -------
    tuple of pandas.DataFrame or TransitionTable
        A tuple containing:
        - oil_table : DataFrame
            Cashflow table for OIL stream.
        - gas_table : DataFrame
            Cashflow table for GAS stream.
        - consolidated_table : DataFrame
            Consolidated cashflow table combining OIL and GAS.
    """

    if isinstance(contract, Transition):
        return _get_transition_table(contract)
    return _get_single_contract_table(contract)
