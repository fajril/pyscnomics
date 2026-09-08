from dataclasses import dataclass, field
from datetime import date
from typing import TypeAlias, TypeGuard, TypeVar

import numpy as np
import numpy.typing as npt

from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, CostOfSales, LBT
from pyscnomics.econ.selection import FluidType
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.summary_profile import PostRunSummaryProfile
from pyscnomics.contracts.run_arguments import (
    validate_cost_recovery_run_arguments,
    validate_gross_split_run_arguments,
)
from pyscnomics.contracts.psc_tools import (
    get_cost_to_be_recovered,
    get_unrecovered_cost,
)


class TransitionException(ValueError):
    """Raised when Transition lifecycle requirements are not satisfied."""


FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
NumericArray: TypeAlias = FloatArray | IntArray
Array: TypeAlias = npt.NDArray[np.generic]
_Cost = TypeVar("_Cost")
_Value = TypeVar("_Value")


@dataclass(frozen=True)
class _CapitalFields:
    cost: FloatArray
    expense_year: IntArray
    cost_allocation: list[FluidType]
    description: list[str]
    tax_portion: FloatArray
    tax_discount: float | FloatArray
    pis_year: IntArray
    salvage_value: FloatArray
    useful_life: FloatArray
    depreciation_factor: FloatArray
    is_ic_applied: list[bool]


@dataclass(frozen=True)
class _IntangibleFields:
    cost: FloatArray
    expense_year: IntArray
    cost_allocation: list[FluidType]
    description: list[str]
    tax_portion: FloatArray
    tax_discount: float | FloatArray


@dataclass(frozen=True)
class _OpexFields:
    expense_year: IntArray
    cost_allocation: list[FluidType]
    description: list[str]
    tax_portion: FloatArray
    tax_discount: float | FloatArray
    fixed_cost: FloatArray
    prod_rate: FloatArray
    cost_per_volume: FloatArray


@dataclass(frozen=True)
class _AsrFields:
    cost: FloatArray
    expense_year: IntArray
    cost_allocation: list[FluidType]
    description: list[str]
    tax_portion: FloatArray
    tax_discount: float | FloatArray
    final_year: IntArray
    future_rate: float | FloatArray


@dataclass(frozen=True)
class _CostOfSalesFields:
    cost: FloatArray
    expense_year: IntArray
    cost_allocation: list[FluidType]
    description: list[str]
    tax_portion: FloatArray
    tax_discount: float | FloatArray


@dataclass(frozen=True)
class _LbtFields:
    cost: FloatArray
    expense_year: IntArray
    cost_allocation: list[FluidType]
    description: list[str]
    tax_portion: FloatArray
    tax_discount: float | FloatArray
    final_year: IntArray
    utilized_land_area: FloatArray
    utilized_building_area: FloatArray
    njop_land: FloatArray
    njop_building: FloatArray
    gross_revenue: FloatArray


@dataclass(frozen=True)
class _TransitionRunProfiles:
    """Validated common profiles read from a completed contract run."""

    project_years: IntArray
    oil_lifting: Lifting
    gas_lifting: Lifting
    sulfur_lifting: Lifting
    electricity_lifting: Lifting
    co2_lifting: Lifting
    oil_revenue: FloatArray
    gas_revenue: FloatArray
    sulfur_revenue: FloatArray
    electricity_revenue: FloatArray
    co2_revenue: FloatArray
    oil_wap_price: FloatArray
    gas_wap_price: FloatArray
    sulfur_wap_price: FloatArray
    electricity_wap_price: FloatArray
    co2_wap_price: FloatArray
    oil_sunk_cost: FloatArray
    gas_sunk_cost: FloatArray
    oil_undepreciated_asset: float
    gas_undepreciated_asset: float
    oil_ddmo: FloatArray
    gas_ddmo: FloatArray
    oil_government_take: FloatArray
    gas_government_take: FloatArray
    consolidated_taxable_income: FloatArray
    consolidated_ctr_net_share: FloatArray
    consolidated_cashflow: FloatArray
    consolidated_ddmo: FloatArray
    consolidated_tax_payment: FloatArray
    consolidated_government_take: FloatArray
    tax_rate_arr: FloatArray
    oil_capital_expenditures_post_tax: FloatArray
    gas_capital_expenditures_post_tax: FloatArray
    oil_capital_expenditures_pre_tax: FloatArray
    gas_capital_expenditures_pre_tax: FloatArray
    oil_intangible_expenditures_post_tax: FloatArray
    gas_intangible_expenditures_post_tax: FloatArray
    oil_intangible_expenditures_pre_tax: FloatArray
    gas_intangible_expenditures_pre_tax: FloatArray
    oil_opex_expenditures_post_tax: FloatArray
    gas_opex_expenditures_post_tax: FloatArray
    oil_opex_expenditures_pre_tax: FloatArray
    gas_opex_expenditures_pre_tax: FloatArray
    oil_asr_expenditures_post_tax: FloatArray
    gas_asr_expenditures_post_tax: FloatArray
    oil_asr_expenditures_pre_tax: FloatArray
    gas_asr_expenditures_pre_tax: FloatArray
    oil_cost_of_sales_expenditures_post_tax: FloatArray
    gas_cost_of_sales_expenditures_post_tax: FloatArray
    oil_cost_of_sales_expenditures_pre_tax: FloatArray
    gas_cost_of_sales_expenditures_pre_tax: FloatArray
    oil_lbt_expenditures_post_tax: FloatArray
    gas_lbt_expenditures_post_tax: FloatArray
    oil_lbt_expenditures_pre_tax: FloatArray
    gas_lbt_expenditures_pre_tax: FloatArray
    oil_capital_indirect_tax: FloatArray
    gas_capital_indirect_tax: FloatArray
    oil_intangible_indirect_tax: FloatArray
    gas_intangible_indirect_tax: FloatArray
    oil_opex_indirect_tax: FloatArray
    gas_opex_indirect_tax: FloatArray
    oil_asr_indirect_tax: FloatArray
    gas_asr_indirect_tax: FloatArray
    oil_lbt_indirect_tax: FloatArray
    gas_lbt_indirect_tax: FloatArray
    oil_cost_of_sales_indirect_tax: FloatArray
    gas_cost_of_sales_indirect_tax: FloatArray

    @classmethod
    def from_contract(
        cls, contract: CostRecovery | GrossSplit
    ) -> "_TransitionRunProfiles":
        oil_columns = contract.get_oil_table_columns()
        gas_columns = contract.get_gas_table_columns()
        consolidated_columns = contract.get_consolidated_table_columns()
        (
            oil_undepreciated_asset,
            gas_undepreciated_asset,
            tax_rate_arr,
        ) = contract.get_transition_fiscal_inputs()

        if isinstance(contract, CostRecovery):
            contractor_net_share_key = "contractor_net_share"
            cashflow_key = "cashflow"
            tax_payment_key = "tax_payment"
        else:
            contractor_net_share_key = "net_ctr_share"
            cashflow_key = "ctr_cashflow"
            tax_payment_key = "tax"

        return cls(
            project_years=_require_project_years(contract.project_years),
            oil_lifting=contract.get_lifting_by_fluid(FluidType.OIL),
            gas_lifting=contract.get_lifting_by_fluid(FluidType.GAS),
            sulfur_lifting=contract.get_lifting_by_fluid(FluidType.SULFUR),
            electricity_lifting=contract.get_lifting_by_fluid(FluidType.ELECTRICITY),
            co2_lifting=contract.get_lifting_by_fluid(FluidType.CO2),
            oil_revenue=_table_float(oil_columns, "revenue"),
            gas_revenue=_table_float(gas_columns, "revenue"),
            sulfur_revenue=_table_float(consolidated_columns, "revenue_sulfur"),
            electricity_revenue=_table_float(
                consolidated_columns, "revenue_electricity"
            ),
            co2_revenue=_table_float(consolidated_columns, "revenue_co2"),
            oil_wap_price=_table_float(oil_columns, "price"),
            gas_wap_price=_table_float(gas_columns, "price"),
            sulfur_wap_price=_table_float(consolidated_columns, "price_sulfur"),
            electricity_wap_price=_table_float(
                consolidated_columns, "price_electricity"
            ),
            co2_wap_price=_table_float(consolidated_columns, "price_co2"),
            oil_sunk_cost=_table_float(oil_columns, "sunk_cost"),
            gas_sunk_cost=_table_float(gas_columns, "sunk_cost"),
            oil_undepreciated_asset=oil_undepreciated_asset,
            gas_undepreciated_asset=gas_undepreciated_asset,
            oil_ddmo=_table_float(oil_columns, "ddmo"),
            gas_ddmo=_table_float(gas_columns, "ddmo"),
            oil_government_take=_table_float(oil_columns, "government_take"),
            gas_government_take=_table_float(gas_columns, "government_take"),
            consolidated_taxable_income=_table_float(
                consolidated_columns, "taxable_income"
            ),
            consolidated_ctr_net_share=_table_float(
                consolidated_columns, contractor_net_share_key
            ),
            consolidated_cashflow=_table_float(consolidated_columns, cashflow_key),
            consolidated_ddmo=_table_float(consolidated_columns, "ddmo"),
            consolidated_tax_payment=_table_float(
                consolidated_columns, tax_payment_key
            ),
            consolidated_government_take=_table_float(
                consolidated_columns, "government_take"
            ),
            tax_rate_arr=tax_rate_arr,
            oil_capital_expenditures_post_tax=_table_float(
                oil_columns, "capital_postonstream"
            ),
            gas_capital_expenditures_post_tax=_table_float(
                gas_columns, "capital_postonstream"
            ),
            oil_capital_expenditures_pre_tax=_table_float(
                oil_columns, "capital_expenditures_pre_tax"
            ),
            gas_capital_expenditures_pre_tax=_table_float(
                gas_columns, "capital_expenditures_pre_tax"
            ),
            oil_intangible_expenditures_post_tax=_table_float(
                oil_columns, "intangible_postonstream"
            ),
            gas_intangible_expenditures_post_tax=_table_float(
                gas_columns, "intangible_postonstream"
            ),
            oil_intangible_expenditures_pre_tax=_table_float(
                oil_columns, "intangible_expenditures_pre_tax"
            ),
            gas_intangible_expenditures_pre_tax=_table_float(
                gas_columns, "intangible_expenditures_pre_tax"
            ),
            oil_opex_expenditures_post_tax=_table_float(
                oil_columns, "opex_postonstream"
            ),
            gas_opex_expenditures_post_tax=_table_float(
                gas_columns, "opex_postonstream"
            ),
            oil_opex_expenditures_pre_tax=_table_float(
                oil_columns, "opex_expenditures_pre_tax"
            ),
            gas_opex_expenditures_pre_tax=_table_float(
                gas_columns, "opex_expenditures_pre_tax"
            ),
            oil_asr_expenditures_post_tax=_table_float(oil_columns, "asr_postonstream"),
            gas_asr_expenditures_post_tax=_table_float(gas_columns, "asr_postonstream"),
            oil_asr_expenditures_pre_tax=_table_float(
                oil_columns, "asr_expenditures_pre_tax"
            ),
            gas_asr_expenditures_pre_tax=_table_float(
                gas_columns, "asr_expenditures_pre_tax"
            ),
            oil_cost_of_sales_expenditures_post_tax=_table_float(
                oil_columns, "cost_of_sales_postonstream"
            ),
            gas_cost_of_sales_expenditures_post_tax=_table_float(
                gas_columns, "cost_of_sales_postonstream"
            ),
            oil_cost_of_sales_expenditures_pre_tax=_table_float(
                oil_columns, "cost_of_sales_expenditures_pre_tax"
            ),
            gas_cost_of_sales_expenditures_pre_tax=_table_float(
                gas_columns, "cost_of_sales_expenditures_pre_tax"
            ),
            oil_lbt_expenditures_post_tax=_table_float(oil_columns, "lbt_postonstream"),
            gas_lbt_expenditures_post_tax=_table_float(gas_columns, "lbt_postonstream"),
            oil_lbt_expenditures_pre_tax=_table_float(
                oil_columns, "lbt_expenditures_pre_tax"
            ),
            gas_lbt_expenditures_pre_tax=_table_float(
                gas_columns, "lbt_expenditures_pre_tax"
            ),
            oil_capital_indirect_tax=_table_float(oil_columns, "capital_indirect_tax"),
            gas_capital_indirect_tax=_table_float(gas_columns, "capital_indirect_tax"),
            oil_intangible_indirect_tax=_table_float(
                oil_columns, "intangible_indirect_tax"
            ),
            gas_intangible_indirect_tax=_table_float(
                gas_columns, "intangible_indirect_tax"
            ),
            oil_opex_indirect_tax=_table_float(oil_columns, "opex_indirect_tax"),
            gas_opex_indirect_tax=_table_float(gas_columns, "opex_indirect_tax"),
            oil_asr_indirect_tax=_table_float(oil_columns, "asr_indirect_tax"),
            gas_asr_indirect_tax=_table_float(gas_columns, "asr_indirect_tax"),
            oil_lbt_indirect_tax=_table_float(oil_columns, "lbt_indirect_tax"),
            gas_lbt_indirect_tax=_table_float(gas_columns, "lbt_indirect_tax"),
            oil_cost_of_sales_indirect_tax=_table_float(
                oil_columns, "cost_of_sales_indirect_tax"
            ),
            gas_cost_of_sales_indirect_tax=_table_float(
                gas_columns, "cost_of_sales_indirect_tax"
            ),
        )


@dataclass(frozen=True)
class _TransitionSchemeProfiles:
    """Normalized fiscal profiles shared by both transition schemes."""

    oil_deductible_cost: FloatArray
    gas_deductible_cost: FloatArray
    oil_unrecovered_cost: FloatArray
    gas_unrecovered_cost: FloatArray
    oil_contractor_ets: FloatArray
    gas_contractor_ets: FloatArray
    oil_government_ets: FloatArray
    gas_government_ets: FloatArray
    oil_contractor_ftp: FloatArray
    gas_contractor_ftp: FloatArray
    oil_government_ftp: FloatArray
    gas_government_ftp: FloatArray
    contractor_ftp: FloatArray
    government_ftp: FloatArray
    oil_tax_payment: FloatArray
    gas_tax_payment: FloatArray
    oil_contractor_take: FloatArray
    gas_contractor_take: FloatArray
    oil_cashflow: FloatArray
    gas_cashflow: FloatArray

    @classmethod
    def from_contract(
        cls,
        contract: CostRecovery | GrossSplit,
        *,
        include_gross_split_government_share: bool,
    ) -> "_TransitionSchemeProfiles":
        oil = contract.get_oil_table_columns()
        gas = contract.get_gas_table_columns()
        consolidated = contract.get_consolidated_table_columns()

        if isinstance(contract, CostRecovery):
            oil_deductible = _table_float(oil, "cost_recovery_after_tf")
            gas_deductible = _table_float(gas, "cost_recovery_after_tf")
            return cls(
                oil_deductible_cost=oil_deductible,
                gas_deductible_cost=gas_deductible,
                oil_unrecovered_cost=_table_float(oil, "unrec_after_transfer"),
                gas_unrecovered_cost=_table_float(gas, "unrec_after_transfer"),
                oil_contractor_ets=_table_float(oil, "ets_after_transfer"),
                gas_contractor_ets=_table_float(gas, "ets_after_transfer"),
                oil_government_ets=_table_float(oil, "government_share"),
                gas_government_ets=_table_float(gas, "government_share"),
                oil_contractor_ftp=_table_float(oil, "ftp_ctr"),
                gas_contractor_ftp=_table_float(gas, "ftp_ctr"),
                oil_government_ftp=_table_float(oil, "ftp_gov"),
                gas_government_ftp=_table_float(gas, "ftp_gov"),
                contractor_ftp=_table_float(consolidated, "ftp_ctr"),
                government_ftp=_table_float(consolidated, "ftp_gov"),
                oil_tax_payment=_table_float(oil, "tax_payment"),
                gas_tax_payment=_table_float(gas, "tax_payment"),
                oil_contractor_take=(
                    _table_float(oil, "contractor_net_share") + oil_deductible
                ),
                gas_contractor_take=(
                    _table_float(gas, "contractor_net_share") + gas_deductible
                ),
                oil_cashflow=_table_float(oil, "cashflow"),
                gas_cashflow=_table_float(gas, "cashflow"),
            )

        zero_profile = np.zeros_like(contract.project_years, dtype=float)
        government_ftp = (
            _table_float(consolidated, "government_share")
            if include_gross_split_government_share
            else zero_profile
        )
        return cls(
            oil_deductible_cost=_table_float(oil, "deductible_cost"),
            gas_deductible_cost=_table_float(gas, "deductible_cost"),
            oil_unrecovered_cost=_table_float(oil, "carry_forward_cost"),
            gas_unrecovered_cost=_table_float(gas, "carry_forward_cost"),
            oil_contractor_ets=_table_float(oil, "contractor_share"),
            gas_contractor_ets=_table_float(gas, "contractor_share"),
            oil_government_ets=zero_profile,
            gas_government_ets=zero_profile,
            oil_contractor_ftp=zero_profile,
            gas_contractor_ftp=zero_profile,
            oil_government_ftp=zero_profile,
            gas_government_ftp=zero_profile,
            contractor_ftp=zero_profile,
            government_ftp=government_ftp,
            oil_tax_payment=_table_float(oil, "tax"),
            gas_tax_payment=_table_float(gas, "tax"),
            oil_contractor_take=_table_float(oil, "net_ctr_share"),
            gas_contractor_take=_table_float(gas, "net_ctr_share"),
            oil_cashflow=_table_float(oil, "ctr_cashflow"),
            gas_cashflow=_table_float(gas, "ctr_cashflow"),
        )


@dataclass(frozen=True)
class TransitionFiscalProfiles:
    """Public, read-only fiscal outputs from a completed transition run."""

    tax_payment: FloatArray
    cashflow: FloatArray
    government_take: FloatArray
    oil_undepreciated_asset: float
    gas_undepreciated_asset: float
    consolidated_undepreciated_asset: float
    oil_government_ets: FloatArray
    gas_government_ets: FloatArray
    government_ftp: FloatArray


def _require_tuple(value: tuple[_Cost, ...] | None, name: str) -> tuple[_Cost, ...]:
    if value is None:
        raise ValueError(f"{name} must be initialized before Transition.run")
    return value


def _require_value(value: _Value | None, name: str) -> _Value:
    """Return a lifecycle-initialized value, preserving the established failure mode."""
    if value is None:
        raise TransitionException(f"{name} must be initialized before Transition.run")
    return value


def _normalize_capital(item: CapitalCost) -> _CapitalFields:
    return _CapitalFields(
        _require_value(item.cost, "cost"),
        _require_value(item.expense_year, "expense_year"),
        _require_value(item.cost_allocation, "cost_allocation"),
        _require_value(item.description, "description"),
        _require_value(item.tax_portion, "tax_portion"),
        _require_value(item.tax_discount, "tax_discount"),
        _require_value(item.pis_year, "pis_year"),
        _require_value(item.salvage_value, "salvage_value"),
        _require_value(item.useful_life, "useful_life"),
        _require_value(item.depreciation_factor, "depreciation_factor"),
        _require_value(item.is_ic_applied, "is_ic_applied"),
    )


def _normalize_intangible(item: Intangible) -> _IntangibleFields:
    return _IntangibleFields(
        _require_value(item.cost, "cost"),
        _require_value(item.expense_year, "expense_year"),
        _require_value(item.cost_allocation, "cost_allocation"),
        _require_value(item.description, "description"),
        _require_value(item.tax_portion, "tax_portion"),
        _require_value(item.tax_discount, "tax_discount"),
    )


def _normalize_opex(item: OPEX) -> _OpexFields:
    return _OpexFields(
        _require_value(item.expense_year, "expense_year"),
        _require_value(item.cost_allocation, "cost_allocation"),
        _require_value(item.description, "description"),
        _require_value(item.tax_portion, "tax_portion"),
        _require_value(item.tax_discount, "tax_discount"),
        _require_value(item.fixed_cost, "fixed_cost"),
        _require_value(item.prod_rate, "prod_rate"),
        _require_value(item.cost_per_volume, "cost_per_volume"),
    )


def _normalize_asr(item: ASR) -> _AsrFields:
    return _AsrFields(
        _require_value(item.cost, "cost"),
        _require_value(item.expense_year, "expense_year"),
        _require_value(item.cost_allocation, "cost_allocation"),
        _require_value(item.description, "description"),
        _require_value(item.tax_portion, "tax_portion"),
        _require_value(item.tax_discount, "tax_discount"),
        _require_value(item.final_year, "final_year"),
        _require_value(item.future_rate, "future_rate"),
    )


def _normalize_cost_of_sales(item: CostOfSales) -> _CostOfSalesFields:
    return _CostOfSalesFields(
        _require_value(item.cost, "cost"),
        _require_value(item.expense_year, "expense_year"),
        _require_value(item.cost_allocation, "cost_allocation"),
        _require_value(item.description, "description"),
        _require_value(item.tax_portion, "tax_portion"),
        _require_value(item.tax_discount, "tax_discount"),
    )


def _normalize_cost_of_sales_from_asr(item: _AsrFields) -> _CostOfSalesFields:
    return _CostOfSalesFields(
        item.cost,
        item.expense_year,
        item.cost_allocation,
        item.description,
        item.tax_portion,
        item.tax_discount,
    )


def _normalize_lbt(item: LBT) -> _LbtFields:
    return _LbtFields(
        _require_value(item.cost, "cost"),
        _require_value(item.expense_year, "expense_year"),
        _require_value(item.cost_allocation, "cost_allocation"),
        _require_value(item.description, "description"),
        _require_value(item.tax_portion, "tax_portion"),
        _require_value(item.tax_discount, "tax_discount"),
        _require_int_array(item.final_year, "final_year"),
        _require_float_array(item.utilized_land_area, "utilized_land_area"),
        _require_float_array(item.utilized_building_area, "utilized_building_area"),
        _require_float_array(item.njop_land, "njop_land"),
        _require_float_array(item.njop_building, "njop_building"),
        _require_float_array(item.gross_revenue, "gross_revenue"),
    )


def _table_float(columns: dict[str, NumericArray], key: str) -> FloatArray:
    return _require_float_array(columns[key], key)


def _require_float_array(value: object, name: str) -> FloatArray:
    if not _is_float_profile(value):
        raise TypeError(f"{name} must be a one-dimensional float64 ndarray")
    return value


def _require_int_array(value: object, name: str) -> IntArray:
    if not _is_project_year_profile(value):
        raise TypeError(f"{name} must be a one-dimensional int64 ndarray")
    return value


def _require_transition_scalar(value: float | None, name: str) -> float:
    if value is None:
        raise TransitionException(f"{name} is unavailable before Transition.run()")
    return value


def _require_project_years(value: IntArray | None) -> IntArray:
    if not _is_project_year_profile(value):
        raise ValueError("project_years must be a one-dimensional int64 array")
    return value


def _require_lifting(value: Lifting | None, name: str) -> Lifting:
    if value is None:
        raise TransitionException(f"{name} is unavailable before Transition.run()")
    return value


def _is_float_profile(value: object) -> TypeGuard[FloatArray]:
    return (
        isinstance(value, np.ndarray)
        and value.ndim == 1
        and value.dtype == np.dtype(np.float64)
    )


def _is_project_year_profile(value: object) -> TypeGuard[IntArray]:
    return (
        isinstance(value, np.ndarray)
        and value.ndim == 1
        and value.dtype == np.dtype(np.int64)
    )


def _require_adjustable_argument(value: object, name: str) -> int | float | Array:
    if isinstance(value, (int, float)) or _is_array(value):
        return value
    raise TypeError(f"{name} must be a float or NumPy ndarray")


def _is_array(value: object) -> TypeGuard[Array]:
    return isinstance(value, np.ndarray)


def _run_transition_contract(
    contract: CostRecovery | GrossSplit,
    arguments: dict[str, object],
) -> None:
    """Validate a dynamic argument mapping before contract execution."""

    if isinstance(contract, CostRecovery):
        contract.run(**validate_cost_recovery_run_arguments(arguments))
    else:
        contract.run(**validate_gross_split_run_arguments(arguments))


def adjust_rows(
    original_value: int | float | Array,
    first_contract: bool,
    prior_rows: FloatArray,
    post_rows: FloatArray,
):
    """
    Function to adjust an array based on the given condition, tailored for transition contract.
    Parameters
    ----------
    original_value: float | Array
        The original value of the data.
    first_contract: bool
        The condition whether the row is existed in first or second contract in Transition scheme.
    prior_rows: FloatArray
        The array that will be placed before the given value.
    post_rows:
        The array that will be placed after the given value.
    Returns
    -------
    adjusted_values: float | Array
        The array containing the adjusted value.

    """
    if isinstance(original_value, (int, float)):
        adjusted_values = original_value

    elif _is_array(original_value) and first_contract:
        adjusted_values = np.concatenate((original_value, post_rows))

    elif _is_array(original_value) and not first_contract:
        adjusted_values = np.concatenate((prior_rows, original_value))

    else:
        adjusted_values = original_value

    return adjusted_values


def adjusting_contract_arguments(
    arguments_dict: dict[str, object],
    first_contract: bool,
    prior_rows: FloatArray,
    post_rows: FloatArray,
):
    """
    The function used to adjust the contract arguments before parsed to the transition contract.
    Parameters
    ----------
    arguments_dict: dict
        The dictionary containing the arguments of a contract.
    first_contract: bool
        The condition whether the row is existed in first or second contract in Transition scheme.
    prior_rows: FloatArray
        The array that will be placed before the given value.
    post_rows:
        The array that will be placed after the given value.
    Returns
    -------
    arguments_dict: dict
        The dictionary containing adjusted arguments.
    """
    for argument_name in (
        "tax_rate",
        "vat_rate",
        "lbt_rate",
        "inflation_rate",
        "cum_production_split_offset",
    ):
        if argument_name in arguments_dict:
            arguments_dict[argument_name] = adjust_rows(
                original_value=_require_adjustable_argument(
                    arguments_dict[argument_name], argument_name
                ),
                first_contract=first_contract,
                prior_rows=prior_rows,
                post_rows=post_rows,
            )

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
    argument_contract1: dict[str, object]
    argument_contract2: dict[str, object]

    # Attributes to define later
    _contract1_transitioned: CostRecovery | GrossSplit | None = field(
        default=None, init=False, repr=False
    )
    _contract2_transitioned: CostRecovery | GrossSplit | None = field(
        default=None, init=False, repr=False
    )

    project_years: IntArray | None = field(default=None, init=False, repr=False)

    _oil_lifting: Lifting | None = field(default=None, init=False, repr=False)
    _gas_lifting: Lifting | None = field(default=None, init=False, repr=False)
    _sulfur_lifting: Lifting | None = field(default=None, init=False, repr=False)
    _electricity_lifting: Lifting | None = field(default=None, init=False, repr=False)
    _co2_lifting: Lifting | None = field(default=None, init=False, repr=False)

    _oil_revenue: FloatArray | None = field(default=None, init=False, repr=False)
    _gas_revenue: FloatArray | None = field(default=None, init=False, repr=False)
    _sulfur_revenue: FloatArray | None = field(default=None, init=False, repr=False)
    _electricity_revenue: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _co2_revenue: FloatArray | None = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with pre tax expenditures for each cost elements)
    _oil_capital_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_capital_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_intangible_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_intangible_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_opex_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_opex_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_asr_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_asr_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_cost_of_sales_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_cost_of_sales_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_lbt_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_lbt_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    # Attributes to be defined later (associated with indirect taxes for each cost element)
    _oil_capital_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_capital_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_intangible_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_intangible_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_opex_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_opex_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_asr_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_asr_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_lbt_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_lbt_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_cost_of_sales_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_cost_of_sales_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    # Attributes to be defined later (associated with pre tax expenditures for each cost elements)
    _oil_capital_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_capital_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_intangible_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_intangible_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_opex_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_opex_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_asr_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_asr_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_lbt_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_lbt_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_cost_of_sales_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_cost_of_sales_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    # Attributes to be defined later (associated with total expenditures and indirect taxes for each fluid)
    _oil_total_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_total_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_total_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_total_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_total_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_total_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    _oil_wap_price: FloatArray | None = field(default=None, init=False, repr=False)
    _gas_wap_price: FloatArray | None = field(default=None, init=False, repr=False)
    _sulfur_wap_price: FloatArray | None = field(default=None, init=False, repr=False)
    _electricity_wap_price: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _co2_wap_price: FloatArray | None = field(default=None, init=False, repr=False)

    _oil_sunk_cost: FloatArray | None = field(default=None, init=False, repr=False)
    _gas_sunk_cost: FloatArray | None = field(default=None, init=False, repr=False)
    _oil_deductible_cost: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_deductible_cost: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_unrec_cost: FloatArray | None = field(default=None, init=False, repr=False)
    _gas_unrec_cost: FloatArray | None = field(default=None, init=False, repr=False)
    _oil_ctr_ets: FloatArray | None = field(default=None, init=False, repr=False)
    _gas_ctr_ets: FloatArray | None = field(default=None, init=False, repr=False)
    _oil_gov_ets: NumericArray | None = field(default=None, init=False, repr=False)
    _gas_gov_ets: NumericArray | None = field(default=None, init=False, repr=False)
    _net_operating_profit: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _ctr_net_cashflow: FloatArray | None = field(default=None, init=False, repr=False)
    _ctr_ftp: FloatArray | None = field(default=None, init=False, repr=False)
    _gov_ftp: FloatArray | None = field(default=None, init=False, repr=False)
    _oil_ddmo: FloatArray | None = field(default=None, init=False, repr=False)
    _gas_ddmo: FloatArray | None = field(default=None, init=False, repr=False)
    _ddmo: FloatArray | None = field(default=None, init=False, repr=False)
    _tax_payment: FloatArray | None = field(default=None, init=False, repr=False)
    _oil_effective_tax_payment: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_effective_tax_payment: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_ctr_take: FloatArray | None = field(default=None, init=False, repr=False)
    _gas_ctr_take: FloatArray | None = field(default=None, init=False, repr=False)
    _oil_government_take: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_government_take: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _government_take: FloatArray | None = field(default=None, init=False, repr=False)
    _oil_cashflow: FloatArray | None = field(default=None, init=False, repr=False)
    _gas_cashflow: FloatArray | None = field(default=None, init=False, repr=False)
    _oil_undepreciated_asset: float | None = field(default=None, init=False, repr=False)
    _gas_undepreciated_asset: float | None = field(default=None, init=False, repr=False)

    _oil_ftp_ctr: FloatArray | None = field(default=None, init=False, repr=False)
    _gas_ftp_ctr: FloatArray | None = field(default=None, init=False, repr=False)
    _oil_ftp_gov: FloatArray | None = field(default=None, init=False, repr=False)
    _gas_ftp_gov: FloatArray | None = field(default=None, init=False, repr=False)

    # Consolidated Attributes
    _consolidated_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_revenue: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_sunk_cost: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_undepreciated_asset: float | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_cashflow: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_ddmo: FloatArray | None = field(default=None, init=False, repr=False)
    _consolidated_tax_payment: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_government_take: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    _consolidated_ctr_ftp: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_gov_ftp: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    @staticmethod
    def _parse_dataclass(
        contract: CostRecovery | GrossSplit,
        start_date_trans: date,
        end_date_trans: date,
        lifting: tuple[Lifting, ...],
        capital: tuple[CapitalCost, ...],
        intangible: tuple[Intangible, ...],
        opex: tuple[OPEX, ...],
        asr: tuple[ASR, ...],
        cost_of_sales: tuple[CostOfSales, ...],
        lbt: tuple[LBT, ...],
    ) -> CostRecovery | GrossSplit:

        # Condition where only one fluid is produced
        produced_fluid = [
            item.fluid_type
            for item in _require_tuple(contract.lifting, "contract.lifting")
        ]

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
                approval_year=contract.approval_year,
                oil_onstream_date=oil_onstream_date,
                gas_onstream_date=gas_onstream_date,
                lifting=lifting,
                capital_cost=capital,
                intangible_cost=intangible,
                opex=opex,
                asr_cost=asr,
                cost_of_sales=cost_of_sales,
                lbt_cost=lbt,
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
                gas_dmo_holiday_duration=contract.gas_dmo_holiday_duration,
            )
        else:
            new_contract = GrossSplit(
                start_date=start_date_trans,
                end_date=end_date_trans,
                approval_year=contract.approval_year,
                oil_onstream_date=oil_onstream_date,
                gas_onstream_date=gas_onstream_date,
                lifting=lifting,
                capital_cost=capital,
                intangible_cost=intangible,
                opex=opex,
                asr_cost=asr,
                cost_of_sales=cost_of_sales,
                lbt_cost=lbt,
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
        condition = (
            self.contract1.end_date.day == 31 and self.contract1.end_date.month == 12
        )
        if condition:
            # additional_year = 1
            additional_year = 0
            end_date_condition = 1
        else:
            additional_year = 0
            end_date_condition = 0

        # project_years_trans = np.arange(start_date_trans.year, end_date_trans.year + 1 + additional_year)
        # project_duration_trans = end_date_trans.year - start_date_trans.year + 1 + additional_year

        # Defining the array of years between the prior contract to the new contract
        years_to_new = np.arange(
            self.contract1.end_date.year + 1,
            self.contract2.end_date.year + 1 + additional_year - end_date_condition,
        )
        years_to_prior = np.arange(
            self.contract1.start_date.year,
            self.contract2.start_date.year + additional_year - end_date_condition,
        )

        len_to_new = len(years_to_new)
        len_to_prior = len(years_to_prior)

        # Defining the row gap and description between the prior contract and the new contract
        zeros_to_new = np.zeros_like(years_to_new, dtype=float)
        zeros_to_prior = np.zeros_like(years_to_prior, dtype=float)
        desc_to_new = ["-"] * len_to_new
        desc_to_prior = ["-"] * len_to_prior
        fluid_to_new = [FluidType.OIL] * len_to_new
        fluid_to_prior = [FluidType.OIL] * len_to_prior
        ic_to_new = [False] * len_to_new
        ic_to_prior = [False] * len_to_prior

        contract1_lifting = _require_tuple(self.contract1.lifting, "contract1.lifting")
        contract1_capital = tuple(
            _normalize_capital(item)
            for item in _require_tuple(
                self.contract1.capital_cost, "contract1.capital_cost"
            )
        )
        contract1_intangible = tuple(
            _normalize_intangible(item)
            for item in _require_tuple(
                self.contract1.intangible_cost, "contract1.intangible_cost"
            )
        )
        contract1_opex = tuple(
            _normalize_opex(item)
            for item in _require_tuple(self.contract1.opex, "contract1.opex")
        )
        contract1_asr = tuple(
            _normalize_asr(item)
            for item in _require_tuple(self.contract1.asr_cost, "contract1.asr_cost")
        )
        contract1_cost_of_sales = tuple(
            _normalize_cost_of_sales(item)
            for item in _require_tuple(
                self.contract1.cost_of_sales, "contract1.cost_of_sales"
            )
        )
        contract1_lbt = tuple(
            _normalize_lbt(item)
            for item in _require_tuple(self.contract1.lbt_cost, "contract1.lbt_cost")
        )
        contract2_lifting = _require_tuple(self.contract2.lifting, "contract2.lifting")
        contract2_capital = tuple(
            _normalize_capital(item)
            for item in _require_tuple(
                self.contract2.capital_cost, "contract2.capital_cost"
            )
        )
        contract2_intangible = tuple(
            _normalize_intangible(item)
            for item in _require_tuple(
                self.contract2.intangible_cost, "contract2.intangible_cost"
            )
        )
        contract2_opex = tuple(
            _normalize_opex(item)
            for item in _require_tuple(self.contract2.opex, "contract2.opex")
        )
        contract2_asr = tuple(
            _normalize_asr(item)
            for item in _require_tuple(self.contract2.asr_cost, "contract2.asr_cost")
        )
        contract2_lbt = tuple(
            _normalize_lbt(item)
            for item in _require_tuple(self.contract2.lbt_cost, "contract2.lbt_cost")
        )

        # Modifying the contract1
        # Changing the start_date and end_date of contract1
        new_lifting_1st = [
            Lifting(
                start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                lifting_rate=np.concatenate((lift.lifting_rate, zeros_to_new)),
                price=np.concatenate((lift.price, zeros_to_new)),
                prod_year=np.concatenate((lift.prod_year, years_to_new)),
                fluid_type=lift.fluid_type,
                ghv=np.concatenate(
                    (_require_float_array(lift.ghv, "lifting.ghv"), zeros_to_new)
                ),
                prod_rate=np.concatenate(
                    (
                        _require_float_array(lift.prod_rate, "lifting.prod_rate"),
                        zeros_to_new,
                    )
                ),
                prod_rate_baseline=np.concatenate(
                    (
                        _require_float_array(
                            lift.prod_rate_baseline, "lifting.prod_rate_baseline"
                        ),
                        zeros_to_new,
                    )
                ),
            )
            for lift in contract1_lifting
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract1 tangible
        new_tang_1st = [
            CapitalCost(
                start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                cost=np.concatenate((tang.cost, zeros_to_new)),
                expense_year=np.concatenate((tang.expense_year, years_to_new)).astype(
                    int
                ),
                cost_allocation=tang.cost_allocation + fluid_to_new,
                description=tang.description + desc_to_new,
                tax_portion=np.concatenate((tang.tax_portion, zeros_to_new)),
                tax_discount=np.concatenate((tang.tax_discount, zeros_to_new)),
                pis_year=np.concatenate(
                    (
                        tang.pis_year,
                        np.full_like(years_to_new, tang.pis_year[-1]),
                    )
                ).astype(int),
                salvage_value=np.concatenate((tang.salvage_value, zeros_to_new)),
                useful_life=np.concatenate(
                    (
                        tang.useful_life,
                        np.full_like(zeros_to_new, tang.useful_life[-1]),
                    )
                ),
                depreciation_factor=np.concatenate(
                    (tang.depreciation_factor, zeros_to_new)
                ),
                is_ic_applied=tang.is_ic_applied + ic_to_new,
            )
            for tang in contract1_capital
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract1 intangible
        new_intang_1st = [
            Intangible(
                start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                cost=np.concatenate((intang.cost, zeros_to_new)),
                expense_year=np.concatenate((intang.expense_year, years_to_new)).astype(
                    int
                ),
                cost_allocation=intang.cost_allocation + fluid_to_new,
                description=intang.description + desc_to_new,
                tax_portion=np.concatenate((intang.tax_portion, zeros_to_new)),
                tax_discount=np.concatenate((intang.tax_discount, zeros_to_new)),
            )
            for intang in contract1_intangible
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract1 opex
        new_opex_1st = [
            OPEX(
                start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                expense_year=np.concatenate((opx.expense_year, years_to_new)).astype(
                    int
                ),
                cost_allocation=opx.cost_allocation + fluid_to_new,
                description=opx.description + desc_to_new,
                fixed_cost=np.concatenate((opx.fixed_cost, zeros_to_new)),
                prod_rate=np.concatenate((opx.prod_rate, zeros_to_new)),
                cost_per_volume=np.concatenate((opx.cost_per_volume, zeros_to_new)),
                tax_portion=np.concatenate((opx.tax_portion, zeros_to_new)),
                tax_discount=np.concatenate((opx.tax_discount, zeros_to_new)),
            )
            for opx in contract1_opex
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract1 asr
        new_asr_1st = [
            ASR(
                start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                cost=np.concatenate((asr.cost, zeros_to_new)),
                expense_year=np.concatenate((asr.expense_year, years_to_new)).astype(
                    int
                ),
                cost_allocation=asr.cost_allocation + fluid_to_new,
                description=asr.description + desc_to_new,
                tax_portion=np.concatenate((asr.tax_portion, zeros_to_new)),
                tax_discount=np.concatenate((asr.tax_discount, zeros_to_new)),
                final_year=np.concatenate((asr.final_year, years_to_new)),
                future_rate=np.concatenate((asr.future_rate, zeros_to_new)),
            )
            for asr in contract1_asr
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract1 Cost of Sales
        new_cost_of_sales_1st = [
            CostOfSales(
                start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                cost=np.concatenate((cos.cost, zeros_to_new)),
                expense_year=np.concatenate((cos.expense_year, years_to_new)).astype(
                    int
                ),
                cost_allocation=cos.cost_allocation + fluid_to_new,
                description=cos.description + desc_to_new,
                tax_portion=np.concatenate((cos.tax_portion, zeros_to_new)),
                tax_discount=np.concatenate((cos.tax_discount, zeros_to_new)),
            )
            for cos in contract1_cost_of_sales
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract1 Land and Building Tax
        new_lbt_1st = [
            LBT(
                start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                cost=np.concatenate((bt.cost, zeros_to_new)),
                expense_year=np.concatenate((bt.expense_year, years_to_new)).astype(
                    int
                ),
                cost_allocation=bt.cost_allocation + fluid_to_new,
                description=bt.description + desc_to_new,
                tax_portion=np.concatenate((bt.tax_portion, zeros_to_new)),
                tax_discount=np.concatenate((bt.tax_discount, zeros_to_new)),
                final_year=np.concatenate((bt.final_year, years_to_new)),
                utilized_land_area=np.concatenate(
                    (bt.utilized_land_area, zeros_to_new)
                ),
                utilized_building_area=np.concatenate(
                    (bt.utilized_building_area, zeros_to_new)
                ),
                njop_land=np.concatenate((bt.njop_land, zeros_to_new)),
                njop_building=np.concatenate((bt.njop_building, zeros_to_new)),
                gross_revenue=np.concatenate((bt.gross_revenue, zeros_to_new)),
            )
            for bt in contract1_lbt
        ]

        # Modifying the contract2
        # Changing the start_date and end_date of contract2

        # Concatenating the zeros_to_new and years_to_new to the contract2 lifting
        new_lifting_2nd = [
            Lifting(
                start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                lifting_rate=np.concatenate((zeros_to_prior, lift.lifting_rate)),
                price=np.concatenate((zeros_to_prior, lift.price)),
                prod_year=np.concatenate((years_to_prior, lift.prod_year)).astype(int),
                fluid_type=lift.fluid_type,
                ghv=np.concatenate(
                    (zeros_to_prior, _require_float_array(lift.ghv, "lifting.ghv"))
                ),
                prod_rate=np.concatenate(
                    (
                        zeros_to_prior,
                        _require_float_array(lift.prod_rate, "lifting.prod_rate"),
                    )
                ),
                prod_rate_baseline=np.concatenate(
                    (
                        zeros_to_prior,
                        _require_float_array(
                            lift.prod_rate_baseline, "lifting.prod_rate_baseline"
                        ),
                    )
                ),
            )
            for lift in contract2_lifting
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract2 tangible
        new_tang_2nd = [
            CapitalCost(
                start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                cost=np.concatenate((zeros_to_prior, tang.cost)),
                expense_year=np.concatenate((years_to_prior, tang.expense_year)).astype(
                    int
                ),
                cost_allocation=fluid_to_prior + tang.cost_allocation,
                description=desc_to_prior + tang.description,
                tax_portion=np.concatenate((zeros_to_prior, tang.tax_portion)),
                tax_discount=np.concatenate((zeros_to_prior, tang.tax_discount)),
                pis_year=np.concatenate(
                    (
                        np.full_like(years_to_prior, tang.pis_year[0]),
                        tang.pis_year,
                    )
                ).astype(int),
                salvage_value=np.concatenate((zeros_to_prior, tang.salvage_value)),
                useful_life=np.concatenate(
                    (
                        np.full_like(zeros_to_prior, tang.useful_life[0]),
                        tang.useful_life,
                    )
                ),
                depreciation_factor=np.concatenate(
                    (zeros_to_prior, tang.depreciation_factor)
                ),
                is_ic_applied=ic_to_prior + tang.is_ic_applied,
            )
            for tang in contract2_capital
        ]
        # Concatenating the zeros_to_new and years_to_new to the contract2 intangible
        new_intang_2nd = [
            Intangible(
                start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                cost=np.concatenate((zeros_to_prior, intang.cost)),
                expense_year=np.concatenate(
                    (years_to_prior, intang.expense_year)
                ).astype(int),
                cost_allocation=fluid_to_prior + intang.cost_allocation,
                description=desc_to_prior + intang.description,
                tax_portion=np.concatenate((zeros_to_prior, intang.tax_portion)),
                tax_discount=np.concatenate((zeros_to_prior, intang.tax_discount)),
            )
            for intang in contract2_intangible
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract2 opex
        new_opex_2nd = [
            OPEX(
                start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                expense_year=np.concatenate((years_to_prior, opx.expense_year)).astype(
                    int
                ),
                cost_allocation=fluid_to_prior + opx.cost_allocation,
                description=desc_to_prior + opx.description,
                tax_portion=np.concatenate((zeros_to_prior, opx.tax_portion)),
                tax_discount=np.concatenate((zeros_to_prior, opx.tax_discount)),
                fixed_cost=np.concatenate((zeros_to_prior, opx.fixed_cost)),
                prod_rate=np.concatenate((zeros_to_prior, opx.prod_rate)),
                cost_per_volume=np.concatenate((zeros_to_prior, opx.cost_per_volume)),
            )
            for opx in contract2_opex
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract2 asr
        new_asr_2nd = [
            ASR(
                start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                cost=np.concatenate((zeros_to_prior, asr.cost)),
                expense_year=np.concatenate((years_to_prior, asr.expense_year)).astype(
                    int
                ),
                cost_allocation=fluid_to_prior + asr.cost_allocation,
                description=desc_to_prior + asr.description,
                tax_portion=np.concatenate((zeros_to_prior, asr.tax_portion)),
                tax_discount=np.concatenate((zeros_to_prior, asr.tax_discount)),
                final_year=np.concatenate((years_to_prior, asr.final_year)),
                future_rate=np.concatenate((zeros_to_prior, asr.future_rate)),
            )
            for asr in contract2_asr
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract2 Cost Of Sales
        new_cost_of_sales_2nd = [
            CostOfSales(
                start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                cost=np.concatenate((zeros_to_prior, cos.cost)),
                expense_year=np.concatenate((years_to_prior, cos.expense_year)).astype(
                    int
                ),
                cost_allocation=fluid_to_prior + cos.cost_allocation,
                tax_portion=np.concatenate((zeros_to_prior, cos.tax_portion)),
                tax_discount=np.concatenate((zeros_to_prior, cos.tax_discount)),
            )
            for cos in (
                _normalize_cost_of_sales_from_asr(item) for item in contract2_asr
            )
        ]

        # Concatenating the zeros_to_new and years_to_new to the contract2 Land and Building Tax
        new_lbt_2nd = [
            LBT(
                start_year=start_date_trans.year,
                end_year=end_date_trans.year,
                cost=np.concatenate((zeros_to_prior, bt.cost)),
                expense_year=np.concatenate((years_to_prior, bt.expense_year)).astype(
                    int
                ),
                cost_allocation=fluid_to_prior + bt.cost_allocation,
                tax_portion=np.concatenate((zeros_to_prior, bt.tax_portion)),
                tax_discount=np.concatenate((zeros_to_prior, bt.tax_discount)),
                final_year=np.concatenate((years_to_prior, bt.final_year)),
                utilized_land_area=np.concatenate(
                    (zeros_to_prior, bt.utilized_land_area)
                ),
                utilized_building_area=np.concatenate(
                    (zeros_to_prior, bt.utilized_building_area)
                ),
                njop_land=np.concatenate((zeros_to_prior, bt.njop_land)),
                njop_building=np.concatenate((zeros_to_prior, bt.njop_building)),
                gross_revenue=np.concatenate((zeros_to_prior, bt.gross_revenue)),
            )
            for bt in contract2_lbt
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
            cost_of_sales=tuple(new_cost_of_sales_1st),
            lbt=tuple(new_lbt_1st),
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
            cost_of_sales=tuple(new_cost_of_sales_2nd),
            lbt=tuple(new_lbt_2nd),
        )

        # Adjusting the contract arguments
        new_argument_contract1 = adjusting_contract_arguments(
            arguments_dict=self.argument_contract1,
            first_contract=True,
            prior_rows=zeros_to_prior,
            post_rows=zeros_to_new,
        )

        new_argument_contract2 = adjusting_contract_arguments(
            arguments_dict=self.argument_contract2,
            first_contract=False,
            prior_rows=zeros_to_prior,
            post_rows=zeros_to_new,
        )
        # Executing the new contract
        # for i in self.argument_contract1.keys():
        #     print(i, ': ', self.argument_contract1[i])
        # input()

        _run_transition_contract(contract1_new, new_argument_contract1)
        _run_transition_contract(contract2_new, new_argument_contract2)
        contract1_transitioned = contract1_new
        contract2_transitioned = contract2_new
        contract2_pre_transfer_profiles = _TransitionRunProfiles.from_contract(
            contract2_transitioned
        )

        # Defining the unrecoverable cost from the prior contract
        first_consolidated_columns = (
            contract1_transitioned.get_consolidated_table_columns()
        )
        if isinstance(contract1_transitioned, CostRecovery):
            latest_unrec = float(
                _table_float(first_consolidated_columns, "unrec_after_transfer").flat[
                    -1
                ]
            )
        else:
            latest_unrec = float(
                _table_float(
                    first_consolidated_columns, "carry_forward_cost_after_tf"
                ).flat[-1]
            )
        unrec_cost_prior = np.zeros_like(
            contract2_transitioned.project_years, dtype=float
        )
        unrec_cost_prior[len(years_to_prior)] = latest_unrec

        # Calculating the Unrecoverable cost in the transitioned cashflow
        unrec_trans = get_unrecovered_cost(
            revenue=contract2_pre_transfer_profiles.consolidated_taxable_income,
            depreciation=unrec_cost_prior,
            non_capital=np.zeros_like(
                contract2_transitioned.project_years, dtype=float
            ),
            ftp_ctr=np.zeros_like(
                contract2_pre_transfer_profiles.project_years, dtype=float
            ),
            ftp_gov=np.zeros_like(
                contract2_pre_transfer_profiles.project_years, dtype=float
            ),
            ic=np.zeros_like(
                contract2_pre_transfer_profiles.project_years, dtype=float
            ),
        )

        cost_to_be_recovered_trans = get_cost_to_be_recovered(
            unrecovered_cost=unrec_trans
        )

        cost_recovery_trans = CostRecovery.calculate_cost_recovery(
            revenue=contract2_pre_transfer_profiles.consolidated_taxable_income,
            ftp=np.zeros_like(
                contract2_pre_transfer_profiles.project_years, dtype=float
            ),
            ic=np.zeros_like(
                contract2_pre_transfer_profiles.project_years, dtype=float
            ),
            depreciation=unrec_cost_prior,
            non_capital=np.zeros_like(
                contract2_pre_transfer_profiles.project_years, dtype=float
            ),
            cost_to_be_recovered=cost_to_be_recovered_trans,
            cr_cap_rate=1.0,
        )

        # Calculate taxable income after considering transferred unrecoverable cost from contract1
        taxable_income_trans = CostRecovery.calculate_ets_before_transfer(
            revenue=contract2_pre_transfer_profiles.consolidated_taxable_income,
            ftp_ctr=np.zeros_like(
                contract2_pre_transfer_profiles.project_years, dtype=float
            ),
            ftp_gov=np.zeros_like(
                contract2_pre_transfer_profiles.project_years, dtype=float
            ),
            ic=np.zeros_like(
                contract2_pre_transfer_profiles.project_years, dtype=float
            ),
            cost_recovery=cost_recovery_trans,
        )

        # Calculate tax payment of adjusted taxable income
        tax_payment_transition = (
            taxable_income_trans * contract2_pre_transfer_profiles.tax_rate_arr
        )

        # Calculate adjusted consolidated profiles for contract2.
        contractor_net_share_transition = (
            contract2_pre_transfer_profiles.consolidated_taxable_income
            - (unrec_cost_prior * unrec_portion)
            - tax_payment_transition
        )
        cashflow_transition = (
            contract2_pre_transfer_profiles.consolidated_cashflow
            + contract2_pre_transfer_profiles.consolidated_tax_payment
            - tax_payment_transition
        )
        government_take_transition = (
            contract2_pre_transfer_profiles.consolidated_government_take
            - contract2_pre_transfer_profiles.consolidated_tax_payment
            + tax_payment_transition
        )
        contract2_transitioned.apply_transition_adjustment(
            contractor_net_share=contractor_net_share_transition,
            cashflow=cashflow_transition,
            government_take=government_take_transition,
            tax_payment=tax_payment_transition,
        )

        # Parsing new contract to the attributes of Transition dataclass
        self._contract1_transitioned = contract1_transitioned
        self._contract2_transitioned = contract2_transitioned

        # Populating transition attributes
        first_profiles = _TransitionRunProfiles.from_contract(contract1_transitioned)
        second_profiles = _TransitionRunProfiles.from_contract(contract2_transitioned)

        # WAP Price
        self._oil_wap_price = second_profiles.oil_wap_price
        self._gas_wap_price = second_profiles.gas_wap_price
        self._sulfur_wap_price = second_profiles.sulfur_wap_price
        self._electricity_wap_price = second_profiles.electricity_wap_price
        self._co2_wap_price = second_profiles.co2_wap_price

        # Lifting
        self._oil_lifting = first_profiles.oil_lifting + second_profiles.oil_lifting

        self._gas_lifting = first_profiles.gas_lifting + second_profiles.gas_lifting

        self._sulfur_lifting = (
            first_profiles.sulfur_lifting + second_profiles.sulfur_lifting
        )

        self._electricity_lifting = (
            first_profiles.electricity_lifting + second_profiles.electricity_lifting
        )

        self._co2_lifting = first_profiles.co2_lifting + second_profiles.co2_lifting

        # Revenue
        self._oil_revenue = first_profiles.oil_revenue + second_profiles.oil_revenue

        self._gas_revenue = first_profiles.gas_revenue + second_profiles.gas_revenue

        self._sulfur_revenue = (
            first_profiles.sulfur_revenue + second_profiles.sulfur_revenue
        )

        self._electricity_revenue = (
            first_profiles.electricity_revenue + second_profiles.electricity_revenue
        )

        self._co2_revenue = first_profiles.co2_revenue + second_profiles.co2_revenue

        # Tangible
        self._oil_capital_expenditures_post_tax = (
            first_profiles.oil_capital_expenditures_post_tax
            + second_profiles.oil_capital_expenditures_post_tax
        )

        self._gas_capital_expenditures_post_tax = (
            first_profiles.gas_capital_expenditures_post_tax
            + second_profiles.gas_capital_expenditures_post_tax
        )

        self._oil_capital_expenditures_pre_tax = (
            first_profiles.oil_capital_expenditures_pre_tax
            + second_profiles.oil_capital_expenditures_pre_tax
        )

        self._gas_capital_expenditures_pre_tax = (
            first_profiles.gas_capital_expenditures_pre_tax
            + second_profiles.gas_capital_expenditures_pre_tax
        )

        # Intangible
        self._oil_intangible_expenditures_post_tax = (
            first_profiles.oil_intangible_expenditures_post_tax
            + second_profiles.oil_intangible_expenditures_post_tax
        )

        self._gas_intangible_expenditures_post_tax = (
            first_profiles.gas_intangible_expenditures_post_tax
            + second_profiles.gas_intangible_expenditures_post_tax
        )

        self._oil_intangible_expenditures_pre_tax = (
            first_profiles.oil_intangible_expenditures_pre_tax
            + second_profiles.oil_intangible_expenditures_pre_tax
        )

        self._gas_intangible_expenditures_pre_tax = (
            first_profiles.gas_intangible_expenditures_pre_tax
            + second_profiles.gas_intangible_expenditures_pre_tax
        )

        # Opex
        self._oil_opex_expenditures_post_tax = (
            first_profiles.oil_opex_expenditures_post_tax
            + second_profiles.oil_opex_expenditures_post_tax
        )

        self._gas_opex_expenditures_post_tax = (
            first_profiles.gas_opex_expenditures_post_tax
            + second_profiles.gas_opex_expenditures_post_tax
        )

        self._oil_opex_expenditures_pre_tax = (
            first_profiles.oil_opex_expenditures_pre_tax
            + second_profiles.oil_opex_expenditures_pre_tax
        )

        self._gas_opex_expenditures_pre_tax = (
            first_profiles.gas_opex_expenditures_pre_tax
            + second_profiles.gas_opex_expenditures_pre_tax
        )

        # ASR
        self._oil_asr_expenditures_post_tax = (
            first_profiles.oil_asr_expenditures_post_tax
            + second_profiles.oil_asr_expenditures_post_tax
        )

        self._gas_asr_expenditures_post_tax = (
            first_profiles.gas_asr_expenditures_post_tax
            + second_profiles.gas_asr_expenditures_post_tax
        )

        self._oil_asr_expenditures_pre_tax = (
            first_profiles.oil_asr_expenditures_pre_tax
            + second_profiles.oil_asr_expenditures_pre_tax
        )

        self._gas_asr_expenditures_pre_tax = (
            first_profiles.gas_asr_expenditures_pre_tax
            + second_profiles.gas_asr_expenditures_pre_tax
        )

        # Cost Of Sales
        self._oil_cost_of_sales_expenditures_post_tax = (
            first_profiles.oil_cost_of_sales_expenditures_post_tax
            + second_profiles.oil_cost_of_sales_expenditures_post_tax
        )

        self._gas_cost_of_sales_expenditures_post_tax = (
            first_profiles.gas_cost_of_sales_expenditures_post_tax
            + second_profiles.gas_cost_of_sales_expenditures_post_tax
        )

        self._oil_cost_of_sales_expenditures_pre_tax = (
            first_profiles.oil_cost_of_sales_expenditures_post_tax
            + second_profiles.oil_cost_of_sales_expenditures_post_tax
        )

        self._gas_cost_of_sales_expenditures_pre_tax = (
            first_profiles.gas_cost_of_sales_expenditures_pre_tax
            + second_profiles.gas_cost_of_sales_expenditures_pre_tax
        )

        # Land and Building Tax
        self._oil_lbt_expenditures_post_tax = (
            first_profiles.oil_lbt_expenditures_post_tax
            + second_profiles.oil_lbt_expenditures_post_tax
        )

        self._gas_lbt_expenditures_post_tax = (
            first_profiles.gas_lbt_expenditures_post_tax
            + second_profiles.gas_lbt_expenditures_post_tax
        )

        self._oil_lbt_expenditures_pre_tax = (
            first_profiles.oil_lbt_expenditures_pre_tax
            + second_profiles.oil_lbt_expenditures_pre_tax
        )

        self._gas_lbt_expenditures_pre_tax = (
            first_profiles.gas_lbt_expenditures_pre_tax
            + second_profiles.gas_lbt_expenditures_pre_tax
        )

        # Indirect Taxes
        self._oil_capital_indirect_tax = (
            first_profiles.oil_capital_indirect_tax
            + second_profiles.oil_capital_indirect_tax
        )

        self._gas_capital_indirect_tax = (
            first_profiles.gas_capital_indirect_tax
            + second_profiles.gas_capital_indirect_tax
        )

        self._oil_intangible_indirect_tax = (
            first_profiles.oil_intangible_indirect_tax
            + second_profiles.oil_intangible_indirect_tax
        )

        self._gas_intangible_indirect_tax = (
            first_profiles.gas_intangible_indirect_tax
            + second_profiles.gas_intangible_indirect_tax
        )

        self._oil_opex_indirect_tax = (
            first_profiles.oil_opex_indirect_tax + second_profiles.oil_opex_indirect_tax
        )

        self._gas_opex_indirect_tax = (
            first_profiles.gas_opex_indirect_tax + second_profiles.gas_opex_indirect_tax
        )

        self._oil_asr_indirect_tax = (
            first_profiles.oil_asr_indirect_tax + second_profiles.oil_asr_indirect_tax
        )

        self._gas_asr_indirect_tax = (
            first_profiles.gas_asr_indirect_tax + second_profiles.gas_asr_indirect_tax
        )

        self._oil_lbt_indirect_tax = (
            first_profiles.oil_lbt_indirect_tax + second_profiles.oil_lbt_indirect_tax
        )

        self._gas_lbt_indirect_tax = (
            first_profiles.gas_lbt_indirect_tax + second_profiles.gas_lbt_indirect_tax
        )

        self._oil_cost_of_sales_indirect_tax = (
            first_profiles.oil_cost_of_sales_indirect_tax
            + second_profiles.oil_cost_of_sales_indirect_tax
        )

        self._gas_cost_of_sales_indirect_tax = (
            first_profiles.gas_cost_of_sales_indirect_tax
            + second_profiles.gas_cost_of_sales_indirect_tax
        )

        # Total indirect taxes for OIL and GAS
        self._oil_total_indirect_tax = (
            self._oil_capital_indirect_tax
            + self._oil_intangible_indirect_tax
            + self._oil_opex_indirect_tax
            + self._oil_asr_indirect_tax
            + self._oil_lbt_indirect_tax
            + self._oil_cost_of_sales_indirect_tax
        )

        self._gas_total_indirect_tax = (
            self._gas_capital_indirect_tax
            + self._gas_intangible_indirect_tax
            + self._gas_opex_indirect_tax
            + self._gas_asr_indirect_tax
            + self._gas_lbt_indirect_tax
            + self._gas_cost_of_sales_indirect_tax
        )

        # Undepreciated Asset
        self._oil_undepreciated_asset = (
            first_profiles.oil_undepreciated_asset
            + second_profiles.oil_undepreciated_asset
        )
        self._gas_undepreciated_asset = (
            first_profiles.gas_undepreciated_asset
            + second_profiles.gas_undepreciated_asset
        )
        self._consolidated_undepreciated_asset = (
            self._oil_undepreciated_asset + self._gas_undepreciated_asset
        )

        # Sunk Cost
        self._oil_sunk_cost = (
            first_profiles.oil_sunk_cost + second_profiles.oil_sunk_cost
        )
        self._gas_sunk_cost = (
            first_profiles.gas_sunk_cost + second_profiles.gas_sunk_cost
        )

        first_scheme = _TransitionSchemeProfiles.from_contract(
            contract1_transitioned,
            include_gross_split_government_share=False,
        )
        second_scheme = _TransitionSchemeProfiles.from_contract(
            contract2_transitioned,
            include_gross_split_government_share=True,
        )

        _oil_deductible_cost_1 = first_scheme.oil_deductible_cost
        _gas_deductible_cost_1 = first_scheme.gas_deductible_cost
        _oil_deductible_cost_2 = second_scheme.oil_deductible_cost
        _gas_deductible_cost_2 = second_scheme.gas_deductible_cost
        _oil_unrec_cost_2 = second_scheme.oil_unrecovered_cost
        _gas_unrec_cost_2 = second_scheme.gas_unrecovered_cost
        _oil_ctr_ets_1 = first_scheme.oil_contractor_ets
        _gas_ctr_ets_1 = first_scheme.gas_contractor_ets
        _oil_ctr_ets_2 = second_scheme.oil_contractor_ets
        _gas_ctr_ets_2 = second_scheme.gas_contractor_ets
        _oil_gov_ets_1 = first_scheme.oil_government_ets
        _gas_gov_ets_1 = first_scheme.gas_government_ets
        _oil_gov_ets_2 = second_scheme.oil_government_ets
        _gas_gov_ets_2 = second_scheme.gas_government_ets
        _oil_ctr_ftp_1 = first_scheme.oil_contractor_ftp
        _gas_ctr_ftp_1 = first_scheme.gas_contractor_ftp
        _oil_ctr_ftp_2 = second_scheme.oil_contractor_ftp
        _gas_ctr_ftp_2 = second_scheme.gas_contractor_ftp
        _oil_gov_ftp_1 = first_scheme.oil_government_ftp
        _gas_gov_ftp_1 = first_scheme.gas_government_ftp
        _oil_gov_ftp_2 = second_scheme.oil_government_ftp
        _gas_gov_ftp_2 = second_scheme.gas_government_ftp
        _ctr_ftp_1 = first_scheme.contractor_ftp
        _gov_ftp_1 = first_scheme.government_ftp
        _ctr_ftp_2 = second_scheme.contractor_ftp
        _gov_ftp_2 = second_scheme.government_ftp
        _oil_effective_tax_payment_1 = first_scheme.oil_tax_payment
        _gas_effective_tax_payment_1 = first_scheme.gas_tax_payment
        _oil_effective_tax_payment_2 = second_scheme.oil_tax_payment
        _gas_effective_tax_payment_2 = second_scheme.gas_tax_payment
        _oil_ctr_take_1 = first_scheme.oil_contractor_take
        _gas_ctr_take_1 = first_scheme.gas_contractor_take
        _oil_ctr_take_2 = second_scheme.oil_contractor_take
        _gas_ctr_take_2 = second_scheme.gas_contractor_take
        _oil_cashflow_1 = first_scheme.oil_cashflow
        _gas_cashflow_1 = first_scheme.gas_cashflow
        _oil_cashflow_2 = second_scheme.oil_cashflow
        _gas_cashflow_2 = second_scheme.gas_cashflow

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

        # Effective Tax Payment
        self._oil_effective_tax_payment = (
            _oil_effective_tax_payment_1 + _oil_effective_tax_payment_2
        )
        self._gas_effective_tax_payment = (
            _gas_effective_tax_payment_1 + _gas_effective_tax_payment_2
        )

        # DDMO
        self._oil_ddmo = first_profiles.oil_ddmo + second_profiles.oil_ddmo
        self._gas_ddmo = first_profiles.gas_ddmo + second_profiles.gas_ddmo

        # Contractor Take
        self._oil_ctr_take = _oil_ctr_take_1 + _oil_ctr_take_2
        self._gas_ctr_take = _gas_ctr_take_1 + _gas_ctr_take_2

        # Cashflow
        self._oil_cashflow = _oil_cashflow_1 + _oil_cashflow_2
        self._gas_cashflow = _gas_cashflow_1 + _gas_cashflow_2

        # Contractor Net Operating Profit
        self._net_operating_profit = (
            first_profiles.consolidated_ctr_net_share
            + second_profiles.consolidated_ctr_net_share
        )

        # Contractor Net Cash Flow
        self._ctr_net_cashflow = (
            first_profiles.consolidated_cashflow + second_profiles.consolidated_cashflow
        )

        # FTP
        self._oil_ftp_ctr = _oil_ctr_ftp_1 + _oil_ctr_ftp_2
        self._gas_ftp_ctr = _gas_ctr_ftp_1 + _gas_ctr_ftp_2
        self._oil_ftp_gov = _oil_gov_ftp_1 + _oil_gov_ftp_2
        self._gas_ftp_gov = _gas_gov_ftp_1 + _gas_gov_ftp_2
        self._ctr_ftp = _ctr_ftp_1 + _ctr_ftp_2
        self._gov_ftp = _gov_ftp_1 + _gov_ftp_2

        # DDMO
        self._ddmo = (
            first_profiles.consolidated_ddmo + second_profiles.consolidated_ddmo
        )

        # Tax
        self._tax_payment = (
            first_profiles.consolidated_tax_payment
            + second_profiles.consolidated_tax_payment
        )

        # Government Take
        self._oil_government_take = (
            first_profiles.oil_government_take + second_profiles.oil_government_take
        )
        self._gas_government_take = (
            first_profiles.gas_government_take + second_profiles.gas_government_take
        )

        self._government_take = (
            first_profiles.consolidated_government_take
            + second_profiles.consolidated_government_take
        )

        # Consolidated Attributes
        # Consolidated Revenue
        self._consolidated_revenue = self._oil_revenue + self._gas_revenue

        # Consolidated Sunk Cost
        self._consolidated_sunk_cost = self._oil_sunk_cost + self._gas_sunk_cost

        # Consolidated Cash Flow
        self._consolidated_cashflow = (
            first_profiles.consolidated_cashflow + second_profiles.consolidated_cashflow
        )

        # Consolidated DDMO
        self._consolidated_ddmo = (
            first_profiles.consolidated_ddmo + second_profiles.consolidated_ddmo
        )

        # Consolidated Tax Payment
        self._consolidated_tax_payment = (
            first_profiles.consolidated_tax_payment
            + second_profiles.consolidated_tax_payment
        )

        # Consolidated Government Take
        self._consolidated_government_take = (
            first_profiles.consolidated_government_take
            + second_profiles.consolidated_government_take
        )

        # Consolidated Indirect Taxes
        self._consolidated_indirect_tax = (
            self._oil_total_indirect_tax + self._gas_total_indirect_tax
        )

        # Project Years
        self.project_years = np.copy(contract1_transitioned.project_years)

    def get_post_run_summary_profile(self) -> PostRunSummaryProfile:
        """Return named reporting data from a completed contract run."""
        _ = self.get_transitioned_contracts()
        return PostRunSummaryProfile(
            contract_type="transition",
            project_years=_require_project_years(self.project_years),
            oil_lifting=_require_lifting(self._oil_lifting, "_oil_lifting"),
            gas_lifting=_require_lifting(self._gas_lifting, "_gas_lifting"),
            sulfur_lifting=_require_lifting(self._sulfur_lifting, "_sulfur_lifting"),
            electricity_lifting=_require_lifting(
                self._electricity_lifting, "_electricity_lifting"
            ),
            co2_lifting=_require_lifting(self._co2_lifting, "_co2_lifting"),
            oil_revenue=_require_float_array(self._oil_revenue, "_oil_revenue"),
            gas_revenue=_require_float_array(self._gas_revenue, "_gas_revenue"),
            sulfur_revenue=_require_float_array(
                self._sulfur_revenue, "_sulfur_revenue"
            ),
            electricity_revenue=_require_float_array(
                self._electricity_revenue, "_electricity_revenue"
            ),
            co2_revenue=_require_float_array(self._co2_revenue, "_co2_revenue"),
            gas_wap_price=_require_float_array(self._gas_wap_price, "_gas_wap_price"),
            oil_capital_expenditures_pre_tax=_require_float_array(
                self._oil_capital_expenditures_pre_tax,
                "_oil_capital_expenditures_pre_tax",
            ),
            oil_intangible_expenditures_pre_tax=_require_float_array(
                self._oil_intangible_expenditures_pre_tax,
                "_oil_intangible_expenditures_pre_tax",
            ),
            oil_opex_expenditures_pre_tax=_require_float_array(
                self._oil_opex_expenditures_pre_tax, "_oil_opex_expenditures_pre_tax"
            ),
            oil_asr_expenditures_pre_tax=_require_float_array(
                self._oil_asr_expenditures_pre_tax, "_oil_asr_expenditures_pre_tax"
            ),
            oil_lbt_expenditures_pre_tax=_require_float_array(
                self._oil_lbt_expenditures_pre_tax, "_oil_lbt_expenditures_pre_tax"
            ),
            oil_cost_of_sales_expenditures_pre_tax=_require_float_array(
                self._oil_cost_of_sales_expenditures_pre_tax,
                "_oil_cost_of_sales_expenditures_pre_tax",
            ),
            gas_capital_expenditures_pre_tax=_require_float_array(
                self._gas_capital_expenditures_pre_tax,
                "_gas_capital_expenditures_pre_tax",
            ),
            gas_intangible_expenditures_pre_tax=_require_float_array(
                self._gas_intangible_expenditures_pre_tax,
                "_gas_intangible_expenditures_pre_tax",
            ),
            gas_opex_expenditures_pre_tax=_require_float_array(
                self._gas_opex_expenditures_pre_tax, "_gas_opex_expenditures_pre_tax"
            ),
            gas_asr_expenditures_pre_tax=_require_float_array(
                self._gas_asr_expenditures_pre_tax, "_gas_asr_expenditures_pre_tax"
            ),
            gas_lbt_expenditures_pre_tax=_require_float_array(
                self._gas_lbt_expenditures_pre_tax, "_gas_lbt_expenditures_pre_tax"
            ),
            gas_cost_of_sales_expenditures_pre_tax=_require_float_array(
                self._gas_cost_of_sales_expenditures_pre_tax,
                "_gas_cost_of_sales_expenditures_pre_tax",
            ),
            oil_capital_expenditures_post_tax=_require_float_array(
                self._oil_capital_expenditures_post_tax,
                "_oil_capital_expenditures_post_tax",
            ),
            oil_intangible_expenditures_post_tax=_require_float_array(
                self._oil_intangible_expenditures_post_tax,
                "_oil_intangible_expenditures_post_tax",
            ),
            oil_opex_expenditures_post_tax=_require_float_array(
                self._oil_opex_expenditures_post_tax, "_oil_opex_expenditures_post_tax"
            ),
            oil_asr_expenditures_post_tax=_require_float_array(
                self._oil_asr_expenditures_post_tax, "_oil_asr_expenditures_post_tax"
            ),
            oil_lbt_expenditures_post_tax=_require_float_array(
                self._oil_lbt_expenditures_post_tax, "_oil_lbt_expenditures_post_tax"
            ),
            oil_cost_of_sales_expenditures_post_tax=_require_float_array(
                self._oil_cost_of_sales_expenditures_post_tax,
                "_oil_cost_of_sales_expenditures_post_tax",
            ),
            gas_capital_expenditures_post_tax=_require_float_array(
                self._gas_capital_expenditures_post_tax,
                "_gas_capital_expenditures_post_tax",
            ),
            gas_intangible_expenditures_post_tax=_require_float_array(
                self._gas_intangible_expenditures_post_tax,
                "_gas_intangible_expenditures_post_tax",
            ),
            gas_opex_expenditures_post_tax=_require_float_array(
                self._gas_opex_expenditures_post_tax, "_gas_opex_expenditures_post_tax"
            ),
            gas_asr_expenditures_post_tax=_require_float_array(
                self._gas_asr_expenditures_post_tax, "_gas_asr_expenditures_post_tax"
            ),
            gas_lbt_expenditures_post_tax=_require_float_array(
                self._gas_lbt_expenditures_post_tax, "_gas_lbt_expenditures_post_tax"
            ),
            gas_cost_of_sales_expenditures_post_tax=_require_float_array(
                self._gas_cost_of_sales_expenditures_post_tax,
                "_gas_cost_of_sales_expenditures_post_tax",
            ),
            oil_capital_indirect_tax=_require_float_array(
                self._oil_capital_indirect_tax, "_oil_capital_indirect_tax"
            ),
            oil_intangible_indirect_tax=_require_float_array(
                self._oil_intangible_indirect_tax, "_oil_intangible_indirect_tax"
            ),
            oil_opex_indirect_tax=_require_float_array(
                self._oil_opex_indirect_tax, "_oil_opex_indirect_tax"
            ),
            oil_asr_indirect_tax=_require_float_array(
                self._oil_asr_indirect_tax, "_oil_asr_indirect_tax"
            ),
            oil_lbt_indirect_tax=_require_float_array(
                self._oil_lbt_indirect_tax, "_oil_lbt_indirect_tax"
            ),
            oil_cost_of_sales_indirect_tax=_require_float_array(
                self._oil_cost_of_sales_indirect_tax, "_oil_cost_of_sales_indirect_tax"
            ),
            gas_capital_indirect_tax=_require_float_array(
                self._gas_capital_indirect_tax, "_gas_capital_indirect_tax"
            ),
            gas_intangible_indirect_tax=_require_float_array(
                self._gas_intangible_indirect_tax, "_gas_intangible_indirect_tax"
            ),
            gas_opex_indirect_tax=_require_float_array(
                self._gas_opex_indirect_tax, "_gas_opex_indirect_tax"
            ),
            gas_asr_indirect_tax=_require_float_array(
                self._gas_asr_indirect_tax, "_gas_asr_indirect_tax"
            ),
            gas_lbt_indirect_tax=_require_float_array(
                self._gas_lbt_indirect_tax, "_gas_lbt_indirect_tax"
            ),
            gas_cost_of_sales_indirect_tax=_require_float_array(
                self._gas_cost_of_sales_indirect_tax, "_gas_cost_of_sales_indirect_tax"
            ),
            oil_sunk_cost=_require_float_array(self._oil_sunk_cost, "_oil_sunk_cost"),
            gas_sunk_cost=_require_float_array(self._gas_sunk_cost, "_gas_sunk_cost"),
            oil_cashflow=_require_float_array(self._oil_cashflow, "_oil_cashflow"),
            gas_cashflow=_require_float_array(self._gas_cashflow, "_gas_cashflow"),
            oil_total_indirect_tax=_require_float_array(
                self._oil_total_indirect_tax, "_oil_total_indirect_tax"
            ),
            gas_total_indirect_tax=_require_float_array(
                self._gas_total_indirect_tax, "_gas_total_indirect_tax"
            ),
            consolidated_revenue=_require_float_array(
                self._consolidated_revenue, "_consolidated_revenue"
            ),
            consolidated_sunk_cost=_require_float_array(
                self._consolidated_sunk_cost, "_consolidated_sunk_cost"
            ),
            consolidated_cashflow=_require_float_array(
                self._consolidated_cashflow, "_consolidated_cashflow"
            ),
            oil_deductible_cost=_require_float_array(
                self._oil_deductible_cost, "_oil_deductible_cost"
            ),
            gas_deductible_cost=_require_float_array(
                self._gas_deductible_cost, "_gas_deductible_cost"
            ),
            oil_unrec_cost=_require_float_array(
                self._oil_unrec_cost, "_oil_unrec_cost"
            ),
            gas_unrec_cost=_require_float_array(
                self._gas_unrec_cost, "_gas_unrec_cost"
            ),
            oil_ctr_ets=_require_float_array(self._oil_ctr_ets, "_oil_ctr_ets"),
            gas_ctr_ets=_require_float_array(self._gas_ctr_ets, "_gas_ctr_ets"),
            oil_gov_ets=_require_float_array(self._oil_gov_ets, "_oil_gov_ets"),
            gas_gov_ets=_require_float_array(self._gas_gov_ets, "_gas_gov_ets"),
            oil_effective_tax_payment=_require_float_array(
                self._oil_effective_tax_payment, "_oil_effective_tax_payment"
            ),
            gas_effective_tax_payment=_require_float_array(
                self._gas_effective_tax_payment, "_gas_effective_tax_payment"
            ),
            oil_ddmo=_require_float_array(self._oil_ddmo, "_oil_ddmo"),
            gas_ddmo=_require_float_array(self._gas_ddmo, "_gas_ddmo"),
            oil_ctr_take=_require_float_array(self._oil_ctr_take, "_oil_ctr_take"),
            gas_ctr_take=_require_float_array(self._gas_ctr_take, "_gas_ctr_take"),
            oil_government_take=_require_float_array(
                self._oil_government_take, "_oil_government_take"
            ),
            gas_government_take=_require_float_array(
                self._gas_government_take, "_gas_government_take"
            ),
            ctr_ftp=_require_float_array(self._ctr_ftp, "_ctr_ftp"),
            gov_ftp=_require_float_array(self._gov_ftp, "_gov_ftp"),
            ctr_net_cashflow=_require_float_array(
                self._ctr_net_cashflow, "_ctr_net_cashflow"
            ),
            net_operating_profit=_require_float_array(
                self._net_operating_profit, "_net_operating_profit"
            ),
            consolidated_ddmo=_require_float_array(
                self._consolidated_ddmo, "_consolidated_ddmo"
            ),
            consolidated_tax_payment=_require_float_array(
                self._consolidated_tax_payment, "_consolidated_tax_payment"
            ),
            consolidated_government_take=_require_float_array(
                self._consolidated_government_take, "_consolidated_government_take"
            ),
        )

    def get_transitioned_contracts(
        self,
    ) -> tuple[CostRecovery | GrossSplit, CostRecovery | GrossSplit]:
        first = self._contract1_transitioned
        second = self._contract2_transitioned
        if first is None or second is None:
            raise TransitionException(
                "Transition.run() must be called before table generation"
            )
        return first, second

    def get_fiscal_profiles(self) -> TransitionFiscalProfiles:
        """Return validated fiscal outputs from a completed transition run."""

        _ = self.get_transitioned_contracts()
        return TransitionFiscalProfiles(
            tax_payment=_require_float_array(
                self._consolidated_tax_payment, "consolidated tax payment"
            ),
            cashflow=_require_float_array(
                self._consolidated_cashflow, "consolidated cashflow"
            ),
            government_take=_require_float_array(
                self._consolidated_government_take, "consolidated government take"
            ),
            oil_undepreciated_asset=_require_transition_scalar(
                self._oil_undepreciated_asset, "oil undepreciated asset"
            ),
            gas_undepreciated_asset=_require_transition_scalar(
                self._gas_undepreciated_asset, "gas undepreciated asset"
            ),
            consolidated_undepreciated_asset=_require_transition_scalar(
                self._consolidated_undepreciated_asset,
                "consolidated undepreciated asset",
            ),
            oil_government_ets=_require_float_array(
                self._oil_gov_ets, "oil government ETS"
            ),
            gas_government_ets=_require_float_array(
                self._gas_gov_ets, "gas government ETS"
            ),
            government_ftp=_require_float_array(self._gov_ftp, "government FTP"),
        )
