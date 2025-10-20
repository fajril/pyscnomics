"""
Collection of functions to administer sensitivity analysis.

This Code is made to detached the sensitivity module in the previous version
which depended with the excel into fully able to be run in python.
"""

import numpy as np
import pandas as pd

from pyscnomics.econ import CostOfSales, Lifting, FluidType
# from pyscnomics.io.aggregator import Aggregate
# from pyscnomics.optimize.adjuster import AdjustData

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT
from pyscnomics.tools.summary import get_summary


class SensitivityException(Exception):
    """ Exception to be raised for a misuse of Sensitivity Method """

    pass


def _get_multipliers(
    min_deviation: float,
    max_deviation: float,
    base_value: float = 1.0,
    step: int = 10,
) -> np.ndarray:
    """
    Generate an array of multipliers symmetrically distributed around a base value.

    The multipliers are created using linear spacing between
    ``base_value - min_deviation`` and ``base_value + max_deviation``.

    The lower and upper ranges are combined into a single one-dimensional
    array, excluding the duplicated base value at the center.

    Parameters
    ----------
    min_deviation : float
        The deviation below the base value.
        Must be non-negative.
    max_deviation : float
        The deviation above the base value.
        Must be non-negative.
    base_value : float, optional
        The central reference value around which the multipliers are generated.
        Default is ``1.0``.
    step : int, optional
        The number of intervals used to generate linearly spaced values
        on each side of the base value. Must be greater than zero.
        Default is ``10``.

    Returns
    -------
    np.ndarray
        A one-dimensional NumPy array containing the combined multipliers.
        The array starts from ``base_value - min_deviation`` and ends at
        ``base_value + max_deviation``.

    Raises
    ------
    SensitivityException
        If ``min_deviation`` or ``max_deviation`` is negative, or if ``step``
        is less than or equal to zero.
    """

    # Throw exceptions for unsuitable input data
    if min_deviation < 0 or max_deviation < 0:
        raise SensitivityException("Deviations must be non negative.")

    if step <= 0:
        raise SensitivityException("Step must be greater than zero.")

    # Specify the minimum and maximum values
    min_val = base_value - min_deviation
    max_val = base_value + max_deviation

    # Create an array of minimum and maximum values
    lower_mults = np.linspace(min_val, base_value, step + 1)
    upper_mults = np.linspace(base_value, max_val, step + 1)

    # Combine `min_multipliers` with `max_multipliers into a single array`
    return np.concatenate((lower_mults, upper_mults[1:]))


def _prepare_adjusted_parameters_single_contract(
    contract: BaseProject | CostRecovery | GrossSplit,
    adjustment_value: float,
    element: str,
) -> dict:
    """
    Prepare adjusted economic parameters for a single contract instance.

    This function applies a proportional adjustment to one category of cost or
    production-related data within a given contract object. The specific parameter
    to adjust (CAPEX, OPEX, oil/gas price, or oil/gas lifting rate) is controlled
    by the ``element`` argument. All other parameters remain unchanged.

    The function supports contract types based on the Indonesian Production
    Sharing Contract (PSC) schemes: ``BaseProject``, ``CostRecovery``, and ``GrossSplit``.

    Parameters
    ----------
    contract : BaseProject or CostRecovery or GrossSplit
        The contract instance containing economic data to be adjusted. The object
        must define attributes such as ``capital_cost``, ``intangible_cost``,
        ``opex``, and ``lifting``.
    adjustment_value : float
        The multiplier applied to the selected parameter. For instance, ``1.1``
        represents a +10% increase, while ``0.9`` represents a −10% decrease.
    element : {"CAPEX", "OPEX", "OILPRICE", "GASPRICE", "OILLIFTING", "GASLIFTING"}
        The target economic component to adjust:

        - ``"CAPEX"`` — Scales both capital and intangible costs.
        - ``"OPEX"`` — Scales operating costs (fixed and per-volume).
        - ``"OILPRICE"`` — Scales the oil price component in the lifting data.
        - ``"GASPRICE"`` — Scales the gas price component in the lifting data.
        - ``"OILLIFTING"`` — Scales the oil lifting rate component.
        - ``"GASLIFTING"`` — Scales the gas lifting rate component.

    Returns
    -------
    dict
        A dictionary of adjusted contract components containing the same structure
        as the original contract attributes:
        ```python
        {
            "capital": tuple[CapitalCost],
            "intangible": tuple[Intangible],
            "opex": tuple[OPEX],
            "lifting": tuple[Lifting],
        }
        ```
        Each tuple contains new instances of the corresponding data class, updated
        according to the specified adjustment factor.

    Notes
    -----
    - The adjustment is **multiplicative**, not additive.
    - Unselected components are returned unchanged.
    - This function is designed for **economic sensitivity analysis** in upstream
      oil and gas modeling, where parameter variation is required to assess project
      robustness.
    - Internally, a helper function ``_adjust_lifting()`` is used to handle lifting-
      related modifications efficiently.
    """

    # Default values
    capital_adjusted = contract.capital_cost
    intangible_adjusted = contract.intangible_cost
    opex_adjusted = contract.opex
    lifting_adjusted = contract.lifting

    # Helper function to assist lifting adjustment
    def _adjust_lifting(fluid_type: FluidType, target_field: str):
        return tuple(
            [
                Lifting(
                    start_year=lft.start_year,
                    end_year=lft.end_year,
                    lifting_rate=(
                        lft.lifting_rate * adjustment_value
                        if target_field == "lifting_rate" and lft.fluid_type is fluid_type
                        else lft.lifting_rate
                    ),
                    price=(
                        lft.price * adjustment_value
                        if target_field == "price" and lft.fluid_type is fluid_type
                        else lft.price
                    ),
                    prod_year=lft.prod_year,
                    fluid_type=lft.fluid_type,
                    ghv=lft.ghv,
                    # prod_rate=lft.prod_rate,
                    prod_rate_baseline=lft.prod_rate_baseline,
                )
                for lft in contract.lifting
            ]
        )

    # Adjustment to CAPEX cost (capital + intangible costs),
    # modify the `cost` attribute of class CapitalCost and Intangible
    if element == "CAPEX":
        capital_adjusted = tuple(
            [
                CapitalCost(
                    start_year=cap.start_year,
                    end_year=cap.end_year,
                    expense_year=cap.expense_year,
                    cost=cap.cost * adjustment_value,
                    cost_allocation=cap.cost_allocation,
                    cost_type=cap.cost_type,
                    description=cap.description,
                    tax_portion=cap.tax_portion,
                    tax_discount=cap.tax_discount,
                    pis_year=cap.pis_year,
                    salvage_value=cap.salvage_value,
                    useful_life=cap.useful_life,
                    depreciation_factor=cap.depreciation_factor,
                    is_ic_applied=cap.is_ic_applied,
                )
                for cap in contract.capital_cost
            ]
        )

        intangible_adjusted = tuple(
            [
                Intangible(
                    start_year=intang.start_year,
                    end_year=intang.end_year,
                    cost=intang.cost * adjustment_value,
                    cost_type=intang.cost_type,
                    expense_year=intang.expense_year,
                    cost_allocation=intang.cost_allocation,
                    description=intang.description,
                    tax_portion=intang.tax_portion,
                    tax_discount=intang.tax_discount,
                )
                for intang in contract.intangible_cost
            ]
        )

    # Adjustment to OPEX cost, modify the `cost` attribute of class OPEX
    elif element == "OPEX":
        opex_adjusted = tuple(
            [
                OPEX(
                    start_year=opx.start_year,
                    end_year=opx.end_year,
                    expense_year=opx.expense_year,
                    cost_allocation=opx.cost_allocation,
                    cost_type=opx.cost_type,
                    description=opx.description,
                    tax_portion=opx.tax_portion,
                    tax_discount=opx.tax_discount,
                    fixed_cost=opx.fixed_cost * adjustment_value,
                    prod_rate=opx.prod_rate,
                    cost_per_volume=opx.cost_per_volume * adjustment_value,
                )
                for opx in contract.opex
            ]
        )

    # Adjustment to oil price; modify the `price` attribute of class Lifting
    elif element == "OILPRICE":
        lifting_adjusted = _adjust_lifting(fluid_type=FluidType.OIL, target_field="price")

    # Adjustment to gas price; modify the `price` attribute of class Lifting
    elif element == "GASPRICE":
        lifting_adjusted = _adjust_lifting(fluid_type=FluidType.GAS, target_field="price")

    # Adjustment to oil lifting; modify the `lifting_rate` attribute of class Lifting
    elif element == "OILLIFTING":
        lifting_adjusted = _adjust_lifting(
            fluid_type=FluidType.OIL, target_field="lifting_rate"
        )

    # Adjustment to gas lifting; modify the `lifting_rate` attribute of class Lifting
    elif element == "GASLIFTING":
        lifting_adjusted = _adjust_lifting(
            fluid_type=FluidType.GAS, target_field="lifting_rate"
        )

    else:
        raise SensitivityException(f"Invalid element: {element!r}")

    return {
        "capital": capital_adjusted,
        "intangible": intangible_adjusted,
        "opex": opex_adjusted,
        "lifting": lifting_adjusted,
    }


def _prepare_onstream_dates_single_contract(
    contract: BaseProject | CostRecovery | GrossSplit
) -> dict:
    """
    Prepare oil and gas onstream dates for a single contract instance.

    This function determines the onstream dates for oil and gas by checking
    whether the corresponding revenue attributes (`_oil_revenue` and `_gas_revenue`)
    contain nonzero values. If the total revenue for a fluid type is zero,
    the onstream date is set to ``None``. Otherwise, the onstream date is retrieved
    from the contract's respective onstream date attribute.

    Parameters
    ----------
    contract : BaseProject or CostRecovery or GrossSplit
        The contract instance containing oil and gas revenue information,
        as well as the corresponding onstream date attributes.

    Returns
    -------
    dict
        A dictionary containing the oil and gas onstream dates, with the following keys:

        - ``"oil"`` : datetime or None
          The oil onstream date, or ``None`` if `_oil_revenue` equals zero.
        - ``"gas"`` : datetime or None
          The gas onstream date, or ``None`` if `_gas_revenue` equals zero.

    Notes
    -----
    - The function dynamically accesses contract attributes using :func:`getattr`.
    - It assumes the contract object exposes attributes:
      ``_oil_revenue``, ``_gas_revenue``, ``oil_onstream_date``, and ``gas_onstream_date``.
    """
    def _get_onstream_date(target_str: str, default: str):
        value = getattr(contract, target_str)
        total = value.sum() if hasattr(value, "sum") else value
        return None if total == 0 else getattr(contract, default)

    return {
        "oil": _get_onstream_date(target_str="_oil_revenue", default="oil_onstream_date"),
        "gas": _get_onstream_date(target_str="_gas_revenue", default="gas_onstream_date"),
    }


def _adjust_element_single_contract(
    contract: BaseProject | CostRecovery | GrossSplit,
    contract_arguments: dict,
    element: str,
    adjustment_value: float,
    run_contract: bool = False,
):
    """
    Adjusts a single parameter of a PSC contract and creates a new adjusted contract instance.

    This function modifies a specified element (e.g., lifting, cost, or onstream parameter)
    for a given contract instance (`BaseProject`, `CostRecovery`, or `GrossSplit`).
    It reconstructs a new contract with updated parameters and optionally executes
    the economic run with the provided contract arguments.

    Parameters
    ----------
    contract : BaseProject or CostRecovery or GrossSplit
        The contract instance to be adjusted.
    contract_arguments : dict
        Arguments required for running the contract model.
        Passed to the contract's ``run()`` method if ``run_contract`` is ``True``.
    element : str
        The name of the parameter element to be adjusted.
        Typically refers to a key in the contract parameter dictionary.
    adjustment_value : float
        The adjustment factor or value applied to the specified element.
    run_contract : bool, default=False
        Whether to execute the contract immediately after adjustment.

    Returns
    -------
    BaseProject or CostRecovery or GrossSplit
        A new contract instance with adjusted parameters. If ``run_contract=True``,
        the returned instance will contain post-run results.
    """

    # Prepare adjusted parameters
    params_adjusted = _prepare_adjusted_parameters_single_contract(
        contract=contract,
        adjustment_value=adjustment_value,
        element=element,
    )

    # Prepare onstream dates
    onstream_date = _prepare_onstream_dates_single_contract(contract=contract)

    # Specify the required arguments to create an instance of BaseProject
    base_project_kwargs = {
        # Base parameters
        "start_date": contract.start_date,
        "end_date": contract.end_date,
        "oil_onstream_date": onstream_date["oil"],
        "gas_onstream_date": onstream_date["gas"],
        "approval_year": contract.approval_year,
        "is_pod_1": contract.is_pod_1,

        # Lifting and costs
        "lifting": params_adjusted["lifting"],
        "capital_cost": params_adjusted["capital"],
        "intangible_cost": params_adjusted["intangible"],
        "opex": params_adjusted["opex"],
        "asr_cost": contract.asr_cost,
        "lbt_cost": contract.lbt_cost,
        "cost_of_sales": contract.cost_of_sales,
    }

    # Create a new instance of BaseProject
    if isinstance(contract, BaseProject):
        contract_adjusted = BaseProject(**base_project_kwargs)

    # Create a new instance of CostRecovery
    elif isinstance(contract, CostRecovery):

        # Specify the required arguments to create an instance of CostRecovery
        cost_recovery_kwargs = {
            # Base parameters, lifting, and costs
            **base_project_kwargs,

            # FTP
            "oil_ftp_is_available": contract.oil_ftp_is_available,
            "oil_ftp_is_shared": contract.oil_ftp_is_shared,
            "oil_ftp_portion": contract.oil_ftp_portion,
            "gas_ftp_is_available": contract.gas_ftp_is_available,
            "gas_ftp_is_shared": contract.gas_ftp_is_shared,
            "gas_ftp_portion": contract.gas_ftp_portion,

            # Split
            "tax_split_type": contract.tax_split_type,
            "condition_dict": contract.condition_dict,
            "indicator_rc_icp_sliding": contract.indicator_rc_icp_sliding,
            "oil_ctr_pretax_share": contract.oil_ctr_pretax_share,
            "gas_ctr_pretax_share": contract.gas_ctr_pretax_share,

            # Investment credit and cap rate
            "oil_ic_rate": contract.oil_ic_rate,
            "gas_ic_rate": contract.gas_ic_rate,
            "ic_is_available": contract.ic_is_available,
            "oil_cr_cap_rate": contract.oil_cr_cap_rate,
            "gas_cr_cap_rate": contract.gas_cr_cap_rate,

            # DMO
            "oil_dmo_volume_portion": contract.oil_dmo_volume_portion,
            "oil_dmo_fee_portion": contract.oil_dmo_fee_portion,
            "oil_dmo_holiday_duration": contract.oil_dmo_holiday_duration,
            "gas_dmo_volume_portion": contract.gas_dmo_volume_portion,
            "gas_dmo_fee_portion": contract.gas_dmo_fee_portion,
            "gas_dmo_holiday_duration": contract.gas_dmo_holiday_duration,

            # Carry forward depreciation
            "oil_carry_forward_depreciation": contract.oil_carry_forward_depreciation,
            "gas_carry_forward_depreciation": contract.gas_carry_forward_depreciation,
        }

        contract_adjusted = CostRecovery(**cost_recovery_kwargs)

    # Create a new instance of GrossSplit
    elif isinstance(contract, GrossSplit):

        # Specify the required arguments to create an instance of GrossSplit
        gross_split_kwargs = {
            # Base parameters, lifting, and costs
            **base_project_kwargs,

            # Field and reservoir properties
            "field_status": contract.field_status,
            "field_loc": contract.field_loc,
            "res_depth": contract.res_depth,
            "infra_avail": contract.infra_avail,
            "res_type": contract.res_type,
            "api_oil": contract.api_oil,
            "domestic_use": contract.domestic_use,
            "prod_stage": contract.prod_stage,
            "co2_content": contract.co2_content,
            "h2s_content": contract.h2s_content,
            "field_reserves_2024": contract.field_reserves_2024,
            "infra_avail_2024": contract.infra_avail_2024,
            "field_loc_2024": contract.field_loc_2024,
            "split_ministry_disc": contract.split_ministry_disc,

            # DMO parameters
            "oil_dmo_volume_portion": contract.oil_dmo_volume_portion,
            "oil_dmo_fee_portion": contract.oil_dmo_fee_portion,
            "oil_dmo_holiday_duration": contract.oil_dmo_holiday_duration,
            "gas_dmo_volume_portion": contract.gas_dmo_volume_portion,
            "gas_dmo_fee_portion": contract.gas_dmo_fee_portion,
            "gas_dmo_holiday_duration": contract.gas_dmo_holiday_duration,

            # Carry forward depreciation
            "oil_carry_forward_depreciation": contract.oil_carry_forward_depreciation,
            "gas_carry_forward_depreciation": contract.gas_carry_forward_depreciation,
        }

        contract_adjusted = GrossSplit(**gross_split_kwargs)

    else:
        raise SensitivityException(
            f"Invalid contract type: {contract.__class__.__qualname__}"
        )

    if run_contract:
        contract_adjusted.run(**contract_arguments)

    return contract_adjusted


def _adjust_element_single_contract_old(
    contract: BaseProject | CostRecovery | GrossSplit,
    contract_arguments: dict,
    element: str,
    adjustment_value: float,
    run_contract: bool = False,
) -> CostRecovery | GrossSplit:
    """
    The function to adjust the element of the contract based on the adjustment_value.

    Parameters
    ----------
    contract: BaseProject | CostRecovery | GrossSplit | Transition
        The contract which will be adjusted.
    contract_arguments: dict
        The contract arguments.
    element: str
        The element of the contract that will be adjusted.
    adjustment_value: float
        The adjustment value which will be multiplied by the corresponding element.
    run_contract:bool
        The option to run the contract or not.

    Returns
    -------
    contract_adjusted: CostRecovery | GrossSplit
        The contract that has been adjusted.
    """

    capital_adjusted = contract.capital_cost
    intangible_adjusted = contract.intangible_cost
    opex_adjusted = contract.opex
    asr_adjusted = contract.asr_cost
    lbt_adjusted = contract.lbt_cost
    cos_adjusted = contract.cost_of_sales
    lifting_adjusted = contract.lifting

    if element == "CAPEX":
        # Adjusting the Capital Cost of the contract
        capital_adjusted = tuple(
            [
                CapitalCost(
                    start_year=tan.start_year,
                    end_year=tan.end_year,
                    cost=tan.cost * adjustment_value,
                    expense_year=tan.expense_year,
                    cost_allocation=tan.cost_allocation,
                    description=tan.description,
                    tax_portion=tan.tax_portion,
                    tax_discount=tan.tax_discount,
                    pis_year=tan.pis_year,
                    salvage_value=tan.salvage_value,
                    useful_life=tan.useful_life,
                    depreciation_factor=tan.depreciation_factor,
                    is_ic_applied=tan.is_ic_applied,
                )
                for tan in contract.capital_cost
            ]
        )

        # Adjusting the Intangible cost of the contract
        intangible_adjusted = tuple(
            [
                Intangible(
                    start_year=intang.start_year,
                    end_year=intang.end_year,
                    cost=intang.cost * adjustment_value,
                    expense_year=intang.expense_year,
                    cost_allocation=intang.cost_allocation,
                    description=intang.description,
                    tax_portion=intang.tax_portion,
                    tax_discount=intang.tax_discount,
                )
                for intang in contract.intangible_cost
            ]
        )

    elif element == "OPEX":
        # Adjusting the OPEX cost of the contract
        opex_adjusted = tuple(
            [
                OPEX(
                    start_year=opx.start_year,
                    end_year=opx.end_year,
                    expense_year=opx.expense_year,
                    cost_allocation=opx.cost_allocation,
                    description=opx.description,
                    tax_portion=opx.tax_portion,
                    tax_discount=opx.tax_discount,
                    fixed_cost=opx.fixed_cost * adjustment_value,
                    prod_rate=opx.prod_rate,
                    cost_per_volume=opx.cost_per_volume,
                )
                for opx in contract.opex
            ]
        )

        # Adjusting the ASR cost of the contract
        asr_adjusted = tuple(
            [
                ASR(
                    start_year=asr.start_year,
                    end_year=asr.end_year,
                    cost=asr.cost * adjustment_value,
                    expense_year=asr.expense_year,
                    cost_allocation=asr.cost_allocation,
                    description=asr.description,
                    tax_portion=asr.tax_portion,
                    tax_discount=asr.tax_discount,
                    final_year=asr.final_year,
                    future_rate=asr.future_rate,
                )
                for asr in contract.asr_cost
            ]
        )

        # Adjusting the LBT of the contract
        lbt_adjusted = tuple(
            [
                LBT(
                    start_year=bt.start_year,
                    end_year=bt.end_year,
                    expense_year=bt.expense_year,
                    cost=bt.cost * adjustment_value,
                    cost_allocation=bt.cost_allocation,
                    description=bt.description,
                    tax_portion=bt.tax_portion,
                    tax_discount=bt.tax_discount,
                    final_year=bt.final_year,
                    utilized_land_area=bt.utilized_land_area,
                    utilized_building_area=bt.utilized_building_area,
                    njop_land=bt.njop_land,
                    njop_building=bt.njop_building,
                    gross_revenue=bt.gross_revenue,
                )
                for bt in contract.lbt_cost
            ]
        )

        # Adjusting the Intangible cost of the contract
        cos_adjusted = tuple(
            [
                CostOfSales(
                    start_year=cos.start_year,
                    end_year=cos.end_year,
                    cost=cos.cost * adjustment_value,
                    expense_year=cos.expense_year,
                    cost_allocation=cos.cost_allocation,
                    description=cos.description,
                    tax_portion=cos.tax_portion,
                    tax_discount=cos.tax_discount,
                )
                for cos in contract.cost_of_sales
            ]
        )

    elif element == "OILPRICE":
        # Adjusting the oil price
        lifting_adjusted = tuple(
            [
                Lifting(
                    start_year=lift.start_year,
                    end_year=lift.end_year,
                    lifting_rate=lift.lifting_rate,
                    price=(
                        lift.price * adjustment_value
                        if lift.fluid_type is FluidType.OIL
                        else lift.price
                    ),
                    prod_year=lift.prod_year,
                    fluid_type=lift.fluid_type,
                    ghv=lift.ghv,
                    prod_rate=lift.prod_rate,
                    prod_rate_baseline=lift.prod_rate_baseline,
                )
                for lift in contract.lifting
            ]
        )

    elif element == "GASPRICE":
        # Adjusting the oil price
        lifting_adjusted = tuple(
            [
                Lifting(
                    start_year=lift.start_year,
                    end_year=lift.end_year,
                    lifting_rate=lift.lifting_rate,
                    price=(
                        lift.price * adjustment_value
                        if lift.fluid_type is FluidType.GAS
                        else lift.price
                    ),
                    prod_year=lift.prod_year,
                    fluid_type=lift.fluid_type,
                    ghv=lift.ghv,
                    prod_rate=lift.prod_rate,
                    prod_rate_baseline=lift.prod_rate_baseline,
                )
                for lift in contract.lifting
            ]
        )

    elif element == "LIFTING":
        # Adjusting the lifting
        lifting_adjusted = tuple(
            [
                Lifting(
                    start_year=lift.start_year,
                    end_year=lift.end_year,
                    lifting_rate=lift.lifting_rate * adjustment_value,
                    price=lift.price,
                    prod_year=lift.prod_year,
                    fluid_type=lift.fluid_type,
                    ghv=lift.ghv,
                    # prod_rate=lift.prod_rate,
                    # prod_rate is being by passed in the routine of sensitivity,
                    # thus it will be filled with the lifting value
                    prod_rate_baseline=lift.prod_rate_baseline,
                )
                for lift in contract.lifting
            ]
        )

    elif element == "OILLIFTING":
        # Adjusting the lifting
        lifting_adjusted = tuple(
            [
                Lifting(
                    start_year=lift.start_year,
                    end_year=lift.end_year,
                    lifting_rate=(
                        lift.lifting_rate * adjustment_value
                        if lift.fluid_type is FluidType.OIL
                        else lift.lifting_rate
                    ),
                    price=lift.price,
                    prod_year=lift.prod_year,
                    fluid_type=lift.fluid_type,
                    ghv=lift.ghv,
                    # prod_rate=lift.prod_rate,
                    # prod_rate is being by passed in the routine of sensitivity,
                    # thus it will be filled with the lifting value
                    prod_rate_baseline=lift.prod_rate_baseline,
                )
                for lift in contract.lifting
            ]
        )

    elif element == "GASLIFTING":
        # Adjusting the lifting
        lifting_adjusted = tuple(
            [
                Lifting(
                    start_year=lift.start_year,
                    end_year=lift.end_year,
                    lifting_rate=(
                        lift.lifting_rate * adjustment_value
                        if lift.fluid_type is FluidType.GAS
                        else lift.lifting_rate
                    ),
                    price=lift.price,
                    prod_year=lift.prod_year,
                    fluid_type=lift.fluid_type,
                    ghv=lift.ghv,
                    # prod_rate=lift.prod_rate,
                    # prod_rate is being by passed in the routine of sensitivity,
                    # thus it will be filled with the lifting value
                    prod_rate_baseline=lift.prod_rate_baseline,
                )
                for lift in contract.lifting
            ]
        )

    else:
        raise SensitivityException(f"The element value, {element}, is not recognized.")

    # Adjust the contract with the adjusted element
    # On stream date treatment
    if np.sum(contract._oil_revenue) == 0:
        oil_onstream_date = None
    else:
        oil_onstream_date = contract.oil_onstream_date

    if np.sum(contract._gas_revenue) == 0:
        gas_onstream_date = None
    else:
        gas_onstream_date = contract.gas_onstream_date

    # When the contract is CostRecovery, parsing back the adjusted cost elements
    # to the cost recovery contract
    if isinstance(contract, CostRecovery):
        contract_adjusted = CostRecovery(
            start_date=contract.start_date,
            end_date=contract.end_date,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=lifting_adjusted,
            capital_cost=capital_adjusted,
            intangible_cost=intangible_adjusted,
            opex=opex_adjusted,
            asr_cost=asr_adjusted,
            lbt_cost=lbt_adjusted,
            cost_of_sales=cos_adjusted,
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

            oil_carry_forward_depreciation=contract.oil_carry_forward_depreciation,
            gas_carry_forward_depreciation=contract.gas_carry_forward_depreciation,
        )

    # When the contract is GrossSplit, parsing back the adjusted cost elements
    # to the gross split contract
    elif isinstance(contract, GrossSplit):
        contract_adjusted = GrossSplit(
            start_date=contract.start_date,
            end_date=contract.end_date,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=contract.lifting,
            capital_cost=capital_adjusted,
            intangible_cost=intangible_adjusted,
            opex=opex_adjusted,
            asr_cost=asr_adjusted,
            lbt_cost=lbt_adjusted,
            cost_of_sales=cos_adjusted,
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

            oil_carry_forward_depreciation=contract.oil_carry_forward_depreciation,
            gas_carry_forward_depreciation=contract.gas_carry_forward_depreciation,
        )

    elif isinstance(contract, BaseProject):
        contract_adjusted = BaseProject(
            start_date=contract.start_date,
            end_date=contract.end_date,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=contract.lifting,
            capital_cost=capital_adjusted,
            intangible_cost=intangible_adjusted,
            opex=opex_adjusted,
            asr_cost=asr_adjusted,
            lbt_cost=lbt_adjusted,
            cost_of_sales=cos_adjusted,
        )

    # When the contract is not recognized, raise an exception
    else:
        raise SensitivityException(
            f"The contract type, {type(contract)}, is not recognized."
        )
    if run_contract is True:
        contract_adjusted.run(**contract_arguments)
    else:
        pass

    return contract_adjusted


def _adjust_element_transition_contract(
    contract: Transition,
    contract_arguments: dict,
    element: str,
    adjustment_value: float,
    run_contract: bool = False,
) -> Transition:
    """
    The function to adjust the element of the transition contract based on the adjustment_value.

    Parameters
    ----------
    contract: Transition
        The contract which will be adjusted.
    contract_arguments: dict
        The contract arguments.
    element: str
        The element of the contract that will be adjusted.
    adjustment_value: float
        The adjustment value which will be multiplied by the corresponding element.
    run_contract:bool
        The option to run the contract or not.

    Returns
    -------
    contract_adjusted: Transition
        The contract that has been adjusted.
    """
    # Retrieving the first contract
    contract_1_adjusted = _adjust_element_single_contract(
        contract=contract.contract1,
        contract_arguments=contract.argument_contract1,
        element=element,
        adjustment_value=adjustment_value,
        run_contract=run_contract,
    )

    # Retrieving the second contract
    contract_2_adjusted = _adjust_element_single_contract(
        contract=contract.contract2,
        contract_arguments=contract.argument_contract2,
        element=element,
        adjustment_value=adjustment_value,
        run_contract=run_contract,
    )

    # Reconstructing the transition contract
    contract_adjusted = Transition(
        contract1=contract_1_adjusted,
        contract2=contract_2_adjusted,
        argument_contract1=contract.argument_contract1,
        argument_contract2=contract.argument_contract2,
    )

    # Condition when the run_contract is true
    if run_contract is True:
        contract_adjusted.run(**contract_arguments)
    else:
        pass

    return contract_adjusted


def _adjust_contract(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    contract_arguments: dict,
    element: str,
    adjustment_value: float,
    run_contract: bool = False,
):
    """
    Function act as the wrapper to get sensitivity of a contract.

    Parameters
    ----------
    contract: BaseProject | CostRecovery | GrossSplit | Transition
        The contract which will be adjusted.
    contract_arguments: dict
        The contract arguments.
    element: str
        The element of the contract that will be adjusted.
    adjustment_value: float
        The adjustment value which will be multiplied by the corresponding element.
    run_contract:bool
        The option to run the contract or not.

    Returns
    -------

    """
    if isinstance(contract, Transition):
        return _adjust_element_transition_contract(
            contract=contract,
            contract_arguments=contract_arguments,
            element=element,
            adjustment_value=adjustment_value,
            run_contract=run_contract,
        )
    else:
        return _adjust_element_single_contract(
            contract=contract,
            contract_arguments=contract_arguments,
            element=element,
            adjustment_value=adjustment_value,
            run_contract=run_contract,
        )


def sensitivity_psc(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    contract_arguments: dict,
    summary_arguments: dict,
    min_deviation: float,
    max_deviation: float,
    base_value: float = 1.0,
    step: int = 10,
    dataframe_output: bool = True,
) -> dict | pd.DataFrame:
    """
    The function to get the sensitivity analysis of a contract.

    Parameters
    ----------
    contract: BaseProject | CostRecovery | Gross Split | Transition
        The contract that the sensitivity will be retrieved.
    contract_arguments: dict
        The contract arguments of the contract.
    summary_arguments: dict
        The summary arguments.
    min_deviation: float
        The minimum deviation from the base value.
    max_deviation: float
        The maximum deviation from the base value.
    base_value: float, optional
        The base value for the multipliers. Default is 1.0.
    step: int, optional
        The number of steps to create multipliers. Default is 10.
    dataframe_output: bool
        The option whether the output in a dataframe form or dictionary.
    Returns
    -------
    out: dict | pd.DataFrame
        The sensitivity result
    """

    # Get the multipliers
    multipliers = _get_multipliers(
        min_deviation=min_deviation,
        max_deviation=max_deviation,
        base_value=base_value,
        step=step,
    )

    # Adjust the element of the contract and contain it in a list
    psc_adjusted_dict = {
        element: {
            mul: _adjust_contract(
                contract=contract,
                contract_arguments=contract_arguments,
                element=element,
                adjustment_value=mul,
                run_contract=True,
            )
            for mul in multipliers
        }
        for element in ["CAPEX", "OPEX", "OILPRICE", "GASPRICE", "OILLIFTING", "GASLIFTING"]
    }

    # print('\t')
    # print(f'Filetype: {type(psc_adjusted_dict["CAPEX"][0.8])}')
    # print(f'Length: {len(psc_adjusted_dict["CAPEX"][0.8])}')
    # print('psc_adjusted_dict = \n', psc_adjusted_dict["CAPEX"][0.8])

    # Get the summary of each contract in psc_adjusted_dict and contain it in a dictionary
    summary_adjusted_dict = {
        element: {
            mul: psc_adjusted_dict[element][mul].get_summary(**summary_arguments)
            for mul in psc_adjusted_dict[element]
        }
        for element in psc_adjusted_dict.keys()
    }

    print('\t')
    print(f'Filetype: {type(summary_adjusted_dict["CAPEX"][1.2])}')
    print(f'Length: {len(summary_adjusted_dict["CAPEX"][1.2])}')
    print('summary_adjusted_dict["CAPEX"][1.2] = \n', summary_adjusted_dict["CAPEX"][1.2])

    # summary_adjusted_dict = {
    #     element: {
    #         mul: get_summary(
    #             **{**summary_arguments, "contract": psc_adjusted_dict[element][mul]}
    #         )
    #         for mul in psc_adjusted_dict[element]
    #     }
    #     for element in psc_adjusted_dict.keys()
    # }

    # Retrieve the value for NPV, IRR, PI, POT, Gov Take, and Contractor Share
    indicator_list = [
        "ctr_npv",
        "ctr_irr",
        "ctr_pi",
        "ctr_pot",
        "gov_take",
        "ctr_net_share",
    ]

    # Transform summary_adjusted_dict into the following structure
    sensitivity_result = {
        indicator: {
            multiplier: {
                element: summary_adjusted_dict[element][multiplier].get(indicator, None)
                for element in summary_adjusted_dict
            }
            for multiplier in summary_adjusted_dict[next(iter(summary_adjusted_dict))]
        }
        for indicator in indicator_list
    }

    # print('\t')
    # print(f'Filetype: {type(sensitivity_result)}')
    # print(f'Length: {len(sensitivity_result)}')
    # print('sensitivity_result = \n', sensitivity_result)

    if dataframe_output is True:
        # Transform result dictionary into DataFrames
        sensitivity_result_df = {
            indicator: pd.DataFrame.from_dict(data, orient="index")
            .reset_index()
            .rename(columns={"index": "Factor"})
            .set_index("Factor")
            for indicator, data in sensitivity_result.items()
        }

        print('\t')
        print(f'Filetype: {type(sensitivity_result_df)}')
        print(f'Length: {len(sensitivity_result_df)}')
        print('sensitivity_result_df = \n', sensitivity_result_df)

        # return sensitivity_result_df

    # else:
    #     return sensitivity_result
