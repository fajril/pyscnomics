"""
Handles calculations associated with PSC Gross Split.
"""

import logging
import warnings
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts import psc_tools
from pyscnomics.econ.costs import (
    CapitalCost,
    Intangible,
    OPEX,
    ASR,
    LBT,
    CostOfSales,
)
from pyscnomics.econ.selection import (
    FluidType,
    GrossSplitRegime,
    TaxRegime,
    DeprMethod,
    OtherRevenue,
    InitialYearAmortizationIncurred,
    InitialYearDepreciationIncurred,
    InflationAppliedTo,
    VariableSplit522017,
    VariableSplit082017,
    VariableSplit132024,
    SunkCostMethod,
    NPVSelection,
    DiscountingMode,
    ContractType,
)
from pyscnomics.econ.depreciation import unit_of_production_rate, unit_of_production_book_value
from pyscnomics.econ.indicator import (
    irr,
    npv_nominal_terms,
    npv_real_terms,
    npv_skk_nominal_terms,
    npv_skk_real_terms,
    npv_point_forward,
    pot_psc,
)

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)


class GrossSplitException(Exception):
    """ Exception to be raised for a misuse of GrossSplit class """

    pass


class SunkCostException(Exception):
    """ Exception to be raised for a misuse of sunkcost method """

    pass


class CumulativeProductionSplitException(Exception):
    """ Exception to be raised for a misuse of Cumulative Production Split Offset """

    pass


class GrossSplitSummaryException(Exception):
    """ Exception to be raised for a misuse of get_summary() method """

    pass


@dataclass
class GrossSplit(BaseProject):
    """
    Dataclass that represents Gross Split (GS) contract.

    Parameters
    ----------
    field_status:
        Status of the field development (e.g., "No POD", "POD I").
    field_loc:
        Location of the field (e.g., onshore, offshore shallow, offshore deepwater).
    res_depth:
        Reservoir depth category in meters.
    infra_avail:
        Infrastructure availability near the field (e.g., "Well Developed", "Limited").
    res_type:
        Reservoir type (conventional or unconventional).
    api_oil:
        API gravity of produced oil, expressed in categories (e.g., "<25", "25–<40").
    domestic_use :
        Portion of production allocated for domestic use, in percentage categories.
    prod_stage :
        Production stage (primary, secondary, or tertiary recovery).
    co2_content :
        CO₂ content of produced hydrocarbons, in mole percentage categories.
    h2s_content :
        H₂S content of produced hydrocarbons, in ppm categories.
    field_reserves_2024 :
        Field reserves category under the 2024 regulation.
    infra_avail_2024 :
        Infrastructure availability category under the 2024 regulation.
    field_loc_2024 :
        Field location category under the 2024 regulation.
    split_ministry_disc : float, default=0.0
        Ministry discretion adjustment to contractor split (fraction of revenue).
    oil_dmo_volume_portion : float or np.ndarray, default=0.25
        Portion of oil production subject to DMO (Domestic Market Obligation).
    oil_dmo_fee_portion : float or np.ndarray, default=1.0
        Fraction of oil price paid as DMO fee during the holiday period.
    oil_dmo_holiday_duration : int, default=60
        Duration of the oil DMO holiday in months.
    gas_dmo_volume_portion : float or np.ndarray, default=1.0
        Portion of gas production subject to DMO.
    gas_dmo_fee_portion : float or np.ndarray, default=1.0
        Fraction of gas price paid as DMO fee during the holiday period.
    gas_dmo_holiday_duration : int, default=60
        Duration of the gas DMO holiday in months.
    oil_carry_forward_depreciation : int, float, or np.ndarray, default=0.0
        Unrecovered depreciation carried forward for oil-related costs.
    gas_carry_forward_depreciation : int, float, or np.ndarray, default=0.0
        Unrecovered depreciation carried forward for gas-related costs.
    conversion_boe_to_scf : float, default=5.615
        Conversion factor from barrels of oil equivalent (BOE) to standard
        cubic feet (SCF).
    """

    # Arguments associated with variable split components
    field_status: (
        str
        | VariableSplit522017.FieldStatus
        | VariableSplit082017.FieldStatus
        | None
    ) = field(default="No POD")

    field_loc: (
        str
        | VariableSplit522017.FieldLocation
        | VariableSplit082017.FieldLocation
        | VariableSplit132024.FieldLocation
    ) = field(default="Onshore")

    res_depth: (
        str
        | VariableSplit522017.ReservoirDepth
        | VariableSplit082017.ReservoirDepth
        | None
    ) = field(default="<=2500")

    infra_avail: (
        str
        | VariableSplit522017.InfrastructureAvailability
        | VariableSplit082017.InfrastructureAvailability
        | VariableSplit132024.InfrastructureAvailability
    ) = field(default="Well Developed")

    res_type: (
        str
        | VariableSplit522017.ReservoirType
        | VariableSplit082017.ReservoirType
        | None
    ) = field(default="Conventional")

    api_oil: (
        str
        | VariableSplit522017.APIOil
        | VariableSplit082017.APIOil
        | None
    ) = field(default="<25")

    domestic_use: (
        str
        | VariableSplit522017.DomesticUse
        | VariableSplit082017.DomesticUse
        | None
    ) = field(default="50<=x<70")

    prod_stage: (
        str
        | VariableSplit522017.ProductionStage
        | VariableSplit082017.ProductionStage
        | None
    ) = field(default="Secondary")

    co2_content: (
        str
        | VariableSplit522017.CO2Content
        | VariableSplit082017.CO2Content
        | None
    ) = field(default="<5")

    h2s_content: (
        str
        | VariableSplit522017.H2SContent
        | VariableSplit082017.H2SContent
        | None
    ) = field(default="<100")

    field_reserves_2024: (
        str
        | VariableSplit132024.FieldReservesAmount
        | None
    ) = field(default=VariableSplit132024.FieldReservesAmount.HIGH)

    infra_avail_2024: (
        str
        | VariableSplit132024.InfrastructureAvailability
    ) = field(default="available")

    field_loc_2024: (
        str
        | VariableSplit132024.FieldLocation
    ) = field(default="Onshore")

    # Argument associated with ministry discretion
    split_ministry_disc: float = field(default=0.0)

    # Arguments associated with DMO
    oil_dmo_volume_portion: float | np.ndarray = field(default=0.25)
    oil_dmo_fee_portion: float | np.ndarray = field(default=1.0)
    oil_dmo_holiday_duration: int = field(default=60)
    gas_dmo_volume_portion: float | np.ndarray = field(default=1.0)
    gas_dmo_fee_portion: float | np.ndarray = field(default=1.0)
    gas_dmo_holiday_duration: int = field(default=60)

    # Arguments associated with carry forward depreciation
    oil_carry_forward_depreciation: int | float | np.ndarray = field(default=0.0)
    gas_carry_forward_depreciation: int | float | np.ndarray = field(default=0.0)

    # Attribute associated with unit conversion
    conversion_boe_to_scf: float = field(default=5.615, init=False, repr=False)

    # Attributes associated with carry forward depreciation
    _oil_carry_forward_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _gas_carry_forward_depreciation: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with depreciations and undepreciated assets
    _oil_depreciations: dict = field(default_factory=lambda: {}, init=False, repr=False)
    _gas_depreciations: dict = field(default_factory=lambda: {}, init=False, repr=False)
    _oil_undepreciated_assets: dict = field(default_factory=lambda: {}, init=False, repr=False)
    _gas_undepreciated_assets: dict = field(default_factory=lambda: {}, init=False, repr=False)
    _oil_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _gas_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _oil_sum_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)
    _gas_sum_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with amortization
    _oil_amortizations: dict = field(default_factory=lambda: {}, init=False, repr=False)
    _gas_amortizations: dict = field(default_factory=lambda: {}, init=False, repr=False)
    _oil_amortization: np.ndarray = field(default=None, init=False, repr=False)
    _gas_amortization: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with non-depreciable costs
    _oil_non_depreciables: dict = field(default_factory=lambda: {}, init=False, repr=False)
    _gas_non_depreciables: dict = field(default_factory=lambda: {}, init=False, repr=False)

    # Attributes associated with split
    _cumulative_prod: np.ndarray = field(default=None, init=False, repr=False)
    _oil_base_split: np.ndarray = field(default=None, init=False, repr=False)
    _gas_base_split: np.ndarray = field(default=None, init=False, repr=False)
    _variable_split: float = field(default=0, init=False, repr=False)
    _var_split_array: np.ndarray = field(default=None, init=False, repr=False)
    _oil_prog_price_split: np.ndarray = field(default=None, init=False, repr=False)
    _oil_prog_cum_split: np.ndarray = field(default=None, init=False, repr=False)
    _gas_prog_price_split: np.ndarray = field(default=None, init=False, repr=False)
    _gas_prog_cum_split: np.ndarray = field(default=None, init=False, repr=False)
    _oil_prog_split: np.ndarray = field(default=None, init=False, repr=False)
    _gas_prog_split: np.ndarray = field(default=None, init=False, repr=False)
    _ministry_discretion_arr: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ctr_split: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_split: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with shares
    _oil_ctr_share_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_share_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _oil_gov_share: np.ndarray = field(default=None, init=False, repr=False)
    _gas_gov_share: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with core business logic
    _oil_cost_tobe_deducted: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cost_tobe_deducted: np.ndarray = field(default=None, init=False, repr=False)
    _oil_carward_deduct_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_carward_deduct_cost: np.ndarray = field(default=None, init=False, repr=False)
    _oil_deductible_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_deductible_cost: np.ndarray = field(default=None, init=False, repr=False)
    _transfer_to_oil: np.ndarray = field(default=None, init=False, repr=False)
    _transfer_to_gas: np.ndarray = field(default=None, init=False, repr=False)
    _oil_carward_cost_aftertf: np.ndarray = field(default=None, init=False, repr=False)
    _gas_carward_cost_aftertf: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ctr_share_after_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_share_after_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _oil_net_operating_profit: np.ndarray = field(default=None, init=False, repr=False)
    _gas_net_operating_profit: np.ndarray = field(default=None, init=False, repr=False)
    _oil_profit_pre_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_profit_pre_transfer: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with DMO
    _oil_dmo_volume: np.ndarray = field(default=None, init=False, repr=False)
    _oil_dmo_fee: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ddmo: np.ndarray = field(default=None, init=False, repr=False)
    _gas_dmo_volume: np.ndarray = field(default=None, init=False, repr=False)
    _gas_dmo_fee: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ddmo: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with taxable income
    _oil_taxable_income: np.ndarray = field(default=None, init=False, repr=False)
    _gas_taxable_income: np.ndarray = field(default=None, init=False, repr=False)
    _tax_rate_arr: np.ndarray = field(default=None, init=False, repr=False)
    _oil_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_tax: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with shares
    _oil_ctr_net_share: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_net_share: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ctr_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _oil_government_take: np.ndarray = field(default=None, init=False, repr=False)
    _gas_government_take: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with 100% contractor split warning
    _oil_ctr_split_prior_bracket: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_split_prior_bracket: np.ndarray = field(default=None, init=False, repr=False)
    _oil_year_maximum_ctr_split: np.ndarray = field(default=None, init=False, repr=False)
    _gas_year_maximum_ctr_split: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with consolidated profiles
    _consolidated_carry_forward_depreciation: np.ndarray = field(
        default=None, init=False, repr=False
    )

    _consolidated_depreciations: dict = field(
        default_factory=lambda: {}, init=False, repr=False
    )
    _consolidated_undepreciated_assets: dict = field(
        default_factory=lambda: {}, init=False, repr=False
    )
    _consolidated_amortizations: dict = field(
        default_factory=lambda: {}, init=False, repr=False
    )
    _consolidated_non_depreciables: dict = field(
        default_factory=lambda: {}, init=False, repr=False
    )

    _consolidated_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_amortization: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_sum_undepreciated_asset: np.ndarray = field(
        default=None, init=False, repr=False
    )

    _consolidated_ctr_share_before_tf: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_gov_share_before_tf: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_tobe_deducted: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_carward_deduct_cost: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_deductible_cost: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_carward_cost_aftertf: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_ctr_share_after_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_net_operating_profit: np.ndarray = field(
        default=None, init=False, repr=False
    )

    _consolidated_dmo_volume: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_dmo_fee: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ddmo: np.ndarray = field(default=None, init=False, repr=False)

    _consolidated_taxable_income: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_tax_payment: np.ndarray = field(default=None, init=False, repr=False)

    _consolidated_ctr_net_share: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_government_take: np.ndarray = field(default=None, init=False, repr=False)

    def _check_attributes(self) -> None:
        """
        Validate and normalize project attributes for gross split calculations.

        Checks performed:
        - Fractional attributes are within [0.0, 1.0].
        - Discrete attributes (DMO holiday durations) are non-negative.
        - Carry forward depreciation attributes are numeric, non-negative,
          do not exceed project duration, and are converted to arrays matching
          project years.

        Raises
        ------
        GrossSplitException
            If any attribute violates the above rules.
        """

        # Prepare fractional attributes
        min_fraction = 0.0
        max_fraction = 1.0

        fraction_attributes = (
            ("split_ministry_disc", self.split_ministry_disc),
            ("oil_dmo_volume_portion", self.oil_dmo_volume_portion),
            ("oil_dmo_fee_portion", self.oil_dmo_fee_portion),
            ("gas_dmo_volume_portion", self.gas_dmo_volume_portion),
            ("gas_dmo_fee_portion", self.gas_dmo_fee_portion)
        )

        for attr_name, attr in fraction_attributes:
            if not (min_fraction <= attr <= max_fraction):
                range_type = "exceeding 1.0" if attr > max_fraction else "below 0.0"
                raise GrossSplitException(
                    f"The {attr_name} value ({attr}) is {range_type}. "
                    f"Allowed range: {min_fraction} - {max_fraction}."
                )

        # Prepare attributes associated with DMO holiday
        discrete_attributes = (
            ("oil_dmo_holiday_duration", self.oil_dmo_holiday_duration),
            ("gas_dmo_holiday_duration", self.gas_dmo_holiday_duration)
        )

        for attr_name, attr in discrete_attributes:
            if attr < 0:
                raise GrossSplitException(
                    f"The {attr_name} value, {attr}, is below 0. "
                    f"The minimum value for this attribute is 0."
                )

        # Prepare attributes associated with carry forward depreciation
        carward_depr = {
            "oil_carry_forward_depreciation": self.oil_carry_forward_depreciation,
            "gas_carry_forward_depreciation": self.gas_carry_forward_depreciation,
        }

        for attr_name, value in carward_depr.items():
            # Check for inappropriate data type
            if not isinstance(value, (np.ndarray, float, int)):
                raise GrossSplitException(
                    f"Attribute {attr_name} must be provided in the form of "
                    f"numpy array, float, or integer, not "
                    f"({value.__class__.__qualname__})"
                )

            # Check for negative values
            if np.any(value < 0):
                raise GrossSplitException(f"The {attr_name} contains negative values")

            # Check length exceeding project duration
            if isinstance(value, np.ndarray) and len(value) > self.project_duration:
                raise GrossSplitException(
                    f"The {attr_name} length ({len(value)}) exceeds "
                    f"project duration ({self.project_duration})."
                )

            # Modify 'carward_depr' to array with length equals to project duration
            if isinstance(value, (np.ndarray, float, int)):
                arr = np.atleast_1d(value).astype(float)
                result = np.zeros_like(self.project_years, dtype=float)
                result[:len(arr)] = arr
                carward_depr[attr_name] = result

        self._oil_carry_forward_depreciation = carward_depr["oil_carry_forward_depreciation"]
        self._gas_carry_forward_depreciation = carward_depr["gas_carry_forward_depreciation"]

    def _check_cum_production_split_offset(
        self,
        cum_production_split_offset: float | np.ndarray | None,
    ) -> None:
        """
        Validate the cumulative production split offset.

        Ensures that if a cumulative production split offset is provided as a
        NumPy array, its length matches the project duration.

        Parameters
        ----------
        cum_production_split_offset : float, np.ndarray, or None
            The cumulative production split offset to validate. Can be:
            - A single float value (scalar)
            - A NumPy array of values for each project year
            - None (no offset provided)

        Raises
        ------
        CumulativeProductionSplitException
            If ``cum_production_split_offset`` is a NumPy array and its length
            does not match the project duration.

        Returns
        -------
        None
            This method only validates the input and raises an exception if invalid.
        """

        if isinstance(cum_production_split_offset, np.ndarray):
            if len(cum_production_split_offset) != self.project_duration:
                raise CumulativeProductionSplitException(
                    f"Length of the cum_production_split_offset: "
                    f"({len(cum_production_split_offset)}) "
                    f"is different from the length of project years: "
                    f"{self.project_duration}"
                )

    def _validate_sunkcost_non_pod_1(self) -> None:
        """
        Validate sunk cost arrays for non–POD I contracts.

        This method checks whether all sunk cost components are zero when the
        contract is not POD I. The validation is performed on both depreciable and
        non-depreciable sunk cost arrays for oil and gas.

        In strict mode (``is_strict``), the presence of any non-zero sunk cost in a
        non–POD I contract results in a ``SunkCostException``. In non-strict mode,
        the method instead emits a warning for each cost category found to contain
        non-zero values.

        The following attributes are validated:

        - ``_oil_depreciable_sunk_cost``
        - ``_gas_depreciable_sunk_cost``
        - ``_oil_non_depreciable_sunk_cost``
        - ``_gas_non_depreciable_sunk_cost``

        Notes
        -----
        - POD I contracts are exempt from this validation.
        - The method does not modify any sunk cost or preonstream arrays; it performs
          validation only.
        """

        def _all_zeros(arr: np.ndarray) -> bool:
            return np.all(arr == 0)

        if not self.is_pod_1:
            zero_sunk_costs = {
                "oil_depreciable": _all_zeros(self._oil_depreciable_sunk_cost),
                "gas_depreciable": _all_zeros(self._gas_depreciable_sunk_cost),
                "oil_non_depreciable": _all_zeros(self._oil_non_depreciable_sunk_cost),
                "gas_non_depreciable": _all_zeros(self._gas_non_depreciable_sunk_cost),
            }

            values = zero_sunk_costs.values()

            # Strict mode: raise an error
            if self.is_strict:
                if not all(values):
                    raise SunkCostException(f"Cannot have sunk cost in non-POD I contract.")

            # Loose mode: display a warning message
            else:
                for key, val in zero_sunk_costs.items():
                    if not val:
                        message = f"Found {key} sunk cost for non-POD I contract."
                        logging.warning(message)

    def _check_invalid_non_capital_costs(
        self,
        obj_non_capital: Intangible | OPEX | ASR | LBT | CostOfSales,
        obj_name: str,
        is_strict: bool,
    ) -> None:
        """
        Validate non-capital sunk costs and non-capital preonstream costs for
        PSC Gross Split POD I contracts.

        Checks whether non-capital sunk cost objects contain nonzero values.
        In strict mode, raises an exception; otherwise, records a warning and
        continues execution.

        Parameters
        ----------
        obj_non_capital : Intangible | OPEX | ASR | LBT | CostOfSales
            Non-capital sunk cost object to be validated.
        obj_name : str
            Human-readable name of the cost component.
        is_strict : bool
            If True, raise an error on invalid sunk costs; otherwise, issue a warning.
        """

        # Raise an error: invalid object in "obj_non_capital"
        if not isinstance(obj_non_capital, (Intangible, OPEX, ASR, LBT, CostOfSales)):
            raise GrossSplitException(
                f"Found an invalid object ({obj_non_capital.__class__.__qualname__}). "
                f"Object must be an instance of Intangible/OPEX/ASR/LBT/CostOfSales."
            )

        # Identify nonzero sunk costs
        name = obj_name
        nonzero_mask = (obj_non_capital.cost != 0)

        # Exit early if all sunk costs are zeros
        if not np.any(nonzero_mask):
            return

        # Extract invalid costs alongside their corresponding expense years
        invalid_costs = obj_non_capital.cost[nonzero_mask]
        invalid_years = obj_non_capital.expense_year[nonzero_mask]

        # Pair invalid expense years with invalid costs
        invalid = [f"{yr}: {cst}" for yr, cst in zip(invalid_years, invalid_costs)]

        # Specify messages to be displayed
        msg_error = (
            f"Cannot allow {name!r}: {invalid} in PSC Gross Split (GS) POD I contract. "
            f"Conceptually, all sunk costs in PSC GS POD I should be classified as capital "
            f"cost--with AMORTIZATION APPLIED TO THEM--, not non-capital (Intangible/OPEX/ASR"
            f"LBT/CostOfSales) costs."
        )

        msg_warning = (
            f"Found {name!r}: {invalid} in PSC Gross Split (GS) POD I contract. Conceptually, "
            f"all sunk costs in PSC GS POD I should be classified as capital cost--with "
            f"AMORTIZATION APPLIED TO THEM--, not non-capital (Intangible/OPEX/ASR/LBT/"
            f"CostOfSales) costs. Classifying sunk costs as non-capital costs in PSC GS POD I "
            f"contract may produce an incorrect calculation, where direct charges applied to "
            f"them WITHOUT AMORTIZATION."
        )

        # Strict mode: raise an error and stop execution
        if is_strict:
            raise GrossSplitException(msg_error)

        # Loose mode: record a warning message, validate PIS years with onstream year,
        # and continue execution
        else:
            self.warning_messages.append(
                (ContractType.GROSS_SPLIT.value, msg_warning)
            )

    def _check_no_sunk_costs(
        self,
        sc_object: CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales,
        sc_name: str,
        is_capital: bool,
        is_strict: bool,
        onstream_year: int,
    ) -> None:
        """
        Validate absence of sunk costs for Non-POD I PSC Gross Split contracts.

        Ensures that no capital or non-capital sunk costs are present. In strict
        mode, raises an exception; otherwise, records a warning and optionally
        validates capital PIS years against the onstream year.

        Parameters
        ----------
        sc_object : CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales
            Sunk cost object to be validated.
        sc_name : str
            Human-readable name of the cost component.
        is_capital : bool
            Flag indicating whether the object represents capital cost.
        is_strict : bool
            If True, raise an error on detected sunk costs; otherwise, issue a warning.
        onstream_year : int
            Project onstream year used for capital PIS year validation.
        """

        # Raise an error: invalid data input for "sc_object"
        if not isinstance(sc_object, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            raise GrossSplitException(
                f"Found an invalid sunk cost object ({sc_object.__class__.__qualname__}). "
                f"The object must be an instance of CapitalCost/Intangible/OPEX/ASR/"
                f"LBT/CostOfSales."
            )

        # Identify nonzero sunk costs
        nonzero_mask = (sc_object.cost != 0)

        # Exit early if all sunk costs are zeros
        if not np.any(nonzero_mask):
            return

        # Extract invalid costs and their corresponding expense years
        invalid_costs = sc_object.cost[nonzero_mask]
        invalid_years = sc_object.expense_year[nonzero_mask]

        # Pair invalid expense years with invalid costs
        invalid = [f"{yr}: {cst}" for yr, cst in zip(invalid_years, invalid_costs)]

        # Specify messages to be displayed
        msg_error = (
            f"Cannot allow {sc_name!r}: {invalid} in Non-POD I PSC Gross Split (GS) contract. "
            f"Conceptually, Non-POD I PSC GS MUST NOT contain any sunk costs, either as "
            f"capital or non-capital sunk costs."
        )

        msg_warning = (
            f"Found {sc_name!r}: {invalid} in Non-POD I PSC Gross Split (GS) contract. "
            f"Conceptually, Non-POD I PSC GS SHOULD NOT contain any sunk costs, either as "
            f"capital or non-capital sunk costs. Having sunk costs in Non-POD I GS contract "
            f"leads to incorrect calculation, in which capital sunk costs are subject to "
            f"depreciations and non-capital sunk costs are subject to direct charges."
        )

        # Strict mode: raise an error and stop execution
        if is_strict:
            raise GrossSplitException(msg_error)

        # Loose mode: record a warning message, validate PIS years with onstream year,
        # and continue execution
        else:
            self.warning_messages.append(
                (ContractType.GROSS_SPLIT.value, msg_warning)
            )

            # For capital sunk cost, check whether PIS years < onstream year
            if is_capital:
                self._check_capital_pis_years(
                    obj_capital=sc_object,
                    obj_name=sc_name,
                    is_amortization=False,
                    is_strict=False,
                    onstream_year=onstream_year,
                )

    def _prepare_amortization(self) -> None:
        """
        Validate amortization-related cost classifications under PSC Gross Split.

        Performs amortization-specific validation for POD I contracts only.

        For POD I:
        - Validate capital sunk costs and pre-onstream costs PIS years vs onstream year
        - Validate that non-capital sunk costs and pre-onstream costs are not amortized

        For Non-POD I contracts, no amortization validation is applied.

        Validation behavior respects strict vs non-strict mode.
        """

        # Specify onstream year
        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])

        # Helper method to check whether capital costs PIS years < onstream year
        def _validate_capital_pis_years(mapping):
            for obj, name, is_amort in mapping:
                self._check_capital_pis_years(
                    obj_capital=obj,
                    obj_name=name,
                    is_amortization=is_amort,
                    is_strict=self.is_strict,
                    onstream_year=onstream_yr,
                )

        # Helper method to validate non-capital costs
        def _validate_non_capital_costs(mapping):
            for obj, name in mapping:
                self._check_invalid_non_capital_costs(
                    obj_non_capital=obj,
                    obj_name=name,
                    is_strict=self.is_strict,
                )

        # Define several mapping variables
        mapping_capital_sunk_costs = [
            (self._oil_capital_sunk_cost, "oil_capital_sunk_cost", True),
            (self._gas_capital_sunk_cost, "gas_capital_sunk_cost", True),
        ]

        mapping_capital_preonstreams = [
            (self._oil_capital_preonstream, "oil_capital_preonstream", True),
            (self._gas_capital_preonstream, "gas_capital_preonstream", True),
        ]

        mapping_non_capital_sunk_costs = [
            # Oil
            (self._oil_intangible_sunk_cost, "oil_intangible_sunk_cost"),
            (self._oil_opex_sunk_cost, "oil_opex_sunk_cost"),
            (self._oil_asr_sunk_cost, "oil_asr_sunk_cost"),
            (self._oil_lbt_sunk_cost, "oil_lbt_sunk_cost"),
            (self._oil_cost_of_sales_sunk_cost, "oil_cost_of_sales_sunk_cost"),

            # Gas
            (self._gas_intangible_sunk_cost, "gas_intangible_sunk_cost"),
            (self._gas_opex_sunk_cost, "gas_opex_sunk_cost"),
            (self._gas_asr_sunk_cost, "gas_asr_sunk_cost"),
            (self._gas_lbt_sunk_cost, "gas_lbt_sunk_cost"),
            (self._gas_cost_of_sales_sunk_cost, "gas_cost_of_sales_sunk_cost"),
        ]

        mapping_non_capital_preonstreams = [
            # Oil
            (self._oil_intangible_preonstream, "oil_intangible_preonstream"),
            (self._oil_opex_preonstream, "oil_opex_preonstream"),
            (self._oil_asr_preonstream, "oil_asr_preonstream"),
            (self._oil_lbt_preonstream, "oil_lbt_preonstream"),
            (self._oil_cost_of_sales_preonstream, "oil_cost_of_sales_preonstream"),

            # Gas
            (self._gas_intangible_preonstream, "gas_intangible_preonstream"),
            (self._gas_opex_preonstream, "gas_opex_preonstream"),
            (self._gas_asr_preonstream, "gas_asr_preonstream"),
            (self._gas_lbt_preonstream, "gas_lbt_preonstream"),
            (self._gas_cost_of_sales_preonstream, "gas_cost_of_sales_preonstream"),
        ]

        # Non-POD I Gross Split:
        # Amortization validation is not applicable
        if not self.is_pod_1:
            return

        # POD I Gross Split:
        # 1) Validate capital sunk costs PIS years
        # 2) Validate non-capital sunk costs
        else:
            # Check capital sunk costs: whether PIS years < onstream year
            _validate_capital_pis_years(mapping_capital_sunk_costs)

            # Check capital preonstream costs: whether PIS years < onstream year
            _validate_capital_pis_years(mapping_capital_preonstreams)

            # Check whether sunk costs are treated as non-capital costs
            _validate_non_capital_costs(mapping_non_capital_sunk_costs)

            # Check whether preonstream costs are treated as non-capital costs
            _validate_non_capital_costs(mapping_non_capital_preonstreams)

    def _get_amortization(
        self,
        year_inflation: np.ndarray,
        inflation_rate: np.ndarray | int | float,
        tax_rate: np.ndarray | float,
        inflation_rate_applied_to: InflationAppliedTo,
    ) -> None:
        """
        Compute oil and gas amortization schedules by cost type.

        Initializes amortization arrays for all cost types and fluids. If the
        project is POD-1, sunk and pre-onstream amortizations are calculated
        using unit-of-production logic with inflation and tax adjustments.
        Post-onstream amortization is set to zero.

        Parameters
        ----------
        year_inflation : np.ndarray
            Inflation base years aligned with project years.
        inflation_rate : np.ndarray or float or int
            Inflation rate(s) applied to eligible costs.
        tax_rate : np.ndarray or float
            Indirect tax rate(s).
        inflation_rate_applied_to : InflationAppliedTo
            Specifies which cost components are subject to inflation.

        Returns
        -------
        None
            Updates the following attributes in place:

            - ``_oil_amortizations``
            - ``_gas_amortizations``

        Notes
        -----
        - Only POD-1 projects generate amortization charges.
        - Supported cost types: ``sunk_cost``, ``preonstream``, ``postonstream``.
        - Post-onstream amortization is always zero.
        """
        
        def _zeros():
            return np.zeros_like(self.project_years, dtype=float)

        # Initialize amortization containers for all cost types
        cost_types_all = ["sunk_cost", "preonstream", "postonstream"]
        self._oil_amortizations = {c: _zeros() for c in cost_types_all}
        self._gas_amortizations = {c: _zeros() for c in cost_types_all}

        # Only POD 1 projects generate amortization
        if not self.is_pod_1:
            return

        # Prepare data prior to calculating amortization
        self._prepare_amortization()

        # Mapping: fluid → (cost type, capital cost object, lifting object)
        amort_mapping = {
            "oil": (
                ("sunk_cost", self._oil_capital_sunk_cost, self._oil_lifting),
                ("preonstream", self._oil_capital_preonstream, self._oil_lifting),
            ),
            "gas": (
                ("sunk_cost", self._gas_capital_sunk_cost, self._gas_lifting),
                ("preonstream", self._gas_capital_preonstream, self._gas_lifting),
            ),
        }

        # Define intermediate attribute --> "amortizations": dict
        # Only these cost types generate amortization charges
        cost_types_amort = ["sunk_cost", "preonstream"]
        amortizations = {f: {c: None for c in cost_types_amort} for f in amort_mapping.keys()}

        # Common arguments passed to amortization calculator
        kwargs_amort = {
            "year_inflation": year_inflation,
            "inflation_rate": inflation_rate,
            "tax_rate": tax_rate,
            "inflation_rate_applied_to": inflation_rate_applied_to,
        }

        # Compute amortization per fluid per cost type
        for f, mapping in amort_mapping.items():
            for c, obj_c, obj_l in mapping:
                amortizations[f][c] = obj_c.total_amortization_rate(
                    prod=obj_l.get_lifting_rate_ghv_arr(),
                    **kwargs_amort,
                )

        # Assign computed amortizations; post-onstream remains zero
        self._oil_amortizations = amortizations["oil"]
        self._gas_amortizations = amortizations["gas"]
        self._oil_amortizations["postonstream"] = _zeros()
        self._gas_amortizations["postonstream"] = _zeros()

    def _prepare_depreciation(self) -> None:
        """
        Validate depreciation-related constraints for PSC Gross Split contracts.

        For Non-POD I contracts, this method disallows all sunk costs and
        validates capital pre-onstream PIS years against the onstream year.
        POD I contracts bypass depreciation-specific validation.
        """

        # Specify onstream year
        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])

        # All sunk costs (capital and non-capital) that must NOT exist
        # in Non-POD I Gross Split contracts
        mapping_sunk_costs = [
            # Oil
            (self._oil_capital_sunk_cost, "oil_capital_sunk_cost", True),
            (self._oil_intangible_sunk_cost, "oil_intangible_sunk_cost", False),
            (self._oil_opex_sunk_cost, "oil_opex_sunk_cost", False),
            (self._oil_asr_sunk_cost, "oil_asr_sunk_cost", False),
            (self._oil_lbt_sunk_cost, "oil_lbt_sunk_cost", False),
            (self._oil_cost_of_sales_sunk_cost, "oil_cost_of_sales_sunk_cost", False),

            # Gas
            (self._gas_capital_sunk_cost, "gas_capital_sunk_cost", True),
            (self._gas_intangible_sunk_cost, "gas_intangible_sunk_cost", False),
            (self._gas_opex_sunk_cost, "gas_opex_sunk_cost", False),
            (self._gas_asr_sunk_cost, "gas_asr_sunk_cost", False),
            (self._gas_lbt_sunk_cost, "gas_lbt_sunk_cost", False),
            (self._gas_cost_of_sales_sunk_cost, "gas_cost_of_sales_sunk_cost", False),
        ]

        # Capital pre-onstream costs whose PIS years must not precede onstream year
        mapping_capital_preonstreams = [
            (self._oil_capital_preonstream, "oil_capital_preonstream", False),
            (self._gas_capital_preonstream, "gas_capital_preonstream", False),
        ]

        # POD I Gross Split:
        # Depreciation validation is not applicable
        if self.is_pod_1:
            return

        # Non-POD I Gross Split:
        # 1) Disallow all sunk costs
        # 2) Validate capital pre-onstream PIS years
        else:
            for obj, name, is_cap in mapping_sunk_costs:
                self._check_no_sunk_costs(
                    sc_object=obj,
                    sc_name=name,
                    is_capital=is_cap,
                    is_strict=self.is_strict,
                    onstream_year=onstream_yr,
                )

            for obj, name, is_amort in mapping_capital_preonstreams:
                self._check_capital_pis_years(
                    obj_capital=obj,
                    obj_name=name,
                    is_amortization=is_amort,
                    is_strict=self.is_strict,
                    onstream_year=onstream_yr,
                )

    def _get_depreciation(
        self,
        depr_method: DeprMethod,
        decline_factor: float | int,
        year_inflation: np.ndarray,
        inflation_rate: np.ndarray | int | float,
        tax_rate: np.ndarray | float,
        inflation_rate_applied_to: InflationAppliedTo,
    ) -> None:

        # Preliminary assessments prior to depreciation calculation
        self._prepare_depreciation()

        # Define the mapping between fluids, cost types, and capital objects
        depr_mapping = {
            "oil": [
                ("sunk_cost", self._oil_capital_sunk_cost),
                ("preonstream", self._oil_capital_preonstream),
                ("postonstream", self._oil_capital_postonstream),
            ],
            "gas": [
                ("sunk_cost", self._gas_capital_sunk_cost),
                ("preonstream", self._gas_capital_preonstream),
                ("postonstream", self._gas_capital_postonstream),
            ],
        }

        # Define intermediate attributes:
        # "depreciations": dict and "undepreciated_assets": dict
        fluids = ["oil", "gas"]
        cost_types = ["postonstream", "preonstream", "sunk_cost"]

        depreciations = {f: {c: None for c in cost_types} for f in fluids}
        undepreciated_assets = {f: {c: None for c in cost_types} for f in fluids}

        self._oil_capital_sunk_cost.total_depreciation_rate(
            depr_method=depr_method,
            decline_factor=decline_factor,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            tax_rate=tax_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
        )

        # print('\t')
        # print(f'Filetype: {type(self._oil_capital_sunk_cost)}')
        # print(f'Length: {len(self._oil_capital_sunk_cost)}')
        # print('_oil_capital_sunk_cost = ', self._oil_capital_sunk_cost)

        # (
        #     depreciations["oil"]["sunk_cost"],
        #     undepreciated_assets["oil"]["sunk_cost"]
        # ) = self._oil_capital_sunk_cost.total_depreciation_rate(
        #     depr_method=depr_method,
        #     decline_factor=decline_factor,
        #     year_inflation=year_inflation,
        #     inflation_rate=inflation_rate,
        #     tax_rate=tax_rate,
        #     inflation_rate_applied_to=inflation_rate_applied_to,
        # )

        # for (f, mapping) in depr_mapping.items():
        #     for (c, obj, depr_type) in mapping:
        #         (
        #             depreciations[f][c],
        #             undepreciated_assets[f][c],
        #         ) = obj.total_depreciation_rate(
        #             oil_onstream_year=self.oil_onstream_date.year,
        #             gas_onstream_year=self.gas_onstream_date.year,
        #             initial_depreciation_year=depr_type,
        #             depr_method=depr_method,
        #             decline_factor=decline_factor,
        #             year_inflation=year_inflation,
        #             inflation_rate=inflation_rate,
        #             tax_rate=tax_rate,
        #             inflation_rate_applied_to=inflation_rate_applied_to,
        #         )
        #
        # # Set initial values for attributes "_oil_depreciations" and "_gas_depreciations"
        # self._oil_depreciations = {
        #     c: np.zeros_like(self.project_years, dtype=float) for c in cost_types
        # }
        #
        # self._gas_depreciations = {
        #     c: np.zeros_like(self.project_years, dtype=float) for c in cost_types
        # }
        #
        # # Set initial values for attributes "_oil_undepreciated_assets"
        # # and "_gas_undepreciated_assets"
        # self._oil_undepreciated_assets = {c: None for c in cost_types}
        # self._gas_undepreciated_assets = {c: None for c in cost_types}
        #
        # # Set several attributes according to contract type (POD I or non POD I)
        # if self.is_pod_1:
        #
        #     # Set attributes "_oil_depreciations" and "_gas_depreciations"
        #     self._oil_depreciations["postonstream"] = depreciations["oil"]["postonstream"]
        #     self._gas_depreciations["postonstream"] = depreciations["gas"]["postonstream"]
        #
        #     # Set attributes "_oil_undepreciated_assets" and "gas_undepreciated_assets"
        #     for f in fluids:
        #         for i, j in [
        #             ("postonstream", undepreciated_assets[f]["postonstream"]),
        #             ("preonstream", np.zeros([1, 1], dtype=float)),
        #             ("sunk_cost", np.zeros([1, 1], dtype=float))
        #         ]:
        #             getattr(self, f"_{f}_undepreciated_assets")[i] = j
        #
        # else:
        #
        #     # Set attributes "_oil_depreciations", "_gas_depreciations",
        #     # "_oil_undepreciated_assets", and "gas_undepreciated_assets"
        #     for f, f_depr, f_undepr in [
        #         ("oil", self._oil_depreciations, self._oil_undepreciated_assets),
        #         ("gas", self._gas_depreciations, self._gas_undepreciated_assets),
        #     ]:
        #         for c in cost_types:
        #             f_depr[c] = depreciations[f][c]
        #             f_undepr[c] = undepreciated_assets[f][c]

    def _get_modified_depreciations(self, sum_undepreciated_cost: bool) -> None:
        """
        Modify oil and gas depreciation and undepreciated asset schedules.

        This method applies up to three sequential adjustments to the internal
        depreciation and undepreciated asset arrays for both oil and gas:

        1. **Tolerance cleanup**: Very small undepreciated asset values
           (below a fixed tolerance of ``1e-5``) are set to zero to
           eliminate numerical noise.
        2. **Final-year adjustment** (optional): If
           ``sum_undepreciated_cost=True``, all remaining undepreciated
           balances are summed and transferred into the final year of the
           corresponding depreciation schedule. After this transfer,
           the undepreciated asset arrays are reset to zeros.
        3. **Depreciation reclassification** (*only for non-POD I contracts*):
           If ``is_pod_1`` is False, all ``preonstream`` cost depreciations
           are combined into ``sunk_cost`` depreciations. Once transferred,
           the ``preonstream`` depreciation arrays are reset to zeros.

        Parameters
        ----------
        sum_undepreciated_cost : bool
            If True, any remaining undepreciated costs are added to the last
            project year’s depreciation value and the undepreciated assets
            are cleared. If False, undepreciated balances remain as-is
            (apart from tolerance cleanup).

        Returns
        -------
        None
            The method updates the following attributes in place:

            - ``_oil_depreciations``
            - ``_gas_depreciations``
            - ``_oil_undepreciated_assets``
            - ``_gas_undepreciated_assets``

        Notes
        -----
        - Affects the following cost types: ``postonstream``, ``preonstream``,
          and ``sunk_cost``.
        - The tolerance threshold is fixed at ``1e-5``.
        - Depreciations are updated only in the last year if
          ``sum_undepreciated_cost=True``.
        - For contracts that are **not POD I**, all ``preonstream``
          depreciations are reclassified into ``sunk_cost`` and cleared from
          their original arrays.
        """

        undepre_assets = [self._oil_undepreciated_assets, self._gas_undepreciated_assets]
        cost_types = ["postonstream", "preonstream", "sunk_cost"]

        # Treatment for small order of number (example 1e-5) in undepreciated assets
        tol = 1.0e-5
        for asset in undepre_assets:
            for c in cost_types:
                asset[c][asset[c] < tol] = 0.0

        # Treatment whether the undepreciated asset is summed up in
        # the last year of the contract
        if sum_undepreciated_cost:
            for depr, undepr in [
                (self._oil_depreciations, self._oil_undepreciated_assets),
                (self._gas_depreciations, self._gas_undepreciated_assets)
            ]:
                for c in cost_types:
                    depr[c][-1] += undepr[c].sum()
                    undepr[c] = np.zeros([1, 1], dtype=float)

        # Modify depreciations for non POD I contract
        if not self.is_pod_1:

            # Combine sunk cost and preonstream cost depreciations for non POD I contract,
            # then assign the results as the "modified" sunk cost depreciations
            self._oil_depreciations["sunk_cost"] += self._oil_depreciations["preonstream"]
            self._gas_depreciations["sunk_cost"] += self._gas_depreciations["preonstream"]

            # Assign preonstream cost depreciations as zeros
            self._oil_depreciations["preonstream"] = np.zeros_like(self.project_years, dtype=float)
            self._gas_depreciations["preonstream"] = np.zeros_like(self.project_years, dtype=float)

    def _wrapper_base_split(self, regime: GrossSplitRegime) -> None:
        """
        Assign contractor base split arrays for oil and gas according to the
        selected Gross Split fiscal regime.

        This method validates the input fiscal regime and sets the attributes
        ``_oil_base_split`` and ``_gas_base_split``. Each attribute is an array
        with the same shape as ``project_years``, filled with the corresponding
        contractor base split percentage for oil and gas.

        Parameters
        ----------
        regime : GrossSplitRegime
            Fiscal regime under which the contractor base split is determined.
            Must be an instance of ``GrossSplitRegime``.

        Raises
        ------
        GrossSplitException
            If ``regime`` is not an instance of ``GrossSplitRegime``.
        KeyError
            If the given ``regime`` is not defined in the contractor base split
            mapping.

        Attributes Set
        --------------
        _oil_base_split : numpy.ndarray
            Array of contractor base split percentages for oil,
            shaped like ``project_years``.
        _gas_base_split : numpy.ndarray
            Array of contractor base split percentages for gas,
            shaped like ``project_years``.

        Notes
        -----
        The contractor base split values are defined as tuples of
        ``(oil_split, gas_split)`` for each fiscal regime:
          - PERMEN ESDM 8/2017: (0.43, 0.48)
          - PERMEN ESDM 52/2017: (0.43, 0.48)
          - PERMEN ESDM 20/2019: (0.43, 0.48)
          - PERMEN ESDM 12/2020: (0.43, 0.48)
          - PERMEN ESDM 13/2024: (0.47, 0.49)
        """

        # Raise an exception: inappropriate data input for parameter `regime`
        if not isinstance(regime, GrossSplitRegime):
            raise GrossSplitException(
                f"Parameter regime must be an instance of GrossSplitRegime, "
                f"not a/an {regime.__class__.__qualname__}"
            )

        # Contractor base splits based on fiscal regime in tuple (oil, gas)
        ctr_base_splits = {
            GrossSplitRegime.PERMEN_ESDM_8_2017: (0.43, 0.48),
            GrossSplitRegime.PERMEN_ESDM_52_2017: (0.43, 0.48),
            GrossSplitRegime.PERMEN_ESDM_20_2019: (0.43, 0.48),
            GrossSplitRegime.PERMEN_ESDM_12_2020: (0.43, 0.48),
            GrossSplitRegime.PERMEN_ESDM_13_2024: (0.47, 0.49),
        }

        # Lookup regime and handle unknown regime
        try:
            oil_split, gas_split = ctr_base_splits[regime]
        except KeyError:
            raise GrossSplitException(
                f"Gross split regime ({regime}) is not recognized."
            )

        # Set attributes "_oil_base_split" and "_gas_base_split"
        for f, f_split in {"oil": oil_split, "gas": gas_split}.items():
            setattr(
                self,
                f"_{f}_base_split",
                np.full_like(self.project_years, fill_value=f_split, dtype=float)
            )

    def _get_var_split_08_2017(self) -> float:
        """
        Compute the Variable Split adjustment factor based on the parameters
        defined in Gross Split PSC Regulation (Permen ESDM No. 08/2017).

        The method evaluates multiple project characteristics (e.g., field status,
        location, reservoir depth, CO₂/H₂S content, oil API gravity, domestic use,
        and production stage) and assigns numerical adjustments according to
        predefined regulatory tables. Both string-based keys (for backward
        compatibility) and Enum-based keys (preferred) are supported.

        The total Variable Split is the sum of all applicable adjustment factors.

        Returns
        -------
        float
            The total value of the Variable Split adjustment factor.

        Notes
        -----
        - The calculation follows the adjustment scheme of Permen ESDM No. 08/2017.
        - Adjustment components include:
            * Field development status
            * Field location and depth
            * Reservoir depth and type
            * Infrastructure availability
            * Gas contaminants (CO₂ and H₂S)
            * Oil API gravity
            * Domestic market obligation (DMO) share
            * Production stage (primary, secondary, tertiary)
        - Positive values increase the contractor's share, while negative
          values decrease it.
        - The result is stored internally in ``self._variable_split``.
        """

        # Components of variable split as a dictionary with strings as the corresponding key.
        # The string parameters is kept for backward compatibility.
        params_str = {
            "field_status": {
                "POD I": 0.05,
                "POD II": 0.0,
                "POFD": 0.0,
                "No POD": -0.05,
            },
            "field_loc": {
                "Onshore": 0.0,
                "Offshore (0<h<=20)": 0.08,
                "Offshore (20<h<=50)": 0.1,
                "Offshore (50<h<=150)": 0.12,
                "Offshore (150<h<=1000)": 0.14,
                "Offshore (h>1000)": 0.16,
            },
            "res_depth": {
                "<=2500": 0.0,
                ">2500": 0.01,
            },
            "infra_avail": {
                "Well Developed": 0.0,
                "New Frontier": 0.02,
            },
            "res_type": {
                "Conventional": 0.0,
                "Non Conventional": 0.16,
            },
            "co2_content": {
                "<5": 0.0,
                "5<=x<10": 0.005,
                "10<=x<20": 0.01,
                "20<=x<40": 0.015,
                "40<=x<60": 0.02,
                "x>=60": 0.04,
            },
            "h2s_content": {
                "<100": 0.0,
                "100<=x<300": 0.005,
                "300<=x<500": 0.0075,
                "x>=500": 0.01,
            },
            "api_oil": {
                "<25": 0.01,
                ">=25": 0.0,
            },
            "domestic_use": {
                "<30": 0.0,
                "30<=x<50": 0.02,
                "50<=x<70": 0.03,
                "70<=x<100": 0.04,
            },
            "prod_stage": {
                "Primary": 0.0,
                "Secondary": 0.03,
                "Tertiary": 0.05,
            }
        }

        # Components of variable split as a dictionary with Enum as the corresponding key.
        params_enum = {
            "field_status": {
                VariableSplit082017.FieldStatus.POD_I: 0.05,
                VariableSplit082017.FieldStatus.POD_II: 0.0,
                VariableSplit082017.FieldStatus.POFD: 0.0,
                VariableSplit082017.FieldStatus.NO_POD: -0.05,
            },
            "field_loc": {
                VariableSplit082017.FieldLocation.ONSHORE: 0.0,
                VariableSplit082017.FieldLocation.OFFSHORE_0_UNTIL_LESSEQUAL_20: 0.08,
                VariableSplit082017.FieldLocation.OFFSHORE_20_UNTIL_LESSEQUAL_50: 0.1,
                VariableSplit082017.FieldLocation.OFFSHORE_50_UNTIL_LESSEQUAL_150: 0.12,
                VariableSplit082017.FieldLocation.OFFSHORE_150_UNTIL_LESSEQUAL_1000: 0.14,
                VariableSplit082017.FieldLocation.OFFSHORE_GREATERTHAN_1000: 0.16,
            },
            "res_depth": {
                VariableSplit082017.ReservoirDepth.LESSEQUAL_2500: 0.0,
                VariableSplit082017.ReservoirDepth.GREATERTHAN_2500: 0.01,
            },
            "infra_avail": {
                VariableSplit082017.InfrastructureAvailability.WELL_DEVELOPED: 0.0,
                VariableSplit082017.InfrastructureAvailability.NEW_FRONTIER: 0.02,
            },
            "res_type": {
                VariableSplit082017.ReservoirType.CONVENTIONAL: 0.0,
                VariableSplit082017.ReservoirType.NON_CONVENTIONAL: 0.16,
            },
            "co2_content": {
                VariableSplit082017.CO2Content.LESSTHAN_5: 0.0,
                VariableSplit082017.CO2Content.EQUAL_5_UNTIL_LESSTHAN_10: 0.005,
                VariableSplit082017.CO2Content.EQUAL_10_UNTIL_LESSTHAN_20: 0.01,
                VariableSplit082017.CO2Content.EQUAL_20_UNTIL_LESSTHAN_40: 0.015,
                VariableSplit082017.CO2Content.EQUAL_40_UNTIL_LESSTHAN_60: 0.02,
                VariableSplit082017.CO2Content.EQUALGREATERTHAN_60: 0.04,
            },
            "h2s_content": {
                VariableSplit082017.H2SContent.LESSTHAN_100: 0.0,
                VariableSplit082017.H2SContent.EQUAL_100_UNTIL_LESSTHAN_300: 0.005,
                VariableSplit082017.H2SContent.EQUAL_300_UNTIL_LESSTHAN_500: 0.0075,
                VariableSplit082017.H2SContent.EQUALGREATERTHAN_500: 0.01,
            },
            "api_oil": {
                VariableSplit082017.APIOil.LESSTHAN_25: 0.01,
                VariableSplit082017.APIOil.EQUALGREATERTHAN_25: 0.0,
            },
            "domestic_use": {
                VariableSplit082017.DomesticUse.LESSTHAN_30: 0.0,
                VariableSplit082017.DomesticUse.EQUAL_30_UNTIL_LESSTHAN_50: 0.02,
                VariableSplit082017.DomesticUse.EQUAL_50_UNTIL_LESSTHAN_70: 0.03,
                VariableSplit082017.DomesticUse.EQUAL_70_UNTIL_LESSTHAN_100: 0.04,
            },
            "prod_stage": {
                VariableSplit082017.ProductionStage.PRIMARY: 0.0,
                VariableSplit082017.ProductionStage.SECONDARY: 0.03,
                VariableSplit082017.ProductionStage.TERTIARY: 0.05,
            }
        }

        source_dict = {
            "field_status": self.field_status,
            "field_loc": self.field_loc,
            "res_depth": self.res_depth,
            "infra_avail": self.infra_avail,
            "res_type": self.res_type,
            "api_oil": self.api_oil,
            "domestic_use": self.domestic_use,
            "prod_stage": self.prod_stage,
            "co2_content": self.co2_content,
            "h2s_content": self.h2s_content,
        }

        variable_split = np.array(
            [
                params_str[key][param] if isinstance(param, str)
                else params_enum[key][param] for key, param in
                source_dict.items()
            ],
            dtype=float
        )

        self._variable_split = float(np.sum(variable_split))

    def _get_var_split_52_2017(self) -> float:
        """
        Compute the Variable Split adjustment factor based on the parameters
        defined in Gross Split PSC Regulation (Permen ESDM No. 52/2017).

        The method evaluates multiple project characteristics (e.g., field status,
        location, reservoir depth, CO₂/H₂S content, oil API gravity, domestic use,
        and production stage) and assigns numerical adjustments according to
        predefined regulatory tables. Both string-based keys (for backward
        compatibility) and Enum-based keys (preferred) are supported.

        The total Variable Split is the sum of all applicable adjustment factors.

        Returns
        -------
        float
            The total value of the Variable Split adjustment factor.

        Notes
        -----
        - The calculation follows the adjustment scheme of Permen ESDM No. 52/2017.
        - Adjustment components include:
            * Field development status
            * Field location and water depth
            * Reservoir depth and type
            * Infrastructure availability
            * Gas contaminants (CO₂ and H₂S)
            * Oil API gravity
            * Domestic market obligation (DMO) share
            * Production stage (primary, secondary, tertiary)
        - Compared to Permen ESDM No. 08/2017, this regulation introduces:
            * Different coefficients for field development status (e.g., POD II, No POD)
            * Expanded H₂S content ranges
            * Differentiated frontier incentives (onshore vs. offshore)
            * Higher adjustment factors for advanced production stages
        - Positive values increase the contractor's share, while negative
          values decrease it.
        - The result is stored internally in ``self._variable_split``.
        """

        # Components of variable split as a dictionary with strings as the corresponding key.
        # The string parameters is kept for backward compatibility.
        params_str = {
            "field_status": {
                "POD I": 0.05,
                "POD II": 0.03,
                "No POD": 0,
            },
            "field_loc": {
                "Onshore": 0,
                "Offshore (0<h<=20)": 0.08,
                "Offshore (20<h<=50)": 0.1,
                "Offshore (50<h<=150)": 0.12,
                "Offshore (150<h<=1000)": 0.14,
                "Offshore (h>1000)": 0.16,
            },
            "res_depth": {
                "<=2500": 0,
                ">2500": 0.01,
            },
            "infra_avail": {
                "Well Developed": 0,
                "New Frontier Offshore": 0.02,
                "New Frontier Onshore": 0.04,
            },
            "res_type": {
                "Conventional": 0,
                "Non Conventional": 0.16,
            },
            "co2_content": {
                "<5": 0,
                "5<=x<10": 0.005,
                "10<=x<20": 0.01,
                "20<=x<40": 0.015,
                "40<=x<60": 0.02,
                "x>=60": 0.04,
            },
            "h2s_content": {
                "<100": 0,
                "100<=x<1000": 0.01,
                "1000<=x<2000": 0.02,
                "2000<=x<3000": 0.03,
                "3000<=x<4000": 0.04,
                "x>=4000": 0.05,
            },
            "api_oil": {
                "<25": 0.01,
                ">=25": 0,
            },
            "domestic_use": {
                "30<=x<50": 0.02,
                "50<=x<70": 0.03,
                "70<=x<100": 0.04,
            },
            "prod_stage": {
                "Primary": 0.0,
                "Secondary": 0.06,
                "Tertiary": 0.1,
            }
        }

        # Components of variable split as a dictionary with Enum as the corresponding key.
        params_enum = {
            "field_status": {
                VariableSplit522017.FieldStatus.POD_I: 0.05,
                VariableSplit522017.FieldStatus.POD_II: 0.03,
                VariableSplit522017.FieldStatus.NO_POD: 0.0,
            },
            "field_loc": {
                VariableSplit522017.FieldLocation.ONSHORE: 0.0,
                VariableSplit522017.FieldLocation.OFFSHORE_0_UNTIL_LESSEQUAL_20: 0.08,
                VariableSplit522017.FieldLocation.OFFSHORE_20_UNTIL_LESSEQUAL_50: 0.1,
                VariableSplit522017.FieldLocation.OFFSHORE_50_UNTIL_LESSEQUAL_150: 0.12,
                VariableSplit522017.FieldLocation.OFFSHORE_150_UNTIL_LESSEQUAL_1000: 0.14,
                VariableSplit522017.FieldLocation.OFFSHORE_GREATERTHAN_1000: 0.16,
            },
            "res_depth": {
                VariableSplit522017.ReservoirDepth.LESSEQUAL_2500: 0.0,
                VariableSplit522017.ReservoirDepth.GREATERTHAN_2500: 0.01,
            },
            "infra_avail": {
                VariableSplit522017.InfrastructureAvailability.WELL_DEVELOPED: 0.0,
                VariableSplit522017.InfrastructureAvailability.NEW_FRONTIER_OFFSHORE: 0.02,
                VariableSplit522017.InfrastructureAvailability.NEW_FRONTIER_ONSHORE: 0.04,
            },
            "res_type": {
                VariableSplit522017.ReservoirType.CONVENTIONAL: 0.0,
                VariableSplit522017.ReservoirType.NON_CONVENTIONAL: 0.16,
            },
            "co2_content": {
                VariableSplit522017.CO2Content.LESSTHAN_5: 0.0,
                VariableSplit522017.CO2Content.EQUAL_5_UNTIL_LESSTHAN_10: 0.005,
                VariableSplit522017.CO2Content.EQUAL_10_UNTIL_LESSTHAN_20: 0.01,
                VariableSplit522017.CO2Content.EQUAL_20_UNTIL_LESSTHAN_40: 0.015,
                VariableSplit522017.CO2Content.EQUAL_40_UNTIL_LESSTHAN_60: 0.02,
                VariableSplit522017.CO2Content.EQUALGREATERTHAN_60: 0.04,
            },
            "h2s_content": {
                VariableSplit522017.H2SContent.LESSTHAN_100: 0.0,
                VariableSplit522017.H2SContent.EQUAL_100_UNTIL_LESSTHAN_1000: 0.01,
                VariableSplit522017.H2SContent.EQUAL_1000_UNTIL_LESSTHAN_2000: 0.02,
                VariableSplit522017.H2SContent.EQUAL_2000_UNTIL_LESSTHAN_3000: 0.03,
                VariableSplit522017.H2SContent.EQUAL_3000_UNTIL_LESSTHAN_4000: 0.04,
                VariableSplit522017.H2SContent.EQUALGREATERTHAN_4000: 0.05,
            },
            "api_oil": {
                VariableSplit522017.APIOil.LESSTHAN_25: 0.01,
                VariableSplit522017.APIOil.EQUALGREATERTHAN_25: 0.0,
            },
            "domestic_use": {
                VariableSplit522017.DomesticUse.EQUAL_30_UNTIL_LESSTHAN_50: 0.02,
                VariableSplit522017.DomesticUse.EQUAL_50_UNTIL_LESSTHAN_70: 0.03,
                VariableSplit522017.DomesticUse.EQUAL_70_UNTIL_LESSTHAN_100: 0.04,
            },
            "prod_stage": {
                VariableSplit522017.ProductionStage.PRIMARY: 0.0,
                VariableSplit522017.ProductionStage.SECONDARY: 0.06,
                VariableSplit522017.ProductionStage.TERTIARY: 0.1,
            }
        }

        source_dict = {
            "field_status": self.field_status,
            "field_loc": self.field_loc,
            "res_depth": self.res_depth,
            "infra_avail": self.infra_avail,
            "res_type": self.res_type,
            "api_oil": self.api_oil,
            "domestic_use": self.domestic_use,
            "prod_stage": self.prod_stage,
            "co2_content": self.co2_content,
            "h2s_content": self.h2s_content,
        }

        variable_split = np.array(
            [
                params_str[key][param] if isinstance(param, str)
                else params_enum[key][param] for key, param in
                source_dict.items()
            ],
            dtype=float
        )

        self._variable_split = float(np.sum(variable_split))

    def _get_var_split_13_2024(
        self,
        reservoir_type: VariableSplit132024.ReservoirType
    ) -> float:
        """
        Compute the Variable Split adjustment factor based on the parameters
        defined in Gross Split PSC Regulation (Permen ESDM No. 13/2024).

        The calculation depends on the reservoir type (conventional vs.
        unconventional) and applies different adjustment schemes:

        - For **conventional reservoirs (MK)**, the adjustment factor is derived
          from three components:
            * Field location (onshore, shallow offshore, deep offshore, ultra-deep offshore)
            * Infrastructure availability (available, partially available, not available)
            * Field reserves size (low, medium, high)
          Each component contributes a coefficient, and their sum forms the final
          Variable Split.

        - For **unconventional reservoirs (MNK)**, a fixed adjustment factor of
          ``0.46`` is applied.

        Parameters
        ----------
        reservoir_type : VariableSplit132024.ReservoirType
            The reservoir classification. Must be an instance of
            ``VariableSplit132024.ReservoirType``. Accepted values are:
            - ``ReservoirType.MK`` : Conventional (Migas Konvensional)
            - ``ReservoirType.MNK`` : Unconventional (Migas Non-Konvensional)

        Returns
        -------
        float
            The total value of the Variable Split adjustment factor.

        Raises
        ------
        GrossSplitException
            If ``reservoir_type`` is not an instance of
            ``VariableSplit132024.ReservoirType`` or is otherwise unrecognized.

        Notes
        -----
        - The calculation follows the adjustment scheme of Permen ESDM No. 13/2024.
        - Adjustment logic differs fundamentally from earlier regulations:
            * For conventional reservoirs, coefficients are based on project
              characteristics.
            * For unconventional reservoirs, a fixed incentive factor is used.
        - The result is stored internally in ``self._variable_split``.
        """

        # Raise an exception for incorrect reservoir type
        if not isinstance(reservoir_type, VariableSplit132024.ReservoirType):
            raise GrossSplitException(
                f"Variable reservoir_type must be an instance of "
                f"VariableSplit132024.ReservoirType, not "
                f"{reservoir_type.__class__.__qualname__}"
            )

        # Configuration for conventional reservoir
        # MK = Migas Konvensional (Conventional Oil & Gas)
        if reservoir_type == VariableSplit132024.ReservoirType.MK:

            params_str = {
                "field_loc_2024": {
                    "Onshore": 0.11,
                    "shallow_offshore": 0.12,
                    "deep_offshore": 0.13,
                    "ultradeep_offshore": 0.14,
                },
                "infra_avail_2024": {
                    "available": 0.10,
                    "partially_available": 0.11,
                    "not_available": 0.13,
                },
                "field_reserves_2024": {
                    "low": 0.14,
                    "medium": 0.13,
                    "high": 0.12,
                }
            }

            params_enum = {
                "field_loc_2024": {
                    VariableSplit132024.FieldLocation.ONSHORE: 0.11,
                    VariableSplit132024.FieldLocation.SHALLOW_OFFSHORE: 0.12,
                    VariableSplit132024.FieldLocation.DEEP_OFFSHORE: 0.13,
                    VariableSplit132024.FieldLocation.ULTRADEEP_OFFSHORE: 0.14,
                },
                "infra_avail_2024": {
                    VariableSplit132024.InfrastructureAvailability.AVAILABLE: 0.10,
                    VariableSplit132024.InfrastructureAvailability.PARTIALLY_AVAILABLE: 0.11,
                    VariableSplit132024.InfrastructureAvailability.NOT_AVAILABLE: 0.13,
                },
                "field_reserves_2024": {
                    VariableSplit132024.FieldReservesAmount.LOW: 0.14,
                    VariableSplit132024.FieldReservesAmount.MEDIUM: 0.13,
                    VariableSplit132024.FieldReservesAmount.HIGH: 0.12,
                }
            }

            source_dict = {
                "field_loc_2024": self.field_loc_2024,
                "infra_avail_2024": self.infra_avail_2024,
                "field_reserves_2024": self.field_reserves_2024,
            }

            variable_split = np.array(
                [
                    params_str[key][param] if isinstance(param, str)
                    else params_enum[key][param] for key, param in
                    source_dict.items()
                ],
                dtype=float
            )

            self._variable_split = float(np.sum(variable_split))

        # Configuration for unconventional reservoir
        # MNK = Migas Non-Konvensional (Unconventional Oil & Gas)
        elif reservoir_type == VariableSplit132024.ReservoirType.MNK:
            self._variable_split = 0.46

        else:
            raise GrossSplitException(
                f"Variable reservoir_type ({reservoir_type}) is unrecognized."
            )

    def _wrapper_variable_split(
        self,
        regime: GrossSplitRegime,
        reservoir_type: VariableSplit132024.ReservoirType,
    ) -> float:
        """
        Dispatch the calculation of the Variable Split adjustment factor
        according to the applicable Gross Split PSC regulation.

        This method acts as a wrapper around the regulation-specific
        implementations:

        - ``_get_var_split_08_2017`` → for Permen ESDM No. 08/2017
        - ``_get_var_split_52_2017`` → for Permen ESDM No. 52/2017,
          No. 20/2019, and No. 12/2020 (identical variable split scheme)
        - ``_get_var_split_13_2024`` → for Permen ESDM No. 13/2024, which
          requires explicit specification of the reservoir type
          (conventional vs. unconventional)

        Parameters
        ----------
        regime : GrossSplitRegime
            The selected Gross Split PSC regulation. Must be one of:
            - ``GrossSplitRegime.PERMEN_ESDM_8_2017``
            - ``GrossSplitRegime.PERMEN_ESDM_52_2017``
            - ``GrossSplitRegime.PERMEN_ESDM_20_2019``
            - ``GrossSplitRegime.PERMEN_ESDM_12_2020``
            - ``GrossSplitRegime.PERMEN_ESDM_13_2024``

        reservoir_type : VariableSplit132024.ReservoirType
            The reservoir classification (only required for
            ``PERMEN_ESDM_13_2024``). Must be one of:
            - ``ReservoirType.MK`` : Conventional (Migas Konvensional)
            - ``ReservoirType.MNK`` : Unconventional (Migas Non-Konvensional)

        Returns
        -------
        float
            The computed Variable Split adjustment factor, as determined by
            the selected regulation.

        Raises
        ------
        GrossSplitException
            If the given ``regime`` is not recognized or if an invalid
            ``reservoir_type`` is provided for ``PERMEN_ESDM_13_2024``.

        Notes
        -----
        - Regulations No. 52/2017, 20/2019, and 12/2020 share the same
          variable split formula; differences between them do not affect
          this calculation and must be checked in the official documents.
        - This wrapper ensures that downstream economic calculations can
          access a single entry point for all supported Gross Split regimes,
          while keeping the regime-specific logic encapsulated in dedicated
          helper methods.
        """

        # For fiscal regime following PERMEN ESDM No. 8 Year 2017
        if regime == GrossSplitRegime.PERMEN_ESDM_8_2017:
            variable_split_func = self._get_var_split_08_2017()

        # For fiscal regime following PERMEN ESDM No. 52 Year 2017
        # or PERMEN ESDM No. 20 Year 2019 or PERMEN ESDM No. 12 Year 2020
        elif (
            regime == GrossSplitRegime.PERMEN_ESDM_52_2017
            or regime == GrossSplitRegime.PERMEN_ESDM_20_2019
            or regime == GrossSplitRegime.PERMEN_ESDM_12_2020
        ):
            variable_split_func = self._get_var_split_52_2017()

        # For fiscal regime following PERMEN ESDM No. 13 Year 2024
        elif regime == GrossSplitRegime.PERMEN_ESDM_13_2024:
            variable_split_func = self._get_var_split_13_2024(
                reservoir_type=reservoir_type
            )

        else:
            raise GrossSplitException(
                f"The Gross Split regime, {regime}, is not recognized"
            )

        return variable_split_func

    @staticmethod
    def _get_prog_price_split_08_2017(fluid: FluidType, price: float) -> float:
        """
        Compute the progressive split adjustment for oil based on
        price thresholds defined in PERMEN ESDM No. 08/2017.

        This function assigns a progressive split (ps) value depending
        on the oil price bracket. The adjustment is only applicable
        for oil; for other fluids, the adjustment is zero.

        Parameters
        ----------
        fluid : FluidType
            Type of hydrocarbon fluid. Only `FluidType.OIL`
            triggers progressive split adjustments.
        price : float
            Market price of the fluid (USD/bbl). The progressive
            split value is determined by predefined price intervals.

        Returns
        -------
        ps : float
            Progressive split adjustment value:

            - `< 40`   →  0.075
            - `[40, 55)` →  0.05
            - `[55, 70)` →  0.025
            - `[70, 85)` →  0.0
            - `[85, 100)` → -0.025
            - `[100, 115)` → -0.05
            - `>= 115` → -0.075

            For non-oil fluids, returns 0.0.
        """

        # Components of progressive split based on PERMEN ESDM 08/2017 (Appendix B)
        # OIL price
        if fluid == FluidType.OIL:
            if price < 40:
                ps = 0.075
            elif 40 <= price < 55:
                ps = 0.05
            elif 55 <= price < 70:
                ps = 0.025
            elif 70 <= price < 85:
                ps = 0.0
            elif 85 <= price < 100:
                ps = -0.025
            elif 100 <= price < 115:
                ps = -0.05
            elif 115 <= price:
                ps = -0.075
            else:
                ps = 0.0

        # Others (non-OIL)
        else:
            ps = 0.0

        return ps

    @staticmethod
    def _get_prog_cum_split_08_2017(cum: float) -> float:
        """
        Compute the progressive split adjustment based on cumulative production
        thresholds defined in PERMEN ESDM No. 08/2017.

        The adjustment (px) is determined by the cumulative production
        volume range. If no cumulative value is provided (``None``),
        the adjustment defaults to zero.

        Parameters
        ----------
        cum : float or None
            Cumulative production volume in thousand barrels of oil equivalent (MBOE).
            If ``None``, the function returns 0.0.

        Returns
        -------
        px : float
            Cumulative progressive split adjustment value:

            - ``None``         → 0.0
            - ``0 < cum < 1,000``     → 0.05
            - ``1,000 ≤ cum < 10,000`` → 0.04
            - ``10,000 ≤ cum < 20,000`` → 0.03
            - ``20,000 ≤ cum < 50,000`` → 0.02
            - ``50,000 ≤ cum < 150,000`` → 0.01
            - ``cum ≥ 150,000``       → 0.0

        Raises
        ------
        ValueError
            If `cum` is provided but does not fall within the recognized ranges.

        Notes
        -----
        Thresholds and adjustment values are taken from Appendix B of
        PERMEN ESDM No. 08/2017.
        """

        # Cumulative Progressive Split
        if cum is None:
            px = 0.0
        elif 0 <= cum < 1_000:
            px = 0.05
        elif 1_000 <= cum < 10_000:
            px = 0.04
        elif 10_000 <= cum < 20_000:
            px = 0.03
        elif 20_000 <= cum < 50_000:
            px = 0.02
        elif 50_000 <= cum < 150_000:
            px = 0.01
        elif 150_000 <= cum:
            px = 0.0
        else:
            raise ValueError('No Regulation exist regarding the cumulative value')

        return px

    @staticmethod
    def _get_prog_price_split_52_2017(fluid: FluidType, price: float) -> float:
        """
        Calculate the progressive split adjustment for oil or gas based on the
        Gross Split Regime PERMEN ESDM No. 52/2017 (Appendix B).

        For **oil**, the progressive split is determined by the reference price
        relative to a fixed benchmark of USD 85 per barrel. For **gas**, the
        progressive split depends on the gas price relative to threshold
        values of USD 7 and USD 10 per MMBTU.

        Parameters
        ----------
        fluid : FluidType
            The type of hydrocarbon fluid. Supported values are:
            - ``FluidType.OIL``: oil progressive split calculation.
            - ``FluidType.GAS``: gas progressive split calculation.
        price : float
            The reference price of the hydrocarbon fluid:
            - For oil: price in USD/barrel.
            - For gas: price in USD/MMBTU.

        Returns
        -------
        ps : float
            The progressive split value as a fraction to be applied according
            to PERMEN ESDM No. 52/2017.

        Raises
        ------
        ValueError
            If an unsupported fluid type is provided.

        Notes
        -----
        - Oil: progressive split decreases linearly with price above zero,
          benchmarked at USD 85/barrel.
        - Gas: progressive split is positive when price < USD 7/MMBTU,
          zero when 7 ≤ price ≤ 10, and negative when price > USD 10/MMBTU.
        """

        # Components of progressive split based on PERMEN ESDM 52/2017 (Appendix B)
        # OIL price
        if fluid == FluidType.OIL:
            ps = np.where(price > 0, (85 - price) * 0.25 / 100, 0.0)

        # GAS price
        elif fluid == FluidType.GAS:
            if price < 7:
                ps = (7 - price) * 2.5 / 100
            elif 7 <= price <= 10:
                ps = 0.0
            elif price > 10:
                ps = (10 - price) * 2.5 / 100
            else:
                ps = 0.0

        else:
            raise ValueError('Unknown fluid type')

        return ps

    @staticmethod
    def _get_prog_cum_split_52_2017(cum: float) -> float:
        """
        Calculate the progressive cumulative split under PERMEN ESDM No. 52/2017.

        The progressive cumulative split is determined based on cumulative production
        volume (cum), expressed in thousand BOE (MBOE). Specific thresholds and split
        rates apply according to Appendix B of the regulation.

        Parameters
        ----------
        cum : float
            Cumulative production volume used to determine the progressive split,
            expressed in MBOE.

        Returns
        -------
        float
            Progressive cumulative split value based on cumulative production.

        Raises
        ------
        ValueError
            If the cumulative value does not fall under any
            regulation threshold.

        Notes
        -----
        - The progressive cumulative split decreases as cumulative production increases.
        - Thresholds are defined in terms of MBOE ranges:
          * 0 ≤ cum < 30,000 → 0.10
          * 30,000 ≤ cum < 60,000 → 0.09
          * 60,000 ≤ cum < 90,000 → 0.08
          * 90,000 ≤ cum < 125,000 → 0.06
          * 125,000 ≤ cum < 175,000 → 0.04
          * cum ≥ 175,000 → 0.00
        - If ``cum`` is ``None``, the split defaults to 0.
        - This method only handles scalar inputs. For arrays, consider vectorizing with
          ``np.vectorize`` or an equivalent looping mechanism.
        """

        # Cumulative Progressive Split
        if cum is None:
            px = 0.0
        elif np.logical_and(np.greater_equal(cum, 0), np.less(cum, 30_000)):
            px = 0.1
        elif np.logical_and(np.greater_equal(cum, 30_000), np.less(cum, 60_000)):
            px = 0.09
        elif np.logical_and(np.greater_equal(cum, 60_000), np.less(cum, 90_000)):
            px = 0.08
        elif np.logical_and(np.greater_equal(cum, 90_000), np.less(cum, 125_000)):
            px = 0.06
        elif np.logical_and(np.greater_equal(cum, 125_000), np.less(cum, 175_000)):
            px = 0.04
        elif np.greater_equal(cum, 175_000):
            px = 0.0
        else:
            raise ValueError('No Regulation exist regarding the cumulative value')

        return px

    @staticmethod
    def _get_prog_price_split_13_2024(fluid: FluidType, price: float) -> float:
        """
        Calculate the progressive price split under PERMEN ESDM No. 13/2024 regulation.

        The progressive price split is determined based on fluid type (oil or gas) and
        the market price. Specific thresholds and linear interpolation rules apply to
        calculate the split as stipulated by the regulation.

        Parameters
        ----------
        fluid : FluidType
            Type of fluid, either ``FluidType.OIL`` or ``FluidType.GAS``.
        price : float or ndarray
            Market price of the fluid:

            - Oil: expressed in USD/bbl.
            - Gas: expressed in USD/MMBTU.

        Returns
        -------
        float
            Progressive price split value corresponding to the given fluid type and price.

        Raises
        ------
        ValueError
            If the fluid type is unknown.

        Notes
        -----
        - **Oil price split rules:**

          * price ≤ 45 → 0.05
          * 45 < price < 65 → linearly decreasing from 0.05 to 0.0
          * 65 ≤ price ≤ 85 → 0.0
          * 85 < price < 105 → linearly decreasing from 0.0 to -0.05
          * price ≥ 105 → -0.05

        - **Gas price split rules:**

          * price ≤ 4 → 0.05
          * 4 < price < 7 → linearly decreasing from 0.05 to 0.0
          * 7 ≤ price ≤ 10 → 0.0
          * 10 < price < 13 → linearly decreasing from 0.0 to -0.05
          * price ≥ 13 → -0.05

        - This method assumes scalar input for ``price``. To apply over arrays,
          vectorization (e.g., with ``np.vectorize``) is recommended.
        """

        # Components of progressive split based on PERMEN ESDM 13/2024
        # OIL price
        if fluid == FluidType.OIL:
            if price <= 45:
                ps = 0.05
            elif 45 < price < 65:
                ps = -0.0025 * (price - 45) + 0.05  # In the form of y2 = m * (x2 - x1) + y1
            elif 65 <= price <= 85:
                ps = 0.0
            elif 85 < price < 105:
                ps = -0.0025 * (price - 85) + 0.0  # In the form of y2 = m * (x2 - x1) + y1
            elif 105 <= price:
                ps = -0.05
            else:
                ps = 0.0

        # GAS price
        elif fluid == FluidType.GAS:
            if price <= 4:
                ps = 0.05
            elif 4 < price < 7:
                ps = (-1.6667 * price + 11.6667)/100
            elif 7 <= price <= 10:
                ps = 0.0
            elif 10 < price < 13:
                # ps = -0.0025 * (price - 10) + 0.0  # In the form of y2 = m * (x2 - x1) + y1
                ps = (-1.6667 * price + 16.6667)/100

            elif 13 <= price:
                ps = -0.05
            else:
                ps = 0.0

        else:
            raise ValueError("Unknown fluid type")

        return ps

    def _wrapper_progressive_split(
        self,
        fluid: FluidType,
        price: float,
        cum: float | None,
        regime: GrossSplitRegime,
    ) -> tuple:
        """
        Wrapper function to calculate the progressive split adjustment due to
        fluid price and cumulative production.

        This method selects the appropriate regulation-specific functions for
        determining progressive price split and cumulative production split under
        the specified gross split regime.

        Parameters
        ----------
        fluid : FluidType
            Type of fluid, either ``FluidType.OIL`` or ``FluidType.GAS``.
        price : float
            Fluid price. Expressed in USD/bbl for oil and USD/MMBTU for gas.
        cum : float or None
            Cumulative production volume, in units consistent with the selected
            fluid type. If ``None``, cumulative adjustment may be skipped depending
            on the regime.
        regime : GrossSplitRegime
            Fiscal regime under which the progressive split adjustment is applied.
            Supported regimes are:

            - ``PERMEN_ESDM_8_2017``
            - ``PERMEN_ESDM_52_2017``
            - ``PERMEN_ESDM_20_2019``
            - ``PERMEN_ESDM_12_2020``
            - ``PERMEN_ESDM_13_2024``

        Returns
        -------
        tuple of float or ndarray
            A tuple containing:

            - ``prog_price_split`` : float or ndarray
              Progressive split adjustment due to price.
            - ``prog_cum_split`` : float or ndarray
              Progressive split adjustment due to cumulative production.

        Raises
        ------
        ValueError
            If the gross split regime is not recognized.

        Notes
        -----
        - For ``PERMEN_ESDM_08_2017``:
          Both price- and cumulative-based progressive splits are applied.
        - For ``PERMEN_ESDM_52_2017``, ``PERMEN_ESDM_20_2019``,
          and ``PERMEN_ESDM_12_2020``:
          Progressive splits are determined using the respective regulation thresholds.
        - For ``PERMEN_ESDM_13_2024``:
          Only price-based progressive split applies. The cumulative adjustment
          is explicitly set to zero.
        """

        # Specify split adjustment due to oil price and cumulative production
        # for fiscal regime PERMEN ESDM 08/2017
        if regime == GrossSplitRegime.PERMEN_ESDM_8_2017:
            prog_price_split = self._get_prog_price_split_08_2017(
                fluid=fluid, price=price
            )
            prog_cum_split = self._get_prog_cum_split_08_2017(cum=cum)

        # Specify split adjustment due to oil price and cumulative production
        # for fiscal regime PERMEN ESDM 52/2017, PERMEN ESDM 20/2019, and
        # PERMEN ESDM 12/2020
        elif (
            regime == GrossSplitRegime.PERMEN_ESDM_52_2017
            or regime == GrossSplitRegime.PERMEN_ESDM_20_2019
            or regime == GrossSplitRegime.PERMEN_ESDM_12_2020
        ):
            prog_price_split = self._get_prog_price_split_52_2017(
                fluid=fluid, price=price
            )
            prog_cum_split = self._get_prog_cum_split_52_2017(cum=cum)

        elif regime == GrossSplitRegime.PERMEN_ESDM_13_2024:
            prog_price_split = self._get_prog_price_split_13_2024(
                fluid=fluid, price=price
            )
            prog_cum_split = np.zeros_like(prog_price_split, dtype=float)

        else:
            prog_price_split = ValueError("Gross split regime is not recognized")
            prog_cum_split = ValueError("Gross split regime is not recognized")

        return prog_price_split, prog_cum_split

    def _get_total_contractor_split(
        self,
        regime: GrossSplitRegime,
        reservoir_type: VariableSplit132024.ReservoirType,
        ministry_discretion: np.ndarray,
    ) -> None:
        """
        Calculate total contractor split for oil and gas.

        This method computes the contractor's share of production (oil and gas)
        by summing the base split, variable split, ministry discretion adjustment,
        and, when applicable, the progressive split.

        For unconventional reservoirs (MNK) under PERMEN ESDM 13/2024, the
        progressive split is excluded.

        Parameters
        ----------
        regime : GrossSplitRegime
            Fiscal regime applied to the production sharing calculation.
        reservoir_type : VariableSplit132024.ReservoirType
            Reservoir classification, e.g., conventional (MK) or unconventional (MNK).
        ministry_discretion : numpy.ndarray
            Adjustment array representing the ministry's discretionary split.
            Must be broadcastable to the shape of the base split arrays.

        Returns
        -------
        None
            This method modifies the following attributes in place:

            - ``self._oil_ctr_split`` : numpy.ndarray
              Contractor split for oil.
            - ``self._gas_ctr_split`` : numpy.ndarray
              Contractor split for gas.

        Notes
        -----
        - For **PERMEN ESDM 13/2024** with **MNK reservoir type**, the progressive
          split (``self._oil_prog_split`` and ``self._gas_prog_split``) is excluded
          from the calculation.
        - For all other regimes, the progressive split is included.
        """

        # Procedures for unconventional reservoir (MNK) in PERMEN ESDM 13/2024
        if (
            regime == GrossSplitRegime.PERMEN_ESDM_13_2024
            and reservoir_type == VariableSplit132024.ReservoirType.MNK
        ):
            # Calculate total OIL contractor split
            self._oil_ctr_split = (
                self._oil_base_split + self._var_split_array + ministry_discretion
            )

            # Calculate total GAS contractor split
            self._gas_ctr_split = (
                self._gas_base_split + self._var_split_array + ministry_discretion
            )

        # Procedures for other fiscal regimes
        else:
            self._oil_ctr_split = (
                self._oil_base_split
                + self._var_split_array
                + self._oil_prog_split
                + ministry_discretion
            )

            self._gas_ctr_split = (
                self._gas_base_split
                + self._var_split_array
                + self._gas_prog_split
                + ministry_discretion
            )

    def _get_year_maximum_split(self, ctr_split: np.ndarray, fluid: str):
        """
        Identify project years where the contractor split is greater than or equal to 100%.

        Parameters
        ----------
        ctr_split : np.ndarray
            Array of contractor split values corresponding to each project year.
        fluid : str
            Fluid type associated with the contractor split (e.g., "oil" or "gas").

        Returns
        -------
        np.ndarray
            Array of project years where the contractor split is greater than or equal to 100%.
            Returns an empty array if no such years exist.

        Notes
        -----
        A warning is raised if the contractor split is greater than or equal to 100%
        for any project year.
        """

        mask = ctr_split >= 1.0
        years_exceed = self.project_years[mask]

        if np.any(mask):
            warnings.warn(
                f"The {fluid} contractor split >= 100% in years: {years_exceed}",
                UserWarning
            )
            warnings.simplefilter("default", UserWarning)

        return years_exceed

    def _allocate_sunk_cost(
        self, sunk_cost: np.ndarray, preonstream: np.ndarray
    ) -> np.ndarray:
        """
        Allocate sunk cost and preonstream expenditures to the onstream year.

        This method aggregates the total value of ``sunk_cost`` and
        ``preonstream`` arrays into a single bulk value and assigns it to
        the project year corresponding to the earlier of the oil or gas
        onstream dates. All other years are set to zero.

        Parameters
        ----------
        sunk_cost : np.ndarray
            Array of sunk cost values across project years.
        preonstream : np.ndarray
            Array of preonstream cost values across project years.

        Returns
        -------
        np.ndarray
            A 1D array aligned with ``project_years`` where the summed bulk
            value of ``sunk_cost`` and ``preonstream`` is positioned at the
            onstream year, with zeros elsewhere.

        Raises
        ------
        ValueError
            If the onstream year does not uniquely match exactly one entry
            in ``project_years``.

        Notes
        -----
        - The onstream year is determined as the earlier of
          ``oil_onstream_date.year`` and ``gas_onstream_date.year``.
        - Both ``sunk_cost`` and ``preonstream`` arrays are expected to
          align with ``project_years`` in length.
        """

        # Calculate bulk value
        bulk_value = float(np.sum(sunk_cost + preonstream))

        # Determine the location of onstream year in project years array
        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])
        match = np.flatnonzero(self.project_years == onstream_yr)

        # Expected only a single match; raise an exception if multiple matches occurs
        if match.size != 1:
            raise ValueError(f"Expected one onstream year match, got {match.size} instead.")

        onstream_id = int(match[0])

        # Create a new array with bulk value positioned at the onstream year
        arr = np.zeros_like(self.project_years, dtype=float)
        arr[onstream_id] = bulk_value

        return arr

    def _get_cost_to_be_deducted(self, sunk_cost_method: SunkCostMethod) -> None:
        """
        Compute the total cost to be deducted for oil and gas.

        This method aggregates all deductible costs for both oil and gas,
        including depreciations, amortizations, sunk costs, carry-forward
        depreciation, and various post-tax expenditures.

        The calculation depends on whether the project is under POD I or
        non-POD I, and on the chosen sunk cost accounting method.

        Parameters
        ----------
        sunk_cost_method : SunkCostMethod
            Method to allocate sunk costs. Supported options are:
            - ``SunkCostMethod.DEPRECIATED_TANGIBLE`` : Non-depreciable sunk costs
              are included together with depreciations and amortizations.
            - ``SunkCostMethod.POOLED_1ST_YEAR`` : Depreciable sunk costs are
              pooled at the onstream year and combined with non-depreciable
              sunk costs, amortizations, and postonstream depreciations.

        Returns
        -------
        None
            Results are stored in the following instance attributes:
            - ``self._oil_cost_tobe_deducted`` : ndarray of oil costs to be deducted.
            - ``self._gas_cost_tobe_deducted`` : ndarray of gas costs to be deducted.

        Raises
        ------
        KeyError
            If ``sunk_cost_method`` is not recognized.

        Notes
        -----
        - For POD I, costs include all depreciations and amortizations
          plus carry-forward depreciation and post-tax expenditures.
        - For non-POD I cases, the inclusion of sunk costs depends on
          the selected accounting method.
        - Post-tax expenditures included are intangible, opex, ASR, LBT,
          and cost of sales.

        See Also
        --------
        _allocate_sunk_cost : Helper for allocating sunk costs to the onstream year.
        """

        # Calculates the sum of depreciations and amortizations, namely
        # sunk cost + preonstream + postonstream, for OIL and GAS
        oil_total_depr = np.array(list(self._oil_depreciations.values())).sum(axis=0)
        oil_total_amor = np.array(list(self._oil_amortizations.values())).sum(axis=0)
        gas_total_depr = np.array(list(self._gas_depreciations.values())).sum(axis=0)
        gas_total_amor = np.array(list(self._gas_amortizations.values())).sum(axis=0)

        def _oil_common(*extra):
            return (
                np.sum(extra, axis=0)
                + self._oil_carry_forward_depreciation
                + self._oil_intangible_expenditures_post_tax
                + self._oil_opex_expenditures_post_tax
                + self._oil_asr_expenditures_post_tax
                + self._oil_lbt_expenditures_post_tax
                + self._oil_cost_of_sales_expenditures_post_tax
            )

        def _gas_common(*extra):
            return (
                np.sum(extra, axis=0)
                + self._gas_carry_forward_depreciation
                + self._gas_intangible_expenditures_post_tax
                + self._gas_opex_expenditures_post_tax
                + self._gas_asr_expenditures_post_tax
                + self._gas_lbt_expenditures_post_tax
                + self._gas_cost_of_sales_expenditures_post_tax
            )

        # Specify cost to be deducted for POD I
        if self.is_pod_1:
            self._oil_cost_tobe_deducted = _oil_common(oil_total_depr, oil_total_amor)
            self._gas_cost_tobe_deducted = _gas_common(gas_total_depr, gas_total_amor)

        # Specify cost to be deducted for non POD I
        else:
            oil_non_dep = self._allocate_sunk_cost(
                sunk_cost=self._oil_non_depreciable_sunk_cost,
                preonstream=self._oil_non_depreciable_preonstream,
            )
            gas_non_dep = self._allocate_sunk_cost(
                sunk_cost=self._gas_non_depreciable_sunk_cost,
                preonstream=self._gas_non_depreciable_preonstream,
            )

            # Option 1: DEPRECIATED TANGIBLE
            if sunk_cost_method == SunkCostMethod.DEPRECIATED_TANGIBLE:
                self._oil_cost_tobe_deducted = _oil_common(
                    oil_total_depr, oil_total_amor, oil_non_dep
                )
                self._gas_cost_tobe_deducted = _gas_common(
                    gas_total_depr, gas_total_amor, gas_non_dep
                )

            # Option 2: POOLED AT ONSTREAM YEAR
            elif sunk_cost_method == SunkCostMethod.POOLED_1ST_YEAR:
                oil_dep = self._allocate_sunk_cost(
                    sunk_cost=self._oil_depreciable_sunk_cost,
                    preonstream=self._oil_depreciable_preonstream,
                )
                gas_dep = self._allocate_sunk_cost(
                    sunk_cost=self._gas_depreciable_sunk_cost,
                    preonstream=self._gas_depreciable_preonstream,
                )

                self._oil_cost_tobe_deducted = _oil_common(
                    self._oil_depreciations["postonstream"],
                    oil_total_amor,
                    oil_dep,
                    oil_non_dep,
                )
                self._gas_cost_tobe_deducted = _gas_common(
                    self._gas_depreciations["postonstream"],
                    gas_total_amor,
                    gas_dep,
                    gas_non_dep,
                )

            else:
                raise KeyError(
                    f"Unrecognized sunk cost method: {sunk_cost_method.__class__.__qualname__}"
                )

    @staticmethod
    def _get_deductible_cost(
        ctr_gross_share: np.ndarray,
        cost_tobe_deducted: np.ndarray,
        carward_deduct_cost: np.ndarray
    ) -> np.ndarray:
        """
        Calculate deductible cost limited by contractor gross share.

        This method determines the deductible cost in each project year by
        comparing the contractor's gross share against the sum of current
        year costs and the carried-forward deductible costs from previous
        years. The deductible cost is capped at the contractor's gross share.

        Parameters
        ----------
        ctr_gross_share : np.ndarray
            Contractor's gross share per project year.
        cost_tobe_deducted : np.ndarray
            Current year cost values to be deducted.
        carward_deduct_cost : np.ndarray
            Deductible costs carried forward from previous years. The first
            year is padded with zero since there is no prior year.

        Returns
        -------
        np.ndarray
            Deductible costs per project year. Each value is the lesser of
            the contractor's gross share or the sum of the current year cost
            and the carried-forward cost.

        Notes
        -----
        - The carried-forward deductible cost array is shifted by one year,
          with the first element set to zero.
        - Ensures that deductible costs never exceed the contractor's gross
          share in any given year.
        """

        carward_deduct_cost = np.concatenate((np.zeros(1), carward_deduct_cost[:-1]))

        return np.where(
            ctr_gross_share > (cost_tobe_deducted + carward_deduct_cost),
            cost_tobe_deducted + carward_deduct_cost,
            ctr_gross_share
        )

    def _get_tax_by_regime(self, tax_regime) -> np.ndarray:
        """
        Determine the tax rate array based on the tax regime and project years.

        This method computes the tax rates for the project years depending on the specified
        `tax_regime`. It uses predefined tax configurations for certain years (2013, 2016, 2020)
        and allows for specific tax regimes that override these configurations. For tax regimes
        like `NAILED_DOWN`, the tax rate is fixed based on the start year of the project.

        Parameters
        ----------
        tax_regime : TaxRegime
            The tax regime to be applied. It determines how tax rates are selected and can be
            one of the following:
            -   `TaxRegime.UU_07_2021`
            -   `TaxRegime.UU_02_2020`
            -   `TaxRegime.UU_36_2008`
            -   `TaxRegime.NAILED_DOWN`

        Returns
        -------
        tax_rate_arr : np.ndarray
            A 1D array of tax rates for each project year. The tax rate is determined based on
            the tax regime and the project's starting year in relation to the predefined tax
            configurations.
        """
        tax_config = {
            2013: 0.44,
            2016: 0.42,
            2020: 0.40
        }

        tax_rate_arr = np.full_like(
            self.project_years, fill_value=tax_config[min(tax_config)], dtype=float
        )

        for year in tax_config:
            indices = np.array(np.where(self.project_years >= year)).ravel()
            tax_rate_arr[indices] = tax_config[year]

        if tax_regime == TaxRegime.UU_07_2021:
            tax_rate_arr = np.full_like(self.project_years, fill_value=0.40, dtype=float)

        if tax_regime == TaxRegime.UU_02_2020:
            tax_rate_arr = np.full_like(self.project_years, fill_value=0.42, dtype=float)

        if tax_regime == TaxRegime.UU_36_2008:
            tax_rate_arr = np.full_like(self.project_years, fill_value=0.44, dtype=float)

        if tax_regime == TaxRegime.NAILED_DOWN:
            if self.start_date.year >= max(tax_config):
                tax_rate_arr = np.full_like(
                    self.project_years, fill_value=tax_config[max(tax_config)], dtype=float
                )
            else:
                tax_rate_arr = np.full_like(
                    self.project_years, fill_value=tax_config[min(tax_config)], dtype=float
                )

        return tax_rate_arr

    def _get_consolidated_profiles(self) -> None:
        """
        Consolidate oil and gas profiles into combined attributes.

        This method aggregates oil and gas data for lifting, prices, revenue,
        sunk costs, expenditures (pre-tax, post-tax), indirect taxes,
        total expenses, depreciation/amortization, cost recovery logic,
        tax calculations, and cashflows into consolidated project-level
        attributes.

        Notes
        -----
        The consolidated attributes are computed as element-wise sums of the
        corresponding oil and gas attributes. No return value is produced;
        all results are stored in instance variables prefixed with
        ``_consolidated_``.
        """

        # Attributes associated with consolidated lifting
        self._consolidated_lifting = (
            self._oil_lifting.get_lifting_rate_ghv_arr()
            + self._gas_lifting.get_lifting_rate_ghv_arr()
        )
        self._consolidated_wap_price = self._oil_wap_price + self._gas_wap_price
        self._consolidated_revenue = self._oil_revenue + self._gas_revenue

        # Attributes associated with consolidated sunk cost
        self._consolidated_depreciable_sunk_cost = (
            self._oil_depreciable_sunk_cost + self._gas_depreciable_sunk_cost
        )
        self._consolidated_non_depreciable_sunk_cost = (
            self._oil_non_depreciable_sunk_cost + self._gas_non_depreciable_sunk_cost
        )
        self._consolidated_sunk_cost = self._oil_sunk_cost + self._gas_sunk_cost

        # Attributes associated with consolidated preonstream
        self._consolidated_depreciable_preonstream = (
            self._oil_depreciable_preonstream + self._gas_depreciable_preonstream
        )
        self._consolidated_non_depreciable_preonstream = (
            self._oil_non_depreciable_preonstream + self._gas_non_depreciable_preonstream
        )
        self._consolidated_preonstream = self._oil_preonstream + self._gas_preonstream

        categories = [
            "capital",
            "intangible",
            "opex",
            "asr",
            "lbt",
            "cost_of_sales"
        ]

        # Attributes associated with consolidated expenditures pre tax,
        # consolidated indirect tax, and consolidated expenditures post tax
        for categ in categories:
            oil_pre_tax = getattr(self, f"_oil_{categ}_expenditures_pre_tax")
            gas_pre_tax = getattr(self, f"_gas_{categ}_expenditures_pre_tax")
            oil_indirect_tax = getattr(self, f"_oil_{categ}_indirect_tax")
            gas_indirect_tax = getattr(self, f"_gas_{categ}_indirect_tax")
            oil_post_tax = getattr(self, f"_oil_{categ}_expenditures_post_tax")
            gas_post_tax = getattr(self, f"_gas_{categ}_expenditures_post_tax")

            # Set attributes associated with consolidated expenditures pre tax
            setattr(
                self, f"_consolidated_{categ}_expenditures_pre_tax", oil_pre_tax + gas_pre_tax
            )

            # Set attributes associated with consolidated indirect taxes
            setattr(
                self, f"_consolidated_{categ}_indirect_tax", oil_indirect_tax + gas_indirect_tax
            )

            # Set attributes associated with consolidated expenditures post tax
            setattr(
                self, f"_consolidated_{categ}_expenditures_post_tax", oil_post_tax + gas_post_tax
            )

        self._consolidated_total_expenditures_pre_tax = (
            self._oil_total_expenditures_pre_tax + self._gas_total_expenditures_pre_tax
        )

        self._consolidated_total_indirect_tax = (
            self._oil_total_indirect_tax + self._gas_total_indirect_tax
        )

        self._consolidated_total_expenditures_post_tax = (
            self._oil_total_expenditures_post_tax + self._gas_total_expenditures_post_tax
        )

        # Attribute associated with carry forward depreciation
        self._consolidated_carry_forward_depreciation = (
            self._oil_carry_forward_depreciation + self._gas_carry_forward_depreciation
        )

        # Attributes associated with depreciations, undepreciated_assets, and amortizations
        cost_types = ["sunk_cost", "preonstream", "postonstream"]

        self._consolidated_depreciations = {
            c: self._oil_depreciations[c] + self._gas_depreciations[c]
            for c in cost_types
        }

        self._consolidated_undepreciated_assets = {
            c: self._oil_undepreciated_assets[c] + self._gas_undepreciated_assets[c]
            for c in cost_types
        }

        self._consolidated_sum_undepreciated_asset = self._oil_sum_undepreciated_asset \
            + self._gas_sum_undepreciated_asset

        self._consolidated_amortizations = {
            c: self._oil_amortizations[c] + self._gas_amortizations[c]
            for c in cost_types
        }

        self._consolidated_depreciation = self._oil_depreciation + self._gas_depreciation
        self._consolidated_amortization = self._oil_amortization + self._gas_amortization

        # Attributes associated with core business logics
        self._consolidated_ctr_share_before_tf = (
            self._oil_ctr_share_before_transfer + self._gas_ctr_share_before_transfer
        )
        self._consolidated_gov_share_before_tf = self._oil_gov_share + self._gas_gov_share
        self._consolidated_cost_tobe_deducted = (
            self._oil_cost_tobe_deducted + self._gas_cost_tobe_deducted
        )
        self._consolidated_carward_deduct_cost = (
            self._oil_carward_deduct_cost + self._gas_carward_deduct_cost
        )
        self._consolidated_deductible_cost = (
            self._oil_deductible_cost + self._gas_deductible_cost
        )
        self._consolidated_carward_cost_aftertf = (
            self._oil_carward_cost_aftertf + self._gas_carward_cost_aftertf
        )
        self._consolidated_ctr_share_after_transfer = (
            self._oil_ctr_share_after_transfer + self._gas_ctr_share_after_transfer
        )
        self._consolidated_net_operating_profit = (
            self._oil_net_operating_profit + self._gas_net_operating_profit
        )

        # Attributes associated with DMO
        self._consolidated_dmo_volume = self._oil_dmo_volume + self._gas_dmo_volume
        self._consolidated_dmo_fee = self._oil_dmo_fee + self._gas_dmo_fee
        self._consolidated_ddmo = self._oil_ddmo + self._gas_ddmo

        # Attributes associated with taxable income
        self._consolidated_taxable_income = (
            self._oil_taxable_income + self._gas_taxable_income
        )
        self._consolidated_tax_payment = self._oil_tax + self._gas_tax

        # Attributes associated with contractor and government take
        self._consolidated_ctr_net_share = self._oil_ctr_net_share + self._gas_ctr_net_share
        self._consolidated_government_take = (
            self._oil_government_take + self._gas_government_take
        )

        # Attributes associated with consolidated investments
        self._consolidated_capital = self._oil_capital + self._gas_capital
        self._consolidated_non_capital = self._oil_non_capital + self._gas_non_capital
        self._consolidated_total_expenses = self._oil_total_expenses + self._gas_total_expenses

        # Attribute associated with consolidated cashflow
        self._consolidated_cashflow = self._oil_ctr_cashflow + self._gas_ctr_cashflow

        # =================================================================================

        self._consolidated_opex = (
            self._oil_opex_expenditures_post_tax + self._gas_opex_expenditures_post_tax
        )
        self._consolidated_asr = (
            self._oil_asr_expenditures_post_tax + self._gas_asr_expenditures_post_tax
        )
        self._consolidated_lbt = (
            self._oil_lbt_expenditures_post_tax + self._gas_lbt_expenditures_post_tax
        )
        self._consolidated_non_capital = self._oil_non_capital + self._gas_non_capital

    def run(
        self,
        sulfur_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        electricity_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_OIL_REVENUE,
        co2_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        vat_rate: np.ndarray | float = 0.0,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo | None = InflationAppliedTo.CAPEX,
        cum_production_split_offset: float | np.ndarray | None = 0.0,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        sum_undepreciated_cost: bool = False,
        is_dmo_end_weighted: bool = False,
        tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
        effective_tax_rate: float | np.ndarray = 0.22,
        amortization: bool = False,
        sunk_cost_method: SunkCostMethod = SunkCostMethod.DEPRECIATED_TANGIBLE,
        regime: GrossSplitRegime = GrossSplitRegime.PERMEN_ESDM_13_2024,
        reservoir_type_permen_2024: VariableSplit132024.ReservoirType = (
            VariableSplit132024.ReservoirType.MK
        ),
        initial_amortization_year: InitialYearAmortizationIncurred = (
            InitialYearAmortizationIncurred.ONSTREAM_YEAR
        ),
    ):
        """
        Run the full economic evaluation workflow under a Gross Split PSC scheme.

        This method executes the main sequence of calculations, including
        revenues, expenditures, taxes, contractor/government split, deductible
        costs, DMO (Domestic Market Obligation), cash flows, and government take.
        It modifies the model’s internal state by populating time-series arrays
        for oil, gas, and consolidated results.

        Parameters
        ----------
        sulfur_revenue : OtherRevenue, default=OtherRevenue.ADDITION_TO_GAS_REVENUE
            Treatment of sulfur revenue (added to gas revenue or handled separately).
        electricity_revenue : OtherRevenue, default=OtherRevenue.ADDITION_TO_OIL_REVENUE
            Treatment of electricity revenue.
        co2_revenue : OtherRevenue, default=OtherRevenue.ADDITION_TO_GAS_REVENUE
            Treatment of CO₂ revenue.
        vat_rate : float or np.ndarray, default=0.0
            Value-added tax (VAT) rate applied to indirect taxes.
        year_inflation : np.ndarray, optional
            Year-specific inflation factors. If None, a constant rate is applied.
        inflation_rate : float or np.ndarray, default=0.0
            Inflation rate applied to selected expenditures.
        inflation_rate_applied_to : InflationAppliedTo or None, default=InflationAppliedTo.CAPEX
            Specifies which cost components are subject to inflation adjustments.
        cum_production_split_offset : float, np.ndarray, or None, default=0.0
            Offset applied to cumulative production for progressive split calculation.
            Can be a scalar (applied to the first year only) or a time-series array.
        depr_method : DeprMethod, default=DeprMethod.PSC_DB
            Depreciation method to apply (e.g., PSC Declining Balance).
        decline_factor : float or int, default=2
            Decline factor for declining-balance depreciation.
        sum_undepreciated_cost : bool, default=False
            Whether to sum undepreciated costs across categories before adjustment.
        is_dmo_end_weighted : bool, default=False
            If True, DMO holiday adjustment is applied at the end of the period.
        tax_regime : TaxRegime, default=TaxRegime.NAILED_DOWN
            Tax regime to use if effective tax rate is not explicitly provided.
        effective_tax_rate : float or np.ndarray, default=0.22
            Effective tax rate. If None, determined by `tax_regime`.
        amortization : bool, default=False
            Whether amortization of certain costs is enabled.
        sunk_cost_method : SunkCostMethod, default=SunkCostMethod.DEPRECIATED_TANGIBLE
            Method for treating sunk costs in the calculation.
        regime : GrossSplitRegime, default=GrossSplitRegime.PERMEN_ESDM_13_2024
            Gross Split regime definition to apply.
        reservoir_type_permen_2024 : VariableSplit132024.ReservoirType, default=MK
            Reservoir classification for variable split (as defined in Permen 13/2024).
        initial_amortization_year : InitialYearAmortizationIncurred, default=ONSTREAM_YEAR
            Defines when amortization begins (e.g., onstream year).

        Returns
        -------
        None
            The method updates internal attributes in-place, including:
            - ``_oil_*`` : Oil-related revenues, expenditures, splits, taxes, cash flows.
            - ``_gas_*`` : Gas-related revenues, expenditures, splits, taxes, cash flows.
            - ``_consolidated_*`` : Aggregated oil + gas profiles.

        Notes
        -----
        - This is the **main entry point** for running the model after setting
          production, cost, and contract parameters.
        - The method modifies internal state only; results are retrieved through
          :meth:`get_results` or other accessors.
        - PSC-specific terminology:
          * *Deductible cost* ≈ "cost recovery" in cost-recovery PSC.
          * *Contractor split* refers to gross split allocation before and after transfer.
          * *DMO* corresponds to domestic market obligation adjustments.
        """

        # Perform initial check to several input arguments
        self._check_attributes()

        # WAP (Weighted Average Price) for each produced fluid
        self._get_wap_price()

        # Validate sunk cost, pre-onstream, and post-onstream objects
        self._get_cost_objects_validation()

        # Calculate pre tax expenditures
        self._get_expenditures_pre_tax(
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
        )

        # Calculate indirect taxes
        self._get_indirect_taxes(tax_rate=vat_rate)

        # Calculate post tax expenditures
        self._get_expenditures_post_tax()

        # Calculate other revenue
        self._get_other_revenue(
            sulfur_revenue=sulfur_revenue,
            electricity_revenue=electricity_revenue,
            co2_revenue=co2_revenue,
        )

        # Total pre-tax expenditures
        self._oil_total_expenditures_pre_tax = (
            self._oil_capital_expenditures_pre_tax
            + self._oil_intangible_expenditures_pre_tax
            + self._oil_opex_expenditures_pre_tax
            + self._oil_asr_expenditures_pre_tax
            + self._oil_lbt_expenditures_pre_tax
            + self._oil_cost_of_sales_expenditures_pre_tax
        )

        self._gas_total_expenditures_pre_tax = (
            self._gas_capital_expenditures_pre_tax
            + self._gas_intangible_expenditures_pre_tax
            + self._gas_opex_expenditures_pre_tax
            + self._gas_asr_expenditures_pre_tax
            + self._gas_lbt_expenditures_pre_tax
            + self._gas_cost_of_sales_expenditures_pre_tax
        )

        # Total indirect taxes
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

        # Total expenditures post-tax
        self._oil_total_expenditures_post_tax = (
            self._oil_capital_expenditures_post_tax
            + self._oil_intangible_expenditures_post_tax
            + self._oil_opex_expenditures_post_tax
            + self._oil_asr_expenditures_post_tax
            + self._oil_lbt_expenditures_post_tax
            + self._oil_cost_of_sales_expenditures_post_tax
        )

        self._gas_total_expenditures_post_tax = (
            self._gas_capital_expenditures_post_tax
            + self._gas_intangible_expenditures_post_tax
            + self._gas_opex_expenditures_post_tax
            + self._gas_asr_expenditures_post_tax
            + self._gas_lbt_expenditures_post_tax
            + self._gas_cost_of_sales_expenditures_post_tax
        )

        # Prepare sunk costs, preonstream costs, and postonstream costs
        self._get_sunkcost_array()
        self._get_preonstream_array()
        self._get_postonstream_array()

        self._get_amortization(
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            tax_rate=vat_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
        )

        print('\t')
        print(f'Filetype: {type(self._oil_amortizations)}')
        print(f'Length: {len(self._oil_amortizations)}')
        print('_oil_amortizations = \n', self._oil_amortizations)

        # self._get_depreciation(
        #     depr_method=depr_method,
        #     decline_factor=decline_factor,
        #     year_inflation=year_inflation,
        #     inflation_rate=inflation_rate,
        #     tax_rate=vat_rate,
        #     inflation_rate_applied_to=inflation_rate_applied_to
        # )

        # # Modify sunk cost and preonstream cost for non-POD I contract
        # if not self.is_pod_1:
        #     self._validated_sunkcost_non_pod_1()
        #
        #     """
        #     Former approach
        #     ---------------
        #     # Combine sunk cost and preonstream cost for non-POD I contract,
        #     # Assign the result as the "modified" sunk cost
        #     self._oil_sunk_cost += self._oil_preonstream
        #     self._gas_sunk_cost += self._gas_preonstream
        #
        #     # Assign preonstream cost as zeros
        #     self._oil_preonstream = np.zeros_like(self.project_years, dtype=float)
        #     self._gas_preonstream = np.zeros_like(self.project_years, dtype=float)
        #     """
        #
        # # Calculate depreciations and undepreciated assets
        # self._get_depreciation(
        #     depr_method=depr_method,
        #     decline_factor=decline_factor,
        #     year_inflation=year_inflation,
        #     inflation_rate=inflation_rate,
        #     tax_rate=vat_rate,
        #     inflation_rate_applied_to=inflation_rate_applied_to,
        # )
        #
        # # Modify depreciations, accounting for various adjusments
        # self._get_modified_depreciations(sum_undepreciated_cost=sum_undepreciated_cost)
        # self._oil_sum_undepreciated_asset = np.sum(
        #     [np.sum(v) for v in self._oil_undepreciated_assets.values()]
        # )
        # self._gas_sum_undepreciated_asset = np.sum(
        #     [np.sum(v) for v in self._gas_undepreciated_assets.values()]
        # )
        #
        # # Calculate amortizations
        # self._get_amortization(
        #     salvage_value=0.0,
        #     initial_amortization_year=initial_amortization_year,
        # )
        #
        # self._oil_depreciation = np.sum([v for v in self._oil_depreciations.values()], axis=0)
        # self._gas_depreciation = np.sum([v for v in self._gas_depreciations.values()], axis=0)
        # self._oil_amortization = np.sum([v for v in self._oil_amortizations.values()], axis=0)
        # self._gas_amortization = np.sum([v for v in self._gas_amortizations.values()], axis=0)
        #
        # # Specify base split
        # self._wrapper_base_split(regime=regime)
        #
        # # Specify variable split
        # self._wrapper_variable_split(
        #     regime=regime,
        #     reservoir_type=reservoir_type_permen_2024,
        # )
        #
        # self._var_split_array = np.full_like(
        #     self.project_years, fill_value=self._variable_split, dtype=float
        # )
        #
        # # Calculate gas production in MMBOE. Based on discussion with YRA on June 28th, 2024,
        # # cumulative production used for progressive cumulative production split is defined as
        # # the sum of production rate and production baseline.
        # prod_gas_boe = (
        #     self._gas_lifting.get_prod_rate_total_arr() * 1_000. / self.conversion_boe_to_scf
        # )
        #
        # # Check if the cumulative production split offset length is the same
        # # with the length of project years
        # self._check_cum_production_split_offset(
        #     cum_production_split_offset=cum_production_split_offset
        # )
        #
        # # Adjustment in cumulative production split offset
        # # when parameter "cum_production_split_offset" is given as a single value.
        # if (
        #     isinstance(cum_production_split_offset, float)
        #     or isinstance(cum_production_split_offset, int)
        # ):
        #     offset_arr = np.full_like(self.project_years, fill_value=0.0, dtype=float)
        #     offset_arr[0] = cum_production_split_offset
        #
        # else:
        #     offset_arr = np.full_like(self.project_years, fill_value=0.0, dtype=float)
        #
        # # Calculating the cumulative production
        # self._cumulative_prod = np.cumsum(
        #     self._oil_lifting.get_prod_rate_total_arr()
        #     + prod_gas_boe
        #     + offset_arr
        # )
        #
        # # Specify progressive split
        # vectorized_get_prog_split = np.vectorize(self._wrapper_progressive_split)
        #
        # # Condition when the cum_production_split_offset is filled with np.ndarray
        # if (
        #     isinstance(cum_production_split_offset, np.ndarray)
        #     and len(cum_production_split_offset) > 1
        # ):
        #     # Define attributes "_oil_prog_price_split", "_oil_prog_cum_split",
        #     # "_oil_prog_split"
        #     (
        #         self._oil_prog_price_split,
        #         self._oil_prog_cum_split
        #     ) = vectorized_get_prog_split(
        #         fluid=self._oil_lifting.fluid_type,
        #         price=self._oil_lifting.get_price_arr(),
        #         cum=None,
        #         regime=regime
        #     )
        #
        #     self._oil_prog_split = self._oil_prog_price_split + cum_production_split_offset
        #
        #     # Define attributes "_gas_prog_price_split", "_gas_prog_cum_split",
        #     # "_gas_prog_split"
        #     (
        #         self._gas_prog_price_split,
        #         self._gas_prog_cum_split
        #     ) = vectorized_get_prog_split(
        #         fluid=self._gas_lifting.fluid_type,
        #         price=self._gas_lifting.get_price_arr(),
        #         cum=None,
        #         regime=regime
        #     )
        #
        #     self._gas_prog_split = self._gas_prog_price_split + cum_production_split_offset
        #
        # # Condition when the cum_production_split_offset is not filled
        # else:
        #     # Define attributes "_oil_prog_price_split", "_oil_prog_cum_split",
        #     # "_oil_prog_split"
        #     (
        #         self._oil_prog_price_split,
        #         self._oil_prog_cum_split
        #     ) = vectorized_get_prog_split(
        #         fluid=self._oil_lifting.fluid_type,
        #         price=self._oil_lifting.get_price_arr(),
        #         cum=self._cumulative_prod,
        #         regime=regime,
        #     )
        #
        #     self._oil_prog_split = self._oil_prog_price_split + self._oil_prog_cum_split
        #
        #     # Define attributes "_gas_prog_price_split", "_gas_prog_cum_split",
        #     # "_gas_prog_split"
        #     (
        #         self._gas_prog_price_split,
        #         self._gas_prog_cum_split
        #     ) = vectorized_get_prog_split(
        #         fluid=self._gas_lifting.fluid_type,
        #         price=self._gas_lifting.get_price_arr(),
        #         cum=self._cumulative_prod,
        #         regime=regime,
        #     )
        #
        #     self._gas_prog_split = self._gas_prog_price_split + self._gas_prog_cum_split
        #
        # # Ministerial discretion
        # self._ministry_discretion_arr = np.full_like(
        #     self.project_years, fill_value=self.split_ministry_disc, dtype=float
        # )
        #
        # # Total Contractor Split
        # self._get_total_contractor_split(
        #     regime=regime,
        #     reservoir_type=reservoir_type_permen_2024,
        #     ministry_discretion=self._ministry_discretion_arr,
        # )
        #
        # # Adjustment when total contractor split exceeds 100%
        # self._oil_ctr_split_prior_bracket = np.copy(self._oil_ctr_split)
        # self._gas_ctr_split_prior_bracket = np.copy(self._gas_ctr_split)
        #
        # self._oil_year_maximum_ctr_split = self._get_year_maximum_split(
        #     ctr_split=self._oil_ctr_split,
        #     fluid="oil"
        # )
        #
        # self._gas_year_maximum_ctr_split = self._get_year_maximum_split(
        #     ctr_split=self._gas_ctr_split,
        #     fluid="gas"
        # )
        #
        # # Set maximum contractor split to be 100%
        # self._oil_ctr_split[self._oil_ctr_split > 1.0] = 1.0
        # self._gas_ctr_split[self._gas_ctr_split > 1.0] = 1.0
        #
        # # Contractor Share
        # self._oil_ctr_share_before_transfer = self._oil_revenue * self._oil_ctr_split
        # self._gas_ctr_share_before_transfer = self._gas_revenue * self._gas_ctr_split
        #
        # # Government Share
        # self._oil_gov_share = self._oil_revenue - self._oil_ctr_share_before_transfer
        # self._gas_gov_share = self._gas_revenue - self._gas_ctr_share_before_transfer
        #
        # # Calculate capital and non-capital investments
        # self._get_investments()
        #
        # # Cost to be deducted
        # self._get_cost_to_be_deducted(sunk_cost_method=sunk_cost_method)
        #
        # # Oil carry forward deductible cost (in PSC Cost Recovery called Unrecovered Cost)
        # zeros = np.zeros_like(self.project_years, dtype=float)
        # oil_total_depr = np.array(list(self._oil_depreciations.values())).sum(axis=0)
        # oil_total_amor = np.array(list(self._oil_amortizations.values())).sum(axis=0)
        #
        # self._oil_carward_deduct_cost = psc_tools.get_unrecovered_cost(
        #     depreciation=(
        #         oil_total_depr + oil_total_amor + self._oil_carry_forward_depreciation
        #     ),
        #     non_capital=self._oil_non_capital,
        #     revenue=self._oil_ctr_share_before_transfer,
        #     ftp_ctr=zeros,
        #     ftp_gov=zeros,
        #     ic=zeros,
        # )
        #
        # # Gas carry forward deductible cost (in PSC Cost Recovery called Unrecovered Cost)
        # gas_total_depr = np.array(list(self._gas_depreciations.values())).sum(axis=0)
        # gas_total_amor = np.array(list(self._gas_amortizations.values())).sum(axis=0)
        #
        # self._gas_carward_deduct_cost = psc_tools.get_unrecovered_cost(
        #     depreciation=(
        #         gas_total_depr + gas_total_amor + self._gas_carry_forward_depreciation
        #     ),
        #     non_capital=self._gas_non_capital,
        #     revenue=self._gas_ctr_share_before_transfer,
        #     ftp_ctr=zeros,
        #     ftp_gov=zeros,
        #     ic=zeros,
        # )
        #
        # # Deductible cost (In PSC Cost Recovery called "cost recovery")
        # self._oil_deductible_cost = self._get_deductible_cost(
        #     ctr_gross_share=self._oil_ctr_share_before_transfer,
        #     cost_tobe_deducted=self._oil_cost_tobe_deducted,
        #     carward_deduct_cost=self._oil_carward_deduct_cost
        # )
        #
        # self._gas_deductible_cost = self._get_deductible_cost(
        #     ctr_gross_share=self._gas_ctr_share_before_transfer,
        #     cost_tobe_deducted=self._gas_cost_tobe_deducted,
        #     carward_deduct_cost=self._gas_carward_deduct_cost
        # )
        #
        # # Transfer
        # self._transfer_to_oil, self._transfer_to_gas = psc_tools.get_transfer(
        #     oil_unrecovered=self._oil_carward_deduct_cost,
        #     gas_unrecovered=self._gas_carward_deduct_cost,
        #     oil_ets_pretransfer=self._oil_ctr_share_before_transfer,
        #     gas_ets_pretransfer=self._gas_ctr_share_before_transfer,
        # )
        #
        # # Carry Forward Deductible Cost After Transfer
        # self._oil_carward_cost_aftertf = self._oil_carward_deduct_cost - self._transfer_to_gas
        # self._gas_carward_cost_aftertf = self._gas_carward_deduct_cost - self._transfer_to_oil
        #
        # # Contractor Share After Transfer
        # self._oil_ctr_share_after_transfer = (
        #     self._oil_ctr_share_before_transfer
        #     + self._transfer_to_oil
        #     - self._transfer_to_gas
        # )
        #
        # self._gas_ctr_share_after_transfer = (
        #     self._gas_ctr_share_before_transfer
        #     + self._transfer_to_gas
        #     - self._transfer_to_oil
        # )
        #
        # # Contractor Net Operating Profit
        # self._oil_net_operating_profit = (
        #     self._oil_ctr_share_after_transfer - self._oil_deductible_cost
        # )
        # self._gas_net_operating_profit = (
        #     self._gas_ctr_share_after_transfer - self._gas_deductible_cost
        # )
        #
        # # ====================================================================================
        #
        # """
        # # For debugging purpose only
        # _gas_tobe_deducted = (
        #     gas_total_depr
        #     + gas_total_amor
        #     + self._gas_carry_forward_depreciation
        #     + self._gas_non_capital
        # )
        # _oil_tobe_deducted = (
        #     oil_total_depr
        #     + oil_total_amor
        #     + self._oil_carry_forward_depreciation
        #     + self._oil_non_capital
        # )
        #
        # col1 = [
        #     'ctr_before_trf',
        #     'tobe_deducted',
        #     'carward_deduct',
        #     'deductible',
        #     'transfer_to',
        #     'carward_after_trf',
        #     'ctr_after_trf',
        #     'net_profit'
        # ]
        #
        # fluids = ['oil', 'gas']
        # gs1_df = pd.DataFrame(
        #     columns=pd.MultiIndex.from_product([col1, fluids]),
        #     index=self.project_years
        # )
        # gs1_df[('ctr_before_trf', 'oil')] = self._oil_ctr_share_before_transfer
        # gs1_df[('ctr_before_trf', 'gas')] = self._gas_ctr_share_before_transfer
        # gs1_df[('tobe_deducted', 'oil')] = _oil_tobe_deducted
        # gs1_df[('tobe_deducted', 'gas')] = _gas_tobe_deducted
        # gs1_df[('deductible', 'oil')] = self._oil_deductible_cost
        # gs1_df[('deductible', 'gas')] = self._gas_deductible_cost
        # gs1_df[('carward_deduct', 'oil')] = self._oil_carward_deduct_cost
        # gs1_df[('carward_deduct', 'gas')] = self._gas_carward_deduct_cost
        # gs1_df[('transfer_to', 'oil')] = self._transfer_to_oil
        # gs1_df[('transfer_to', 'gas')] = self._transfer_to_gas
        # gs1_df[('carward_after_trf', 'oil')] = self._oil_carward_cost_aftertf
        # gs1_df[('carward_after_trf', 'gas')] = self._gas_carward_cost_aftertf
        # gs1_df[('ctr_after_trf', 'oil')] = self._oil_ctr_share_after_transfer
        # gs1_df[('ctr_after_trf', 'gas')] = self._gas_ctr_share_after_transfer
        # gs1_df[('net_profit', 'oil')] = self._oil_net_operating_profit
        # gs1_df[('net_profit', 'gas')] = self._gas_net_operating_profit
        # """
        #
        # # ====================================================================================
        #
        # allocation = np.stack(
        #     [self._oil_ctr_share_before_transfer, self._gas_ctr_share_before_transfer],
        #     axis=1
        # )
        # tobe_deducted = np.stack(
        #     [self._oil_cost_tobe_deducted, self._gas_cost_tobe_deducted],
        #     axis=1
        # )
        # deductible = np.zeros_like(allocation)
        # undeductible_pre_trf = np.zeros_like(allocation)
        # undeductible_post_trf = np.zeros_like(allocation)
        # net_pre_trf = np.zeros_like(allocation)
        # net_post_trf = np.zeros_like(allocation)
        # trf_to_og = np.zeros_like(allocation)
        #
        # prev_undeductible = np.zeros(2)
        #
        # for i in range(self.project_duration):
        #     tobe_deducted_year = tobe_deducted[i] + prev_undeductible
        #     deductible[i] = np.minimum(allocation[i], tobe_deducted_year)
        #     undeductible_pre_trf[i] = np.maximum(0, tobe_deducted_year - allocation[i])
        #     net_pre_trf[i] = np.maximum(0, allocation[i] - tobe_deducted_year)
        #
        #     net_after = net_pre_trf[i].copy()
        #     undeduct_after = undeductible_pre_trf[i].copy()
        #
        #     # Transfer between oil and gas
        #     trf_to = np.zeros(2)
        #
        #     if net_pre_trf[i, 0] > 0 and undeductible_pre_trf[i, 1] > 0:
        #         """
        #         Transfer from oil to gas
        #         if oil net profit > 0 and gas undeductible cost > 0
        #         then oil profit is transferred to cover gas the undeductible cost
        #         """
        #         trf_to[1] = np.minimum(net_pre_trf[i, 0], undeductible_pre_trf[i, 1])
        #         net_after[0] -= trf_to[1]
        #         undeduct_after[1] -= trf_to[1]
        #
        #     elif net_pre_trf[i, 1] > 0 and undeductible_pre_trf[i, 0] > 0:
        #         """
        #         Transfer from gas to oil
        #         if gas net profit > 0 and oil undeductible cost > 0
        #         then gas profit is transferred to cover oil undeductible cost
        #         """
        #         trf_to[0] = np.minimum(net_pre_trf[i, 1], undeductible_pre_trf[i, 0])
        #         net_after[1] -= trf_to[0]
        #         undeduct_after[0] -= trf_to[0]
        #
        #     net_post_trf[i] = net_after
        #     undeductible_post_trf[i] = undeduct_after
        #     trf_to_og[i] = trf_to
        #
        #     prev_undeductible = undeductible_post_trf[i]
        #
        # """
        # cols2 = [
        #     'allocation',
        #     'tobe_deducted',
        #     'deductible',
        #     'undeductible_pre_trf',
        #     'net_pre_trf',
        #     'trf_to_og',
        #     'undeductible_post_trf',
        #     'net_post_trf'
        # ]
        #
        # fluids = ['oil', 'gas']
        #
        # gs2_df = pd.DataFrame(
        #     columns=pd.MultiIndex.from_product([cols2, fluids]),
        #     index=self.project_years
        # )
        # gs2_df[('allocation', 'oil')] = allocation[:, 0]
        # gs2_df[('allocation', 'gas')] = allocation[:, 1]
        # gs2_df[('tobe_deducted', 'oil')] = tobe_deducted[:, 0]
        # gs2_df[('tobe_deducted', 'gas')] = tobe_deducted[:, 1]
        # gs2_df[('deductible', 'oil')] = deductible[:, 0]
        # gs2_df[('deductible', 'gas')] = deductible[:, 1]
        # gs2_df[('undeductible_pre_trf', 'oil')] = undeductible_pre_trf[:, 0]
        # gs2_df[('undeductible_pre_trf', 'gas')] = undeductible_pre_trf[:, 1]
        # gs2_df[('net_pre_trf', 'oil')] = net_pre_trf[:, 0]
        # gs2_df[('net_pre_trf', 'gas')] = net_pre_trf[:, 1]
        # gs2_df[('trf_to_og', 'oil')] = trf_to_og[:, 0]
        # gs2_df[('trf_to_og', 'gas')] = trf_to_og[:, 1]
        # gs2_df[('undeductible_post_trf', 'oil')] = undeductible_post_trf[:, 0]
        # gs2_df[('undeductible_post_trf', 'gas')] = undeductible_post_trf[:, 1]
        # gs2_df[('net_post_trf', 'oil')] = net_post_trf[:, 0]
        # gs2_df[('net_post_trf', 'gas')] = net_post_trf[:, 1]
        # """
        #
        # # ====================================================================================
        #
        # self._oil_deductible_cost = deductible[:, 0]
        # self._gas_deductible_cost = deductible[:, 1]
        # self._oil_carward_deduct_cost = undeductible_pre_trf[:, 0]
        # self._gas_carward_deduct_cost = undeductible_pre_trf[:, 1]
        # self._transfer_to_oil = trf_to_og[:, 0]
        # self._transfer_to_gas = trf_to_og[:, 1]
        # self._oil_profit_pre_transfer = net_pre_trf[:, 0]
        # self._gas_profit_pre_transfer = net_pre_trf[:, 1]
        # self._oil_carward_cost_aftertf = undeductible_post_trf[:, 0]
        # self._gas_carward_cost_aftertf = undeductible_post_trf[:, 1]
        # self._oil_net_operating_profit = net_post_trf[:, 0]
        # self._gas_net_operating_profit = net_post_trf[:, 1]
        #
        # # DMO
        # self._oil_dmo_volume, self._oil_dmo_fee, self._oil_ddmo = psc_tools.get_dmo_gross_split(
        #     onstream_date=self.oil_onstream_date,
        #     start_date=self.start_date,
        #     project_years=self.project_years,
        #     dmo_holiday_duration=self.oil_dmo_holiday_duration,
        #     dmo_volume_portion=self.oil_dmo_volume_portion,
        #     dmo_fee_portion=self.oil_dmo_fee_portion,
        #     price=self._oil_wap_price,
        #     unrecovered_cost=self._oil_carward_cost_aftertf,
        #     is_dmo_end_weighted=is_dmo_end_weighted,
        #     net_operating_profit=self._oil_net_operating_profit,
        #     contractor_share=self._oil_ctr_share_after_transfer,
        # )
        #
        # self._gas_dmo_volume, self._gas_dmo_fee, self._gas_ddmo = psc_tools.get_dmo_gross_split(
        #     onstream_date=self.gas_onstream_date,
        #     start_date=self.start_date,
        #     project_years=self.project_years,
        #     dmo_holiday_duration=self.gas_dmo_holiday_duration,
        #     dmo_volume_portion=self.gas_dmo_volume_portion,
        #     dmo_fee_portion=self.gas_dmo_fee_portion,
        #     price=self._gas_wap_price,
        #     unrecovered_cost=self._gas_carward_cost_aftertf,
        #     is_dmo_end_weighted=is_dmo_end_weighted,
        #     net_operating_profit=self._gas_net_operating_profit,
        #     contractor_share=self._gas_ctr_share_after_transfer,
        # )
        #
        # # Taxable Income
        # self._oil_taxable_income = self._oil_net_operating_profit - self._oil_ddmo
        # self._gas_taxable_income = self._gas_net_operating_profit - self._gas_ddmo
        #
        # # Tax Payment
        # # Generating tax array based on the tax regime if tax_rate argument is None
        # if effective_tax_rate is None:
        #     self._tax_rate_arr = self._get_tax_by_regime(tax_regime=tax_regime)
        #
        # else:
        #     # Generating Tax array if tax_rate argument is a single value not array
        #     if isinstance(effective_tax_rate, float) or isinstance(effective_tax_rate, int):
        #         self._tax_rate_arr = np.full_like(
        #             self.project_years, effective_tax_rate, dtype=float
        #         )
        #
        #     # Tac rate argument is given as an array
        #     elif isinstance(effective_tax_rate, np.ndarray):
        #         self._tax_rate_arr = effective_tax_rate
        #
        # self._oil_tax = self._oil_taxable_income * self._tax_rate_arr
        # self._gas_tax = self._gas_taxable_income * self._tax_rate_arr
        #
        # # Contractor Net Share
        # self._oil_ctr_net_share = self._oil_taxable_income - self._oil_tax
        # self._gas_ctr_net_share = self._gas_taxable_income - self._gas_tax
        #
        # # Contractor Cash Flow
        # self._oil_ctr_cashflow = (
        #     self._oil_ctr_share_before_transfer
        #     - self._oil_total_expenses
        #     - self._oil_ddmo
        #     - self._oil_tax
        # )
        #
        # self._gas_ctr_cashflow = (
        #     self._gas_ctr_share_before_transfer
        #     - self._gas_total_expenses
        #     - self._gas_ddmo
        #     - self._gas_tax
        # )
        #
        # # Government Take
        # self._oil_government_take = self._oil_gov_share + self._oil_ddmo + self._oil_tax
        # self._gas_government_take = self._gas_gov_share + self._gas_ddmo + self._gas_tax
        #
        # # Prepare consolidated attributes
        # self._get_consolidated_profiles()

    def get_summary(
        self,
        discount_rate: float = 0.1,
        npv_mode: NPVSelection = NPVSelection.NPV_SKK_REAL_TERMS,
        discounting_mode: DiscountingMode = DiscountingMode.END_YEAR,
        discount_rate_start_year: int | None = None,
        inflation_rate: np.ndarray | float = 0.0,
        profitability_discounted: bool = False,
    ) -> dict:
        """
        Generate a comprehensive project economic summary under the Gross Split
        PSC scheme.

        This method computes key production, cost, tax, and profitability indicators,
        and returns the overall project summary in dictionary form.

        The summary includes metrics such as lifting volumes, gross revenues,
        investment costs, NPV, IRR, PI, and government take.

        Parameters
        ----------
        discount_rate : float, default=0.1
            The discount rate applied for NPV calculations.

        npv_mode : NPVSelection, default=NPVSelection.NPV_SKK_REAL_TERMS
            The NPV calculation mode. Options include:
            - ``NPV_SKK_REAL_TERMS`` : SKK Migas real-term NPV.
            - ``NPV_SKK_NOMINAL_TERMS`` : SKK Migas nominal-term NPV.
            - ``NPV_NOMINAL_TERMS`` : Standard nominal-term NPV.
            - ``NPV_REAL_TERMS`` : Standard real-term NPV.
            - ``NPV_POINT_FORWARD`` : Point-forward NPV.

        discounting_mode : DiscountingMode, default=DiscountingMode.END_YEAR
            The timing convention for discounting cashflows (e.g., start-year or end-year).

        discount_rate_start_year : int or None, optional
            The reference year for discounting. If ``None``, the project start year is used.
            Must lie between project start and end years.

        inflation_rate : float or ndarray, default=0.0
            Inflation rate(s) applied for real-term NPV calculations. Can be a scalar or
            array of yearly inflation rates.

        profitability_discounted : bool, default=False
            If ``True``, the profitability index (PI) is computed using discounted
            investment (NPV-based). If ``False``, the PI uses undiscounted investment.

        Returns
        -------
        summary : dict
            A dictionary containing the key project summary indicators.

        Raises
        ------
        GrossSplitSummaryException
            If `discount_rate_start_year` is before the project start year or
            after the project end year.

        Notes
        -----
        - This method must run the full model first via :meth:`run` before generating
          the summary results.
        - Economic indicators are computed according to SKK Migas evaluation conventions.
        """

        # Prepare discount rate start year
        if discount_rate_start_year is None:
            discount_rate_start_year = self.start_date.year

        # Cannot have discount rate year before the start year of the project
        if discount_rate_start_year < self.start_date.year:
            raise GrossSplitSummaryException(
                f"The discounting reference year ({discount_rate_start_year}) "
                f"is before start year of the project ({self.start_date.year})."
            )

        # Cannot have discount rate year after the end year of the project
        if discount_rate_start_year > self.end_date.year:
            raise GrossSplitSummaryException(
                f"The discounting reference year ({discount_rate_start_year}) "
                f"is after the end year of the project ({self.end_date.year})."
            )

        # Prepare OIL lifting summary
        oil_lifting_ghv = self._oil_lifting.get_lifting_rate_ghv_arr()
        oil_lifting_ghv_sum = np.sum(oil_lifting_ghv, dtype=float)
        oil_wap_sum = self._calc_division(
            numerator=self._oil_revenue.sum(dtype=float), denominator=oil_lifting_ghv_sum
        )

        # Prepare GAS lifting summary
        gas_lifting_ghv = self._gas_lifting.get_lifting_rate_ghv_arr()
        gas_lifting_ghv_sum = np.sum(gas_lifting_ghv, dtype=float)
        gas_wap_sum = self._calc_division(
            numerator=np.sum(self._gas_wap_price * gas_lifting_ghv),
            denominator=gas_lifting_ghv_sum,
        )

        # Prepare gross revenue summary
        oil_gross_revenue_sum = self._oil_revenue.sum(dtype=float)
        gas_gross_revenue_sum = self._gas_revenue.sum(dtype=float)
        total_gross_revenue_sum = oil_gross_revenue_sum + gas_gross_revenue_sum

        # Prepare sunk cost summary
        sunk_cost_sum = np.sum(self._oil_sunk_cost + self._gas_sunk_cost, dtype=float)

        # Prepare preonstream costs
        preonstream_map = {
            "oil_intangible": self._oil_intangible_preonstream,
            "gas_intangible": self._gas_intangible_preonstream,
            "oil_opex": self._oil_opex_preonstream,
            "gas_opex": self._gas_opex_preonstream,
            "oil_asr": self._oil_asr_preonstream,
            "gas_asr": self._gas_asr_preonstream,
            "oil_lbt": self._oil_lbt_preonstream,
            "gas_lbt": self._gas_lbt_preonstream,
        }

        preonstream_costs = {
            key: val.expenditures_pre_tax() for key, val in preonstream_map.items()
        }

        # Prepare tangible and intangible cost summary
        tangible_cost = (
            self._oil_capital_expenditures_post_tax
            + self._gas_capital_expenditures_post_tax
            + self._oil_depreciable_preonstream
            + self._gas_depreciable_preonstream
        )

        intangible_cost = (
            self._oil_intangible_expenditures_post_tax
            + self._gas_intangible_expenditures_post_tax
            + preonstream_costs["oil_intangible"]
            + preonstream_costs["gas_intangible"]
        )

        tangible_sum = tangible_cost.sum(dtype=float)
        intangible_sum = intangible_cost.sum(dtype=float)
        investment_sum = tangible_sum + intangible_sum

        # Prepare OPEX summary
        opex_cost = (
            self._oil_opex_expenditures_post_tax
            + self._gas_opex_expenditures_post_tax
            + preonstream_costs["oil_opex"]
            + preonstream_costs["gas_opex"]
        )

        opex_sum = opex_cost.sum(dtype=float)

        # Prepare ASR summary
        asr_cost = (
            self._oil_asr_expenditures_post_tax
            + self._gas_asr_expenditures_post_tax
            + preonstream_costs["oil_asr"]
            + preonstream_costs["gas_asr"]
        )

        asr_sum = asr_cost.sum(dtype=float)

        # Prepare LBT summary
        lbt_cost = (
            self._oil_lbt_expenditures_post_tax
            + self._gas_lbt_expenditures_post_tax
            + preonstream_costs["oil_lbt"]
            + preonstream_costs["gas_lbt"]
        )

        lbt_sum = lbt_cost.sum(dtype=float)

        # Prepare indirect taxes summary
        oil_indirect_tax_sum = self._oil_total_indirect_tax.sum(dtype=float)
        gas_indirect_tax_sum = self._gas_total_indirect_tax.sum(dtype=float)

        # Prepare carry forward depreciation summary
        oil_carward_depreciation_sum = self._oil_carry_forward_depreciation.sum(dtype=float)
        gas_carward_depreciation_sum = self._gas_carry_forward_depreciation.sum(dtype=float)
        total_carward_depreciation_sum = (
            oil_carward_depreciation_sum + gas_carward_depreciation_sum
        )

        # Prepare undepreciated assets summary
        oil_undepreciated_asset_sum = np.sum(
            [undepr.sum() for undepr in self._oil_undepreciated_assets.values()]
        )
        gas_undepreciated_asset_sum = np.sum(
            [undepr.sum() for undepr in self._gas_undepreciated_assets.values()]
        )
        total_undepreciated_asset_sum = (
            oil_undepreciated_asset_sum + gas_undepreciated_asset_sum
        )

        # Prepare government DDMO summary
        gov_ddmo = self._consolidated_ddmo.sum(dtype=float)

        # Prepare government take summary
        gov_take_income_sum = self._consolidated_tax_payment.sum(dtype=float)
        gov_take_sum = self._consolidated_government_take.sum(dtype=float)
        gov_take_over_gross_rev = self._calc_division(
            numerator=gov_take_sum, denominator=total_gross_revenue_sum
        )

        # Calculate IRR
        ctr_irr = irr(cashflow=self._consolidated_cashflow)

        # Calculate NPV
        # NPV method: SKK real terms
        if npv_mode == NPVSelection.NPV_SKK_REAL_TERMS:

            # Contractor NPV
            ctr_npv = npv_skk_real_terms(
                cashflow=self._consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_skk_real_terms(
                cashflow=tangible_cost + intangible_cost,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Government take NPV
            gov_take_npv = npv_skk_real_terms(
                cashflow=self._consolidated_government_take,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

        # NPV method: SKK nominal terms
        elif npv_mode == NPVSelection.NPV_SKK_NOMINAL_TERMS:

            # Contractor NPV
            ctr_npv = npv_skk_nominal_terms(
                cashflow=self._consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_skk_nominal_terms(
                cashflow=tangible_cost + intangible_cost,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                discounting_mode=discounting_mode,
            )

            # Government take NPV
            gov_take_npv = npv_skk_nominal_terms(
                cashflow=self._consolidated_government_take,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                discounting_mode=discounting_mode,
            )

        # NPV method: nominal terms
        elif npv_mode == NPVSelection.NPV_NOMINAL_TERMS:

            # Contractor NPV
            ctr_npv = npv_nominal_terms(
                cashflow=self._consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_nominal_terms(
                cashflow=tangible_cost + intangible_cost,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Government take NPV
            gov_take_npv = npv_nominal_terms(
                cashflow=self._consolidated_government_take,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

        # NPV method: real terms
        elif npv_mode == NPVSelection.NPV_REAL_TERMS:

            # Contractor NPV
            ctr_npv = npv_real_terms(
                cashflow=self._consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                inflation_rate=inflation_rate,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_real_terms(
                cashflow=tangible_cost + intangible_cost,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                inflation_rate=inflation_rate,
                discounting_mode=discounting_mode,
            )

            # Government take NPV
            gov_take_npv = npv_real_terms(
                cashflow=self._consolidated_government_take,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                inflation_rate=inflation_rate,
                discounting_mode=discounting_mode,
            )

        # NPV method: point forward
        else:
            # Contractor NPV
            ctr_npv = npv_point_forward(
                cashflow=self._consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_point_forward(
                cashflow=tangible_cost + intangible_cost,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Government take NPV
            gov_take_npv = npv_point_forward(
                cashflow=self._consolidated_government_take,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

        # Contractor present value ratio to the investment NPV
        # Profitability Index is calculated using discounted investment
        if profitability_discounted:
            ctr_pv_ratio = self._calc_division(numerator=ctr_npv, denominator=investment_npv)

        else:
            investment_pi = np.sum(tangible_cost + intangible_cost)
            ctr_pv_ratio = self._calc_division(numerator=ctr_npv, denominator=investment_pi)

        ctr_pi = 1 + ctr_pv_ratio

        # Contractor POT
        ctr_pot = pot_psc(
            cashflow=self._consolidated_cashflow,
            cashflow_years=self.project_years,
            reference_year=discount_rate_start_year,
        )

        # Prepare deductible cost summary
        deductible_cost_sum = self._consolidated_deductible_cost.sum(dtype=float)
        deductible_cost_over_gross_rev = self._calc_division(
            numerator=deductible_cost_sum, denominator=total_gross_revenue_sum
        )

        # Prepare carry forward deductible cost summary
        carry_forward_deductible_cost = self._consolidated_carward_deduct_cost[-1]
        carry_forcost_over_gross_share = self._calc_division(
            numerator=carry_forward_deductible_cost, denominator=total_gross_revenue_sum
        )
        carry_forcost_over_deductible_cost = self._calc_division(
            numerator=carry_forward_deductible_cost, denominator=deductible_cost_sum
        )

        # Prepare contractor gross share summary
        ctr_gross_share_sum = self._consolidated_ctr_share_before_tf.sum(dtype=float)

        # Prepare government gross share summary
        gov_gross_share_sum = self._consolidated_gov_share_before_tf.sum(dtype=float)

        # Prepare contractor net share summary
        ctr_net_share_sum = self._consolidated_ctr_net_share.sum(dtype=float)
        ctr_net_share_over_gross_share = self._calc_division(
            numerator=ctr_net_share_sum, denominator=total_gross_revenue_sum
        )

        # Prepare contractor net cashflow summary
        # For NPV POINT FORWARD
        if npv_mode == NPVSelection.NPV_POINT_FORWARD:
            # Modify contractor net cashflow since the cashflow before discount rate
            # start year is neglected
            ref_year_arr = np.full_like(
                self._consolidated_cashflow, fill_value=discount_rate_start_year
            )
            cashflow_point_forward = np.where(
                self.project_years >= ref_year_arr,
                self._consolidated_cashflow,
                0,
            )
            gross_revenue_point_forward = np.where(
                self.project_years >= ref_year_arr,
                self._consolidated_cashflow,
                0,
            )

            # Contractor net cashflow
            ctr_net_cashflow_sum = cashflow_point_forward.sum(dtype=float)
            ctr_net_cashflow_over_gross_rev = self._calc_division(
                numerator=ctr_net_cashflow_sum,
                denominator=gross_revenue_point_forward.sum(dtype=float),
            )

        # For other NPV calculation methods
        else:
            ctr_net_cashflow_sum = self._consolidated_cashflow.sum(dtype=float)
            ctr_net_cashflow_over_gross_rev = self._calc_division(
                numerator=ctr_net_cashflow_sum, denominator=total_gross_revenue_sum
            )

        return {
            "lifting_oil": oil_lifting_ghv_sum,
            "oil_wap": oil_wap_sum,
            "lifting_gas": gas_lifting_ghv_sum,
            "gas_wap": gas_wap_sum,
            "gross_revenue": total_gross_revenue_sum,
            "gross_revenue_oil": oil_gross_revenue_sum,
            "gross_revenue_gas": gas_gross_revenue_sum,
            "ctr_gross_share": ctr_gross_share_sum,
            "gov_gross_share": gov_gross_share_sum,
            "investment": investment_sum,
            "oil_capex": self._oil_capital.sum(),
            "gas_capex": self._gas_capital.sum(),
            "sunk_cost": sunk_cost_sum,
            "tangible": tangible_sum,
            "intangible": intangible_sum,
            "opex_asr_lbt": opex_sum + asr_sum + lbt_sum,
            "opex": opex_sum,
            "asr": asr_sum,
            "lbt": lbt_sum,
            "cost_recovery/deductible_cost": deductible_cost_sum,
            "cost_recovery_over_gross_rev": deductible_cost_over_gross_rev,
            "unrec_cost": carry_forward_deductible_cost,
            "unrec_over_costrec": carry_forcost_over_deductible_cost,
            "unrec_over_gross_rev": carry_forcost_over_gross_share,
            "ctr_net_share": ctr_net_share_sum,
            "ctr_net_share_over_gross_share": ctr_net_share_over_gross_share,
            "ctr_net_cashflow": ctr_net_cashflow_sum,
            "ctr_net_cashflow_over_gross_rev": ctr_net_cashflow_over_gross_rev,
            "ctr_npv": ctr_npv,
            "ctr_npv_sunk_cost_pooled": ctr_npv,
            "ctr_irr": ctr_irr,
            "ctr_irr_sunk_cost_pooled": ctr_irr,
            "ctr_pot": ctr_pot,
            "ctr_pv_ratio": ctr_pv_ratio,
            "ctr_pi": ctr_pi,
            "gov_ddmo": gov_ddmo,
            "gov_tax_income": gov_take_income_sum,
            "gov_take": gov_take_sum,
            "gov_take_over_gross_rev": gov_take_over_gross_rev,
            "gov_take_npv": gov_take_npv,
            "gov_ftp_share": 0.0,
            "undepreciated_asset_oil": oil_undepreciated_asset_sum,
            "undepreciated_asset_gas": gas_undepreciated_asset_sum,
            "undepreciated_asset_total": total_undepreciated_asset_sum,
            "total_indirect_taxes": oil_indirect_tax_sum + gas_indirect_tax_sum,
            "oil_indirect_taxes": oil_indirect_tax_sum,
            "gas_indirect_taxes": gas_indirect_tax_sum,
            "total_carry_forward_depreciation": total_carward_depreciation_sum,
            "oil_carry_forward_depreciation": oil_carward_depreciation_sum,
            "gas_carry_forward_depreciation": gas_carward_depreciation_sum,
        }
