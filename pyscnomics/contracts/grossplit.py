"""
Handles calculations associated with PSC Gross Split.
"""
import warnings
from dataclasses import dataclass, field
import numpy as np

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts import psc_tools
from pyscnomics.econ.selection import (FluidType,
                                       GrossSplitRegime,
                                       TaxRegime,
                                       TaxType,
                                       DeprMethod,
                                       OtherRevenue,
                                       InflationAppliedTo,
                                       VariableSplit522017,
                                       VariableSplit082017,
                                       VariableSplit132024)
from pyscnomics.econ.results import CashFlow
from pyscnomics.econ.depreciation import unit_of_production_rate


class GrossSplitException(Exception):
    """Exception to raise for a misuse of GrossSplit class"""

    pass


class SunkCostException(Exception):
    """Exception to raise for a misuse of Sunk Cost Method"""

    pass


class CumulativeProductionSplitException(Exception):
    """Exception to raise for a misuse of Cumulative Production Split Offset"""

    pass


@dataclass
class GrossSplit(BaseProject):
    """
    Dataclass that represents Gross Split (GS) contract.

    Parameters
    ----------
    field_status: str | VariableSplit522017.FieldStatus | VariableSplit082017.FieldStatus
        The field status of the corresponding contract.
    field_loc: str | VariableSplit522017.FieldLocation | VariableSplit082017.FieldLocation
        The field location of the corresponding contract.
    res_depth: str | VariableSplit522017.ReservoirDepth | VariableSplit082017.ReservoirDepth
        The reservoir depth of the corresponding contract
    infra_avail: str | VariableSplit522017.InfrastructureAvailability | VariableSplit082017.InfrastructureAvailability
        The infrastructure availability of the corresponding contract.
    res_type: str | VariableSplit522017.ReservoirType | VariableSplit082017.ReservoirType
        The reservoir type of the corresponding contract.
    api_oil: str | VariableSplit522017.APIOil | VariableSplit082017.APIOil
        The Oil's API of the corresponding contract.
    domestic_use: str | VariableSplit522017.DomesticUse | VariableSplit082017.DomesticUse
        The domestic use or local content of the corresponding contract.
    prod_stage: str | VariableSplit522017.ProductionStage | VariableSplit082017.ProductionStage
        The production stage of the corresponding contract.
    co2_content: str | VariableSplit522017.CO2Content | VariableSplit082017.CO2Content
        The CO2 content of the corresponding contract.
    h2s_content: str | VariableSplit522017.H2SContent | VariableSplit082017.H2SContent
        The H2S content of the corresponding contract.
    base_split_ctr_oil: float
        The contractor base split for oil of the corresponding contract.
    base_split_ctr_gas: float
        The contractor base split for gas of the corresponding contract.
    split_ministry_disc: float
        The ministerial discretion split of the corresponding contract.
    oil_dmo_volume_portion: float
        The Oil's Domestic Market Obligation (DMO) volume portion.
    oil_dmo_fee_portion: float
        The Oil's DMO fee portion.
    oil_dmo_holiday_duration: int
        The duration of the Oil DMO Holiday in month unit.
    gas_dmo_volume_portion: float
        The Gas's Domestic Market Obligation (DMO) volume portion.
    gas_dmo_fee_portion: float
        The Gas's DMO fee portion.
    gas_dmo_holiday_duration: int
        The duration of the Gas's DMO Holiday in month unit.
    """
    field_status: str | VariableSplit522017.FieldStatus | VariableSplit082017.FieldStatus | None = field(default='No POD')
    field_loc: str | VariableSplit522017.FieldLocation | VariableSplit082017.FieldLocation | VariableSplit132024.FieldLocation= field(default='Onshore')
    res_depth: str | VariableSplit522017.ReservoirDepth | VariableSplit082017.ReservoirDepth | None  = field(default='<=2500')
    infra_avail: str | VariableSplit522017.InfrastructureAvailability | VariableSplit082017.InfrastructureAvailability | VariableSplit132024.InfrastructureAvailability = field(default='Well Developed')
    res_type: str | VariableSplit522017.ReservoirType | VariableSplit082017.ReservoirType | None = field(default='Conventional')
    api_oil: str | VariableSplit522017.APIOil | VariableSplit082017.APIOil | None = field(default='<25')
    domestic_use: str | VariableSplit522017.DomesticUse | VariableSplit082017.DomesticUse | None = field(default='50<=x<70')
    prod_stage: str | VariableSplit522017.ProductionStage | VariableSplit082017.ProductionStage | None = field(default='Secondary')
    co2_content: str | VariableSplit522017.CO2Content | VariableSplit082017.CO2Content | None = field(default='<5')
    h2s_content: str | VariableSplit522017.H2SContent | VariableSplit082017.H2SContent | None = field(default='<100')
    field_reserves: str | VariableSplit132024.FieldReservesAmount | None = field(default=VariableSplit132024.FieldReservesAmount.HIGH)

    base_split_ctr_oil: float = field(default=0.43)
    base_split_ctr_gas: float = field(default=0.48)
    split_ministry_disc: float = field(default=0.08)

    oil_dmo_volume_portion: float = field(default=0.25)
    oil_dmo_fee_portion: float = field(default=1.0)
    oil_dmo_holiday_duration: int = field(default=60)

    gas_dmo_volume_portion: float = field(default=1.0)
    gas_dmo_fee_portion: float = field(default=1.0)
    gas_dmo_holiday_duration: int = field(default=60)

    conversion_boe_to_scf: float = field(default=5.615, init=False, repr=False)

    _oil_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _gas_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _oil_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)
    _gas_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)

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
    _oil_ctr_split: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_split: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ctr_share_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_share_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _oil_gov_share: np.ndarray = field(default=None, init=False, repr=False)
    _gas_gov_share: np.ndarray = field(default=None, init=False, repr=False)
    _oil_total_expenses: np.ndarray = field(default=None, init=False, repr=False)
    _gas_total_expenses: np.ndarray = field(default=None, init=False, repr=False)

    _oil_amortization: np.ndarray = field(default=None, init=False, repr=False)
    _gas_amortization: np.ndarray = field(default=None, init=False, repr=False)

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

    _oil_dmo_volume: np.ndarray = field(default=None, init=False, repr=False)
    _oil_dmo_fee: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ddmo: np.ndarray = field(default=None, init=False, repr=False)

    _gas_dmo_volume: np.ndarray = field(default=None, init=False, repr=False)
    _gas_dmo_fee: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ddmo: np.ndarray = field(default=None, init=False, repr=False)

    _oil_taxable_income: np.ndarray = field(default=None, init=False, repr=False)
    _gas_taxable_income: np.ndarray = field(default=None, init=False, repr=False)

    _tax_rate_arr: np.ndarray = field(default=None, init=False, repr=False)

    _oil_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_tax: np.ndarray = field(default=None, init=False, repr=False)

    _oil_ctr_net_share: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_net_share: np.ndarray = field(default=None, init=False, repr=False)

    _oil_ctr_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_cashflow: np.ndarray = field(default=None, init=False, repr=False)

    _oil_government_take: np.ndarray = field(default=None, init=False, repr=False)
    _gas_government_take: np.ndarray = field(default=None, init=False, repr=False)

    # Consolidated Attributes
    _consolidated_capital_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_intangible_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_opex_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_asr_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_lbt_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_cost_of_sales_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)

    _consolidated_capital_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_intangible_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_opex_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_asr_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_lbt_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_cost_of_sales_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)

    _consolidated_capital_cost: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_intangible: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_opex: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_asr: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_lbt: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_non_capital: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ctr_share_before_tf: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_gov_share_before_tf: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_total_expenses: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_cost_tobe_deducted: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_carward_deduct_cost: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_deductible_cost: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_carward_cost_aftertf: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ctr_share_after_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_net_operating_profit: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_dmo_volume: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_dmo_fee: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ddmo: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_taxable_income: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_tax_payment: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ctr_net_share: np.ndarray = field(default=None, init=False, repr=False)

    _consolidated_amortization: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes fof containing the 100% contractor split warning
    _oil_ctr_split_prior_bracket: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_split_prior_bracket: np.ndarray = field(default=None, init=False, repr=False)
    _oil_year_maximum_ctr_split: np.ndarray = field(default=None, init=False, repr=False)
    _gas_year_maximum_ctr_split: np.ndarray = field(default=None, init=False, repr=False)


    def _check_attributes(self):
        """
        Function to check the Gross Split input.
        -------

        """
        # Defining any attributes that in the form of fraction
        fraction_attributes = (
            ('base_split_ctr_oil', self.base_split_ctr_oil),
            ('base_split_ctr_gas', self.base_split_ctr_gas),
            ('split_ministry_disc', self.split_ministry_disc),
            ('oil_dmo_volume_portion', self.oil_dmo_volume_portion),
            ('oil_dmo_fee_portion', self.oil_dmo_fee_portion),
            ('gas_dmo_volume_portion', self.gas_dmo_volume_portion),
            ('gas_dmo_fee_portion', self.gas_dmo_fee_portion)
        )

        for attr_name, attr in fraction_attributes:
            if attr > 1.0 or attr < 0.0:
                range_type = 'exceeding 1.0' if attr > 1.0 else 'below 0.0'
                raise GrossSplitException(
                    f"The {attr_name} value, {attr}, is {range_type}. "
                    f"The allowed range for this attribute is between 0.0 and 1.0."
                )

        discrete_attributes = (
            ('oil_dmo_holiday_duration', self.oil_dmo_holiday_duration),
            ('gas_dmo_holiday_duration', self.gas_dmo_holiday_duration)
        )

        for attr_name, attr in discrete_attributes:
            if attr < 0:
                raise GrossSplitException(
                    f"The {attr_name} value, {attr}, is below 0. "
                    f"The minimum value for this attribute is 0."
                )

    def _wrapper_variable_split(self,
                                regime: GrossSplitRegime = GrossSplitRegime.PERMEN_ESDM_20_2019):
        """
        Function to wrap the variable split function.

        Parameters
        ----------
        regime: GrossSplitRegime
         The selection of the Gross Split Regime

        Returns
        -------
        variable_split_func
            The function of the variable split

        Notes
        -------
        (1) Gross Split Regime PERMEN ESDM No. 52 Tahun 2017, PERMEN ESDM No. 20 Tahun 2019,
        and PERMEN ESDM No. 12 Tahun 2020 are having the same variable split value. The complete differences
        could be seen in official documents.
        """

        if regime == GrossSplitRegime.PERMEN_ESDM_8_2017:
            variable_split_func = self._get_var_split_08_2017()

        elif (regime == GrossSplitRegime.PERMEN_ESDM_52_2017 or
              regime == GrossSplitRegime.PERMEN_ESDM_20_2019 or
              regime == GrossSplitRegime.PERMEN_ESDM_12_2020):
            variable_split_func = self._get_var_split_52_2017()

        elif regime == GrossSplitRegime.PERMEN_ESDM_13_2024:
            variable_split_func = self._get_var_split_13_2024()

        else:
            raise GrossSplitException(
                f"The Gross Split regime, {regime}, is not recognized"
            )

        return variable_split_func

    def _get_var_split_08_2017(self):
        """
        A function to get the value of Variable Split based on the given parameters.

        Returns
        -------
        _variable_split: float
            The value of variable split.
        """
        # The string parameters is being keep to backward compatibility
        params_str = {
            'field_status': {
                'POD I': 0.05,
                'POD II': 0.0,
                'POFD': 0.0,
                'No POD': -0.05,
            },
            'field_loc': {
                'Onshore': 0.0,
                'Offshore (0<h<=20)': 0.08,
                'Offshore (20<h<=50)': 0.1,
                'Offshore (50<h<=150)': 0.12,
                'Offshore (150<h<=1000)': 0.14,
                'Offshore (h>1000)': 0.16,
            },
            'res_depth': {
                '<=2500': 0.0,
                '>2500': 0.01,
            },
            'infra_avail': {
                'Well Developed': 0.0,
                'New Frontier': 0.02,
            },
            'res_type': {
                'Conventional': 0.0,
                'Non Conventional': 0.16,
            },
            'co2_content': {
                '<5': 0.0,
                '5<=x<10': 0.005,
                '10<=x<20': 0.01,
                '20<=x<40': 0.015,
                '40<=x<60': 0.02,
                'x>=60': 0.04,
            },
            'h2s_content': {
                '<100': 0.0,
                '100<=x<300': 0.005,
                '300<=x<500': 0.0075,
                'x>=500': 0.01,
            },
            'api_oil': {
                '<25': 0.01,
                '>=25': 0.0,
            },
            'domestic_use': {
                '<30': 0.0,
                '30<=x<50': 0.02,
                '50<=x<70': 0.03,
                '70<=x<100': 0.04,
            },
            'prod_stage': {
                'Primary': 0.0,
                'Secondary': 0.03,
                'Tertiary': 0.05,
            }
        }

        params_enum = {
            'field_status': {
                VariableSplit082017.FieldStatus.POD_I: 0.05,
                VariableSplit082017.FieldStatus.POD_II: 0.0,
                VariableSplit082017.FieldStatus.POFD: 0.0,
                VariableSplit082017.FieldStatus.NO_POD: -0.05,
            },
            'field_loc': {
                VariableSplit082017.FieldLocation.ONSHORE: 0.0,
                VariableSplit082017.FieldLocation.OFFSHORE_0_UNTIL_LESSEQUAL_20: 0.08,
                VariableSplit082017.FieldLocation.OFFSHORE_20_UNTIL_LESSEQUAL_50: 0.1,
                VariableSplit082017.FieldLocation.OFFSHORE_50_UNTIL_LESSEQUAL_150: 0.12,
                VariableSplit082017.FieldLocation.OFFSHORE_150_UNTIL_LESSEQUAL_1000: 0.14,
                VariableSplit082017.FieldLocation.OFFSHORE_GREATERTHAN_1000: 0.16,
            },
            'res_depth': {
                VariableSplit082017.ReservoirDepth.LESSEQUAL_2500: 0.0,
                VariableSplit082017.ReservoirDepth.GREATERTHAN_2500: 0.01,
            },
            'infra_avail': {
                VariableSplit082017.InfrastructureAvailability.WELL_DEVELOPED: 0.0,
                VariableSplit082017.InfrastructureAvailability.NEW_FRONTIER: 0.02,
            },
            'res_type': {
                VariableSplit082017.ReservoirType.CONVENTIONAL: 0.0,
                VariableSplit082017.ReservoirType.NON_CONVENTIONAL: 0.16,
            },
            'co2_content': {
                VariableSplit082017.CO2Content.LESSTHAN_5: 0.0,
                VariableSplit082017.CO2Content.EQUAL_5_UNTIL_LESSTHAN_10: 0.005,
                VariableSplit082017.CO2Content.EQUAL_10_UNTIL_LESSTHAN_20: 0.01,
                VariableSplit082017.CO2Content.EQUAL_20_UNTIL_LESSTHAN_40: 0.015,
                VariableSplit082017.CO2Content.EQUAL_40_UNTIL_LESSTHAN_60: 0.02,
                VariableSplit082017.CO2Content.EQUALGREATERTHAN_60: 0.04,
            },
            'h2s_content': {
                VariableSplit082017.H2SContent.LESSTHAN_100: 0.0,
                VariableSplit082017.H2SContent.EQUAL_100_UNTIL_LESSTHAN_300: 0.005,
                VariableSplit082017.H2SContent.EQUAL_300_UNTIL_LESSTHAN_500: 0.0075,
                VariableSplit082017.H2SContent.EQUALGREATERTHAN_500: 0.01,
            },
            'api_oil': {
                VariableSplit082017.APIOil.LESSTHAN_25: 0.01,
                VariableSplit082017.APIOil.EQUALGREATERTHAN_25: 0.0,
            },
            'domestic_use': {
                VariableSplit082017.DomesticUse.LESSTHAN_30: 0.0,
                VariableSplit082017.DomesticUse.EQUAL_30_UNTIL_LESSTHAN_50: 0.02,
                VariableSplit082017.DomesticUse.EQUAL_50_UNTIL_LESSTHAN_70: 0.03,
                VariableSplit082017.DomesticUse.EQUAL_70_UNTIL_LESSTHAN_100: 0.04,
            },
            'prod_stage': {
                VariableSplit082017.ProductionStage.PRIMARY: 0.0,
                VariableSplit082017.ProductionStage.SECONDARY: 0.03,
                VariableSplit082017.ProductionStage.TERTIARY: 0.05,
            }
        }
        source_dict = {
            'field_status': self.field_status,
            'field_loc': self.field_loc,
            'res_depth': self.res_depth,
            'infra_avail': self.infra_avail,
            'res_type': self.res_type,
            'api_oil': self.api_oil,
            'domestic_use': self.domestic_use,
            'prod_stage': self.prod_stage,
            'co2_content': self.co2_content,
            'h2s_content': self.h2s_content,
        }

        variable_split = np.array([
            params_str[key][param] if isinstance(param, str) else params_enum[key][param]
            for key, param in source_dict.items()
        ], dtype=float)

        self._variable_split = float(np.sum(variable_split))

    def _get_var_split_52_2017(self):
        """
        A function to get the value of Variable Split based on the given parameters.

        Returns
        -------
        _variable_split: float
            The value of variable split.
        """
        params_str = {
            'field_status': {
                'POD I': 0.05,
                'POD II': 0.03,
                'No POD': 0,
            },
            'field_loc': {
                'Onshore': 0,
                'Offshore (0<h<=20)': 0.08,
                'Offshore (20<h<=50)': 0.1,
                'Offshore (50<h<=150)': 0.12,
                'Offshore (150<h<=1000)': 0.14,
                'Offshore (h>1000)': 0.16,
            },
            'res_depth': {
                '<=2500': 0,
                '>2500': 0.01,
            },
            'infra_avail': {
                'Well Developed': 0,
                'New Frontier Offshore': 0.02,
                'New Frontier Onshore': 0.04,
            },
            'res_type': {
                'Conventional': 0,
                'Non Conventional': 0.16,
            },
            'co2_content': {
                '<5': 0,
                '5<=x<10': 0.005,
                '10<=x<20': 0.01,
                '20<=x<40': 0.015,
                '40<=x<60': 0.02,
                'x>=60': 0.04,
            },
            'h2s_content': {
                '<100': 0,
                '100<=x<1000': 0.01,
                '1000<=x<2000': 0.02,
                '2000<=x<3000': 0.03,
                '3000<=x<4000': 0.04,
                'x>=4000': 0.05,
            },
            'api_oil': {
                '<25': 0.01,
                '>=25': 0,
            },
            'domestic_use': {
                '30<=x<50': 0.02,
                '50<=x<70': 0.03,
                '70<=x<100': 0.04,
            },
            'prod_stage': {
                'Primary': 0,
                'Secondary': 0.06,
                'Tertiary': 0.1,
            }
        }

        params_enum = {
            'field_status': {
                VariableSplit522017.FieldStatus.POD_I: 0.05,
                VariableSplit522017.FieldStatus.POD_II: 0.03,
                VariableSplit522017.FieldStatus.NO_POD: 0.0,
            },
            'field_loc': {
                VariableSplit522017.FieldLocation.ONSHORE: 0.0,
                VariableSplit522017.FieldLocation.OFFSHORE_0_UNTIL_LESSEQUAL_20: 0.08,
                VariableSplit522017.FieldLocation.OFFSHORE_20_UNTIL_LESSEQUAL_50: 0.1,
                VariableSplit522017.FieldLocation.OFFSHORE_50_UNTIL_LESSEQUAL_150: 0.12,
                VariableSplit522017.FieldLocation.OFFSHORE_150_UNTIL_LESSEQUAL_1000: 0.14,
                VariableSplit522017.FieldLocation.OFFSHORE_GREATERTHAN_1000: 0.16,
            },
            'res_depth': {
                VariableSplit522017.ReservoirDepth.LESSEQUAL_2500: 0.0,
                VariableSplit522017.ReservoirDepth.GREATERTHAN_2500: 0.01,
            },
            'infra_avail': {
                VariableSplit522017.InfrastructureAvailability.WELL_DEVELOPED: 0.0,
                VariableSplit522017.InfrastructureAvailability.NEW_FRONTIER_OFFSHORE: 0.02,
                VariableSplit522017.InfrastructureAvailability.NEW_FRONTIER_ONSHORE: 0.04,
            },
            'res_type': {
                VariableSplit522017.ReservoirType.CONVENTIONAL: 0.0,
                VariableSplit522017.ReservoirType.NON_CONVENTIONAL: 0.16,
            },
            'co2_content': {
                VariableSplit522017.CO2Content.LESSTHAN_5: 0.0,
                VariableSplit522017.CO2Content.EQUAL_5_UNTIL_LESSTHAN_10: 0.005,
                VariableSplit522017.CO2Content.EQUAL_10_UNTIL_LESSTHAN_20: 0.01,
                VariableSplit522017.CO2Content.EQUAL_20_UNTIL_LESSTHAN_40: 0.015,
                VariableSplit522017.CO2Content.EQUAL_40_UNTIL_LESSTHAN_60: 0.02,
                VariableSplit522017.CO2Content.EQUALGREATERTHAN_60: 0.04,
            },
            'h2s_content': {
                VariableSplit522017.H2SContent.LESSTHAN_100: 0.0,
                VariableSplit522017.H2SContent.EQUAL_100_UNTIL_LESSTHAN_1000: 0.01,
                VariableSplit522017.H2SContent.EQUAL_1000_UNTIL_LESSTHAN_2000: 0.02,
                VariableSplit522017.H2SContent.EQUAL_2000_UNTIL_LESSTHAN_3000: 0.03,
                VariableSplit522017.H2SContent.EQUAL_3000_UNTIL_LESSTHAN_4000: 0.04,
                VariableSplit522017.H2SContent.EQUALGREATERTHAN_4000: 0.05,
            },
            'api_oil': {
                VariableSplit522017.APIOil.LESSTHAN_25: 0.01,
                VariableSplit522017.APIOil.EQUALGREATERTHAN_25: 0.0,
            },
            'domestic_use': {
                VariableSplit522017.DomesticUse.EQUAL_30_UNTIL_LESSTHAN_50: 0.02,
                VariableSplit522017.DomesticUse.EQUAL_50_UNTIL_LESSTHAN_70: 0.03,
                VariableSplit522017.DomesticUse.EQUAL_70_UNTIL_LESSTHAN_100: 0.04,
            },
            'prod_stage': {
                VariableSplit522017.ProductionStage.PRIMARY: 0.0,
                VariableSplit522017.ProductionStage.SECONDARY: 0.06,
                VariableSplit522017.ProductionStage.TERTIARY: 0.1,
            }
        }

        source_dict = {
            'field_status': self.field_status,
            'field_loc': self.field_loc,
            'res_depth': self.res_depth,
            'infra_avail': self.infra_avail,
            'res_type': self.res_type,
            'api_oil': self.api_oil,
            'domestic_use': self.domestic_use,
            'prod_stage': self.prod_stage,
            'co2_content': self.co2_content,
            'h2s_content': self.h2s_content,
        }

        variable_split = np.array([
            params_str[key][param] if isinstance(param, str) else params_enum[key][param]
            for key, param in source_dict.items()
        ], dtype=float)

        self._variable_split = float(np.sum(variable_split))

    def _get_var_split_13_2024(self):
        """
        A function to get the value of Variable Split based on the given parameters.

        Returns
        -------
        _variable_split: float
            The value of variable split.
        """
        params_str = {
            'field_loc': {
                'onshore': 0.11,
                'shallow_offshore': 0.12,
                'deep_offshore': 0.13,
                'ultradeep_offshore': 0.14,
            },
            'infra_avail': {
                'available': 0.10,
                'partially_available': 0.11,
                'not_available': 0.13,
            },
            'field_reserves':{
                'low': 0.14,
                'medium': 0.13,
                'high': 0.12,
            }
        }

        params_enum = {
            'field_loc': {
                VariableSplit132024.FieldLocation.ONSHORE: 0.11,
                VariableSplit132024.FieldLocation.SHALLOW_OFFSHORE: 0.12,
                VariableSplit132024.FieldLocation.DEEP_OFFSHORE: 0.13,
                VariableSplit132024.FieldLocation.ULTRADEEP_OFFSHORE: 0.14,
            },
            'infra_avail': {
                VariableSplit132024.InfrastructureAvailability.AVAILABLE: 0.10,
                VariableSplit132024.InfrastructureAvailability.PARTIALLY_AVAILABLE: 0.11,
                VariableSplit132024.InfrastructureAvailability.NOT_AVAILABLE: 0.13,
            },
            'field_reserves':{
                VariableSplit132024.FieldReservesAmount.LOW: 0.14,
                VariableSplit132024.FieldReservesAmount.MEDIUM: 0.13,
                VariableSplit132024.FieldReservesAmount.HIGH: 0.12,
            }
        }

        source_dict = {
            'field_loc': self.field_loc,
            'infra_avail': self.infra_avail,
            'field_reserves': self.field_reserves,
        }

        variable_split = np.array([
            params_str[key][param] if isinstance(param, str) else params_enum[key][param]
            for key, param in source_dict.items()
        ], dtype=float)

        self._variable_split = float(np.sum(variable_split))

    def _wrapper_progressive_split(
            self,
            fluid: FluidType,
            price: np.ndarray,
            cum: np.ndarray,
            regime: GrossSplitRegime = GrossSplitRegime.PERMEN_ESDM_20_2019):

        if (regime == GrossSplitRegime.PERMEN_ESDM_52_2017 or
                regime == GrossSplitRegime.PERMEN_ESDM_20_2019 or
                regime == GrossSplitRegime.PERMEN_ESDM_12_2020):
            prog_price_split = self._get_prog_price_split_52_2017(fluid, price)
            prog_cum_split = self._get_prog_cum_split_52_2017(cum)

        elif regime == GrossSplitRegime.PERMEN_ESDM_8_2017:
            prog_price_split = self._get_prog_price_split_08_2017(fluid, price)
            prog_cum_split = self._get_prog_cum_split_08_2017(cum)

        elif regime == GrossSplitRegime.PERMEN_ESDM_13_2024:
            prog_price_split = self._get_prog_price_split_13_2024(fluid, price)
            prog_cum_split = cum * 0 # Multiplied by 0 since there is no terms in the Regulation and filling the vectorization

        else:
            prog_price_split = ValueError('Not Recognized Gross Split Regime')
            prog_cum_split = ValueError('Not Recognized Gross Split Regime')

        return prog_price_split, prog_cum_split

    @staticmethod
    def _get_prog_price_split_08_2017(
            fluid: FluidType,
            price: np.ndarray
    ):
        # Indonesia's Ministry Regulations No.08 The Year of 2017. At Appendix B Progressive Component
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
                ps = 0
        else:
            ps = 0

        return ps

    @staticmethod
    def _get_prog_price_split_52_2017(
            fluid: FluidType,
            price: np.ndarray
    ):
        # Indonesia's Ministry Regulations No.52 The Year of 2017. At Appendix B Progressive Component
        if fluid == FluidType.OIL:
            ps = np.where(price > 0,
                          (85 - price) * 0.25 / 100,
                          0)

        elif fluid == FluidType.GAS:
            if price < 7:
                ps = (7 - price) * 2.5 / 100
            elif (7 < price) and (price < 10):
                ps = 0
            elif price > 10:
                ps = (10 - price) * 2.5 / 100
            else:
                ps = 0
        else:
            raise ValueError('Unknown fluid type')

        return ps

    @staticmethod
    def _get_prog_price_split_13_2024(
            fluid: FluidType,
            price: np.ndarray
    ):
        if fluid == FluidType.OIL:
            if price <= 45:
                ps = 0.05
            elif 46 < price <= 65:
                ps = -0.0025 * (price - 45) + 0.05 # In the form of y2 = m * (x2 - x1) + y1
            elif 65 < price <= 85:
                ps = 0.0
            elif 85 < price <= 105:
                ps = -0.0025 * (price - 85) + 0.0 # In the form of y2 = m * (x2 - x1) + y1
            elif 105 < price:
                ps = -0.05
            else:
                ps = 0

        elif fluid == FluidType.GAS:
            if price <= 4:
                ps = 0.05
            elif 4 < price <= 7:
                ps = -0.0025 * (price - 4) + 0.05 # In the form of y2 = m * (x2 - x1) + y1
            elif 7 < price <= 10:
                ps = 0.0
            elif 10 < price <= 13:
                ps = -0.0025 * (price - 10) + 0.0 # In the form of y2 = m * (x2 - x1) + y1
            elif 13 < price:
                ps = -0.05
            else:
                ps = 0

        else:
            raise ValueError('Unknown fluid type')

        return ps


    @staticmethod
    def _get_prog_cum_split_08_2017(
            cum: np.ndarray | None,
    ):
        # Cumulative Progressive Split
        if cum is None:
            px = 0
        elif 0 < cum < 1000:
            px = 0.05
        elif 1000 <= cum < 10000:
            px = 0.04
        elif 10000 <= cum < 20000:
            px = 0.03
        elif 20000 <= cum < 50000:
            px = 0.02
        elif 50000 <= cum < 150000:
            px = 0.01
        elif 150000 <= cum:
            px = 0
        else:
            raise ValueError('No Regulation exist regarding the cumulative value')

        return px

    @staticmethod
    def _get_prog_cum_split_52_2017(
            cum: np.ndarray | None
    ):
        # Cumulative Progressive Split
        if cum is None:
            px = 0
        elif np.logical_and(np.greater_equal(cum, 0), np.less(cum, 30000)):
            px = 0.1
        elif np.logical_and(np.greater_equal(cum, 30000), np.less(cum, 60000)):
            px = 0.09
        elif np.logical_and(np.greater_equal(cum, 60000), np.less(cum, 90000)):
            px = 0.08
        elif np.logical_and(np.greater_equal(cum, 90000), np.less(cum, 125000)):
            px = 0.06
        elif np.logical_and(np.greater_equal(cum, 125000), np.less(cum, 175000)):
            px = 0.04
        elif np.greater_equal(cum, 175000):
            px = 0
        else:
            raise ValueError('No Regulation exist regarding the cumulative value')

        return px

    @staticmethod
    def _get_deductible_cost(ctr_gross_share, cost_tobe_deducted, carward_deduct_cost):

        carward_deduct_cost = np.concatenate((np.zeros(1), carward_deduct_cost[:-1]))

        return np.where(ctr_gross_share > (cost_tobe_deducted + carward_deduct_cost),
                        cost_tobe_deducted + carward_deduct_cost,
                        ctr_gross_share)

    # Todo (20 March 2025): Fix the sunk cost calculation method
    # def _get_sunk_cost(self, sunk_cost_reference_year: int):
    #     oil_cost_raw = (
    #             self._oil_capital_expenditures_post_tax
    #             + self._oil_non_capital
    #     )
    #     self._oil_sunk_cost = oil_cost_raw[
    #                           : (sunk_cost_reference_year - self.start_date.year + 1)
    #                           ]
    #
    #     gas_cost_raw = (
    #             self._gas_capital_expenditures_post_tax
    #             + self._gas_non_capital
    #     )
    #     self._gas_sunk_cost = gas_cost_raw[
    #                           : (sunk_cost_reference_year - self.start_date.year + 1)
    #                           ]
    #
    #     if sunk_cost_reference_year == self.start_date.year:
    #         self._oil_sunk_cost = np.zeros_like(self.project_years)
    #         self._gas_sunk_cost = np.zeros_like(self.project_years)

    def _get_year_maximum_split(
            self,
            ctr_split: np.ndarray,
            fluid: str
    ):
        """
        Function to get the years of when the contractor have maximum split 100% or more.

        Parameters
        ----------
        ctr_split: np.ndarray
            The array of the contractor split
        fluid: str
            The fluid type that the contractor split being observed.

        Returns
        -------
        out: np.ndarray
            The array of the years when the contractor have the maximum split 100%.

        """

        indices = (np.argwhere(ctr_split >= 1.0)).flatten()
        years_of_max = self.project_years[indices]
        if len(years_of_max)>0:
            warnings.warn(
                f"The {fluid} contractor split equal more than 100% are in the following years {years_of_max} ",
                UserWarning)
            warnings.simplefilter("default", UserWarning)
        else:
            pass

        return years_of_max

    def run(self,
            sulfur_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
            electricity_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_OIL_REVENUE,
            co2_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
            is_dmo_end_weighted=False,
            regime: GrossSplitRegime = GrossSplitRegime.PERMEN_ESDM_20_2019,
            tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
            effective_tax_rate: float | np.ndarray = 0.22,
            sunk_cost_reference_year: int = None,
            depr_method: DeprMethod = DeprMethod.PSC_DB,
            decline_factor: float | int = 2,
            year_inflation: np.ndarray = None,
            vat_rate: np.ndarray | float = 0.0,
            inflation_rate: np.ndarray | float = 0.0,
            inflation_rate_applied_to: InflationAppliedTo | None = InflationAppliedTo.CAPEX,
            cum_production_split_offset: float | np.ndarray | None = 0.0,
            amortization: bool = False,
            sum_undepreciated_cost: bool = False
            ):

        # Configure Sunk Cost Reference Year
        if sunk_cost_reference_year is None:
            sunk_cost_reference_year = self.start_date.year

        if sunk_cost_reference_year > self.oil_onstream_date.year:
            raise SunkCostException(
                f"Sunk Cost Reference Year {sunk_cost_reference_year} "
                f"is after the on stream date: {self.oil_onstream_date}"
            )

        if sunk_cost_reference_year > self.gas_onstream_date.year:
            raise SunkCostException(
                f"Sunk Cost Reference Year {sunk_cost_reference_year} "
                f"is after the on stream date: {self.gas_onstream_date}"
            )

        if sunk_cost_reference_year < self.start_date.year:
            raise SunkCostException(
                f"Sunk Cost Reference Year {sunk_cost_reference_year} "
                f"is before the project start date: {self.start_date}"
            )

        if sunk_cost_reference_year > self.end_date.year:
            raise SunkCostException(
                f"Sunk Cost Reference Year {sunk_cost_reference_year} "
                f"is after the project end date: {self.end_date}"
            )

        # Checking if the Cumulative Production Split Offset length is same with the project years
        if isinstance(cum_production_split_offset, np.ndarray):
            if len(cum_production_split_offset) != len(self.project_years):
                raise CumulativeProductionSplitException(
                    f"Length of the cum_production_split_offset: {len(cum_production_split_offset)} "
                    f"is not the same with Length of the project years: {len(self.project_years)}"
                )

        # Get the WAP Price
        self._get_wap_price()

        # Calculate expenditures for every cost components
        self._get_expenditures_pre_tax(
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
        )

        # Calculate indirect taxes
        self._get_indirect_taxes(tax_rate=vat_rate)

        # Calculate post tax expenditures
        self._get_expenditures_post_tax(
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            tax_rate=vat_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
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

        # Get The Other Revenue as the chosen selection
        self._get_other_revenue(sulfur_revenue=sulfur_revenue,
                                electricity_revenue=electricity_revenue,
                                co2_revenue=co2_revenue)

        # Depreciation (tangible cost)
        (
            self._oil_depreciation,
            self._oil_undepreciated_asset,
        ) = self._oil_capital_cost.total_depreciation_rate(
            depr_method=depr_method,
            decline_factor=decline_factor,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            tax_rate=vat_rate,
        )

        (
            self._gas_depreciation,
            self._gas_undepreciated_asset,
        ) = self._gas_capital_cost.total_depreciation_rate(
            depr_method=depr_method,
            decline_factor=decline_factor,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            tax_rate=vat_rate,
        )

        # Treatment for small order of number, in example 1e-15
        self._oil_undepreciated_asset = np.where(
            self._oil_undepreciated_asset < 1.0e-5, 0, self._oil_undepreciated_asset)
        self._gas_undepreciated_asset = np.where(
            self._gas_undepreciated_asset < 1.0e-5, 0, self._gas_undepreciated_asset)

        # Treatment of the un-depreciated asset to be summed up in the last year of the contract or not
        if sum_undepreciated_cost is True:
            self._oil_depreciation[-1] = self._oil_depreciation[-1] + self._oil_undepreciated_asset
            self._gas_depreciation[-1] = self._gas_depreciation[-1] + self._gas_undepreciated_asset

            self._oil_undepreciated_asset = np.zeros_like(self.project_years, dtype=float)
            self._gas_undepreciated_asset = np.zeros_like(self.project_years, dtype=float)
        else:
            pass

        # Non Capital Cost
        self._oil_non_capital = (
                self._oil_intangible_expenditures_post_tax
                + self._oil_opex_expenditures_post_tax
                + self._oil_asr_expenditures_post_tax
                + self._oil_lbt_expenditures_post_tax
        )

        self._gas_non_capital = (
                self._gas_intangible_expenditures_post_tax
                + self._gas_opex_expenditures_post_tax
                + self._gas_asr_expenditures_post_tax
                + self._gas_lbt_expenditures_post_tax
        )

        # Get Sunk Cost
        self._get_sunk_cost(sunk_cost_reference_year)

        # Amortization Cost
        if amortization is True:
            self._oil_amortization = unit_of_production_rate(
                start_year_project=self.start_date.year,
                cost=float(np.sum(self._oil_sunk_cost)),
                prod=self._oil_lifting.get_lifting_rate_arr(),
                prod_year=self.project_years,
                salvage_value=0.0,
                amortization_len=self.project_duration,)

            self._gas_amortization = unit_of_production_rate(
                start_year_project=self.start_date.year,
                cost=float(np.sum(self._gas_sunk_cost)),
                prod=self._gas_lifting.get_lifting_rate_arr(),
                prod_year=self.project_years,
                salvage_value=0.0,
                amortization_len=self.project_duration,)

            self._oil_sunk_cost = np.zeros_like(self.project_years)
            self._gas_sunk_cost = np.zeros_like(self.project_years)

        else:
            self._oil_amortization = np.zeros_like(self.project_years)
            self._gas_amortization = np.zeros_like(self.project_years)

        # Variable Split. -> Will set the value of _variable_split
        self._wrapper_variable_split(regime=regime)

        # Base Split
        self._oil_base_split = np.full_like(self.project_years, fill_value=self.base_split_ctr_oil, dtype=float)
        self._gas_base_split = np.full_like(self.project_years, fill_value=self.base_split_ctr_gas, dtype=float)

        # Variable Split
        self._var_split_array = np.full_like(self.project_years, fill_value=self._variable_split, dtype=float)

        # Calculating the gas production in MMBOE
        # Based on the discussion on 28 of June 2024, with YRA in Whatsapp,
        # The cum. production used for progressive cum. production split is the sum of production rate and baseline

        prod_gas_boe = self._gas_lifting.get_prod_rate_total_arr() / self.conversion_boe_to_scf

        # Condition when the offset cumulative production array when user input is float
        if isinstance(cum_production_split_offset, float) or isinstance(cum_production_split_offset, int):
            offset_arr = np.full_like(self.project_years, fill_value=0.0, dtype=float)
            offset_arr[0] = cum_production_split_offset

        else:
            offset_arr = np.full_like(self.project_years, fill_value=0.0, dtype=float)

        # Calculating the cumulative production
        self._cumulative_prod = np.cumsum(self._oil_lifting.get_prod_rate_total_arr() + prod_gas_boe + offset_arr)

        # Progressive Split
        vectorized_get_prog_split = np.vectorize(self._wrapper_progressive_split)

        # Condition when the cum_production_split_offset is filled with np.ndarray
        if isinstance(cum_production_split_offset, np.ndarray) and len(cum_production_split_offset) > 1:
            self._oil_prog_price_split, self._oil_prog_cum_split = vectorized_get_prog_split(
                fluid=self._oil_lifting.fluid_type,
                price=self._oil_lifting.get_price_arr(),
                cum=None,
                regime=regime
            )

            self._oil_prog_split = self._oil_prog_price_split + cum_production_split_offset

            self._gas_prog_price_split, self._gas_prog_cum_split = vectorized_get_prog_split(
                fluid=self._gas_lifting.fluid_type,
                price=self._gas_lifting.get_price_arr(),
                cum=None,
                regime=regime
            )

            self._gas_prog_split = self._gas_prog_price_split + cum_production_split_offset

        # Condition when the cum_production_split_offset is not filled
        else:
            self._oil_prog_price_split, self._oil_prog_cum_split = vectorized_get_prog_split(
                fluid=self._oil_lifting.fluid_type,
                price=self._oil_lifting.get_price_arr(),
                cum=self._cumulative_prod,
                regime=regime
            )
            self._oil_prog_split = self._oil_prog_price_split + self._oil_prog_cum_split

            self._gas_prog_price_split, self._gas_prog_cum_split = vectorized_get_prog_split(
                fluid=self._gas_lifting.fluid_type,
                price=self._gas_lifting.get_price_arr(),
                cum=self._cumulative_prod,
                regime=regime
            )
            self._gas_prog_split = self._gas_prog_price_split + self._gas_prog_cum_split

        # Ministerial Discretion
        minis_disc_array = np.full_like(self.project_years, fill_value=self.split_ministry_disc, dtype=float)

        # Total Contractor Split
        self._oil_ctr_split = (self._oil_base_split + self._var_split_array + self._oil_prog_split +
                               minis_disc_array)
        self._gas_ctr_split = (self._gas_base_split + self._var_split_array + self._gas_prog_split +
                               minis_disc_array)

        self._oil_ctr_split_prior_bracket = np.copy(self._oil_ctr_split)
        self._gas_ctr_split_prior_bracket = np.copy(self._gas_ctr_split)

        # Add the condition to show the contractor split more than 100%
        self._oil_year_maximum_ctr_split = self._get_year_maximum_split(
            ctr_split=self._oil_ctr_split,
            fluid="Oil"
        )

        self._gas_year_maximum_ctr_split = self._get_year_maximum_split(
            ctr_split=self._gas_ctr_split,
            fluid="Gas"
        )

        # Condition to limit the contractor split is 1.0 at maximum
        self._oil_ctr_split = np.where(
            self._oil_ctr_split > np.full_like(self._oil_ctr_split, 1.0),
            np.full_like(self._oil_ctr_split, 1.0),
            self._oil_ctr_split
        )

        self._gas_ctr_split = np.where(
            self._gas_ctr_split > np.full_like(self._gas_ctr_split, 1.0),
            np.full_like(self._gas_ctr_split, 1.0),
            self._gas_ctr_split
        )

        # Contractor Share
        self._oil_ctr_share_before_transfer = self._oil_revenue * self._oil_ctr_split
        self._gas_ctr_share_before_transfer = self._gas_revenue * self._gas_ctr_split

        # Government Share
        self._oil_gov_share = self._oil_revenue - self._oil_ctr_share_before_transfer
        self._gas_gov_share = self._gas_revenue - self._gas_ctr_share_before_transfer

        # Total Investment
        self._oil_total_expenses = (
                self._oil_capital_expenditures_post_tax +
                self._oil_intangible_expenditures_post_tax +
                self._oil_opex_expenditures_post_tax +
                self._oil_asr_expenditures_post_tax +
                self._oil_lbt_expenditures_post_tax
        )

        self._gas_total_expenses = (
                self._gas_capital_expenditures_post_tax +
                self._gas_intangible_expenditures_post_tax +
                self._gas_opex_expenditures_post_tax +
                self._gas_asr_expenditures_post_tax +
                self._gas_lbt_expenditures_post_tax
        )

        # Cost to be Deducted
        self._oil_cost_tobe_deducted = (
                self._oil_depreciation +
                self._oil_intangible_expenditures_post_tax +
                self._oil_opex_expenditures_post_tax +
                self._oil_asr_expenditures_post_tax +
                self._oil_lbt_expenditures_post_tax
        )

        self._gas_cost_tobe_deducted = (
                self._gas_depreciation +
                self._gas_intangible_expenditures_post_tax +
                self._gas_opex_expenditures_post_tax +
                self._gas_asr_expenditures_post_tax +
                self._gas_lbt_expenditures_post_tax
        )

        # Carry Forward Deductible Cost (In PSC Cost Recovery called Unrecovered Cost)
        self._oil_carward_deduct_cost = psc_tools.get_unrecovered_cost(depreciation=self._oil_depreciation,
                                                                       non_capital=self._oil_non_capital,
                                                                       revenue=self._oil_ctr_share_before_transfer,
                                                                       ftp_ctr=np.zeros_like(self.project_years),
                                                                       ftp_gov=np.zeros_like(self.project_years),
                                                                       ic=np.zeros_like(self.project_years))

        self._gas_carward_deduct_cost = psc_tools.get_unrecovered_cost(depreciation=self._gas_depreciation,
                                                                       non_capital=self._gas_non_capital,
                                                                       revenue=self._gas_ctr_share_before_transfer,
                                                                       ftp_ctr=np.zeros_like(self.project_years),
                                                                       ftp_gov=np.zeros_like(self.project_years),
                                                                       ic=np.zeros_like(self.project_years))

        # Deductible Cost (In PSC Cost Recovery called Cost Recovery)
        self._oil_deductible_cost = self._get_deductible_cost(ctr_gross_share=self._oil_ctr_share_before_transfer,
                                                              cost_tobe_deducted=self._oil_cost_tobe_deducted,
                                                              carward_deduct_cost=self._oil_carward_deduct_cost)

        self._gas_deductible_cost = self._get_deductible_cost(ctr_gross_share=self._gas_ctr_share_before_transfer,
                                                              cost_tobe_deducted=self._gas_cost_tobe_deducted,
                                                              carward_deduct_cost=self._gas_carward_deduct_cost)

        # Transfer
        self._transfer_to_oil, self._transfer_to_gas = psc_tools.get_transfer(
            oil_unrecovered=self._oil_carward_deduct_cost,
            gas_unrecovered=self._gas_carward_deduct_cost,
            oil_ets_pretransfer=self._oil_ctr_share_before_transfer,
            gas_ets_pretransfer=self._gas_ctr_share_before_transfer)

        # Carry Forward Deductible Cost After Transfer
        self._oil_carward_cost_aftertf = self._oil_carward_deduct_cost - self._transfer_to_gas
        self._gas_carward_cost_aftertf = self._gas_carward_deduct_cost - self._transfer_to_oil

        # Contractor Share After Transfer
        self._oil_ctr_share_after_transfer = (self._oil_ctr_share_before_transfer +
                                              self._transfer_to_oil -
                                              self._transfer_to_gas)

        self._gas_ctr_share_after_transfer = (self._gas_ctr_share_before_transfer +
                                              self._transfer_to_gas -
                                              self._transfer_to_oil)

        # Contractor Net Operating Profit
        self._oil_net_operating_profit = self._oil_ctr_share_after_transfer - self._oil_deductible_cost
        self._gas_net_operating_profit = self._gas_ctr_share_after_transfer - self._gas_deductible_cost

        # DMO
        self._oil_dmo_volume, self._oil_dmo_fee, self._oil_ddmo = psc_tools.get_dmo_gross_split(
            onstream_date=self.oil_onstream_date,
            start_date=self.start_date,
            project_years=self.project_years,
            dmo_holiday_duration=self.oil_dmo_holiday_duration,
            dmo_volume_portion=self.oil_dmo_volume_portion,
            dmo_fee_portion=self.oil_dmo_fee_portion,
            price=self._oil_wap_price,
            unrecovered_cost=self._oil_carward_cost_aftertf,
            is_dmo_end_weighted=is_dmo_end_weighted,
            net_operating_profit=self._oil_net_operating_profit,
            contractor_share=self._oil_ctr_share_after_transfer,)

        self._gas_dmo_volume, self._gas_dmo_fee, self._gas_ddmo = psc_tools.get_dmo_gross_split(
            onstream_date=self.gas_onstream_date,
            start_date=self.start_date,
            project_years=self.project_years,
            dmo_holiday_duration=self.gas_dmo_holiday_duration,
            dmo_volume_portion=self.gas_dmo_volume_portion,
            dmo_fee_portion=self.gas_dmo_fee_portion,
            price=self._gas_wap_price,
            unrecovered_cost=self._gas_carward_cost_aftertf,
            is_dmo_end_weighted=is_dmo_end_weighted,
            net_operating_profit=self._gas_net_operating_profit,
            contractor_share=self._gas_ctr_share_after_transfer,)

        # Taxable Income
        self._oil_taxable_income = self._oil_net_operating_profit - self._oil_ddmo
        self._gas_taxable_income = self._gas_net_operating_profit - self._gas_ddmo

        # Tax Payment
        # Generating Tax array if tax_rate argument is a single value not array
        if isinstance(effective_tax_rate, float) or isinstance(effective_tax_rate, int):
            self._tax_rate_arr = np.full_like(self.project_years, effective_tax_rate, dtype=float)

        # Generating Tax array based on the tax regime if tax_rate argument is None
        if effective_tax_rate is None:
            self._tax_rate_arr = self._get_tax_by_regime(tax_regime=tax_regime)

        elif isinstance(effective_tax_rate, np.ndarray):
            self._tax_rate_arr = effective_tax_rate

        self._oil_tax = self._oil_taxable_income * self._tax_rate_arr
        self._gas_tax = self._gas_taxable_income * self._tax_rate_arr

        # Contractor Net Share
        self._oil_ctr_net_share = self._oil_taxable_income - self._oil_tax
        self._gas_ctr_net_share = self._gas_taxable_income - self._gas_tax

        # Contractor Cash Flow
        self._oil_ctr_cashflow = (self._oil_ctr_share_before_transfer - self._oil_total_expenses - self._oil_ddmo -
                                  self._oil_tax)
        self._gas_ctr_cashflow = (self._gas_ctr_share_before_transfer - self._gas_total_expenses - self._gas_ddmo -
                                  self._gas_tax)

        # Government Take
        self._oil_government_take = self._oil_gov_share + self._oil_ddmo + self._oil_tax
        self._gas_government_take = self._gas_gov_share + self._gas_ddmo + self._gas_tax

        # Consolidated attributes
        self._consolidated_capital_expenditures_pre_tax = self._oil_capital_expenditures_pre_tax + self._gas_capital_expenditures_pre_tax
        self._consolidated_intangible_expenditures_pre_tax = self._oil_intangible_expenditures_pre_tax + self._gas_intangible_expenditures_pre_tax
        self._consolidated_opex_expenditures_pre_tax = self._oil_opex_expenditures_pre_tax + self._gas_opex_expenditures_pre_tax
        self._consolidated_asr_expenditures_pre_tax = self._oil_asr_expenditures_pre_tax + self._gas_asr_expenditures_pre_tax
        self._consolidated_lbt_expenditures_pre_tax = self._oil_lbt_expenditures_pre_tax + self._gas_lbt_expenditures_pre_tax
        self._consolidated_cost_of_sales_expenditures_pre_tax = self._oil_cost_of_sales_expenditures_pre_tax + self._gas_cost_of_sales_expenditures_pre_tax
        self._consolidated_expenditures_pre_tax = (
                self._consolidated_capital_expenditures_pre_tax +
                self._consolidated_intangible_expenditures_pre_tax +
                self._consolidated_opex_expenditures_pre_tax +
                self._consolidated_asr_expenditures_pre_tax +
                self._consolidated_lbt_expenditures_pre_tax +
                self._consolidated_cost_of_sales_expenditures_pre_tax
        )

        self._consolidated_capital_indirect_tax = self._oil_capital_indirect_tax + self._gas_capital_indirect_tax
        self._consolidated_intangible_indirect_tax = self._oil_intangible_indirect_tax + self._gas_intangible_indirect_tax
        self._consolidated_opex_indirect_tax = self._oil_opex_indirect_tax + self._gas_opex_indirect_tax
        self._consolidated_asr_indirect_tax = self._oil_asr_indirect_tax + self._gas_asr_indirect_tax
        self._consolidated_lbt_indirect_tax = self._oil_lbt_indirect_tax + self._gas_lbt_indirect_tax
        self._consolidated_cost_of_sales_indirect_tax = self._oil_cost_of_sales_indirect_tax + self._gas_cost_of_sales_indirect_tax

        self._consolidated_amortization = self._oil_amortization + self._gas_amortization

        self._consolidated_revenue = self._oil_revenue + self._gas_revenue
        self._consolidated_capital_cost = self._oil_capital_expenditures_post_tax + self._gas_capital_expenditures_post_tax
        self._consolidated_intangible = self._oil_intangible_expenditures_post_tax + self._gas_intangible_expenditures_post_tax
        self._consolidated_sunk_cost = self._oil_sunk_cost + self._gas_sunk_cost
        self._consolidated_opex = self._oil_opex_expenditures_post_tax + self._gas_opex_expenditures_post_tax
        self._consolidated_asr = self._oil_asr_expenditures_post_tax + self._gas_asr_expenditures_post_tax
        self._consolidated_lbt = self._oil_lbt_expenditures_post_tax + self._gas_lbt_expenditures_post_tax
        self._consolidated_non_capital = self._oil_non_capital + self._gas_non_capital
        self._consolidated_depreciation = self._oil_depreciation + self._gas_depreciation
        self._consolidated_undepreciated_asset = self._oil_undepreciated_asset + self._gas_undepreciated_asset
        self._consolidated_indirect_tax = self._oil_total_indirect_tax + self._gas_total_indirect_tax
        self._consolidated_ctr_share_before_tf = self._oil_ctr_share_before_transfer + self._gas_ctr_share_before_transfer
        self._consolidated_gov_share_before_tf = self._oil_gov_share + self._gas_gov_share
        self._consolidated_total_expenses = self._oil_total_expenses + self._gas_total_expenses
        self._consolidated_cost_tobe_deducted = self._oil_cost_tobe_deducted + self._gas_cost_tobe_deducted
        self._consolidated_carward_deduct_cost = self._oil_carward_deduct_cost + self._gas_carward_deduct_cost
        self._consolidated_deductible_cost = self._oil_deductible_cost + self._gas_deductible_cost
        self._consolidated_carward_cost_aftertf = self._oil_carward_cost_aftertf + self._gas_carward_cost_aftertf
        self._consolidated_ctr_share_after_transfer = self._oil_ctr_share_after_transfer + self._gas_ctr_share_after_transfer
        self._consolidated_net_operating_profit = self._oil_net_operating_profit + self._gas_net_operating_profit
        self._consolidated_dmo_volume = self._oil_dmo_volume + self._gas_dmo_volume
        self._consolidated_dmo_fee = self._oil_dmo_fee + self._gas_dmo_fee
        self._consolidated_ddmo = self._oil_ddmo + self._gas_ddmo
        self._consolidated_taxable_income = self._oil_taxable_income + self._gas_taxable_income
        self._consolidated_tax_payment = self._oil_tax + self._gas_tax
        self._consolidated_ctr_net_share = self._oil_ctr_net_share + self._gas_ctr_net_share
        self._consolidated_government_take = self._oil_government_take + self._gas_government_take
        self._consolidated_cashflow = self._oil_ctr_cashflow + self._gas_ctr_cashflow