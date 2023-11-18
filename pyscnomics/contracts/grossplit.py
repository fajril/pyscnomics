"""
Handles calculations associated with PSC Gross Split.
"""
from dataclasses import dataclass, field
import numpy as np

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts import psc_tools
from pyscnomics.econ.selection import FluidType, GrossSplitRegime, TaxRegime, TaxType, DeprMethod, OtherRevenue
from pyscnomics.econ.results import CashFlow


class GrossSplitException(Exception):
    """Exception to raise for a misuse of GrossSplit class"""

    pass


class SunkCostException(Exception):
    """Exception to raise for a misuse of Sunk Cost Method"""

    pass


@dataclass
class GrossSplit(BaseProject):
    field_status: str = field(default='No POD')
    field_loc: str = field(default='Onshore')
    res_depth: str = field(default='<=2500')
    infra_avail: str = field(default='Well Developed')
    res_type: str = field(default='Conventional')
    api_oil: str = field(default='<25')
    domestic_use: str = field(default='50<=x<70')
    prod_stage: str = field(default='Secondary')
    co2_content: str = field(default='<5')
    h2s_content: str = field(default='<100')

    base_split_ctr_oil: float = field(default=0.43)
    base_split_ctr_gas: float = field(default=0.48)
    split_ministry_disc: float = field(default=0.08)

    oil_dmo_volume_portion: float = field(default=0.25)
    oil_dmo_fee_portion: float = field(default=1.0)
    oil_dmo_holiday_duration: int = field(default=60)

    gas_dmo_volume_portion: float = field(default=1.0)
    gas_dmo_fee_portion: float = field(default=1.0)
    gas_dmo_holiday_duration: int = field(default=60)

    conversion_bboe2bscf: float = field(default=5.6)

    _oil_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _gas_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _oil_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)
    _gas_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)
    _oil_non_capital: np.ndarray = field(default=None, init=False, repr=False)
    _gas_non_capital: np.ndarray = field(default=None, init=False, repr=False)

    _cumulative_prod: np.ndarray = field(default=None, init=False, repr=False)

    _oil_base_split: np.ndarray = field(default=None, init=False, repr=False)
    _gas_base_split: np.ndarray = field(default=None, init=False, repr=False)
    _variable_split: float = field(default=0, init=False, repr=False)
    _var_split_array: np.ndarray = field(default=None, init=False, repr=False)
    _oil_prog_split: np.ndarray = field(default=None, init=False, repr=False)
    _gas_prog_split: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ctr_split: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_split: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ctr_share_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_share_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _oil_gov_share: np.ndarray = field(default=None, init=False, repr=False)
    _gas_gov_share: np.ndarray = field(default=None, init=False, repr=False)

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

    _oil_cashflow: CashFlow = field(default=None, init=False, repr=False)
    _gas_cashflow: CashFlow = field(default=None, init=False, repr=False)

    # Consolidated Attributes
    _consolidated_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_tangible: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_intangible: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_opex: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_asr: np.ndarray = field(default=None, init=False, repr=False)
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
    _consolidated_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_government_take: np.ndarray = field(default=None, init=False, repr=False)

    def _wrapper_variable_split(self,
                                regime: GrossSplitRegime = GrossSplitRegime.PERMEN_ESDM_20_2019):

        if regime == GrossSplitRegime.PERMEN_ESDM_8_2017:
            variable_split_func = self._get_var_split_08_2017()

        elif (regime == GrossSplitRegime.PERMEN_ESDM_52_2017 or
              GrossSplitRegime.PERMEN_ESDM_20_2019 or
              GrossSplitRegime.PERMEN_ESDM_12_2020):
            variable_split_func = self._get_var_split_52_2017()

        else:
            variable_split_func = ValueError('Not Recognized Gross Split Regime')

        return variable_split_func

    def _get_var_split_08_2017(self):
        """
        A function to get the value of Variable Split based on the given parameters.

        Returns
        -------
        _variable_split: float
            The value of variable split.
        """
        params = {
            'field_status': {
                'POD I': 0.05,
                'POD II': 0,
                'POFD': 0,
                'No POD': -0.05,
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
                'New Frontier': 0.02,
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
                '100<=x<300': 0.005,
                '300<=x<500': 0.0075,
                'x>=500': 0.01,
            },
            'api_oil': {
                '<25': 0.01,
                '>=25': 0,
            },
            'domestic_use': {
                '<30': 0,
                '30<=x<50': 0.02,
                '50<=x<70': 0.03,
                '70<=x<100': 0.04,
            },
            'prod_stage': {
                'Primer': 0,
                'Secondary': 0.03,
                'Tertiary': 0.05,
            }
        }

        self._variable_split = sum(params[param][getattr(self, param)] for param in params)

    def _get_var_split_52_2017(self):
        """
        A function to get the value of Variable Split based on the given parameters.

        Returns
        -------
        _variable_split: float
            The value of variable split.
        """
        params = {
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
                'Primer': 0,
                'Secondary': 0.06,
                'Tertiary': 0.1,
            }
        }

        self._variable_split = sum(params[param][getattr(self, param)] for param in params)

    def _wrapper_progressive_split(self,
                                   fluid: FluidType,
                                   price: float,
                                   cum: float,
                                   regime: GrossSplitRegime = GrossSplitRegime.PERMEN_ESDM_20_2019):

        if regime == GrossSplitRegime.PERMEN_ESDM_20_2019:
            prog_split = self._get_prog_split_52_2017(fluid, price, cum)

        elif regime == GrossSplitRegime.PERMEN_ESDM_8_2017:
            prog_split = self._get_prog_split_08_2017(fluid, price, cum)

        else:
            prog_split = ValueError('Not Recognized Gross Split Regime')

        return prog_split

    @staticmethod
    def _get_prog_split_08_2017(fluid: FluidType, price: float, cum: float):
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

        # Cumulative Progressive Split
        if 0 < cum < 1:
            px = 0.05
        elif 1 <= cum < 10:
            px = 0.04
        elif 10 <= cum < 20:
            px = 0.03
        elif 20 <= cum < 50:
            px = 0.02
        elif 50 <= cum < 150:
            px = 0.01
        elif 150 <= cum:
            px = 0
        else:
            raise ValueError('No Regulation exist regarding the cumulative value')

        ps = ps + px
        return ps

    @staticmethod
    def _get_prog_split_52_2017(fluid: FluidType, price: float, cum: float):
        # Indonesia's Ministry Regulations No.52 The Year of 2017. At Appendix B Progressive Component
        if fluid == FluidType.OIL:
            ps = (85 - price) * 0.25 / 100

        elif fluid == FluidType.GAS:
            if price < 7:
                ps = (7 - price) * 2.5 / 100
            elif price > 10:
                ps = (10 - price) * 2.5 / 100
            else:
                ps = 0
        else:
            raise ValueError('Unknown fluid type')

        # Cumulative Progressive Split
        if np.less(cum, 30):
            px = 0.1
        elif np.logical_and(np.less_equal(30, cum), cum < np.less(cum, 60)):
            px = 0.09
        elif np.logical_and(np.less_equal(60, cum), cum < np.less(cum, 90)):
            px = 0.08
        elif np.logical_and(np.less_equal(90, cum), cum < np.less(cum, 125)):
            px = 0.06
        elif np.logical_and(np.less_equal(125, cum), cum < np.less(cum, 175)):
            px = 0.04
        elif np.greater(cum, 175):
            px = 0
        else:
            raise ValueError('No Regulation exist regarding the cumulative value')

        ps = ps + px
        return ps

    @staticmethod
    def _get_deductible_cost(ctr_gross_share, cost_tobe_deducted, carward_deduct_cost):

        carward_deduct_cost = np.concatenate((np.zeros(1), carward_deduct_cost[:-1]))

        return np.where(ctr_gross_share > (cost_tobe_deducted + carward_deduct_cost),
                        cost_tobe_deducted + carward_deduct_cost,
                        ctr_gross_share)

    def _get_sunk_cost(self, discount_rate_year: int):

        oil_cost_raw = self._oil_depreciation + self._oil_undepreciated_asset + self._oil_non_capital
        self._oil_sunk_cost = oil_cost_raw[:(discount_rate_year - self.start_date.year + 1)]

        gas_cost_raw = self._gas_depreciation + self._gas_undepreciated_asset + self._gas_non_capital
        self._gas_sunk_cost = gas_cost_raw[:(discount_rate_year - self.start_date.year + 1)]

        if discount_rate_year == self.start_date.year:
            self._oil_sunk_cost = np.zeros(1)
            self._gas_sunk_cost = np.zeros(1)

    def run(self,
            sulfur_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
            electricity_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_OIL_REVENUE,
            co2_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
            is_dmo_end_weighted=False,
            regime: GrossSplitRegime = GrossSplitRegime.PERMEN_ESDM_20_2019,
            tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
            tax_rate: float | np.ndarray = 0.22,
            discount_rate_year: int | None = None,
            depr_method: DeprMethod = DeprMethod.PSC_DB,
            decline_factor: float | int = 2,
            year_ref: int = None,
            tax_type: TaxType = TaxType.VAT,
            vat_rate: np.ndarray | float = 0.0,
            lbt_rate: np.ndarray | float = 0.0,
            inflation_rate: np.ndarray | float = 0.0,
            future_rate: float = 0.02,
            ):

        if discount_rate_year is None:
            discount_rate_year = self.start_date.year

        if discount_rate_year < self.start_date.year:
            raise SunkCostException(
                f"start_date year {self.start_date} "
                f"is after the discount rate year: {self.end_date}"
            )

        # Configure year reference
        if year_ref is None:
            year_ref = self.start_date.year

        # Get the WAP Price
        self._get_wap_price()

        # Calculate expenditures for every cost components
        self._get_expenditures(
            year_ref=year_ref,
            tax_type=tax_type,
            vat_rate=vat_rate,
            lbt_rate=lbt_rate,
            inflation_rate=inflation_rate,
            future_rate=future_rate,
        )

        # Get The Other Revenue as the chosen selection
        self._get_other_revenue(sulfur_revenue=sulfur_revenue,
                                electricity_revenue=electricity_revenue,
                                co2_revenue=co2_revenue)

        # Depreciation (tangible cost)
        (
            self._oil_depreciation,
            self._oil_undepreciated_asset,
        ) = self._oil_tangible.total_depreciation_rate(
            depr_method=depr_method,
            decline_factor=decline_factor,
            year_ref=year_ref,
            tax_type=tax_type,
            vat_rate=vat_rate,
            lbt_rate=lbt_rate,
            inflation_rate=inflation_rate,
        )

        (
            self._gas_depreciation,
            self._gas_undepreciated_asset,
        ) = self._gas_tangible.total_depreciation_rate(
            depr_method=depr_method,
            decline_factor=decline_factor,
            year_ref=year_ref,
            tax_type=tax_type,
            vat_rate=vat_rate,
            lbt_rate=lbt_rate,
            inflation_rate=inflation_rate,
        )

        # Non Capital Cost
        self._oil_non_capital = (
                self._oil_intangible_expenditures
                + self._oil_opex_expenditures
                + self._oil_asr_expenditures
        )

        self._gas_non_capital = (
                self._gas_intangible_expenditures
                + self._gas_opex_expenditures
                + self._gas_asr_expenditures
        )

        # Get Sunk Cost
        self._get_sunk_cost(discount_rate_year)

        # Variable Split. -> Will set the value of _variable_split
        self._wrapper_variable_split(regime=regime)

        # Base Split
        self._oil_base_split = np.full_like(self.project_years, fill_value=self.base_split_ctr_oil, dtype=float)
        self._gas_base_split = np.full_like(self.project_years, fill_value=self.base_split_ctr_gas, dtype=float)

        # Variable Split
        self._var_split_array = np.full_like(self.project_years, fill_value=self._variable_split, dtype=float)

        # Cumulative Production
        self._cumulative_prod = np.cumsum(self._oil_lifting.lifting_rate +
                                          (self._gas_lifting.lifting_rate / self.conversion_bboe2bscf))

        # Progressive Split
        vectorized_get_prog_split = np.vectorize(self._wrapper_progressive_split)

        self._oil_prog_split = vectorized_get_prog_split(
            fluid=self._oil_lifting.fluid_type,
            price=self._oil_lifting.price,
            cum=self._cumulative_prod,
            regime=regime
        )

        self._gas_prog_split = vectorized_get_prog_split(
            fluid=self._gas_lifting.fluid_type,
            price=self._gas_lifting.price,
            cum=self._cumulative_prod,
            regime=regime
        )

        # Ministerial Discretion
        minis_disc_array = np.full_like(self.project_years, fill_value=self.split_ministry_disc, dtype=float)

        # Total Contractor Split
        self._oil_ctr_split = (self._oil_base_split + self._var_split_array + self._oil_prog_split +
                               minis_disc_array)
        self._gas_ctr_split = (self._gas_base_split + self._var_split_array + self._gas_prog_split +
                               minis_disc_array)

        # Contractor Share
        self._oil_ctr_share_before_transfer = self._oil_revenue * self._oil_ctr_split
        self._gas_ctr_share_before_transfer = self._gas_revenue * self._gas_ctr_split

        # Government Share
        self._oil_gov_share = self._oil_revenue - self._oil_ctr_share_before_transfer
        self._gas_gov_share = self._gas_revenue - self._gas_ctr_share_before_transfer

        # Total Investment
        self._oil_total_expenses = (self._oil_tangible.expenditures() + self._oil_intangible.expenditures() +
                                    self._oil_opex.expenditures() + self._oil_asr.expenditures())
        self._gas_total_expenses = (self._gas_tangible.expenditures() + self._gas_intangible.expenditures() +
                                    self._gas_opex.expenditures() + self._gas_asr.expenditures())

        # Cost to be Deducted
        self._oil_cost_tobe_deducted = (self._oil_depreciation + self._oil_intangible.expenditures() +
                                        self._oil_opex.expenditures() + self._oil_asr.expenditures())
        self._gas_cost_tobe_deducted = (self._gas_depreciation + self._gas_intangible.expenditures() +
                                        self._gas_opex.expenditures() + self._gas_asr.expenditures())

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
        self._oil_dmo_volume, self._oil_dmo_fee, self._oil_ddmo = psc_tools.get_dmo(
            onstream_date=self.oil_onstream_date,
            start_date=self.start_date,
            project_years=self.project_years,
            dmo_holiday_duration=self.oil_dmo_holiday_duration,
            dmo_volume_portion=self.oil_dmo_volume_portion,
            dmo_fee_portion=self.oil_dmo_fee_portion,
            lifting=self._oil_lifting,
            price=self._oil_wap_price,
            ctr_pretax_share=1.0,
            unrecovered_cost=self._oil_carward_cost_aftertf,
            is_dmo_end_weighted=is_dmo_end_weighted)

        self._gas_dmo_volume, self._gas_dmo_fee, self._gas_ddmo = psc_tools.get_dmo(
            onstream_date=self.gas_onstream_date,
            start_date=self.start_date,
            project_years=self.project_years,
            dmo_holiday_duration=self.gas_dmo_holiday_duration,
            dmo_volume_portion=self.gas_dmo_volume_portion,
            dmo_fee_portion=self.gas_dmo_fee_portion,
            lifting=self._gas_lifting,
            price=self._gas_wap_price,
            ctr_pretax_share=1.0,
            unrecovered_cost=self._gas_carward_cost_aftertf,
            is_dmo_end_weighted=is_dmo_end_weighted)

        # Taxable Income
        self._oil_taxable_income = self._oil_net_operating_profit - self._oil_ddmo
        self._gas_taxable_income = self._gas_net_operating_profit - self._gas_ddmo

        # Tax Payment
        # Generating Tax array if tax_rate argument is a single value not array
        if isinstance(tax_rate, float):
            self._tax_rate_arr = np.full_like(self.project_years, tax_rate, dtype=float)

        # Generating Tax array based on the tax regime if tax_rate argument is None
        if tax_rate is None:
            self._tax_rate_arr = self._get_tax_by_regime(tax_regime=tax_regime)

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
        self._consolidated_revenue = self._oil_revenue + self._gas_revenue
        self._consolidated_tangible = self._oil_tangible.expenditures() + self._gas_tangible.expenditures()
        self._consolidated_intangible = self._oil_intangible.expenditures() + self._gas_intangible.expenditures()
        self._consolidated_sunk_cost = self._oil_sunk_cost + self._gas_sunk_cost
        self._consolidated_opex = self._oil_opex.expenditures() + self._gas_opex.expenditures()
        self._consolidated_asr = self._oil_asr.expenditures() + self._gas_asr.expenditures()
        self._consolidated_non_capital = self._oil_non_capital + self._gas_non_capital
        self._consolidated_depreciation = self._oil_depreciation + self._gas_depreciation
        self._consolidated_undepreciated_asset = self._oil_undepreciated_asset + self._gas_undepreciated_asset
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

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):
        # Between two instances of Gross Split
        if isinstance(other, GrossSplit):
            return all(
                (
                    np.allclose(self._oil_lifting.lifting_rate, other._oil_lifting.lifting_rate),
                    np.allclose(self._gas_lifting.lifting_rate, other._gas_lifting.lifting_rate),
                    np.allclose(self._oil_revenue, other._oil_revenue),
                    np.allclose(self._gas_revenue, other._gas_revenue),
                    np.allclose(self._oil_tangible.expenditures(), other._oil_tangible.expenditures()),
                    np.allclose(self._gas_tangible.expenditures(), other._gas_tangible.expenditures()),
                    np.allclose(self._oil_intangible.expenditures(), other._oil_intangible.expenditures()),
                    np.allclose(self._gas_intangible.expenditures(), other._gas_intangible.expenditures()),
                    np.allclose(self._oil_opex.expenditures(), other._oil_opex.expenditures()),
                    np.allclose(self._gas_opex.expenditures(), other._gas_opex.expenditures()),
                    np.allclose(self._oil_asr.expenditures(), other._oil_asr.expenditures()),
                    np.allclose(self._gas_asr.expenditures(), other._gas_asr.expenditures()),
                )
            )

        else:
            return False

    def __lt__(self, other):
        if isinstance(other, GrossSplit):

            self.run()
            other.run()

            self_total_base_cashflow = self._oil_base_cashflow + self._gas_base_cashflow
            other_total_base_cashflow = other._oil_base_cashflow + other._gas_base_cashflow

            return self_total_base_cashflow.irr() < other_total_base_cashflow.irr()

        else:
            raise GrossSplitException

    def __le__(self, other):
        if isinstance(other, GrossSplit):
            self.run()
            other.run()

            self_total_base_cashflow = self._oil_base_cashflow + self._gas_base_cashflow
            other_total_base_cashflow = other._oil_base_cashflow + other._gas_base_cashflow

            return self_total_base_cashflow.irr() <= other_total_base_cashflow.irr()

        else:
            raise GrossSplitException

    def __gt__(self, other):
        if isinstance(other, GrossSplit):

            self.run()
            other.run()

            self_total_base_cashflow = self._oil_base_cashflow + self._gas_base_cashflow
            other_total_base_cashflow = other._oil_base_cashflow + other._gas_base_cashflow

            return self_total_base_cashflow.irr() > other_total_base_cashflow.irr()

        else:
            raise GrossSplitException

    def __ge__(self, other):
        if isinstance(other, GrossSplit):

            self.run()
            other.run()

            self_total_base_cashflow = self._oil_base_cashflow + self._gas_base_cashflow
            other_total_base_cashflow = other._oil_base_cashflow + other._gas_base_cashflow

            return self_total_base_cashflow.irr() >= other_total_base_cashflow.irr()

        else:
            raise GrossSplitException
