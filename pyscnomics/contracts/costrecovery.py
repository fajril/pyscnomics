"""
Handles calculations associated with PSC Cost Recovery.
"""

from dataclasses import dataclass, field
import numpy as np
from functools import reduce

import pandas as pd

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts import psc_tools
from pyscnomics.econ.selection import (
    FluidType,
    InflationAppliedTo,
    TaxRegime,
    FTPTaxRegime,
    TaxSplitTypeCR,
    TaxPaymentMode,
    TaxType,
    DeprMethod,
    OtherRevenue
)


class SunkCostException(Exception):
    """Exception to raise for a misuse of Sunk Cost Method"""

    pass


@dataclass
class CostRecovery(BaseProject):
    oil_ftp_is_available: bool = field(default=True)
    oil_ftp_is_shared: bool = field(default=True)
    oil_ftp_portion: float = field(default=0.2)

    gas_ftp_is_available: bool = field(default=True)
    gas_ftp_is_shared: bool = field(default=True)
    gas_ftp_portion: float = field(default=0.2)

    tax_split_type: TaxSplitTypeCR = field(default=TaxSplitTypeCR.CONVENTIONAL)
    condition_dict: dict = field(default_factory=dict)
    indicator_rc_icp_sliding: list[float] = field(default_factory=list)
    oil_ctr_pretax_share: float | np.ndarray = field(default=0.25)
    gas_ctr_pretax_share: float | np.ndarray = field(default=0.5)

    oil_ic_rate: float = field(default=0.0)
    gas_ic_rate: float = field(default=0.0)
    ic_is_available: bool = field(default=False)
    oil_cr_cap_rate: float = field(default=1.0)
    gas_cr_cap_rate: float = field(default=1.0)

    oil_dmo_volume_portion: float = field(default=0.25)
    oil_dmo_fee_portion: float = field(default=0.25)
    oil_dmo_holiday_duration: int = field(default=60)

    gas_dmo_volume_portion: float = field(default=1.0)
    gas_dmo_fee_portion: float = field(default=1.0)
    gas_dmo_holiday_duration: int = field(default=60)

    _oil_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _gas_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _oil_undepreciated_asset: float = field(default=None, init=False, repr=False)
    _gas_undepreciated_asset: float = field(default=None, init=False, repr=False)

    _oil_non_capital: np.ndarray = field(default=None, init=False, repr=False)
    _gas_non_capital: np.ndarray = field(default=None, init=False, repr=False)

    _split_ctr_oil: np.ndarray = field(default=None, init=False, repr=False)
    _split_ctr_gas: np.ndarray = field(default=None, init=False, repr=False)

    _oil_ic: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ic_unrecovered: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ic_paid: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ic: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ic_unrecovered: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ic_paid: np.ndarray = field(default=None, init=False, repr=False)

    _oil_unrecovered_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_unrecovered_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )

    _oil_cost_to_be_recovered: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cost_to_be_recovered: np.ndarray = field(default=None, init=False, repr=False)

    _oil_cost_recovery: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cost_recovery: np.ndarray = field(default=None, init=False, repr=False)

    _oil_ets_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ets_before_transfer: np.ndarray = field(default=None, init=False, repr=False)

    _transfer_to_oil: np.ndarray = field(default=None, init=False, repr=False)
    _transfer_to_gas: np.ndarray = field(default=None, init=False, repr=False)

    _oil_unrecovered_after_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_unrecovered_after_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )

    _oil_cost_to_be_recovered_after_tf: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_cost_to_be_recovered_after_tf: np.ndarray = field(
        default=None, init=False, repr=False
    )

    _oil_cost_recovery_after_tf: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_cost_recovery_after_tf: np.ndarray = field(
        default=None, init=False, repr=False
    )

    _oil_ets_after_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ets_after_transfer: np.ndarray = field(default=None, init=False, repr=False)

    _oil_contractor_share: np.ndarray = field(default=None, init=False, repr=False)
    _oil_government_share: np.ndarray = field(default=None, init=False, repr=False)
    _gas_contractor_share: np.ndarray = field(default=None, init=False, repr=False)
    _gas_government_share: np.ndarray = field(default=None, init=False, repr=False)

    _oil_dmo_volume: np.ndarray = field(default=None, init=False, repr=False)
    _oil_dmo_fee: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ddmo: np.ndarray = field(default=None, init=False, repr=False)
    _gas_dmo_volume: np.ndarray = field(default=None, init=False, repr=False)
    _gas_dmo_fee: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ddmo: np.ndarray = field(default=None, init=False, repr=False)

    _oil_taxable_income: np.ndarray = field(default=None, init=False, repr=False)
    _gas_taxable_income: np.ndarray = field(default=None, init=False, repr=False)

    _tax_rate_arr: np.ndarray = field(default=None, init=False, repr=False)

    _oil_tax_payment: np.ndarray = field(default=None, init=False, repr=False)
    _gas_tax_payment: np.ndarray = field(default=None, init=False, repr=False)

    _oil_ctr_net_share: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_net_share: np.ndarray = field(default=None, init=False, repr=False)

    _oil_contractor_take: np.ndarray = field(default=None, init=False, repr=False)
    _gas_contractor_take: np.ndarray = field(default=None, init=False, repr=False)
    _oil_government_take: np.ndarray = field(default=None, init=False, repr=False)
    _gas_government_take: np.ndarray = field(default=None, init=False, repr=False)

    _oil_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cashflow: np.ndarray = field(default=None, init=False, repr=False)

    # Consolidated Attributes
    _consolidated_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_tangible: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_intangible: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_opex: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_asr: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_non_capital: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_undepreciated_asset: np.ndarray | float = field(
        default=None, init=False, repr=False
    )
    _consolidated_ftp: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ftp_ctr: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ftp_gov: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ic: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ic_unrecovered: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_ic_paid: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_unrecovered_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_recovery_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_ets_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_unrecovered_after_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_to_be_recovered_after_tf: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_recovery_after_tf: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_ets_after_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_contractor_share: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_government_share: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_dmo_volume: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_dmo_fee: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ddmo: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_taxable_income: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_tax_due: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_unpaid_tax_balance: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_tax_payment: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ctr_net_share: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_contractor_take: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_government_take: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_cashflow: np.ndarray = field(default=None, init=False, repr=False)

    def _get_rc_icp_pretax(self):
        # Extract relevant values from the condition_dict dictionary
        bot_limits = np.array(
            [self.condition_dict[c]["bot_limit"] for c in self.condition_dict]
        )
        top_limits = np.array(
            [self.condition_dict[c]["top_limit"] for c in self.condition_dict]
        )
        ctr_oil_values = np.array(
            [self.condition_dict[c]["ctr_oil"] for c in self.condition_dict]
        )
        ctr_gas_values = np.array(
            [self.condition_dict[c]["ctr_gas"] for c in self.condition_dict]
        )

        # Create conditions using vectorized comparisons
        conditions = (bot_limits < self.indicator_rc_icp_sliding) & (
                self.indicator_rc_icp_sliding <= top_limits
        )

        # Calculate corresponding values using np.select()
        self.oil_ctr_pretax_share = np.select(
            conditions, ctr_oil_values, default=np.nan
        )
        self.gas_ctr_pretax_share = np.select(
            conditions, ctr_gas_values, default=np.nan
        )

    def _get_ftp(self):
        self._oil_ftp_ctr = np.zeros_like(self._oil_revenue)
        self._gas_ftp_ctr = np.zeros_like(self._gas_revenue)
        self._oil_ftp_gov = np.zeros_like(self._oil_revenue)
        self._gas_ftp_gov = np.zeros_like(self._gas_revenue)

        if self.oil_ftp_is_available:
            self._oil_ftp = self.oil_ftp_portion * self._oil_revenue
            if self.oil_ftp_is_shared:
                self._oil_ftp_ctr = self.oil_ctr_pretax_share * self._oil_ftp
            self._oil_ftp_gov = self._oil_ftp - self._oil_ftp_ctr

        if self.gas_ftp_is_available:
            self._gas_ftp = self.gas_ftp_portion * self._gas_revenue
            if self.gas_ftp_is_shared:
                self._gas_ftp_ctr = self.gas_ctr_pretax_share * self._gas_ftp
            self._gas_ftp_gov = self._gas_ftp - self._gas_ftp_ctr

    def _get_ic(
            self,
            revenue: np.ndarray,
            ftp: np.ndarray,
            cost_alloc: FluidType,
            ic_rate: float,
    ):

        # Condition where fluid is Oil:
        if cost_alloc == FluidType.GAS:
            tangible_class = self._gas_tangible
        else:
            tangible_class = self._oil_tangible

        # Applying the IC calculation to only true value
        ic_arr = np.where(np.asarray(tangible_class.is_ic_applied) == True,
                          tangible_class.cost * ic_rate,
                          0)

        ic_expenses = np.bincount(
            tangible_class.expense_year - tangible_class.start_year, weights=ic_arr
        )
        zeros = np.zeros(self.project_duration - len(ic_expenses))
        ic_total = np.concatenate((ic_expenses, zeros))

        # Unrec IC
        ic_unrecovered = np.cumsum(ic_total) - np.cumsum(revenue - ftp)
        ic_unrecovered = np.where(ic_unrecovered > 0, ic_unrecovered, 0)

        ic_unrec_next = np.roll(ic_unrecovered, 1)
        ic_unrec_next[0] = 0
        ic_paid = np.minimum(revenue - ftp, ic_total + ic_unrec_next)

        return ic_total, ic_unrecovered, ic_paid

    @staticmethod
    def _get_cost_recovery(
            revenue: np.ndarray,
            ftp: np.ndarray,
            ic: np.ndarray,
            depreciation: np.ndarray,
            non_capital: np.ndarray,
            cost_to_be_recovered: np.ndarray,
            cr_cap_rate: float,
    ) -> np.ndarray:
        """
        A function to get the array of cost recovery.

        Parameters
        ----------
        revenue: np.ndarray
            The array containing the revenue.
        ftp: np.ndarray
            The array containing the First Tranche Petroleum (FTP).
        ic: np.ndarray
            The array containing the Paid Investment Credit (IC).
        depreciation: np.ndarray
            The array containing the depreciated expenditures.
        non_capital: np.ndarray
            The array containing the non-capital expenditures.
            non_capital expenditures is consisting of:
            Intangible, Operating Expenditures (OPEX) and Abandonment Site and Restoration (ASR) Expenditures.
        cost_to_be_recovered: np.ndarray
            The array containing the cost to be recovered.
        cr_cap_rate: float
            The cost recovery cap rate.

        Returns
        -------
        out: np.ndarray
            The array of cost recovery.
        """

        return np.minimum(
            revenue - ftp - ic,
            ((depreciation + non_capital + cost_to_be_recovered) * cr_cap_rate),
        )
        # There is possibility for a bug in Cap rate
        # return (depreciation + non_capital + cost_to_be_recovered) * cr_cap_rate

    @staticmethod
    def _get_ets_before_transfer(
            revenue: np.ndarray,
            ftp_ctr: np.ndarray,
            ftp_gov: np.ndarray,
            ic: np.ndarray,
            cost_recovery: np.ndarray,
    ) -> np.ndarray:
        """
        A function to get the Equity To be Shared (ETS) before transfer.

        Parameters
        ----------
        revenue: np.ndarray
            The array containing the revenue.
        ftp_ctr: np.ndarray
            The array containing the Contractor's First Tranche Petroleum (FTP).
        ftp_gov: np.ndarray
            The array containing the Government's FTP.
        ic: np.ndarray
            The array containing the Paid Investment Credit (IC).
        cost_recovery
            The array containing the cost recovery.

        Returns
        -------
        out: np.ndarray
            The array of ETS before transfer.
        """

        result = revenue - (ftp_ctr + ftp_gov) - ic - cost_recovery
        tol = np.full_like(result, fill_value=1.0e-12)
        return np.where(result < tol, 0, result)

    @staticmethod
    def _get_ets_after_transfer(
            ets_before_tf: np.ndarray,
            revenue: np.ndarray,
            ftp_ctr: np.ndarray,
            ftp_gov: np.ndarray,
            ic: np.ndarray,
            cost_recovery: np.ndarray,
            transferred_in: np.ndarray,
            transferred_out: np.ndarray,
    ) -> np.ndarray:
        """
        A function to get the Equity To be Shared (ETS) before transfer.

        Parameters
        ----------
        revenue: np.ndarray
            The array containing the revenue.
        ftp_ctr: np.ndarray
            The array containing the Contractor's First Tranche Petroleum (FTP).
        ftp_gov: np.ndarray
            The array containing the Government's FTP.
        ic: np.ndarray
            The array containing the Paid Investment Credit (IC).
        cost_recovery: np.ndarray
            The array containing the cost recovery.
        transferred_in: np.ndarray
            The transferred cost into the cashflow.
        transferred_out: np.ndarray
            The transferred cost out from the cashflow.


        Returns
        -------
        out: np.ndarray
            The array of ETS before transfer.
        """
        ets_after_transfer = np.where(ets_before_tf > 0,
                                      revenue
                                      - (ftp_ctr + ftp_gov)
                                      - ic
                                      - cost_recovery
                                      - transferred_out
                                      + transferred_in
                                      ,
                                      0)

        tol = np.full_like(ets_after_transfer, fill_value=1.0e-14)
        return np.where(ets_after_transfer < tol, 0, ets_after_transfer)

    @staticmethod
    def _get_equity_share(ets, pretax_ctr):
        r"""
        Use to calculate equity share (ES).

        .. math::

            ES_{ctr \, (t)} &= pretax_ctr \times ETS_{(t)} \\
            ES_{gov \, (t)} &= (1 - pretax_ctr) \times ETS_{(t)}

        Parameters
        ----------
        ets : np.ndarray
            equity to be shared
        pretax_ctr: float
            pretax contractor

        Returns
        -------
        out: tup
            * ES_ctr : np.ndarray
                Equity share of contractor
            * ES_gov : np.ndarray
                Equity share of government
        """
        contractor_share = pretax_ctr * ets
        government_share = (1 - pretax_ctr) * ets

        return contractor_share, government_share

    @staticmethod
    # def _get_ftp_tax_payment(
    #         unrec: np.ndarray,
    #         ftp: np.ndarray,
    #         tax_rate: float,
    #         ftp_tax_regime: FTPTaxRegime = FTPTaxRegime.PDJP_20_2017,
    # ):
    #     """
    #     A funtion to get the array of tax payment of the First Tranche Petroleum (FTP).
    #
    #     Parameters
    #     ----------
    #     unrec: np.ndarray
    #         The array containing the Unrecovered Cost.
    #     ftp: np.ndarray
    #         The array containing the contractor's FTP.
    #     tax_rate: float
    #         The tax rate value.
    #     ftp_tax_regime: TaxSplitTypeCR
    #         The regime configuration used.
    #
    #     Returns
    #     -------
    #     ftp_tax_payment: np.ndarray
    #         The array of Tax Payment of the FTP.
    #     """
    #     if ftp_tax_regime == FTPTaxRegime.PDJP_20_2017:
    #         cum_ftp = np.cumsum(ftp)
    #         cum_ftp = np.where(cum_ftp > unrec, cum_ftp, 0)
    #         tax_of_ftp = np.concatenate((np.zeros(1), np.diff(cum_ftp * tax_rate)))
    #         ftp_tax_payment = np.where(tax_of_ftp > 0, tax_of_ftp, 0)
    #
    #     else:
    #         ftp_tax_payment = ftp * tax_rate
    #
    #     return ftp_tax_payment

    @staticmethod
    def _get_tax_payment(
            ctr_share: np.ndarray,
            taxable_income: np.ndarray,
            tax_rate: np.ndarray,
            ftp_ctr: np.ndarray,
            unrec_cost: np.ndarray,
            ftp_tax_regime: FTPTaxRegime = FTPTaxRegime.PRE_2017,
    ):

        ftp_considered_b5 = ftp_ctr

        # Tax payment prior to PDJP 2017
        if ftp_tax_regime == FTPTaxRegime.PRE_2017:
            applied_tax = np.where(ctr_share > 0, 1, 0)
            cum_ti = np.cumsum(taxable_income)

            applied_tax_prior = np.concatenate((np.zeros(1), applied_tax))[0:-1]

            ctr_tax = np.where(
                np.logical_and(applied_tax == 1, applied_tax_prior == 0),
                cum_ti * tax_rate,
                np.where(
                    np.logical_and(applied_tax == 1, applied_tax_prior == 1),
                    taxable_income * tax_rate,
                    0,
                ),
            )

        # Tax payment after PDJP 2017
        elif ftp_tax_regime == FTPTaxRegime.PDJP_20_2017:
            # Defining the array used in calculation of ftp tax payment
            ftp_cum_b2 = np.cumsum(ftp_ctr)
            ftp_prior_b3 = np.zeros_like(ftp_ctr, dtype=float)
            ftp_diff_b4 = np.zeros_like(ftp_ctr, dtype=float)
            ftp_considered_b5 = np.zeros_like(ftp_ctr, dtype=float)

            # Looping throughout the ftp array
            for i, value in enumerate(ftp_ctr):
                if i == 0:
                    ftp_prior_b3[i] = 0

                ftp_prior_b3[i] = ftp_prior_b3[i - 1] + ftp_considered_b5[i - 1]

                ftp_diff_b4[i] = np.where(ftp_cum_b2[i] > unrec_cost[i],
                                          ftp_cum_b2[i] - ftp_prior_b3[i],
                                          0)
                ftp_considered_b5[i] = np.where(ftp_diff_b4[i] > unrec_cost[i],
                                                ftp_diff_b4[i] - unrec_cost[i],
                                                0)

            # Calculating the ftp tax payment
            ftp_tax_paid = ftp_considered_b5 * tax_rate

            # Calculating the tax from taxable income
            ti_tax = (taxable_income - ftp_ctr) * tax_rate

            # Calculating ctr_tax
            ctr_tax = ftp_tax_paid + ti_tax

        else:
            ctr_tax = taxable_income * tax_rate

        return ctr_tax, ftp_considered_b5

    @staticmethod
    def _unpaid_and_tax_balance(tax_payment: np.ndarray, ets_ctr: np.ndarray):
        unpaid_tax = np.zeros_like(ets_ctr, dtype=float)
        unpaid_tax[0] = 0
        ctr_tax = np.zeros_like(ets_ctr, dtype=float)

        for i in range(1, len(unpaid_tax)):
            unpaid_tax[i] = max(0, unpaid_tax[i - 1] + tax_payment[i] - ets_ctr[i])
            ctr_tax[i] = min(ets_ctr[i], unpaid_tax[i - 1] + tax_payment[i])

        return unpaid_tax, ctr_tax

    def _get_sunk_cost(self, discount_rate_year: int):
        oil_cost_raw = (
                self._oil_depreciation
                + self._oil_undepreciated_asset
                + self._oil_non_capital
        )
        self._oil_sunk_cost = oil_cost_raw[
                              : (discount_rate_year - self.start_date.year + 1)
                              ]

        gas_cost_raw = (
                self._gas_depreciation
                + self._gas_undepreciated_asset
                + self._gas_non_capital
        )
        self._gas_sunk_cost = gas_cost_raw[
                              : (discount_rate_year - self.start_date.year + 1)
                              ]

        if discount_rate_year == self.start_date.year:
            self._oil_sunk_cost = np.zeros(1)
            self._gas_sunk_cost = np.zeros(1)

    def run(
            self,
            sulfur_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
            electricity_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_OIL_REVENUE,
            co2_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
            is_dmo_end_weighted: bool = False,
            tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
            tax_rate: float | np.ndarray | None = None,
            tax_payment_method: TaxPaymentMode = TaxPaymentMode.TAX_DUE_MODE,
            ftp_tax_regime=FTPTaxRegime.PDJP_20_2017,
            discount_rate_year: int = None,
            depr_method: DeprMethod = DeprMethod.PSC_DB,
            decline_factor: float | int = 2,
            year_ref: int = None,
            tax_type: TaxType = TaxType.VAT,
            vat_rate: np.ndarray | float = 0.0,
            lbt_rate: np.ndarray | float = 0.0,
            inflation_rate: np.ndarray | float = 0.0,
            future_rate: float = 0.02,
            inflation_rate_applied_to: InflationAppliedTo = InflationAppliedTo.CAPEX
    ):
        # Configure discount rate
        if discount_rate_year is None:
            discount_rate_year = self.start_date.year

        if discount_rate_year < self.start_date.year:
            raise SunkCostException(
                f"start_date year {self.start_date} "
                f"is after the discount rate year: {discount_rate_year}"
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
            inflation_rate_applied_to=inflation_rate_applied_to
        )

        # Get The Other Revenue as the chosen selection
        self._get_other_revenue(sulfur_revenue=sulfur_revenue,
                                electricity_revenue=electricity_revenue,
                                co2_revenue=co2_revenue)

        # Calculate FTP
        self._get_ftp()

        # Defining the PreTax Split, whether using conventional PreTax or Sliding
        if self.tax_split_type is not TaxSplitTypeCR.CONVENTIONAL:
            self._get_rc_icp_pretax()

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

        # Non-capital costs (intangible + opex + asr)
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

        # Filtering for only the cost that in the bracket of the project years
        self._oil_depreciation = self._oil_depreciation[
                                 : (self.end_date.year - self.start_date.year + 1)
                                 ]
        self._gas_depreciation = self._gas_depreciation[
                                 : (self.end_date.year - self.start_date.year + 1)
                                 ]

        # Get Sunk Cost
        self._get_sunk_cost(discount_rate_year)

        # Investment credit
        self._oil_ic, self._oil_ic_unrecovered, self._oil_ic_paid = self._get_ic(
            revenue=self._oil_revenue,
            ftp=self._oil_ftp,
            cost_alloc=FluidType.OIL,
            ic_rate=self.oil_ic_rate,
        )

        self._gas_ic, self._gas_ic_unrecovered, self._gas_ic_paid = self._get_ic(
            revenue=self._gas_revenue,
            ftp=self._gas_ftp,
            cost_alloc=FluidType.GAS,
            ic_rate=self.gas_ic_rate,
        )

        # self._oil_ic, self._oil_ic_unrecovered, self._oil_ic_paid = np.zeros_like(self.project_years), np.zeros_like(
        #     self.project_years), np.zeros_like(self.project_years)
        #
        # self._gas_ic, self._gas_ic_unrecovered, self._gas_ic_paid = np.zeros_like(self.project_years), np.zeros_like(
        #     self.project_years), np.zeros_like(self.project_years)

        # Unrecovered cost before transfer/consolidation
        self._oil_unrecovered_before_transfer = psc_tools.get_unrecovered_cost(
            depreciation=self._oil_depreciation,
            non_capital=self._oil_non_capital,
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            ic=self._oil_ic_paid,
        )

        self._gas_unrecovered_before_transfer = psc_tools.get_unrecovered_cost(
            depreciation=self._gas_depreciation,
            non_capital=self._gas_non_capital,
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            ic=self._gas_ic_paid,
        )

        # Cost to be recovered
        self._oil_cost_to_be_recovered = psc_tools.get_cost_to_be_recovered(
            unrecovered_cost=self._oil_unrecovered_before_transfer,
        )

        self._gas_cost_to_be_recovered = psc_tools.get_cost_to_be_recovered(
            unrecovered_cost=self._gas_unrecovered_before_transfer,
        )

        # Cost recovery
        self._oil_cost_recovery = self._get_cost_recovery(
            revenue=self._oil_revenue,
            ftp=self._oil_ftp,
            ic=self._oil_ic_paid,
            depreciation=self._oil_depreciation,
            non_capital=self._oil_non_capital,
            cost_to_be_recovered=self._oil_cost_to_be_recovered,
            cr_cap_rate=self.oil_cr_cap_rate,
        )

        self._gas_cost_recovery = self._get_cost_recovery(
            revenue=self._gas_revenue,
            ftp=self._gas_ftp,
            ic=self._gas_ic_paid,
            depreciation=self._gas_depreciation,
            non_capital=self._gas_non_capital,
            cost_to_be_recovered=self._gas_cost_to_be_recovered,
            cr_cap_rate=self.gas_cr_cap_rate,
        )

        # ETS (Equity to be Split) before transfer/consolidation
        self._oil_ets_before_transfer = self._get_ets_before_transfer(
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            ic=self._oil_ic_paid,
            cost_recovery=self._oil_cost_recovery,
        )

        self._gas_ets_before_transfer = self._get_ets_before_transfer(
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            ic=self._gas_ic_paid,
            cost_recovery=self._gas_cost_recovery,
        )

        self._transfer_to_oil, self._transfer_to_gas = psc_tools.get_transfer(
            gas_unrecovered=self._gas_unrecovered_before_transfer,
            oil_unrecovered=self._oil_unrecovered_before_transfer,
            gas_ets_pretransfer=self._gas_ets_before_transfer,
            oil_ets_pretransfer=self._oil_ets_before_transfer,
        )

        # Unrecovered cost after transfer/consolidation
        # self._oil_unrecovered_after_transfer = psc_tools.get_unrec_cost_after_tf(
        #     depreciation=self._oil_depreciation,
        #     non_capital=self._oil_non_capital,
        #     revenue=self._oil_revenue,
        #     ftp_ctr=self._oil_ftp_ctr,
        #     ftp_gov=self._oil_ftp_gov,
        #     ic=self._oil_ic_paid,
        #     transferred_cost=self._transfer_to_oil,
        # )
        #
        # self._gas_unrecovered_after_transfer = psc_tools.get_unrec_cost_after_tf(
        #     depreciation=self._gas_depreciation,
        #     non_capital=self._gas_non_capital,
        #     revenue=self._gas_revenue,
        #     ftp_ctr=self._gas_ftp_ctr,
        #     ftp_gov=self._gas_ftp_gov,
        #     ic=self._gas_ic_paid,
        #     transferred_cost=self._transfer_to_gas,
        # )

        self._oil_unrecovered_after_transfer = psc_tools.get_unrec_cost_after_tf(
            depreciation=self._oil_depreciation,
            non_capital=self._oil_non_capital,
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            ic=self._oil_ic_paid,
            transferred_cost_in=self._transfer_to_oil,
            transferred_cost_out=self._transfer_to_gas,
        )

        self._gas_unrecovered_after_transfer = psc_tools.get_unrec_cost_after_tf(
            depreciation=self._gas_depreciation,
            non_capital=self._gas_non_capital,
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            ic=self._gas_ic_paid,
            transferred_cost_in=self._transfer_to_gas,
            transferred_cost_out=self._transfer_to_oil,
        )

        # Cost to be recovered after transfer
        self._oil_cost_to_be_recovered_after_tf = (
            psc_tools.get_cost_to_be_recovered_after_tf(
                unrecovered_cost=self._oil_unrecovered_after_transfer,
                transferred_cost=self._transfer_to_oil,
            )
        )

        self._gas_cost_to_be_recovered_after_tf = (
            psc_tools.get_cost_to_be_recovered_after_tf(
                unrecovered_cost=self._gas_unrecovered_after_transfer,
                transferred_cost=self._transfer_to_gas,
            )
        )

        # Cost recovery after transfer
        self._oil_cost_recovery_after_tf = (
                self._get_cost_recovery(
                    revenue=self._oil_revenue,
                    ftp=self._oil_ftp,
                    ic=self._oil_ic_paid,
                    depreciation=self._oil_depreciation,
                    non_capital=self._oil_non_capital,
                    cost_to_be_recovered=self._oil_cost_to_be_recovered_after_tf,
                    cr_cap_rate=self.oil_cr_cap_rate,
                )
                + self._transfer_to_oil
        )

        self._gas_cost_recovery_after_tf = (
                self._get_cost_recovery(
                    revenue=self._gas_revenue,
                    ftp=self._gas_ftp,
                    ic=self._gas_ic_paid,
                    depreciation=self._gas_depreciation,
                    non_capital=self._gas_non_capital,
                    cost_to_be_recovered=self._gas_cost_to_be_recovered_after_tf,
                    cr_cap_rate=self.gas_cr_cap_rate,
                )
                + self._transfer_to_gas
        )

        # ETS after Transfer
        self._oil_ets_after_transfer = self._get_ets_after_transfer(
            ets_before_tf=self._oil_ets_before_transfer,
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            ic=self._oil_ic_paid,
            cost_recovery=self._oil_cost_recovery_after_tf,
            transferred_in=self._transfer_to_oil,
            transferred_out=self._transfer_to_gas,
        )

        self._gas_ets_after_transfer = self._get_ets_after_transfer(
            ets_before_tf=self._gas_ets_before_transfer,
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            ic=self._gas_ic_paid,
            cost_recovery=self._gas_cost_recovery_after_tf,
            transferred_in=self._transfer_to_gas,
            transferred_out=self._transfer_to_oil,
        )

        # ES (Equity Share)
        self._oil_contractor_share, self._oil_government_share = self._get_equity_share(
            ets=self._oil_ets_after_transfer, pretax_ctr=self.oil_ctr_pretax_share
        )

        self._gas_contractor_share, self._gas_government_share = self._get_equity_share(
            ets=self._gas_ets_after_transfer, pretax_ctr=self.gas_ctr_pretax_share
        )

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
            ctr_pretax_share=self.oil_ctr_pretax_share,
            unrecovered_cost=self._oil_unrecovered_after_transfer,
            is_dmo_end_weighted=is_dmo_end_weighted,
        )

        self._gas_dmo_volume, self._gas_dmo_fee, self._gas_ddmo = psc_tools.get_dmo(
            onstream_date=self.gas_onstream_date,
            start_date=self.start_date,
            project_years=self.project_years,
            dmo_holiday_duration=self.gas_dmo_holiday_duration,
            dmo_volume_portion=self.gas_dmo_volume_portion,
            dmo_fee_portion=self.gas_dmo_fee_portion,
            lifting=self._gas_lifting,
            price=self._gas_wap_price,
            ctr_pretax_share=self.gas_ctr_pretax_share,
            unrecovered_cost=self._gas_unrecovered_after_transfer,
            is_dmo_end_weighted=is_dmo_end_weighted,
        )

        # Taxable income (also known as Net Contractor Share - NCS)
        self._oil_taxable_income = (
                self._oil_ftp_ctr
                + self._oil_contractor_share
                + self._oil_ic_paid
                - self._oil_ddmo
        )
        self._gas_taxable_income = (
                self._gas_ftp_ctr
                + self._gas_contractor_share
                + self._gas_ic_paid
                - self._gas_ddmo
        )

        # Tax Payment
        # Generating Tax array if tax_rate argument is a single value not array
        if isinstance(tax_rate, float):
            self._tax_rate_arr = np.full_like(self.project_years, tax_rate, dtype=float)

        # Generating Tax array based on the tax regime if tax_rate argument is None
        if tax_rate is None:
            self._tax_rate_arr = self._get_tax_by_regime(tax_regime=tax_regime)

        self._oil_tax_payment, _oil_ftp_considered = self._get_tax_payment(
            ctr_share=self._oil_contractor_share,
            taxable_income=self._oil_taxable_income,
            tax_rate=self._tax_rate_arr,
            ftp_ctr=self._oil_ftp_ctr,
            unrec_cost=self._oil_unrecovered_after_transfer,
            ftp_tax_regime=ftp_tax_regime,
        )

        self._gas_tax_payment, _gas_ftp_considered = self._get_tax_payment(
            ctr_share=self._gas_contractor_share,
            taxable_income=self._gas_taxable_income,
            tax_rate=self._tax_rate_arr,
            ftp_ctr=self._gas_ftp_ctr,
            unrec_cost=self._gas_unrecovered_after_transfer,
            ftp_tax_regime=ftp_tax_regime,
        )

        # Contractor Share
        self._oil_ctr_net_share = self._oil_taxable_income - self._oil_tax_payment
        self._gas_ctr_net_share = self._gas_taxable_income - self._gas_tax_payment

        # Contractor Take by Fluid
        # self._oil_contractor_take = (
        #         self._oil_taxable_income
        #         - self._oil_tax_payment
        #         + self._oil_cost_recovery_after_tf
        # )
        #
        # self._gas_contractor_take = (
        #         self._gas_taxable_income
        #         - self._gas_tax_payment
        #         + self._gas_cost_recovery_after_tf
        # )

        self._oil_contractor_take = (
                self._oil_taxable_income
                - self._oil_tax_payment
                + self._oil_cost_recovery_after_tf
        )

        self._gas_contractor_take = (
                self._gas_taxable_income
                - self._gas_tax_payment
                + self._gas_cost_recovery_after_tf
        )

        # Contractor CashFlow
        self._oil_cashflow = self._oil_contractor_take - (
                self._oil_tangible_expenditures + self._oil_non_capital
        )
        self._gas_cashflow = self._gas_contractor_take - (
                self._gas_tangible_expenditures + self._gas_non_capital
        )

        # Government Take by Fluid
        self._oil_government_take = (
                self._oil_ftp_gov
                + self._oil_government_share
                + self._oil_tax_payment
                + self._oil_ddmo
        )

        self._gas_government_take = (
                self._gas_ftp_gov
                + self._gas_government_share
                + self._gas_tax_payment
                + self._gas_ddmo
        )

        # Consolidated attributes
        self._consolidated_revenue = self._oil_revenue + self._gas_revenue
        self._consolidated_tangible = (
                self._oil_tangible_expenditures + self._gas_tangible_expenditures
        )
        self._consolidated_intangible = (
                self._oil_intangible_expenditures + self._gas_intangible_expenditures
        )
        self._consolidated_sunk_cost = self._oil_sunk_cost + self._gas_sunk_cost
        self._consolidated_opex = (
                self._oil_opex_expenditures + self._gas_opex_expenditures
        )
        self._consolidated_asr = (
                self._oil_asr_expenditures + self._gas_asr_expenditures
        )
        self._consolidated_non_capital = self._oil_non_capital + self._gas_non_capital
        self._consolidated_depreciation = (
                self._oil_depreciation + self._gas_depreciation
        )

        self._consolidated_undepreciated_asset = (
                self._oil_undepreciated_asset + self._gas_undepreciated_asset
        )
        self._consolidated_ftp = self._oil_ftp + self._gas_ftp
        self._consolidated_ftp_ctr = self._oil_ftp_ctr + self._gas_ftp_ctr
        self._consolidated_ftp_gov = self._oil_ftp_gov + self._gas_ftp_gov
        self._consolidated_ic = self._oil_ic + self._gas_ic
        self._consolidated_ic_unrecovered = (
                self._oil_ic_unrecovered + self._gas_ic_unrecovered
        )
        self._consolidated_ic_paid = self._oil_ic_paid + self._gas_ic_paid
        self._consolidated_unrecovered_before_transfer = (
                self._oil_unrecovered_before_transfer
                + self._gas_unrecovered_before_transfer
        )
        self._consolidated_cost_recovery_before_transfer = (
                self._oil_cost_recovery + self._gas_cost_recovery
        )
        self._consolidated_ets_before_transfer = (
                self._oil_ets_before_transfer + self._gas_ets_before_transfer
        )
        self._consolidated_unrecovered_after_transfer = (
                self._oil_unrecovered_after_transfer + self._gas_unrecovered_after_transfer
        )
        self._consolidated_cost_to_be_recovered_after_tf = (
                self._oil_cost_to_be_recovered_after_tf
                + self._gas_cost_to_be_recovered_after_tf
        )
        self._consolidated_cost_recovery_after_tf = (
                self._oil_cost_recovery_after_tf + self._gas_cost_recovery_after_tf
        )
        self._consolidated_ets_after_transfer = (
                self._oil_ets_after_transfer + self._gas_ets_after_transfer
        )
        self._consolidated_contractor_share = (
                self._oil_contractor_share + self._gas_contractor_share
        )
        self._consolidated_government_share = (
                self._oil_government_share + self._gas_government_share
        )
        self._consolidated_dmo_volume = self._oil_dmo_volume + self._gas_dmo_volume
        self._consolidated_dmo_fee = self._oil_dmo_fee + self._gas_dmo_fee
        self._consolidated_ddmo = self._oil_ddmo + self._gas_ddmo
        self._consolidated_taxable_income = (
                self._oil_taxable_income + self._gas_taxable_income
        )

        # Calculating the consolidated tax based on the tax payment mode
        _consolidated_ftp_tax = np.zeros_like(self._consolidated_contractor_share)
        if tax_payment_method is TaxPaymentMode.TAX_DUE_MODE:
            self._consolidated_tax_due, _consolidated_ftp_tax = self._get_tax_payment(
                ctr_share=self._consolidated_contractor_share,
                taxable_income=self._consolidated_taxable_income,
                tax_rate=self._tax_rate_arr,
                ftp_ctr=self._consolidated_ftp_ctr,
                unrec_cost=self._consolidated_unrecovered_after_transfer,
                ftp_tax_regime=FTPTaxRegime.PDJP_20_2017,
            )

            (
                self._consolidated_unpaid_tax_balance,
                self._consolidated_tax_payment,
            ) = self._unpaid_and_tax_balance(
                tax_payment=self._consolidated_tax_due,
                ets_ctr=self._consolidated_contractor_share,
            )

        elif tax_payment_method is TaxPaymentMode.TAX_DIRECT_MODE:
            self._consolidated_tax_payment = (
                    self._oil_tax_payment + self._gas_tax_payment
            )

        self._consolidated_ctr_net_share = (
                self._consolidated_taxable_income - self._consolidated_tax_payment
        )

        self._consolidated_contractor_take = (
                self._consolidated_taxable_income
                - self._consolidated_tax_payment
                + self._consolidated_cost_recovery_after_tf
                - _consolidated_ftp_tax
        )

        self._consolidated_government_take = (
                self._consolidated_ftp_gov
                + self._consolidated_government_share
                + self._consolidated_tax_payment
                + self._consolidated_ddmo
        )

        self._consolidated_cashflow = self._consolidated_contractor_take - (
                self._consolidated_tangible + self._consolidated_non_capital
        )
