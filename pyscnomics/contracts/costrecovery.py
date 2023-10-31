"""
Handles calculations associated with PSC Cost Recovery.
"""

from dataclasses import dataclass, field
import numpy as np
from functools import reduce

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts import psc_tools
from pyscnomics.econ.selection import FluidType, YearReference, TaxRegime, FTPTaxRegime, TaxSplitTypeCR, NPVSelection
from pyscnomics.econ import indicator
from pyscnomics.tools.summary import Summary


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

    _summary: Summary = field(default=None)

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

    _oil_unrecovered_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_unrecovered_before_transfer: np.ndarray = field(default=None, init=False, repr=False)

    _oil_cost_to_be_recovered: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cost_to_be_recovered: np.ndarray = field(default=None, init=False, repr=False)

    _oil_cost_recovery: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cost_recovery: np.ndarray = field(default=None, init=False, repr=False)

    _oil_ets_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ets_before_transfer: np.ndarray = field(default=None, init=False, repr=False)

    _transfer_to_oil: np.ndarray = field(default=None, init=False, repr=False)
    _transfer_to_gas: np.ndarray = field(default=None, init=False, repr=False)

    _oil_unrecovered_after_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_unrecovered_after_transfer: np.ndarray = field(default=None, init=False, repr=False)

    _oil_cost_to_be_recovered_after_tf: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cost_to_be_recovered_after_tf: np.ndarray = field(default=None, init=False, repr=False)

    _oil_cost_recovery_after_tf: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cost_recovery_after_tf: np.ndarray = field(default=None, init=False, repr=False)

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
    _consolidated_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ftp: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ftp_ctr: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ftp_gov: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ic: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ic_unrecovered: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ic_paid: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_unrecovered_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_cost_recovery_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ets_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_unrecovered_after_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_cost_to_be_recovered_after_tf: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_cost_recovery_after_tf: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ets_after_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_contractor_share: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_government_share: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_dmo_volume: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_dmo_fee: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ddmo: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_taxable_income: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_tax_due: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_unpaid_tax_balance: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_tax_payment: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ctr_net_share: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_contractor_take: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_government_take: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_cashflow: np.ndarray = field(default=None, init=False, repr=False)

    def _get_aggregate(self):
        self._oil_lifting = self._get_oil_lifting()
        self._gas_lifting = self._get_gas_lifting()
        self._oil_tangible = self._get_oil_tangible()
        self._gas_tangible = self._get_gas_tangible()
        self._oil_intangible = self._get_oil_intangible()
        self._gas_intangible = self._get_gas_intangible()
        self._oil_opex = self._get_oil_opex()
        self._gas_opex = self._get_gas_opex()
        self._oil_asr = self._get_oil_asr()
        self._gas_asr = self._get_gas_asr()

    def _get_revenue(self):
        self._oil_revenue = self._oil_lifting.revenue()
        self._gas_revenue = self._gas_lifting.revenue()

    def _get_rc_icp_pretax(self):
        # Extract relevant values from the condition_dict dictionary
        bot_limits = np.array([self.condition_dict[c]['bot_limit'] for c in self.condition_dict])
        top_limits = np.array([self.condition_dict[c]['top_limit'] for c in self.condition_dict])
        ctr_oil_values = np.array([self.condition_dict[c]['ctr_oil'] for c in self.condition_dict])
        ctr_gas_values = np.array([self.condition_dict[c]['ctr_gas'] for c in self.condition_dict])

        # Create conditions using vectorized comparisons
        conditions = (bot_limits < self.indicator_rc_icp_sliding) & (self.indicator_rc_icp_sliding <= top_limits)

        # Calculate corresponding values using np.select()
        self.oil_ctr_pretax_share = np.select(conditions, ctr_oil_values, default=np.nan)
        self.gas_ctr_pretax_share = np.select(conditions, ctr_gas_values, default=np.nan)

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
            self, revenue: np.ndarray, ftp: np.ndarray, cost_alloc: FluidType, ic_rate: float
    ) -> tuple:

        if any(i for i in self.tangible_cost if i.is_ic_applied):
            tangible_ic_applied = reduce(
                lambda x, y: x + y,
                (
                    i
                    for i in self.tangible_cost
                    if i.cost_allocation == cost_alloc and i.is_ic_applied
                ),
            )

            ic_total = ic_rate * tangible_ic_applied.expenditures(year_ref=YearReference.PIS_YEAR)

            ic_unrecovered = np.cumsum(ic_total) - np.cumsum(revenue - ftp)
            ic_unrecovered = np.where(ic_unrecovered > 0, ic_unrecovered, 0)

            ic_unrec_next = np.roll(ic_unrecovered, 1)
            ic_unrec_next[0] = 0
            ic_paid = np.minimum(revenue - ftp, ic_total + ic_unrec_next)
            return ic_total, ic_unrecovered, ic_paid

        else:

            return tuple(np.squeeze(np.hsplit(np.zeros((revenue.size, 3)), 3)))

    @staticmethod
    def _get_cost_recovery(revenue: np.ndarray,
                           ftp: np.ndarray,
                           ic: np.ndarray,
                           depreciation: np.ndarray,
                           non_capital: np.ndarray,
                           cost_to_be_recovered: np.ndarray,
                           cr_cap_rate: float) -> np.ndarray:
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

        return np.minimum(revenue - ftp - ic,
                          ((depreciation + non_capital + cost_to_be_recovered) * cr_cap_rate))
        # There is possibility for a bug in Cap rate
        # return (depreciation + non_capital + cost_to_be_recovered) * cr_cap_rate

    @staticmethod
    def _get_ets_before_transfer(revenue: np.ndarray,
                                 ftp_ctr: np.ndarray,
                                 ftp_gov: np.ndarray,
                                 ic: np.ndarray,
                                 cost_recovery: np.ndarray) -> np.ndarray:
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
        return revenue - (ftp_ctr + ftp_gov) - ic - cost_recovery

    @staticmethod
    def _get_ets_after_transfer(revenue: np.ndarray,
                                ftp_ctr: np.ndarray,
                                ftp_gov: np.ndarray,
                                ic: np.ndarray,
                                cost_recovery: np.ndarray,
                                transferred_in: np.ndarray,
                                transferred_out: np.ndarray) -> np.ndarray:
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
        ets_after_transfer = revenue - (ftp_ctr + ftp_gov) - ic - cost_recovery - transferred_out + transferred_in
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
    def _get_ftp_tax_payment(unrec: np.ndarray,
                             ftp: np.ndarray,
                             tax_rate: float,
                             ftp_tax_regime: FTPTaxRegime = FTPTaxRegime.PDJP_20_2017):
        """
        A funtion to get the array of tax payment of the First Tranche Petroleum (FTP).

        Parameters
        ----------
        unrec: np.ndarray
            The array containing the Unrecovered Cost.
        ftp: np.ndarray
            The array containing the contractor's FTP.
        tax_rate: float
            The tax rate value.
        ftp_tax_regime: TaxSplitTypeCR
            The regime configuration used.

        Returns
        -------
        ftp_tax_payment: np.ndarray
            The array of Tax Payment of the FTP.
        """
        if ftp_tax_regime == FTPTaxRegime.PDJP_20_2017:
            cum_ftp = np.cumsum(ftp)
            cum_ftp = np.where(cum_ftp > unrec, cum_ftp, 0)
            tax_of_ftp = np.concatenate((np.zeros(1), np.diff(cum_ftp * tax_rate)))
            ftp_tax_payment = np.where(tax_of_ftp > 0, tax_of_ftp, 0)

        else:
            ftp_tax_payment = ftp * tax_rate

        return ftp_tax_payment

    @staticmethod
    def _get_tax_payment(ctr_share,
                         taxable_income,
                         tax_rate,
                         ftp_tax_regime: FTPTaxRegime = FTPTaxRegime.PDJP_20_2017):
        applied_tax = np.where(ctr_share > 0, 1, 0)
        if ftp_tax_regime == FTPTaxRegime.PDJP_20_2017:
            cum_ti = np.cumsum(taxable_income)

            applied_tax_prior = np.concatenate((np.zeros(1), applied_tax))[0:-1]

            ctr_tax = np.where(np.logical_and(applied_tax == 1, applied_tax_prior == 0),
                               cum_ti * tax_rate,
                               np.where(np.logical_and(applied_tax == 1, applied_tax_prior == 1),
                                        taxable_income * tax_rate,
                                        0))
        else:
            ctr_tax = taxable_income * tax_rate

        return ctr_tax

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

        oil_cost_raw = self._oil_depreciation + self._oil_undepreciated_asset + self._oil_non_capital
        self._oil_sunk_cost = oil_cost_raw[:(discount_rate_year - self.start_date.year + 1)]

        gas_cost_raw = self._gas_depreciation + self._gas_undepreciated_asset + self._gas_non_capital
        self._gas_sunk_cost = gas_cost_raw[:(discount_rate_year - self.start_date.year + 1)]

        if discount_rate_year == self.start_date.year:
            self._oil_sunk_cost = np.zeros(1)
            self._gas_sunk_cost = np.zeros(1)

    def run(self,
            is_dmo_end_weighted: bool = False,
            ctr_tax: float | np.ndarray = None,
            tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
            tax_rate: float = 0.44,
            ftp_tax_regime=FTPTaxRegime.PDJP_20_2017,
            discount_rate_year: int = None,
            discount_rate: float = 0.1,
            npv_mode: NPVSelection = NPVSelection.POINT_FORWARD
            ):

        if discount_rate_year is None:
            discount_rate_year = self.start_date.year

        if discount_rate_year < self.start_date.year:
            raise SunkCostException(
                f"start_date year {self.start_date} "
                f"is after the discount rate year: {self.end_date}"
            )

        self._get_aggregate()

        self._get_wap_price()

        self._get_revenue()
        self._get_ftp()

        # Defining the PreTax Split, wheter using conventional PreTax or Sliding
        if self.tax_split_type is not TaxSplitTypeCR.CONVENTIONAL:
            self._get_rc_icp_pretax()

        # Depreciation (tangible cost)
        (
            self._oil_depreciation,
            self._oil_undepreciated_asset,
        ) = self._oil_tangible.total_depreciation_rate()
        (
            self._gas_depreciation,
            self._gas_undepreciated_asset,
        ) = self._gas_tangible.total_depreciation_rate()

        # Non-capital costs (intangible + opex + asr)
        self._oil_non_capital = (
                self._oil_intangible.expenditures()
                + self._oil_opex.expenditures()
                + self._oil_asr.expenditures()
        )

        self._gas_non_capital = (
                self._gas_intangible.expenditures()
                + self._gas_opex.expenditures()
                + self._gas_asr.expenditures()
        )

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
        self._oil_cost_recovery = self._get_cost_recovery(revenue=self._oil_revenue,
                                                          ftp=self._oil_ftp,
                                                          ic=self._oil_ic_paid,
                                                          depreciation=self._oil_depreciation,
                                                          non_capital=self._oil_non_capital,
                                                          cost_to_be_recovered=self._oil_cost_to_be_recovered,
                                                          cr_cap_rate=self.oil_cr_cap_rate,
                                                          )

        self._gas_cost_recovery = self._get_cost_recovery(revenue=self._gas_revenue,
                                                          ftp=self._gas_ftp,
                                                          ic=self._gas_ic_paid,
                                                          depreciation=self._gas_depreciation,
                                                          non_capital=self._gas_non_capital,
                                                          cost_to_be_recovered=self._gas_cost_to_be_recovered,
                                                          cr_cap_rate=self.gas_cr_cap_rate
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
        self._oil_unrecovered_after_transfer = psc_tools.get_unrec_cost_after_tf(
            depreciation=self._oil_depreciation,
            non_capital=self._oil_non_capital,
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            ic=self._oil_ic_paid,
            transferred_cost=self._transfer_to_oil)

        self._gas_unrecovered_after_transfer = psc_tools.get_unrec_cost_after_tf(
            depreciation=self._gas_depreciation,
            non_capital=self._gas_non_capital,
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            ic=self._gas_ic_paid,
            transferred_cost=self._transfer_to_gas)

        # Cost to be recovered after transfer
        self._oil_cost_to_be_recovered_after_tf = psc_tools.get_cost_to_be_recovered_after_tf(
            unrecovered_cost=self._oil_unrecovered_before_transfer,
            transferred_cost=self._transfer_to_oil
        )

        self._gas_cost_to_be_recovered_after_tf = psc_tools.get_cost_to_be_recovered_after_tf(
            unrecovered_cost=self._gas_unrecovered_before_transfer,
            transferred_cost=self._transfer_to_gas
        )

        # Cost recovery after transfer
        self._oil_cost_recovery_after_tf = self._get_cost_recovery(
            revenue=self._oil_revenue,
            ftp=self._oil_ftp,
            ic=self._oil_ic_paid,
            depreciation=self._oil_depreciation,
            non_capital=self._oil_non_capital,
            cost_to_be_recovered=self._oil_cost_to_be_recovered_after_tf,
            cr_cap_rate=self.oil_cr_cap_rate,
        ) + self._transfer_to_oil

        self._gas_cost_recovery_after_tf = self._get_cost_recovery(
            revenue=self._gas_revenue,
            ftp=self._gas_ftp,
            ic=self._gas_ic_paid,
            depreciation=self._gas_depreciation,
            non_capital=self._gas_non_capital,
            cost_to_be_recovered=self._gas_cost_to_be_recovered_after_tf,
            cr_cap_rate=self.gas_cr_cap_rate
        ) + self._transfer_to_gas

        # ETS after Transfer
        self._oil_ets_after_transfer = self._get_ets_after_transfer(
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            ic=self._oil_ic_paid,
            cost_recovery=self._oil_cost_recovery_after_tf,
            transferred_in=self._transfer_to_oil,
            transferred_out=self._transfer_to_gas
        )

        self._gas_ets_after_transfer = self._get_ets_after_transfer(
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            ic=self._gas_ic_paid,
            cost_recovery=self._gas_cost_recovery_after_tf,
            transferred_in=self._transfer_to_gas,
            transferred_out=self._transfer_to_oil
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
            ctr_pretax_share=self.gas_ctr_pretax_share,
            unrecovered_cost=self._gas_unrecovered_after_transfer,
            is_dmo_end_weighted=is_dmo_end_weighted)

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
        self._oil_tax_payment = self._get_tax_payment(ctr_share=self._oil_contractor_share,
                                                      taxable_income=self._oil_taxable_income,
                                                      tax_rate=tax_rate,
                                                      ftp_tax_regime=ftp_tax_regime)

        self._gas_tax_payment = self._get_tax_payment(ctr_share=self._gas_contractor_share,
                                                      taxable_income=self._gas_taxable_income,
                                                      tax_rate=tax_rate,
                                                      ftp_tax_regime=ftp_tax_regime)

        # Contractor Share
        self._oil_ctr_net_share = self._oil_taxable_income - self._oil_tax_payment
        self._gas_ctr_net_share = self._gas_taxable_income - self._gas_tax_payment

        # Contractor Take by Fluid
        self._oil_contractor_take = (
                self._oil_taxable_income - self._oil_tax_payment + self._oil_cost_recovery_after_tf
        )

        self._gas_contractor_take = (
                self._gas_taxable_income - self._gas_tax_payment + self._gas_cost_recovery_after_tf
        )

        # Contractor CashFlow
        self._oil_cashflow = self._oil_contractor_take - (self._oil_tangible.expenditures() + self._oil_non_capital)
        self._gas_cashflow = self._gas_contractor_take - (self._gas_tangible.expenditures() + self._gas_non_capital)

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
        self._consolidated_tangible = self._oil_tangible.expenditures() + self._gas_tangible.expenditures()
        self._consolidated_intangible = self._oil_intangible.expenditures() + self._gas_intangible.expenditures()
        self._consolidated_sunk_cost = self._oil_sunk_cost + self._gas_sunk_cost
        self._consolidated_opex = self._oil_opex.expenditures() + self._gas_opex.expenditures()
        self._consolidated_asr = self._oil_asr.expenditures() + self._gas_asr.expenditures()
        self._consolidated_non_capital = self._oil_non_capital + self._gas_non_capital
        self._consolidated_depreciation = self._oil_depreciation + self._gas_depreciation
        self._consolidated_undepreciated_asset = self._oil_undepreciated_asset + self._gas_undepreciated_asset
        self._consolidated_ftp = self._oil_ftp + self._gas_ftp
        self._consolidated_ftp_ctr = self._oil_ftp_ctr + self._gas_ftp_ctr
        self._consolidated_ftp_gov = self._oil_ftp_gov + self._gas_ftp_gov
        self._consolidated_ic = self._oil_ic + self._gas_ic
        self._consolidated_ic_unrecovered = self._oil_ic_unrecovered + self._gas_ic_unrecovered
        self._consolidated_ic_paid = self._oil_ic_paid + self._gas_ic_paid
        self._consolidated_unrecovered_before_transfer = (self._oil_unrecovered_before_transfer +
                                                          self._gas_unrecovered_before_transfer)
        self._consolidated_cost_recovery_before_transfer = self._oil_cost_recovery + self._gas_cost_recovery
        self._consolidated_ets_before_transfer = self._oil_ets_before_transfer + self._gas_ets_before_transfer
        self._consolidated_unrecovered_after_transfer = (self._oil_unrecovered_after_transfer +
                                                         self._gas_unrecovered_after_transfer)
        self._consolidated_cost_to_be_recovered_after_tf = (self._oil_cost_to_be_recovered_after_tf +
                                                            self._gas_cost_to_be_recovered_after_tf)
        self._consolidated_cost_recovery_after_tf = self._oil_cost_recovery_after_tf + self._gas_cost_recovery_after_tf
        self._consolidated_ets_after_transfer = self._oil_ets_after_transfer + self._gas_ets_after_transfer
        self._consolidated_contractor_share = self._oil_contractor_share + self._gas_contractor_share
        self._consolidated_government_share = self._oil_government_share + self._gas_government_share
        self._consolidated_dmo_volume = self._oil_dmo_volume + self._gas_dmo_volume
        self._consolidated_dmo_fee = self._oil_dmo_fee + self._gas_dmo_fee
        self._consolidated_ddmo = self._oil_ddmo + self._gas_ddmo
        self._consolidated_taxable_income = self._oil_taxable_income + self._gas_taxable_income

        self._consolidated_tax_due = self._get_tax_payment(ctr_share=self._consolidated_contractor_share,
                                                           taxable_income=self._consolidated_taxable_income,
                                                           tax_rate=tax_rate,
                                                           ftp_tax_regime=ftp_tax_regime)

        self._consolidated_unpaid_tax_balance, self._consolidated_tax_payment = self._unpaid_and_tax_balance(
            tax_payment=self._consolidated_tax_due,
            ets_ctr=self._consolidated_contractor_share)

        self._consolidated_ctr_net_share = self._consolidated_taxable_income - self._consolidated_tax_payment

        self._consolidated_contractor_take = (self._consolidated_taxable_income -
                                              self._consolidated_tax_payment +
                                              self._consolidated_cost_recovery_after_tf)

        self._consolidated_government_take = (
                self._consolidated_ftp_gov
                + self._consolidated_government_share
                + self._consolidated_tax_payment
                + self._consolidated_ddmo
        )

        self._consolidated_cashflow = (self._consolidated_contractor_take -
                                       (self._consolidated_tangible +
                                        self._consolidated_non_capital))
