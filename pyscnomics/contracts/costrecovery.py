"""
Handles calculations associated with PSC Cost Recovery.
"""

from dataclasses import dataclass, field
import numpy as np
from functools import reduce

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts import psc_tools
from pyscnomics.econ.selection import FluidType, YearReference, TaxRegime, FTPTaxRegime, TaxSplitTypeCR
from pyscnomics.econ import indicator


@dataclass
class CostRecovery(BaseProject):
    ftp_is_available: bool = field(default=True)
    ftp_is_shared: bool = field(default=True)
    ftp_portion: float = field(default=0.2)

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
    _oil_ftp_tax_payment: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ftp_tax_payment: np.ndarray = field(default=None, init=False, repr=False)

    _oil_contractor_take: np.ndarray = field(default=None, init=False, repr=False)
    _gas_contractor_take: np.ndarray = field(default=None, init=False, repr=False)
    _oil_government_take: np.ndarray = field(default=None, init=False, repr=False)
    _gas_government_take: np.ndarray = field(default=None, init=False, repr=False)

    _oil_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cashflow: np.ndarray = field(default=None, init=False, repr=False)

    _oil_pot: float = field(default=None, init=False, repr=False)
    _gas_pot: float = field(default=None, init=False, repr=False)

    _oil_npv: float = field(default=None, init=False, repr=False)
    _gas_npv: float = field(default=None, init=False, repr=False)

    _oil_irr: float = field(default=None, init=False, repr=False)
    _gas_irr: float = field(default=None, init=False, repr=False)

    _ctr_take: np.ndarray = field(default=None, init=False, repr=False)
    _ctr_share: np.ndarray = field(default=None, init=False, repr=False)
    _gov_take: np.ndarray = field(default=None, init=False, repr=False)
    _gov_share: np.ndarray = field(default=None, init=False, repr=False)

    _ftp: np.ndarray = field(default=None, init=False, repr=False)
    _ic: np.ndarray = field(default=None, init=False, repr=False)
    _cost_recovery: np.ndarray = field(default=None, init=False, repr=False)

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

        if self.ftp_is_available:
            self._oil_ftp = self.ftp_portion * self._oil_revenue
            self._gas_ftp = self.ftp_portion * self._gas_revenue
            if self.ftp_is_shared:
                self._oil_ftp_ctr = self.oil_ctr_pretax_share * self._oil_ftp
                self._gas_ftp_ctr = self.gas_ctr_pretax_share * self._gas_ftp

            self._oil_ftp_gov = self._oil_ftp - self._oil_ftp_ctr
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

            # FIXME: Find an alternative to np.roll()
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

    def run(self,
            is_dmo_end_weighted=False,
            ctr_tax: float | np.ndarray = None,
            tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
            tax_rate=0.44,   # TODO: Replace with NaN as default value after implementing TaxRegime.
            ftp_tax_regime=FTPTaxRegime.PDJP_20_2017
            ):
        # TODO: Tax rate argument will be deleted then replaced with the value in tax_regime.
        #  Currently, it is used for testing

        self._get_aggregate()
        self._get_revenue()
        self._get_ftp()

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
        self._oil_unrecovered_after_transfer = (
                self._oil_unrecovered_before_transfer - self._transfer_to_oil
        )
        self._gas_unrecovered_after_transfer = (
                self._gas_unrecovered_before_transfer - self._transfer_to_gas
        )

        # ETS (Equity to be Split) after transfer/consolidation
        self._oil_ets_after_transfer = psc_tools.get_ets_after_transfer(
            ets_before_transfer=self._oil_ets_before_transfer,
            trfto=self._transfer_to_oil,
            unrecovered_after_transfer=self._oil_unrecovered_after_transfer,
        )

        self._gas_ets_after_transfer = psc_tools.get_ets_after_transfer(
            ets_before_transfer=self._gas_ets_before_transfer,
            trfto=self._transfer_to_gas,
            unrecovered_after_transfer=self._gas_unrecovered_after_transfer,
        )

        # ES (Equity Share)
        self._oil_contractor_share, self._oil_government_share = self._get_equity_share(
            ets=self._oil_ets_after_transfer, pretax_ctr=self.oil_ctr_pretax_share
        )

        self._gas_contractor_share, self._gas_government_share = self._get_equity_share(
            ets=self._gas_ets_after_transfer, pretax_ctr=self.gas_ctr_pretax_share
        )

        self._oil_dmo_volume, self._oil_dmo_fee, self._oil_ddmo = psc_tools.get_dmo(
            onstream_date=self.oil_onstream_date,
            start_date=self.start_date,
            project_years=self.project_years,
            dmo_holiday_duration=self.oil_dmo_holiday_duration,
            dmo_volume_portion=self.oil_dmo_volume_portion,
            dmo_fee_portion=self.oil_dmo_fee_portion,
            lifting=self._oil_lifting,
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

        self._oil_ftp_tax_payment = self._get_ftp_tax_payment(unrec=self._oil_unrecovered_after_transfer,
                                                              ftp=self._oil_ftp_ctr,
                                                              tax_rate=tax_rate,
                                                              ftp_tax_regime=ftp_tax_regime)

        self._gas_ftp_tax_payment = self._get_ftp_tax_payment(unrec=self._gas_unrecovered_after_transfer,
                                                              ftp=self._gas_ftp_ctr,
                                                              tax_rate=tax_rate,
                                                              ftp_tax_regime=ftp_tax_regime)

        # Contractor Take by Fluid
        self._oil_contractor_take = (
                self._oil_taxable_income - self._oil_ftp_tax_payment + self._oil_cost_recovery
        )

        self._gas_contractor_take = (
                self._gas_taxable_income - self._gas_ftp_tax_payment + self._gas_cost_recovery
        )

        # Government Take by Fluid
        self._oil_government_take = (
                self._oil_ftp_gov
                + self._oil_government_share
                + self._oil_ftp_tax_payment
                + self._oil_ddmo
        )

        self._gas_government_take = (
                self._gas_ftp_gov
                + self._gas_government_share
                + self._gas_ftp_tax_payment
                + self._gas_ddmo
        )

        # Contractor CashFlow
        self._oil_cashflow = self._oil_contractor_take - self._oil_tangible.expenditures() - self._oil_non_capital
        self._gas_cashflow = self._gas_contractor_take - self._gas_tangible.expenditures() - self._gas_non_capital

        # Pay Out Time (POT) and Internal Rate Return (IRR)
        # Condition where there is no revenue from one of Oil and Gas
        if np.sum(self._oil_cashflow) == 0:
            self._oil_pot = 0
            self._oil_irr = 0
        else:
            self._oil_pot = indicator.pot(cashflow=self._oil_cashflow)
            self._oil_irr = indicator.irr(cashflow=self._oil_cashflow)

        if np.sum(self._gas_cashflow) == 0:
            self._gas_pot = 0
            self._gas_irr = 0
        else:
            self._gas_pot = indicator.pot(cashflow=self._gas_cashflow)
            self._gas_irr = indicator.irr(cashflow=self._gas_cashflow)

        # NPV
        self._oil_npv = indicator.npv(cashflow=self._oil_cashflow)
        self._gas_npv = indicator.npv(cashflow=self._gas_cashflow)

        # Combined Contractor Take and Contractor Share, used for testing the consistency
        self._ctr_take = self._oil_contractor_take + self._gas_contractor_take
        self._ctr_share = self._oil_contractor_share + self._gas_contractor_share

        # Combined Government Take and Contractor Share, used for testing the consistency
        self._gov_take = self._oil_government_take + self._gas_government_take
        self._gov_share = self._oil_government_share + self._gas_government_share

        # Combined FTP, IC, Cost_Recovery, used for testing the consistency
        self._ftp = self._oil_ftp + self._gas_ftp
        self._ic = self._oil_ic + self._gas_ic
        self._cost_recovery = self._oil_cost_recovery + self._gas_cost_recovery

        return
