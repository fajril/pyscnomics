"""
Handles calculations associated with PSC Cost Recovery.
"""

from dataclasses import dataclass, field
import numpy as np
import dateutils
from functools import reduce

from pyscnomics.contracts.project import BaseProject
from pyscnomics.econ.selection import FluidType, YearReference, TaxRegime, FTPTaxRegime
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.econ.revenue import Lifting


@dataclass
class CostRecovery(BaseProject):
    ftp_is_available: bool = field(default=True)
    ftp_is_shared: bool = field(default=True)
    ftp_portion: float = field(default=0.2)

    # TODO: accommodate pretax value according to ICP sliding scale and R/C ratio
    # TODO: raise exception to accommodate division by zero when C = 0
    oil_ctr_pretax_share: float = field(default=0.25)
    gas_ctr_pretax_share: float = field(default=0.5)

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

    contractor_tax: np.ndarray = field(default=None)

    # Fields/attributes to be defined later
    _oil_lifting: Lifting = field(default=None, init=False, repr=False)
    _gas_lifting: Lifting = field(default=None, init=False, repr=False)
    _oil_tangible: Tangible = field(default=None, init=False, repr=False)
    _gas_tangible: Tangible = field(default=None, init=False, repr=False)
    _oil_intangible: Intangible = field(default=None, init=False, repr=False)
    _gas_intangible: Intangible = field(default=None, init=False, repr=False)
    _oil_opex: OPEX = field(default=None, init=False, repr=False)
    _gas_opex: OPEX = field(default=None, init=False, repr=False)
    _oil_asr: ASR = field(default=None, init=False, repr=False)
    _gas_asr: ASR = field(default=None, init=False, repr=False)

    _oil_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _gas_revenue: np.ndarray = field(default=None, init=False, repr=False)

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

    def _get_FTP(self):
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

    def _get_IC(
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
    def _get_unrecovered_cost(
            depreciation: np.ndarray,
            non_capital: np.ndarray,
            revenue,
            ftp_ctr,
            ftp_gov,
            invest_credit,
    ) -> np.ndarray:

        # unrecovered_cost = np.cumsum(non_capital) - np.cumsum(
        #     revenue - (ftp_ctr + ftp_gov) - invest_credit
        # )

        unrecovered_cost = np.cumsum(depreciation + non_capital) - np.cumsum(
            revenue - (ftp_ctr + ftp_gov) - invest_credit
        )

        return np.where(unrecovered_cost >= 0, unrecovered_cost, 0)

    @staticmethod
    def _get_cost_to_be_recovered(unrecovered_cost: np.ndarray) -> np.ndarray:
        ctr = np.concatenate((np.zeros(1), -np.diff(unrecovered_cost)))
        return np.where(ctr > 0, ctr, 0)

    @staticmethod
    def _get_cost_recovery(revenue: np.ndarray,
                           ftp: np.ndarray,
                           ic: np.ndarray,
                           depreciation: np.ndarray,
                           non_capital: np.ndarray,
                           cost_to_be_recovered: np.ndarray,
                           cr_cap_rate) -> np.ndarray:

        result = np.minimum(revenue - ftp - ic, ((depreciation + non_capital + cost_to_be_recovered) * cr_cap_rate))
        return result
        # There is possibility for a bug in Cap rate
        # return (depreciation + non_capital + cost_to_be_recovered) * cr_cap_rate

    @staticmethod
    def _get_ets_before_transfer(revenue: np.ndarray,
                                 ftp_ctr: np.ndarray,
                                 ftp_gov: np.ndarray,
                                 invest_credit: np.ndarray,
                                 cost_recovery: np.ndarray) -> np.ndarray:
        return revenue - (ftp_ctr + ftp_gov) - invest_credit - cost_recovery

    @staticmethod
    def _get_transfer(
            gas_unrecovered, oil_unrecovered, gas_ets_pretransfer, oil_ets_pretransfer
    ) -> tuple:

        Trf2oil = np.zeros_like(oil_unrecovered)
        Trf2gas = np.zeros_like(gas_unrecovered)

        # Transfer to oil
        combined_condition_oil = np.logical_and(
            np.greater(oil_unrecovered, 0), np.equal(gas_unrecovered, 0)
        )

        oil_indices = np.argwhere(combined_condition_oil)

        if np.size(oil_indices) > 0:
            Trf2oil[oil_indices] = np.minimum(
                gas_ets_pretransfer[oil_indices], oil_unrecovered[oil_indices]
            )

        # Transfer to gas
        combined_condition_gas = np.logical_and(
            np.equal(oil_unrecovered, 0), np.greater(gas_unrecovered, 0)
        )

        indices_gas = np.argwhere(combined_condition_gas)

        if np.size(indices_gas) > 0:
            Trf2gas[indices_gas] = np.minimum(
                oil_ets_pretransfer[indices_gas], gas_unrecovered[indices_gas]
            )

        return Trf2oil, Trf2gas

    @staticmethod
    def _get_ets_after_transfer(ets_before_transfer, trfto, unrecovered_after_transfer):

        ets_after_transfer = np.zeros_like(ets_before_transfer)

        indices = np.equal(unrecovered_after_transfer, 0)

        if np.size(indices) > 0:
            ets_after_transfer[indices] = ets_before_transfer[indices] + trfto[indices]

        return ets_after_transfer

    @staticmethod
    def _get_equity_share(ETS, pretax_ctr):
        r"""
        Use to calculate equity share (ES).

        .. math::

            ES_{ctr \, (t)} &= pretax_ctr \times ETS_{(t)} \\
            ES_{gov \, (t)} &= (1 - pretax_ctr) \times ETS_{(t)}

        Parameters
        ----------
        ETS : np.ndarray
            equity to be shared
        pretax_ctr: float
            pretax contractor

        Returns
        -------
        out: tup
            * ES_ctr : np.ndarray
                Equity share contractor
            * ES_gov : np.ndarray
                Equity share goverment
        """
        contractor_share = pretax_ctr * ETS
        government_share = (1 - pretax_ctr) * ETS

        return contractor_share, government_share

    # @staticmethod
    def _get_dmo(
            self,
            dmo_holiday_duration: int,
            dmo_volume_portion: float,
            dmo_fee_portion: float,
            lifting: Lifting,
            ctr_pretax_share: float,
            unrecovered_cost: np.ndarray,
            is_dmo_end_weighted
    ) -> tuple:

        # DMO end date
        dmo_end_date = self.onstream_date + dateutils.relativedelta(
            months=dmo_holiday_duration
        )

        # Identify position of dmo start year in project years array
        dmo_indices = self.onstream_date.year - self.start_date.year

        # Calculate DMO volume
        dmo_holiday = np.where(self.project_years >= dmo_end_date.year, False, True)
        dmo_volume = dmo_volume_portion * lifting.lifting_rate * ctr_pretax_share
        dmo_fee = np.where(np.logical_or(unrecovered_cost > 0, ~dmo_holiday),
                           dmo_fee_portion * lifting.price * dmo_volume,
                           lifting.price * dmo_volume)
        # Weighted dmo fee condition if the period of dmo is ended in the middle of the year
        if unrecovered_cost[dmo_indices] > 0 and is_dmo_end_weighted:
            dmo_fee[dmo_indices] = dmo_end_date.month / 12 * lifting.price[dmo_indices] * dmo_volume[dmo_indices] + (1 - dmo_end_date.month / 12) * dmo_volume[dmo_indices] * dmo_fee_portion * lifting.price[dmo_indices]

        # Calculate Net DMO
        ddmo = (dmo_volume * lifting.price) - dmo_fee

        return dmo_volume, dmo_fee, ddmo

    def _get_ftp_tax_payment(self, unrec_arr, ftp_arr, tax_rate, ftp_tax_regime=FTPTaxRegime.PDJP_20_2017):
        if ftp_tax_regime == FTPTaxRegime.PDJP_20_2017:
            cum_ftp = np.cumsum(ftp_arr)
            cum_ftp = np.where(cum_ftp > unrec_arr, cum_ftp, 0)
            tax_of_ftp = np.concatenate((np.zeros(1), np.diff(cum_ftp * tax_rate)))
            ftp_tax_payment = np.where(tax_of_ftp > 0, tax_of_ftp, 0)

        else:
            ftp_tax_payment = ftp_arr * tax_rate

        return ftp_tax_payment

    def run(self,
            is_dmo_end_weighted=False,
            ctr_tax: float | np.ndarray = None,
            tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
            tax_rate=0.44, # TODO: Replace with NaN as default value after implementing TaxRegime.
            ftp_tax_regime=FTPTaxRegime.PDJP_20_2017
            ):
        # TODO: Tax rate argument will be deleted then replaced with the value in tax_regime.
        #  Currently, it is used for testing

        self._get_aggregate()
        self._get_revenue()
        self._get_FTP()

        # Depreciation (tangible cost)
        (
            self._oil_depreciation,
            self._oil_undepreciated_asset,
        ) = self._oil_tangible.psc_depreciation_rate()
        (
            self._gas_depreciation,
            self._gas_undepreciated_asset,
        ) = self._gas_tangible.psc_depreciation_rate()

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
        self._oil_ic, self._oil_ic_unrecovered, self._oil_ic_paid = self._get_IC(
            revenue=self._oil_revenue,
            ftp=self._oil_ftp,
            cost_alloc=FluidType.OIL,
            ic_rate=self.oil_ic_rate,
        )

        self._gas_ic, self._gas_ic_unrecovered, self._gas_ic_paid = self._get_IC(
            revenue=self._gas_revenue,
            ftp=self._gas_ftp,
            cost_alloc=FluidType.GAS,
            ic_rate=self.gas_ic_rate,
        )

        # Unrecovered cost before transfer/consolidation
        self._oil_unrecovered_before_transfer = self._get_unrecovered_cost(
            depreciation=self._oil_depreciation,
            non_capital=self._oil_non_capital,
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            invest_credit=self._oil_ic_paid,
        )

        self._gas_unrecovered_before_transfer = self._get_unrecovered_cost(
            depreciation=self._gas_depreciation,
            non_capital=self._gas_non_capital,
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            invest_credit=self._gas_ic_paid,
        )

        # Cost to be recovered
        self._oil_cost_to_be_recovered = self._get_cost_to_be_recovered(
            unrecovered_cost=self._oil_unrecovered_before_transfer,
        )

        self._gas_cost_to_be_recovered = self._get_cost_to_be_recovered(
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
            invest_credit=self._oil_ic_paid,
            cost_recovery=self._oil_cost_recovery,
        )

        self._gas_ets_before_transfer = self._get_ets_before_transfer(
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            invest_credit=self._gas_ic_paid,
            cost_recovery=self._gas_cost_recovery,
        )

        self._transfer_to_oil, self._transfer_to_gas = self._get_transfer(
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
        self._oil_ets_after_transfer = self._get_ets_after_transfer(
            ets_before_transfer=self._oil_ets_before_transfer,
            trfto=self._transfer_to_oil,
            unrecovered_after_transfer=self._oil_unrecovered_after_transfer,
        )

        self._gas_ets_after_transfer = self._get_ets_after_transfer(
            ets_before_transfer=self._gas_ets_before_transfer,
            trfto=self._transfer_to_gas,
            unrecovered_after_transfer=self._gas_unrecovered_after_transfer,
        )

        # ES (Equity Share)
        self._oil_contractor_share, self._oil_government_share = self._get_equity_share(
            ETS=self._oil_ets_after_transfer, pretax_ctr=self.oil_ctr_pretax_share
        )

        self._gas_contractor_share, self._gas_government_share = self._get_equity_share(
            ETS=self._gas_ets_after_transfer, pretax_ctr=self.gas_ctr_pretax_share
        )

        self._oil_dmo_volume, self._oil_dmo_fee, self._oil_ddmo = self._get_dmo(
            dmo_holiday_duration=self.oil_dmo_holiday_duration,
            dmo_volume_portion=self.oil_dmo_volume_portion,
            dmo_fee_portion=self.oil_dmo_fee_portion,
            lifting=self._oil_lifting,
            ctr_pretax_share=self.oil_ctr_pretax_share,
            unrecovered_cost=self._oil_unrecovered_after_transfer,
            is_dmo_end_weighted=is_dmo_end_weighted)

        self._gas_dmo_volume, self._gas_dmo_fee, self._gas_ddmo = self._get_dmo(
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

        self._oil_ftp_tax_payment = self._get_ftp_tax_payment(unrec_arr=self._oil_unrecovered_after_transfer,
                                                              ftp_arr=self._oil_ftp_ctr,
                                                              tax_rate=tax_rate,
                                                              ftp_tax_regime=ftp_tax_regime)

        self._gas_ftp_tax_payment = self._get_ftp_tax_payment(unrec_arr=self._gas_unrecovered_after_transfer,
                                                              ftp_arr=self._gas_ftp_ctr,
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

        # Combined Contractor Take and Contractor Share
        self._ctr_take = self._oil_contractor_take + self._gas_contractor_take
        self._ctr_share = self._oil_contractor_share + self._gas_contractor_share

        # Combined Government Take and Contractor Share
        self._gov_take = self._oil_government_take + self._gas_government_take
        self._gov_share = self._oil_government_share + self._gas_government_share

        # Combined FTP, IC, Cost_Recovery
        self._ftp = self._oil_ftp + self._gas_ftp
        self._ic = self._oil_ic + self._gas_ic
        self._cost_recovery = self._oil_cost_recovery + self._gas_cost_recovery

        return

