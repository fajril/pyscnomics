"""
Handles calculations associated with PSC Cost Recovery.
"""

from dataclasses import dataclass, field
import numpy as np
import dateutils
from functools import reduce

from pyscnomics.contracts.project import BaseProject
from pyscnomics.econ.selection import FluidType, YearReference
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.econ.revenue import Lifting


@dataclass
class CostRecovery(BaseProject):
    ftp_is_available: bool = field(default=True)
    ftp_is_shared: bool = field(default=True)

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

    def __post_init__(self):
        self.contractor_tax = np.ones(self.project_duration) * 0.4

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
            if self.ftp_is_shared:
                self._oil_ftp_ctr = self.oil_ctr_pretax_share * self._oil_revenue
                self._gas_ftp_ctr = self.gas_ctr_pretax_share * self._gas_revenue

            self._oil_ftp_gov = self._oil_revenue - self._oil_ftp_ctr
            self._gas_ftp_gov = self._gas_revenue - self._gas_ftp_ctr

        self._oil_ftp = self._oil_ftp_ctr + self._oil_ftp_gov
        self._gas_ftp = self._gas_ftp_ctr + self._gas_ftp_gov

    def _get_IC(
        self, revenue: np.ndarray, ftp: float, cost_alloc: FluidType, ic_rate: float
    ) -> np.ndarray:

        tangible_ic_applied = reduce(
            lambda x, y: x + y,
            (
                i
                for i in self.tangible_cost
                if i.cost_allocation == cost_alloc and i.is_ic_applied
            ),
        )

        ic = ic_rate * tangible_ic_applied.expenditures(year_ref=YearReference.PIS_YEAR)

        ic_unrecovered = np.cumsum(ic) - np.cumsum(revenue - ftp)
        ic_unrecovered = np.where(ic_unrecovered > 0, ic_unrecovered, 0)

        # FIXME: Find an alternative to np.roll()
        ic_unrec_next = np.roll(ic_unrecovered, 1)
        ic_unrec_next[0] = 0

        return np.minimum(revenue - ftp, ic + ic_unrec_next)

    @staticmethod
    def _get_unrecovered_cost(
        depreciation: np.ndarray,
        non_capital: np.ndarray,
        revenue,
        ftp_ctr,
        ftp_gov,
        invest_credit,
    ) -> np.ndarray:

        unrecovered_cost = np.cumsum(depreciation + non_capital) - np.cumsum(
            revenue - (ftp_ctr + ftp_gov) - invest_credit
        )

        return np.where(unrecovered_cost >= 0, unrecovered_cost, 0)

    @staticmethod
    def _get_cost_to_be_recovered(
        depreciation,
        non_capital,
        revenue,
        ftp_ctr,
        ftp_gov,
        invest_credit,
        unrecovered_cost,
    ):

        cost_to_be_recovered = (
            revenue - (ftp_ctr + ftp_gov) - invest_credit - depreciation - non_capital
        )

        return np.where(
            np.logical_and(unrecovered_cost > 0, cost_to_be_recovered > 0),
            cost_to_be_recovered,
            0,
        )

    @staticmethod
    def _get_cost_recovery(
        depreciation, non_capital, cost_to_be_recovered, cr_cap_rate
    ):
        return (depreciation + non_capital + cost_to_be_recovered) * cr_cap_rate

    @staticmethod
    def _get_ets_before_transfer(
        revenue, ftp_ctr, ftp_gov, invest_credit, cost_recovery
    ):
        return revenue - (ftp_ctr + ftp_gov) - invest_credit - cost_recovery

    @staticmethod
    def _get_transfer(
        gas_unrecovered, oil_unrecovered, gas_ets_pretransfer, oil_ets_pretransfer
    ):

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
        ES_ctr: np.ndarray,
        unrecovered_cost: np.ndarray,
    ) -> np.ndarray:

        # Instantiate dmo arrays as zeros
        dmo_volume = np.zeros_like(lifting.revenue())
        dmo_fee = np.zeros_like(lifting.revenue())
        ddmo = np.zeros_like(lifting.revenue())

        # DMO end date
        dmo_end_date = self.onstream_date + dateutils.relativedelta(
            months=dmo_holiday_duration
        )

        # Identify position of dmo start year in project years array
        indices = np.argwhere(lifting.project_years == self.onstream_date.year).ravel()

        if len(indices) > 0:

            # Calculate DMO volume
            dmo_volume = min(
                dmo_volume_portion * ctr_pretax_share * lifting.lifting_rate[indices],
                ES_ctr[indices],
            )

            # Calculate DMO fee
            dmo_discounted_price = np.where(
                unrecovered_cost > 0, dmo_fee_portion * lifting.price, lifting.price
            )
            dmo_fee = dmo_discounted_price * dmo_volume

            # DMO fee correction
            dmo_fee_offset = (
                self.onstream_date.year
                - self.start_date.year
                + int(dmo_holiday_duration / 12)
            )

            if unrecovered_cost[dmo_fee_offset] == 0:
                dmo_fee[dmo_fee_offset] = (dmo_end_date.month / 12.0) * dmo_fee + (
                    (12.0 - dmo_end_date) / 12.0
                ) * dmo_fee

            # Calculate Net DMO
            ddmo = (dmo_volume * lifting.price) - dmo_fee

        return dmo_volume, dmo_fee, ddmo

    def run(self):

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
        self._oil_ic_paid = self._get_IC(
            revenue=self._oil_revenue,
            ftp=self._oil_ftp,
            cost_alloc=FluidType.OIL,
            ic_rate=self.oil_ic_rate,
        )

        self._gas_ic_paid = self._get_IC(
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
            depreciation=self._oil_depreciation,
            non_capital=self._oil_non_capital,
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            invest_credit=self._oil_ic_paid,
            unrecovered_cost=self._oil_unrecovered_before_transfer,
        )

        self._gas_cost_to_be_recovered = self._get_cost_to_be_recovered(
            depreciation=self._gas_depreciation,
            non_capital=self._gas_non_capital,
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            invest_credit=self._gas_ic_paid,
            unrecovered_cost=self._gas_unrecovered_before_transfer,
        )

        # Cost recovery
        self._oil_cost_recovery = self._get_cost_recovery(
            depreciation=self._oil_depreciation,
            non_capital=self._oil_non_capital,
            cost_to_be_recovered=self._oil_cost_to_be_recovered,
            cr_cap_rate=self.oil_cr_cap_rate,
        )

        self._gas_cost_recovery = self._get_cost_recovery(
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

        # DMO
        self._oil_dmo_volume, self._oil_dmo_fee, self._oil_ddmo = self._get_dmo(
            dmo_holiday_duration=self.oil_dmo_holiday_duration,
            dmo_volume_portion=self.oil_dmo_volume_portion,
            dmo_fee_portion=self.oil_dmo_fee_portion,
            lifting=self._oil_lifting,
            ctr_pretax_share=self.oil_ctr_pretax_share,
            ES_ctr=self._oil_contractor_share,
            unrecovered_cost=self._oil_unrecovered_after_transfer,
        )

        self._gas_dmo_volume, self._gas_dmo_fee, self._gas_ddmo = self._get_dmo(
            dmo_holiday_duration=self.gas_dmo_holiday_duration,
            dmo_volume_portion=self.gas_dmo_volume_portion,
            dmo_fee_portion=self.gas_dmo_fee_portion,
            lifting=self._gas_lifting,
            ctr_pretax_share=self.gas_ctr_pretax_share,
            ES_ctr=self._gas_contractor_share,
            unrecovered_cost=self._gas_unrecovered_after_transfer,
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

        # Tax payment
        # FIXME: re-check the value of array contractor_tax
        self._oil_tax_payment = self.contractor_tax * self._oil_taxable_income
        self._gas_tax_payment = self.contractor_tax * self._gas_taxable_income

        # Contractor Take
        self._oil_contractor_take = (
            self._oil_taxable_income - self._oil_tax_payment + self._oil_cost_recovery
        )

        self._gas_contractor_take = (
            self._gas_taxable_income - self._gas_tax_payment + self._gas_cost_recovery
        )

        # Government Take
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
