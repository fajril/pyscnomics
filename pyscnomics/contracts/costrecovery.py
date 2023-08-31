"""
Handles calculations associated with PSC Cost Recovery.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import dateutils
from datetime import datetime
from functools import reduce

from pyscnomics.contracts.project import BaseProject
from pyscnomics.econ.selection import FluidType, TaxSplitTypeCR, YearReference
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

    gas_dmo_volume_portion: float = field(default=1.)
    gas_dmo_fee_portion: float = field(default=1.)
    gas_dmo_holiday_duration: int = field(default=60)

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
        depreciation: np.ndarray, non_capital: np.ndarray, revenue, ftp_ctr, ftp_gov, invest_credit
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
        ES_ctr = pretax_ctr * ETS
        ES_gov = (1 - pretax_ctr) * ETS

        return ES_ctr, ES_gov

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
        dmo_end_date = self.onstream_date + dateutils.relativedelta(months=dmo_holiday_duration)

        # Identify position of dmo start year in project years array
        indices = np.argwhere(lifting.project_years == self.onstream_date.year).ravel()

        if len(indices) > 0:

            # Calculate DMO volume
            dmo_volume = min(
                dmo_volume_portion * ctr_pretax_share * lifting.lifting_rate[indices],
                ES_ctr[indices]
            )

            # Calculate DMO fee
            dmo_discounted_price = np.where(
                unrecovered_cost > 0, dmo_fee_portion * lifting.price, lifting.price
            )
            dmo_fee = dmo_discounted_price * dmo_volume

            # DMO fee correction
            dmo_fee_offset = self.onstream_date.year - self.start_date.year + int(dmo_holiday_duration / 12)

            if unrecovered_cost[dmo_fee_offset] == 0:
                dmo_fee[dmo_fee_offset] = \
                    (dmo_end_date.month / 12.) * dmo_fee + ((12. - dmo_end_date) / 12.) * dmo_fee

            # Calculate Net DMO
            ddmo = (dmo_volume * lifting.price) - dmo_fee

        return dmo_volume, dmo_fee, ddmo

    @staticmethod
    def _get_taxable_income(self, contractor_share, ddmo):
        r"""
        Use to calculate taxable income (TI).

        .. math::

            TI_{(t)} = CS_{(t)} - DDMO_{(t)}

        Parameters
        ----------
        CS : np.ndarray
            contractor share
        DDMO : np.ndarray
            differential domestic market obligation

        Return
        ------
        TI : np.ndarray
            taxable income
        """
        return CS - DDMO

    def _get_tax1(self, TI, FTP_con, UR):
        r"""
        Use to calculate tax (tax).

        .. math::

            Tax_{(t)} = \left\{\begin{matrix}
                0, & \sum_{i=0}^{t} FTP_{con} > UR_{(t)} \\
                \text{tax rate} \times TI_{(t)}, & \sum_{i=0}^{t} FTP_{con} \leq UR_{(t)}
                \end{matrix}\right.

        Parameters
        ----------
        TI: np.ndarray
            taxable income
        FTP_con : np.ndarray
            contractor FTP
        UR : np.ndarray
            unrecovered

        Return
        ------
        tax: float
            contractor tax
        """
        tax = np.zeros_like(TI)

        # Calculate cumulative FTP Contractor
        cum_FTP_con = np.cumsum(FTP_con)

        # Calculate tax where Cum FTP_con <= UR
        indices = np.argwhere(cum_FTP_con <= UR)
        tax[indices] = self.tax_ef * TI[indices]

        return tax

    def _get_tax2(self, TI, ES_con):
        r"""
        Use to calculate tax (tax).

        .. math::



        Parameters
        ----------
        TI: np.ndarray
            taxable income
        ES_con : np.ndarray
            equity to be share contractor FTP

        Returns
        -------
        out: tup
            * tax_paid: float
                paid contractor tax
            * tax_carryforward: float
                carryforward contractor tax
        """
        TI_cumsum = np.cumsum(TI)

        calc_tax = np.zeros_like(ES_con)
        tax_paid = np.zeros_like(ES_con)
        tax_carryforward = np.zeros_like(ES_con)

        for idx, ES_val in enumerate(ES_con):
            if idx > 0:
                if ES_con[idx - 1] == 0 and ES_val > 0:
                    calc_tax[idx] = self.tax_ef * TI_cumsum[idx]
                elif ES_con[idx - 1] > 0 and ES_val > 0:
                    calc_tax[idx] = self.tax_ef * TI[idx]

                # Calculate tax total
                tax_total = calc_tax[idx] + tax_carryforward[idx - 1]

                # Calculate tax paid
                tax_paid[idx] = min(tax_total, ES_con[idx])

                # Calculate tax payable
                tax_carryforward[idx] = tax_total - tax_paid[idx]

        return tax_paid, tax_carryforward

    def run(self):

        self._get_aggregate()
        self._get_revenue()
        self._get_FTP()

        # Depreciation (tangible cost)
        oil_depreciation, self._oil_undepreciated_asset = self._oil_tangible.psc_depreciation_rate()
        gas_depreciation, self._gas_undepreciated_asset = self._gas_tangible.psc_depreciation_rate()

        # Non-capital costs (intangible + opex + asr)
        oil_non_capital = (
            self._oil_intangible.expenditures()
            + self._oil_opex.expenditures()
            + self._oil_asr.expenditures()
        )

        gas_non_capital = (
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
        oil_unrecovered_before_transfer = self._get_unrecovered_cost(
            depreciation=oil_depreciation,
            non_capital=oil_non_capital,
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            invest_credit=self._oil_ic_paid,
        )

        gas_unrecovered_before_transfer = self._get_unrecovered_cost(
            depreciation=gas_depreciation,
            non_capital=gas_non_capital,
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            invest_credit=self._gas_ic_paid,
        )

        # Cost to be recovered
        oil_cost_to_be_recovered = self._get_cost_to_be_recovered(
            depreciation=oil_depreciation,
            non_capital=oil_non_capital,
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            invest_credit=self._oil_ic_paid,
            unrecovered_cost=oil_unrecovered_before_transfer,
        )

        gas_cost_to_be_recovered = self._get_cost_to_be_recovered(
            depreciation=gas_depreciation,
            non_capital=gas_non_capital,
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            invest_credit=self._gas_ic_paid,
            unrecovered_cost=gas_unrecovered_before_transfer,
        )

        # Cost recovery
        oil_cost_recovery = self._get_cost_recovery(
            depreciation=oil_depreciation,
            non_capital=oil_non_capital,
            cost_to_be_recovered=oil_cost_to_be_recovered,
            cr_cap_rate=self.oil_cr_cap_rate,
        )

        gas_cost_recovery = self._get_cost_recovery(
            depreciation=gas_depreciation,
            non_capital=gas_non_capital,
            cost_to_be_recovered=gas_cost_to_be_recovered,
            cr_cap_rate=self.gas_cr_cap_rate,
        )

        # ETS (Equity to be Split) before transfer/consolidation
        oil_ets_before_transfer = self._get_ets_before_transfer(
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            invest_credit=self._oil_ic_paid,
            cost_recovery=oil_cost_recovery,
        )

        gas_ets_before_transfer = self._get_ets_before_transfer(
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            invest_credit=self._gas_ic_paid,
            cost_recovery=gas_cost_recovery,
        )

        Trf2oil, Trf2gas = self._get_transfer(
            gas_unrecovered=gas_unrecovered_before_transfer,
            oil_unrecovered=oil_unrecovered_before_transfer,
            gas_ets_pretransfer=gas_ets_before_transfer,
            oil_ets_pretransfer=oil_ets_before_transfer,
        )

        # Unrecovered cost after transfer/consolidation
        oil_unrecovered_after_transfer = oil_unrecovered_before_transfer - Trf2oil
        gas_unrecovered_after_transfer = gas_unrecovered_before_transfer - Trf2gas

        # ETS (Equity to be Split) after transfer/consolidation
        oil_ets_after_transfer = self._get_ets_after_transfer(
            ets_before_transfer=oil_ets_before_transfer,
            trfto=Trf2oil,
            unrecovered_after_transfer=oil_unrecovered_after_transfer,
        )

        gas_ets_after_transfer = self._get_ets_after_transfer(
            ets_before_transfer=gas_ets_before_transfer,
            trfto=Trf2gas,
            unrecovered_after_transfer=gas_unrecovered_after_transfer,
        )

        # ES (Equity Share)
        oil_es_ctr, oil_es_gov = self._get_equity_share(
            ETS=oil_ets_after_transfer, pretax_ctr=self.oil_ctr_pretax_share
        )

        gas_es_ctr, gas_es_gov = self._get_equity_share(
            ETS=gas_ets_after_transfer, pretax_ctr=self.gas_ctr_pretax_share
        )

        # CS (Contractor Share) and GS (Government Share)
        # FIXME: Oil and Gas IC
        oil_ctr_share = self._oil_ftp_ctr + oil_es_ctr  # + oil_ic_paid
        gas_ctr_share = self._gas_ftp_ctr + gas_es_ctr  # + gas_ic_paid
        oil_gov_share = self._oil_ftp_gov + oil_es_gov
        gas_gov_share = self._gas_ftp_gov + gas_es_gov

        # DMO
        dmo_oil = self._get_dmo(
            dmo_holiday_duration=self.oil_dmo_holiday_duration,
            dmo_volume_portion=self.oil_dmo_volume_portion,
            dmo_fee_portion=self.oil_dmo_fee_portion,
            lifting=self._oil_lifting,
            ctr_pretax_share=self.oil_ctr_pretax_share,
            ES_ctr=oil_es_ctr,
            unrecovered_cost=oil_unrecovered_after_transfer
        )

        dmo_gas = self._get_dmo(
            dmo_holiday_duration=self.gas_dmo_holiday_duration,
            dmo_volume_portion=self.gas_dmo_volume_portion,
            dmo_fee_portion=self.gas_dmo_fee_portion,
            lifting=self._gas_lifting,
            ctr_pretax_share=self.gas_ctr_pretax_share,
            ES_ctr=gas_es_ctr,
            unrecovered_cost=gas_unrecovered_after_transfer
        )



    def _get_NCS(self, TI, Tax):
        r"""
        Use to calculate net contractor share (NCS).

        .. math::

            NCS_{(t)} = TI_{(t)} - Tax_{(t)}

        Parameters
        ----------
        TI : np.ndarray
            taxable income
        Tax : np.ndarray
            tax

        Return
        ------
        NCS : np.ndarray
            net contractor share
        """
        return TI - Tax

    def _get_CT(self, NCS, CR):
        r"""
        Use to calculate contractor take (CT).

        .. math::

            CT_{(t)} = NCS_{(t)} + CR_{(t)}

        Parameters
        ----------
        NCS : np.ndarray
            net contractor share
        CR : np.ndarray
            cost recovery

        Return
        ------
        CT : np.ndarray
            contractor take
        """
        return NCS + CR

    # FIXME: Update mengikuti struktur data terbaru
    def _get_Expenditure(self, capex_tangible, capex_intangible, opex):
        r"""
        Use to calculate Expenditure or Cash Out.

        .. math::
            Expenditure_{(t)} = \text{CAPEX}_{tangible \, (t)} +
                                \text{CAPEX}_{intangible \, (t)} +
                                \text{OPEX}_{(t)}

        Parameters
        ----------
        capex_tangible : np.ndarray
            capex tangible
        capex_intangible : np.ndarray
            capex intangible
        opex : np.ndarray
            operating expenditure

        Returns
        -------
        expenditure : np.ndarray
            expenditure
        """
        return capex_tangible + capex_intangible + opex

    def _get_CF(self, CT, Expenditure):
        r"""
        Use to calculate Cashflow (CF).

        .. math::

            CF_{(t)} = CT_{(t)} - Expenditure_{(t)}

        Parameters
        ----------
        CT : np.ndarray
            Contractor Take
        Expenditure : np.ndarray
            expenditure or cash out

        Return
        ------
        CF : np.ndarray
            cashflow
        """
        return CT - Expenditure

    def _get_GOI_Take(self, GS, DDMO, Tax):
        r"""
        Use to calculate Goverment of Indonesia Take (GOI Take).

        .. math::

            GOI_{take \, (t)} = GS_{(t)} + DDMO_{(t)} + Tax_{(t)}

        Parameters
        ----------
        GS : np.ndarray
            goverment share
        DDMO : np.ndarray
            differential domestic market obligation
        Tax : np.ndarray
            tax

        Return
        ------
        GOI_take: np.ndarray
            goverment of indonesia Take
        """
        return GS + DDMO + Tax

    # def run_BeforeTransfer(self, depr_capex, undepr_capex, intangible, sunk_cost):
    def run_BeforeTransfer(self):
        """

        Parameters
        ----------
        depr_capex : TYPE
            DESCRIPTION.
        undepr_capex : TYPE
            DESCRIPTION.
        intangible : TYPE
            DESCRIPTION.
        sunk_cost : TYPE
            DESCRIPTION.
        capex_tangible : TYPE
            DESCRIPTION.
        capex_intangible : TYPE
            DESCRIPTION.
        year_arr : TYPE
            DESCRIPTION.

        Raise
        -----
        Exception
            if the fluid type is not valid.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        # Calculate Revenue
        revenue = self.lifting.revenue()

        # Calculate FTP (Depend on Phase)
        self.FTP_con, self.FTP_gov, FTP = self._get_FTP(revenue)

        # Calculate Revenue after FTP (Depend on Phase)
        REV_after_FTP = self._get_REV_after_FTP(revenue, FTP)

        # Calculate IC (Depend on Phase)
        IC = self._get_IC()

        # Calculate IC Paid and IC UnRec
        self.IC_paid, IC_unrec = self._get_ICpaid_ICunrec(IC, REV_after_FTP)

        # Calculate Revenue after FTP & IC
        REV_after_IC = self._get_REV_after_IC(REV_after_FTP, self.IC_paid)

        # Calculate Cost Recovery Allocation (Depend on Phase)
        CR_allocation = self._get_CR_allocation(REV_after_IC)

        # Calculate OPEX (Depend on Phase)
        self.OPEX = self.cost_obj.get_opex()

        # Calculate CR
        CR = self._get_CR(depr_capex, undepr_capex, self.OPEX, intangible, sunk_cost)

        # Calculate CTR
        self.CTR, self.UR_BeforeTransfer = self._get_CTR_UR(CR, CR_allocation)

        # Calculate ETS PreTransfer
        self.ETS_BeforeTransfer = self._get_ETS_PreTransfer(REV_after_IC, self.CTR)

        # Generate dataframe
        col_lst = [
            "FTP Con",
            "FTP Gov",
            "FTP",
            "REV After FTP",
            "IC",
            "IC paid",
            "IC Unrec",
            "REV After IC",
            "CR Allocation",
            "OPEX",
            "CR",
            "CTR",
            "UR BeforeTransfer",
            "ETS BeforeTransfer",
        ]

        res_arr = np.column_stack(
            (
                self.FTP_con,
                self.FTP_gov,
                FTP,
                REV_after_FTP,
                IC,
                self.IC_paid,
                IC_unrec,
                REV_after_IC,
                CR_allocation,
                self.OPEX,
                CR,
                self.CTR,
                self.UR_BeforeTransfer,
                self.ETS_BeforeTransfer,
            )
        )

        res_df = pd.DataFrame(data=res_arr, columns=col_lst)

        return res_df

    def run_Transfer(self, gas_result_df, oil_result_df):
        # Calculate Transfer
        UR_oil = oil_result_df["UR BeforeTransfer"].to_numpy()
        UR_gas = gas_result_df["UR BeforeTransfer"].to_numpy()
        ETS_oil = oil_result_df["ETS BeforeTransfer"].to_numpy()
        ETS_gas = gas_result_df["ETS BeforeTransfer"].to_numpy()

        oil_result_df["Transfer To"], gas_result_df["Transfer To"] = self._get_Transfer(
            UR_gas, UR_oil, ETS_gas, ETS_oil
        )

        return gas_result_df, oil_result_df

    def run_AfterTransfer(self, capex_tangible, capex_intangible, year_arr, result_df):

        TransferTo = result_df["Transfer To"]

        # Calculate UR After Transfer
        UR_AfterTransfer = self._get_UR_After_Transfer(
            self.UR_BeforeTransfer, TransferTo
        )

        # Calculate ETS After Transfer
        ETS_AfterTransfer = self._get_ETS_After_Transfer(
            self.ETS_BeforeTransfer, TransferTo, UR_AfterTransfer
        )

        # Calculate ES
        ES_con, ES_gov = self._get_ES(ETS_AfterTransfer)

        # Calculate CS
        CS = self._get_CS(self.FTP_con, ES_con, self.IC_paid)

        # Calculate GS
        GS = self._get_GS(self.FTP_gov, ES_gov)

        # Calculate DMO
        DMO = self._get_DMO(ES_con, year_arr)

        # Calculate DMO Fee
        DMO_Fee = self._get_DMO_Fee(DMO)

        # Calculate DDMO
        DDMO = self._get_DDMO(DMO, DMO_Fee)

        # Calculate TI
        TI = self._get_TI(CS, DDMO)

        # Calculate Tax
        Tax_paid, Tax_carryforward = self._get_tax2(TI, ES_con)

        # Calculate NCS
        NCS = self._get_NCS(TI, Tax_paid)

        # Calculate CT
        CT = self._get_CT(NCS, self.CTR)

        # Calculate Expenditure
        Expenditure = self._get_Expenditure(capex_tangible, capex_intangible, self.OPEX)

        # Calculate CF
        CF = self._get_CF(CT, Expenditure)

        # Calculate GOI Take
        GOI_take = self._get_GOI_Take(GS, DDMO, Tax_paid)

        # Generate dataframe
        col_lst = [
            "UR AfterTransfer",
            "ETS AfterTransfer",
            "ES Con",
            "ES Gov",
            "CS",
            "GS",
            "DMO",
            "DMO Fee",
            "DDMO",
            "TI",
            "Tax Paid",
            "Tax Carryforward",
            "NCS",
            "CT",
            "Expenditure",
            "CF",
            "GOI Take",
        ]

        res_arr = np.column_stack(
            (
                UR_AfterTransfer,
                ETS_AfterTransfer,
                ES_con,
                ES_gov,
                CS,
                GS,
                DMO,
                DMO_Fee,
                DDMO,
                TI,
                Tax_paid,
                Tax_carryforward,
                NCS,
                CT,
                Expenditure,
                CF,
                GOI_take,
            )
        )

        res_df = pd.DataFrame(data=res_arr, columns=col_lst)

        result_df = result_df.join(res_df)

        return result_df

    def run(self):
        return

    def get_result(self):
        return


# class CostRecoveryCombine(object):
#     def __init__(self, cr_gas, cr_oil):
#         self.cr_gas = cr_gas
#         self.cr_oil = cr_oil

#     def run(self):
#         cr_phase_lst = [self.cr_gas, self.cr_oil]


#         result = {}

#         # Calculate CR Before Transfer
#         for cr_obj in cr_phase_lst:
#             result[cr_obj.phase] = \
#                 cr_obj.run_BeforeTransfer(depr_capex, undepr_capex, intangible,
#                                           sunk_cost)

#         # Calculate CR Transfer
#         result[self.cr_gas.phase], result[self.cr_oil.phase] = \
#             self.cr_gas.run_Transfer(result[self.cr_gas.phase],
#                                      result[self.cr_oil.phase])

#         # Calculate CR After Transfer
#         for cr_obj in cr_phase_lst:
#             result[cr_obj.phase] = \
#                 cr_obj.run_AfterTransfer(capex_tangible, capex_intangible,
#                                          year_arr, result[cr_obj.phase])

#         # Calculate CR Combined
#         combined_df = pd.DataFrame()

#         for col_name in result[self.cr_gas.phase].columns:
#                 combined_df[col_name] = result[self.cr_gas.phase][col_name] \
#                                         + result[self.cr_oil.phase][col_name]

#         result["Combined"] = combined_df

#         return result


if __name__ == "__main__":
    from pyscnomics.datasets import load_data

    data_CR = load_data("CR_Gas")
