"""
Handles calculations associated with PSC Cost Recovery.
"""

from dataclasses import dataclass, field
import numpy as np

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts import psc_tools
from pyscnomics.econ.selection import (
    FluidType,
    InflationAppliedTo,
    TaxRegime,
    FTPTaxRegime,
    TaxSplitTypeCR,
    DeprMethod,
    OtherRevenue,
)


class SunkCostException(Exception):
    """Exception to raise for a misuse of Sunk Cost Method"""

    pass


class CostRecoveryException(Exception):
    """Exception to raise for a misuse of Cost Recovery attributes"""

    pass


@dataclass
class CostRecovery(BaseProject):
    """
    Dataclass that represents Cost Recovery (CR) contract.

    Parameters
    ----------
    oil_ftp_is_available: bool
        The availability of the Oil's First Tranche Petroleum (FTP).
    oil_ftp_is_shared: bool
        The condition whether the Oil FTP will be shared to the contractor or not.
    oil_ftp_portion: float
        The portion of the Oil FTP in fraction.
    gas_ftp_is_available: bool
        The availability of the Gas's First Tranche Petroleum (FTP).
    gas_ftp_is_shared: bool
        The condition whether the Gas FTP will be shared to the contractor or not.
    gas_ftp_portion: float
        The portion of the Gas FTP in fraction.
    tax_split_type: TaxSplitTypeCR
        The type of the contractor split.
    condition_dict: dict
        The condition dictionary of the split when the tax_split_type is not Conventional.
    indicator_rc_icp_sliding: np.ndarray
        The indicator used in the SLIDING_SCALE or R2C tax_split_type.
    oil_ctr_pretax_share: float | np.ndarray
        The Oil Contractor Pre-Tax Split or Oil Contractor Pre-Tax Split Share.
    gas_ctr_pretax_share: float | np.ndarray
        The Gas Contractor Pre-Tax Split or Oil Contractor Pre-Tax Split Share.
    oil_ic_rate: float
        The Oil's Investment Credit (IC) rate of the contract.
    gas_ic_rate: float
        The Gas's Investment Credit (IC) rate of the contract.
    ic_is_available: bool
        The condition whether if IC is available or not.
    oil_cr_cap_rate: float
        The Oil Cost Recovery cap rate.
    gas_cr_cap_rate: float
        The Gas Cost Recovery cap rate.
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

    oil_ftp_is_available: bool = field(default=True)
    oil_ftp_is_shared: bool = field(default=True)
    oil_ftp_portion: float = field(default=0.2)

    gas_ftp_is_available: bool = field(default=True)
    gas_ftp_is_shared: bool = field(default=True)
    gas_ftp_portion: float = field(default=0.2)

    tax_split_type: TaxSplitTypeCR = field(default=TaxSplitTypeCR.CONVENTIONAL)
    condition_dict: dict = field(default_factory=dict)
    indicator_rc_icp_sliding: np.ndarray = field(default=None)
    oil_ctr_pretax_share: float | np.ndarray = field(default=0.25)
    gas_ctr_pretax_share: float | np.ndarray = field(default=0.5)

    oil_ic_rate: float | np.ndarray = field(default=0.0)
    gas_ic_rate: float | np.ndarray = field(default=0.0)
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
    _oil_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)
    _gas_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)

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

    # Consolidated Attributes
    _consolidated_capital_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_intangible_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_opex_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_asr_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_lbt_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_of_sales_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )

    _consolidated_capital_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_intangible_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_opex_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_asr_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_lbt_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_of_sales_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)

    _consolidated_cost_of_sales: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_capital_cost: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_intangible: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_opex: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_asr: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_lbt: np.ndarray = field(default=None, init=False, repr=False)
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

    def _check_attributes(self):
        """
        Function to check the Cost Recovery input.
        -------

        """
        # Defining any attributes that in the form of fraction
        fraction_attributes = (
            ("oil_ftp_portion", self.oil_ftp_portion),
            ("gas_ftp_portion", self.gas_ftp_portion),
            ("oil_ctr_pretax_share", self.oil_ctr_pretax_share),
            ("gas_ctr_pretax_share", self.gas_ctr_pretax_share),
            ("oil_ic_rate", self.oil_ic_rate),
            ("gas_ic_rate", self.gas_ic_rate),
            ("oil_cr_cap_rate", self.oil_cr_cap_rate),
            ("gas_cr_cap_rate", self.gas_cr_cap_rate),
            ("oil_dmo_volume_portion", self.oil_dmo_volume_portion),
            ("oil_dmo_fee_portion", self.oil_dmo_fee_portion),
            ("gas_dmo_volume_portion", self.gas_dmo_volume_portion),
            ("gas_dmo_fee_portion", self.gas_dmo_fee_portion),
        )

        for attr_name, attr in fraction_attributes:
            if attr > 1.0 or attr < 0.0:
                range_type = "exceeding 1.0" if attr > 1.0 else "below 0.0"
                raise CostRecoveryException(
                    f"The {attr_name} value, {attr}, is {range_type}. "
                    f"The allowed range for this attribute is between 0.0 and 1.0."
                )

        discrete_attributes = (
            ("oil_dmo_holiday_duration", self.oil_dmo_holiday_duration),
            ("gas_dmo_holiday_duration", self.gas_dmo_holiday_duration),
        )

        for attr_name, attr in discrete_attributes:
            if attr < 0:
                raise CostRecoveryException(
                    f"The {attr_name} value, {attr}, is below 0. "
                    f"The minimum value for this attribute is 0."
                )

    def _get_rc_icp_pretax(self):
        """
        A Function to get the value of PreTax Split using Revenue over Cost (RC)
        or Indonesian Crude Price (ICP) sliding scale.

        Notes
        -------
        The structure of the dictionary is as the following:
        condition_dict =
        {
            'condition_1': {'bot_limit': 0,
                            'top_limit': 1,
                            'ctr_oil': 0.3137260,
                            'ctr_gas': 0.5498040,
                            'gov_oil': 0.686274,
                            'gov_gas': 0.45098,
                            'split_ctr_oil': 20,
                            'split_ctr_gas': 35,
                            'split_gov_oil': 80,
                            'split_gov_gas': 65},

            'condition_2': {'bot_limit': 1,
                            'top_limit': 1.2,
                            'ctr_oil': 0.27451,
                            'ctr_gas': 0.509804,
                            'gov_oil': 0.72549,
                            'gov_gas': 0.490196,
                            'split_ctr_oil': 17.5,
                            'split_ctr_gas': 32.5,
                            'split_gov_oil': 82.5,
                            'split_gov_gas': 67.5},

            'condition_3': {'bot_limit': 1.2,
                            'top_limit': 1.4,
                            'ctr_oil': 0.235295,
                            'ctr_gas': 0.470589,
                            'gov_oil': 0.764705,
                            'gov_gas': 0.529411,
                            'split_ctr_oil': 15,
                            'split_ctr_gas': 30,
                            'split_gov_oil': 85,
                            'split_gov_gas': 70}
        }

        Returns
        -------
        Modifying the following Cost Recovery attributes:
            self.oil_ctr_pretax_share
            self.gas_ctr_pretax_share

        """
        # Conditioning the given dictionary
        length_condition = len(self.condition_dict[list(self.condition_dict.keys())[0]])
        enum_value = list(range(1, length_condition + 1))
        condition_dict_new = {}
        for index, step in enumerate(enum_value):
            dict_isi = {
                "bot_limit": self.condition_dict["RC Bottom Limit"][index],
                "top_limit": self.condition_dict["RC Top Limit"][index],
                "ctr_oil": self.condition_dict["Pre Tax CTR Oil"][index],
                "ctr_gas": self.condition_dict["Pre Tax CTR Gas"][index],
            }

            key_string = "condition_" + str(step)
            condition_dict_new[key_string] = dict_isi

        self.condition_dict = condition_dict_new

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
        conditions = (bot_limits[:, np.newaxis] < self.indicator_rc_icp_sliding) & (
            self.indicator_rc_icp_sliding <= top_limits[:, np.newaxis]
        )

        # Calculate corresponding values using np.select()
        self.oil_ctr_pretax_share = np.select(
            conditions, ctr_oil_values[:, np.newaxis], default=np.nan
        )
        self.gas_ctr_pretax_share = np.select(
            conditions, ctr_gas_values[:, np.newaxis], default=np.nan
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
        ic_rate: float | np.ndarray,
    ):
        """
        The function to apply the Investment Credit (IC) into the capital cost
        based on the given IC rate.

        Parameters
        ----------
        revenue: np.ndarray
            The array of revenue.
        ftp: np.ndarray
            The array of First Tranche Petroleum (FTP).
        cost_alloc: FluidType
            The cost allocation of the cost.
        ic_rate: float | np.ndarray
            The IC rate which will be applied to the cost.

        Returns
        -------
        out: tuple
        ic_total: The total of the IC
        ic_unrecovered: The unrecoverable IC
        ic_paid: The IC that has been paid
        """

        # Condition where fluid is Oil:
        if cost_alloc == FluidType.GAS:
            capital_class = self._gas_capital_cost
        else:
            capital_class = self._oil_capital_cost

        # Condition when the ic rate is float
        if isinstance(ic_rate, float):
            ic_rate = np.full_like(capital_class.cost, ic_rate)
        else:
            pass

        # Applying the IC calculation to only true value
        ic_arr = np.where(
            np.asarray(capital_class.is_ic_applied) is True,
            capital_class.cost * ic_rate,
            0,
        )

        ic_expenses = np.bincount(
            capital_class.expense_year - capital_class.start_year, weights=ic_arr
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
            Intangible, Operating Expenditures (OPEX) and
            Abandonment Site and Restoration (ASR) Expenditures.
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
        tol = np.full_like(result, fill_value=1.0e-5)

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
        ets_after_transfer = np.where(
            ets_before_tf >= 0,
            revenue
            - (ftp_ctr + ftp_gov)
            - ic
            - cost_recovery
            - transferred_out
            + transferred_in,
            0,
        )

        tol = np.full_like(ets_after_transfer, fill_value=1.0e-5)

        return np.where(ets_after_transfer < tol, 0, ets_after_transfer)

    @staticmethod
    def _get_equity_share(ets, pretax_ctr) -> tuple:
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
    def _get_tax_payment(
        ctr_share: np.ndarray,
        taxable_income: np.ndarray,
        tax_rate: np.ndarray,
        ftp_ctr: np.ndarray,
        unrec_cost: np.ndarray,
        ftp_tax_regime: FTPTaxRegime = FTPTaxRegime.PRE_PDJP_20_2017,
    ) -> np.ndarray:
        """
        The function to get the tax payment of the Cost Recovery contract based on
        the chosen regime.
        There are three existing regime, they are :
            - PRE_PDJP_20_2017
            - PDJP_20_2017
            - DIRECT_MODE

        Notes
        ----------
        [1] PRE_PDJP_20_2017
            In general, the tax will be applied after the contractor share is not zero.
            The taxable object that occurs at the time prior of the applied tax time
            will be accumulated and weighted on the time when the tax is applied.

        [2] PDJP_20_2017
            The tax will be applied when the cumulative First Tranche Petroleum (FTP)
            is greater than the unrecoverable cost

        [3] DIRECT_MODE
            The tax will be applied regardless the cumulative FTP, unrecoverable cost
            and Contractor Share condition.

        Parameters
        ----------
        ctr_share: np.ndarray
            The contractor equity to be split
        taxable_income: np.ndarray
            The contractor's Taxable Income (TI)
        tax_rate: float | np.ndarray
            The applied effective tax rate
        ftp_ctr: np.ndarray
            The contractor's First Tranche Petroleum (FTP)
        unrec_cost: np.ndarray
            The unrecoverable cost
        ftp_tax_regime: FTPTaxRegime
            The regime used to calculate the tax payment

        Returns
        -------
        ctr_tax: np.ndarray
            The contractor's tax payment
        """

        # Tax payment prior to PDJP 2017
        if ftp_tax_regime == FTPTaxRegime.PRE_PDJP_20_2017:
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

                ftp_diff_b4[i] = np.where(
                    ftp_cum_b2[i] > ftp_prior_b3[i], ftp_cum_b2[i] - ftp_prior_b3[i], 0
                )
                ftp_considered_b5[i] = np.where(
                    ftp_diff_b4[i] > unrec_cost[i], ftp_diff_b4[i] - unrec_cost[i], 0
                )

            # Calculating the ftp tax payment
            ftp_tax_paid = ftp_considered_b5 * tax_rate

            # Calculating the tax from taxable income
            ti_tax = (taxable_income - ftp_ctr) * tax_rate

            # Calculating ctr_tax
            ctr_tax = ftp_tax_paid + ti_tax

        else:
            ctr_tax = taxable_income * tax_rate

        return ctr_tax

    @staticmethod
    def _get_tax_payment_pdjp(
        ftp_ctr: np.ndarray,
        unrec_cost: np.ndarray,
        tax_rate: np.ndarray,
        taxable_income: np.ndarray,
        ctr_share: np.ndarray,
        ddmo: np.ndarray,
    ):
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

            ftp_diff_b4[i] = np.where(
                ftp_cum_b2[i] > ftp_prior_b3[i], ftp_cum_b2[i] - ftp_prior_b3[i], 0
            )
            ftp_considered_b5[i] = np.where(
                ftp_diff_b4[i] > unrec_cost[i], ftp_diff_b4[i] - unrec_cost[i], 0
            )

        # Calculating the ftp tax payment
        ftp_tax_paid = ftp_considered_b5 * tax_rate

        # Calculating the taxable income without ftp
        taxable_income_wo_ftp = np.where(
            taxable_income - ftp_ctr < 0,
            taxable_income - ftp_ctr + ddmo,
            taxable_income - ftp_ctr,
        )

        # Defining Taxing flag
        applied_tax = np.where(ctr_share > 0, 1, 0)
        cum_ti = np.cumsum(taxable_income_wo_ftp)

        applied_tax_prior = np.concatenate((np.zeros(1), applied_tax))[0:-1]

        ctr_ets_tax = np.where(
            np.logical_and(applied_tax == 1, applied_tax_prior == 0),
            cum_ti * tax_rate,
            np.where(
                np.logical_and(applied_tax == 1, applied_tax_prior == 1),
                taxable_income_wo_ftp * tax_rate,
                0,
            ),
        )

        # Tax from FTP and Contractor Equity
        ctr_tax = ftp_tax_paid + ctr_ets_tax

        # Contractor Share without FTP
        ets_and_ftp_ctr = ctr_share + ftp_ctr

        # Initiating the array of unpaid_tax and tax_paid
        unpaid_tax = np.zeros_like(taxable_income, dtype=float)
        tax_paid = np.zeros_like(taxable_income, dtype=float)

        for index, i in enumerate(taxable_income):
            if index == 0:
                tax_paid[index] = np.where(
                    taxable_income[index] > ctr_tax[index] + 0,
                    ctr_tax[index] + 0,
                    ets_and_ftp_ctr[index] - ddmo[index],
                )

                unpaid_tax[index] = np.where(
                    ctr_tax[index] + 0 > tax_paid[index],
                    ctr_tax[index] + 0 - tax_paid[index],
                    0,
                )

            else:
                tax_paid[index] = np.where(
                    taxable_income[index] > ctr_tax[index] + unpaid_tax[index - 1],
                    ctr_tax[index] + unpaid_tax[index - 1],
                    ets_and_ftp_ctr[index] - ddmo[index],
                )

                unpaid_tax[index] = np.where(
                    ctr_tax[index] + unpaid_tax[index - 1] > tax_paid[index],
                    ctr_tax[index] + unpaid_tax[index - 1] - tax_paid[index],
                    0,
                )

        return tax_paid, unpaid_tax

    @staticmethod
    def _unpaid_and_tax_balance(tax_payment: np.ndarray, ets_ctr: np.ndarray):
        """
        The function to get the contractor's unpaid tax and tax payment.

        Parameters
        ----------
        tax_payment: np.ndarray
            The contractors tax payment
        ets_ctr: np.ndarray
            The contractor's share

        Returns
        -------
        unpaid_tax: np.ndarray
            The unpaid contractor's tax
        ctr_tax: np.ndarray
            The contractor's tax payment
        """
        unpaid_tax = np.zeros_like(ets_ctr, dtype=float)
        unpaid_tax[0] = 0
        ctr_tax = np.zeros_like(ets_ctr, dtype=float)

        for i in range(1, len(unpaid_tax)):
            unpaid_tax[i] = max(0, unpaid_tax[i - 1] + tax_payment[i] - ets_ctr[i])
            ctr_tax[i] = min(ets_ctr[i], unpaid_tax[i - 1] + tax_payment[i])

        return unpaid_tax, ctr_tax

    # Todo (20 March 2025): Fix the sunk cost calculation method
    # def _get_sunk_cost(self, sunk_cost_reference_year: int):
    #     oil_cost_raw = (
    #             self._oil_capital_expenditures_post_tax
    #             + self._oil_non_capital
    #     )
    #     self._oil_sunk_cost = oil_cost_raw[
    #                           : (sunk_cost_reference_year - self.start_date.year + 1)
    #                           ]
    #     self._oil_sunk_cost = np.concatenate(
    #     (self._oil_sunk_cost, np.zeros(self.project_years[-1] - sunk_cost_reference_year))
    #     )
    #
    #     gas_cost_raw = (
    #             self._gas_capital_expenditures_post_tax
    #             + self._gas_non_capital
    #     )
    #     self._gas_sunk_cost = gas_cost_raw[
    #                           : (sunk_cost_reference_year - self.start_date.year + 1)
    #                           ]
    #     self._gas_sunk_cost = np.concatenate(
    #         (self._gas_sunk_cost, np.zeros(self.project_years[-1] - sunk_cost_reference_year)))
    #
    #     if sunk_cost_reference_year == self.start_date.year:
    #         self._oil_sunk_cost = np.zeros_like(self.project_years)
    #         self._gas_sunk_cost = np.zeros_like(self.project_years)

    def _apply_cost_of_sales(
        self, oil_applied: bool = False, gas_applied: bool = False
    ):
        """
        The function to apply the cost of sales.

        Parameters
        ----------
        oil_applied: bool
            The condition when oil is being applied by cost of sales.
        gas_applied: bool
            The condition when gas is being applied by cost of sales.

        """
        # Condition while the oil cost of sales is applied while there is no oil revenue
        if oil_applied is True and np.sum(self._oil_revenue) <= 0:
            raise CostRecoveryException(
                f"The oil revenue is zero or below zero throughout the project years. "
            )
        # Condition when the oil cost of sales is applied while there are oil revenues
        elif oil_applied is True and np.sum(self._oil_revenue) > 0:
            self._oil_revenue = (
                self._oil_revenue - self._oil_cost_of_sales_expenditures_post_tax
            )
        else:
            pass

        # Condition while the gas cost of sales is applied while there is no gas revenue
        if gas_applied is True and np.sum(self._gas_revenue) <= 0:
            raise CostRecoveryException(
                f"The gas revenue is zero or below zero throughout the project years."
            )
        # Condition when the gas cost of sales is applied while there are gas revenues
        elif gas_applied is True and np.sum(self._gas_revenue) > 0:
            self._gas_revenue = (
                self._gas_revenue - self._gas_cost_of_sales_expenditures_post_tax
            )
        else:
            pass

    def run(
        self,
        sulfur_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        electricity_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_OIL_REVENUE,
        co2_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        is_dmo_end_weighted: bool = False,
        tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
        effective_tax_rate: float | np.ndarray | None = None,
        ftp_tax_regime=FTPTaxRegime.PDJP_20_2017,
        sunk_cost_reference_year: int = None,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        year_inflation: np.ndarray = None,
        vat_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo | None = InflationAppliedTo.CAPEX,
        post_uu_22_year2001: bool = True,
        oil_cost_of_sales_applied: bool = False,
        gas_cost_of_sales_applied: bool = False,
        sum_undepreciated_cost: bool = False,
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

        # Get the WAP Price
        self._get_wap_price()

        # Calculate pre tax expenditures
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
        self._get_other_revenue(
            sulfur_revenue=sulfur_revenue,
            electricity_revenue=electricity_revenue,
            co2_revenue=co2_revenue,
        )

        # Calculate FTP
        self._get_ftp()

        # Condition when the Cost of Sales for Oil is being applied, which
        # will modify oil or gas revenue
        self._apply_cost_of_sales(
            oil_applied=oil_cost_of_sales_applied, gas_applied=gas_cost_of_sales_applied
        )

        # Defining the PreTax Split, whether using conventional PreTax or Sliding
        if self.tax_split_type is not TaxSplitTypeCR.CONVENTIONAL:
            self._get_rc_icp_pretax()

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
            self._oil_undepreciated_asset < 1.0e-5, 0, self._oil_undepreciated_asset
        )
        self._gas_undepreciated_asset = np.where(
            self._gas_undepreciated_asset < 1.0e-5, 0, self._gas_undepreciated_asset
        )

        # Treatment of the un-depreciated asset to be summed up in the last year
        # of the contract or not
        if sum_undepreciated_cost is True:
            self._oil_depreciation[-1] = (
                self._oil_depreciation[-1] + self._oil_undepreciated_asset
            )
            self._gas_depreciation[-1] = (
                self._gas_depreciation[-1] + self._gas_undepreciated_asset
            )

            self._oil_undepreciated_asset = np.zeros_like(
                self.project_years, dtype=float
            )
            self._gas_undepreciated_asset = np.zeros_like(
                self.project_years, dtype=float
            )
        else:
            pass

        # Non-capital costs (intangible + opex + asr)
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

        # Filtering for only the cost that in the bracket of the project years
        self._oil_depreciation = self._oil_depreciation[
            : (self.end_date.year - self.start_date.year + 1)
        ]
        self._gas_depreciation = self._gas_depreciation[
            : (self.end_date.year - self.start_date.year + 1)
        ]

        # Get Sunk Cost
        self._get_sunk_cost(sunk_cost_reference_year)

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

        (
            self._oil_unrecovered_before_transfer,
            self._oil_cost_to_be_recovered,
            self._oil_cost_recovery,
        ) = psc_tools.get_unrec_cost_2b_recovered_costrec(
            project_years=self.project_years,
            depreciation=self._oil_depreciation,
            non_capital=self._oil_non_capital,
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            ic=self._oil_ic_paid,
            cr_cap_rate=self.oil_cr_cap_rate,
        )

        (
            self._gas_unrecovered_before_transfer,
            self._gas_cost_to_be_recovered,
            self._gas_cost_recovery,
        ) = psc_tools.get_unrec_cost_2b_recovered_costrec(
            project_years=self.project_years,
            depreciation=self._gas_depreciation,
            non_capital=self._gas_non_capital,
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            ic=self._gas_ic_paid,
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

        (
            self._oil_unrecovered_after_transfer,
            self._oil_cost_to_be_recovered_after_tf,
            self._oil_cost_recovery_after_tf,
        ) = psc_tools.get_unrec_cost_2b_recovered_costrec(
            project_years=self.project_years,
            depreciation=self._oil_depreciation,
            non_capital=self._oil_non_capital,
            revenue=self._oil_revenue + self._transfer_to_oil,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            ic=self._oil_ic_paid,
            cr_cap_rate=self.oil_cr_cap_rate,
        )

        (
            self._gas_unrecovered_after_transfer,
            self._gas_cost_to_be_recovered_after_tf,
            self._gas_cost_recovery_after_tf,
        ) = psc_tools.get_unrec_cost_2b_recovered_costrec(
            project_years=self.project_years,
            depreciation=self._gas_depreciation,
            non_capital=self._gas_non_capital,
            revenue=self._gas_revenue + self._transfer_to_gas,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            ic=self._gas_ic_paid,
            cr_cap_rate=self.gas_cr_cap_rate,
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
            ets=self._oil_ets_after_transfer,
            ctr_ets=self._oil_contractor_share,
            ctr_ftp=self._oil_ftp_ctr,
            post_uu_22_year2001=post_uu_22_year2001,
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
            ets=self._gas_ets_after_transfer,
            ctr_ets=self._gas_contractor_share,
            ctr_ftp=self._gas_ftp_ctr,
            post_uu_22_year2001=post_uu_22_year2001,
        )

        # Adjusting DDMO for Pre PDJP
        if ftp_tax_regime is FTPTaxRegime.PRE_PDJP_20_2017:
            self._oil_ddmo = np.where(self._oil_contractor_share > 0, self._oil_ddmo, 0)
            self._gas_ddmo = np.where(self._gas_contractor_share > 0, self._gas_ddmo, 0)

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
        if isinstance(effective_tax_rate, float):
            self._tax_rate_arr = np.full_like(
                self.project_years, effective_tax_rate, dtype=float
            )

        # Generating Tax array based on the tax regime if tax_rate argument is None
        elif effective_tax_rate is None:
            self._tax_rate_arr = self._get_tax_by_regime(tax_regime=tax_regime)

        elif isinstance(effective_tax_rate, np.ndarray):
            self._tax_rate_arr = effective_tax_rate

        else:
            self._tax_rate_arr = np.full_like(
                self.project_years, fill_value=0.0, dtype=float
            )

        self._oil_tax_payment = self._get_tax_payment(
            ctr_share=self._oil_contractor_share,
            taxable_income=self._oil_taxable_income,
            tax_rate=self._tax_rate_arr,
            ftp_ctr=self._oil_ftp_ctr,
            unrec_cost=self._oil_unrecovered_after_transfer,
            ftp_tax_regime=ftp_tax_regime,
        )

        self._gas_tax_payment = self._get_tax_payment(
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
            self._oil_capital_expenditures_post_tax + self._oil_non_capital
        )
        self._gas_cashflow = self._gas_contractor_take - (
            self._gas_capital_expenditures_post_tax + self._gas_non_capital
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

        # Returning the gross revenue
        self._oil_revenue = (
            self._oil_revenue + self._oil_cost_of_sales_expenditures_post_tax
        )
        self._gas_revenue = (
            self._gas_revenue + self._gas_cost_of_sales_expenditures_post_tax
        )

        # Consolidated attributes
        self._consolidated_revenue = self._oil_revenue + self._gas_revenue

        self._consolidated_capital_cost = (
            self._oil_capital_expenditures_post_tax
            + self._gas_capital_expenditures_post_tax
        )

        self._consolidated_cost_of_sales = (
            self._oil_cost_of_sales_expenditures_post_tax
            + self._gas_cost_of_sales_expenditures_post_tax
        )

        self._consolidated_intangible = (
            self._oil_intangible_expenditures_post_tax
            + self._gas_intangible_expenditures_post_tax
        )

        self._consolidated_sunk_cost = self._oil_sunk_cost + self._gas_sunk_cost

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

        self._consolidated_depreciation = (
            self._oil_depreciation + self._gas_depreciation
        )

        self._consolidated_undepreciated_asset = (
            self._oil_undepreciated_asset + self._gas_undepreciated_asset
        )

        self._consolidated_capital_expenditures_pre_tax = (
            self._oil_capital_expenditures_pre_tax
            + self._gas_capital_expenditures_pre_tax
        )
        self._consolidated_intangible_expenditures_pre_tax = (
            self._oil_intangible_expenditures_pre_tax
            + self._gas_intangible_expenditures_pre_tax
        )
        self._consolidated_opex_expenditures_pre_tax = (
            self._oil_opex_expenditures_pre_tax + self._gas_opex_expenditures_pre_tax
        )
        self._consolidated_asr_expenditures_pre_tax = (
            self._oil_asr_expenditures_pre_tax + self._gas_asr_expenditures_pre_tax
        )
        self._consolidated_lbt_expenditures_pre_tax = (
            self._oil_lbt_expenditures_pre_tax + self._gas_lbt_expenditures_pre_tax
        )
        self._consolidated_cost_of_sales_expenditures_pre_tax = (
            self._oil_cost_of_sales_expenditures_pre_tax
            + self._gas_cost_of_sales_expenditures_pre_tax
        )
        self._consolidated_expenditures_pre_tax = (
            self._consolidated_capital_expenditures_pre_tax
            + self._consolidated_intangible_expenditures_pre_tax
            + self._consolidated_opex_expenditures_pre_tax
            + self._consolidated_asr_expenditures_pre_tax
            + self._consolidated_lbt_expenditures_pre_tax
            + self._consolidated_cost_of_sales_expenditures_pre_tax
        )

        self._consolidated_capital_indirect_tax = (
            self._oil_capital_indirect_tax + self._gas_capital_indirect_tax
        )
        self._consolidated_intangible_indirect_tax = (
            self._oil_intangible_indirect_tax + self._gas_intangible_indirect_tax
        )
        self._consolidated_opex_indirect_tax = (
            self._oil_opex_indirect_tax + self._gas_opex_indirect_tax
        )
        self._consolidated_asr_indirect_tax = (
            self._oil_asr_indirect_tax + self._gas_asr_indirect_tax
        )
        self._consolidated_lbt_indirect_tax = (
            self._oil_lbt_indirect_tax + self._gas_lbt_indirect_tax
        )
        self._consolidated_cost_of_sales_indirect_tax = (
            self._oil_cost_of_sales_indirect_tax + self._gas_cost_of_sales_indirect_tax
        )
        self._consolidated_indirect_tax = (
            self._oil_total_indirect_tax + self._gas_total_indirect_tax
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

        # Calculating the consolidated tax based on the ftp tax payment regime
        if ftp_tax_regime is FTPTaxRegime.PDJP_20_2017:
            self._consolidated_tax_payment, self._consolidated_unpaid_tax_balance = (
                self._get_tax_payment_pdjp(
                    ftp_ctr=self._oil_ftp_ctr + self._gas_ftp_ctr,
                    unrec_cost=self._oil_unrecovered_after_transfer
                    + self._gas_unrecovered_after_transfer,
                    tax_rate=self._tax_rate_arr,
                    taxable_income=self._consolidated_taxable_income,
                    ctr_share=self._consolidated_contractor_share,
                    ddmo=self._consolidated_ddmo,
                )
            )

        elif ftp_tax_regime is FTPTaxRegime.PRE_PDJP_20_2017:
            self._consolidated_tax_due = self._get_tax_payment(
                ctr_share=self._consolidated_contractor_share,
                taxable_income=self._consolidated_taxable_income,
                tax_rate=self._tax_rate_arr,
                ftp_ctr=self._consolidated_ftp_ctr,
                unrec_cost=self._consolidated_unrecovered_after_transfer,
                ftp_tax_regime=ftp_tax_regime,
            )
            (
                self._consolidated_unpaid_tax_balance,
                self._consolidated_tax_payment,
            ) = self._unpaid_and_tax_balance(
                tax_payment=self._consolidated_tax_due,
                ets_ctr=self._consolidated_contractor_share,
            )

        else:
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
        )

        self._consolidated_government_take = (
            self._consolidated_ftp_gov
            + self._consolidated_government_share
            + self._consolidated_tax_payment
            + self._consolidated_ddmo
        )

        self._consolidated_cashflow = self._consolidated_contractor_take - (
            self._consolidated_capital_cost
            + self._consolidated_non_capital
            + self._consolidated_cost_of_sales
        )
