"""
Python Script to parse data from Spreadsheet
"""

from dataclasses import dataclass, field

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.io.aggregator import Aggregate


class ParsingException(Exception):
    """ Exception to be raised for a misuse of InitiateContract class """

    pass


@dataclass
class InitiateContract:
    """
    Class for initializing Production Sharing Contract (PSC) configurations.

    Parameters
    ----------
    workbook_path: str
        The path to the workbook containing relevant data.

    Attributes
    ----------
    summary_arguments: dict
        Arguments for summary calculations.
    psc: Union[CostRecovery, GrossSplit, dict]
        Type of Production Sharing Contract: CostRecovery, GrossSplit, or Transition.
    psc_arguments: dict
        Arguments for the Production Sharing Contract.
    contract_type: str
        Type of contract.
    data: Aggregate
        Object containing aggregated data.
    transition: Transition
        An instance of Transition class for PSC transition case.
    transition_arguments: dict
        Arguments for PSC transition case.
    """
    workbook_path: str = field(default=None)

    # Attributes to be defined later
    summary_arguments: dict = field(default=None, init=False, repr=False)
    psc: CostRecovery | GrossSplit | dict = field(default=None, init=False, repr=False)
    psc_arguments: dict = field(default=None, init=False, repr=False)
    contract_type: str = field(default=None, init=False, repr=False)
    data: Aggregate = field(default=None, init=False, repr=False)
    transition: Transition = field(default=None, init=False, repr=False)
    transition_arguments: dict = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Prepare attribute workbook_path
        if self.workbook_path is None:
            self.workbook_path = "Workbook.xlsb"

        # Prepare attribute data
        self.data = Aggregate(workbook_to_read=self.workbook_path)
        self.data.fit()

        # Prepare attribute summary_arguments
        self.summary_arguments = {
            "reference_year": self.data.general_config_data.discount_rate_start_year,
            "inflation_rate": self.data.fiscal_config_data.inflation_rate,
            "discount_rate": self.data.general_config_data.discount_rate,
            "npv_mode": self.data.fiscal_config_data.npv_mode,
            "discounting_mode": self.data.fiscal_config_data.discounting_mode,
        }

        # Prepare attribute contract type
        self.contract_type = self.data.general_config_data.type_of_contract

    def _get_single_contract_project(self) -> tuple:
        """
        Get details for a single Production Sharing Contract with BaseProject.

        Returns
        -------
        psc: BaseProject
            Object representing the BaseProject contract.
        psc_arguments: dict
            Arguments specific to the Cost Recovery contract.
        summary_arguments: dict
            Arguments for the contract summary.
        data: Aggregate
            The arranged data collected as an instance of Aggregate class.

        Notes
        -----
        This method creates an instance of BaseProject class.
        It populates the attributes of BaseProject object with relevant data.
        """
        # Prepare total lifting data
        lifting_total = (
            self.data.oil_lifting_aggregate_total
            + self.data.gas_lifting_aggregate_total
            + self.data.sulfur_lifting_aggregate
            + self.data.electricity_lifting_aggregate
            + self.data.co2_lifting_aggregate
        )

        # Create an instance of Project
        self.psc = BaseProject(
            start_date=self.data.general_config_data.start_date_project,
            end_date=self.data.general_config_data.end_date_project,
            oil_onstream_date=self.data.general_config_data.oil_onstream_date,
            gas_onstream_date=self.data.general_config_data.gas_onstream_date,
            lifting=lifting_total,
            tangible_cost=self.data.tangible_cost_aggregate,
            intangible_cost=self.data.intangible_cost_aggregate,
            opex=self.data.opex_aggregate,
            asr_cost=self.data.asr_cost_aggregate,
        )

        # Specify arguments for Project
        self.psc_arguments = {
            "sulfur_revenue": self.data.fiscal_config_data.sulfur_revenue_config,
            "electricity_revenue": self.data.fiscal_config_data.electricity_revenue_config,
            "co2_revenue": self.data.fiscal_config_data.co2_revenue_config,
            "sunk_cost_reference_year": self.data.fiscal_config_data.sunk_cost_reference_year,
            # "year_ref": None,
            # "tax_type": None,
            "vat_rate": self.data.fiscal_config_data.vat_rate,
            "lbt_rate": self.data.fiscal_config_data.lbt_rate,
            "inflation_rate": self.data.fiscal_config_data.inflation_rate,
            "future_rate": float(self.data.fiscal_config_data.asr_future_rate),
            "inflation_rate_applied_to": self.data.general_config_data.inflation_rate_applied_to,
        }

        # Filling the summary contract argument
        self.summary_arguments["contract"] = self.psc

        return self.psc, self.psc_arguments, self.summary_arguments, self.data

    def _get_single_contract_cr(self) -> tuple:
        """
        Get details for a single Production Sharing Contract with Cost Recovery (PSC CR).

        Returns
        -------
        psc: CostRecovery
            Object representing the Cost Recovery contract.
        psc_arguments: dict
            Arguments specific to the Cost Recovery contract.
        summary_arguments: dict
            Arguments for the contract summary.
        data: Aggregate
            The arranged data collected as an instance of Aggregate class.

        Notes
        -----
        This method creates an instance of the PSC CostRecovery class.
        It populates the attributes of PSC CostRecovery object with relevant data.
        """
        # Prepare total lifting data
        lifting_total = (
            self.data.oil_lifting_aggregate_total
            + self.data.gas_lifting_aggregate_total
            + self.data.sulfur_lifting_aggregate
            + self.data.electricity_lifting_aggregate
            + self.data.co2_lifting_aggregate
        )

        # Create an instance of PSC CR
        self.psc = CostRecovery(
            start_date=self.data.general_config_data.start_date_project,
            end_date=self.data.general_config_data.end_date_project,
            oil_onstream_date=self.data.general_config_data.oil_onstream_date,
            gas_onstream_date=self.data.general_config_data.gas_onstream_date,
            lifting=lifting_total,
            tangible_cost=self.data.tangible_cost_aggregate,
            intangible_cost=self.data.intangible_cost_aggregate,
            opex=self.data.opex_aggregate,
            asr_cost=self.data.asr_cost_aggregate,
            oil_ftp_is_available=self.data.psc_cr_data.oil_ftp_availability,
            oil_ftp_is_shared=self.data.psc_cr_data.oil_ftp_is_shared,
            oil_ftp_portion=self.data.psc_cr_data.oil_ftp_portion,
            gas_ftp_is_available=self.data.psc_cr_data.gas_ftp_availability,
            gas_ftp_is_shared=self.data.psc_cr_data.gas_ftp_is_shared,
            gas_ftp_portion=self.data.psc_cr_data.gas_ftp_portion,
            tax_split_type=self.data.psc_cr_data.split_type,
            condition_dict=self.data.psc_cr_data.icp_sliding_scale,
            indicator_rc_icp_sliding=self.data.psc_cr_data.indicator_rc_split_sliding_scale,
            oil_ctr_pretax_share=self.data.psc_cr_data.oil_ctr_pretax,
            gas_ctr_pretax_share=self.data.psc_cr_data.gas_ctr_pretax,
            oil_ic_rate=self.data.psc_cr_data.ic_oil,
            gas_ic_rate=self.data.psc_cr_data.ic_gas,
            ic_is_available=self.data.psc_cr_data.ic_availability,
            oil_cr_cap_rate=self.data.psc_cr_data.oil_cr_cap_rate,
            gas_cr_cap_rate=self.data.psc_cr_data.gas_cr_cap_rate,
            oil_dmo_volume_portion=self.data.psc_cr_data.oil_dmo_volume,
            oil_dmo_fee_portion=self.data.psc_cr_data.oil_dmo_fee,
            oil_dmo_holiday_duration=self.data.psc_cr_data.oil_dmo_period,
            gas_dmo_volume_portion=self.data.psc_cr_data.gas_dmo_volume,
            gas_dmo_fee_portion=self.data.psc_cr_data.gas_dmo_fee,
            gas_dmo_holiday_duration=self.data.psc_cr_data.gas_dmo_period,
        )

        # Specify arguments for PSC CR
        self.psc_arguments = {
            "sulfur_revenue": self.data.fiscal_config_data.sulfur_revenue_config,
            "electricity_revenue": self.data.fiscal_config_data.electricity_revenue_config,
            "co2_revenue": self.data.fiscal_config_data.co2_revenue_config,
            "is_dmo_end_weighted": self.data.psc_cr_data.dmo_is_weighted,
            "tax_regime": self.data.fiscal_config_data.tax_mode,
            "tax_rate": self.data.fiscal_config_data.tax_rate,
            "ftp_tax_regime": self.data.fiscal_config_data.tax_payment_config,
            "sunk_cost_reference_year": self.data.fiscal_config_data.sunk_cost_reference_year,
            "depr_method": self.data.fiscal_config_data.depreciation_method,
            "decline_factor": self.data.fiscal_config_data.decline_factor,
            "vat_rate": self.data.fiscal_config_data.vat_rate,
            "lbt_rate": self.data.fiscal_config_data.lbt_rate,
            "inflation_rate": self.data.fiscal_config_data.inflation_rate,
            "future_rate": float(self.data.fiscal_config_data.asr_future_rate),
            "inflation_rate_applied_to": self.data.general_config_data.inflation_rate_applied_to,
            # "year_ref": None,
            # "tax_type": None,
        }

        # Filling the summary contract argument
        self.summary_arguments["contract"] = self.psc

        return self.psc, self.psc_arguments, self.summary_arguments, self.data

    def _get_single_contract_gs(self) -> tuple:
        """
        Get details for a single Production Sharing Contract with Gross Split (PSC GS).

        Returns
        -------
        psc: CostRecovery
            Object representing the Gross Split contract.
        psc_arguments: dict
            Arguments specific to the Gross Split contract.
        summary_arguments: dict
            Arguments for the contract summary.
        data: Aggregate
            The arranged data collected as an instance of Aggregate class.

        Notes
        -----
        This method creates an instance of the PSC GrossSplit class.
        It populates the attributes of PSC GrossSplit object with relevant data.
        """
        # Prepare total lifting data
        lifting_total = (
            self.data.oil_lifting_aggregate_total
            + self.data.gas_lifting_aggregate_total
            + self.data.sulfur_lifting_aggregate
            + self.data.electricity_lifting_aggregate
            + self.data.co2_lifting_aggregate
        )

        # Create an instance of PSC GS
        self.psc = GrossSplit(
            start_date=self.data.general_config_data.start_date_project,
            end_date=self.data.general_config_data.end_date_project,
            oil_onstream_date=self.data.general_config_data.oil_onstream_date,
            gas_onstream_date=self.data.general_config_data.gas_onstream_date,
            lifting=lifting_total,
            tangible_cost=self.data.tangible_cost_aggregate,
            intangible_cost=self.data.intangible_cost_aggregate,
            opex=self.data.opex_aggregate,
            asr_cost=self.data.asr_cost_aggregate,
            field_status=self.data.psc_gs_data.field_status,
            field_loc=self.data.psc_gs_data.field_location,
            res_depth=self.data.psc_gs_data.reservoir_depth,
            infra_avail=self.data.psc_gs_data.infrastructure_availability,
            res_type=self.data.psc_gs_data.reservoir_type,
            api_oil=self.data.psc_gs_data.oil_api,
            domestic_use=self.data.psc_gs_data.domestic_content_use,
            prod_stage=self.data.psc_gs_data.production_stage,
            co2_content=self.data.psc_gs_data.co2_content,
            h2s_content=self.data.psc_gs_data.h2s_content,
            base_split_ctr_oil=self.data.psc_gs_data.oil_base_split,
            base_split_ctr_gas=self.data.psc_gs_data.gas_base_split,
            split_ministry_disc=self.data.psc_gs_data.ministry_discretion_split,
            oil_dmo_volume_portion=self.data.psc_gs_data.oil_dmo_volume,
            oil_dmo_fee_portion=self.data.psc_gs_data.oil_dmo_fee,
            oil_dmo_holiday_duration=self.data.psc_gs_data.oil_dmo_period,
            gas_dmo_volume_portion=self.data.psc_gs_data.gas_dmo_volume,
            gas_dmo_fee_portion=self.data.psc_gs_data.gas_dmo_fee,
            gas_dmo_holiday_duration=self.data.psc_gs_data.gas_dmo_period,
        )

        # Specify arguments for PSC GS
        self.psc_arguments = {
            "sulfur_revenue": self.data.fiscal_config_data.sulfur_revenue_config,
            "electricity_revenue": self.data.fiscal_config_data.electricity_revenue_config,
            "co2_revenue": self.data.fiscal_config_data.co2_revenue_config,
            "is_dmo_end_weighted": self.data.psc_gs_data.dmo_is_weighted,
            "tax_regime": self.data.fiscal_config_data.tax_mode,
            "tax_rate": self.data.fiscal_config_data.tax_rate,
            "sunk_cost_reference_year": self.data.fiscal_config_data.sunk_cost_reference_year,
            "depr_method": self.data.fiscal_config_data.depreciation_method,
            "decline_factor": self.data.fiscal_config_data.decline_factor,
            "vat_rate": self.data.fiscal_config_data.vat_rate,
            "lbt_rate": self.data.fiscal_config_data.lbt_rate,
            "inflation_rate": self.data.fiscal_config_data.inflation_rate,
            "future_rate": float(self.data.fiscal_config_data.asr_future_rate),
            "inflation_rate_applied_to": self.data.general_config_data.inflation_rate_applied_to,
            # "regime": None,
            # "year_ref": None,
            # "tax_type": None,
        }

        # Filling the summary contract argument
        self.summary_arguments["contract"] = self.psc

        return self.psc, self.psc_arguments, self.summary_arguments, self.data

    def _get_transition_contract_cr_to_cr(self) -> tuple:
        """
        Get details for a transition Production Sharing Contract from
        Cost Recovery to Cost Recovery (CR-CR).

        Returns
        -------
        transition: Transition
            Object representing the transition from Cost Recovery to Cost Recovery.
        transition_arguments: dict
            Arguments specific to the transition.
        summary_arguments: dict
            Arguments for the transition summary.
        data: Aggregate
            The arranged data collected as an instance of Aggregate class.

        Notes
        -----
        This method defines configurations for PSC transition CR to CR. It creates
        instances and populates the attributes of CostRecovery objects for both cases,
        respectively.
        """
        # Define keys for transition CR-CR
        keys = ["Cost Recovery Config", "Cost Recovery Config (2)"]

        # Prepare total lifting data
        lifting_total = {
            psc: (
                self.data.oil_lifting_aggregate_total[psc]
                + self.data.gas_lifting_aggregate_total[psc]
                + self.data.sulfur_lifting_aggregate[psc]
                + self.data.electricity_lifting_aggregate[psc]
                + self.data.co2_lifting_aggregate[psc]
            )
            for psc in ["PSC 1", "PSC 2"]
        }

        # Create an instance of PSC transition CR-CR
        self.psc = {
            "PSC 1": CostRecovery(
                start_date=self.data.general_config_data.start_date_project,
                end_date=self.data.general_config_data.end_date_project,
                oil_onstream_date=self.data.general_config_data.oil_onstream_date,
                gas_onstream_date=self.data.general_config_data.gas_onstream_date,
                lifting=lifting_total["PSC 1"],
                tangible_cost=self.data.tangible_cost_aggregate["PSC 1"],
                intangible_cost=self.data.intangible_cost_aggregate["PSC 1"],
                opex=self.data.opex_aggregate["PSC 1"],
                asr_cost=self.data.asr_cost_aggregate["PSC 1"],
                oil_ftp_is_available=self.data.psc_transition_cr_to_cr[keys[0]].oil_ftp_availability,
                oil_ftp_is_shared=self.data.psc_transition_cr_to_cr[keys[0]].oil_ftp_is_shared,
                oil_ftp_portion=self.data.psc_transition_cr_to_cr[keys[0]].oil_ftp_portion,
                gas_ftp_is_available=self.data.psc_transition_cr_to_cr[keys[0]].gas_ftp_availability,
                gas_ftp_is_shared=self.data.psc_transition_cr_to_cr[keys[0]].gas_ftp_is_shared,
                gas_ftp_portion=self.data.psc_transition_cr_to_cr[keys[0]].gas_ftp_portion,
                tax_split_type=self.data.psc_transition_cr_to_cr[keys[0]].split_type,
                condition_dict=self.data.psc_transition_cr_to_cr[keys[0]].icp_sliding_scale,
                indicator_rc_icp_sliding=(
                    self.data.psc_transition_cr_to_cr[keys[0]].indicator_rc_split_sliding_scale
                ),
                oil_ctr_pretax_share=self.data.psc_transition_cr_to_cr[keys[0]].oil_ctr_pretax,
                gas_ctr_pretax_share=self.data.psc_transition_cr_to_cr[keys[0]].gas_ctr_pretax,
                oil_ic_rate=self.data.psc_transition_cr_to_cr[keys[0]].ic_oil,
                gas_ic_rate=self.data.psc_transition_cr_to_cr[keys[0]].ic_gas,
                ic_is_available=self.data.psc_transition_cr_to_cr[keys[0]].ic_availability,
                oil_cr_cap_rate=self.data.psc_transition_cr_to_cr[keys[0]].oil_cr_cap_rate,
                gas_cr_cap_rate=self.data.psc_transition_cr_to_cr[keys[0]].gas_cr_cap_rate,
                oil_dmo_volume_portion=self.data.psc_transition_cr_to_cr[keys[0]].oil_dmo_volume,
                oil_dmo_fee_portion=self.data.psc_transition_cr_to_cr[keys[0]].oil_dmo_fee,
                oil_dmo_holiday_duration=self.data.psc_transition_cr_to_cr[keys[0]].oil_dmo_period,
                gas_dmo_volume_portion=self.data.psc_transition_cr_to_cr[keys[0]].gas_dmo_volume,
                gas_dmo_fee_portion=self.data.psc_transition_cr_to_cr[keys[0]].gas_dmo_fee,
                gas_dmo_holiday_duration=self.data.psc_transition_cr_to_cr[keys[0]].gas_dmo_period,
            ),
            "PSC 2": CostRecovery(
                start_date=self.data.general_config_data.start_date_project_second,
                end_date=self.data.general_config_data.end_date_project_second,
                oil_onstream_date=self.data.general_config_data.oil_onstream_date_second,
                gas_onstream_date=self.data.general_config_data.gas_onstream_date_second,
                lifting=lifting_total["PSC 2"],
                tangible_cost=self.data.tangible_cost_aggregate["PSC 2"],
                intangible_cost=self.data.intangible_cost_aggregate["PSC 2"],
                opex=self.data.opex_aggregate["PSC 2"],
                asr_cost=self.data.asr_cost_aggregate["PSC 2"],
                oil_ftp_is_available=self.data.psc_transition_cr_to_cr[keys[1]].oil_ftp_availability,
                oil_ftp_is_shared=self.data.psc_transition_cr_to_cr[keys[1]].oil_ftp_is_shared,
                oil_ftp_portion=self.data.psc_transition_cr_to_cr[keys[1]].oil_ftp_portion,
                gas_ftp_is_available=self.data.psc_transition_cr_to_cr[keys[1]].gas_ftp_availability,
                gas_ftp_is_shared=self.data.psc_transition_cr_to_cr[keys[1]].gas_ftp_is_shared,
                gas_ftp_portion=self.data.psc_transition_cr_to_cr[keys[1]].gas_ftp_portion,
                tax_split_type=self.data.psc_transition_cr_to_cr[keys[1]].split_type,
                condition_dict=self.data.psc_transition_cr_to_cr[keys[1]].icp_sliding_scale,
                indicator_rc_icp_sliding=(
                    self.data.psc_transition_cr_to_cr[keys[1]].indicator_rc_split_sliding_scale
                ),
                oil_ctr_pretax_share=self.data.psc_transition_cr_to_cr[keys[1]].oil_ctr_pretax,
                gas_ctr_pretax_share=self.data.psc_transition_cr_to_cr[keys[1]].gas_ctr_pretax,
                oil_ic_rate=self.data.psc_transition_cr_to_cr[keys[1]].ic_oil,
                gas_ic_rate=self.data.psc_transition_cr_to_cr[keys[1]].ic_gas,
                ic_is_available=self.data.psc_transition_cr_to_cr[keys[1]].ic_availability,
                oil_cr_cap_rate=self.data.psc_transition_cr_to_cr[keys[1]].oil_cr_cap_rate,
                gas_cr_cap_rate=self.data.psc_transition_cr_to_cr[keys[1]].gas_cr_cap_rate,
                oil_dmo_volume_portion=self.data.psc_transition_cr_to_cr[keys[1]].oil_dmo_volume,
                oil_dmo_fee_portion=self.data.psc_transition_cr_to_cr[keys[1]].oil_dmo_fee,
                oil_dmo_holiday_duration=self.data.psc_transition_cr_to_cr[keys[1]].oil_dmo_period,
                gas_dmo_volume_portion=self.data.psc_transition_cr_to_cr[keys[1]].gas_dmo_volume,
                gas_dmo_fee_portion=self.data.psc_transition_cr_to_cr[keys[1]].gas_dmo_fee,
                gas_dmo_holiday_duration=self.data.psc_transition_cr_to_cr[keys[1]].gas_dmo_period,
            ),
        }

        # Specify arguments for PSC transition CR-CR
        self.psc_arguments = {
            "PSC 1": {
                "sulfur_revenue": self.data.fiscal_config_data.sulfur_revenue_config["PSC 1"],
                "electricity_revenue": self.data.fiscal_config_data.electricity_revenue_config["PSC 1"],
                "co2_revenue": self.data.fiscal_config_data.co2_revenue_config["PSC 1"],
                "is_dmo_end_weighted": self.data.psc_transition_cr_to_cr[keys[0]].dmo_is_weighted,
                "tax_regime": self.data.fiscal_config_data.tax_mode["PSC 1"],
                "tax_rate": self.data.fiscal_config_data.tax_rate["PSC 1"],
                "ftp_tax_regime": self.data.fiscal_config_data.tax_payment_config["PSC 1"],
                "sunk_cost_reference_year": self.data.fiscal_config_data.sunk_cost_reference_year,
                "depr_method": self.data.fiscal_config_data.depreciation_method["PSC 1"],
                "decline_factor": self.data.fiscal_config_data.decline_factor["PSC 1"],
                "vat_rate": self.data.fiscal_config_data.vat_rate["PSC 1"],
                "lbt_rate": self.data.fiscal_config_data.lbt_rate["PSC 1"],
                "inflation_rate": self.data.fiscal_config_data.inflation_rate["PSC 1"],
                "future_rate": float(self.data.fiscal_config_data.asr_future_rate["PSC 1"]),
                "inflation_rate_applied_to": self.data.general_config_data.inflation_rate_applied_to,
                # "year_ref": None,
                # "tax_type": None,
            },
            "PSC 2": {
                "sulfur_revenue": self.data.fiscal_config_data.sulfur_revenue_config["PSC 2"],
                "electricity_revenue": self.data.fiscal_config_data.electricity_revenue_config["PSC 2"],
                "co2_revenue": self.data.fiscal_config_data.co2_revenue_config["PSC 2"],
                "is_dmo_end_weighted": self.data.psc_transition_cr_to_cr[keys[1]].dmo_is_weighted,
                "tax_regime": self.data.fiscal_config_data.tax_mode["PSC 2"],
                "tax_rate": self.data.fiscal_config_data.tax_rate["PSC 2"],
                "ftp_tax_regime": self.data.fiscal_config_data.tax_payment_config["PSC 2"],
                "sunk_cost_reference_year": self.data.fiscal_config_data.sunk_cost_reference_year,
                "depr_method": self.data.fiscal_config_data.depreciation_method["PSC 2"],
                "decline_factor": self.data.fiscal_config_data.decline_factor["PSC 2"],
                "vat_rate": self.data.fiscal_config_data.vat_rate["PSC 2"],
                "lbt_rate": self.data.fiscal_config_data.lbt_rate["PSC 2"],
                "inflation_rate": self.data.fiscal_config_data.inflation_rate["PSC 2"],
                "future_rate": float(self.data.fiscal_config_data.asr_future_rate["PSC 2"]),
                "inflation_rate_applied_to": self.data.general_config_data.inflation_rate_applied_to,
                # "year_ref": None,
                # "tax_type": None,
            },
        }

        # Create an instance of transition object
        self.transition = Transition(
            contract1=self.psc["PSC 1"],
            contract2=self.psc["PSC 2"],
            argument_contract1=self.psc_arguments["PSC 1"],
            argument_contract2=self.psc_arguments["PSC 2"],
        )

        self.summary_arguments["contract"] = self.transition
        self.transition_arguments = {
            "unrec_portion": self.data.fiscal_config_data.transferred_unrec_cost
        }

        return self.transition, self.transition_arguments, self.summary_arguments, self.data

    def _get_transition_contract_cr_to_gs(self) -> tuple:
        """
        Get details for a transition Production Sharing Contract from
        Cost Recovery to Gross Split (CR-GS).

        Returns
        -------
        transition: Transition
            Object representing the transition from Cost Recovery to Gross Split.
        transition_arguments: dict
            Arguments specific to the transition.
        summary_arguments: dict
            Arguments for the transition summary.
        data: Aggregate
            The arranged data collected as an instance of Aggregate class.

        Notes
        -----
        This method defines configurations for PSC transition CR to GS. It creates
        instances and populates the attributes of CostRecovery and GrossSplit objects,
        respectively.
        """
        # Define keys for transition CR-GS
        keys = ["Cost Recovery Config", "Gross Split Config"]

        # Prepare total lifting data
        lifting_total = {
            psc: (
                self.data.oil_lifting_aggregate_total[psc]
                + self.data.gas_lifting_aggregate_total[psc]
                + self.data.sulfur_lifting_aggregate[psc]
                + self.data.electricity_lifting_aggregate[psc]
                + self.data.co2_lifting_aggregate[psc]
            )
            for psc in ["PSC 1", "PSC 2"]
        }

        # Create an instance of PSC transition CR-GS
        self.psc = {
            "PSC 1": CostRecovery(
                start_date=self.data.general_config_data.start_date_project,
                end_date=self.data.general_config_data.end_date_project,
                oil_onstream_date=self.data.general_config_data.oil_onstream_date,
                gas_onstream_date=self.data.general_config_data.gas_onstream_date,
                lifting=lifting_total["PSC 1"],
                tangible_cost=self.data.tangible_cost_aggregate["PSC 1"],
                intangible_cost=self.data.intangible_cost_aggregate["PSC 1"],
                opex=self.data.opex_aggregate["PSC 1"],
                asr_cost=self.data.asr_cost_aggregate["PSC 1"],
                oil_ftp_is_available=self.data.psc_transition_cr_to_gs[keys[0]].oil_ftp_availability,
                oil_ftp_is_shared=self.data.psc_transition_cr_to_gs[keys[0]].oil_ftp_is_shared,
                oil_ftp_portion=self.data.psc_transition_cr_to_gs[keys[0]].oil_ftp_portion,
                gas_ftp_is_available=self.data.psc_transition_cr_to_gs[keys[0]].gas_ftp_availability,
                gas_ftp_is_shared=self.data.psc_transition_cr_to_gs[keys[0]].gas_ftp_is_shared,
                gas_ftp_portion=self.data.psc_transition_cr_to_gs[keys[0]].gas_ftp_portion,
                tax_split_type=self.data.psc_transition_cr_to_gs[keys[0]].split_type,
                condition_dict=self.data.psc_transition_cr_to_gs[keys[0]].icp_sliding_scale,
                indicator_rc_icp_sliding=(
                    self.data.psc_transition_cr_to_gs[keys[0]].indicator_rc_split_sliding_scale
                ),
                oil_ctr_pretax_share=self.data.psc_transition_cr_to_gs[keys[0]].oil_ctr_pretax,
                gas_ctr_pretax_share=self.data.psc_transition_cr_to_gs[keys[0]].gas_ctr_pretax,
                oil_ic_rate=self.data.psc_transition_cr_to_gs[keys[0]].ic_oil,
                gas_ic_rate=self.data.psc_transition_cr_to_gs[keys[0]].ic_gas,
                ic_is_available=self.data.psc_transition_cr_to_gs[keys[0]].ic_availability,
                oil_cr_cap_rate=self.data.psc_transition_cr_to_gs[keys[0]].oil_cr_cap_rate,
                gas_cr_cap_rate=self.data.psc_transition_cr_to_gs[keys[0]].gas_cr_cap_rate,
                oil_dmo_volume_portion=self.data.psc_transition_cr_to_gs[keys[0]].oil_dmo_volume,
                oil_dmo_fee_portion=self.data.psc_transition_cr_to_gs[keys[0]].oil_dmo_fee,
                oil_dmo_holiday_duration=self.data.psc_transition_cr_to_gs[keys[0]].oil_dmo_period,
                gas_dmo_volume_portion=self.data.psc_transition_cr_to_gs[keys[0]].gas_dmo_volume,
                gas_dmo_fee_portion=self.data.psc_transition_cr_to_gs[keys[0]].gas_dmo_fee,
                gas_dmo_holiday_duration=self.data.psc_transition_cr_to_gs[keys[0]].gas_dmo_period,
            ),
            "PSC 2": GrossSplit(
                start_date=self.data.general_config_data.start_date_project_second,
                end_date=self.data.general_config_data.end_date_project_second,
                oil_onstream_date=self.data.general_config_data.oil_onstream_date_second,
                gas_onstream_date=self.data.general_config_data.gas_onstream_date_second,
                lifting=lifting_total["PSC 2"],
                tangible_cost=self.data.tangible_cost_aggregate["PSC 2"],
                intangible_cost=self.data.intangible_cost_aggregate["PSC 2"],
                opex=self.data.opex_aggregate["PSC 2"],
                asr_cost=self.data.asr_cost_aggregate["PSC 2"],
                field_status=self.data.psc_transition_cr_to_gs[keys[1]].field_status,
                field_loc=self.data.psc_transition_cr_to_gs[keys[1]].field_location,
                res_depth=self.data.psc_transition_cr_to_gs[keys[1]].reservoir_depth,
                infra_avail=self.data.psc_transition_cr_to_gs[keys[1]].infrastructure_availability,
                res_type=self.data.psc_transition_cr_to_gs[keys[1]].reservoir_type,
                api_oil=self.data.psc_transition_cr_to_gs[keys[1]].oil_api,
                domestic_use=self.data.psc_transition_cr_to_gs[keys[1]].domestic_content_use,
                prod_stage=self.data.psc_transition_cr_to_gs[keys[1]].production_stage,
                co2_content=self.data.psc_transition_cr_to_gs[keys[1]].co2_content,
                h2s_content=self.data.psc_transition_cr_to_gs[keys[1]].h2s_content,
                base_split_ctr_oil=self.data.psc_transition_cr_to_gs[keys[1]].oil_base_split,
                base_split_ctr_gas=self.data.psc_transition_cr_to_gs[keys[1]].gas_base_split,
                split_ministry_disc=self.data.psc_transition_cr_to_gs[keys[1]].ministry_discretion_split,
                oil_dmo_volume_portion=self.data.psc_transition_cr_to_gs[keys[1]].oil_dmo_volume,
                oil_dmo_fee_portion=self.data.psc_transition_cr_to_gs[keys[1]].oil_dmo_fee,
                oil_dmo_holiday_duration=self.data.psc_transition_cr_to_gs[keys[1]].oil_dmo_period,
                gas_dmo_volume_portion=self.data.psc_transition_cr_to_gs[keys[1]].gas_dmo_volume,
                gas_dmo_fee_portion=self.data.psc_transition_cr_to_gs[keys[1]].gas_dmo_fee,
                gas_dmo_holiday_duration=self.data.psc_transition_cr_to_gs[keys[1]].gas_dmo_period,
            ),
        }

        # Specify arguments for PSC transition CR-GS
        self.psc_arguments = {
            "PSC 1": {
                "sulfur_revenue": self.data.fiscal_config_data.sulfur_revenue_config["PSC 1"],
                "electricity_revenue": self.data.fiscal_config_data.electricity_revenue_config["PSC 1"],
                "co2_revenue": self.data.fiscal_config_data.co2_revenue_config["PSC 1"],
                "is_dmo_end_weighted": self.data.psc_transition_cr_to_gs[keys[0]].dmo_is_weighted,
                "tax_regime": self.data.fiscal_config_data.tax_mode["PSC 1"],
                "tax_rate": self.data.fiscal_config_data.tax_rate["PSC 1"],
                "ftp_tax_regime": self.data.fiscal_config_data.tax_payment_config["PSC 1"],
                "sunk_cost_reference_year": self.data.fiscal_config_data.sunk_cost_reference_year,
                "depr_method": self.data.fiscal_config_data.depreciation_method["PSC 1"],
                "decline_factor": self.data.fiscal_config_data.decline_factor["PSC 1"],
                "vat_rate": self.data.fiscal_config_data.vat_rate["PSC 1"],
                "lbt_rate": self.data.fiscal_config_data.lbt_rate["PSC 1"],
                "inflation_rate": self.data.fiscal_config_data.inflation_rate["PSC 1"],
                "future_rate": float(self.data.fiscal_config_data.asr_future_rate["PSC 1"]),
                "inflation_rate_applied_to": self.data.general_config_data.inflation_rate_applied_to,
                # "year_ref": None,
                # "tax_type": None,
            },
            "PSC 2": {
                "sulfur_revenue": self.data.fiscal_config_data.sulfur_revenue_config["PSC 2"],
                "electricity_revenue": self.data.fiscal_config_data.electricity_revenue_config["PSC 2"],
                "co2_revenue": self.data.fiscal_config_data.co2_revenue_config["PSC 2"],
                "is_dmo_end_weighted": self.data.psc_transition_cr_to_gs[keys[1]].dmo_is_weighted,
                "tax_regime": self.data.fiscal_config_data.tax_mode["PSC 2"],
                "tax_rate": self.data.fiscal_config_data.tax_rate["PSC 2"],
                "sunk_cost_reference_year": self.data.fiscal_config_data.sunk_cost_reference_year,
                "depr_method": self.data.fiscal_config_data.depreciation_method["PSC 2"],
                "decline_factor": self.data.fiscal_config_data.decline_factor["PSC 2"],
                "vat_rate": self.data.fiscal_config_data.vat_rate["PSC 2"],
                "lbt_rate": self.data.fiscal_config_data.lbt_rate["PSC 2"],
                "inflation_rate": self.data.fiscal_config_data.inflation_rate["PSC 2"],
                "future_rate": float(self.data.fiscal_config_data.asr_future_rate["PSC 2"]),
                "inflation_rate_applied_to": self.data.general_config_data.inflation_rate_applied_to,
                # "regime": None,
                # "year_ref": None,
                # "tax_type": None,
            }
        }

        # Create an instance of transition object
        self.transition = Transition(
            contract1=self.psc["PSC 1"],
            contract2=self.psc["PSC 2"],
            argument_contract1=self.psc_arguments["PSC 1"],
            argument_contract2=self.psc_arguments["PSC 2"],
        )

        self.summary_arguments["contract"] = self.transition
        self.transition_arguments = {
            "unrec_portion": self.data.fiscal_config_data.transferred_unrec_cost
        }

        return self.transition, self.transition_arguments, self.summary_arguments, self.data

    def _get_transition_contract_gs_to_gs(self) -> tuple:
        """
        Get details for a transition Production Sharing Contract from
        Gross Split to Gross Split (GS-GS).

        Returns
        -------
        transition: Transition
            Object representing the transition from Gross Split to Gross Split.
        transition_arguments: dict
            Arguments specific to the transition.
        summary_arguments: dict
            Arguments for the transition summary.
        data: Aggregate
            The arranged data collected as an instance of Aggregate class.

        Notes
        -----
        This method defines configurations for PSC transition GS to GS. It creates
        instances and populates the attributes of GrossSplit objects for both cases,
        respectively.
        """
        # Define keys for transition GS-GS
        keys = ["Gross Split Config", "Gross Split Config (2)"]

        # Prepare total lifting data
        lifting_total = {
            psc: (
                self.data.oil_lifting_aggregate_total[psc]
                + self.data.gas_lifting_aggregate_total[psc]
                + self.data.sulfur_lifting_aggregate[psc]
                + self.data.electricity_lifting_aggregate[psc]
                + self.data.co2_lifting_aggregate[psc]
            )
            for psc in ["PSC 1", "PSC 2"]
        }

        # Create an instance of PSC transition GS-GS
        self.psc = {
            "PSC 1": GrossSplit(
                start_date=self.data.general_config_data.start_date_project,
                end_date=self.data.general_config_data.end_date_project,
                oil_onstream_date=self.data.general_config_data.oil_onstream_date,
                gas_onstream_date=self.data.general_config_data.gas_onstream_date,
                lifting=lifting_total["PSC 1"],
                tangible_cost=self.data.tangible_cost_aggregate["PSC 1"],
                intangible_cost=self.data.intangible_cost_aggregate["PSC 1"],
                opex=self.data.opex_aggregate["PSC 1"],
                asr_cost=self.data.asr_cost_aggregate["PSC 1"],
                field_status=self.data.psc_transition_gs_to_gs[keys[0]].field_status,
                field_loc=self.data.psc_transition_gs_to_gs[keys[0]].field_location,
                res_depth=self.data.psc_transition_gs_to_gs[keys[0]].reservoir_depth,
                infra_avail=self.data.psc_transition_gs_to_gs[keys[0]].infrastructure_availability,
                res_type=self.data.psc_transition_gs_to_gs[keys[0]].reservoir_type,
                api_oil=self.data.psc_transition_gs_to_gs[keys[0]].oil_api,
                domestic_use=self.data.psc_transition_gs_to_gs[keys[0]].domestic_content_use,
                prod_stage=self.data.psc_transition_gs_to_gs[keys[0]].production_stage,
                co2_content=self.data.psc_transition_gs_to_gs[keys[0]].co2_content,
                h2s_content=self.data.psc_transition_gs_to_gs[keys[0]].h2s_content,
                base_split_ctr_oil=self.data.psc_transition_gs_to_gs[keys[0]].oil_base_split,
                base_split_ctr_gas=self.data.psc_transition_gs_to_gs[keys[0]].gas_base_split,
                split_ministry_disc=self.data.psc_transition_gs_to_gs[keys[0]].ministry_discretion_split,
                oil_dmo_volume_portion=self.data.psc_transition_gs_to_gs[keys[0]].oil_dmo_volume,
                oil_dmo_fee_portion=self.data.psc_transition_gs_to_gs[keys[0]].oil_dmo_fee,
                oil_dmo_holiday_duration=self.data.psc_transition_gs_to_gs[keys[0]].oil_dmo_period,
                gas_dmo_volume_portion=self.data.psc_transition_gs_to_gs[keys[0]].gas_dmo_volume,
                gas_dmo_fee_portion=self.data.psc_transition_gs_to_gs[keys[0]].gas_dmo_fee,
                gas_dmo_holiday_duration=self.data.psc_transition_gs_to_gs[keys[0]].gas_dmo_period,
            ),
            "PSC 2": GrossSplit(
                start_date=self.data.general_config_data.start_date_project_second,
                end_date=self.data.general_config_data.end_date_project_second,
                oil_onstream_date=self.data.general_config_data.oil_onstream_date_second,
                gas_onstream_date=self.data.general_config_data.gas_onstream_date_second,
                lifting=lifting_total["PSC 2"],
                tangible_cost=self.data.tangible_cost_aggregate["PSC 2"],
                intangible_cost=self.data.intangible_cost_aggregate["PSC 2"],
                opex=self.data.opex_aggregate["PSC 2"],
                asr_cost=self.data.asr_cost_aggregate["PSC 2"],
                field_status=self.data.psc_transition_gs_to_gs[keys[1]].field_status,
                field_loc=self.data.psc_transition_gs_to_gs[keys[1]].field_location,
                res_depth=self.data.psc_transition_gs_to_gs[keys[1]].reservoir_depth,
                infra_avail=self.data.psc_transition_gs_to_gs[keys[1]].infrastructure_availability,
                res_type=self.data.psc_transition_gs_to_gs[keys[1]].reservoir_type,
                api_oil=self.data.psc_transition_gs_to_gs[keys[1]].oil_api,
                domestic_use=self.data.psc_transition_gs_to_gs[keys[1]].domestic_content_use,
                prod_stage=self.data.psc_transition_gs_to_gs[keys[1]].production_stage,
                co2_content=self.data.psc_transition_gs_to_gs[keys[1]].co2_content,
                h2s_content=self.data.psc_transition_gs_to_gs[keys[1]].h2s_content,
                base_split_ctr_oil=self.data.psc_transition_gs_to_gs[keys[1]].oil_base_split,
                base_split_ctr_gas=self.data.psc_transition_gs_to_gs[keys[1]].gas_base_split,
                split_ministry_disc=self.data.psc_transition_gs_to_gs[keys[1]].ministry_discretion_split,
                oil_dmo_volume_portion=self.data.psc_transition_gs_to_gs[keys[1]].oil_dmo_volume,
                oil_dmo_fee_portion=self.data.psc_transition_gs_to_gs[keys[1]].oil_dmo_fee,
                oil_dmo_holiday_duration=self.data.psc_transition_gs_to_gs[keys[1]].oil_dmo_period,
                gas_dmo_volume_portion=self.data.psc_transition_gs_to_gs[keys[1]].gas_dmo_volume,
                gas_dmo_fee_portion=self.data.psc_transition_gs_to_gs[keys[1]].gas_dmo_fee,
                gas_dmo_holiday_duration=self.data.psc_transition_gs_to_gs[keys[1]].gas_dmo_period,
            ),
        }

        # Specify arguments for PSC transition GS-GS
        self.psc_arguments = {
            "PSC 1": {
                "sulfur_revenue": self.data.fiscal_config_data.sulfur_revenue_config["PSC 1"],
                "electricity_revenue": self.data.fiscal_config_data.electricity_revenue_config["PSC 1"],
                "co2_revenue": self.data.fiscal_config_data.co2_revenue_config["PSC 1"],
                "is_dmo_end_weighted": self.data.psc_transition_gs_to_gs[keys[0]].dmo_is_weighted,
                "tax_regime": self.data.fiscal_config_data.tax_mode["PSC 1"],
                "tax_rate": self.data.fiscal_config_data.tax_rate["PSC 1"],
                "sunk_cost_reference_year": self.data.fiscal_config_data.sunk_cost_reference_year,
                "depr_method": self.data.fiscal_config_data.depreciation_method["PSC 1"],
                "decline_factor": self.data.fiscal_config_data.decline_factor["PSC 1"],
                "vat_rate": self.data.fiscal_config_data.vat_rate["PSC 1"],
                "lbt_rate": self.data.fiscal_config_data.lbt_rate["PSC 1"],
                "inflation_rate": self.data.fiscal_config_data.inflation_rate["PSC 1"],
                "future_rate": float(self.data.fiscal_config_data.asr_future_rate["PSC 1"]) ,
                "inflation_rate_applied_to": self.data.general_config_data.inflation_rate_applied_to,
                # "regime": None,
                # "year_ref": None,
                # "tax_type": None,
            },
            "PSC 2": {
                "sulfur_revenue": self.data.fiscal_config_data.sulfur_revenue_config["PSC 2"],
                "electricity_revenue": self.data.fiscal_config_data.electricity_revenue_config["PSC 2"],
                "co2_revenue": self.data.fiscal_config_data.co2_revenue_config["PSC 2"],
                "is_dmo_end_weighted": self.data.psc_transition_gs_to_gs[keys[1]].dmo_is_weighted,
                "tax_regime": self.data.fiscal_config_data.tax_mode["PSC 2"],
                "tax_rate": self.data.fiscal_config_data.tax_rate["PSC 2"],
                "sunk_cost_reference_year": self.data.fiscal_config_data.sunk_cost_reference_year,
                "depr_method": self.data.fiscal_config_data.depreciation_method["PSC 2"],
                "decline_factor": self.data.fiscal_config_data.decline_factor["PSC 2"],
                "vat_rate": self.data.fiscal_config_data.vat_rate["PSC 2"],
                "lbt_rate": self.data.fiscal_config_data.lbt_rate["PSC 2"],
                "inflation_rate": self.data.fiscal_config_data.inflation_rate["PSC 2"],
                "future_rate": float(self.data.fiscal_config_data.asr_future_rate["PSC 2"]),
                "inflation_rate_applied_to": self.data.general_config_data.inflation_rate_applied_to,
                # "regime": None,
                # "year_ref": None,
                # "tax_type": None,
            },
        }

        # Create an instance of transition object
        self.transition = Transition(
            contract1=self.psc["PSC 1"],
            contract2=self.psc["PSC 2"],
            argument_contract1=self.psc_arguments["PSC 1"],
            argument_contract2=self.psc_arguments["PSC 2"],
        )

        self.summary_arguments["contract"] = self.transition
        self.transition_arguments = {
            "unrec_portion": self.data.fiscal_config_data.transferred_unrec_cost
        }

        return self.transition, self.transition_arguments, self.summary_arguments, self.data

    def _get_transition_contract_gs_to_cr(self) -> tuple:
        """
        Get details for a transition Production Sharing Contract from
        Gross Split to Cost Recovery (GS-CR).

        Returns
        -------
        transition: Transition
            Object representing the transition from Gross Split to Cost Recovery.
        transition_arguments: dict
            Arguments specific to the transition.
        summary_arguments: dict
            Arguments for the transition summary.
        data: Aggregate
            The arranged data collected as an instance of Aggregate class.

        Notes
        -----
        This method defines configurations for PSC transition GS to CR. It creates
        instances and populates the attributes of GrossSplit and CostRecovery objects,
        respectively.
        """
        # Define keys for transition GS-CR
        keys = ["Gross Split Config", "Cost Recovery Config"]

        # Prepare total lifting data
        lifting_total = {
            psc: (
                self.data.oil_lifting_aggregate_total[psc]
                + self.data.gas_lifting_aggregate_total[psc]
                + self.data.sulfur_lifting_aggregate[psc]
                + self.data.electricity_lifting_aggregate[psc]
                + self.data.co2_lifting_aggregate[psc]
            )
            for psc in ["PSC 1", "PSC 2"]
        }

        # Create an instance of PSC transition GS-CR
        self.psc = {
            "PSC 1": GrossSplit(
                start_date=self.data.general_config_data.start_date_project,
                end_date=self.data.general_config_data.end_date_project,
                oil_onstream_date=self.data.general_config_data.oil_onstream_date,
                gas_onstream_date=self.data.general_config_data.gas_onstream_date,
                lifting=lifting_total["PSC 1"],
                tangible_cost=self.data.tangible_cost_aggregate["PSC 1"],
                intangible_cost=self.data.intangible_cost_aggregate["PSC 1"],
                opex=self.data.opex_aggregate["PSC 1"],
                asr_cost=self.data.asr_cost_aggregate["PSC 1"],
                field_status=self.data.psc_transition_gs_to_cr[keys[0]].field_status,
                field_loc=self.data.psc_transition_gs_to_cr[keys[0]].field_location,
                res_depth=self.data.psc_transition_gs_to_cr[keys[0]].reservoir_depth,
                infra_avail=self.data.psc_transition_gs_to_cr[keys[0]].infrastructure_availability,
                res_type=self.data.psc_transition_gs_to_cr[keys[0]].reservoir_type,
                api_oil=self.data.psc_transition_gs_to_cr[keys[0]].oil_api,
                domestic_use=self.data.psc_transition_gs_to_cr[keys[0]].domestic_content_use,
                prod_stage=self.data.psc_transition_gs_to_cr[keys[0]].production_stage,
                co2_content=self.data.psc_transition_gs_to_cr[keys[0]].co2_content,
                h2s_content=self.data.psc_transition_gs_to_cr[keys[0]].h2s_content,
                base_split_ctr_oil=self.data.psc_transition_gs_to_cr[keys[0]].oil_base_split,
                base_split_ctr_gas=self.data.psc_transition_gs_to_cr[keys[0]].gas_base_split,
                split_ministry_disc=self.data.psc_transition_gs_to_cr[keys[0]].ministry_discretion_split,
                oil_dmo_volume_portion=self.data.psc_transition_gs_to_cr[keys[0]].oil_dmo_volume,
                oil_dmo_fee_portion=self.data.psc_transition_gs_to_cr[keys[0]].oil_dmo_fee,
                oil_dmo_holiday_duration=self.data.psc_transition_gs_to_cr[keys[0]].oil_dmo_period,
                gas_dmo_volume_portion=self.data.psc_transition_gs_to_cr[keys[0]].gas_dmo_volume,
                gas_dmo_fee_portion=self.data.psc_transition_gs_to_cr[keys[0]].gas_dmo_fee,
                gas_dmo_holiday_duration=self.data.psc_transition_gs_to_cr[keys[0]].gas_dmo_period,
            ),
            "PSC 2": CostRecovery(
                start_date=self.data.general_config_data.start_date_project_second,
                end_date=self.data.general_config_data.end_date_project_second,
                oil_onstream_date=self.data.general_config_data.oil_onstream_date_second,
                gas_onstream_date=self.data.general_config_data.gas_onstream_date_second,
                lifting=lifting_total["PSC 2"],
                tangible_cost=self.data.tangible_cost_aggregate["PSC 2"],
                intangible_cost=self.data.intangible_cost_aggregate["PSC 2"],
                opex=self.data.opex_aggregate["PSC 2"],
                asr_cost=self.data.asr_cost_aggregate["PSC 2"],
                oil_ftp_is_available=self.data.psc_transition_gs_to_cr[keys[1]].oil_ftp_availability,
                oil_ftp_is_shared=self.data.psc_transition_gs_to_cr[keys[1]].oil_ftp_is_shared,
                oil_ftp_portion=self.data.psc_transition_gs_to_cr[keys[1]].oil_ftp_portion,
                gas_ftp_is_available=self.data.psc_transition_gs_to_cr[keys[1]].gas_ftp_availability,
                gas_ftp_is_shared=self.data.psc_transition_gs_to_cr[keys[1]].gas_ftp_is_shared,
                gas_ftp_portion=self.data.psc_transition_gs_to_cr[keys[1]].gas_ftp_portion,
                tax_split_type=self.data.psc_transition_gs_to_cr[keys[1]].split_type,
                condition_dict=self.data.psc_transition_gs_to_cr[keys[1]].icp_sliding_scale,
                indicator_rc_icp_sliding=(
                    self.data.psc_transition_gs_to_cr[keys[1]].indicator_rc_split_sliding_scale
                ),
                oil_ctr_pretax_share=self.data.psc_transition_gs_to_cr[keys[1]].oil_ctr_pretax,
                gas_ctr_pretax_share=self.data.psc_transition_gs_to_cr[keys[1]].gas_ctr_pretax,
                oil_ic_rate=self.data.psc_transition_gs_to_cr[keys[1]].ic_oil,
                gas_ic_rate=self.data.psc_transition_gs_to_cr[keys[1]].ic_gas,
                ic_is_available=self.data.psc_transition_gs_to_cr[keys[1]].ic_availability,
                oil_cr_cap_rate=self.data.psc_transition_gs_to_cr[keys[1]].oil_cr_cap_rate,
                gas_cr_cap_rate=self.data.psc_transition_gs_to_cr[keys[1]].gas_cr_cap_rate,
                oil_dmo_volume_portion=self.data.psc_transition_gs_to_cr[keys[1]].oil_dmo_volume,
                oil_dmo_fee_portion=self.data.psc_transition_gs_to_cr[keys[1]].oil_dmo_fee,
                oil_dmo_holiday_duration=self.data.psc_transition_gs_to_cr[keys[1]].oil_dmo_period,
                gas_dmo_volume_portion=self.data.psc_transition_gs_to_cr[keys[1]].gas_dmo_volume,
                gas_dmo_fee_portion=self.data.psc_transition_gs_to_cr[keys[1]].gas_dmo_fee,
                gas_dmo_holiday_duration=self.data.psc_transition_gs_to_cr[keys[1]].gas_dmo_period,
            ),
        }

        # Specify arguments for PSC transition GS-CR
        self.psc_arguments = {
            "PSC 1": {
                "sulfur_revenue": self.data.fiscal_config_data.sulfur_revenue_config["PSC 1"],
                "electricity_revenue": self.data.fiscal_config_data.electricity_revenue_config["PSC 1"],
                "co2_revenue": self.data.fiscal_config_data.co2_revenue_config["PSC 1"],
                "is_dmo_end_weighted": self.data.psc_transition_gs_to_cr[keys[0]].dmo_is_weighted,
                "tax_regime": self.data.fiscal_config_data.tax_mode["PSC 1"],
                "tax_rate": self.data.fiscal_config_data.tax_rate["PSC 1"],
                "sunk_cost_reference_year": self.data.fiscal_config_data.sunk_cost_reference_year,
                "depr_method": self.data.fiscal_config_data.depreciation_method["PSC 1"],
                "decline_factor": self.data.fiscal_config_data.decline_factor["PSC 1"],
                "vat_rate": self.data.fiscal_config_data.vat_rate["PSC 1"],
                "lbt_rate": self.data.fiscal_config_data.lbt_rate["PSC 1"],
                "inflation_rate": self.data.fiscal_config_data.inflation_rate["PSC 1"],
                "future_rate": float(self.data.fiscal_config_data.asr_future_rate["PSC 1"]),
                "inflation_rate_applied_to": self.data.general_config_data.inflation_rate_applied_to,
                # "regime": None,
                # "year_ref": None,
                # "tax_type": None,
            },
            "PSC 2": {
                "sulfur_revenue": self.data.fiscal_config_data.sulfur_revenue_config["PSC 2"],
                "electricity_revenue": self.data.fiscal_config_data.electricity_revenue_config["PSC 2"],
                "co2_revenue": self.data.fiscal_config_data.co2_revenue_config["PSC 2"],
                "is_dmo_end_weighted": self.data.psc_transition_gs_to_cr[keys[1]].dmo_is_weighted,
                "tax_regime": self.data.fiscal_config_data.tax_mode["PSC 2"],
                "tax_rate": self.data.fiscal_config_data.tax_rate["PSC 2"],
                "ftp_tax_regime": self.data.fiscal_config_data.tax_payment_config["PSC 2"],
                "sunk_cost_reference_year": self.data.fiscal_config_data.sunk_cost_reference_year,
                "depr_method": self.data.fiscal_config_data.depreciation_method["PSC 2"],
                "decline_factor": self.data.fiscal_config_data.decline_factor["PSC 2"],
                "vat_rate": self.data.fiscal_config_data.vat_rate["PSC 2"],
                "lbt_rate": self.data.fiscal_config_data.lbt_rate["PSC 2"],
                "inflation_rate": self.data.fiscal_config_data.inflation_rate["PSC 2"],
                "future_rate": float(self.data.fiscal_config_data.asr_future_rate["PSC 2"]),
                "inflation_rate_applied_to": self.data.general_config_data.inflation_rate_applied_to,
                # "year_ref": None,
                # "tax_type": None,
            },
        }

        # Create an instance of transition object
        self.transition = Transition(
            contract1=self.psc["PSC 1"],
            contract2=self.psc["PSC 2"],
            argument_contract1=self.psc_arguments["PSC 1"],
            argument_contract2=self.psc_arguments["PSC 2"],
        )

        self.summary_arguments["contract"] = self.transition
        self.transition_arguments = {
            "unrec_portion": self.data.fiscal_config_data.transferred_unrec_cost
        }

        return self.transition, self.transition_arguments, self.summary_arguments, self.data

    def activate(self):
        """
        Activate the Production Sharing Contract based on its type.

        Returns
        -------
        contract: BaseProject | CostRecovery | GrossSplit | Transition
            Object representing the activated contract.

        Notes
        -----
        This method activates the Production Sharing Contract (PSC) based on
        the specified contract type. It selects the appropriate method to
        retrieve details for a single contract or a transition between contracts.

        (1) If the contract type is "Project", the method returns objects to run
            a single BaseProject contract case.
        (2) If the contract type is "PSC Cost Recovery (CR)," the method returns
            objects to run a single Cost Recovery contract case.
        (3) If the contract type is "PSC Gross Split (GS)," the method returns
            objects to run a single Gross Split contract case.
        (4) If the contract type is "Transition CR - CR," the method returns
            objects to run a transition from Cost Recovery to Cost Recovery case.
        (5) If the contract type is "Transition CR - GS," the method returns
            objects to run a transition from Cost Recovery to Gross Split case.
        (6) If the contract type is "Transition GS - CR," the method returns
            objects to run a transition from Gross Split to Cost Recovery case.
        (7) If the contract type is "Transition GS - GS," the method returns
            objects to run a transition from Gross Split to Gross Split case.
        """
        if self.contract_type == "Project":
            return self._get_single_contract_project()

        elif self.contract_type == "PSC Cost Recovery (CR)":
            return self._get_single_contract_cr()

        elif self.contract_type == "PSC Gross Split (GS)":
            return self._get_single_contract_gs()

        elif self.contract_type == "Transition CR - CR":
            return self._get_transition_contract_cr_to_cr()

        elif self.contract_type == "Transition CR - GS":
            return self._get_transition_contract_cr_to_gs()

        elif self.contract_type == "Transition GS - GS":
            return self._get_transition_contract_gs_to_gs()

        elif self.contract_type == "Transition GS - CR":
            return self._get_transition_contract_gs_to_cr()

        else:
            raise ParsingException(
                f"The filled contract type ({self.contract_type}) is not available. "
                f"The available contract types are: "
                f"(1) Project, "
                f"(2) PSC Cost Recovery (CR), "
                f"(3) PSC Gross Split (GS), "
                f"(4) Transition CR - CR, "
                f"(5) Transition CR - GS, "
                f"(6) Transition GS - GS, "
                f"(7) Transition GS - CR."
            )
