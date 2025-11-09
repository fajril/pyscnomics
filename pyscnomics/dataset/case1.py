"""
CASE 1
"""

import numpy as np
from datetime import date
from dataclasses import dataclass, field

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.econ.selection import (
    FluidType,
    CostType,
    VariableSplit082017,
    # VariableSplit522017,
    VariableSplit132024,
    TaxSplitTypeCR,
    ContractType,
    OtherRevenue,
    TaxRegime,
    FTPTaxRegime,
    DeprMethod,
    SunkCostMethod,
    # InflationAppliedTo,
    GrossSplitRegime,
    InitialYearAmortizationIncurred,
    NPVSelection,
    DiscountingMode,
)
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import (
    CapitalCost,
    Intangible,
    OPEX,
    ASR,
    # LBT,
    # CostOfSales,
)
from pyscnomics.io.getattr import (
    convert_object,
    construct_lifting_attr,
    construct_cost_attr,
)


@dataclass
class Case1:
    """
    Example case 1
    ---------------
    A dataclass which stores data representing a case study for economic
    PSC evaluation.

    Parameters
    ----------
    contract_type: ContractType
        Type of contract. Selection: BaseProject, CostRecovery, or GrossSplit.
    """

    contract_type: ContractType

    # Attributes associated with lifting
    lifting: dict = field(default_factory=lambda: {}, init=False, repr=False)

    # Attributes associated with costs
    capital: dict = field(default_factory=lambda: {}, init=False, repr=False)
    intangible: dict = field(default_factory=lambda: {}, init=False, repr=False)
    opex: dict = field(default_factory=lambda: {}, init=False, repr=False)
    asr: dict = field(default_factory=lambda: {}, init=False, repr=False)

    setup_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)
    class_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)
    contract_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)
    summary_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)

    def __post_init__(self):
        """
        Handles the following operations:
            -   Prepare lifting data
            -   Prepare capital data
            -   Prepare intangible data
            -   Prepare opex data
            -   Prepare asr data
            -   Prepare lbt data
            -   Prepare cost of sales data
        """

        self.get_lifting()
        self.get_capital()
        self.get_intangible()
        self.get_opex()
        self.get_asr()
        self.get_setup_arguments()
        self.get_class_arguments()
        self.get_contract_arguments()
        self.get_summary_arguments()

    def get_lifting(self) -> None:
        """
        Initialize and assign lifting data for oil and gas.

        Creates raw lifting data dictionaries for each fluid type and
        stores them in `self.lifting`.

        Returns
        -------
        None
            Updates `self.lifting` with oil and gas lifting data.
        """

        # Prepare lifting data: OIL
        lifting_oil = {
            "start_year": 2023,
            "end_year": 2037,
            "prod_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029,
                    2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "lifting_rate": np.array(
                [
                    460.240475451, 873.845020492, 802.981698938, 932.529841824,
                    928.962143147, 962.198861205, 592.127288123, 688.191243627,
                    609.136555531, 459.317272429, 422.929806095, 324.406696975
                ]
            ),
            "price": np.array(
                [
                    65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65
                ]
            ),
            "fluid_type": FluidType.OIL,
        }

        # Prepare lifting data: GAS
        lifting_gas = {
            "start_year": 2023,
            "end_year": 2037,
            "prod_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029,
                    2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "lifting_rate": np.array(
                [
                    1.00495513, 1.96626503, 2.5581599, 4.76693091,
                    6.0547015, 6.25418605, 4.04543102, 4.58236067,
                    4.26727342, 2.46265306, 2.34562227, 2.10730217
                ]
            ),
            "price": np.array(
                [
                    5.28767105, 5.23662554, 5.20574913, 5.50581852,
                    5.50165717, 5.50819591, 5.66529057, 5.61780989,
                    5.70251656, 5.68642526, 5.69206885, 5.66643698
                ]
            ),
            "ghv": np.array(
                [
                    1010, 1010, 1010, 1010, 1010, 1010,
                    1010, 1010, 1010, 1010, 1010, 1010,
                ]
            ),
            "fluid_type": FluidType.GAS,
        }

        # Store lifting as the class's attribute: "lifting"
        self.lifting = {
            "oil": lifting_oil,
            "gas": lifting_gas,
        }

    def get_capital(self) -> None:
        """
        Initialize and assign capital cost data for oil and gas.

        Creates raw capital cost dictionaries for each fluid type and
        stores them in `self.capital`.

        Returns
        -------
        None
            Updates `self.capital` with oil and gas capital cost data.
        """

        # Prepare capital data: OIL
        capital_oil = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
                ]
            ),
            "cost": np.array(
                [
                    25714.707321, 21578.906603, 18978.850357, 15273.457471,
                    7640.015402, 194.226795, 419.762154, 262.961878, 556.462722,
                    406.618528
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
                ]
            ),
            "pis_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
                ]
            ),
            "useful_life": np.array(
                [
                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5
                ]
            ),
            "depreciation_factor": np.array(
                [
                    0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25
                ]
            ),
            "tax_portion": np.array(
                [
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                ]
            ),
        }

        # Prepare capital data: GAS
        capital_gas = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
                ]
            ),
            "cost": np.array(
                [
                    4613.344573, 3950.905221, 4890.837055, 6679.491296, 4256.871068,
                    108.051922, 252.454555, 152.843852, 345.419185, 192.630464
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                ]
            ),
            "pis_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
                ]
            ),
            "useful_life": np.array(
                [
                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5
                ]
            ),
            "depreciation_factor": np.array(
                [
                    0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25
                ]
            ),
            "tax_portion": np.array(
                [
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                ]
            ),
        }

        # Store capital costs as the class's attribute: "capital"
        self.capital = {
            "oil": capital_oil,
            "gas": capital_gas,
        }

    def get_intangible(self) -> None:
        """
        Initialize and assign intangible cost data for oil and gas.

        Creates raw intangible cost dictionaries for each fluid type and
        stores them in `self.intangible`.

        Returns
        -------
        None
            Updates `self.intangible` with oil and gas intangible cost data.
        """

        # Prepare intangible cost: OIL
        intangible_oil = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
                ]
            ),
            "cost": np.array(
                [
                    46849.00562, 41944.72775, 22099.70561, 23079.23951, 21928.15442,
                    718.934075, 978.88238, 830.661181, 1863.113433, 1594.436516
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
                ]
            ),
            "tax_portion": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }

        # Prepare instangible cost: GAS
        intangible_gas = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
                ]
            ),
            "cost": np.array(
                [
                    8404.941309, 7679.705323, 5695.079368, 10093.16847, 12217.94999,
                    399.956189, 588.722239, 482.813158, 1156.510758, 755.344445
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
                ]
            ),
            "tax_portion": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }

        # Store intangible costs as the class's attribute: "intangible"
        self.intangible = {
            "oil": intangible_oil,
            "gas": intangible_gas,
        }

    def get_opex(self) -> None:
        """
        Initialize and assign OPEX data for oil and gas.

        Creates raw operating expenditure dictionaries for each fluid type and
        stores them in `self.opex`.

        Returns
        -------
        None
            Updates `self.opex` with oil and gas OPEX data.
        """

        # Prepare opex: OIL
        opex_oil = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "fixed_cost": np.array(
                [
                    785.7212242, 4351.968932, 4703.91775, 6325.825984, 6390.700982,
                    6575.827111, 4434.656934, 4938.484606, 8351.939437, 3282.075895,
                    3226.479026, 3468.367892
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
                ]
            ),
            "tax_portion": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }

        # Prepare opex: GAS
        opex_gas = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "fixed_cost": np.array(
                [
                    140.96224, 796.806673, 1212.196461, 2766.452827, 3560.776868,
                    3658.253024, 2667.104051, 2870.44273, 5184.39062, 1554.842585,
                    1582.693399, 1983.719401
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
                ]
            ),
            "tax_portion": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }

        # Store opex as the class's attributes: "opex"
        self.opex = {
            "oil": opex_oil,
            "gas": opex_gas,
        }

    def get_asr(self) -> None:
        """
        Initialize and assign ASR cost data for oil and gas.

        Creates raw Abandonment and Site Restoration (ASR) cost dictionaries
        for each fluid type and stores them in `self.asr`.

        Returns
        -------
        None
            Updates `self.asr` with oil and gas ASR cost data.
        """

        # Prepare ASR: OIL
        oil_asr = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "cost": np.array(
                [
                    173.28832, 316.4429143, 387.4743104, 688.6020255, 717.0204906,
                    726.1726918, 708.4741216, 779.1434817, 770.3641929, 730.4658999
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                ]
            ),
            "final_year": np.array(
                [
                    2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "tax_portion": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }

        # Prepare ASR: GAS
        gas_asr = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "cost": np.array(
                [
                    44.65628428, 138.3889467, 215.8933058, 383.081915,
                    431.2325133, 422.0803121, 439.7788823, 369.1095222,
                    377.8888109, 417.7871039,
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
                ]
            ),
            "final_year": np.array(
                [
                    2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "tax_portion": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }

        # Store ASR costs as the class's attribute: "asr"
        self.asr = {
            "oil": oil_asr,
            "gas": gas_asr,
        }

    def get_setup_arguments(self) -> dict:
        """
        Define and return setup-related project arguments.

        Initializes key project setup parameters such as start and end dates,
        onstream dates, approval year, and POD status.

        Returns
        -------
        dict
            Dictionary containing project setup parameters including
            start/end dates, onstream dates, approval year, and POD flag.
        """

        self.setup_arguments = {
            "start_date": date(year=2023, month=1, day=1),
            "end_date": date(year=2037, month=12, day=31),
            "oil_onstream_date": date(year=2024, month=1, day=1),
            "gas_onstream_date": date(year=2024, month=1, day=1),
            "approval_year": 2026,
            "is_pod_1": False,
        }

    def get_class_arguments(self) -> dict:
        """
        Define and assign keyword arguments specific to the contract type.

        Builds predefined argument dictionaries for Base Project, Cost Recovery,
        and Gross Split contracts. Includes parameters governing FTP, DMO,
        tax splits, investment credits, depreciation, and variable split factors.

        Returns
        -------
        dict
            Contract-specific keyword argument dictionary used for initializing
            the corresponding contract class.

        Raises
        ------
        ValueError
            If the specified contract type is not recognized.

        Notes
        -----
        The selected argument set depends on ``self.contract_type``.
        - Cost Recovery contracts include fiscal and FTP configurations.
        - Gross Split contracts include field, reservoir, and variable split parameters.
        - Base Project contracts are initialized with minimal or default arguments.
        """

        # Cost recovery
        kwargs_cost_recovery = {
            # FTP
            "oil_ftp_is_available": True,
            "oil_ftp_is_shared": True,
            "oil_ftp_portion": 0.2,
            "gas_ftp_is_available": True,
            "gas_ftp_is_shared": True,
            "gas_ftp_portion": 0.2,

            # Tax split
            "tax_split_type": TaxSplitTypeCR.CONVENTIONAL,
            "condition_dict": dict,
            "indicator_rc_icp_sliding": None,
            "oil_ctr_pretax_share": 0.25,
            "gas_ctr_pretax_share": 0.5,

            # Investment credit
            "oil_ic_rate": 0.0,
            "gas_ic_rate": 0.0,
            "ic_is_available": False,
            "oil_cr_cap_rate": 1.0,
            "gas_cr_cap_rate": 1.0,

            # DMO
            "oil_dmo_volume_portion": 0.25,
            "oil_dmo_fee_portion": 0.25,
            "oil_dmo_holiday_duration": 60,
            "gas_dmo_volume_portion": 1.0,
            "gas_dmo_fee_portion": 1.0,
            "gas_dmo_holiday_duration": 60,

            # Carry forward depreciation
            "oil_carry_forward_depreciation": 0.0,
            "gas_carry_forward_depreciation": 0.0,
        }

        # Gross split
        VS_08 = VariableSplit082017
        VS_13 = VariableSplit132024

        kwargs_gross_split = {
            # Field and reservoir properties
            "field_status": VS_08.FieldStatus.NO_POD,
            "field_loc": VS_08.FieldLocation.ONSHORE,
            "res_depth": VS_08.ReservoirDepth.LESSEQUAL_2500,
            "infra_avail": VS_08.InfrastructureAvailability.WELL_DEVELOPED,
            "res_type": VS_08.ReservoirType.CONVENTIONAL,
            "api_oil": VS_08.APIOil.LESSTHAN_25,
            "domestic_use": VS_08.DomesticUse.EQUAL_50_UNTIL_LESSTHAN_70,
            "prod_stage": VS_08.ProductionStage.SECONDARY,
            "co2_content": VS_08.CO2Content.LESSTHAN_5,
            "h2s_content": VS_08.H2SContent.LESSTHAN_100,
            "field_reserves_2024": VS_13.FieldReservesAmount.MEDIUM,
            "infra_avail_2024": VS_13.InfrastructureAvailability.PARTIALLY_AVAILABLE,
            "field_loc_2024": VS_13.FieldLocation.ONSHORE,

            # Ministry discretion
            "split_ministry_disc": 0.08,

            # DMO
            "oil_dmo_volume_portion": 0.25,
            "oil_dmo_fee_portion": 1.0,
            "gas_dmo_volume_portion": 1.0,
            "gas_dmo_fee_portion": 1.0,
            "oil_dmo_holiday_duration": 60,
            "gas_dmo_holiday_duration": 60,

            # Carry forward depreciation
            "oil_carry_forward_depreciation": 0.0,
            "gas_carry_forward_depreciation": 0.0,
        }

        # Mapping contract kwargs based on contract type
        class_args_map = {
            ContractType.BASE_PROJECT: {},
            ContractType.COST_RECOVERY: kwargs_cost_recovery,
            ContractType.GROSS_SPLIT: kwargs_gross_split,
        }

        try:
            self.class_arguments = class_args_map[self.contract_type]

        except KeyError:
            raise ValueError(f"Unrecognized contract type: {self.contract_type!r}")

    def get_contract_arguments(self) -> dict:
        """
        Build and assign contract arguments based on the contract type.

        Constructs predefined argument sets for Base Project, Cost Recovery,
        and Gross Split contracts covering fiscal, depreciation, and revenue settings.

        Returns
        -------
        dict
            Argument dictionary for the current contract type.

        Raises
        ------
        ValueError
            If the contract type is not recognized.
        """

        # Base project
        args_base_project = {
            "sulfur_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "electricity_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "tax_rate": 0.0,
            "year_inflation": None,
            "inflation_rate": 0.0,
            "inflation_rate_applied_to": None,
        }

        # Cost recovery
        args_cost_recovery = {
            **args_base_project,
            "is_dmo_end_weighted": False,
            "tax_regime": TaxRegime.NAILED_DOWN,
            "effective_tax_rate": None,
            "ftp_tax_regime": FTPTaxRegime.PDJP_20_2017,
            "depr_method": DeprMethod.PSC_DB,
            "decline_factor": 2,
            "post_uu_22_year2001": True,
            "oil_cost_of_sales_applied": False,
            "gas_cost_of_sales_applied": False,
            "sum_undepreciated_cost": False,
            "sunk_cost_method": SunkCostMethod.DEPRECIATED_TANGIBLE,
        }

        # Base project
        args_gross_split = {
            **args_base_project,
            "cum_production_split_offset": 0.0,
            "depr_method": DeprMethod.PSC_DB,
            "decline_factor": 2,
            "sum_undepreciated_cost": False,
            "is_dmo_end_weighted": False,
            "tax_regime": TaxRegime.NAILED_DOWN,
            "effective_tax_rate": 0.22,
            "amortization": False,
            "sunk_cost_method": SunkCostMethod.DEPRECIATED_TANGIBLE,
            "regime": GrossSplitRegime.PERMEN_ESDM_13_2024,
            "reservoir_type_permen_2024": VariableSplit132024.ReservoirType.MK,
            "initial_amortization_year": InitialYearAmortizationIncurred.ONSTREAM_YEAR,
        }

        # Pooled args
        ctr_args = {
            ContractType.BASE_PROJECT: args_base_project,
            ContractType.COST_RECOVERY: args_cost_recovery,
            ContractType.GROSS_SPLIT: args_gross_split,
        }

        try:
            self.contract_arguments = ctr_args[self.contract_type]

        except KeyError:
            raise ValueError(f"Unrecognized contract type: {self.contract_type!r}")

    def get_summary_arguments(self) -> dict:
        """
        Define and assign project summary arguments.

        Sets key economic parameters for NPV and profitability evaluation,
        including discount rate, inflation, and discounting mode.

        Returns
        -------
        dict
            Dictionary containing summary argument values.
        """

        self.summary_arguments = {
            "discount_rate": 0.1,
            "npv_mode": NPVSelection.NPV_SKK_NOMINAL_TERMS,
            "discounting_mode": DiscountingMode.END_YEAR,
            "discount_rate_start_year": 2023,
            "inflation_rate": 0.0,
            "profitability_discounted": False,
        }

    def as_dict(self) -> dict:
        """
        Convert all contract-related attributes into a structured dictionary.

        The method converts setup, summary, contract, and class arguments, as well as
        cost and lifting data, into serializable dictionary forms. Each component is
        processed through helper converters and class constructors depending on the
        contract type.

        Returns
        -------
        dict
            A dictionary containing all converted and structured contract data,
            including setup, summary, contract arguments, cost components,
            and lifting information.
        """

        def _converter(source: dict):
            return {key: convert_object(objects=val) for key, val in source.items()}

        def _construct_cost_attributes(source: dict, Cls):
            items = tuple([Cls(**val) for val in source.values()])
            return construct_cost_attr(cost=items)

        # Convert "setup_arguments", "summary_arguments", and "contract_arguments"
        setup = _converter(source=self.setup_arguments)
        summary_arguments = _converter(source=self.summary_arguments)
        contract_arguments = _converter(source=self.contract_arguments)

        # Convert "class_arguments"
        converter_class_args = _converter(source=self.class_arguments)
        grosssplit = (
            converter_class_args
            if self.contract_type == ContractType.GROSS_SPLIT
            else None
        )
        cost_recovery = (
            converter_class_args
            if self.contract_type == ContractType.COST_RECOVERY
            else None
        )

        # Convert "lifting"
        lifting = construct_lifting_attr(
            lifting=tuple([Lifting(**lft) for lft in self.lifting.values()])
        )

        # Convert "capital", "intangible", "opex", "asr"
        capital = _construct_cost_attributes(self.capital, CapitalCost)
        intangible = _construct_cost_attributes(self.intangible, Intangible)
        opex = _construct_cost_attributes(self.opex, OPEX)
        asr = _construct_cost_attributes(self.asr, ASR)

        # Mapping converted data
        mapping_data = (
            ("setup", setup),
            ("summary_arguments", summary_arguments),
            ("contract_arguments", contract_arguments),
            ("grosssplit", grosssplit),
            ("cost_recovery", cost_recovery),
            ("lifting", lifting),
            ("capital", capital),
            ("intangible", intangible),
            ("opex", opex),
            ("asr", asr),
        )

        return {key: val for key, val in mapping_data}

    def as_class(self) -> CostRecovery | GrossSplit | BaseProject:
        """
        Convert stored attributes into a contract class instance.

        Creates per-fluid instances of lifting and cost components, merges them
        with setup and contract-specific arguments, and returns the corresponding
        contract object (BaseProject, CostRecovery, or GrossSplit).

        Returns
        -------
        CostRecovery | GrossSplit | BaseProject
            Initialized contract instance.

        Raises
        ------
        ValueError
            If the specified contract type is not recognized.
        """

        fluids = ["oil", "gas"]

        # Create per fluid instances for lifting and each cost category
        instances = {
            "lifting": {fl: Lifting(**self.lifting[fl]) for fl in fluids},
            "capital": {fl: CapitalCost(**self.capital[fl]) for fl in fluids},
            "intangible": {fl: Intangible(**self.intangible[fl]) for fl in fluids},
            "opex": {fl: OPEX(**self.opex[fl]) for fl in fluids},
            "asr": {fl: ASR(**self.asr[fl]) for fl in fluids},
        }

        # Construct tuples from the created instances
        instances_as_tuple = {
            "lifting": tuple([lft for lft in instances["lifting"].values()]),
            "capital_cost": tuple([cap for cap in instances["capital"].values()]),
            "intangible_cost": tuple([itg for itg in instances["intangible"].values()]),
            "opex": tuple([op for op in instances["opex"].values()]),
            "asr_cost": tuple([ar for ar in instances["asr"].values()]),
        }

        # Merge keyword arguments to create an instance of contract
        kwargs_merged = {
            **self.setup_arguments,     # setup arguments
            **instances_as_tuple,       # lifting and costs arguments
            **self.class_arguments,     # class's arguments
        }

        # Create and return the appropriate contract instance
        if self.contract_type == ContractType.COST_RECOVERY:
            return CostRecovery(**kwargs_merged)
        elif self.contract_type == ContractType.GROSS_SPLIT:
            return GrossSplit(**kwargs_merged)
        elif self.contract_type == ContractType.BASE_PROJECT:
            return BaseProject(**kwargs_merged)
        else:
            raise ValueError(f"Unrecognized contract type: {self.contract_type!r}")
