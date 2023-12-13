"""
Prepare data input loaded from a target Excel file.
"""

from dataclasses import dataclass, field, InitVar
from datetime import datetime
from pyxlsb import convert_date
import numpy as np
import pandas as pd

from pyscnomics.econ.selection import TaxSplitTypeCR
from pyscnomics.tools.helper import (
    get_inflation_applied_converter,
    get_tax_payment_converter,
    get_tax_regime_converter,
    get_npv_mode_converter,
    get_discounting_mode_converter,
    get_depreciation_method_converter,
    get_array_from_target,
    get_lifting_data_split_simple,
    get_lifting_data_split_advanced,
    get_cost_data_split,
    get_to_list_converter,
    get_fluidtype_converter,
    get_boolean_converter,
    get_split_type_converter,
)


class GeneralConfigDataException(Exception):
    """ Exception to be raised for inappropriate use of GeneralConfigData class """

    pass


class FiscalConfigDataException(Exception):
    """ Exception to be raised for inappropriate use of FiscalConfigData class """

    pass


class OilLiftingDataException(Exception):
    """ Exception to be raised for an inappropriate use of OilLiftingData class """

    pass


class GasLiftingDataException(Exception):
    """ Exception to be raised for an inappropriate use of GasLiftingData class """

    pass


class LPGPropaneLiftingDataException(Exception):
    """ Exception to be raised for an inappropriate use of LPGPropaneLiftingData class """

    pass


class LPGButaneLiftingDataException(Exception):
    """ Exception to be raised for an inappropriate use of LPGButaneLiftingData class """

    pass


class SulfurLiftingDataException(Exception):
    """ Exception to be raised for an inappropriate use of SulfurLiftingData class """

    pass


class ElectricityLiftingDataException(Exception):
    """ Exception to be raised for an inappropriate use of ElectricityLiftingData class """

    pass


class CO2LiftingDataException(Exception):
    """ Exception to be raised for an inappropriate use of CO2LiftingData class """

    pass


class TangibleCostDataException(Exception):
    """ Exception to be raised for an inappropriate use of TangibleCostData class """

    pass


class IntangibleCostDataException(Exception):
    """ Exception to be raised for an inappropriate use of IntangibleCostData class """

    pass


class OPEXDataException(Exception):
    """ Exception to be raised for an inappropriate use of OPEXData class """

    pass


class ASRCostDataException(Exception):
    """ Exception to be raised for an inappropriate use of ASRCostData class """
    pass


class MonteCarloDataException(Exception):
    """ Exception to be raised for an inappropriate use of MonteCarloData class """

    pass


class OptimizationDataException(Exception):
    """ Exception to be raised for an inappropriate use of OptimizationData class """

    pass


@dataclass
class GeneralConfigData:
    """
    A dataclass representing general configuration information for an oil and gas economic project.

    Attributes
    ----------
    start_date_project: date
        The start date of the project.
    end_date_project: date
        The end date of the project.
    start_date_project_second: date
        The start date of the second project (for PSC transition).
    end_date_project_second: date
        The end date of the second project (for PSC transition).
    type_of_contract: str
        The type of contract associated with the project.
    oil_onstream_date: date, optional
        The onstream date for oil production (defaults to None).
    gas_onstream_date: date, optional
        The onstream date for gas production (defaults to None).
    discount_rate_start_year: int, optional
        The start year for applying discount rate (defaults to None).
    discount_rate: float
        The discount rate applied to project cash flows (defaults to 0.1).
    inflation_rate_applied_to: str, optional
        The attribute to which inflation rate is applied (defaults to None).
    vat_discount: float
        The value-added tax discount (defaults to 0.0).
    lbt_discount: float
        The land and building tax discount (defaults to 0.0).
    gsa_number: int
        The number of GSA available.
    """
    start_date_project: datetime
    end_date_project: datetime
    start_date_project_second: datetime
    end_date_project_second: datetime
    type_of_contract: str
    oil_onstream_date: datetime
    gas_onstream_date: datetime
    discount_rate_start_year: int
    discount_rate: float
    inflation_rate_applied_to: str
    vat_discount: float
    lbt_discount: float
    gsa_number: int

    # Attributes associated with duration of the project
    project_duration: int = field(default=None, init=False)
    project_years: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        # Prepare attribute vat_discount
        if self.vat_discount is None:
            self.vat_discount = 0.0
        self.vat_discount = float(self.vat_discount)

        # Prepare attribute lbt_discount
        if self.lbt_discount is None:
            self.lbt_discount = 0.0
        self.lbt_discount = float(self.lbt_discount)

        # Prepare attribute gsa_number
        self.gsa_number = int(self.gsa_number)

        # Prepare attribute inflation_rate_applied_to
        self.inflation_rate_applied_to = (
            get_inflation_applied_converter(target=self.inflation_rate_applied_to)
        )

        # Prepare attributes associated with datetime
        self.start_date_project, self.end_date_project = list(
            map(convert_date, [self.start_date_project, self.end_date_project])
        )

        # Prepare attribute project_years and project_duration
        if "Transition" in self.type_of_contract:
            self.start_date_project_second, self.end_date_project_second = list(
                map(convert_date, [self.start_date_project_second, self.end_date_project_second])
            )

            if self.end_date_project.year < self.start_date_project.year:
                raise GeneralConfigDataException(
                    f"Start year of the first contract ({self.start_date_project.year}) "
                    f"is after the end year of the first contract ({self.end_date_project.year})."
                )

            if self.end_date_project_second.year < self.start_date_project_second.year:
                raise GeneralConfigDataException(
                    f"Start year of the second contract ({self.start_date_project_second.year}) "
                    f"is after the end year of the second contract ({self.end_date_project_second.year})."
                )

            if self.end_date_project.year > self.start_date_project_second.year:
                raise GeneralConfigDataException(
                    f"End year of the first contract ({self.end_date_project.year}) "
                    f"is after the start year of the second contract "
                    f"({self.start_date_project_second.year})."
                )

            self.project_duration = (
                    self.end_date_project_second.year - self.start_date_project.year + 1
            )
            self.project_years = np.arange(
                self.start_date_project.year, self.end_date_project_second.year + 1, 1
            )

            if self.oil_onstream_date is not None:
                self.oil_onstream_date = {
                    "PSC 1": convert_date(date=self.oil_onstream_date),
                    "PSC 2": self.start_date_project_second,
                }

            if self.gas_onstream_date is not None:
                self.gas_onstream_date = {
                    "PSC 1": convert_date(date=self.gas_onstream_date),
                    "PSC 2": self.start_date_project_second,
                }

        else:
            if self.end_date_project.year >= self.start_date_project.year:
                self.project_duration = self.end_date_project.year - self.start_date_project.year + 1
                self.project_years = np.arange(
                    self.start_date_project.year, self.end_date_project.year + 1, 1
                )

            else:
                raise GeneralConfigDataException(
                    f"start year {self.start_date_project.year} "
                    f"is after the end year: {self.end_date_project.year}"
                )

            if self.oil_onstream_date is not None:
                self.oil_onstream_date = convert_date(date=self.oil_onstream_date)

            if self.gas_onstream_date is not None:
                self.gas_onstream_date = convert_date(date=self.gas_onstream_date)


@dataclass
class FiscalConfigData:
    """
    A dataclass representing fiscal configuration information for an oil and gas economic project.

    Attributes
    ----------
    tax_payment_method: str
        The method of tax payment.
    tax_ftp_regime: str
        The tax regime for FTP.
    npv_mode: str
        The mode for Net Present Value (NPV) calculation.
    discounting_mode: str
        The mode for discounting.
    future_rate_asr: float
        The future rate for ASR.
    depreciation_method: str
        The method of depreciation.
    decline_factor: float | int
        The decline factor for economic analysis.
    transferred_unrec_cost: float
        The transferred unrecoverable cost.
    tax_rate_second_contract: float
        The tax rate for the second contract
    tax_mode: str
        The mode for tax calculation.
    tax_rate_init: float
        Initial value for tax rate (used in User Input - Single Value mode).
    multi_tax: dict
        Dictionary for multi-year tax rates (used in User Input - Multi Value mode)
    inflation_rate_mode: str
        The mode for inflation rate calculation.
    inflation_rate_init: float
        Initial value for inflation rate (used in User Input - Single Value mode).
    multi_inflation: dict
        Dictionary for multi-year inflation rates (used in User Input - Multi Value mode)
    vat_mode: str
        The mode for VAT calculation.
    vat_rate_init: float
        Initial value for VAT rate (used in User Input - Single Value mode).
    multi_vat: dict
        Dictionary for multi-year VAT rates (used in User Input - Multi Value mode)
    lbt_mode: str
        The mode for LBT calculation.
    lbt_rate_init: float
        Initial value for LBT rate (used in User Input - Single Value mode).
    multi_lbt: float
        Multi-year LBT rates (used in User Input - Multi Value mode)
    project_years: np.ndarray
        Array representing project years
    tax_rate: None | float | np.ndarray
        Calculated tax rate based on the specified mode.
    inflation_rate: float | np.ndarray
        Calculated inflation rate based on the specified mode.
    vat_rate: float | np.ndarray
        Calculated VAT rate based on the specified mode.
    lbt_rate: float | np.ndarray
        Calculated LBT rate based on the specified mode.
    """
    tax_payment_method: str
    tax_ftp_regime: str
    npv_mode: str
    discounting_mode: str
    future_rate_asr: float
    depreciation_method: str
    decline_factor: float | int
    transferred_unrec_cost: float
    tax_rate_second_contract: float

    # Attributes associated with tax
    tax_mode: str
    tax_rate_init: InitVar[float] = field(repr=False)
    multi_tax: InitVar[dict] = field(repr=False)

    # Attributes associated with inflation
    inflation_rate_mode: str
    inflation_rate_init: InitVar[float] = field(repr=False)
    multi_inflation: InitVar[dict] = field(repr=False)

    # Attributes associated with VAT
    vat_mode: str
    vat_rate_init: InitVar[float] = field(repr=False)
    multi_vat: InitVar[dict] = field(repr=False)

    # Attributes associated with LBT
    lbt_mode: str
    lbt_rate_init: InitVar[float] = field(repr=False)
    multi_lbt: InitVar[float] = field(repr=False)

    # Attributes associated with project duration
    project_years: np.ndarray

    # Attributes to be defined later
    tax_rate: None | float | np.ndarray = field(default=None, init=False)
    inflation_rate: float | np.ndarray = field(default=None, init=False)
    vat_rate: float | np.ndarray = field(default=None, init=False)
    lbt_rate: float | np.ndarray = field(default=None, init=False)

    def __post_init__(
        self,
        tax_rate_init,
        multi_tax,
        inflation_rate_init,
        multi_inflation,
        vat_rate_init,
        multi_vat,
        lbt_rate_init,
        multi_lbt,
    ):
        # Prepare attribute tax_payment_method
        self.tax_payment_method = get_tax_payment_converter(target=self.tax_payment_method)

        # Prepare attribute tax_ftp_regime
        self.tax_ftp_regime = get_tax_regime_converter(target=self.tax_ftp_regime)

        # Prepare attribute npv_mode
        self.npv_mode = get_npv_mode_converter(target=self.npv_mode)

        # Prepare attribute discounting_mode
        self.discounting_mode = get_discounting_mode_converter(target=self.discounting_mode)

        # Prepare attribute depreciation_method
        self.depreciation_method = get_depreciation_method_converter(target=self.depreciation_method)

        # Prepare attribute tax_rate
        if self.tax_mode == "User Input - Single Value":
            if pd.isna(tax_rate_init):
                self.tax_rate = 0.44
            else:
                self.tax_rate = float(tax_rate_init)

        elif self.tax_mode == "Nailed Down" or self.tax_mode == "Prevailing":
            self.tax_rate = None

        elif self.tax_mode == "User Input - Multi Value":
            self.tax_rate = get_array_from_target(target=multi_tax, project_years=self.project_years)

        # Prepare attribute inflation_rate
        if self.inflation_rate_mode == "User Input - Single Value":
            if pd.isna(inflation_rate_init):
                self.inflation_rate = 0.02
            else:
                self.inflation_rate = float(inflation_rate_init)

        elif self.inflation_rate_mode == "User Input - Multi Value":
            self.inflation_rate = (
                get_array_from_target(target=multi_inflation, project_years=self.project_years)
            )

        # Prepare attribute vat_rate
        if self.vat_mode == "User Input - Single Value":
            if pd.isna(vat_rate_init):
                self.vat_rate = 0.12
            else:
                self.vat_rate = float(vat_rate_init)

        elif self.vat_mode == "User Input - Multi Value":
            self.vat_rate = get_array_from_target(target=multi_vat, project_years=self.project_years)

        # Prepare attribute lbt_rate
        if self.lbt_mode == "User Input - Single Value":
            if pd.isna(lbt_rate_init):
                self.lbt_rate = 0.02
            else:
                self.lbt_rate = float(lbt_rate_init)

        elif self.lbt_mode == "User Input - Multi Value":
            self.lbt_rate = get_array_from_target(target=multi_lbt, project_years=self.project_years)


@dataclass
class OilLiftingData:
    """
    A dataclass representing oil lifting information for an oil and gas economic project.

    Attributes
    ----------
    prod_year_init: dict
        Dictionary containing production years data.
    oil_lifting_rate : dict
        Dictionary containing oil lifting rate data.
    oil_price : dict
        Dictionary containing oil price data.
    condensate_lifting_rate : dict
        Dictionary containing condensate lifting rate data.
    condensate_price : dict
        Dictionary containing condensate price data.
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    type_of_contract: str
        A string depicting the type of PSC contract.
    end_date_project: date
        The end date of the project.
    start_date_project_second: date
        The start date of the second project (for PSC transition).

    Notes
    -----
    This dataclass is used to store and organize information related to oil lifting.
    """
    prod_year_init: InitVar[dict] = field(repr=False)
    oil_lifting_rate: dict
    oil_price: dict
    condensate_lifting_rate: dict
    condensate_price: dict

    # Attributes associated with project duration
    project_duration: int = field(repr=False)
    project_years: np.ndarray = field(repr=False)

    # Attributes associated with PSC transition
    type_of_contract: str = field(repr=False)
    end_date_project: datetime = field(repr=False)
    start_date_project_second: datetime = field(repr=False)

    # Attributes to be defined later
    prod_year: dict = field(default=None, init=False)

    def __post_init__(self, prod_year_init: dict):
        # Prepare attribute prod_year
        if not isinstance(prod_year_init, dict):
            raise OilLiftingDataException(
                f"Attribute prod_year must be provided in the form of dictionary. "
                f"The current datatype of prod_year is "
                f"{prod_year_init.__class__.__qualname__}"
            )

        for ws in prod_year_init.keys():
            if prod_year_init[ws] is None:
                prod_year_init[ws] = np.int_(self.project_years)
            else:
                prod_year_init[ws] = np.int_(prod_year_init[ws])

        self.prod_year = prod_year_init.copy()

        # Prepare attribute oil_lifting_rate
        if not isinstance(self.oil_lifting_rate, dict):
            raise OilLiftingDataException(
                f"Attribute oil_lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of oil_lifting_rate is "
                f"{self.oil_lifting_rate.__class__.__qualname__}"
            )

        oil_lifting_rate_nan = {}
        for ws in self.oil_lifting_rate.keys():
            if self.oil_lifting_rate[ws] is None:
                self.oil_lifting_rate[ws] = np.zeros_like(self.project_years, dtype=np.float_)
                oil_lifting_rate_nan[ws] = None
            else:
                oil_lifting_rate_nan[ws] = np.argwhere(pd.isna(self.oil_lifting_rate[ws])).ravel()
                if len(oil_lifting_rate_nan[ws]) > 0:
                    self.oil_lifting_rate[ws][oil_lifting_rate_nan[ws]] = (
                        np.zeros(len(oil_lifting_rate_nan[ws]), dtype=np.float_)
                    )
                else:
                    self.oil_lifting_rate[ws] = np.float_(self.oil_lifting_rate[ws])

        # Prepare attribute oil_price
        if not isinstance(self.oil_price, dict):
            raise OilLiftingDataException(
                f"Attribute oil_price must be provided in the form of dictionary. "
                f"The current datatype of oil_price is "
                f"{self.oil_price.__class__.__qualname__}"
            )

        oil_price_nan = {}
        for ws in self.oil_price.keys():
            if self.oil_price[ws] is None:
                self.oil_price[ws] = np.zeros_like(self.project_years, dtype=np.float_)
                oil_price_nan[ws] = None
            else:
                oil_price_nan[ws] = np.argwhere(pd.isna(self.oil_price[ws])).ravel()
                if len(oil_price_nan[ws]) > 0:
                    self.oil_price[ws][oil_price_nan[ws]] = (
                        np.zeros(len(oil_price_nan[ws]), dtype=np.float_)
                    )
                else:
                    self.oil_price[ws] = np.float_(self.oil_price[ws])

        # Prepare attribute condensate_lifting_rate
        if not isinstance(self.condensate_lifting_rate, dict):
            raise OilLiftingDataException(
                f"Attribute condensate_lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of condensate_lifting_rate is "
                f"{self.condensate_lifting_rate.__class__.__qualname__}"
            )

        condensate_lifting_rate_nan = {}
        for ws in self.condensate_lifting_rate.keys():
            if self.condensate_lifting_rate[ws] is None:
                self.condensate_lifting_rate[ws] = np.zeros_like(self.project_years, dtype=np.float_)
                condensate_lifting_rate_nan[ws] = None
            else:
                condensate_lifting_rate_nan[ws] = np.argwhere(
                    pd.isna(self.condensate_lifting_rate[ws])
                ).ravel()

                if len(condensate_lifting_rate_nan[ws]) > 0:
                    self.condensate_lifting_rate[ws][condensate_lifting_rate_nan[ws]] = (
                        np.zeros(len(condensate_lifting_rate_nan[ws]), dtype=np.float_)
                    )
                else:
                    self.condensate_lifting_rate[ws] = np.float_(self.condensate_lifting_rate[ws])

        # Prepare attribute condensate_price
        if not isinstance(self.condensate_price, dict):
            raise OilLiftingDataException(
                f"Attribute condensate_price must be provided in the form of dictionary. "
                f"The current datatype of condensate_price is "
                f"{self.condensate_price.__class__.__qualname__}"
            )

        condensate_price_nan = {}
        for ws in self.condensate_price.keys():
            if self.condensate_price[ws] is None:
                self.condensate_price[ws] = np.zeros_like(self.project_years, dtype=np.float_)
                condensate_price_nan[ws] = None
            else:
                condensate_price_nan[ws] = np.argwhere(pd.isna(self.condensate_price[ws])).ravel()
                if len(condensate_price_nan[ws]) > 0:
                    self.condensate_price[ws][condensate_price_nan[ws]] = (
                        np.zeros(len(condensate_price_nan[ws]), dtype=np.float_)
                    )
                else:
                    self.condensate_price[ws] = np.float_(self.condensate_price[ws])

        # Adjust data for transition case
        if "Transition" in self.type_of_contract:
            # Modify attributes "prod_year", "oil_lifting_rate", "oil_price",
            # "condensate_lifting_rate", and "condensate_price"
            target_attrs = {
                "attr": [
                    self.prod_year,
                    self.oil_lifting_rate,
                    self.oil_price,
                    self.condensate_lifting_rate,
                    self.condensate_price,
                ],
                "status": [
                    False,
                    True,
                    False,
                    True,
                    False,
                ]
            }

            (
                self.prod_year,
                self.oil_lifting_rate,
                self.oil_price,
                self.condensate_lifting_rate,
                self.condensate_price,
            ) = [
                get_lifting_data_split_simple(
                    target_attr=i,
                    is_target_attr_volume=j,
                    prod_year_init=prod_year_init,
                    end_date_contract_1=self.end_date_project,
                    start_date_contract_2=self.start_date_project_second,
                )
                for i, j in zip(target_attrs["attr"], target_attrs["status"])
            ]


@dataclass
class GasLiftingData:
    """
    A dataclass representing gas lifting information for an oil and gas economic project.

    Attributes
    ----------
    gsa_number: int
        The number of GSA.
    prod_year_init: dict
        Dictionary containing production years data.
    prod_rate: dict
        Dictionary containing gas lifting rate data.
    lifting_rate: dict
        Dictionary containing gas GSA lifting rate data.
    ghv: dict
        Dictionary containing gas GSA ghv data.
    price: dict
        Dictionary containing gas GSA price data.
    type_of_contract: str
        The type of contract associated with the project.
    end_date_project: date
        The end date of the project.
    start_date_project_second: date
        The start date of the second project (for PSC transition).
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    """
    gsa_number: int
    prod_year_init: InitVar[dict] = field(repr=False)
    prod_rate: dict
    lifting_rate: dict
    ghv: dict
    price: dict

    # Attributes associated with PSC transition
    type_of_contract: str = field(repr=False)
    end_date_project: datetime = field(repr=False)
    start_date_project_second: datetime = field(repr=False)

    # Attributes associated with project duration
    project_duration: int = field(repr=False)
    project_years: np.ndarray = field(repr=False)

    # Attributes to be defined later
    prod_year: dict = field(default=None, init=False)

    def __post_init__(self, prod_year_init):
        # Prepare attribute prod_year
        if not isinstance(prod_year_init, dict):
            raise GasLiftingDataException(
                f"Attribute prod_year must be provided in the form of dictionary. "
                f"The current datatype of prod_year is "
                f"{prod_year_init.__class__.__qualname__}"
            )

        for ws in prod_year_init.keys():
            if prod_year_init[ws] is None:
                prod_year_init[ws] = np.int_(self.project_years)
            else:
                prod_year_init[ws] = np.int_(prod_year_init[ws])

        self.prod_year = prod_year_init.copy()

        # Prepare attribute gas_prod_rate
        if not isinstance(self.prod_rate, dict):
            raise GasLiftingDataException(
                f"Attribute gas_prod_rate must be provided in the form of dictionary. "
                f"The current datatype of gas_prod_rate is "
                f"{self.prod_rate.__class__.__qualname__}"
            )

        prod_rate_nan = {}
        for ws in self.prod_rate.keys():
            if self.prod_rate[ws] is None:
                self.prod_rate[ws] = np.zeros_like(self.project_years, dtype=np.float_)
                prod_rate_nan[ws] = None
            else:
                prod_rate_nan[ws] = np.argwhere(pd.isna(self.prod_rate[ws])).ravel()
                if len(prod_rate_nan[ws]) > 0:
                    self.prod_rate[ws][prod_rate_nan[ws]] = (
                        np.zeros(len(prod_rate_nan[ws]), dtype=np.float_)
                    )
                else:
                    self.prod_rate[ws] = np.float_(self.prod_rate[ws])

        # Prepare attribute lifting_rate
        if not isinstance(self.lifting_rate, dict):
            raise GasLiftingDataException(
                f"Attribute lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of lifting_rate is "
                f"{self.lifting_rate.__class__.__qualname__}"
            )

        lifting_rate_nan = {}
        for ws in self.lifting_rate.keys():
            lifting_rate_nan[ws] = {}
            for gsa in self.lifting_rate[ws].keys():
                if self.lifting_rate[ws][gsa] is None:
                    self.lifting_rate[ws][gsa] = np.zeros_like(self.project_years, dtype=np.float_)
                    lifting_rate_nan[ws][gsa] = None
                else:
                    lifting_rate_nan[ws][gsa] = np.argwhere(pd.isna(self.lifting_rate[ws][gsa])).ravel()
                    if len(lifting_rate_nan[ws][gsa]) > 0:
                        self.lifting_rate[ws][gsa][lifting_rate_nan[ws][gsa]] = (
                            np.zeros(len(lifting_rate_nan[ws][gsa]), dtype=np.float_)
                        )
                    else:
                        self.lifting_rate[ws][gsa] = np.float_(self.lifting_rate[ws][gsa])

        # Prepare attribute ghv
        if not isinstance(self.ghv, dict):
            raise GasLiftingDataException(
                f"Attribute ghv must be provided in the form of dictionary. "
                f"The current datatype of ghv is "
                f"{self.ghv.__class__.__qualname__}"
            )

        ghv_nan = {}
        for ws in self.ghv.keys():
            ghv_nan[ws] = {}
            for gsa in self.ghv[ws].keys():
                if self.ghv[ws][gsa] is None:
                    self.ghv[ws][gsa] = np.zeros_like(self.project_years, dtype=np.float_)
                    ghv_nan[ws][gsa] = None
                else:
                    ghv_nan[ws][gsa] = np.argwhere(pd.isna(self.ghv[ws][gsa])).ravel()
                    if len(ghv_nan[ws][gsa]) > 0:
                        self.ghv[ws][gsa][ghv_nan[ws][gsa]] = (
                            np.zeros(len(ghv_nan[ws][gsa]), dtype=np.float_)
                        )
                    else:
                        self.ghv[ws][gsa] = np.float_(self.ghv[ws][gsa])

        # Prepare attribute price
        if not isinstance(self.price, dict):
            raise GasLiftingDataException(
                f"Attribute price must be provided in the form of dictionary. "
                f"The current datatype of price is "
                f"{self.price.__class__.__qualname__}"
            )

        price_nan = {}
        for ws in self.price.keys():
            price_nan[ws] = {}
            for gsa in self.price[ws].keys():
                if self.price[ws][gsa] is None:
                    self.price[ws][gsa] = np.zeros_like(self.project_years, dtype=np.float_)
                    price_nan[ws][gsa] = None
                else:
                    price_nan[ws][gsa] = np.argwhere(pd.isna(self.price[ws][gsa])).ravel()
                    if len(price_nan[ws][gsa]) > 0:
                        self.price[ws][gsa][price_nan[ws][gsa]] = (
                            np.zeros(len(price_nan[ws][gsa]), dtype=np.float_)
                        )
                    else:
                        self.price[ws][gsa] = np.float_(self.price[ws][gsa])

        # Adjust data for PSC transition case
        if "Transition" in self.type_of_contract:
            # Modify attributes "prod_year" and "prod_rate"
            target_attrs_general = {
                "attr": [self.prod_year, self.prod_rate],
                "status": [False, True],
            }

            (
                self.prod_year,
                self.prod_rate,
            ) = [
                get_lifting_data_split_simple(
                    target_attr=i,
                    is_target_attr_volume=j,
                    prod_year_init=prod_year_init,
                    end_date_contract_1=self.end_date_project,
                    start_date_contract_2=self.start_date_project_second,
                )
                for i, j in zip(target_attrs_general["attr"], target_attrs_general["status"])
            ]

            # Modify attributes "lifting_rate", "ghv", and "price"
            target_attrs_gsa = {
                "attr": [self.lifting_rate, self.ghv, self.price],
                "status": [True, False, False],
            }

            (
                self.lifting_rate,
                self.ghv,
                self.price,
            ) = [
                get_lifting_data_split_advanced(
                    target_attr=i,
                    is_target_attr_volume=j,
                    prod_year_init=prod_year_init,
                    end_date_contract_1=self.end_date_project,
                    start_date_contract_2=self.start_date_project_second,
                    gsa_number=self.gsa_number,
                )
                for i, j in zip(target_attrs_gsa["attr"], target_attrs_gsa["status"])
            ]


@dataclass
class LPGPropaneLiftingData:
    """
    A dataclass representing LPG Propane lifting information for an oil and gas economic project.

    Attributes
    ----------
    prod_year_init: dict
        Dictionary containing production years data.
    lifting_rate: dict
        Dictionary containing LPG Propane lifting rate data.
    price: dict
        Dictionary containing LPG Propane price data.
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    type_of_contract: str
        A string depicting the type of PSC contract.
    end_date_project: date
        The end date of the project.
    start_date_project_second: date
        The start date of the second project (for PSC transition).
    """
    prod_year_init: InitVar[dict] = field(repr=False)
    lifting_rate: dict
    price: dict

    # Attributes associated with project duration
    project_duration: int = field(repr=False)
    project_years: np.ndarray = field(repr=False)

    # Attributes associated with PSC transition
    type_of_contract: str = field(repr=False)
    end_date_project: datetime = field(repr=False)
    start_date_project_second: datetime = field(repr=False)

    # Attributes to be defined later
    prod_year: dict = field(default=None, init=False)

    def __post_init__(self, prod_year_init):
        # Prepare attribute prod_year
        if not isinstance(prod_year_init, dict):
            raise LPGPropaneLiftingDataException(
                f"Attribute prod_year must be provided in the form of dictionary. "
                f"The current datatype of prod_year is "
                f"{prod_year_init.__class__.__qualname__}"
            )

        for ws in prod_year_init.keys():
            if prod_year_init[ws] is None:
                prod_year_init[ws] = np.int_(self.project_years)
            else:
                prod_year_init[ws] = np.int_(prod_year_init[ws])

        self.prod_year = prod_year_init.copy()

        # Prepare attribute lifting_rate
        if not isinstance(self.lifting_rate, dict):
            raise LPGPropaneLiftingDataException(
                f"Attribute lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of lifting_rate is "
                f"{self.lifting_rate.__class__.__qualname__}"
            )

        lifting_rate_nan = {}
        for ws in self.lifting_rate.keys():
            if self.lifting_rate[ws] is None:
                self.lifting_rate[ws] = np.zeros_like(self.project_years, dtype=np.float_)
                lifting_rate_nan[ws] = None
            else:
                lifting_rate_nan[ws] = np.argwhere(pd.isna(self.lifting_rate[ws])).ravel()
                if len(lifting_rate_nan[ws]) > 0:
                    self.lifting_rate[ws][lifting_rate_nan[ws]] = (
                        np.zeros(len(lifting_rate_nan[ws]), dtype=np.float_)
                    )
                else:
                    self.lifting_rate[ws] = np.float_(self.lifting_rate[ws])

        # Prepare attribute price
        if not isinstance(self.price, dict):
            raise LPGPropaneLiftingDataException(
                f"Attribute price must be provided in the form of dictionary. "
                f"The current datatype of price is "
                f"{self.price.__class__.__qualname__}"
            )

        price_nan = {}
        for ws in self.price.keys():
            if self.price[ws] is None:
                self.price[ws] = np.zeros_like(self.project_years, dtype=np.float_)
                price_nan[ws] = None
            else:
                price_nan[ws] = np.argwhere(pd.isna(self.price[ws])).ravel()
                if len(price_nan[ws]) > 0:
                    self.price[ws][price_nan[ws]] = np.zeros(len(price_nan[ws]), dtype=np.float_)
                else:
                    self.price[ws] = np.float_(self.price[ws])

        # Adjust data for transition case
        if "Transition" in self.type_of_contract:
            # Modify attributes "prod_year", "lifting_rate", "price",
            target_attrs = {
                "attr": [self.prod_year, self.lifting_rate, self.price],
                "status": [False, True, False],
            }

            (
                self.prod_year,
                self.lifting_rate,
                self.price,
            ) = [
                get_lifting_data_split_simple(
                    target_attr=i,
                    is_target_attr_volume=j,
                    prod_year_init=prod_year_init,
                    end_date_contract_1=self.end_date_project,
                    start_date_contract_2=self.start_date_project_second,
                )
                for i, j in zip(target_attrs["attr"], target_attrs["status"])
            ]


@dataclass
class LPGButaneLiftingData:
    """
    A dataclass representing LPG Butane lifting information for an oil and gas economic project.

    Attributes
    ----------
    prod_year_init: dict
        Dictionary containing production years data.
    lifting_rate: dict
        Dictionary containing LPG Butane lifting rate data.
    price: dict
        Dictionary containing LPG Butane price data.
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    type_of_contract: str
        A string depicting the type of PSC contract.
    end_date_project: date
        The end date of the project.
    start_date_project_second: date
        The start date of the second project (for PSC transition).
    """
    prod_year_init: InitVar[dict] = field(repr=False)
    lifting_rate: dict
    price: dict

    # Attributes associated with project duration
    project_duration: int = field(repr=False)
    project_years: np.ndarray = field(repr=False)

    # Attributes associated with PSC transition
    type_of_contract: str = field(repr=False)
    end_date_project: datetime = field(repr=False)
    start_date_project_second: datetime = field(repr=False)

    # Attributes to be defined later
    prod_year: dict = field(default=None, init=False)

    def __post_init__(self, prod_year_init):
        # Prepare attribute prod_year
        if not isinstance(prod_year_init, dict):
            raise LPGButaneLiftingDataException(
                f"Attribute prod_year must be provided in the form of dictionary. "
                f"The current datatype of prod_year is "
                f"{prod_year_init.__class__.__qualname__}"
            )

        for ws in prod_year_init.keys():
            if prod_year_init[ws] is None:
                prod_year_init[ws] = np.int_(self.project_years)
            else:
                prod_year_init[ws] = np.int_(prod_year_init[ws])

        self.prod_year = prod_year_init.copy()

        # Prepare attribute lifting_rate
        if not isinstance(self.lifting_rate, dict):
            raise LPGButaneLiftingDataException(
                f"Attribute lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of lifting_rate is "
                f"{self.lifting_rate.__class__.__qualname__}"
            )

        lifting_rate_nan = {}
        for ws in self.lifting_rate.keys():
            if self.lifting_rate[ws] is None:
                self.lifting_rate[ws] = np.zeros_like(self.project_years, np.float_)
                lifting_rate_nan[ws] = None
            else:
                lifting_rate_nan[ws] = np.argwhere(pd.isna(self.lifting_rate[ws])).ravel()
                if len(lifting_rate_nan[ws]) > 0:
                    self.lifting_rate[ws][lifting_rate_nan[ws]] = (
                        np.zeros(len(lifting_rate_nan[ws]), dtype=np.float_)
                    )
                else:
                    self.lifting_rate[ws] = np.float_(self.lifting_rate[ws])

        # Prepare attribute price
        if not isinstance(self.price, dict):
            raise LPGButaneLiftingDataException(
                f"Attribute price must be provided in the form of dictionary. "
                f"The current datatype of price is "
                f"{self.price.__class__.__qualname__}"
            )

        price_nan = {}
        for ws in self.price.keys():
            if self.price[ws] is None:
                self.price[ws] = np.zeros_like(self.project_years, np.float_)
                price_nan[ws] = None
            else:
                price_nan[ws] = np.argwhere(pd.isna(self.price[ws])).ravel()
                if len(price_nan[ws]) > 0:
                    self.price[ws][price_nan[ws]] = np.zeros(len(price_nan[ws]), dtype=np.float_)
                else:
                    self.price[ws] = np.float_(self.price[ws])

        # Adjust data for transition case
        if "Transition" in self.type_of_contract:
            target_attrs = {
                "attr": [self.prod_year, self.lifting_rate, self.price],
                "status": [False, True, False],
            }

            (
                self.prod_year,
                self.lifting_rate,
                self.price,
            ) = [
                get_lifting_data_split_simple(
                    target_attr=i,
                    is_target_attr_volume=j,
                    prod_year_init=prod_year_init,
                    end_date_contract_1=self.end_date_project,
                    start_date_contract_2=self.start_date_project_second,
                )
                for i, j in zip(target_attrs["attr"], target_attrs["status"])
            ]


@dataclass
class SulfurLiftingData:
    """
    A dataclass representing sulfur lifting information for an oil and gas economic project.

    Attributes
    ----------
    prod_year_init: dict
        Dictionary containing production years data.
    lifting_rate: dict
        Dictionary containing sulfur lifting rate data.
    price: dict
        Dictionary containing sulfur price data.
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    type_of_contract: str
        A string depicting the type of PSC contract.
    end_date_project: date
        The end date of the project.
    start_date_project_second: date
        The start date of the second project (for PSC transition).
    """
    prod_year_init: InitVar[dict] = field(repr=False)
    lifting_rate: dict
    price: dict

    # Attributes associated with project duration
    project_duration: int = field(repr=False)
    project_years: np.ndarray = field(repr=False)

    # Attributes associated with PSC transition
    type_of_contract: str = field(repr=False)
    end_date_project: datetime = field(repr=False)
    start_date_project_second: datetime = field(repr=False)

    # Attributes to be defined later
    prod_year: dict = field(default=None, init=False)

    def __post_init__(self, prod_year_init):
        # Prepare attribute prod_year
        if not isinstance(prod_year_init, dict):
            raise SulfurLiftingDataException(
                f"Attribute prod_year must be provided in the form of dictionary. "
                f"The current datatype of prod_year is "
                f"{prod_year_init.__class__.__qualname__}"
            )

        for ws in prod_year_init.keys():
            if prod_year_init[ws] is None:
                prod_year_init[ws] = np.int_(self.project_years)
            else:
                prod_year_init[ws] = np.int_(prod_year_init[ws])

        self.prod_year = prod_year_init.copy()

        # Prepare attribute lifting_rate
        if not isinstance(self.lifting_rate, dict):
            raise SulfurLiftingDataException(
                f"Attribute lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of lifting_rate is "
                f"{self.lifting_rate.__class__.__qualname__}"
            )

        lifting_rate_nan = {}
        for ws in self.lifting_rate.keys():
            if self.lifting_rate[ws] is None:
                self.lifting_rate[ws] = np.zeros_like(self.project_years, dtype=np.float_)
                lifting_rate_nan[ws] = None
            else:
                lifting_rate_nan[ws] = np.argwhere(pd.isna(self.lifting_rate[ws])).ravel()
                if len(lifting_rate_nan[ws]) > 0:
                    self.lifting_rate[ws][lifting_rate_nan[ws]] = (
                        np.zeros(len(lifting_rate_nan[ws]), dtype=np.float_)
                    )
                else:
                    self.lifting_rate[ws] = np.float_(self.lifting_rate[ws])

        # Prepare attribute price
        if not isinstance(self.price, dict):
            raise SulfurLiftingDataException(
                f"Attribute price must be provided in the form of dictionary. "
                f"The current datatype of price is "
                f"{self.price.__class__.__qualname__}"
            )

        price_nan = {}
        for ws in self.price.keys():
            if self.price[ws] is None:
                self.price[ws] = np.zeros_like(self.project_years, dtype=np.float_)
                price_nan[ws] = None
            else:
                price_nan[ws] = np.argwhere(pd.isna(self.price[ws])).ravel()
                if len(price_nan[ws]) > 0:
                    self.price[ws][price_nan[ws]] = np.zeros(len(price_nan[ws]), dtype=np.float_)
                else:
                    self.price[ws] = np.float_(self.price[ws])

        # Adjust data for transition case
        if "Transition" in self.type_of_contract:
            target_attrs = {
                "attr": [self.prod_year, self.lifting_rate, self.price],
                "status": [False, True, False],
            }

            (
                self.prod_year,
                self.lifting_rate,
                self.price,
            ) = [
                get_lifting_data_split_simple(
                    target_attr=i,
                    is_target_attr_volume=j,
                    prod_year_init=prod_year_init,
                    end_date_contract_1=self.end_date_project,
                    start_date_contract_2=self.start_date_project_second,
                )
                for i, j in zip(target_attrs["attr"], target_attrs["status"])
            ]


@dataclass
class ElectricityLiftingData:
    """
    A dataclass representing electricity lifting information for an oil and gas economic project.

    Attributes
    ----------
    prod_year_init: dict
        Dictionary containing production years data.
    lifting_rate: dict
        Dictionary containing electricity lifting rate data.
    price: dict
        Dictionary containing electricity price data.
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    type_of_contract: str
        A string depicting the type of PSC contract.
    end_date_project: date
        The end date of the project.
    start_date_project_second: date
        The start date of the second project (for PSC transition).
    """
    prod_year_init: InitVar[dict] = field(repr=False)
    lifting_rate: dict
    price: dict

    # Attributes associated with project duration
    project_duration: int = field(repr=False)
    project_years: np.ndarray = field(repr=False)

    # Attributes associated with PSC transition
    type_of_contract: str = field(repr=False)
    end_date_project: datetime = field(repr=False)
    start_date_project_second: datetime = field(repr=False)

    # Attributes to be defined later
    prod_year: dict = field(default=None, init=False)

    def __post_init__(self, prod_year_init):
        # Prepare attribute prod_year
        if not isinstance(prod_year_init, dict):
            raise ElectricityLiftingDataException(
                f"Attribute prod_year must be provided in the form of dictionary. "
                f"The current datatype of prod_year is "
                f"{prod_year_init.__class__.__qualname__}"
            )

        for ws in prod_year_init.keys():
            if prod_year_init[ws] is None:
                prod_year_init[ws] = np.int_(self.project_years)
            else:
                prod_year_init[ws] = np.int_(prod_year_init[ws])

        self.prod_year = prod_year_init.copy()

        # Prepare attribute lifting_rate
        if not isinstance(self.lifting_rate, dict):
            raise ElectricityLiftingDataException(
                f"Attribute lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of lifting_rate is "
                f"{self.lifting_rate.__class__.__qualname__}"
            )

        lifting_rate_nan = {}
        for ws in self.lifting_rate.keys():
            if self.lifting_rate[ws] is None:
                self.lifting_rate[ws] = np.zeros_like(self.project_years, dtype=np.float_)
                lifting_rate_nan[ws] = None
            else:
                lifting_rate_nan[ws] = np.argwhere(pd.isna(self.lifting_rate[ws])).ravel()
                if len(lifting_rate_nan[ws]) > 0:
                    self.lifting_rate[ws][lifting_rate_nan[ws]] = (
                        np.zeros(len(lifting_rate_nan[ws]), dtype=np.float_)
                    )
                else:
                    self.lifting_rate[ws] = np.float_(self.lifting_rate[ws])

        # Prepare attribute price
        if not isinstance(self.price, dict):
            raise ElectricityLiftingDataException(
                f"Attribute price must be provided in the form of dictionary. "
                f"The current datatype of price is "
                f"{self.price.__class__.__qualname__}"
            )

        price_nan = {}
        for ws in self.price.keys():
            if self.price[ws] is None:
                self.price[ws] = np.zeros_like(self.project_years, dtype=np.float_)
                price_nan[ws] = None
            else:
                price_nan[ws] = np.argwhere(pd.isna(self.price[ws])).ravel()
                if len(price_nan[ws]) > 0:
                    self.price[ws][price_nan[ws]] = np.zeros(len(price_nan[ws]), dtype=np.float_)
                else:
                    self.price[ws] = np.float_(self.price[ws])

        # Adjust data for transition case
        if "Transition" in self.type_of_contract:
            target_attrs = {
                "attr": [self.prod_year, self.lifting_rate, self.price],
                "status": [False, True, False],
            }

            (
                self.prod_year,
                self.lifting_rate,
                self.price,
            ) = [
                get_lifting_data_split_simple(
                    target_attr=i,
                    is_target_attr_volume=j,
                    prod_year_init=prod_year_init,
                    end_date_contract_1=self.end_date_project,
                    start_date_contract_2=self.start_date_project_second,
                )
                for i, j in zip(target_attrs["attr"], target_attrs["status"])
            ]


@dataclass
class CO2LiftingData:
    """
    A dataclass representing CO2 lifting information for an oil and gas economic project.

    Attributes
    ----------
    prod_year_init: dict
        Dictionary containing production years data.
    lifting_rate: dict
        Dictionary containing CO2 lifting rate data.
    price: dict
        Dictionary containing CO2 price data.
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    type_of_contract: str
        A string depicting the type of PSC contract.
    end_date_project: date
        The end date of the project.
    start_date_project_second: date
        The start date of the second project (for PSC transition).
    """
    prod_year_init: InitVar[dict] = field(repr=False)
    lifting_rate: dict
    price: dict

    # Attributes associated with project duration
    project_duration: int = field(repr=False)
    project_years: np.ndarray = field(repr=False)

    # Attributes associated with PSC transition
    type_of_contract: str = field(repr=False)
    end_date_project: datetime = field(repr=False)
    start_date_project_second: datetime = field(repr=False)

    # Attributes to be defined later
    prod_year: dict = field(default=None, init=False)

    def __post_init__(self, prod_year_init):
        # Prepare attribute prod_year
        if not isinstance(prod_year_init, dict):
            raise CO2LiftingDataException(
                f"Attribute prod_year must be provided in the form of dictionary. "
                f"The current datatype of prod_year is "
                f"{prod_year_init.__class__.__qualname__}"
            )

        for ws in prod_year_init.keys():
            if prod_year_init[ws] is None:
                prod_year_init[ws] = np.int_(self.project_years)
            else:
                prod_year_init[ws] = np.int_(prod_year_init[ws])

        self.prod_year = prod_year_init.copy()

        # Prepare attribute lifting_rate
        if not isinstance(self.lifting_rate, dict):
            raise CO2LiftingDataException(
                f"Attribute lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of lifting_rate is "
                f"{self.lifting_rate.__class__.__qualname__}"
            )

        lifting_rate_nan = {}
        for ws in self.lifting_rate.keys():
            if self.lifting_rate[ws] is None:
                self.lifting_rate[ws] = np.zeros_like(self.project_years, dtype=np.float_)
                lifting_rate_nan[ws] = None
            else:
                lifting_rate_nan[ws] = np.argwhere(pd.isna(self.lifting_rate[ws])).ravel()
                if len(lifting_rate_nan[ws]) > 0:
                    self.lifting_rate[ws][lifting_rate_nan[ws]] = (
                        np.zeros(len(lifting_rate_nan[ws]), dtype=np.float_)
                    )
                else:
                    self.lifting_rate[ws] = np.float_(self.lifting_rate[ws])

        # Prepare attribute price
        if not isinstance(self.price, dict):
            raise CO2LiftingDataException(
                f"Attribute price must be provided in the form of dictionary. "
                f"The current datatype of price is "
                f"{self.price.__class__.__qualname__}"
            )

        price_nan = {}
        for ws in self.price.keys():
            if self.price[ws] is None:
                self.price[ws] = np.zeros_like(self.project_years, dtype=np.float_)
                price_nan[ws] = None
            else:
                price_nan[ws] = np.argwhere(pd.isna(self.price[ws])).ravel()
                if len(price_nan[ws]) > 0:
                    self.price[ws][price_nan[ws]] = np.zeros(len(price_nan[ws]), dtype=np.float_)
                else:
                    self.price[ws] = np.float_(self.price[ws])

        # Adjust data for transition case
        if "Transition" in self.type_of_contract:
            target_attrs = {
                "attr": [self.prod_year, self.lifting_rate, self.price],
                "status": [False, True, False],
            }

            (
                self.prod_year,
                self.lifting_rate,
                self.price,
            ) = [
                get_lifting_data_split_simple(
                    target_attr=i,
                    is_target_attr_volume=j,
                    prod_year_init=prod_year_init,
                    end_date_contract_1=self.end_date_project,
                    start_date_contract_2=self.start_date_project_second,
                )
                for i, j in zip(target_attrs["attr"], target_attrs["status"])
            ]


@dataclass
class TangibleCostData:
    """
    A dataclass representing tangible cost information for an oil and gas economic project.

    Attributes
    ----------
    expense_year: np.ndarray
        Array of expense years.
    cost: np.ndarray
        Array of costs.
    cost_allocation: list
        List of cost allocations.
    pis_year: np.ndarray
        Array of PIS years.
    useful_life: np.ndarray
        Array of useful life.
    depreciation_factor: np.ndarray
        Array of depreciation factors.
    salvage_value: np.ndarray
        Array of salvage values.
    is_ic_applied: list
        List indicating whether IC (Investment Credit) is applied.
    vat_portion: np.ndarray
        Array of VAT (Value Added Tax) portions.
    lbt_portion: np.ndarray
        Array of LBT (Land and Building Tax) portions.
    description: list
        List of descriptions.
    data_length: int
        Length of the captured data.
    project_years: np.ndarray
        Array of project years.
    type_of_contract: str
        A string depicting the type of PSC contract.
    end_date_project: date
        The end date of the project.
    start_date_project_second: date
        The start date of the second project (for PSC transition).
    """
    expense_year_init: InitVar[np.ndarray] = field(repr=False)
    cost: np.ndarray
    cost_allocation: list
    pis_year: np.ndarray
    useful_life: np.ndarray
    depreciation_factor: np.ndarray
    salvage_value: np.ndarray
    is_ic_applied: list
    vat_portion: np.ndarray
    lbt_portion: np.ndarray
    description: list

    # Attribute associated with project duration
    project_years: np.ndarray = field(repr=False)

    # Attributes associated with PSC transition
    type_of_contract: str = field(repr=False)
    end_date_project: datetime = field(repr=False)
    start_date_project_second: datetime = field(repr=False)

    # Attributes to be defined later
    expense_year: np.ndarray = field(default=None, init=False)

    def __post_init__(self, expense_year_init):
        # Prepare attribute expense_year
        if not isinstance(expense_year_init, np.ndarray):
            raise TangibleCostDataException(
                f"Expense year data must be given in the form of numpy.ndarray. "
                f"The current expense_year data is given in the form of: "
                f"({expense_year_init.__class__.__qualname__})."
            )

        expense_year_init = np.int_(expense_year_init)
        self.expense_year = expense_year_init.copy()

        # Prepare attribute cost_allocation
        if not isinstance(self.cost_allocation, list):
            raise TangibleCostDataException(
                f"Cost allocation data must be given in the form of list. "
                f"The current cost_allocation data is given in the form of: "
                f"({self.cost_allocation.__class__.__qualname__})."
            )

        cost_allocation_nan = np.argwhere(pd.isna(self.cost_allocation)).ravel()
        if len(cost_allocation_nan) > 0:
            self.cost_allocation = np.array(self.cost_allocation)
            self.cost_allocation[cost_allocation_nan] = ["Oil" for i in range(len(cost_allocation_nan))]
            self.cost_allocation = [get_fluidtype_converter(target=i) for i in self.cost_allocation]
        else:
            self.cost_allocation = [get_fluidtype_converter(target=i) for i in self.cost_allocation]

        # Prepare attribute cost
        if not isinstance(self.cost, np.ndarray):
            raise TangibleCostDataException(
                f"Cost data must be given in the form of numpy.ndarray. "
                f"The current cost data is given in the form of: "
                f"({self.cost.__class__.__qualname__})."
            )

        cost_nan = np.argwhere(pd.isna(self.cost)).ravel()
        if len(cost_nan) > 0:
            self.cost[cost_nan] = np.zeros(len(cost_nan), dtype=np.float_)
        else:
            self.cost = np.float_(self.cost)

        print('\t')
        print(f'Filetype: {type(self.cost)}')
        print(f'Length: {len(self.cost)}')
        print('cost = ', self.cost)

    #     # Prepare attribute pis_year
    #     if not isinstance(self.pis_year, np.ndarray):
    #         raise TangibleCostDataException(
    #             f"PIS year data must be given in the form of numpy.ndarray. "
    #             f"The current pis_year data is given in the form of: "
    #             f"({self.pis_year.__class__.__qualname__})."
    #         )
    #
    #     pis_year_nan = np.argwhere(np.isnan(self.pis_year)).ravel()
    #
    #     if len(pis_year_nan) > 0:
    #         self.pis_year[pis_year_nan] = expense_year_init[pis_year_nan]
    #
    #     else:
    #         self.pis_year = self.pis_year
    #
    #     # Prepare attribute useful_life
    #     if not isinstance(self.useful_life, np.ndarray):
    #         raise TangibleCostDataException(
    #             f"Useful life data must be given in the form of numpy.ndarray. "
    #             f"The current useful_life data is given in the form of: "
    #             f"({self.useful_life.__class__.__qualname__})."
    #         )
    #
    #     useful_life_nan = np.argwhere(np.isnan(self.useful_life)).ravel()
    #
    #     if len(useful_life_nan) > 0:
    #         self.useful_life[useful_life_nan] = np.repeat(5.0, len(useful_life_nan))
    #
    #     else:
    #         self.useful_life = np.float_(self.useful_life)
    #
    #     # Prepare attribute depreciation_factor
    #     if not isinstance(self.depreciation_factor, np.ndarray):
    #         raise TangibleCostDataException(
    #             f"Depreciation factor data must be given in the form of numpy.ndarray. "
    #             f"The current depreciation_factor data is given in the form of: "
    #             f"({self.depreciation_factor.__class__.__qualname__})."
    #         )
    #
    #     depreciation_factor_nan = np.argwhere(np.isnan(self.depreciation_factor)).ravel()
    #
    #     if len(depreciation_factor_nan) > 0:
    #         self.depreciation_factor[depreciation_factor_nan] = (
    #             np.repeat(0.5, len(depreciation_factor_nan))
    #         )
    #
    #     else:
    #         self.depreciation_factor = np.float_(self.depreciation_factor)
    #
    #     # Prepare attribute salvage_value
    #     if not isinstance(self.salvage_value, np.ndarray):
    #         raise TangibleCostDataException(
    #             f"Salvage value data must be given in the form of numpy.ndarray. "
    #             f"The current salvage_value data is given in the form of: "
    #             f"({self.salvage_value.__class__.__qualname__})."
    #         )
    #
    #     salvage_value_nan = np.argwhere(np.isnan(self.salvage_value)).ravel()
    #
    #     if len(salvage_value_nan) > 0:
    #         self.salvage_value[salvage_value_nan] = np.zeros(len(salvage_value_nan), dtype=np.float_)
    #
    #     else:
    #         self.salvage_value = np.float_(self.salvage_value)
    #
    #     # Prepare attribute is_ic_applied
    #     if not isinstance(self.is_ic_applied, list):
    #         raise TangibleCostDataException(
    #             f"Is IC applied data must be given in the form of list. "
    #             f"The current is_ic_applied data is given in the form of: "
    #             f"({self.is_ic_applied.__class__.__qualname__})."
    #         )
    #
    #     is_ic_applied_nan = np.argwhere(pd.isna(self.is_ic_applied)).ravel()
    #
    #     if len(is_ic_applied_nan) > 0:
    #         for i in is_ic_applied_nan:
    #             self.is_ic_applied[i] = "No"
    #
    #     else:
    #         self.is_ic_applied = self.is_ic_applied
    #
    #     self.is_ic_applied = [get_boolean_converter(target=i) for i in self.is_ic_applied]
    #
    #     # Prepare attribute vat_portion
    #     if not isinstance(self.vat_portion, np.ndarray):
    #         raise TangibleCostDataException(
    #             f"VAT portion data must be given in the form of numpy.ndarray. "
    #             f"The current vat_portion data is given in the form of: "
    #             f"({self.vat_portion.__class__.__qualname__})."
    #         )
    #
    #     vat_portion_nan = np.argwhere(np.isnan(self.vat_portion)).ravel()
    #
    #     if len(vat_portion_nan) > 0:
    #         self.vat_portion[vat_portion_nan] = np.ones(len(vat_portion_nan), dtype=np.float_)
    #
    #     else:
    #         self.vat_portion = np.float_(self.vat_portion)
    #
    #     # Prepare attribute lbt_portion
    #     if not isinstance(self.lbt_portion, np.ndarray):
    #         raise TangibleCostDataException(
    #             f"LBT portion data must be given in the form of numpy.ndarray. "
    #             f"The current lbt_portion data is given in the form of: "
    #             f"({self.lbt_portion.__class__.__qualname__})."
    #         )
    #
    #     lbt_portion_nan = np.argwhere(np.isnan(self.lbt_portion)).ravel()
    #
    #     if len(lbt_portion_nan) > 0:
    #         self.lbt_portion[lbt_portion_nan] = np.ones(len(lbt_portion_nan), dtype=np.float_)
    #
    #     else:
    #         self.lbt_portion = np.float_(self.lbt_portion)
    #
    #     # Prepare attribute description
    #     if not isinstance(self.description, list):
    #         raise TangibleCostDataException(
    #             f"Description data must be given in the form of list. "
    #             f"The current description data is given in the form of: "
    #             f"({self.description.__class__.__qualname__})."
    #         )
    #
    #     description_nan = np.argwhere(pd.isna(self.description)).ravel()
    #
    #     if len(description_nan) > 0:
    #         for i in description_nan:
    #             self.description[i] = " "
    #
    #     else:
    #         self.description = self.description
    #
    #     # Adjust data for transition case
    #     if "Transition" in self.type_of_contract:
    #         # Modify the values of the attributes
    #         target_attrs = {
    #             "attr": [
    #                 self.expense_year,
    #                 self.cost,
    #                 self.cost_allocation,
    #                 self.pis_year,
    #                 self.useful_life,
    #                 self.depreciation_factor,
    #                 self.salvage_value,
    #                 self.is_ic_applied,
    #                 self.vat_portion,
    #                 self.lbt_portion,
    #                 self.description,
    #             ],
    #             "status": [
    #                 False,
    #                 True,
    #                 False,
    #                 False,
    #                 False,
    #                 False,
    #                 False,
    #                 False,
    #                 False,
    #                 False,
    #                 False,
    #             ],
    #         }
    #
    #         (
    #             self.expense_year,
    #             self.cost,
    #             self.cost_allocation,
    #             self.pis_year,
    #             self.useful_life,
    #             self.depreciation_factor,
    #             self.salvage_value,
    #             self.is_ic_applied,
    #             self.vat_portion,
    #             self.lbt_portion,
    #             self.description,
    #         ) = [
    #             get_cost_data_split(
    #                 target_attr=i,
    #                 is_target_attr_volume=j,
    #                 expense_year_init=expense_year_init,
    #                 end_date_contract_1=self.end_date_project,
    #                 start_date_contract_2=self.start_date_project_second,
    #             )
    #             for i, j in zip(target_attrs["attr"], target_attrs["status"])
    #         ]
    #
    #         # Prepare attributes with datatype list
    #         (
    #             self.cost_allocation,
    #             self.is_ic_applied,
    #             self.description,
    #         ) = list(
    #             map(get_to_list_converter, [self.cost_allocation, self.is_ic_applied, self.description])
    #         )


@dataclass
class IntangibleCostData:
    """
    A dataclass representing intangible cost information for an oil and gas economic project.

    Attributes
    ----------
    expense_year: np.ndarray
        Array of expense years.
    cost_allocation: list
        List of cost allocations.
    cost: np.ndarray
        Array of costs.
    vat_portion: np.ndarray
        Array of VAT (Value Added Tax) portions.
    lbt_portion: np.ndarray
        Array of LBT (Land and Building Tax) portions.
    description: list
        List of descriptions.
    data_length: int
        Length of the captured data.
    project_years: np.ndarray
        Array of project years.
    type_of_contract: str
        A string depicting the type of PSC contract.
    end_date_project: date
        The end date of the project.
    start_date_project_second: date
        The start date of the second project (for PSC transition).
    """
    expense_year_init: InitVar[np.ndarray] = field(repr=False)
    cost_allocation: list
    cost: np.ndarray
    vat_portion: np.ndarray
    lbt_portion: np.ndarray
    description: list

    # Attribute associated with length of the captured data
    data_length: int = field(repr=False)

    # Attribute associated with project duration
    project_years: np.ndarray = field(repr=False)

    # Attributes associated with PSC transition
    type_of_contract: str = field(repr=False)
    end_date_project: datetime = field(repr=False)
    start_date_project_second: datetime = field(repr=False)

    # Attributes to be defined later
    expense_year: np.ndarray = field(default=None, init=False)

    def __post_init__(self, expense_year_init):
        # Prepare attribute expense_year
        if not isinstance(expense_year_init, np.ndarray):
            raise IntangibleCostDataException(
                f"Expense year data must be given in the form of numpy.ndarray. "
                f"The current expense_year data is given in the form of: "
                f"({expense_year_init.__class__.__qualname__})."
            )

        expense_year_nan = np.argwhere(np.isnan(expense_year_init)).ravel()

        if len(expense_year_nan) > 0:
            raise IntangibleCostDataException(
                f"Expense year data is incomplete. Please re-check the expense_year data. "
                f"The number of expense_year data must be ({self.data_length}), "
                f"not ({self.data_length - len(expense_year_nan)})."
            )

        else:
            expense_year_init = np.int_(expense_year_init)
            self.expense_year = expense_year_init.copy()

        # Prepare attribute cost_allocation
        if not isinstance(self.cost_allocation, list):
            raise IntangibleCostDataException(
                f"Cost allocation data must be given in the form of list. "
                f"The current cost_allocation data is given in the form of: "
                f"({self.cost_allocation.__class__.__qualname__})."
            )

        cost_allocation_nan = np.argwhere(pd.isna(self.cost_allocation)).ravel()

        if len(cost_allocation_nan) > 0:
            raise IntangibleCostDataException(
                f"Cost allocation data is incomplete. "
                f"Please re-check cost_allocation data. "
                f"The number of cost_allocation data must be ({self.data_length}), "
                f"not ({self.data_length - len(cost_allocation_nan)})."
            )

        else:
            self.cost_allocation = [get_fluidtype_converter(target=i) for i in self.cost_allocation]

        # Prepare attribute cost
        if not isinstance(self.cost, np.ndarray):
            raise IntangibleCostDataException(
                f"Cost data must be given in the form of numpy.ndarray. "
                f"The current cost data is given in the form of: "
                f"({self.cost.__class__.__qualname__})."
            )

        cost_nan = np.argwhere(np.isnan(self.cost)).ravel()

        if len(cost_nan) > 0:
            self.cost[cost_nan] = np.zeros(len(cost_nan), dtype=np.float_)

        else:
            self.cost = np.float_(self.cost)

        # Prepare attribute vat_portion
        if not isinstance(self.vat_portion, np.ndarray):
            raise IntangibleCostDataException(
                f"VAT portion data must be given in the form of numpy.ndarray. "
                f"The current vat_portion data is given in the form of: "
                f"({self.vat_portion.__class__.__qualname__})."
            )

        vat_portion_nan = np.argwhere(np.isnan(self.vat_portion)).ravel()

        if len(vat_portion_nan) > 0:
            self.vat_portion[vat_portion_nan] = np.ones(len(vat_portion_nan), dtype=np.float_)

        else:
            self.vat_portion = np.float_(self.vat_portion)

        # Prepare attribute lbt_portion
        if not isinstance(self.lbt_portion, np.ndarray):
            raise IntangibleCostDataException(
                f"LBT portion data must be given in the form of numpy.ndarray. "
                f"The current lbt_portion data is given in the form of: "
                f"({self.lbt_portion.__class__.__qualname__})."
            )

        lbt_portion_nan = np.argwhere(np.isnan(self.lbt_portion)).ravel()

        if len(lbt_portion_nan) > 0:
            self.lbt_portion[lbt_portion_nan] = np.ones(len(lbt_portion_nan), dtype=np.float_)

        else:
            self.lbt_portion = np.float_(self.lbt_portion)

        # Prepare attribute description
        if not isinstance(self.description, list):
            raise IntangibleCostDataException(
                f"Description data must be given in the form of list. "
                f"The current description data is given in the form of: "
                f"({self.description.__class__.__qualname__})."
            )

        description_nan = np.argwhere(pd.isna(self.description)).ravel()

        if len(description_nan) > 0:
            for i in description_nan:
                self.description[i] = " "

        else:
            self.description = self.description

        # Adjust data for transition case
        if "Transition" in self.type_of_contract:
            # Modify the values of the attributes
            target_attrs = {
                "attr": [
                    self.expense_year,
                    self.cost_allocation,
                    self.cost,
                    self.vat_portion,
                    self.lbt_portion,
                    self.description,
                ],
                "status": [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                ],
            }

            (
                self.expense_year,
                self.cost_allocation,
                self.cost,
                self.vat_portion,
                self.lbt_portion,
                self.description,
            ) = [
                get_cost_data_split(
                    target_attr=i,
                    is_target_attr_volume=j,
                    expense_year_init=expense_year_init,
                    end_date_contract_1=self.end_date_project,
                    start_date_contract_2=self.start_date_project_second,
                )
                for i, j in zip(target_attrs["attr"], target_attrs["status"])
            ]

            # Prepare attributes with datatype list
            self.cost_allocation, self.description = list(
                map(get_to_list_converter, [self.cost_allocation, self.description])
            )


@dataclass
class OPEXData:
    """
    A dataclass representing opex information for an oil and gas economic project.

    Attributes
    ----------
    expense_year: np.ndarray
        Array of expense years.
    cost_allocation: list
        List of cost allocations.
    fixed_cost: np.ndarray
        Array of fixed costs.
    prod_rate: np.ndarray
        Array of production rates.
    cost_per_volume: np.ndarray
        Array of costs per volume.
    vat_portion: np.ndarray
        Array of VAT (Value Added Tax) portions.
    lbt_portion: np.ndarray
        Array of LBT (Land and Building Tax) portions.
    description: list
        List of descriptions.
    data_length: int
        Length of the captured data.
    project_years: np.ndarray
        Array of project years.
    type_of_contract: str
        A string depicting the type of PSC contract.
    end_date_project: date
        The end date of the project.
    start_date_project_second: date
        The start date of the second project (for PSC transition).
    """
    expense_year_init: InitVar[np.ndarray] = field(repr=False)
    cost_allocation: list
    fixed_cost: np.ndarray
    prod_rate: np.ndarray
    cost_per_volume: np.ndarray
    vat_portion: np.ndarray
    lbt_portion: np.ndarray
    description: list

    # Attribute associated with length of the captured data
    data_length: int = field(repr=False)

    # Attribute associated with project duration
    project_years: np.ndarray = field(repr=False)

    # Attributes associated with PSC transition
    type_of_contract: str = field(repr=False)
    end_date_project: datetime = field(repr=False)
    start_date_project_second: datetime = field(repr=False)

    # Attributes to be defined later
    expense_year: np.ndarray = field(default=None, init=False)

    def __post_init__(self, expense_year_init):
        # Prepare attribute expense year
        if not isinstance(expense_year_init, np.ndarray):
            raise OPEXDataException(
                f"Expense year data must be given in the form of numpy.ndarray. "
                f"The current expense_year data is given in the form of: "
                f"({expense_year_init.__class__.__qualname__})."
            )

        expense_year_nan = np.argwhere(np.isnan(expense_year_init)).ravel()

        if len(expense_year_nan) > 0:
            raise OPEXDataException(
                f"Expense year data is incomplete. Please re-check the expense_year data. "
                f"The number of expense_year data must be ({self.data_length}), "
                f"not ({self.data_length - len(expense_year_nan)})."
            )

        else:
            expense_year_init = np.int_(expense_year_init)
            self.expense_year = expense_year_init.copy()

        # Prepare attribute cost_allocation
        if not isinstance(self.cost_allocation, list):
            raise OPEXDataException(
                f"Cost allocation data must be given in the form of list. "
                f"The current cost_allocation data is given in the form of: "
                f"({self.cost_allocation.__class__.__qualname__})."
            )

        cost_allocation_nan = np.argwhere(pd.isna(self.cost_allocation)).ravel()

        if len(cost_allocation_nan) > 0:
            raise OPEXDataException(
                f"Cost allocation data is incomplete. "
                f"Please re-check cost_allocation data. "
                f"The number of cost_allocation data must be ({self.data_length}), "
                f"not ({self.data_length - len(cost_allocation_nan)})."
            )

        else:
            self.cost_allocation = [get_fluidtype_converter(target=i) for i in self.cost_allocation]

        # Prepare attribute fixed_cost
        if not isinstance(self.fixed_cost, np.ndarray):
            raise OPEXDataException(
                f"Fixed cost data must be given in the form of numpy.ndarray. "
                f"The current fixed_cost data is given in the form of: "
                f"({self.fixed_cost.__class__.__qualname__})."
            )

        fixed_cost_nan = np.argwhere(np.isnan(self.fixed_cost)).ravel()

        if len(fixed_cost_nan) > 0:
            self.fixed_cost[fixed_cost_nan] = np.zeros(len(fixed_cost_nan), dtype=np.float_)

        else:
            self.fixed_cost = np.float_(self.fixed_cost)

        # Prepare attribute prod_rate
        if not isinstance(self.prod_rate, np.ndarray):
            raise OPEXDataException(
                f"Production rate data must be given in the form of numpy.ndarray. "
                f"The current prod_rate data is given in the form of: "
                f"({self.prod_rate.__class__.__qualname__})."
            )

        prod_rate_nan = np.argwhere(np.isnan(self.prod_rate)).ravel()

        if len(prod_rate_nan) > 0:
            self.prod_rate[prod_rate_nan] = np.zeros(len(prod_rate_nan), dtype=np.float_)

        else:
            self.prod_rate = np.float_(self.prod_rate)

        # Prepare attribute cost_per_volume
        if not isinstance(self.cost_per_volume, np.ndarray):
            raise OPEXDataException(
                f"Cost per volume data must be given in the form of numpy.ndarray. "
                f"The current cost_per_volume data is given in the form of: "
                f"({self.cost_per_volume.__class__.__qualname__})."
            )

        cost_per_volume_nan = np.argwhere(np.isnan(self.cost_per_volume)).ravel()

        if len(cost_per_volume_nan) > 0:
            self.cost_per_volume[cost_per_volume_nan] = np.zeros(len(cost_per_volume_nan), dtype=np.float_)

        else:
            self.cost_per_volume = np.float_(self.cost_per_volume)

        # Prepare attribute vat_portion
        if not isinstance(self.vat_portion, np.ndarray):
            raise OPEXDataException(
                f"VAT portion data must be given in the form of numpy.ndarray. "
                f"The current vat_portion data is given in the form of: "
                f"({self.vat_portion.__class__.__qualname__})."
            )

        vat_portion_nan = np.argwhere(np.isnan(self.vat_portion)).ravel()

        if len(vat_portion_nan) > 0:
            self.vat_portion[vat_portion_nan] = np.ones(len(vat_portion_nan), dtype=np.float_)

        else:
            self.vat_portion = np.float_(self.vat_portion)

        # Prepare attribute lbt_portion
        if not isinstance(self.lbt_portion, np.ndarray):
            raise OPEXDataException(
                f"LBT portion data must be given in the form of numpy.ndarray. "
                f"The current lbt_portion data is given in the form of: "
                f"({self.lbt_portion.__class__.__qualname__})."
            )

        lbt_portion_nan = np.argwhere(np.isnan(self.lbt_portion)).ravel()

        if len(lbt_portion_nan) > 0:
            self.lbt_portion[lbt_portion_nan] = np.ones(len(lbt_portion_nan), dtype=np.float_)

        else:
            self.lbt_portion = np.float_(self.lbt_portion)

        # Prepare attribute description
        if not isinstance(self.description, list):
            raise OPEXDataException(
                f"Description data must be given in the form of list. "
                f"The current description data is given in the form of: "
                f"({self.description.__class__.__qualname__})."
            )

        description_nan = np.argwhere(pd.isna(self.description)).ravel()

        if len(description_nan) > 0:
            for i in description_nan:
                self.description[i] = " "

        else:
            self.description = self.description

        # Adjust data for transition case
        if "Transition" in self.type_of_contract:
            # Modify the values of the attributes
            target_attrs = {
                "attr": [
                    self.expense_year,
                    self.cost_allocation,
                    self.fixed_cost,
                    self.prod_rate,
                    self.cost_per_volume,
                    self.vat_portion,
                    self.lbt_portion,
                    self.description,
                ],
                "status": [
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False
                ]
            }

            (
                self.expense_year,
                self.cost_allocation,
                self.fixed_cost,
                self.prod_rate,
                self.cost_per_volume,
                self.vat_portion,
                self.lbt_portion,
                self.description,
            ) = [
                get_cost_data_split(
                    target_attr=i,
                    is_target_attr_volume=j,
                    expense_year_init=expense_year_init,
                    end_date_contract_1=self.end_date_project,
                    start_date_contract_2=self.start_date_project_second,
                )
                for i, j in zip(target_attrs["attr"], target_attrs["status"])
            ]

            # Prepare attributes with datatype list
            self.cost_allocation, self.description = list(
                map(get_to_list_converter, [self.cost_allocation, self.description])
            )


@dataclass
class ASRCostData:
    """
    A dataclass representing ASR cost information for an oil and gas economic project.

    Attributes
    ----------
    expense_year: np.ndarray
        Array of expense years.
    cost_allocation: list
        List of cost allocations.
    cost: np.ndarray
        Array of costs.
    vat_portion: np.ndarray
        Array of VAT (Value Added Tax) portions.
    lbt_portion: np.ndarray
        Array of LBT (Land and Building Tax) portions.
    description: list
        List of descriptions.
    data_length: int
        Length of the captured data.
    project_years: np.ndarray
        Array of project years.
    type_of_contract: str
        A string depicting the type of PSC contract.
    end_date_project: date
        The end date of the project.
    start_date_project_second: date
        The start date of the second project (for PSC transition).
    """
    expense_year_init: InitVar[np.ndarray] = field(repr=False)
    cost_allocation: list
    cost: np.ndarray
    vat_portion: np.ndarray
    lbt_portion: np.ndarray
    description: list

    # Attribute associated with length of the captured data
    data_length: int = field(repr=False)

    # Attribute associated with project duration
    project_years: np.ndarray = field(repr=False)

    # Attributes associated with PSC transition
    type_of_contract: str = field(repr=False)
    end_date_project: datetime = field(repr=False)
    start_date_project_second: datetime = field(repr=False)

    # Attributes to be defined later
    expense_year: np.ndarray = field(default=None, init=False)

    def __post_init__(self, expense_year_init):
        # Prepare attribute expense_year
        if not isinstance(expense_year_init, np.ndarray):
            raise ASRCostDataException(
                f"Expense year data must be given in the form of numpy.ndarray. "
                f"The current expense_year data is given in the form of: "
                f"({expense_year_init.__class__.__qualname__})."
            )

        expense_year_nan = np.argwhere(np.isnan(expense_year_init)).ravel()

        if len(expense_year_nan) > 0:
            raise ASRCostDataException(
                f"Expense year data is incomplete. Please re-check the expense_year data. "
                f"The number of expense_year data must be ({self.data_length}), "
                f"not ({self.data_length - len(expense_year_nan)})."
            )

        else:
            expense_year_init = np.int_(expense_year_init)
            self.expense_year = expense_year_init.copy()

        # Prepare attribute cost_allocation
        if not isinstance(self.cost_allocation, list):
            raise ASRCostDataException(
                f"Cost allocation data must be given in the form of list. "
                f"The current cost_allocation data is given in the form of: "
                f"({self.cost_allocation.__class__.__qualname__})."
            )

        cost_allocation_nan = np.argwhere(pd.isna(self.cost_allocation)).ravel()

        if len(cost_allocation_nan) > 0:
            raise ASRCostDataException(
                f"Cost allocation data is incomplete. "
                f"Please re-check cost_allocation data. "
                f"The number of cost_allocation data must be ({self.data_length}), "
                f"not ({self.data_length - len(cost_allocation_nan)})."
            )

        else:
            self.cost_allocation = [get_fluidtype_converter(target=i) for i in self.cost_allocation]

        # Prepare attribute cost
        if not isinstance(self.cost, np.ndarray):
            raise ASRCostDataException(
                f"Cost data must be given in the form of numpy.ndarray. "
                f"The current cost data is given in the form of: "
                f"({self.cost.__class__.__qualname__})."
            )

        cost_nan = np.argwhere(np.isnan(self.cost)).ravel()

        if len(cost_nan) > 0:
            self.cost[cost_nan] = np.zeros(len(cost_nan), dtype=np.float_)

        else:
            self.cost = np.float_(self.cost)

        # Prepare attribute vat_portion
        if not isinstance(self.vat_portion, np.ndarray):
            raise ASRCostDataException(
                f"VAT portion data must be given in the form of numpy.ndarray. "
                f"The current vat_portion data is given in the form of: "
                f"({self.vat_portion.__class__.__qualname__})."
            )

        vat_portion_nan = np.argwhere(np.isnan(self.vat_portion)).ravel()

        if len(vat_portion_nan) > 0:
            self.vat_portion[vat_portion_nan] = np.ones(len(vat_portion_nan), dtype=np.float_)

        else:
            self.vat_portion = np.float_(self.vat_portion)

        # Prepare attribute lbt_portion
        if not isinstance(self.lbt_portion, np.ndarray):
            raise ASRCostDataException(
                f"LBT portion data must be given in the form of numpy.ndarray. "
                f"The current lbt_portion data is given in the form of: "
                f"({self.lbt_portion.__class__.__qualname__})."
            )

        lbt_portion_nan = np.argwhere(np.isnan(self.lbt_portion)).ravel()

        if len(lbt_portion_nan) > 0:
            self.lbt_portion[lbt_portion_nan] = np.ones(len(lbt_portion_nan), dtype=np.float_)

        else:
            self.lbt_portion = np.float_(self.lbt_portion)

        # Prepare attribute description
        if not isinstance(self.description, list):
            raise ASRCostDataException(
                f"Description data must be given in the form of list. "
                f"The current description data is given in the form of: "
                f"({self.description.__class__.__qualname__})."
            )

        description_nan = np.argwhere(pd.isna(self.description)).ravel()

        if len(description_nan) > 0:
            for i in description_nan:
                self.description[i] = " "

        else:
            self.description = self.description

        # Adjust data for transition case
        if "Transition" in self.type_of_contract:
            # Modify the values of the attributes
            target_attrs = {
                "attr": [
                    self.expense_year,
                    self.cost_allocation,
                    self.cost,
                    self.vat_portion,
                    self.lbt_portion,
                    self.description,
                ],
                "status": [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                ]
            }

            (
                self.expense_year,
                self.cost_allocation,
                self.cost,
                self.vat_portion,
                self.lbt_portion,
                self.description,
            ) = [
                get_cost_data_split(
                    target_attr=i,
                    is_target_attr_volume=j,
                    expense_year_init=expense_year_init,
                    end_date_contract_1=self.end_date_project,
                    start_date_contract_2=self.start_date_project_second,
                )
                for i, j in zip(target_attrs["attr"], target_attrs["status"])
            ]

            # Prepare attributes with datatype list
            self.cost_allocation, self.description = list(
                map(get_to_list_converter, [self.cost_allocation, self.description])
            )


@dataclass
class PSCCostRecoveryData:
    """
    A dataclass representing attributes associated with Production Sharing Contract
    (PSC) cost recovery.

    Attributes
    ----------
    oil_ftp_availability: str
        Availability status of oil for FTP.
    oil_ftp_is_shared: str
        Shared status of oil for FTP.
    oil_ftp_portion: float
        Portion of oil for FTP.
    gas_ftp_availability: str
        Availability status of gas for FTP.
    gas_ftp_is_shared: str
        Shared status of gas for FTP.
    gas_ftp_portion: float
        Portion of gas for FTP.
    split_type: str
        Type of pretax split configuration.
    oil_ctr_pretax: float
        Pretax contribution of oil.
    gas_ctr_pretax: float
        Pretax contribution of gas.
    ic_availability: str
        Availability status of investment credit (IC).
    ic_oil: float
        Investment credit for oil.
    ic_gas: float
        Investment credit for gas.
    oil_cr_cap_rate: float
        Cap rate for oil cost recovery.
    gas_cr_cap_rate: float
        Cap rate for gas cost recovery.
    dmo_is_weighted: str
        Weighted status of general DMO.
    oil_dmo_holiday: str
        Holiday status for DMO oil.
    oil_dmo_period: int | float
        Period for DMO oil.
    oil_dmo_start_production: datetime
        Start production date for DMO oil.
    oil_dmo_volume: float
        Volume for DMO oil.
    oil_dmo_fee: float
        Fee for DMO oil.
    gas_dmo_holiday: str
        Holiday status for DMO gas.
    gas_dmo_period: int | float
        Period for DMO gas.
    gas_dmo_start_production: datetime
        Start production date for DMO gas.
    gas_dmo_volume: float
        Volume for DMO gas.
    gas_dmo_fee: float
        Fee for DMO gas.
    rc_split_init: InitVar[np.ndarray]
        Initial values for rc_split attributes.
    icp_sliding_scale_init: InitVar[np.ndarray]
        Initial values for icp_sliding_scale attributes.
    indicator_rc_split_sliding_scale_init: InitVar[np.ndarray]
        Initial values for indicator_rc_split_sliding_scale attributes.
    rc_split: dict
        Dictionary storing rc_split attributes (to be defined later).
    icp_sliding_scale: dict
        Dictionary storing icp_sliding_scale attributes (to be defined later).
    indicator_rc_split_sliding_scale: dict
        Dictionary storing indicator_rc_split_sliding_scale attributes (to be defined later).

    Notes
    -----
    This class is designed to store data related to a Production Sharing Contract
    (PSC) cost recovery scenario.
    """
    # Attributes associated with FTP
    oil_ftp_availability: str
    oil_ftp_is_shared: str
    oil_ftp_portion: float
    gas_ftp_availability: str
    gas_ftp_is_shared: str
    gas_ftp_portion: float

    # Attributes associated with pretax split configuration
    split_type: str
    oil_ctr_pretax: float
    gas_ctr_pretax: float

    # Attributes associated with investment credit
    ic_availability: str
    ic_oil: float
    ic_gas: float

    # Attributes associated with cap
    oil_cr_cap_rate: float
    gas_cr_cap_rate: float

    # Attribute associated with general DMO
    dmo_is_weighted: str

    # Attributes associated with DMO oil
    oil_dmo_holiday: str
    oil_dmo_period: int | float
    oil_dmo_start_production: datetime
    oil_dmo_volume: float
    oil_dmo_fee: float

    # Attributes associated with DMO gas
    gas_dmo_holiday: str
    gas_dmo_period: int | float
    gas_dmo_start_production: datetime
    gas_dmo_volume: float
    gas_dmo_fee: float

    # Attributes associated with rc_split and icp_sliding_scale
    rc_split_init: InitVar[np.ndarray] = field(repr=False)
    icp_sliding_scale_init: InitVar[np.ndarray] = field(repr=False)
    indicator_rc_split_sliding_scale_init: InitVar[np.ndarray] = field(repr=False)

    # Attributes to be defined later on
    rc_split: dict = field(default=None, init=False)
    icp_sliding_scale: dict = field(default=None, init=False)
    indicator_rc_split_sliding_scale: dict = field(default=None, init=False)

    def __post_init__(
        self,
        rc_split_init,
        icp_sliding_scale_init,
        indicator_rc_split_sliding_scale_init,
    ):
        # Convert attributes to boolean
        self.oil_ftp_availability = get_boolean_converter(target=self.oil_ftp_availability)
        self.oil_ftp_is_shared = get_boolean_converter(target=self.oil_ftp_is_shared)
        self.gas_ftp_availability = get_boolean_converter(target=self.gas_ftp_availability)
        self.gas_ftp_is_shared = get_boolean_converter(target=self.gas_ftp_is_shared)
        self.ic_availability = get_boolean_converter(target=self.ic_availability)
        self.dmo_is_weighted = get_boolean_converter(target=self.dmo_is_weighted)
        self.oil_dmo_holiday = get_boolean_converter(target=self.oil_dmo_holiday)
        self.gas_dmo_holiday = get_boolean_converter(target=self.gas_dmo_holiday)

        # Prepare attribute split_type
        self.split_type = get_split_type_converter(target=self.split_type)

        # Prepare attribute rc_split
        if self.split_type == TaxSplitTypeCR.R2C:
            rc_split_attrs = [
                "RC Bottom Limit",
                "RC Top Limit",
                "Pre Tax CTR Oil",
                "Pre Tax CTR Gas",
            ]
            rc_split = {key: rc_split_init[:, i] for i, key in enumerate(rc_split_attrs)}

            self.rc_split = {
                key: np.array(list(filter(lambda i: i is not None, rc_split[key])))
                for key in rc_split_attrs
            }

            indicator_attrs = ["Year", "Indicator"]
            indicator_split = {
                key: indicator_rc_split_sliding_scale_init[:, i]
                for i, key in enumerate(indicator_attrs)
            }

            self.indicator_rc_split_sliding_scale = {
                key: np.array(list(filter(lambda i: i is not None, indicator_split[key])))
                for key in indicator_attrs
            }

        # Prepare attribute icp_sliding_scale
        if self.split_type == TaxSplitTypeCR.SLIDING_SCALE:
            icp_attrs = [
                "ICP Bottom Limit",
                "ICP Top Limit",
                "Pre Tax CTR Oil",
                "Pre Tax CTR Gas",
            ]
            icp_sliding_scale = {
                key: icp_sliding_scale_init[:, i] for i, key in enumerate(icp_attrs)
            }

            self.icp_sliding_scale = {
                key: np.array(list(filter(lambda i: i is not None, icp_sliding_scale[key])))
                for key in icp_attrs
            }

            indicator_attrs = ["Year", "Indicator"]
            indicator_split = {
                key: indicator_rc_split_sliding_scale_init[:, i]
                for i, key in enumerate(indicator_attrs)
            }

            self.indicator_rc_split_sliding_scale = {
                key: np.array(list(filter(lambda i: i is not None, indicator_split[key])))
                for key in indicator_attrs
            }


@dataclass
class PSCGrossSplitData:
    """
    A dataclass representing attributes associated with Production Sharing Contract
    (PSC) gross split.

    Attributes
    ----------
    field_status: str
        The status of the field.
    field_location: str
        The location of the field.
    reservoir_depth: str
        The depth of the reservoir.
    infrastructure_availability: str
        The availability of infrastructure.
    reservoir_type: str
        The type of reservoir.
    co2_content: str
        The CO2 content in the production.
    h2s_content: str
        The H2S content in the production.
    oil_api: str
        The API gravity of the oil.
    domestic_content_use: str
        The use of domestic content.
    production_stage: str
        The stage of production.
    ministry_discretion_split: float
        The split determined by ministry discretion.
    oil_base_split: float
        Base split for oil.
    gas_base_split: float
        Base split for gas.
    dmo_is_weighted: str
        Weighting information for general DMO.
    oil_dmo_holiday: str
        Holiday information for DMO oil.
    oil_dmo_period: int | float
        The period of DMO oil production.
    oil_dmo_start_production: date
        The start date of DMO oil production.
    oil_dmo_volume: float
        The volume of DMO oil production.
    oil_dmo_fee: float
        The fee associated with DMO oil production.
    gas_dmo_holiday: str
        Holiday information for DMO gas.
    gas_dmo_period: int | float
        The period of DMO gas production.
    gas_dmo_start_production: date
        The start date of DMO gas production.
    gas_dmo_volume: float
        The volume of DMO gas production.
    gas_dmo_fee: float
        The fee associated with DMO gas production.

    Notes
    -----
    This class is designed to store data related to a Production Sharing Contract
    (PSC) gross split scenario.
    """
    # Attributes associated with split configuration
    field_status: str
    field_location: str
    reservoir_depth: str
    infrastructure_availability: str
    reservoir_type: str
    co2_content: str
    h2s_content: str
    oil_api: str
    domestic_content_use: str
    production_stage: str
    ministry_discretion_split: float

    # Attribute associated with split
    oil_base_split: float
    gas_base_split: float

    # Attribute associated with general DMO
    dmo_is_weighted: str

    # Attributes associated with DMO oil
    oil_dmo_holiday: str
    oil_dmo_period: int | float
    oil_dmo_start_production: datetime
    oil_dmo_volume: float
    oil_dmo_fee: float

    # Attributes associated with DMO gas
    gas_dmo_holiday: str
    gas_dmo_period: int | float
    gas_dmo_start_production: datetime
    gas_dmo_volume: float
    gas_dmo_fee: float

    def __post_init__(self):
        # Convert attributes to boolean
        self.dmo_is_weighted = get_boolean_converter(target=self.dmo_is_weighted)
        self.oil_dmo_holiday = get_boolean_converter(target=self.oil_dmo_holiday)
        self.gas_dmo_holiday = get_boolean_converter(target=self.gas_dmo_holiday)


@dataclass
class SensitivityData:
    """
    A dataclass representing attributes associated with sensitivity analysis.

    Attributes
    ----------
    parameter: list
        A list of parameters or values for which sensitivity analysis is conducted.
    percentage_min: float
        The minimum percentage value for sensitivity analysis.
    percentage_max: float
        The maximum percentage value for sensitivity analysis.
    step: int
        The step size for the sensitivity analysis.
    """
    parameter: list
    percentage_min: float
    percentage_max: float
    step: int


@dataclass
class MonteCarloData:
    """
    A dataclass representing attributes associated with montecarlo analysis.

    Attributes
    ----------
    parameter: list
        A list of parameters or variables for the simulation.
    distribution: list
        A list of probability distributions corresponding to each parameter.
    min_values: np.ndarray
        An array of minimum values for each parameter.
    max_values: np.ndarray
        An array of maximum values for each parameter.
    std_dev: np.ndarray
        An array of standard deviations for each parameter.
    """
    parameter: list
    distribution: list
    min_values: np.ndarray
    max_values: np.ndarray
    std_dev: np.ndarray

    # Attribute associated with length of the captured data
    data_length: int

    def __post_init__(self):
        # Prepare attribute min_values
        if not isinstance(self.min_values, np.ndarray):
            raise MonteCarloDataException(
                f"Minimum values data must be given in the form of numpy.ndarray. "
                f"The current min_values data is given in the form of: "
                f"({self.min_values.__class__.__qualname__})."
            )

        min_values_nan = list(filter(lambda i: i is np.nan, self.min_values))

        if len(min_values_nan) > 0:
            raise MonteCarloDataException(
                f"Minimum values data is incomplete. Please re-check the min_values data. "
                f"The number of min_values data must be ({self.data_length}), "
                f"not ({self.data_length - len(min_values_nan)})."
            )

        else:
            self.min_values = self.min_values

        # Prepare attribute max_values
        if not isinstance(self.max_values, np.ndarray):
            raise MonteCarloDataException(
                f"Maximum values data must be given in the form of numpy.ndarray. "
                f"The current max_values data is given in the form of: "
                f"({self.max_values.__class__.__qualname__})."
            )

        max_values_nan = list(filter(lambda i: i is np.nan, self.max_values))

        if len(max_values_nan) > 0:
            raise MonteCarloDataException(
                f"Maximum values data is incomplete. Please re-check the max_values data. "
                f"The number of max_values data must be ({self.data_length}), "
                f"not ({self.data_length - len(max_values_nan)})."
            )

        else:
            self.max_values = self.max_values

        # Prepare attribute std_dev
        if not isinstance(self.std_dev, np.ndarray):
            raise MonteCarloDataException(
                f"Standard deviation data must be given in the form of numpy.ndarray. "
                f"The current std_dev data is given in the form of: "
                f"({self.std_dev.__class__.__qualname__})."
            )

        std_dev_nan = list(filter(lambda i: i is np.nan, self.std_dev))

        if len(std_dev_nan) > 0:
            raise MonteCarloDataException(
                f"Standard deviation data is incomplete. Please re-check the std_dev data. "
                f"The number of std_dev data must be ({self.data_length}), "
                f"not ({self.data_length - len(std_dev_nan)})."
            )

        else:
            self.std_dev = self.std_dev

        # Conditions to raise exception
        for i, val in enumerate(self.min_values):
            if self.min_values[i] == self.max_values[i]:
                raise MonteCarloDataException(
                    f"Maximum and minimum value(s) are the same."
                )

            if self.min_values[i] > self.max_values[i]:
                raise MonteCarloDataException(
                    f"Maximum values of parameter(s) are less "
                    f"than their corresponding minimum values."
                )


@dataclass
class OptimizationData:
    """
    A dataclass representing attributes associated with optimization study.

    Attributes
    ----------
    target: dict
        The target data.
    data_cr_init: dict
        The initial cost recovery data.
    data_gs_init: dict
        The initial gross split data.
    """
    target: dict
    data_cr_init: dict = field(repr=False)
    data_gs_init: dict = field(repr=False)

    # Attributes to be defined later
    data_cr: dict = field(default=None, init=False)
    data_gs: dict = field(default=None, init=False)

    def __post_init__(self):
        # Prepare attribute data_cr
        # Step #1: raise exception for inappropriate data input
        check_data_cr = [i <= j for i, j in zip(self.data_cr_init["max"], self.data_cr_init["min"])]

        if True in check_data_cr:
            raise OptimizationDataException(
                f"Error in cost recovery data input. "
                f"The maximum value(s) must be larger than the minimum value(s). "
                f"Max: ({self.data_cr_init['max']}), "
                f"Min: ({self.data_cr_init['min']})."
            )

        # Step #2: filter out 'None' values
        id_cr_unsorted = np.array(
            [i for i, val in enumerate(self.data_cr_init["priority"]) if val is not None]
        )
        data_cr_unsorted = {key: self.data_cr_init[key][id_cr_unsorted] for key in self.data_cr_init}

        # Step #3: sorted the data based on the values of "priority"
        id_cr_sorted = np.argsort(data_cr_unsorted["priority"])
        self.data_cr = {key: data_cr_unsorted[key][id_cr_sorted] for key in data_cr_unsorted}

        # Prepare attribute data_gs
        # Step #1: raise exception for inappropriate data input
        check_data_gs = [i <= j for i, j in zip(self.data_gs_init["max"], self.data_gs_init["min"])]

        if True in check_data_gs:
            raise OptimizationDataException(
                f"Error in cost recovery data input. "
                f"The maximum value(s) must be larger than the minimum value(s). "
                f"Max: ({self.data_gs_init['max']}), "
                f"Min: ({self.data_gs_init['min']})."
            )

        # Step #2: filter out 'None' values
        id_gs_unsorted = np.array(
            [i for i, val in enumerate(self.data_gs_init["priority"]) if val is not None]
        )
        data_gs_unsorted = {key: self.data_gs_init[key][id_gs_unsorted] for key in self.data_gs_init}

        # Step #3: sorted the data based on the values of "priority"
        id_gs_sorted = np.argsort(data_gs_unsorted["priority"])
        self.data_gs = {key: data_gs_unsorted[key][id_gs_sorted] for key in data_gs_unsorted}
