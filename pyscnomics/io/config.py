"""
Prepare data input loaded from a target Excel file.
"""

from dataclasses import dataclass, field, InitVar
from datetime import date
import numpy as np


class ExcelDataException(Exception):
    """ Exception to be raised for inappropriate data input in Excel """

    pass


class OilLiftingDataException(Exception):
    """ Exception to be raised for inappropriate use of OilLifingData class """

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
    """

    start_date_project: date
    end_date_project: date
    type_of_contract: str
    oil_onstream_date: date = field(default=None)
    gas_onstream_date: date = field(default=None)
    discount_rate_start_year: int = field(default=None)
    discount_rate: float = field(default=0.1)
    inflation_rate_applied_to: str = field(default=None)
    vat_discount: float = field(default=0.0)
    lbt_discount: float = field(default=0.0)
    gsa_number: int = field(default=1)

    # Attributes associated with duration of the project
    project_duration: int = field(default=None, init=False)
    project_years: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        # Check for inappropriate start and end year project
        if self.end_date_project.year >= self.start_date_project.year:
            self.project_duration = self.end_date_project.year - self.start_date_project.year + 1
            self.project_years = np.arange(
                self.start_date_project.year, self.end_date_project.year + 1, 1
            )

        else:
            raise ExcelDataException(
                f"start year {self.start_date_project.year} "
                f"is after the end year: {self.end_date_project.year}"
            )


@dataclass
class FiscalConfigData:
    """
    A dataclass representing fiscal configuration information for an oil and gas economic project.

    Attributes
    ----------
    tax_mode: str
        The tax mode for fiscal configuration.
    tax_payment_method: str
        The method of tax payment.
    tax_psc_cost_recovery: str
        The basis of tax configuration for PSC cost recovery.
    npv_mode: str
        The Net Present Value (NPV) calculation mode.
    future_rate_asr: float
        The future rate used in ASR cost calculation.
    depreciation_method: str
        The depreciation method to use.
    """

    tax_mode: str
    tax_payment_method: str
    tax_psc_cost_recovery: str
    npv_mode: str
    future_rate_asr: float
    depreciation_method: str

    # Attributes to be defined later
    tax_rate: float | dict = field(default=0.44)
    year_arr: np.ndarray = field(default=None, repr=False)
    tax_rate_arr: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        # Configure attribute tax_rate
        if self.tax_mode == "User Input - Single Value":
            self.tax_rate = self.tax_rate

        if self.tax_mode == "Nailed Down" or self.tax_mode == "Prevailing":
            self.tax_rate = None

        if self.tax_mode == "User Input - Multi Value":
            if not isinstance(self.year_arr, np.ndarray):
                raise ExcelDataException

            if not isinstance(self.tax_rate_arr, np.ndarray):
                raise ExcelDataException

            if len(self.year_arr) != len(self.tax_rate_arr):
                raise ExcelDataException

            self.tax_rate = {
                "Year": self.year_arr,
                "Tax Rate": self.tax_rate_arr,
            }


@dataclass
class OilLiftingData:
    """
    A dataclass representing oil lifting information for an oil and gas economic project.

    Attributes
    ----------
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    prod_year: dict
        Dictionary containing production years data.
    oil_lifting_rate : dict
        Dictionary containing oil lifting rate data.
    oil_price : dict
        Dictionary containing oil price data.
    condensate_lifting_rate : dict
        Dictionary containing condensate lifting rate data.
    condensate_price : dict
        Dictionary containing condensate price data.

    Notes
    -----
    This dataclass is used to store and organize information related to oil lifting.
    """
    project_duration: int
    project_years: np.ndarray
    prod_year: dict
    oil_lifting_rate: dict
    oil_price: dict
    condensate_lifting_rate: dict
    condensate_price: dict


@dataclass
class GasLiftingData:
    """
    A dataclass representing gas lifting information for an oil and gas economic project.

    Attributes
    ----------
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    gas_gsa_number: int
        The number of GSA.
    prod_year: dict
        Dictionary containing production years data.
    gas_lifting_rate: dict
        Dictionary containing gas lifting rate data.
    gas_ghv: dict
        Dictionary containing gas ghv data.
    gas_price: dict
        Dictionary containing gas price data.
    """
    project_duration: int
    project_years: np.ndarray
    gas_gsa_number: int
    prod_year: dict
    gas_lifting_rate: dict
    gas_ghv: dict
    gas_price: dict



