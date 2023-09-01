"""
Specify fluid type and depreciation method.
"""

from enum import Enum


class DeprMethod(Enum):

    """
    Enumeration of depreciation methods.

    Attributes
    ----------
    SL : str
        Represents the straight-line depreciation method
    DB : str
        Represents the declining balance depreciation method
    UOP : str
        Represents the units of production depreciation method
    """

    SL = "sl"
    DB = "db"
    PSC = "psc db"
    UOP = "uop"


class FluidType(Enum):

    """
    Enumeration of fluid types for depreciation calculation.

    Attributes
    ----------
    ALL : str
        Represents all fluid types
    OIL : str
        Represents oil as the fluid type
    GAS : str
        Represents gas as the fluid type
    """

    ALL = "all"
    OIL = "oil"
    GAS = "gas"
    SULFUR = "sulfur"
    ELECTRICITY = "electricity"
    CO2 = "co2"


class TaxSplitTypeCR(Enum):
    """
    Enumeration of split type for cost recovery.

    Attributes
    ----------
    CONVENTIONAL : str
        Represents conventional split type
    SLIDING_SCALE : str
        Represents sliding scale split type
    R2C : str
        Represents revenue to cost split type

    """

    CONVENTIONAL = "conventional"
    SLIDING_SCALE = "ICP sliding scale"
    R2C = "R/C"


class YearReference(Enum):
    """
    Enumeration of different year references used for financial calculations.

    This enumeration defines symbolic names for various year references that are commonly used
    in financial calculations, such as PIS (Payment in Service) year and expense year.

    Attributes:
    -----------
    PIS_YEAR : str
        Symbolic name for the Payment in Service (PIS) year reference

    EXPENSE_YEAR : str
        Symbolic name for the expense year reference

    Notes:
    ------
    The `YearReference` enumeration provides a convenient way to reference specific years when
    performing financial calculations or representing different types of fiscal year references.
    """

    PIS_YEAR = "pis_year"
    EXPENSE_YEAR = "expense_year"


class TaxRegime(Enum):
    """
    Enumeration class representing different tax regimes for oil and gas economic analysis.

    Attributes:
    ----------
    UU_36_2008 (str): Tax regime under "UU No.36 Tahun 2008"
    UU_02_2020 (str): Tax regime under "UU No.02 Tahun 2020"
    UU_07_2021 (str): Tax regime under "UU No.07 Tahun 2021"
    NAILED_DOWN (str): Special tax regime designated as "nailed down"
    PREVAILING (str): General tax regime referred to as "prevailing"
    """

    UU_36_2008 = "UU No.36 Tahun 2008"
    UU_02_2020 = "UU No.02 Tahun 2020"
    UU_07_2021 = "UU No.07 Tahun 2021"
    NAILED_DOWN = "nailed down"
    PREVAILING = "prevailing"
