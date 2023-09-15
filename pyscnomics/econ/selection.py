"""
Specify constants for several processes using Enum class.
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
    PSC : str
        Represents the PSC double declining balance method
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
    SULFUR: str
        Represents sulfur as the fluid type
    ELECTRICITY: str
        Represent electricity as the fluid type
    CO2: str
        Represents CO2 as the fluid type
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
    Enumeration of different year references used for depreciation calculations.

    Attributes:
    -----------
    PIS_YEAR : str
        Symbolic name for the Placed in Service (PIS) year reference

    EXPENSE_YEAR : str
        Symbolic name for the expense year reference
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
    NAILED_DOWN = "nailed down"
    PREVAILING = "prevailing"
    UU_36_2008 = "UU No.36 Tahun 2008"
    UU_02_2020 = "UU No.02 Tahun 2020"
    UU_07_2021 = "UU No.07 Tahun 2021"


class FTPTaxRegime(Enum):
    PDJP_20_2017 = "Peraturan DIRJEN PAJAK No. PER-20/PJ/2017"
    PRE_2017 = "Pre Peraturan DIRJEN PAJAK No. PER-20/PJ/2017"


class GrossSplitRegime(Enum):
    PERMEN_ESDM_8_2017 = 'Peraturan Menteri ESDM No. 8 Tahun 2017'
    PERMEN_ESDM_52_2017 = 'Peraturan Menteri ESDM No. 52 Tahun 2017'
    PERMEN_ESDM_20_2019 = 'Peraturan Menteri ESDM No. 20 Tahun 2019'
    PERMEN_ESDM_12_2020 = 'Peraturan Menteri ESDM No. 12 Tahun 2020'
