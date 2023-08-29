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
        Represents the straight-line depreciation method.
    DB : str
        Represents the declining balance depreciation method.
    UOP : str
        Represents the units of production depreciation method.
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
        Represents all fluid types.
    OIL : str
        Represents oil as the fluid type.
    GAS : str
        Represents gas as the fluid type.

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
        Represents conventional split type.
    SLIDING_SCALE : str
        Represents sliding scale split type.
    R2C : str
        Represents revenue to cost split type.

    """

    CONVENTIONAL = "conventional"
    SLIDING_SCALE = "ICP sliding scale"
    R2C = "R/C"
