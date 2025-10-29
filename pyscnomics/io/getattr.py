"""
Handles the routine to get the attributes of a contract object.
"""
import numpy as np

from datetime import date
# from enum import Enum

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.econ import (
    TaxSplitTypeCR,
    NPVSelection,
    DiscountingMode,
    OtherRevenue,
    DeprMethod,
    FTPTaxRegime,
    TaxRegime,
    InflationAppliedTo,
    GrossSplitRegime,
)
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import (
    CapitalCost,
    Intangible,
    OPEX,
    ASR,
    LBT,
    CostOfSales
)
from pyscnomics.econ.selection import (
    FluidType,
    CostType,
    VariableSplit082017,
    VariableSplit522017,
    VariableSplit132024,
    SunkCostMethod,
    InitialYearAmortizationIncurred,
)


class GetAttrException(Exception):
    """Exception to raise for a misuse of Get Attributes Method"""

    pass


def _helper_convert_enum_to_str(enum_target, enum_type, enum_mapping) -> str:
    """
    Convert an enum member to its corresponding string representation.

    This helper function provides a generic way to map enum members to
    string values with proper type validation and error handling.

    Parameters
    ----------
    enum_target : Any
        The enum member to convert to string representation.
    enum_type : type
        The expected enum class type for validation.
    enum_mapping : dict
        Dictionary mapping enum members to their string representations.
        Keys should be instances of `enum_type`, values should be strings.

    Returns
    -------
    str
        String representation of the enum member as defined in `enum_mapping`.

    Raises
    ------
    GetAttrException
        If `enum_target` is not an instance of `enum_type`.
        If `enum_target` is not found as a key in `enum_mapping`.

    Notes
    -----
    This function is designed as an internal helper to reduce code duplication
    when converting multiple enum types to string representations.
    """

    # Filter incorrect input
    if not isinstance(enum_target, enum_type):
        raise GetAttrException(
            f"Parameter must be a {enum_type.__name__!r} instance, "
            f"not {enum_target.__class__.__qualname__!r}"
        )

    try:
        return enum_mapping[enum_target]
    except KeyError:
        raise GetAttrException(f"Invalid {enum_type.__name__!r} object: {enum_target!r} ")


def convert_enum_fluid(objects: FluidType) -> str:
    """
    Convert FluidType enum member to its corresponding string representation.

    Parameters
    ----------
    objects : FluidType
        The FluidType enum member to convert to string.

    Returns
    -------
    str
        String representation of the FluidType enum member.
        Possible values: 'Oil', 'Gas', 'Sulfur', 'Electricity', 'CO2'.

    Raises
    ------
    GetAttrException
        If `objects` is not a FluidType instance.
        If `objects` is not a recognized FluidType enum member.

    Notes
    -----
    This function uses a dictionary-based mapping approach for optimal
    performance (O(1) lookup time) compared to the previous if-elif chain.

    The implementation delegates to a shared helper function
    `_helper_convert_enum_to_str` for consistent error handling and validation
    across similar enum converters.
    """

    """
    Former approach:
    if objects is FluidType.OIL:
        result = 'Oil'
    elif objects is FluidType.GAS:
        result = 'Gas'
    elif objects is FluidType.SULFUR:
        result = 'Sulfur'
    elif objects is FluidType.ELECTRICITY:
        result = 'Electricity'
    elif objects is FluidType.CO2:
        result = 'CO2'
    else:
        raise GetAttrException(
            f"{objects} is not recognized"
        )
    return result
    """

    # Core mapping: an instance of FluidType -> its corresponding string
    mapping = {
        FluidType.OIL: "Oil",
        FluidType.GAS: "Gas",
        FluidType.SULFUR: "Sulfur",
        FluidType.ELECTRICITY: "Electricity",
        FluidType.CO2: "CO2",
    }

    return _helper_convert_enum_to_str(
        enum_target=objects, enum_type=FluidType, enum_mapping=mapping
    )


def convert_enum_cost_type(objects: CostType) -> str:
    """
    Convert a CostType enum member to its string representation.

    Parameters
    ----------
    objects : CostType
        Target cost type enum.

    Returns
    -------
    str
        Corresponding cost type string.

    Notes
    -----
    The conversion uses a predefined mapping and `_helper_convert_enum_to_str`.
    """

    # Core mapping: an instance of CostType -> its corresponding string
    mapping = {
        CostType.SUNK_COST: "sunk_cost",
        CostType.PRE_ONSTREAM_COST: "preonstream_cost",
        CostType.POST_ONSTREAM_COST: "postonstream_cost",
    }

    return _helper_convert_enum_to_str(
        enum_target=objects, enum_type=CostType, enum_mapping=mapping
    )


def convert_enum_taxsplit(objects: TaxSplitTypeCR) -> str:
    """
    Convert TaxSplitTypeCR enum member to its corresponding string representation.

    Parameters
    ----------
    objects : TaxSplitTypeCR
        The TaxSplitTypeCR enum member to convert to string.

    Returns
    -------
    str
        String representation of the TaxSplitTypeCR enum member.
        Possible values: 'Conventional', 'ICP Sliding Scale', 'R/C'.

    Raises
    ------
    GetAttrException
        If `objects` is not a TaxSplitTypeCR instance.
        If `objects` is not a recognized TaxSplitTypeCR enum member.

    Notes
    -----
    This function uses a dictionary-based mapping approach for optimal
    performance (O(1) lookup time) compared to the previous if-elif chain.

    The implementation delegates to a shared helper function `_helper_convert_enum_to_str`
    for consistent error handling and validation across similar enum converters.
    """

    """
    # Former approach
    if objects is TaxSplitTypeCR.CONVENTIONAL:
        result = 'Conventional'
    elif objects is TaxSplitTypeCR.SLIDING_SCALE:
        result = 'ICP Sliding Scale'
    elif objects is TaxSplitTypeCR.R2C:
        result = 'R/C'
    else:
        raise GetAttrException(
            f"{objects} is not recognized"
        )
    return result
    """

    # Core mapping
    mapping = {
        TaxSplitTypeCR.CONVENTIONAL: "Conventional",
        TaxSplitTypeCR.SLIDING_SCALE: "ICP Sliding Scale",
        TaxSplitTypeCR.R2C: "R/C"
    }

    return _helper_convert_enum_to_str(
        enum_target=objects, enum_type=TaxSplitTypeCR, enum_mapping=mapping
    )


def convert_enum_npv(objects: NPVSelection) -> str:
    """
    Convert NPVSelection enum member to its corresponding string representation.

    Parameters
    ----------
    objects : NPVSelection
        The NPVSelection enum member to convert to string.

    Returns
    -------
    str
        String representation of the NPVSelection enum member.
        Possible values:
        - 'SKK Full Cycle Real Terms'
        - 'SKK Full Cycle Nominal Terms'
        - 'Full Cycle Real Terms'
        - 'Full Cycle Nominal Terms'
        - 'Point Forward'

    Raises
    ------
    GetAttrException
        If `objects` is not a NPVSelection instance.
        If `objects` is not a recognized NPVSelection enum member.

    Notes
    -----
    This function replaces a previous inefficient approach that used linear
    search through dictionary keys. The current implementation uses direct
    dictionary lookup for optimal O(1) performance.

    The function delegates to `_helper_convert_enum_to_str` for consistent
    type validation and error handling across all enum conversion functions.
    """

    """
    # Former approach
    attrs = {
        NPVSelection.NPV_SKK_REAL_TERMS:"SKK Full Cycle Real Terms",
        NPVSelection.NPV_SKK_NOMINAL_TERMS:"SKK Full Cycle Nominal Terms",
        NPVSelection.NPV_REAL_TERMS:"Full Cycle Real Terms",
        NPVSelection.NPV_NOMINAL_TERMS:"Full Cycle Nominal Terms",
        NPVSelection.NPV_POINT_FORWARD:"Point Forward",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]
    """

    # Core mapping
    mapping = {
        NPVSelection.NPV_SKK_REAL_TERMS: "SKK Full Cycle Real Terms",
        NPVSelection.NPV_SKK_NOMINAL_TERMS: "SKK Full Cycle Nominal Terms",
        NPVSelection.NPV_REAL_TERMS: "Full Cycle Real Terms",
        NPVSelection.NPV_NOMINAL_TERMS: "Full Cycle Nominal Terms",
        NPVSelection.NPV_POINT_FORWARD: "Point Forward",
    }

    return _helper_convert_enum_to_str(
        enum_target=objects, enum_type=NPVSelection, enum_mapping=mapping
    )


def convert_enum_discountingmode(objects: DiscountingMode) -> str:
    """
    Convert DiscountingMode enum member to its corresponding string representation.

    Parameters
    ----------
    objects : DiscountingMode
        The DiscountingMode enum member to convert to string.

    Returns
    -------
    str
        String representation of the DiscountingMode enum member.
        Possible values:
        - 'End Year'
        - 'Mid Year'

    Raises
    ------
    GetAttrException
        If `objects` is not a DiscountingMode instance.
        If `objects` is not a recognized DiscountingMode enum member.

    Notes
    -----
    This function replaces a previous inefficient approach that used linear
    search through dictionary keys. The current implementation uses direct
    dictionary lookup for optimal O(1) performance.

    The function delegates to `_helper_convert_enum_to_str` for consistent
    type validation and error handling across all enum conversion functions.
    """

    """
    # Former approach
    attrs = {
        DiscountingMode.END_YEAR: "End Year",
        DiscountingMode.MID_YEAR: "Mid Year",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]
    """

    # Core mapping
    mapping = {
        DiscountingMode.END_YEAR: "End Year",
        DiscountingMode.MID_YEAR: "Mid Year",
    }

    return _helper_convert_enum_to_str(
        enum_target=objects, enum_type=DiscountingMode, enum_mapping=mapping
    )


def convert_enum_otherrevenue(objects: OtherRevenue) -> str:
    """
    Convert OtherRevenue enum member to its corresponding string representation.

    Parameters
    ----------
    objects : OtherRevenue
        The OtherRevenue enum member to convert to string.

    Returns
    -------
    str
        String representation of the OtherRevenue enum member.
        Possible values:
        - 'Addition to Oil Revenue'
        - 'Addition to Gas Revenue'
        - 'Reduction to Oil OPEX'
        - 'Reduction to Gas OPEX'

    Raises
    ------
    GetAttrException
        If `objects` is not a OtherRevenue instance.
        If `objects` is not a recognized OtherRevenue enum member.

    Notes
    -----
    This function replaces a previous inefficient approach that used linear
    search through dictionary keys. The current implementation uses direct
    dictionary lookup for optimal O(1) performance.

    The function delegates to `_helper_convert_enum_to_str` for consistent
    type validation and error handling across all enum conversion functions.
    """

    """
    # Former approach
    attrs = {
        OtherRevenue.ADDITION_TO_OIL_REVENUE: "Addition to Oil Revenue",
        OtherRevenue.ADDITION_TO_GAS_REVENUE: "Addition to Gas Revenue",
        OtherRevenue.REDUCTION_TO_OIL_OPEX: "Reduction to Oil OPEX",
        OtherRevenue.REDUCTION_TO_GAS_OPEX: "Reduction to Gas OPEX",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]
    """

    # Core mapping
    mapping = {
        OtherRevenue.ADDITION_TO_OIL_REVENUE: "Addition to Oil Revenue",
        OtherRevenue.ADDITION_TO_GAS_REVENUE: "Addition to Gas Revenue",
        OtherRevenue.REDUCTION_TO_OIL_OPEX: "Reduction to Oil OPEX",
        OtherRevenue.REDUCTION_TO_GAS_OPEX: "Reduction to Gas OPEX",
    }

    return _helper_convert_enum_to_str(
        enum_target=objects, enum_type=OtherRevenue, enum_mapping=mapping
    )


def convert_enum_depreciationmethod(objects: DeprMethod) -> str:
    """
    Convert DeprMethod enum member to its corresponding string representation.

    Parameters
    ----------
    objects : DeprMethod
        The DeprMethod enum member to convert to string.

    Returns
    -------
    str
        String representation of the DeprMethod enum member.
        Possible values:
        - 'PSC Declining Balance'
        - 'Declining Balance'
        - 'Unit Of Production'
        - 'Straight Line'

    Raises
    ------
    GetAttrException
        If `objects` is not a DeprMethod instance.
        If `objects` is not a recognized DeprMethod enum member.

    Notes
    -----
    This function replaces a previous inefficient approach that used linear
    search through dictionary keys. The current implementation uses direct
    dictionary lookup for optimal O(1) performance.

    The function delegates to `_helper_convert_enum_to_str` for consistent
    type validation and error handling across all enum conversion functions.
    """

    """
    # Former approach
    attrs = {
        DeprMethod.PSC_DB: "PSC Declining Balance",
        DeprMethod.DB: "Declining Balance",
        DeprMethod.UOP: "Unit Of Production",
        DeprMethod.SL: "Straight Line",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]
    """

    # Core mapping
    mapping = {
        DeprMethod.PSC_DB: "PSC Declining Balance",
        DeprMethod.DB: "Declining Balance",
        DeprMethod.UOP: "Unit Of Production",
        DeprMethod.SL: "Straight Line",
    }

    return _helper_convert_enum_to_str(
        enum_target=objects, enum_type=DeprMethod, enum_mapping=mapping
    )


def convert_enum_ftptaxregime(objects: FTPTaxRegime) -> str:
    """
    Convert FTPTaxRegime enum member to its corresponding string representation.

    Parameters
    ----------
    objects : FTPTaxRegime
        The FTPTaxRegime enum member to convert to string.

    Returns
    -------
    str
        String representation of the FTPTaxRegime enum member.
        Possible values:
        - 'PDJP No.20 Tahun 2017'
        - 'Pre PDJP No.20 Tahun 2017'

    Raises
    ------
    GetAttrException
        If `objects` is not a FTPTaxRegime instance.
        If `objects` is not a recognized FTPTaxRegime enum member.

    Notes
    -----
    This function replaces a previous inefficient approach that used linear
    search through dictionary keys. The current implementation uses direct
    dictionary lookup for optimal O(1) performance.

    The function delegates to `_helper_convert_enum_to_str` for consistent
    type validation and error handling across all enum conversion functions.
    """

    """
    # Former approach
    attrs = {
        FTPTaxRegime.PDJP_20_2017: "PDJP No.20 Tahun 2017",
        FTPTaxRegime.PRE_PDJP_20_2017: "Pre PDJP No.20 Tahun 2017",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]
    """

    # Core mapping
    mapping = {
        FTPTaxRegime.PDJP_20_2017: "PDJP No.20 Tahun 2017",
        FTPTaxRegime.PRE_PDJP_20_2017: "Pre PDJP No.20 Tahun 2017",
    }

    return _helper_convert_enum_to_str(
        enum_target=objects, enum_type=FTPTaxRegime, enum_mapping=mapping
    )


def convert_enum_taxregime(objects: TaxRegime) -> str:
    """
    Convert TaxRegime enum member to its corresponding string representation.

    Parameters
    ----------
    objects : TaxRegime
        The TaxRegime enum member to convert to string.

    Returns
    -------
    str
        String representation of the TaxRegime enum member.
        Possible values:
        - 'nailed down'
        - 'prevailing'
        - 'UU No.36 Tahun 2008'
        - 'UU No.02 Tahun 2020'
        - 'UU No.07 Tahun 2021'

    Raises
    ------
    GetAttrException
        If `objects` is not a TaxRegime instance.
        If `objects` is not a recognized TaxRegime enum member.

    Notes
    -----
    This function replaces a previous inefficient approach that used linear
    search through dictionary keys. The current implementation uses direct
    dictionary lookup for optimal O(1) performance.

    The function delegates to `_helper_convert_enum_to_str` for consistent
    type validation and error handling across all enum conversion functions.
    """

    """
    # Former approach
    attrs = {
        TaxRegime.NAILED_DOWN: "nailed down",
        TaxRegime.PREVAILING: "prevailing",
        TaxRegime.UU_36_2008: "UU No.36 Tahun 2008",
        TaxRegime.UU_02_2020: "UU No.02 Tahun 2020",
        TaxRegime.UU_07_2021: "UU No.07 Tahun 2021",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]
    """

    # Core mapping
    mapping = {
        TaxRegime.NAILED_DOWN: "nailed down",
        TaxRegime.PREVAILING: "prevailing",
        TaxRegime.UU_36_2008: "UU No.36 Tahun 2008",
        TaxRegime.UU_02_2020: "UU No.02 Tahun 2020",
        TaxRegime.UU_07_2021: "UU No.07 Tahun 2021",
    }

    return _helper_convert_enum_to_str(
        enum_target=objects, enum_type=TaxRegime, enum_mapping=mapping
    )


def convert_enum_inflationappliedto(objects: InflationAppliedTo) -> str:
    """
    Convert InflationAppliedTo enum member to its corresponding string representation.

    Parameters
    ----------
    objects : InflationAppliedTo
        The InflationAppliedTo enum member to convert to string.

    Returns
    -------
    str
        String representation of the InflationAppliedTo enum member.
        Possible values:
        - 'CAPEX'
        - 'OPEX'
        - 'CAPEX AND OPEX'

    Raises
    ------
    GetAttrException
        If `objects` is not a InflationAppliedTo instance.
        If `objects` is not a recognized InflationAppliedTo enum member.

    Notes
    -----
    This function replaces a previous inefficient approach that used linear
    search through dictionary keys. The current implementation uses direct
    dictionary lookup for optimal O(1) performance.

    The function delegates to `_helper_convert_enum_to_str` for consistent
    type validation and error handling across all enum conversion functions.
    """

    """
    # Former approach
    attrs = {
        InflationAppliedTo.CAPEX: "CAPEX",
        InflationAppliedTo.OPEX: "OPEX",
        InflationAppliedTo.CAPEX_AND_OPEX: "CAPEX AND OPEX",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]
    """

    # Core mapping
    mapping = {
        InflationAppliedTo.CAPEX: "CAPEX",
        InflationAppliedTo.OPEX: "OPEX",
        InflationAppliedTo.CAPEX_AND_OPEX: "CAPEX AND OPEX",
    }

    return _helper_convert_enum_to_str(
        enum_target=objects, enum_type=InflationAppliedTo, enum_mapping=mapping
    )


def convert_enum_gsregime(objects: GrossSplitRegime) -> str:
    """
    Convert GrossSplitRegime enum member to its corresponding string representation.

    Parameters
    ----------
    objects : GrossSplitRegime
        The GrossSplitRegime enum member to convert to string.

    Returns
    -------
    str
        String representation of the GrossSplitRegime enum member.
        Possible values:
        - 'PERMEN_ESDM_8_2017'
        - 'PERMEN_ESDM_52_2017'
        - 'PERMEN_ESDM_20_2019'
        - 'PERMEN_ESDM_12_2020'
        - 'PERMEN_ESDM_13_2024'

    Raises
    ------
    GetAttrException
        If `objects` is not a GrossSplitRegime instance.
        If `objects` is not a recognized GrossSplitRegime enum member.

    Notes
    -----
    This function replaces a previous inefficient approach that used linear
    search through dictionary keys. The current implementation uses direct
    dictionary lookup for optimal O(1) performance.

    The function delegates to `_helper_convert_enum_to_str` for consistent
    type validation and error handling across all enum conversion functions.
    """

    """
    # Former approach
    attrs = {
        GrossSplitRegime.PERMEN_ESDM_8_2017: "PERMEN_ESDM_8_2017",
        GrossSplitRegime.PERMEN_ESDM_52_2017: "PERMEN_ESDM_52_2017",
        GrossSplitRegime.PERMEN_ESDM_20_2019: "PERMEN_ESDM_20_2019",
        GrossSplitRegime.PERMEN_ESDM_12_2020: "PERMEN_ESDM_12_2020",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]
    """

    # Core mapping
    mapping = {
        GrossSplitRegime.PERMEN_ESDM_8_2017: "PERMEN_ESDM_8_2017",
        GrossSplitRegime.PERMEN_ESDM_52_2017: "PERMEN_ESDM_52_2017",
        GrossSplitRegime.PERMEN_ESDM_20_2019: "PERMEN_ESDM_20_2019",
        GrossSplitRegime.PERMEN_ESDM_12_2020: "PERMEN_ESDM_12_2020",
        GrossSplitRegime.PERMEN_ESDM_13_2024: "PERMEN_ESDM_13_2024",
    }

    return _helper_convert_enum_to_str(
        enum_target=objects, enum_type=GrossSplitRegime, enum_mapping=mapping
    )


def convert_enum_sunk_cost_method(objects: SunkCostMethod) -> str:
    """
    Convert a `SunkCostMethod` enum to its corresponding string representation.

    Parameters
    ----------
    objects : SunkCostMethod
        The sunk cost method enum to be converted.

    Returns
    -------
    str
        The string representation of the given `SunkCostMethod` value.
    """

    # Core mapping
    mapping = {
        SunkCostMethod.DEPRECIATED_TANGIBLE: "depreciated_tangible",
        SunkCostMethod.POOLED_1ST_YEAR: "pooled_first_year",
    }

    return _helper_convert_enum_to_str(
        enum_target=objects, enum_type=SunkCostMethod, enum_mapping=mapping
    )


def convert_enum_initial_amortization_year(
    objects: InitialYearAmortizationIncurred
) -> str:
    """
    Convert InitialYearAmortizationIncurred enum to string representation.

    Maps enum values to their corresponding string identifiers for use in
    amortization calculations and data processing.

    Parameters
    ----------
    objects : InitialYearAmortizationIncurred
        Enum value representing the initial year basis for amortization.

    Returns
    -------
    str
        String representation of the enum value:
        - "onstream_year" for ONSTREAM_YEAR
        - "approval_year" for APPROVAL_YEAR
    """

    # Core mapping
    mapping = {
        InitialYearAmortizationIncurred.ONSTREAM_YEAR: "onstream_year",
        InitialYearAmortizationIncurred.APPROVAL_YEAR: "approval_year",
    }

    return _helper_convert_enum_to_str(
        enum_target=objects,
        enum_type=InitialYearAmortizationIncurred,
        enum_mapping=mapping,
    )


def convert_object(objects):
    """
    Convert various object types to their serializable representations.

    This function serves as a universal converter that handles multiple data types
    including dates, NumPy arrays, and various enum types used in the financial
    modeling system. It ensures objects are converted to JSON-serializable formats.

    Parameters
    ----------
    objects : Any
        The object to convert to a serializable representation.
        Supported types include:
        - date objects
        - numpy.ndarray
        - FluidType enum
        - TaxSplitTypeCR enum
        - NPVSelection enum
        - DiscountingMode enum
        - OtherRevenue enum
        - DeprMethod enum
        - FTPTaxRegime enum
        - TaxRegime enum
        - InflationAppliedTo enum
        - GrossSplitRegime enum
        - VariableSplit522017 enums
        - VariableSplit082017 enums
        - VariableSplit132024 enums

    Returns
    -------
    Any
        Serializable representation of the input object:
        - date -> str in 'YYYY-MM-DD' format
        - numpy.ndarray -> list
        - Enum types -> str representation via dedicated converters
        - VariableSplit enums -> direct enum value
        - Other types -> returned as-is

    Notes
    -----
    This function provides a centralized conversion point for serialization
    purposes, particularly useful for preparing data for JSON serialization
    or API responses.

    The function uses dedicated converter functions for specific enum types
    to ensure consistent string representations across the application.
    """

    # Object is date
    if isinstance(objects, date):
        return objects.strftime('%Y-%m-%d')

    # Object is numpy.ndarray
    elif isinstance(objects, np.ndarray):
        return objects.tolist()

    # Object is FluidType
    elif isinstance(objects, FluidType):
        return convert_enum_fluid(objects=objects)

    # Object is CostType
    elif isinstance(objects, CostType):
        return convert_enum_cost_type(objects=objects)

    # Object is TaxSplitTypeCR
    elif isinstance(objects, TaxSplitTypeCR):
        return convert_enum_taxsplit(objects=objects)

    # Object is NPVSelection
    elif isinstance(objects, NPVSelection):
        return convert_enum_npv(objects=objects)

    # Object is DiscountingMode
    elif isinstance(objects, DiscountingMode):
        return convert_enum_discountingmode(objects=objects)

    # Object is OtherRevenue
    elif isinstance(objects, OtherRevenue):
        return convert_enum_otherrevenue(objects=objects)

    # Object is DeprMethod
    elif isinstance(objects, DeprMethod):
        return convert_enum_depreciationmethod(objects=objects)

    # Object is FTPTaxRegime
    elif isinstance(objects, FTPTaxRegime):
        return convert_enum_ftptaxregime(objects=objects)

    # Object is TaxRegime
    elif isinstance(objects, TaxRegime):
        return convert_enum_taxregime(objects=objects)

    # Object is InflationAppliedTo
    elif isinstance(objects, InflationAppliedTo):
        return convert_enum_inflationappliedto(objects=objects)

    # Object is GrossSplitRegime
    elif isinstance(objects, GrossSplitRegime):
        return convert_enum_gsregime(objects=objects)

    # Object is SunkCostMethod
    elif isinstance(objects, SunkCostMethod):
        return convert_enum_sunk_cost_method(objects=objects)

    # Object is InitialAmortizationIncurred
    elif isinstance(objects, InitialYearAmortizationIncurred):
        return convert_enum_initial_amortization_year(objects=objects)

    # Object is VariableSplit522017
    elif isinstance(objects, (
        VariableSplit522017.FieldStatus,
        VariableSplit522017.FieldLocation,
        VariableSplit522017.ReservoirDepth,
        VariableSplit522017.InfrastructureAvailability,
        VariableSplit522017.ReservoirType,
        VariableSplit522017.CO2Content,
        VariableSplit522017.H2SContent,
        VariableSplit522017.APIOil,
        VariableSplit522017.DomesticUse,
        VariableSplit522017.ProductionStage,
    )):
        return objects.value

    # Object is VariableSplit082017
    elif isinstance(objects, (
        VariableSplit082017.FieldStatus,
        VariableSplit082017.FieldLocation,
        VariableSplit082017.ReservoirDepth,
        VariableSplit082017.InfrastructureAvailability,
        VariableSplit082017.ReservoirType,
        VariableSplit082017.CO2Content,
        VariableSplit082017.H2SContent,
        VariableSplit082017.APIOil,
        VariableSplit082017.DomesticUse,
        VariableSplit082017.ProductionStage,
    )):
        return objects.value

    # Object is VariableSplit132024
    elif isinstance(objects, (
        VariableSplit132024.InfrastructureAvailability,
        VariableSplit132024.FieldReservesAmount,
        VariableSplit132024.FieldLocation,
        VariableSplit132024.ReservoirType,
    )):
        return objects.value

    else:
        return objects


def _prepare_args(target_arg, default, contract_arguments: dict):
    return contract_arguments.get(target_arg, default)


def construct_lifting_attr(lifting: tuple[Lifting]) -> dict:
    """
    Construct a dictionary containing processed lifting attributes for each fluid type.

    This function iterates through a tuple of `Lifting` objects, extracts their
    attributes, converts each attribute using `convert_object()`, and returns
    a dictionary mapping descriptive fluid type names (e.g., "Oil 0", "Gas 1")
    to the corresponding processed lifting data.

    Parameters
    ----------
    lifting : tuple of Lifting
        A tuple containing one or more `Lifting` objects. Each object represents
        lifting data for a specific fluid type.

    Returns
    -------
    dict
        A dictionary where:
        - Keys are strings describing each fluid type with an index suffix
          (e.g., `"Oil 0"`, `"Gas 1"`).
        - Values are dictionaries of processed lifting attributes for each fluid type.

    Notes
    -----
    - Each attribute of the `Lifting` object is converted using the
      `convert_object()` function.
    - The function assumes that each `Lifting` object has a `fluid_type`
      attribute with a `.value` property that can be converted to string.
    """

    fluid_types = (
        [
            (str(lift.fluid_type.value) + ' ' + str(index)).capitalize()
            for index, lift in enumerate(lifting)
        ]
    )

    liftings = [vars(lift) for lift in lifting]

    for lift in liftings:
        for key, item in lift.items():
            lift[key] = convert_object(objects=item)

    return dict(zip(fluid_types, liftings))


def construct_cost_attr(
    cost: (
        tuple[CapitalCost]
        | tuple[Intangible]
        | tuple[OPEX]
        | tuple[ASR]
        | tuple[LBT]
        | tuple[CostOfSales]
    )
):
    """
    Construct a dictionary of processed cost attributes from cost objects.

    Parameters
    ----------
    cost : tuple of CapitalCost, Intangible, OPEX, ASR, LBT, or CostOfSales
        Tuple of cost objects representing different expenditure types.

    Returns
    -------
    dict
        Dictionary mapping each cost entry (e.g., "Cost 0", "Cost 1") to its
        processed attributes.

    Notes
    -----
    Each cost object's attributes are converted using `convert_object()`,
    with `cost_allocation` handled by `convert_enum_fluid()` and `cost_type`
    by `convert_enum_cost_type()`.
    """

    # Define key for each cost instances
    cost_key = ["Cost " + str(index) for index, _ in enumerate(cost)]

    # Convert every cost instances into their correponding dictionaries
    costs = [vars(cst) for cst in cost]

    # Modify cost attributes (as dictionaries)
    for cst in costs:
        for key, val in cst.items():
            if key == "cost_allocation":
                cst[key] = [convert_enum_fluid(objects=fluid) for fluid in cst[key]]

            elif key == "cost_type":
                cst[key] = [convert_enum_cost_type(objects=ct) for ct in cst[key]]

            else:
                cst[key] = convert_object(objects=val)

    """
    Former approach
    ---------------
        for cst in costs:
        for key, item in cst.items():
            if key == "cost_allocation":
                cst[key] = [convert_enum_fluid(objects=fluid) for fluid in cst[key]]
            else:
                cst[key] = convert_object(objects=item)
    """

    return dict(zip(cost_key, costs))


def construct_setup_attr(contract: BaseProject | CostRecovery | GrossSplit | Transition):
    """
    Construct a standardized dictionary of setup attributes for a contract.

    Parameters
    ----------
    contract : BaseProject | CostRecovery | GrossSplit | Transition
        Contract object containing lifting data, contract dates, and onstream information.

    Returns
    -------
    dict
        Dictionary containing formatted setup attributes:
        - start_date : str
        - end_date : str
        - oil_onstream_date : str or None
        - gas_onstream_date : str or None
        - approval_year : int or None
        - is_pod_1 : bool

    Notes
    -----
    - Dates are formatted as "DD/MM/YYYY".
    - Onstream dates are included only if the fluid type is produced.
    - Attribute assignment uses `_prepare_setup_attr` for validation.
    """

    fluid_produced = [lift.fluid_type for lift in contract.lifting]
    contract_attrs = list(vars(contract).keys())

    def _get_date(fluid_type: FluidType, onstream_date: date):
        """
        Get formatted onstream date if fluid type is produced.

        Helper function to validate fluid production and format the
        corresponding onstream date.

        Parameters
        ----------
        fluid_type : FluidType
            Type of fluid (OIL or GAS) to check for production
        onstream_date : datetime
            Date when production for the fluid type begins

        Returns
        -------
        str or None
            Formatted date string "DD/MM/YYYY" if fluid type is produced,
            otherwise None
        """
        return (
            None if fluid_type not in fluid_produced
            else onstream_date.strftime("%d/%m/%Y")
        )

    def _prepare_setup_attr(target, source, default):
        """
        Select appropriate attribute value from source or fallback to default.

        Parameters
        ----------
        target : str
            Attribute name to check in the contract attribute list.
        source : Any
            Candidate value to assign if valid (not None).
        default : Any
            Default value to use if `target` not found or `source` is None.

        Returns
        -------
        Any
            Selected attribute value, either `source` or `default`.
        """
        return (
            default if (target not in contract_attrs) or (source is None)
            else source
        )

    # Mapping variables
    args = {
        "oil": (FluidType.OIL, contract.oil_onstream_date),
        "gas": (FluidType.GAS, contract.gas_onstream_date),
        "approval_year": ("approval_year", contract.approval_year, None),
        "is_pod_1": ("is_pod_1", contract.is_pod_1, False),
    }

    return {
        "start_date": contract.start_date.strftime("%d/%m/%Y"),
        "end_date": contract.end_date.strftime("%d/%m/%Y"),
        "oil_onstream_date": _get_date(*args["oil"]),
        "gas_onstream_date": _get_date(*args["gas"]),
        "approval_year": _prepare_setup_attr(*args["approval_year"]),
        "is_pod_1": _prepare_setup_attr(*args["is_pod_1"]),
    }


def construct_summary_arguments_attr(summary_arguments: dict):
    """
    Process summary arguments by converting all values using convert_object.

    This function iterates through all key-value pairs in the input dictionary
    and applies the `convert_object` function to each value, modifying the
    dictionary in place.

    Parameters
    ----------
    summary_arguments : dict
        Dictionary containing summary arguments where values need to be
        processed/converted. The dictionary will be modified in place.

    Returns
    -------
    dict
        The same dictionary object with all values processed by `convert_object`.
        The dictionary is modified in place and returned for method chaining.

    Notes
    -----
    - This function modifies the input dictionary in place
    - The specific conversion behavior depends on the `convert_object` function
    """

    for key, value in summary_arguments.items():
        summary_arguments[key] = convert_object(objects=value)

    return summary_arguments


def construct_costrecovery_attr(contract: CostRecovery):
    """
    Extract and convert CostRecovery contract attributes to a dictionary.

    Processes CostRecovery contract object attributes across multiple categories
    including FTP, splits, investment credits, DMO, and depreciation. All values
    are processed through convert_object.

    Parameters
    ----------
    contract : CostRecovery
        CostRecovery contract object containing attributes for FTP, tax splits,
        investment credits, DMO, and depreciation settings.

    Returns
    -------
    dict
        Dictionary containing converted CostRecovery attributes with keys for:
        - FTP settings (oil/gas availability, sharing, portions)
        - Tax splits and contractor shares
        - Investment credit rates and cap rates
        - DMO volumes, fees, and holidays
        - Carry forward depreciation values
    """

    cr_setup = {
        # FTP
        "oil_ftp_is_available": contract.oil_ftp_is_available,
        "oil_ftp_is_shared": contract.oil_ftp_is_shared,
        "oil_ftp_portion": contract.oil_ftp_portion,
        "gas_ftp_is_available": contract.gas_ftp_is_available,
        "gas_ftp_is_shared": contract.gas_ftp_is_shared,
        "gas_ftp_portion": contract.gas_ftp_portion,

        # Split
        "tax_split_type": contract.tax_split_type,
        "condition_dict": contract.condition_dict,
        "indicator_rc_icp_sliding": contract.indicator_rc_icp_sliding,
        "oil_ctr_pretax_share": contract.oil_ctr_pretax_share,
        "gas_ctr_pretax_share": contract.gas_ctr_pretax_share,

        # Investment credit and cap rate
        "oil_ic_rate": contract.oil_ic_rate,
        "gas_ic_rate": contract.gas_ic_rate,
        "ic_is_available": contract.ic_is_available,
        "oil_cr_cap_rate": contract.oil_cr_cap_rate,
        "gas_cr_cap_rate": contract.gas_cr_cap_rate,

        # DMO
        "oil_dmo_volume_portion": contract.oil_dmo_volume_portion,
        "oil_dmo_fee_portion": contract.oil_dmo_fee_portion,
        "oil_dmo_holiday_duration": contract.oil_dmo_holiday_duration,
        "gas_dmo_volume_portion": contract.gas_dmo_volume_portion,
        "gas_dmo_fee_portion": contract.gas_dmo_fee_portion,
        "gas_dmo_holiday_duration": contract.gas_dmo_holiday_duration,

        # Carry forward depreciation
        "oil_carry_forward_depreciation": contract.oil_carry_forward_depreciation,
        "gas_carry_forward_depreciation": contract.gas_carry_forward_depreciation,
    }

    for key, value in cr_setup.items():
        cr_setup[key] = convert_object(objects=value)

    return cr_setup


def construct_costrecovery_arguments_attr(contract_arguments: dict):
    """
    Construct processed Cost Recovery contract arguments with default values applied.

    Parameters
    ----------
    contract_arguments : dict
        Dictionary containing user-defined Cost Recovery parameters.

    Returns
    -------
    dict
        Dictionary of processed contract arguments where missing keys are filled
        with default values and all entries are converted using `convert_object()`.

    Notes
    -----
    - Default values are defined in an internal mapping.
    - Each argument is processed using `convert_object()` for type consistency.
    """

    mapping_args = {
        "sulfur_revenue": ("sulfur_revenue", "Addition to Oil Revenue"),
        "electricity_revenue": ("electricity_revenue", "Addition to Oil Revenue"),
        "co2_revenue": ("co2_revenue", "Addition to Oil Revenue"),
        "vat_rate": ("vat_rate", 0.0),
        "inflation_rate": ("inflation_rate", 0.0),
        "inflation_rate_applied_to": ("inflation_rate_applied_to", None),
        "is_dmo_end_weighted": ("is_dmo_end_weighted", False),
        "tax_regime": ("tax_regime", "nailed down"),
        "effective_tax_rate": ("effective_tax_rate", None),
        "ftp_tax_regime": ("ftp_tax_regime", "PDJP No.20 Tahun 2017"),
        "depr_method": ("depr_method", "PSC Declining Balance"),
        "decline_factor": ("decline_factor", 2),
        "post_uu_22_year2001": ("post_uu_22_year2001", True),
        "oil_cost_of_sales_applied": ("oil_cost_of_sales_applied", False),
        "gas_cost_of_sales_applied": ("gas_cost_of_sales_applied", False),
        "sum_undepreciated_cost": ("sum_undepreciated_cost", True),
        "sunk_cost_method": ("sunk_cost_method", "depreciated_tangible"),
    }

    cr_arguments = {
        key: _prepare_args(*val, contract_arguments) for key, val in mapping_args.items()
    }

    for key, value in cr_arguments.items():
        cr_arguments[key] = convert_object(objects=value)

    return cr_arguments


def construct_grosssplit_attr(contract: GrossSplit):
    """
    Construct a dictionary of processed Gross Split contract attributes.

    Parameters
    ----------
    contract : GrossSplit
        The Gross Split contract object containing field, reservoir, DMO,
        and depreciation parameters.

    Returns
    -------
    dict
        Dictionary of processed Gross Split attributes, where all values
        are converted using `convert_object()`.

    Notes
    -----
    - Includes key contract attributes such as field/reservoir properties,
      DMO parameters, and carry-forward depreciation.
    - Ensures type consistency by processing all values with `convert_object()`.
    """

    gs_setup = {
        # Field and reservoir properties
        "field_status": contract.field_status,
        "field_loc": contract.field_loc,
        "res_depth": contract.res_depth,
        "infra_avail": contract.infra_avail,
        "res_type": contract.res_type,
        "api_oil": contract.api_oil,
        "domestic_use": contract.domestic_use,
        "prod_stage": contract.prod_stage,
        "co2_content": contract.co2_content,
        "h2s_content": contract.h2s_content,
        "field_reserves_2024": contract.field_reserves_2024,
        "infra_avail_2024": contract.infra_avail_2024,
        "field_loc_2024": contract.field_loc_2024,
        "split_ministry_disc": contract.split_ministry_disc,

        # DMO parameters
        "oil_dmo_volume_portion": contract.oil_dmo_volume_portion,
        "oil_dmo_fee_portion": contract.oil_dmo_fee_portion,
        "oil_dmo_holiday_duration": contract.oil_dmo_holiday_duration,
        "gas_dmo_volume_portion": contract.gas_dmo_volume_portion,
        "gas_dmo_fee_portion": contract.gas_dmo_fee_portion,
        "gas_dmo_holiday_duration": contract.gas_dmo_holiday_duration,

        # Carry forward depreciation
        "oil_carry_forward_depreciation": contract.oil_carry_forward_depreciation,
        "gas_carry_forward_depreciation": contract.gas_carry_forward_depreciation,
    }

    for key, value in gs_setup.items():
        gs_setup[key] = convert_object(objects=value)

    return gs_setup


def construct_grosssplit_arguments_attr(contract_arguments: dict) -> dict:
    """
    Construct processed Gross Split contract arguments with default values applied.

    Parameters
    ----------
    contract_arguments : dict
        Dictionary containing user-defined Gross Split parameters.

    Returns
    -------
    dict
        Dictionary of processed contract arguments where missing keys are filled
        with default values and all entries are converted using `convert_object()`.

    Notes
    -----
    - Default values are defined in an internal mapping.
    - Ensures type consistency through `convert_object()`.
    """

    mapping_args = {
        "sulfur_revenue": ("sulfur_revenue", "Addition to Oil Revenue"),
        "electricity_revenue": ("electricity_revenue", "Addition to Oil Revenue"),
        "co2_revenue": ("co2_revenue", "Addition to Oil Revenue"),
        "vat_rate": ("vat_rate", 0.0),
        "inflation_rate": ("inflation_rate", 0.0),
        "inflation_rate_applied_to": ("inflation_rate_applied_to", None),
        "cum_production_split_offset": ("cum_production_split_offset", None),
        "depr_method": ("depr_method", "PSC Declining Balance"),
        "decline_factor": ("decline_factor", 2),
        "sum_undepreciated_cost": ("sum_undepreciated_cost", True),
        "is_dmo_end_weighted": ("is_dmo_end_weighted", False),
        "tax_regime": ("tax_regime", "nailed down"),
        "effective_tax_rate": ("effective_tax_rate", 0.22),
        "amortization": ("amortization", False),
        "sunk_cost_method": ("sunk_cost_method", "depreciated_tangible"),
        "regime": ("regime", "PERMEN_ESDM_12_2020"),
        "reservoir_type_permen_2024": ("reservoir_type_permen_2024", "conventional"),
        "initial_amortization_year": ("initial_amortization_year", "onstream_year"),
    }

    gs_arguments = {
        key: _prepare_args(*val, contract_arguments) for key, val in mapping_args.items()
    }

    for key, value in gs_arguments.items():
        gs_arguments[key] = convert_object(objects=value)

    return gs_arguments


def construct_baseproject_arguments_attr(contract_arguments: dict) -> dict:
    """
    Process BaseProject contract arguments with mapping and conversion.

    Maps and converts contract arguments for BaseProject using predefined
    mappings and applies object conversion to all values.

    Parameters
    ----------
    contract_arguments : dict
        Raw contract arguments dictionary.

    Returns
    -------
    dict
        Processed BaseProject arguments with keys: sulfur_revenue,
        electricity_revenue, co2_revenue, tax_rate, year_inflation,
        inflation_rate, inflation_rate_applied_to.
    """

    mapping_args = {
        "sulfur_revenue": ("sulfur_revenue", "Addition to Oil Revenue"),
        "electricity_revenue": ("electricity_revenue", "Addition to Oil Revenue"),
        "co2_revenue": ("co2_revenue", "Addition to Oil Revenue"),
        "tax_rate": ("tax_rate", 0.0),
        "year_inflation": ("year_inflation", None),
        "inflation_rate": ("inflation_rate", 0.0),
        "inflation_rate_applied_to": ("inflation_rate_applied_to", None),
    }

    bp_arguments = {
        key: _prepare_args(*val, contract_arguments) for key, val in mapping_args.items()
    }

    for key, value in bp_arguments.items():
        bp_arguments[key] = convert_object(objects=value)

    return bp_arguments


def construct_transition_attr(contract: Transition):
    """
    Construct a dictionary of processed Transition contract attributes
    """

    if isinstance(contract.contract1, CostRecovery):
        contract_1 = construct_costrecovery_attr(contract=contract.contract1)

    elif isinstance(contract.contract1, GrossSplit):
        contract_1 = construct_grosssplit_attr(contract=contract.contract1)

    else:
        raise GetAttrException(
            f"The contract: {type(contract.contract1)} type is not recognized"
        )

    if isinstance(contract.contract1, CostRecovery):
        contract_2 = construct_costrecovery_attr(contract=contract.contract2)

    elif isinstance(contract.contract1, GrossSplit):
        contract_2 = construct_grosssplit_attr(contract=contract.contract2)

    else:
        raise GetAttrException(
            f"The contract: {type(contract.contract2)} type is not recognized"
        )

    return contract_1, contract_2


def construct_transition_arguments(contract_arguments: dict):
    """
    Construct processed Transition contract arguments with default values applied
    """

    trans_arguments = {
        "unrec_portion": contract_arguments["unrec_portion"],
    }

    for key, value in trans_arguments.items():
        trans_arguments[key] = convert_object(objects=value)

    return trans_arguments


def get_contract_attributes(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    contract_arguments: dict,
    summary_arguments: dict,
) -> dict:
    """
    Construct a comprehensive dictionary of contract attributes in JSON-compatible format.

    This function aggregates various components of a petroleum contract—
    including setup, arguments, lifting data, and cost structures—into a
    unified dictionary ready for JSON serialization.

    Parameters
    ----------
    contract : BaseProject or CostRecovery or GrossSplit or Transition
        The contract object whose attributes will be extracted and processed.
    contract_arguments : dict
        Dictionary containing user-defined or default contract argument values.
    summary_arguments : dict
        Dictionary containing summary-level configuration of the contract.

    Returns
    -------
    dict
        A dictionary containing all contract attributes in a JSON-compatible
        structure. The content and structure depend on the contract type.

    Notes
    -----
    - Automatically detects contract type and calls the corresponding
      attribute-construction functions.
    - Includes setup, summary arguments, contract-specific attributes,
      lifting data, and cost structures.
    - Ensures that all elements are processed through `convert_object()`
      for JSON compatibility.
    """

    # Constructing the setup and summary arguments key
    attr = {
        "setup": construct_setup_attr(contract=contract),
        "summary_arguments": construct_summary_arguments_attr(
            summary_arguments=summary_arguments
        ),
    }

    # Constructing the contract config key
    if isinstance(contract, CostRecovery):
        attr["costrecovery"] = construct_costrecovery_attr(contract=contract)
        attr["contract_arguments"] = construct_costrecovery_arguments_attr(
            contract_arguments=contract_arguments
        )

    elif isinstance(contract, GrossSplit):
        attr["grosssplit"] = construct_grosssplit_attr(contract=contract)
        attr["contract_arguments"] = construct_grosssplit_arguments_attr(
            contract_arguments=contract_arguments
        )

    elif isinstance(contract, Transition):
        attr["contract_1"], attr["contract_2"] = construct_transition_attr(contract=contract)
        attr["contract_arguments"] = construct_transition_arguments(
            contract_arguments=contract_arguments
        )

    elif isinstance(contract, BaseProject):
        attr["contract_arguments"] = construct_baseproject_arguments_attr(
            contract_arguments=contract_arguments
        )

    # Mapping for lifting and costs assignments
    mapping_lifting_costs = (
        ("lifting", construct_lifting_attr, contract.lifting),
        ("capital", construct_cost_attr, contract.capital_cost),
        ("intangible", construct_cost_attr, contract.intangible_cost),
        ("opex", construct_cost_attr, contract.opex),
        ("asr", construct_cost_attr, contract.asr_cost),
        ("lbt", construct_cost_attr, contract.lbt_cost),
        ("cost_of_sales", construct_cost_attr, contract.cost_of_sales),
    )

    # Add `attr` members for lifting and costs
    for (section, builder_func, source_data) in mapping_lifting_costs:
        attr[section] = builder_func(source_data)

    return attr
