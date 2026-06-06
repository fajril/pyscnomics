"""
A collection of operations to run optimization module.
"""

# import numpy as np
# from datetime import date
# from pyscnomics.example import ExampleCase
# from pyscnomics.contracts.project import BaseProject
# from pyscnomics.contracts.costrecovery import CostRecovery
# from pyscnomics.contracts.grossplit import GrossSplit
# from pyscnomics.econ.selection import (
#     TaxSplitTypeCR,
#     OtherRevenue,
#     InflationAppliedTo,
#     TaxRegime,
#     FTPTaxRegime,
#     DeprMethod,
#     SunkCostMethod,
#     InitialYearAmortizationIncurred,
#     GrossSplitRegime,
#     VariableSplit082017,
#     VariableSplit132024,
#     NPVSelection,
#     DiscountingMode,
# )


def optimization_arguments_dict():
    """
    Returns a dictionary containing default optimization settings.

    Returns
    -------
    dict
        Dictionary with target parameter, target value, and optimization
        variable configuration (name, minimum, and maximum limits).
    """
    return {
        "target_parameter": "IRR",
        "target_value": 0.3,
        "dict_optimization": {
            "parameter": "VAT Discount",
            "min": 0.1,
            "max": 0.6,
        },
    }
