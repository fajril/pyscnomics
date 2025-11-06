# """
# A collection of function to generate a synthetic case
# """
#
# import numpy as np
# from datetime import date
# from dataclasses import dataclass, field
#
# from pyscnomics.econ.selection import FluidType, CostType
# from pyscnomics.econ.revenue import Lifting
# from pyscnomics.econ.costs import (
#     CapitalCost,
#     Intangible,
#     OPEX,
#     ASR,
#     LBT,
#     CostOfSales,
# )
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
#
#
# def get_cost_kwargs_dummy():
#     """
#     Construct dummy kwargs to create an instance of cost.
#     """
#
#     kwargs_dummy = {
#         "oil": {
#             "start_year": 2023,
#             "end_year": 2032,
#             "expense_year": np.array(
#                 [
#                     2023, 2024, 2025, 2026, 2027,
#                     2028, 2029, 2030, 2031, 2032,
#                 ]
#             ),
#             "cost_allocation": (
#                 [
#                     FluidType.OIL, FluidType.OIL,
#                     FluidType.OIL, FluidType.OIL,
#                     FluidType.OIL, FluidType.OIL,
#                     FluidType.OIL, FluidType.OIL,
#                     FluidType.OIL, FluidType.OIL,
#                 ]
#             ),
#             "cost_type": (
#                 [
#                     CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
#                     CostType.PRE_ONSTREAM_COST, CostType.SUNK_COST,
#                     CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
#                     CostType.POST_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
#                     CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
#                 ]
#             ),
#             "tax_portion": np.array(
#                 [
#                     1, 1, 1, 1, 1,
#                     1, 1, 1, 1, 1,
#                 ]
#             ),
#         },
#         "gas": {
#             "start_year": 2023,
#             "end_year": 2032,
#             "expense_year": np.array(
#                 [
#                     2023, 2024, 2025, 2026, 2027,
#                     2028, 2029, 2030, 2031, 2032,
#                 ]
#             ),
#             "cost_allocation": (
#                 [
#                     FluidType.GAS, FluidType.GAS,
#                     FluidType.GAS, FluidType.GAS,
#                     FluidType.GAS, FluidType.GAS,
#                     FluidType.GAS, FluidType.GAS,
#                     FluidType.GAS, FluidType.GAS,
#                 ]
#             ),
#             "cost_type": (
#                 [
#                     CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
#                     CostType.PRE_ONSTREAM_COST, CostType.SUNK_COST,
#                     CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
#                     CostType.POST_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
#                     CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
#                 ]
#             ),
#             "tax_portion": np.array(
#                 [
#                     1, 1, 1, 1, 1,
#                     1, 1, 1, 1, 1,
#                 ]
#             ),
#         },
#     }
#
#     return kwargs_dummy
#
#
# @dataclass
# class PrepareLiftingCostsAsClass:
#     """
#     Prepare lifting and costs data
#     """
#
#     lifting: dict = field(default_factory=lambda: {}, init=False, repr=False)
#     capital_cost: dict = field(default_factory=lambda : {}, init=False, repr=False)
#     intangible_cost: dict = field(default_factory=lambda : {}, init=False, repr=False)
#     opex: dict = field(default_factory=lambda : {}, init=False, repr=False)
#     asr_cost: dict = field(default_factory=lambda : {}, init=False, repr=False)
#     lbt_cost: dict = field(default_factory=lambda : {}, init=False, repr=False)
#     cost_of_sales: dict = field(default_factory=lambda : {}, init=False, repr=False)
#
#     def __post_init__(self):
#         self._get_lifting_data()
#         self._get_capital_data()
#         self._get_intangible_data()
#         self._get_opex_data()
#         self._get_asr_data()
#         self._get_lbt_data()
#         self._get_cost_of_sales_data()
#
#     def _get_lifting_data(self) -> None:
#         """
#         Prepare lifting data
#         """
#
#         # Lifting data for OIL
#         lifting_oil = {
#             PoolData.DUMMY: Lifting(
#                 start_year=2023,
#                 end_year=2032,
#                 prod_year=np.array([2030, 2031, 2032]),
#                 lifting_rate=np.array([100, 100, 100]),
#                 price=np.array([120, 120, 120]),
#                 fluid_type=FluidType.OIL,
#             ),
#             PoolData.BENUANG: Lifting(
#                 start_year=2023,
#                 end_year=2037,
#                 prod_year=np.array(
#                     [
#                         2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032,
#                         2033, 2034, 2035
#                     ]
#                 ),
#                 lifting_rate=np.array(
#                     [
#                         460.240475451, 873.845020492, 802.981698938, 932.529841824,
#                         928.962143147, 962.198861205, 592.127288123, 688.191243627,
#                         609.136555531, 459.317272429, 422.929806095, 324.406696975
#                     ]
#                 ),
#                 price=np.array(
#                     [65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65]
#                 ),
#                 fluid_type=FluidType.OIL,
#             ),
#         }
#
#         # Lifting data for GAS
#         lifting_gas = {
#             PoolData.DUMMY: Lifting(
#                 start_year=2023,
#                 end_year=2032,
#                 prod_year=np.array([2029, 2030, 2031]),
#                 lifting_rate=np.array([10, 10, 10]),
#                 price=np.array([1, 1, 1]),
#                 fluid_type=FluidType.GAS,
#             ),
#             PoolData.BENUANG: Lifting(
#                 start_year=2023,
#                 end_year=2037,
#                 prod_year=np.array(
#                     [
#                         2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032,
#                         2033, 2034, 2035
#                     ]
#                 ),
#                 lifting_rate=np.array(
#                     [
#                         1.00495513, 1.96626503, 2.5581599, 4.76693091, 6.0547015,
#                         6.25418605, 4.04543102, 4.58236067, 4.26727342, 2.46265306,
#                         2.34562227, 2.10730217
#                     ]
#                 ),
#                 price=np.array(
#                     [
#                         5.28767105, 5.23662554, 5.20574913, 5.50581852, 5.50165717,
#                         5.50819591, 5.66529057, 5.61780989, 5.70251656, 5.68642526,
#                         5.69206885, 5.66643698
#                     ]
#                 ),
#                 ghv=np.array(
#                     [
#                         1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010,
#                         1010, 1010
#                     ]
#                 ),
#                 fluid_type=FluidType.GAS,
#             ),
#         }
#
#         # Lifting data for SULFUR
#         lifting_sulfur = {
#             PoolData.DUMMY: Lifting(
#                 start_year=2023,
#                 end_year=2032,
#                 prod_year=np.array([2030, 2031, 2032]),
#                 lifting_rate=np.array([10, 10, 10]),
#                 price=np.array([1, 1, 1]),
#                 fluid_type=FluidType.SULFUR,
#             ),
#             PoolData.BENUANG: None,
#         }
#
#         # Lifting data for electricity
#         lifting_electricity = {
#             PoolData.DUMMY: None,
#             PoolData.BENUANG: None,
#         }
#
#         # Lifting data for CO2
#         lifting_co2 = {
#             PoolData.DUMMY: None,
#             PoolData.BENUANG: None,
#         }
#
#         # Specify attribute lifting
#         self.lifting = {
#             "oil": lifting_oil,
#             "gas": lifting_gas,
#             "sulfur": lifting_sulfur,
#             "electricity": lifting_electricity,
#             "co2": lifting_co2,
#         }
#
#     def _get_capital_data(self) -> None:
#         """
#         Prepare capital cost data
#         """
#
#         # Capital cost data (OIL)
#         capital_cost_oil = {
#             PoolData.DUMMY: CapitalCost(
#                 **get_cost_kwargs_dummy()["oil"],
#                 cost=np.array(
#                     [
#                         200, 200, 200, 200,
#                         50, 50, 50, 50, 50, 50,
#                     ]
#                 )
#             ),
#             PoolData.BENUANG: CapitalCost(
#                 start_year=2023,
#                 end_year=2037,
#                 expense_year=np.array(
#                     [
#                         2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
#                     ]
#                 ),
#                 cost=np.array(
#                     [
#                         25714.707321, 21578.906603, 18978.850357, 15273.457471,
#                         7640.015402, 194.226795, 419.762154, 262.961878, 556.462722,
#                         406.618528
#                     ]
#                 ),
#                 cost_allocation=(
#                     [
#                         FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
#                         FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
#                         FluidType.OIL, FluidType.OIL
#                     ]
#                 ),
#                 cost_type=(
#                     [
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
#                     ]
#                 ),
#                 pis_year=np.array(
#                     [
#                         2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
#                     ]
#                 ),
#                 useful_life=np.array(
#                     [
#                         5, 5, 5, 5, 5, 5, 5, 5, 5, 5
#                     ]
#                 ),
#                 depreciation_factor=np.array(
#                     [
#                         0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25
#                     ]
#                 ),
#                 tax_portion=np.array(
#                     [
#                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1
#                     ]
#                 ),
#             ),
#         }
#
#         # Capital cost data (GAS)
#         capital_cost_gas = {
#             PoolData.DUMMY: CapitalCost(
#                 **get_cost_kwargs_dummy()["gas"],
#                 cost=np.array(
#                     [
#                         20, 20, 20, 20,
#                         5, 5, 5, 5, 5, 5,
#                     ]
#                 )
#             ),
#             PoolData.BENUANG: CapitalCost(
#                 start_year=2023,
#                 end_year=2037,
#                 expense_year=np.array(
#                     [
#                         2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
#                     ]
#                 ),
#                 cost_allocation=(
#                     [
#                         FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
#                         FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
#                         FluidType.GAS, FluidType.GAS
#                     ]
#                 ),
#                 cost_type=(
#                     [
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                     ]
#                 ),
#                 cost=np.array(
#                     [
#                         4613.344573, 3950.905221, 4890.837055, 6679.491296, 4256.871068,
#                         108.051922, 252.454555, 152.843852, 345.419185, 192.630464
#                     ]
#                 ),
#                 pis_year=np.array(
#                     [
#                         2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
#                     ]
#                 ),
#                 useful_life=np.array(
#                     [
#                         5, 5, 5, 5, 5, 5, 5, 5, 5, 5
#                     ]
#                 ),
#                 depreciation_factor=np.array(
#                     [
#                         0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25
#                     ]
#                 ),
#                 tax_portion=np.array(
#                     [
#                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1
#                     ]
#                 ),
#             ),
#         }
#
#         # Assign attribute capital_cost
#         self.capital_cost = {
#             "oil": capital_cost_oil,
#             "gas": capital_cost_gas,
#         }
#
#     def _get_intangible_data(self) -> None:
#         """
#         Prepare intangible cost data
#         """
#
#         # Intangible cost data (OIL)
#         intangible_oil = {
#             PoolData.DUMMY: Intangible(
#                 **get_cost_kwargs_dummy()["oil"],
#                 cost=np.array(
#                     [
#                         200, 200, 200, 200,
#                         50, 50, 50, 50, 50, 50,
#                     ]
#                 ),
#             ),
#             PoolData.BENUANG: Intangible(
#                 start_year=2023,
#                 end_year=2037,
#                 expense_year=np.array(
#                     [
#                         2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
#                     ]
#                 ),
#                 cost=np.array(
#                     [
#                         46849.00562, 41944.72775, 22099.70561, 23079.23951, 21928.15442,
#                         718.934075, 978.88238, 830.661181, 1863.113433, 1594.436516
#                     ]
#                 ),
#                 cost_allocation=(
#                     [
#                         FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
#                         FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
#                         FluidType.OIL, FluidType.OIL
#                     ]
#                 ),
#                 cost_type=(
#                     [
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
#                     ]
#                 ),
#                 tax_portion=np.array(
#                     [
#                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#                     ]
#                 ),
#             ),
#         }
#
#         # Intangible cost data (GAS)
#         intangible_gas = {
#             PoolData.DUMMY: Intangible(
#                 **get_cost_kwargs_dummy()["gas"],
#                 cost=np.array(
#                     [
#                         20, 20, 20, 20,
#                         5, 5, 5, 5, 5, 5,
#                     ]
#                 )
#             ),
#             PoolData.BENUANG: Intangible(
#                 start_year=2023,
#                 end_year=2037,
#                 expense_year=np.array(
#                     [
#                         2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
#                     ]
#                 ),
#                 cost=np.array(
#                     [
#                         8404.941309, 7679.705323, 5695.079368, 10093.16847, 12217.94999,
#                         399.956189, 588.722239, 482.813158, 1156.510758, 755.344445
#                     ]
#                 ),
#                 cost_allocation=(
#                     [
#                         FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
#                         FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
#                         FluidType.GAS, FluidType.GAS
#                     ]
#                 ),
#                 cost_type=(
#                     [
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
#                     ]
#                 ),
#                 tax_portion=np.array(
#                     [
#                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#                     ]
#                 ),
#             ),
#         }
#
#         # Assign attribute intangible_cost
#         self.intangible_cost = {
#             "oil": intangible_oil,
#             "gas": intangible_gas
#         }
#
#     def _get_opex_data(self) -> None:
#         """
#         Prepare opex data
#         """
#
#         # OPEX data (OIL)
#         opex_oil = {
#             PoolData.DUMMY: OPEX(
#                 **get_cost_kwargs_dummy()["oil"],
#                 fixed_cost=np.array(
#                     [
#                         200, 200, 200, 200,
#                         50, 50, 50, 50, 50, 50,
#                     ]
#                 )
#             ),
#             PoolData.BENUANG: OPEX(
#                 start_year=2023,
#                 end_year=2037,
#                 expense_year=np.array(
#                     [
#                         2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
#                     ]
#                 ),
#                 fixed_cost=np.array(
#                     [
#                         785.7212242, 4351.968932, 4703.91775, 6325.825984, 6390.700982,
#                         6575.827111, 4434.656934, 4938.484606, 8351.939437, 3282.075895,
#                         3226.479026, 3468.367892
#                     ]
#                 ),
#                 cost_allocation=(
#                     [
#                         FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
#                         FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
#                         FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL
#                     ]
#                 ),
#                 cost_type=(
#                     [
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
#                     ]
#                 ),
#                 tax_portion=np.array(
#                     [
#                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#                     ]
#                 ),
#             ),
#         }
#
#         # OPEX data (GAS)
#         opex_gas = {
#             PoolData.DUMMY: OPEX(
#                 **get_cost_kwargs_dummy()["gas"],
#                 fixed_cost=np.array(
#                     [
#                         20, 20, 20, 20,
#                         5, 5, 5, 5, 5, 5,
#                     ]
#                 )
#             ),
#             PoolData.BENUANG: OPEX(
#                 start_year=2023,
#                 end_year=2037,
#                 expense_year=np.array(
#                     [
#                         2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
#                     ]
#                 ),
#                 fixed_cost=np.array(
#                     [
#                         140.96224, 796.806673, 1212.196461, 2766.452827, 3560.776868,
#                         3658.253024, 2667.104051, 2870.44273, 5184.39062, 1554.842585,
#                         1582.693399, 1983.719401
#                     ]
#                 ),
#                 cost_allocation=(
#                     [
#                         FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
#                         FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
#                         FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS
#                     ]
#                 ),
#                 cost_type=(
#                     [
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
#                     ]
#                 ),
#                 tax_portion=np.array(
#                     [
#                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#                     ]
#                 ),
#             ),
#         }
#
#         # Assign attribute OPEX
#         self.opex = {
#             "oil": opex_oil,
#             "gas": opex_gas,
#         }
#
#     def _get_asr_data(self) -> None:
#         """
#         Prepare ASR data
#         """
#
#         # ASR data (OIL)
#         asr_oil = {
#             PoolData.DUMMY: ASR(
#                 **get_cost_kwargs_dummy()["oil"],
#                 cost=np.array(
#                     [
#                         200, 200, 200, 200,
#                         50, 50, 50, 50, 50, 50,
#                     ]
#                 ),
#             ),
#             PoolData.BENUANG: ASR(
#                 start_year=2023,
#                 end_year=2037,
#                 expense_year=np.array(
#                     [
#                         2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
#                     ]
#                 ),
#                 cost=np.array(
#                     [
#                         173.28832, 316.4429143, 387.4743104, 688.6020255, 717.0204906,
#                         726.1726918, 708.4741216, 779.1434817, 770.3641929, 730.4658999
#                     ]
#                 ),
#                 cost_allocation=(
#                     [
#                         FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
#                         FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
#                         FluidType.OIL, FluidType.OIL
#                     ]
#                 ),
#                 cost_type=(
#                     [
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                     ]
#                 ),
#                 final_year=np.array(
#                     [
#                         2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
#                     ]
#                 ),
#                 tax_portion=np.array(
#                     [
#                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#                     ]
#                 ),
#             ),
#         }
#
#         # ASR data (GAS)
#         asr_gas = {
#             PoolData.DUMMY: ASR(
#                 **get_cost_kwargs_dummy()["gas"],
#                 cost=np.array(
#                     [
#                         20, 20, 20, 20,
#                         5, 5, 5, 5, 5, 5,
#                     ]
#                 )
#             ),
#             PoolData.BENUANG: ASR(
#                 start_year=2023,
#                 end_year=2037,
#                 expense_year=np.array(
#                     [
#                         2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
#                     ]
#                 ),
#                 cost=np.array(
#                     [
#                         44.65628428, 138.3889467, 215.8933058, 383.081915,
#                         431.2325133, 422.0803121, 439.7788823, 369.1095222,
#                         377.8888109, 417.7871039,
#                     ]
#                 ),
#                 cost_allocation=(
#                     [
#                         FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
#                         FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
#                         FluidType.GAS, FluidType.GAS
#                     ]
#                 ),
#                 cost_type=(
#                     [
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
#                         CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
#                     ]
#                 ),
#                 final_year=np.array(
#                     [
#                         2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
#                     ]
#                 ),
#                 tax_portion=np.array(
#                     [
#                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#                     ]
#                 ),
#             ),
#         }
#
#         # Assign attribute asr_cost
#         self.asr_cost = {
#             "oil": asr_oil,
#             "gas": asr_gas,
#         }
#
#     def _get_lbt_data(self) -> None:
#         """
#         Prepare LBT data
#         """
#
#         # LBT data (OIL)
#         lbt_oil = {
#             PoolData.DUMMY: LBT(
#                 **get_cost_kwargs_dummy()["oil"],
#                 cost=np.array(
#                     [
#                         200, 200, 200, 200,
#                         50, 50, 50, 50, 50, 50,
#                     ]
#                 )
#             ),
#             PoolData.BENUANG: None,
#         }
#
#         # LBT data (GAS)
#         lbt_gas = {
#             PoolData.DUMMY: LBT(
#                 **get_cost_kwargs_dummy()["gas"],
#                 cost=np.array(
#                     [
#                         20, 20, 20, 20,
#                         5, 5, 5, 5, 5, 5,
#                     ]
#                 )
#             ),
#             PoolData.BENUANG: None,
#         }
#
#         # Assign attribute lbt_cost
#         self.lbt_cost = {
#             "oil": lbt_oil,
#             "gas": lbt_gas,
#         }
#
#     def _get_cost_of_sales_data(self) -> None:
#         """
#         Prepare cost of sales data
#         """
#
#         # Cost of sales data (OIL)
#         cos_oil = {
#             PoolData.DUMMY: CostOfSales(
#                 **get_cost_kwargs_dummy()["oil"],
#                 cost=np.array(
#                     [
#                         200, 200, 200, 200,
#                         50, 50, 50, 50, 50, 50,
#                     ]
#                 )
#             ),
#             PoolData.BENUANG: None,
#         }
#
#         # Cost of sales data (GAS)
#         cos_gas = {
#             PoolData.DUMMY: CostOfSales(
#                 **get_cost_kwargs_dummy()["gas"],
#                 cost=np.array(
#                     [
#                         20, 20, 20, 20,
#                         5, 5, 5, 5, 5, 5,
#                     ]
#                 )
#             ),
#             PoolData.BENUANG: None,
#         }
#
#         # Assign attribute cost_of_sales
#         self.cost_of_sales = {
#             "oil": cos_oil,
#             "gas": cos_gas,
#         }
