"""
Collection of classes or modules to be disposed in the future version.
"""

# @dataclass
# class SunkCost(GeneralCost):
#     """
#     Manages sunk cost.
#
#     This class inherits the attributes from class GeneralCost.
#
#     Parameters
#     ----------
#     pod1_year: int
#         The year of POD I approval.
#     investment_type: list[SunkCostInvestmentType]
#         The type of investment (TANGIBLE or INTANGIBLE).
#     salvage_value: np.ndarray
#         The value of the asset at the end of useful period.
#     depreciation_period: np.ndarray
#         The period of depreciation (analogue to useful_life in class CapitalCost).
#     depreciation_factor: np.ndarray
#         The value of depreciation factor to be used in PSC_DB depreciation method.
#         Default value is 0.5 for the entire project duration.
#     """
#
#     # Local arguments
#     onstream_year: int = field(default=None)
#     pod1_year: int = field(default=None)
#     investment_type: list[SunkCostInvestmentType] = field(default=None)
#     salvage_value: np.ndarray = field(default=None)
#     depreciation_period: np.ndarray = field(default=None)
#     depreciation_factor: np.ndarray = field(default=None)
#
#     # Overridden argument
#     cost: np.ndarray = field(default=None)
#
#     def __post_init__(self):
#         """
#         Handles the following operations/procedures:
#         - Prepare attribute project_duration and project_years,
#         - Prepare attribute pod1_year,
#         - Prepare attribute expense_year,
#         - Raise an error: expense year is after the end year of the project,
#         - Raise an error: expense year is before the start year of the project,
#         - Prepare attribute cost,
#         - Prepare attribute investment_type,
#         - Prepare attribute cost_allocation,
#         - Prepare attribute description,
#         - Prepare attribute salvage_value,
#         - Prepare attribute depreciation_period,
#         - Prepare attribute depreciation_factor,
#         - Prepare attribute tax_portion,
#         - Prepare attribute tax_discount,
#         - Check input data for unequal length of arrays,
#         """
#
#         # Prepare attribute project_duration and project_years
#         if self.end_year >= self.start_year:
#             self.project_duration = self.end_year - self.start_year + 1
#             self.project_years = np.arange(self.start_year, self.end_year + 1, 1)
#
#         else:
#             raise SunkCostException(
#                 f"start year {self.start_year} is after the end year {self.end_year} "
#                 f"of the project"
#             )
#
#         # Prepare attribute pod1_year
#         if self.pod1_year is None:
#             raise SunkCostException(f"Missing data for pod1_year: {self.pod1_year}")
#
#         else:
#             if not isinstance(self.pod1_year, int):
#                 raise SunkCostException(
#                     f"Attribute pod1_year must be provided as an int, "
#                     f"not as a/an {self.pod1_year.__class__.__qualname__}"
#                 )
#
#             if self.pod1_year < self.start_year:
#                 raise SunkCostException(
#                     f"POD I year ({self.pod1_year}) is before the start year "
#                     f"of the project ({self.start_year})"
#                 )
#
#             if self.pod1_year > self.end_year:
#                 raise SunkCostException(
#                     f"POD I year ({self.pod1_year}) is after the end year "
#                     f"of the project ({self.end_year})"
#                 )
#
#         # Prepare attribute expense_year
#         if not isinstance(self.expense_year, np.ndarray):
#             raise SunkCostException(
#                 f"Attribute expense_year must be provided as a numpy.ndarray, "
#                 f"not as a/an {self.expense_year.__class__.__qualname__}"
#             )
#
#         else:
#             expense_year_nan_sum = np.sum(pd.isna(self.expense_year), dtype=np.float64)
#             if expense_year_nan_sum > 0:
#                 raise SunkCostException(
#                     f"Missing values in array expense_year: {self.expense_year}"
#                 )
#
#         self.expense_year = self.expense_year.astype(int)
#
#         # Raise an error: expense year is after the end year of the project
#         if np.max(self.expense_year) > self.end_year:
#             raise SunkCostException(
#                 f"Expense year ({np.max(self.expense_year)}) "
#                 f"is after the end year of the project ({self.end_year})"
#             )
#
#         # Raise an error: expense_year is before the start year of the project
#         if np.min(self.expense_year) < self.start_year:
#             raise SunkCostException(
#                 f"Expense year ({np.min(self.expense_year)}) "
#                 f"is before the start year of the project ({self.start_year})"
#             )
#
#         # Prepare attribute cost
#         if self.cost is None:
#             self.cost = np.zeros_like(self.project_years, dtype=np.float64)
#
#         else:
#             if not isinstance(self.cost, np.ndarray):
#                 raise SunkCostException(
#                     f"Attribute cost must be provided as a numpy.ndarray, "
#                     f"not as a/an {self.cost.__class__.__qualname__}"
#                 )
#
#             else:
#                 cost_nan_id = np.argwhere(pd.isna(self.cost)).ravel()
#                 if len(cost_nan_id) > 0:
#                     self.cost[cost_nan_id] = np.zeros(len(cost_nan_id))
#
#         self.cost = self.cost.astype(np.float64)
#
#         # Prepare attribute investment_type
#         if self.investment_type is None:
#             self.investment_type = [
#                 SunkCostInvestmentType.TANGIBLE for _ in range(len(self.expense_year))
#             ]
#
#         else:
#             if not isinstance(self.investment_type, list):
#                 raise SunkCostException(
#                     f"Attribute investment_type must be given as a list, "
#                     f"not as a/an {self.investment_type.__class__.__qualname__}"
#                 )
#
#             self.investment_type = [
#                 SunkCostInvestmentType.TANGIBLE if pd.isna(val) else val
#                 for _, val in enumerate(self.investment_type)
#             ]
#
#         # Prepare attribute cost_allocation
#         if self.cost_allocation is None:
#             self.cost_allocation = [
#                 FluidType.OIL for _ in range(len(self.expense_year))
#             ]
#
#         else:
#             if not isinstance(self.cost_allocation, list):
#                 raise SunkCostException(
#                     f"Attribute cost_allocation must be given as a list, "
#                     f"not as a/an {self.cost_allocation.__class__.__qualname__}"
#                 )
#
#             self.cost_allocation = [
#                 FluidType.OIL if pd.isna(val) else val
#                 for _, val in enumerate(self.cost_allocation)
#             ]
#
#         # Prepare attribute description
#         if self.description is None:
#             self.description = [" " for _ in range(len(self.expense_year))]
#
#         else:
#             if not isinstance(self.description, list):
#                 raise SunkCostException(
#                     f"Attribute description must be provided as a list, "
#                     f"not as a/an {self.description.__class__.__qualname__}"
#                 )
#
#             self.description = [
#                 " " if pd.isna(val) else val for _, val in enumerate(self.description)
#             ]
#
#         # Prepare attribute salvage_value
#         if self.salvage_value is None:
#             self.salvage_value = np.zeros_like(self.expense_year)
#
#         else:
#             if not isinstance(self.salvage_value, np.ndarray):
#                 raise SunkCostException(
#                     f"Attribute salvage_value must be provided as a numpy.ndarray, "
#                     f"not as an/a {self.salvage_value.__class__.__qualname__}"
#                 )
#
#             salvage_value_nan_id = np.argwhere(pd.isna(self.salvage_value)).ravel()
#             if len(salvage_value_nan_id) > 0:
#                 self.salvage_value[salvage_value_nan_id] = np.zeros(
#                     len(salvage_value_nan_id)
#                 )
#
#         self.salvage_value = self.salvage_value.astype(np.float64)
#
#         # Prepare attribute depreciation_period
#         if self.depreciation_period is None:
#             self.depreciation_period = np.repeat(5.0, len(self.expense_year))
#
#         else:
#             if not isinstance(self.depreciation_period, np.ndarray):
#                 raise SunkCostException(
#                     f"Attribute depreciation_period must be given as a numpy.ndarray, "
#                     f"not as an/a {self.depreciation_period.__class__.__qualname__}"
#                 )
#
#             depreciation_period_nan_id = np.argwhere(
#                 pd.isna(self.depreciation_period)
#             ).ravel()
#             if len(depreciation_period_nan_id) > 0:
#                 self.depreciation_period[depreciation_period_nan_id] = np.repeat(
#                     5.0, len(depreciation_period_nan_id)
#                 )
#
#         self.depreciation_period = self.depreciation_period.astype(np.float64)
#
#         # Prepare attribute depreciation_factor
#         if self.depreciation_factor is None:
#             self.depreciation_factor = np.repeat(0.5, len(self.expense_year))
#
#         else:
#             if not isinstance(self.depreciation_factor, np.ndarray):
#                 raise SunkCostException(
#                     f"Attribute depreciation_factor must be given as a numpy.ndarray, "
#                     f"not as an/a {self.depreciation_factor.__class__.__qualname__}"
#                 )
#
#             depreciation_factor_nan_id = np.argwhere(
#                 pd.isna(self.depreciation_factor)
#             ).ravel()
#             if len(depreciation_factor_nan_id) > 0:
#                 self.depreciation_factor[depreciation_factor_nan_id] = np.repeat(
#                     0.5, len(depreciation_factor_nan_id)
#                 )
#
#             depreciation_factor_large = np.sum(
#                 self.depreciation_factor > 1.0, dtype=int
#             )
#             depreciation_factor_negative = np.sum(
#                 self.depreciation_factor < 0.0, dtype=int
#             )
#             if depreciation_factor_large > 0 or depreciation_factor_negative > 0:
#                 raise SunkCostException(
#                     f"The value of depreciation_factor must be within the "
#                     f"following interval: 0 < depreciation_factor < 1, "
#                     f"depreciation_factor: {self.depreciation_factor}"
#                 )
#
#         self.depreciation_factor = self.depreciation_factor.astype(np.float64)
#
#         # Prepare attribute tax_portion
#         if self.tax_portion is None:
#             self.tax_portion = np.zeros_like(self.expense_year)
#
#         else:
#             if not isinstance(self.tax_portion, np.ndarray):
#                 raise SunkCostException(
#                     f"Attribute tax_portion must be given as a numpy.ndarray, "
#                     f"not as a/an {self.tax_portion.__class__.__qualname__}"
#                 )
#
#             tax_portion_nan_id = np.argwhere(pd.isna(self.tax_portion)).ravel()
#             if len(tax_portion_nan_id) > 0:
#                 self.tax_portion[tax_portion_nan_id] = np.zeros(len(tax_portion_nan_id))
#
#             tax_portion_large = np.sum(self.tax_portion > 1.0, dtype=np.float64)
#             tax_portion_negative = np.sum(self.tax_portion < 0.0, dtype=np.float64)
#             if tax_portion_large > 0 or tax_portion_negative > 0:
#                 raise SunkCostException(
#                     f"The value of tax_portion must be: 0 < tax_portion < 1, "
#                     f"tax_portion: {self.tax_portion}"
#                 )
#
#         self.tax_portion = self.tax_portion.astype(np.float64)
#
#         # Prepare attribute tax_discount
#         if not isinstance(self.tax_discount, (float, int, np.ndarray)):
#             raise SunkCostException(
#                 f"Attribute tax_discount must be provided as a float/int "
#                 f"or as a numpy.ndarray, not as a/an "
#                 f"{self.tax_discount.__class__.__qualname__}"
#             )
#
#         if isinstance(self.tax_discount, (float, int)):
#             if self.tax_discount < 0 or self.tax_discount > 1:
#                 raise SunkCostException(
#                     f"Attribute tax_discount must be between 0 and 1"
#                 )
#
#             self.tax_discount = np.repeat(self.tax_discount, len(self.expense_year))
#
#         elif isinstance(self.tax_discount, np.ndarray):
#             tax_discount_nan_id = np.argwhere(pd.isna(self.tax_discount)).ravel()
#             if len(tax_discount_nan_id) > 0:
#                 self.tax_discount[tax_discount_nan_id] = np.zeros(
#                     len(tax_discount_nan_id)
#                 )
#
#             tax_discount_large = np.sum(self.tax_discount > 1.0, dtype=np.float64)
#             tax_discount_negative = np.sum(self.tax_discount < 0.0, dtype=np.float64)
#             if tax_discount_large > 0 or tax_discount_negative > 0:
#                 raise SunkCostException(
#                     f"The value of tax_discount must be: 0 < tax_discount < 1, "
#                     f"tax_discount: {self.tax_discount}"
#                 )
#
#         self.tax_discount = self.tax_discount.astype(np.float64)
#
#         # Check input data for unequal length of arrays
#         arr_reference = len(self.expense_year)
#
#         if not all(
#             len(arr) == arr_reference
#             for arr in [
#                 self.cost,
#                 self.investment_type,
#                 self.cost_allocation,
#                 self.description,
#                 self.salvage_value,
#                 self.depreciation_period,
#                 self.depreciation_factor,
#                 self.tax_portion,
#                 self.tax_discount,
#             ]
#         ):
#             raise SunkCostException(
#                 f"Unequal length of arrays: "
#                 f"expense_year: {len(self.expense_year)}, "
#                 f"cost: {len(self.cost)}, "
#                 f"investment_type: {len(self.investment_type)}, "
#                 f"cost_allocation: {len(self.cost_allocation)}, "
#                 f"description: {len(self.description)}, "
#                 f"salvage_value: {len(self.salvage_value)}, "
#                 f"depreciation_period: {len(self.depreciation_period)}, "
#                 f"depreciation_factor: {len(self.depreciation_factor)}, "
#                 f"tax_portion: {len(self.tax_portion)}, "
#                 f"tax_discount: {len(self.tax_discount)} "
#             )
#
#     def _get_sunk_cost_id(self, fluid_type: FluidType) -> np.ndarray:
#         """
#         Get the indices of sunk costs for a specific fluid type (OIL or GAS).
#
#         The method identifies which costs are considered sunk cost based on the
#         relationship between the POD I approval year and the onstream year, and
#         filters them by the specified fluid type (OIL or GAS).
#
#         Parameters
#         ----------
#         fluid_type : FluidType
#             The fluid type to filter by (must be either OIL or GAS).
#
#         Returns
#         -------
#         np.ndarray
#             Array of indices representing sunk costs for the specified fluid type.
#
#         Raises
#         ------
#         SunkCostException
#             If POD I year occurs after the onstream year, which is an invalid scenario.
#
#         Notes
#         -----
#         The sunk cost determination depends on the timing relationship:
#         1.  If POD I year equals onstream year:
#             costs expensed before/on onstream year are considered as sunk cost.
#         2.  If POD I year is before onstream year:
#             costs expensed before/on POD I year are sunk considered as sunk cost.
#
#         The method filters these sunk costs by the specified fluid type (OIL or GAS).
#         """
#
#         # Year of POD I approval equals to onstream year
#         if self.pod1_year == self.onstream_year:
#
#             # Determine sunk cost ID for a particular fluid type (OIL or GAS)
#             sc_id = np.argwhere(self.expense_year <= self.onstream_year).ravel()
#             sc_cost_allocation_id = np.array(
#                 [self.cost_allocation[val] for _, val in enumerate(sc_id)]
#             )
#             sc_fluid_type_id = np.array(
#                 [
#                     sc_id[i]
#                     for i, val in enumerate(sc_cost_allocation_id)
#                     if val == fluid_type
#                 ]
#             )
#
#         # Year of POD I approval is before the onstream year
#         elif self.pod1_year < self.onstream_year:
#
#             # Determine sunk cost ID for a particular fluid type (OIL or GAS)
#             sc_id = np.argwhere(self.expense_year <= self.pod1_year).ravel()
#             sc_cost_allocation_id = np.array(
#                 [self.cost_allocation[val] for _, val in enumerate(sc_id)]
#             )
#             sc_fluid_type_id = np.array(
#                 [
#                     sc_id[i]
#                     for i, val in enumerate(sc_cost_allocation_id)
#                     if val == fluid_type
#                 ]
#             )
#
#         else:
#             raise SunkCostException(f"Cannot have POD I year after the onstream year")
#
#         return sc_fluid_type_id
#
#     def _get_preonstream_cost_id(self, fluid_type: FluidType) -> np.ndarray:
#         """
#         Get the indices of pre-onstream costs for a specific fluid type (OIL or GAS).
#
#         Identifies which costs are considered pre-onstream cost based on the relationship
#         between the POD I approval year and the onstream year, and filters them by the
#         specified fluid type (OIL or GAS).
#
#         Parameters
#         ----------
#         fluid_type : FluidType
#             The fluid type to filter by (must be either OIL or GAS).
#
#         Returns
#         -------
#         np.ndarray
#             Array of indices representing pre-onstream costs for the specified fluid type.
#             Returns empty array if POD I year equals onstream year.
#
#         Raises
#         ------
#         SunkCostException
#             If POD I year occurs after the onstream year, which is an invalid scenario.
#
#         Notes
#         -----
#         The pre-onstream cost determination depends on the timing relationship:
#         1.  If POD I year equals onstream year:
#             returns empty array (no pre-onstream costs)
#         2.  If POD I year is before onstream year:
#             costs expensed between POD I year and onstream year
#
#         The method filters these pre-onstream costs by the specified fluid type (OIL or GAS).
#         """
#
#         # Year of POD I approval equals to onstream year
#         if self.pod1_year == self.onstream_year:
#
#             # Determine pre-onstream cost ID for a particular fluid type (OIL or GAS)
#             poc_fluid_type_id = np.array([])
#
#         # Year of POD I approval is before the onstream year
#         elif self.pod1_year < self.onstream_year:
#
#             # Determine pre-onstream cost ID for a particular fluid type (OIL or GAS)
#             poc_id = np.argwhere(self.expense_year >= self.pod1_year).ravel()
#             poc_cost_allocation_id = np.array(
#                 [self.cost_allocation[val] for _, val in enumerate(poc_id)]
#             )
#             poc_fluid_type_id = np.array(
#                 [
#                     poc_id[i]
#                     for i, val in enumerate(poc_cost_allocation_id)
#                     if val == fluid_type
#                 ]
#             )
#
#         else:
#             raise SunkCostException(f"Cannot have POD I year after the onstream year")
#
#         return poc_fluid_type_id
#
#     def _get_sunk_cost_investment_id(
#         self,
#         fluid_type: FluidType,
#         investment_config: SunkCostInvestmentType,
#     ) -> np.ndarray:
#         """
#         Get the indices of sunk costs filtered by both fluid type and investment type.
#
#         This method first identifies sunk costs for a specific fluid type (OIL or GAS),
#         then further filters them by the specified investment type (Tangible or Intangible).
#
#         Parameters
#         ----------
#         fluid_type : FluidType
#             The fluid type to filter by (must be either OIL or GAS).
#         investment_config : SunkCostInvestmentType
#             The investment type to filter by. Must be one of:
#             - SunkCostInvestmentType.TANGIBLE: For tangible investments
#             - SunkCostInvestmentType.INTANGIBLE: For intangible investments
#
#         Returns
#         -------
#         np.ndarray
#             Array of indices representing sunk costs that match both the specified
#             fluid type and investment type. Returns empty array if no matching costs found.
#
#         Notes
#         -----
#         The method works in two stages:
#         1. First calls `_get_sunk_cost_id` to get sunk costs for the specified fluid type
#         2. Then filters these costs by the specified investment type (Tangible/Intangible)
#
#         If no sunk costs exist for the fluid type, returns an empty array immediately.
#
#         See Also
#         --------
#         _get_sunk_cost_id : Method used to get sunk costs by fluid type.
#         SunkCostInvestmentType : Enum defining the investment type options.
#         """
#
#         # Identify sunk cost id for a particular fluid type (OIL or GAS)
#         sc_fluid_type_id = self._get_sunk_cost_id(fluid_type=fluid_type)
#
#         # Determine sunk cost id for a particular investment type
#         # (TANGIBLE or INTANGIBLE)
#         if len(sc_fluid_type_id) == 0:
#             sc_investment_id = np.array([])
#
#         else:
#             sc_investment_id = np.array(
#                 [
#                     sc_fluid_type_id[i]
#                     for i, val in enumerate(
#                         np.array(self.investment_type)[sc_fluid_type_id]
#                     )
#                     if val == investment_config
#                 ]
#             )
#
#         return sc_investment_id
#
#     def _get_preonstream_cost_investment_id(
#         self,
#         fluid_type: FluidType,
#         investment_config: SunkCostInvestmentType,
#     ) -> np.ndarray:
#         """
#         Get the indices of pre-onstream costs filtered by both fluid type and investment type.
#
#         This method first identifies pre-onstream costs for a specific fluid type (OIL or GAS),
#         then further filters them by the specified investment type (Tangible or Intangible).
#
#         Parameters
#         ----------
#         fluid_type : FluidType
#             The fluid type to filter by (must be either OIL or GAS).
#         investment_config : SunkCostInvestmentType
#             The investment type to filter by. Must be one of:
#             - SunkCostInvestmentType.TANGIBLE: For tangible investments
#             - SunkCostInvestmentType.INTANGIBLE: For intangible investments
#
#         Returns
#         -------
#         np.ndarray
#             Array of indices representing pre-onstream costs that match both the specified
#             fluid type and investment type. Returns empty array if:
#             - No pre-onstream costs exist for the fluid type, or
#             - No costs match the specified investment type
#
#         Notes
#         -----
#         The method works in two stages:
#         1. First calls `_get_preonstream_cost_id` to get pre-onstream costs for the fluid type
#         2. Then filters these costs by the specified investment type (Tangible/Intangible)
#
#         See Also
#         --------
#         _get_preonstream_cost_id : Method used to get pre-onstream costs by fluid type.
#         SunkCostInvestmentType : Enum defining the investment type options.
#         _get_sunk_cost_investment_id : Similar method for sunk costs.
#         """
#
#         # Identify pre-onstream cost id for a particular fluid type (OIL or GAS)
#         poc_fluid_type_id = self._get_preonstream_cost_id(fluid_type=fluid_type)
#
#         # Determine preonstream cost id for a particular investment type
#         # (TANGIBLE or INTANGIBLE)
#         if len(poc_fluid_type_id) == 0:
#             poc_investment_id = np.array([])
#
#         else:
#             poc_investment_id = np.array(
#                 [
#                     poc_fluid_type_id[i]
#                     for i, val in enumerate(
#                         np.array(self.investment_type)[poc_fluid_type_id]
#                     )
#                     if val == investment_config
#                 ]
#             )
#
#         return poc_investment_id
#
#     def get_indirect_tax(self, tax_rate: np.ndarray | float = 0.0) -> np.ndarray:
#         """
#         Calculate indirect tax for the project.
#
#         This method computes the indirect tax by calling the `calc_indirect_tax` function
#         with the object's attributes as parameters.
#
#         Parameters
#         ----------
#         tax_rate : np.ndarray or float, optional
#             The tax rate(s) to apply. Can be a single float or an array of rates.
#             Default is 0.0 (no tax).
#
#         Returns
#         -------
#         np.ndarray
#             Array of calculated indirect tax values for each project year.
#
#         Notes
#         -----
#         The actual calculation is delegated to the `calc_indirect_tax` function which uses:
#         - The project's start year
#         - Cost values
#         - Expense year information
#         - Project duration
#         - Tax portion
#         - Provided tax rate(s)
#         - Tax discount information
#
#         See Also
#         --------
#         calc_indirect_tax : The function that performs the actual tax calculation.
#         """
#
#         return calc_indirect_tax(
#             start_year=self.start_year,
#             cost=self.cost,
#             expense_year=self.expense_year,
#             project_years=self.project_years,
#             tax_portion=self.tax_portion,
#             tax_rate=tax_rate,
#             tax_discount=self.tax_discount,
#         )
#
#     def get_sunk_cost_investment_array(
#         self,
#         fluid_type: FluidType,
#         investment_config: SunkCostInvestmentType,
#         tax_rate: np.ndarray | float = 0.0,
#     ) -> np.ndarray:
#         """
#         Calculate sunk cost investment array adjusted for tax and allocated by expense year.
#
#         This method computes an array of sunk costs for a specific fluid and investment type,
#         adjusted for VAT/indirect tax, and distributed across project years based on when
#         the expenses occurred.
#
#         Parameters
#         ----------
#         fluid_type : FluidType
#             The fluid type to filter by (OIL or GAS).
#         investment_config : SunkCostInvestmentType
#             The investment type to filter by (TANGIBLE or INTANGIBLE).
#         tax_rate : np.ndarray or float, optional
#             The tax rate(s) to apply for VAT adjustment. Can be a single float or array.
#             Default is 0.0 (no tax).
#
#         Returns
#         -------
#         np.ndarray
#             An array of length `project_duration` containing:
#             - Sunk costs adjusted for tax, allocated to their respective expense years
#             - Zeros for years with no expenses
#             - All zeros if no matching costs found
#
#         Notes
#         -----
#         The method performs the following operations:
#         1. Adjusts costs by adding indirect tax (VAT)
#         2. Identifies relevant sunk costs using fluid and investment type filters
#         3. Allocates costs to expense years using bincount
#         4. Pads with zeros for remaining project duration
#
#         The output array length matches the project duration, with costs positioned
#         according to (expense_year - start_year).
#         """
#
#         # Adjust cost by VAT
#         cost_adjusted_by_vat = self.cost + self.get_indirect_tax(tax_rate=tax_rate)
#
#         # Identify sunk cost indices for a particular fluid type (OIL or GAS)
#         # and for a particular investment type (TANGIBLE or INTANGIBLE)
#         sc_investment_id = self._get_sunk_cost_investment_id(
#             fluid_type=fluid_type,
#             investment_config=investment_config,
#         )
#
#         # For non-empty "sc_investment_id" array
#         if len(sc_investment_id) > 0:
#
#             # Extract sunk cost for a particular fluid type (OIL or GAS) and
#             # for a particular investment type (TANGIBLE or INTANGIBLE)
#             sc_investment = cost_adjusted_by_vat[sc_investment_id]
#
#             # Allocate the extracted sunk cost by their associated
#             # expense year in array project_years
#             sc_investment_expenses = np.bincount(
#                 self.expense_year[sc_investment_id] - self.start_year,
#                 weights=sc_investment,
#             )
#             zeros = np.zeros(self.project_duration - len(sc_investment_expenses))
#             sc_investment_array = np.concatenate(
#                 (sc_investment_expenses, zeros), dtype=np.float64
#             )
#
#         # For empty "sc_investment_id" array
#         else:
#             sc_investment_array = np.zeros_like(self.project_years, dtype=np.float64)
#
#         return sc_investment_array
#
#     def get_preonstream_cost_investment_array(
#         self,
#         fluid_type: FluidType,
#         investment_config: SunkCostInvestmentType,
#         tax_rate: np.ndarray | float = 0.0,
#     ) -> np.ndarray:
#         """
#         Calculate pre-onstream cost investment array adjusted for tax and allocated
#         by expense year.
#
#         This method computes an array of pre-onstream costs for a specific fluid
#         and investment type, adjusted for VAT/indirect tax, and distributed across
#         project years based on when the expenses occurred.
#
#         Parameters
#         ----------
#         fluid_type : FluidType
#             The fluid type to filter by (OIL or GAS).
#         investment_config : SunkCostInvestmentType
#             The investment type to filter by (TANGIBLE or INTANGIBLE).
#         tax_rate : np.ndarray or float, optional
#             The tax rate(s) to apply for VAT adjustment. Can be a single float or array.
#             Default is 0.0 (no tax).
#
#         Returns
#         -------
#         np.ndarray
#             An array of length `project_duration` containing:
#             - Pre-onstream costs adjusted for tax, allocated to their respective expense years
#             - Zeros for years with no expenses
#             - All zeros if no matching costs found
#
#         Notes
#         -----
#         The method performs the following operations:
#         1. Adjusts costs by adding indirect tax (VAT)
#         2. Identifies relevant pre-onstream costs using fluid and investment type filters
#         3. Allocates costs to expense years using bincount
#         4. Pads with zeros for remaining project duration
#
#         The output array length matches the project duration, with costs positioned
#         according to (expense_year - start_year). Pre-onstream costs are defined as
#         costs incurred between POD I approval and onstream date.
#         """
#
#         # Adjust cost by VAT
#         cost_adjusted_by_vat = self.cost + self.get_indirect_tax(tax_rate=tax_rate)
#
#         # Identify the preonstream cost indices for a particular fluid type (OIL or GAS)
#         # and for a particular investment type (TANGIBLE or INTANGIBLE)
#         poc_investment_id = self._get_preonstream_cost_investment_id(
#             fluid_type=fluid_type,
#             investment_config=investment_config,
#         )
#
#         # For non-empty "poc_investment_id" array
#         if len(poc_investment_id) > 0:
#
#             # Extract preonstream cost for a particular fluid type (OIL or GAS) and
#             # for a particular investment type (TANGIBLE or INTANGIBLE)
#             poc_investment = cost_adjusted_by_vat[poc_investment_id]
#
#             # Allocate the extracted preonstream cost by their associated
#             # expense year in array project_years
#             poc_investment_expenses = np.bincount(
#                 self.expense_year[poc_investment_id] - self.start_year,
#                 weights=poc_investment,
#             )
#             zeros = np.zeros(self.project_duration - len(poc_investment_expenses))
#             poc_investment_array = np.concatenate(
#                 (poc_investment_expenses, zeros), dtype=np.float64
#             )
#
#         # For empty "poc_investment_id" array
#         else:
#             poc_investment_array = np.zeros_like(self.project_years, dtype=np.float64)
#
#         return poc_investment_array
#
#     @staticmethod
#     def get_investment_bulk(cost_investment_array: np.ndarray) -> float:
#         """
#         Compute the total sum of all investments in the given array.
#
#          Parameters
#          ----------
#          cost_investment_array : np.ndarray
#              Array containing investment cost values to be summed.
#              This parameter could be one of the following:
#              - sc_oil_tangible_array
#              - sc_oil_intangible_array
#              - sc_gas_tangible_array
#              - sc_gas_intangible_array
#              - poc_oil_tangible_array
#              - poc_oil_intangible_array
#              - poc_gas_tangible_array
#              - poc_gas_intangible_array
#
#          Returns
#          -------
#          float
#              The total sum of all investment costs in the input array.
#              The sum is computed using float64 precision to maintain numerical accuracy.
#         """
#
#         return np.sum(cost_investment_array, dtype=np.float64)
#
#     def get_amortization_charge(
#         self,
#         cost_bulk: float,
#         prod: np.ndarray,
#         prod_year: np.ndarray,
#         salvage_value: float = 0.0,
#         amortization_len: int = 0,
#     ) -> np.ndarray:
#         """
#         Calculate amortization charges using the unit of production method.
#
#         This method computes the amortization expense based on actual production
#         volumes each year relative to total production over the asset's life.
#
#         Parameters
#         ----------
#         cost_bulk : float
#             The total sunk cost or preonstream cost to be amortized.
#             This parameter could be one of the following:
#             - sc_oil_tangible_bulk
#             - sc_oil_intangible_bulk
#             - sc_gas_tangible_bulk
#             - sc_gas_intangible_bulk
#             - poc_oil_tangible_bulk
#             - poc_oil_intangible_bulk
#             - poc_gas_tangible_bulk
#             - poc_gas_intangible_bulk
#         prod : np.ndarray
#             Array of production volumes for each year.
#         prod_year : np.ndarray
#             Array of years corresponding to the production volumes.
#             Must start at the same year as the asset's onstream year.
#         salvage_value : float, optional
#             The estimated residual value at the end of amortization (default=0.0).
#         amortization_len : int, optional
#             The maximum number of years over which to amortize (default=0 for no limit).
#
#         Returns
#         -------
#         np.ndarray
#             Array of amortization charges for each production year.
#
#         Raises
#         ------
#         SunkCostException
#             If the first production year doesn't match the asset's onstream year.
#
#         Notes
#         -----
#         - The calculation is delegated to `unit_of_production_rate()` function.
#         - Amortization continues until either:
#             * All production is accounted for, or
#             * The amortization length is reached (if specified)
#         """
#
#         # The start of production year must be the same with the onstream year
#         if np.min(prod_year) != self.onstream_year:
#             raise SunkCostException(
#                 f"Unequal value for the start of production ({np.min(prod_year)}) "
#                 f"and the onstream year ({self.onstream_year})"
#             )
#
#         # Calculate amortization charge by calling function unit_of_production_rate()
#         return depr.unit_of_production_rate(
#             start_year_project=self.start_year,
#             cost=cost_bulk,
#             prod_year=prod_year,
#             prod=prod,
#             salvage_value=salvage_value,
#             amortization_len=amortization_len,
#         )
#
#     def get_amortization_book_value(
#         self,
#         cost_investment_array: np.ndarray,
#         cost_bulk: float,
#         prod: np.ndarray,
#         prod_year: np.ndarray,
#         salvage_value: float = 0.0,
#         amortization_len: int = 0,
#     ) -> np.ndarray:
#         """
#         Calculate the amortization book value over time for an investment.
#
#         The book value is computed as the cumulative investment costs minus
#         the cumulative amortization charges, showing the remaining undepreciated
#         value of the asset over time.
#
#         Parameters
#         ----------
#         cost_investment_array : np.ndarray
#             Array of investment costs for each period.
#             This parameter could be one of the following:
#             - sc_oil_tangible_array
#             - sc_oil_intangible_array
#             - sc_gas_tangible_array
#             - sc_gas_intangible_array
#             - poc_oil_tangible_array
#             - poc_oil_intangible_array
#             - poc_gas_tangible_array
#             - poc_gas_intangible_array
#         cost_bulk : float
#             Total capital cost to be amortized (depreciated).
#             This parameter could be one of the following:
#             - sc_oil_tangible_bulk
#             - sc_oil_intangible_bulk
#             - sc_gas_tangible_bulk
#             - sc_gas_intangible_bulk
#             - poc_oil_tangible_bulk
#             - poc_oil_intangible_bulk
#             - poc_gas_tangible_bulk
#             - poc_gas_intangible_bulk
#         prod : np.ndarray
#             Array of production volumes for each year.
#         prod_year : np.ndarray
#             Array of years corresponding to the production volumes.
#             Must start at the same year as the asset's onstream year.
#         salvage_value : float, optional
#             The estimated residual value at the end of amortization (default=0.0).
#         amortization_len : int, optional
#             The maximum number of years over which to amortize (default=0 for no limit).
#
#         Returns
#         -------
#         np.ndarray
#             Array of book values for each period, calculated as:
#             cumulative_investment - cumulative_amortization
#
#         Notes
#         -----
#         - Relies on `get_amortization_charge()` to compute the amortization schedule.
#         - The book value will never go below the salvage value.
#         - Both investment costs and amortization charges are cumulatively summed.
#         """
#
#         # Calculate amortization charge
#         amortization_charge = self.get_amortization_charge(
#             cost_bulk=cost_bulk,
#             prod=prod,
#             prod_year=prod_year,
#             salvage_value=salvage_value,
#             amortization_len=amortization_len,
#         )
#
#         # Calculate amortization book value
#         return np.cumsum(cost_investment_array) - np.cumsum(amortization_charge)
#
#     def get_sunk_cost_tangible_depreciation_charge(
#         self,
#         fluid_type: FluidType,
#         depr_method: DeprMethod = DeprMethod.PSC_DB,
#         decline_factor: float | int = 2,
#         tax_rate: np.ndarray | float = 0.0,
#     ) -> tuple:
#         """
#         Calculate depreciation rates for tangible sunk costs based on specified method.
#
#         Computes depreciation charges for tangible assets associated with either oil or gas
#         production, applying the selected depreciation method. Returns both the annual
#         depreciation charges and any remaining undepreciated asset value.
#
#         Parameters
#         ----------
#         fluid_type : FluidType
#             The fluid type (OIL or GAS) for which to calculate depreciation.
#         depr_method : DeprMethod, optional
#             The depreciation method to apply (default: PSC_DB). Options:
#             - SL: Straight Line
#             - DB: Declining Balance
#             - PSC_DB: PSC Declining Balance
#         decline_factor : float | int, optional
#             The acceleration factor for declining balance methods (default: 2).
#         tax_rate : np.ndarray | float, optional
#             Tax rate(s) to adjust costs (default: 0.0). Can be a single value or array.
#
#         Returns
#         -------
#         tuple[np.ndarray, float]
#             A tuple containing:
#             - total_depreciation_charge: np.ndarray
#                 Annual depreciation charges for each project year
#             - undepreciated_asset: float
#                 Remaining asset value after all depreciation
#
#         Raises
#         ------
#         SunkCostException
#             If an unrecognized depreciation method is specified.
#
#         Notes
#         -----
#         - Costs are adjusted by indirect taxes (VAT) before depreciation calculation.
#         - Depreciation schedules are aligned with expense years for each cost component.
#         - Three depreciation methods are supported:
#             1. Straight Line (SL)
#             2. Declining Balance (DB)
#             3. PSC Declining Balance (PSC_DB)
#         - If no tangible assets exist for the specified fluid type, returns zero arrays.
#         """
#
#         # Adjust cost by VAT
#         cost_adjusted_by_vat = self.cost + self.get_indirect_tax(tax_rate=tax_rate)
#
#         # Identify sunk cost tangible indices for a particular fluid type (OIL or GAS)
#         sc_tangible_id = self._get_sunk_cost_investment_id(
#             fluid_type=fluid_type,
#             investment_config=SunkCostInvestmentType.TANGIBLE,
#         )
#
#         # Operations to be conducted when a particular sunk cost tangible
#         # indices are available
#         if len(sc_tangible_id) > 0:
#
#             # Calculate depreciation for every cost element in array cost_adjusted_by_vat
#             # Depreciation method is straight line
#             if depr_method == DeprMethod.SL:
#                 depreciation_charge = np.array(
#                     [
#                         depr.straight_line_depreciation_rate(
#                             cost=c,
#                             salvage_value=sv,
#                             useful_life=dp,
#                             depreciation_len=self.project_duration,
#                         )
#                         for c, sv, dp in zip(
#                             cost_adjusted_by_vat[sc_tangible_id],
#                             self.salvage_value[sc_tangible_id],
#                             self.depreciation_period[sc_tangible_id],
#                         )
#                     ]
#                 )
#
#             # Depreciation method is declining balance
#             elif depr_method == DeprMethod.DB:
#                 depreciation_charge = np.array(
#                     [
#                         depr.declining_balance_depreciation_rate(
#                             cost=c,
#                             salvage_value=sv,
#                             useful_life=dp,
#                             decline_factor=decline_factor,
#                             depreciation_len=self.project_duration,
#                         )
#                         for c, sv, dp in zip(
#                             cost_adjusted_by_vat[sc_tangible_id],
#                             self.salvage_value[sc_tangible_id],
#                             self.depreciation_period[sc_tangible_id],
#                         )
#                     ]
#                 )
#
#             # Depreciation method is PSC declining balance
#             elif depr_method == DeprMethod.PSC_DB:
#                 depreciation_charge = np.array(
#                     [
#                         depr.psc_declining_balance_depreciation_rate(
#                             cost=c,
#                             depreciation_factor=df,
#                             useful_life=dp,
#                             depreciation_len=self.project_duration,
#                         )
#                         for c, df, dp in zip(
#                             cost_adjusted_by_vat[sc_tangible_id],
#                             self.depreciation_factor[sc_tangible_id],
#                             self.depreciation_period[sc_tangible_id],
#                         )
#                     ]
#                 )
#
#             else:
#                 raise SunkCostException(
#                     f"Depreciation method ({depr_method}) is not recognized"
#                 )
#
#             # The relative difference between expense_year and start_year
#             shift_indices = self.expense_year[sc_tangible_id] - self.start_year
#
#             # Modify depreciation_charge so that expenditures are aligned
#             # with the corresponding expense_year
#             depreciation_charge = np.array(
#                 [
#                     np.concatenate((np.zeros(i), row[:-i])) if i > 0 else row
#                     for row, i in zip(depreciation_charge, shift_indices)
#                 ]
#             )
#
#             # Calculate total depreciation charge for all cost components
#             # and the associated undepreciated asset (if any)
#             total_depreciation_charge = depreciation_charge.sum(axis=0, keepdims=False)
#             undepreciated_asset = np.sum(cost_adjusted_by_vat[sc_tangible_id]) - np.sum(
#                 total_depreciation_charge
#             )
#
#         # Operations to be conducted when a particular sunk cost tangible
#         # indices are unavailable
#         else:
#             total_depreciation_charge = np.zeros_like(
#                 self.project_years, dtype=np.float64
#             )
#             undepreciated_asset = 0.0
#
#         return total_depreciation_charge, undepreciated_asset
#
#     def get_preonstream_cost_tangible_depreciation_charge(
#         self,
#         fluid_type: FluidType,
#         depr_method: DeprMethod = DeprMethod.PSC_DB,
#         decline_factor: float | int = 2,
#         tax_rate: np.ndarray | float = 0.0,
#     ) -> tuple:
#         """
#         Calculate depreciation rates for tangible pre-onstream costs based on specified method.
#
#         Computes depreciation charges for tangible assets incurred before production start
#         (pre-onstream) for either oil or gas projects. Returns both the annual depreciation
#         charges and any remaining undepreciated asset value.
#
#         Parameters
#         ----------
#         fluid_type : FluidType
#             The fluid type (OIL or GAS) for which to calculate depreciation.
#         depr_method : DeprMethod, optional
#             The depreciation method to apply (default: PSC_DB). Options:
#             - SL: Straight Line
#             - DB: Declining Balance
#             - PSC_DB: PSC Declining Balance
#         decline_factor : float | int, optional
#             The acceleration factor for declining balance methods (default: 2).
#         tax_rate : np.ndarray | float, optional
#             Tax rate(s) to adjust costs (default: 0.0). Can be a single value or array.
#
#         Returns
#         -------
#         tuple[np.ndarray, float]
#             A tuple containing:
#             - total_depreciation_charge: np.ndarray
#                 Annual depreciation charges for each project year, aligned with expense years
#             - undepreciated_asset: float
#                 Remaining asset value after all depreciation charges
#
#         Raises
#         ------
#         SunkCostException
#             If an unrecognized depreciation method is specified.
#
#         Notes
#         -----
#         - Costs are adjusted by indirect taxes (VAT) before depreciation calculation
#         - Depreciation schedules are time-shifted to align with actual expense years
#         - Handles three depreciation methods:
#             1. Straight Line (equal annual charges)
#             2. Declining Balance (accelerated depreciation)
#             3. PSC-specific Declining Balance
#         - Returns zero arrays if no pre-onstream tangible costs exist for the fluid type
#         - For PSC_DB method, uses predefined depreciation factors instead of salvage values
#         """
#
#         # Adjust cost by VAT
#         cost_adjusted_by_vat = self.cost + self.get_indirect_tax(tax_rate=tax_rate)
#
#         # Identify preonstream cost tangible indices for a particular fluid type (OIL or GAS)
#         poc_tangible_id = self._get_preonstream_cost_investment_id(
#             fluid_type=fluid_type,
#             investment_config=SunkCostInvestmentType.TANGIBLE,
#         )
#
#         # Operations to be conducted when a particular preonstream
#         # cost tangible indices are available
#         if len(poc_tangible_id) > 0:
#
#             # Calculate depreciation for every cost element in array cost_adjusted_by_vat
#             # Depreciation method is straight line
#             if depr_method == DeprMethod.SL:
#                 depreciation_charge = np.array(
#                     [
#                         depr.straight_line_depreciation_rate(
#                             cost=c,
#                             salvage_value=sv,
#                             useful_life=dp,
#                             depreciation_len=self.project_duration,
#                         )
#                         for c, sv, dp, in zip(
#                             cost_adjusted_by_vat[poc_tangible_id],
#                             self.salvage_value[poc_tangible_id],
#                             self.depreciation_period[poc_tangible_id],
#                         )
#                     ]
#                 )
#
#             # Depreciation method is declining balance
#             elif depr_method == DeprMethod.DB:
#                 depreciation_charge = np.array(
#                     [
#                         depr.declining_balance_depreciation_rate(
#                             cost=c,
#                             salvage_value=sv,
#                             useful_life=dp,
#                             decline_factor=decline_factor,
#                             depreciation_len=self.project_duration,
#                         )
#                         for c, sv, dp in zip(
#                             cost_adjusted_by_vat[poc_tangible_id],
#                             self.salvage_value[poc_tangible_id],
#                             self.depreciation_period[poc_tangible_id],
#                         )
#                     ]
#                 )
#
#             # Depreciation method is PSC declining balance
#             elif depr_method == DeprMethod.PSC_DB:
#                 depreciation_charge = np.array(
#                     [
#                         depr.psc_declining_balance_depreciation_rate(
#                             cost=c,
#                             depreciation_factor=df,
#                             useful_life=dp,
#                             depreciation_len=self.project_duration,
#                         )
#                         for c, df, dp, in zip(
#                             cost_adjusted_by_vat[poc_tangible_id],
#                             self.depreciation_factor[poc_tangible_id],
#                             self.depreciation_period[poc_tangible_id],
#                         )
#                     ]
#                 )
#
#             else:
#                 raise SunkCostException(
#                     f"Depreciation method ({depr_method}) is not recognized"
#                 )
#
#             # The relative difference between expense_year and start_year
#             shift_indices = self.expense_year[poc_tangible_id] - self.start_year
#
#             # Modify depreciation_charge so that expenditures are aligned
#             # with the corresponding expense_year
#             depreciation_charge = np.array(
#                 [
#                     np.concatenate((np.zeros(i), row[:-i])) if i > 0 else row
#                     for row, i in zip(depreciation_charge, shift_indices)
#                 ]
#             )
#
#             # Calculate total depreciation charge for all cost components
#             # and the associated undepreciated asset (if any)
#             total_depreciation_charge = depreciation_charge.sum(axis=0, keepdims=False)
#             undepreciated_asset = np.sum(
#                 cost_adjusted_by_vat[poc_tangible_id]
#             ) - np.sum(total_depreciation_charge)
#
#         # Operations to be conducted when a particular preonstream
#         # cost tangible indices are unavailable
#         else:
#             total_depreciation_charge = np.zeros_like(
#                 self.project_years, dtype=np.float64
#             )
#             undepreciated_asset = 0.0
#
#         return total_depreciation_charge, undepreciated_asset
#
#     def get_sunk_cost_tangible_depreciation_book_value(
#         self,
#         fluid_type: FluidType,
#         depr_method: DeprMethod = DeprMethod.PSC_DB,
#         decline_factor: float | int = 2,
#         tax_rate: np.ndarray | float = 0.0,
#     ) -> np.ndarray:
#         """
#         Calculate the book value over time for tangible sunk costs after depreciation.
#
#         The book value represents the remaining undepreciated value of tangible assets
#         for a specific fluid type (OIL or GAS) over the project duration, computed as
#         the cumulative investment costs minus cumulative depreciation charges.
#
#         Parameters
#         ----------
#         fluid_type : FluidType
#             The fluid type (OIL or GAS) for which to calculate book values.
#         depr_method : DeprMethod, optional
#             The depreciation method to apply (default: PSC_DB). Options:
#             - SL: Straight Line
#             - DB: Declining Balance
#             - PSC_DB: PSC Declining Balance
#         decline_factor : float | int, optional
#             The acceleration factor for declining balance methods (default: 2).
#         tax_rate : np.ndarray | float, optional
#             Tax rate(s) to adjust costs (default: 0.0). Can be a single value or array.
#
#         Returns
#         -------
#         np.ndarray
#             Array of book values for each period, calculated as:
#             cumulative_investment - cumulative_depreciation
#
#         Notes
#         -----
#         - Combines results from:
#             * get_sunk_cost_investment_array() for cumulative investment costs
#             * get_sunk_cost_tangible_depreciation_charge() for depreciation charges
#         - Uses tax-adjusted costs for accurate valuation
#         - Time-aligned with project duration years
#         """
#
#         # Calculate sunk cost tangible array for a particular fluid type (OIL or GAS)
#         sc_tangible_array = self.get_sunk_cost_investment_array(
#             fluid_type=fluid_type,
#             investment_config=SunkCostInvestmentType.TANGIBLE,
#             tax_rate=tax_rate,
#         )
#
#         # Calculate sunk cost tangible depreciation charge for a particular
#         # fluid type (OIL or GAS)
#         sc_tangible_depreciation_charge = (
#             self.get_sunk_cost_tangible_depreciation_charge(
#                 fluid_type=fluid_type,
#                 depr_method=depr_method,
#                 decline_factor=decline_factor,
#                 tax_rate=tax_rate,
#             )[0]
#         )
#
#         return np.cumsum(sc_tangible_array) - np.cumsum(sc_tangible_depreciation_charge)
#
#     def get_preonstream_cost_tangible_depreciation_book_value(
#         self,
#         fluid_type: FluidType,
#         depr_method: DeprMethod = DeprMethod.PSC_DB,
#         decline_factor: float | int = 2,
#         tax_rate: np.ndarray | float = 0.0,
#     ) -> np.ndarray:
#         """
#         Calculate the book value over time for pre-onstream tangible costs after
#         depreciation.
#
#         Computes the remaining undepreciated value of tangible assets incurred before
#         production start (pre-onstream) for a specific fluid type, showing how the
#         asset value declines over the project duration.
#
#         Parameters
#         ----------
#         fluid_type : FluidType
#             The fluid type (OIL or GAS) for which to calculate book values.
#         depr_method : DeprMethod, optional
#             The depreciation method to apply (default: PSC_DB). Options:
#             - SL: Straight Line
#             - DB: Declining Balance
#             - PSC_DB: PSC Declining Balance
#         decline_factor : float | int, optional
#             The acceleration factor for declining balance methods (default: 2).
#         tax_rate : np.ndarray | float, optional
#             Tax rate(s) to adjust costs (default: 0.0). Can be a single value or array.
#
#         Returns
#         -------
#         np.ndarray
#             Array of book values for each period, calculated as:
#             cumulative_preonstream_costs - cumulative_depreciation_charges
#
#         Notes
#         -----
#         - Relies on:
#             * get_preonstream_cost_investment_array() for cumulative pre-onstream costs
#             * get_preonstream_cost_tangible_depreciation_charge() for depreciation charges
#         - Book value represents the remaining capitalizable value of pre-production assets
#         - Automatically handles time-alignment between costs and depreciation schedules
#         - Returns zero array if no pre-onstream costs exist for the specified fluid type
#         """
#
#         # Calculate preonstream cost tangible array for a particular fluid type (OIL or GAS)
#         poc_tangible_array = self.get_preonstream_cost_investment_array(
#             fluid_type=fluid_type,
#             investment_config=SunkCostInvestmentType.TANGIBLE,
#             tax_rate=tax_rate,
#         )
#
#         poc_tangible_depreciation_charge = (
#             self.get_preonstream_cost_tangible_depreciation_charge(
#                 fluid_type=fluid_type,
#                 depr_method=depr_method,
#                 decline_factor=decline_factor,
#                 tax_rate=tax_rate,
#             )[0]
#         )
#
#         return np.cumsum(poc_tangible_array) - np.cumsum(
#             poc_tangible_depreciation_charge
#         )
#
#     def __eq__(self, other):
#         # Between two instances of SunkCost
#         if isinstance(other, SunkCost):
#             return all(
#                 (
#                     self.start_year == other.start_year,
#                     self.end_year == other.end_year,
#                     self.onstream_year == other.onstream_year,
#                     self.pod1_year == other.pod1_year,
#                     np.allclose(self.expense_year, other.expense_year),
#                     np.allclose(self.cost, other.cost),
#                     np.allclose(self.salvage_value, other.salvage_value),
#                     np.allclose(self.depreciation_period, other.depreciation_period),
#                     np.allclose(self.depreciation_factor, other.depreciation_factor),
#                     np.allclose(self.tax_portion, other.tax_portion),
#                     np.allclose(self.tax_discount, other.tax_discount),
#                     self.cost_allocation == other.cost_allocation,
#                     self.investment_type == other.investment_type,
#                 )
#             )
#
#         # Between an instance of SunkCost and an integer/float
#         elif isinstance(other, (int, float)):
#             return np.sum(self.cost) == other
#
#         else:
#             return False
#
#     def __lt__(self, other):
#         # Between an instance of SunkCost with another instance of
#         # CapitalCost/Intangible/OPEX/ASR/LBT/SunkCost
#         if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, SunkCost)):
#             return np.sum(self.cost) < np.sum(other.cost)
#
#         # Between an instance of SunkCost and an integer/a float
#         elif isinstance(other, (int, float)):
#             return np.sum(self.cost) < other
#
#         else:
#             raise SunkCostException(
#                 f"Must compare an instance of SunkCost with another instance of "
#                 f"CapitalCost/Intangible/OPEX/ASR/LBT/SunkCost, an integer, or a float."
#             )
#
#     def __le__(self, other):
#         # Between an instance of SunkCost with another instance of
#         # CapitalCost/Intangible/OPEX/ASR/LBT/SunkCost
#         if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, SunkCost)):
#             return np.sum(self.cost) <= np.sum(other.cost)
#
#         # Between an instance of SunkCost and an integer/a float
#         elif isinstance(other, (int, float)):
#             return np.sum(self.cost) <= other
#
#         else:
#             raise SunkCostException(
#                 f"Must compare an instance of SunkCost with another instance of "
#                 "CapitalCost/Intangible/OPEX/ASR/LBT/SunkCost, an integer, or a float."
#             )
#
#     def __gt__(self, other):
#         # Between an instance of SunkCost with another instance of
#         # CapitalCost/Intangible/OPEX/ASR/LBT/SunkCost
#         if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, SunkCost)):
#             return np.sum(self.cost) > np.sum(other.cost)
#
#         # Between an instance of SunkCost and an integer/a float
#         elif isinstance(other, (int, float)):
#             return np.sum(self.cost) > other
#
#         else:
#             raise SunkCostException(
#                 f"Must compare an instance of SunkCost with another instance of "
#                 f"CapitalCost/Intangible/OPEX/ASR/LBT/SunkCost, an integer, or a float."
#             )
#
#     def __ge__(self, other):
#         # Between an instance of SunkCost with another instance of
#         # CapitalCost/Intangible/OPEX/ASR/LBT/SunkCost
#         if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, SunkCost)):
#             return np.sum(self.cost) >= np.sum(other.cost)
#
#         # Between an instance of SunkCost and an integer/a float
#         elif isinstance(other, (int, float)):
#             return np.sum(self.cost) >= other
#
#         else:
#             raise SunkCostException(
#                 f"Must compare an instance of SunkCost with another instance of "
#                 f"CapitalCost/Intangible/OPEX/ASR/LBT/SunkCost, an integer, or a float."
#             )
#
#     def __add__(self, other):
#         # Only allows addition between an instance of SunkCost
#         # and another instance of SunkCost
#         if isinstance(other, SunkCost):
#             start_year_combined = min(self.start_year, other.start_year)
#             end_year_combined = max(self.end_year, other.end_year)
#             onstream_year_combined = None
#             pod1_year_combined = min(self.pod1_year, other.pod1_year)
#             expense_year_combined = np.concatenate(
#                 (self.expense_year, other.expense_year)
#             )
#             cost_combined = np.concatenate((self.cost, other.cost))
#             salvage_value_combined = np.concatenate(
#                 (self.salvage_value, other.salvage_value)
#             )
#             depreciation_period_combined = np.concatenate(
#                 (self.depreciation_period, other.depreciation_period)
#             )
#             depreciation_factor_combined = np.concatenate(
#                 (self.depreciation_factor, other.depreciation_factor)
#             )
#             tax_portion_combined = np.concatenate((self.tax_portion, other.tax_portion))
#             tax_discount_combined = np.concatenate(
#                 (self.tax_discount, other.tax_discount)
#             )
#             cost_allocation_combined = self.cost_allocation + other.cost_allocation
#             description_combined = self.description + other.description
#             investment_type_combined = self.investment_type + other.investment_type
#
#             return SunkCost(
#                 start_year=start_year_combined,
#                 end_year=end_year_combined,
#                 onstream_year=onstream_year_combined,
#                 pod1_year=pod1_year_combined,
#                 expense_year=expense_year_combined,
#                 cost=cost_combined,
#                 cost_allocation=cost_allocation_combined,
#                 description=description_combined,
#                 salvage_value=salvage_value_combined,
#                 depreciation_period=depreciation_period_combined,
#                 depreciation_factor=depreciation_factor_combined,
#                 tax_portion=tax_portion_combined,
#                 tax_discount=tax_discount_combined,
#                 investment_type=investment_type_combined,
#             )
#
#         else:
#             raise SunkCostException(
#                 f"Must add an instance of SunkCost with another instance of SunkCost, "
#                 f"{other}: ({other.__class__.__qualname__}) is not an instance of SunkCost."
#             )
#
#     def __iadd__(self, other):
#         return self.__add__(other)
#
#     def __sub__(self, other):
#         # Only allows subtraction between an instance of SunkCost
#         # and another instance of SunkCost
#         if isinstance(other, SunkCost):
#             start_year_combined = min(self.start_year, other.start_year)
#             end_year_combined = max(self.end_year, other.end_year)
#             onstream_year_combined = None
#             pod1_year_combined = min(self.pod1_year, other.pod1_year)
#             expense_year_combined = np.concatenate(
#                 (self.expense_year, other.expense_year)
#             )
#             cost_combined = np.concatenate((self.cost, -other.cost))
#             salvage_value_combined = np.concatenate(
#                 (self.salvage_value, other.salvage_value)
#             )
#             depreciation_period_combined = np.concatenate(
#                 (self.depreciation_period, other.depreciation_period)
#             )
#             depreciation_factor_combined = np.concatenate(
#                 (self.depreciation_factor, other.depreciation_factor)
#             )
#             tax_portion_combined = np.concatenate((self.tax_portion, other.tax_portion))
#             tax_discount_combined = np.concatenate(
#                 (self.tax_discount, other.tax_discount)
#             )
#             cost_allocation_combined = self.cost_allocation + other.cost_allocation
#             description_combined = self.description + other.description
#             investment_type_combined = self.investment_type + other.investment_type
#
#             return SunkCost(
#                 start_year=start_year_combined,
#                 end_year=end_year_combined,
#                 onstream_year=onstream_year_combined,
#                 pod1_year=pod1_year_combined,
#                 expense_year=expense_year_combined,
#                 cost=cost_combined,
#                 cost_allocation=cost_allocation_combined,
#                 description=description_combined,
#                 salvage_value=salvage_value_combined,
#                 depreciation_period=depreciation_period_combined,
#                 depreciation_factor=depreciation_factor_combined,
#                 tax_portion=tax_portion_combined,
#                 tax_discount=tax_discount_combined,
#                 investment_type=investment_type_combined,
#             )
#
#         else:
#             raise SunkCostException(
#                 f"Must subtract an instance of SunkCost with another instance of "
#                 f"SunkCost, {other}: ({other.__class__.__qualname__}) is not an "
#                 f"instance of SunkCost."
#             )
#
#     def __rsub__(self, other):
#         return self.__sub__(other)
#
#     def __mul__(self, other):
#         # Multiplication is allowed only with an integer or a float
#         if isinstance(other, (int, float)):
#             return SunkCost(
#                 start_year=self.start_year,
#                 end_year=self.end_year,
#                 onstream_year=None,
#                 pod1_year=self.pod1_year,
#                 expense_year=self.expense_year,
#                 cost=self.cost * other,
#                 cost_allocation=self.cost_allocation,
#                 description=self.description,
#                 salvage_value=self.salvage_value,
#                 depreciation_period=self.depreciation_period,
#                 depreciation_factor=self.depreciation_factor,
#                 tax_portion=self.tax_portion,
#                 tax_discount=self.tax_discount,
#                 investment_type=self.investment_type,
#             )
#
#         else:
#             raise SunkCostException(
#                 f"Must multiply with an integer or a float. "
#                 f"{other}: ({other.__class__.__qualname__}) is not an integer "
#                 f"nor a float."
#             )
#
#     def __rmul__(self, other):
#         return self.__mul__(other)
#
#     def __truediv__(self, other):
#         # Between an instance of SunkCost with another instance of
#         # CapitalCost/Intangible/OPEX/ASR/LBT/SunkCost
#         if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, SunkCost)):
#             return np.sum(self.cost) / np.sum(other.cost)
#
#         # Between an instance of SunkCost and an integer or a float
#         elif isinstance(other, (int, float)):
#             # Cannot divide by zero
#             if other == 0:
#                 raise SunkCostException(f"Cannot divide by zero")
#
#             else:
#                 return SunkCost(
#                     start_year=self.start_year,
#                     end_year=self.end_year,
#                     onstream_year=None,
#                     pod1_year=self.pod1_year,
#                     expense_year=self.expense_year,
#                     cost=self.cost / other,
#                     cost_allocation=self.cost_allocation,
#                     description=self.description,
#                     salvage_value=self.salvage_value,
#                     depreciation_period=self.depreciation_period,
#                     depreciation_factor=self.depreciation_factor,
#                     tax_portion=self.tax_portion,
#                     tax_discount=self.tax_discount,
#                     investment_type=self.investment_type,
#                 )
#
#         else:
#             raise SunkCostException(
#                 f"Must divide with an instance of CapitalCost/Intangible/OPEX/"
#                 f"ASR/LBT/SunkCost, an integer, or a float. "
#                 f"{other}: ({other.__class__.__qualname__}) is not an intance of "
#                 f"CapitalCost/Intangible/OPEX/ASR/LBT/SunkCost nor an integer "
#                 f"nor a float."
#             )



def _get_oil_capital(self) -> CapitalCost:
    """
    Determines total oil CapitalCost from the number of oil CapitalCost instances in
    attribute self.capital_cost_total.

    Returns
    -------
    CapitalCost
        An instance of CapitalCost that only includes FluidType.OIL as the associated
        cost_allocation that has been combined altogether following the rules prescribed
        in the dunder method __add__() of CapitalCost class.

    Notes
    -----
    The core operations are as follows:
    (1) Check the attribute cost_allocation in attribute self.capital_cost_total,
    (2) If OIL is not available as an instance in attribute self.capital_cost_total,
        then establish a new instance of OIL CapitalCost with the following attribute set
        to zero: cost.
    (3) Identify index location where cost_allocation is FluidType.OIL in attribute
        self.capital_cost_total,
    (4) Create a new instance of CapitalCost with only FluidType.OIL as its cost_allocation.
    """

    if FluidType.OIL not in self.capital_cost_total.cost_allocation:
        return CapitalCost(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=np.array([self.start_date.year]),
            cost=np.array([0]),
            cost_allocation=[FluidType.OIL],
        )

    else:
        # Configure indices of OIL capital cost
        mask = np.logical_and(
            np.array(self.capital_cost_total.cost_allocation) == FluidType.OIL,
            np.array(self.capital_cost_total.is_sunkcost) == False
        )

        oil_capital_id = np.flatnonzero(mask)

        start_year = self.capital_cost_total.start_year
        end_year = self.capital_cost_total.end_year
        expense_year = self.capital_cost_total.expense_year[oil_capital_id]
        cost = self.capital_cost_total.cost[oil_capital_id]
        cost_allocation = np.array(self.capital_cost_total.cost_allocation)[oil_capital_id]
        description = np.array(self.capital_cost_total.description)[oil_capital_id]
        is_sunkcost = np.array(self.capital_cost_total.is_sunkcost)[oil_capital_id]
        tax_portion = self.capital_cost_total.tax_portion[oil_capital_id]
        tax_discount = self.capital_cost_total.tax_discount[oil_capital_id]
        pis_year = self.capital_cost_total.pis_year[oil_capital_id]
        salvage_value = self.capital_cost_total.salvage_value[oil_capital_id]
        useful_life = self.capital_cost_total.useful_life[oil_capital_id]
        depreciation_factor = self.capital_cost_total.depreciation_factor[oil_capital_id]
        is_ic_applied = np.array(self.capital_cost_total.is_ic_applied)[oil_capital_id]

        return CapitalCost(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost=cost,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            is_sunkcost=is_sunkcost.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
            pis_year=pis_year,
            salvage_value=salvage_value,
            useful_life=useful_life,
            depreciation_factor=depreciation_factor,
            is_ic_applied=is_ic_applied.tolist(),
        )



def _get_gas_capital(self) -> CapitalCost:
    """
    Determines total gas CapitalCost from the number of gas CapitalCost instances in
    attribute self.capital_cost_total.

    Returns
    -------
    CapitalCost
        An instance of CapitalCost that only includes FluidType.GAS as the associated
        cost_allocation that has been combined altogether following the rules prescribed
        in the dunder method __add__() of CapitalCost class.

    Notes
    -----
    The core operations are as follows:
    (1) Check the attribute cost_allocation in attribute self.capital_cost_total,
    (2) If GAS is not available as an instance in attribute self.capital_cost_total,
        then establish a new instance of GAS CapitalCost with the following attribute set
        to zero: cost.
    (3) Identify index location where cost_allocation is FluidType.GAS in attribute
        self.capital_cost_total,
    (4) Create a new instance of CapitalCost with only FluidType.GAS as its cost_allocation.
    """

    if FluidType.GAS not in self.capital_cost_total.cost_allocation:
        return CapitalCost(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=np.array([self.start_date.year]),
            cost=np.array([0]),
            cost_allocation=[FluidType.GAS],
        )

    else:
        gas_capital_id = np.argwhere(
            np.array(self.capital_cost_total.cost_allocation) == FluidType.GAS
        ).ravel()

        start_year = self.capital_cost_total.start_year
        end_year = self.capital_cost_total.end_year
        expense_year = self.capital_cost_total.expense_year[gas_capital_id]
        cost = self.capital_cost_total.cost[gas_capital_id]
        cost_allocation = np.array(self.capital_cost_total.cost_allocation)[gas_capital_id]
        description = np.array(self.capital_cost_total.description)[gas_capital_id]
        tax_portion = self.capital_cost_total.tax_portion[gas_capital_id]
        tax_discount = self.capital_cost_total.tax_discount[gas_capital_id]
        pis_year = self.capital_cost_total.pis_year[gas_capital_id]
        salvage_value = self.capital_cost_total.salvage_value[gas_capital_id]
        useful_life = self.capital_cost_total.useful_life[gas_capital_id]
        depreciation_factor = self.capital_cost_total.depreciation_factor[gas_capital_id]
        is_ic_applied = np.array(self.capital_cost_total.is_ic_applied)[gas_capital_id]

        return CapitalCost(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost=cost,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
            pis_year=pis_year,
            salvage_value=salvage_value,
            useful_life=useful_life,
            depreciation_factor=depreciation_factor,
            is_ic_applied=is_ic_applied.tolist(),
        )
        
        
        
def _get_oil_intangible(self) -> Intangible:
    """
    Determines total oil Intangible from the number of oil Intangible instances in
    attribute self.intangible_cost_total.

    Returns
    -------
    Intangible
        An instance of Intangible that only includes FluidType.OIL as the associated
        cost_allocation that has been combined altogether following the rules prescribed
        in the dunder method __add__() of Intangible class.

    Notes
    -----
    The core operations are as follows:
    (1) Check the attribute cost_allocation in attribute self.intangible_cost_total,
    (2) If OIL is not available as an instance in attribute self.intangible_cost_total,
        then establish a new instance of OIL Intangible with the following attribute set
        to zero: cost.
    (3) Identify index location where cost_allocation is FluidType.OIL in attribute
        self.intangible_cost_total,
    (4) Create a new instance of Intangible with only FluidType.OIL as its cost_allocation.
    """

    if FluidType.OIL not in self.intangible_cost_total.cost_allocation:
        return Intangible(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=np.array([self.start_date.year]),
            cost=np.array([0]),
            cost_allocation=[FluidType.OIL],
        )

    else:
        oil_intangible_id = np.argwhere(
            np.array(self.intangible_cost_total.cost_allocation) == FluidType.OIL
        ).ravel()

        start_year = self.intangible_cost_total.start_year
        end_year = self.intangible_cost_total.end_year
        expense_year = self.intangible_cost_total.expense_year[oil_intangible_id]
        cost = self.intangible_cost_total.cost[oil_intangible_id]
        cost_allocation = np.array(
            self.intangible_cost_total.cost_allocation
        )[oil_intangible_id]
        description = np.array(self.intangible_cost_total.description)[oil_intangible_id]
        tax_portion = self.intangible_cost_total.tax_portion[oil_intangible_id]
        tax_discount = self.intangible_cost_total.tax_discount[oil_intangible_id]

        return Intangible(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost=cost,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
        )
            
            
            
def _get_gas_intangible(self) -> Intangible:
    """
    Determines total gas Intangible from the number of gas Intangible instances in
    attribute self.intangible_cost_total.

    Returns
    -------
    Intangible
        An instance of Intangible that only includes FluidType.GAS as the associated
        cost_allocation that has been combined altogether following the rules prescribed
        in the dunder method __add__() of Intangible class.

    Notes
    -----
    The core operations are as follows:
    (1) Check the attribute cost_allocation in attribute self.intangible_cost_total,
    (2) If GAS is not available as an instance in attribute self.intangible_cost_total,
        then establish a new instance of GAS Intangible with the following attribute set
        to zero: cost.
    (3) Identify index location where cost_allocation is FluidType.GAS in attribute
        self.intangible_cost_total,
    (4) Create a new instance of Intangible with only FluidType.GAS as its cost_allocation.
    """

    if FluidType.GAS not in self.intangible_cost_total.cost_allocation:
        return Intangible(
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            expense_year=np.array([self.start_date.year]),
            cost=np.array([0]),
            cost_allocation=[FluidType.GAS],
        )

    else:
        gas_intangible_id = np.argwhere(
            np.array(self.intangible_cost_total.cost_allocation) == FluidType.GAS
        ).ravel()

        start_year = self.intangible_cost_total.start_year
        end_year = self.intangible_cost_total.end_year
        expense_year = self.intangible_cost_total.expense_year[gas_intangible_id]
        cost = self.intangible_cost_total.cost[gas_intangible_id]
        cost_allocation = np.array(self.intangible_cost_total.cost_allocation)[gas_intangible_id]
        description = np.array(self.intangible_cost_total.description)[gas_intangible_id]
        tax_portion = self.intangible_cost_total.tax_portion[gas_intangible_id]
        tax_discount = self.intangible_cost_total.tax_discount[gas_intangible_id]

        return Intangible(
            start_year=start_year,
            end_year=end_year,
            expense_year=expense_year,
            cost=cost,
            cost_allocation=cost_allocation.tolist(),
            description=description.tolist(),
            tax_portion=tax_portion,
            tax_discount=tax_discount,
        )
        
        