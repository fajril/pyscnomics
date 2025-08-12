import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", 2000)
pd.set_option("display.max_colwidth", None)



def _cashflow_get_attrs_by_type(self, cashflow_type: CashflowType):

    if cashflow_type == CashflowType.OIL:
        return [
            self.project_years,
            self._oil_lifting.get_lifting_rate_arr(),
            self._oil_wap_price,
            self._oil_revenue,
            self._oil_sunk_cost,
            self._oil_capital_expenditures_pre_tax,
            self._oil_intangible_expenditures_pre_tax,
            self._oil_opex_expenditures_pre_tax,
            self._oil_asr_expenditures_pre_tax,
            self._oil_lbt_expenditures_pre_tax,
            self._oil_cost_of_sales_expenditures_pre_tax,
        ]

    elif cashflow_type == CashflowType.GAS:
        return [
            self.project_years,
            self._gas_lifting.get_lifting_rate_arr(),
            self._gas_wap_price,
            self._gas_revenue,
            self._gas_sunk_cost,
            self._gas_capital_expenditures_pre_tax,
            self._gas_intangible_expenditures_pre_tax,
            self._gas_opex_expenditures_pre_tax,
            self._gas_asr_expenditures_pre_tax,
            self._gas_lbt_expenditures_pre_tax,
            self._gas_cost_of_sales_expenditures_pre_tax,
        ]

    elif cashflow_type == CashflowType.CONSOLIDATED:
        return [
            self.project_years,
            self._oil_lifting.get_lifting_rate_arr() + self._gas_lifting.get_lifting_rate_arr(),
            self._oil_wap_price + self._gas_wap_price,
            self._oil_revenue + self._gas_revenue,
            self._oil_sunk_cost + self._gas_sunk_cost,
            self._oil_capital_expenditures_pre_tax + self._gas_capital_expenditures_pre_tax,
            self._oil_intangible_expenditures_pre_tax + self._gas_intangible_expenditures_pre_tax,
            self._oil_opex_expenditures_pre_tax + self._gas_opex_expenditures_pre_tax,
            self._oil_asr_expenditures_pre_tax + self._gas_asr_expenditures_pre_tax,
            self._oil_lbt_expenditures_pre_tax + self._gas_lbt_expenditures_pre_tax,
            self._oil_cost_of_sales_expenditures_pre_tax + self._gas_cost_of_sales_expenditures_pre_tax,
        ]

    else:
        raise BaseProjectException(
            f"Unsupported cashflow_type: {cashflow_type}"
        )

def _cashflow_get_results(self, cashflow_type: CashflowType):

    # if not self._run_completed:
    #     raise Warning(
    #         f"Please run cashflow_calculate() first before requesting results"
    #     )
        # raise RuntimeError(
        #     f"Please run cashflow_calculate() first before requesting results"
        # )

    # Prepare the initial dataframe with default values of zeros
    keys = [
        "Project Years",
        "Lifting",
        "Price",
        "Revenue",
        "Sunk Cost",
        "Capital Cost Pre Tax",
        "Intangible Pre Tax",
        "OPEX Pre Tax",
        "ASR Pre Tax",
        "LBT Pre Tax",
        "Cost of Sales Pre Tax",
    ]

    results = pd.DataFrame(np.nan, index=range(self.project_duration), columns=keys)

    # Modify the dataframe with calculated values
    modified_attrs = self._cashflow_get_attrs_by_type(cashflow_type=cashflow_type)

    for key, val in zip(keys, modified_attrs):
        results[key] = val

    return results

def cashflow_calculate(
    self,
    sulfur_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
    electricity_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_OIL_REVENUE,
    co2_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
    tax_rate: np.ndarray | float = 0.0,
    year_inflation: np.ndarray = None,
    inflation_rate: np.ndarray | float = 0.0,
    inflation_rate_applied_to: InflationAppliedTo = None,
):

    kwargs = {
        "sulfur_revenue": sulfur_revenue,
        "electricity_revenue": electricity_revenue,
        "co2_revenue": co2_revenue,
        "tax_rate": tax_rate,
        "year_inflation": year_inflation,
        "inflation_rate": inflation_rate,
        "inflation_rate_applied_to": inflation_rate_applied_to,
    }

    # Run calculatin and remember params
    self.run(**kwargs)
    self._last_run_params = kwargs
    self._run_completed = True

def get_oil_cashflow_table(self):
    return self._cashflow_get_results(cashflow_type=CashflowType.OIL)

def get_gas_cashflow_table(self):
    return self._cashflow_get_results(cashflow_type=CashflowType.GAS)

def get_consolidated_cashflow_table(self):
    return self._cashflow_get_results(cashflow_type=CashflowType.CONSOLIDATED)
