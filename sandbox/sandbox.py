import numpy as np
from pyscnomics.econ import indicator
from pyscnomics.econ.selection import DiscountingMode

# Cashflow
cashflow_years = np.arange(2018, 2029)
cashflow = np.array([-100000, 0, 0, 0, 0, 0, 0, 50000, 50000, 50000, 50000])
discount_rate = 0.1
reference_year = 2023

# Nominal Terms
npv_nominal_terms = indicator.npv_nominal_terms(cashflow=cashflow,
                                                cashflow_years=cashflow_years,
                                                discount_rate=discount_rate,
                                                reference_year=reference_year,
                                                discounting_mode=DiscountingMode.END_YEAR)

# Real Terms
inflation_rate = 0.03
npv_real_terms = indicator.npv_real_terms(cashflow=cashflow,
                                          cashflow_years=cashflow_years,
                                          discount_rate=discount_rate,
                                          reference_year=reference_year,
                                          inflation_rate=inflation_rate,
                                          discounting_mode=DiscountingMode.END_YEAR)

# SKK Nominal Terms
npv_skk_nominal_terms = indicator.npv_skk_nominal_terms(cashflow=cashflow,
                                                        cashflow_years=cashflow_years,
                                                        discount_rate=discount_rate,
                                                        discounting_mode=DiscountingMode.END_YEAR)

# SKK Real Terms
npv_skk_real_terms = indicator.npv_skk_real_terms(cashflow=cashflow,
                                                  cashflow_years=cashflow_years,
                                                  discount_rate=discount_rate,
                                                  reference_year=reference_year,
                                                  discounting_mode=DiscountingMode.END_YEAR)

# Point Forward
npv_point_forward = indicator.npv_point_forward(cashflow=cashflow,
                                                cashflow_years=cashflow_years,
                                                discount_rate=discount_rate,
                                                reference_year=reference_year,
                                                discounting_mode=DiscountingMode.END_YEAR)

# Sort the nested dictionary by the second key (assuming each inner dictionary has at least two keys)
sorted_nested_dict = dict(sorted(nested_dict.items(), key=lambda x: x[1]['Priority']))

# Print the sorted nested dictionary
print(sorted_nested_dict)