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


print('npv_nominal_terms :', npv_nominal_terms)
print('npv_real_terms :', npv_real_terms)
print('npv_skk_nominal_terms :', npv_skk_nominal_terms)
print('npv_skk_real_terms :', npv_skk_real_terms)
print('npv_point_forward :', npv_point_forward)

# POT
cashflow_psc = np.array(
    [-0.256683836, -0.673626192, -6.517022729, -0.386445703, -0.666439491, -8.706499027, -0.253577600, -0.128721715,
     -0.084977677, -0.572136045, -5.876060455, -1.430334000, -5.554723774, -36.792930707, 20.719872793, 21.207345549,
     21.069649586, 8.199755855, 5.989491246, 7.595050463, 7.223124711, 6.932400441, 3.556406469, 6.819592533,
     6.034993688, 5.108545699, 4.154169710, 2.793870918, 1.867523855, 1.224588341, 0.000000000, 0.000000000,
     0.000000000, 0.000000000, 0.000000000, 0.000000000])
cashflow_years = np.arange(2010, 2046)

pot = indicator.pot_psc(cashflow=cashflow_psc,
                        cashflow_years=cashflow_years,
                        reference_year=2022)

print('pot', pot)
