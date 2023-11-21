import numpy as np

from pyscnomics.econ.selection import DiscountingMode
from pyscnomics.econ.indicator import (npv_nominal_terms,
                                       npv_real_terms,
                                       npv_skk_nominal_terms,
                                       npv_skk_real_terms,
                                       npv_point_forward,
                                       pot_psc)

# Cashflow
cashflow_years = np.arange(2018, 2029)
cashflow = np.array([-100000, 0, 0, 0, 0, 0, 0, 50000, 50000, 50000, 50000])
discount_rate = 0.1
reference_year = 2023
inflation_rate = 0.03


def test_npv_nominal_terms():
    base = 44084.7930158769
    calc = npv_nominal_terms(cashflow=cashflow,
                             cashflow_years=cashflow_years,
                             discount_rate=discount_rate,
                             reference_year=reference_year,
                             discounting_mode=DiscountingMode.END_YEAR)
    np.testing.assert_allclose(base, calc)


def test_npv_real_terms():
    base = 28157.385585876900
    calc = npv_real_terms(cashflow=cashflow,
                          cashflow_years=cashflow_years,
                          discount_rate=discount_rate,
                          reference_year=reference_year,
                          inflation_rate=inflation_rate,
                          discounting_mode=DiscountingMode.END_YEAR)

    np.testing.assert_allclose(base, calc)


def test_npv_skk_nominal_terms():
    base = -10534.679687877200
    calc = npv_skk_nominal_terms(cashflow=cashflow,
                                 cashflow_years=cashflow_years,
                                 discount_rate=discount_rate,
                                 discounting_mode=DiscountingMode.END_YEAR)

    np.testing.assert_allclose(base, calc)


def test_npv_skk_real_terms():
    base = -16966.206984123100
    calc = npv_skk_real_terms(cashflow=cashflow,
                              cashflow_years=cashflow_years,
                              discount_rate=discount_rate,
                              reference_year=reference_year,
                              discounting_mode=DiscountingMode.END_YEAR)

    np.testing.assert_allclose(base, calc)


def test_npv_point_forward():
    base = 144084.793015877000
    calc = npv_point_forward(cashflow=cashflow,
                             cashflow_years=cashflow_years,
                             discount_rate=discount_rate,
                             reference_year=reference_year,
                             discounting_mode=DiscountingMode.END_YEAR)

    np.testing.assert_allclose(base, calc)


def test_pot():
    cashflow_psc = np.array(
        [-0.256683836, -0.673626192, -6.517022729, -0.386445703, -0.666439491, -8.706499027, -0.253577600, -0.128721715,
         -0.084977677, -0.572136045, -5.876060455, -1.430334000, -5.554723774, -36.792930707, 20.719872793,
         21.207345549, 21.069649586, 8.199755855, 5.989491246, 7.595050463, 7.223124711, 6.932400441, 3.556406469,
         6.819592533, 6.034993688, 5.108545699, 4.154169710, 2.793870918, 1.867523855, 1.224588341, 0.000000000,
         0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000])
    cashflow_years_pot = np.arange(2010, 2046)
    reference_year_pot = 2022

    base = 5.59798256
    calc = pot_psc(cashflow=cashflow_psc,
                   cashflow_years=cashflow_years_pot,
                   reference_year=reference_year_pot)

    np.testing.assert_allclose(base, calc)
