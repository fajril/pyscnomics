"""
A unit testing for depreciation module.
"""

import numpy as np
from pyscnomics.dataset.sample import load_data, load_testing
from pyscnomics.econ.selection import InflationAppliedTo

psc = load_data(dataset_type='small_oil', contract_type='gross_split')
tax_rate = np.array([0.11, 0.11, 0.11, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
inflation_rate = np.array([0, 0, 0.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
psc.run(tax_rate=0.22,
        vat_rate=tax_rate,
        inflation_rate=inflation_rate,
        inflation_rate_applied_to=InflationAppliedTo.CAPEX)



def test_tangible():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_tangible'))
    calc = psc._oil_tangible.expenditures()
    np.testing.assert_allclose(base, calc)


def test_intangible():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_intangible'))
    calc = psc._oil_intangible.expenditures()
    np.testing.assert_allclose(base, calc)


def test_opex():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_opex'))
    calc = psc._oil_opex.expenditures()
    np.testing.assert_allclose(base, calc)


def test_lifting():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_lifting'))
    calc = psc._oil_lifting.lifting_rate
    np.testing.assert_allclose(base, calc)


def test_price():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_price'))
    calc = psc._oil_lifting.price
    np.testing.assert_allclose(base, calc)


def test_revenue():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_revenue'))
    calc = psc._oil_revenue
    np.testing.assert_allclose(base, calc)


def test_depreciation():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_depreciation'))
    calc = np.array(psc._oil_depreciation)
    np.testing.assert_allclose(base, calc)


def test_non_capital():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_non_capital'))
    calc = np.array(psc._oil_non_capital)
    np.testing.assert_allclose(base, calc)


def test_cumulative_prod():
    base = np.asarray(load_testing(dataset_type='small_oil', key='cumulative_prod'))
    calc = np.array(psc._cumulative_prod)
    np.testing.assert_allclose(base, calc)


def test_prog_split():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_prog_split'))
    calc = np.array(psc._oil_prog_split)
    np.testing.assert_allclose(base, calc)


def test_oil_ctr_split():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_ctr_split'))
    calc = np.array(psc._oil_ctr_split)
    np.testing.assert_allclose(base, calc)


def test_oil_ctr_share():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_ctr_share'))
    calc = np.array(psc._oil_ctr_share_before_transfer)
    np.testing.assert_allclose(base, calc)


def test_oil_gov_share():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_gov_share'))
    calc = np.array(psc._oil_gov_share)
    np.testing.assert_allclose(base, calc)


def test_oil_total_expense():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_total_expenses'))
    calc = np.array(psc._oil_total_expenses)
    np.testing.assert_allclose(base, calc)


def test_oil_cost_tobe_deducted():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_cost_tobe_deducted'))
    calc = np.array(psc._oil_cost_tobe_deducted)
    np.testing.assert_allclose(base, calc)


def test_oil_carry_forward_deductible_cost():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_carward_deduct_cost'))
    calc = np.array(psc._oil_carward_deduct_cost)
    np.testing.assert_allclose(base, calc)


def test_oil_deductible_cost():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_deductible_cost'))
    calc = np.array(psc._oil_deductible_cost)
    np.testing.assert_allclose(base, calc)


def test_oil_net_operating_profit():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_net_operating_profit'))
    calc = np.array(psc._oil_net_operating_profit)
    np.testing.assert_allclose(base, calc)


def test_oil_dmo_volume():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_dmo_volume'))
    calc = np.array(psc._oil_dmo_volume)
    np.testing.assert_allclose(base, calc)


def test_oil_dmo_fee():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_dmo_fee'))
    calc = np.array(psc._oil_dmo_fee)
    np.testing.assert_allclose(base, calc)


def test_oil_ddmo():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_ddmo'))
    calc = np.array(psc._oil_ddmo)
    np.testing.assert_allclose(base, calc)


def test_oil_taxable_income():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_taxable_income'))
    calc = np.array(psc._oil_taxable_income)
    np.testing.assert_allclose(base, calc)


def test_oil_tax():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_tax'))
    calc = np.array(psc._oil_tax)
    np.testing.assert_allclose(base, calc)


def test_oil_ctr_net_share():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_ctr_net_share'))
    calc = np.array(psc._oil_ctr_net_share)
    np.testing.assert_allclose(base, calc)


def test_oil_ctr_cashflow():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_ctr_cashflow'))
    calc = np.array(psc._oil_ctr_cashflow)
    np.testing.assert_allclose(base, calc)


def test_oil_gov_take():
    base = np.asarray(load_testing(dataset_type='small_oil', key='oil_gov_take'))
    calc = np.array(psc._oil_government_take)
    np.testing.assert_allclose(base, calc)