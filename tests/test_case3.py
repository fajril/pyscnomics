"""
A unit testing for depreciation module.
"""

import numpy as np
from pyscnomics.dataset.sample import load_data, load_testing
from pyscnomics.econ.selection import InflationAppliedTo

# Defining the contract object
psc = load_data(dataset_type='case3', contract_type='gross_split')

# Defining the contract arguments
tax_rate = 0.22
sunk_cost_reference_year = 2022
vat_rate = np.array([0.11, 0.11, 0.11, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
inflation_rate = 0.0
future_rate = 0.0
inflation_rate_applied_to = None

# Defining the contract arguments
contract_arguments = {
    "effective_tax_rate": tax_rate,
    "sunk_cost_reference_year": sunk_cost_reference_year,
    "vat_rate": vat_rate,
    "inflation_rate": inflation_rate,
    "inflation_rate_applied_to": inflation_rate_applied_to,
}

# Running the contract with contract arguments
psc.run(**contract_arguments)


def test_tangible():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_tangible'))
    calc = psc._oil_capital_expenditures_post_tax
    np.testing.assert_allclose(base, calc)


def test_intangible():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_intangible'))
    calc = psc._oil_intangible_expenditures_post_tax
    np.testing.assert_allclose(base, calc)


def test_opex():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_opex'))
    calc = psc._oil_opex_expenditures_post_tax
    np.testing.assert_allclose(base, calc)


def test_lifting():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_lifting'))
    calc = psc._oil_lifting.get_lifting_rate_arr()
    np.testing.assert_allclose(base, calc)


def test_price():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_price'))
    calc = psc._oil_lifting.get_price_arr()
    np.testing.assert_allclose(base, calc)


def test_revenue():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_revenue'))
    calc = psc._oil_revenue
    np.testing.assert_allclose(base, calc)


def test_depreciation():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_depreciation'))
    calc = np.array(psc._oil_depreciation)
    np.testing.assert_allclose(base, calc)


def test_non_capital():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_non_capital'))
    calc = np.array(psc._oil_non_capital)
    np.testing.assert_allclose(base, calc)


def test_cumulative_prod():
    base = np.asarray(load_testing(dataset_type='case3', key='cumulative_prod'))
    calc = np.array(psc._cumulative_prod)
    np.testing.assert_allclose(base, calc)


def test_prog_split():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_prog_split'))
    calc = np.array(psc._oil_prog_split)
    np.testing.assert_allclose(base, calc)


def test_oil_ctr_split():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_ctr_split'))
    calc = np.array(psc._oil_ctr_split)
    np.testing.assert_allclose(base, calc)


def test_oil_ctr_share():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_ctr_share'))
    calc = np.array(psc._oil_ctr_share_before_transfer)
    np.testing.assert_allclose(base, calc)


def test_oil_gov_share():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_gov_share'))
    calc = np.array(psc._oil_gov_share)
    np.testing.assert_allclose(base, calc)


def test_oil_total_expense():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_total_expenses'))
    calc = np.array(psc._oil_total_expenses)
    np.testing.assert_allclose(base, calc)


def test_oil_cost_tobe_deducted():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_cost_tobe_deducted'))
    calc = np.array(psc._oil_cost_tobe_deducted)
    np.testing.assert_allclose(base, calc)


def test_oil_carry_forward_deductible_cost():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_carward_deduct_cost'))
    calc = np.array(psc._oil_carward_deduct_cost)
    np.testing.assert_allclose(base, calc)


def test_oil_deductible_cost():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_deductible_cost'))
    calc = np.array(psc._oil_deductible_cost)
    np.testing.assert_allclose(base, calc)


def test_oil_net_operating_profit():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_net_operating_profit'))
    calc = np.array(psc._oil_net_operating_profit)
    np.testing.assert_allclose(base, calc)


def test_oil_dmo_volume():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_dmo_volume'))
    calc = np.array(psc._oil_dmo_volume)
    np.testing.assert_allclose(base, calc)


def test_oil_dmo_fee():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_dmo_fee'))
    calc = np.array(psc._oil_dmo_fee)
    np.testing.assert_allclose(base, calc)


def test_oil_ddmo():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_ddmo'))
    calc = np.array(psc._oil_ddmo)
    np.testing.assert_allclose(base, calc)


def test_oil_taxable_income():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_taxable_income'))
    calc = np.array(psc._oil_taxable_income)
    np.testing.assert_allclose(base, calc)


def test_oil_tax():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_tax'))
    calc = np.array(psc._oil_tax)
    np.testing.assert_allclose(base, calc)


def test_oil_ctr_net_share():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_ctr_net_share'))
    calc = np.array(psc._oil_ctr_net_share)
    np.testing.assert_allclose(base, calc)


def test_oil_ctr_cashflow():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_ctr_cashflow'))
    calc = np.array(psc._oil_ctr_cashflow)
    np.testing.assert_allclose(base, calc)


def test_oil_gov_take():
    base = np.asarray(load_testing(dataset_type='case3', key='oil_gov_take'))
    calc = np.array(psc._oil_government_take)
    np.testing.assert_allclose(base, calc)