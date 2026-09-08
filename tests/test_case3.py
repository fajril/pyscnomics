"""
A unit testing for depreciation module.
"""

import numpy as np

from pyscnomics.dataset.sample import load_data, load_testing
from pyscnomics.econ.selection import FluidType, GrossSplitRegime


# Defining the contract object
contract = load_data(dataset_type="case3", contract_type="gross_split")

# Defining the contract arguments
tax_rate = 0.22
vat_rate = np.array(
    [
        0.11,
        0.11,
        0.11,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
    ]
)
inflation_rate = 0.0
future_rate = 0.0
inflation_rate_applied_to = None

# Defining the contract arguments
# Running the contract with contract arguments
contract.run(
    effective_tax_rate=tax_rate,
    vat_rate=vat_rate,
    inflation_rate=inflation_rate,
    inflation_rate_applied_to=inflation_rate_applied_to,
    regime=GrossSplitRegime.PERMEN_ESDM_12_2020,
)
oil_columns = contract.get_oil_table_columns()


def test_tangible():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_tangible"))
    calc = oil_columns["capital_postonstream"]
    np.testing.assert_allclose(base, calc)


def test_intangible():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_intangible"))
    calc = oil_columns["intangible_postonstream"]
    np.testing.assert_allclose(base, calc)


def test_opex():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_opex"))
    calc = oil_columns["opex_postonstream"]
    np.testing.assert_allclose(base, calc)


def test_lifting():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_lifting"))
    calc = oil_columns["lifting"]
    np.testing.assert_allclose(base, calc)


def test_price():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_price"))
    calc = oil_columns["price"]
    np.testing.assert_allclose(base, calc)


def test_revenue():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_revenue"))
    calc = oil_columns["revenue"]
    np.testing.assert_allclose(base, calc)


def test_depreciation():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_depreciation"))
    calc = np.array(
        oil_columns["depreciations_sunk_cost"]
        + oil_columns["depreciations_preonstream"]
        + oil_columns["depreciations_postonstream"]
    )
    np.testing.assert_allclose(base, calc)


def test_non_capital():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_non_capital"))
    calc = np.array(oil_columns["expenses_non_capital"])
    np.testing.assert_allclose(base, calc)


def test_cumulative_prod():
    base = np.asarray(load_testing(dataset_type="case3", key="cumulative_prod"))
    oil_lifting = contract.get_lifting_by_fluid(FluidType.OIL)
    gas_lifting = contract.get_lifting_by_fluid(FluidType.GAS)
    calc = np.cumsum(
        oil_lifting.get_prod_rate_total_arr()
        + gas_lifting.get_prod_rate_total_arr() * 1_000.0 / 5.615
    )
    np.testing.assert_allclose(base, calc)


def test_prog_split():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_prog_split"))
    calc = np.array(oil_columns["progressive_split"])
    np.testing.assert_allclose(base, calc)


def test_oil_ctr_split():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_ctr_split"))
    calc = np.array(oil_columns["contractor_split"])
    np.testing.assert_allclose(base, calc)


def test_oil_ctr_share():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_ctr_share"))
    calc = np.array(oil_columns["contractor_share"])
    np.testing.assert_allclose(base, calc)


def test_oil_gov_share():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_gov_share"))
    calc = np.array(oil_columns["government_share"])
    np.testing.assert_allclose(base, calc)


def test_oil_total_expense():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_total_expenses"))
    calc = np.array(oil_columns["expenses_total"])
    np.testing.assert_allclose(base, calc)


def test_oil_cost_tobe_deducted():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_cost_tobe_deducted"))
    calc = np.array(oil_columns["cost_to_be_deducted"])
    np.testing.assert_allclose(base, calc)


def test_oil_carry_forward_deductible_cost():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_carward_deduct_cost"))
    calc = np.array(oil_columns["carry_forward_cost"])
    np.testing.assert_allclose(base, calc)


def test_oil_deductible_cost():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_deductible_cost"))
    calc = np.array(oil_columns["deductible_cost"])
    np.testing.assert_allclose(base, calc)


def test_oil_net_operating_profit():
    base = np.asarray(
        load_testing(dataset_type="case3", key="oil_net_operating_profit")
    )
    calc = np.array(oil_columns["ctr_net_operating_profit"])
    np.testing.assert_allclose(base, calc)


def test_oil_dmo_volume():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_dmo_volume"))
    calc = np.array(oil_columns["dmo_volume"])
    np.testing.assert_allclose(base, calc)


def test_oil_dmo_fee():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_dmo_fee"))
    calc = np.array(oil_columns["dmo_fee"])
    np.testing.assert_allclose(base, calc)


def test_oil_ddmo():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_ddmo"))
    calc = np.array(oil_columns["ddmo"])
    np.testing.assert_allclose(base, calc)


def test_oil_taxable_income():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_taxable_income"))
    calc = np.array(oil_columns["taxable_income"])
    np.testing.assert_allclose(base, calc)


def test_oil_tax():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_tax"))
    calc = np.array(oil_columns["tax"])
    np.testing.assert_allclose(base, calc)


def test_oil_ctr_net_share():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_ctr_net_share"))
    calc = np.array(oil_columns["net_ctr_share"])
    np.testing.assert_allclose(base, calc)


def test_oil_ctr_cashflow():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_ctr_cashflow"))
    calc = np.array(oil_columns["ctr_cashflow"])
    np.testing.assert_allclose(base, calc)


def test_oil_gov_take():
    base = np.asarray(load_testing(dataset_type="case3", key="oil_gov_take"))
    calc = np.array(oil_columns["government_take"])
    np.testing.assert_allclose(base, calc)


def test_table_column_providers_expose_each_stream():
    contract = load_data(dataset_type="case3", contract_type="gross_split")
    contract.run(
        effective_tax_rate=tax_rate,
        vat_rate=vat_rate,
        inflation_rate=inflation_rate,
        inflation_rate_applied_to=inflation_rate_applied_to,
        regime=GrossSplitRegime.PERMEN_ESDM_12_2020,
    )

    for columns in (
        contract.get_oil_table_columns(),
        contract.get_gas_table_columns(),
        contract.get_consolidated_table_columns(),
    ):
        assert tuple(columns)[:4] == ("years", "lifting", "price", "revenue")
        assert tuple(columns)[-2:] == ("ctr_cashflow", "cum_cashflow")
