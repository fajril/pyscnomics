import numpy as np

from pyscnomics.dataset.sample import load_data, load_testing
from pyscnomics.econ.selection import FTPTaxRegime

psc = load_data(dataset_type='case1', contract_type='cost_recovery')

# Editing the CostRecovery attribute as the corresponding case 1
# FTP
psc.oil_ftp_is_available = True
psc.oil_ftp_is_shared = True
psc.oil_ftp_portion = 0.2
psc.gas_ftp_is_available = True
psc.gas_ftp_is_shared = True
psc.gas_ftp_portion = 0.2

# Split Pre Tax
psc.oil_ctr_pretax_share = 0.34722220
psc.gas_ctr_pretax_share = 0.5208330

# DMO
psc.oil_dmo_volume_portion = 0.25
psc.oil_dmo_fee_portion = 0.25
psc.oil_dmo_holiday_duration = 60
psc.gas_dmo_volume_portion = 0.25
psc.gas_dmo_fee_portion = 1
psc.gas_dmo_holiday_duration = 60

tax_rate = 0.424

psc.run(tax_rate=tax_rate, ftp_tax_regime=FTPTaxRegime.PRE_PDJP_20_2017)


def test_revenue():
    """
    Test module to test value of revenue calculated by pyscnomics.
    """

    oil_revenue = psc._oil_revenue
    oil_revenue_base = np.asarray(load_testing(dataset_type='case1', key='oil_revenue'))

    gas_revenue = psc._gas_revenue
    gas_revenue_base = np.asarray(load_testing(dataset_type='case1', key='gas_revenue'))

    np.testing.assert_allclose(oil_revenue, oil_revenue_base, atol=1e-6)
    np.testing.assert_allclose(gas_revenue, gas_revenue_base, atol=1e-6)


def test_depreciation():
    """
    Test module for testing the Depreciation.
    """

    oil_depreciation = psc._oil_depreciation
    oil_depreciation_base = load_testing(dataset_type='case1', key='oil_depreciation')
    gas_depreciation = psc._gas_depreciation
    gas_depreciation_base = load_testing(dataset_type='case1', key='gas_depreciation')

    np.testing.assert_allclose(oil_depreciation, oil_depreciation_base, atol=1e-6)
    np.testing.assert_allclose(gas_depreciation, gas_depreciation_base, atol=1e-6)


def test_non_capital():
    oil_non_capital = psc._oil_non_capital
    oil_non_capital_base = load_testing(dataset_type='case1', key='oil_non_capital')
    gas_non_capital = psc._gas_non_capital
    gas_non_capital_base = load_testing(dataset_type='case1', key='gas_non_capital')

    np.testing.assert_allclose(oil_non_capital, oil_non_capital_base, atol=1e-6)
    np.testing.assert_allclose(gas_non_capital, gas_non_capital_base, atol=1e-6)


def test_ftp():
    oil_ftp = psc._oil_ftp
    oil_ftp_base = load_testing(dataset_type='case1', key='oil_ftp')
    gas_ftp = psc._gas_ftp
    gas_ftp_base = load_testing(dataset_type='case1', key='gas_ftp')

    np.testing.assert_allclose(oil_ftp, oil_ftp_base, atol=1e-6)
    np.testing.assert_allclose(gas_ftp, gas_ftp_base, atol=1e-6)


def test_ftp_ctr():
    oil_ftp_ctr = psc._oil_ftp_ctr
    oil_ftp_ctr_base = load_testing(dataset_type='case1', key='oil_ftp_ctr')
    gas_ftp_ctr = psc._gas_ftp_ctr
    gas_ftp_ctr_base = load_testing(dataset_type='case1', key='gas_ftp_ctr')

    np.testing.assert_allclose(oil_ftp_ctr, oil_ftp_ctr_base, atol=1e-6)
    np.testing.assert_allclose(gas_ftp_ctr, gas_ftp_ctr_base, atol=1e-6)


def test_ftp_gov():
    oil_ftp_gov = psc._oil_ftp_gov
    oil_ftp_gov_base = load_testing(dataset_type='case1', key='oil_ftp_gov')
    gas_ftp_gov = psc._gas_ftp_gov
    gas_ftp_gov_base = load_testing(dataset_type='case1', key='gas_ftp_gov')

    np.testing.assert_allclose(oil_ftp_gov, oil_ftp_gov_base, atol=1e-6)
    np.testing.assert_allclose(gas_ftp_gov, gas_ftp_gov_base, atol=1e-6)


def test_ic():
    """
    Test module for testing the Investment Credit (IC).
    """

    oil_ic = psc._oil_ic
    oil_ic_unrecovered = psc._oil_ic_unrecovered
    oil_ic_paid = psc._oil_ic_paid
    gas_ic = psc._gas_ic
    gas_ic_unrecovered = psc._gas_ic_unrecovered
    gas_ic_paid = psc._gas_ic_paid

    oil_ic_base = load_testing(dataset_type='case1', key='oil_ic')
    oil_ic_unrecovered_base = load_testing(dataset_type='case1', key='oil_ic_unrec')
    oil_ic_paid_base = load_testing(dataset_type='case1', key='oil_ic_paid')
    gas_ic_base = load_testing(dataset_type='case1', key='gas_ic')
    gas_ic_unrecovered_base = load_testing(dataset_type='case1', key='gas_ic_unrec')
    gas_ic_paid_base = load_testing(dataset_type='case1', key='gas_ic_paid')

    np.testing.assert_allclose(oil_ic, oil_ic_base, atol=1e-6)
    np.testing.assert_allclose(oil_ic_unrecovered, oil_ic_unrecovered_base, atol=1e-6)
    np.testing.assert_allclose(oil_ic_paid, oil_ic_paid_base, atol=1e-6)
    np.testing.assert_allclose(gas_ic, gas_ic_base, atol=1e-6)
    np.testing.assert_allclose(gas_ic_unrecovered, gas_ic_unrecovered_base, atol=1e-6)
    np.testing.assert_allclose(gas_ic_paid, gas_ic_paid_base, atol=1e-6)


def test_unrec_before_transfer():
    """
    Test module for testing the Unrecovered Cost Before Transfer.
    """

    oil_unrec_before_transfer = psc._oil_unrecovered_before_transfer
    oil_unrec_before_transfer_base = load_testing(dataset_type='case1', key='oil_unrec_cost')
    gas_unrec_before_transfer = psc._gas_unrecovered_before_transfer
    gas_unrec_before_transfer_base = load_testing(dataset_type='case1', key='gas_unrec_cost')

    np.testing.assert_allclose(oil_unrec_before_transfer, oil_unrec_before_transfer_base, atol=1e-6)
    np.testing.assert_allclose(gas_unrec_before_transfer, gas_unrec_before_transfer_base, atol=1e-6)


def test_cost_tobe_recovered():
    """
    Test module for testing the Cost To Be Recover.
    """

    oil_cost_to_be_recovered = psc._oil_cost_to_be_recovered
    gas_cost_to_be_recovered = psc._gas_cost_to_be_recovered

    oil_cost_to_be_recovered_base = load_testing(dataset_type='case1', key='oil_cost_to_be_recovered')
    gas_cost_to_be_recovered_base = load_testing(dataset_type='case1', key='gas_cost_to_be_recovered')

    np.testing.assert_allclose(oil_cost_to_be_recovered, oil_cost_to_be_recovered_base, atol=1e-6)
    np.testing.assert_allclose(gas_cost_to_be_recovered, gas_cost_to_be_recovered_base, atol=1e-6)


def test_cost_recovery():
    # Calculated result
    oil_cost_recovery = psc._oil_cost_recovery
    gas_cost_recovery = psc._gas_cost_recovery

    # Expected result
    oil_cost_recovery_base = load_testing(dataset_type='case1', key='oil_cost_recovery')
    gas_cost_recovery_base = load_testing(dataset_type='case1', key='gas_cost_recovery')

    # Execute testing
    np.testing.assert_allclose(oil_cost_recovery, oil_cost_recovery_base, atol=1e-6)
    np.testing.assert_allclose(gas_cost_recovery, gas_cost_recovery_base, atol=1e-6)


def test_ets_before_transfer():
    # Calculated result
    oil_ets_before_transfer = psc._oil_ets_before_transfer
    gas_ets_before_transfer = psc._gas_ets_before_transfer

    # Expected result
    oil_ets_before_transfer_base = load_testing(dataset_type='case1', key='oil_ets_before_transfer')
    gas_ets_before_transfer_base = load_testing(dataset_type='case1', key='gas_ets_before_transfer')

    # Execute testing
    np.testing.assert_allclose(oil_ets_before_transfer, oil_ets_before_transfer_base, atol=1e-6)
    np.testing.assert_allclose(gas_ets_before_transfer, gas_ets_before_transfer_base, atol=1e-6)


def test_transfer():
    # Calculated result
    transfer_to_gas = psc._transfer_to_gas
    transfer_to_oil = psc._transfer_to_oil

    # Expected result
    transfer_to_gas_base = load_testing(dataset_type='case1', key='oil_transfer_to_gas')
    transfer_to_oil_base = load_testing(dataset_type='case1', key='gas_transfer_to_oil')

    # Execute testing
    np.testing.assert_allclose(transfer_to_gas, transfer_to_gas_base, atol=1e-6)
    np.testing.assert_allclose(transfer_to_oil, transfer_to_oil_base, atol=1e-6)


def test_unrecovered_after_transfer():
    # Calculated result
    oil_unrecovered_after_transfer = psc._oil_unrecovered_after_transfer
    gas_unrecovered_after_transfer = psc._gas_unrecovered_after_transfer

    # Expected result
    oil_unrecovered_after_transfer_base = load_testing(dataset_type='case1', key='oil_unrec_after_transfer')
    gas_unrecovered_after_transfer_base = load_testing(dataset_type='case1', key='gas_unrec_after_transfer')

    # Execute testing
    np.testing.assert_allclose(oil_unrecovered_after_transfer, oil_unrecovered_after_transfer_base, atol=1e-6)
    np.testing.assert_allclose(gas_unrecovered_after_transfer, gas_unrecovered_after_transfer_base, atol=1e-6)


def test_ets_after_transfer():
    # Calculated result
    oil_ets_after_transfer = psc._oil_ets_after_transfer
    gas_ets_after_transfer = psc._gas_ets_after_transfer

    # Expected result
    oil_ets_after_transfer_base = load_testing(dataset_type='case1', key='oil_ets_after_transfer')
    gas_ets_after_transfer_base = load_testing(dataset_type='case1', key='gas_ets_after_transfer')

    # Execute testing
    np.testing.assert_allclose(oil_ets_after_transfer, oil_ets_after_transfer_base, atol=1e-6)
    np.testing.assert_allclose(gas_ets_after_transfer, gas_ets_after_transfer_base, atol=1e-6)


def test_equity_share():
    # Calculated result
    oil_contractor_share = psc._oil_contractor_share
    oil_government_share = psc._oil_government_share

    gas_contractor_share = psc._gas_contractor_share
    gas_government_share = psc._gas_government_share

    # Expected result
    oil_contractor_share_base = load_testing(dataset_type='case1', key='oil_ctr_share')
    oil_government_share_base = load_testing(dataset_type='case1', key='oil_gov_share')
    gas_contractor_share_base = load_testing(dataset_type='case1', key='gas_ctr_share')
    gas_government_share_base = load_testing(dataset_type='case1', key='gas_gov_share')

    # Execute testing
    np.testing.assert_allclose(oil_contractor_share, oil_contractor_share_base, atol=1e-6)
    np.testing.assert_allclose(oil_government_share, oil_government_share_base, atol=1e-6)
    np.testing.assert_allclose(gas_contractor_share, gas_contractor_share_base, atol=1e-6)
    np.testing.assert_allclose(gas_government_share, gas_government_share_base, atol=1e-6)


def test_dmo():
    # Calculated result
    oil_dmo_volume = psc._oil_dmo_volume
    oil_dmo_fee = psc._oil_dmo_fee
    oil_ddmo = psc._oil_ddmo

    gas_dmo_volume = psc._gas_dmo_volume
    gas_dmo_fee = psc._gas_dmo_fee
    gas_ddmo = psc._gas_ddmo

    # Expected result
    oil_dmo_volume_base = load_testing(dataset_type='case1', key='oil_dmo_volume')
    oil_dmo_fee_base = load_testing(dataset_type='case1', key='oil_dmo_fee')
    oil_ddmo_base = load_testing(dataset_type='case1', key='oil_ddmo')

    gas_dmo_volume_base = load_testing(dataset_type='case1', key='gas_dmo_volume')
    gas_dmo_fee_base = load_testing(dataset_type='case1', key='gas_dmo_fee')
    gas_ddmo_base = load_testing(dataset_type='case1', key='gas_ddmo')

    # Execute testing
    np.testing.assert_allclose(oil_dmo_volume, oil_dmo_volume_base, atol=1e-6)
    np.testing.assert_allclose(oil_dmo_fee, oil_dmo_fee_base, atol=1e-6)
    np.testing.assert_allclose(oil_ddmo, oil_ddmo_base, atol=1e-6)

    np.testing.assert_allclose(gas_dmo_volume, gas_dmo_volume_base, atol=1e-6)
    np.testing.assert_allclose(gas_dmo_fee, gas_dmo_fee_base, atol=1e-6)
    np.testing.assert_allclose(gas_ddmo, gas_ddmo_base, atol=1e-6)


def test_taxable_income():
    # Calculated result
    oil_taxable_income = psc._oil_taxable_income
    gas_taxable_income = psc._gas_taxable_income

    # Expected result
    oil_taxable_income_base = load_testing(dataset_type='case1', key='oil_taxable_income')
    gas_taxable_income_base = load_testing(dataset_type='case1', key='gas_taxable_income')

    # Execute testing
    np.testing.assert_allclose(oil_taxable_income, oil_taxable_income_base, atol=1e-6)
    np.testing.assert_allclose(gas_taxable_income, gas_taxable_income_base, atol=1e-6)


def test_tax():
    # Calculated result
    oil_tax = psc._oil_tax_payment
    gas_tax = psc._gas_tax_payment

    # Expected result
    oil_tax_base = load_testing(dataset_type='case1', key='oil_tax')
    gas_tax_base = load_testing(dataset_type='case1', key='gas_tax')

    # Execute testing
    np.testing.assert_allclose(oil_tax, oil_tax_base, atol=1e-6)
    np.testing.assert_allclose(gas_tax, gas_tax_base, atol=1e-6)


def test_ctr_net_share():
    # Calculated result
    oil_ctr_share_after_tax = psc._oil_ctr_net_share
    gas_ctr_share_after_tax = psc._gas_ctr_net_share

    # Expected result
    oil_ctr_share_after_tax_base = load_testing(dataset_type='case1', key='oil_ctr_net_share')
    gas_ctr_share_after_tax_base = load_testing(dataset_type='case1', key='gas_ctr_net_share')

    # Execute testing
    np.testing.assert_allclose(oil_ctr_share_after_tax, oil_ctr_share_after_tax_base, atol=1e-6)
    np.testing.assert_allclose(gas_ctr_share_after_tax, gas_ctr_share_after_tax_base, atol=1e-6)


def test_ctr_take():
    # Calculated result
    oil_ctr_share_after_tax = psc._oil_ctr_net_share
    gas_ctr_share_after_tax = psc._gas_ctr_net_share

    # Expected result
    oil_ctr_share_after_tax_base = load_testing(dataset_type='case1', key='oil_ctr_net_share')
    gas_ctr_share_after_tax_base = load_testing(dataset_type='case1', key='gas_ctr_net_share')

    # Execute testing
    np.testing.assert_allclose(oil_ctr_share_after_tax, oil_ctr_share_after_tax_base, atol=1e-6)
    np.testing.assert_allclose(gas_ctr_share_after_tax, gas_ctr_share_after_tax_base, atol=1e-6)


def test_ctr_cashflow():
    # Calculated result
    oil_ctr_cashflow = psc._oil_cashflow
    gas_ctr_cashflow = psc._gas_cashflow

    # Expected result
    oil_ctr_cashflow_base = load_testing(dataset_type='case1', key='oil_ctr_cashflow')
    gas_ctr_cashflow_base = load_testing(dataset_type='case1', key='gas_ctr_cashflow')

    # Execute testing
    np.testing.assert_allclose(oil_ctr_cashflow, oil_ctr_cashflow_base, atol=1e-6)
    np.testing.assert_allclose(gas_ctr_cashflow, gas_ctr_cashflow_base, atol=1e-6)


def test_goi_take():
    # Calculated result
    oil_goi_take = psc._oil_government_take
    gas_goi_take = psc._gas_government_take

    # Expected result
    oil_goi_take_base = load_testing(dataset_type='case1', key='oil_goi_take')
    gas_goi_take_base = load_testing(dataset_type='case1', key='gas_goi_take')

    # Execute testing
    np.testing.assert_allclose(oil_goi_take, oil_goi_take_base, atol=1e-6)
    np.testing.assert_allclose(gas_goi_take, gas_goi_take_base, atol=1e-6)


def test_consistency():
    # Top side result
    gross_revenue_topside = psc._oil_revenue + psc._gas_revenue

    # Bottom side result
    gross_revenue_botside = psc._oil_contractor_take + psc._gas_contractor_take + psc._oil_government_take + psc._gas_government_take

    # Execute testing
    np.testing.assert_allclose(gross_revenue_topside, gross_revenue_botside, atol=1e-6)


def test_consistency_ets():
    # Calculated result Oil
    oil_revenue = (psc._oil_ftp +
                   psc._oil_ic +
                   psc._oil_cost_recovery +
                   psc._oil_ets_before_transfer)

    # Calculated result Gas
    gas_revenue = (psc._gas_ftp +
                   psc._gas_ic +
                   psc._gas_cost_recovery +
                   psc._gas_ets_before_transfer)

    # Expected result Oil
    oil_revenue_base = psc._oil_revenue
    gas_revenue_base = psc._gas_revenue

    np.testing.assert_allclose(oil_revenue, oil_revenue_base)
    np.testing.assert_allclose(gas_revenue, gas_revenue_base)


def test_consolidated_revenue():
    # Calculated result
    engine = psc._consolidated_revenue

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_revenue')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_depreciation():
    # Calculated result
    engine = psc._consolidated_depreciation

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_depreciation')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_non_capital():
    # Calculated result
    engine = psc._consolidated_non_capital

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_non_capital')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ftp():
    # Calculated result
    engine = psc._consolidated_ftp

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_ftp')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ftp_ctr():
    # Calculated result
    engine = psc._consolidated_ftp_ctr

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_ftp_ctr')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ftp_gov():
    # Calculated result
    engine = psc._consolidated_ftp_gov

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_ftp_gov')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ic():
    # Calculated result
    engine = psc._consolidated_ic

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_ic')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ic_unrec():
    # Calculated result
    engine = psc._consolidated_ic_unrecovered

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_ic_unrec')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ic_paid():
    # Calculated result
    engine = psc._consolidated_ic_paid

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_ic_paid')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_unrec_cost():
    # Calculated result
    engine = psc._consolidated_unrecovered_before_transfer

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_unrec_cost')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_cost_recovery_before_transfer():
    # Calculated result
    engine = psc._consolidated_cost_recovery_before_transfer

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_cost_recovery_before_transfer')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ets_before_transfer():
    # Calculated result
    engine = psc._consolidated_ets_before_transfer

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_ets_before_transfer')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_unrec_after_transfer():
    # Calculated result
    engine = psc._consolidated_unrecovered_after_transfer

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_unrec_after_transfer')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_cost_recovery_after_transfer():
    # Calculated result
    engine = psc._consolidated_cost_recovery_after_tf

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_cost_recovery_after_transfer')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ets_after_transfer():
    # Calculated result
    engine = psc._consolidated_ets_after_transfer

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_ets_after_transfer')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ctr_share():
    # Calculated result
    engine = psc._consolidated_contractor_share

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_ctr_share')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_gov_share():
    # Calculated result
    engine = psc._consolidated_government_share

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_gov_share')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_dmo_volume():
    # Calculated result
    engine = psc._consolidated_dmo_volume

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_dmo_volume')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_dmo_fee():
    # Calculated result
    engine = psc._consolidated_dmo_fee

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_dmo_fee')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ddmo():
    # Calculated result
    engine = psc._consolidated_ddmo

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_ddmo')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_taxable_income():
    # Calculated result
    engine = psc._consolidated_taxable_income

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_taxable_income')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_tax_due():
    # Calculated result
    engine = psc._consolidated_tax_due

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_tax_due')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_unpaid_tax_balance():
    # Calculated result
    engine = psc._consolidated_unpaid_tax_balance

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_unpaid_tax_balance')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_tax_payment():
    # Calculated result
    engine = psc._consolidated_tax_payment

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_tax_payment')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ctr_net_share():
    # Calculated result
    engine = psc._consolidated_ctr_net_share

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_ctr_net_share')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_gov_take():
    # Calculated result
    engine = psc._consolidated_government_take

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_gov_take')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_cashflow():
    # Calculated result
    engine = psc._consolidated_cashflow

    # Expected result
    base = load_testing(dataset_type='case1', key='cnsltd_cashflow')

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)



