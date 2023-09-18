import numpy as np

from pyscnomics.dataset.sample_refactored import load_dataset, load_testing


def test_revenue():
    """
    Test module to test value of revenue calculated by pyscnomics.
    """

    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    revenue = psc_result['XX']
    revenue_base = load_testing(dataset_type='small_gas', key='Revenue')

    np.testing.assert_allclose(revenue, revenue_base)


def test_depreciation():
    """
    Test module for testing the Depreciation.
    """
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    gas_depreciation = psc_result['XX']
    gas_depreciation_base = load_testing(dataset_type='small_gas', key='Depreciation')

    np.testing.assert_allclose(gas_depreciation, gas_depreciation_base)


def test_non_capital():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    gas_non_capital = psc_result['XX']
    gas_non_capital_base = load_testing(dataset_type='small_gas', key='Non Capital')

    np.testing.assert_allclose(gas_non_capital, gas_non_capital_base)


def test_ftp():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    gas_ftp = psc_result['XX']
    gas_ftp_base = load_testing(dataset_type='small_gas', key='FTP')

    np.testing.assert_allclose(gas_ftp, gas_ftp_base)

def test_ftp_ctr():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    gas_ftp_ctr = psc_result['XX']
    gas_ftp_ctr_base = load_testing(dataset_type='small_gas', key='FTP CTR')

    np.testing.assert_allclose(gas_ftp_ctr, gas_ftp_ctr_base)

def test_ftp_gov():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    gas_ftp_gov = psc_result['XX']
    gas_ftp_gov_base = load_testing(dataset_type='small_gas', key='FTP GOV')

    np.testing.assert_allclose(gas_ftp_gov, gas_ftp_gov_base)

def test_ic():
    """
    Test module for testing the Investment Credit (IC).
    """
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    gas_ic = psc_result['XX']
    gas_ic_unrecovered = psc_result['XX']
    gas_ic_paid = psc_result['XX']

    gas_ic_base = load_testing(dataset_type='small_gas', key='IC')
    gas_ic_unrecovered_base = load_testing(dataset_type='small_gas', key='IC Unrec')
    gas_ic_paid_base = load_testing(dataset_type='small_gas', key='IC Paid')

    np.testing.assert_allclose(gas_ic, gas_ic_base)
    np.testing.assert_allclose(gas_ic_unrecovered, gas_ic_unrecovered_base)
    np.testing.assert_allclose(gas_ic_paid, gas_ic_paid_base)


def test_unrec_before_transfer():
    """
    Test module for testing the Unrecovered Cost Before Transfer.
    """
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    unrec_before_transfer = psc_result['XX']
    unrec_before_transfer_base = psc_result['XX']

    np.testing.assert_allclose(unrec_before_transfer, unrec_before_transfer_base)


def test_cost_tobe_recovered():
    """
    Test module for testing the Cost To Be Recover.
    """
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    gas_cost_to_be_recovered = psc_result['XX']
    gas_cost_to_be_recovered_base = load_testing(dataset_type='small_gas', key='Cost To Be Recovered')

    np.testing.assert_allclose(gas_cost_to_be_recovered, gas_cost_to_be_recovered_base)


def test_cost_recovery():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    # Calculated result
    gas_cost_recovery = psc_result['XX']

    # Expected result
    gas_cost_recovery_base = load_testing(dataset_type='small_gas', key='Cost Recovery')

    # Execute testing
    np.testing.assert_allclose(gas_cost_recovery, gas_cost_recovery_base)


def test_ets_before_transfer():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    # Calculated result
    gas_ets_before_transfer = psc_result['XX']

    # Expected result
    gas_ets_before_transfer_base = load_testing(dataset_type='small_gas', key='ETS Before Transfer')

    # Execute testing
    np.testing.assert_allclose(gas_ets_before_transfer, gas_ets_before_transfer_base)


def test_transfer():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    # Calculated result
    transfer = psc_result['XX']

    # Expected result
    transfer_base = load_testing(dataset_type='small_gas', key='Transfer')

    # Execute testing
    np.testing.assert_allclose(transfer, transfer_base)


def test_unrecovered_after_transfer():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    # Calculated result
    gas_unrecovered_after_transfer = psc_result['XX']

    # Expected result
    gas_unrecovered_after_transfer_base = load_testing(dataset_type='small_gas', key='Unrec After Transfer')

    # Execute testing
    np.testing.assert_allclose(gas_unrecovered_after_transfer, gas_unrecovered_after_transfer_base)


def test_ets_after_transfer():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    # Calculated result
    gas_ets_after_transfer = psc_result['XX']

    # Expected result
    gas_ets_after_transfer_base = load_testing(dataset_type='small_gas', key='ETS After Transfer')

    # Execute testing
    np.testing.assert_allclose(gas_ets_after_transfer, gas_ets_after_transfer_base)


def test_equity_share():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    # Calculated result
    gas_contractor_share = psc_result['XX']
    gas_government_share = psc_result['XX']

    # Expected result
    gas_contractor_share_base = load_testing(dataset_type='small_gas', key='Contractor Share')
    gas_government_share_base = load_testing(dataset_type='small_gas', key='Government Share')

    # Execute testing
    np.testing.assert_allclose(gas_contractor_share_base, gas_contractor_share)
    np.testing.assert_allclose(gas_government_share, gas_government_share_base)


def test_dmo():
    # TODO: Make the dmo test for gas.
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    # Calculated result
    gas_dmo_volume = psc_result['XX']
    gas_dmo_fee = psc_result['XX']
    gas_ddmo = psc_result['XX']

    # Expected result
    gas_dmo_volume_base = load_testing(dataset_type='small_gas', key='DMO')
    gas_dmo_fee_base = load_testing(dataset_type='small_gas', key='DMO Fee')
    gas_ddmo_base = load_testing(dataset_type='small_gas', key='Net DMO')

    # Execute testing
    np.testing.assert_allclose(gas_dmo_volume, gas_dmo_volume_base)
    np.testing.assert_allclose(gas_dmo_fee, gas_dmo_fee_base)
    np.testing.assert_allclose(gas_ddmo, gas_ddmo_base)


def test_taxable_income():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    # Calculated result
    gas_taxable_income = psc_result['XX']

    # Expected result
    gas_taxable_income_base = load_testing(dataset_type='small_gas', key='Contr. Taxable Income')

    # Execute testing
    np.testing.assert_allclose(gas_taxable_income, gas_taxable_income_base)


def test_tax_payment():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    # Calculated result
    gas_tax_payment = psc_result['XX']

    # Expected result
    gas_tax_payment_base = load_testing(dataset_type='small_gas', key='Contr. Tax')

    # Execute testing
    np.testing.assert_allclose(gas_tax_payment, gas_tax_payment_base)


def test_consistency_revenue_ct_gt():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    gross_revenue_calc = psc_result['XX'] + psc_result['XX']
    gross_revenue_engine = psc_result['XX'] + psc_result['XX']

    np.testing.assert_allclose(gross_revenue_calc, gross_revenue_engine)


def test_consistency_ets():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    revenue = psc_result['XX'] + psc_result['XX'] + psc_result['XX'] + psc_result['XX'] + psc_result['XX']
    revenue_base = psc_result['XX'] + psc_result['XX']

    np.testing.assert_allclose(revenue, revenue_base)


def test_contractor_take():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    # Calculated result
    gas_contractor_take = psc_result['XX']

    # Expected result
    gas_contractor_take_base = load_testing(dataset_type='small_gas', key='Contr. Net Share')

    # Execute testing
    np.testing.assert_allclose(gas_contractor_take, gas_contractor_take_base)


def test_government_take():
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    # Calculated result
    gas_government_take = psc_result['XX']

    # Expected result
    gas_government_take_base = load_testing(dataset_type='small_gas', key='GOI Take')

    # Execute testing
    np.testing.assert_allclose(gas_government_take, gas_government_take_base)


def test_sum_of_gov_take():
    """
    Test module to test the consistency of Government Take (GT), Taxable Income (TI), Cost Recovery (CR) vs revenue.
         Revenue = GT + TI + CR
    """
    data = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc_result = data.run()

    revenue = psc_result['XX']
    gov_take = psc_result['XX']
    taxable_income = psc_result['XX']
    cost_recovery = psc_result['XX']
    revenue_calc = gov_take + taxable_income + cost_recovery

    np.testing.assert_allclose(revenue, revenue_calc)
