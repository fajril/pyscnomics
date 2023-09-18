import numpy as np

from pyscnomics.dataset.sample import load_dataset, load_testing


def test_revenue():
    """
    Test module to test value of revenue calculated by pyscnomics.
    """

    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    revenue = psc._gas_revenue
    revenue_base = load_testing(dataset_type='small_gas', key='Revenue')

    np.testing.assert_allclose(revenue, revenue_base)


def test_depreciation():
    """
    Test module for testing the Depreciation.
    """
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    gas_depreciation = psc._gas_depreciation
    gas_depreciation_base = load_testing(dataset_type='small_gas', key='Depreciation')

    np.testing.assert_allclose(gas_depreciation, gas_depreciation_base)


def test_non_capital():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    gas_non_capital = psc._gas_non_capital
    gas_non_capital_base = load_testing(dataset_type='small_gas', key='Non Capital')

    np.testing.assert_allclose(gas_non_capital, gas_non_capital_base)


def test_ftp():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    gas_ftp = psc._gas_ftp
    gas_ftp_base = load_testing(dataset_type='small_gas', key='FTP')

    np.testing.assert_allclose(gas_ftp, gas_ftp_base)

def test_ftp_ctr():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    gas_ftp_ctr = psc._gas_ftp_ctr
    gas_ftp_ctr_base = load_testing(dataset_type='small_gas', key='FTP CTR')

    np.testing.assert_allclose(gas_ftp_ctr, gas_ftp_ctr_base)

def test_ftp_gov():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    gas_ftp_gov = psc._gas_ftp_gov
    gas_ftp_gov_base = load_testing(dataset_type='small_gas', key='FTP GOV')

    np.testing.assert_allclose(gas_ftp_gov, gas_ftp_gov_base)

def test_ic():
    """
    Test module for testing the Investment Credit (IC).
    """
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    gas_ic = psc._gas_ic
    gas_ic_unrecovered = psc._gas_ic_unrecovered
    gas_ic_paid = psc._gas_ic_paid

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
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    unrec_before_transfer = psc._gas_unrecovered_before_transfer
    unrec_before_transfer_base = load_testing(dataset_type='small_gas', key='Unrec Cost')

    np.testing.assert_allclose(unrec_before_transfer, unrec_before_transfer_base)


def test_cost_tobe_recovered():
    """
    Test module for testing the Cost To Be Recover.
    """
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    gas_cost_to_be_recovered = psc._gas_cost_to_be_recovered
    gas_cost_to_be_recovered_base = load_testing(dataset_type='small_gas', key='Cost To Be Recovered')

    np.testing.assert_allclose(gas_cost_to_be_recovered, gas_cost_to_be_recovered_base)


def test_cost_recovery():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    # Calculated result
    gas_cost_recovery = psc._gas_cost_recovery

    # Expected result
    gas_cost_recovery_base = load_testing(dataset_type='small_gas', key='Cost Recovery')

    # Execute testing
    np.testing.assert_allclose(gas_cost_recovery, gas_cost_recovery_base)


def test_ets_before_transfer():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    # Calculated result
    gas_ets_before_transfer = psc._gas_ets_before_transfer

    # Expected result
    gas_ets_before_transfer_base = load_testing(dataset_type='small_gas', key='ETS Before Transfer')

    # Execute testing
    np.testing.assert_allclose(gas_ets_before_transfer, gas_ets_before_transfer_base)


def test_transfer():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    # Calculated result
    transfer = psc._transfer_to_oil

    # Expected result
    transfer_base = load_testing(dataset_type='small_gas', key='Transfer')

    # Execute testing
    np.testing.assert_allclose(transfer, transfer_base)


def test_unrecovered_after_transfer():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    # Calculated result
    gas_unrecovered_after_transfer = psc._gas_unrecovered_after_transfer

    # Expected result
    gas_unrecovered_after_transfer_base = load_testing(dataset_type='small_gas', key='Unrec After Transfer')

    # Execute testing
    np.testing.assert_allclose(gas_unrecovered_after_transfer, gas_unrecovered_after_transfer_base)


def test_ets_after_transfer():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    # Calculated result
    gas_ets_after_transfer = psc._gas_ets_after_transfer

    # Expected result
    gas_ets_after_transfer_base = load_testing(dataset_type='small_gas', key='ETS After Transfer')

    # Execute testing
    np.testing.assert_allclose(gas_ets_after_transfer, gas_ets_after_transfer_base)


def test_equity_share():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    # Calculated result
    gas_contractor_share = psc._gas_contractor_share
    gas_government_share = psc._gas_government_share

    # Expected result
    gas_contractor_share_base = load_testing(dataset_type='small_gas', key='Contractor Share')
    gas_government_share_base = load_testing(dataset_type='small_gas', key='Government Share')

    # Execute testing
    np.testing.assert_allclose(gas_contractor_share_base, gas_contractor_share)
    np.testing.assert_allclose(gas_government_share, gas_government_share_base)


def test_dmo():
    # TODO: Make the dmo test for gas.
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    # Calculated result
    gas_dmo_volume = psc._gas_dmo_volume
    gas_dmo_fee = psc._gas_dmo_fee
    gas_ddmo = psc._gas_ddmo

    # Expected result
    gas_dmo_volume_base = load_testing(dataset_type='small_gas', key='DMO')
    gas_dmo_fee_base = load_testing(dataset_type='small_gas', key='DMO Fee')
    gas_ddmo_base = load_testing(dataset_type='small_gas', key='Net DMO')

    # Execute testing
    np.testing.assert_allclose(gas_dmo_volume, gas_dmo_volume_base)
    np.testing.assert_allclose(gas_dmo_fee, gas_dmo_fee_base)
    np.testing.assert_allclose(gas_ddmo, gas_ddmo_base)


def test_taxable_income():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    # Calculated result
    gas_taxable_income = psc._gas_taxable_income

    # Expected result
    gas_taxable_income_base = load_testing(dataset_type='small_gas', key='Contr. Taxable Income')

    # Execute testing
    np.testing.assert_allclose(gas_taxable_income, gas_taxable_income_base)


def test_tax_payment():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    # Calculated result
    gas_tax_payment = psc._gas_tax_payment

    # Expected result
    gas_tax_payment_base = load_testing(dataset_type='small_gas', key='Contr. Tax')

    # Execute testing
    np.testing.assert_allclose(gas_tax_payment, gas_tax_payment_base)


def test_consistency_revenue_ct_gt():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    gross_revenue_calc = psc._ctr_take + psc._gov_take
    gross_revenue_engine = psc._oil_revenue + psc._gas_revenue

    np.testing.assert_allclose(gross_revenue_calc, gross_revenue_engine)


def test_consistency_ets():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    revenue = psc._ftp + psc._gas_ic + psc._cost_recovery + psc._ctr_share + psc._gov_share
    revenue_base = psc._oil_revenue + psc._gas_revenue

    np.testing.assert_allclose(revenue, revenue_base)


def test_contractor_take():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    # Calculated result
    gas_contractor_take = psc._gas_contractor_take

    # Expected result
    gas_contractor_take_base = load_testing(dataset_type='small_gas', key='Contr. Net Share')

    # Execute testing
    np.testing.assert_allclose(gas_contractor_take, gas_contractor_take_base)


def test_government_take():
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    # Calculated result
    gas_government_take = psc._gas_government_take

    # Expected result
    gas_government_take_base = load_testing(dataset_type='small_gas', key='GOI Take')

    # Execute testing
    np.testing.assert_allclose(gas_government_take, gas_government_take_base)


def test_sum_of_gov_take():
    """
    Test module to test the consistency of Government Take (GT), Taxable Income (TI), Cost Recovery (CR) vs revenue.
         Revenue = GT + TI + CR
    """
    psc = load_dataset(dataset_type='small_gas', contract_type='cost_recovery')
    psc.run()

    revenue = psc._gas_revenue
    gov_take = psc._gas_government_take
    taxable_income = psc._gas_taxable_income
    cost_recovery = psc._gas_cost_recovery
    revenue_calc = gov_take + taxable_income + cost_recovery

    np.testing.assert_allclose(revenue, revenue_calc)
