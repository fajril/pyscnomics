import os

import numpy as np

from pyscnomics.dataset.sample import load_data, load_testing


def test_revenue():
    """
    Test module to test value of revenue calculated by pyscnomics.
    """

    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    revenue = psc._gas_revenue
    revenue_base = load_testing(dataset='gas_real', key='Revenue Gas (MMUS$)')

    np.testing.assert_allclose(revenue, revenue_base)

def test_ftp():
    """
    Test module for testing the First Trench Petroleum (FTP).
    """
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    gas_ftp = psc._gas_ftp
    gas_ftp_base = load_testing(dataset='gas_real', key='FTP Gas')

    np.testing.assert_allclose(gas_ftp, gas_ftp_base)

def test_ic():
    """
    Test module for testing the Investment Credit (IC).
    """
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    gas_ic = psc._gas_ic
    gas_ic_unrecovered = psc._gas_ic_unrecovered
    gas_ic_paid = psc._gas_ic_paid

    gas_ic_base = load_testing(dataset='gas_real', key='IC Gas')
    gas_ic_unrecovered_base = load_testing(dataset='gas_real', key='IC Unrec.')
    gas_ic_paid_base = load_testing(dataset='gas_real', key='IC Paid')

    np.testing.assert_allclose(gas_ic, gas_ic_base)
    np.testing.assert_allclose(gas_ic_unrecovered, gas_ic_unrecovered_base)
    np.testing.assert_allclose(gas_ic_paid, gas_ic_paid_base)

def test_depreciation():
    """
    Test module for testing the Investment Credit (IC).
    """
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    gas_depreciation = psc._gas_depreciation
    gas_depreciation_base = load_testing(dataset='gas_real', key='Deprec. Gas')
    np.testing.assert_allclose(gas_depreciation, gas_depreciation_base)

def test_non_capital():
    """
    Test module for testing The Non-Capital Cost.
    """
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    gas_depreciation = psc._gas_non_capital
    gas_depreciation_base = load_testing(dataset='gas_real', key='Opex Engine')
    np.testing.assert_allclose(gas_depreciation, gas_depreciation_base)

def test_unrec_before_transfer():
    """
    Test module for testing the Unrecovered Cost Before Transfer.
    """
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    unrec_before_transfer = psc._gas_unrecovered_before_transfer
    unrec_before_transfer_base = load_testing(dataset='gas_real', key='UnRec. Gas')

    np.testing.assert_allclose(unrec_before_transfer, unrec_before_transfer_base)

def test_cost_tobe_recovered():
    """
    Test module for testing the Cost To Be Recover.
    """
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    gas_cost_to_be_recovered = psc._gas_cost_to_be_recovered
    gas_cost_to_be_recovered_base = load_testing(dataset='gas_real', key='Cost To Be Recovered')

    np.testing.assert_allclose(gas_cost_to_be_recovered, gas_cost_to_be_recovered_base)

def test_cost_recovery():
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    # Calculated result
    gas_cost_recovery = psc._gas_cost_recovery

    # Expected result
    gas_cost_recovery_base = load_testing(dataset='gas_real', key='Cost Rec. Gas')

    # Execute testing
    np.testing.assert_allclose(gas_cost_recovery, gas_cost_recovery_base)


def test_ets_before_transfer():
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    # Calculated result
    gas_ets_before_transfer = psc._gas_ets_before_transfer

    # Expected result
    gas_ets_before_transfer_base = load_testing(dataset='gas_real', key='ETS After Transferred')

    # Execute testing
    np.testing.assert_allclose(gas_ets_before_transfer, gas_ets_before_transfer_base)


def test_transfer():
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    # Calculated result
    transfer_to_gas = psc._transfer_to_gas

    # Expected result
    transfer_to_gas_base = load_testing(dataset='gas_real', key='Transfered from  Oil')

    # Execute testing
    np.testing.assert_allclose(transfer_to_gas, transfer_to_gas_base)


def test_unrecovered_after_transfer():
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    # Calculated result
    gas_unrecovered_after_transfer = psc._gas_unrecovered_after_transfer

    # Expected result
    gas_unrecovered_after_transfer_base = load_testing(dataset='gas_real', key='UnRec. Gas after transfered')

    # Execute testing
    np.testing.assert_allclose(gas_unrecovered_after_transfer, gas_unrecovered_after_transfer_base)

def test_ets_after_transfer():
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    # Calculated result
    gas_ets_after_transfer = psc._gas_ets_after_transfer

    # Expected result
    gas_ets_after_transfer_base = load_testing(dataset='gas_real', key='ETS Gas Contr.')

    # Execute testing
    np.testing.assert_allclose(gas_ets_after_transfer, gas_ets_after_transfer_base)


def test_equity_share():
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    # Calculated result
    gas_contractor_share = psc._gas_contractor_share
    gas_government_share = psc._gas_government_share

    # Expected result
    gas_contractor_share_base = load_testing(dataset='gas_real', key='')
    gas_government_share_base = load_testing(dataset='gas_real', key='')

    # Execute testing
    np.testing.assert_allclose(gas_contractor_share, gas_government_share_base)
    np.testing.assert_allclose(gas_government_share, gas_government_share_base)


def test_dmo():
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    # Calculated result
    gas_dmo_volume = psc._gas_dmo_volume
    gas_dmo_fee = psc._gas_dmo_fee
    gas_ddmo = psc._gas_ddmo

    # Expected result
    gas_dmo_volume_base = load_testing(dataset='gas_real', key='DMO')
    gas_dmo_fee_base = load_testing(dataset='gas_real', key='DMO Fee')
    gas_ddmo_base = load_testing(dataset='gas_real', key='Net DMO')

    # Execute testing
    np.testing.assert_allclose(gas_dmo_volume, gas_dmo_volume_base)
    np.testing.assert_allclose(gas_dmo_fee, gas_dmo_fee_base)
    np.testing.assert_allclose(gas_ddmo, gas_ddmo_base)


def test_taxable_income():
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    # Calculated result
    gas_taxable_income = psc._gas_taxable_income

    # Expected result
    gas_taxable_income_base = load_testing(dataset='gas_real', key='Contr. Taxable Income')

    # Execute testing
    np.testing.assert_allclose(gas_taxable_income, gas_taxable_income_base)


def test_tax_payment():
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    # Calculated result
    gas_tax_payment = psc._gas_tax_payment

    # Expected result
    gas_tax_payment_base = load_testing(dataset='gas_real', key='Contr. Tax')

    # Execute testing
    np.testing.assert_allclose(gas_tax_payment, gas_tax_payment_base)


def test_contractor_take():
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    # Calculated result
    gas_contractor_take = psc._gas_contractor_take

    # Expected result
    gas_contractor_take_base = load_testing(dataset='gas_real', key='Contr. Net Share')

    # Execute testing
    np.testing.assert_allclose(gas_contractor_take, gas_contractor_take_base)


def test_government_take():
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    # Calculated result
    gas_government_take = psc._gas_government_take

    # Expected result
    gas_government_take_base = load_testing(dataset='gas_real', key='GOI Take')

    # Execute testing
    np.testing.assert_allclose(gas_government_take, gas_government_take_base)


def test_sum_of_gov_take():
    """
    Test module to test the consistency of Government Take (GT), Taxable Income (TI), Cost Recovery (CR) vs revenue.
         Revenue = GT + TI + CR
    """
    psc = load_data(dataset='gas_real', contract='cost_recovery')
    psc.run()

    revenue = psc._gas_revenue
    gov_take = psc._gas_government_take
    taxable_income = psc._gas_taxable_income
    cost_recovery = psc._gas_cost_recovery
    revenue_calc = gov_take + taxable_income + cost_recovery

    np.testing.assert_allclose(revenue, revenue_calc)



