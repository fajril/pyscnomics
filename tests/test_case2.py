import numpy as np

from pyscnomics.dataset.sample import load_data, load_testing
from pyscnomics.econ.selection import DeprMethod, FTPTaxRegime


# Defining the contract object
contract = load_data(dataset_type="case2", contract_type="cost_recovery")

# Defining the contract arguments
tax_rate = 0.405
ftp_tax_regime = FTPTaxRegime.DIRECT_MODE
depr_method = DeprMethod.PSC_DB
future_rate = 0.0
post_uu_22_year2001 = True

# Running the contract with contract arguments
contract.run(
    effective_tax_rate=tax_rate,
    ftp_tax_regime=ftp_tax_regime,
    depr_method=depr_method,
    post_uu_22_year2001=post_uu_22_year2001,
)
oil_columns = contract.get_oil_table_columns()
gas_columns = contract.get_gas_table_columns()
consolidated_columns = contract.get_consolidated_table_columns()
summary = contract.get_post_run_summary_profile()


def test_revenue():
    """
    Test module to test value of revenue calculated by pyscnomics.
    """

    oil_revenue = oil_columns["revenue"]
    oil_revenue_base = np.asarray(load_testing(dataset_type="case2", key="oil_revenue"))

    gas_revenue = gas_columns["revenue"]
    gas_revenue_base = np.asarray(load_testing(dataset_type="case2", key="gas_revenue"))

    np.testing.assert_allclose(oil_revenue, oil_revenue_base, atol=1e-6)
    np.testing.assert_allclose(gas_revenue, gas_revenue_base, atol=1e-6)


def test_depreciation():
    """
    Test module for testing the Depreciation.
    """

    oil_depreciation = (
        oil_columns["depreciations_sunk_cost"]
        + oil_columns["depreciations_preonstream"]
        + oil_columns["depreciations_postonstream"]
    )
    oil_depreciation_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_depreciation")
    )
    gas_depreciation = (
        gas_columns["depreciations_sunk_cost"]
        + gas_columns["depreciations_preonstream"]
        + gas_columns["depreciations_postonstream"]
    )
    gas_depreciation_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_depreciation")
    )

    np.testing.assert_allclose(oil_depreciation, oil_depreciation_base, atol=1e-6)
    np.testing.assert_allclose(gas_depreciation, gas_depreciation_base, atol=1e-6)


def test_non_capital():
    oil_non_capital = oil_columns["expenses_non_capital"]
    oil_non_capital_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_non_capital")
    )
    gas_non_capital = gas_columns["expenses_non_capital"]
    gas_non_capital_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_non_capital")
    )

    np.testing.assert_allclose(oil_non_capital, oil_non_capital_base, atol=1e-6)
    np.testing.assert_allclose(gas_non_capital, gas_non_capital_base, atol=1e-6)


def test_ftp():
    oil_ftp = oil_columns["ftp"]
    oil_ftp_base = np.asarray(load_testing(dataset_type="case2", key="oil_ftp"))
    gas_ftp = gas_columns["ftp"]
    gas_ftp_base = np.asarray(load_testing(dataset_type="case2", key="gas_ftp"))

    np.testing.assert_allclose(oil_ftp, oil_ftp_base, atol=1e-6)
    np.testing.assert_allclose(gas_ftp, gas_ftp_base, atol=1e-6)


def test_ftp_ctr():
    oil_ftp_ctr = oil_columns["ftp_ctr"]
    oil_ftp_ctr_base = np.asarray(load_testing(dataset_type="case2", key="oil_ftp_ctr"))
    gas_ftp_ctr = gas_columns["ftp_ctr"]
    gas_ftp_ctr_base = np.asarray(load_testing(dataset_type="case2", key="gas_ftp_ctr"))

    np.testing.assert_allclose(oil_ftp_ctr, oil_ftp_ctr_base, atol=1e-6)
    np.testing.assert_allclose(gas_ftp_ctr, gas_ftp_ctr_base, atol=1e-6)


def test_ftp_gov():
    oil_ftp_gov = oil_columns["ftp_gov"]
    oil_ftp_gov_base = np.asarray(load_testing(dataset_type="case2", key="oil_ftp_gov"))
    gas_ftp_gov = gas_columns["ftp_gov"]
    gas_ftp_gov_base = np.asarray(load_testing(dataset_type="case2", key="gas_ftp_gov"))

    np.testing.assert_allclose(oil_ftp_gov, oil_ftp_gov_base, atol=1e-6)
    np.testing.assert_allclose(gas_ftp_gov, gas_ftp_gov_base, atol=1e-6)


def test_ic():
    """
    Test module for testing the Investment Credit (IC).
    """

    oil_ic = np.zeros_like(oil_columns["investment_credit"])
    oil_ic_unrecovered = np.zeros_like(oil_columns["investment_credit"])
    oil_ic_paid = oil_columns["investment_credit"]
    gas_ic = np.zeros_like(gas_columns["investment_credit"])
    gas_ic_unrecovered = np.zeros_like(gas_columns["investment_credit"])
    gas_ic_paid = gas_columns["investment_credit"]

    oil_ic_base = np.asarray(load_testing(dataset_type="case2", key="oil_ic"))
    oil_ic_unrecovered_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_ic_unrec")
    )
    oil_ic_paid_base = np.asarray(load_testing(dataset_type="case2", key="oil_ic_paid"))
    gas_ic_base = np.asarray(load_testing(dataset_type="case2", key="gas_ic"))
    gas_ic_unrecovered_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_ic_unrec")
    )
    gas_ic_paid_base = np.asarray(load_testing(dataset_type="case2", key="gas_ic_paid"))

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

    oil_unrec_before_transfer = oil_columns["unrecovered_cost"]
    oil_unrec_before_transfer_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_unrec_cost")
    )
    gas_unrec_before_transfer = gas_columns["unrecovered_cost"]
    gas_unrec_before_transfer_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_unrec_cost")
    )

    np.testing.assert_allclose(
        oil_unrec_before_transfer, oil_unrec_before_transfer_base, atol=1e-6
    )
    np.testing.assert_allclose(
        gas_unrec_before_transfer, gas_unrec_before_transfer_base, atol=1e-6
    )


def test_cost_recovery():
    # Calculated result
    oil_cost_recovery = oil_columns["cost_recovery"]
    gas_cost_recovery = gas_columns["cost_recovery"]

    # Expected result
    oil_cost_recovery_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_cost_recovery")
    )
    gas_cost_recovery_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_cost_recovery")
    )

    # Execute testing
    np.testing.assert_allclose(oil_cost_recovery, oil_cost_recovery_base, atol=1e-6)
    np.testing.assert_allclose(gas_cost_recovery, gas_cost_recovery_base, atol=1e-6)


def test_cost_recovery_balance_uses_prior_post_transfer_unrecovered():
    balances = (
        (
            oil_columns["cost_recovery"],
            oil_columns["unrecovered_cost"],
            oil_columns["depreciations_sunk_cost"]
            + oil_columns["depreciations_preonstream"]
            + oil_columns["depreciations_postonstream"],
            oil_columns["non_depreciables_sunk_cost"]
            + oil_columns["non_depreciables_preonstream"]
            + oil_columns["non_depreciables_postonstream"],
            oil_columns["unrec_after_transfer"],
        ),
        (
            gas_columns["cost_recovery"],
            gas_columns["unrecovered_cost"],
            gas_columns["depreciations_sunk_cost"]
            + gas_columns["depreciations_preonstream"]
            + gas_columns["depreciations_postonstream"],
            gas_columns["non_depreciables_sunk_cost"]
            + gas_columns["non_depreciables_preonstream"]
            + gas_columns["non_depreciables_postonstream"],
            gas_columns["unrec_after_transfer"],
        ),
    )

    for (
        cost_recovery,
        unrecovered_before,
        depreciation,
        non_capital,
        unrecovered_after,
    ) in balances:
        prior_unrecovered_after = np.concatenate(([0.0], unrecovered_after[:-1]))
        np.testing.assert_allclose(
            cost_recovery + unrecovered_before,
            depreciation + non_capital + prior_unrecovered_after,
            atol=1e-6,
        )


def test_ets_before_transfer():
    # Calculated result
    oil_ets_before_transfer = oil_columns["ets_before_transfer"]
    gas_ets_before_transfer = gas_columns["ets_before_transfer"]

    # Expected result
    oil_ets_before_transfer_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_ets_before_transfer")
    )
    gas_ets_before_transfer_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_ets_before_transfer")
    )

    # Execute testing
    np.testing.assert_allclose(
        oil_ets_before_transfer, oil_ets_before_transfer_base, atol=1e-6
    )
    np.testing.assert_allclose(
        gas_ets_before_transfer, gas_ets_before_transfer_base, atol=1e-6
    )


def test_transfer():
    # Calculated result
    transfer_to_gas = oil_columns["transfer_to_gas"]
    transfer_to_oil = gas_columns["transfer_to_oil"]

    # Expected result
    transfer_to_gas_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_transfer_to_gas")
    )
    transfer_to_oil_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_transfer_to_oil")
    )

    # Execute testing
    np.testing.assert_allclose(transfer_to_gas, transfer_to_gas_base, atol=1e-6)
    np.testing.assert_allclose(transfer_to_oil, transfer_to_oil_base, atol=1e-6)


def test_unrecovered_after_transfer():
    # Calculated result
    oil_unrecovered_after_transfer = oil_columns["unrec_after_transfer"]
    gas_unrecovered_after_transfer = gas_columns["unrec_after_transfer"]

    # Expected result
    oil_unrecovered_after_transfer_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_unrec_after_transfer")
    )
    gas_unrecovered_after_transfer_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_unrec_after_transfer")
    )

    # Execute testing
    np.testing.assert_allclose(
        oil_unrecovered_after_transfer, oil_unrecovered_after_transfer_base, atol=1e-6
    )
    np.testing.assert_allclose(
        gas_unrecovered_after_transfer, gas_unrecovered_after_transfer_base, atol=1e-6
    )


def test_ets_after_transfer():
    # Calculated result
    oil_ets_after_transfer = oil_columns["ets_after_transfer"]
    gas_ets_after_transfer = gas_columns["ets_after_transfer"]

    # Expected result
    oil_ets_after_transfer_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_ets_after_transfer")
    )
    gas_ets_after_transfer_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_ets_after_transfer")
    )

    # Execute testing
    np.testing.assert_allclose(
        oil_ets_after_transfer, oil_ets_after_transfer_base, atol=1e-6
    )
    np.testing.assert_allclose(
        gas_ets_after_transfer, gas_ets_after_transfer_base, atol=1e-6
    )


def test_equity_share():
    # Calculated result
    oil_contractor_share = oil_columns["contractor_share"]
    oil_government_share = oil_columns["government_share"]

    gas_contractor_share = gas_columns["contractor_share"]
    gas_government_share = gas_columns["government_share"]

    # Expected result
    oil_contractor_share_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_ctr_share")
    )
    oil_government_share_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_gov_share")
    )
    gas_contractor_share_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_ctr_share")
    )
    gas_government_share_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_gov_share")
    )

    # Execute testing
    np.testing.assert_allclose(
        oil_contractor_share, oil_contractor_share_base, atol=1e-6
    )
    np.testing.assert_allclose(
        oil_government_share, oil_government_share_base, atol=1e-6
    )
    np.testing.assert_allclose(
        gas_contractor_share, gas_contractor_share_base, atol=1e-6
    )
    np.testing.assert_allclose(
        gas_government_share, gas_government_share_base, atol=1e-6
    )


def test_dmo():
    # Calculated result
    oil_dmo_volume = oil_columns["dmo_volume"]
    oil_dmo_fee = oil_columns["dmo_fee"]
    oil_ddmo = oil_columns["ddmo"]

    gas_dmo_volume = gas_columns["dmo_volume"]
    gas_dmo_fee = gas_columns["dmo_fee"]
    gas_ddmo = gas_columns["ddmo"]

    # Expected result
    oil_dmo_volume_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_dmo_volume")
    )
    oil_dmo_fee_base = np.asarray(load_testing(dataset_type="case2", key="oil_dmo_fee"))
    oil_ddmo_base = np.asarray(load_testing(dataset_type="case2", key="oil_ddmo"))

    gas_dmo_volume_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_dmo_volume")
    )
    gas_dmo_fee_base = np.asarray(load_testing(dataset_type="case2", key="gas_dmo_fee"))
    gas_ddmo_base = np.asarray(load_testing(dataset_type="case2", key="gas_ddmo"))

    # Execute testing
    np.testing.assert_allclose(oil_dmo_volume, oil_dmo_volume_base, atol=1e-6)
    np.testing.assert_allclose(oil_dmo_fee, oil_dmo_fee_base, atol=1e-6)
    np.testing.assert_allclose(oil_ddmo, oil_ddmo_base, atol=1e-6)

    np.testing.assert_allclose(gas_dmo_volume, gas_dmo_volume_base, atol=1e-6)
    np.testing.assert_allclose(gas_dmo_fee, gas_dmo_fee_base, atol=1e-6)
    np.testing.assert_allclose(gas_ddmo, gas_ddmo_base, atol=1e-6)


def test_taxable_income():
    # Calculated result
    oil_taxable_income = oil_columns["taxable_income"]
    gas_taxable_income = gas_columns["taxable_income"]

    # Expected result
    oil_taxable_income_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_taxable_income")
    )
    gas_taxable_income_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_taxable_income")
    )

    # Execute testing
    np.testing.assert_allclose(oil_taxable_income, oil_taxable_income_base, atol=1e-6)
    np.testing.assert_allclose(gas_taxable_income, gas_taxable_income_base, atol=1e-6)


def test_tax():
    # Calculated result
    oil_tax = oil_columns["tax_payment"]
    gas_tax = gas_columns["tax_payment"]

    # Expected result
    oil_tax_base = np.asarray(load_testing(dataset_type="case2", key="oil_tax"))
    gas_tax_base = np.asarray(load_testing(dataset_type="case2", key="gas_tax"))

    # Execute testing
    np.testing.assert_allclose(oil_tax, oil_tax_base, atol=1e-6)
    np.testing.assert_allclose(gas_tax, gas_tax_base, atol=1e-6)


def test_ctr_net_share():
    # Calculated result
    oil_ctr_share_after_tax = oil_columns["contractor_net_share"]
    gas_ctr_share_after_tax = gas_columns["contractor_net_share"]

    # Expected result
    oil_ctr_share_after_tax_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_ctr_net_share")
    )
    gas_ctr_share_after_tax_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_ctr_net_share")
    )

    # Execute testing
    np.testing.assert_allclose(
        oil_ctr_share_after_tax, oil_ctr_share_after_tax_base, atol=1e-6
    )
    np.testing.assert_allclose(
        gas_ctr_share_after_tax, gas_ctr_share_after_tax_base, atol=1e-6
    )


def test_ctr_take():
    # Calculated result
    oil_ctr_share_after_tax = oil_columns["contractor_net_share"]
    gas_ctr_share_after_tax = gas_columns["contractor_net_share"]

    # Expected result
    oil_ctr_share_after_tax_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_ctr_net_share")
    )
    gas_ctr_share_after_tax_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_ctr_net_share")
    )

    # Execute testing
    np.testing.assert_allclose(
        oil_ctr_share_after_tax, oil_ctr_share_after_tax_base, atol=1e-6
    )
    np.testing.assert_allclose(
        gas_ctr_share_after_tax, gas_ctr_share_after_tax_base, atol=1e-6
    )


def test_ctr_cashflow():
    # Calculated result
    oil_ctr_cashflow = oil_columns["cashflow"]
    gas_ctr_cashflow = gas_columns["cashflow"]

    # Expected result
    oil_ctr_cashflow_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_ctr_cashflow")
    )
    gas_ctr_cashflow_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_ctr_cashflow")
    )

    # Execute testing
    np.testing.assert_allclose(oil_ctr_cashflow, oil_ctr_cashflow_base, atol=1e-6)
    np.testing.assert_allclose(gas_ctr_cashflow, gas_ctr_cashflow_base, atol=1e-6)


def test_goi_take():
    # Calculated result
    oil_goi_take = oil_columns["government_take"]
    gas_goi_take = gas_columns["government_take"]

    # Expected result
    oil_goi_take_base = np.asarray(
        load_testing(dataset_type="case2", key="oil_goi_take")
    )
    gas_goi_take_base = np.asarray(
        load_testing(dataset_type="case2", key="gas_goi_take")
    )

    # Execute testing
    np.testing.assert_allclose(oil_goi_take, oil_goi_take_base, atol=1e-6)
    np.testing.assert_allclose(gas_goi_take, gas_goi_take_base, atol=1e-6)


def test_consistency():
    oil_contractor_take = summary.oil_contractor_take
    gas_contractor_take = summary.gas_contractor_take
    assert oil_contractor_take is not None
    assert gas_contractor_take is not None

    # Top side result
    gross_revenue_topside = oil_columns["revenue"] + gas_columns["revenue"]

    # Bottom side result
    gross_revenue_botside = (
        oil_contractor_take
        + gas_contractor_take
        + oil_columns["government_take"]
        + gas_columns["government_take"]
    )

    # Execute testing
    np.testing.assert_allclose(gross_revenue_topside, gross_revenue_botside, atol=1e-6)


def test_consistency_ets():
    # Calculated result Oil
    oil_revenue = (
        oil_columns["ftp"]
        + np.zeros_like(oil_columns["investment_credit"])
        + oil_columns["cost_recovery"]
        + oil_columns["ets_before_transfer"]
    )

    # Calculated result Gas
    gas_revenue = (
        gas_columns["ftp"]
        + np.zeros_like(gas_columns["investment_credit"])
        + gas_columns["cost_recovery"]
        + gas_columns["ets_before_transfer"]
    )

    # Expected result Oil
    oil_revenue_base = oil_columns["revenue"]
    gas_revenue_base = gas_columns["revenue"]

    np.testing.assert_allclose(oil_revenue, oil_revenue_base)
    np.testing.assert_allclose(gas_revenue, gas_revenue_base)


def test_consolidated_revenue():
    # Calculated result
    engine = consolidated_columns["revenue"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_revenue"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_depreciation():
    # Calculated result
    engine = (
        consolidated_columns["depreciations_sunk_cost"]
        + consolidated_columns["depreciations_preonstream"]
        + consolidated_columns["depreciations_postonstream"]
    )

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_depreciation"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_non_capital():
    # Calculated result
    engine = consolidated_columns["expenses_non_capital"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_non_capital"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ftp():
    # Calculated result
    engine = consolidated_columns["ftp"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_ftp"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ftp_ctr():
    # Calculated result
    engine = consolidated_columns["ftp_ctr"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_ftp_ctr"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ftp_gov():
    # Calculated result
    engine = consolidated_columns["ftp_gov"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_ftp_gov"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ic():
    # Calculated result
    engine = np.zeros_like(consolidated_columns["investment_credit"])

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_ic"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ic_unrec():
    # Calculated result
    engine = np.zeros_like(consolidated_columns["investment_credit"])

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_ic_unrec"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ic_paid():
    # Calculated result
    engine = consolidated_columns["investment_credit"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_ic_paid"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_unrec_cost():
    # Calculated result
    engine = consolidated_columns["unrecovered_cost"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_unrec_cost"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_cost_recovery_before_transfer():
    # Calculated result
    engine = consolidated_columns["cost_recovery"]

    # Expected result
    base = np.asarray(
        load_testing(dataset_type="case2", key="cnsltd_cost_recovery_before_transfer")
    )

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ets_before_transfer():
    # Calculated result
    engine = consolidated_columns["ets_before_transfer"]

    # Expected result
    base = np.asarray(
        load_testing(dataset_type="case2", key="cnsltd_ets_before_transfer")
    )

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_unrec_after_transfer():
    # Calculated result
    engine = consolidated_columns["unrec_after_transfer"]

    # Expected result
    base = np.asarray(
        load_testing(dataset_type="case2", key="cnsltd_unrec_after_transfer")
    )

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_cost_recovery_after_transfer():
    # Calculated result
    engine = consolidated_columns["cost_recovery_after_tf"]

    # Expected result
    base = np.asarray(
        load_testing(dataset_type="case2", key="cnsltd_cost_recovery_after_transfer")
    )

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ets_after_transfer():
    # Calculated result
    engine = consolidated_columns["ets_after_transfer"]

    # Expected result
    base = np.asarray(
        load_testing(dataset_type="case2", key="cnsltd_ets_after_transfer")
    )

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ctr_share():
    # Calculated result
    engine = consolidated_columns["contractor_share"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_ctr_share"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_gov_share():
    # Calculated result
    engine = consolidated_columns["government_share"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_gov_share"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_dmo_volume():
    # Calculated result
    engine = consolidated_columns["dmo_volume"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_dmo_volume"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_dmo_fee():
    # Calculated result
    engine = consolidated_columns["dmo_fee"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_dmo_fee"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ddmo():
    # Calculated result
    engine = consolidated_columns["ddmo"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_ddmo"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_taxable_income():
    # Calculated result
    engine = consolidated_columns["taxable_income"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_taxable_income"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_tax_due():
    assert load_testing(dataset_type="case2", key="cnsltd_tax_due") is None


def test_consolidated_unpaid_tax_balance():
    assert load_testing(dataset_type="case2", key="cnsltd_unpaid_tax_balance") is None


def test_consolidated_tax_payment():
    # Calculated result
    engine = consolidated_columns["tax_payment"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_tax_payment"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_ctr_net_share():
    # Calculated result
    engine = consolidated_columns["contractor_net_share"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_ctr_net_share"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_gov_take():
    # Calculated result
    engine = consolidated_columns["government_take"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_gov_take"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)


def test_consolidated_cashflow():
    # Calculated result
    engine = consolidated_columns["cashflow"]

    # Expected result
    base = np.asarray(load_testing(dataset_type="case2", key="cnsltd_cashflow"))

    # Execute testing
    np.testing.assert_allclose(engine, base, atol=1e-6)
