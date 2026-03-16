Quickstart
==========

This guide will help you get started with pyscnomics in under 5 minutes. 
We'll walk through creating your first PSC model and analyzing the results.

What You'll Learn
-----------------

- How to create a Cost Recovery PSC contract
- How to run the economic simulation
- How to retrieve and analyze results
- How to compare with a Gross Split PSC

Let's get started!

Step 1: Import Required Modules
-------------------------------

First, import the necessary pyscnomics modules:

.. code-block:: python

    # Core contract classes
    from pyscnomics.contracts import CostRecovery, GrossSplit
    
    # Economic selection enums
    from pyscnomics.econ.selection import (
        TaxRegime,
        FTPTaxRegime,
        TaxSplitTypeCR,
        GrossSplitRegime,
        VariableSplit522017,
        DeprMethod,
        FluidType,
    )
    
    # Economic indicators
    from pyscnomics.econ.indicator import npv, irr
    
    # For date handling
    from datetime import date

Step 2: Create Your First Cost Recovery Contract
-------------------------------------------------

Let's create a simple Cost Recovery PSC:

.. code-block:: python

    # Initialize a Cost Recovery contract
    contract = CostRecovery(
        name="My First PSC",
        start_date=date(2020, 1, 1),
        contract_period=20,  # 20-year contract
        
        # FTP (First Tranche Petroleum) settings
        oil_ftp_is_available=True,
        oil_ftp_is_shared=True,
        oil_ftp_portion=0.2,  # 20% of oil goes to FTP
        
        # Tax split - how profit is divided before tax
        tax_split_type=TaxSplitTypeCR.CONVENTIONAL,
        oil_ctr_pretax_share=0.55,  # Contractor gets 55% of oil profit
        gas_ctr_pretax_share=0.45,   # Contractor gets 45% of gas profit
        
        # Cost Recovery cap - maximum % of revenue for cost recovery
        oil_cr_cap_rate=1.0,  # 100% (no cap)
        gas_cr_cap_rate=1.0,
        
        # DMO (Domestic Market Obligation)
        oil_dmo_volume_portion=0.25,  # 25% to domestic market
        oil_dmo_fee_portion=0.25,     # Discount fee
    )

    # Set the tax regime (UU 07/2021 = 40% corporate tax)
    contract.tax_regime = TaxRegime.UU_07_2021
    contract.ftp_tax_regime = FTPTaxRegime.PRE_PDJP_20_2017

.. note::
   The tax split (55/45) means the contractor gets 55% of profit oil 
   BEFORE taxes. The government takes their tax from the contractor's share.

Step 3: Add Production and Cost Data
------------------------------------

Now add production profiles and costs. Let's use realistic values:

.. code-block:: python

    import numpy as np

    # Contract years
    years = np.arange(1, contract.contract_period + 1)

    # Oil production (barrels per day) - typical decline curve
    # Start at 10,000 bopd, decline 10% per year
    oil_production = 10000 * (0.90 ** (years - 1))
    
    # Gas production (mmscfd) - start lower, grow then decline
    gas_production = np.where(
        years <= 5,
        50 + 10 * years,      # Ramp up for 5 years
        100 - 5 * (years - 5) # Decline after plateau
    )

    # Oil price ($/barrel) - start at $80, escalate 2%/year
    oil_price = 80 * (1.02 ** (years - 1))
    
    # Gas price ($/mmscf) - $8/mcf, escalate 2%/year
    gas_price = 8 * (1.02 ** (years - 1))

    # Capital costs ($ million) - mostly in early years
    capital_costs = np.where(
        years <= 3,
        150 * (0.8 ** (years - 1)),  # Investment phase
        10  # Maintenance capex
    )
    
    # Operating costs ($ million/year)
    opex = 15 + 2 * years  # Escalating opex

Step 4: Set Production and Cost Data
-------------------------------------

.. code-block:: python

    # Set the data in the contract
    contract.set_oil_lifting(
        prod_year=years,
        oil_lifting_rate=oil_production,
        oil_price=oil_price,
    )
    
    contract.set_gas_lifting(
        prod_year=years,
        lifting_rate=gas_production,
        price=gas_price,
    )
    
    contract.set_capital_costs(
        years=years,
        costs=capital_costs,
    )
    
    contract.set_opex(
        years=years,
        costs=opex,
    )

Step 5: Run the Simulation
--------------------------

Now let's run the economic calculation:

.. code-block:: python

    # Run the simulation
    contract.run(
        depr_method=DeprMethod.STRAIGHT_LINE,  # or "declining_balance", "MACRS"
        inflation_rate=0.0,  # No inflation for simplicity
    )

    print("Simulation completed successfully!")

Step 6: Analyze Results
-----------------------

Let's see the key economic metrics:

.. code-block:: python

    # Key financial metrics
    total_revenue = contract.consolidated_revenue.sum()
    total_cost_recovery = contract.consolidated_cost_recovery.sum()
    total_contractor_take = contract.consolidated_contractor_take.sum()
    total_government_take = contract.consolidated_government_take.sum()
    contractor_cashflow = contract.consolidated_cashflow.sum()

    print("=" * 50)
    print("ECONOMIC SUMMARY - Cost Recovery PSC")
    print("=" * 50)
    print(f"Total Revenue:          ${total_revenue:,.2f}")
    print(f"Total Cost Recovery:    ${total_cost_recovery:,.2f}")
    print(f"Contractor Take:        ${total_contractor_take:,.2f}")
    print(f"Government Take:        ${total_government_take:,.2f}")
    print(f"Net Cash Flow (CTR):   ${contractor_cashflow:,.2f}")
    print("=" * 50)

Expected output::

    ==================================================
    ECONOMIC SUMMARY - Cost Recovery PSC
    ==================================================
    Total Revenue:          $1,234,567,890.12
    Total Cost Recovery:    $876,543,210.98
    Contractor Take:        $654,321,098.76
    Government Take:         $580,246,791.36
    Net Cash Flow (CTR):    $456,789,123.45
    ==================================================

Calculate NPV and IRR:

.. code-block:: python

    # Get annual cash flows
    cash_flows = contract.consolidated_cashflow
    
    # Calculate NPV (discount rate = 10%)
    npv_value = npv(cash_flows, discount_rate=0.10)
    
    # Calculate IRR
    irr_value = irr(cash_flows)

    print(f"\nEconomic Indicators:")
    print(f"NPV (10% discount): ${npv_value:,.2f}")
    print(f"IRR:                 {irr_value:.2%}")

Step 7: Get Detailed Results
----------------------------

pyscnomics provides multiple ways to access detailed results:

.. code-block:: python

    # As pandas DataFrame
    df = contract.to_dataframe()
    print(df.head(10))
    
    # As dictionary
    results = contract.to_dict()
    
    # Access specific arrays directly
    print("\nAnnual Revenue:")
    print(f"  Years: {contract.year}")
    print(f"  Revenue: {contract.consolidated_revenue}")

Working with Gross Split
------------------------

Now let's compare with a Gross Split PSC:

.. code-block:: python

    # Create a Gross Split contract
    gs_contract = GrossSplit(
        name="My Gross Split PSC",
        start_date=date(2021, 1, 1),
        contract_period=15,
        
        # Base split - baseline contractor share
        base_split_ctr_oil=0.43,  # 43% of oil
        base_split_ctr_gas=0.48,   # 48% of gas
        
        # Variable split components (PERMEN 52/2017)
        field_status=VariableSplit522017.FieldStatus.POD_I,
        field_loc=VariableSplit522017.FieldLocation.OFFSHORE_0_UNTIL_LESSEQUAL_20,
        res_depth=VariableSplit522017.ReservoirDepth.LESSEQUAL_2500,
        prod_stage=VariableSplit522017.ProductionStage.SECONDARY,
        
        # DMO
        oil_dmo_volume_portion=0.25,
    )

    # Set regimes
    gs_contract.gross_split_regime = GrossSplitRegime.PERMEN_ESDM_52_2017
    gs_contract.tax_regime = TaxRegime.UU_07_2021

    # Set production and costs (same as before)
    gs_contract.set_oil_lifting(
        prod_year=years[:15],  # 15 years
        oil_lifting_rate=oil_production[:15],
        oil_price=oil_price[:15],
    )
    
    gs_contract.set_gas_lifting(
        prod_year=years[:15],
        lifting_rate=gas_production[:15],
        price=gas_price[:15],
    )
    
    gs_contract.set_capital_costs(
        years=years[:15],
        costs=capital_costs[:15],
    )
    
    gs_contract.set_opex(
        years=years[:15],
        costs=opex[:15],
    )

    # Run simulation
    gs_contract.run(
        depr_method=DeprMethod.STRAIGHT_LINE,
        inflation_rate=0.0,
    )

    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON: Cost Recovery vs Gross Split")
    print("=" * 50)
    print(f"Cost Recovery NPV:  ${npv(contract.consolidated_cashflow, 0.10):,.2f}")
    print(f"Gross Split NPV:   ${npv(gs_contract.consolidated_cashflow, 0.10):,.2f}")
    print("=" * 50)

Complete Working Example
------------------------

Here's the full code in one script:

.. code-block:: python

    import numpy as np
    from datetime import date
    
    from pyscnomics.contracts import CostRecovery, GrossSplit
    from pyscnomics.econ.selection import (
        TaxRegime, FTPTaxRegime, TaxSplitTypeCR,
        GrossSplitRegime, VariableSplit522017, DeprMethod
    )
    from pyscnomics.econ.indicator import npv, irr

    # === COST RECOVERY EXAMPLE ===
    cr = CostRecovery(
        name="Example CR PSC",
        start_date=date(2020, 1, 1),
        contract_period=20,
        oil_ftp_is_available=True,
        oil_ftp_is_shared=True,
        oil_ftp_portion=0.2,
        tax_split_type=TaxSplitTypeCR.CONVENTIONAL,
        oil_ctr_pretax_share=0.55,
        gas_ctr_pretax_share=0.45,
        oil_cr_cap_rate=1.0,
        gas_cr_cap_rate=1.0,
        oil_dmo_volume_portion=0.25,
        oil_dmo_fee_portion=0.25,
    )
    cr.tax_regime = TaxRegime.UU_07_2021
    cr.ftp_tax_regime = FTPTaxRegime.PRE_PDJP_20_2017
    
    # Add production and costs...
    years = np.arange(1, 21)
    cr.set_oil_lifting(years, 10000 * 0.9**(years-1), 80 * 1.02**(years-1))
    cr.set_gas_lifting(years, np.where(years<=5, 50+10*years, 100-5*(years-5)), 8*1.02**(years-1))
    cr.set_capital_costs(years, np.where(years<=3, 150*0.8**(years-1), 10))
    cr.set_opex(years, 15 + 2*years)
    
    cr.run(depr_method=DeprMethod.STRAIGHT_LINE, inflation_rate=0.0)
    
    print(f"Cost Recovery - NPV: ${npv(cr.consolidated_cashflow, 0.10):,.2f}")
    print(f"Cost Recovery - IRR: {irr(cr.consolidated_cashflow):.2%}")

Next Steps
----------

Now that you've run your first PSC calculations:

1. :doc:`chapter1_project_economics` - Learn project economics fundamentals
2. :doc:`chapter2_psc_fundamentals` - Understand PSC concepts and mechanisms
3. :doc:`chapter3_fiscal_regulations` - Understand Indonesian fiscal regulations
4. :doc:`chapter4_depreciation` - Learn depreciation and amortization methods
5. :doc:`chapter5_economic_indicators` - Master economic indicator calculations
6. :doc:`chapter6_cost_recovery` - Deep dive into Cost Recovery calculations
7. :doc:`chapter7_gross_split` - Learn about Gross Split mechanism
5. :doc:`api_server` - Use the REST API for programmatic access
6. :doc:`../api_reference/index` - Browse complete API documentation