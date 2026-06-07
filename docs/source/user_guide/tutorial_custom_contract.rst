Tutorial: Building a Custom Contract
=====================================

This tutorial walks through creating a Gross Split contract from scratch —
without relying on the sample dataset. By the end you will know how to
configure a contract, run it, and inspect the results.


Step 1: Import the classes
--------------------------

.. code-block:: python

   from pyscnomics.contracts.gross_split import GrossSplit
   from pyscnomics.contracts.costrecovery import CostRecovery
   from pyscnomics.tools.table import get_table
   from pyscnomics.tools.summary import get_summary
   from pyscnomics.econ.selection import (
       GrossSplitRegime,
       VariableSplit522017,
   )


Step 2: Configure production and price profiles
------------------------------------------------

A contract needs production volumes and commodity prices as NumPy arrays,
one value per year:

.. code-block:: python

   import numpy as np

   # 10-year production profile (oil in BOPD, gas in MMSCFD)
   years = 10
   oil_production = np.array([5000, 8000, 10000, 10000, 9000, 8000, 7000, 5000, 3000, 1000])
   gas_production = np.array([0] * years)

   # Price profile (USD/BO for oil, USD/MMSCF for gas)
   oil_price = np.array([70, 72, 74, 76, 78, 75, 73, 71, 70, 68])
   gas_price = np.array([0] * years)

   # Cost profiles (USD)
   capex = np.array([50_000_000, 30_000_000, 0, 0, 0, 0, 0, 0, 0, 0])
   opex = np.array([5_000_000] * years)


Step 3: Create a Gross Split contract
--------------------------------------

.. code-block:: python

   psc = GrossSplit(
       field_status="POD I",
       field_loc="Onshore",
       res_depth="<=2500",
       infra_avail="Well Developed",
       res_type="Conventional",
       api_oil="<25",
       domestic_use="50<=x<70",
       prod_stage="Secondary",
       co2_content="<5",
       h2s_content="<100",
   )

All variable-split parameters have sensible defaults. Override only the ones
relevant to your field.


Step 4: Run the contract
------------------------

.. code-block::python

   contract_arguments = {
       "regime": GrossSplitRegime.PERMEN_ESDM_52_2017,
       "oil_production": oil_production,
       "gas_production": gas_production,
       "oil_price": oil_price,
       "gas_price": gas_price,
       "capex_oil": capex,
       "capex_gas": np.zeros(years),
       "opex_oil": opex,
       "opex_gas": np.zeros(years),
   }

   psc.run(**contract_arguments)


Step 5: Inspect results
-----------------------

.. code-block:: python

   # Cashflow tables (oil, gas, consolidated)
   oil_table, gas_table, consolidated = get_table(contract=psc)
   print(consolidated.head())

   # Economic summary
   summary = psc.get_summary(reference_year=0)
   print(f"IRR:        {summary['ctr_irr']:.2%}")
   print(f"NPV:        ${summary['ctr_npv']:,.0f}")
   print(f"PI:         {summary['ctr_pi']:.2f}")
   print(f"Government: ${summary['gov_take']:,.0f}")


Next steps
----------

- Run **optimization** on your contract — see :doc:`/api_reference/optimize`
- Run **sensitivity** analysis to see how parameters affect IRR — see :doc:`chapter8_incentives_framework`
- Serve your contract via the **API** — see :doc:`api_server`
