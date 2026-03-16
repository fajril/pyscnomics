Chapter 3: Fiscal Regulations and Taxation
==========================================

This chapter provides a comprehensive overview of the fiscal regulations and taxation frameworks implemented in pyscnomics. Understanding these regulations is essential for accurately modeling Production Sharing Contracts (PSCs) under Indonesian petroleum fiscal regimes.

3.1 Introduction to Indonesian Petroleum Fiscal Framework
----------------------------------------------------------

Indonesia's petroleum fiscal regime has evolved significantly over decades, with regulations issued by various government bodies:

- **Ministry of Energy and Mineral Resources (ESDM)**: Issues PERMEN regulations governing PSC terms
- **People's Representative Council (DPR)**: Enacts UU (Undang-Undang) laws on taxation
- **Directorate General of Taxes (DJP)**: Issues PDJP regulations on tax implementation

pyscnomics implements these regulations through enumeration classes and configuration parameters, allowing users to select the appropriate regime for their contract modeling.

3.2 PERMEN ESDM Regulations
---------------------------

PERMEN (Peraturan Menteri) regulations issued by the Minister of Energy and Mineral Resources define the operational and fiscal terms of PSCs. pyscnomics supports multiple PERMEN regimes for both Cost Recovery and Gross Split contracts.

3.2.1 Gross Split Regimes
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Gross Split PSC was introduced through PERMEN ESDM No. 8/2017 as a new mechanism for oil and gas contracts. While initially mandatory for new contracts, subsequent regulations (PERMEN ESDM No. 12/2020 and No. 13/2024) introduced flexibility, allowing both Cost Recovery and Gross Split schemes to coexist. Contractors can now choose or convert between schemes based on project characteristics.

Supported Regimes in pyscnomics
"""""""""""""""""""""""""""""""

.. list-table:: Gross Split Regimes in pyscnomics
   :header-rows: 1

   * - Regime Enum
     - Regulation
     - Variable Components
     - Default Split (Oil/Gas)
   * - ``PERMEN_ESDM_8_2017``
     - PERMEN ESDM No. 8 Tahun 2017
     - 9 components
     - 43% / 48%
   * - ``PERMEN_ESDM_52_2017``
     - PERMEN ESDM No. 52 Tahun 2017
     - 9 components + Progressive
     - 43% / 48%
   * - ``PERMEN_ESDM_20_2019``
     - PERMEN ESDM No. 20 Tahun 2019
     - 5 components (Variable Split 5)
     - 43% / 48%
   * - ``PERMEN_ESDM_12_2020``
     - PERMEN ESDM No. 12 Tahun 2020
     - 5 components
     - 43% / 48%
   * - ``PERMEN_ESDM_13_2024``
     - PERMEN ESDM No. 13 Tahun 2024
     - 3 components
     - 43% / 48%

How to Configure in pyscnomics
""""""""""""""""""""""""""""""""""

.. code-block:: python

   from pyscnomics.contracts import GrossSplit
   from pyscnomics.econ.selection import GrossSplitRegime

   # Create contract and set regime
   contract = GrossSplit(
       start_date=date(2024, 1, 1),
       end_date=date(2043, 12, 31),
       # ... other parameters
   )
   
   # Set the regulatory regime
   contract.gross_split_regime = GrossSplitRegime.PERMEN_ESDM_52_2017
   
   # Run calculation
   contract.run()

Regime-Specific Variable Split Components
""""""""""""""""""""""""""""""""""""""""""

Each PERMEN regime defines specific variable split components that adjust the contractor's base split:

**PERMEN ESDM 8/2017 (Initial Regulation)**

Implements 9 variable components:

.. list-table:: Field Status Component (8/2017)
   :header-rows: 1

   * - Status
     - Enum Value
     - Split Adjustment
   * - POD I
     - ``FieldStatus.POD_I``
     - +5%
   * - POD II
     - ``FieldStatus.POD_II``
     - 0%
   * - POFD
     - ``FieldStatus.POFD``
     - 0%
   * - No POD
     - ``FieldStatus.NO_POD``
     - -5%

.. list-table:: Field Location Component (8/2017)
   :header-rows: 1

   * - Location
     - Enum Value
     - Split Adjustment
   * - Onshore
     - ``FieldLocation.ONSHORE``
     - 0%
   * - Offshore 0 < h ≤ 20m
     - ``FieldLocation.OFFSHORE_0_UNTIL_LESSEQUAL_20``
     - +8%
   * - Offshore 20 < h ≤ 50m
     - ``FieldLocation.OFFSHORE_20_UNTIL_LESSEQUAL_50``
     - +10%
   * - Offshore 50 < h ≤ 150m
     - ``FieldLocation.OFFSHORE_50_UNTIL_LESSEQUAL_150``
     - +12%
   * - Offshore 150 < h ≤ 1000m
     - ``FieldLocation.OFFSHORE_150_UNTIL_LESSEQUAL_1000``
     - +14%
   * - Offshore h > 1000m
     - ``FieldLocation.OFFSHORE_GREATERTHAN_1000``
     - +16%

.. list-table:: Reservoir Depth Component (8/2017)
   :header-rows: 1

   * - Depth
     - Enum Value
     - Split Adjustment
   * - ≤ 2500m
     - ``ReservoirDepth.LESSEQUAL_2500``
     - 0%
   * - > 2500m
     - ``ReservoirDepth.GREATERTHAN_2500``
     - +1%

.. list-table:: Infrastructure Availability Component (8/2017)
   :header-rows: 1

   * - Status
     - Enum Value
     - Split Adjustment
   * - Well Developed
     - ``InfrastructureAvailability.WELL_DEVELOPED``
     - 0%
   * - New Frontier Offshore
     - ``InfrastructureAvailability.NEW_FRONTIER_OFFSHORE``
     - +2%
   * - New Frontier Onshore
     - ``InfrastructureAvailability.NEW_FRONTIER_ONSHORE``
     - +2%

.. list-table:: Reservoir Type Component (8/2017)
   :header-rows: 1

   * - Type
     - Enum Value
     - Split Adjustment
   * - Conventional
     - ``ReservoirType.CONVENTIONAL``
     - 0%
   * - Non-Conventional
     - ``ReservoirType.NON_CONVENTIONAL``
     - +16%

.. list-table:: CO2 Content Component (8/2017)
   :header-rows: 1

   * - CO2 Content
     - Enum Value
     - Split Adjustment
   * - < 5%
     - ``CO2Content.LESSTHAN_5``
     - 0%
   * - 5% ≤ x < 10%
     - ``CO2Content.EQUAL_5_UNTIL_LESSTHAN_10``
     - +0.5%
   * - 10% ≤ x < 20%
     - ``CO2Content.EQUAL_10_UNTIL_LESSTHAN_20``
     - +1%
   * - 20% ≤ x < 40%
     - ``CO2Content.EQUAL_20_UNTIL_LESSTHAN_40``
     - +1.5%
   * - 40% ≤ x < 60%
     - ``CO2Content.EQUAL_40_UNTIL_LESSTHAN_60``
     - +2%
   * - x ≥ 60%
     - ``CO2Content.EQUALGREATERTHAN_60``
     - +4%

.. list-table:: H2S Content Component (8/2017)
   :header-rows: 1

   * - H2S Content (ppm)
     - Enum Value
     - Split Adjustment
   * - < 100
     - ``H2SContent.LESSTHAN_100``
     - 0%
   * - 100 ≤ x < 300
     - ``H2SContent.EQUAL_100_UNTIL_LESSTHAN_300``
     - +0.5%
   * - 300 ≤ x < 500
     - ``H2SContent.EQUAL_300_UNTIL_LESSTHAN_500``
     - +0.75%
   * - x ≥ 500
     - ``H2SContent.EQUALGREATERTHAN_500``
     - +1%

.. list-table:: API Oil Gravity Component (8/2017)
   :header-rows: 1

   * - API Gravity
     - Enum Value
     - Split Adjustment
   * - < 25 (Heavy Oil)
     - ``APIOil.LESSTHAN_25``
     - +1%
   * - ≥ 25 (Medium/Light Oil)
     - ``APIOil.EQUALGREATERTHAN_25``
     - 0%

.. list-table:: Domestic Use Component (8/2017)
   :header-rows: 1

   * - Domestic Use
     - Enum Value
     - Split Adjustment
   * - < 30%
     - ``DomesticUse.LESSTHAN_30``
     - 0%
   * - 30% ≤ x < 50%
     - ``DomesticUse.EQUAL_30_UNTIL_LESSTHAN_50``
     - +2%
   * - 50% ≤ x < 70%
     - ``DomesticUse.EQUAL_50_UNTIL_LESSTHAN_70``
     - +3%
   * - 70% ≤ x < 100%
     - ``DomesticUse.EQUAL_70_UNTIL_LESSTHAN_100``
     - +4%

**PERMEN ESDM 52/2017 (Enhanced with Progressive Split)**

Includes all 8/2017 components plus:

10. **Progressive Split (Price-Based)**:

    - **Oil**: When price deviates from $85/bbl
      
      .. math::
         
         PS_{oil} = \frac{(85 - P_{oil}) \times 0.25}{100}
      
      Where :math:`P_{oil}` is the oil price in USD/barrel.
      
      *Higher prices reduce contractor share; lower prices increase it.*
    
    - **Gas**: When price deviates from $7-10/MMBtu band
      
      .. math::
         
         PS_{gas} = 
         \begin{cases}
         \frac{(7 - P_{gas}) \times 2.5}{100} & \text{if } P_{gas} < 7 \\
         0 & \text{if } 7 \leq P_{gas} \leq 10 \\
         \frac{(10 - P_{gas}) \times 2.5}{100} & \text{if } P_{gas} > 10
         \end{cases}
      
      Where :math:`P_{gas}` is the gas price in USD/MMBtu.

11. **Progressive Split (Cumulative Production-Based)**:
    
    Adjusts contractor share based on cumulative production in MMBOE:
    
    .. list-table:: Cumulative Production Progressive Split
       :header-rows: 1
       
       * - Cumulative Production (MMBOE)
         - Split Adjustment
       * - 0 - 30
         - +10%
       * - 30 - 60
         - +9%
       * - 60 - 90
         - +8%
       * - 90 - 125
         - +6%
       * - 125 - 175
         - +4%
       * - > 175
         - 0%

**PERMEN ESDM 20/2019 (Variable Split 5 - Simplified)**

Reduces components to 5 key variables:

1. **Field Location**: Same as 8/2017
2. **Reservoir Depth**: Same as 8/2017
3. **Infrastructure Availability**: Same as 8/2017
4. **Reservoir Type**: Same as 8/2017
5. **Field Reserves**: New component based on recoverable reserves

Removes: CO2 content, H2S content (extended ranges), API oil, domestic use, production stage

**PERMEN ESDM 12/2020 (Refinements)**

Similar to 20/2019 with minor adjustments to component weightings and implementation details.

**PERMEN ESDM 13/2024 (Latest Simplified Regulation)**

Further simplifies to 3 variable components:

.. list-table:: Field Location Component (13/2024)
   :header-rows: 1

   * - Location
     - Enum Value
     - Split Adjustment
   * - Onshore
     - ``FieldLocation.ONSHORE``
     - 0%
   * - Shallow Offshore (h < 500m)
     - ``FieldLocation.SHALLOW_OFFSHORE``
     - +8%
   * - Deep Offshore (500 ≤ h ≤ 1000m)
     - ``FieldLocation.DEEP_OFFSHORE``
     - +12%
   * - Ultra-Deep Offshore (h > 1000m)
     - ``FieldLocation.ULTRADEEP_OFFSHORE``
     - +16%

.. list-table:: Infrastructure Availability Component (13/2024)
   :header-rows: 1

   * - Availability Status
     - Enum Value
     - Split Adjustment
   * - Not Available (x = 0%)
     - ``InfrastructureAvailability.NOT_AVAILABLE``
     - +4%
   * - Partially Available (0 < x ≤ 50%)
     - ``InfrastructureAvailability.PARTIALLY_AVAILABLE``
     - +2%
   * - Available (x > 50%)
     - ``InfrastructureAvailability.AVAILABLE``
     - 0%

.. list-table:: Field Reserves Amount Component (13/2024)
   :header-rows: 1

   * - Reserves Amount
     - Enum Value
     - Split Adjustment
   * - Low (< 20 MMBOE)
     - ``FieldReservesAmount.LOW``
     - +3%
   * - Medium (20-60 MMBOE)
     - ``FieldReservesAmount.MEDIUM``
     - +1.5%
   * - High (> 60 MMBOE)
     - ``FieldReservesAmount.HIGH``
     - 0%

Notable Changes in 13/2024:
- Removed ministerial discretion adjustment
- Simplified field location categories
- Introduced field reserves as key variable

3.2.2 Cost Recovery Regulatory Framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While Cost Recovery PSCs are no longer issued for new contracts in Indonesia, pyscnomics maintains support for legacy contracts and transition modeling.

Cost Recovery Fiscal Terms
"""""""""""""""""""""""""""""""

Cost Recovery PSCs operate under these key fiscal mechanisms:

**First Tranche Petroleum (FTP)**

FTP is a priority share of production allocated before cost recovery:

- **Oil FTP Portion**: Default 20% of gross revenue (configurable)
- **Gas FTP Portion**: Default 20% of gross revenue (configurable)
- **FTP Sharing**: Can be shared between contractor and government or allocated entirely to government

**Cost Recovery Cap**

Limits the amount of costs that can be recovered in any period:

- **Oil Cost Recovery Cap**: Default 100% (no cap, configurable)
- **Gas Cost Recovery Cap**: Default 100% (no cap, configurable)
- **Historical caps**: Earlier contracts often had 80-85% caps

**Investment Credit (IC)**

Additional incentive that accelerates contractor recovery:

- **Oil IC Rate**: Default 0% (disabled), configurable up to typical values of 10-20%
- **Gas IC Rate**: Default 0% (disabled), configurable up to typical values of 10-20%

**Equity Split (Pre-Tax)**

Distribution of remaining revenue after FTP and cost recovery:

- **Oil Contractor Share**: Default 25%, configurable
- **Gas Contractor Share**: Default 50%, configurable

**Domestic Market Obligation (DMO)**

Contractor obligation to supply domestic market:

- **Oil DMO Volume Portion**: Default 25% of production
- **Oil DMO Fee Portion**: Default 25% (discount from market price)
- **Oil DMO Holiday**: Default 60 months from onstream date
- **Gas DMO Volume Portion**: Default 100% of production
- **Gas DMO Fee Portion**: Default 100% (typically no discount for gas)
- **Gas DMO Holiday**: Default 60 months from onstream date

3.3 Tax Regimes
---------------

pyscnomics implements corporate income tax regimes as defined by Indonesian tax laws (Undang-Undang or UU).

3.3.1 Tax Regime Enumeration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Tax Regimes in pyscnomics
   :header-rows: 1

   * - Regime Enum
     - Legal Basis
     - Tax Rate
     - Default in pyscnomics
   * - ``UU_36_2008``
     - UU No. 36 Tahun 2008
     - 44%
     - No
   * - ``UU_02_2020``
     - UU No. 02 Tahun 2020
     - 42%
     - No
   * - ``UU_07_2021``
     - UU No. 07 Tahun 2021
     - 40%
     - **Yes**
   * - ``NAILED_DOWN``
     - Contract-Specific
     - Variable
     - No
   * - ``PREVAILING``
     - Current Law
     - Current Rate
     - No

Configuration in pyscnomics
"""""""""""""""""""""""""""""""""

.. code-block:: python

   from pyscnomics.contracts import CostRecovery, GrossSplit
   from pyscnomics.econ.selection import TaxRegime

   # For Cost Recovery contract
   cr_contract = CostRecovery(
       # ... parameters
   )
   cr_contract.tax_regime = TaxRegime.UU_07_2021  # 40% tax rate

   # For Gross Split contract
   gs_contract = GrossSplit(
       # ... parameters
   )
   gs_contract.tax_regime = TaxRegime.UU_07_2021  # 40% tax rate

Tax Regime Details
""""""""""""""""""""""

**UU No. 36 Tahun 2008**

- **Corporate Income Tax Rate**: 44%
- **Effective Period**: 2008 - 2020
- **Historical Context**: Reduced from earlier rates (previously 45%)
- **Usage**: Legacy contracts signed during this period

**UU No. 02 Tahun 2020**

- **Corporate Income Tax Rate**: 42%
- **Effective Period**: 2020 - 2021
- **Context**: Part of broader tax reform initiatives
- **Usage**: Transitional period contracts

**UU No. 07 Tahun 2021 (Default)**

- **Corporate Income Tax Rate**: 40%
- **Effective Period**: 2021 - Present
- **Current Status**: Prevailing rate for all new PSCs
- **pyscnomics Default**: This is the default tax regime if not explicitly specified

**NAILED_DOWN**

- **Tax Rate**: Fixed at contract signing rate
- **Purpose**: Provides tax rate certainty for contract duration
- **Usage**: Some legacy contracts negotiated with fixed rates
- **Configuration**: Rate must be specified in contract terms

**PREVAILING**

- **Tax Rate**: Automatically follows current law
- **Purpose**: Ensures compliance with latest tax regulations
- **Usage**: Contracts that automatically adopt tax law changes
- **Behavior**: pyscnomics applies the most recent tax rate from available regimes

3.3.2 Tax Calculation Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pyscnomics supports two primary tax calculation modes:

Tax Direct Mode
""""""""""""""""""

Tax is calculated directly from taxable income in each period:

.. math::

   \text{Tax}_t = \max(0, \text{TI}_t \times \tau)

Where:

- :math:`\text{TI}_t` = Taxable Income. Contractor revenue minus deductible costs in period t
- :math:`\tau` = Corporate tax rate
- Tax can be zero if taxable income is negative (loss)

Tax Due Mode
""""""""""""""""

Tax is treated as an unrecoverable cost that can be weighted/carried forward:

- Tax liability is calculated when taxable income is positive
- Unpaid tax amounts can be carried forward to future periods
- More conservative approach that reflects tax payment timing

**Default in pyscnomics**: Tax Direct Mode is the default calculation method.

Configuration:

.. code-block:: python

   from pyscnomics.econ.selection import TaxPaymentMode

   contract.tax_payment_mode = TaxPaymentMode.TAX_DIRECT_MODE
   # or
   contract.tax_payment_mode = TaxPaymentMode.TAX_DUE_MODE

3.4 PDJP Regulations (Tax Directorate)
---------------------------------------

The Directorate General of Taxes (DJP) issues regulations (PER-XX/PJ/YYYY) that govern tax implementation details for PSCs.

3.4.1 FTP Tax Regimes
^^^^^^^^^^^^^^^^^^^^^^

PDJP regulations specify how First Tranche Petroleum (FTP) is treated for tax purposes:

Supported FTP Tax Regimes
"""""""""""""""""""""""""""""""""""""""""

.. list-table:: FTP Tax Regimes in pyscnomics
   :header-rows: 1

   * - Regime Enum
     - Regulation
     - FTP Tax Treatment
     - Default in pyscnomics
   * - ``PRE_PDJP_20_2017``
     - Pre-PDJP 2017
     - FTP taxed as government share
     - No
   * - ``PDJP_20_2017``
     - PER-20/PJ/2017
     - FTP may have different tax treatment
     - **Yes**
   * - ``DIRECT_MODE``
     - Direct calculation
     - Tax paid directly without regime-specific rules
     - No

Configuration in pyscnomics
""""""""""""""""""""""""""""""""""

.. code-block:: python

   from pyscnomics.econ.selection import FTPTaxRegime

   contract = CostRecovery(
       # ... parameters
   )
   contract.ftp_tax_regime = FTPTaxRegime.PDJP_20_2017

3.4.2 Tax Treatment of PSC Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different fiscal components have specific tax treatments under PDJP regulations:

**Cost Recovery Tax Treatment**

- Recovered costs reduce taxable income
- Unrecovered costs carried forward
- Depreciation/amortization schedules affect timing

**FTP Tax Treatment**

Under PER-20/PJ/2017:

- FTP allocated to government is not contractor income
- FTP allocated to contractor (if shared) is taxable income
- Tax treatment varies by contract vintage

**Investment Credit Tax Treatment**

- IC is typically treated as reduction to government take
- May have specific tax implications depending on regime
- Not generally treated as taxable income to contractor

3.5 Value Added Tax (VAT) and Land Building Tax (LBT)
------------------------------------------------------

pyscnomics supports indirect taxes that affect project economics.

3.5.1 VAT (Value Added Tax)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VAT applies to goods and services procured for the project:

- **Default VAT Rate**: 11% (as per current Indonesian regulations)
- **Application**: Applied to eligible capital and operating costs
- **Treatment**: Input VAT may be creditable or treated as project cost
- **Configurable**: Users can adjust rates based on specific contract terms

Configuration:

.. code-block:: python

   # VAT rate can be configured per cost component
   capital_cost = CapitalCost(
       start_year=2020,
       end_year=2030,
       expense_year=np.array([2020, 2021, 2022]),
       cost=np.array([10_000_000, 5_000_000, 3_000_000]),
       # ... other parameters
   )
   
   # Set VAT discount/exemption if applicable
   contract.vat_discount = 0.0  # No discount (default)
   contract.vat_discount = 0.5  # 50% discount/exemption

3.5.2 LBT (Land and Building Tax)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Local tax on land and building usage:

- **Rate**: Varies by region and land classification
- **Application**: Applied to facility land area and buildings
- **Payment**: Annual obligation
- **Configurable**: Users specify rates based on local regulations

Configuration:

.. code-block:: python

   from pyscnomics.econ.costs import LBT

   lbt = LBT(
       start_year=2020,
       end_year=2030,
       expense_year=np.array([2020]),
       cost=np.array([500_000]),  # Annual LBT cost
       # ... other parameters
   )
   
   # Set LBT discount if applicable
   contract.lbt_discount = 0.0  # No discount (default)

3.6 Default Values in pyscnomics
---------------------------------

When configuring fiscal parameters, pyscnomics uses these default values if not explicitly specified:

**Gross Split Defaults**

.. list-table:: Gross Split Default Parameters
   :header-rows: 1

   * - Parameter
     - Default Value
     - Description
   * - ``base_split_ctr_oil``
     - 0.43 (43%)
     - Base contractor split for oil
   * - ``base_split_ctr_gas``
     - 0.48 (48%)
     - Base contractor split for gas
   * - ``split_ministry_disc``
     - 0.08 (8%)
     - Ministerial discretion adjustment (13/2024: 0%)
   * - ``oil_dmo_volume_portion``
     - 0.25 (25%)
     - Oil DMO volume obligation
   * - ``oil_dmo_fee_portion``
     - 1.0 (100%)
     - Oil DMO fee portion
   * - ``oil_dmo_holiday_duration``
     - 60 months
     - Oil DMO holiday period
   * - ``gas_dmo_volume_portion``
     - 1.0 (100%)
     - Gas DMO volume obligation
   * - ``gas_dmo_fee_portion``
     - 1.0 (100%)
     - Gas DMO fee portion
   * - ``gas_dmo_holiday_duration``
     - 60 months
     - Gas DMO holiday period

**Cost Recovery Defaults**

.. list-table:: Cost Recovery Default Parameters
   :header-rows: 1

   * - Parameter
     - Default Value
     - Description
   * - ``oil_ftp_portion``
     - 0.20 (20%)
     - Oil FTP percentage
   * - ``gas_ftp_portion``
     - 0.20 (20%)
     - Gas FTP percentage
   * - ``oil_ftp_is_available``
     - True
     - Oil FTP enabled
   * - ``oil_ftp_is_shared``
     - True
     - Oil FTP shared with contractor
   * - ``gas_ftp_is_available``
     - True
     - Gas FTP enabled
   * - ``gas_ftp_is_shared``
     - True
     - Gas FTP shared with contractor
   * - ``oil_ctr_pretax_share``
     - 0.25 (25%)
     - Oil contractor pre-tax split
   * - ``gas_ctr_pretax_share``
     - 0.50 (50%)
     - Gas contractor pre-tax split
   * - ``oil_cr_cap_rate``
     - 1.0 (100%)
     - Oil cost recovery cap
   * - ``gas_cr_cap_rate``
     - 1.0 (100%)
     - Gas cost recovery cap
   * - ``oil_ic_rate``
     - 0.0 (0%)
     - Oil investment credit rate (disabled)
   * - ``gas_ic_rate``
     - 0.0 (0%)
     - Gas investment credit rate (disabled)
   * - ``ic_is_available``
     - False
     - Investment credit disabled by default

**Tax and Fiscal Defaults**

.. list-table:: Tax and Fiscal Default Parameters
   :header-rows: 1

   * - Parameter
     - Default Value
     - Description
   * - ``tax_regime``
     - ``TaxRegime.UU_07_2021``
     - Current tax law (40%)
   * - ``tax_payment_mode``
     - ``TaxPaymentMode.TAX_DIRECT_MODE``
     - Direct tax calculation
   * - ``ftp_tax_regime``
     - ``FTPTaxRegime.PDJP_20_2017``
     - Current PDJP regulation
   * - ``vat_rate``
     - 0.11 (11%)
     - Current Indonesian VAT rate
   * - ``vat_discount``
     - 0.0 (0%)
     - No VAT discount by default
   * - ``lbt_discount``
     - 0.0 (0%)
     - No LBT discount by default

3.7 Summary
-----------

pyscnomics provides comprehensive implementation of Indonesian petroleum fiscal regulations:

1. **PERMEN ESDM Regulations**: Full support for all Gross Split regimes (8/2017 through 13/2024) with accurate variable split calculations
2. **Tax Regimes**: All major tax law revisions (UU 36/2008, 02/2020, 07/2021) plus special regimes
3. **PDJP Regulations**: FTP tax treatment according to current and historical regulations
4. **Indirect Taxes**: VAT and LBT with configurable rates and discounts
5. **Defaults**: Conservative defaults based on current regulations (UU 07/2021 tax rate, PERMEN 13/2024 variable split)

When modeling contracts, always verify that the selected regime matches the actual contract terms, as legacy contracts may operate under older regulations.

Next Steps
^^^^^^^^^^

- For depreciation and amortization methods: See :doc:`chapter4_depreciation`
- For economic indicator calculations: See :doc:`chapter5_economic_indicators`
- For Cost Recovery contract details: See :doc:`chapter6_cost_recovery`
- For Gross Split contract details: See :doc:`chapter7_gross_split`
