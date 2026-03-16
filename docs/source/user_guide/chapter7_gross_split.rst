Chapter 7: Gross Split PSC Mechanism
=====================================

This chapter examines the Gross Split PSC, the modern petroleum fiscal regime
introduced by the Indonesian government in 2017 as an alternative mechanism for
oil and gas contracts. While initially mandatory for new contracts, subsequent
regulations (PERMEN ESDM No. 12/2020 and No. 13/2024) established flexibility,
allowing both Gross Split and Cost Recovery schemes to coexist. Contractors can
now choose the scheme that best fits their project characteristics or convert
between schemes.

7.1 Introduction to Gross Split
---------------------------------

In a Gross Split PSC (also called "Gross Split" or "GS"), there is **no cost
recovery mechanism**. Instead, the total revenue (after deducting DMO and other
obligations) is split directly between the contractor and the government based
on a variable split formula.

This approach simplifies the fiscal terms and provides more predictable returns
for contractors, while placing more risk on the contractor's side.

7.1.1 Historical Context
^^^^^^^^^^^^^^^^^^^^^^^^^

The Gross Split PSC was introduced through:

1. **PERMEN ESDM 8/2017**: Initial gross split regulation
2. **PERMEN ESDM 52/2017**: Enhanced with progressive split components
3. **PERMEN ESDM 20/2019**: Variable Split 5 (5 variable components)
4. **PERMEN ESDM 12/2020**: Further refinements
5. **PERMEN ESDM 13/2024**: Latest regulation with simplified variable split

7.1.2 Features Implemented in pyscnomics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pyscnomics implements comprehensive Gross Split support with:

- Multiple regulatory regimes (PERMEN ESDM 8/2017, 52/2017, 20/2019, 12/2020, 13/2024)
- Comprehensive variable split components based on field characteristics
- Progressive split (price-based and cumulative production-based)
- Separate oil and gas handling
- DMO (Domestic Market Obligation) with fee and holiday provisions
- Unit of Production (UOP) amortization for intangible costs

7.2 Key Differences from Cost Recovery
---------------------------------------

The fundamental difference between Cost Recovery and Gross Split:

+-------------------+---------------------------+---------------------------+
| Aspect            | Cost Recovery PSC         | Gross Split PSC           |
+===================+===========================+===========================+
| Cost Recovery     | Yes, with cap limits      | **None**                  |
+-------------------+---------------------------+---------------------------+
| Profit Split      | After cost recovery       | Direct from gross revenue |
+-------------------+---------------------------+---------------------------+
| Risk Profile      | Government shares risk    | Contractor bears more risk|
+-------------------+---------------------------+---------------------------+
| Government Take   | Variable (increases with  | More predictable          |
|                   | profitability)           |                           |
+-------------------+---------------------------+---------------------------+
| Tax Deductions    | Recovered via cost recovery| OPEX & Amortization      |
+-------------------+---------------------------+---------------------------+
| Contractor Share  | CR + FTP + ETS Split     | Revenue × Split %         |
+-------------------+---------------------------+---------------------------+

7.3 Glossary of Gross Split Terms
-----------------------------

.. glossary::
   :sorted:

   Base Split
      The baseline revenue split percentage before applying variable components.
      For oil: typically 43%; for gas: typically 48%.

   Variable Split
      Additional split adjustments based on field characteristics such as
      location, reservoir depth, infrastructure availability, and fluid properties.

   Progressive Split
      Additional split adjustments based on oil/gas prices (price progressive)
      and cumulative production (cumulative progressive).

   Ministerial Discretion
      An additional split adjustment (typically up to 8%) granted at the
      Ministry's discretion based on project specific factors.

   Amortization
      The systematic allocation of intangible costs over time, similar to
      depreciation but for intangible assets. In Gross Split, amortization
      uses the Unit of Production (UOP) method.

   Gross Split Rate
      The effective government share in the gross split, used to calculate
      the contractor's tax-deductible costs.

Gross Split Regimes
-------------------

pyscnomics supports multiple regulatory regimes:

.. list-table:: Supported Gross Split Regimes
   :header-rows: 1

   * - Regime
     - Regulation
     - Variable Components
     - Notes
   * - ``PERMEN_ESDM_8_2017``
     - PERMEN ESDM No. 8 Tahun 2017
     - 9 components
     - Initial gross split regulation
   * - ``PERMEN_ESDM_52_2017``
     - PERMEN ESDM No. 52 Tahun 2017
     - 9 + Progressive
     - Added progressive split
   * - ``PERMEN_ESDM_20_2019``
     - PERMEN ESDM No. 20 Tahun 2019
     - 5 components
     - Simplified to Variable Split 5
   * - ``PERMEN_ESDM_12_2020``
     - PERMEN ESDM No. 12 Tahun 2020
     - 5 components
     - Refinements
   * - ``PERMEN_ESDM_13_2024``
     - PERMEN ESDM No. 13 Tahun 2024
     - 3 components
     - Latest simplified regulation

Base Split
----------

The Base Split is the foundation of the gross split calculation. It represents
the baseline revenue split before applying variable and progressive components:

.. math::

   S_{c,oil}^{\text{base}} = \texttt{base\_split\_ctr\_oil}

   S_{c,gas}^{\text{base}} = \texttt{base\_split\_ctr\_gas}

Where:

- :math:`S_{c,oil}^{\text{base}}` = Contractor base split for oil
- :math:`S_{c,gas}^{\text{base}}` = Contractor base split for gas

pyscnomics Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Base Split Configuration
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``base_split_ctr_oil``
     - float
     - 0.43
     - Base contractor split for oil (43%)
   * - ``base_split_ctr_gas``
     - float
     - 0.48
     - Base contractor split for gas (48%)
   * - ``split_ministry_disc``
     - float
     - 0.08
     - Ministerial discretion adjustment (8%)

Base Split Calculation Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given:

- Base split oil: 43%
- Base split gas: 48%
- Ministerial discretion: 8%

The base contractor share before variable components:

- Oil: 43%
- Gas: 48%

Plus ministerial discretion (if applicable):

- Oil: 43% + 8% = 51%
- Gas: 48% + 8% = 56%

Variable Split Components
------------------------

The Variable Split adjusts the contractor's share based on field characteristics.
pyscnomics implements comprehensive variable split components:

Field Status
^^^^^^^^^^^^

*(PERMEN ESDM 8/2017 and 52/2017)*

.. list-table:: Field Status Component
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

Field Location
^^^^^^^^^^^^^

.. list-table:: Field Location Component
   :header-rows: 1

   * - Location
     - Enum Value (8/2017, 52/2017)
     - Enum Value (13/2024)
     - Split Adjustment
   * - Onshore
     - ``FieldLocation.ONSHORE``
     - ``FieldLocation.ONSHORE``
     - 0%
   * - Offshore 0 < h ≤ 20m
     - ``FieldLocation.OFFSHORE_0_UNTIL_LESSEQUAL_20``
     - N/A
     - +8%
   * - Offshore 20 < h ≤ 50m
     - ``FieldLocation.OFFSHORE_20_UNTIL_LESSEQUAL_50``
     - N/A
     - +10%
   * - Offshore 50 < h ≤ 150m
     - ``FieldLocation.OFFSHORE_50_UNTIL_LESSEQUAL_150``
     - N/A
     - +12%
   * - Offshore 150 < h ≤ 1000m
     - ``FieldLocation.OFFSHORE_150_UNTIL_LESSEQUAL_1000``
     - N/A
     - +14%
   * - Offshore h > 1000m
     - ``FieldLocation.OFFSHORE_GREATERTHAN_1000``
     - N/A
     - +16%
   * - Shallow Offshore (h < 500m)
     - N/A
     - ``FieldLocation.SHALLOW_OFFSHORE``
     - +8%
   * - Deep Offshore (500 ≤ h ≤ 1000m)
     - N/A
     - ``FieldLocation.DEEP_OFFSHORE``
     - +12%
   * - Ultra-Deep Offshore (h > 1000m)
     - N/A
     - ``FieldLocation.ULTRADEEP_OFFSHORE``
     - +16%

Reservoir Depth
^^^^^^^^^^^^^^^

.. list-table:: Reservoir Depth Component
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

Infrastructure Availability
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Infrastructure Availability Component
   :header-rows: 1

   * - Status (8/2017, 52/2017)
     - Status (13/2024)
     - Split Adjustment
   * - Well Developed
     - N/A
     - 0%
   * - New Frontier Offshore
     - N/A
     - +2%
   * - New Frontier Onshore
     - N/A
     - +2%
   * - Not Available (x = 0%)
     - ``InfrastructureAvailability.NOT_AVAILABLE``
     - +4%
   * - Partially Available (0 < x ≤ 50%)
     - ``InfrastructureAvailability.PARTIALLY_AVAILABLE``
     - +2%
   * - Available (x > 50%)
     - ``InfrastructureAvailability.AVAILABLE``
     - 0%

Reservoir Type
^^^^^^^^^^^^^^

.. list-table:: Reservoir Type Component
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

CO2 Content
^^^^^^^^^^^

*(PERMEN ESDM 52/2017 and later)*

.. list-table:: CO2 Content Component
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

H2S Content
^^^^^^^^^^^

*(PERMEN ESDM 52/2017 and later)*

.. list-table:: H2S Content Component
   :header-rows: 1

   * - H2S Content (ppm)
     - Enum Value (52/2017)
     - Enum Value (20/2019)
     - Split Adjustment
   * - < 100
     - ``H2SContent.LESSTHAN_100``
     - ``H2SContent.LESSTHAN_100``
     - 0%
   * - 100 ≤ x < 1000
     - ``H2SContent.EQUAL_100_UNTIL_LESSTHAN_1000``
     - ``H2SContent.EQUAL_100_UNTIL_LESSTHAN_1000``
     - +1%
   * - 1000 ≤ x < 2000
     - ``H2SContent.EQUAL_1000_UNTIL_LESSTHAN_2000``
     - ``H2SContent.EQUAL_1000_UNTIL_LESSTHAN_2000``
     - +2%
   * - 2000 ≤ x < 3000
     - N/A
     - ``H2SContent.EQUAL_2000_UNTIL_LESSTHAN_3000``
     - +3%
   * - 3000 ≤ x < 4000
     - N/A
     - ``H2SContent.EQUAL_3000_UNTIL_LESSTHAN_4000``
     - +4%
   * - x ≥ 4000
     - N/A
     - ``H2SContent.EQUALGREATERTHAN_4000``
     - +5%

API Oil Gravity
^^^^^^^^^^^^^^^

.. list-table:: API Oil Gravity Component
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

Domestic Use
^^^^^^^^^^^^

.. list-table:: Domestic Use Component
   :header-rows: 1

   * - Domestic Use
     - Enum Value (8/2017)
     - Enum Value (52/2017, 20/2019)
     - Split Adjustment
   * - < 30%
     - ``DomesticUse.LESSTHAN_30``
     - N/A
     - 0%
   * - 30% ≤ x < 50%
     - ``DomesticUse.EQUAL_30_UNTIL_LESSTHAN_50``
     - N/A
     - +2%
   * - 50% ≤ x < 70%
     - ``DomesticUse.EQUAL_50_UNTIL_LESSTHAN_70``
     - ``DomesticUse.EQUAL_50_UNTIL_LESSTHAN_70``
     - +3%
   * - 70% ≤ x < 100%
     - ``DomesticUse.EQUAL_70_UNTIL_LESSTHAN_100``
     - ``DomesticUse.EQUAL_70_UNTIL_LESSTHAN_100``
     - +4%

Production Stage
^^^^^^^^^^^^^^^

.. list-table:: Production Stage Component
   :header-rows: 1

   * - Stage
     - Enum Value
     - Split Adjustment
   * - Primary
     - ``ProductionStage.PRIMARY``
     - 0%
   * - Secondary
     - ``ProductionStage.SECONDARY``
     - +3% to +6%
   * - Tertiary
     - ``ProductionStage.TERTIARY``
     - +5% to +10%

Field Reserves (PERMEN 13/2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Field Reserves Component (13/2024)
   :header-rows: 1

   * - Reserves (MMBOE)
     - Enum Value
     - Split Adjustment
   * - Low (x < 20)
     - ``FieldReservesAmount.LOW``
     - +3%
   * - Medium (20 ≤ x ≤ 60)
     - ``FieldReservesAmount.MEDIUM``
     - +1.5%
   * - High (x > 60)
     - ``FieldReservesAmount.HIGH``
     - 0%

Variable Split Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The total variable split is calculated as:

.. math::

   \text{VS} = \sum_{i} V_i

Where :math:`V_i` represents each variable component based on the selected regime.

pyscnomics Implementation:

.. code-block:: python

   from pyscnomics.contracts import GrossSplit
   from pyscnomics.econ.selection import (
       VariableSplit522017,
       VariableSplit082017,
       VariableSplit132024,
   )

   # Example with PERMEN 52/2017 components
   contract = GrossSplit(
       field_status=VariableSplit522017.FieldStatus.POD_I,           # +5%
       field_loc=VariableSplit522017.FieldLocation.OFFSHORE_20_50,   # +10%
       res_depth=VariableSplit522017.ReservoirDepth.GREATERTHAN_2500, # +1%
       infra_avail=VariableSplit522017.InfrastructureAvailability.WELL_DEVELOPED,  # 0%
       res_type=VariableSplit522017.ReservoirType.CONVENTIONAL,      # 0%
       co2_content=VariableSplit522017.CO2Content.LESSTHAN_5,         # 0%
       h2s_content=VariableSplit522017.H2SContent.LESSTHAN_100,       # 0%
       api_oil=VariableSplit522017.APIOil.GREATERTHAN_25,            # 0%
       domestic_use=VariableSplit522017.DomesticUse.EQUAL_50_70,      # +3%
       prod_stage=VariableSplit522017.ProductionStage.SECONDARY,     # +3-6%
   )
   # Total variable split ≈ +22% to +25%

Progressive Split Components
----------------------------

*(PERMEN ESDM 52/2017 and later)*

The Progressive Split provides additional contractor share adjustments based on
oil/gas prices and cumulative production.

Price Progressive Split
^^^^^^^^^^^^^^^^^^^^^^^

For Oil:

.. math::

   PS_{oil} = \frac{(85 - P_{oil}) \times 0.25}{100}

For Gas:

.. math::

   PS_{gas} = 
   \begin{cases}
   \frac{(7 - P_{gas}) \times 2.5}{100} & \text{if } P_{gas} < 7 \\
   0 & \text{if } 7 \leq P_{gas} \leq 10 \\
   \frac{(10 - P_{gas}) \times 2.5}{100} & \text{if } P_{gas} > 10
   \end{cases}

Where:

- :math:`P_{oil}` = Oil price in USD/barrel
- :math:`P_{gas}` = Gas price in USD/MMBtu

**Interpretation:**

- Higher oil prices → Lower contractor share (government captures upside)
- Lower gas prices (< 7) → Higher contractor share (government supports)
- Mid-range gas prices (7-10) → No adjustment
- Higher gas prices (> 10) → Lower contractor share

Cumulative Progressive Split
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Based on cumulative production in BOE:

.. list-table:: Cumulative Progressive Split
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

Total Progressive Split
^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \text{PS}_{total} = \text{PS}_{price} + \text{PS}_{cumulative}

Final Split Calculation
------------------------

The final gross split is calculated as:

.. math::

   S_{c,oil} &= S_{c,oil}^{\text{base}} + \text{VS}_{oil} + \text{PS}_{oil} + \text{MS}_{oil} \\
   S_{c,gas} &= S_{c,gas}^{\text{base}} + \text{VS}_{gas} + \text{PS}_{gas} + \text{MS}_{gas} \\
   S_g &= 100\% - S_c

Where:

- :math:`S_c` = Contractor total split percentage
- :math:`S_g` = Government total split percentage
- :math:`\text{VS}` = Variable split adjustment
- :math:`\text{PS}` = Progressive split adjustment
- :math:`\text{MS}` = Ministerial split adjustment

pyscnomics calculates this separately for oil and gas.

Gross Split Calculation Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Gross Split calculation follows these steps:

1. **Revenue Calculation**: Calculate gross revenue from production × prices
2. **DMO Application**: Deduct DMO portion (or set to 0 if within holiday period)
3. **Base Split**: Apply base split percentages (43% oil, 48% gas)
4. **Variable Split**: Add adjustments based on field characteristics
5. **Progressive Split**: Add price-based and production-based adjustments
6. **Ministerial Discretion**: Add any ministry-granted adjustments
7. **Revenue Split**: Calculate contractor and government shares
8. **Tax Calculation**: Calculate taxable income and apply tax rate
9. **Net Cash Flow**: Final contractor cash flow after tax

Revenue Calculation
-------------------

Gross Revenue:

.. math::

   R_t &= R_{oil,t} + R_{gas,t} \\
   R_{oil,t} &= Q_{oil,t} \times P_{oil,t} \\
   R_{gas,t} &= Q_{gas,t} \times P_{gas,t}

DMO Deduction:

.. math::

   R_{split,t} = R_t \times (1 - \rho_{\text{DMO}})

pyscnomics Configuration Parameters for DMO:

.. list-table:: DMO Configuration
   :header-rows: 1

   * - Parameter
     - Symbol
     - Type
     - Default
     - Description
   * - ``oil_dmo_volume_portion``
     - :math:`\rho_{\text{DMO,oil}}`
     - float
     - 0.25
     - Portion of oil for domestic market
   * - ``oil_dmo_fee_portion``
     - :math:`\phi_{\text{DMO,oil}}`
     - float
     - 1.0
     - Fee portion for oil DMO
   * - ``oil_dmo_holiday_duration``
     - :math:`m_{oil}`
     - int
     - 60
     - Holiday duration in months
   * - ``gas_dmo_volume_portion``
     - :math:`\rho_{\text{DMO,gas}}`
     - float
     - 1.0
     - Portion of gas for domestic market
   * - ``gas_dmo_fee_portion``
     - :math:`\phi_{\text{DMO,gas}}`
     - float
     - 1.0
     - Fee portion for gas DMO
   * - ``gas_dmo_holiday_duration``
     - :math:`m_{gas}`
     - int
     - 60
     - Holiday duration in months

DMO Calculation:

.. math::

   V_{\text{DMO},oil,t} &= Q_{oil,t} \times \rho_{\text{DMO,oil}} \\
   \text{DMO}_{\text{fee},oil,t} &= V_{\text{DMO},oil,t} \times (P_{market,t} - P_{DMO,t}) \times \phi_{\text{DMO,oil}} \\
   V_{\text{DMO},t} &= 0 \quad \text{if} \quad t \leq m

Revenue Split:

.. math::

   \text{CS}_{c,t} &= R_{split,t} \times S_c \\
   \text{GS}_{g,t} &= R_{split,t} \times (1 - S_c)

Amortization (UOP Method)
--------------------------

Unlike Cost Recovery where depreciation is used, Gross Split uses **amortization**
for intangible costs based on Unit of Production:

.. math::

   \text{Amortization}_t = \frac{Q_t}{Q_{total}} \times \texttt{Intangible Cost} \times 2

pyscnomics Implementation:

.. code-block:: python

   from pyscnomics.econ.depreciation import unit_of_production_rate

   amortization = unit_of_production_rate(
       start_year_project=2020,
       cost=20_000_000,  # Intangible costs
       prod=np.array([1000, 950, 900, ...]),  # Production profile
       prod_year=np.array([2020, 2021, 2022, ...]),
       salvage_value=0,
       amortization_len=20
   )

Tax Calculation
---------------

Under Gross Split, OPEX, Amortization, and other deductible costs are
tax-deductible:

.. math::

   \text{TI}_t &= \text{CS}_{c,t} - \text{OPEX}_t - \text{Amortization}_t - \text{ASR}_t - \text{LBT}_t - \text{DMO}_{\text{fee},t} \\
   \text{Tax}_t &= \max(0, \text{TI}_t \times \tau)

Where:

- :math:`\tau` = Corporate tax rate (40% under UU 07/2021)

pyscnomics supports the following tax regimes:

.. list-table:: Tax Regimes for Gross Split
   :header-rows: 1

   * - Regime
     - Tax Rate
     - Description
   * - ``UU_36_2008``
     - 44%
     - Tax regime under UU No.36 Tahun 2008
   * - ``UU_02_2020``
     - 42%
     - Tax regime under UU No.02 Tahun 2020
   * - ``UU_07_2021``
     - 40%
     - Tax regime under UU No.07 Tahun 2021 (prevailing)
   * - ``NAILED_DOWN``
     - Varies
     - Fixed rate based on contract signing year

Carry-Forward Costs
-------------------

pyscnomics supports carry-forward deductible costs (similar to unrecovered costs
in Cost Recovery):

.. math::

   D_{\text{carried},t} &= D_{t-1} - \min(D_{t-1}, \text{CS}_{c,t-1}) \\
   D_{\text{total},t} &= D_t + D_{\text{carried},t} \\
   \text{TI}_t &= \text{CS}_{c,t} - \min(D_{\text{total},t}, \text{CS}_{c,t})

Where:

- :math:`D_t` = Deductible costs in year t
- :math:`D_{\text{carried},t}` = Carried forward deductible costs at year t
- :math:`D_{\text{total},t}` = Total deductible costs (current + carried forward)
- :math:`\text{TI}_t` = Taxable Income in year t
- :math:`\text{CS}_{c,t}` = Contractor revenue in year t

Net Contractor Cash Flow
-------------------------

The final net cash flow to the contractor:

.. math::

   \text{CF}_{c,t} = \text{CS}_{c,t} - \text{Tax}_t

Or equivalently:

.. math::

   \text{CF}_{c,t} = R_{split,t} \times S_c - \tau \times \max(0, R_{split,t} \times S_c - D_t)

pyscnomics calculates this as:

.. code-block:: python

   cashflow = contractor_revenue - tax_payment

Using pyscnomics for Gross Split
---------------------------------

Complete Example:

.. code-block:: python

   import numpy as np
   from datetime import date
   from pyscnomics.contracts import GrossSplit
   from pyscnomics.econ.selection import (
       GrossSplitRegime,
       VariableSplit522017,
       TaxRegime,
       DeprMethod,
       FluidType,
   )
   from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX
   from pyscnomics.econ.revenue import Lifting

   # Define production profiles
   project_years = np.arange(2022, 2042)
   
   oil_lifting = Lifting(
       start_year=2022,
       end_year=2041,
       prod_year=project_years,
       lifting_rate=np.array([8000] * 20),
       price=np.array([80.0] * 20),
       fluid_type=FluidType.OIL,
   )

   gas_lifting = Lifting(
       start_year=2022,
       end_year=2041,
       prod_year=project_years,
       lifting_rate=np.array([40000] * 20),
       price=np.array([10.0] * 20),
       fluid_type=FluidType.GAS,
   )

   # Define intangible costs (amortized via UOP)
   intangible_cost = Intangible(
       start_year=2022,
       end_year=2041,
       expense_year=np.array([2022]),
       cost=np.array([15_000_000]),
       cost_allocation=[FluidType.OIL],
   )

   # Define OPEX
   opex = OPEX(
       start_year=2022,
       end_year=2041,
       expense_year=np.array([2023]),
       fixed_cost=np.array([3_000_000]),
       cost_allocation=[FluidType.OIL],
   )

   # Create Gross Split contract
   contract = GrossSplit(
       start_date=date(2022, 1, 1),
       end_date=date(2041, 12, 31),
       oil_onstream_date=date(2023, 1, 1),
       gas_onstream_date=date(2023, 1, 1),
       lifting=(oil_lifting, gas_lifting),
       intangible_cost=(intangible_cost,),
       opex=(opex,),
       
       # Base split
       base_split_ctr_oil=0.43,
       base_split_ctr_gas=0.48,
       split_ministry_disc=0.08,
       
       # Variable split components (PERMEN 52/2017)
       field_status=VariableSplit522017.FieldStatus.POD_I,           # +5%
       field_loc=VariableSplit522017.FieldLocation.OFFSHORE_20_50, # +10%
       res_depth=VariableSplit522017.ReservoirDepth.LESSEQUAL_2500, # 0%
       infra_avail=VariableSplit522017.InfrastructureAvailability.WELL_DEVELOPED,  # 0%
       res_type=VariableSplit522017.ReservoirType.CONVENTIONAL,     # 0%
       co2_content=VariableSplit522017.CO2Content.LESSTHAN_5,      # 0%
       h2s_content=VariableSplit522017.H2SContent.LESSTHAN_100,    # 0%
       api_oil=VariableSplit522017.APIOil.GREATERTHAN_25,         # 0%
       domestic_use=VariableSplit522017.DomesticUse.EQUAL_50_70,    # +3%
       prod_stage=VariableSplit522017.ProductionStage.SECONDARY,  # +3-6%
       
       # DMO
       oil_dmo_volume_portion=0.25,
       oil_dmo_fee_portion=1.0,
       oil_dmo_holiday_duration=60,
       gas_dmo_volume_portion=1.0,
       gas_dmo_fee_portion=1.0,
       gas_dmo_holiday_duration=60,
   )

   # Set regime and tax
   contract.gross_split_regime = GrossSplitRegime.PERMEN_ESDM_52_2017
   contract.tax_regime = TaxRegime.UU_07_2021

   # Run calculation
   contract.run(
       depr_method=DeprMethod.UOP,
       inflation_rate=0.02,
   )

   # Access results using get_table
   from pyscnomics.tools.table import get_table
   table_oil, table_gas, table_consolidated = get_table(contract)

   print(f"Total Revenue: ${table_consolidated['Revenue'].sum():,.2f}")
   print(f"Contractor Split (Oil): {table_oil['Contractor_Share'].iloc[-1] / table_oil['Revenue'].iloc[-1]:.2%}")
   print(f"Contractor Split (Gas): {table_gas['Contractor_Share'].iloc[-1] / table_gas['Revenue'].iloc[-1]:.2%}")
   print(f"Total Contractor Revenue: ${table_consolidated['Contractor_Net_Share'].sum():,.2f}")
   print(f"Total Tax Paid: ${table_consolidated['Tax_Payment'].sum():,.2f}")
   print(f"Net Cash Flow: ${table_consolidated['Cashflow'].sum():,.2f}")

   # Get detailed results
   print(table_consolidated)

PERMEN ESDM 13/2024 (Latest Regulation)
---------------------------------------

The newest regulation simplifies the variable split:

.. code-block:: python

   import numpy as np
   from datetime import date
   from pyscnomics.contracts import GrossSplit
   from pyscnomics.econ.selection import (
       GrossSplitRegime,
       VariableSplit132024,
       TaxRegime,
   )

   # Simplified variable split for 2024
   contract = GrossSplit(
       start_date=date(2024, 1, 1),
       end_date=date(2043, 12, 31),
       oil_onstream_date=date(2025, 1, 1),
       gas_onstream_date=date(2025, 1, 1),
       
       # Base split (no ministerial discretion in 2024)
       base_split_ctr_oil=0.43,
       base_split_ctr_gas=0.48,
       split_ministry_disc=0.0,
       
       # Simplified variable split (3 components)
       field_loc=VariableSplit132024.FieldLocation.SHALLOW_OFFSHORE,
       infra_avail=VariableSplit132024.InfrastructureAvailability.AVAILABLE,
       field_reserves=VariableSplit132024.FieldReservesAmount.HIGH,
       
       # DMO
       oil_dmo_volume_portion=0.25,
       oil_dmo_fee_portion=1.0,
   )

   contract.gross_split_regime = GrossSplitRegime.PERMEN_ESDM_13_2024
   contract.tax_regime = TaxRegime.UU_07_2021

   contract.run(depr_method=DeprMethod.UOP, inflation_rate=0.0)

Common Pitfalls and Troubleshooting
----------------------------------

1. **Split Exceeds 100%**
   
   The sum of base + variable + progressive + ministerial may exceed 100%.
   pyscnomics will issue a warning:
   
   .. code-block:: python
   
      warnings.warn("Contractor split exceeds 100%")

2. **Negative Progressive Split**
   
   At high oil prices, the price progressive split can become negative.
   The total split is clipped to ensure minimum government share.

3. **UOP Amortization Issues**
   
   Ensure production profile is properly aligned with intangible cost years:
   
   .. code-block:: python
   
      from pyscnomics.tools.table import get_table
   
      # Get results using get_table
      table_oil, table_gas, table_consolidated = get_table(contract)
      
      # Check amortization schedule
      print(table_consolidated['Amortization'])

4. **DMO Holiday Not Applied**
   
   The holiday is calculated from the onstream date:
   
   .. code-block:: python
   
      # 60 months = 5 years from onstream
      contract.oil_dmo_holiday_duration = 60