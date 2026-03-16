Mathematical Notation Conventions
==================================

This chapter establishes the mathematical notation conventions used throughout
the pyscnomics documentation. Consistent notation ensures clarity and prevents
ambiguity when working with complex petroleum economic calculations.

General Conventions
--------------------

Variables and Symbols
^^^^^^^^^^^^^^^^^^^^^^

**Scalar Variables**

- Lowercase italic letters represent scalar variables: :math:`t`, :math:`i`, :math:`n`
- Uppercase italic letters represent scalar constants or totals: :math:`T`, :math:`N`, :math:`R`
- Greek letters represent parameters or ratios: :math:`\alpha`, :math:`\beta`, :math:`\gamma`, :math:`\tau`

**Time Indexing**

The subscript :math:`t` denotes the time period (typically year):

- :math:`X_t` = Value of variable X in year :math:`t`
- :math:`t = 0` represents the base year or project start
- :math:`t = 1, 2, 3, \ldots, T` represents subsequent years
- :math:`T` represents the final year of the project

**Fluid Type Subscripts**

When distinguishing between oil and gas:

- :math:`X_{oil}` or :math:`X_o` = Value for oil
- :math:`X_{gas}` or :math:`X_g` = Value for gas
- :math:`f` = Fluid type index (oil, gas, sulfur, electricity, CO2)

**Production by Fluid Type:**

When production is expressed by fluid type:

.. math::

   R_t = \sum_{f} Q_{f,t} \cdot P_{f,t}

Where:
- :math:`Q_{f,t}` = Production of fluid type f in year t
- :math:`P_{f,t}` = Price of fluid type f in year t
- :math:`f \in \{oil, gas, sulfur, electricity, CO_2\}`

**Party Subscripts**

When distinguishing between parties:

- :math:`X_c` or :math:`X_{ctr}` = Contractor's portion
- :math:`X_g` or :math:`X_{gov}` = Government's portion

Functions and Operators
^^^^^^^^^^^^^^^^^^^^^^^^

**Mathematical Functions**

.. list-table:: Mathematical Functions
   :header-rows: 1

   * - Notation
     - Description
     - Example
   * - :math:`\min(a, b)`
     - Minimum of two values
     - :math:`\min(100, 80) = 80`
   * - :math:`\max(a, b)`
     - Maximum of two values
     - :math:`\max(0, -50) = 0`
   * - :math:`\lceil x \rceil`
     - Ceiling function (round up)
     - :math:`\lceil 4.2 \rceil = 5`
   * - :math:`\lfloor x \rfloor`
     - Floor function (round down)
     - :math:`\lfloor 4.8 \rfloor = 4`
   * - :math:`\sum_{i=1}^{n}`
     - Summation from i=1 to n
     - :math:`\sum_{i=1}^{3} i = 6`
   * - :math:`\prod_{i=1}^{n}`
     - Product from i=1 to n
     - :math:`\prod_{i=1}^{3} i = 6`

**Conditional Expressions**

Piecewise functions use cases notation:

.. math::

   f(x) = \begin{cases}
   a & \text{if condition 1} \\
   b & \text{if condition 2} \\
   c & \text{otherwise}
   \end{cases}

Financial and Economic Notation
---------------------------------

Revenue and Production
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Revenue Notation
   :header-rows: 1

   * - Symbol
     - Description
     - Unit
   * - :math:`R_t`
     - Revenue (Gross Revenue) in year t
     - MUSD
   * - :math:`Q_t`
     - Production quantity (lifting) in year t
     - MSTB/year or BSCF/year
   * - :math:`P_t`
     - Price per unit in year t
     - MUSD/MSTB or MUSD/BSCF
   * - :math:`\text{WAP}_t`
     - Weighted Average Price in year t
     - MUSD/unit

**Revenue Formula:**

.. math::

   R_t = Q_t \cdot P_t

Costs
^^^^^^

.. list-table:: Cost Notation
   :header-rows: 1

   * - Symbol
     - Description
     - Unit
   * - :math:`C_{c,t}`
     - Capital costs (tangible) in year t
     - MUSD
   * - :math:`\text{NC}_t`
     - Non-capital costs in year t
     - MUSD
   * - :math:`D_t`
     - Depreciation charge in year t
     - MUSD
   * - :math:`\text{OPEX}_t`
     - Operating expenses in year t
     - MUSD
   * - :math:`\text{IC}_t`
     - Investment Credit in year t
     - MUSD

**Total Operating Costs:**

.. math::

   \text{Costs}_t = D_t + \text{NC}_t = D_t + \text{OPEX}_t + \texttt{Intangible}_t + \text{ASR}_t + \text{LBT}_t

**Expenditure Variants:**

.. list-table:: Expenditure Notation
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`E_{pre-tax,t}`
     - Pre-tax expenditures in year t
   * - :math:`E_{inflated,t}`
     - Inflation-adjusted expenditures in year t
   * - :math:`E_{post-tax,t}`
     - Post-tax expenditures in year t
   * - :math:`\text{SC}_t`
     - Sunk costs in year t (costs incurred before project start)
   * - :math:`\text{POC}_t`
     - Pre-onstream costs in year t (costs before production begins)

**Cash Flow:**

.. math::

   \text{CF}_t = R_t - \text{SC}_t - \text{POC}_t - E_{post-tax,t}

PSC-Specific Notation
----------------------

First Tranche Petroleum (FTP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: FTP Notation
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`\text{FTP}_t`
     - Total FTP in year t
   * - :math:`\text{FTP}_{c,t}`
     - Contractor's share of FTP in year t
   * - :math:`\text{FTP}_{g,t}`
     - Government's share of FTP in year t
   * - :math:`S_c`
     - Contractor's split ratio
   * - :math:`\alpha_{ctr}`
     - Contractor's pre-tax share (alternative notation)

**FTP Split Formula:**

.. math::

   \text{FTP}_{c,t} = \text{FTP}_t \cdot S_c

Cost Recovery
^^^^^^^^^^^^^^

.. list-table:: Cost Recovery Notation
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`\text{CR}_t`
     - Cost Recovery in year t
   * - :math:`\text{UC}_t`
     - Unrecovered Costs at end of year t
   * - :math:`\text{CTR}_t`
     - Cost to be Recovered in year t
   * - :math:`\gamma_{CR}`
     - Cost Recovery cap rate (typically 1.0)

**Cost Recovery Formula:**

.. math::

   \text{CR}_t = \min\left(R_t - \text{FTP}_t - \text{IC}_t,\ (D_t + \text{NC}_t + \text{CTR}_t) \cdot \gamma_{CR}\right)

Equity and Profit Split
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Equity Notation
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`\text{ETS}_t`
     - Equity to be Split in year t
   * - :math:`\text{CS}_t`
     - Contractor Share in year t
   * - :math:`\text{GS}_t`
     - Government Share in year t
   * - :math:`S_c`
     - Contractor split ratio

**Equity Split Formula:**

.. math::

   \text{CS}_t = \text{ETS}_t \cdot S_c

Taxation
^^^^^^^^^

.. list-table:: Tax Notation
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`\text{TI}_t`
     - Taxable Income in year t
   * - :math:`\text{Tax}_t`
     - Tax Payment in year t
   * - :math:`\tau`
     - Tax rate (e.g., 0.40 for 40%)
   * - :math:`\text{PPh}`
     - Corporate Income Tax (CIT) rate
   * - :math:`\text{PBDR}`
     - Branch Profit Tax (BPT) rate
   * - :math:`D_t`
     - Deductible costs in year t (Gross Split)
   * - :math:`D_{\text{carried},t}`
     - Carried forward deductible costs at year t
   * - :math:`D_{\text{total},t}`
     - Total deductible costs (current + carried forward) in year t

**Basic Tax Formula:**

.. math::

   \text{Tax}_t = \text{TI}_t \cdot \tau

**Contractor Tax with PBDR:**

.. math::

   \text{Effective Tax Rate} = (1 - \text{PPh}) \cdot \text{PBDR} + \text{PPh}

**Indirect Taxes:**

.. math::

   \text{Tax}_{indirect,t} = E_{taxable,t} \cdot \tau_{indirect}

Where:
- :math:`\tau_{indirect}` = Indirect tax rate (VAT, import duties)

DMO (Domestic Market Obligation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: DMO Notation
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`V_{\text{DMO},t}`
     - DMO volume in year t
   * - :math:`V_{\text{DMO},oil,t}`
     - Oil DMO volume in year t
   * - :math:`V_{\text{DMO},gas,t}`
     - Gas DMO volume in year t
   * - :math:`\text{DMO}_{\text{fee},t}`
     - DMO fee payment in year t
   * - :math:`\text{DMO}_{\text{fee},oil,t}`
     - Oil DMO fee payment in year t
   * - :math:`\text{DMO}_{\text{fee},gas,t}`
     - Gas DMO fee payment in year t
   * - :math:`\text{DDMO}_t`
     - Difference DMO in year t
   * - :math:`\rho_{\text{DMO}}`
     - DMO volume portion (percentage of production subject to DMO)
   * - :math:`\rho_{\text{DMO,oil}}`
     - Oil DMO volume portion
   * - :math:`\rho_{\text{DMO,gas}}`
     - Gas DMO volume portion
   * - :math:`\phi_{\text{DMO}}`
     - DMO fee portion (discount applied to DMO price)
   * - :math:`\phi_{\text{DMO,oil}}`
     - Oil DMO fee portion
   * - :math:`\phi_{\text{DMO,gas}}`
     - Gas DMO fee portion
   * - :math:`m`
     - Month when DMO holiday ends
   * - :math:`m_{oil}`
     - Month when oil DMO holiday ends
   * - :math:`m_{gas}`
     - Month when gas DMO holiday ends
   * - :math:`R_{split,t}`
     - Revenue after DMO deduction in year t

**DMO Difference Formula:**

.. math::

   \text{DDMO}_t = P_t \cdot V_{\text{DMO},t} - \text{DMO}_{\text{fee},t}

Contractor and Government Take
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Take Notation
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`\text{CT}_t`
     - Contractor Take in year t
   * - :math:`\text{GT}_t`
     - Government Take in year t
   * - :math:`\text{CF}_{c,t}`
     - Contractor Cash Flow in year t

**Contractor Take:**

.. math::

   \text{CT}_t = \text{CR}_t + \text{FTP}_{c,t} + \text{CS}_t - \text{Tax}_t

**Government Take:**

.. math::

   \text{GT}_t = \text{FTP}_{g,t} + \text{GS}_t + \text{DDMO}_t + \text{Tax}_t

Economic Indicators
--------------------

.. list-table:: Economic Indicator Notation
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`\text{NPV}`
     - Net Present Value
   * - :math:`\text{IRR}`
     - Internal Rate of Return
   * - :math:`\text{POT}`
     - Payout Time
   * - :math:`r`
     - Discount rate (hurdle rate)
   * - :math:`i`
     - Inflation rate
   * - :math:`\text{PI}`
     - Profitability Index

**NPV Formula:**

.. math::

   \text{NPV} = \sum_{t=0}^{T} \frac{\text{CF}_t}{(1 + r)^t}

For detailed information on units and currency conventions used throughout this
documentation, see :doc:`chapter0b_units_currency`.

**Gross Split Specific Notation:**

.. list-table:: Gross Split Notation
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`S_c`
     - Contractor split ratio (also :math:`S_{c,t}` for time-varying)
   * - :math:`S_g`
     - Government split ratio (:math:`S_g = 1 - S_c`)
   * - :math:`S_{c}^{\text{base}}`
     - Contractor base split (before variable/progressive adjustments)
   * - :math:`\text{VS}`
     - Variable split adjustment
   * - :math:`\text{PS}`
     - Progressive split adjustment (price-based and cumulative)
   * - :math:`\text{PS}_{price}`
     - Price progressive split
   * - :math:`\text{PS}_{cumulative}`
     - Cumulative production progressive split
   * - :math:`\text{MS}`
     - Ministerial split adjustment (discretion)
   * - :math:`\text{Amortization}_t`
     - Amortization expense in year t (UOP method)
   * - :math:`R_{split,t}`
     - Revenue after DMO deduction in year t
   * - :math:`V_{\text{DMO},t}`
     - DMO volume in year t
   * - :math:`\text{DMO}_{\text{fee},t}`
     - DMO fee payment in year t
   * - :math:`\rho_{\text{DMO}}`
     - DMO volume portion (percentage of production subject to DMO)
   * - :math:`\phi_{\text{DMO}}`
     - DMO fee portion (discount applied to DMO price)
   * - :math:`m`
     - Month when DMO holiday ends
   * - :math:`D_t`
     - Deductible costs in year t (Gross Split)
   * - :math:`D_{\text{carried},t}`
     - Carried forward deductible costs at year t
   * - :math:`D_{\text{total},t}`
     - Total deductible costs (current + carried forward) in year t

Alphabetical Reference
----------------------

.. list-table:: Notation Quick Reference
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\alpha_{ctr}`
     - Contractor pre-tax share ratio
   * - :math:`\gamma_{CR}`
     - Cost Recovery cap rate
   * - :math:`\gamma_{dep}`
     - Depreciation factor (PSC_DB method)
   * - :math:`\delta`
     - Discount factor
   * - :math:`\tau`
     - Tax rate
   * - :math:`\tau_{indirect}`
     - Indirect tax rate (VAT, import duties)
   * - :math:`C_c`
     - Capital costs (eligible)
   * - :math:`\text{CF}_{c,t}`
     - Contractor Cash Flow
   * - :math:`\text{CR}_t`
     - Cost Recovery
   * - :math:`\text{CS}_t`
     - Contractor Share
   * - :math:`\text{CT}_t`
     - Contractor Take
   * - :math:`\text{CTR}_t`
     - Cost to be Recovered
   * - :math:`D_t`
     - Depreciation charge (Cost Recovery) / Deductible costs (Gross Split)
   * - :math:`\text{DDMO}_t`
     - Difference DMO
   * - :math:`\text{DMO}_{\text{fee},t}`
     - DMO fee payment
   * - :math:`E_{pre-tax,t}`
     - Pre-tax expenditures
   * - :math:`E_{inflated,t}`
     - Inflation-adjusted expenditures
   * - :math:`E_{post-tax,t}`
     - Post-tax expenditures
   * - :math:`\text{ETS}_t`
     - Equity to be Split
   * - :math:`\text{FTP}_t`
     - First Tranche Petroleum
   * - :math:`\text{GS}_t`
     - Government Share
   * - :math:`\text{GT}_t`
     - Government Take
   * - :math:`\text{IC}_t`
     - Investment Credit
   * - :math:`\text{IRR}`
     - Internal Rate of Return
   * - :math:`k`
     - Decline factor (DB depreciation method)
   * - :math:`\text{MS}`
     - Ministerial split adjustment (discretion)
   * - :math:`N`
     - Useful life (years)
   * - :math:`\text{NC}_t`
     - Non-capital costs
   * - :math:`\text{NPV}`
     - Net Present Value
   * - :math:`P_t`
     - Price per unit
   * - :math:`\text{PBDR}`
     - Branch Profit Tax rate
   * - :math:`\text{PPh}`
     - Corporate Income Tax rate
   * - :math:`\text{POT}`
     - Payout Time
   * - :math:`PS`
     - Progressive split adjustment
   * - :math:`Q_t`
     - Production quantity
   * - :math:`r`
     - Discount rate
   * - :math:`R_t`
     - Revenue
   * - :math:`R_{split,t}`
     - Revenue after DMO deduction (Gross Split)
   * - :math:`\text{Amortization}_t`
     - Amortization expense in year t
   * - :math:`S`
     - Salvage value
   * - :math:`S_c`
     - Contractor split ratio
   * - :math:`S_g`
     - Government split ratio
   * - :math:`S_{c}^{\text{base}}`
     - Contractor base split
   * - :math:`\text{SC}_t`
     - Sunk costs
   * - :math:`\text{POC}_t`
     - Pre-onstream costs
   * - :math:`T`
     - Final project year
   * - :math:`\text{TI}_t`
     - Taxable Income
   * - :math:`t`
     - Time index (year)
   * - :math:`\text{UC}_t`
     - Unrecovered Costs
   * - :math:`V_{\text{DMO},t}`
     - DMO volume
   * - :math:`\text{VS}`
     - Variable split adjustment
   * - :math:`\text{WAP}_t`
     - Weighted Average Price

Notation Variations
--------------------

Some notations may appear in different forms throughout the documentation:

**Subscript Variations:**

- :math:`\text{FTP}_{c,t}` ≡ :math:`\text{FTP}_{ctr,t}` (contractor FTP)
- :math:`\text{FTP}_{g,t}` ≡ :math:`\text{FTP}_{gov,t}` (government FTP)
- :math:`S_c` ≡ :math:`\alpha_{ctr}` (contractor split ratio)
- :math:`C_c` ≡ :math:`C_{ctr}` (capital costs)

**Fluid Type Indicators:**

- :math:`X_{oil,t}` ≡ :math:`X_{o,t}` (oil-specific)
- :math:`X_{gas,t}` ≡ :math:`X_{g,t}` (gas-specific)

**Context-Dependent Notation:**

In some contexts, simpler notation is used for clarity:

- :math:`R` instead of :math:`R_t` when the time period is clear from context
- :math:`\text{FTP}` instead of :math:`\text{FTP}_t` for general discussions
- :math:`\text{CR}` instead of :math:`\text{CR}_t` in formulas where time is implicit

Next Steps
-----------

With this notation foundation established, you are ready to proceed with:

- :doc:`chapter1_project_economics` - Petroleum project economics fundamentals
- :doc:`chapter2_psc_fundamentals` - Production Sharing Contract concepts
- :doc:`chapter6_cost_recovery` - Detailed Cost Recovery mechanics (heavy use of this notation)

For API-specific notation and class attributes, refer to the :doc:`../api_reference/index`.
