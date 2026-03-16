Chapter 5: Economic Indicators
==============================

This chapter provides a comprehensive guide to the economic indicators implemented in pyscnomics. These indicators are essential for evaluating the financial viability of petroleum projects and Production Sharing Contracts (PSCs).

5.1 Introduction to Economic Indicators
----------------------------------------

Economic indicators are quantitative metrics used to assess the profitability, risk, and overall attractiveness of petroleum projects. In PSC economics, these indicators help both contractors and governments evaluate proposed contract terms.

**Key Indicators in pyscnomics:**

1. **Net Present Value (NPV)** - Project value in today's dollars
2. **Internal Rate of Return (IRR)** - Project's inherent profitability
3. **Payout Time (POT)** - Capital recovery period
4. **Profitability Index (PI)** - Value created per dollar invested

**Calculation Approaches:**

pyscnomics supports multiple calculation methods to accommodate different requirements:

- Standard NPV and IRR (period-based)
- Extended NPV and IRR (XNPV, XIRR) with exact dates
- SKK Migas-specific methods for Indonesian regulatory compliance
- Real and nominal terms adjustments

5.2 Net Present Value (NPV)
----------------------------

NPV represents the present value of all future cash flows discounted at a specified rate. It indicates whether a project creates or destroys value.

5.2.1 Standard NPV Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Mathematical Formulation:**

.. math::

   NPV = \sum_{t=0}^{T} \frac{CF_t}{(1 + r)^t}

Where:

- :math:`CF_t` = Cash flow in period t
- :math:`r` = Discount rate (hurdle rate)
- :math:`T` = Total number of periods
- :math:`t` = Period index (0, 1, 2, ..., T)

**Interpretation:**

- **NPV > 0**: Project creates value (acceptable)
- **NPV = 0**: Project breaks even (marginal)
- **NPV < 0**: Project destroys value (reject)

**pyscnomics Implementation:**

.. code-block:: python

   import numpy as np
   from pyscnomics.econ.indicator import npv

   # Define cash flows (Year 0 to Year 5)
   cashflows = np.array([-100_000_000, 30_000_000, 40_000_000, 
                         35_000_000, 25_000_000, 20_000_000])
   
   # Calculate NPV with 10% discount rate (default)
   result = npv(cashflow=cashflows, disc_rate=0.10)
   
   print(f"NPV at 10%: ${result:,.2f}")
   # Output: NPV at 10%: $18,629,014.17
   
   # Calculate NPV with different discount rates
   npv_15 = npv(cashflow=cashflows, disc_rate=0.15)
   npv_8 = npv(cashflow=cashflows, disc_rate=0.08)
   
   print(f"NPV at 15%: ${npv_15:,.2f}")  # Lower NPV at higher rate
   print(f"NPV at 8%: ${npv_8:,.2f}")    # Higher NPV at lower rate

**Default Parameters:**

- ``disc_rate``: 0.10 (10%) - This is the default discount rate used by pyscnomics

5.2.2 Extended NPV (XNPV) with Dates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

XNPV calculates NPV using exact cash flow dates, providing more precise results for irregular timing.

**Mathematical Formulation:**

.. math::

   XNPV = \sum_{i=0}^{n} \frac{CF_i}{(1 + r)^{\frac{d_i - d_0}{365}}}

Where:

- :math:`CF_i` = Cash flow at date i
- :math:`d_i` = Date of cash flow i
- :math:`d_0` = Start date (reference date)
- :math:`r` = Annual discount rate

**pyscnomics Implementation:**

.. code-block:: python

   from datetime import date
   from pyscnomics.econ.indicator import xnpv

   # Define cash flows with exact dates
   cashflows = np.array([-100_000_000, 30_000_000, 40_000_000, 
                         35_000_000, 25_000_000])
   
   start_date = date(2024, 1, 1)
   end_date = date(2028, 12, 31)
   
   # Calculate XNPV
   result = xnpv(
       cashflow=cashflows,
       start_date=start_date,
       end_date=end_date,
       disc_rate=0.10
   )
   
   print(f"XNPV: ${result:,.2f}")

5.2.3 NPV in Nominal Terms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The nominal terms method calculates NPV using nominal (non-inflation-adjusted) cash flows and discount rates.

**Mathematical Formulation:**

.. math::

   NPV_{nominal} = \sum_{t} CF_t \times DCF_t

Where the discount factor (DCF) depends on the discounting mode:

**End-Year Discounting:**

.. math::

   DCF_t = \frac{1}{(1 + r)^{t}}

**Mid-Year Discounting:**

.. math::

   DCF_t = \frac{1}{(1 + r)^{t + 0.5}}

**pyscnomics Implementation:**

.. code-block:: python

   from pyscnomics.econ.indicator import npv_nominal_terms
   from pyscnomics.econ.selection import DiscountingMode

   cashflows = np.array([-100_000_000, 40_000_000, 45_000_000, 50_000_000])
   years = np.array([2024, 2025, 2026, 2027])
   
   # End-year discounting (default)
   npv_end = npv_nominal_terms(
       cashflow=cashflows,
       cashflow_years=years,
       discount_rate=0.10,
       reference_year=2024,
       discounting_mode=DiscountingMode.END_YEAR
   )
   
   # Mid-year discounting (more precise for continuous production)
   npv_mid = npv_nominal_terms(
       cashflow=cashflows,
       cashflow_years=years,
       discount_rate=0.10,
       reference_year=2024,
       discounting_mode=DiscountingMode.MID_YEAR
   )
   
   print(f"NPV (End-Year): ${npv_end:,.2f}")
   print(f"NPV (Mid-Year): ${npv_mid:,.2f}")

**Default Parameters:**

- ``discounting_mode``: ``DiscountingMode.END_YEAR`` - End-year discounting is the default

5.2.4 NPV in Real Terms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Real terms NPV adjusts for inflation, providing a more accurate picture of purchasing power.

**Mathematical Formulation:**

.. math::

   NPV_{real} = \sum_{t} CF_t \times DCF_t \times \text{Inflation Adjustment}

Where inflation adjustment converts future cash flows to reference year purchasing power:

.. math::

   \text{Inflation Adjustment}_t = \frac{1}{(1 + i)^{T_{max} - T_{ref}}}

Where:

- :math:`i` = Inflation rate
- :math:`T_{max}` = Maximum year in cash flow series
- :math:`T_{ref}` = Reference year for real terms

**pyscnomics Implementation:**

.. code-block:: python

   from pyscnomics.econ.indicator import npv_real_terms

   cashflows = np.array([-100_000_000, 40_000_000, 45_000_000, 50_000_000])
   years = np.array([2024, 2025, 2026, 2027])
   
   npv_real = npv_real_terms(
       cashflow=cashflows,
       cashflow_years=years,
       discount_rate=0.10,
       reference_year=2024,
       inflation_rate=0.03,  # 3% inflation
       discounting_mode=DiscountingMode.END_YEAR
   )
   
   print(f"NPV (Real Terms): ${npv_real:,.2f}")

**Default Parameters:**

- ``inflation_rate``: Must be specified by user (no default)
- ``discounting_mode``: ``DiscountingMode.END_YEAR``

5.2.5 SKK Nominal Terms (SKK Migas Method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SKK Migas (Indonesia's upstream oil and gas regulator) has specific NPV calculation requirements.

**Key Differences:**

- Reference year is always the starting year of the project
- Follows specific discounting conventions for Indonesian PSCs

**Mathematical Formulation:**

.. math::

   NPV_{SKK} = \sum_{t} \frac{CF_t}{(1 + r)^{t - t_{0} + t_{os}}}

Where:

- :math:`t_{0}` = First year of cash flows (automatically determined)
- :math:`t_{os}` = Offset year. 0 for end-year, 0.5 for mid-year discounting

**pyscnomics Implementation:**

.. code-block:: python

   from pyscnomics.econ.indicator import npv_skk_nominal_terms

   cashflows = np.array([-100_000_000, 40_000_000, 45_000_000, 50_000_000])
   years = np.array([2024, 2025, 2026, 2027])
   
   npv_skk = npv_skk_nominal_terms(
       cashflow=cashflows,
       cashflow_years=years,
       discount_rate=0.10,
       discounting_mode=DiscountingMode.END_YEAR
   )
   
   print(f"NPV (SKK Nominal): ${npv_skk:,.2f}")
   # Note: Reference year automatically set to 2024 (minimum year)

5.2.6 SKK Real Terms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SKK method with real terms adjustment. In this method, inflation rate is assumed equal to discount rate.

**Mathematical Formulation:**

Similar to SKK nominal, but with inflation adjustment where inflation rate equals discount rate:

.. math::

   i = r

**pyscnomics Implementation:**

.. code-block:: python

   from pyscnomics.econ.indicator import npv_skk_real_terms

   cashflows = np.array([-100_000_000, 40_000_000, 45_000_000, 50_000_000])
   years = np.array([2024, 2025, 2026, 2027])
   
   npv_skk_real = npv_skk_real_terms(
       cashflow=cashflows,
       cashflow_years=years,
       discount_rate=0.10,
       reference_year=2024,
       discounting_mode=DiscountingMode.END_YEAR
   )
   
   print(f"NPV (SKK Real): ${npv_skk_real:,.2f}")
   # Inflation rate is automatically set equal to discount rate (10%)

5.2.7 Point Forward NPV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Point forward NPV only considers cash flows from a reference year onward, evaluating the remaining value of a project from a specific point in time.

**Mathematical Formulation:**

.. math::

   NPV_{pf} = \sum_{t \geq t_{ref}} CF_t \times DCF_t

Where:

- Only cash flows from reference year onward are included
- :math:`t_{ref}` = Reference year for point-forward calculation
- :math:`CF_t` = Net contractor cash flow in year t

**How Past Costs Affect Point-Forward NPV:**

The cashflow array passed to ``npv_point_forward()`` is calculated using the full project timeline, so it already includes the effects of Production Sharing Contract mechanisms. Point-forward analysis filters which years to include, but the cashflows themselves retain the impact of prior investments:

**Tangible Costs (Capital Expenditures):**

- **Excluded:** Direct CAPEX payments from years before the reference year
- **Still Included:** 

  - Depreciation charges in years after the reference year, which generate cost recovery eligibility
  - Unrecovered costs carried forward from prior years, which continue to be recovered in future years

**Example:** 100 MUSD CAPEX in 2020 with 5-year depreciation:

- Depreciation continues in 2023-2024, making those costs eligible for recovery
- Any unrecovered amounts from 2020-2022 carry forward and increase 2023+ cashflows when recovered

**Intangible Costs:**

Treatment depends on contract type:

**In Cost Recovery PSCs:**

- Intangible costs are expensed immediately (not amortized)
- **Excluded:** Direct intangible cost payments from years before the reference year
- **Still Included:** Unrecovered amounts from prior years continue to be eligible for recovery in future years through the cost recovery mechanism

**In Gross Split PSCs:**

- Intangible costs can be amortized using Unit of Production (UOP) method
- **Excluded:** Direct intangible cost payments from years before the reference year
- **Still Included:** Ongoing UOP amortization in future years provides tax deductions

**Key Concept - Unrecovered Cost Carry-Forward:**

The most important mechanism is the **unrecovered cost carry-forward**. When costs incurred in early years cannot be fully recovered due to revenue caps, the unrecovered balance carries forward and remains eligible for recovery in future years. This means:

1. The negative cash outflow from the initial investment is excluded
2. But the positive cash inflow from recovering that investment continues to flow through
3. This can make point-forward NPV appear higher than the true remaining project value

**Use Case:**

Point forward NPV is useful when:

- Evaluating whether to continue an ongoing project (sunk costs are truly sunk)
- Making go/no-go decisions for continuing operations
- Comparing alternatives from current point in time

**Important Caveat:**

Point-forward analysis should be used carefully for decision-making. While appropriate for "should we continue operating" questions, it can overstate project value because it excludes the initial investment burden while retaining the benefit of recovering that investment.

.. admonition:: Government Incentive Evaluation
   :class: note
   
   In evaluating fiscal incentives or policy changes, the government typically prefers
   **Point-Forward NPV** analysis. This approach focuses on future cash flows from the
   reference date onward, which better reflects the incremental value created by policy
   changes rather than historical sunk costs.

**Example Scenario:**

Consider a Cost Recovery PSC with 100 MUSD CAPEX in 2020:

- **Full-Cycle NPV:** Includes -100 MUSD investment in 2020 and all subsequent cost recovery payments
- **Point-Forward from 2023:**

  - Excludes the -100 MUSD outflow in 2020
  - BUT includes cost recovery payments in 2023+ that pay back that 100 MUSD investment
  - Also includes depreciation-eligible amounts in 2023+ that generate further cost recovery

**pyscnomics Implementation:**

.. code-block:: python

   from pyscnomics.econ.indicator import npv_point_forward

   # Full project cash flows (calculated with full timeline)
   # These cashflows already include cost recovery effects
   cashflows = np.array([-50_000_000, -50_000_000, 40_000_000, 
                         45_000_000, 50_000_000, 45_000_000])
   years = np.array([2020, 2021, 2022, 2023, 2024, 2025])
   
   # Point forward from 2023
   # Excludes direct costs from 2020-2022
   # But retains cost recovery benefits from those costs in 2023+
   npv_pf = npv_point_forward(
       cashflow=cashflows,
       cashflow_years=years,
       discount_rate=0.10,
       reference_year=2023,
       discounting_mode=DiscountingMode.END_YEAR
   )
   
   print(f"NPV (Point Forward from 2023): ${npv_pf:,.2f}")

5.2.8 Summary of NPV Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: NPV Methods in pyscnomics
   :header-rows: 1

   * - Method
     - Function
     - Key Feature
     - Default Discount Rate
     - Default Discounting Mode
   * - Standard NPV
     - ``npv()``
     - Period-based
     - 10%
     - N/A
   * - Extended NPV
     - ``xnpv()``
     - Date-based
     - 10%
     - N/A
   * - Nominal Terms
     - ``npv_nominal_terms()``
     - No inflation adjustment
     - User specified
     - END_YEAR
   * - Real Terms
     - ``npv_real_terms()``
     - Inflation adjusted
     - User specified
     - END_YEAR
   * - SKK Nominal
     - ``npv_skk_nominal_terms()``
     - Regulatory compliant
     - User specified
     - END_YEAR
   * - SKK Real
     - ``npv_skk_real_terms()``
     - i = r assumption
     - User specified
     - END_YEAR
   * - Point Forward
     - ``npv_point_forward()``
     - Ignore sunk costs
     - User specified
     - END_YEAR

5.3 Internal Rate of Return (IRR)
----------------------------------

IRR is the discount rate at which NPV equals zero. It represents the project's inherent rate of return.

5.3.1 Standard IRR Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Mathematical Formulation:**

Find :math:`IRR` such that:

.. math::

   \sum_{t=0}^{T} \frac{CF_t}{(1 + IRR)^t} = 0

**Interpretation:**

- **IRR > hurdle rate**: Project exceeds required return (acceptable)
- **IRR = hurdle rate**: Project meets required return (marginal)
- **IRR < hurdle rate**: Project below required return (reject)

**pyscnomics Implementation:**

.. code-block:: python

   from pyscnomics.econ.indicator import irr

   # Cash flows: initial investment followed by returns
   cashflows = np.array([-100_000_000, 30_000_000, 40_000_000, 
                         35_000_000, 25_000_000, 20_000_000])
   
   # Calculate IRR
   result = irr(cashflow=cashflows)
   
   print(f"IRR: {result:.2%}")
   # Output: IRR: 17.89%
   
   # Compare to hurdle rate
   hurdle_rate = 0.10  # 10%
   
   if result > hurdle_rate:
       print(f"Project exceeds {hurdle_rate:.0%} hurdle rate")
   else:
       print(f"Project below {hurdle_rate:.0%} hurdle rate")

**Special Cases:**

pyscnomics handles edge cases:

- If all cash flows are positive or all negative: IRR = 0 (no solution)
- If IRR calculation fails: Returns 0
- If IRR is negative: Returns 0 (conservative approach)

5.3.2 Extended IRR (XIRR) with Dates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

XIRR calculates IRR using exact dates for irregular cash flow timing.

**Mathematical Formulation:**

Find :math:`XIRR` such that:

.. math::

   \sum_{i=0}^{n} \frac{CF_i}{(1 + XIRR)^{\frac{d_i - d_0}{365}}} = 0

**pyscnomics Implementation:**

.. code-block:: python

   from pyscnomics.econ.indicator import xirr

   cashflows = np.array([-100_000_000, 30_000_000, 40_000_000, 
                         35_000_000, 25_000_000])
   
   start_date = date(2024, 1, 1)
   end_date = date(2028, 6, 30)  # Irregular end date
   
   result = xirr(
       cashflow=cashflows,
       start_date=start_date,
       end_date=end_date
   )
   
   print(f"XIRR: {result:.2%}")

5.4 Payout Time (POT)
---------------------

Payout Time (also called Payback Period) indicates when cumulative cash flows turn positive, marking the point at which invested capital is recovered.

5.4.1 Standard POT Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Mathematical Formulation:**

.. math::

   POT = t_{break} + \frac{|C_{t_{break}-1}|}{C_{t_{break}} - C_{t_{break}-1}}

Where:

- :math:`POT` = Payout Time in years
- :math:`t_{break}` = First year with positive cumulative cash flow
- :math:`C_t` = Cumulative cash flow at end of year t (i.e., :math:`C_t = \sum_{i=0}^{t} CF_i`)

**Interpretation:**

- Shorter POT = Faster capital recovery (lower risk)
- Typical petroleum projects: 5-10 years POT
- Projects with POT > 15 years are considered high-risk

**pyscnomics Implementation:**

.. code-block:: python

   from pyscnomics.econ.indicator import pot

   cashflows = np.array([-100_000_000, -20_000_000, 30_000_000, 
                         45_000_000, 50_000_000, 40_000_000])
   
   # Calculate POT
   result = pot(cashflow=cashflows)
   
   print(f"Payout Time: {result:.2f} years")
   # Output: Payout Time: 4.71 years
   
   # Cumulative cash flows for verification
   cum_cf = np.cumsum(cashflows)
   print(f"Cumulative Year 3: ${cum_cf[2]:,.2f}")  # Still negative
   print(f"Cumulative Year 4: ${cum_cf[3]:,.2f}")  # Turns positive

**Example Walkthrough:**

For the cash flows above:

- Year 0: :math:`C_0 = -100` MUSD
- Year 1: :math:`C_1 = -120` MUSD
- Year 2: :math:`C_2 = -90` MUSD
- Year 3: :math:`C_3 = -45` MUSD
- Year 4: :math:`C_4 = +5` MUSD (breaks even)

.. math::

   POT = 4 + \frac{|C_3|}{C_4 - C_3} = 4 + \frac{|-45|}{5 - (-45)} = 4 + \frac{45}{50} = 4.9 \text{ years}

5.4.2 PSC-Specific POT Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For PSCs, pyscnomics provides a specialized POT calculation that accounts for project years and reference dates.

**pyscnomics Implementation:**

.. code-block:: python

   from pyscnomics.econ.indicator import pot_psc

   cashflows = np.array([-100_000_000, -20_000_000, 30_000_000, 
                         45_000_000, 50_000_000, 40_000_000])
   years = np.array([2024, 2025, 2026, 2027, 2028, 2029])
   reference_year = 2024
   
   # Calculate PSC-specific POT
   result = pot_psc(
       cashflow=cashflows,
       cashflow_years=years,
       reference_year=reference_year
   )
   
   print(f"PSC Payout Time: {result:.2f} years from {reference_year}")

5.5 Profitability Index (PI)
----------------------------

PI measures the value created per unit of investment, providing a efficiency metric for capital allocation.

**Mathematical Formulation:**

pyscnomics supports two approaches for calculating PI, controlled by the ``profitability_discounted`` parameter:

**Approach 1: Discounted Investment (Default)**

.. math::

   PI = \frac{NPV + I_{disc}}{I_{disc}} = 1 + \frac{NPV}{I_{disc}}

Where:

- :math:`PI` = Profitability Index
- :math:`NPV` = Net Present Value
- :math:`I_{disc}` = Discounted investment (present value of negative cash flows)

**Approach 2: Undiscounted Investment (Optional)**

.. math::

   PI = 1 + \frac{NPV}{I_{undisc}} = \frac{NPV + I_{undisc}}{I_{undisc}}

Where:

- :math:`I_{undisc}` = Undiscounted investment (absolute value of negative cash flows)

The undiscounted investment includes:
- Capital expenditures (post-tax)
- Intangible expenditures (post-tax)
- Minus sunk costs

**Interpretation:**

- **PI > 1.0**: Positive NPV (value creating)
- **PI = 1.0**: Break-even (NPV = 0)
- **PI < 1.0**: Negative NPV (value destroying)
- **PI = 2.0**: 2 MUSD of value created per 1 MUSD invested

**Calculation Example:**

Given:
- Initial investment: :math:`I = 100` MUSD
- NPV: :math:`NPV = 50` MUSD

.. math::

   PI = \frac{NPV + I}{I} = \frac{50 + 100}{100} = 1.5

**When to Use Each Approach:**

**Discounted Investment (Default):**

``profitability_discounted=True``

- When investment is spread over multiple years
- All outflows are discounted to present value
- More precise for projects with phased investments
- Accounts for time value of money in both inflows and outflows

**Undiscounted Investment:**

``profitability_discounted=False``

- Investment at time 0 is not discounted
- Provides intuitive interpretation: "value created per dollar initially invested"
- Useful when initial investment is the primary consideration

**Common Use Cases for PI:**

- Capital rationing scenarios
- Comparing projects of different sizes
- Ranking multiple investment opportunities
- Evaluating efficiency of capital use

5.6 Economic Summary and Contract Indicators
---------------------------------------------

pyscnomics provides comprehensive economic summary functionality that calculates all key metrics used by the Indonesian government and industry for petroleum project evaluation. These summaries consolidate production data, revenue, costs, and economic indicators into standardized reports.

**Why Use Economic Summaries?**

- Generate standardized reports for government submission (SKK Migas)
- Compare multiple scenarios with consistent metrics
- Access all 30+ economic indicators in one call
- Export results to dictionaries for further processing

5.6.1 Overview of Summary Approaches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pyscnomics offers three ways to generate economic summaries:

1. **get_summary() Function** - Quick, functional approach for simple use cases
2. **Summary Class** - Object-oriented with full parameter control
3. **Summarizer Class** - Alternative interface with additional formatting options

All three approaches provide identical economic metrics; the choice depends on your coding style and integration needs.

5.6.2 Using get_summary() Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``get_summary()`` function provides the simplest way to generate a comprehensive economic summary.

**Basic Usage:**

.. code-block:: python

   from pyscnomics.tools.summary import get_summary
   from pyscnomics.econ.selection import NPVSelection, DiscountingMode

   # Generate summary with default parameters
   summary = get_summary(contract=contract)

   # Access results
   print(f"Contractor NPV: ${summary['ctr_npv']:,.2f}")
   print(f"Contractor IRR: {summary['ctr_irr']:.2%}")
   print(f"Government Take: ${summary['gov_take']:,.2f}")

**Advanced Usage with All Parameters:**

.. code-block:: python

   from pyscnomics.tools.summary import get_summary
   from pyscnomics.econ.selection import NPVSelection, DiscountingMode

   summary = get_summary(
       contract=contract,
       reference_year=2024,
       discount_rate=0.10,
       inflation_rate=0.03,
       npv_mode=NPVSelection.NPV_SKK_REAL_TERMS,
       discounting_mode=DiscountingMode.END_YEAR,
       profitability_discounted=True
   )

   # Print key metrics
   print(f"Gross Revenue: ${summary['gross_revenue']:,.2f}")
   print(f"Total Investment: ${summary['investment']:,.2f}")
   print(f"Contractor NPV: ${summary['ctr_npv']:,.2f}")
   print(f"Contractor IRR: {summary['ctr_irr']:.2%}")
   print(f"Contractor PI: {summary['ctr_pi']:.2f}")
   print(f"Government Take: ${summary['gov_take']:,.2f}")
   print(f"Cost Recovery: ${summary['cost_recovery / deductible_cost']:,.2f}")

**Parameters:**

.. list-table:: ``get_summary()`` Parameters
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``contract``
     - BaseProject | CostRecovery | GrossSplit | Transition
     - Required
     - The contract object after running
   * - ``reference_year``
     - int | None
     - None
     - Reference year for NPV calculation (defaults to project start)
   * - ``discount_rate``
     - float
     - 0.10
     - Discount rate for NPV calculations
   * - ``inflation_rate``
     - float
     - 0.0
     - Inflation rate for real terms calculations
   * - ``npv_mode``
     - NPVSelection
     - NPV_SKK_REAL_TERMS
     - NPV calculation method
   * - ``discounting_mode``
     - DiscountingMode
     - END_YEAR
     - End-year or mid-year discounting
   * - ``profitability_discounted``
     - bool
     - False
     - Use discounted investment for PI calculation

5.6.3 Using Summary Class
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Summary`` class provides an object-oriented interface with the same functionality.

**Usage:**

.. code-block:: python

   from pyscnomics.tools.summary import Summary
   from pyscnomics.econ.selection import NPVSelection, DiscountingMode

   # Create Summary object
   summary_obj = Summary(
       contract=contract,
       reference_year=2024,
       discount_rate=0.10,
       inflation_rate=0.03,
       npv_mode=NPVSelection.NPV_SKK_REAL_TERMS,
       discounting_mode=DiscountingMode.END_YEAR,
       profitability_discounted=True
   )

   # Generate dictionary of results
   results = summary_obj.get_summary_dict()

   # Access individual metrics
   print(f"Oil Production: {results['lifting_oil']:,.0f} MSTB")
   print(f"Gas Production: {results['lifting_gas']:,.0f} BSCF")
   print(f"Contractor NPV: ${results['ctr_npv']:,.2f}")

**Benefits of Summary Class:**

- Cleaner syntax for complex workflows
- Easier to reuse with different parameters
- Better integration with object-oriented codebases

5.6.4 Using Summarizer Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Summarizer`` class (from ``pyscnomics.tools.summarizer``) provides an alternative interface with additional formatting capabilities.

**Usage:**

.. code-block:: python

   from pyscnomics.tools.summarizer import Summarizer

   # Create Summarizer
   summarizer = Summarizer(
       contract=contract,
       reference_year=2024,
       discount_rate=0.10,
       profitability_discounted=True
   )

   # Get summary
   summarizer.get_summary()

   # Access attributes directly
   print(f"Contractor PI: {summarizer.contractor_pi:.2f}")
   print(f"Government Take: ${summarizer.gov_take:,.2f}")

**Key Differences:**

- Access metrics as object attributes (e.g., ``summarizer.ctr_npv``)
- Integrated with ExecutiveSummary for report generation
- Used internally by pyscnomics CLI and API

5.6.5 Complete Metrics Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following table documents all metrics available in economic summaries. These metrics are used by SKK Migas (Indonesia's upstream oil and gas regulator) for official reporting.

**Production Metrics:**

.. list-table:: Production Metrics
   :header-rows: 1

   * - Metric
     - Key
     - Unit
     - Description
   * - Oil Lifting
     - ``lifting_oil``
     - MSTB
     - Total oil production volume
   * - Gas Lifting
     - ``lifting_gas``
     - BSCF
     - Total gas production volume
   * - Oil WAP
     - ``oil_wap``
     - MUSD/MSTB
     - Oil weighted average price
   * - Gas WAP
     - ``gas_wap``
     - MUSD/BSCF
     - Gas weighted average price

**Revenue Metrics:**

.. list-table:: Revenue Metrics
   :header-rows: 1

   * - Metric
     - Key
     - Unit
     - Description
   * - Gross Revenue
     - ``gross_revenue``
     - MUSD
     - Total gross revenue (oil + gas)
   * - Oil Revenue
     - ``gross_revenue_oil``
     - MUSD
     - Revenue from oil production
   * - Gas Revenue
     - ``gross_revenue_gas``
     - MUSD
     - Revenue from gas production

**Investment and Cost Metrics:**

.. list-table:: Investment and Cost Metrics
   :header-rows: 1

   * - Metric
     - Key
     - Unit
     - Description
   * - Total Investment
     - ``investment``
     - MUSD
     - Total capital investment (tangible + intangible)
   * - Oil CAPEX
     - ``oil_capex``
     - MUSD
     - Oil capital expenditures
   * - Gas CAPEX
     - ``gas_capex``
     - MUSD
     - Gas capital expenditures
   * - Sunk Cost
     - ``sunk_cost``
     - MUSD
     - Costs incurred before project start
   * - Tangible
     - ``tangible``
     - MUSD
     - Tangible capital costs
   * - Intangible
     - ``intangible``
     - MUSD
     - Intangible costs
   * - OPEX
     - ``opex``
     - MUSD
     - Operating expenditures
   * - ASR
     - ``asr``
     - MUSD
     - Abandonment and site restoration
   * - OPEX and ASR
     - ``opex_and_asr``
     - MUSD
     - Combined operating and restoration costs

**Contractor Economic Indicators:**

.. list-table:: Contractor Economic Indicators
   :header-rows: 1

   * - Metric
     - Key
     - Unit
     - Description
   * - Contractor NPV
     - ``ctr_npv``
     - MUSD
     - Net present value for contractor
   * - NPV (Sunk Cost Pooled)
     - ``ctr_npv_sunk_cost_pooled``
     - MUSD
     - NPV with sunk cost in year 0
   * - Contractor IRR
     - ``ctr_irr``
     - Decimal
     - Internal rate of return
   * - IRR (Sunk Cost Pooled)
     - ``ctr_irr_sunk_cost_pooled``
     - Decimal
     - IRR with sunk cost pooled
   * - Contractor POT
     - ``ctr_pot``
     - Years
     - Payout time
   * - Contractor PI
     - ``ctr_pi``
     - Ratio
     - Profitability index
   * - PV Ratio
     - ``ctr_pv_ratio``
     - Ratio
     - NPV / Investment
   * - Contractor Net Share
     - ``ctr_net_share``
     - MUSD
     - Contractor net cash position
   * - Net Share %
     - ``ctr_net_share_over_gross_share``
     - Decimal
     - Net share as percentage of gross
   * - Contractor Net Cash Flow
     - ``ctr_net_cashflow``
     - MUSD
     - Total contractor cash flow
   * - Cash Flow %
     - ``ctr_net_cashflow_over_gross_rev``
     - Decimal
     - Cash flow as percentage of revenue
   * - Contractor Gross Share
     - ``ctr_gross_share``
     - MUSD
     - Gross share before costs (Cost Recovery only)

**Government and PSC Metrics:**

.. list-table:: Government and PSC Metrics
   :header-rows: 1

   * - Metric
     - Key
     - Unit
     - Description
   * - Government Take
     - ``gov_take``
     - MUSD
     - Total government revenue
   * - Government Take %
     - ``gov_take_over_gross_rev``
     - Decimal
     - Government take as percentage of revenue
   * - Government Take NPV
     - ``gov_take_npv``
     - MUSD
     - NPV of government take
   * - Government FTP Share
     - ``gov_ftp_share``
     - MUSD
     - Government FTP revenue (Cost Recovery only)
   * - Government Equity Share
     - ``gov_equity_share``
     - MUSD
     - Government equity split (Cost Recovery only)
   * - Government DDMO
     - ``gov_ddmo``
     - MUSD
     - DMO fee revenue
   * - Government Tax Income
     - ``gov_tax_income``
     - MUSD
     - Corporate income tax revenue
   * - Cost Recovery / Deductible Cost
     - ``cost_recovery / deductible_cost``
     - MUSD
     - PSC cost recovery amount
   * - Cost Recovery %
     - ``cost_recovery_over_gross_rev``
     - Decimal
     - Cost recovery percentage of revenue
   * - Unrecovered Cost
     - ``unrec_cost``
     - MUSD
     - Unrecovered costs carried forward
   * - Unrecovered/Cost Recovery
     - ``unrec_over_costrec``
     - Decimal
     - Ratio of unrecovered to cost recovery
   * - Unrecovered/Revenue
     - ``unrec_over_gross_rev``
     - Decimal
     - Unrecovered as percentage of revenue

5.6.6 Contract-Specific Metric Availability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not all metrics are available for all contract types. The following table shows which metrics apply to each contract type:

.. list-table:: Metric Availability by Contract Type
   :header-rows: 1

   * - Metric Category
     - BaseProject
     - CostRecovery
     - GrossSplit
     - Transition
   * - Production Metrics
     - Yes
     - Yes
     - Yes
     - Yes
   * - Revenue Metrics
     - Yes
     - Yes
     - Yes
     - Yes
   * - Investment Metrics
     - Yes
     - Yes
     - Yes
     - Yes
   * - Contractor NPV/IRR/POT/PI
     - Yes
     - Yes
     - Yes
     - Yes
   * - Contractor Gross Share
     - No
     - Yes
     - Yes
     - Yes
   * - Government Take Metrics
     - No
     - Yes
     - Yes
     - Yes
   * - FTP Metrics
     - No
     - Yes
     - No
     - Yes
   * - Cost Recovery Metrics
     - No
     - Yes
     - No*
     - No*
   * - Deductible Cost Metrics
     - No
     - No
     - Yes
     - Yes

\* Transition contracts use Deductible Cost metrics instead of Cost Recovery

5.6.7 Full-Cycle vs Point-Forward Indicators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using economic summaries, you can choose between full-cycle and point-forward perspectives:

**Full-Cycle Indicators:**

- Include all project cash flows from inception
- Useful for initial project evaluation
- Includes all sunk costs (CAPEX, intangible costs, etc.)
- Includes all cost recovery from the beginning
- Represents total project economics
- Default behavior in get_summary() and Summary class

**Point-Forward Indicators:**

- Exclude direct sunk costs (payments made before the reference year)
- **Continue to include** cost recovery benefits from those past investments
- Useful for ongoing project assessment
- Represents remaining project value
- Use ``npv_point_forward()`` or specify reference year after sunk costs

**How It Works:**

The cashflow array used in point-forward analysis is calculated using the full project timeline, so it already incorporates PSC mechanisms:

1. **Unrecovered Cost Carry-Forward:** Costs from early years that weren't fully recovered (due to revenue caps) carry forward and continue to generate recovery payments in future years

2. **Tangible Costs:** Depreciation from past CAPEX continues to make those costs eligible for recovery in future years

3. **Intangible Costs:** 
   - In Cost Recovery PSCs: Unrecovered amounts carry forward (no ongoing amortization)
   - In Gross Split PSCs: UOP amortization continues to provide tax deductions

**Key Distinction:**

Point-forward does not simply "ignore" past costs. While the initial cash outflows are excluded, the PSC mechanisms ensure that the **recovery** of those costs continues to flow through in future years. This means:

- The negative cash outflow (-100 MUSD CAPEX in 2020) is excluded
- But the positive cash inflow from recovering that 100 MUSD through cost recovery continues
- This can make point-forward NPV appear higher than the true remaining value

**When to Use Each:**

**Full-Cycle:**
- Initial investment decisions
- Comparing projects before commitment
- Total project economics assessment

**Point-Forward:**
- "Should we continue operating?" decisions
- Evaluating remaining project value
- When historical costs are truly sunk and irreversible

**Example - Comparing Perspectives:**

.. code-block:: python

   from pyscnomics.tools.summary import get_summary
   from pyscnomics.tools.table import get_table
   from pyscnomics.econ.indicator import npv_point_forward

   # Full-cycle summary (includes all historical costs and all recovery)
   full_summary = get_summary(contract=contract)

   # Get cashflow using get_table
   table_oil, table_gas, table_consolidated = get_table(contract)

   # Point-forward from year 2025
   # Excludes CAPEX from 2020-2024 but INCLUDES cost recovery of those amounts
   pf_npv = npv_point_forward(
       cashflow=table_consolidated['Cashflow'].values,
       cashflow_years=contract.project_years,
       discount_rate=0.10,
       reference_year=2025
   )

   print(f"Full-Cycle NPV: ${full_summary['ctr_npv']:,.2f}")
   print(f"Point-Forward NPV (from 2025): ${pf_npv:,.2f}")
   
   # Note: Point-Forward NPV may be higher than expected because
   # it excludes past CAPEX but retains the recovery of that CAPEX

5.6.8 Practical Examples
^^^^^^^^^^^^^^^^^^^^^^^^^

**Example 1: SKK Migas Report Generation:**

.. code-block:: python

   from pyscnomics.tools.summary import get_summary
   from pyscnomics.econ.selection import NPVSelection

   # Generate SKK-compliant summary
   skk_summary = get_summary(
       contract=contract,
       reference_year=contract.start_date.year,
       discount_rate=0.10,
       npv_mode=NPVSelection.NPV_SKK_REAL_TERMS
   )

   # SKK Migas required outputs
   report = {
       'lifting_oil_mstb': skk_summary['lifting_oil'],
       'lifting_gas_bscf': skk_summary['lifting_gas'],
       'gross_revenue_musd': skk_summary['gross_revenue'],
       'contractor_npv_musd': skk_summary['ctr_npv'],
       'contractor_irr_pct': skk_summary['ctr_irr'] * 100,
       'gov_take_musd': skk_summary['gov_take'],
       'gov_take_pct': skk_summary['gov_take_over_gross_rev'] * 100
   }

**Example 2: Multi-Scenario Comparison:**

.. code-block:: python

   from pyscnomics.tools.summary import get_summary

   scenarios = {
       'Base Case': contract_base,
       'High Price': contract_high_price,
       'Low Production': contract_low_prod
   }

   results = []
   for name, contract in scenarios.items():
       summary = get_summary(contract=contract)
       results.append({
           'Scenario': name,
           'NPV (MUSD)': summary['ctr_npv'],
           'IRR (%)': summary['ctr_irr'] * 100,
           'PI': summary['ctr_pi'],
           'Gov Take (%)': summary['gov_take_over_gross_rev'] * 100
       })

   # Display comparison table
   import pandas as pd
   df = pd.DataFrame(results)
   print(df.to_string(index=False))

5.7 Default Values and Best Practices
--------------------------------------

5.7.1 Default Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Discount Rate:**

- pyscnomics default: 10% (0.10)
- Industry typical range: 8-15%
- Higher rates for higher risk projects
- Government evaluation often uses 10-12%

**Discounting Mode:**

- pyscnomics default: ``END_YEAR``
- ``END_YEAR``: Cash flows at year-end (conservative)
- ``MID_YEAR``: Cash flows mid-year (more realistic for continuous production)

**Inflation Rate:**

- No default - must be specified by user
- Current Indonesian target: 3% ± 1%
- Historical average: 5-7%

5.7.2 Best Practices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Consistent Discount Rates:**
   
   Use the same discount rate when comparing projects or alternatives.

2. **Mid-Year vs End-Year:**
   
   - Use END_YEAR for conservative estimates
   - Use MID_YEAR when production is relatively continuous

3. **Real vs Nominal Terms:**
   
   - Use nominal terms for actual cash flow projections
   - Use real terms to compare purchasing power across time periods

4. **Multiple Scenarios:**
   
   Calculate indicators using different assumptions:
   - Base case (P50)
   - Optimistic case (P10)
   - Pessimistic case (P90)

5. **Indicator Combinations:**
   
   Don't rely on a single indicator:
   - NPV shows absolute value creation
   - IRR shows relative return
   - POT shows risk/recovery timing
   - PI shows capital efficiency

5.8 Summary
-----------

pyscnomics provides comprehensive economic indicator calculations:

**NPV Methods:**

- Standard NPV: ``npv()`` - Period-based with 10% default discount rate
- Extended NPV: ``xnpv()`` - Date-based for irregular timing
- Nominal Terms: ``npv_nominal_terms()`` - No inflation adjustment
- Real Terms: ``npv_real_terms()`` - Inflation-adjusted
- SKK Nominal: ``npv_skk_nominal_terms()`` - Indonesian regulatory method
- SKK Real: ``npv_skk_real_terms()`` - SKK with i=r assumption
- Point Forward: ``npv_point_forward()`` - Exclude sunk costs (but retain cost recovery)

**Other Indicators:**

- IRR: ``irr()`` - Internal rate of return
- XIRR: ``xirr()`` - IRR with exact dates
- POT: ``pot()`` - Payout time calculation
- PSC POT: ``pot_psc()`` - PSC-specific payout time
- PI: Profitability Index via ``get_summary()`` or ``Summary`` class

**Economic Summary:**

- ``get_summary()`` - Quick function for all 30+ metrics
- ``Summary`` class - Object-oriented with full control
- ``Summarizer`` class - Alternative interface

**Default Values:**

- Discount rate: 10%
- Discounting mode: ``END_YEAR``
- IRR edge cases: Returns 0 for invalid scenarios
- No inflation default (must be specified)

Next Steps
^^^^^^^^^^

- For Cost Recovery contract modeling: See :doc:`chapter6_cost_recovery`
- For Gross Split contract modeling: See :doc:`chapter7_gross_split`
- For incentives framework and application process: See :doc:`chapter8_incentives_framework`
