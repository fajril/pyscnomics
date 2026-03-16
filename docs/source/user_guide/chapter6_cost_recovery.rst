Chapter 6: Cost Recovery PSC Mechanism
========================================

This chapter provides a comprehensive treatment of the Cost Recovery PSC,
the traditional fiscal regime that dominated Indonesian petroleum contracts
from the 1970s until 2017.

6.1 Introduction to Cost Recovery
----------------------------------

In a Cost Recovery PSC (also called "Cost Recovery" or "CR"), the contractor
first recovers their invested capital and operating costs from the produced
hydrocarbons ("cost recovery"), and then the remaining profit ("profit oil/gas")
is split between the contractor and the government according to agreed percentages.

This mechanism provides the contractor with a degree of cost recovery guarantee,
making it attractive for high-risk exploration activities. However, the
government's take increases as project profitability improves.

6.1.1 Historical Context
^^^^^^^^^^^^^^^^^^^^^^^^^

The Cost Recovery PSC was the dominant fiscal regime in Indonesia from the
1970s until 2017. Under this regime, contractors could recover:

- 100% of legitimate capital investments (through depreciation)
- 100% of operating costs
- Additional incentives like Investment Credit

However, the government retained significant upside through the profit split
mechanism, especially when oil prices were high.

6.1.2 Features Implemented in pyscnomics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pyscnomics implements a comprehensive Cost Recovery model with support for:

- Separate oil and gas handling with independent calculations
- Multiple tax regimes (UU 36/2008 at 44%, UU 02/2020 at 42%, UU 07/2021 at 40%)
- FTP (First Tranche Petroleum) with configurable sharing between contractor/government
- Investment Credit (IC) for qualifying capital expenditures
- DMO (Domestic Market Obligation) with fee and holiday provisions
- Cost Recovery caps for both oil and gas (typically 100%)
- Multiple tax split types (Conventional, Sliding Scale ICP, R2C)
- Cost transfer between oil and gas (cross-asset recovery)
- Multiple FTP tax regimes (PRE_PDJP_20_2017, PDJP_20_2017, DIRECT_MODE)

6.2 The Cost Recovery Calculation Flow
---------------------------------------

Understanding the Cost Recovery mechanism requires following the sequence of
calculations performed annually. The following diagram illustrates the
complete flow:

.. mermaid::

   flowchart TD
       A[Gross Revenue] --> B[FTP Allocation]
       B --> C[Investment Credit]
       C --> D[Cost Recovery]
       D --> E[Equity to be Shared]
       E --> F[Profit Split]
       F --> G[DMO Calculation]
       G --> H[Tax Calculation]
       H --> I[Contractor Take]
       
       style A fill:#e1f5ff
       style I fill:#d4edda

The calculation is performed annually, with each year's results depending on
the unrecovered costs carried forward from previous years.

6.3 Revenue Calculation
------------------------

Before diving into the cost recovery mechanism, it's essential to understand how
revenue is calculated in pyscnomics.

6.3.1 Gross Revenue (R)
^^^^^^^^^^^^^^^^^^^^^^^

The total revenue from selling all produced hydrocarbons:

.. math::

   R_t = R_{oil,t} + R_{gas,t}

   R_{oil,t} = Q_{oil,t} \times P_{oil,t}

   R_{gas,t} = Q_{gas,t} \times P_{gas,t}

Where:

- :math:`R_t` = Total gross revenue in year t
- :math:`R_{oil,t}` = Oil revenue in year t
- :math:`R_{gas,t}` = Gas revenue in year t
- :math:`Q_{oil,t}` = Oil lifting (production sold) in year t
- :math:`P_{oil,t}` = Oil weighted average price (WAP) in year t
- :math:`Q_{gas,t}` = Gas lifting (production sold) in year t
- :math:`P_{gas,t}` = Gas weighted average price (WAP) in year t

**In pyscnomics**, revenue is calculated through the Lifting class:

.. code-block:: python

   from pyscnomics.econ.revenue import Lifting
   from pyscnomics.econ.selection import FluidType

   oil_lifting = Lifting(
       start_year=2020,
       end_year=2040,
       prod_year=np.array([2020, 2021, ...]),
       lifting_rate=np.array([10000, 9500, ...]),  # barrels per day
       price=np.array([75.0, 72.0, ...]),  # USD per barrel
       fluid_type=FluidType.OIL,
   )

   revenue = oil_lifting.revenue()  # Returns revenue array

6.3.2 Weighted Average Price (WAP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When multiple lifting transactions occur with different prices in the same year,
pyscnomics calculates a volume-weighted average price:

.. math::

   \text{WAP}_{fluid} = \frac{\sum_{i} (V_i \times P_i)}{\sum_{i} V_i}

This ensures revenue accurately reflects the mix of lifting transactions.

6.4 First Tranche Petroleum (FTP)
----------------------------------

Before cost recovery, a portion of total petroleum is allocated as FTP, which
is split directly between contractor and government. FTP serves as the government's
priority revenue stream.

.. admonition:: Contract Reference: PSC Section 6.4
   :class: info
   
   **First Tranche Petroleum**
   
   "Notwithstanding anything to the contrary elsewhere contained in this CONTRACT, 
   SKK MIGAS and CONTRACTOR shall be entitled to first take and receive each Year 
   a quantity of Petroleum twenty percent (20%) of the Petroleum production of each 
   such Year, called the 'First Tranche Petroleum', before any deduction for recovery 
   of Operating Costs..."

6.4.1 FTP Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pyscnomics provides these FTP configuration options:

.. list-table:: FTP Configuration Parameters
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``oil_ftp_is_available``
     - bool
     - True
     - Whether FTP applies to oil
   * - ``oil_ftp_is_shared``
     - bool
     - True
     - Whether FTP is shared with contractor
   * - ``oil_ftp_portion``
     - float
     - 0.2
     - FTP portion (20% typical)
   * - ``gas_ftp_is_available``
     - bool
     - True
     - Whether FTP applies to gas
   * - ``gas_ftp_is_shared``
     - bool
     - True
     - Whether FTP is shared with contractor
   * - ``gas_ftp_portion``
     - float
     - 0.2
     - FTP portion for gas

6.4.2 FTP Calculation Formula
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Step 1: Calculate Total FTP**

The total FTP is calculated as a portion of gross revenue:

.. math::

   \text{FTP}_{oil,t} = R_{oil,t} \times \text{ftp\_portion}_{oil}

   \text{FTP}_{gas,t} = R_{gas,t} \times \text{ftp\_portion}_{gas}

**Step 2: Split FTP Between Parties**

.. admonition:: Contract Reference: PSC Section 6.4.2
   :class: info
   
   "Such FTP for each Calendar Year is shared for Crude Oil between SKK MIGAS 
   and CONTRACTOR in accordance with the sharing splits provided under 
   paragraph 6.2.3."

When FTP is shared, it's allocated based on the contractor's pre-tax share:

.. math::

   \text{FTP}_{ctr,oil,t} = \text{FTP}_{oil,t} \times \alpha_{oil}

   \text{FTP}_{gov,oil,t} = \text{FTP}_{oil,t} \times (1 - \alpha_{oil})

Where:

- :math:`\alpha_{oil}` = Contractor pre-tax share for oil (e.g., 0.25 or 25%)
- :math:`\text{FTP}_{ctr,oil,t}` = Contractor's portion of FTP
- :math:`\text{FTP}_{gov,oil,t}` = Government's portion of FTP

**Alternative: Non-Shared FTP**

When FTP is not shared (``oil_ftp_is_shared = False``):

.. math::

   \text{FTP}_{ctr,oil,t} = 0

   \text{FTP}_{gov,oil,t} = \text{FTP}_{oil,t}

.. admonition:: Important Note: Post-2008 FTP Treatment
   :class: warning
   
   For contracts signed after 2008, an important clarification applies:
   
   "Operating Cost shall not be recovered from CONTRACTOR share of FTP. 
   The CONTRACTOR share from FTP is exempt from cost recovery. For avoidance 
   of doubt the CONTRACTOR share of FTP is subject to Indonesia Income Tax Law."
   
   This means:
   - Contractor's FTP share does NOT reduce unrecovered costs
   - Contractor's FTP share IS subject to income tax
   - This can create a paradox where IC actually reduces contractor take (see Section 6.5.3)

**Practical Example:**

Given:

- Oil Revenue: 100,000 MUSD
- FTP Portion: 20%
- Contractor Pre-tax Share: 25%

Calculation:

- Total FTP = 100,000 MUSD × 0.20 = 20,000 MUSD
- Contractor FTP = 20,000 MUSD × 0.25 = 5,000 MUSD
- Government FTP = 20,000 MUSD × 0.75 = 15,000 MUSD

6.5 Investment Credit (IC)
---------------------------

Investment Credit provides additional compensation for capital investments,
particularly for deepwater or challenging environments. The IC allows the
contractor to recover a percentage of capital costs directly from revenue
before the regular cost recovery calculation.

.. admonition:: Contract Reference: PSC Exhibit C, Article II
   :class: info
   
   "CONTRACTOR may recover an investment credit amounting to 17% of the 
   capital investment costs directly required for developing Crude Oil 
   production facilities... of each new field Operating Costs, commencing 
   in the earliest production Year or Years before tax deduction..."

6.5.1 IC Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: IC Configuration Parameters
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``ic_is_available``
     - bool
     - False
     - Whether IC is available
   * - ``oil_ic_rate``
     - float
     - 0.0
     - IC rate for oil capital costs
   * - ``gas_ic_rate``
     - float
     - 0.0
     - IC rate for gas capital costs

6.5.2 IC Calculation Formula
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

IC is calculated on qualifying capital expenditures:

.. math::

   \text{IC}_t = \text{IC}_{\text{rate}} \cdot C_{c,t}

Where:

- :math:`\text{IC}_{\text{rate}}` = Investment Credit rate specified in contract (typically 10-20%)
- :math:`C_{c,t}` = Eligible capital costs (investment capital costs) in year t
- Only capital costs where ``is_ic_applied = True`` are eligible

In pyscnomics, IC eligibility is set per cost item:

.. code-block:: python

   capital_cost = CapitalCost(
       start_year=2020,
       end_year=2030,
       expense_year=np.array([2020, 2021]),
       cost=np.array([50_000_000, 30_000_000]),
       is_ic_applied=[True, False],  # Only first item eligible for IC
       # ... other parameters
   )

6.5.3 IC Recovery Logic and Unrecovered IC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike regular cost recovery, IC has its own unrecovered tracking mechanism:

**Step 1: Calculate Unrecovered IC**

.. math::

   \text{IC}_{\text{unrec}, t} = \sum_{i=1}^{t} \text{IC}_i - \sum_{i=1}^{t} (R_i - \text{FTP}_i)

   \text{IC}_{\text{unrec}, t} = \max(0, \text{IC}_{\text{unrec}, t})

**Step 2: Calculate IC Paid**

.. math::

   \text{IC}_{\text{paid}, t} = \min(R_t - \text{FTP}_t, \text{IC}_t + \text{IC}_{\text{unrec}, t-1})

This ensures IC doesn't exceed available revenue after FTP.

6.5.4 The IC Paradox: When IC Reduces Contractor Take
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Critical Consideration
   :class: warning
   
   Investment Credit can paradoxically reduce the contractor's net take when
   operating costs exceed revenue. This occurs because:
   1. IC reduces the amount available for regular cost recovery
   2. IC is subject to tax even when there's no profit
   3. The tax on IC can exceed the benefit of accelerated recovery

**Example: IC Impact on Contractor Take**

Consider a contract with:

- Revenue: **100 MUSD**
- FTP (20%): **20 MUSD**
- Operating Costs: **80 MUSD**
- IC Rate: 10% on **100 MUSD** capital = **10 MUSD** IC
- Tax Rate: 44%
- After-tax split: 85% contractor / 15% government

.. list-table:: Impact of Investment Credit
   :header-rows: 1

   * - Description
     - Without IC
     - With IC
   * - Revenue
     - 100 MUSD
     - 100 MUSD
   * - FTP (20%)
     - 20 MUSD
     - 20 MUSD
   * - IC
     - 0 MUSD
     - 10 MUSD
   * - Cost Recovery (CR)
     - 80 MUSD
     - 70 MUSD
   * - Contractor FTP Share (25%)
     - 5 MUSD
     - 5 MUSD
   * - Contractor Share (after-tax)
     - 5 MUSD
     - 15 MUSD
   * - Tax
     - 2.2 MUSD
     - 6.6 MUSD
   * - **Contractor Take (CR + FTP - Tax)**
     - **82.8 MUSD**
     - **78.4 MUSD**

In this scenario, the contractor actually receives **4.4 MUSD LESS** with IC enabled!
This occurs because:

- IC reduces cost recovery capacity by **10 MUSD**
- The **10 MUSD** IC is taxed at 44% = **4.4 MUSD** additional tax
- Net effect: **-10 MUSD** + **5.357 MUSD** (additional share) - **4.4 MUSD** (tax) = **-9.043 MUSD**

.. admonition:: Recommendation
   :class: tip
   
   For contracts with IC provisions, contractors should consider:
   1. Requesting delayed IC application until revenue exceeds operating costs
   2. Evaluating whether IC provides net benefit given tax implications
   3. Negotiating IC timing provisions in the contract

6.6 Cost Classification in PSC
-------------------------------

Understanding cost classification is essential for accurate cost recovery
calculations. PSC contracts categorize costs into specific groups with
different recovery treatments.

6.6.1 Operating Costs Definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Contract Reference: PSC Exhibit C, Article II
   :class: info
   
   "For any Year in which commercial production occurs, Operating Costs consist of:
   
   (a) current Year Non Capital Costs;
   (b) current Year depreciation for Capital Costs;
   (c) current Year allowed recovery of prior Years' unrecovered Operating Costs;
   (d) current Year allowed recovery of prior Years' unrecovered Operating Costs 
       with respect to Exploratory Expenditures..."

6.6.2 Non-Capital Costs
^^^^^^^^^^^^^^^^^^^^^^^^

Non-Capital Costs are fully recovered in the year they are incurred:

**Primary Categories:**

1. **Operations**: Direct field operating expenses
2. **Office, Services, and General Administration**: Overhead costs
3. **Production Services**: Support services for production
4. **Exploratory Expenditures** (if not previously recovered):
   - Exploratory drilling
   - Data acquisition
5. **Training**: Personnel training for Indonesian nationals

**Additional Operating Costs:**

- **Overhead Allocation**: Corporate overhead allocated to project
- **Gas Costs**: Specific gas-related operating costs
- **Inventory Accounting**: Materials and supplies
- **Insurance**: Property and liability insurance
- **Claims**: Insurance and contractual claims
- **Abandonment and Site Restoration (ASR)**: Future abandonment obligations

In pyscnomics, Non-Capital Costs include:

.. math::

   \text{NC}_t = \texttt{Intangible}_t + \text{OPEX}_t + \text{ASR}_t + \text{LBT}_t

Where:
- **Intangible**: Geophysical, drilling permits, and other intangible costs
- **OPEX**: Operating expenses (fixed + variable components)
- **ASR**: Abandonment and Site Restoration provisions
- **LBT**: Land and Building Tax

6.6.3 Capital Costs (Tangible Assets)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Capital Costs are recovered through depreciation over the asset's useful life.

**Primary Categories:**

1. **Construction Utilities and Auxiliaries**: Supporting infrastructure
2. **Construction Housing and Welfare**: Worker facilities
3. **Production Facilities**: Processing and treatment equipment
4. **Movables**: Equipment and tools
5. **Development Wells**: Production wells drilling costs

.. admonition:: Contract Reference: PSC Exhibit C, Article III
   :class: info
   
   **Depreciation Groups and Useful Lives:**
   
   **GROUP 1** (50% depreciation factor) - 5 Year Useful Life:
   - Automobile: 1.5 years
   - Trucks-light (≤13,000 lbs)
   - Trucks-heavy (>13,000 lbs)
   - Aircraft
   - Construction Equipment
   - Furniture and Office Equipment
   
   **GROUP 2** (25% depreciation factor) - 10 Year Useful Life:
   - Construction utilities and auxiliaries
   - Platform and Storage Plant
   - Construction housing and welfare
   - Production facilities
   - Railroad cars and locomotives
   - Vessels, barges, tugs
   - Drilling and production tools, equipment and instruments

6.7 Depreciation Methods
-------------------------

Capital costs are recovered through depreciation. pyscnomics supports multiple
depreciation methods as detailed in :doc:`chapter4_depreciation`.

6.7.1 PSC Declining Balance Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The standard PSC depreciation method uses declining balance:

.. math::

   D_t = D_f \times C \times (1 - D_f)^{t-1}

Where:
- :math:`D_t` = Depreciation charge in year t
- :math:`D_f` = Depreciation factor (0.50 for Group 1, 0.25 for Group 2)
- :math:`C` = Initial capital cost
- :math:`t` = Year index

**Final Year Adjustment:**

At the end of the asset's useful life, the remaining book value is fully depreciated:

.. math::

   D_{t=\lceil \text{ul} \rceil} = C_c - \sum_{t=1}^{\lceil \text{ul} - 1 \rceil} D_t

Where:
- :math:`\lceil \text{ul} \rceil` denotes the ceiling of useful life (rounded up to nearest integer)
- :math:`D_{t=\lceil \text{ul} \rceil}` is the depreciation charge in the final year of useful life

**Example Depreciation Schedule:**

Consider a Capital Cost of 100,000 MUSD with:
- Depreciation Factor: 25%
- Useful Life: 5 years

.. list-table:: Depreciation Schedule Example
   :header-rows: 1

   * - Year
     - Formula
     - Depreciation Charge
     - Remaining Book Value
   * - 1
     - :math:`0.25 \times 100,000`
     - $25,000.00
     - $75,000.00
   * - 2
     - :math:`0.25 \times 75,000`
     - $18,750.00
     - $56,250.00
   * - 3
     - :math:`0.25 \times 56,250`
     - $14,062.50
     - $42,187.50
   * - 4
     - :math:`0.25 \times 42,187.50`
     - $10,546.88
     - $31,640.62
   * - 5 (Final)
     - Remaining balance
     - $31,640.62
     - $0.00
   * - **Total**
     -
     - **$100,000.00**
     -

.. admonition:: Important Note
   :class: warning
   
   If the PSC contract ends before the depreciation period is complete, the
   remaining undepreciated costs cannot be recovered unless specifically
   agreed with the Government.

6.8 Unrecovered Costs and Cost to be Recovered
-----------------------------------------------

6.8.1 Unrecovered Costs (UC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When eligible costs exceed available cost recovery in a given year, the
difference becomes Unrecovered Costs:

**Calculation:**

.. math::

   \text{UC}_t = \max\left(0, \sum_{i=1}^{t}(D_i + \text{NC}_i) - \sum_{i=1}^{t}(R_i - \text{FTP}_i - \text{IC}_i)\right)

Where:
- :math:`D_i` = Depreciation in year i
- :math:`\text{NC}_i` = Non-capital costs in year i
- :math:`\text{UC}_t` is always ≥ 0

In pyscnomics implementation:

.. code-block:: python

   unrecovered_cost = np.cumsum(depreciation + non_capital) - np.cumsum(
       revenue - (ftp_ctr + ftp_gov) - ic
   )
   unrecovered_cost = np.where(unrecovered_cost >= 0, unrecovered_cost, 0)

6.8.2 Cost to be Recovered (CTR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Cost to be Recovered represents unrecovered costs from previous years
that are eligible for recovery in the current year:

.. math::

   \text{CTR}_t = \max(0, \text{UC}_{t-1} - \text{UC}_t)

Or equivalently:

.. math::

   \text{CTR}_t = 
   \begin{cases}
   \text{UC}_{t-1} - \text{UC}_t & \text{if } t > 1 \text{ and } \text{UC}_{t-1} > \text{UC}_t \\
   0 & \text{otherwise}
   \end{cases}

In pyscnomics:

.. code-block:: python

   def get_cost_to_be_recovered(unrecovered_cost: np.ndarray) -> np.ndarray:
       ctr = np.concatenate((np.zeros(1), -np.diff(unrecovered_cost)))
       return np.where(ctr > 0, ctr, 0)

6.9 Cost Recovery Calculation
------------------------------

The actual cost recovery is calculated as the minimum of:
1. Available revenue after FTP and IC
2. Eligible costs (depreciation + non-capital + cost to be recovered)

6.9.1 Cost Recovery Formula
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \text{CR}_t = \min\left(R_t - \text{FTP}_t - \text{IC}_t, (D_t + \text{NC}_t + \text{CTR}_t) \times \gamma_{CR}\right)

Where:
- :math:`\text{CR}_t` = Cost Recovery in year t
- :math:`D_t` = Depreciation in year t
- :math:`\text{NC}_t` = Non-capital costs in year t
- :math:`\text{CTR}_t` = Cost to be recovered in year t
- :math:`\gamma_{CR}` = Cost recovery cap rate (default 1.0 = 100%)

.. admonition:: Source Code Implementation
   :class: note
   
   The pyscnomics implementation in ``costrecovery.py``:
   
   .. code-block:: python
   
      @staticmethod
      def _get_cost_recovery(
          revenue: np.ndarray,
          ftp: np.ndarray,
          ic: np.ndarray,
          depreciation: np.ndarray,
          non_capital: np.ndarray,
          cost_to_be_recovered: np.ndarray,
          cr_cap_rate: float,
      ) -> np.ndarray:
          return np.minimum(
              revenue - ftp - ic,
              (depreciation + non_capital + cost_to_be_recovered) * cr_cap_rate,
          )

6.9.2 Cost Recovery Cap
^^^^^^^^^^^^^^^^^^^^^^^^

The cost recovery cap limits the maximum percentage of gross revenue that
can be used for cost recovery:

**Configuration:**

- ``oil_cr_cap_rate`` (default: 1.0 = 100%)
- ``gas_cr_cap_rate`` (default: 1.0 = 100%)

Historical contracts often had caps of 80-85%, but modern contracts
typically allow 100% cost recovery.

6.9.3 Example Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Cost Recovery Example Over 4 Years
   :header-rows: 1

   * - Year
     - Revenue
     - Eligible Costs
     - Max CR
     - CR Taken
     - UC_new
     - UC_cum
   * - 1
      - 100 MUSD
      - 80 MUSD
      - 100 MUSD
      - 80 MUSD
      - 0 MUSD
      - 0 MUSD
    * - 2
      - 80 MUSD
      - 90 MUSD
      - 80 MUSD
      - 80 MUSD
      - 10 MUSD
      - 10 MUSD
    * - 3
      - 80 MUSD
      - 85 MUSD
      - 80 MUSD
      - 80 MUSD
      - 15 MUSD
      - 25 MUSD
    * - 4
      - 100 MUSD
      - 70 MUSD + 25 MUSD CTR
      - 100 MUSD
      - 95 MUSD
      - 0 MUSD
      - 0 MUSD

6.10 Equity to be Shared (ETS)
-------------------------------

After deducting FTP, IC, and Cost Recovery, the remaining revenue is the
Equity to be Shared (ETS), also called "Profit Petroleum":

6.10.1 ETS Calculation
^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \text{ETS}_t = R_t - (\text{FTP}_{ctr,t} + \text{FTP}_{gov,t}) - \text{IC}_t - \text{CR}_t

Or equivalently:

.. math::

   \text{ETS}_t = R_t - \text{FTP}_{total,t} - \text{IC}_t - \text{CR}_t

Where:
- :math:`\text{FTP}_{total,t}` = Total FTP (both contractor and government portions)

In pyscnomics implementation:

.. code-block:: python

   @staticmethod
   def _get_ets_before_transfer(
       revenue: np.ndarray,
       ftp_ctr: np.ndarray,
       ftp_gov: np.ndarray,
       ic: np.ndarray,
       cost_recovery: np.ndarray,
   ) -> np.ndarray:
       result = revenue - (ftp_ctr + ftp_gov) - ic - cost_recovery
       tol = np.full_like(result, fill_value=1.0e-5)
       return np.where(result < tol, 0, result)

.. admonition:: Note
   :class: note
   
   A tolerance of :math:`10^{-5}` is applied: if ETS < :math:`10^{-5}`, 
   it is set to zero to avoid floating-point artifacts.

6.11 Profit Split
-----------------

The ETS is split between contractor and government based on the agreed
pre-tax split ratio.

6.11.1 Split Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Split Configuration Parameters
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``tax_split_type``
     - TaxSplitTypeCR
     - CONVENTIONAL
     - Type of tax split
   * - ``oil_ctr_pretax_share``
     - float
     - 0.25
     - Contractor pre-tax share for oil
   * - ``gas_ctr_pretax_share``
     - float
     - 0.50
     - Contractor pre-tax share for gas

6.11.2 Conventional Split
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Contract Reference: PSC Section 6.2 and 6.3
   :class: info
   
   **Oil Split (Section 6.2.3):**
   "...SKK MIGAS and CONTRACTOR shall be entitled to take and receive each Year, 
   respectively fifty eight point three three three three percent (58.3333%) for 
   SKK MIGAS and forty one point six six six seven percent (41.6667%) for CONTRACTOR."
   
   **Gas Split (Section 6.3.2):**
   "...SKK MIGAS forty one point six six six seven percent (41.6667%) and 
   CONTRACTOR fifty eight point three three three three percent (58.3333%)."

The conventional split uses fixed percentages:

.. math::

   \text{CS}_t = \alpha_{ctr} \times \text{ETS}_t

   \text{GS}_t = (1 - \alpha_{ctr}) \times \text{ETS}_t

Where:
- :math:`\text{CS}_t` = Contractor Share in year t
- :math:`\text{GS}_t` = Government Share in year t
- :math:`\alpha_{ctr}` = Contractor pre-tax share (e.g., 0.416667 for oil)

.. admonition:: Critical Note: Pre-Tax vs After-Tax Split
   :class: warning
   
   The split percentages specified in PSC contracts are **PRE-TAX**, not after-tax.
   
   For example, a contract may specify 85% : 15% after-tax split. To achieve this
   with a 44% tax rate, the pre-tax split would be approximately 41.67% : 58.33%.
   
   **Formula:**
   
   .. math::
   
      \alpha_{\text{after-tax}} = \alpha_{\text{pre-tax}} \times (1 - \tau) + \tau
   
   Where :math:`\tau` is the tax rate.
   
   When tax regulations change, the after-tax split will change even if the
   pre-tax split remains constant. Always use the **pre-tax split** from the
   contract for pyscnomics calculations.

6.11.3 Sliding Scale / R2C Split
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some contracts use variable splits based on the R/C (Revenue/Cost) ratio:

.. math::

   \alpha_{ctr} = f\left(\frac{R}{C}\right)

pyscnomics supports sliding scale through the ``condition_dict`` parameter:

.. code-block:: python

   condition_dict = {
       'RC Bottom Limit': [0, 1.0, 1.2, 1.4],
       'RC Top Limit': [1.0, 1.2, 1.4, 999],
       'Pre Tax CTR Oil': [0.3137260, 0.27451, 0.235295, 0.196078],
       'Pre Tax CTR Gas': [0.5498040, 0.509804, 0.470589, 0.431373],
   }

6.12 Domestic Market Obligation (DMO)
-------------------------------------

The DMO requires the contractor to supply a portion of production to the
domestic market at a discounted price.

.. admonition:: Contract Reference: PSC Section 5.2.19
   :class: info
   
   "CONTRACTOR shall after commercial production commences, fulfill its 
   obligation towards the supply of the domestic market. CONTRACTOR agrees 
   to sell and deliver to GOI a portion of the share of Crude Oil..."

6.12.1 DMO Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: DMO Configuration Parameters
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``oil_dmo_volume_portion``
     - float
     - 0.25
     - Portion of oil for domestic market (25%)
   * - ``oil_dmo_fee_portion``
     - float
     - 0.25
     - Fee portion for oil DMO (25% of market price)
   * - ``oil_dmo_holiday_duration``
     - int
     - 60
     - Holiday duration in months (60 = 5 years)
   * - ``gas_dmo_volume_portion``
     - float
     - 1.0
     - Portion of gas for domestic market (100%)
   * - ``gas_dmo_fee_portion``
     - float
     - 1.0
     - Fee portion for gas DMO (100%)
   * - ``gas_dmo_holiday_duration``
     - int
     - 60
     - Holiday duration in months

6.12.2 DMO Volume Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   V_{\text{DMO},t} = \rho_{\text{DMO}} \times Q_t \times \alpha_{\text{ctr}}

Where:

- :math:`V_{\text{DMO},t}` = DMO volume in year t
- :math:`\rho_{\text{DMO}}` = DMO volume portion (e.g., 0.25 for 25%)
- :math:`Q_t` = Total production in year t
- :math:`\alpha_{\text{ctr}}` = Contractor pre-tax share

6.12.3 DMO Fee Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The DMO fee depends on whether unrecovered costs exist and whether the
contract is in the DMO holiday period:

.. math::

   \text{DMO}_{\text{fee},t} =
   \begin{cases}
   P_t \times V_{\text{DMO},t} & \text{if } \text{UC}_t > 0 \text{ or } t \leq t_{\text{holiday}} \\
   \phi_{\text{DMO}} \times P_t \times V_{\text{DMO},t} & \text{if } \text{UC}_t = 0 \text{ and } t > t_{\text{holiday}}
   \end{cases}

Where:

- :math:`\phi_{\text{DMO}}` = DMO fee portion (discount factor, e.g., 0.25 for 25% of market price)
- Full price applies when unrecovered costs exist or during holiday
- Discounted price applies after holiday when costs are recovered

6.12.4 Difference DMO (DDMO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Difference DMO represents the value of the DMO concession:

.. math::

   \text{DDMO}_t = (P_t \times V_{\text{DMO},t}) - \text{DMO}_{\text{fee},t}

Or equivalently:

.. math::

   \text{DDMO}_t = (1 - \phi_{\text{DMO}}) \times P_t \times V_{\text{DMO},t}

The DDMO is added to government take and represents an indirect tax on the contractor.

6.12.5 DMO Holiday Weighting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the DMO holiday ends mid-year, the DMO fee is prorated:

.. math::

   \text{DMO}_{\text{fee},t} = \frac{m}{12} \times P_t \times V_{\text{DMO},t} + \frac{12-m}{12} \times \phi_{\text{DMO}} \times P_t \times V_{\text{DMO},t}

Where:

- :math:`m` = Month when holiday ends (1-12)
- :math:`\phi_{\text{DMO}}` = DMO fee portion
- First term: Full price for months 1 to m
- Second term: Discounted price for months m+1 to 12

**Example:** If holiday ends on April 30 (month 4):

.. math::

   \text{DMO}_{\text{fee},t} = \frac{4}{12} \times (P \times V) + \frac{8}{12} \times (0.25 \times P \times V)

6.13 Tax Calculation
--------------------

6.13.1 Taxable Income (TI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Taxable Income is calculated as:

.. math::

   \text{TI}_t = \text{FTP}_{ctr,t} + \text{CS}_t - \text{DDMO}_t

Where:
- :math:`\text{FTP}_{ctr,t}` = Contractor's FTP share
- :math:`\text{CS}_t` = Contractor Share (from ETS)
- :math:`\text{DDMO}_t` = Difference DMO

6.13.2 Tax Payment
^^^^^^^^^^^^^^^^^^^

The tax payment depends on the selected tax regime:

**Direct Mode:**

.. math::

   \text{Tax}_t = \text{TI}_t \times \tau

**Pre-PDJP 2017 Mode:**

Tax is applied when cumulative contractor share becomes positive:

.. math::

   \text{Tax}_t = 
   \begin{cases}
   \text{TI}_{\text{cum},t} \times \tau & \text{first positive year} \\
   \text{TI}_t \times \tau & \text{subsequent years} \\
   0 & \text{if no positive TI}
   \end{cases}

**PDJP 2017 Mode:**

More complex calculation considering FTP vs unrecovered costs. See pyscnomics
source code for detailed implementation.

Where:
- :math:`\tau` = Tax rate (0.40 for UU 07/2021, 0.42 for UU 02/2020, 0.44 for UU 36/2008)

**Contractor Tax Formula (with PBDR):**

The complete Contractor Tax calculation includes both Corporate Income Tax (PPh/CIT) and Branch Profit Tax (PBDR/BPT):

.. math::

   \text{Contractor Tax Rate} = (1 - \text{PPh}) \times \text{PBDR} + \text{PPh}

Where:
- :math:`\text{PPh}` = Corporate Income Tax rate (e.g., 0.40 or 40%)
- :math:`\text{PBDR}` = Branch Profit Tax rate (typically 0.20 or 20% after tax holidays)

**Example:** If CIT = 40% and BPT = 20%:

.. math::

   \text{Effective Tax Rate} = (1 - 0.40) \times 0.20 + 0.40 = 0.12 + 0.40 = 0.52 \text{ or } 52\%

.. note::

   The effective tax rate combines both CIT on profits and BPT on post-tax profits 
   remitted to the foreign contractor.

6.14 Contractor and Government Take
------------------------------------

6.14.1 Contractor Take
^^^^^^^^^^^^^^^^^^^^^^^

The contractor's total take consists of:

.. math::

   \text{CT}_t = \text{CR}_t + \text{FTP}_{c,t} + \text{CS}_t - \text{Tax}_t

Or equivalently using Taxable Income:

.. math::

   \text{CT}_t = \text{TI}_t - \text{Tax Payment}_t + \text{CR}_t

Where:
- :math:`\text{CT}_t` = Contractor Take in year t
- :math:`\text{TI}_t` = Taxable Income (includes FTP and Contractor Share)

In cash flow terms:

.. math::

   \text{CF}_{c,t} = \text{CT}_t - C_{c,t} - \text{NC}_t

Where:
- :math:`\text{CF}_{c,t}` = Contractor Cash Flow in year t
- :math:`C_{c,t}` = Contractor capital expenditures in year t
- :math:`\text{NC}_t` = Contractor non-capital expenditures in year t

6.14.2 Government Take
^^^^^^^^^^^^^^^^^^^^^^^

The government's total take consists of:

.. math::

   \text{GT}_t = \text{FTP}_{g,t} + \text{GS}_t + \text{DDMO}_t + \text{Tax Payment}_t

6.14.3 Government Take Indicators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**GT Ratio:**

.. math::

   \text{GT}_{\text{ratio}} = \frac{\sum_{t=1}^{\text{PSC}} \text{GT}_t}{\sum_{t=1}^{\text{PSC}} (R_t - \text{CTR}_t)}

**Government Share of Gross Revenue (GSGR):**

.. math::

   \text{GSGR} = \frac{\sum_{t=1}^{\text{PSC}} \text{GT}_t}{\sum_{t=1}^{\text{PSC}} R_t}

6.15 Full-Cycle vs Point-Forward Analysis
------------------------------------------

When calculating economic indicators (NPV, IRR), two approaches are used:

**Full-Cycle:**
- :math:`t = 0` is set at the start of investment
- Includes all costs from project inception
- Provides complete project economics
- More conservative (higher discounting of early costs)

**Point-Forward:**
- :math:`t = 0` is set when POD (Plan of Development) is approved
- Excludes pre-POD exploration and appraisal costs
- Makes project appear more economic
- Often used for investment decisions after discovery

.. admonition:: Important Consideration
   :class: warning
   
   Point-forward analysis can significantly improve project economics by excluding
   sunk costs. Always clarify which approach is being used when comparing projects.

6.16 Cost Transfer Between Oil and Gas
---------------------------------------

pyscnomics supports transferring unrecovered costs and cost recovery between
oil and gas pools. This is important when one fluid type has insufficient
revenue to recover costs.

The transfer mechanism:

1. Calculate unrecovered costs for oil and gas separately
2. If oil has excess cost recovery capacity and gas has unrecovered costs,
   transfer from oil to gas (and vice versa)
3. Recalculate ETS after transfer

This allows costs incurred in a gas field to be recovered from oil revenue
(if the contract allows), maximizing cost recovery efficiency.

6.17 Complete Calculation Sequence Summary
-------------------------------------------

The complete cost recovery calculation follows these steps:

**Step 1: Calculate Gross Revenue**

.. math::

   R_t = Q_t \times P_t

**Step 2: Calculate FTP**

.. math::

   \text{FTP}_t = R_t \times \text{ftp\_portion}
   \text{FTP}_{ctr,t} = \text{FTP}_t \times \alpha_{ctr}
   \text{FTP}_{gov,t} = \text{FTP}_t \times (1 - \alpha_{ctr})

**Step 3: Calculate Investment Credit**

.. math::

   \text{IC}_t = \sum_{i} C_{i,t} \times \text{ic\_rate}_i

**Step 4: Calculate Depreciation**

.. math::

   D_t = \text{DepreciationMethod}(C, \text{useful\_life}, t)

**Step 5: Calculate Non-Capital Costs**

.. math::

   \text{NC}_t = \texttt{Intangible}_t + \text{OPEX}_t + \text{ASR}_t + \text{LBT}_t

**Step 6: Calculate Unrecovered Costs**

.. math::

   \text{UC}_t = \max\left(0, \sum_{i=1}^{t}(D_i + \text{NC}_i) - \sum_{i=1}^{t}(R_i - \text{FTP}_i - \text{IC}_i)\right)

**Step 7: Calculate Cost to be Recovered**

.. math::

   \text{CTR}_t = \max(0, \text{UC}_{t-1} - \text{UC}_t)

**Step 8: Calculate Cost Recovery**

.. math::

   \text{CR}_t = \min(R_t - \text{FTP}_t - \text{IC}_t, (D_t + \text{NC}_t + \text{CTR}_t) \times \gamma_{CR})

**Step 9: Calculate Equity to be Shared**

.. math::

   \text{ETS}_t = R_t - \text{FTP}_t - \text{IC}_t - \text{CR}_t

**Step 10: Calculate Profit Split**

.. math::

   \text{CS}_t = \text{ETS}_t \times \alpha_{ctr}
   \text{GS}_t = \text{ETS}_t \times (1 - \alpha_{ctr})

**Step 11: Calculate DMO**

.. math::

   V_{\text{DMO},t} = \text{dmo\_portion} \times Q_t \times \alpha_{ctr}
   \text{DMO}_{\text{fee},t} = \text{calculate\_dmo\_fee}(V_{\text{DMO},t}, \text{UC}_t, t)
   \text{DDMO}_t = (P_t \times V_{\text{DMO},t}) - \text{DMO}_{\text{fee},t}

**Step 12: Calculate Tax**

.. math::

   \text{TI}_t = \text{FTP}_{ctr,t} + \text{CS}_t - \text{DDMO}_t
   \text{Tax}_t = \text{TI}_t \times \tau

**Step 13: Calculate Final Takes**

.. math::

   \text{CTR}_{\text{take},t} = \text{CR}_t + \text{FTP}_{ctr,t} + \text{CS}_t - \text{Tax}_t
   \text{GOV}_{\text{take},t} = \text{FTP}_{gov,t} + \text{GS}_t + \text{DDMO}_t + \text{Tax}_t

6.18 Using pyscnomics for Cost Recovery
----------------------------------------

Here's a complete example of creating and running a Cost Recovery contract:

.. code-block:: python

   import numpy as np
   from datetime import date
   from pyscnomics.contracts import CostRecovery
   from pyscnomics.econ.selection import (
       TaxSplitTypeCR, TaxRegime, FTPTaxRegime, DeprMethod, FluidType
   )
   from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT
   from pyscnomics.econ.revenue import Lifting

   # Define production profiles
   project_years = np.arange(2020, 2041)
   
   oil_lifting = Lifting(
       start_year=2020,
       end_year=2040,
       prod_year=project_years,
       lifting_rate=np.array([10000] * 21),  # barrels per day
       price=np.array([75.0] * 21),  # USD per barrel
       fluid_type=FluidType.OIL,
   )

   gas_lifting = Lifting(
       start_year=2020,
       end_year=2040,
       prod_year=project_years,
       lifting_rate=np.array([50000] * 21),  # mscf per day
       price=np.array([8.0] * 21),  # USD per mscf
       fluid_type=FluidType.GAS,
   )

   # Define capital costs
   capital_cost = CapitalCost(
       start_year=2020,
       end_year=2040,
       expense_year=np.array([2020, 2021, 2022]),
       cost=np.array([50_000_000, 30_000_000, 20_000_000]),
       cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL],
       useful_life=np.array([10, 10, 10]),
       pis_year=np.array([2020, 2021, 2022]),
   )

   # Define OPEX
   opex = OPEX(
       start_year=2020,
       end_year=2040,
       expense_year=np.array([2023]),
       fixed_cost=np.array([5_000_000]),
       cost_allocation=[FluidType.OIL],
   )

   # Create Cost Recovery contract
   contract = CostRecovery(
       start_date=date(2020, 1, 1),
       end_date=date(2040, 12, 31),
       oil_onstream_date=date(2023, 1, 1),
       gas_onstream_date=date(2023, 1, 1),
       lifting=(oil_lifting, gas_lifting),
       capital_cost=(capital_cost,),
       opex=(opex,),
       
       # FTP settings
       oil_ftp_is_available=True,
       oil_ftp_is_shared=True,
       oil_ftp_portion=0.2,
       gas_ftp_is_available=True,
       gas_ftp_is_shared=True,
       gas_ftp_portion=0.2,
       
       # Tax split
       tax_split_type=TaxSplitTypeCR.CONVENTIONAL,
       oil_ctr_pretax_share=0.55,
       gas_ctr_pretax_share=0.45,
       
       # Cost Recovery cap
       oil_cr_cap_rate=1.0,
       gas_cr_cap_rate=1.0,
       
       # DMO
       oil_dmo_volume_portion=0.25,
       oil_dmo_fee_portion=0.25,
       oil_dmo_holiday_duration=60,
       gas_dmo_volume_portion=1.0,
       gas_dmo_fee_portion=1.0,
       gas_dmo_holiday_duration=60,
   )

   # Run the calculation
   contract.run(
       depr_method=DeprMethod.STRAIGHT_LINE,
       inflation_rate=0.0,
       tax_regime=TaxRegime.UU_07_2021,
       ftp_tax_regime=FTPTaxRegime.PRE_PDJP_20_2017,
       vat_rate=0.11,
       lbt_rate=0.0,
   )

   # Access results using get_table
   from pyscnomics.tools.table import get_table
   table_oil, table_gas, table_consolidated = get_table(contract)

   print(f"Total Revenue: ${table_consolidated['Revenue'].sum():,.2f}")
   print(f"Total Cost Recovery: ${table_consolidated['Cost_Recovery'].sum():,.2f}")
   print(f"Total Contractor Take: ${table_consolidated['Contractor_Net_Share'].sum():,.2f}")
   print(f"Total Government Take: ${table_consolidated['Government_Take'].sum():,.2f}")

   # Get detailed results as DataFrame
   print(table_consolidated)

For detailed output, use:

.. code-block:: python

   # Get all results as dictionary using get_table
   table_oil, table_gas, table_consolidated = get_table(contract)
   
   # Access specific arrays from DataFrame
   print(table_consolidated['Cost_Recovery'])
   print(table_consolidated['Contractor_Share'])
   print(table_consolidated['Government_Take'])

6.19 Common Pitfalls and Troubleshooting
-----------------------------------------

1. **Cost Recovery Cap Limit Reached**

   The Cost Recovery Cap is a government-imposed limit (e.g., 80% of revenue) that restricts the maximum amount contractors can recover in any given year. If eligible costs exceed this cap, the excess costs become **unrecovered costs** that carry forward to future years.

   .. admonition:: Understanding the Cap
      :class: info

      The cap exists to ensure the government receives revenue even in early years when costs are high. It is not something contractors can "exceed" - rather, it limits their annual recovery.

   .. code-block:: python

      from pyscnomics.tools.table import get_table

      # Unrecovered costs carry forward automatically
      # Get results using get_table
      table_oil, table_gas, table_consolidated = get_table(contract)
      
      # Review unrecovered cost balance:
      print(table_consolidated['Unrecovered_Cost'])

      # If negotiating contract terms, consider cap rate:
      contract.oil_cr_cap_rate = 0.80  # 80% cap (standard)

2. **Negative ETS**
   
   If ETS becomes negative, it's set to zero. This can happen when:
   - Revenue is too low
   - FTP + IC + CR exceeds revenue
   
   Check using: ``table_consolidated['ETS_After_Transfer']``

3. **Tax Calculation Issues**
   
   Ensure the correct FTP tax regime is selected:
   
   .. code-block:: python
   
      # For contracts signed before 2017
      ftp_tax_regime=FTPTaxRegime.PRE_PDJP_20_2017
      
      # For contracts signed 2017 onwards
      ftp_tax_regime=FTPTaxRegime.PDJP_20_2017

4. **DMO Holiday Not Applied**
   
   The holiday is calculated from the onstream date:
   
   .. code-block:: python
   
      # 60 months = 5 years from onstream
      contract.oil_dmo_holiday_duration = 60

6.20 References and Further Reading
------------------------------------

For additional information, see:

- :doc:`chapter3_fiscal_regulations` - Tax regimes and PDJP regulations
- :doc:`chapter4_depreciation` - Detailed depreciation methods
- :doc:`chapter5_economic_indicators` - NPV, IRR, and other indicators
- :doc:`chapter7_gross_split` - Comparison with Gross Split regime
