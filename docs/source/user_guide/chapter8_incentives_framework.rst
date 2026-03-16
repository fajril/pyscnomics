Chapter 8: Incentives Framework and Application Process
======================================================

This chapter examines the incentives framework for petroleum projects under
KEPMEN 199/K/HK.02/MEM.M/2021 (Pedoman Pemberian Insentif Kegiatan Usaha Hulu
Minyak dan Gas Bumi). Understanding this framework is essential for petroleum
economists who need to evaluate project eligibility for fiscal incentives and
navigate the application process with SKK Migas.

.. admonition:: Regulatory Reference
   :class: info

   This chapter is based on KEPMEN 199/K/HK.02/MEM.M/2021 issued by the
   Minister of Energy and Mineral Resources on October 18, 2021. The regulation
   establishes guidelines for granting incentives to upstream oil and gas
   business activities.

8.1 Overview of the Incentives Framework
-----------------------------------------

Purpose and Legal Basis
^^^^^^^^^^^^^^^^^^^^^^^^

KEPMEN 199/2021 was established to:

1. **Optimize upstream oil and gas activities** through targeted fiscal incentives
2. **Maintain and increase production** by making marginal projects economically viable
3. **Sustain investment** in the petroleum sector
4. **Contribute to state revenue** while ensuring contractor profitability

The regulation operates under the authority of the Ministry of Energy and Mineral
Resources (ESDM) and establishes SKK Migas as the implementing agency responsible
for evaluation and recommendation.

.. admonition:: Key Principle
   :class: tip

   **Minimum Incentive Principle**: Incentives are granted at the minimum level
   required to achieve optimal economic impact for both the Government and the
   Contractor (Diktum KETIGA).

Scope and Applicability
^^^^^^^^^^^^^^^^^^^^^^^^

Incentives under KEPMEN 199/2021 apply to:

**Eligible Projects:**

- **POD I** (First Plan of Development) for new Working Areas
- **Subsequent PODs** including revisions for existing Working Areas
- **Extension areas** and transfer areas

**Contract Types:**

- Cost Recovery PSCs (all variants)
- Gross Split PSCs (all PERMEN regimes)

.. list-table:: Incentive Scope by Contract Type
   :header-rows: 1

   * - Incentive Type
     - Cost Recovery PSC
     - Gross Split PSC
   * - Profit Split Adjustment
     - Yes (negotiable split)
     - Yes (variable/progressive split)
   * - First Tranche Petroleum (FTP)
     - Yes
     - N/A
   * - Investment Credit (IC)
     - Yes
     - N/A
   * - DMO Fee Adjustment
     - Yes
     - Yes (via fee portion)

8.2 Types of Incentives Available
----------------------------------

Cost Recovery PSC Incentives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Cost Recovery contracts, the following incentive mechanisms are available
under KEPMEN 199/2021:

**1. Profit Split Adjustment**

The contractor-government profit sharing ratio can be adjusted from the
standard terms. This is the most significant incentive available, as it
directly affects the long-term revenue distribution.

**2. First Tranche Petroleum (FTP) Adjustment**

- FTP portion can be modified from standard rates
- FTP sharing between contractor and government can be adjusted
- Applied before cost recovery calculations

**3. Investment Credit (IC)**

- Additional cost recovery for qualifying capital investments
- Typically 10-20% of eligible capital costs
- Accelerates recovery of development investments

See :doc:`chapter6_cost_recovery` for detailed technical implementation.

**4. DMO Fee Adjustment**

- DMO fee portion can be reduced from standard rates
- Holiday periods can be extended
- Applies to oil and gas sold to domestic markets

**5. Accelerated Depreciation**

- Depreciation schedules can be accelerated
- Increases early-year cost recovery
- Improves project cash flow timing

Gross Split PSC Incentives
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Gross Split contracts, incentives are provided through the split mechanism:

**1. Base Split Adjustment**

The contractor's base split can be increased from standard rates:

- Oil: Standard 43% can be increased
- Gas: Standard 48% can be increased

**2. Variable Split Components**

Additional percentage points based on field characteristics:

.. list-table:: Variable Split Incentives (PERMEN 52/2017)
   :header-rows: 1

   * - Component
     - Maximum Adjustment
     - Typical Incentive Conditions
   * - Field Status
     - +5% (POD I)
     - New field development
   * - Field Location
     - +16% (ultra-deep offshore)
     - >1000m water depth
   * - Reservoir Depth
     - +1% (>2500m)
     - Deep reservoirs
   * - Infrastructure
     - +4% (not available)
     - Frontier areas
   * - Reservoir Type
     - +16% (unconventional)
     - CBM, shale, etc.
   * - H2S Content
     - +5% (>4000 ppm)
     - High impurities

**3. Progressive Split**

Price-based and cumulative production-based adjustments that provide
additional contractor share when conditions warrant.

**4. Ministerial Discretion**

Up to 8% additional split can be granted at the Minister's discretion
for project-specific factors (removed in PERMEN 13/2024).

See :doc:`chapter7_gross_split` for detailed technical implementation.

8.3 Evaluation Criteria
------------------------

The evaluation process uses two categories of criteria:

Kriteria Umum (General Criteria)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Economic Thresholds:**

Projects are evaluated based on Internal Rate of Return (IRR) or
Profitability Index (PI) against established benchmarks.

.. admonition:: Threshold Determination
   :class: important

   Thresholds are determined through **ekonometrika** (econometric analysis)
   using historical POD data from SKK Migas:

   - **Lower bound**: P90 (90th percentile) of industry IRR/PI
   - **Upper bound**: P50 (50th percentile) of industry IRR/PI

**Economic Parameters Used:**

.. list-table:: Evaluation Metrics
   :header-rows: 1

   * - Parameter
     - Description
     - Formula
   * - IRR
     - Internal Rate of Return
     - :math:`\sum_{t=0}^{T} \frac{CF_t}{(1 + IRR)^t} = 0`
   * - PI
     - Profitability Index
     - :math:`PI = \frac{NPV + I}{I}`
   * - R/C
     - Revenue over Cost
     - :math:`R/C = \frac{\sum GR}{\sum OC}`

Where:

- :math:`CF_t` = Cash flow in year t
- :math:`NPV` = Net Present Value
- :math:`I` = Investment
- :math:`GR` = Gross Revenue
- :math:`OC` = Operating Costs

Kriteria Khusus (Special Criteria)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Technical Aspects:**

Projects may qualify based on technical complexity:

1. **Deepwater Operations** (>500m water depth)
2. **High Pressure High Temperature (HPHT)**

   - High Pressure reservoirs
   - High Temperature reservoirs

3. **High Impurities**

   - CO2 content >5%
   - H2S content >100 ppm
   - Nitrogen, Mercury content

4. **Unconventional Resources**

   - Coal Bed Methane (CBM)
   - Shale oil/gas
   - Tight gas

**Non-Technical Aspects:**

1. **Remote/Frontier Locations**

   - Outermost regions
   - Remote islands
   - Areas with limited infrastructure

2. **Policy Implementation**

   - Supports government investment promotion policies
   - Creates significant multiplier effects
   - Strategic national importance

3. **Other Special Circumstances**

   - Comparable urgency and specificity
   - National energy security considerations

8.4 The 4-Quadrant Economic Mapping
------------------------------------

The evaluation uses a four-quadrant classification system based on Revenue
over Cost (R/C) ratio and IRR/PI metrics.

Quadrant Classification
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 4-Quadrant Economic Classification Matrix
   :header-rows: 1
   :widths: 15 40 40
   :class: quadrant-table

   * - 
     - **Low R/C**
     - **High R/C**
   * - **High IRR/PI**
     - **Quadrant IV** (Gray)
       
       Low R/C
       
       No Incentive
     - **Quadrant III** (Green)
       
       High R/C
       
       No Incentive Needed
   * - **Low IRR/PI**
     - **Quadrant I** (Red)
       
       Low R/C
       
       Maximum Incentive
     - **Quadrant II** (Yellow)
       
       High R/C
       
       Moderate Incentive

**Quadrant I: Low R/C, Low IRR/PI**

- **Characteristics**: Marginal economics on both metrics
- **Incentive Priority**: Highest
- **Typical Projects**: Frontier areas, deepwater, HPHT
- **Recommended Action**: Maximum incentive package

**Quadrant II: High R/C, Low IRR/PI**

- **Characteristics**: Good cost efficiency but poor returns
- **Incentive Priority**: Moderate
- **Typical Projects**: High-cost developments with good production
- **Recommended Action**: Targeted incentives (IC, accelerated depreciation)

**Quadrant III: High R/C, High IRR/PI**

- **Characteristics**: Strong economics on all metrics
- **Incentive Priority**: None required
- **Typical Projects**: Conventional onshore/standard offshore
- **Recommended Action**: No incentives granted

**Quadrant IV: Low R/C, High IRR/PI**

- **Characteristics**: Poor cost efficiency but good returns
- **Incentive Priority**: None required
- **Typical Projects**: Rare - usually indicates data issues
- **Recommended Action**: No incentives granted

Decision Flow
^^^^^^^^^^^^^

.. mermaid::

   flowchart TD
       A[Project Proposal] --> B{Meets Kriteria Umum?}
       B -->|Yes| C[Quadrant Classification]
       B -->|No| D{Meets Kriteria Khusus?}
       D -->|Yes| E[Case-by-Case Assessment]
       D -->|No| F[No Incentive Granted]
       C --> G{Quadrant I or II?}
       G -->|Yes| H[Recommend Incentive Package]
       G -->|No| D
       E --> H
       H --> I[SKK Migas Recommendation]
       I --> J[Minister Approval]
       J --> K[Incentive Granted]

8.5 IRR vs PI Selection
------------------------

The choice between IRR and PI depends on project characteristics.

When to Use IRR
^^^^^^^^^^^^^^^^

**Appropriate for:**

- **POD I** (First Plan of Development)
- Projects with **S-curve cash flows**
- Fields with **no existing production**

**Characteristics:**

- Initial negative cash flows (investment phase)
- Subsequent positive cash flows (production phase)
- Clear break-even point

**Example:**

A new offshore field development:

- Years 1-3: -500 MUSD (drilling, facilities)
- Years 4-20: +80 to +150 MUSD/year (production)
- IRR captures the return on invested capital

When to Use PI
^^^^^^^^^^^^^^^

**Appropriate for:**

- **Brownfield expansions**
- **Workovers/redevelopment**
- Projects with **existing production revenue**

**Characteristics:**

- Revenue from existing operations
- Incremental investment
- Need to compare efficiency of capital use

**Example:**

An EOR project in a mature field:

- Existing production: 10,000 bbl/day
- Incremental investment: 50 MUSD
- Additional production: +2,000 bbl/day
- PI measures value created per MUSD invested

Statistical Analysis
^^^^^^^^^^^^^^^^^^^^

Both metrics use P-statistics for threshold determination:

.. list-table:: Statistical Thresholds
   :header-rows: 1

   * - Percentile
     - Interpretation
     - Use Case
   * - P90
     - Conservative estimate
     - Lower incentive threshold
   * - P50
     - Median outcome
     - Upper incentive threshold
   * - P10
     - Optimistic estimate
     - Not used for incentives

8.6 Application Process Flow
-----------------------------

The incentive application follows a structured process with specific timelines
and documentation requirements.

Step-by-Step Process
^^^^^^^^^^^^^^^^^^^^^

**Step 1: Contractor Submission**

The contractor submits a formal request to SKK Migas including:

- Formal application letter
- Complete POD documentation
- Economic analysis with base case and sensitivity scenarios
- Justification for requested incentives
- Technical and commercial data

**Step 2: SKK Migas Initial Review**

SKK Migas conducts preliminary verification:

- Completeness of documentation
- Data consistency
- Scenario reasonableness
- Preliminary eligibility assessment

**Step 3: Economic Evaluation**

If preliminary eligible, SKK Migas performs detailed analysis:

.. mermaid::

   flowchart LR
       A[Calculate IRR/PI] --> B[Calculate R/C]
       B --> C[Quadrant Mapping]
       C --> D{Meets Criteria?}
       D -->|Yes| E[Determine Incentive Level]
       D -->|No| F[Case-by-Case Review]
       F --> G{Justified?}
       G -->|Yes| E
       G -->|No| H[Reject Application]

**Step 4: Scenario Development**

For qualifying projects, SKK Migas develops incentive scenarios:

- Minimum incentive required to achieve target IRR/PI
- Alternative incentive packages
- Impact on government take
- Recommendation to Minister

**Step 5: Timeline Compliance**

SKK Migas must provide recommendations within:

- **5 working days** after receiving complete documentation
- Expedited processing available for strategic projects

**Step 6: Ministerial Decision**

The Minister of ESDM reviews SKK Migas recommendations and:

- Approves incentive package
- Modifies incentive terms
- Rejects the application

Required Documentation
^^^^^^^^^^^^^^^^^^^^^^^

**Economic Analysis:**

- Base case economic model
- Sensitivity analysis (price, cost, production)
- Without-incentive scenario
- With-incentive scenario(s)
- IRR/PI calculations

**Technical Documentation:**

- Reservoir characterization
- Development plan
- Cost estimates (CAPEX, OPEX)
- Production profiles

**Commercial Data:**

- Price assumptions
- Market analysis
- Contract terms

8.7 Practical Examples
-----------------------

Example 1: Deepwater Development (Quadrant I)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Project Characteristics:**

- Location: Offshore, 1,200m water depth
- Type: New field (POD I)
- Reserves: 50 MMBOE
- Development cost: 1,500 MUSD

**Base Case Economics:**

- IRR: 8%
- R/C: 1.3
- Quadrant: I (Low R/C, Low IRR)

**Evaluation:**

.. code-block:: python

   from pyscnomics.econ.indicator import irr
   
   # Calculate base case IRR
   cashflows = np.array([-500, -800, -200,  # Investment phase
                         150, 200, 250, 300, 280, 250])  # Production
   
   base_irr = irr(cashflow=cashflows)
   print(f"Base case IRR: {base_irr:.1%}")
   # Output: Base case IRR: 8.2%
   
   # Target IRR: 12% (industry threshold)
   # Required incentive: Investment Credit + Increased Split

**Recommended Incentive Package:**

- Investment Credit: 20% on qualifying CAPEX
- Profit split adjustment: +5% contractor share
- Resulting IRR: 12.5%

Example 2: Brownfield EOR (Using PI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Project Characteristics:**

- Location: Mature onshore field
- Type: EOR implementation
- Incremental reserves: 10 MMBOE
- Investment: 100 MUSD

**Base Case Economics:**

- PI: 1.2
- NPV: 20 MUSD
- Existing production: 5,000 bbl/day

**Evaluation:**

Since this is a brownfield project with existing revenue, PI is the
appropriate metric. The project creates 1.2 MUSD value per 1 MUSD invested.

**Recommended Incentive:**

- Accelerated depreciation (front-load cost recovery)
- DMO fee reduction (25% to 10%)
- Resulting PI: 1.4

Example 3: Frontier Area (Kriteria Khusus)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Project Characteristics:**

- Location: Eastern Indonesia frontier area
- Type: POD I
- Infrastructure: None available
- Economic results: Marginal

**Evaluation:**

Base case economics fall below thresholds, but project meets Kriteria Khusus:

- Remote/outermost location
- Limited infrastructure
- Strategic energy security value

**Process:**

- Standard evaluation: Does not meet Kriteria Umum
- Case-by-case assessment: Approved
- Recommended package: Base split adjustment + DMO holiday extension

8.8 Monitoring and Review
--------------------------

Post-Approval Obligations
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once incentives are granted, contractors must:

**1. Reporting Requirements**

- Semi-annual progress reports to SKK Migas
- Annual economic performance reports
- Immediate notification of material changes

**2. Compliance Monitoring**

SKK Migas monitors:

- Actual vs. planned development
- Production achievement
- Cost management
- Economic performance

**3. Review Triggers**

SKK Migas may recommend incentive review if:

- Contractor fails to meet agreed obligations
- Project economics exceed approved parameters significantly
- Material breach of contract terms

Incentive Withdrawal
^^^^^^^^^^^^^^^^^^^^

Incentives may be withdrawn or modified if:

1. **Non-compliance**: Contractor fails to execute approved POD
2. **Excessive profitability**: Actual IRR/PI exceeds approved thresholds by >20%
3. **Contract violations**: Breach of PSC terms

The review process requires:

- SKK Migas assessment
- Formal recommendation to Minister
- Contractor right to respond

Case-by-Case Procedures
^^^^^^^^^^^^^^^^^^^^^^^^

For projects evaluated under Kriteria Khusus:

**Extended Review Process:**

1. Detailed technical assessment
2. Economic modeling with special parameters
3. Multi-stakeholder consultation
4. Higher-level approval authority

**Timeline:**

- May extend beyond standard 5-day period
- Depends on complexity and data availability

8.9 Summary and Key Takeaways
------------------------------

**Key Points:**

1. **Minimum Incentive Principle**: Incentives granted at lowest level to achieve
economic viability

2. **Two-Track Evaluation**: Kriteria Umum (standard) or Kriteria Khusus (special cases)

3. **4-Quadrant System**: R/C vs IRR/PI mapping determines incentive priority

4. **Metric Selection**: IRR for greenfield POD I, PI for brownfield projects

5. **Timeline**: 5 working days for SKK Migas recommendation

6. **Monitoring**: Semi-annual reporting and compliance review

**When to Seek Incentives:**

- Project IRR < industry P90 threshold
- Quadrant I or II classification
- Technical complexity (HPHT, deepwater, unconventional)
- Frontier/remote locations

**When Not to Seek Incentives:**

- Project already economically viable (Quadrant III)
- Standard terms sufficient
- No qualifying technical or locational factors

**Next Steps:**

- For Cost Recovery incentive implementation: See :doc:`chapter6_cost_recovery`
- For Gross Split incentive implementation: See :doc:`chapter7_gross_split`
- For economic indicator calculation: See :doc:`chapter5_economic_indicators`

8.10 References
---------------

**Primary Regulation:**

- KEPMEN 199/K/HK.02/MEM.M/2021: Pedoman Pemberian Insentif Kegiatan Usaha
  Hulu Minyak dan Gas Bumi (October 18, 2021)

**Related Regulations:**

- PERMEN ESDM 8/2017: Kontrak Bagi Hasil Gross Split
- PERMEN ESDM 52/2017: Progressive Split Amendment
- PERMEN ESDM 20/2019: Variable Split 5
- PERMEN ESDM 12/2020: Contract Flexibility
- PERMEN ESDM 13/2024: Latest Gross Split Regulation

**Related Documentation:**

- :doc:`chapter6_cost_recovery` - Technical implementation of Cost Recovery incentives
- :doc:`chapter7_gross_split` - Technical implementation of Gross Split incentives
- :doc:`chapter5_economic_indicators` - IRR, PI, and NPV calculations
- :doc:`chapter3_fiscal_regulations` - Tax and regulatory framework
