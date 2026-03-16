Chapter 4: Depreciation and Amortization
=========================================

This chapter provides a comprehensive guide to the depreciation and amortization methods implemented in pyscnomics. Understanding these methods is crucial for accurate tax calculations and economic analysis of Production Sharing Contracts (PSCs).

4.1 Introduction to Depreciation in PSC Economics
--------------------------------------------------

Depreciation and amortization are accounting methods used to allocate the cost of tangible and intangible assets over their useful lives. In petroleum economics, these mechanisms serve two primary purposes:

1. **Tax Deduction**: Depreciation/amortization expenses reduce taxable income, thereby reducing tax liability
2. **Cost Recovery**: In Cost Recovery PSCs, depreciation affects the timing and amount of cost recovery

**Key Differences:**

- **Depreciation**: Applies to tangible assets (capital costs, equipment, facilities)
- **Amortization**: Applies to intangible assets (drilling costs, exploration expenses, permits)

4.2 Depreciation Methods in pyscnomics
---------------------------------------

pyscnomics supports four depreciation/amortization methods through the ``DeprMethod`` enumeration:

.. list-table:: Depreciation Methods in pyscnomics
   :header-rows: 1

   * - Method Enum
     - Method Name
     - Use Case
     - Default Parameters
   * - ``DeprMethod.SL``
     - Straight-Line
     - Tangible assets with predictable life
     - Useful life required
   * - ``DeprMethod.DB``
     - Declining Balance
     - Assets with higher early-year usage
     - Decline factor = 2.0 (double)
   * - ``DeprMethod.PSC_DB``
     - PSC Declining Balance
     - Indonesian PSC standard method
     - Depreciation factor = 0.5
   * - ``DeprMethod.UOP``
     - Unit of Production
     - Intangible assets, Gross Split
     - Based on production profile

4.3 Straight-Line Depreciation (SL)
------------------------------------

The straight-line method allocates an equal amount of depreciation expense for each period over the asset's useful life. It is the simplest and most commonly used depreciation method.

4.3.1 Mathematical Formulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The annual depreciation charge is calculated as:

.. math::

   d = \frac{C_c - S}{N}

Where:
- :math:`d` = Annual depreciation charge
- :math:`C_c` = Asset cost (initial capital expenditure)
- :math:`S` = Salvage value (residual value at end of useful life)
- :math:`N` = Useful life in years

The book value at any time t is:

.. math::

   BV_t = C_c - t \times d

4.3.2 Example Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Scenario**: Offshore platform installation

- Cost (:math:`C_c`): $100,000,000
- Salvage value (:math:`S`): $10,000,000
- Useful life (:math:`N`): 20 years

Annual depreciation:

.. math::

   d = \frac{100,000,000 - 10,000,000}{20} = \frac{90,000,000}{20} = 4,500,000 \text{ per year}

Book value schedule:

- Year 0: $100,000,000
- Year 1: $95,500,000
- Year 5: $77,500,000
- Year 10: $55,000,000
- Year 20: $10,000,000 (salvage value)

4.3.3 pyscnomics Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from pyscnomics.econ.depreciation import (
       straight_line_depreciation_rate,
       straight_line_book_value
   )

   # Calculate depreciation charges
   depreciation_charge = straight_line_depreciation_rate(
       cost=100_000_000,           # Asset cost
       salvage_value=10_000_000,   # Residual value
       useful_life=20,             # Years
       depreciation_len=25         # Extend array to 25 years (optional)
   )
   
   print(f"Annual depreciation: ${depreciation_charge[0]:,.2f}")
   # Output: Annual depreciation: $4,500,000.00
   
   # Calculate book values
   book_value = straight_line_book_value(
       cost=100_000_000,
       salvage_value=10_000_000,
       useful_life=20,
       depreciation_len=25
   )
   
   print(f"Book value Year 10: ${book_value[9]:,.2f}")
   # Output: Book value Year 10: $55,000,000.00

4.3.4 When to Use Straight-Line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Straight-line depreciation is most appropriate when:

- The asset generates consistent economic benefits each year
- Maintenance costs are relatively stable over the asset life
- There is no significant technological obsolescence risk
- Tax regulations permit or require this method

**Default Parameters in pyscnomics**:

- No default useful life - must be specified by user
- Salvage value defaults to 0 if not specified
- Depreciation charge is evenly distributed across all periods

4.4 Declining Balance Depreciation (DB)
---------------------------------------

The declining balance method is an accelerated depreciation method where depreciation expense is higher in the early years and decreases over time. This better reflects the actual usage pattern of many assets.

4.4.1 Mathematical Formulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The depreciation charge for each period is calculated as:

.. math::

   d_t = \frac{k}{N} \times BV_{t-1}

Where:
- :math:`d_t` = Depreciation charge in year t
- :math:`k` = Decline factor (commonly 2.0 for double-declining balance)
- :math:`N` = Useful life
- :math:`BV_{t-1}` = Book value at beginning of year t
- :math:`BV_0 = C_c` (initial asset cost)

The book value is calculated as:

.. math::

   BV_t = BV_{t-1} - d_t = BV_{t-1} \times \left(1 - \frac{k}{N}\right)

4.4.2 Example Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Scenario**: Processing facility with double-declining balance

- Cost (:math:`C_c`): $50,000,000
- Salvage value (:math:`S`): $5,000,000
- Useful life (:math:`N`): 10 years
- Decline factor (:math:`k`): 2.0 (double-declining)

Calculation:

- **Year 1**: 
  - Depreciation rate = 2/10 = 20%
  - Depreciation = $50,000,000 × 0.20 = $10,000,000
  - Book value = $40,000,000

- **Year 2**: 
  - Depreciation = $40,000,000 × 0.20 = $8,000,000
  - Book value = $32,000,000

- **Year 3**: 
  - Depreciation = $32,000,000 × 0.20 = $6,400,000
  - Book value = $25,600,000

The depreciation continues until the book value approaches the salvage value.

4.4.3 Switch to Straight-Line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In practice, the declining balance method often switches to straight-line when the straight-line depreciation on the remaining book value exceeds the declining balance amount. This ensures the asset depreciates fully to salvage value.

4.4.4 pyscnomics Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pyscnomics.econ.depreciation import (
       declining_balance_depreciation_rate,
       declining_balance_book_value
   )

   # Calculate declining balance depreciation
   depreciation_charge = declining_balance_depreciation_rate(
       cost=50_000_000,            # Asset cost
       salvage_value=5_000_000,    # Residual value
       useful_life=10,             # Years
       decline_factor=2.0,         # Double-declining (default)
       depreciation_len=15         # Extend to 15 years (optional)
   )
   
   print(f"Year 1 depreciation: ${depreciation_charge[0]:,.2f}")
   # Output: Year 1 depreciation: $10,000,000.00
   
   print(f"Year 2 depreciation: ${depreciation_charge[1]:,.2f}")
   # Output: Year 2 depreciation: $8,000,000.00

**Default Parameters**:

- Decline factor defaults to 2.0 (double-declining balance)
- Can be set to 1.0 for standard declining balance
- Higher factors result in faster depreciation

4.4.5 When to Use Declining Balance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Declining balance is appropriate when:

- Assets lose more value in early years due to wear and tear
- Maintenance costs increase as assets age
- Technological obsolescence is significant
- Tax benefits of early depreciation are desirable

4.5 PSC Declining Balance
------------------------------------

The PSC Declining Balance method is a specialized depreciation method used specifically in Indonesian Production Sharing Contracts. It follows a specific formula defined in PSC regulations.

4.5.1 Mathematical Formulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The depreciation charge is calculated as:

.. math::

   d_t = \gamma_{dep} \times C_c \times (1 - \gamma_{dep})^{t-1}

Where:
- :math:`d_t` = Depreciation charge in year t
- :math:`\gamma_{dep}` = Depreciation factor (default 0.5 in pyscnomics)
- :math:`C_c` = Initial asset cost
- :math:`t` = Year number (1, 2, 3, ...)

The book value is:

.. math::

   BV_t = C_c \times (1 - \gamma_{dep})^t

4.5.2 Example Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Scenario**: PSC-compliant depreciation

- Cost (:math:`C_c`): $80,000,000
- Depreciation factor (:math:`\gamma_{dep}`): 0.5
- Useful life: Determined by factor

Calculation:

- **Year 1**: 

  - Depreciation = 0.5 × $80,000,000 × (1-0.5)⁰ = $40,000,000
  - Book value = $80,000,000 × 0.5 = $40,000,000

- **Year 2**: 

  - Depreciation = 0.5 × $80,000,000 × 0.5¹ = $20,000,000
  - Book value = $80,000,000 × 0.5² = $20,000,000

- **Year 3**: 

  - Depreciation = 0.5 × $80,000,000 × 0.5² = $10,000,000
  - Book value = $80,000,000 × 0.5³ = $10,000,000

The pattern continues with each year's depreciation being half of the previous year's book value.

4.5.3 pyscnomics Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pyscnomics.econ.depreciation import (
       psc_declining_balance_depreciation_rate,
       psc_declining_balance_book_value
   )

   # Calculate PSC declining balance depreciation
   depreciation_charge = psc_declining_balance_depreciation_rate(
       cost=80_000_000,            # Asset cost
       useful_life=8,              # Useful life in years
       depreciation_factor=0.5,    # PSC standard factor (default)
       depreciation_len=10         # Extend to 10 years (optional)
   )
   
   print(f"Year 1 depreciation: ${depreciation_charge[0]:,.2f}")
   # Output: Year 1 depreciation: $40,000,000.00
   
   print(f"Year 2 depreciation: ${depreciation_charge[1]:,.2f}")
   # Output: Year 2 depreciation: $20,000,000.00
   
   # Get book values
   book_value = psc_declining_balance_book_value(
       cost=80_000_000,
       useful_life=8,
       depreciation_factor=0.5
   )
   
   print(f"Book value Year 1: ${book_value[0]:,.2f}")
   # Output: Book value Year 1: $40,000,000.00

**Default Parameters**:

- Depreciation factor defaults to 0.5
- Must be between 0 and 1 (inclusive)
- pyscnomics raises ``DepreciationException`` if factor is outside this range

4.5.4 When to Use PSC Declining Balance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use PSC_DB when:

- Modeling Indonesian PSC contracts specifically
- The contract requires PSC-compliant depreciation
- Regulatory compliance is mandatory
- Default factor of 0.5 is typically required by regulations

4.6 Unit of Production (UOP) Amortization
-----------------------------------------

The Unit of Production method is primarily used for amortizing intangible costs in oil and gas projects. It allocates costs based on actual production volumes rather than time.

4.6.1 Mathematical Formulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The amortization charge for each period is calculated as:

.. math::

   a_t = \frac{Q_t}{Q_{total}} \times (C_c - S) \times 2

Where:
- :math:`a_t` = Amortization charge in year t
- :math:`Q_t` = Production volume in year t
- :math:`Q_{total}` = Total expected production over asset life
- :math:`C_c` = Asset cost (typically intangible costs)
- :math:`S` = Salvage value (usually 0 for intangibles)
- :math:`\times 2` = Double UOP method (standard in many PSCs)

4.6.2 Example Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Scenario**: Geological survey and drilling costs

- Cost (:math:`C_c`): $20,000,000 (intangible exploration costs)
- Salvage value (:math:`S`): $0
- Production profile (:math:`Q_t`): [1000, 950, 900, 850, 800, 750, 700, 650] barrels/day
- Total production (:math:`Q_{total}`): 5,600 barrels/day over 8 years

Calculation:

.. math::

   \text{Amortization base} = \frac{20,000,000 - 0}{5,600} \times 2 = 7,142.86 \text{ per barrel}

Yearly amortization:

- **Year 1**: 1,000 × 7,142.86 = $7,142,860
- **Year 2**: 950 × 7,142.86 = $6,785,717
- **Year 3**: 900 × 7,142.86 = $6,428,574
- **Year 8**: 650 × 7,142.86 = $4,642,859

Total amortization over 8 years equals $20,000,000 (the full cost).

4.6.3 pyscnomics Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from pyscnomics.econ.depreciation import (
       unit_of_production_rate,
       unit_of_production_book_value
   )

   # Define production profile
   production = np.array([1000, 950, 900, 850, 800, 750, 700, 650])
   production_years = np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027])

   # Calculate UOP amortization
   amortization_charge = unit_of_production_rate(
       start_year_project=2020,    # Project start year
       cost=20_000_000,            # Intangible cost
       prod=production,            # Production array
       prod_year=production_years, # Production years
       salvage_value=0.0,          # No salvage for intangibles
       amortization_len=10         # Extend to 10 years (optional)
   )
   
   print(f"Year 1 amortization: ${amortization_charge[0]:,.2f}")
   # Output: Year 1 amortization: $7,142,857.14
   
   print(f"Total amortization: ${amortization_charge.sum():,.2f}")
   # Output: Total amortization: $20,000,000.00
   
   # Calculate book values
   book_value = unit_of_production_book_value(
       start_year_project=2020,
       cost=20_000_000,
       prod=production,
       prod_year=production_years,
       salvage_value=0.0,
       amortization_len=10
   )
   
   print(f"Book value Year 4: ${book_value[3]:,.2f}")
   # Output: Book value Year 4: $5,357,142.86

**Key Parameters**:

- ``start_year_project``: First year of the project
- ``prod``: Production values for each year
- ``prod_year``: Calendar years corresponding to production
- ``amortization_len``: Total length of amortization array

4.6.4 Validation and Error Handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pyscnomics performs validation on UOP inputs:

.. code-block:: python

   # This will raise UnitOfProductionException if:
   # - Production year is before project start year
   # - Production year is after project end year
   # - Salvage value exceeds cost
   # - Total production is zero or negative
   # - Production and year arrays have different lengths

4.6.5 When to Use UOP Amortization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use UOP when:

- Amortizing intangible costs in Gross Split PSCs (required method)
- Production varies significantly over the asset life
- Costs should be matched to actual resource extraction
- Intangible assets have no time-based useful life

4.7 Using Depreciation in Contracts
-------------------------------------

4.7.1 Cost Recovery Contracts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Cost Recovery PSCs, depreciation affects tax deductions:

.. code-block:: python

   from pyscnomics.contracts import CostRecovery
   from pyscnomics.econ.selection import DeprMethod
   from pyscnomics.econ.costs import CapitalCost

   # Define capital cost with tangible assets
   capital = CapitalCost(
       start_year=2020,
       end_year=2030,
       expense_year=np.array([2020, 2021]),
       cost=np.array([100_000_000, 50_000_000]),
       useful_life=np.array([20, 15]),  # Different assets, different lives
       # ... other parameters
   )

   # Create contract and run with depreciation method
   contract = CostRecovery(
       start_date=date(2020, 1, 1),
       end_date=date(2030, 12, 31),
       capital_cost=(capital,),
       # ... other parameters
   )

   # Run with selected depreciation method
   contract.run(
       depr_method=DeprMethod.SL,  # Straight-line
       # depr_method=DeprMethod.DB,     # Declining balance
       # depr_method=DeprMethod.PSC_DB, # PSC declining balance
   )

4.7.2 Gross Split Contracts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Gross Split PSCs, UOP amortization is used for intangible costs:

.. code-block:: python

   from pyscnomics.contracts import GrossSplit
   from pyscnomics.econ.selection import DeprMethod
   from pyscnomics.econ.costs import Intangible

   # Define intangible costs
   intangible = Intangible(
       start_year=2020,
       end_year=2030,
       expense_year=np.array([2020]),
       cost=np.array([20_000_000]),
       # ... other parameters
   )

   # Create contract
   contract = GrossSplit(
       start_date=date(2020, 1, 1),
       end_date=date(2030, 12, 31),
       intangible_cost=(intangible,),
       # ... other parameters
   )

   # Run with UOP method (required for Gross Split)
   contract.run(
       depr_method=DeprMethod.UOP,  # Unit of Production
   )

4.8 Default Values and Configuration
-------------------------------------

**Default Depreciation Method**:

- If not specified, pyscnomics uses ``DeprMethod.SL`` (Straight-Line)
- For Gross Split contracts, ``DeprMethod.UOP`` should be explicitly set

**Default Parameters by Method**:

.. list-table:: Depreciation Default Parameters
   :header-rows: 1

   * - Parameter
     - Default Value
     - Applies To
   * - ``decline_factor`` (DB)
     - 2.0
     - Double-declining balance
   * - ``depreciation_factor`` (PSC_DB)
     - 0.5
     - PSC declining balance
   * - ``salvage_value`` (all methods)
     - 0.0
     - All methods
   * - ``depreciation_len`` (all methods)
     - 0 (matches useful life)
     - All methods

**Useful Life Considerations**:

- No default useful life. It must be specified
- Typical ranges in petroleum industry:
  - Offshore platforms: 20-25 years
  - Processing facilities: 15-20 years
  - Drilling equipment: 10-15 years
  - Vehicles and minor equipment: 5-10 years

4.9 Summary
-----------

pyscnomics provides comprehensive depreciation and amortization support:

1. **Straight-Line (SL)**: Equal depreciation each year; simplest method
2. **Declining Balance (DB)**: Accelerated depreciation; higher early-year expenses
3. **PSC Declining Balance**: Indonesian PSC-specific method; factor of 0.5
4. **Unit of Production (UOP)**: Production-based amortization; required for Gross Split intangibles

**Key Points**:

- Depreciation affects tax calculations by reducing taxable income
- Method selection depends on contract type and regulatory requirements
- Default parameters are conservative and follow industry standards
- UOP is mandatory for Gross Split intangible cost amortization

Next Steps
^^^^^^^^^^

- For economic indicator calculations using depreciation: See :doc:`chapter5_economic_indicators`
- For Cost Recovery contract modeling: See :doc:`chapter6_cost_recovery`
- For Gross Split contract modeling: See :doc:`chapter7_gross_split`
