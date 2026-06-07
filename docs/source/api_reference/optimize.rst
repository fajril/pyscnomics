Optimization Module
==================

.. automodule:: pyscnomics.optimize
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyscnomics.optimize.adjuster
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyscnomics.optimize.optimization
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyscnomics.optimize.optimization_transition
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyscnomics.optimize.sensitivity
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyscnomics.optimize.uncertainty
   :members:
   :undoc-members:
   :show-inheritance:


Optimization Parameters
-----------------------

The ``OptimizationParameter`` enum (from ``pyscnomics.econ.selection``) defines
which fiscal parameters the optimizer can sweep. Use these in the
``dict_optimization`` argument of :func:`~pyscnomics.optimize.optimization.optimize_psc`.

.. list-table::
   :header-rows: 1
   :widths: 30 50

   * - Parameter
     - Description
   * - ``OIL_CTR_PRETAX``
     - Oil contractor pre-tax split fraction
   * - ``GAS_CTR_PRETAX``
     - Gas contractor pre-tax split fraction
   * - ``OIL_FTP_PORTION``
     - Oil First Tranche Petroleum portion
   * - ``GAS_FTP_PORTION``
     - Gas First Tranche Petroleum portion
   * - ``OIL_IC``
     - Oil Investment Credit
   * - ``GAS_IC``
     - Gas Investment Credit
   * - ``OIL_DMO_FEE``
     - Oil Domestic Market Obligation fee
   * - ``GAS_DMO_FEE``
     - Gas Domestic Market Obligation fee
   * - ``VAT_RATE``
     - Value-Added Tax rate
   * - ``EFFECTIVE_TAX_RATE``
     - Effective tax rate
   * - ``MINISTERIAL_DISCRETION``
     - Ministerial discretion adjustment
   * - ``VAT_DISCOUNT``
     - VAT discount factor
   * - ``LBT_DISCOUNT``
     - Land & Building Tax discount factor
   * - ``DEPRECIATION_ACCELERATION``
     - Depreciation acceleration factor


Optimization Targets
--------------------

The ``OptimizationTarget`` enum defines which economic indicator the optimizer
tries to reach:

.. list-table::
   :header-rows: 1
   :widths: 15 50

   * - Target
     - Description
   * - ``IRR``
     - Internal Rate of Return (default)
   * - ``NPV``
     - Net Present Value
   * - ``PI``
     - Profitability Index


Usage Example
-------------

.. code-block:: python

   from pyscnomics.optimize.optimization import optimize_psc
   from pyscnomics.econ.selection import OptimizationParameter, OptimizationTarget
   import numpy as np

   params, values, result, contracts = optimize_psc(
       dict_optimization={
           "parameter": [
               OptimizationParameter.EFFECTIVE_TAX_RATE,
               OptimizationParameter.VAT_RATE,
           ],
           "min": np.array([0.30, 0.05]),
           "max": np.array([0.50, 0.15]),
       },
       contract=psc,
       contract_arguments={},
       target_optimization_value=0.10,
       summary_argument={"reference_year": 0},
       target_parameter=OptimizationTarget.IRR,
   )
