Changelog
=========

Version 1.4.0
-------------

**New Features**

- **API Service** — PySCnomics now ships with a built-in FastAPI server (``pyscnomics --api 1``).
  30 REST endpoints covering contract calculation, optimization, sensitivity, uncertainty, and more.
  Interactive docs available at ``/docs``.
- **Gross Split Optimization** — regime-based split optimization across all 5 Permen ESDM regimes
  (8/2017, 52/2017, 20/2019, 12/2020, 13/2024). See :doc:`/api_reference/optimize`.

**Improvements**

- Default API port changed from 8000 to 9999 to avoid conflicts with common dev servers.
- Dependency upgrades: FastAPI 0.136, Starlette 1.2, Pydantic 2.13, Pillow 12.2, lxml 6.1.

**Breaking Changes**

- Module ``pyscnomics.contracts.grossplit`` renamed to ``pyscnomics.contracts.gross_split``
  (snake_case convention). Update your imports accordingly.

**Documentation**

- New user guide chapters 1–8 covering PSC economics from fundamentals to advanced analysis.
- New API server guide with usage examples.
- ReadTheDocs integration at https://pyscnomics.readthedocs.io.

Version 1.3.0
-------------

- Transition contract variants (4 modes between Cost Recovery and Gross Split).
- Uncertainty analysis with Monte Carlo simulation.
- Sensitivity analysis with deviation sweeps.
