Installation
============

This guide will walk you through installing pyscnomics and setting up your development environment.

What is pyscnomics?
-------------------

pyscnomics is a Python library for modeling and analyzing Production Sharing Contracts (PSC) 
commonly used in Indonesia's oil and gas industry. It provides:

- **Cost Recovery PSC modeling** - Traditional fiscal regime with cost recovery mechanism
- **Gross Split PSC modeling** - Modern fiscal regime with direct revenue splitting
- **Economic analysis** - NPV, IRR, payback period calculations
- **Optimization tools** - Sensitivity analysis, uncertainty quantification
- **REST API** - Programmatic access via FastAPI

System Requirements
-------------------

- **Python**: 3.11 or higher
- **Operating System**: Windows, macOS, or Linux
- **Package Manager**: uv (recommended)

.. note::
   Python 3.13+ is required for full type hints support and latest features.

Installing Python
-----------------

If you don't have Python installed, download it from python.org:

.. code-block:: bash

    # Verify Python installation
    python3 --version
    # Should output: Python 3.13.x

Installing uv
-------------

uv is a fast, modern Python package manager. Install it using:

.. code-block:: bash

    # macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Windows (PowerShell)
    irm https://astral.sh/uv/install.ps1 | iex

    # Or via pip
    pip install uv

Verify installation:

.. code-block:: bash

    uv --version

Installing pyscnomics
----------------------

Option 1: Install from PyPI (recommended for users)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Create a new project
    uv init my-psc-project
    cd my-psc-project

    # Add pyscnomics
    uv add pyscnomics

Option 2: Install from source (for development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/your-org/pyscnomics.git
    cd pyscnomics

    # Install in editable mode
    uv pip install -e .

Installing Dependencies
-----------------------

pyscnomics has several optional dependencies for different features:

.. code-block:: bash

    # Core dependencies (required)
    uv sync

    # All dependencies including API, visualization, optimization
    uv sync --all-extras

    # Or install specific extras
    uv add pyscnomics[api]      # FastAPI server
    uv add pyscnomics[optimize] # Optimization tools
    uv add pyscnomics[io]       # Excel I/O support
    uv add pyscnomics[dev]      # Development tools

Available extras:

- ``api`` - FastAPI web server and REST API
- ``optimize`` - Sensitivity and uncertainty analysis
- ``io`` - Excel file reading/writing (requires Microsoft Excel)
- ``viz`` - Visualization tools
- ``dev`` - Development and testing tools
- ``all`` - All extras above

Verifying Installation
---------------------

Verify pyscnomics is installed correctly:

.. code-block:: python

    import pyscnomics

    print(pyscnomics.__version__)
    # Should output: 1.0.0 or higher

    # Check available modules
    from pyscnomics.contracts import CostRecovery, GrossSplit
    from pyscnomics.econ.indicator import npv, irr

    print("Installation successful!")

Setting Up Development Environment
-----------------------------------

If you want to contribute to pyscnomics:

.. code-block:: bash

    # Clone repository
    git clone https://github.com/your-org/pyscnomics.git
    cd pyscnomics

    # Install with all extras
    uv sync --all-extras

    # Install pre-commit hooks
    uv run pre-commit install

    # Run tests
    uv run pytest

Building Documentation
----------------------

To build the documentation locally:

.. code-block:: bash

    # Navigate to docs directory
    cd docs

    # Build HTML documentation
    sphinx-build -b html source build

    # Or use live reload during development
    sphinx-autobuild source build

The documentation will be available in ``docs/build/index.html``.

Common Installation Issues
--------------------------

**Issue: "ModuleNotFoundError: No module named 'pyscnomics'"**

Solution: Make sure you installed pyscnomics in the correct environment:

.. code-block:: bash

    # Check which Python uv is using
    uv python list
    uv python pin

**Issue: Missing dependencies**

Solution: Re-run sync with all extras:

.. code-block:: bash

    uv sync --all-extras

**Issue: Permission denied during installation**

Solution: Use a virtual environment:

.. code-block:: bash

    uv venv
    source .venv/bin/activate  # Linux/macOS
    # or
    .venv\Scripts\activate     # Windows
    uv sync --all-extras

Next Steps
----------

Now that pyscnomics is installed, continue to:

1. :doc:`quickstart` - Run your first PSC calculation
2. :doc:`chapter1_project_economics` - Learn project economics fundamentals
3. :doc:`chapter2_psc_fundamentals` - Understand PSC concepts
4. :doc:`chapter3_fiscal_regulations` - Learn Indonesian fiscal regulations
5. :doc:`chapter4_depreciation` - Understand depreciation methods
6. :doc:`chapter5_economic_indicators` - Master economic indicators
7. :doc:`chapter6_cost_recovery` - Learn Cost Recovery mechanism
8. :doc:`chapter7_gross_split` - Learn Gross Split mechanism