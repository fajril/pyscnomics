CLI Reference
=============

PySCnomics provides a command-line interface for running simulations and
starting the API server.


Usage
-----

.. code-block:: bash

   pyscnomics [OPTIONS]


Options
-------

``-p, --path TEXT``
   Path to the Microsoft Excel workbook with the PySCnomics template.
   Required for ``Standard``, ``Sensitivity``, ``Optimization``, and
   ``Uncertainty`` modes.

``-m, --mode TEXT``
   Simulation mode. One of:

   - ``Standard`` — run a single contract calculation
   - ``Sensitivity`` — parameter sweep analysis
   - ``Optimization`` — find parameters that achieve a target economic indicator
   - ``Uncertainty`` — Monte Carlo simulation

``-api, --api INTEGER``
   Enable or disable the built-in FastAPI server.

   - ``1`` — start the API server (default)
   - ``0`` — run in file-only mode

``-port, --port INTEGER``
   Port for the API server. Default: ``9999``.


Examples
--------

Start the API server on the default port:

.. code-block:: bash

   pyscnomics --api 1

Start on a custom port:

.. code-block:: bash

   pyscnomics --api 1 --port 8080

Run a standard simulation from an Excel template:

.. code-block:: bash

   pyscnomics --path /path/to/template.xlsx --mode Standard

Run sensitivity analysis:

.. code-block:: bash

   pyscnomics --path /path/to/template.xlsx --mode Sensitivity


API Endpoints
-------------

Once the server is running, the following endpoints are available:

.. list-table::
   :header-rows: 1
   :widths: 15 10 50

   * - Category
     - Method
     - Path
   * - Root
     - GET
     - ``/api/``
   * - Contract
     - POST
     - ``/api/costrecovery``, ``/api/grosssplit``, ``/api/transition``
   * - Summary
     - POST
     - ``/api/{contract}/detailed_summary``
   * - Tables
     - POST
     - ``/api/{contract}/table``
   * - Optimization
     - POST
     - ``/api/{contract}/optimization``
   * - Sensitivity
     - POST
     - ``/api/{contract}/sensitivity``
   * - Uncertainty
     - POST
     - ``/api/{contract}/uncertainty``
   * - Split
     - POST
     - ``/api/grosssplit/split``, ``/api/transition/split``
   * - Econ Limit
     - POST
     - ``/api/econlimit``
   * - Expenditures
     - POST
     - ``/api/asr_expenditures``, ``/api/lbt_expenditures``
   * - LTP & RPD
     - POST
     - ``/api/ltp``, ``/api/rpd``

Interactive documentation (Swagger UI) is available at ``/docs`` when the
server is running.
