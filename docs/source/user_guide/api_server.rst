REST API Server
===============

pyscnomics provides a FastAPI-based REST API for programmatic access to PSC 
calculations. This is ideal for building web applications, integrating with 
other systems, or providing a service-oriented architecture.

Why Use the REST API?
---------------------

- **Web Applications**: Build a UI on top of pyscnomics
- **Microservices**: Integrate PSC calculations into larger systems
- **Automation**: Script complex workflows
- **Team Access**: Share calculations via HTTP endpoints

Prerequisites
-------------

Make sure the API dependencies are installed:

.. code-block:: bash

    uv sync --all-extras

Starting the API Server
----------------------

Run the API server:

.. code-block:: bash

    uv run uvicorn pyscnomics.api.main:app --reload

The server will start at ``http://localhost:8000``.

For production, use gunicorn with uvicorn workers:

.. code-block:: bash

    uv run gunicorn pyscnomics.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

Interactive Documentation
-------------------------

FastAPI automatically generates interactive API documentation. Access it at:

**Swagger UI**: http://localhost:8000/docs

.. image:: https://fastapi.tiangolo.com/img/index/index-01-swagger-ui.png
   :alt: Swagger UI Demo
   :width: 600

This allows you to:
- Browse all available endpoints
- Try endpoints directly in the browser
- See request/response schemas
- View authentication requirements

**ReDoc**: http://localhost:8000/redoc

.. image:: https://fastapi.tiangolo.com/img/index/index-02-redoc.png
   :alt: ReDoc Demo
   :width: 600

API Endpoints
-------------

The API provides the following main endpoints:

+------------------+------------------------------------------+
| Endpoint         | Description                              |
+==================+==========================================+
| ``GET /``        | Check API version                        |
+------------------+------------------------------------------+
| ``POST /costrecovery`` | Calculate Cost Recovery PSC    |
+------------------+------------------------------------------+
| ``POST /costrecovery/detailed`` | Get detailed CR results |
+------------------+------------------------------------------+
| ``POST /costrecovery/table`` | Get results as table    |
+------------------+------------------------------------------+
| ``POST /grosssplit``   | Calculate Gross Split PSC       |
+------------------+------------------------------------------+
| ``POST /grosssplit/detailed`` | Get detailed GS results |
+------------------+------------------------------------------+
| ``POST /grosssplit/table`` | Get GS results as table   |
+------------------+------------------------------------------+
| ``POST /transition``   | Calculate PSC Transition        |
+------------------+------------------------------------------+
| ``POST /baseproject``  | Calculate Base Project          |
+------------------+------------------------------------------+
| ``POST /baseproject/table`` | Get Base Project results |
+------------------+------------------------------------------+
| ``GET  /ltp``     | Forecast using LTP model           |
+------------------+------------------------------------------+
| ``GET  /rpd``     | Forecast using RPD model           |
+------------------+------------------------------------------+

Example Usage
-------------

Using Python requests
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import requests

    # Define the API URL
    url = "http://localhost:8000/costrecovery"

    # Prepare the request data
    data = {
        "setup": {
            "start_date": "2020-01-01",
            "contract_period": 20,
            "name": "My PSC",
        },
        "costrecovery": {
            "oil_ftp_is_available": True,
            "oil_ftp_portion": 0.2,
            "oil_ctr_pretax_share": 0.55,
            "oil_cr_cap_rate": 1.0,
            "tax_split_type": "CONVENTIONAL",
        },
        "summary_arguments": {
            "tax_regime": "UU_07_2021",
            "ftp_tax_regime": "PRE_PDJP_20_2017",
            "depr_method": "STRAIGHT_LINE",
            "inflation_rate": 0.0,
        },
        "contract_arguments": {
            "oil_dmo_volume_portion": 0.25,
            "oil_dmo_fee_portion": 0.25,
        },
        "capital": {
            "years": [1, 2, 3, 4, 5],
            "costs": [100, 50, 20, 10, 10],
        },
        "tangible": {
            "years": [1, 2, 3],
            "costs": [80, 40, 15],
        },
        "intangible": {
            "years": [1, 2],
            "costs": [20, 10],
        },
        "opex": {
            "years": list(range(1, 21)),
            "costs": [15] * 20,
        },
    }

    # Make the request
    response = requests.post(url, json=data)

    # Check response
    if response.status_code == 200:
        result = response.json()
        print(f"NPV: ${result['npv']:,.2f}")
        print(f"IRR: {result['irr']:.2%}")
    else:
        print(f"Error: {response.text}")

Using curl
~~~~~~~~~~

.. code-block:: bash

    curl -X POST "http://localhost:8000/costrecovery" \
      -H "Content-Type: application/json" \
      -d '{
        "setup": {"start_date": "2020-01-01", "contract_period": 20},
        "costrecovery": {"oil_ftp_is_available": true, "oil_ftp_portion": 0.2},
        "summary_arguments": {"tax_regime": "UU_07_2021"}
      }'

Request/Response Format
-----------------------

The API uses Pydantic models for request/response validation. Here's a typical 
request structure:

.. code-block:: python

    {
        "setup": {
            "start_date": "2020-01-01",      # Contract start date
            "contract_period": 20,             # Years
            "name": "My PSC"                  # Optional name
        },
        "costrecovery": {                     # Cost Recovery parameters
            "oil_ftp_is_available": True,
            "oil_ftp_portion": 0.2,
            "tax_split_type": "CONVENTIONAL",
            "oil_ctr_pretax_share": 0.55,
            "gas_ctr_pretax_share": 0.45,
            "oil_cr_cap_rate": 1.0,
            "gas_cr_cap_rate": 1.0,
        },
        "summary_arguments": {                 # Economic parameters
            "tax_regime": "UU_07_2021",
            "ftp_tax_regime": "PRE_PDJP_20_2017",
            "depr_method": "STRAIGHT_LINE",
            "inflation_rate": 0.0,
            "discount_rate": 0.10,
        },
        "contract_arguments": {               # Contract terms
            "oil_dmo_volume_portion": 0.25,
            "oil_dmo_fee_portion": 0.25,
            "oil_dmo_holiday_duration": 60,
        },
        "capital": {                          # Capital costs
            "years": [1, 2, 3],
            "costs": [100, 50, 20]
        },
        "tangible": {                         # Tangible costs
            "years": [1, 2],
            "costs": [80, 40]
        },
        "intangible": {                       # Intangible costs
            "years": [1, 2],
            "costs": [20, 10]
        },
        "opex": {                             # Operating costs
            "years": [1, 2, 3, 4, 5],
            "costs": [15, 15, 16, 16, 17]
        },
        "asr": {                              # Abandonment costs
            "years": [20],
            "costs": [50]
        }
    }

Typical response:

.. code-block:: python

    {
        "name": "My PSC",
        "npv": 123456789.12,
        "irr": 0.1523,
        "payout_period": 5.5,
        "total_revenue": 987654321.10,
        "total_cost_recovery": 345678901.23,
        "contractor_take": 567890123.45,
        "government_take": 419764197.65,
        "cash_flows": [ -100000000, 25000000, ... ],
        "years": [1, 2, 3, ... ]
    }

Error Handling
-------------

The API returns appropriate HTTP status codes:

- ``200`` - Success
- ``422`` - Validation Error (invalid request data)
- ``500`` - Internal Server Error

Error responses include details:

.. code-block:: python

    {
        "detail": [
            {
                "loc": ["body", "setup", "start_date"],
                "msg": "invalid date format",
                "type": "value_error"
            }
        ]
    }

Deployment Considerations
------------------------

Production Deployment
~~~~~~~~~~~~~~~~~~~~~

1. **Use Gunicorn with Uvicorn workers**::

    uv run gunicorn pyscnomics.api.main:app \
        -w 4 \
        -k uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:8000

2. **Environment variables**::

    export HOST=0.0.0.0
    export PORT=8000
    export WORKERS=4

3. **CORS configuration** - If accessing from a frontend:

   The API supports CORS. Configure in main.py if needed.

4. **Authentication** - Add using FastAPI's dependency injection:

   .. code-block:: python

       from fastapi import Depends, HTTPException, status
       from fastapi.security import HTTPBearer

       security = HTTPBearer()

       async def verify_token(token: str = Depends(security)):
           if token != "your-secret-token":
               raise HTTPException(status_code=401)
           return token

OpenAPI Schema
--------------

The API automatically generates an OpenAPI (Swagger) schema at:

- **JSON**: http://localhost:8000/openapi.json
- **YAML**: http://localhost:8000/openapi.yaml

This can be used to:
- Generate client libraries
- Integrate with API management tools
- Create custom documentation

Next Steps
----------

1. Explore interactive docs at http://localhost:8000/docs
2. Try the endpoints with sample data
3. Build your own frontend application
4. Check the :doc:`../api_reference/index` for detailed API documentation