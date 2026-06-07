# PySCnomics

> An economic engine for calculating PSC (Production Sharing Contract) schemes in Indonesia.

PySCnomics is a Python package for assessing the economic feasibility of oil and gas projects following Indonesian PSC schemes, including Cost Recovery, Gross Split, and their transition variants. Developed jointly by SKK Migas and Institut Teknologi Bandung (ITB).

- 📖 **Full documentation:** [https://pyscnomics.readthedocs.io](https://pyscnomics.readthedocs.io)
- 🚀 **API service:** 30 endpoints via `pyscnomics --api 1`
- 🐍 **Python:** 3.11+


## Installation

Install from PyPI:

```bash
pip install pyscnomics
```

Or with [uv](https://github.com/astral-sh/uv) (recommended for development):

```bash
uv pip install pyscnomics
```


## Key Features

PySCnomics evaluates all Indonesian PSC contract types and regime variations:

- **Base Project** — economic indicators without fiscal terms
- **Cost Recovery** — pre-2017 fiscal regime
- **Gross Split** — post-2017 fiscal regime (Permen ESDM 8/2017, 52/2017, 20/2019, 12/2020, 13/2024)
- **Transition** — four variants between Cost Recovery and Gross Split

Advanced analysis modules:

- **Optimization** — regime-based Gross Split contract optimization
- **Sensitivity** — parameter sweep with deviation analysis
- **Uncertainty** — Monte Carlo simulation
- **Depreciation, Inflation, Cost Taxing, Production Profile Generation**


## Quick Start

Generate a sample contract and view its cashflow table:

```python
from pyscnomics.dataset.object_sample import generate_contract_sample
from pyscnomics.econ.selection import ContractSample
from pyscnomics.tools.table import get_table

# Initiate contract object
psc = generate_contract_sample(case=ContractSample.CASE_1)

# Get cashflow table
tables = get_table(contract=psc)
print(tables)
```


## Optimization (Gross Split)

Optimize a Gross Split contract toward a target IRR, NPV, or PI by sweeping fiscal parameters:

```python
from pyscnomics.dataset.object_sample import generate_contract_sample
from pyscnomics.econ.selection import ContractSample, OptimizationParameter, OptimizationTarget
from pyscnomics.optimize.optimization import optimize_psc
import numpy as np

psc = generate_contract_sample(case=ContractSample.CASE_1)

# Configure optimization: which parameters to sweep and their bounds
dict_optimization = {
    "parameter": [
        OptimizationParameter.EFFECTIVE_TAX_RATE,
        OptimizationParameter.VAT_RATE,
    ],
    "min": np.array([0.30, 0.05]),
    "max": np.array([0.50, 0.15]),
}

# Run optimization
params, values, result, contracts = optimize_psc(
    dict_optimization=dict_optimization,
    contract=psc,
    contract_arguments={},
    target_optimization_value=0.10,   # 10% IRR target
    summary_argument={"reference_year": 2023},
    target_parameter=OptimizationTarget.IRR,
)

print("Optimized parameters:", params)
print("Optimized values:", values)
print("Resulting IRR:", result)
```

Supported regimes (configured in `pyscnomics/contracts/grossplit.py`):

- `PERMEN_ESDM_8_2017` — base split (0.43, 0.48)
- `PERMEN_ESDM_52_2017` — base split (0.43, 0.48)
- `PERMEN_ESDM_20_2019` — base split (0.43, 0.48)
- `PERMEN_ESDM_12_2020` — base split (0.43, 0.48)
- `PERMEN_ESDM_13_2024` — base split (0.47, 0.49)


## Sensitivity Analysis

Sweep a parameter range and evaluate the effect on contract indicators:

```python
from pyscnomics.dataset.object_sample import generate_contract_sample
from pyscnomics.econ.selection import ContractSample
from pyscnomics.optimize.sensitivity import sensitivity_psc

psc = generate_contract_sample(case=ContractSample.CASE_1)

result = sensitivity_psc(
    contract=psc,
    contract_arguments={'effective_tax_rate': 0.40},
    summary_arguments={'reference_year': 2023},
    min_deviation=0.2,
    max_deviation=0.2,
    base_value=1,
    step=10,
)

for key in result.keys():
    print(key, result[key])
```


## API Service

PySCnomics can run as a FastAPI service, exposing all 30 endpoints programmatically. This is the recommended way to integrate PySCnomics into web apps, notebooks, or pipelines.

### Start the server

```bash
pyscnomics --api 1 --port 9999
```

- `--api 1` is the default; pass `--api 0` to disable API mode
- `--port 9999` is the default; change if 9999 is in use

The server will be available at `http://localhost:9999`. Interactive docs at `http://localhost:9999/docs`.

### Endpoint categories

| Category | Method | Path | Count |
|----------|--------|------|-------|
| Root | `GET` | `/api/` | 1 |
| Contract calculation | `POST` | `/api/{contract}` | 3 |
| Detailed summary | `POST` | `/api/{contract}/detailed_summary` | 3 |
| Tables | `POST` | `/api/{contract}/table` | 4 |
| Optimization | `POST` | `/api/{contract}/optimization` | 3 |
| Sensitivity | `POST` | `/api/{contract}/sensitivity` | 4 |
| Uncertainty | `POST` | `/api/{contract}/uncertainty` | 3 |
| Split calculation | `POST` | `/api/{contract}/split` | 2 |
| Econ limit & expenditures | `POST` | `/api/econlimit`, `/api/asr_expenditures`, `/api/lbt_expenditures` | 3 |
| LTP & RPD | `POST` | `/api/ltp`, `/api/rpd` | 2 |
| **Total** | | | **30** |

Where `{contract}` is one of: `costrecovery`, `grosssplit`, `transition`, `baseproject`.

See the OpenAPI schema at `http://localhost:9999/openapi.json` for full details.


## Development

Clone the repository:

```bash
git clone https://github.com/fajril/pyscnomics.git
cd pyscnomics
uv sync
```

Run the API locally:

```bash
uv run pyscnomics --api 1
```

Run tests:

```bash
uv run pytest
```

Build documentation (Sphinx):

```bash
uv sync --extra docs
uv run sphinx-build docs/source docs/build
```


## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Major changes should be discussed in an issue first.

Public forks in active use:

- [adhimmulia/pyscnomics-Extended](https://github.com/adhimmulia/pyscnomics-Extended)
- [aguswe/pyscnomics](https://github.com/aguswe/pyscnomics)


## Citation

If PySCnomics is used in academic work, please cite the project. Author list and contact information are in [AUTHORS.md](AUTHORS.md).


## License

This project is licensed under the Apache Software License. See [LICENSE](LICENSE) for details.
