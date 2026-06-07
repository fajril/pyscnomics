# PySCnomics

> An economic engine for calculating PSC (Production Sharing Contract) schemes in Indonesia.

PySCnomics is a Python package for assessing the economic feasibility of oil and gas projects following Indonesian PSC schemes, including Cost Recovery, Gross Split, and their transition variants. Developed jointly by SKK Migas and Institut Teknologi Bandung (ITB).

- 📖 **Documentation:** [https://pyscnomics.readthedocs.io](https://pyscnomics.readthedocs.io)
- 🐍 **Python:** 3.11+


## Installation

```bash
pip install pyscnomics
```

## Key Features

- **Base Project**, **Cost Recovery**, **Gross Split**, and **Transition** contract types
- **Optimization**, **Sensitivity**, and **Uncertainty** analysis
- Depreciation, inflation, cost taxing, and production profile generation
- REST API service with 30 endpoints (FastAPI)

## Quick Start

```python
from pyscnomics.dataset.object_sample import generate_contract_sample
from pyscnomics.econ.selection import ContractSample
from pyscnomics.tools.table import get_table

psc = generate_contract_sample(case=ContractSample.CASE_1)
tables = get_table(contract=psc)
print(tables)
```

## API Service

```bash
pyscnomics --api 1 --port 9999
```

Interactive docs available at `http://localhost:9999/docs` once the server is running.

## Development

```bash
git clone https://github.com/fajril/pyscnomics.git
cd pyscnomics
uv sync
uv run pytest
```

## Contributing

Contributions are welcome. Please open an issue first for major changes.

Public forks in active use:
- [adhimmulia/pyscnomics-Extended](https://github.com/adhimmulia/pyscnomics-Extended)
- [aguswe/pyscnomics](https://github.com/aguswe/pyscnomics)

## License

Licensed under the Apache Software License. See [LICENSE](LICENSE).
