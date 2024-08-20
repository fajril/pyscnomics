# Overview

PySCnomics is a package contains tailored functionalities for assessing economic feasibility of oil and gas projects following the state-of-the-art Production Sharing Contract (PSC) schemes in Indonesia. 

Developed through a collaborative research between Indonesia's Special Task Force for Upstream Oil and Gas Business Activities (SKK Migas) and the Department of Petroleum Engineering at Institut Teknologi Bandung (ITB), PySCnomics stands as a reliable solution for industry professionals.


## Installation
To install PySCnomics, simply run:

`pip install pyscnomics`


## Key Features
PySCnomics offers comprehensive capabilities to evaluate the feasibility of various Indonesian PSC contracts, including a wide range of regime variations of each contract: 
- Base Project
- Cost Recovery
- Gross Split
- Transition Cost Recovery - Cost Recovery
- Transition Cost Recovery - Gross Split
- Transition Gross Split - Gross Split 
- Transition Gross Split - Cost Recovery

Beyond feasibility assessment, PySCnomics provides advanced tools for:
- PSC Contract Optimization
- PSC Contract Sensitivity
- PSC Contract Uncertainty Analysis

To further streamline the assessment process, PySCnomics includes specialized modules for:
- Depreciation
- Inflation
- Cost Taxing
- Production Profile Generation
- API service
- And much more...

## Quick Start
Create a new file `sample.py` with the following code:

```python
from pyscnomics.dataset.object_sample import generate_contract_sample
from pyscnomics.econ.selection import ContractSample
from pyscnomics.tools.table import get_table

# Initiating Contract Object
psc = generate_contract_sample(case=ContractSample.CASE_1)

# Get the cashflow table from the contract
tables = get_table(contract=psc)
print(tables)
```

## License
This project is licensed under the terms of the Apache Software license. See the [License](https://github.com/fajril/pyscnomics/blob/main/LICENSE) file for details.





