# Contributing to PySCnomics

This file provides guidelines for contributors working on PySCnomics.

---

## 1. Build, Test & Development Commands

### Installation

#### Using uv (recommended)

```bash
# Install uv if not already installed
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create virtual environment and install dependencies
uv sync
```

#### Using pip (alternative)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate     # Windows

# Install in development mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_case1.py

# Run a single test function
uv run pytest tests/test_case1.py::test_revenue

# Run tests with verbose output
uv run pytest -v

# Run tests matching a pattern
uv run pytest -k "test_revenue"
```

### Running the API Service

```bash
# Start API server (default port 8000)
uv run pyscnomics --api 1

# Start API on custom port
uv run pyscnomics --api 1 --port 9999
```

### Building Documentation

```bash
# Build HTML documentation
cd docs
uv run make html

# Or directly with sphinx-build
uv run sphinx-build -b html source _build

# View the generated documentation
open _build/index.html  # macOS
# xdg-open _build/index.html  # Linux
```

---

## 2. Code Style Guidelines

### General Principles

- **Python Version**: 3.11+ (use modern syntax like `|` union types, `dataclass` field defaults)
- **Docstring Style**: Google-style docstrings
- **Type Hints**: Use type hints throughout; prefer modern Python 3.11+ syntax

### Import Organization

Order imports in this sequence (separated by blank lines):

1. Standard library (`datetime`, `functools`, `enum`, `typing`)
2. Third-party packages (`numpy`, `pandas`, `fastapi`)
3. Local application imports (`pyscnomics.*`)

```python
# Correct example
from dataclasses import dataclass, field
from typing import Union, Tuple
import numpy as np

from pyscnomics.contracts.project import BaseProject
from pyscnomics.econ.selection import FluidType, CostType
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `BaseProject`, `CostRecovery`, `ExecutiveSummary` |
| Functions/methods | snake_case | `run()`, `get_table()`, `calculate_depreciation()` |
| Variables | snake_case | `oil_revenue`, `discount_rate` |
| Constants | UPPER_CASE | `MAX_ITERATIONS`, `DEFAULT_RATE` |
| Enum values | UPPER_CASE | `FluidType.OIL`, `DeprMethod.SL` |
| Private attributes | snake_case with `_` prefix | `_oil_revenue`, `_calculate()` |

### Type Hints

Use Python 3.11+ union syntax:

```python
# Good (Python 3.11+)
def process(contract: BaseProject | CostRecovery) -> tuple[np.ndarray, float]:
    ...

# Acceptable (legacy)
from typing import Union
def process(contract: Union[BaseProject, CostRecovery]) -> Tuple[np.ndarray, float]:
    ...
```

### Custom Exceptions

Define module-specific exceptions with clear docstrings:

```python
class BaseProjectException(Exception):
    """Exception raised for misuse of BaseProject class"""
    pass


class DepreciationException(Exception):
    """Raised when an invalid depreciation method is used"""
    pass
```

### Dataclass Usage

Prefer `@dataclass` for data containers. Use `field()` for defaults and configuration:

```python
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Cashflow:
    contract: BaseProject | CostRecovery | GrossSplit
    reference_year: int | None = None
    inflation_rate: float = 0.0
    discount_rate: float = 0.1
    
    # Computed fields (not init, not repr)
    cashflow: np.ndarray = field(default=None, init=False, repr=False)
    npv: float = field(default=None, init=False, repr=False)
```

### Docstrings

Use Google-style docstrings with Parameters, Returns, and Notes sections:

```python
def straight_line_depreciation_rate(
    cost: float,
    salvage_value: float,
    useful_life: int,
    depreciation_len: int = 0,
) -> np.ndarray:
    """
    Calculate the straight-line depreciation charge for each period.

    Parameters
    ----------
    cost : float
        Cost of the asset.
    salvage_value : float
        Remaining value after depreciation.
    useful_life : int
        Duration for depreciation.
    depreciation_len : int, optional
        Length beyond useful life. Defaults to 0.

    Returns
    -------
    numpy.ndarray
        Depreciation charge for each period.

    Notes
    -----
    The straight-line method allocates equal depreciation for each period.
    """
```

### Enum Usage

Use `Enum` for related constants. Document each value:

```python
class FluidType(Enum):
    """
    Enumeration of fluid types for depreciation calculation.

    Attributes
    ----------
    ALL : str
        Represents all fluid types.
    OIL : str
        Represents oil as the fluid type.
    GAS : str
        Represents gas as the fluid type.
    """

    ALL = "all"
    OIL = "oil"
    GAS = "gas"
```

### Error Handling

- Raise descriptive custom exceptions for domain-specific errors
- Use specific exception types rather than generic `Exception`
- Include context in error messages:

```python
if not valid:
    raise DepreciationException(f"Invalid depreciation method: {method}")
```

### File Organization

Follow the existing module structure:

```
pyscnomics/
├── contracts/       # Contract classes (BaseProject, CostRecovery, GrossSplit, Transition)
├── econ/            # Economic calculations (depreciation, revenue, costs, indicator)
├── optimize/        # Optimization modules (sensitivity, uncertainty)
├── tools/           # Utility functions (table, summary, summarizer)
├── io/              # Input/output (parsing, spreadsheet, config)
├── api/             # FastAPI service (main, router, adapter)
└── dataset/         # Sample data and test fixtures
```

### Pandas Display Settings

When working with pandas in modules that need display configuration:

```python
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
```

---

## 3. Testing Guidelines

- Test files go in `tests/` directory
- Name test files as `test_<module>.py`
- Name test functions with `test_` prefix
- Use `numpy.testing.assert_allclose()` for floating-point comparisons with `atol`:
  ```python
  np.testing.assert_allclose(actual, expected, atol=1e-6)
  ```

---

## 4. Adding New Features

When adding new functionality:

1. Follow the existing class hierarchy (BaseProject → CostRecovery/GrossSplit)
2. Use existing Enum values from `pyscnomics.econ.selection` when applicable
3. Add appropriate type hints to all public methods
4. Write docstrings for all public functions
5. Add tests in the `tests/` directory
6. Use existing test data patterns from `pyscnomics.dataset.sample`
