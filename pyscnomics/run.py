
import numpy as np
from pyscnomics.econ.costs import PreOnstreamCost


posc = PreOnstreamCost(
    start_year=2023,
    end_year=2032,
    onstream_year=2027,
    expense_year=np.array([2023, 2024, 2025, 2026, 2027]),
    cost=np.array([100, 100, 100, 50, 50]),
)