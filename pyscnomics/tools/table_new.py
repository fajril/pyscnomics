"""
A collection of procedures to generate cashflow of a contract in the form of DataFrame
"""

import pandas as pd
import numpy as np
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition


def get_table(contract: CostRecovery | GrossSplit | BaseProject | Transition) -> tuple:

    if isinstance(contract, CostRecovery):
        pass

    elif isinstance(contract, GrossSplit):
        pass

    elif isinstance(contract, BaseProject):
        pass

    elif isinstance(contract, Transition):
        pass
