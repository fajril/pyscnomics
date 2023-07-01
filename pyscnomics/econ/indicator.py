import numpy as np

def npv(cashflow: np.ndarray, disc_rate: float):
    """ Calculate Net Present Value
    """
    year = np.arange(0, len(cashflow))
    return (cashflow * np.power(np.repeat((1 + disc_rate), len(cashflow)), -year)).sum()

def irr()