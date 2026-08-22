from datetime import date

import numpy as np
import pytest

import pyscnomics.econ.indicator as indicator

cashflow = np.array(
    [
        -100.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
    ]
)
start_date = date(2023, 7, 15)
end_date = date(2030, 7, 14)


def test_pot():
    result = 2.5
    cash = np.array([-100.0, 50.0, 40.0, 20.0])
    calc = indicator.pot(cashflow=cash)
    assert isinstance(calc, float)
    np.testing.assert_allclose(calc, result)


def test_npv():
    result = 143.420940884647
    calc = indicator.npv(cashflow=cashflow)
    np.testing.assert_allclose(calc, result)


def test_xnpv():
    result = 154.734614499426  # excel xnpv result
    calc = indicator.xnpv(
        cashflow=cashflow,
        start_date=start_date,
        end_date=end_date,
    )
    np.testing.assert_allclose(result, calc)


def test_irr():
    result = 0.46557123187676  # excel irr result
    calc = indicator.irr(cashflow=cashflow)
    np.testing.assert_allclose(calc, result)


def test_xirr():
    result = 0.621820921  # excel xirr result
    calc = indicator.xirr(
        cashflow=cashflow,
        start_date=start_date,
        end_date=end_date,
    )
    np.testing.assert_allclose(calc, result)


def test_invalid_calculations_do_not_become_zero():
    invalid_cashflow = np.array([-100.0, np.nan, 150.0])
    start_date = date(2020, 1, 1)
    end_date = date(2022, 1, 1)

    with pytest.raises(ValueError):
        _ = indicator.npv(cashflow=np.array([np.nan]))

    with pytest.raises(ValueError):
        _ = indicator.xnpv(
            cashflow=invalid_cashflow,
            start_date=start_date,
            end_date=end_date,
        )

    with pytest.raises(ValueError):
        _ = indicator.xirr(
            cashflow=invalid_cashflow,
            start_date=start_date,
            end_date=end_date,
        )
