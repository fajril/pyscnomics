import numpy as np


class RPDModel:
    """Oil Ramp Up - Plateau - Decline Model
    =========================================

    Forecast model based on ramp up, plateau, and decline characteristics.
    ramp up period is assumed to follow linear model.
    Decline period is based on Arps Exponential Decline.

    Parameters
    ---
    year_rampup : int
        Number of year from onstream to peak/plateau rate (yr)
    drate : float
        Arps yearly decline rate (1/yr)
    q_plateau_ratio : float
        Ratio of plateau rate and volume (1/yr)
    q_min_ratio : float
        Ratio of minimum rate at abandoned year and volume (1/yr)
    """

    def __init__(
        self,
        year_rampup: int = 2,
        drate: float = 0.08,
        q_plateau_ratio: float = 0.1,
        q_min_ratio: float = 0.05,
    ):
        self._year_rampup = year_rampup
        self._drate = drate
        self._q_plateau_ratio = q_plateau_ratio
        self._q_min_ratio = q_min_ratio
        self._q_plateau = 0
        self._q_min = 0

    def predict(self, volume: float) -> np.ndarray:
        """Predict the forecast based on RPD Model
        ===========================================

        Calculates the production forecast.

        Parameter
        ---
        volume : float
            Resources volume.

        Return
        ---
        rate : ndarray
            array of production forecast
        """
        self._q_plateau = volume * self._q_plateau_ratio
        self._q_min = volume * self._q_min_ratio
        rate_rampup = self._calc_rampup()
        rate_decline = self._calc_decline()
        rate_plateau = self._calc_plateau(volume, rate_rampup.sum(), rate_decline.sum())
        rate = np.concatenate([rate_rampup, rate_plateau, rate_decline])
        return self._adj_rate(volume, rate)

    def cumprod(self, volume: float) -> np.ndarray:
        """Calculate the cumulative production forecast based on RPD Model
        ===========================================

        Parameters
        ----------
        volume : float
            Resources volume.

        Returns
        -------
        cumprod : ndarray
            Cumulative production forecast.
        """
        return np.cumsum(self.predict(volume))


    def _calc_rampup(self) -> np.ndarray:
        if self._year_rampup == 0:
            return np.asarray([])
        slope = self._q_plateau / self._year_rampup
        return np.arange(self._year_rampup + 1) * slope

    def _calc_decline(self) -> np.ndarray:
        year_decline = np.log(self._q_min / self._q_plateau) / (-self._drate)
        year = np.arange(year_decline)
        rate = self._q_plateau * np.exp(-self._drate * year)
        return rate

    def _calc_plateau(
        self, vol_total: float, vol_rampup: float, vol_decline: float
    ) -> np.ndarray:
        vol_plateau = vol_total - vol_rampup - vol_decline
        if vol_plateau < 0:
            return np.asarray([])
        year_plateau = int(vol_plateau / self._q_plateau)
        return np.ones(year_plateau) * self._q_plateau

    def _adj_rate(self, volume: float, rate: np.ndarray) -> np.ndarray:
        rate_adj = rate * volume / np.sum(rate)
        rate_adj = np.round(rate_adj, 3)
        rate_adj[-1] += volume - rate_adj.sum()
        return rate_adj
