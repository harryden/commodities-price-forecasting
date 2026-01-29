"""
Time series transformations including differencing and seasonal decomposition.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


class Transformer:
    """
    Extracts seasonal component from differenced series using STL decomposition.
    Provides inverse transform to reconstruct forecasts.
    """

    def __init__(self, seasonal_period: int = None):
        self.seasonal_period = seasonal_period
        self.returns_: pd.Series = None
        self.seasonal_: np.ndarray = None
        self.residuals_: pd.Series = None

    def fit(self, price_series: pd.Series) -> "Transformer":
        """Fit transformer on price series, extracting seasonality from differences."""
        r = price_series.diff().dropna()
        self.returns_ = r

        if self.seasonal_period:
            stl = STL(r, period=self.seasonal_period, robust=True).fit()
            self.seasonal_ = stl.seasonal.values
        else:
            self.seasonal_ = np.zeros(len(r))

        resid_vals = r.values - self.seasonal_
        self.residuals_ = pd.Series(resid_vals, index=r.index)
        return self

    def transform(self) -> pd.Series:
        """Return the residual series (differenced - seasonal)."""
        return self.residuals_

    def inverse_transform(self, resid_forecast: np.ndarray) -> np.ndarray:
        """Add seasonal component back to residual forecasts."""
        h = len(resid_forecast)

        if self.seasonal_period and len(self.seasonal_) >= self.seasonal_period:
            block = self.seasonal_[-self.seasonal_period:]
            reps = int(np.ceil(h / len(block)))
            seasonal_future = np.tile(block, reps)[:h]
        else:
            seasonal_future = np.zeros(h)

        return resid_forecast + seasonal_future
