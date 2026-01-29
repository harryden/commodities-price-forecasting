"""
Forecasting models for time series prediction.

Implements mean models (Random Walk, ARIMA) and volatility models (Constant, ARCH, GARCH).
"""

import abc
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA


class ForecastModel(abc.ABC):
    """Abstract base class for mean forecasting models."""

    @abc.abstractmethod
    def fit(self, resid: pd.Series) -> None:
        """Fit model parameters from residual series."""
        ...

    @abc.abstractmethod
    def forecast(self, h: int) -> np.ndarray:
        """Produce h-step ahead forecasts."""
        ...


class VolatilityModel(abc.ABC):
    """Abstract base class for volatility forecasting models."""

    @abc.abstractmethod
    def fit(self, resid: pd.Series) -> None:
        """Fit volatility parameters from residual series."""
        ...

    @abc.abstractmethod
    def forecast_var(self, h: int) -> np.ndarray:
        """Produce h-step ahead variance forecasts."""
        ...


# --- Mean Models ---

class RWNoDrift(ForecastModel):
    """Random walk with no drift: forecast zero change at all horizons."""

    def __init__(self, transformer):
        pass

    def fit(self, resid):
        pass

    def forecast(self, h):
        return np.zeros(h)


class RWWithDrift(ForecastModel):
    """Random walk with constant drift equal to historical mean."""

    def __init__(self, transformer):
        self.mu = 0.0

    def fit(self, resid):
        self.mu = float(resid.mean())

    def forecast(self, h):
        return np.full(h, self.mu)


class ARIMAForecaster(ForecastModel):
    """ARIMA(p,d,q) forecaster with configurable order and trend."""

    def __init__(self, transformer, order_fn=lambda x: (1, 0, 0), trend='c'):
        self.order_fn = order_fn
        self.trend = trend
        self.model = None

    def fit(self, resid):
        data = resid.reset_index(drop=True)
        order = self.order_fn(data)
        self.model = ARIMA(data, order=order, trend=self.trend).fit(method='innovations_mle')

    def forecast(self, h):
        return self.model.get_forecast(steps=h).predicted_mean.values


# --- Volatility Models ---

class ConstantVar(VolatilityModel):
    """Constant variance model: forecast historical variance at all horizons."""

    def __init__(self, transformer):
        self.var = None

    def fit(self, resid):
        self.var = float(resid.var())

    def forecast_var(self, h):
        return np.full(h, self.var)


class ARCH1(VolatilityModel):
    """ARCH(1) conditional variance model."""

    def __init__(self, transformer):
        self.res = None
        self.scale = 100

    def fit(self, resid):
        resid_scaled = self.scale * resid
        self.res = arch_model(resid_scaled, vol='Arch', p=1).fit(disp='off')

    def forecast_var(self, h):
        f = self.res.forecast(horizon=h)
        return f.variance.values[-1] / (self.scale ** 2)


class ARCH2(VolatilityModel):
    """ARCH(2) conditional variance model."""

    def __init__(self, transformer):
        self.res = None
        self.scale = 100

    def fit(self, resid):
        resid_scaled = self.scale * resid
        self.res = arch_model(resid_scaled, vol='Arch', p=2).fit(disp='off')

    def forecast_var(self, h):
        f = self.res.forecast(horizon=h)
        return f.variance.values[-1] / (self.scale ** 2)


class GARCH11(VolatilityModel):
    """GARCH(1,1) conditional variance model."""

    def __init__(self, transformer):
        self.res = None
        self.scale = 100

    def fit(self, resid):
        resid_scaled = self.scale * resid
        self.res = arch_model(resid_scaled, vol='Garch', p=1, q=1).fit(disp='off')

    def forecast_var(self, h):
        f = self.res.forecast(horizon=h)
        return f.variance.values[-1] / (self.scale ** 2)
