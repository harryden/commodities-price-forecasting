"""
Rolling-origin (sliding window) evaluation for time series forecasting.

Implements proper temporal cross-validation by iterating forecast origins
through time and collecting predictions at each step.
"""

import numpy as np
import pandas as pd
from loader import DataLoader
from transforms import Transformer


class EvaluationRunner:
    """
    Evaluates forecasting models using rolling-origin validation.

    At each origin t0, fits models on data up to t0 and forecasts
    h steps ahead, then moves forward by 'step' observations.
    """

    def __init__(
        self,
        data_path: str,
        seasonal_period: int,
        mean_ctor,
        vol_ctor,
        horizon: int = 200,
        start_frac: float = 0.8,
        step: int = 100,
    ):
        self.loader = DataLoader(data_path)
        self.transformer = Transformer(seasonal_period)
        self.mean_ctor = mean_ctor
        self.vol_ctor = vol_ctor
        self.horizon = horizon
        self.start_frac = start_frac
        self.step = step

    def run(self, commodity: str) -> pd.DataFrame:
        """Run rolling-origin evaluation for a commodity."""
        series = self.loader.get_series(commodity)
        tf = self.transformer.fit(series)
        resid = tf.transform()

        mean_model = self.mean_ctor(tf)
        vol_model = self.vol_ctor(tf)

        n = len(resid)
        start = int(n * self.start_frac)
        records = []

        for t0 in range(start, n - self.horizon, self.step):
            train = resid.iloc[:t0]
            last_price = series.iloc[t0]
            origin_dt = series.index[t0]

            # Fit and forecast mean
            mean_model.fit(train)
            r_for = mean_model.forecast(self.horizon)

            # Fit and forecast variance
            vol_model.fit(train)
            v_for = vol_model.forecast_var(self.horizon)

            # Transform back to price space
            full_r = tf.inverse_transform(r_for)
            price_f = last_price + np.cumsum(full_r)
            price_var = v_for

            # Collect results for each horizon step
            for i in range(self.horizon):
                records.append({
                    'model': f"{mean_model.__class__.__name__}_{vol_model.__class__.__name__}",
                    'origin': origin_dt,
                    'step': i + 1,
                    'forecast_price': price_f[i],
                    'actual_price': series.iloc[t0 + i + 1],
                    'return_variance': v_for[i],
                    'forecast_price_variance': price_var[i]
                })

        return pd.DataFrame(records)
