"""
Evaluation metrics for time series forecasting.
"""

import pandas as pd
import numpy as np


class MetricsEvaluator:
    """Computes point forecast and variance forecast metrics."""

    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df.copy()

    def sliding_window_metrics(self) -> pd.DataFrame:
        """Compute metrics for each sliding window origin."""
        df = self.df.copy()
        df['error'] = df['forecast_price'] - df['actual_price']
        df['abs_error'] = df['error'].abs()
        df['sq_error'] = df['error'] ** 2
        df['ape'] = df['abs_error'] / df['actual_price']

        return (
            df
            .groupby(['model', 'origin'])
            .agg(
                RMSE=('sq_error', lambda x: np.sqrt(x.mean())),
                MAE=('abs_error', 'mean'),
                MAPE=('ape', 'mean'),
                MdAPE=('ape', 'median'),
            )
            .reset_index()
        )

    def point_metrics(self) -> pd.DataFrame:
        """Aggregate metrics across all windows for each model."""
        df = self.df.copy()
        df['error'] = df['forecast_price'] - df['actual_price']
        df['abs_error'] = df['error'].abs()
        df['sq_error'] = df['error'] ** 2
        df['ape'] = df['abs_error'] / df['actual_price']

        return (
            df
            .groupby('model')
            .agg(
                RMSE=('sq_error', lambda x: np.sqrt(x.mean())),
                MAE=('abs_error', 'mean'),
                MAPE=('ape', 'mean'),
                MdAPE=('ape', 'median'),
            )
        )

    def variance_metrics(self) -> pd.DataFrame:
        """Evaluate variance forecast accuracy."""
        df = self.df.copy()
        df['realized_var'] = (df['actual_price'] - df['forecast_price']) ** 2
        df['error_var'] = df['forecast_price_variance'] - df['realized_var']
        df['mse_var'] = df['error_var'] ** 2
        df['mae_var'] = df['error_var'].abs()

        return df.groupby('model').agg(
            MSE_var=('mse_var', 'mean'),
            MAE_var=('mae_var', 'mean'),
        )

    def all_metrics(self, levels=[0.8, 0.9, 0.95]) -> pd.DataFrame:
        """Combine point and variance metrics."""
        pm = self.point_metrics()
        vm = self.variance_metrics()
        return pm.join(vm)
