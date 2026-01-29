"""
Future forecasting with confidence intervals.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from loader import DataLoader
from transforms import Transformer


class ForecastRunner:
    """Generates future forecasts with confidence interval bands."""

    def __init__(
        self,
        data_path: str,
        seasonal_period: int,
        mean_ctor,
        vol_ctor,
        horizon: int = 200,
        output_dir: str = "outputs",
    ):
        self.loader = DataLoader(data_path)
        self.transformer = Transformer(seasonal_period)
        self.mean_ctor = mean_ctor
        self.vol_ctor = vol_ctor
        self.horizon = horizon
        self.output_dir = output_dir

    def run(self, commodity: str):
        """Generate point forecasts and variance estimates."""
        series = self.loader.get_series(commodity)
        tf = self.transformer.fit(series)
        resid = tf.transform()

        mean_model = self.mean_ctor(tf)
        vol_model = self.vol_ctor(tf)

        mean_model.fit(resid)
        vol_model.fit(resid)

        r_for = mean_model.forecast(self.horizon)
        v_for = vol_model.forecast_var(self.horizon)

        full_r = tf.inverse_transform(r_for)
        last_price = series.iloc[-1]
        price_f = last_price + np.cumsum(full_r)

        idx = range(len(series), len(series) + self.horizon)
        forecast_series = pd.Series(price_f, index=idx, name='forecast')
        variance_series = pd.Series(v_for, index=idx, name='variance')

        return series, forecast_series, variance_series

    def plot(self, commodity: str, ci_levels=[0.8, 0.9, 0.95]):
        """Generate forecasts and save plots with confidence intervals."""
        history, forecast_series, variance_series = self.run(commodity)
        os.makedirs(self.output_dir, exist_ok=True)

        cum_var = variance_series.cumsum().values
        fc_values = forecast_series.values

        result_df = pd.DataFrame({
            'forecast': forecast_series,
            'variance': variance_series,
        })

        # Compute CI bounds
        for level in ci_levels:
            alpha = 1.0 - level
            z = norm.ppf(1 - alpha / 2)
            lower = fc_values - z * np.sqrt(cum_var)
            upper = fc_values + z * np.sqrt(cum_var)
            pct = int(level * 100)
            result_df[f'lower_{pct}'] = lower
            result_df[f'upper_{pct}'] = upper

        csv_path = os.path.join(self.output_dir, 'forecast_with_cis.csv')
        result_df.to_csv(csv_path, index=True)

        plot_colors = ['tab:orange'] * len(ci_levels)
        sorted_levels = sorted(ci_levels, reverse=True)

        def save_zoom_plot(history_part, label, filename):
            fig, ax = plt.subplots(figsize=(12, 4))
            n = min(len(history_part), len(forecast_series))
            idx_fc = forecast_series.index[:n]

            ax.plot(history_part.index, history_part.values, label='History')
            ax.plot(idx_fc, forecast_series.values[:n], label='Forecast', color='orange')

            for j, level in enumerate(sorted_levels):
                pct = int(level * 100)
                lower_vals = result_df[f'lower_{pct}'].values[:n]
                upper_vals = result_df[f'upper_{pct}'].values[:n]
                ax.fill_between(
                    idx_fc, lower_vals, upper_vals,
                    color=plot_colors[j],
                    alpha=0.3 - 0.1 * j,
                    label=f'{pct}% CI'
                )

            ax.set_title(f'Forecast for {commodity} ({label})')
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Price')
            ax.legend(loc='upper left')

            ymin = min(history_part.min(), lower_vals.min()) * 0.90
            ymax = max(history_part.max(), upper_vals.max()) * 1.10
            ax.set_ylim([ymin, ymax])

            plt.tight_layout()
            path = os.path.join(self.output_dir, filename)
            plt.savefig(path)
            plt.close(fig)
            return path

        path_full = save_zoom_plot(history, 'full history', 'forecast_full.png')
        path_200 = save_zoom_plot(history.iloc[-200:], 'last 200 points', 'forecast_200.png')
        path_20 = save_zoom_plot(history.iloc[-20:], 'last 20 points', 'forecast_20.png')

        return csv_path, path_full, path_200, path_20
