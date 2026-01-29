"""
Diagnostic plots for model residual analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


class DiagnosticPlotter:
    """Generates diagnostic plots for forecast residuals."""

    def __init__(self, results_df: pd.DataFrame, output_dir: str, combo_name: str):
        self.df = results_df.copy()
        self.output_dir = output_dir
        self.combo_dir = os.path.join(output_dir, combo_name, 'diagnostics')
        os.makedirs(self.combo_dir, exist_ok=True)

    def _get_residuals(self, model: str) -> np.ndarray:
        """Extract residuals from first forecast origin for a model."""
        first_origin = (
            self.df[self.df['model'] == model]
            .sort_values(['origin', 'step'])
            .groupby('origin')
            .head(1)['origin']
            .iloc[0]
        )

        sub = self.df[(self.df['model'] == model) & (self.df['origin'] == first_origin)]
        sub = sub.sort_values('step')
        return (sub['forecast_price'] - sub['actual_price']).values

    def plot_residuals(self, model: str):
        """Plot residuals over forecast horizon."""
        res = self._get_residuals(model)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(res)
        ax.axhline(0, color='k', linestyle='--')
        ax.set_title(f"Residuals for {model}")
        fig.savefig(os.path.join(self.combo_dir, f"{model}_residuals.png"))
        plt.close(fig)

    def plot_acf(self, model: str, lags: int = 40):
        """Plot autocorrelation of residuals."""
        res = self._get_residuals(model)
        fig = plot_acf(res, lags=lags)
        plt.title(f"ACF of Residuals for {model}")
        fig.savefig(os.path.join(self.combo_dir, f"{model}_acf.png"))
        plt.close(fig)

    def plot_acf_squared(self, model: str, lags: int = 40):
        """Plot autocorrelation of squared residuals (checks for ARCH effects)."""
        res = self._get_residuals(model)
        fig = plot_acf(res**2, lags=lags)
        plt.title(f"ACF of Squared Residuals for {model}")
        fig.savefig(os.path.join(self.combo_dir, f"{model}_acf_squared.png"))
        plt.close(fig)
