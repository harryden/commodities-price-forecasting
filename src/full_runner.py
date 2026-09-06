"""
Main orchestrator for the forecasting pipeline.

Coordinates data loading, model fitting, evaluation, diagnostics, and forecasting.
"""

import os
import pandas as pd
from loader import DataLoader
from transforms import Transformer
from evaluation_runner import EvaluationRunner
from metrics import MetricsEvaluator
from diagnostics import DiagnosticPlotter
from forecast_runner import ForecastRunner
from aggregator import ResultsAggregator


class FullRunner:
    """Runs the complete forecasting and evaluation pipeline."""

    def __init__(
        self,
        commodity: str,
        data_path: str,
        output_root: str,
        mean_ctor,
        var_ctor,
        seasonal_period: int = None,
        horizon: int = 200,
        start_frac: float = 0.8,
        step: int = 10,
        ci_levels: list = [0.8, 0.9, 0.95],
    ):
        self.commodity = commodity
        self.data_path = data_path
        self.output_root = output_root
        self.mean_ctor = mean_ctor
        self.var_ctor = var_ctor
        self.vol_ctor = var_ctor
        self.seasonal_period = seasonal_period
        self.horizon = horizon
        self.start_frac = start_frac
        self.step = step
        self.ci_levels = ci_levels

    def combo_name(self):
        """Generate descriptive name for this model combination."""
        name = f"MeanModel:{self.mean_ctor.__name__}_VarModel:{self.var_ctor.__name__}"
        name += f"_Season:{self.seasonal_period}" if self.seasonal_period else "_Season:None"
        return name

    def run(self, progress_bar: bool = False):
        """Execute the full pipeline."""
        combo = self.combo_name()
        out_dir = os.path.join(self.output_root, self.commodity, combo)
        eval_dir = os.path.join(out_dir, 'evaluation')
        plots_dir = os.path.join(eval_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Rolling-origin evaluation
        fr = EvaluationRunner(
            data_path=self.data_path,
            seasonal_period=self.seasonal_period,
            mean_ctor=self.mean_ctor,
            vol_ctor=self.vol_ctor,
            horizon=self.horizon,
            start_frac=self.start_frac,
            step=self.step
        )
        df_fc = fr.run(self.commodity)
        df_fc.to_csv(os.path.join(eval_dir, 'forecast_results.csv'), index=False)

        # Metrics
        me = MetricsEvaluator(df_fc)
        df_metrics = me.all_metrics(levels=self.ci_levels)
        df_metrics.to_csv(os.path.join(eval_dir, 'metrics.csv'))

        df_metrics_window = me.sliding_window_metrics()
        df_metrics_window.to_csv(os.path.join(eval_dir, 'metrics_window.csv'), index=False)

        # Diagnostics
        dp = DiagnosticPlotter(df_fc, plots_dir, combo)
        for m in df_fc['model'].unique():
            dp.plot_residuals(m)
            dp.plot_acf(m)
            dp.plot_acf_squared(m)

        # Future forecasting with CIs
        ff = ForecastRunner(
            data_path=self.data_path,
            seasonal_period=self.seasonal_period,
            mean_ctor=self.mean_ctor,
            vol_ctor=self.vol_ctor,
            horizon=self.horizon,
            output_dir=out_dir,
        )
        ff.plot(self.commodity, ci_levels=self.ci_levels)

        return out_dir

    def aggregate(self):
        """Aggregate results across all model combinations."""
        agg = ResultsAggregator(
            root_dir=self.output_root,
            commodity=self.commodity
        )
        df_agg = agg.aggregate()
        path = os.path.join(self.output_root, self.commodity, 'aggregated_metrics.csv')
        df_agg.to_csv(path, index=False)
        return df_agg
