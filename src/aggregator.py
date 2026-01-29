"""
Aggregates results across model combinations for comparison.
"""

from pathlib import Path
import pandas as pd


class ResultsAggregator:
    """Collects and combines metrics from multiple model runs."""

    def __init__(self, root_dir: str, commodity: str):
        self.commodity_path = Path(root_dir) / commodity

    def aggregate(self) -> pd.DataFrame:
        """Aggregate metrics.csv files from all model combinations."""
        records = []

        for combo_dir in self.commodity_path.iterdir():
            metrics_file = combo_dir / "evaluation" / "metrics.csv"
            if metrics_file.exists():
                df = (
                    pd.read_csv(metrics_file, index_col=0)
                    .reset_index()
                    .rename(columns={"index": "model"})
                )
                df["model_combo"] = combo_dir.name
                records.append(df)

        if not records:
            raise FileNotFoundError(f"No metrics.csv files found under {self.commodity_path}")

        agg_df = pd.concat(records, ignore_index=True)
        agg_df = agg_df.sort_values('RMSE', ascending=True)
        return agg_df
