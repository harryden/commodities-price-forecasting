"""
Data loading and preprocessing for time series data.
"""

import pandas as pd
import numpy as np


class DataLoader:
    """Loads and cleans commodity price data from CSV."""

    def __init__(self, path: str):
        self.path = path

    def get_series(self, commodity: str) -> pd.Series:
        """Load and clean a single commodity series."""
        df = pd.read_csv(self.path, index_col='day')
        s = df[commodity].copy()

        # Replace placeholder value with NaN
        s.replace(1000.0, np.nan, inplace=True)

        # Interpolate missing values
        s = s.interpolate(method='linear', limit_area='inside')

        return s.dropna()
