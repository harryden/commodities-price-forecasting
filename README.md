# Commodity Price Forecasting

Rolling-origin evaluation framework for time series forecasting. Compares ARIMA and GARCH models against random walk baselines on commodity price data.

## Overview

This project implements proper temporal cross-validation for evaluating forecasting models. The key insight: **random walk baselines are surprisingly hard to beat** at multi-step horizons.

**Key Features:**
- **Rolling-origin validation**: No data leakage—models only see past data at each forecast origin
- **Modular architecture**: Separate components for loading, transforms, models, evaluation, and diagnostics
- **Multiple model classes**: Random walk, ARIMA(p,d,q), and GARCH family volatility models
- **Comprehensive metrics**: RMSE, MAE, MAPE for point forecasts; MSE/MAE for variance forecasts
- **Diagnostic plots**: Residual ACF and squared-ACF to detect model misspecification

## Architecture

```
FullRunner (orchestrator)
├── DataLoader          → Load and clean price series
├── Transformer         → Differencing + optional STL decomposition
├── EvaluationRunner    → Rolling-origin backtesting loop
│   ├── ForecastModel   → Mean models (RW, ARIMA)
│   └── VolatilityModel → Variance models (Constant, ARCH, GARCH)
├── MetricsEvaluator    → Point and variance metrics
├── DiagnosticPlotter   → Residual analysis plots
└── ForecastRunner      → Future forecasts with confidence intervals
```

### Rolling-Origin Evaluation

At each origin `t0`, the model:
1. Fits on data `[0, t0)`
2. Forecasts `h` steps ahead
3. Compares against actual values `[t0+1, t0+h]`
4. Moves origin forward by `step` observations

This prevents look-ahead bias that plagues naive train/test splits.

## Requirements

```
pandas
numpy
scipy
statsmodels
arch
matplotlib
tqdm
```

## Usage

```bash
cd notebooks
jupyter notebook experiment.ipynb
```

Or programmatically:

```python
from models import RWWithDrift, ConstantVar
from full_runner import FullRunner

runner = FullRunner(
    commodity='guitars',
    data_path='data/interpolated_spiff_data.csv',
    output_root='output',
    mean_ctor=RWWithDrift,
    var_ctor=ConstantVar,
    horizon=200,
    start_frac=0.8,
    step=100
)
runner.run()
```

## Project Structure

```
commodities-price-forecasting/
├── src/
│   ├── models.py            # Mean and volatility model classes
│   ├── loader.py            # Data loading and cleaning
│   ├── transforms.py        # Differencing and seasonal decomposition
│   ├── evaluation_runner.py # Rolling-origin validation loop
│   ├── metrics.py           # Forecast accuracy metrics
│   ├── diagnostics.py       # Residual diagnostic plots
│   ├── forecast_runner.py   # Future forecasting with CIs
│   ├── aggregator.py        # Combine results across models
│   └── full_runner.py       # Main pipeline orchestrator
├── notebooks/
│   └── experiment.ipynb     # Example experiment
├── data/
│   └── interpolated_spiff_data.csv
└── output/                  # Generated results (git-ignored)
```

## Background

This project was built for TMS088 (Financial Time Series) at Chalmers University, Spring 2025.

**My Contribution:** I was solely responsible for Task 3 (Extrapolation) of a 6-person group project. This included:
- Designing and implementing the entire forecasting pipeline (all code in this repo)
- Building the rolling-origin validation framework
- Systematic model comparison (Random Walk → ARIMA → GARCH)
- Writing the methodology, results, and analysis for Task 3 in the final report

The complete group project analyzed 7 commodity time series across 4 tasks: data analysis, interpolation (filling gaps), extrapolation (forecasting 200 days), and investment strategies. See [full report (26 pages)](docs/project_report.pdf) for complete context.

**Key Finding:** Random walk baselines consistently outperformed more complex models—improvements from ARIMA and GARCH were marginal (<2% in most cases). This aligns with efficient market hypothesis. Proper validation methodology matters more than model sophistication.
