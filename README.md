# Commodity Price Forecasting

Rolling-origin evaluation framework for time series forecasting. Evaluates Random Walk and ARIMA mean models paired with constant variance baselines on commodity price data; ARCH and GARCH volatility models are implemented in `src/models.py` and evaluated in the accompanying report.

## Overview

This project implements proper temporal cross-validation for evaluating forecasting models. The key insight: **random walk baselines are surprisingly hard to beat** at multi-step horizons.

**Key Features:**
- **Rolling-origin validation**: Models only see observations prior to each forecast origin, preventing look-ahead leakage
- **Modular architecture**: Separate components for loading, transforms, models, evaluation, and diagnostics
- **Multiple model classes**: Random walk, ARIMA(p,d,q), and ConstantVar variance models in the committed experiment; ARCH and GARCH family classes are implemented in `src/models.py`
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

Note on execution: `notebooks/experiment.ipynb` is committed unexecuted (`execution_count: null`, output directory gitignored, no output CSVs). In `src/full_runner.py`, programmatic execution via `runner.run()` references `self.vol_ctor` on lines 64 and 92 while `__init__` sets `self.var_ctor`, requiring an attribute alignment before running end-to-end. Benchmark values and model selections are cited directly from `docs/project_report.pdf`.

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

## Context

Built for TMS088 (Financial Time Series) at Chalmers University of Technology, Spring 2025.

In a 6-person group project analyzing 7 commodity series across four tasks (data analysis, interpolation, extrapolation, and trading strategies), this repository implements the Task 3 extrapolation component:
- Rolling-origin validation framework for multi-step price forecasting
- Implementation of mean models (random walk with/without drift, ARIMA) and volatility models (ARCH, GARCH)
- Evaluation metrics and residual diagnostic plotting (ACF, squared ACF)

The full 26-page project report is available at [`docs/project_report.pdf`](docs/project_report.pdf).

**Report Findings:** In the report's model selection (Table 22), random walk baselines were selected for 5 of the 7 commodity series. For series where ARIMA or GARCH models provided the best fit (Tranquillity and Slingshots), error improvements over random walk were modest (<2% in most series).
