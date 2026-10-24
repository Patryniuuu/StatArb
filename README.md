# Statistical Arbitrage Through Cointegration: Pairs Trading and Johansen Baskets

An educational quantitative-finance project exploring cointegration-based statistical arbitrage through mathematical implementation, rolling walk-forward backtesting, and empirical analysis.

The project implements and compares two approaches to relative-value trading:

- **Bivariate pairs trading**, based on the Engle–Granger cointegration framework.
- **Three-asset synthetic baskets**, selected using the Johansen trace procedure.

The main objective is to understand and implement an end-to-end quantitative research workflow, from econometric modeling and signal generation to execution assumptions, transaction costs, and performance evaluation.

> **Research prototype — not production trading software.** The reported results are historical backtest outputs, not evidence of a guaranteed or persistent investment edge.

## Project Overview

Individual stock prices are generally modeled as non-stationary time series. Cointegration analysis investigates whether a linear combination of multiple price series exhibits stationary behavior, potentially creating opportunities for mean-reversion strategies.

This project explores how such relationships can be identified, converted into trading signals, and evaluated under a rolling out-of-sample testing framework.

The implementation uses a predefined universe of 36 US equities grouped into four sectors:

- Technology and semiconductors
- Banks and financial services
- Energy and oil
- Consumer staples

The research focuses on the relationship between econometric model selection, mean-reversion dynamics, execution assumptions, and transaction costs.

## Key Features

- **Engle–Granger pairs trading:** OLS estimation, residual-spread construction, ADF testing, and additional statistical filters.
- **Johansen cointegration baskets:** Analysis of three-asset combinations using the Johansen trace procedure and estimated cointegrating vectors.
- **Multi-stage candidate filtering:** Return correlation, regression fit, hedge-ratio bounds, half-life estimates, stationarity tests, and ECM-based adjustment criteria for pairs.
- **Walk-forward evaluation:** Rolling training and out-of-sample testing windows configured separately for daily and hourly data.
- **Rolling Z-scores:** Dynamic spread normalization and a state-based trading signal generator.
- **Execution simulation:** Signals generated from closing-price data and trades mapped to the next bar's opening price.
- **Transaction-cost modeling:** Configurable commission, slippage, and annualized short-borrowing assumptions.
- **Performance visualization:** Matplotlib charts for portfolio equity, drawdowns, Z-scores, and strategy comparisons.

## Methodology

### 1. Bivariate Pairs Trading

The Engle–Granger workflow estimates a linear relationship between the log-prices of two assets using ordinary least squares (OLS). The regression residuals define the candidate spread.

Candidate pairs are evaluated using a sequence of filters, including:

- Return correlation
- OLS beta and \(R^2\) constraints
- An AR(1)-based half-life estimate
- An augmented Dickey–Fuller (ADF) test with MacKinnon cointegration p-values
- Error-correction-based criteria assessing the adjustment of the two assets

Pairs that survive the filters are ranked and selected subject to portfolio-selection constraints.

### 2. Multivariate Johansen Baskets

The basket strategy examines combinations of three assets within the predefined sectors.

The Johansen trace procedure is used to test for cointegration. Candidate baskets are then evaluated using their estimated cointegrating vectors and the mean-reversion characteristics of the resulting synthetic spreads.

Surviving baskets are ranked and passed to the portfolio-selection procedure.

### 3. Signal Generation

The signal generator operates on rolling Z-scores of the candidate spreads.

Depending on the configured parameters, the strategy can open a position when the spread crosses an entry threshold and close it when a specified exit condition is met. The implementation also includes stop-loss, holding-period, and end-of-window liquidation rules.

### 4. Walk-Forward Backtesting

The system separates model estimation and candidate selection from the subsequent out-of-sample trading window.

The current configuration defines the following profiles:

| Parameter | Daily profile (`1d`) | Hourly profile (`1h`) |
|---|---|---|
| Training window | 12 months | 3 months |
| Test window | 3 months | 1 month |
| Configured OOS start | 2020-01-01 | 2025-05-01 |
| Configured OOS end | 2025-12-31 | 2026-09-30 |

These dates describe the current values in `config.py`; they may differ from the configuration used to generate the results in the accompanying report.

The walk-forward design is intended to reduce look-ahead bias by separating training and testing periods and by modeling execution at the next bar's open. It does not, by itself, eliminate every possible source of information leakage, model-selection bias, or overfitting.

### 5. Execution and Transaction Costs

The current default configuration specifies:

| Parameter | Default |
|---|---:|
| Initial capital | $100,000 |
| Commission | 3 basis points per side |
| Slippage | 3 basis points per side |
| Annualized short-borrowing rate | 1% |
| Base allocation per trade | 10% |

These values are simulation assumptions, not broker-verified quotes. Actual commissions, bid-ask spreads, market impact, liquidity, and stock-borrow availability can differ materially.

## Selected Results Reported in the Project Report

The following figures are taken from the project's report and refer to specific historical experiment configurations.

| Scenario | Strategy | Interval | Trades | Net PnL | Profit factor | Sharpe ratio |
|---|---|---|---:|---:|---:|---:|
| S1 | Engle–Granger pairs, excluding Technology | Daily | 53 | -$2,821.92 | 0.43 | -0.92 |
| S8 | Johansen basket, all four sectors | Daily | 103 | +$2,618.08 | 1.51 | 0.66 |
| S6 | Johansen basket | Hourly | 142 | -$1,587.25 | 0.63 | -1.54 |

The reported results illustrate that performance depends on the selected model, market universe, time scale, and transaction-cost assumptions. In particular, the reported hourly strategy loses money after modeled costs despite positive gross PnL in the basket scenario.

**Reproducibility note:** These figures should not be assumed to reproduce from the current repository configuration without matching the code version, input data, parameters, and experiment dates used to produce the report.

## Methodological Limitations

This project is an implementation and evaluation exercise, rather than a formal demonstration of a tradable market anomaly.

Important limitations include:

- **Realized-trade risk metrics:** The reported Sharpe ratio and maximum drawdown are calculated from realized trade PnL rather than continuous daily mark-to-market revaluation. Interim losses on open positions may therefore be understated.
- **Portfolio neutrality:** Cointegrating weights do not automatically guarantee dollar neutrality or neutrality to market beta.
- **Multiple testing:** The candidate-filtering cascade is a heuristic. It does not provide a formal guarantee that false discoveries are controlled.
- **Historical information:** The predefined stock universe and market-cap-based pair orientation are not necessarily based on point-in-time historical data. This can introduce survivorship or look-ahead concerns.
- **Simplified execution model:** Fixed transaction-cost and borrowing assumptions do not capture all real-world execution constraints.
- **Historical backtest scope:** The tested universe, time periods, and parameter choices limit the generality of the conclusions.

The results should be interpreted as findings from the tested configurations, not as proof that one method is universally superior or that the strategy will remain profitable in live markets.

## Project Structure

```text
StatArb/
├── data/
│   └── data_loader.py          # Market data retrieval and preprocessing
├── research/
│   ├── cointegration_math.py   # OLS, ADF, Johansen, half-life, and ECM calculations
│   └── market_scanner.py       # Candidate-pair and basket scanning
├── execution/
│   ├── zscore_calculator.py    # Rolling spread normalization
│   ├── signal_generator.py     # Entry and exit state machine
│   └── backtest_engine.py      # Trade PnL, costs, and performance metrics
├── visualization/
│   └── plotter.py              # Equity, drawdown, and strategy-comparison plots
├── config.py                   # Intervals, dates, filters, and execution parameters
├── main.py                     # Main walk-forward execution entry point
├── requirements.txt            # Python dependencies
├── test.py
├── test.ipynb
└── test_backtest_engine.ipynb
```

## Installation

### Requirements

- Python 3.10 or newer
- Internet access for downloading market data through `yfinance`

Create and activate a virtual environment.

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project's direct dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scipy statsmodels yfinance
```

For reproducible installations, the tested dependency versions should be recorded in `requirements.txt`.

## Usage

Run the main pipeline:

```bash
python main.py
```

The script runs the configured walk-forward pipeline and displays strategy performance visualizations.

Before running an experiment, review `config.py`. Relevant parameters include:

- `ACTIVE_INTERVAL`: daily (`1d`) or hourly (`1h`) data
- `EXECUTION_MODE`: pairs, baskets, or both
- `TOP_K_PAIRS` and `TOP_K_BASKETS`: number of selected candidates
- `ACTIVE_SECTORS`: sectors included in the experiment
- Training and out-of-sample date ranges
- Statistical filters, signal thresholds, and transaction-cost assumptions

The project retrieves market data through Yahoo Finance. Data availability, historical coverage, and returned values may change. Exact reproduction of an earlier experiment requires matching its configuration and data inputs.

## Authors

- **Patryk Kubik** — [GitHub](https://github.com/Patryniuuu)
- **Bartosz Obłoj** - [GitHub](https://github.com/oblojb)


