# Stable Pair Strategy Analytics

This project provides analytics utilities and a plotting workflow for evaluating dynamic price bands used by a stable pair liquidity strategy. It ingests swap-level trade data, computes rolling price bounds, and measures how often prices remain within those bands under a range of parameters.

## Features
- Calculate rolling log-price statistics (mean, volatility, ATH) and derive upper/lower bounds for configurable `K` multipliers.
- Resample trade data to minute cadence to evaluate monitoring counters and rebalance triggers defined in the product requirements.
- Generate visualizations and CSV reports covering price trajectory, in-band coverage, re-entry durations, and counter breaches.

## Getting Started
1. Install dependencies with [Poetry](https://python-poetry.org/):
   ```bash
   poetry install
   ```
2. Run the analytics script against the bundled LBTC/WBTC swap data:
   ```bash
   poetry run python scripts/plot_price_bounds.py
   ```
   Outputs are written under `reports/`:
   - `price_bounds.png` — price curve with dynamic bounds for each `K`.
   - `price_bounds_coverage.csv` — swap and minute coverage ratios.
   - `price_bounds_reentry.csv` — minute-based outside streak summaries.
   - `price_bounds_rebalance.csv` — rebalance counts for each `(K, counter)` pair.

## Command Line Interface
`scripts/plot_price_bounds.py` accepts the following options:

- `--data-glob`: glob pattern for trade CSV files. Defaults to `data/LBTC-WBTC_*.csv`.
- `--output`: path for the generated plot (PNG). Defaults to `reports/price_bounds.png`. Associated CSV reports inherit the same stem (e.g., `price_bounds_coverage.csv`).

Example:
```bash
poetry run python scripts/plot_price_bounds.py \
  --data-glob "data/LBTC-WBTC_*.csv" \
  --output "reports/lbtc_wbtc_bounds.png"
```

## Repository Layout
- `src/stable_pair_strategy/` — analytics library (`calculate_price_bounds`, etc.).
- `scripts/` — executable helpers, including the plotting CLI.
- `data/` — sample swap datasets used for local testing.
- `reports/` — generated plots and CSV summaries.

