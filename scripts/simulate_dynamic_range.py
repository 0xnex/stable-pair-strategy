"""Simulate dynamic price range adjustments following the updated strategy PRD."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd


PriceSource = Literal["price", "ratio"]


@dataclass
class SimulationConfig:
    k: float
    lookback: str = "7D"
    start: str = "2025-06-26T00:00:00+00:00"
    end: str = "2025-08-25T00:00:00+00:00"
    price_source: PriceSource = "price"
    out_of_range_threshold_minutes: int = 120
    reentry_debounce_minutes: int = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-glob",
        default="data/LBTC-WBTC_*.csv",
        help="Glob pattern identifying trade CSV files (default: data/LBTC-WBTC_*.csv).",
    )
    parser.add_argument(
        "--k",
        type=float,
        required=True,
        help="K multiplier used in the volatility band calculation.",
    )
    parser.add_argument(
        "--start",
        default="2025-06-26T00:00:00+00:00",
        help="Simulation window start (ISO timestamp, default 2025-06-26T00:00:00+00:00).",
    )
    parser.add_argument(
        "--end",
        default="2025-08-25T00:00:00+00:00",
        help="Simulation window end (ISO timestamp, default 2025-08-25T00:00:00+00:00).",
    )
    parser.add_argument(
        "--lookback",
        default="7D",
        help="Rolling lookback window for statistics (default 7D).",
    )
    parser.add_argument(
        "--price-source",
        choices=("price", "ratio"),
        default="price",
        help="Field used to derive spot price (default: price column).",
    )
    parser.add_argument(
        "--oor-threshold",
        type=int,
        default=120,
        help="Minutes outside the band before triggering a rebalance (default 120).",
    )
    parser.add_argument(
        "--reentry-debounce",
        type=int,
        default=5,
        help="Consecutive minutes inside the band required to reset the out-of-range timer (default 5).",
    )
    parser.add_argument(
        "--output",
        default="reports/dynamic_range_simulation.csv",
        help="Path for the summary CSV report.",
    )
    return parser.parse_args()


def load_trade_data(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise FileNotFoundError("No CSV files were found for the provided pattern.")
    data = pd.concat(frames, ignore_index=True)
    timestamps = pd.to_datetime(data["timestamp"], utc=True, format="mixed", errors="coerce")
    data = data.assign(timestamp=timestamps).dropna(subset=["timestamp"])
    return data.sort_values("timestamp").reset_index(drop=True)


def select_price_series(data: pd.DataFrame, source: PriceSource) -> pd.Series:
    if source == "price":
        series = data["price"].astype(float)
    else:
        ratio = data["token_a_amount"].astype(float) / data["token_b_amount"].astype(float)
        series = ratio.replace([math.inf, -math.inf], pd.NA)
    prices = pd.Series(series.values, index=data["timestamp"])
    return prices.dropna().sort_index()


def prepare_minute_prices(series: pd.Series) -> pd.Series:
    # Use the last observed trade price within each minute and forward-fill gaps
    minute = series.resample("1min").last().ffill()
    return minute.dropna()


def compute_rolling_bounds(prices: pd.Series, cfg: SimulationConfig) -> pd.DataFrame:
    log_price = prices.apply(lambda x: math.log(x) if x and x > 0 else float("nan"))
    mu = log_price.rolling(cfg.lookback).mean()
    sigma = log_price.rolling(cfg.lookback).std(ddof=0)
    ath = log_price.diff().abs().rolling(cfg.lookback).mean()
    width = pd.concat([cfg.k * sigma, ath], axis=1).max(axis=1)
    lower = (mu - width).apply(math.exp)
    upper = (mu + width).apply(math.exp)
    frame = pd.DataFrame({"price": prices, "lower": lower, "upper": upper})
    return frame.dropna()


def run_simulation(minute_prices: pd.Series, cfg: SimulationConfig) -> dict[str, float]:
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)

    # Ensure we have enough history for the initial lookback window
    required_history_start = start - pd.Timedelta(cfg.lookback)
    minute_prices = minute_prices.loc[(minute_prices.index >= required_history_start) & (minute_prices.index < end)]

    bands = compute_rolling_bounds(minute_prices, cfg)
    bands = bands.loc[(bands.index >= start) & (bands.index < end)]

    if bands.empty:
        raise ValueError("No data available within the requested simulation window after applying the lookback.")

    current_lower = bands["lower"].iloc[0]
    current_upper = bands["upper"].iloc[0]

    inside_minutes = 0
    out_of_range_minutes = 0
    inside_streak = 0
    adjustments = 0

    for timestamp, row in bands.iterrows():
        price = row["price"]
        in_band = current_lower <= price <= current_upper

        if in_band:
            inside_minutes += 1
            inside_streak += 1
            if inside_streak >= cfg.reentry_debounce_minutes:
                out_of_range_minutes = 0
        else:
            inside_streak = 0
            out_of_range_minutes += 1
            if out_of_range_minutes > cfg.out_of_range_threshold_minutes:
                adjustments += 1
                current_lower = row["lower"]
                current_upper = row["upper"]
                out_of_range_minutes = 0
                inside_streak = 0
                if not (current_lower <= price <= current_upper):
                    out_of_range_minutes = 1

    total_minutes = len(bands)
    coverage = inside_minutes / total_minutes if total_minutes else float("nan")

    return {
        "k": cfg.k,
        "price_source": cfg.price_source,
        "start": cfg.start,
        "end": cfg.end,
        "minutes_total": total_minutes,
        "minutes_inside": inside_minutes,
        "coverage": coverage,
        "adjustments": adjustments,
        "oor_threshold_minutes": cfg.out_of_range_threshold_minutes,
        "reentry_debounce_minutes": cfg.reentry_debounce_minutes,
    }


def main() -> None:
    args = parse_args()

    cfg = SimulationConfig(
        k=args.k,
        lookback=args.lookback,
        start=args.start,
        end=args.end,
        price_source=args.price_source,
        out_of_range_threshold_minutes=args.oor_threshold,
        reentry_debounce_minutes=args.reentry_debounce,
    )

    paths = sorted(Path(".").glob(args.data_glob))
    raw = load_trade_data(paths)
    price_series = select_price_series(raw, cfg.price_source)
    minute_prices = prepare_minute_prices(price_series)

    summary = run_simulation(minute_prices, cfg)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_path, index=False)

    print("Simulation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
