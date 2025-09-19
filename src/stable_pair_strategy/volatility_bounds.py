"""Utilities for computing dynamic price bounds for stable pair strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class BoundsConfig:
    """Configuration for computing rolling price bounds."""

    window: str = "7D"
    k_values: Iterable[float] = (0.3, 0.5, 0.8, 1.0, 1.2, 1.5)


def _prepare_log_price(data: pd.DataFrame) -> pd.Series:
    """Construct a time-indexed log-price series from raw swap data."""

    required_columns = {"timestamp", "token_a_amount", "token_b_amount"}
    missing = required_columns.difference(data.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {cols}")

    ts = pd.to_datetime(data["timestamp"], utc=True, format="mixed", errors="coerce")
    frame = (
        data.assign(timestamp=ts)
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    price_ratio = frame["token_a_amount"].astype(float) / frame["token_b_amount"].astype(float)
    log_price = np.log(price_ratio.replace(0, np.nan))
    return log_price.dropna()


def calculate_price_bounds(data: pd.DataFrame, config: BoundsConfig | None = None) -> pd.DataFrame:
    """Compute rolling price bounds for a range of K scaling factors.

    Parameters
    ----------
    data:
        Raw swap data containing timestamped token amounts.
    config:
        Optional configuration specifying the rolling window and K values.

    Returns
    -------
    pandas.DataFrame
        Time-indexed frame with the spot price and lower/upper bands for each K.
    """

    cfg = config or BoundsConfig()
    log_price = _prepare_log_price(data)

    if log_price.empty:
        raise ValueError("No valid log-price observations were found in the data frame.")

    window = cfg.window
    mu = log_price.rolling(window=window, min_periods=1).mean()
    sigma = log_price.rolling(window=window, min_periods=2).std(ddof=0).fillna(0.0)
    delta = log_price.diff().abs()
    ath = delta.rolling(window=window, min_periods=1).mean().fillna(0.0)

    output = pd.DataFrame(index=log_price.index)
    output["price"] = np.exp(log_price)
    output["mu"] = mu
    output["sigma"] = sigma
    output["ath"] = ath

    for k in cfg.k_values:
        width = np.maximum(k * sigma.to_numpy(), ath.to_numpy())
        lower = np.exp(mu.to_numpy() - width)
        upper = np.exp(mu.to_numpy() + width)
        label = f"k_{k:.1f}".replace(".", "p")
        output[f"lower_{label}"] = lower
        output[f"upper_{label}"] = upper

    return output


__all__: List[str] = ["BoundsConfig", "calculate_price_bounds"]
