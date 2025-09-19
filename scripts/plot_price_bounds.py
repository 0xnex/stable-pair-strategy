"""Generate price volatility bands for the stable pair strategy test data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stable_pair_strategy.volatility_bounds import BoundsConfig, calculate_price_bounds


def load_trade_data(paths: Iterable[Path]) -> pd.DataFrame:
    """Read and concatenate swap records from the provided CSV paths."""

    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise FileNotFoundError("No CSV files were found for the provided pattern.")
    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-glob",
        default="data/LBTC-WBTC_*.csv",
        help=(
            "Glob pattern identifying trade CSV files (default: data/LBTC-WBTC_*.csv). "
            "Override if you want to target a different dataset."
        ),
    )
    parser.add_argument(
        "--output",
        default="reports/price_bounds.png",
        help="Path to save the generated chart.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(Path(".").glob(args.data_glob))
    data = load_trade_data(paths)

    config = BoundsConfig(k_values=(0.3, 0.5, 0.8, 1.0, 1.2, 1.5))
    counter_thresholds = (30, 45, 60, 75, 90, 120)

    bounds = calculate_price_bounds(data, config=config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(bounds.index, bounds["price"], label="Spot price", color="black", linewidth=1.5)

    for k in config.k_values:
        label = f"k_{k:.1f}".replace(".", "p")
        lower_col = f"lower_{label}"
        upper_col = f"upper_{label}"
        ax.plot(bounds.index, bounds[lower_col], label=f"Lower K={k}")
        ax.plot(bounds.index, bounds[upper_col], label=f"Upper K={k}")

    ax.set_title("Price trajectory with dynamic bounds")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price (token_a / token_b)")
    ax.legend(loc="upper right", ncol=2, fontsize="small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    inside_map_swaps = build_inside_map(bounds, config.k_values)
    inside_map_minutes = resample_inside_map(inside_map_swaps, freq="1min")

    run_lengths_minutes = compute_outside_run_lengths(inside_map_minutes)

    coverage = summarise_coverage(inside_map_swaps, inside_map_minutes)
    coverage_path = output_path.with_name(f"{output_path.stem}_coverage.csv")
    coverage.to_csv(coverage_path, index=False)

    reentry = summarise_run_lengths(run_lengths_minutes, basis="minute")
    reentry_path = output_path.with_name(f"{output_path.stem}_reentry.csv")
    reentry.to_csv(reentry_path, index=False)

    rebalance = summarise_rebalance_thresholds(
        inside_map_minutes, run_lengths_minutes, counter_thresholds
    )
    rebalance_path = output_path.with_name(f"{output_path.stem}_rebalance.csv")
    rebalance.to_csv(rebalance_path, index=False)

    print("Price coverage summary (swap + minute basis):")
    print(coverage.to_string(index=False))
    print("\nMinute-based re-entry statistics (consecutive minutes outside before re-entry):")
    print(reentry.to_string(index=False))
    print("\nRebalance triggers and exposure stats (minute counters):")
    print(rebalance.to_string(index=False))


def build_inside_map(bounds: pd.DataFrame, k_values: Iterable[float]) -> dict[float, pd.Series]:
    """Return a mapping of K to boolean series indicating in-range swaps."""

    price = bounds["price"]
    inside_map: dict[float, pd.Series] = {}
    for k in k_values:
        label = f"k_{k:.1f}".replace(".", "p")
        lower = bounds[f"lower_{label}"]
        upper = bounds[f"upper_{label}"]
        inside_map[k] = price.between(lower, upper, inclusive="both")

    return inside_map


def compute_coverage(inside_map: dict[float, pd.Series]) -> pd.DataFrame:
    """Calculate the share of swaps whose price sits within each K-band."""

    if not inside_map:
        return pd.DataFrame(columns=["k", "inside_count", "total", "coverage"])

    total = len(next(iter(inside_map.values())))
    records = []
    for k, inside in inside_map.items():
        inside_count = int(inside.sum())
        coverage = inside_count / total if total else float("nan")
        records.append(
            {
                "k": k,
                "inside_count": inside_count,
                "total": total,
                "coverage": coverage,
            }
        )

    return pd.DataFrame.from_records(records)


def resample_inside_map(
    inside_map: dict[float, pd.Series], freq: str = "1min"
) -> dict[float, pd.Series]:
    """Resample inside/outside indicators onto a regular time grid."""

    resampled: dict[float, pd.Series] = {}
    for k, series in inside_map.items():
        # ensure monotonic index and drop duplicate timestamps before resampling
        ordered = series.sort_index()
        deduped = ordered[~ordered.index.duplicated(keep="last")]
        resampled_series = deduped.resample(freq).ffill().dropna()
        resampled[k] = resampled_series.astype(bool)
    return resampled


def summarise_coverage(
    inside_swaps: dict[float, pd.Series],
    inside_minutes: dict[float, pd.Series],
) -> pd.DataFrame:
    """Combine swap-level and minute-level coverage statistics."""

    coverage_swap = compute_coverage(inside_swaps)
    coverage_swap["basis"] = "swap"

    coverage_minute = compute_coverage(inside_minutes)
    coverage_minute["basis"] = "minute"

    return pd.concat([coverage_swap, coverage_minute], ignore_index=True)


def compute_outside_run_lengths(inside_map: dict[float, pd.Series]) -> dict[float, list[int]]:
    """Collect run lengths (in swaps) when price stays outside the band."""

    run_lengths: dict[float, list[int]] = {}
    for k, inside in inside_map.items():
        run_lengths[k] = _extract_outside_runs(inside)

    return run_lengths


def _extract_outside_runs(inside: pd.Series) -> list[int]:
    """Return lengths of all consecutive outside stretches."""

    runs: list[int] = []
    count = 0
    for flag in inside:
        if not flag:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if count:
        runs.append(count)
    return runs


def summarise_run_lengths(
    run_lengths: dict[float, list[int]], *, basis: str
) -> pd.DataFrame:
    """Summarise re-entry characteristics for each K band."""

    records = []
    for k, lengths in run_lengths.items():
        if lengths:
            series = pd.Series(lengths, dtype="float")
            records.append(
                {
                    "k": k,
                    "events": len(lengths),
                    f"mean_{basis}s_outside": series.mean(),
                    f"median_{basis}s_outside": series.median(),
                    f"max_{basis}s_outside": series.max(),
                }
            )
        else:
            records.append(
                {
                    "k": k,
                    "events": 0,
                    f"mean_{basis}s_outside": float("nan"),
                    f"median_{basis}s_outside": float("nan"),
                    f"max_{basis}s_outside": float("nan"),
                }
            )

    frame = pd.DataFrame.from_records(records)
    frame["basis"] = basis
    return frame


def summarise_rebalance_thresholds(
    inside_map: dict[float, pd.Series],
    run_lengths: dict[float, list[int]],
    thresholds: Iterable[int],
) -> pd.DataFrame:
    """Aggregate rebalance metrics for each K and counter threshold."""

    records = []
    for k, inside in inside_map.items():
        lengths = run_lengths.get(k, [])
        total = len(inside)
        inside_count = int(inside.sum())
        outside_count = total - inside_count
        max_outside = max(lengths) if lengths else 0

        for threshold in thresholds:
            rebalance_count = sum(length > threshold for length in lengths)
            records.append(
                {
                    "k": k,
                    "counter_threshold": threshold,
                    "rebalance_count": rebalance_count,
                    "outside_minutes": outside_count,
                    "inside_minutes": inside_count,
                    "total_minutes": total,
                    "max_consecutive_outside_minutes": max_outside,
                }
            )

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    main()
