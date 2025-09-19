"""Stable pair strategy analytics package."""

from .volatility_bounds import BoundsConfig, calculate_price_bounds

__all__ = ["BoundsConfig", "calculate_price_bounds"]
