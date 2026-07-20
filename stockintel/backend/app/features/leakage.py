"""Empirical look-ahead detection.

Feature code is easy to *believe* is causal and easy to get subtly wrong: one
`center=True`, one `shift(-1)`, one `bfill` and the reported accuracy jumps to
something that looks like a breakthrough and is actually a bug. Comments do not
catch that. This module does, by testing the property directly.

The probe: a feature at time `t` is causal if truncating the input series after
`t` does not change it. Recompute features from a truncated frame and compare
the tail value against the same cell computed from the full history. Any
disagreement means the feature saw the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)

FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class LeakageReport:
    """Result of a look-ahead probe."""

    passed: bool
    checked_columns: int
    checked_positions: int
    offending_columns: tuple[str, ...]
    max_absolute_difference: float

    def summary(self) -> str:
        if self.passed:
            return (
                f"No look-ahead detected: {self.checked_columns} feature(s) "
                f"stable across {self.checked_positions} truncation point(s)."
            )
        return (
            f"LOOK-AHEAD DETECTED in {len(self.offending_columns)} column(s): "
            f"{', '.join(self.offending_columns[:8])} "
            f"(max difference {self.max_absolute_difference:.3e})."
        )


def probe_lookahead(
    frame: pd.DataFrame,
    build: FeatureBuilder,
    *,
    probe_points: int = 5,
    tolerance: float = 1e-9,
) -> LeakageReport:
    """Check that features depend only on past and present data.

    Args:
        frame: Full OHLCV history.
        build: Feature builder taking a frame and returning features indexed
            identically (call with `drop_warmup=False`).
        probe_points: How many truncation points to test, spread over the
            final third of the series.
        tolerance: Absolute difference treated as floating-point noise.

    Returns:
        A `LeakageReport`; inspect `.passed`.
    """
    full_features = build(frame)

    start = int(len(frame) * 0.66)
    end = len(frame) - 1
    if end <= start:
        raise ValueError("Series too short to probe for look-ahead.")

    cut_positions = np.linspace(start, end, num=probe_points, dtype=int)

    offenders: set[str] = set()
    max_difference = 0.0
    positions_checked = 0

    for cut in cut_positions:
        truncated_features = build(frame.iloc[: cut + 1])
        if truncated_features.empty:
            continue

        positions_checked += 1
        timestamp = frame.index[cut]

        if timestamp not in truncated_features.index or timestamp not in full_features.index:
            continue

        truncated_row = truncated_features.loc[timestamp]
        full_row = full_features.loc[timestamp]

        for column in full_features.columns:
            if column not in truncated_row:
                continue
            a, b = full_row[column], truncated_row[column]

            # NaN on both sides is agreement (both inside the warm-up window).
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) != pd.isna(b):
                offenders.add(column)
                continue

            difference = abs(float(a) - float(b))
            max_difference = max(max_difference, difference)
            if difference > tolerance:
                offenders.add(column)

    report = LeakageReport(
        passed=not offenders,
        checked_columns=full_features.shape[1],
        checked_positions=positions_checked,
        offending_columns=tuple(sorted(offenders)),
        max_absolute_difference=max_difference,
    )

    if report.passed:
        logger.info(report.summary())
    else:
        logger.error(report.summary())
    return report


def assert_no_lookahead(frame: pd.DataFrame, build: FeatureBuilder, **kwargs) -> None:
    """Raise if any feature reads the future. For use in tests and training."""
    report = probe_lookahead(frame, build, **kwargs)
    if not report.passed:
        raise AssertionError(report.summary())
