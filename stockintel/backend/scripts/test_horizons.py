"""Does a longer horizon plus abstention produce real out-of-sample skill?

Tests the hypothesis from the 1d/5d comparison: ranking signal exists
(AUC 0.52-0.55) but the argmax threshold destroys it, and longer horizons
should carry a better signal-to-noise ratio.

Reports forced-choice and selective accuracy side by side, so the effect of
abstention alone is visible. Effective sample size is printed for every
horizon, because overlapping labels make the nominal row count misleading.

Run:  .venv/Scripts/python.exe -m scripts.test_horizons
"""

from __future__ import annotations

import json
import sys

import numpy as np

from app.backtesting.harness import run_selective_walk_forward, run_walk_forward
from app.backtesting.splits import WalkForwardConfig
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.data.market.prices import fetch_history
from app.data.market.registry import get_market_provider
from app.features.targets import (
    OUTLOOK_5D,
    OUTLOOK_10D,
    OUTLOOK_20D,
    align_features_and_target,
    build_target,
    class_balance,
    effective_sample_size,
)
from app.features.technical import build_features
from app.models.tabular import LightGBMModel, LogisticRegressionModel

SYMBOLS = [("us", "AAPL"), ("us", "NVDA"), ("us", "MSFT"),
           ("india", "RELIANCE.NS"), ("india", "TCS.NS"), ("india", "INFY.NS")]
TARGETS = [OUTLOOK_5D, OUTLOOK_10D, OUTLOOK_20D]
MODELS = [("LightGBM", LightGBMModel), ("LogReg", LogisticRegressionModel)]


def main() -> int:
    configure_logging("ERROR")
    settings = get_settings()
    results: dict[str, dict] = {}
    summary_rows: list[tuple] = []

    for market_id, symbol in SYMBOLS:
        provider = get_market_provider(market_id)
        try:
            history = fetch_history(symbol, period="10y")
        except Exception as exc:
            print(f"SKIP {symbol}: {exc}")
            continue

        features, _ = build_features(
            history.frame,
            trading_days_per_year=provider.conventions.trading_days_per_year,
        )

        for spec in TARGETS:
            target, _ = build_target(history.frame["Close"], spec)
            x, y_series = align_features_and_target(features, target)
            y = y_series.to_numpy().astype(int)

            if len(x) < 600:
                continue

            balance = class_balance(y_series, spec)
            ess = effective_sample_size(len(x), spec.horizon_days)
            config = WalkForwardConfig(n_folds=5, horizon=spec.horizon_days)

            print(f"\n{'=' * 92}")
            print(f"{symbol}  |  {spec.name}  |  {len(x)} rows  "
                  f"(effective n ~ {ess} independent windows)")
            print(f"  class balance: " + "  ".join(f"{k}={v:.3f}" for k, v in balance.items()))
            print(f"{'=' * 92}")
            print(f"{'model':<12} {'forced':>8} {'base':>8} {'skill':>8} {'AUC':>7}  |  "
                  f"{'cover':>7} {'sel.acc':>8} {'sel.base':>9} {'sel.skill':>10} {'edge':>6}")
            print("-" * 92)

            for label, factory in MODELS:
                try:
                    forced = run_walk_forward(
                        lambda f=factory: f(spec.n_classes, settings.random_seed),
                        x, y, spec.class_labels, config,
                    )
                    selective = run_selective_walk_forward(
                        lambda f=factory: f(spec.n_classes, settings.random_seed),
                        x, y, spec.class_labels, config,
                    )
                except Exception as exc:
                    print(f"{label:<12} FAILED: {exc}")
                    continue

                agg = forced.aggregate
                auc = agg.get("roc_auc_mean")

                if selective.get("abstains_always") or not selective.get("available"):
                    sel_text = f"{'0.00':>7} {'—':>8} {'—':>9} {'—':>10} {'no':>6}"
                    sel_record = selective
                else:
                    sel_text = (
                        f"{selective['coverage']:>7.3f} "
                        f"{selective['selective_accuracy']:>8.4f} "
                        f"{selective['baseline_accuracy']:>9.4f} "
                        f"{selective['skill_score']:>+10.4f} "
                        f"{('YES' if selective['has_edge'] else 'no'):>6}"
                    )
                    sel_record = selective
                    summary_rows.append((
                        symbol, spec.name, label,
                        selective["coverage"], selective["selective_accuracy"],
                        selective["baseline_accuracy"], selective["skill_score"],
                        selective["has_edge"], ess,
                    ))

                print(
                    f"{label:<12} "
                    f"{agg['accuracy_mean']:>8.4f} "
                    f"{agg['baseline_accuracy_mean']:>8.4f} "
                    f"{agg['skill_score_mean']:>+8.4f} "
                    f"{(f'{auc:.3f}' if auc else 'n/a'):>7}  |  {sel_text}"
                )

                results[f"{symbol}|{spec.name}|{label}"] = {
                    "forced": agg, "selective": sel_record,
                    "effective_sample_size": ess, "class_balance": balance,
                }

    # --- Verdict -----------------------------------------------------------
    print(f"\n{'=' * 92}")
    print("VERDICT")
    print(f"{'=' * 92}")

    if not summary_rows:
        print("No configuration produced a usable selective model.")
    else:
        by_horizon: dict[str, list] = {}
        for row in summary_rows:
            by_horizon.setdefault(row[1], []).append(row)

        for horizon_name in sorted(by_horizon):
            rows = by_horizon[horizon_name]
            with_edge = [r for r in rows if r[7]]
            mean_skill = float(np.mean([r[6] for r in rows]))
            mean_cover = float(np.mean([r[3] for r in rows]))
            print(
                f"{horizon_name:<14} {len(with_edge)}/{len(rows)} configs show an edge  |  "
                f"mean selective skill {mean_skill:+.4f}  |  mean coverage {mean_cover:.1%}"
            )

    output = settings.resolved_artifact_dir / "horizon_comparison.json"
    output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nFull report written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
