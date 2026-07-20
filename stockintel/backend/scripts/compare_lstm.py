"""Does the LSTM beat the tabular models, or the naive baseline?

Runs the same purged walk-forward protocol used for every other model, so the
comparison is apples-to-apples. Whatever it finds is what the platform reports.

Run:  .venv/Scripts/python.exe -m scripts.compare_lstm
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from app.backtesting.harness import run_walk_forward
from app.backtesting.splits import WalkForwardConfig
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.data.market.prices import fetch_history
from app.data.market.registry import get_market_provider
from app.features.targets import (
    OUTLOOK_5D,
    OUTLOOK_20D,
    align_features_and_target,
    build_target,
)
from app.features.technical import build_features
from app.models.baselines import MajorityClassBaseline
from app.models.lstm import LSTMModel
from app.models.tabular import LightGBMModel, LogisticRegressionModel

SYMBOLS = [("us", "AAPL"), ("us", "NVDA"), ("india", "RELIANCE.NS"), ("india", "TCS.NS")]
TARGETS = [OUTLOOK_5D, OUTLOOK_20D]

MODELS = [
    ("Majority baseline", MajorityClassBaseline),
    ("Logistic Regression", LogisticRegressionModel),
    ("LightGBM", LightGBMModel),
    ("LSTM", LSTMModel),
]


def main() -> int:
    configure_logging("ERROR")
    settings = get_settings()
    results: dict[str, dict] = {}
    by_model: dict[str, list[float]] = {name: [] for name, _ in MODELS}
    wins: dict[str, int] = {name: 0 for name, _ in MODELS}
    total_configs = 0

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
            config = WalkForwardConfig(n_folds=5, horizon=spec.horizon_days)
            total_configs += 1

            print(f"\n{'=' * 84}")
            print(f"{symbol}  |  {spec.name}  |  {len(x)} rows x {x.shape[1]} features")
            print(f"{'=' * 84}")
            print(f"{'model':<22} {'acc':>8} {'base':>8} {'skill':>9} {'AUC':>7} "
                  f"{'MCC':>7} {'secs':>7}  {'folds won':>10}")
            print("-" * 84)

            for label, factory in MODELS:
                started = time.perf_counter()
                try:
                    result = run_walk_forward(
                        lambda f=factory: f(spec.n_classes, settings.random_seed),
                        x, y, spec.class_labels, config,
                    )
                except Exception as exc:
                    print(f"{label:<22} FAILED: {str(exc)[:50]}")
                    continue

                elapsed = time.perf_counter() - started
                agg = result.aggregate
                auc = agg.get("roc_auc_mean")

                by_model[label].append(agg["skill_score_mean"])
                if agg["accuracy_mean"] > agg["baseline_accuracy_mean"]:
                    wins[label] += 1

                print(
                    f"{label:<22} "
                    f"{agg['accuracy_mean']:>8.4f} "
                    f"{agg['baseline_accuracy_mean']:>8.4f} "
                    f"{agg['skill_score_mean']:>+9.4f} "
                    f"{(f'{auc:.3f}' if auc else 'n/a'):>7} "
                    f"{agg['matthews_corrcoef_mean']:>+7.3f} "
                    f"{elapsed:>7.1f}  {agg['consistency']:>10}"
                )

                results[f"{symbol}|{spec.name}|{label}"] = agg

    # --- Verdict -----------------------------------------------------------
    print(f"\n{'=' * 84}")
    print(f"VERDICT — mean skill across {total_configs} configurations")
    print(f"{'=' * 84}")
    print(f"{'model':<22} {'mean skill':>12} {'configs beating baseline':>26}")
    print("-" * 84)

    for label, _ in MODELS:
        skills = by_model[label]
        if not skills:
            continue
        print(
            f"{label:<22} {np.mean(skills):>+12.4f} "
            f"{f'{wins[label]}/{len(skills)}':>26}"
        )

    lstm_skill = np.mean(by_model["LSTM"]) if by_model["LSTM"] else float("nan")
    lgbm_skill = np.mean(by_model["LightGBM"]) if by_model["LightGBM"] else float("nan")
    print()
    if np.isnan(lstm_skill) or np.isnan(lgbm_skill):
        print("Insufficient results to compare.")
    elif lstm_skill > lgbm_skill:
        print(f"LSTM outperformed LightGBM by {lstm_skill - lgbm_skill:+.4f} mean skill.")
    else:
        print(f"LSTM did NOT outperform LightGBM ({lstm_skill:+.4f} vs {lgbm_skill:+.4f}).")

    output = settings.resolved_artifact_dir / "lstm_comparison.json"
    output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nFull report written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
