"""Walk-forward model comparison across markets and horizons.

Run:  .venv/Scripts/python.exe -m scripts.compare_models

Prints the honest out-of-sample answer to "does anything beat the baseline?"
and writes the full report to .artifacts/model_comparison.json. Whatever it
finds is what the dashboard will report.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from app.backtesting.harness import build_regime_labels, run_walk_forward
from app.backtesting.splits import WalkForwardConfig
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.data.market.prices import fetch_history
from app.data.market.registry import get_market_provider
from app.features.targets import DIRECTION_1D, OUTLOOK_5D, align_features_and_target, build_target
from app.features.technical import build_features
from app.models.baselines import MajorityClassBaseline, MomentumBaseline
from app.models.tabular import LightGBMModel, LogisticRegressionModel, RandomForestModel

SYMBOLS = [("us", "AAPL"), ("us", "NVDA"), ("india", "RELIANCE.NS"), ("india", "TCS.NS")]
TARGETS = [DIRECTION_1D, OUTLOOK_5D]

MODEL_FACTORIES = [
    ("Majority class", MajorityClassBaseline),
    ("Momentum persistence", MomentumBaseline),
    ("Logistic Regression", LogisticRegressionModel),
    ("Random Forest", RandomForestModel),
    ("LightGBM", LightGBMModel),
]


def main() -> int:
    configure_logging("WARNING")
    settings = get_settings()
    all_results: dict[str, dict] = {}

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
        regimes = build_regime_labels(features)

        for spec in TARGETS:
            target, _ = build_target(history.frame["Close"], spec)
            x, y = align_features_and_target(features, target)
            y = y.to_numpy().astype(int)

            config = WalkForwardConfig(n_folds=5, horizon=spec.horizon_days)

            print(f"\n{'=' * 78}")
            print(f"{symbol}  |  {spec.name}  |  {len(x)} rows  "
                  f"{x.index[0].date()} -> {x.index[-1].date()}")
            print(f"{'=' * 78}")
            print(f"{'model':<24} {'acc':>7} {'base':>7} {'skill':>8} "
                  f"{'MCC':>7} {'AUC':>7}  {'folds won':>10}")
            print("-" * 78)

            key = f"{symbol}|{spec.name}"
            all_results[key] = {"symbol": symbol, "market": market_id,
                                "target": spec.to_dict(), "rows": len(x), "models": {}}

            for label, factory in MODEL_FACTORIES:
                try:
                    result = run_walk_forward(
                        lambda f=factory: f(spec.n_classes, settings.random_seed),
                        x, y, spec.class_labels, config, regime_series=regimes,
                    )
                except Exception as exc:
                    print(f"{label:<24} FAILED: {exc}")
                    continue

                agg = result.aggregate
                auc = agg.get("roc_auc_mean")
                print(
                    f"{label:<24} "
                    f"{agg['accuracy_mean']:>7.4f} "
                    f"{agg['baseline_accuracy_mean']:>7.4f} "
                    f"{agg['skill_score_mean']:>+8.4f} "
                    f"{agg['matthews_corrcoef_mean']:>+7.3f} "
                    f"{(f'{auc:.3f}' if auc else '   n/a'):>7}  "
                    f"{agg['consistency']:>10}"
                )
                all_results[key]["models"][label] = result.to_dict()

    output = settings.resolved_artifact_dir / "model_comparison.json"
    output.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nFull report written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
