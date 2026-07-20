"""Final head-to-head: does the ensemble beat its own components, or a baseline?

Every model runs the identical purged walk-forward protocol. The ensemble's
inner stacking folds are nested inside each outer training window, so its
score is directly comparable to the single models'.

Run:  .venv/Scripts/python.exe -m scripts.compare_final
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
from app.models.ensemble import EnsembleConfig, StackedEnsemble
from app.models.tabular import LightGBMModel, LogisticRegressionModel

SYMBOLS = [("us", "AAPL"), ("us", "NVDA"), ("us", "MSFT"),
           ("india", "RELIANCE.NS"), ("india", "TCS.NS")]
TARGETS = [OUTLOOK_5D, OUTLOOK_20D]


def build_models(n_classes: int, seed: int, horizon: int):
    def ensemble():
        return StackedEnsemble(
            n_classes, seed,
            base_factories=[
                ("logreg", lambda: LogisticRegressionModel(n_classes, seed)),
                ("lgbm", lambda: LightGBMModel(n_classes, seed)),
            ],
            config=EnsembleConfig(inner_folds=3, horizon=horizon),
        )

    return [
        ("Majority baseline", lambda: MajorityClassBaseline(n_classes, seed)),
        ("Logistic Regression", lambda: LogisticRegressionModel(n_classes, seed)),
        ("LightGBM", lambda: LightGBMModel(n_classes, seed)),
        ("Most Possible (stack)", ensemble),
    ]


def main() -> int:
    configure_logging("ERROR")
    settings = get_settings()

    results: dict[str, dict] = {}
    skills: dict[str, list[float]] = {}
    accuracies: dict[str, list[float]] = {}
    wins: dict[str, int] = {}
    stack_reports: list[dict] = []

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

            print(f"\n{'=' * 86}")
            print(f"{symbol}  |  {spec.name}  |  {len(x)} rows")
            print(f"{'=' * 86}")
            print(f"{'model':<24} {'acc':>8} {'base':>8} {'skill':>9} {'AUC':>7} "
                  f"{'MCC':>7} {'secs':>7}  {'folds won':>10}")
            print("-" * 86)

            for label, factory in build_models(
                spec.n_classes, settings.random_seed, spec.horizon_days
            ):
                started = time.perf_counter()
                try:
                    result = run_walk_forward(
                        factory, x, y, spec.class_labels, config
                    )
                except Exception as exc:
                    print(f"{label:<24} FAILED: {str(exc)[:44]}")
                    continue

                elapsed = time.perf_counter() - started
                agg = result.aggregate
                auc = agg.get("roc_auc_mean")

                skills.setdefault(label, []).append(agg["skill_score_mean"])
                accuracies.setdefault(label, []).append(agg["accuracy_mean"])
                wins.setdefault(label, 0)
                if agg["accuracy_mean"] > agg["baseline_accuracy_mean"]:
                    wins[label] += 1

                print(
                    f"{label:<24} "
                    f"{agg['accuracy_mean']:>8.4f} "
                    f"{agg['baseline_accuracy_mean']:>8.4f} "
                    f"{agg['skill_score_mean']:>+9.4f} "
                    f"{(f'{auc:.3f}' if auc else 'n/a'):>7} "
                    f"{agg['matthews_corrcoef_mean']:>+7.3f} "
                    f"{elapsed:>7.1f}  {agg['consistency']:>10}"
                )
                results[f"{symbol}|{spec.name}|{label}"] = agg

            # Inspect what the meta-learner actually learned to weight.
            try:
                stack = StackedEnsemble(
                    spec.n_classes, settings.random_seed,
                    base_factories=[
                        ("logreg", lambda: LogisticRegressionModel(spec.n_classes, 42)),
                        ("lgbm", lambda: LightGBMModel(spec.n_classes, 42)),
                    ],
                    config=EnsembleConfig(inner_folds=3, horizon=spec.horizon_days),
                )
                stack.fit(x, y)
                report = stack.stacking_report()
                report["config"] = f"{symbol}|{spec.name}"
                stack_reports.append(report)
                print(f"{'':>24} meta weights: {report['base_model_weights']} "
                      f"(oof rows {report['oof_rows_available']})")
            except Exception as exc:
                print(f"{'':>24} stacking report unavailable: {exc}")

    # --- Verdict -----------------------------------------------------------
    print(f"\n{'=' * 86}")
    print("VERDICT")
    print(f"{'=' * 86}")
    print(f"{'model':<24} {'mean acc':>10} {'mean skill':>12} {'beat baseline':>16}")
    print("-" * 86)

    for label in skills:
        print(
            f"{label:<24} {np.mean(accuracies[label]):>10.4f} "
            f"{np.mean(skills[label]):>+12.4f} "
            f"{f'{wins[label]}/{len(skills[label])}':>16}"
        )

    stack_skill = np.mean(skills.get("Most Possible (stack)", [np.nan]))
    lgbm_skill = np.mean(skills.get("LightGBM", [np.nan]))
    print()
    if np.isnan(stack_skill) or np.isnan(lgbm_skill):
        print("Insufficient results to compare.")
    elif stack_skill > lgbm_skill:
        print(f"Stacking improved on its best component by {stack_skill - lgbm_skill:+.4f}.")
    else:
        print(
            f"Stacking did NOT improve on LightGBM alone "
            f"({stack_skill:+.4f} vs {lgbm_skill:+.4f})."
        )

    if stack_skill <= 0:
        print("The ensemble does not demonstrate out-of-sample skill. "
              "The platform will report NO_EDGE.")

    print("\nMeta-learner weights across configurations:")
    for report in stack_reports:
        print(f"  {report['config']:<30} {report['base_model_weights']}")

    output = settings.resolved_artifact_dir / "final_comparison.json"
    output.write_text(
        json.dumps({"results": results, "stacking": stack_reports}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nFull report written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
