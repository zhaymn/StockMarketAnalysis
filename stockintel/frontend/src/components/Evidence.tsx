"use client";

import type { AnalysisResponse } from "@/lib/types";
import { formatDate, formatNumber, formatPercent, humanise } from "@/lib/format";
import { Badge, Card, CompareBar, EmptyState, InfoTip, Section, Stat, StatGrid } from "./ui";

/**
 * Model evidence.
 *
 * Accuracy is never shown without its baseline alongside — that pairing is the
 * whole point of the section, since an accuracy figure alone is unfalsifiable.
 */
export function ModelEvidence({ analysis }: { analysis: AnalysisResponse }) {
  const { evidence, target } = analysis.prediction;
  const wf = evidence.walk_forward;
  const calibration = evidence.calibration;
  const beatsBaseline = wf.accuracy_mean > wf.baseline_accuracy_mean;

  return (
    <Section
      title="Model evidence"
      subtitle={`Purged walk-forward validation with a ${target.horizon_days}-day purge and embargo around every test block, so no training label overlaps a test-period price.`}
    >
      <div className="grid gap-5 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <div className="mb-5 flex items-center justify-between">
            <h3 className="eyebrow">
              Out-of-sample accuracy vs baseline
              <InfoTip text="The baseline always predicts the most common class. Any model must beat it to be worth anything." />
            </h3>
            <Badge tone={beatsBaseline ? "bullish" : "warning"}>
              {beatsBaseline ? "Beats baseline" : "Does not beat baseline"}
            </Badge>
          </div>

          <div className="mb-2 flex items-baseline justify-between">
            <span className="text-sm text-text-secondary">Model</span>
            <span className="tnum text-2xl font-bold text-text-primary">
              {formatPercent(wf.accuracy_mean, 1)}
              <span className="ml-2 text-sm font-normal text-text-muted">
                ± {formatPercent(wf.accuracy_std, 1)}
              </span>
            </span>
          </div>
          <CompareBar
            value={wf.accuracy_mean}
            reference={wf.baseline_accuracy_mean}
            max={Math.max(wf.accuracy_mean, wf.baseline_accuracy_mean) * 1.5}
          />
          <div className="mt-2 flex items-baseline justify-between">
            <span className="text-sm text-text-secondary">Naive baseline</span>
            <span className="tnum text-lg font-semibold text-warning">
              {formatPercent(wf.baseline_accuracy_mean, 1)}
            </span>
          </div>

          <div className="mt-5 border-t border-line pt-4">
            <StatGrid columns={4}>
              <Stat
                label="Skill score"
                value={formatNumber(wf.skill_score_mean, 3)}
                hint="0 = no better than baseline"
                valueClassName={wf.skill_score_mean > 0 ? "text-bullish" : "text-bearish"}
                size="sm"
              />
              <Stat
                label="ROC-AUC"
                value={formatNumber(wf.roc_auc_mean, 3)}
                hint="0.5 = no ranking signal"
                size="sm"
              />
              <Stat
                label="Matthews corr."
                value={formatNumber(wf.matthews_corrcoef_mean, 3)}
                hint="Robust to class imbalance"
                size="sm"
              />
              <Stat label="Macro F1" value={formatNumber(wf.f1_macro_mean, 3)} size="sm" />
            </StatGrid>
          </div>

          <p className="mt-4 border-t border-line pt-3 text-xs text-text-muted">
            {wf.consistency} across {wf.n_folds} folds, {wf.total_test_samples.toLocaleString()}{" "}
            out-of-sample test observations.
          </p>
        </Card>

        <Card>
          <h3 className="eyebrow mb-4">
            Probability calibration
            <InfoTip text="Of all predictions given a 70% probability, roughly 70% should occur. Probabilities are displayed only if they pass this check." />
          </h3>

          {calibration ? (
            <>
              <Badge tone={calibration.is_calibrated ? "bullish" : "warning"}>
                {calibration.is_calibrated ? "Calibrated" : "Not calibrated"}
              </Badge>

              <div className="mt-4 space-y-3">
                <Stat
                  label="Brier skill score"
                  value={formatNumber(calibration.brier_skill_score, 3)}
                  hint="Above 0 beats the base rate"
                  valueClassName={
                    calibration.brier_skill_score > 0 ? "text-bullish" : "text-bearish"
                  }
                  size="sm"
                />
                <Stat
                  label="Brier score"
                  value={formatNumber(calibration.brier_score, 4)}
                  hint={`Baseline ${formatNumber(calibration.baseline_brier, 4)}`}
                  size="sm"
                />
                <Stat
                  label="Mean calibration error"
                  value={formatPercent(calibration.expected_calibration_error, 1)}
                  size="sm"
                />
              </div>

              <p className="mt-4 border-t border-line pt-3 text-xs leading-relaxed text-text-secondary">
                {calibration.interpretation}
              </p>
            </>
          ) : (
            <p className="text-sm text-text-muted">
              Too few out-of-sample predictions to assess calibration.
            </p>
          )}
        </Card>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <RegimeBreakdown analysis={analysis} />
        <TopFeatures analysis={analysis} />
      </div>
    </Section>
  );
}

function RegimeBreakdown({ analysis }: { analysis: AnalysisResponse }) {
  const regimes = analysis.prediction.evidence.regime_breakdown ?? {};
  const entries = Object.entries(regimes);

  if (entries.length === 0) {
    return (
      <Card>
        <h3 className="eyebrow mb-4">Performance by market regime</h3>
        <p className="text-sm text-text-muted">
          Not enough observations in any single regime to report reliably.
        </p>
      </Card>
    );
  }

  return (
    <Card>
      <h3 className="eyebrow mb-4">
        Performance by market regime
        <InfoTip text="A model that works only in calm markets is dangerous precisely when a prediction matters most." />
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[380px] text-sm">
          <thead>
            <tr className="border-b border-line text-left">
              <th className="pb-2 font-medium text-text-muted">Regime</th>
              <th className="pb-2 text-right font-medium text-text-muted">n</th>
              <th className="pb-2 text-right font-medium text-text-muted">Accuracy</th>
              <th className="pb-2 text-right font-medium text-text-muted">Baseline</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([regime, stats]) => (
              <tr key={regime} className="border-b border-line/60 last:border-0">
                <td className="py-2 text-text-secondary">{regime}</td>
                <td className="tnum py-2 text-right text-text-muted">{stats.n_samples}</td>
                <td
                  className={`tnum py-2 text-right font-semibold ${
                    stats.beats_baseline ? "text-bullish" : "text-bearish"
                  }`}
                >
                  {formatPercent(stats.accuracy, 1)}
                </td>
                <td className="tnum py-2 text-right text-text-muted">
                  {formatPercent(stats.baseline_accuracy, 1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function TopFeatures({ analysis }: { analysis: AnalysisResponse }) {
  const features = analysis.prediction.evidence.top_features ?? [];

  if (features.length === 0) {
    return (
      <Card>
        <h3 className="eyebrow mb-4">Most influential features</h3>
        <p className="text-sm text-text-muted">
          This model does not expose feature importances.
        </p>
      </Card>
    );
  }

  const max = Math.max(...features.map((f) => f.importance));

  return (
    <Card>
      <h3 className="eyebrow mb-4">
        Most influential features
        <InfoTip text="Gain-based importance: total loss reduction attributed to each feature across the ensemble." />
      </h3>
      <div className="space-y-2.5">
        {features.slice(0, 8).map((feature) => (
          <div key={feature.feature}>
            <div className="mb-1 flex items-baseline justify-between gap-3">
              <span className="truncate text-xs text-text-secondary">
                {feature.feature.replace(/_/g, " ")}
              </span>
              <span className="tnum shrink-0 text-xs text-text-muted">
                {formatPercent(feature.importance, 1)}
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-030">
              <div
                className="h-full rounded-full bg-lime"
                style={{ width: `${(feature.importance / max) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

/** Bullish / bearish / risk factors, each citing the value that produced it. */
export function WhyThisPrediction({ analysis }: { analysis: AnalysisResponse }) {
  const { factors, interpretation } = analysis.prediction;
  const isEmpty =
    factors.bullish.length === 0 &&
    factors.bearish.length === 0 &&
    factors.risks.length === 0;

  if (isEmpty) {
    return (
      <Section title="Why this prediction?">
        <EmptyState
          title="No factors identified"
          message="No signal in the current data crossed the thresholds used to report a factor."
        />
      </Section>
    );
  }

  return (
    <Section
      title="Why this prediction?"
      subtitle="Each factor is derived from a computed signal and shows the value behind it."
    >
      <div className="grid gap-5 lg:grid-cols-3">
        <FactorColumn title="Bullish factors" tone="bullish" factors={factors.bullish} />
        <FactorColumn title="Bearish factors" tone="bearish" factors={factors.bearish} />
        <FactorColumn title="Risk factors" tone="warning" factors={factors.risks} />
      </div>

      <Card className="mt-5">
        <h3 className="eyebrow mb-3">Final interpretation</h3>
        <p className="text-sm leading-relaxed text-text-secondary">{interpretation}</p>
      </Card>
    </Section>
  );
}

function FactorColumn({
  title,
  tone,
  factors,
}: {
  title: string;
  tone: "bullish" | "bearish" | "warning";
  factors: { factor: string; evidence: string }[];
}) {
  const accents = {
    bullish: "border-l-bullish",
    bearish: "border-l-bearish",
    warning: "border-l-warning",
  };
  const titleColors = {
    bullish: "text-bullish",
    bearish: "text-bearish",
    warning: "text-warning",
  };

  return (
    <Card className="flex flex-col">
      <h3 className={`eyebrow mb-4 ${titleColors[tone]}`}>
        {title} ({factors.length})
      </h3>
      {factors.length === 0 ? (
        <p className="text-sm text-text-faint">None identified.</p>
      ) : (
        <ul className="space-y-3">
          {factors.map((factor) => (
            <li key={factor.factor} className={`border-l-2 pl-3 ${accents[tone]}`}>
              <div className="text-sm font-medium text-text-primary">{factor.factor}</div>
              <div className="tnum mt-0.5 text-xs leading-relaxed text-text-muted">
                {factor.evidence}
              </div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

/** Model transparency: what ran, on what data, validated how. */
export function ModelTransparency({ analysis }: { analysis: AnalysisResponse }) {
  const { prediction, data, feature_groups, market } = analysis;
  const evidence = prediction.evidence;

  return (
    <Section
      title="Model transparency"
      subtitle="Everything about how this prediction was produced."
    >
      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <h3 className="eyebrow mb-4">Configuration</h3>
          <dl className="space-y-2.5 text-sm">
            {[
              ["Model", prediction.model_label],
              ["Prediction target", prediction.target.description],
              ["Horizon", `${prediction.target.horizon_days} trading days`],
              ["Benchmark", market.conventions.benchmark_label],
              ["Currency", market.conventions.currency_code],
              ["Timezone", market.conventions.timezone],
            ].map(([label, value]) => (
              <div key={label} className="flex flex-wrap justify-between gap-x-4 gap-y-1 border-b border-line/60 pb-2 last:border-0">
                <dt className="text-text-muted">{label}</dt>
                <dd className="max-w-sm text-right text-text-primary">{value}</dd>
              </div>
            ))}
          </dl>
        </Card>

        <Card>
          <h3 className="eyebrow mb-4">Data &amp; validation</h3>
          <dl className="space-y-2.5 text-sm">
            {[
              ["Training period", `${formatDate(evidence.training_period[0])} – ${formatDate(evidence.training_period[1])}`],
              ["Sessions available", data.sessions.toLocaleString()],
              ["Modelled observations", evidence.n_samples.toLocaleString()],
              ["Effective sample size", evidence.effective_sample_size.toLocaleString()],
              ["Walk-forward folds", String(evidence.walk_forward.n_folds)],
              ["Data freshness", humanise(data.freshness)],
            ].map(([label, value]) => (
              <div key={label} className="flex flex-wrap justify-between gap-x-4 gap-y-1 border-b border-line/60 pb-2 last:border-0">
                <dt className="text-text-muted">{label}</dt>
                <dd className="tnum text-right text-text-primary">{value}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-xs leading-relaxed text-text-faint">
            Effective sample size divides observations by the horizon, because
            overlapping forward-return windows are not independent. It is the honest
            denominator for these metrics.
          </p>
        </Card>
      </div>

      <Card className="mt-5">
        <h3 className="eyebrow mb-4">Input signal groups</h3>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {feature_groups.map((group) => (
            <div key={group.name} className="rounded-lg border border-line bg-ink-000 p-3">
              <div className="flex items-center gap-2">
                <span className="text-bullish" aria-hidden="true">✓</span>
                <span className="text-sm font-medium text-text-primary">{group.name}</span>
                <span className="tnum ml-auto text-xs text-text-muted">
                  {group.n_features}
                </span>
              </div>
              <p className="mt-1.5 text-xs leading-relaxed text-text-muted">
                {group.description}
              </p>
            </div>
          ))}
          <div className="rounded-lg border border-line border-dashed bg-ink-000 p-3">
            <div className="flex items-center gap-2">
              <span className="text-text-faint" aria-hidden="true">○</span>
              <span className="text-sm font-medium text-text-muted">News sentiment</span>
            </div>
            <p className="mt-1.5 text-xs leading-relaxed text-text-faint">
              Not yet included as a model feature — pending validation that it adds
              out-of-sample skill.
            </p>
          </div>
        </div>
      </Card>
    </Section>
  );
}
