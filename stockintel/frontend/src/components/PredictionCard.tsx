"use client";

import type { AnalysisResponse, Prediction } from "@/lib/types";
import { formatPercent, humanise } from "@/lib/format";
import { Badge, Card, CompareBar, InfoTip } from "./ui";

const RISK_TONE = {
  LOW: "bullish",
  MODERATE: "neutral",
  ELEVATED: "warning",
  HIGH: "bearish",
} as const;

const DIRECTION_TONE: Record<string, "bullish" | "bearish" | "neutral"> = {
  BULLISH: "bullish",
  UP: "bullish",
  BEARISH: "bearish",
  DOWN: "bearish",
  NEUTRAL: "neutral",
};

/**
 * The headline prediction.
 *
 * Four verdicts get four genuinely different treatments. NO_EDGE is not styled
 * as an error: it is the honest and most common outcome for price-only models,
 * so it gets a first-class layout that leads with the evidence rather than an
 * apology.
 */
export function PredictionCard({ analysis }: { analysis: AnalysisResponse }) {
  const { prediction } = analysis;

  switch (prediction.verdict) {
    case "DIRECTIONAL":
      return <DirectionalPrediction analysis={analysis} />;
    case "ABSTAINED":
      return <AbstainedPrediction prediction={prediction} />;
    case "NO_EDGE":
      return <NoEdgePrediction prediction={prediction} />;
    default:
      return <InsufficientData prediction={prediction} />;
  }
}

function VerdictShell({
  eyebrow,
  badge,
  children,
  accent,
}: {
  eyebrow: string;
  badge: React.ReactNode;
  children: React.ReactNode;
  accent: string;
}) {
  return (
    <Card className={`border-l-2 ${accent}`}>
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <span className="eyebrow">{eyebrow}</span>
        {badge}
      </div>
      {children}
    </Card>
  );
}

function DirectionalPrediction({ analysis }: { analysis: AnalysisResponse }) {
  const { prediction } = analysis;
  const tone = DIRECTION_TONE[prediction.direction ?? ""] ?? "neutral";
  const range = prediction.expected_return_range;

  return (
    <VerdictShell
      eyebrow={`${prediction.model_label} · ${prediction.target.horizon_days}-day outlook`}
      badge={
        <Badge tone={RISK_TONE[prediction.risk_level]}>
          {humanise(prediction.risk_level)} risk
        </Badge>
      }
      accent={tone === "bullish" ? "border-l-bullish" : tone === "bearish" ? "border-l-bearish" : "border-l-neutral"}
    >
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_2fr]">
        <div>
          <div
            className={`text-4xl font-bold tracking-tight sm:text-5xl ${
              tone === "bullish"
                ? "text-bullish"
                : tone === "bearish"
                  ? "text-bearish"
                  : "text-neutral"
            }`}
          >
            {prediction.direction}
          </div>

          {prediction.probability !== null ? (
            <div className="mt-3">
              <span className="tnum text-2xl font-semibold text-text-primary">
                {formatPercent(prediction.probability, 1)}
              </span>
              <span className="ml-2 text-sm text-text-secondary">
                estimated probability
                <InfoTip text="Calibrated on held-out data: of all predictions assigned this probability, approximately this share historically occurred." />
              </span>
            </div>
          ) : (
            <div className="mt-3 rounded-lg border border-warning/25 bg-warning-dim/40 px-3 py-2">
              <div className="text-xs font-semibold text-warning">
                PROBABILITY NOT CALIBRATED
              </div>
              <p className="mt-1 text-xs leading-relaxed text-text-secondary">
                {prediction.probability_withheld_reason}
              </p>
            </div>
          )}
        </div>

        <div>
          {range ? (
            <div>
              <div className="eyebrow mb-3">
                Historical outcome range
                <InfoTip text={range.basis} />
              </div>
              <QuantileBar range={range} />
              <p className="mt-3 text-xs leading-relaxed text-text-muted">
                Distribution of what actually happened on the {range.n_observations}{" "}
                historical days that resolved to this class — not a model-generated
                confidence band.
              </p>
            </div>
          ) : (
            <p className="text-sm text-text-muted">
              Too few historical observations of this class to quote a defensible
              return range.
            </p>
          )}
        </div>
      </div>

      <p className="mt-6 border-t border-line pt-4 text-sm leading-relaxed text-text-secondary">
        {prediction.interpretation}
      </p>
    </VerdictShell>
  );
}

function QuantileBar({
  range,
}: {
  range: NonNullable<Prediction["expected_return_range"]>;
}) {
  const span = range.p90 - range.p10 || 1;
  const position = (value: number) => ((value - range.p10) / span) * 100;

  return (
    <div>
      <div className="relative h-10">
        {/* p10–p90 */}
        <div className="absolute top-4 h-2 w-full rounded-full bg-ink-030" />
        {/* p25–p75 interquartile */}
        <div
          className="absolute top-4 h-2 rounded-full bg-lime/40"
          style={{
            left: `${position(range.p25)}%`,
            width: `${position(range.p75) - position(range.p25)}%`,
          }}
        />
        {/* median */}
        <div
          className="absolute top-2.5 h-5 w-0.5 bg-lime"
          style={{ left: `${position(range.median)}%` }}
        />
      </div>
      <div className="tnum mt-1 flex justify-between text-xs">
        <span className="text-bearish">{formatPercent(range.p10, 1, true)}</span>
        <span className="font-semibold text-lime">
          {formatPercent(range.median, 1, true)}
        </span>
        <span className="text-bullish">{formatPercent(range.p90, 1, true)}</span>
      </div>
      <div className="mt-1 flex justify-between text-[10px] uppercase tracking-wider text-text-faint">
        <span>10th pct</span>
        <span>median</span>
        <span>90th pct</span>
      </div>
    </div>
  );
}

/**
 * The honest no-signal state — now with the model's indicative lean.
 *
 * Every stock fails the skill gate, which left this state showing nothing but a
 * refusal. It now leads with the model's raw directional lean so there is always
 * a visible signal, explicitly tagged as indicative and NOT a validated call,
 * with the accuracy-vs-baseline evidence kept right beside it. That is the line
 * between "here's the model's tendency, but it hasn't earned your trust" and the
 * confident-looking lie this platform is built to avoid.
 */
function NoEdgePrediction({ prediction }: { prediction: Prediction }) {
  const wf = prediction.evidence.walk_forward;
  const lean = prediction.indicative_direction;
  const leanTone = DIRECTION_TONE[lean ?? ""] ?? "neutral";

  return (
    <VerdictShell
      eyebrow={`${prediction.model_label} · ${prediction.target.horizon_days}-day outlook`}
      badge={<Badge tone="warning">No demonstrated edge</Badge>}
      accent="border-l-warning"
    >
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_1fr]">
        <div>
          {lean ? (
            <>
              <span className="mb-3 inline-flex items-center gap-1.5 rounded-md border border-warning/30 bg-warning-dim/40 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider text-warning">
                Indicative signal · not a validated call
              </span>
              <div
                className={`text-3xl font-bold tracking-tight sm:text-4xl ${
                  leanTone === "bullish"
                    ? "text-bullish"
                    : leanTone === "bearish"
                      ? "text-bearish"
                      : "text-neutral"
                }`}
              >
                LEANS {lean}
              </div>
              {prediction.indicative_probability !== null && (
                <div className="tnum mt-1.5 text-xs text-text-muted">
                  raw model probability {formatPercent(prediction.indicative_probability, 0)} —
                  uncalibrated, shown for context only
                </div>
              )}
            </>
          ) : (
            <div className="text-3xl font-bold tracking-tight text-text-primary sm:text-4xl">
              NO DIRECTIONAL CALL
            </div>
          )}
          <p className="mt-3 max-w-xl text-sm leading-relaxed text-text-secondary">
            {prediction.interpretation}
          </p>
        </div>

        <div className="rounded-lg border border-line bg-ink-000 p-4">
          <div className="eyebrow mb-3">Why no call was issued</div>

          <div className="mb-2 flex items-baseline justify-between">
            <span className="text-xs text-text-muted">Model accuracy</span>
            <span className="tnum text-sm font-semibold text-text-primary">
              {formatPercent(wf.accuracy_mean, 1)}
            </span>
          </div>
          <CompareBar
            value={wf.accuracy_mean}
            reference={wf.baseline_accuracy_mean}
            max={Math.max(wf.accuracy_mean, wf.baseline_accuracy_mean) * 1.6}
          />
          <div className="mt-2 flex items-baseline justify-between">
            <span className="text-xs text-text-muted">Naive baseline</span>
            <span className="tnum text-sm font-semibold text-warning">
              {formatPercent(wf.baseline_accuracy_mean, 1)}
            </span>
          </div>

          <p className="mt-4 border-t border-line pt-3 text-xs leading-relaxed text-text-muted">
            {wf.consistency}. A model must beat the naive baseline consistently
            out-of-sample before this platform will issue a directional call.
          </p>
        </div>
      </div>
    </VerdictShell>
  );
}

function AbstainedPrediction({ prediction }: { prediction: Prediction }) {
  return (
    <VerdictShell
      eyebrow={`${prediction.model_label} · ${prediction.target.horizon_days}-day outlook`}
      badge={<Badge tone="neutral">Abstained today</Badge>}
      accent="border-l-neutral"
    >
      <div className="text-3xl font-bold tracking-tight text-text-primary sm:text-4xl">
        NO CALL TODAY
      </div>
      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-text-secondary">
        {prediction.interpretation}
      </p>
      <p className="mt-4 border-t border-line pt-4 text-xs leading-relaxed text-text-muted">
        This model has demonstrated skill historically, but only acts when its
        confidence clears a threshold fitted on held-out data. Forcing a call every
        session measurably degrades accuracy.
      </p>
    </VerdictShell>
  );
}

function InsufficientData({ prediction }: { prediction: Prediction }) {
  return (
    <VerdictShell
      eyebrow="Prediction unavailable"
      badge={<Badge tone="warning">Insufficient data</Badge>}
      accent="border-l-warning"
    >
      <div className="text-2xl font-bold tracking-tight text-text-primary">
        INSUFFICIENT HISTORY
      </div>
      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-text-secondary">
        {prediction.interpretation ||
          "There is not enough price history for this security to train and validate a model."}
      </p>
    </VerdictShell>
  );
}
