"use client";

import type { AnalysisResponse } from "@/lib/types";
import {
  formatCompact,
  formatNumber,
  formatPercent,
  formatPrice,
  humanise,
  signColor,
  UNAVAILABLE,
} from "@/lib/format";
import { Card, EmptyState, InfoTip, Section, Stat, StatGrid } from "./ui";

const RETURN_LABELS: Record<string, string> = {
  "1d": "1 day",
  "5d": "5 days",
  "1m": "1 month",
  "3m": "3 months",
  "6m": "6 months",
  "1y": "1 year",
};

export function KeyStatistics({ analysis }: { analysis: AnalysisResponse }) {
  const { analytics, market } = analysis;
  const symbol = market.conventions.currency_symbol;
  const { price, volatility, momentum, risk } = analytics;

  return (
    <Section title="Key statistics">
      <Card>
        <StatGrid columns={5}>
          <Stat
            label="52-week range"
            value={
              price.low_52w === null || price.high_52w === null
                ? UNAVAILABLE
                : `${formatPrice(price.low_52w, symbol)} – ${formatPrice(price.high_52w, symbol)}`
            }
            hint={
              price.has_full_52w_history
                ? undefined
                : `Only ${price.range_window_sessions} sessions available`
            }
            size="sm"
          />
          <Stat
            label="Position in range"
            value={formatPercent(price.position_in_52w_range, 0)}
            hint="0% at the low, 100% at the high"
            size="sm"
          />
          <Stat
            label="Volatility (21d)"
            value={formatPercent(volatility.realised_21d, 1)}
            hint={`${humanise(volatility.regime)} regime`}
            valueClassName={
              volatility.regime === "ELEVATED" ? "text-warning" : "text-text-primary"
            }
            size="sm"
          />
          <Stat
            label="RSI (14)"
            value={formatNumber(momentum.rsi_14, 1)}
            hint={humanise(momentum.rsi_state)}
            size="sm"
          />
          <Stat
            label="Max drawdown"
            value={formatPercent(risk.max_drawdown, 1)}
            hint="Worst peak-to-trough decline"
            valueClassName="text-bearish"
            size="sm"
          />
        </StatGrid>
      </Card>
    </Section>
  );
}

export function DetailedAnalytics({ analysis }: { analysis: AnalysisResponse }) {
  return (
    <Section
      title="Detailed analytics"
      subtitle="Every figure is computed from the retrieved price series. Metrics the data cannot support are shown as unavailable rather than defaulted."
    >
      <div className="grid gap-5 lg:grid-cols-2">
        <ReturnsCard analysis={analysis} />
        <VolatilityCard analysis={analysis} />
        <MomentumCard analysis={analysis} />
        <VolumeCard analysis={analysis} />
        <RiskCard analysis={analysis} />
        <BenchmarkCard analysis={analysis} />
      </div>
    </Section>
  );
}

function CardTitle({ children, tip }: { children: React.ReactNode; tip?: string }) {
  return (
    <h3 className="eyebrow mb-4">
      {children}
      {tip && <InfoTip text={tip} />}
    </h3>
  );
}

function ReturnsCard({ analysis }: { analysis: AnalysisResponse }) {
  const { returns } = analysis.analytics;

  return (
    <Card>
      <CardTitle>Trailing returns</CardTitle>
      <div className="space-y-2.5">
        {Object.entries(RETURN_LABELS).map(([key, label]) => {
          const value = returns[key];
          return (
            <div key={key} className="flex items-center justify-between gap-4">
              <span className="text-sm text-text-secondary">{label}</span>
              <div className="flex flex-1 items-center justify-end gap-3">
                {value !== null && value !== undefined && (
                  <div className="relative h-1.5 w-24 overflow-hidden rounded-full bg-ink-030">
                    <div
                      className={`absolute top-0 h-full ${value >= 0 ? "left-1/2 bg-bullish" : "right-1/2 bg-bearish"}`}
                      style={{ width: `${Math.min(50, Math.abs(value) * 100)}%` }}
                    />
                    <div className="absolute left-1/2 top-0 h-full w-px bg-line-strong" />
                  </div>
                )}
                <span
                  className={`tnum w-20 text-right text-sm font-semibold ${signColor(value)}`}
                >
                  {formatPercent(value, 2, true)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function VolatilityCard({ analysis }: { analysis: AnalysisResponse }) {
  const { volatility } = analysis.analytics;

  return (
    <Card>
      <CardTitle tip={`Annualised using ${volatility.annualisation_basis} trading days, the count for this market.`}>
        Volatility
      </CardTitle>
      <StatGrid columns={2}>
        <Stat label="21-day realised" value={formatPercent(volatility.realised_21d, 1)} size="sm" />
        <Stat label="1-year realised" value={formatPercent(volatility.realised_252d, 1)} size="sm" />
        <Stat
          label="ATR (14)"
          value={formatPercent(volatility.atr_14_percent, 2)}
          hint="As a share of price"
          size="sm"
        />
        <Stat
          label="Downside volatility"
          value={formatPercent(volatility.downside_volatility, 1)}
          hint="Negative returns only"
          size="sm"
        />
      </StatGrid>
      <div className="mt-4 border-t border-line pt-3">
        <span className="text-xs text-text-muted">
          Current volatility is{" "}
          <span
            className={`font-semibold ${
              volatility.regime === "ELEVATED"
                ? "text-warning"
                : volatility.regime === "SUBDUED"
                  ? "text-bullish"
                  : "text-text-primary"
            }`}
          >
            {formatNumber(volatility.regime_ratio, 2)}×
          </span>{" "}
          its own one-year average — a {humanise(volatility.regime).toLowerCase()} regime.
        </span>
      </div>
    </Card>
  );
}

function MomentumCard({ analysis }: { analysis: AnalysisResponse }) {
  const { momentum } = analysis.analytics;

  if (!momentum.available) {
    return (
      <Card>
        <CardTitle>Momentum</CardTitle>
        <p className="text-sm text-text-muted">Momentum indicators unavailable.</p>
      </Card>
    );
  }

  return (
    <Card>
      <CardTitle>Momentum &amp; trend</CardTitle>
      <StatGrid columns={2}>
        <Stat label="RSI (14)" value={formatNumber(momentum.rsi_14, 1)} hint={humanise(momentum.rsi_state)} size="sm" />
        <Stat
          label="MACD state"
          value={momentum.macd_state === "BULLISH_CROSSOVER" ? "Bullish" : momentum.macd_state === "BEARISH_CROSSOVER" ? "Bearish" : UNAVAILABLE}
          hint={`Histogram ${formatNumber(momentum.macd_histogram, 4)}`}
          valueClassName={momentum.macd_state === "BULLISH_CROSSOVER" ? "text-bullish" : "text-bearish"}
          size="sm"
        />
      </StatGrid>

      <div className="mt-4 space-y-2 border-t border-line pt-4">
        {[
          ["vs SMA 20", momentum.distance_from_sma_20],
          ["vs SMA 50", momentum.distance_from_sma_50],
          ["vs SMA 200", momentum.distance_from_sma_200],
        ].map(([label, value]) => (
          <div key={label as string} className="flex items-center justify-between">
            <span className="text-sm text-text-secondary">{label as string}</span>
            <span className={`tnum text-sm font-semibold ${signColor(value as number | null)}`}>
              {formatPercent(value as number | null, 2, true)}
            </span>
          </div>
        ))}
        <div className="flex items-center justify-between border-t border-line pt-2">
          <span className="text-sm text-text-secondary">Trend</span>
          <span className="text-sm font-semibold text-text-primary">
            {humanise(momentum.moving_average_trend)}
          </span>
        </div>
      </div>
    </Card>
  );
}

function VolumeCard({ analysis }: { analysis: AnalysisResponse }) {
  const { volume } = analysis.analytics;

  if (!volume.available) {
    return (
      <Card>
        <CardTitle>Volume</CardTitle>
        <p className="text-sm text-text-muted">
          {volume.reason ?? "Volume data unavailable for this security."}
        </p>
      </Card>
    );
  }

  return (
    <Card>
      <CardTitle>Volume</CardTitle>
      <StatGrid columns={2}>
        <Stat label="Latest session" value={formatCompact(volume.latest)} size="sm" />
        <Stat label="20-day average" value={formatCompact(volume.average_20d)} size="sm" />
        <Stat
          label="Relative volume"
          value={`${formatNumber(volume.relative_volume, 2)}×`}
          hint="Versus the 20-day average"
          valueClassName={
            (volume.relative_volume ?? 0) > 1.5 ? "text-warning" : "text-text-primary"
          }
          size="sm"
        />
        <Stat label="Trend" value={humanise(volume.trend)} size="sm" />
      </StatGrid>
    </Card>
  );
}

function RiskCard({ analysis }: { analysis: AnalysisResponse }) {
  const { risk } = analysis.analytics;

  return (
    <Card>
      <CardTitle>Risk</CardTitle>
      <StatGrid columns={2}>
        <Stat
          label="Max drawdown"
          value={formatPercent(risk.max_drawdown, 1)}
          valueClassName="text-bearish"
          size="sm"
        />
        <Stat label="Current drawdown" value={formatPercent(risk.current_drawdown, 1)} size="sm" />
        <Stat
          label="Beta"
          value={formatNumber(risk.beta, 2)}
          hint={
            risk.beta === null
              ? "Insufficient benchmark overlap"
              : `vs ${analysis.analytics.benchmark.benchmark_label}`
          }
          size="sm"
        />
        <Stat
          label="Return / volatility"
          value={formatNumber(risk.sharpe_like_ratio, 2)}
          hint="Sharpe-like, no risk-free rate"
          size="sm"
        />
      </StatGrid>
      <p className="mt-4 border-t border-line pt-3 text-xs leading-relaxed text-text-faint">
        {risk.sharpe_note}
      </p>
    </Card>
  );
}

function BenchmarkCard({ analysis }: { analysis: AnalysisResponse }) {
  const { benchmark } = analysis.analytics;

  if (!benchmark.available) {
    return (
      <Card>
        <CardTitle>Benchmark comparison</CardTitle>
        <p className="text-sm text-text-muted">
          {benchmark.reason ?? "Benchmark comparison unavailable."}
        </p>
      </Card>
    );
  }

  const performance = benchmark.relative_performance ?? {};

  return (
    <Card>
      <CardTitle>Versus {benchmark.benchmark_label}</CardTitle>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[320px] text-sm">
          <thead>
            <tr className="border-b border-line text-left">
              <th className="pb-2 font-medium text-text-muted">Period</th>
              <th className="pb-2 text-right font-medium text-text-muted">Stock</th>
              <th className="pb-2 text-right font-medium text-text-muted">Index</th>
              <th className="pb-2 text-right font-medium text-text-muted">Excess</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(RETURN_LABELS).map(([key, label]) => {
              const row = performance[key];
              return (
                <tr key={key} className="border-b border-line/60 last:border-0">
                  <td className="py-2 text-text-secondary">{label}</td>
                  <td className={`tnum py-2 text-right ${signColor(row?.stock)}`}>
                    {formatPercent(row?.stock ?? null, 1, true)}
                  </td>
                  <td className={`tnum py-2 text-right ${signColor(row?.benchmark)}`}>
                    {formatPercent(row?.benchmark ?? null, 1, true)}
                  </td>
                  <td className={`tnum py-2 text-right font-semibold ${signColor(row?.excess)}`}>
                    {formatPercent(row?.excess ?? null, 1, true)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

export function Fundamentals({ analysis }: { analysis: AnalysisResponse }) {
  const { profile } = analysis;
  const fundamentals = profile.fundamentals;

  if (!fundamentals) {
    return (
      <Section title="Fundamentals">
        <EmptyState
          title="Data unavailable"
          message={profile.reason ?? "No fundamental data was returned for this security."}
        />
      </Section>
    );
  }

  const rows: [string, string][] = [
    ["Market cap", formatCompact(fundamentals.market_cap)],
    ["Trailing P/E", formatNumber(fundamentals.trailing_pe, 2)],
    ["Forward P/E", formatNumber(fundamentals.forward_pe, 2)],
    ["EPS (trailing)", formatNumber(fundamentals.eps_trailing, 2)],
    ["Profit margin", formatPercent(fundamentals.profit_margin, 1)],
    ["Revenue growth", formatPercent(fundamentals.revenue_growth, 1)],
    ["Earnings growth", formatPercent(fundamentals.earnings_growth, 1)],
    ["Free cash flow", formatCompact(fundamentals.free_cash_flow)],
  ];

  const unavailable = profile.unavailable_fields ?? [];

  return (
    <Section
      title="Fundamentals"
      subtitle={profile.industry ? `${profile.sector} · ${profile.industry}` : undefined}
    >
      <Card>
        <StatGrid columns={4}>
          {rows.map(([label, value]) => (
            <Stat key={label} label={label} value={value} size="sm" />
          ))}
        </StatGrid>
        {unavailable.length > 0 && (
          <p className="mt-5 border-t border-line pt-3 text-xs text-text-faint">
            Not reported by the data provider for this security:{" "}
            {unavailable.join(", ").replace(/_/g, " ")}.
          </p>
        )}
      </Card>
    </Section>
  );
}
