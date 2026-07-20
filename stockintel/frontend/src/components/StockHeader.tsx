"use client";

import type { AnalysisResponse } from "@/lib/types";
import { formatDateTime, formatPercent, formatPrice, signColor } from "@/lib/format";
import { Badge } from "./ui";

const FRESHNESS_COPY: Record<string, { label: string; tone: "bullish" | "neutral" | "warning" }> = {
  DELAYED: { label: "Delayed quote", tone: "neutral" },
  END_OF_DAY: { label: "End of day", tone: "neutral" },
  CACHED: { label: "Cached", tone: "warning" },
  UNAVAILABLE: { label: "Unavailable", tone: "warning" },
};

export function StockHeader({ analysis }: { analysis: AnalysisResponse }) {
  const { analytics, market, session, data, profile } = analysis;
  const price = analytics.price;
  const symbol = market.conventions.currency_symbol;
  const freshness = FRESHNESS_COPY[data.freshness] ?? FRESHNESS_COPY.UNAVAILABLE;
  const isOpen = session.status === "OPEN";

  return (
    <div className="border-b border-line bg-ink-010">
      <div className="mx-auto max-w-[1600px] px-4 py-6 sm:px-6 lg:px-8">
        <div className="flex flex-wrap items-start justify-between gap-6">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="tnum text-3xl font-bold tracking-tight text-text-primary sm:text-4xl">
                {analysis.symbol}
              </h1>
              <Badge tone="lime">{market.market_label}</Badge>
              {profile.sector && <Badge tone="neutral">{profile.sector}</Badge>}
            </div>
            {/* Long company names must wrap, not clip or overflow the header. */}
            <p className="mt-1.5 max-w-xl break-words text-sm text-text-secondary">
              {profile.name ?? "Company name unavailable"}
            </p>
          </div>

          <div className="flex flex-wrap items-end gap-x-8 gap-y-4">
            <div>
              <div className="tnum text-3xl font-bold tracking-tight text-text-primary sm:text-4xl">
                {formatPrice(price.current, symbol)}
              </div>
              <div
                className={`tnum mt-1 text-sm font-semibold ${signColor(price.change_percent)}`}
              >
                {price.change !== null && price.change > 0 ? "+" : ""}
                {formatPrice(price.change, symbol)} ({formatPercent(price.change_percent, 2, true)})
              </div>
            </div>

            <div className="flex flex-col gap-1.5 text-right">
              <div className="flex items-center justify-end gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${isOpen ? "bg-bullish" : "bg-text-faint"}`}
                  aria-hidden="true"
                />
                <span className="text-xs font-medium text-text-secondary">
                  {session.status_label}
                </span>
              </div>
              <Badge tone={freshness.tone} className="justify-end">
                {freshness.label}
              </Badge>
              <span className="text-xs text-text-faint">
                Data as of {formatDateTime(session.exchange_time)}
              </span>
            </div>
          </div>
        </div>

        {!data.quality.is_clean && data.quality.warnings.length > 0 && (
          <div className="mt-4 rounded-lg border border-warning/25 bg-warning-dim/30 px-4 py-3">
            <div className="eyebrow text-warning">Data quality notice</div>
            <ul className="mt-1.5 space-y-1">
              {data.quality.warnings.map((warning) => (
                <li key={warning} className="text-xs leading-relaxed text-text-secondary">
                  {warning}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
