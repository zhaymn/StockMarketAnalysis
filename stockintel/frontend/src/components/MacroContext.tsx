"use client";

import { api } from "@/lib/api";
import { useAsyncData } from "@/lib/useAsyncData";
import { formatNumber, formatDate } from "@/lib/format";
import type { MacroSnapshot } from "@/lib/types";
import { Badge, Card, EmptyState, Section, Skeleton, Stat } from "./ui";

/**
 * Macro context from FRED.
 *
 * Every reading shows its observation date and flags itself when stale.
 * FRED's Indian coverage lags materially behind its US coverage — several
 * series are a year or more behind — and presenting those beside a live share
 * price without saying so would misrepresent them as current.
 */
export function MacroContext({ marketId }: { marketId: string }) {
  const { data, error, isLoading } = useAsyncData<MacroSnapshot>(
    marketId,
    (signal) => api.macro(marketId, signal),
  );

  if (isLoading) {
    return (
      <Section title="Macro context">
        <Skeleton className="h-40 w-full" />
      </Section>
    );
  }

  if (error?.isNotConfigured) {
    return (
      <Section title="Macro context">
        <EmptyState
          tone="warning"
          title="Macro data not configured"
          message={error.payload.reason ?? error.message}
          action={
            <div className="mt-3 w-full rounded-lg border border-line bg-ink-000 p-4">
              <div className="eyebrow mb-2">To enable this section</div>
              <ol className="space-y-1.5 text-sm text-text-secondary">
                <li>
                  1. Get a free key at{" "}
                  <a
                    href={error.payload.obtain_at}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-lime underline underline-offset-2 hover:text-lime-dim"
                  >
                    {error.payload.obtain_at}
                  </a>
                </li>
                <li>
                  2. Add{" "}
                  <code className="tnum rounded bg-ink-030 px-1.5 py-0.5 text-xs text-lime">
                    {error.payload.env_var}=your_key
                  </code>{" "}
                  to{" "}
                  <code className="tnum rounded bg-ink-030 px-1.5 py-0.5 text-xs text-text-primary">
                    stockintel/backend/.env
                  </code>
                </li>
                <li>3. Restart the backend</li>
              </ol>
            </div>
          }
        />
      </Section>
    );
  }

  if (error || !data) {
    return (
      <Section title="Macro context">
        <EmptyState
          title="Macro data unavailable"
          message={error?.message ?? "Could not retrieve macroeconomic series."}
          tone="warning"
        />
      </Section>
    );
  }

  const available = data.observations.filter((o) => o.available);

  return (
    <Section
      title="Macro context"
      subtitle="Measured economic series, not headlines. Each reading shows the date it was published."
      action={
        data.n_stale > 0 ? (
          <Badge tone="warning">
            {data.n_stale} of {data.n_available} readings stale
          </Badge>
        ) : (
          <Badge tone="neutral">{data.source}</Badge>
        )
      }
    >
      <Card>
        {available.length === 0 ? (
          <p className="text-sm text-text-muted">
            No macro series were available for this market.
          </p>
        ) : (
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {data.observations.map((observation) => (
              <div key={observation.series_id} className="min-w-0">
                {observation.available ? (
                  <>
                    <Stat
                      label={observation.label}
                      value={`${formatNumber(observation.value, 2)}${
                        observation.unit ? ` ${observation.unit}` : ""
                      }`}
                      valueClassName={
                        observation.is_stale ? "text-text-muted" : "text-text-primary"
                      }
                      size="sm"
                    />
                    <div className="mt-1 flex flex-wrap items-center gap-2">
                      <span className="tnum text-xs text-text-faint">
                        {formatDate(observation.observation_date)}
                      </span>
                      {observation.is_stale && (
                        <Badge tone="warning">
                          {Math.round((observation.age_days ?? 0) / 30)} months old
                        </Badge>
                      )}
                      {observation.change !== null && !observation.is_stale && (
                        <span
                          className={`tnum text-xs ${
                            observation.change > 0
                              ? "text-bullish"
                              : observation.change < 0
                                ? "text-bearish"
                                : "text-text-muted"
                          }`}
                        >
                          {observation.change > 0 ? "+" : ""}
                          {formatNumber(observation.change, 2)}
                        </span>
                      )}
                    </div>
                    <p className="mt-1.5 text-xs leading-relaxed text-text-faint">
                      {observation.description}
                    </p>
                  </>
                ) : (
                  <>
                    <Stat label={observation.label} value="—" size="sm" />
                    <p className="mt-1 text-xs text-text-faint">
                      {observation.unavailable_reason ?? "Unavailable"}
                    </p>
                  </>
                )}
              </div>
            ))}
          </div>
        )}

        {data.coverage_note && (
          <p className="mt-6 border-t border-line pt-4 text-xs leading-relaxed text-warning">
            {data.coverage_note}
          </p>
        )}
      </Card>
    </Section>
  );
}
