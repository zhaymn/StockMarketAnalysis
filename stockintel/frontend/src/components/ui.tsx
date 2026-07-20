/**
 * Shared UI primitives.
 *
 * `Stat` is the workhorse: it takes a possibly-null value and renders the
 * em-dash unavailable state itself, so no call site has to remember to. That
 * makes "missing data looks missing" a property of the component rather than a
 * rule everyone must follow.
 */

import type { ReactNode } from "react";
import { UNAVAILABLE } from "@/lib/format";

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    // min-w-0 is load-bearing: as a grid/flex child this defaults to
    // min-width:auto, which refuses to shrink below its content and lets wide
    // tables push the whole page into horizontal scroll. With it, wide content
    // scrolls inside its own overflow-x-auto wrapper and the body never does.
    <div
      className={`min-w-0 rounded-xl border border-line bg-ink-010 p-5 sm:p-6 ${className}`}
    >
      {children}
    </div>
  );
}

export function Section({
  title,
  subtitle,
  children,
  action,
  id,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  action?: ReactNode;
  id?: string;
}) {
  return (
    <section id={id} className="scroll-mt-24">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-text-primary sm:text-xl">
            {title}
          </h2>
          {subtitle && (
            <p className="mt-1 max-w-2xl text-sm leading-relaxed text-text-secondary">
              {subtitle}
            </p>
          )}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

/**
 * One labelled statistic.
 *
 * `value` accepts a pre-formatted string; formatters in lib/format already
 * return the em dash for null, and this dims the text when they do.
 */
export function Stat({
  label,
  value,
  hint,
  valueClassName = "",
  size = "md",
}: {
  label: string;
  value: string;
  hint?: string;
  valueClassName?: string;
  size?: "sm" | "md" | "lg";
}) {
  const isUnavailable = value === UNAVAILABLE;
  const sizes = {
    sm: "text-base",
    md: "text-xl",
    lg: "text-3xl sm:text-4xl",
  };

  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs font-medium tracking-wide text-text-muted">{label}</span>
      <span
        className={`tnum font-semibold ${sizes[size]} ${
          isUnavailable ? "text-text-faint" : valueClassName || "text-text-primary"
        }`}
        title={isUnavailable ? "Data unavailable from the provider" : undefined}
      >
        {value}
      </span>
      {hint && <span className="text-xs leading-snug text-text-faint">{hint}</span>}
    </div>
  );
}

/** Equal-height grid of stats with consistent gaps. */
export function StatGrid({
  children,
  columns = 4,
}: {
  children: ReactNode;
  columns?: 2 | 3 | 4 | 5;
}) {
  const map = {
    2: "sm:grid-cols-2",
    3: "sm:grid-cols-2 lg:grid-cols-3",
    4: "sm:grid-cols-2 lg:grid-cols-4",
    5: "sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5",
  };
  return <div className={`grid grid-cols-1 gap-5 ${map[columns]}`}>{children}</div>;
}

export function Badge({
  children,
  tone = "neutral",
  className = "",
}: {
  children: ReactNode;
  tone?: "lime" | "bullish" | "bearish" | "neutral" | "warning";
  className?: string;
}) {
  const tones = {
    lime: "bg-lime-faint text-lime border-lime-dim/40",
    bullish: "bg-bullish-dim text-bullish border-bullish/30",
    bearish: "bg-bearish-dim text-bearish border-bearish/30",
    warning: "bg-warning-dim text-warning border-warning/30",
    neutral: "bg-ink-030 text-text-secondary border-line-strong",
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-semibold tracking-wide ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

/**
 * Empty state for unavailable data or unconfigured integrations.
 *
 * Deliberately informative rather than apologetic: it says what is missing and
 * what would fix it, which is the difference between an honest state and a
 * dead end.
 */
export function EmptyState({
  title,
  message,
  action,
  tone = "neutral",
}: {
  title: string;
  message: string;
  action?: ReactNode;
  tone?: "neutral" | "warning";
}) {
  return (
    <Card className={tone === "warning" ? "border-warning/25" : ""}>
      <div className="flex flex-col items-start gap-2">
        <span
          className={`eyebrow ${tone === "warning" ? "text-warning" : "text-text-muted"}`}
        >
          {title}
        </span>
        <p className="max-w-2xl text-sm leading-relaxed text-text-secondary">{message}</p>
        {action}
      </div>
    </Card>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded-md bg-ink-030 ${className}`}
      aria-hidden="true"
    />
  );
}

/** Small info tooltip for methodology notes. */
export function InfoTip({ text }: { text: string }) {
  return (
    <span
      tabIndex={0}
      role="note"
      aria-label={text}
      title={text}
      className="ml-1.5 inline-flex h-4 w-4 cursor-help items-center justify-center rounded-full border border-line-strong text-[10px] font-bold text-text-muted transition-colors hover:border-lime hover:text-lime"
    >
      {/* Hidden from the accessibility tree so the glyph does not concatenate
          onto the heading it annotates; aria-label carries the real content. */}
      <span aria-hidden="true">i</span>
    </span>
  );
}

/** Horizontal bar for comparing a value against a reference. */
export function CompareBar({
  value,
  reference,
  max,
}: {
  value: number;
  reference: number;
  max?: number;
}) {
  const ceiling = max ?? Math.max(value, reference) * 1.25;
  const valuePercent = Math.min(100, (value / ceiling) * 100);
  const referencePercent = Math.min(100, (reference / ceiling) * 100);
  const beatsReference = value > reference;

  return (
    <div className="relative h-2 w-full overflow-hidden rounded-full bg-ink-030">
      <div
        className={`h-full rounded-full ${beatsReference ? "bg-bullish" : "bg-bearish"}`}
        style={{ width: `${valuePercent}%` }}
      />
      {/* Baseline marker — the bar the value has to clear. */}
      <div
        className="absolute top-0 h-full w-0.5 bg-text-primary"
        style={{ left: `${referencePercent}%` }}
        title={`Baseline: ${(reference * 100).toFixed(1)}%`}
      />
    </div>
  );
}
