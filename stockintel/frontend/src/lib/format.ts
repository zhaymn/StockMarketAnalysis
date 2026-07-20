/**
 * Display formatting.
 *
 * Every formatter takes `number | null` and renders null as an em dash. This
 * is the UI half of the no-fake-data rule: a missing metric must look missing,
 * never like a real zero.
 */

/** The single placeholder for unavailable data, used everywhere. */
export const UNAVAILABLE = "—";

export function formatNumber(
  value: number | null | undefined,
  digits = 2,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return UNAVAILABLE;
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatPrice(
  value: number | null | undefined,
  currencySymbol = "",
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return UNAVAILABLE;
  return `${currencySymbol}${value.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

export function formatPercent(
  value: number | null | undefined,
  digits = 2,
  withSign = false,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return UNAVAILABLE;
  const sign = withSign && value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(digits)}%`;
}

/** Compact large numbers: 4.91T, 17.9B, 312M. */
export function formatCompact(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return UNAVAILABLE;

  const abs = Math.abs(value);
  const units: [number, string][] = [
    [1e12, "T"],
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "K"],
  ];

  for (const [threshold, suffix] of units) {
    if (abs >= threshold) return `${(value / threshold).toFixed(2)}${suffix}`;
  }
  return value.toFixed(0);
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return UNAVAILABLE;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return UNAVAILABLE;
  return date.toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return UNAVAILABLE;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return UNAVAILABLE;
  return date.toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function relativeTime(isoString: string): string {
  const then = new Date(isoString).getTime();
  if (Number.isNaN(then)) return UNAVAILABLE;

  const minutes = Math.floor((Date.now() - then) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

/** Tailwind text colour for a signed value. Green/red, never lime. */
export function signColor(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "text-text-muted";
  }
  if (value > 0) return "text-bullish";
  if (value < 0) return "text-bearish";
  return "text-text-secondary";
}

/** Human label for an enum-ish backend string: STRONG_UPTREND -> Strong uptrend. */
export function humanise(value: string | null | undefined): string {
  if (!value) return UNAVAILABLE;
  const spaced = value.replace(/_/g, " ").toLowerCase();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}
