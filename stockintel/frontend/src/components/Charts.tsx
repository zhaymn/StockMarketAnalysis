"use client";

import { useEffect, useRef, useState } from "react";
import {
  AreaSeries,
  CandlestickSeries,
  ColorType,
  createChart,
  HistogramSeries,
  LineSeries,
  type UTCTimestamp,
} from "lightweight-charts";
import { api } from "@/lib/api";
import { useAsyncData } from "@/lib/useAsyncData";
import type { ChartResponse } from "@/lib/types";
import { Card, EmptyState, Section, Skeleton } from "./ui";

const RANGES = ["1m", "3m", "6m", "1y", "5y", "max"] as const;
type Range = (typeof RANGES)[number];

const PALETTE = {
  background: "#000000",
  grid: "#1a1a1a",
  text: "#737373",
  border: "#1f1f1f",
  lime: "#c1df1f",
  bullish: "#4ade80",
  bearish: "#f87171",
  sma20: "#c1df1f",
  sma50: "#60a5fa",
  sma200: "#a78bfa",
  // Forecast uses a distinct hue from every historical series, so the
  // projection cannot be confused with an indicator line.
  forecast: "#f59e0b",
  forecastBand: "#7c5306",
};

function toTime(dateString: string): UTCTimestamp {
  return (new Date(dateString).getTime() / 1000) as UTCTimestamp;
}

/** Shared chart options so every panel aligns on the same visual grammar. */
function baseOptions(height: number) {
  return {
    height,
    // autoSize makes the chart track its container via an internal
    // ResizeObserver. Without it the chart keeps its creation-time width and
    // props the container open, so the container can never shrink and the
    // chart never gets a resize signal -- a deadlock that overflows the page
    // horizontally on mobile.
    autoSize: true,
    layout: {
      background: { type: ColorType.Solid, color: PALETTE.background },
      textColor: PALETTE.text,
      fontFamily: "var(--font-geist-mono), monospace",
      fontSize: 11,
      attributionLogo: false,
    },
    grid: {
      vertLines: { color: PALETTE.grid },
      horzLines: { color: PALETTE.grid },
    },
    rightPriceScale: { borderColor: PALETTE.border },
    timeScale: { borderColor: PALETTE.border, timeVisible: false },
    crosshair: {
      vertLine: { color: PALETTE.text, labelBackgroundColor: PALETTE.lime },
      horzLine: { color: PALETTE.text, labelBackgroundColor: PALETTE.lime },
    },
  };
}

export function ChartSection({
  marketId,
  symbol,
  target,
  mode,
}: {
  marketId: string;
  symbol: string;
  target: string;
  mode: string;
}) {
  const [range, setRange] = useState<Range>("1y");

  const { data, error, isLoading } = useAsyncData<ChartResponse>(
    `${marketId}|${symbol}|${range}|${target}|${mode}`,
    (signal) => api.chart(marketId, symbol, range, target, mode, signal),
  );

  return (
    <Section
      title="Price &amp; technical charts"
      action={
        <div className="flex rounded-lg border border-line bg-ink-000 p-1">
          {RANGES.map((option) => (
            <button
              key={option}
              onClick={() => setRange(option)}
              className={`rounded px-3 py-1.5 text-xs font-semibold uppercase transition-colors ${
                option === range
                  ? "bg-lime text-ink-000"
                  : "text-text-secondary hover:bg-ink-030 hover:text-text-primary"
              }`}
            >
              {option}
            </button>
          ))}
        </div>
      }
    >
      {error ? (
        <EmptyState
          title="Chart unavailable"
          message={error.message ?? "Could not load chart data."}
          tone="warning"
        />
      ) : isLoading && !data ? (
        <div className="space-y-5">
          <Skeleton className="h-[420px] w-full" />
          <div className="grid gap-5 lg:grid-cols-2">
            <Skeleton className="h-[220px] w-full" />
            <Skeleton className="h-[220px] w-full" />
          </div>
        </div>
      ) : data ? (
        <div className="space-y-5">
          <Card className="p-3 sm:p-4">
            <PriceChart data={data} />
          </Card>
          <div className="grid gap-5 lg:grid-cols-2">
            <Card className="p-3 sm:p-4">
              <RsiChart data={data} />
            </Card>
            <Card className="p-3 sm:p-4">
              <MacdChart data={data} />
            </Card>
          </div>
          <Card className="p-3 sm:p-4">
            <VolatilityChart data={data} />
          </Card>
        </div>
      ) : null}
    </Section>
  );
}

function ChartFrame({
  title,
  subtitle,
  legend,
  containerRef,
  height,
}: {
  title: string;
  subtitle?: string;
  legend?: { label: string; color: string }[];
  containerRef: React.RefObject<HTMLDivElement | null>;
  height: number;
}) {
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <h3 className="eyebrow">{title}</h3>
          {subtitle && <p className="mt-1 text-xs text-text-faint">{subtitle}</p>}
        </div>
        {legend && (
          <div className="flex flex-wrap gap-3">
            {legend.map((item) => (
              <span key={item.label} className="flex items-center gap-1.5 text-xs text-text-muted">
                <span
                  className="h-0.5 w-4 rounded-full"
                  style={{ backgroundColor: item.color }}
                  aria-hidden="true"
                />
                {item.label}
              </span>
            ))}
          </div>
        )}
      </div>
      <div ref={containerRef} style={{ height }} />
    </div>
  );
}

function PriceChart({ data }: { data: ChartResponse }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      ...baseOptions(400),
      rightPriceScale: { borderColor: PALETTE.border, scaleMargins: { top: 0.08, bottom: 0.28 } },
    });

    // v5 API: addSeries(SeriesDefinition, options) -- v4's addCandlestickSeries()
    // and friends were removed.
    const candles = chart.addSeries(CandlestickSeries, {
      upColor: PALETTE.bullish,
      downColor: PALETTE.bearish,
      borderUpColor: PALETTE.bullish,
      borderDownColor: PALETTE.bearish,
      wickUpColor: PALETTE.bullish,
      wickDownColor: PALETTE.bearish,
    });

    const candleData = data.dates
      .map((date, index) => ({
        time: toTime(date),
        open: data.ohlc.open[index],
        high: data.ohlc.high[index],
        low: data.ohlc.low[index],
        close: data.ohlc.close[index],
      }))
      // Bars with any missing leg are dropped, not interpolated.
      .filter(
        (bar): bar is { time: UTCTimestamp; open: number; high: number; low: number; close: number } =>
          bar.open !== null && bar.high !== null && bar.low !== null && bar.close !== null,
      );
    candles.setData(candleData);

    const addMovingAverage = (values: (number | null)[], color: string) => {
      const series = chart.addSeries(LineSeries, {
        color,
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      series.setData(
        data.dates
          .map((date, index) => ({ time: toTime(date), value: values[index] }))
          .filter((point): point is { time: UTCTimestamp; value: number } => point.value !== null),
      );
    };

    addMovingAverage(data.moving_averages.sma_20, PALETTE.sma20);
    addMovingAverage(data.moving_averages.sma_50, PALETTE.sma50);
    addMovingAverage(data.moving_averages.sma_200, PALETTE.sma200);

    // Volume on its own overlay scale, pinned to the lower quarter so it never
    // competes with price for vertical space.
    const volume = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    volume.setData(
      data.dates
        .map((date, index) => {
          const value = data.volume[index];
          const open = data.ohlc.open[index];
          const close = data.ohlc.close[index];
          if (value === null) return null;
          const rising = open !== null && close !== null && close >= open;
          return {
            time: toTime(date),
            value,
            color: rising ? "rgba(74,222,128,0.35)" : "rgba(248,113,113,0.35)",
          };
        })
        .filter((point): point is { time: UTCTimestamp; value: number; color: string } => point !== null),
    );

    // --- Forecast overlay -------------------------------------------------
    // Drawn only when a directional call exists, and drawn as DASHED lines so
    // it can never be mistaken for the solid actual data beside it. It is the
    // historical outcome distribution for the predicted class, not a price
    // path: a single projected line would imply a precision the model does
    // not have.
    const forecast = data.forecast;
    if (forecast?.available) {
      const anchorTime = toTime(forecast.anchor_date);

      const addProjection = (
        values: number[],
        color: string,
        width: 1 | 2,
      ) => {
        const series = chart.addSeries(LineSeries, {
          color,
          lineWidth: width,
          lineStyle: 2, // dashed
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        series.setData([
          // Start at the anchor so the cone visibly originates from the last
          // real close rather than floating unattached.
          { time: anchorTime, value: forecast.anchor_close },
          ...forecast.dates.map((date, index) => ({
            time: toTime(date),
            value: values[index],
          })),
        ]);
      };

      addProjection(forecast.upper, PALETTE.forecastBand, 1);
      addProjection(forecast.lower, PALETTE.forecastBand, 1);
      addProjection(forecast.median, PALETTE.forecast, 2);
    }

    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [data]);

  const forecast = data.forecast;

  return (
    <div>
      <ChartFrame
        title="Price history"
        subtitle={
          forecast?.available
            ? "Solid: actual historical data. Dashed: projected outcome range."
            : "Candlesticks with moving averages and volume. All values are actual historical data."
        }
        legend={[
          { label: "SMA 20", color: PALETTE.sma20 },
          { label: "SMA 50", color: PALETTE.sma50 },
          { label: "SMA 200", color: PALETTE.sma200 },
          ...(forecast?.available
            ? [{ label: `${forecast.horizon_days}d range`, color: PALETTE.forecast }]
            : []),
        ]}
        containerRef={containerRef}
        height={400}
      />

      {forecast?.available && (
        <p className="mt-3 border-t border-line pt-3 text-xs leading-relaxed text-text-muted">
          <span className="font-semibold text-text-secondary">
            The dashed cone is not a price forecast.
          </span>{" "}
          {forecast.caveat} {forecast.basis}
        </p>
      )}
    </div>
  );
}

function RsiChart({ data }: { data: ChartResponse }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || !data.rsi_14) return;

    const chart = createChart(containerRef.current, baseOptions(200));
    const series = chart.addSeries(LineSeries, {
      color: PALETTE.lime,
      lineWidth: 2,
      priceLineVisible: false,
    });

    series.setData(
      data.dates
        .map((date, index) => ({ time: toTime(date), value: data.rsi_14![index] }))
        .filter((point): point is { time: UTCTimestamp; value: number } => point.value !== null),
    );

    // Conventional 70/30 bands.
    series.createPriceLine({ price: 70, color: PALETTE.bearish, lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "" });
    series.createPriceLine({ price: 30, color: PALETTE.bullish, lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "" });

    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [data]);

  if (!data.rsi_14) {
    return <EmptyState title="RSI unavailable" message="RSI could not be computed for this range." />;
  }

  return (
    <ChartFrame
      title="RSI (14)"
      subtitle="Above 70 conventionally overbought, below 30 oversold."
      containerRef={containerRef}
      height={200}
    />
  );
}

function MacdChart({ data }: { data: ChartResponse }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || !data.macd.macd) return;

    const chart = createChart(containerRef.current, baseOptions(200));

    const histogram = chart.addSeries(HistogramSeries, { priceLineVisible: false });
    histogram.setData(
      data.dates
        .map((date, index) => {
          const value = data.macd.histogram?.[index] ?? null;
          if (value === null) return null;
          return {
            time: toTime(date),
            value,
            color: value >= 0 ? "rgba(74,222,128,0.5)" : "rgba(248,113,113,0.5)",
          };
        })
        .filter((point): point is { time: UTCTimestamp; value: number; color: string } => point !== null),
    );

    const line = (values: (number | null)[] | null, color: string) => {
      if (!values) return;
      const series = chart.addSeries(LineSeries, { color, lineWidth: 2, priceLineVisible: false });
      series.setData(
        data.dates
          .map((date, index) => ({ time: toTime(date), value: values[index] }))
          .filter((point): point is { time: UTCTimestamp; value: number } => point.value !== null),
      );
    };

    line(data.macd.macd, PALETTE.lime);
    line(data.macd.signal, PALETTE.sma50);

    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [data]);

  if (!data.macd.macd) {
    return <EmptyState title="MACD unavailable" message="MACD could not be computed for this range." />;
  }

  return (
    <ChartFrame
      title="MACD (12, 26, 9)"
      subtitle="Normalised by price so values compare across securities."
      legend={[
        { label: "MACD", color: PALETTE.lime },
        { label: "Signal", color: PALETTE.sma50 },
      ]}
      containerRef={containerRef}
      height={200}
    />
  );
}

function VolatilityChart({ data }: { data: ChartResponse }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || !data.volatility_21d) return;

    const chart = createChart(containerRef.current, baseOptions(200));
    const series = chart.addSeries(AreaSeries, {
      lineColor: PALETTE.lime,
      topColor: "rgba(193,223,31,0.25)",
      bottomColor: "rgba(193,223,31,0.01)",
      lineWidth: 2,
      priceLineVisible: false,
    });

    series.setData(
      data.dates
        .map((date, index) => ({ time: toTime(date), value: data.volatility_21d![index] }))
        .filter((point): point is { time: UTCTimestamp; value: number } => point.value !== null),
    );

    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [data]);

  if (!data.volatility_21d) {
    return <EmptyState title="Volatility unavailable" message="Rolling volatility could not be computed." />;
  }

  return (
    <ChartFrame
      title="Rolling volatility (21-day, annualised)"
      containerRef={containerRef}
      height={200}
    />
  );
}
