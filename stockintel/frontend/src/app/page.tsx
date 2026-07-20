"use client";

import { useCallback, useEffect, useState } from "react";
import { api, ApiRequestError } from "@/lib/api";
import { useAsyncData } from "@/lib/useAsyncData";
import type {
  AnalysisResponse,
  Market,
  ModelMode,
  Stock,
  TargetSpec,
} from "@/lib/types";
import { ControlBar } from "@/components/ControlBar";
import { StockHeader } from "@/components/StockHeader";
import { PredictionCard } from "@/components/PredictionCard";
import { ChartSection } from "@/components/Charts";
import { DetailedAnalytics, Fundamentals, KeyStatistics } from "@/components/Analytics";
import { ModelEvidence, ModelTransparency, WhyThisPrediction } from "@/components/Evidence";
import { NewsSection } from "@/components/NewsSection";
import { Card, EmptyState, Skeleton } from "@/components/ui";

const DEFAULT_STOCK: Record<string, Stock> = {
  us: { symbol: "AAPL", name: "Apple Inc.", exchange: "NASDAQ", label: "Apple Inc. (AAPL)" },
  india: {
    symbol: "RELIANCE.NS",
    name: "Reliance Industries Limited",
    exchange: "NSE",
    label: "Reliance Industries Limited (RELIANCE.NS)",
  },
};

/**
 * The dashboard.
 *
 * Section order follows the brief deliberately: prediction, then the evidence
 * behind it, then deep analytics, then news context. Users see the claim, then
 * immediately what supports or contradicts it.
 */
export default function Dashboard() {
  const [markets, setMarkets] = useState<Market[]>([]);
  const [modes, setModes] = useState<ModelMode[]>([]);
  const [targets, setTargets] = useState<TargetSpec[]>([]);
  const [disclaimer, setDisclaimer] = useState("");

  const [marketId, setMarketId] = useState("us");
  const [stock, setStock] = useState<Stock | null>(DEFAULT_STOCK.us);
  const [mode, setMode] = useState("most_possible");
  const [target, setTarget] = useState("outlook_5d");

  const [bootError, setBootError] = useState<string | null>(null);

  // Keyed on the full selection, so switching any control refetches and stale
  // results for a previous selection are never rendered.
  const {
    data: analysis,
    error,
    isLoading,
  } = useAsyncData<AnalysisResponse>(
    stock ? `${marketId}|${stock.symbol}|${target}|${mode}` : null,
    (signal) => api.analysis(marketId, stock!.symbol, target, mode, signal),
  );

  // Bootstrap: markets and model metadata.
  useEffect(() => {
    const controller = new AbortController();

    Promise.all([api.markets(controller.signal), api.models(controller.signal)])
      .then(([marketData, modelData]) => {
        setMarkets(marketData.markets);
        setModes(modelData.modes);
        setTargets(modelData.targets);
        setDisclaimer(modelData.disclaimer);
      })
      .catch((cause) => {
        if (cause instanceof DOMException) return;
        setBootError(
          cause instanceof ApiRequestError
            ? (cause.payload.detail ?? cause.message)
            : "Could not reach the API.",
        );
      });

    return () => controller.abort();
  }, []);

  const handleMarketChange = useCallback((nextMarket: string) => {
    setMarketId(nextMarket);
    // Switching markets must also switch the stock — currency, benchmark and
    // ticker conventions all change with it.
    setStock(DEFAULT_STOCK[nextMarket] ?? null);
  }, []);

  if (bootError) {
    return (
      <main className="mx-auto max-w-3xl px-6 py-24">
        <EmptyState
          tone="warning"
          title="Backend unavailable"
          message={bootError}
          action={
            <code className="mt-3 block rounded-lg bg-ink-020 p-4 text-xs text-lime">
              cd stockintel/backend
              <br />
              .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
            </code>
          }
        />
      </main>
    );
  }

  return (
    <div className="min-h-screen bg-ink-000">
      <TopBar />

      {markets.length > 0 && (
        <ControlBar
          markets={markets}
          activeMarket={marketId}
          onMarketChange={handleMarketChange}
          activeStock={stock}
          onStockChange={setStock}
          modes={modes}
          activeMode={mode}
          onModeChange={setMode}
          targets={targets}
          activeTarget={target}
          onTargetChange={setTarget}
          isLoading={isLoading}
        />
      )}

      {analysis && <StockHeader analysis={analysis} />}

      <main className="mx-auto max-w-[1600px] space-y-10 px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
        {isLoading && !analysis ? (
          <LoadingState />
        ) : error ? (
          <EmptyState
            tone="warning"
            title={error.code.replace(/_/g, " ").toUpperCase()}
            message={`${error.message}${error.payload.detail ? ` ${error.payload.detail}` : ""}`}
          />
        ) : analysis ? (
          <>
            <PredictionCard analysis={analysis} />
            <KeyStatistics analysis={analysis} />
            <ChartSection marketId={marketId} symbol={analysis.symbol} />
            <DetailedAnalytics analysis={analysis} />
            <Fundamentals analysis={analysis} />
            <ModelEvidence analysis={analysis} />
            <WhyThisPrediction analysis={analysis} />
            <NewsSection symbol={analysis.symbol} />
            <ModelTransparency analysis={analysis} />
            <Disclaimer text={disclaimer} />
          </>
        ) : null}
      </main>
    </div>
  );
}

function TopBar() {
  return (
    <header className="border-b border-line bg-ink-000">
      <div className="mx-auto flex max-w-[1600px] items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-lime">
            <span className="text-sm font-bold text-ink-000">SI</span>
          </div>
          <div>
            <span className="text-sm font-semibold tracking-tight text-text-primary">
              StockIntel
            </span>
            <span className="ml-2 text-xs text-text-muted">
              Prediction &amp; market intelligence
            </span>
          </div>
        </div>
        <span className="hidden text-xs text-text-faint sm:block">
          Predictions are withheld where models show no validated edge
        </span>
      </div>
    </header>
  );
}

function LoadingState() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-56 w-full" />
      <Skeleton className="h-32 w-full" />
      <Skeleton className="h-96 w-full" />
    </div>
  );
}

function Disclaimer({ text }: { text: string }) {
  return (
    <Card className="border-line-strong">
      <h3 className="eyebrow mb-2">Important</h3>
      <p className="text-sm leading-relaxed text-text-secondary">{text}</p>
      <p className="mt-3 text-xs leading-relaxed text-text-faint">
        Market data is delayed and provided for analysis only. Model performance
        measured on historical data does not guarantee future results. This platform
        withholds directional calls where a model fails to beat a naive baseline
        out-of-sample.
      </p>
    </Card>
  );
}
