"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { useAsyncData } from "@/lib/useAsyncData";
import type { Market, ModelMode, Stock, TargetSpec } from "@/lib/types";

/** Market, stock, model and horizon selection. */
export function ControlBar({
  markets,
  activeMarket,
  onMarketChange,
  activeStock,
  onStockChange,
  modes,
  activeMode,
  onModeChange,
  targets,
  activeTarget,
  onTargetChange,
  isLoading,
}: {
  markets: Market[];
  activeMarket: string;
  onMarketChange: (marketId: string) => void;
  activeStock: Stock | null;
  onStockChange: (stock: Stock) => void;
  modes: ModelMode[];
  activeMode: string;
  onModeChange: (mode: string) => void;
  targets: TargetSpec[];
  activeTarget: string;
  onTargetChange: (target: string) => void;
  isLoading: boolean;
}) {
  return (
    <div className="border-b border-line bg-ink-010/80 backdrop-blur-sm">
      <div className="mx-auto max-w-[110rem] px-4 py-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          <div className="lg:col-span-2">
            <FieldLabel>Market</FieldLabel>
            <MarketSelector
              markets={markets}
              active={activeMarket}
              onChange={onMarketChange}
            />
          </div>

          <div className="lg:col-span-4">
            <FieldLabel>Stock</FieldLabel>
            <StockSearch
              marketId={activeMarket}
              active={activeStock}
              onSelect={onStockChange}
            />
          </div>

          <div className="lg:col-span-4">
            <FieldLabel>Prediction model</FieldLabel>
            <ModelSelector modes={modes} active={activeMode} onChange={onModeChange} />
          </div>

          <div className="lg:col-span-2">
            <FieldLabel>Horizon</FieldLabel>
            <HorizonSelector
              targets={targets}
              active={activeTarget}
              onChange={onTargetChange}
              disabled={isLoading}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return <div className="eyebrow mb-2">{children}</div>;
}

function MarketSelector({
  markets,
  active,
  onChange,
}: {
  markets: Market[];
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div
      role="tablist"
      aria-label="Market"
      className="flex rounded-lg border border-line bg-ink-000 p-1"
    >
      {markets.map((market) => {
        const isActive = market.market_id === active;
        return (
          <button
            key={market.market_id}
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(market.market_id)}
            className={`flex-1 rounded-md px-3 py-2 text-sm font-semibold transition-colors ${
              isActive
                ? "bg-lime text-ink-000"
                : "text-text-secondary hover:bg-ink-030 hover:text-text-primary"
            }`}
          >
            {market.market_label}
          </button>
        );
      })}
    </div>
  );
}

function StockSearch({
  marketId,
  active,
  onSelect,
}: {
  marketId: string;
  active: Stock | null;
  onSelect: (stock: Stock) => void;
}) {
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Debounce the query so a live search does not fire on every keystroke.
  // setState here is inside a timer callback, not the effect body, so it does
  // not trigger the cascading-render pattern.
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query.trim()), 250);
    return () => clearTimeout(timer);
  }, [query]);

  // Below two characters, show the browsable directory instead of searching.
  const isSearch = debouncedQuery.length >= 2;
  const { data, isLoading: isSearching } = useAsyncData<{ stocks: Stock[] }>(
    `${marketId}|${isSearch ? debouncedQuery : "__directory__"}`,
    (signal) =>
      isSearch
        ? api.search(marketId, debouncedQuery, signal)
        : api.directory(marketId, signal),
  );

  const results = data?.stocks ?? [];

  // Close on outside click.
  useEffect(() => {
    function handleClick(event: MouseEvent) {
      if (!containerRef.current?.contains(event.target as Node)) setIsOpen(false);
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  return (
    <div ref={containerRef} className="relative">
      <input
        type="text"
        value={query}
        onChange={(event) => {
          setQuery(event.target.value);
          setIsOpen(true);
        }}
        onFocus={() => setIsOpen(true)}
        placeholder={active ? `${active.name} (${active.symbol})` : "Search company or ticker…"}
        aria-label="Search for a stock"
        className="w-full rounded-lg border border-line bg-ink-000 px-3 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:border-lime focus:outline-none"
      />

      {isOpen && (
        <div className="absolute z-50 mt-2 max-h-80 w-full overflow-y-auto rounded-lg border border-line-strong bg-ink-020 shadow-2xl shadow-black/60">
          {isSearching && results.length === 0 ? (
            <div className="px-3 py-4 text-sm text-text-muted">Searching…</div>
          ) : results.length === 0 ? (
            <div className="px-3 py-4 text-sm text-text-muted">
              No matching stocks found.
            </div>
          ) : (
            results.map((stock) => (
              <button
                key={stock.symbol}
                onClick={() => {
                  onSelect(stock);
                  setIsOpen(false);
                  setQuery("");
                }}
                className="flex w-full items-center justify-between gap-3 border-b border-line px-3 py-2.5 text-left last:border-0 hover:bg-ink-030"
              >
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-text-primary">
                    {stock.name}
                  </span>
                  <span className="tnum text-xs text-text-muted">{stock.symbol}</span>
                </span>
                <span className="shrink-0 text-xs text-text-faint">{stock.exchange}</span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

function ModelSelector({
  modes,
  active,
  onChange,
}: {
  modes: ModelMode[];
  active: string;
  onChange: (id: string) => void;
}) {
  const activeMode = modes.find((mode) => mode.id === active);

  return (
    <div>
      <div className="flex flex-wrap gap-2">
        {modes.map((mode) => {
          const isActive = mode.id === active;
          return (
            <button
              key={mode.id}
              onClick={() => onChange(mode.id)}
              className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? "border-lime bg-lime-faint text-lime"
                  : "border-line bg-ink-000 text-text-secondary hover:border-line-strong hover:text-text-primary"
              }`}
            >
              {mode.label}
              {mode.recommended && (
                <span
                  className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                    isActive ? "bg-lime text-ink-000" : "bg-ink-030 text-text-muted"
                  }`}
                >
                  Default
                </span>
              )}
            </button>
          );
        })}
      </div>
      {activeMode && (
        <p className="mt-2 text-xs leading-relaxed text-text-muted">
          {activeMode.description}
        </p>
      )}
    </div>
  );
}

function HorizonSelector({
  targets,
  active,
  onChange,
  disabled,
}: {
  targets: TargetSpec[];
  active: string;
  onChange: (name: string) => void;
  disabled: boolean;
}) {
  return (
    <select
      value={active}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value)}
      aria-label="Prediction horizon"
      className="w-full rounded-lg border border-line bg-ink-000 px-3 py-2.5 text-sm text-text-primary focus:border-lime focus:outline-none disabled:opacity-50"
    >
      {targets.map((target) => (
        <option key={target.name} value={target.name}>
          {target.horizon_days === 1
            ? "1 trading day"
            : `${target.horizon_days} trading days`}
        </option>
      ))}
    </select>
  );
}
