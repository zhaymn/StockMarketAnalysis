/**
 * Backend API client.
 *
 * Errors are surfaced as `ApiRequestError` carrying the backend's structured
 * payload, so components can switch on `code` to render the right honest state
 * (NOT CONFIGURED, DATA UNAVAILABLE, INSUFFICIENT HISTORY) rather than a
 * generic failure message.
 */

import type {
  AnalysisResponse,
  ApiError,
  ChartResponse,
  IntegrationStatus,
  MacroSnapshot,
  Market,
  ModelMode,
  Stock,
  TargetSpec,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiRequestError extends Error {
  constructor(
    readonly status: number,
    readonly payload: ApiError,
  ) {
    super(payload.message);
    this.name = "ApiRequestError";
  }

  get code(): string {
    return this.payload.code;
  }

  /** True when the failure is a missing API key rather than a real fault. */
  get isNotConfigured(): boolean {
    return this.payload.code === "not_configured";
  }
}

async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${BASE_URL}${path}`, { signal });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
    throw new ApiRequestError(0, {
      code: "network_error",
      message: "Could not reach the StockIntel API.",
      detail: `Is the backend running at ${BASE_URL}? Start it with: uvicorn app.main:app --port 8000`,
    });
  }

  if (!response.ok) {
    let payload: ApiError;
    try {
      payload = (await response.json()) as ApiError;
    } catch {
      payload = {
        code: "unknown_error",
        message: `Request failed with status ${response.status}.`,
      };
    }
    throw new ApiRequestError(response.status, payload);
  }

  return (await response.json()) as T;
}

export const api = {
  markets: (signal?: AbortSignal) =>
    request<{ markets: Market[]; default_market: string }>("/api/markets", signal),

  directory: (marketId: string, signal?: AbortSignal) =>
    request<{ stocks: Stock[]; count: number }>(
      `/api/markets/${marketId}/directory`,
      signal,
    ),

  search: (marketId: string, query: string, signal?: AbortSignal) =>
    request<{ stocks: Stock[]; count: number }>(
      `/api/markets/${marketId}/search?q=${encodeURIComponent(query)}`,
      signal,
    ),

  models: (signal?: AbortSignal) =>
    request<{
      modes: ModelMode[];
      default_mode: string;
      targets: TargetSpec[];
      default_target: string;
      disclaimer: string;
    }>("/api/models", signal),

  integrations: (signal?: AbortSignal) =>
    request<{ integrations: Record<string, IntegrationStatus> }>(
      "/api/integrations",
      signal,
    ),

  analysis: (
    marketId: string,
    symbol: string,
    target: string,
    mode: string,
    signal?: AbortSignal,
  ) =>
    request<AnalysisResponse>(
      `/api/analysis/${marketId}/${encodeURIComponent(symbol)}?target=${target}&mode=${mode}`,
      signal,
    ),

  chart: (
    marketId: string,
    symbol: string,
    range: string,
    target: string,
    mode: string,
    signal?: AbortSignal,
  ) =>
    request<ChartResponse>(
      `/api/analysis/${marketId}/${encodeURIComponent(symbol)}/chart`
        + `?range=${range}&target=${target}&mode=${mode}`,
      signal,
    ),

  macro: (marketId: string, signal?: AbortSignal) =>
    request<MacroSnapshot>(`/api/macro/${marketId}`, signal),

  news: (symbol: string, signal?: AbortSignal) =>
    request<Record<string, unknown>>(`/api/news/${encodeURIComponent(symbol)}`, signal),
};
