/**
 * Types mirroring the backend API payloads.
 *
 * Nullable fields are nullable on purpose: the backend returns `null` for any
 * metric it could not compute rather than defaulting to zero, and the UI is
 * required to render those as DATA UNAVAILABLE. Typing them as `number` would
 * let a `?? 0` slip in and turn missing data into a fabricated value.
 */

export type Verdict = "DIRECTIONAL" | "NO_EDGE" | "ABSTAINED" | "INSUFFICIENT_DATA";
export type RiskLevel = "LOW" | "MODERATE" | "ELEVATED" | "HIGH";
export type Freshness = "DELAYED" | "END_OF_DAY" | "CACHED" | "UNAVAILABLE";

export interface MarketConventions {
  currency_code: string;
  currency_symbol: string;
  benchmark_symbol: string;
  benchmark_label: string;
  timezone: string;
  open_time: string;
  close_time: string;
  trading_days_per_year: number;
}

export interface Market {
  market_id: string;
  market_label: string;
  conventions: MarketConventions;
}

export interface Stock {
  symbol: string;
  name: string;
  exchange: string;
  label: string;
}

export interface TargetSpec {
  name: string;
  horizon_days: number;
  n_classes: number;
  class_labels: string[];
  description: string;
}

export interface ModelMode {
  id: string;
  label: string;
  recommended: boolean;
  description: string;
}

export interface Factor {
  factor: string;
  evidence: string;
}

export interface WalkForwardAggregate {
  n_folds: number;
  total_test_samples: number;
  accuracy_mean: number;
  accuracy_std: number;
  baseline_accuracy_mean: number;
  skill_score_mean: number;
  f1_macro_mean: number;
  matthews_corrcoef_mean: number;
  roc_auc_mean: number | null;
  folds_beating_baseline: number;
  consistency: string;
}

export interface CalibrationReport {
  brier_score: number;
  baseline_brier: number;
  brier_skill_score: number;
  is_calibrated: boolean;
  expected_calibration_error: number;
  max_calibration_error: number;
  reliability_bins: {
    bin_lower: number;
    bin_upper: number;
    mean_predicted: number;
    observed_frequency: number;
    count: number;
  }[];
  interpretation: string;
}

export interface RegimeStats {
  n_samples: number;
  accuracy: number;
  baseline_accuracy: number;
  skill_score: number;
  beats_baseline: boolean;
}

export interface Prediction {
  verdict: Verdict;
  symbol: string;
  target: TargetSpec;
  model_mode: string;
  model_label: string;
  direction: string | null;
  probability: number | null;
  probability_is_calibrated: boolean;
  probability_withheld_reason: string | null;
  expected_return_range: {
    p10: number;
    p25: number;
    median: number;
    p75: number;
    p90: number;
    n_observations: number;
    basis: string;
  } | null;
  risk_level: RiskLevel;
  evidence: {
    walk_forward: WalkForwardAggregate;
    baseline: WalkForwardAggregate;
    calibration: CalibrationReport | null;
    regime_breakdown: Record<string, RegimeStats>;
    class_balance: Record<string, number>;
    n_samples: number;
    effective_sample_size: number;
    training_period: [string, string];
    top_features: { feature: string; importance: number }[];
  };
  factors: { bullish: Factor[]; bearish: Factor[]; risks: Factor[] };
  interpretation: string;
  data_timestamp: string | null;
}

export interface Analytics {
  price: {
    current: number | null;
    previous_close: number | null;
    change: number | null;
    change_percent: number | null;
    high_52w: number | null;
    low_52w: number | null;
    range_window_sessions: number;
    has_full_52w_history: boolean;
    position_in_52w_range: number | null;
    last_close_date: string;
    day_high: number | null;
    day_low: number | null;
  };
  returns: Record<string, number | null>;
  volatility: {
    realised_21d: number | null;
    realised_63d: number | null;
    realised_252d: number | null;
    atr_14_percent: number | null;
    downside_volatility: number | null;
    regime_ratio: number | null;
    regime: string;
    annualisation_basis: number;
  };
  momentum: {
    available: boolean;
    rsi_14?: number | null;
    rsi_state?: string;
    macd?: number | null;
    macd_signal?: number | null;
    macd_histogram?: number | null;
    macd_state?: string;
    distance_from_sma_20?: number | null;
    distance_from_sma_50?: number | null;
    distance_from_sma_200?: number | null;
    moving_average_trend?: string;
    trend_strength?: number | null;
  };
  volume: {
    available: boolean;
    reason?: string;
    latest?: number | null;
    average_20d?: number | null;
    average_60d?: number | null;
    relative_volume?: number | null;
    trend?: string;
  };
  risk: {
    max_drawdown: number | null;
    current_drawdown: number | null;
    sharpe_like_ratio: number | null;
    sharpe_note: string;
    beta: number | null;
    correlation_with_benchmark: number | null;
    benchmark_overlap_sessions: number;
  };
  benchmark: {
    available: boolean;
    benchmark_symbol: string;
    benchmark_label: string;
    reason?: string;
    overlap_sessions?: number;
    relative_performance?: Record<
      string,
      { stock: number | null; benchmark: number | null; excess: number | null } | null
    >;
  };
}

export interface AnalysisResponse {
  symbol: string;
  market: Market;
  session: {
    status: string;
    status_label: string;
    exchange_time: string;
    timezone: string;
    next_open: string | null;
    last_session_date: string | null;
  };
  data: {
    freshness: Freshness;
    fetched_at: number;
    served_from_cache: boolean;
    quality: {
      rows_received: number;
      rows_usable: number;
      dropped_null_close: number;
      is_clean: boolean;
      warnings: string[];
    };
    first_session: string;
    last_session: string;
    sessions: number;
  };
  profile: {
    available?: boolean;
    reason?: string;
    name?: string;
    sector?: string | null;
    industry?: string | null;
    summary?: string;
    fundamentals?: Record<string, number | null>;
    unavailable_fields?: string[];
  };
  prediction: Prediction;
  analytics: Analytics;
  feature_groups: { name: string; description: string; n_features: number }[];
}

export interface ChartResponse {
  symbol: string;
  range: string;
  currency: string;
  dates: string[];
  ohlc: {
    open: (number | null)[];
    high: (number | null)[];
    low: (number | null)[];
    close: (number | null)[];
  };
  volume: (number | null)[];
  moving_averages: {
    sma_20: (number | null)[];
    sma_50: (number | null)[];
    sma_200: (number | null)[];
  };
  rsi_14: (number | null)[] | null;
  macd: {
    macd: (number | null)[] | null;
    signal: (number | null)[] | null;
    histogram: (number | null)[] | null;
  };
  volatility_21d: (number | null)[] | null;
}

export interface IntegrationStatus {
  configured: boolean;
  provider: string;
  requires_key: boolean;
  env_var?: string;
  obtain_at?: string;
  free_tier?: string;
  note: string;
}

/** Structured error from the backend. `code` selects the UI empty state. */
export interface ApiError {
  code: string;
  message: string;
  detail?: string;
  integration?: string;
  env_var?: string;
  obtain_at?: string;
  reason?: string;
}
