"""The main analysis endpoint: everything the dashboard renders for one stock."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from app.core.errors import DataUnavailableError
from app.core.logging import get_logger
from app.data.fundamentals.profile import fetch_profile
from app.data.market.calendar import classify_freshness, get_session_info
from app.data.market.prices import fetch_history
from app.data.market.registry import get_market_provider
from app.features.technical import build_features
from app.services.analytics import build_analytics
from app.services.prediction import predict

logger = get_logger(__name__)
router = APIRouter(prefix="/api/analysis", tags=["analysis"])

#: Chart windows in trading sessions.
CHART_RANGES = {"1m": 21, "3m": 63, "6m": 126, "1y": 252, "5y": 1260, "max": 100_000}


def _load_benchmark(symbol: str) -> pd.Series | None:
    """Benchmark closes, or None if unavailable.

    Benchmark failure must not sink the whole analysis — beta and relative
    performance simply report as unavailable.
    """
    try:
        return fetch_history(symbol, period="10y").frame["Close"]
    except Exception as exc:
        logger.warning("Benchmark %s unavailable: %s", symbol, exc)
        return None


@router.get("/{market_id}/{symbol}")
async def analyse(
    market_id: str,
    symbol: str,
    target: str = Query("outlook_5d"),
    mode: str = Query("most_possible"),
    period: str = Query("10y"),
) -> dict[str, object]:
    """Full analysis for one stock: header, prediction, analytics, evidence."""
    provider = get_market_provider(market_id)
    conventions = provider.conventions

    history = fetch_history(symbol, period=period)
    frame = history.frame

    if len(frame) < 250:
        raise DataUnavailableError(
            f"'{symbol}' has only {len(frame)} sessions of history.",
            detail=(
                "At least 250 sessions are required to compute the full feature set "
                "and validate a model. Analytics for shorter histories would be "
                "computed from partial indicator windows and are not shown."
            ),
        )

    features, feature_groups = build_features(
        frame, trading_days_per_year=conventions.trading_days_per_year
    )
    benchmark_close = _load_benchmark(conventions.benchmark_symbol)

    analytics = build_analytics(frame, features, conventions, benchmark_close).to_dict()

    session = get_session_info(conventions, last_session_date=history.last_session_date)
    freshness = classify_freshness(
        served_from_cache=history.served_from_cache, session_status=session.status
    )

    try:
        profile = fetch_profile(symbol)
        profile_payload = profile.to_dict()
    except Exception as exc:
        logger.info("No profile for %s: %s", symbol, exc)
        profile_payload = {"available": False, "reason": str(exc)}

    prediction = predict(
        symbol=symbol,
        frame=frame,
        features=features,
        conventions=conventions,
        analytics=analytics,
        target_name=target,
        model_mode=mode,
    )

    return {
        "symbol": symbol,
        "market": provider.to_dict(),
        "session": session.to_dict(),
        "data": {
            "freshness": freshness.value,
            "fetched_at": history.fetched_at,
            "served_from_cache": history.served_from_cache,
            "quality": history.quality.to_dict(),
            "first_session": frame.index[0].date().isoformat(),
            "last_session": frame.index[-1].date().isoformat(),
            "sessions": len(frame),
        },
        "profile": profile_payload,
        "prediction": prediction.to_dict(),
        "analytics": analytics,
        "feature_groups": [
            {"name": g.name, "description": g.description, "n_features": len(g.columns)}
            for g in feature_groups
        ],
    }


def _forecast_projection(
    frame: pd.DataFrame,
    prediction: dict[str, object],
    conventions,
) -> dict[str, object] | None:
    """Project the prediction's historical outcome range forward from the last close.

    Deliberately **not** a forecast price line. The model predicts a direction,
    not a path, so drawing a single projected price would invent a precision
    that does not exist and would sit on the chart looking exactly like the
    actual data beside it.

    What is drawn instead is the empirical distribution of what actually
    happened, historically, over this horizon, on days that resolved to the
    predicted class — the same quantiles shown in the prediction card. It is a
    cone of historical outcomes, not a claim about this particular future.

    Returns None when no directional call was issued, which is the common case.
    Showing a cone under a NO_EDGE verdict would contradict the verdict.
    """
    if prediction.get("verdict") != "DIRECTIONAL":
        return None

    quantiles = prediction.get("expected_return_range")
    if not quantiles:
        return None

    last_close = float(frame["Close"].iloc[-1])
    last_date = frame.index[-1]
    horizon = int(prediction["target"]["horizon_days"])

    # Project over business days. This is a display axis, not a claim about
    # which specific sessions the exchange will hold.
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon
    )

    def path(total_return: float) -> list[float]:
        # Interpolate linearly in log space so the endpoint is exact and the
        # intermediate points are not implied to be predictions in themselves.
        end_log = float(np.log1p(total_return))
        return [
            round(last_close * float(np.exp(end_log * (i + 1) / horizon)), 4)
            for i in range(horizon)
        ]

    return {
        "available": True,
        "anchor_date": last_date.date().isoformat(),
        "anchor_close": round(last_close, 4),
        "dates": [d.date().isoformat() for d in future_dates],
        "median": path(quantiles["median"]),
        "upper": path(quantiles["p90"]),
        "lower": path(quantiles["p10"]),
        "direction": prediction.get("direction"),
        "horizon_days": horizon,
        "basis": quantiles["basis"],
        "caveat": (
            "This is the historical distribution of outcomes for this predicted "
            "class, projected from the last close. It is not a price forecast, "
            "and the band is not a confidence interval for this specific future."
        ),
    }


@router.get("/{market_id}/{symbol}/chart")
async def chart(
    market_id: str,
    symbol: str,
    range: str = Query("1y", pattern="^(1m|3m|6m|1y|5y|max)$"),
    target: str = Query("outlook_5d"),
    mode: str = Query("most_possible"),
) -> dict[str, object]:
    """OHLCV plus indicator series for the charts.

    Returned as parallel arrays keyed by ISO date so the frontend can align
    price, volume and indicator panels on a shared time axis without
    re-deriving anything.
    """
    provider = get_market_provider(market_id)
    conventions = provider.conventions

    history = fetch_history(symbol, period="10y")
    frame = history.frame
    features, _ = build_features(
        frame, trading_days_per_year=conventions.trading_days_per_year, drop_warmup=False
    )

    sessions = CHART_RANGES[range]
    window = frame.iloc[-sessions:] if sessions < len(frame) else frame
    feature_window = features.reindex(window.index)

    close = window["Close"]

    def series(values: pd.Series) -> list[float | None]:
        return [None if pd.isna(v) or not np.isfinite(v) else round(float(v), 6) for v in values]

    # Moving averages are reconstructed from the full frame then windowed, so
    # the SMA-200 at the left edge of a 1-month view is still a true 200-day
    # average rather than a partial one.
    full_close = frame["Close"]
    sma_20 = full_close.rolling(20, min_periods=20).mean().reindex(window.index)
    sma_50 = full_close.rolling(50, min_periods=50).mean().reindex(window.index)
    sma_200 = full_close.rolling(200, min_periods=200).mean().reindex(window.index)

    rsi = feature_window.get("rsi_14")
    rsi_display = (rsi * 50.0 + 50.0) if rsi is not None else None

    return {
        "symbol": symbol,
        "range": range,
        "currency": conventions.currency_code,
        "dates": [d.date().isoformat() for d in window.index],
        "ohlc": {
            "open": series(window["Open"]),
            "high": series(window["High"]),
            "low": series(window["Low"]),
            "close": series(close),
        },
        "volume": [int(v) if pd.notna(v) else None for v in window["Volume"]],
        "moving_averages": {
            "sma_20": series(sma_20),
            "sma_50": series(sma_50),
            "sma_200": series(sma_200),
        },
        "rsi_14": series(rsi_display) if rsi_display is not None else None,
        "macd": {
            "macd": series(feature_window["macd"]) if "macd" in feature_window else None,
            "signal": series(feature_window["macd_signal"]) if "macd_signal" in feature_window else None,
            "histogram": (
                series(feature_window["macd_histogram"])
                if "macd_histogram" in feature_window else None
            ),
        },
        "volatility_21d": (
            series(feature_window["volatility_21d"])
            if "volatility_21d" in feature_window else None
        ),
        "forecast": _forecast_cached(market_id, symbol, frame, features, conventions, target, mode),
    }


def _forecast_cached(
    market_id: str,
    symbol: str,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    conventions,
    target: str,
    mode: str,
) -> dict[str, object] | None:
    """Run the prediction for the chart's forecast overlay.

    Wrapped so a prediction failure degrades the overlay to absent rather than
    failing the whole chart request -- the price history is useful on its own.
    """
    from app.services.analytics import build_analytics
    from app.services.prediction import predict

    try:
        analytics = build_analytics(frame, features, conventions, None).to_dict()
        prediction = predict(
            symbol=symbol, frame=frame, features=features, conventions=conventions,
            analytics=analytics, target_name=target, model_mode=mode,
        ).to_dict()
        return _forecast_projection(frame, prediction, conventions)
    except Exception as exc:
        logger.info("Forecast overlay unavailable for %s: %s", symbol, exc)
        return None
