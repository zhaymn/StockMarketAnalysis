"""End-to-end checks of the data and feature layers against live market data.

These are integration tests by design: the failure modes that matter here
(a provider changing its response shape, a pandas upgrade altering rolling
semantics, an indicator quietly reading the future) are invisible to tests run
against a synthetic fixture. They skip rather than fail when the network is
unavailable, so an offline run stays green.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.errors import UnknownTickerError
from app.data.market.calendar import MarketStatus, get_session_info
from app.data.market.prices import fetch_history
from app.data.market.registry import get_market_provider, list_markets
from app.features.leakage import probe_lookahead
from app.features.targets import (
    DIRECTION_1D,
    OUTLOOK_5D,
    align_features_and_target,
    build_target,
    class_balance,
)
from app.features.technical import build_features

US_SYMBOL = "AAPL"
INDIA_SYMBOL = "RELIANCE.NS"


@pytest.fixture(scope="module")
def us_history():
    try:
        return fetch_history(US_SYMBOL, period="5y")
    except Exception as exc:  # network/provider unavailable
        pytest.skip(f"Market data unavailable: {exc}")


@pytest.fixture(scope="module")
def india_history():
    try:
        return fetch_history(INDIA_SYMBOL, period="5y")
    except Exception as exc:
        pytest.skip(f"Market data unavailable: {exc}")


# --- Market registry -------------------------------------------------------

def test_markets_have_distinct_conventions():
    markets = {m["market_id"]: m for m in list_markets()}
    assert set(markets) == {"us", "india"}

    us = markets["us"]["conventions"]
    india = markets["india"]["conventions"]

    # The whole point of MarketConventions: currencies and benchmarks must
    # never be shared between markets.
    assert us["currency_code"] == "USD" and india["currency_code"] == "INR"
    assert us["benchmark_symbol"] != india["benchmark_symbol"]
    assert us["timezone"] != india["timezone"]


def test_unknown_market_raises():
    with pytest.raises(UnknownTickerError):
        get_market_provider("atlantis")


def test_invalid_ticker_raises_rather_than_returning_empty():
    # Must fail loudly: an empty frame flowing onward would train a model on
    # nothing and report metrics for it.
    with pytest.raises((UnknownTickerError, Exception)):
        fetch_history("NOTAREALTICKER123XYZ", period="1y", use_cache=False)


# --- Price data ------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", ["us_history", "india_history"])
def test_history_is_clean_and_ordered(fixture_name, request):
    history = request.getfixturevalue(fixture_name)
    frame = history.frame

    assert not frame.empty
    assert frame.index.is_monotonic_increasing
    assert not frame.index.has_duplicates
    assert frame["Close"].notna().all(), "Nulls must be dropped, never forward-filled"
    assert (frame["Close"] > 0).all()
    assert (frame["High"] >= frame["Low"]).all()


def test_session_info_infers_holiday_from_data(us_history):
    provider = get_market_provider("us")
    info = get_session_info(
        provider.conventions, last_session_date=us_history.last_session_date
    )
    assert isinstance(info.status, MarketStatus)
    assert info.to_dict()["timezone"] == "America/New_York"


# --- Features --------------------------------------------------------------

def test_features_have_no_lookahead(us_history):
    """The single most important test in this codebase.

    If this fails, every accuracy number the platform reports is inflated.
    """
    report = probe_lookahead(
        us_history.frame,
        lambda f: build_features(f, drop_warmup=False)[0],
        probe_points=4,
    )
    assert report.passed, report.summary()


def test_features_are_finite_and_scale_free(us_history):
    features, groups = build_features(us_history.frame)

    assert not features.empty
    assert len(groups) >= 4

    # No infinities should survive the guards in build_features.
    assert not np.isinf(features.to_numpy(dtype=float)).any()

    # Every feature should be a normalised derivative, not a raw price level.
    # A column whose median magnitude exceeds ~100 is almost certainly a raw
    # price or volume that leaked into the matrix.
    medians = features.abs().median()
    offenders = medians[medians > 100].index.tolist()
    assert not offenders, f"Non-scale-free features leaked in: {offenders}"


def test_rsi_stays_in_range(us_history):
    features, _ = build_features(us_history.frame)
    rsi = features["rsi_14"].dropna()  # normalised to [-1, 1]
    assert rsi.between(-1.0, 1.0).all()


# --- Targets ---------------------------------------------------------------

def test_direction_target_is_balanced_enough_to_learn(us_history):
    target, band = build_target(us_history.frame["Close"], DIRECTION_1D)
    assert band is None

    balance = class_balance(target, DIRECTION_1D)
    assert abs(balance["UP"] - 0.5) < 0.15, f"Implausible class balance: {balance}"


def test_outlook_target_produces_three_usable_classes(india_history):
    target, band = build_target(india_history.frame["Close"], OUTLOOK_5D)
    assert band is not None

    balance = class_balance(target, OUTLOOK_5D)
    # The volatility-scaled band exists precisely so NEUTRAL is a real class.
    for label in ("BEARISH", "NEUTRAL", "BULLISH"):
        assert balance[label] > 0.10, f"Degenerate class {label}: {balance}"


def test_target_tail_is_unresolved(us_history):
    """The last `horizon` rows have no outcome yet and must be NaN."""
    target, _ = build_target(us_history.frame["Close"], OUTLOOK_5D)
    assert target.iloc[-OUTLOOK_5D.horizon_days:].isna().all()


def test_alignment_keeps_x_and_y_index_identical(us_history):
    features, _ = build_features(us_history.frame)
    target, _ = build_target(us_history.frame["Close"], OUTLOOK_5D)

    x, y = align_features_and_target(features, target)
    assert len(x) == len(y) > 100
    assert x.index.equals(y.index)
    assert not x.isna().to_numpy().any()
    assert not y.isna().any()
