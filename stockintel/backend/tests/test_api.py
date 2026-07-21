"""API contract tests.

Exercised in-process via TestClient against live market data. The emphasis is
on the honesty contract: unavailable data must surface as a typed state, an
unconfigured integration must return actionable instructions, and a prediction
must never claim more than the evidence supports.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def analysis_payload():
    response = client.get("/api/analysis/us/AAPL", params={"target": "outlook_5d"})
    if response.status_code != 200:
        pytest.skip(f"Analysis unavailable ({response.status_code}): {response.text[:200]}")
    return response.json()


# --- Meta ------------------------------------------------------------------

def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_integrations_never_leak_key_material():
    payload = client.get("/api/integrations").json()["integrations"]

    for name, status in payload.items():
        assert "configured" in status
        blob = str(status).lower()
        # Availability only -- never the secret itself.
        assert "api_token" not in blob
        assert "secret" not in blob
        if not status["configured"] and status.get("requires_key"):
            # Must tell the user exactly how to fix it.
            assert status["env_var"] and status["obtain_at"]


def test_models_endpoint_marks_a_default_and_carries_a_disclaimer():
    payload = client.get("/api/models").json()
    assert payload["default_mode"] == "most_possible"

    recommended = [m for m in payload["modes"] if m["recommended"]]
    assert len(recommended) == 1
    assert recommended[0]["id"] == "most_possible"
    assert "not" in payload["disclaimer"].lower()

    horizons = {t["horizon_days"] for t in payload["targets"]}
    assert {1, 5, 10, 20}.issubset(horizons)


# --- Markets ---------------------------------------------------------------

def test_markets_expose_distinct_conventions():
    markets = {m["market_id"]: m for m in client.get("/api/markets").json()["markets"]}
    assert set(markets) == {"us", "india"}

    us = markets["us"]["conventions"]
    india = markets["india"]["conventions"]
    assert (us["currency_code"], india["currency_code"]) == ("USD", "INR")
    assert us["benchmark_label"] == "S&P 500"
    assert india["benchmark_label"] == "NIFTY 50"


def test_unknown_market_returns_structured_404():
    response = client.get("/api/markets/atlantis/directory")
    assert response.status_code == 404
    assert response.json()["code"] == "unknown_ticker"


def test_search_requires_a_query():
    assert client.get("/api/markets/us/search").status_code == 422


# --- Analysis --------------------------------------------------------------

def test_analysis_returns_the_full_dashboard_payload(analysis_payload):
    for key in ("symbol", "market", "session", "data", "prediction", "analytics"):
        assert key in analysis_payload

    analytics = analysis_payload["analytics"]
    for section in ("price", "returns", "volatility", "momentum", "volume", "risk", "benchmark"):
        assert section in analytics


def test_freshness_is_never_overstated_as_realtime(analysis_payload):
    # The development feed is delayed; claiming REAL_TIME would be a fabrication.
    assert analysis_payload["data"]["freshness"] in {"DELAYED", "END_OF_DAY", "CACHED"}


def test_currency_matches_the_market(analysis_payload):
    assert analysis_payload["market"]["conventions"]["currency_code"] == "USD"


def test_prediction_withholds_probability_unless_calibrated(analysis_payload):
    prediction = analysis_payload["prediction"]

    assert prediction["verdict"] in {
        "DIRECTIONAL", "NO_EDGE", "ABSTAINED", "INSUFFICIENT_DATA"
    }
    if not prediction["probability_is_calibrated"]:
        assert prediction["probability"] is None
        assert prediction["probability_withheld_reason"]


def test_prediction_always_reports_its_baseline(analysis_payload):
    """Accuracy without its baseline is meaningless — it must always ship both."""
    evidence = analysis_payload["prediction"]["evidence"]
    assert "walk_forward" in evidence and "baseline" in evidence
    assert "baseline_accuracy_mean" in evidence["walk_forward"]
    # Overlapping labels make nominal sample size misleading.
    assert evidence["effective_sample_size"] <= evidence["n_samples"]


def test_no_edge_verdict_makes_no_directional_claim(analysis_payload):
    prediction = analysis_payload["prediction"]
    if prediction["verdict"] in ("NO_EDGE", "ABSTAINED", "INSUFFICIENT_DATA"):
        assert prediction["direction"] is None
        assert prediction["probability"] is None
        assert prediction["interpretation"]


def test_explanation_factors_are_evidence_backed(analysis_payload):
    factors = analysis_payload["prediction"]["factors"]
    for bucket in ("bullish", "bearish", "risks"):
        for item in factors.get(bucket, []):
            # Every claimed factor must cite the number behind it.
            assert item["factor"] and item["evidence"]


def test_invalid_ticker_returns_structured_error():
    response = client.get("/api/analysis/us/NOTAREALTICKER999")
    assert response.status_code in (404, 422)
    assert response.json()["code"] in {"unknown_ticker", "data_unavailable", "insufficient_history"}


def test_indian_market_uses_inr_and_nifty():
    response = client.get("/api/analysis/india/RELIANCE.NS")
    if response.status_code != 200:
        pytest.skip("Indian market data unavailable")

    payload = response.json()
    assert payload["market"]["conventions"]["currency_code"] == "INR"
    assert payload["analytics"]["benchmark"]["benchmark_label"] == "NIFTY 50"


# --- Charts ----------------------------------------------------------------

def test_chart_series_are_aligned_to_the_date_axis():
    response = client.get("/api/analysis/us/AAPL/chart", params={"range": "1y"})
    if response.status_code != 200:
        pytest.skip("Chart data unavailable")

    payload = response.json()
    n = len(payload["dates"])
    assert n > 100

    for key in ("open", "high", "low", "close"):
        assert len(payload["ohlc"][key]) == n
    assert len(payload["volume"]) == n
    for series in payload["moving_averages"].values():
        assert len(series) == n


def test_chart_rejects_an_unknown_range():
    assert client.get("/api/analysis/us/AAPL/chart", params={"range": "42y"}).status_code == 422


# --- News ------------------------------------------------------------------

def test_news_reports_not_configured_rather_than_fabricating():
    from app.core.config import get_settings

    response = client.get("/api/news/AAPL")

    if not get_settings().has_news_provider:
        assert response.status_code == 503
        payload = response.json()
        assert payload["code"] == "not_configured"
        # Must be actionable, not a bare failure.
        assert payload["env_var"] == "MARKETAUX_API_KEY"
        assert "marketaux.com" in payload["obtain_at"]
    elif response.status_code == 429:
        # Configured, but the plan allowance is spent. A real third state on a
        # quota-limited tier, and it must read as rate limiting rather than as
        # a provider fault the user can do nothing about.
        payload = response.json()
        assert payload["code"] == "rate_limited"
        assert "allowance" in payload["message"].lower()
    else:
        assert response.status_code == 200
        payload = response.json()
        assert "sections" in payload and "aggregate_sentiment" in payload
