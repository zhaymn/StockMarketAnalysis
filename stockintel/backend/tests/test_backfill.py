"""Backfill client tests.

The secret-redaction test exists because the first live run leaked the API key
into the logs: `requests`' HTTPError message embeds the full request URL, and
the URL carries `api_token=<secret>`. Log files get shared, shipped and pasted
into issues, so an exception path that prints a credential is a real disclosure
route, not a cosmetic problem.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import Mock, patch

import pytest

from app.core.errors import IntegrationNotConfiguredError, RateLimitedError
from app.data.news import backfill

FAKE_KEY = "SECRETKEY1234567890abcdefghij"


def make_response(status_code: int, payload: dict | None = None) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.ok = 200 <= status_code < 300
    response.json.return_value = payload or {}
    response.text = (
        f"https://api.marketaux.com/v1/news/all?symbols=AAPL&api_token={FAKE_KEY}"
    )
    return response


# --- Secret handling -------------------------------------------------------

def test_http_error_message_never_contains_the_api_key():
    """An error path must not put the credential into an exception message."""
    with patch.object(backfill.requests, "get", return_value=make_response(500)):
        with pytest.raises(RuntimeError) as excinfo:
            backfill._fetch_page("AAPL", None, date(2026, 3, 2), 1, FAKE_KEY)

    assert FAKE_KEY not in str(excinfo.value)
    assert "api_token" not in str(excinfo.value)
    # Still useful for debugging.
    assert "500" in str(excinfo.value)
    assert "2026-03-02" in str(excinfo.value)


def test_backfill_errors_do_not_leak_the_key_into_progress(caplog):
    settings = Mock()
    settings.has_news_provider = True
    settings.marketaux_api_key = FAKE_KEY

    with patch.object(backfill, "get_settings", return_value=settings), \
         patch.object(backfill.requests, "get", return_value=make_response(500)), \
         patch.object(backfill, "get_archive") as archive:
        archive.return_value.completed_days.return_value = set()
        archive.return_value.add_articles.return_value = 0

        progress = backfill.backfill_symbol(
            "AAPL", date(2026, 3, 2), date(2026, 3, 3), max_requests=4
        )

    blob = str(progress.to_dict()) + caplog.text
    assert FAKE_KEY not in blob


# --- Quota handling --------------------------------------------------------

@pytest.mark.parametrize("status", [402, 429])
def test_quota_exhaustion_raises_rate_limited(status):
    """Marketaux uses 402 for an exhausted allowance, not only 429."""
    with patch.object(backfill.requests, "get", return_value=make_response(status)):
        with pytest.raises(RateLimitedError):
            backfill._fetch_page("AAPL", None, date(2026, 3, 2), 1, FAKE_KEY)


def test_quota_exhaustion_stops_cleanly_with_progress_preserved():
    settings = Mock()
    settings.has_news_provider = True
    settings.marketaux_api_key = FAKE_KEY

    with patch.object(backfill, "get_settings", return_value=settings), \
         patch.object(backfill.requests, "get", return_value=make_response(402)), \
         patch.object(backfill, "get_archive") as archive:
        archive.return_value.completed_days.return_value = set()
        archive.return_value.add_articles.return_value = 0

        progress = backfill.backfill_symbol(
            "AAPL", date(2026, 3, 2), date(2026, 3, 6), max_requests=10
        )

    assert progress.stopped_early is not None
    assert "resume" in progress.stopped_early.lower()
    # A quota-truncated run must not record days as complete.
    archive.return_value.mark_day_complete.assert_not_called()


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_key_raises_not_configured(status):
    with patch.object(backfill.requests, "get", return_value=make_response(status)):
        with pytest.raises(IntegrationNotConfiguredError):
            backfill._fetch_page("AAPL", None, date(2026, 3, 2), 1, FAKE_KEY)


# --- Resumability ----------------------------------------------------------

def test_completed_days_are_skipped_on_a_rerun():
    settings = Mock()
    settings.has_news_provider = True
    settings.marketaux_api_key = FAKE_KEY

    payload = {"data": [], "meta": {"returned": 0}}

    with patch.object(backfill, "get_settings", return_value=settings), \
         patch.object(backfill.requests, "get", return_value=make_response(200, payload)) as get, \
         patch.object(backfill, "get_archive") as archive:
        archive.return_value.completed_days.return_value = {"2026-03-02", "2026-03-03"}
        archive.return_value.add_articles.return_value = 0

        progress = backfill.backfill_symbol(
            "AAPL", date(2026, 3, 2), date(2026, 3, 4), max_requests=10
        )

    assert progress.days_skipped == 2
    # Only the one uncovered weekday should have cost a request.
    assert get.call_count == 1


def test_request_budget_is_respected():
    settings = Mock()
    settings.has_news_provider = True
    settings.marketaux_api_key = FAKE_KEY

    payload = {"data": [], "meta": {"returned": 0}}

    with patch.object(backfill, "get_settings", return_value=settings), \
         patch.object(backfill.requests, "get", return_value=make_response(200, payload)) as get, \
         patch.object(backfill, "get_archive") as archive:
        archive.return_value.completed_days.return_value = set()
        archive.return_value.add_articles.return_value = 0

        progress = backfill.backfill_symbol(
            "AAPL", date(2026, 3, 2), date(2026, 3, 20), max_requests=3
        )

    assert progress.requests_made <= 3
    assert get.call_count <= 3
    assert progress.stopped_early is not None


def test_weekends_are_skipped_by_default():
    settings = Mock()
    settings.has_news_provider = True
    settings.marketaux_api_key = FAKE_KEY
    payload = {"data": [], "meta": {"returned": 0}}

    with patch.object(backfill, "get_settings", return_value=settings), \
         patch.object(backfill.requests, "get", return_value=make_response(200, payload)) as get, \
         patch.object(backfill, "get_archive") as archive:
        archive.return_value.completed_days.return_value = set()
        archive.return_value.add_articles.return_value = 0

        # 7-8 March 2026 is a weekend.
        backfill.backfill_symbol("AAPL", date(2026, 3, 6), date(2026, 3, 9), max_requests=20)

    assert get.call_count == 2  # Friday and Monday only


# --- Symbol mapping --------------------------------------------------------

def test_indian_symbols_map_to_bare_ticker_and_country():
    assert backfill._api_symbol("RELIANCE.NS") == ("RELIANCE", "in")
    assert backfill._api_symbol("TCS.BO") == ("TCS", "in")
    assert backfill._api_symbol("AAPL") == ("AAPL", None)
