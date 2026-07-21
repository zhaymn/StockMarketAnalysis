"""FRED macro-data tests.

Two of these pin bugs that shipped and were caught during development, both of
which are exactly the failure mode this project exists to avoid:

* Staleness thresholds set generously enough that a 507-day-old reading
  reported as fresh — tuning the measurement to flatter the data.
* `is_stale` frozen at fetch time and cached, so a series could cross the
  staleness boundary while cached and keep reporting fresh.

The parsing and staleness logic is pure, so it is tested against fixtures with
`date.today()` frozen — no network, and no dependence on how old the real data
happens to be on the day the suite runs.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import Mock, patch

import pytest

from app.core.errors import IntegrationNotConfiguredError
from app.data.macro import fred
from app.data.macro.fred import (
    SeriesSpec,
    _build_observation,
    _refresh_staleness,
    _to_float,
    fetch_macro_snapshot,
)


def obs(date_str: str, value: str) -> dict:
    """A FRED observation as the API returns it."""
    return {"date": date_str, "value": value}


MONTHLY = SeriesSpec("TEST_M", "Test monthly", "%", "desc", stale_after_months=3)
YOY = SeriesSpec(
    "TEST_CPI", "Test CPI", "% y/y", "desc", stale_after_months=3, as_yoy_change=True
)


# --- Value parsing ---------------------------------------------------------

def test_missing_observation_marker_parses_to_none():
    # FRED encodes a missing value as a literal ".".
    assert _to_float(".") is None
    assert _to_float("") is None
    assert _to_float(None) is None
    assert _to_float("not a number") is None


def test_valid_values_parse():
    assert _to_float("3.63") == 3.63
    assert _to_float("0") == 0.0
    assert _to_float("-0.5") == -0.5


# --- Observation construction ---------------------------------------------

def test_latest_value_and_change_are_taken_from_newest_first():
    # FRED returns newest-first; change is latest minus the previous point.
    raw = [obs("2026-07-20", "4.55"), obs("2026-07-19", "4.57"), obs("2026-07-18", "4.50")]
    with _frozen_today(date(2026, 7, 21)):
        result = _build_observation(MONTHLY, raw)

    assert result.value == 4.55
    assert result.observation_date == "2026-07-20"
    assert result.change == pytest.approx(4.55 - 4.57)


def test_missing_values_are_skipped_when_choosing_the_latest():
    raw = [obs("2026-07-20", "."), obs("2026-07-19", "4.57")]
    with _frozen_today(date(2026, 7, 21)):
        result = _build_observation(MONTHLY, raw)

    assert result.value == 4.57
    assert result.observation_date == "2026-07-19"


def test_empty_series_is_unavailable_not_zero():
    result = _build_observation(MONTHLY, [])
    assert not result.is_available
    assert result.value is None
    assert result.unavailable_reason


def test_all_missing_values_is_unavailable():
    raw = [obs("2026-07-20", "."), obs("2026-07-19", ".")]
    result = _build_observation(MONTHLY, raw)
    assert not result.is_available


# --- Year-over-year change -------------------------------------------------

def test_yoy_reports_percentage_change_not_the_raw_index():
    # A CPI index of 330 vs 320 a year earlier is +3.125%, not "330".
    raw = [obs("2026-06-01", "330.0")] + [
        obs(f"2025-{m:02d}-01", "320.0") for m in range(1, 13)
    ]
    with _frozen_today(date(2026, 7, 1)):
        result = _build_observation(YOY, raw)

    assert result.value == pytest.approx((330.0 / 320.0 - 1.0) * 100.0)
    assert result.value == pytest.approx(3.125)


def test_yoy_without_a_year_of_history_is_unavailable():
    # Only three months present — cannot compute a year-over-year change.
    raw = [obs("2026-06-01", "330.0"), obs("2026-05-01", "329.0"), obs("2026-04-01", "328.0")]
    result = _build_observation(YOY, raw)
    assert not result.is_available
    assert "year-over-year" in result.unavailable_reason.lower()


# --- Staleness: the bug this pins ------------------------------------------

def test_staleness_is_driven_by_the_threshold_not_by_convenience():
    """A monthly series far past its threshold must read stale.

    The bug: a threshold generous enough that a 17-month-old CPI reading
    reported as fresh. Here a 3-month-threshold series at ~16 months old must
    be stale, and one inside the threshold must not be.
    """
    fresh = _build_observation(MONTHLY, [obs("2026-06-01", "4.0")])
    stale = _build_observation(MONTHLY, [obs("2025-03-01", "4.0")])

    with _frozen_today(date(2026, 7, 21)):
        fresh = _refresh_staleness(fresh, (MONTHLY,))
        stale = _refresh_staleness(stale, (MONTHLY,))

    assert not fresh.is_stale, "A 50-day-old monthly reading is not stale"
    assert stale.is_stale, "A 16-month-old reading against a 3-month threshold is stale"


def test_threshold_boundary():
    spec = SeriesSpec("B", "boundary", "%", "d", stale_after_months=3)
    # 3 months * 31 = 93 days is the cutoff.
    just_inside = _build_observation(spec, [obs("2026-04-20", "1.0")])   # 92 days
    just_outside = _build_observation(spec, [obs("2026-04-18", "1.0")])  # 94 days

    with _frozen_today(date(2026, 7, 21)):
        just_inside = _refresh_staleness(just_inside, (spec,))
        just_outside = _refresh_staleness(just_outside, (spec,))

    assert not just_inside.is_stale
    assert just_outside.is_stale


def test_refresh_recomputes_age_against_the_current_date():
    """The caching bug: age and staleness must be recomputed on read.

    An observation cached weeks ago with a small age must, when refreshed, show
    the age as of *now* — not the age frozen at fetch time.
    """
    observation = _build_observation(MONTHLY, [obs("2026-05-01", "4.0")])

    # Simulate a value cached when it was young.
    observation.age_days = 5
    observation.is_stale = False

    with _frozen_today(date(2026, 9, 1)):  # ~4 months later
        refreshed = _refresh_staleness(observation, (MONTHLY,))

    assert refreshed.age_days > 100, "age must reflect today, not the fetch time"
    assert refreshed.is_stale, "a 4-month-old monthly reading is stale"


def test_refresh_is_a_noop_for_unavailable_observations():
    spec = MONTHLY
    unavailable = _build_observation(spec, [])
    refreshed = _refresh_staleness(unavailable, (spec,))
    assert not refreshed.is_available
    assert refreshed.age_days is None


# --- Snapshot behaviour ----------------------------------------------------

def test_unconfigured_key_raises_not_configured():
    settings = Mock()
    settings.has_macro_provider = False

    with patch.object(fred, "get_settings", return_value=settings):
        with pytest.raises(IntegrationNotConfiguredError) as excinfo:
            fetch_macro_snapshot("us")

    assert excinfo.value.env_var == "FRED_API_KEY"


def test_a_single_failing_series_does_not_sink_the_snapshot():
    """One bad series degrades to unavailable; the rest still return."""
    settings = Mock()
    settings.has_macro_provider = True
    settings.fred_api_key = "KEY"

    # 14 months of history, so the year-over-year series can compute their
    # change. Without this, the two as_yoy_change series would fail for lack of
    # history rather than because of the injected outage, muddying the test.
    def full_history() -> list[dict]:
        points = []
        for months_ago in range(14):
            month = 6 - months_ago
            year = 2026
            while month <= 0:
                month += 12
                year -= 1
            points.append(obs(f"{year}-{month:02d}-01", str(3.5 - months_ago * 0.02)))
        return points

    failing_series = fred.US_SERIES[1].series_id  # DGS10, a non-YoY series

    def flaky_fetch(spec, api_key):
        if spec.series_id == failing_series:
            raise RuntimeError("simulated FRED outage")
        return full_history()

    with patch.object(fred, "get_settings", return_value=settings), \
         patch.object(fred, "_fetch_series", side_effect=flaky_fetch), \
         patch.object(fred, "get_cache") as cache:
        cache.return_value.get_json.return_value = None

        snapshot = fetch_macro_snapshot("us", use_cache=False)

    available = [o for o in snapshot.observations if o.is_available]
    unavailable = [o for o in snapshot.observations if not o.is_available]

    # Exactly the injected series failed; every other series returned.
    assert len(available) == len(fred.US_SERIES) - 1
    assert len(unavailable) == 1
    assert unavailable[0].series_id == failing_series
    assert unavailable[0].unavailable_reason


def test_provider_error_never_leaks_the_api_key():
    """FRED URLs carry api_key=...; the error must not echo it."""
    response = Mock()
    response.ok = False
    response.status_code = 400
    response.text = "https://api.stlouisfed.org/fred/...?api_key=SECRETKEY123"

    with patch.object(fred.requests, "get", return_value=response):
        with pytest.raises(Exception) as excinfo:
            fred._fetch_series(MONTHLY, "SECRETKEY123")

    assert "SECRETKEY123" not in str(excinfo.value)
    assert "api_key" not in str(excinfo.value)


def test_india_snapshot_carries_a_coverage_note():
    snapshot = fred.MacroSnapshot(market_id="india")
    assert snapshot.to_dict()["coverage_note"] is not None
    us = fred.MacroSnapshot(market_id="us")
    assert us.to_dict()["coverage_note"] is None


# --- helpers ---------------------------------------------------------------

class _frozen_today:
    """Freeze `fred.date.today()` without touching real datetime construction."""

    def __init__(self, today: date) -> None:
        self._today = today
        self._patch = None

    def __enter__(self):
        real_date = date

        class FrozenDate(real_date):
            @classmethod
            def today(cls):
                return self._today

        self._patch = patch.object(fred, "date", FrozenDate)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
