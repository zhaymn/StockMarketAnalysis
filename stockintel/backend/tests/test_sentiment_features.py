"""Point-in-time sentiment feature tests.

The boundary tests are the important ones. An article published one minute
after the close must not reach that session's feature vector: on a 1-day
horizon, letting it through means the model sees part of the outcome it is
predicting. Both markets are tested because their closes sit at different UTC
offsets and one of them observes daylight saving.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.data.market.registry import get_market_provider
from app.data.news.archive import ArchivedArticle
from app.features.sentiment_features import (
    SENTIMENT_FEATURE_COLUMNS,
    assign_articles_to_sessions,
    build_sentiment_features,
    join_sentiment_features,
    session_close_instants,
)

US = get_market_provider("us").conventions
INDIA = get_market_provider("india").conventions


def make_article(published_at: datetime, score: float = 0.5, *, url: str | None = None):
    """An article with a signed sentiment score of `score`."""
    positive = 0.5 + score / 2
    negative = 0.5 - score / 2
    return ArchivedArticle(
        url=url or f"https://example.com/{published_at.isoformat()}",
        symbol="TEST",
        title="headline",
        description="",
        source="test",
        published_at=published_at,
        sentiment_label="POSITIVE" if score > 0 else "NEGATIVE",
        sentiment_positive=positive,
        sentiment_negative=negative,
        sentiment_neutral=0.0,
    )


@pytest.fixture
def sessions():
    # Weekdays only, matching a trading calendar.
    return pd.DatetimeIndex(pd.bdate_range("2026-03-02", periods=10))


# --- Session close instants ------------------------------------------------

def test_us_close_is_correct_utc_instant(sessions):
    closes = session_close_instants(sessions, US)
    first = closes.iloc[0]

    # US DST begins on the second Sunday in March, which is the 8th in 2026,
    # so 2 March is still EST (UTC-5) and a 16:00 ET close is 21:00 UTC.
    assert first.tzinfo is not None
    assert first.hour == 21
    assert first.date().isoformat() == "2026-03-02"


def test_india_close_is_correct_utc_instant(sessions):
    closes = session_close_instants(sessions, INDIA)
    # 15:30 IST = 10:00 UTC, and India does not observe daylight saving.
    assert closes.iloc[0].hour == 10
    assert closes.iloc[0].minute == 0


def test_us_close_shifts_with_daylight_saving():
    """A hardcoded UTC offset would silently break across the DST boundary."""
    winter = pd.DatetimeIndex([pd.Timestamp("2026-01-15")])
    summer = pd.DatetimeIndex([pd.Timestamp("2026-07-15")])

    winter_close = session_close_instants(winter, US).iloc[0]
    summer_close = session_close_instants(summer, US).iloc[0]

    assert winter_close.hour == 21  # EST: 16:00 - (-5) = 21:00 UTC
    assert summer_close.hour == 20  # EDT: 16:00 - (-4) = 20:00 UTC


# --- The boundary rule -----------------------------------------------------

def test_article_just_before_close_lands_on_that_session(sessions):
    close = session_close_instants(sessions, US).iloc[0]
    article = make_article(close - timedelta(minutes=1))

    assigned = assign_articles_to_sessions([article], sessions, US)
    assert assigned.iloc[0] == sessions[0]


def test_article_just_after_close_lands_on_the_next_session(sessions):
    """The leak this module exists to prevent."""
    close = session_close_instants(sessions, US).iloc[0]
    article = make_article(close + timedelta(minutes=1))

    assigned = assign_articles_to_sessions([article], sessions, US)
    assert assigned.iloc[0] == sessions[1], (
        "An article published after the close must not reach that session's "
        "features — that is a peek at the outcome being predicted."
    )


def test_article_exactly_at_close_is_included(sessions):
    close = session_close_instants(sessions, US).iloc[0]
    assigned = assign_articles_to_sessions([make_article(close)], sessions, US)
    assert assigned.iloc[0] == sessions[0]


def test_weekend_article_is_attributed_to_the_next_session(sessions):
    # 2026-03-07 is a Saturday; the next session is Monday the 9th.
    saturday = datetime(2026, 3, 7, 12, 0, tzinfo=timezone.utc)
    assigned = assign_articles_to_sessions([make_article(saturday)], sessions, US)
    assert assigned.iloc[0] == pd.Timestamp("2026-03-09")


def test_article_after_the_final_session_is_dropped(sessions):
    beyond = session_close_instants(sessions, US).iloc[-1] + timedelta(days=5)
    assigned = assign_articles_to_sessions([make_article(beyond)], sessions, US)
    assert pd.isna(assigned.iloc[0]), "No session could have known this article"


def test_same_instant_assigns_differently_per_market(sessions):
    """A 12:00 UTC article is post-close in Mumbai but pre-close in New York."""
    instant = datetime(2026, 3, 3, 12, 0, tzinfo=timezone.utc)
    article = make_article(instant)

    us_session = assign_articles_to_sessions([article], sessions, US).iloc[0]
    india_session = assign_articles_to_sessions([article], sessions, INDIA).iloc[0]

    assert us_session == pd.Timestamp("2026-03-03")     # before 20:00 UTC close
    assert india_session == pd.Timestamp("2026-03-04")  # after 10:00 UTC close


# --- Feature construction --------------------------------------------------

def test_features_are_causal_under_truncation(sessions):
    """Truncating future sessions must not change earlier feature values."""
    closes = session_close_instants(sessions, US)
    articles = [
        make_article(closes.iloc[i] - timedelta(hours=1), score=0.4 + 0.05 * i, url=f"u{i}")
        for i in range(len(sessions))
    ]

    full, _ = build_sentiment_features(articles, sessions, US)
    truncated, _ = build_sentiment_features(articles, sessions[:6], US)

    for column in ("sentiment_score", "sentiment_score_5d", "news_volume"):
        np.testing.assert_allclose(
            full[column].iloc[:6].to_numpy(),
            truncated[column].to_numpy(),
            atol=1e-12,
            err_msg=f"{column} changed when future sessions were removed",
        )


def test_no_news_is_distinguishable_from_neutral_news(sessions):
    closes = session_close_instants(sessions, US)
    # News on the first session only.
    articles = [make_article(closes.iloc[0] - timedelta(hours=1), score=0.0)]

    features, _ = build_sentiment_features(articles, sessions, US)

    assert features["has_news_coverage"].iloc[0] == 1.0
    assert features["has_news_coverage"].iloc[3] == 0.0
    # Both have a 0.0 score, so the flag is the only thing separating them.
    assert features["sentiment_score"].iloc[3] == 0.0


def test_sessions_since_news_counts_up_and_resets(sessions):
    closes = session_close_instants(sessions, US)
    articles = [
        make_article(closes.iloc[0] - timedelta(hours=1), url="a"),
        make_article(closes.iloc[4] - timedelta(hours=1), url="b"),
    ]
    features, _ = build_sentiment_features(articles, sessions, US)
    since = features["sessions_since_news"].tolist()

    assert since[0] == 0
    assert since[1] == 1
    assert since[3] == 3
    assert since[4] == 0  # reset by the second article


def test_features_contain_no_nans(sessions):
    features, _ = build_sentiment_features([], sessions, US)
    assert not features.isna().to_numpy().any()
    assert set(SENTIMENT_FEATURE_COLUMNS).issubset(features.columns)


def test_unscored_articles_are_excluded(sessions):
    closes = session_close_instants(sessions, US)
    unscored = ArchivedArticle(
        url="https://example.com/x", symbol="TEST", title="t", description="",
        source="s", published_at=closes.iloc[0] - timedelta(hours=1),
    )
    _, coverage = build_sentiment_features([unscored], sessions, US)
    assert coverage.total_articles == 0


def test_coverage_flags_sparse_data_as_unusable(sessions):
    _, coverage = build_sentiment_features([], sessions, US)
    assert not coverage.is_usable
    assert coverage.coverage_fraction == 0.0


# --- Joining ---------------------------------------------------------------

def test_join_preserves_the_price_index_and_fills_gaps(sessions):
    technical = pd.DataFrame({"rsi_14": np.linspace(0, 1, len(sessions))}, index=sessions)
    sentiment, _ = build_sentiment_features([], sessions[:5], US)

    joined = join_sentiment_features(technical, sentiment)

    assert joined.index.equals(sessions)
    assert not joined.isna().to_numpy().any()
    assert (joined["has_news_coverage"].iloc[5:] == 0.0).all()
