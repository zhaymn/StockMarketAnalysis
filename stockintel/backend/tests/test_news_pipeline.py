"""Tests for deduplication, relevance and impact derivation.

The competitor-inversion test encodes the brief's own example: negative text
about a rival is potentially *bullish* for the company being analysed. Getting
this wrong makes every competitor story point the wrong way, so it is pinned
here rather than left to the module docstring.

These use constructed articles, not live news: the logic under test is the
mapping from (sentiment, relationship) to impact, which must hold regardless of
what happens to be in the news today.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.data.fundamentals.profile import CompanyProfile
from app.data.news.base import Article, NewsCategory, TextSentiment
from app.data.news.dedupe import deduplicate, jaccard_similarity
from app.services.event_relevance import (
    ImpactDirection,
    ImpactMagnitude,
    Relationship,
    assess_impact,
    score_articles,
)

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


def make_article(
    title: str,
    *,
    description: str = "",
    sentiment: TextSentiment = TextSentiment.NEUTRAL,
    confidence: float = 0.9,
    category: NewsCategory = NewsCategory.COMPANY,
    tagged: list[str] | None = None,
    hours_ago: float = 1.0,
    source: str = "Reuters",
) -> Article:
    scores = {
        TextSentiment.POSITIVE: {"positive": confidence, "neutral": 0.05, "negative": 0.05},
        TextSentiment.NEGATIVE: {"positive": 0.05, "neutral": 0.05, "negative": confidence},
        TextSentiment.NEUTRAL: {"positive": 0.05, "neutral": confidence, "negative": 0.05},
        TextSentiment.UNAVAILABLE: {},
    }[sentiment]

    return Article(
        title=title,
        url=f"https://example.com/{abs(hash(title))}",
        source=source,
        published_at=NOW - timedelta(hours=hours_ago),
        description=description,
        tagged_symbols=tagged or [],
        category=category,
        text_sentiment=sentiment,
        sentiment_confidence=confidence,
        sentiment_scores=scores,
    )


@pytest.fixture
def nvidia():
    return CompanyProfile(
        symbol="NVDA", name="NVIDIA Corporation",
        sector="Technology", industry="Semiconductors",
    )


@pytest.fixture
def reliance():
    return CompanyProfile(
        symbol="RELIANCE.NS", name="Reliance Industries Limited",
        sector="Energy", industry="Oil & Gas Refining",
    )


# --- The central rule ------------------------------------------------------

def test_negative_competitor_news_is_bullish_not_bearish(nvidia):
    """The brief's own example. Negative text, positive implication."""
    article = make_article(
        "AMD suffers major production failure at key foundry partner",
        description="The setback is expected to delay shipments by several months.",
        sentiment=TextSentiment.NEGATIVE,
    )

    impact = assess_impact(article, nvidia, competitor_names=("AMD", "Intel"), use_llm=False)

    assert impact.relationship is Relationship.COMPETITOR
    # The text is negative...
    assert article.text_sentiment is TextSentiment.NEGATIVE
    # ...but the implication for NVIDIA is not.
    assert impact.direction is ImpactDirection.BULLISH
    assert "rival" in impact.reasoning.lower()


def test_positive_competitor_news_is_bearish(nvidia):
    article = make_article(
        "AMD wins major cloud contract with record-breaking chip order",
        sentiment=TextSentiment.POSITIVE,
    )
    impact = assess_impact(article, nvidia, competitor_names=("AMD",), use_llm=False)

    assert impact.relationship is Relationship.COMPETITOR
    assert impact.direction is ImpactDirection.BEARISH


def test_direct_negative_news_stays_bearish(nvidia):
    """Sentiment must carry through unchanged for the company's own news."""
    article = make_article(
        "NVIDIA Corporation misses quarterly earnings estimates badly",
        sentiment=TextSentiment.NEGATIVE, tagged=["NVDA"],
    )
    impact = assess_impact(article, nvidia, use_llm=False)

    assert impact.relationship is Relationship.DIRECT
    assert impact.direction is ImpactDirection.BEARISH
    assert impact.magnitude is ImpactMagnitude.HIGH  # earnings
    assert impact.event_type == "Earnings result"


def test_impact_language_is_probabilistic(nvidia):
    """The brief forbids stating uncertain causation as fact."""
    article = make_article(
        "NVIDIA Corporation announces record data center revenue",
        sentiment=TextSentiment.POSITIVE, tagged=["NVDA"],
    )
    impact = assess_impact(article, nvidia, use_llm=False)

    hedges = ("may", "could", "potentially", "tends to", "historically")
    assert any(h in impact.reasoning.lower() for h in hedges), impact.reasoning
    # And must never promise an outcome.
    for forbidden in ("will rise", "will fall", "guaranteed", "certain to"):
        assert forbidden not in impact.reasoning.lower()


# --- Relationship and relevance -------------------------------------------

def test_neutral_sentiment_yields_uncertain_not_a_guess(nvidia):
    article = make_article(
        "NVIDIA Corporation confirms date for annual shareholder meeting",
        sentiment=TextSentiment.NEUTRAL, tagged=["NVDA"],
    )
    impact = assess_impact(article, nvidia, use_llm=False)
    assert impact.direction is ImpactDirection.UNCERTAIN


def test_unavailable_sentiment_yields_uncertain(nvidia):
    article = make_article("Some NVIDIA Corporation headline",
                           sentiment=TextSentiment.UNAVAILABLE, tagged=["NVDA"])
    impact = assess_impact(article, nvidia, use_llm=False)
    assert impact.direction is ImpactDirection.UNCERTAIN


def test_macro_news_is_mixed_not_confidently_directional(nvidia):
    article = make_article(
        "Federal Reserve raises interest rates by 50 basis points",
        sentiment=TextSentiment.NEGATIVE, category=NewsCategory.MACRO,
    )
    impact = assess_impact(article, nvidia, use_llm=False)

    assert impact.relationship is Relationship.MACRO
    # Honest: we do not measure this company's rate exposure.
    assert impact.direction is ImpactDirection.MIXED
    assert impact.magnitude is ImpactMagnitude.LOW


def test_sector_news_matches_via_sector_keywords(nvidia):
    article = make_article(
        "New semiconductor export restrictions announced for advanced chip sales",
        sentiment=TextSentiment.NEGATIVE, category=NewsCategory.SECTOR,
    )
    impact = assess_impact(article, nvidia, use_llm=False)

    assert impact.relationship is Relationship.SECTOR
    assert impact.direction is ImpactDirection.BEARISH
    # Sector news is downgraded from HIGH — it moves one name less than its own news.
    assert impact.magnitude is not ImpactMagnitude.HIGH


def test_irrelevant_news_is_excluded(nvidia):
    article = make_article(
        "Local council approves new pedestrian crossing on the high street",
        sentiment=TextSentiment.NEUTRAL,
    )
    impact = assess_impact(article, nvidia, use_llm=False)

    assert impact.relationship is Relationship.UNRELATED
    assert impact.relevance_score == 0.0


def test_indian_company_name_variants_match(reliance):
    """'Reliance Industries Limited' must match a headline saying 'Reliance'."""
    article = make_article(
        "Reliance posts record quarterly profit on refining margins",
        sentiment=TextSentiment.POSITIVE,
    )
    impact = assess_impact(article, reliance, use_llm=False)

    assert impact.relationship is Relationship.DIRECT
    assert impact.direction is ImpactDirection.BULLISH


def test_score_articles_filters_and_ranks(nvidia):
    articles = [
        make_article("Unrelated gardening tips for summer", sentiment=TextSentiment.NEUTRAL),
        make_article("NVIDIA Corporation beats earnings estimates",
                     sentiment=TextSentiment.POSITIVE, tagged=["NVDA"]),
        make_article("Semiconductor demand rises across the industry",
                     sentiment=TextSentiment.POSITIVE, category=NewsCategory.SECTOR),
    ]
    scored = score_articles(articles, nvidia, use_llm=False)

    assert len(scored) == 2  # gardening dropped
    # Direct company news must outrank sector news.
    assert scored[0][1].relationship is Relationship.DIRECT


# --- Deduplication ---------------------------------------------------------

def test_syndicated_copies_collapse_to_one():
    title = "NVIDIA reports record quarterly revenue driven by AI demand"
    articles = [
        make_article(title, source="Reuters", hours_ago=3),
        make_article(title + " ", source="CNBC", hours_ago=2),
        make_article("NVIDIA reports record quarterly revenue driven by AI demand.",
                     source="Bloomberg", hours_ago=1),
    ]
    survivors = deduplicate(articles)

    assert len(survivors) == 1
    # Keeps the earliest (closest to the wire) and records the fold-in count.
    assert survivors[0].source == "Reuters"
    assert survivors[0].duplicate_count == 2


def test_distinct_stories_about_one_company_are_kept():
    articles = [
        make_article("NVIDIA reports record quarterly revenue driven by AI demand"),
        make_article("NVIDIA announces new chief financial officer appointment"),
        make_article("NVIDIA faces antitrust investigation in the European Union"),
    ]
    assert len(deduplicate(articles)) == 3


def test_jaccard_similarity_bounds():
    assert jaccard_similarity(set(), {"a"}) == 0.0
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == pytest.approx(1.0)
    assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0
