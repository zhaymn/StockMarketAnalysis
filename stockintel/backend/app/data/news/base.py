"""News provider abstraction.

Keeps the rest of the application independent of any single news vendor. The
`Article` shape is what every downstream stage consumes -- deduplication,
FinBERT sentiment, relevance scoring, impact analysis -- so swapping Marketaux
for Finnhub or Benzinga means writing one new provider class.

The critical field-level decision is that `text_sentiment` and `expected_impact`
are **separate**, and impact is deliberately absent from this module. A provider
reports what was published; inferring what it means for a specific company's
share price is the relevance engine's job. Conflating them is the classic error
that makes "competitor's factory burns down" read as bearish.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class NewsCategory(str, Enum):
    """Which section of the dashboard an article belongs to."""

    COMPANY = "COMPANY"
    SECTOR = "SECTOR"
    MACRO = "MACRO"


class TextSentiment(str, Enum):
    """FinBERT's reading of the *language*, not of price implications."""

    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class Article:
    """One news item, plus whatever analysis stages have been applied."""

    title: str
    url: str
    source: str
    published_at: datetime

    description: str = ""
    #: Ticker symbols the provider associated with this article.
    tagged_symbols: list[str] = field(default_factory=list)
    category: NewsCategory = NewsCategory.COMPANY

    # --- Populated by the sentiment stage ---------------------------------
    text_sentiment: TextSentiment = TextSentiment.UNAVAILABLE
    sentiment_confidence: float | None = None
    sentiment_scores: dict[str, float] = field(default_factory=dict)

    # --- Populated by the relevance stage ---------------------------------
    relevance_score: float | None = None
    relevance_reasons: list[str] = field(default_factory=list)

    #: Set when deduplication folds syndicated copies into this article.
    duplicate_count: int = 0

    @property
    def content_hash(self) -> str:
        """Stable hash of the analysed text.

        Keys the sentiment cache: identical text must never be re-scored, and
        the same story from two outlets shares a hash only if the text matches.
        """
        payload = f"{self.title.strip().lower()}|{self.description.strip().lower()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    @property
    def analysed_text(self) -> str:
        """Text handed to FinBERT.

        Title plus description: FinBERT was trained on financial phrasing at
        roughly headline-to-paragraph length, and full article bodies dilute
        the signal with boilerplate that shifts scores toward neutral.
        """
        if self.description:
            return f"{self.title}. {self.description}"
        return self.title

    def age_hours(self, *, now: datetime | None = None) -> float:
        reference = now or datetime.now(self.published_at.tzinfo)
        return max(0.0, (reference - self.published_at).total_seconds() / 3600.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "description": self.description,
            "category": self.category.value,
            "tagged_symbols": self.tagged_symbols,
            "text_sentiment": self.text_sentiment.value,
            "sentiment_confidence": (
                round(self.sentiment_confidence, 4)
                if self.sentiment_confidence is not None else None
            ),
            "sentiment_scores": {k: round(v, 4) for k, v in self.sentiment_scores.items()},
            "relevance_score": (
                round(self.relevance_score, 4) if self.relevance_score is not None else None
            ),
            "relevance_reasons": self.relevance_reasons,
            "duplicate_count": self.duplicate_count,
        }


class NewsProvider(ABC):
    """A source of financial news."""

    provider_name: str
    requires_api_key: bool = True

    @abstractmethod
    def is_configured(self) -> bool:
        """Whether this provider can actually make requests."""

    @abstractmethod
    def fetch_company_news(
        self, symbol: str, *, limit: int = 25, lookback_days: int = 14
    ) -> list[Article]:
        """News about one company."""

    @abstractmethod
    def fetch_sector_news(
        self, keywords: list[str], *, limit: int = 25, lookback_days: int = 14
    ) -> list[Article]:
        """News about a sector or theme."""

    @abstractmethod
    def fetch_macro_news(self, *, limit: int = 25, lookback_days: int = 7) -> list[Article]:
        """Macroeconomic and policy news."""
