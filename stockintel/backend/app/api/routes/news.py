"""News, sentiment and event-impact endpoints.

Every route here degrades honestly: with no `MARKETAUX_API_KEY` the provider
raises `IntegrationNotConfiguredError`, which the app-level handler renders as
a structured 503 the frontend turns into NEWS API NOT CONFIGURED with setup
instructions. No placeholder articles are ever returned.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.logging import get_logger
from app.data.fundamentals.profile import fetch_profile
from app.data.news.base import NewsCategory
from app.data.news.dedupe import deduplicate
from app.data.news.marketaux import MarketauxProvider
from app.services import sentiment as sentiment_service
from app.services.event_relevance import SECTOR_KEYWORDS, score_articles

logger = get_logger(__name__)
router = APIRouter(prefix="/api/news", tags=["news"])

provider = MarketauxProvider()


@router.get("/{symbol}")
async def news_for_symbol(
    symbol: str,
    limit: int = Query(20, ge=1, le=50),
    lookback_days: int = Query(14, ge=1, le=90),
) -> dict[str, object]:
    """Company, sector and macro news for one stock, with impact analysis."""
    profile = fetch_profile(symbol)

    company = provider.fetch_company_news(symbol, limit=limit, lookback_days=lookback_days)

    sector_keywords = list(SECTOR_KEYWORDS.get(profile.sector or "", ()))
    sector = (
        provider.fetch_sector_news(sector_keywords, limit=limit, lookback_days=lookback_days)
        if sector_keywords else []
    )

    macro = provider.fetch_macro_news(limit=limit, lookback_days=min(lookback_days, 7))

    sections: dict[str, object] = {}
    aggregates: dict[str, object] = {}

    for name, articles in (
        ("company", company), ("sector", sector), ("macro", macro)
    ):
        deduped = deduplicate(articles)
        sentiment_service.analyse_articles(deduped)
        scored = score_articles(deduped, profile)

        sections[name] = [
            {**article.to_dict(), "impact": impact.to_dict()} for article, impact in scored
        ]
        aggregates[name] = sentiment_service.aggregate_sentiment(
            [article for article, _ in scored]
        )

    return {
        "symbol": symbol,
        "company": {"name": profile.name, "sector": profile.sector},
        "sentiment_model": {
            "name": "FinBERT (ProsusAI/finbert)",
            "available": sentiment_service.is_available(),
            "limitations": (
                "Trained on financial phrasing. Measured limitations: hedged "
                "comparatives such as 'fell less than feared' are read as negative, "
                "and some routine corporate-action text is misclassified."
            ),
        },
        "sections": sections,
        "aggregate_sentiment": aggregates,
        "note": (
            "Text sentiment describes the language of an article. Expected impact "
            "describes the possible implication for this company's share price — "
            "these can differ, for example when negative news concerns a competitor."
        ),
    }
