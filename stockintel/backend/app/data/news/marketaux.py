"""Marketaux news provider.

Chosen over the US-centric alternatives because it tags articles with ticker
symbols and indexes NSE/BSE listings, which the India market requires. The
free tier allows 100 requests/day, so caching here is not an optimisation --
without it a handful of page loads exhausts the daily quota.

If no key is configured, every method raises `IntegrationNotConfiguredError`
carrying the instructions the UI shows. It never returns fabricated articles.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import requests

from app.core.config import get_settings
from app.core.errors import (
    IntegrationNotConfiguredError,
    ProviderError,
    RateLimitedError,
)
from app.core.logging import get_logger
from app.data.cache.store import cache_key, get_cache
from app.data.news.base import Article, NewsCategory, NewsProvider

logger = get_logger(__name__)

API_BASE = "https://api.marketaux.com/v1/news/all"
REQUEST_TIMEOUT_SECONDS = 12

#: Exchange suffix -> Marketaux country code, so an NSE ticker searches Indian
#: coverage rather than defaulting to US wires.
SUFFIX_TO_COUNTRY = {".NS": "in", ".BO": "in"}


class MarketauxProvider(NewsProvider):
    """Financial news from Marketaux."""

    provider_name = "Marketaux"
    requires_api_key = True

    def is_configured(self) -> bool:
        return get_settings().has_news_provider

    def _require_key(self) -> str:
        settings = get_settings()
        if not settings.has_news_provider:
            raise IntegrationNotConfiguredError(
                "News provider",
                env_var="MARKETAUX_API_KEY",
                obtain_at="https://www.marketaux.com/",
                reason=(
                    "Required to fetch company, sector and macro news, and therefore "
                    "also news sentiment and event-impact analysis."
                ),
            )
        return settings.marketaux_api_key

    # -- requests -----------------------------------------------------------
    def _get(self, params: dict[str, object], *, cache_namespace: str) -> dict:
        """Cached GET against the Marketaux API."""
        api_key = self._require_key()
        settings = get_settings()
        cache = get_cache()

        # The key deliberately excludes the API token, so rotating a key does
        # not invalidate the whole cache.
        key = cache_key(*[f"{k}={v}" for k, v in sorted(params.items())])

        hit = cache.get_json(cache_namespace, key)
        if hit is not None:
            logger.debug("News cache hit: %s", key)
            return hit.value

        try:
            response = requests.get(
                API_BASE,
                params={**params, "api_token": api_key},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise ProviderError("Could not reach the news provider.", detail=str(exc)) from exc

        # Marketaux reports an exhausted plan allowance as 402, not 429.
        # Without the 402 branch this surfaced as a generic provider error and
        # the UI showed "502 Bad Gateway" to a user who had simply used up
        # their daily quota.
        if response.status_code in (402, 429):
            raise RateLimitedError(
                "News provider allowance exhausted "
                f"(HTTP {response.status_code}). The Marketaux free tier allows "
                "100 requests/day; it resets daily.",
                retry_after=3600,
            )
        if response.status_code in (401, 403):
            raise IntegrationNotConfiguredError(
                "News provider",
                env_var="MARKETAUX_API_KEY",
                obtain_at="https://www.marketaux.com/",
                reason=(
                    f"The configured key was rejected (HTTP {response.status_code}). "
                    "It may be invalid, expired, or over quota."
                ),
            )
        if not response.ok:
            raise ProviderError(
                f"News provider returned HTTP {response.status_code}.",
                detail=response.text[:300],
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError("News provider returned malformed JSON.") from exc

        cache.set_json(
            cache_namespace, key, payload,
            ttl_seconds=settings.cache_ttl_news, fetched_at=time.time(),
        )
        return payload

    # -- parsing ------------------------------------------------------------
    @staticmethod
    def _parse_articles(payload: dict, category: NewsCategory) -> list[Article]:
        articles: list[Article] = []

        for item in payload.get("data", []):
            title = (item.get("title") or "").strip()
            url = item.get("url") or ""
            if not title or not url:
                continue

            published_raw = item.get("published_at") or ""
            try:
                published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            except ValueError:
                logger.debug("Unparseable publish date %r; skipping article.", published_raw)
                continue

            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)

            tagged = [
                entity.get("symbol")
                for entity in item.get("entities", [])
                if entity.get("symbol")
            ]

            articles.append(Article(
                title=title,
                url=url,
                source=item.get("source") or "unknown",
                published_at=published_at,
                description=(item.get("description") or item.get("snippet") or "").strip(),
                tagged_symbols=tagged,
                category=category,
            ))

        return articles

    @staticmethod
    def _since(lookback_days: int) -> str:
        return (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

    # -- public API ---------------------------------------------------------
    def fetch_company_news(
        self, symbol: str, *, limit: int = 25, lookback_days: int = 14
    ) -> list[Article]:
        """News tagged to a specific ticker.

        Marketaux indexes Indian listings under the bare NSE symbol rather than
        the yfinance '.NS' form, so the suffix is stripped and the country
        filter set instead.
        """
        params: dict[str, object] = {
            "symbols": symbol,
            "filter_entities": "true",
            "language": "en",
            "limit": min(limit, 50),
            "published_after": self._since(lookback_days),
        }

        for suffix, country in SUFFIX_TO_COUNTRY.items():
            if symbol.upper().endswith(suffix):
                params["symbols"] = symbol[: -len(suffix)]
                params["countries"] = country
                break

        payload = self._get(params, cache_namespace="news_company")
        return self._parse_articles(payload, NewsCategory.COMPANY)

    def fetch_sector_news(
        self, keywords: list[str], *, limit: int = 25, lookback_days: int = 14
    ) -> list[Article]:
        if not keywords:
            return []

        payload = self._get(
            {
                # Marketaux search accepts a boolean OR expression.
                "search": " | ".join(f'"{keyword}"' for keyword in keywords[:6]),
                "language": "en",
                "limit": min(limit, 50),
                "published_after": self._since(lookback_days),
            },
            cache_namespace="news_sector",
        )
        return self._parse_articles(payload, NewsCategory.SECTOR)

    def fetch_macro_news(self, *, limit: int = 25, lookback_days: int = 7) -> list[Article]:
        """Macro and policy news.

        The search terms are deliberately narrow. The brief is explicit that
        generic world news must not appear, so this targets the transmission
        channels that actually move equity prices -- rates, inflation, central
        banks, tariffs, commodity shocks -- not general current affairs.
        """
        payload = self._get(
            {
                "search": (
                    '"interest rate" | "inflation" | "central bank" | "Federal Reserve" | '
                    '"monetary policy" | "tariff" | "trade policy" | "recession"'
                ),
                "language": "en",
                "limit": min(limit, 50),
                "published_after": self._since(lookback_days),
            },
            cache_namespace="news_macro",
        )
        return self._parse_articles(payload, NewsCategory.MACRO)
