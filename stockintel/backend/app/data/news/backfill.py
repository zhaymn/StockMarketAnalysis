"""Historical news backfill.

Walks a date range day by day, paginating through Marketaux and storing
everything in the local archive.

**Day-at-a-time rather than one long range.** A single wide window with
pagination cannot be resumed: interrupt it and you do not know which articles
you already have. Fetching one calendar day at a time makes progress durable —
each completed day is recorded, and a resumed run skips it. That matters when a
backfill spans days of quota.

**Tier-agnostic.** The free tier caps responses at 3 articles and paid tiers
allow 100; the only difference is how many requests a day costs. The client
reads `meta.returned` rather than assuming, so the same code works on both.

**Quota-aware.** `max_requests` bounds a run so a backfill cannot silently
exhaust a daily allowance. Hitting it stops cleanly with progress saved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import requests

from app.core.config import get_settings
from app.core.errors import IntegrationNotConfiguredError, RateLimitedError
from app.core.logging import get_logger
from app.data.news.archive import ArchivedArticle, get_archive
from app.data.news.marketaux import API_BASE, SUFFIX_TO_COUNTRY

logger = get_logger(__name__)

REQUEST_TIMEOUT_SECONDS = 60
#: Politeness delay between requests. Marketaux does not publish a per-second
#: limit, and hammering it produced read timeouts during development.
INTER_REQUEST_DELAY_SECONDS = 0.35
MAX_PAGES_PER_DAY = 40


@dataclass
class BackfillProgress:
    """Outcome of a backfill run."""

    symbol: str
    days_requested: int = 0
    days_completed: int = 0
    days_skipped: int = 0
    requests_made: int = 0
    articles_stored: int = 0
    stopped_early: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "days_requested": self.days_requested,
            "days_completed": self.days_completed,
            "days_skipped": self.days_skipped,
            "requests_made": self.requests_made,
            "articles_stored": self.articles_stored,
            "stopped_early": self.stopped_early,
            "errors": self.errors[:10],
        }


def _api_symbol(symbol: str) -> tuple[str, str | None]:
    """Marketaux symbol and country filter for a yfinance ticker."""
    for suffix, country in SUFFIX_TO_COUNTRY.items():
        if symbol.upper().endswith(suffix):
            return symbol[: -len(suffix)], country
    return symbol, None


def _fetch_page(
    api_symbol: str,
    country: str | None,
    day: date,
    page: int,
    api_key: str,
) -> tuple[list[dict], int]:
    """One page of articles for one calendar day. Returns (items, returned)."""
    params: dict[str, object] = {
        "symbols": api_symbol,
        "filter_entities": "true",
        "language": "en",
        # Ask for the paid-tier maximum; free tiers silently cap it lower and
        # report the true figure in meta.returned.
        "limit": 100,
        "page": page,
        "published_after": f"{day.isoformat()}T00:00:00",
        "published_before": f"{(day + timedelta(days=1)).isoformat()}T00:00:00",
        "api_token": api_key,
    }
    if country:
        params["countries"] = country

    response = requests.get(API_BASE, params=params, timeout=REQUEST_TIMEOUT_SECONDS)

    # 402 is how Marketaux reports an exhausted plan allowance -- not 429, as
    # would be conventional. Observed in practice: once the daily quota is
    # gone, every subsequent day returns 402, and without this branch the
    # backfill logs a wall of opaque HTTPErrors instead of stopping cleanly.
    if response.status_code in (402, 429):
        raise RateLimitedError(
            "Marketaux plan allowance exhausted "
            f"(HTTP {response.status_code}). Progress is saved; re-run to resume.",
            retry_after=3600,
        )
    if response.status_code in (401, 403):
        raise IntegrationNotConfiguredError(
            "News provider",
            env_var="MARKETAUX_API_KEY",
            obtain_at="https://www.marketaux.com/",
            reason=f"Key rejected during backfill (HTTP {response.status_code}).",
        )
    if not response.ok:
        # Deliberately does NOT use raise_for_status(): its message embeds the
        # full request URL, which contains api_token=<secret>. That message
        # then lands in log files that may be shared or shipped.
        raise RuntimeError(
            f"Marketaux returned HTTP {response.status_code} for {day.isoformat()} "
            f"page {page}"
        )

    payload = response.json()
    return payload.get("data", []), payload.get("meta", {}).get("returned", 0)


def _parse(item: dict, symbol: str) -> ArchivedArticle | None:
    title = (item.get("title") or "").strip()
    url = item.get("url") or ""
    if not title or not url:
        return None

    raw = item.get("published_at") or ""
    try:
        published_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    return ArchivedArticle(
        url=url,
        symbol=symbol,
        title=title,
        description=(item.get("description") or item.get("snippet") or "").strip(),
        source=item.get("source") or "unknown",
        published_at=published_at,
    )


def backfill_symbol(
    symbol: str,
    start: date,
    end: date,
    *,
    max_requests: int | None = None,
    skip_weekends: bool = True,
) -> BackfillProgress:
    """Fetch and archive historical news for one symbol.

    Args:
        symbol: yfinance ticker, e.g. "AAPL" or "RELIANCE.NS".
        start: First calendar day (inclusive).
        end: Last calendar day (inclusive).
        max_requests: Stop after this many API calls, preserving progress.
        skip_weekends: Skip Saturday and Sunday. Weekend news exists and is
            attributed to the next session by the feature layer, but it is
            sparse, and on a quota-limited tier weekday coverage is worth more.

    Returns:
        `BackfillProgress`. Never raises for quota exhaustion -- it stops early
        and reports why, so a caller can resume tomorrow.
    """
    settings = get_settings()
    if not settings.has_news_provider:
        raise IntegrationNotConfiguredError(
            "News provider",
            env_var="MARKETAUX_API_KEY",
            obtain_at="https://www.marketaux.com/",
            reason="Required to backfill historical news.",
        )

    archive = get_archive()
    api_symbol, country = _api_symbol(symbol)
    already_done = archive.completed_days(symbol)

    progress = BackfillProgress(symbol=symbol)
    day = start

    while day <= end:
        if skip_weekends and day.weekday() >= 5:
            day += timedelta(days=1)
            continue

        progress.days_requested += 1
        key = day.isoformat()

        if key in already_done:
            progress.days_skipped += 1
            day += timedelta(days=1)
            continue

        if max_requests is not None and progress.requests_made >= max_requests:
            progress.stopped_early = (
                f"Reached the {max_requests}-request budget at {key}. "
                f"Re-run to resume from here."
            )
            break

        day_articles: list[ArchivedArticle] = []
        page = 1
        exhausted = False

        while page <= MAX_PAGES_PER_DAY:
            if max_requests is not None and progress.requests_made >= max_requests:
                break

            try:
                items, returned = _fetch_page(api_symbol, country, day, page, settings.marketaux_api_key)
                progress.requests_made += 1
            except RateLimitedError as exc:
                progress.stopped_early = f"Rate limited at {key}: {exc.message}"
                exhausted = True
                break
            except Exception as exc:
                # Message is truncated and never interpolates a URL: provider
                # error text can echo the request, api_token included.
                detail = str(exc)[:120]
                progress.errors.append(f"{key} page {page}: {type(exc).__name__}")
                logger.warning("Backfill error on %s page %d: %s", key, page, detail)
                break

            for item in items:
                parsed = _parse(item, symbol)
                if parsed is not None:
                    day_articles.append(parsed)

            # A short page means the day is exhausted.
            if returned == 0 or len(items) < returned or returned < 3:
                break
            if len(items) == 0:
                break

            page += 1
            time.sleep(INTER_REQUEST_DELAY_SECONDS)

        stored = archive.add_articles(symbol, day_articles)
        progress.articles_stored += stored

        if not exhausted:
            # Only mark complete if the day was actually finished, so a run cut
            # short by quota does not record partial coverage as done.
            archive.mark_day_complete(symbol, key, len(day_articles))
            progress.days_completed += 1

        if exhausted:
            break

        day += timedelta(days=1)
        time.sleep(INTER_REQUEST_DELAY_SECONDS)

    logger.info(
        "Backfill %s: %d days completed, %d skipped, %d requests, %d new articles",
        symbol, progress.days_completed, progress.days_skipped,
        progress.requests_made, progress.articles_stored,
    )
    return progress


def score_archived_articles(symbol: str, limit: int = 2000) -> int:
    """Run FinBERT over unscored archived articles. Returns how many scored.

    Kept separate from fetching so the two can run independently: fetching is
    quota-bound, scoring is CPU-bound, and neither should block the other.
    """
    from app.services import sentiment as sentiment_service

    archive = get_archive()
    pending = archive.unscored_articles(symbol, limit=limit)

    if not pending:
        return 0
    if not sentiment_service.is_available():
        logger.warning("FinBERT unavailable; cannot score archived articles.")
        return 0

    scored = 0
    for article in pending:
        text = f"{article.title}. {article.description}".strip()
        result = sentiment_service.analyse_text(text)
        if result.label.value == "UNAVAILABLE":
            continue
        archive.store_sentiment(article.url, symbol, result.label.value, result.scores)
        scored += 1

    logger.info("Scored %d/%d archived articles for %s", scored, len(pending), symbol)
    return scored
