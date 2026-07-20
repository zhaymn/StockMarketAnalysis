"""Live stock search against Yahoo Finance's public search endpoint.

Shared by every market provider so the HTTP and parsing logic lives in one
place. Ported from the repository-root `src/markets/yahoo_search.py`, with a
cache layer added: search is called on every keystroke-driven request, and
Yahoo will rate-limit an uncached implementation quickly.
"""

from __future__ import annotations

import requests

from app.core.config import get_settings
from app.core.logging import get_logger
from app.data.cache.store import cache_key, get_cache
from app.data.market.base import StockResult

logger = get_logger(__name__)

SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
REQUEST_TIMEOUT_SECONDS = 8
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

#: Search results are stable over minutes; a listing does not appear or vanish
#: within a trading session.
SEARCH_TTL_SECONDS = 3600


def yahoo_finance_search(
    query: str,
    allowed_exchanges: dict[str, str],
    limit: int = 20,
    *,
    use_cache: bool = True,
) -> list[StockResult]:
    """Query Yahoo's live search endpoint for equities on specific exchanges.

    Args:
        query: Free-text company name or ticker fragment.
        allowed_exchanges: Yahoo exchange code -> display label. Only quotes
            whose exchange is a key here are returned.
        limit: Maximum results.
        use_cache: Set False to force a live lookup.

    Returns:
        Matching results in Yahoo's ranking order; `[]` on any network or
        parse failure (logged, never raised).
    """
    query = query.strip()
    if not query:
        return []

    key = cache_key("search", query.lower(), sorted(allowed_exchanges), limit)
    cache = get_cache()

    if use_cache:
        hit = cache.get_json("yahoo_search", key)
        if hit is not None:
            return [StockResult(**row) for row in hit.value]

    params = {"q": query, "quotesCount": max(limit, 10), "newsCount": 0, "listsCount": 0}

    try:
        response = requests.get(
            SEARCH_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Yahoo Finance search failed for query %r: %s", query, exc)
        return []

    results: list[StockResult] = []
    for quote in payload.get("quotes", []):
        if quote.get("quoteType") != "EQUITY":
            continue
        exchange_code = quote.get("exchange")
        if exchange_code not in allowed_exchanges:
            continue
        symbol = quote.get("symbol")
        name = quote.get("shortname") or quote.get("longname")
        if not symbol or not name:
            continue
        results.append(
            StockResult(symbol=symbol, name=name, exchange=allowed_exchanges[exchange_code])
        )
        if len(results) >= limit:
            break

    if use_cache and results:
        cache.set_json(
            "yahoo_search",
            key,
            [{"symbol": r.symbol, "name": r.name, "exchange": r.exchange} for r in results],
            ttl_seconds=SEARCH_TTL_SECONDS,
        )

    return results
