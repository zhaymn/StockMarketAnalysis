"""Live stock search against Yahoo Finance's public search endpoint.

Shared by every market provider so the HTTP request and response-parsing
logic exists in exactly one place. Each provider supplies the set of Yahoo
exchange codes it cares about (e.g. ``{"NSI": "NSE", "BSE": "Bombay"}`` for
Indian markets) and this function does the rest.
"""

from __future__ import annotations

import requests

from src.markets.base import StockResult
from src.utils import get_logger

logger = get_logger(__name__)

SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
REQUEST_TIMEOUT_SECONDS = 8
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def yahoo_finance_search(
    query: str,
    allowed_exchanges: dict[str, str],
    limit: int = 20,
) -> list[StockResult]:
    """Query Yahoo Finance's live search endpoint for equities on specific exchanges.

    Args:
        query: Free-text company name or ticker fragment, e.g. ``"Bharti"``.
        allowed_exchanges: Maps Yahoo exchange codes (e.g. ``"NSI"``) to a
            friendly display label (e.g. ``"NSE"``); only quotes whose
            exchange is a key in this dict are returned.
        limit: Maximum number of results to return.

    Returns:
        Matching ``StockResult`` entries, in the order Yahoo ranks them.
        Returns an empty list (and logs a warning) on any network or
        parsing failure rather than raising — a flaky search lookup should
        degrade to "no results," not crash the app.
    """
    query = query.strip()
    if not query:
        return []

    params = {
        "q": query,
        "quotesCount": max(limit, 10),
        "newsCount": 0,
        "listsCount": 0,
    }
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(
            SEARCH_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
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

    return results
