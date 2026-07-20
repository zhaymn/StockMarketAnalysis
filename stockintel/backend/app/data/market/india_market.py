"""Indian equities (NSE-listed, BSE as secondary), sourced from Yahoo Finance.

NSE's own site (`nseindia.com`, `archives.nseindia.com`) blocks automated
requests outright -- verified during development of the root `src/markets`
package: HTTP 403 on the homepage and HTTP 503 on the official equity-list
CSV, even with a realistic browser User-Agent and a cookie-priming request
first. That is NSE's WAF/anti-bot protection, a widely reported limitation of
scripted NSE access, and it is not fixable from application code.

So this provider is built on Yahoo Finance's live search API, which does index
NSE listings under the `"NSI"` exchange code. Ported from the root package's
`indian_market.py`, whose findings and dedupe logic still hold.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import time as dt_time

from app.core.logging import get_logger
from app.data.market.base import MarketConventions, MarketProvider, StockResult
from app.data.market.yahoo_search import yahoo_finance_search

logger = get_logger(__name__)

#: Yahoo exchange codes for Indian listings -> display label.
INDIA_EXCHANGES: dict[str, str] = {"NSI": "NSE", "BSE": "BSE"}

#: Major NSE companies whose global brand name resolves to a foreign ADR in
#: Yahoo's fuzzy search rather than the NSE listing -- confirmed by direct
#: testing (searching "Infosys" surfaces the NYSE ADR and never returns
#: INFY.NS in the top 20). This corrects that specific verified ranking quirk;
#: it is not a general stock directory.
KNOWN_ADR_ALIASES: dict[str, str] = {
    "infosys": "INFY.NS",
    "wipro": "WIPRO.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
}

SEED_QUERIES: tuple[str, ...] = (
    "Reliance Industries", "Tata Consultancy Services", "HDFC Bank",
    "ICICI Bank", "State Bank of India", "Bharti Airtel", "Hindustan Unilever",
    "ITC Limited", "Larsen Toubro", "Kotak Mahindra Bank", "AXISBANK.NS",
    "Bajaj Finance", "Maruti Suzuki", "Sun Pharma", "Tata Motors",
    "Tata Steel", "NTPC", "Adani Enterprises", "Asian Paints", "Titan Company",
    "UltraTech Cement", "Wipro", "HCL Technologies", "Power Grid Corporation",
    "Nestle India", "JSW Steel", "Mahindra Mahindra", "Coal India",
    "IndusInd Bank", "Dr Reddy", "INFY.NS",
)


def _dedupe_preferring_plain_nse(results: list[StockResult]) -> list[StockResult]:
    """Collapse per-exchange duplicates of one company to its best listing.

    Yahoo returns a company's plain NSE listing, its NSE block-deal series
    (`-BL` suffix) and its BSE listing as separate near-identically-named
    quotes. Useful context in search; noise in a browsable directory where each
    company should appear once. Prefers `.NS` over `-BL.NS` over `.BO`.
    """

    def preference_score(result: StockResult) -> int:
        if result.symbol.endswith(".NS") and "-BL" not in result.symbol:
            return 0
        if result.symbol.endswith(".NS"):
            return 1
        return 2

    def normalize_name(name: str) -> str:
        cleaned = name.upper()
        for suffix in (" LIMITED", " LTD.", " LTD", "."):
            cleaned = cleaned.replace(suffix, "")
        return cleaned.strip()

    best_by_name: dict[str, StockResult] = {}
    for result in results:
        key = normalize_name(result.name)
        current = best_by_name.get(key)
        if current is None or preference_score(result) < preference_score(current):
            best_by_name[key] = result
    return list(best_by_name.values())


class IndiaMarketProvider(MarketProvider):
    """NSE/BSE-listed equities."""

    market_id = "india"
    market_label = "India"

    conventions = MarketConventions(
        currency_code="INR",
        currency_symbol="₹",
        benchmark_symbol="^NSEI",
        benchmark_label="NIFTY 50",
        timezone="Asia/Kolkata",
        open_time=dt_time(9, 15),
        close_time=dt_time(15, 30),
        # NSE averages ~250 sessions/year -- more public holidays than the NYSE.
        trading_days_per_year=250,
    )

    def search(self, query: str, limit: int = 20) -> list[StockResult]:
        results = yahoo_finance_search(query, INDIA_EXCHANGES, limit=limit)

        alias_symbol = self._match_known_alias(query)
        if alias_symbol and not any(r.symbol == alias_symbol for r in results):
            alias_hit = yahoo_finance_search(alias_symbol, INDIA_EXCHANGES, limit=1)
            results = alias_hit + results

        return results[:limit]

    @staticmethod
    def _match_known_alias(query: str) -> str | None:
        normalized = query.strip().lower()
        if not normalized:
            return None
        for fragment, symbol in KNOWN_ADR_ALIASES.items():
            if normalized in fragment or fragment in normalized:
                return symbol
        return None

    def list_directory(self) -> list[StockResult]:
        """Starter directory of large-cap NSE stocks, built from live queries.

        Seed queries run concurrently: each is a separate HTTP round-trip and
        sequentially this takes 10+ seconds, too slow on every cache miss.
        """
        seen: set[str] = set()
        directory: list[StockResult] = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(yahoo_finance_search, query, INDIA_EXCHANGES, 3): query
                for query in SEED_QUERIES
            }
            for future in as_completed(futures):
                query = futures[future]
                try:
                    for result in future.result():
                        if result.symbol not in seen:
                            seen.add(result.symbol)
                            directory.append(result)
                except Exception as exc:
                    logger.warning("Seed query %r failed for India directory: %s", query, exc)

        directory = _dedupe_preferring_plain_nse(directory)
        directory.sort(key=lambda r: r.name)
        logger.info(
            "Built India starter directory: %d stocks from %d seed queries",
            len(directory), len(SEED_QUERIES),
        )
        return directory
