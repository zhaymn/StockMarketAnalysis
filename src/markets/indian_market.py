"""Indian equities market provider (NSE-listed, with BSE as a secondary listing).

NSE's own site (``nseindia.com``, ``archives.nseindia.com``) blocks
automated requests outright -- verified directly: HTTP 403 on the homepage
and HTTP 503 on the equity-list CSV, even with a realistic browser
User-Agent and a cookie-priming request first. This is NSE's WAF/anti-bot
protection, a widely reported real-world limitation of scripted NSE access,
not something fixable from application code.

So instead of a bulk NSE directory file, this provider is built entirely on
Yahoo Finance's live search API (``src.markets.yahoo_search``), which does
correctly index NSE-listed equities under the ``"NSI"`` exchange code.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.markets.base import MarketProvider, StockResult
from src.markets.yahoo_search import yahoo_finance_search
from src.utils import get_logger

logger = get_logger(__name__)

#: Yahoo Finance exchange codes for Indian listings, mapped to a display label.
NSE_EXCHANGES: dict[str, str] = {"NSI": "NSE", "BSE": "Bombay"}

#: A small number of major NSE-listed companies whose global brand name
#: resolves to a foreign ADR listing in Yahoo's fuzzy company-name search
#: rather than their NSE listing -- confirmed by direct testing (searching
#: "Infosys" surfaces the NYSE-listed ADR and never returns ``INFY.NS`` in
#: the top 20 results). This tiny alias map exists only to correct that
#: specific, verified ranking quirk; it is not a general stock directory.
KNOWN_ADR_ALIASES: dict[str, str] = {
    "infosys": "INFY.NS",
}

#: Seed queries used to build the starter directory shown before any search
#: is typed. These are real, live Yahoo Finance queries executed at load
#: time (concurrently, and cached by the caller) -- not a static ticker
#: list -- covering large, well-known NSE companies across sectors.
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
    """Collapse per-exchange duplicates of the same company to a single best listing.

    Yahoo's search returns a company's plain NSE listing, its NSE
    block-deal series (``-BL`` suffix), and its BSE listing as separate
    quotes with near-identical names -- fine for ``search()`` (more
    context for the user), but noisy for a default browsable directory
    where each company should appear once. Prefers plain ``.NS`` over
    ``-BL.NS`` variants over ``.BO``.
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
        current_best = best_by_name.get(key)
        if current_best is None or preference_score(result) < preference_score(current_best):
            best_by_name[key] = result
    return list(best_by_name.values())


class IndianMarketProvider(MarketProvider):
    """NSE/BSE-listed equities, sourced live from Yahoo Finance (see module docstring)."""

    market_id = "indian"
    market_label = "Indian Market"

    def search(self, query: str, limit: int = 20) -> list[StockResult]:
        results = yahoo_finance_search(query, NSE_EXCHANGES, limit=limit)

        alias_symbol = self._match_known_alias(query)
        if alias_symbol and not any(r.symbol == alias_symbol for r in results):
            alias_hit = yahoo_finance_search(alias_symbol, NSE_EXCHANGES, limit=1)
            results = alias_hit + results

        return results[:limit]

    @staticmethod
    def _match_known_alias(query: str) -> str | None:
        normalized = query.strip().lower()
        if not normalized:
            return None
        for name_fragment, symbol in KNOWN_ADR_ALIASES.items():
            if normalized in name_fragment or name_fragment in normalized:
                return symbol
        return None

    def list_directory(self) -> list[StockResult]:
        """A starter directory of large-cap NSE stocks, built from live queries.

        Runs the seed queries concurrently since each is a separate HTTP
        round-trip; sequentially this would take 10+ seconds, too slow for
        an interactive app on every cache miss.
        """
        seen_symbols: set[str] = set()
        directory: list[StockResult] = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(yahoo_finance_search, query, NSE_EXCHANGES, 3): query
                for query in SEED_QUERIES
            }
            for future in as_completed(futures):
                query = futures[future]
                try:
                    for result in future.result():
                        if result.symbol not in seen_symbols:
                            seen_symbols.add(result.symbol)
                            directory.append(result)
                except Exception as exc:
                    # A single failed seed query shouldn't sink the whole directory.
                    logger.warning(
                        "Seed query %r failed while building Indian directory: %s", query, exc
                    )

        directory = _dedupe_preferring_plain_nse(directory)
        directory.sort(key=lambda r: r.name)
        logger.info(
            "Built Indian market starter directory: %d stocks from %d seed queries",
            len(directory), len(SEED_QUERIES),
        )
        return directory
