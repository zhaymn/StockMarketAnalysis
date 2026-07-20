"""United States equities (NYSE / NASDAQ / AMEX), sourced from Yahoo Finance."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import time as dt_time

from app.core.logging import get_logger
from app.data.market.base import MarketConventions, MarketProvider, StockResult
from app.data.market.yahoo_search import yahoo_finance_search

logger = get_logger(__name__)

#: Yahoo exchange codes for US listings -> display label.
US_EXCHANGES: dict[str, str] = {
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NYQ": "NYSE",
    "ASE": "NYSE American",
    "PCX": "NYSE Arca",
    "BATS": "Cboe BZX",
}

#: Seed queries for the starter directory. Executed live against Yahoo at load
#: time (concurrently, then cached) -- not a static ticker table baked into
#: source. Chosen for sector breadth, not just the mega-cap technology names.
SEED_QUERIES: tuple[str, ...] = (
    "Apple", "Microsoft", "NVIDIA", "Alphabet", "Amazon.com", "Meta Platforms",
    "Tesla", "Broadcom", "JPMorgan Chase", "Visa", "Eli Lilly", "UnitedHealth",
    "Exxon Mobil", "Johnson & Johnson", "Walmart", "Procter & Gamble",
    "Mastercard", "Home Depot", "Chevron", "Merck", "AbbVie", "Coca-Cola",
    "Advanced Micro Devices", "Netflix", "Adobe", "Salesforce", "Intel",
    "Caterpillar", "Boeing", "Goldman Sachs", "Pfizer", "Disney",
    "Ford Motor", "Starbucks", "Qualcomm",
)


class USMarketProvider(MarketProvider):
    """US-listed equities."""

    market_id = "us"
    market_label = "United States"

    conventions = MarketConventions(
        currency_code="USD",
        currency_symbol="$",
        benchmark_symbol="^GSPC",
        benchmark_label="S&P 500",
        timezone="America/New_York",
        open_time=dt_time(9, 30),
        close_time=dt_time(16, 0),
        # The NYSE calendar averages ~252 sessions per year.
        trading_days_per_year=252,
    )

    def search(self, query: str, limit: int = 20) -> list[StockResult]:
        return yahoo_finance_search(query, US_EXCHANGES, limit=limit)

    def list_directory(self) -> list[StockResult]:
        """Starter directory built from live concurrent seed queries."""
        seen: set[str] = set()
        directory: list[StockResult] = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(yahoo_finance_search, query, US_EXCHANGES, 2): query
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
                    # One failed seed must not sink the whole directory.
                    logger.warning("Seed query %r failed for US directory: %s", query, exc)

        directory.sort(key=lambda r: r.name)
        logger.info(
            "Built US starter directory: %d stocks from %d seed queries",
            len(directory), len(SEED_QUERIES),
        )
        return directory
