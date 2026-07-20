"""Pluggable market-provider abstraction.

Adding a new market (London Stock Exchange, Tokyo, Hong Kong, crypto, ETFs,
mutual funds, commodities, ...) means writing one new class that implements
``MarketProvider`` and registering it in ``src.markets.registry`` — the core
prediction pipeline (``src.preprocessing``, ``src.dataset``, ``src.model``,
``src.train``, ``src.evaluate``, ``src.predict``) never has to change, since
it only ever consumes a plain yfinance-ready ticker string, regardless of
which provider produced it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class StockResult:
    """One stock as surfaced by a market provider's directory or search."""

    symbol: str
    """yfinance-ready ticker, e.g. ``"RELIANCE.NS"`` or ``"AAPL"``."""

    name: str
    """Company name, e.g. ``"Reliance Industries Ltd"``."""

    exchange: str
    """Display label for the listing exchange, e.g. ``"NSE"``, ``"NASDAQ"``, ``"NYSE"``."""

    def label(self) -> str:
        """Human-readable ``"Company Name (SYMBOL)"`` string for dropdown display."""
        return f"{self.name} ({self.symbol})"


class MarketProvider(ABC):
    """A source of tradable stocks for one market, e.g. Indian or International equities."""

    market_id: str
    market_label: str

    @abstractmethod
    def search(self, query: str, limit: int = 20) -> list[StockResult]:
        """Look up stocks matching a free-text company name or ticker symbol query.

        Args:
            query: User-entered search text, e.g. ``"Bhar"`` or ``"AAPL"``.
            limit: Maximum number of results to return.

        Returns:
            Matching stocks, most relevant first. Empty on no matches or a
            transient lookup failure (providers log failures rather than
            raising, so a flaky network call degrades to "no results"
            instead of crashing the app).
        """

    def list_directory(self) -> list[StockResult]:
        """A browsable default list shown before the user types a search query.

        Providers backed by a reliable bulk data source (e.g. a full exchange
        symbol directory file) should override this with the real thing;
        providers that only support live search may leave this as a smaller
        curated starter list, or an empty list if even that isn't available.
        """
        return []
