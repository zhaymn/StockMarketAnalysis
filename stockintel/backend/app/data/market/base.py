"""Pluggable market-provider abstraction.

Adapted from the provider design in the repository-root `src/markets` package,
extended with the market-convention metadata this platform needs: currency,
benchmark index, exchange timezone and trading hours. Those live on the
provider -- not scattered through the UI -- so currencies and conventions can
never be mixed between markets.

Adding a market means implementing `MarketProvider` and adding one line to
`app.data.market.registry`; nothing downstream changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import time as dt_time


@dataclass(frozen=True)
class StockResult:
    """One stock as surfaced by a provider's directory or search."""

    symbol: str
    """yfinance-ready ticker, e.g. `"RELIANCE.NS"` or `"AAPL"`."""

    name: str
    """Company name, e.g. `"Reliance Industries Ltd"`."""

    exchange: str
    """Display label for the listing exchange, e.g. `"NSE"`, `"NASDAQ"`."""

    def label(self) -> str:
        return f"{self.name} ({self.symbol})"

    def to_dict(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "exchange": self.exchange,
            "label": self.label(),
        }


@dataclass(frozen=True)
class MarketConventions:
    """Everything that must change when the user switches markets."""

    currency_code: str
    """ISO 4217 code, e.g. `"USD"`, `"INR"`."""

    currency_symbol: str
    """Display glyph, e.g. `"$"`, `"₹"`."""

    benchmark_symbol: str
    """yfinance ticker for the market benchmark, e.g. `"^GSPC"`, `"^NSEI"`."""

    benchmark_label: str
    """Human name, e.g. `"S&P 500"`, `"NIFTY 50"`."""

    timezone: str
    """IANA exchange timezone, e.g. `"America/New_York"`, `"Asia/Kolkata"`."""

    open_time: dt_time
    close_time: dt_time

    trading_days_per_year: int
    """Used to annualise volatility. US ~252, India ~250."""

    def to_dict(self) -> dict[str, object]:
        return {
            "currency_code": self.currency_code,
            "currency_symbol": self.currency_symbol,
            "benchmark_symbol": self.benchmark_symbol,
            "benchmark_label": self.benchmark_label,
            "timezone": self.timezone,
            "open_time": self.open_time.strftime("%H:%M"),
            "close_time": self.close_time.strftime("%H:%M"),
            "trading_days_per_year": self.trading_days_per_year,
        }


class MarketProvider(ABC):
    """A source of tradable stocks and conventions for one market."""

    market_id: str
    market_label: str
    conventions: MarketConventions

    @abstractmethod
    def search(self, query: str, limit: int = 20) -> list[StockResult]:
        """Look up stocks matching a company name or ticker fragment.

        Implementations log and return `[]` on transient lookup failure, so a
        flaky network call degrades to "no results" rather than an error page.
        """

    def list_directory(self) -> list[StockResult]:
        """Browsable default list shown before the user searches."""
        return []

    def to_dict(self) -> dict[str, object]:
        return {
            "market_id": self.market_id,
            "market_label": self.market_label,
            "conventions": self.conventions.to_dict(),
        }
