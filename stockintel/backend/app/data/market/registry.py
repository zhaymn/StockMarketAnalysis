"""Registry of available market providers.

Adding a market is two steps: implement a `MarketProvider` subclass, then add
one line to `MARKET_REGISTRY`. Nothing else in the application changes.
"""

from __future__ import annotations

from app.core.errors import UnknownTickerError
from app.data.market.base import MarketProvider
from app.data.market.india_market import IndiaMarketProvider
from app.data.market.us_market import USMarketProvider

MARKET_REGISTRY: dict[str, MarketProvider] = {
    provider.market_id: provider
    for provider in (USMarketProvider(), IndiaMarketProvider())
}

DEFAULT_MARKET_ID = "us"


def get_market_provider(market_id: str) -> MarketProvider:
    """Look up a provider by id.

    Raises:
        UnknownTickerError: If `market_id` is not registered.
    """
    provider = MARKET_REGISTRY.get(market_id)
    if provider is None:
        raise UnknownTickerError(
            f"Unknown market '{market_id}'.",
            detail=f"Available markets: {', '.join(sorted(MARKET_REGISTRY))}.",
        )
    return provider


def list_markets() -> list[dict[str, object]]:
    """All registered markets, for the market selector."""
    return [provider.to_dict() for provider in MARKET_REGISTRY.values()]
