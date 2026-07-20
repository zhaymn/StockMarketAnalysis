"""Registry of available market providers.

Adding a new market (London Stock Exchange, Tokyo, Hong Kong, crypto,
ETFs, mutual funds, commodities, ...) is exactly two steps:

1. Implement a new ``MarketProvider`` subclass (see ``src.markets.base``).
2. Add one line to ``MARKET_REGISTRY`` below.

Nothing else in the application -- not the prediction pipeline in
``src.preprocessing`` / ``src.dataset`` / ``src.model`` / ``src.train`` /
``src.evaluate`` / ``src.predict``, not the rest of the dashboard -- needs
to change, since every provider produces the same ``StockResult`` shape and
every downstream consumer only ever needs a plain yfinance-ready ticker
string, regardless of which market it came from.
"""

from __future__ import annotations

from src.markets.base import MarketProvider
from src.markets.indian_market import IndianMarketProvider
from src.markets.international_market import InternationalMarketProvider

MARKET_REGISTRY: dict[str, MarketProvider] = {
    provider.market_id: provider
    for provider in (IndianMarketProvider(), InternationalMarketProvider())
}


def get_market_provider(market_id: str) -> MarketProvider:
    """Look up a registered market provider by id.

    Args:
        market_id: One of ``MARKET_REGISTRY`` keys, e.g. ``"indian"``.

    Returns:
        The corresponding ``MarketProvider`` instance.

    Raises:
        ValueError: If ``market_id`` is not registered.
    """
    if market_id not in MARKET_REGISTRY:
        raise ValueError(
            f"Unknown market_id '{market_id}'. Available: {sorted(MARKET_REGISTRY)}"
        )
    return MARKET_REGISTRY[market_id]
