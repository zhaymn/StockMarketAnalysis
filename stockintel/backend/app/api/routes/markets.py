"""Market selection and stock search."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.logging import get_logger
from app.data.market.registry import get_market_provider, list_markets

logger = get_logger(__name__)
router = APIRouter(prefix="/api/markets", tags=["markets"])


@router.get("")
async def markets() -> dict[str, object]:
    """Available markets and their conventions (currency, benchmark, timezone)."""
    return {"markets": list_markets(), "default_market": "us"}


@router.get("/{market_id}/directory")
async def directory(market_id: str) -> dict[str, object]:
    """Browsable starter list of stocks, shown before the user searches."""
    provider = get_market_provider(market_id)
    stocks = provider.list_directory()
    return {
        "market_id": market_id,
        "market_label": provider.market_label,
        "count": len(stocks),
        "stocks": [stock.to_dict() for stock in stocks],
    }


@router.get("/{market_id}/search")
async def search(
    market_id: str,
    q: str = Query(..., min_length=1, description="Company name or ticker fragment"),
    limit: int = Query(20, ge=1, le=50),
) -> dict[str, object]:
    """Search for stocks within one market."""
    provider = get_market_provider(market_id)
    results = provider.search(q, limit=limit)
    return {
        "market_id": market_id,
        "query": q,
        "count": len(results),
        "stocks": [stock.to_dict() for stock in results],
    }
