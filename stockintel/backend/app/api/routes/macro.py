"""Macroeconomic context from FRED."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.logging import get_logger
from app.data.macro.fred import fetch_macro_snapshot
from app.data.market.registry import get_market_provider

logger = get_logger(__name__)
router = APIRouter(prefix="/api/macro", tags=["macro"])


@router.get("/{market_id}")
async def macro(market_id: str) -> dict[str, object]:
    """Policy rate, inflation, yields and volatility for one market.

    Raises `IntegrationNotConfiguredError` (503) without a FRED key, which the
    frontend renders as an explicit NOT CONFIGURED state.
    """
    get_market_provider(market_id)  # validates the market id
    return fetch_macro_snapshot(market_id).to_dict()
