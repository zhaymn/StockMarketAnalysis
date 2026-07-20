"""Company profile and fundamentals.

Sourced from yfinance, which surfaces Yahoo's company metadata. Coverage is
uneven -- especially for Indian listings -- so every field is optional and the
API reports which ones were actually available. The dashboard renders
DATA UNAVAILABLE for the rest rather than showing a plausible-looking zero.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.errors import DataUnavailableError
from app.core.logging import get_logger
from app.data.cache.store import cache_key, get_cache

logger = get_logger(__name__)


@dataclass
class CompanyProfile:
    """Company identity and sector classification.

    Drives the relevance engine (sector matching needs to know the sector) and
    the fundamentals panel.
    """

    symbol: str
    name: str
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    website: str | None = None
    summary: str = ""
    employees: int | None = None

    # --- Valuation / health, all optional ---------------------------------
    market_cap: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    eps_trailing: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    debt_to_equity: float | None = None
    free_cash_flow: float | None = None
    beta: float | None = None
    dividend_yield: float | None = None

    fetched_at: float = 0.0
    unavailable_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "sector": self.sector,
            "industry": self.industry,
            "country": self.country,
            "website": self.website,
            "summary": self.summary,
            "employees": self.employees,
            "fundamentals": {
                "market_cap": self.market_cap,
                "trailing_pe": self.trailing_pe,
                "forward_pe": self.forward_pe,
                "eps_trailing": self.eps_trailing,
                "profit_margin": self.profit_margin,
                "revenue_growth": self.revenue_growth,
                "earnings_growth": self.earnings_growth,
                "debt_to_equity": self.debt_to_equity,
                "free_cash_flow": self.free_cash_flow,
                "beta": self.beta,
                "dividend_yield": self.dividend_yield,
            },
            "unavailable_fields": self.unavailable_fields,
            "fetched_at": self.fetched_at,
        }


def _coerce_number(value: object) -> float | None:
    """Convert to float, treating placeholders as missing.

    Yahoo returns 0 for genuinely unknown ratios as often as for a true zero.
    A P/E of 0 is not meaningful, so zeros in ratio fields are treated as
    missing rather than displayed as real values.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def fetch_profile(symbol: str, *, use_cache: bool = True) -> CompanyProfile:
    """Fetch a company profile.

    Raises:
        DataUnavailableError: The provider returned no usable metadata.
    """
    settings = get_settings()
    cache = get_cache()
    key = cache_key("profile", symbol.upper())

    if use_cache:
        hit = cache.get_json("fundamentals", key)
        if hit is not None:
            payload = dict(hit.value)
            payload.pop("fundamentals", None)
            return CompanyProfile(**payload)

    try:
        import yfinance as yf

        info = yf.Ticker(symbol).info or {}
    except Exception as exc:
        logger.warning("Profile fetch failed for %s: %s", symbol, exc)
        raise DataUnavailableError(
            f"Could not retrieve company information for '{symbol}'.",
            detail=str(exc),
        ) from exc

    name = info.get("longName") or info.get("shortName")
    if not name:
        raise DataUnavailableError(
            f"No company profile available for '{symbol}'.",
            detail="The data provider returned no company metadata for this symbol.",
        )

    ratio_fields = {
        "market_cap": _coerce_number(info.get("marketCap")),
        "trailing_pe": _coerce_number(info.get("trailingPE")),
        "forward_pe": _coerce_number(info.get("forwardPE")),
        "eps_trailing": _coerce_number(info.get("trailingEps")),
        "profit_margin": _coerce_number(info.get("profitMargins")),
        "revenue_growth": _coerce_number(info.get("revenueGrowth")),
        "earnings_growth": _coerce_number(info.get("earningsGrowth")),
        "debt_to_equity": _coerce_number(info.get("debtToEquity")),
        "free_cash_flow": _coerce_number(info.get("freeCashflow")),
        "beta": _coerce_number(info.get("beta")),
        "dividend_yield": _coerce_number(info.get("dividendYield")),
    }

    profile = CompanyProfile(
        symbol=symbol.upper(),
        name=name,
        sector=info.get("sector") or None,
        industry=info.get("industry") or None,
        country=info.get("country") or None,
        website=info.get("website") or None,
        summary=(info.get("longBusinessSummary") or "").strip(),
        employees=info.get("fullTimeEmployees"),
        fetched_at=time.time(),
        unavailable_fields=[k for k, v in ratio_fields.items() if v is None],
        **ratio_fields,
    )

    if use_cache:
        payload = profile.to_dict()
        payload.update(payload.pop("fundamentals"))
        cache.set_json(
            "fundamentals", key, payload,
            ttl_seconds=settings.cache_ttl_fundamentals, fetched_at=profile.fetched_at,
        )

    if profile.unavailable_fields:
        logger.info(
            "%s: %d fundamental field(s) unavailable: %s",
            symbol, len(profile.unavailable_fields), ", ".join(profile.unavailable_fields),
        )

    return profile
