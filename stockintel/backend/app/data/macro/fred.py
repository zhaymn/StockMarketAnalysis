"""Macroeconomic series from FRED (Federal Reserve Bank of St. Louis).

Supplies the macro context the brief asks for — policy rates, inflation,
yields, volatility — as *measured series* rather than as keyword-matched news
headlines.

**Staleness is a first-class field, not an afterthought.** FRED's coverage is
far better for the US than for India: US series update daily or monthly, while
several Indian series lag by a year or more (its discount-rate series last
published in 2022). Rendering a 2022 policy rate beside a live share price,
with no indication of its age, would be exactly the kind of quiet fabrication
this project exists to avoid. So every observation carries its own date and an
`is_stale` flag, and the UI shows both.

Series are chosen for relevance to equity pricing: the policy rate and yield
curve drive discount rates, inflation drives the policy rate, and VIX is the
market's own volatility expectation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import requests

from app.core.config import get_settings
from app.core.errors import IntegrationNotConfiguredError, ProviderError
from app.core.logging import get_logger
from app.data.cache.store import cache_key, get_cache

logger = get_logger(__name__)

API_BASE = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_TIMEOUT_SECONDS = 30

#: Macro data updates daily at most, and monthly for most series.
CACHE_TTL_SECONDS = 6 * 3600


@dataclass(frozen=True)
class SeriesSpec:
    """One macro series and how to present it."""

    series_id: str
    label: str
    unit: str
    description: str

    #: Months after which an observation is considered stale for this series.
    #: A daily yield going a month without an update is broken; an annual GDP
    #: figure at 11 months old is simply normal.
    stale_after_months: int = 3

    #: Report year-over-year percentage change instead of the level. Index
    #: series like CPI (332.568) are meaningless as levels to a reader.
    as_yoy_change: bool = False


US_SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec("FEDFUNDS", "Policy rate", "%",
               "Effective federal funds rate — the anchor for discount rates.",
               stale_after_months=2),
    SeriesSpec("DGS10", "10-year yield", "%",
               "10-year Treasury yield — the long-horizon discount rate.",
               stale_after_months=1),
    SeriesSpec("T10Y2Y", "Yield curve (10y−2y)", "pp",
               "Negative values have historically preceded recessions.",
               stale_after_months=1),
    SeriesSpec("CPIAUCSL", "Inflation (CPI)", "% y/y",
               "Consumer price inflation — drives policy-rate expectations.",
               stale_after_months=3, as_yoy_change=True),
    SeriesSpec("UNRATE", "Unemployment", "%",
               "Labour-market slack, the other half of the Fed's mandate.",
               stale_after_months=2),
    SeriesSpec("VIXCLS", "VIX", "",
               "The market's own expectation of near-term volatility.",
               stale_after_months=1),
)

#: India's FRED coverage is materially thinner and slower than the US's. The
#: discount-rate series (INTDSRINM193N) is deliberately excluded: it last
#: published in 2022 and would be actively misleading beside a live price.
#: Thresholds are set by each series' own publication frequency, NOT by how
#: old the data happens to be. An earlier draft set them generously enough
#: that a 507-day-old CPI reading reported as fresh -- which is tuning the
#: threshold to flatter the data, the exact failure this project rejects. A
#: monthly series 17 months behind is stale, and the UI should say so.
INDIA_SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec("IRSTCI01INM156N", "Interbank rate", "%",
               "Short-term interbank lending rate — a policy-stance proxy.",
               stale_after_months=4),          # monthly
    SeriesSpec("INDCPIALLMINMEI", "Inflation (CPI)", "% y/y",
               "Consumer price inflation.",
               stale_after_months=4, as_yoy_change=True),   # monthly
    SeriesSpec("FPCPITOTLZGIND", "Inflation (annual)", "%",
               "World Bank annual inflation rate.",
               stale_after_months=15),         # annual
    SeriesSpec("INDGDPRQPSMEI", "GDP growth", "%",
               "Quarterly real GDP growth.",
               stale_after_months=6),          # quarterly
)

SERIES_BY_MARKET: dict[str, tuple[SeriesSpec, ...]] = {
    "us": US_SERIES,
    "india": INDIA_SERIES,
}


@dataclass
class MacroObservation:
    """One series' latest value, with provenance."""

    series_id: str
    label: str
    unit: str
    description: str

    value: float | None
    observation_date: str | None
    change: float | None = None
    """Change against the previous observation, in the same unit."""

    is_stale: bool = False
    age_days: int | None = None
    unavailable_reason: str | None = None

    @property
    def is_available(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "series_id": self.series_id,
            "label": self.label,
            "unit": self.unit,
            "description": self.description,
            "value": round(self.value, 4) if self.value is not None else None,
            "change": round(self.change, 4) if self.change is not None else None,
            "observation_date": self.observation_date,
            "age_days": self.age_days,
            "is_stale": self.is_stale,
            "available": self.is_available,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass
class MacroSnapshot:
    """All macro series for one market."""

    market_id: str
    observations: list[MacroObservation] = field(default_factory=list)
    fetched_at: float = 0.0

    @property
    def stale_count(self) -> int:
        return sum(1 for o in self.observations if o.is_stale and o.is_available)

    def to_dict(self) -> dict[str, object]:
        available = [o for o in self.observations if o.is_available]
        return {
            "market_id": self.market_id,
            "source": "FRED (Federal Reserve Bank of St. Louis)",
            "observations": [o.to_dict() for o in self.observations],
            "n_available": len(available),
            "n_stale": self.stale_count,
            "fetched_at": self.fetched_at,
            "coverage_note": (
                "FRED's coverage of Indian macro series lags materially behind its "
                "US coverage. Values marked stale are the most recent published, "
                "not current readings."
                if self.market_id == "india" else None
            ),
        }


def _fetch_series(spec: SeriesSpec, api_key: str) -> list[dict]:
    """Recent observations for one series, newest first."""
    # 14 points is enough for a year-over-year comparison on monthly data,
    # plus a previous-value change on daily data.
    response = requests.get(
        API_BASE,
        params={
            "series_id": spec.series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 14,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if not response.ok:
        # Message deliberately excludes the URL, which carries the api_key.
        raise ProviderError(
            f"FRED returned HTTP {response.status_code} for {spec.series_id}."
        )

    return response.json().get("observations", [])


def _to_float(raw: str | None) -> float | None:
    """FRED encodes missing observations as '.'."""
    if raw is None or raw in (".", ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _build_observation(spec: SeriesSpec, raw: list[dict]) -> MacroObservation:
    base = MacroObservation(
        series_id=spec.series_id, label=spec.label, unit=spec.unit,
        description=spec.description, value=None, observation_date=None,
    )

    points = [(o["date"], _to_float(o.get("value"))) for o in raw]
    points = [(d, v) for d, v in points if v is not None]

    if not points:
        base.unavailable_reason = "No observations returned."
        return base

    latest_date, latest_value = points[0]

    if spec.as_yoy_change:
        # Compare against roughly a year earlier. Monthly series give index
        # levels, which mean nothing to a reader as raw numbers.
        year_ago = next(
            (v for d, v in points if _months_between(d, latest_date) >= 11), None
        )
        if year_ago is None or year_ago == 0:
            base.unavailable_reason = "Not enough history for a year-over-year change."
            base.observation_date = latest_date
            return base
        value = (latest_value / year_ago - 1.0) * 100.0
        change = None
    else:
        value = latest_value
        change = latest_value - points[1][1] if len(points) > 1 else None

    age_days = (date.today() - date.fromisoformat(latest_date)).days

    return MacroObservation(
        series_id=spec.series_id, label=spec.label, unit=spec.unit,
        description=spec.description, value=value, observation_date=latest_date,
        change=change, age_days=age_days,
        is_stale=age_days > spec.stale_after_months * 31,
    )


def _refresh_staleness(
    observation: MacroObservation, specs: tuple[SeriesSpec, ...]
) -> MacroObservation:
    """Recompute age and staleness against the current date."""
    if observation.observation_date is None:
        return observation

    spec = next((s for s in specs if s.series_id == observation.series_id), None)
    if spec is None:
        return observation

    age_days = (date.today() - date.fromisoformat(observation.observation_date)).days
    observation.age_days = age_days
    observation.is_stale = age_days > spec.stale_after_months * 31
    return observation


def _months_between(earlier: str, later: str) -> int:
    a = date.fromisoformat(earlier)
    b = date.fromisoformat(later)
    return (b.year - a.year) * 12 + (b.month - a.month)


def fetch_macro_snapshot(market_id: str, *, use_cache: bool = True) -> MacroSnapshot:
    """Latest macro readings for one market.

    Raises:
        IntegrationNotConfiguredError: No FRED key configured.

    A single failing series degrades to `available: false` for that series
    rather than failing the whole snapshot.
    """
    settings = get_settings()
    if not settings.has_macro_provider:
        raise IntegrationNotConfiguredError(
            "Macro data",
            env_var="FRED_API_KEY",
            obtain_at="https://fred.stlouisfed.org/docs/api/api_key.html",
            reason="Required for policy rates, inflation, yields and volatility.",
        )

    specs = SERIES_BY_MARKET.get(market_id)
    if not specs:
        return MacroSnapshot(market_id=market_id, fetched_at=time.time())

    cache = get_cache()
    key = cache_key("macro", market_id)

    if use_cache:
        hit = cache.get_json("fred", key)
        if hit is not None:
            snapshot = MacroSnapshot(market_id=market_id, fetched_at=hit.fetched_at)
            # Age and staleness are recomputed on read, never restored from the
            # cache. They are properties of "how old is this now", not of the
            # fetch, so a cached snapshot must not freeze them: a series can
            # cross the staleness boundary while cached, and a changed
            # threshold must take effect immediately rather than after the TTL.
            snapshot.observations = [
                _refresh_staleness(MacroObservation(**row), specs) for row in hit.value
            ]
            return snapshot

    observations: list[MacroObservation] = []
    for spec in specs:
        try:
            observations.append(_build_observation(spec, _fetch_series(spec, settings.fred_api_key)))
        except Exception as exc:
            logger.warning("FRED series %s unavailable: %s", spec.series_id, exc)
            observations.append(
                MacroObservation(
                    series_id=spec.series_id, label=spec.label, unit=spec.unit,
                    description=spec.description, value=None, observation_date=None,
                    unavailable_reason=f"{type(exc).__name__} while fetching.",
                )
            )

    snapshot = MacroSnapshot(
        market_id=market_id, observations=observations, fetched_at=time.time()
    )

    if use_cache and any(o.is_available for o in observations):
        cache.set_json(
            "fred", key,
            [
                {
                    "series_id": o.series_id, "label": o.label, "unit": o.unit,
                    "description": o.description, "value": o.value,
                    "observation_date": o.observation_date, "change": o.change,
                    "unavailable_reason": o.unavailable_reason,
                    # is_stale and age_days are deliberately NOT cached; see
                    # _refresh_staleness.
                }
                for o in observations
            ],
            ttl_seconds=CACHE_TTL_SECONDS, fetched_at=snapshot.fetched_at,
        )

    logger.info(
        "FRED %s: %d/%d series available, %d stale",
        market_id, sum(1 for o in observations if o.is_available),
        len(observations), snapshot.stale_count,
    )
    return snapshot
