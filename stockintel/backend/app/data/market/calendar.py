"""Market session status and data-freshness labelling.

Deliberately **not** backed by a hardcoded exchange-holiday table. Such a table
is correct the day it is written and quietly wrong a year later, at which point
the app confidently reports "OPEN" on Thanksgiving or Diwali. Instead:

* Weekends and clock time are computed from the exchange timezone (reliable,
  rule-based, never stales).
* Exchange holidays are *inferred from the data*: if it is a weekday inside
  session hours but the most recent bar is older than today, the session is
  reported as `CLOSED_NO_SESSION` -- an honest "the exchange published no bar
  for today" rather than a guess about why.

The freshness label is what the UI shows next to a price, so it must never
overstate. `REAL_TIME` is never returned: yfinance is a delayed feed, and
claiming otherwise would be exactly the kind of fabrication the brief forbids.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

from app.data.market.base import MarketConventions


class MarketStatus(str, Enum):
    """Current state of the exchange session."""

    OPEN = "OPEN"
    PRE_OPEN = "PRE_OPEN"
    CLOSED_WEEKEND = "CLOSED_WEEKEND"
    CLOSED_AFTER_HOURS = "CLOSED_AFTER_HOURS"
    CLOSED_NO_SESSION = "CLOSED_NO_SESSION"
    """A weekday within session hours for which the exchange published no bar
    -- an exchange holiday or a trading halt."""


class DataFreshness(str, Enum):
    """How current a price actually is. Never overstated."""

    DELAYED = "DELAYED"
    """Live-ish quote from a delayed feed (yfinance: typically 15 min)."""

    END_OF_DAY = "END_OF_DAY"
    """Settled close for the most recent completed session."""

    CACHED = "CACHED"
    """Served from local cache; see `age_seconds` for how stale."""

    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class SessionInfo:
    """Exchange session state at a point in time."""

    status: MarketStatus
    exchange_time: datetime
    timezone: str
    next_open: datetime | None
    last_session_date: date | None
    """Date of the most recent bar the data source published."""

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "status_label": _STATUS_LABELS[self.status],
            "exchange_time": self.exchange_time.isoformat(),
            "timezone": self.timezone,
            "next_open": self.next_open.isoformat() if self.next_open else None,
            "last_session_date": (
                self.last_session_date.isoformat() if self.last_session_date else None
            ),
        }


_STATUS_LABELS: dict[MarketStatus, str] = {
    MarketStatus.OPEN: "Market open",
    MarketStatus.PRE_OPEN: "Pre-open",
    MarketStatus.CLOSED_WEEKEND: "Closed — weekend",
    MarketStatus.CLOSED_AFTER_HOURS: "Closed — after hours",
    MarketStatus.CLOSED_NO_SESSION: "Closed — no session today",
}


def get_session_info(
    conventions: MarketConventions,
    *,
    last_session_date: date | None = None,
    now: datetime | None = None,
) -> SessionInfo:
    """Determine exchange session state.

    Args:
        conventions: Market conventions supplying timezone and hours.
        last_session_date: Date of the newest available bar. Enables holiday
            inference; omit it and holidays are indistinguishable from open days.
        now: Override for the current instant (testing).
    """
    tz = ZoneInfo(conventions.timezone)
    exchange_now = (now.astimezone(tz) if now else datetime.now(tz))
    current_time = exchange_now.time()
    is_weekend = exchange_now.weekday() >= 5  # 5=Sat, 6=Sun

    if is_weekend:
        status = MarketStatus.CLOSED_WEEKEND
    elif current_time < conventions.open_time:
        status = MarketStatus.PRE_OPEN
    elif current_time > conventions.close_time:
        status = MarketStatus.CLOSED_AFTER_HOURS
    elif last_session_date is not None and last_session_date < exchange_now.date():
        # Weekday, inside hours, yet no bar published for today.
        status = MarketStatus.CLOSED_NO_SESSION
    else:
        status = MarketStatus.OPEN

    return SessionInfo(
        status=status,
        exchange_time=exchange_now,
        timezone=conventions.timezone,
        next_open=_next_open(exchange_now, conventions),
        last_session_date=last_session_date,
    )


def _next_open(exchange_now: datetime, conventions: MarketConventions) -> datetime | None:
    """Next weekday session open. Ignores holidays -- see module docstring.

    Returns an upper bound on how soon trading could resume, which is why the
    UI labels it "next scheduled open" rather than a guarantee.
    """
    candidate = exchange_now
    if candidate.time() >= conventions.open_time:
        candidate = candidate + timedelta(days=1)

    for _ in range(7):
        if candidate.weekday() < 5:
            return candidate.replace(
                hour=conventions.open_time.hour,
                minute=conventions.open_time.minute,
                second=0,
                microsecond=0,
            )
        candidate = candidate + timedelta(days=1)
    return None


def classify_freshness(
    *,
    served_from_cache: bool,
    session_status: MarketStatus,
) -> DataFreshness:
    """Label how current the served price is.

    Note the deliberate absence of a REAL_TIME branch: the development data
    source is a delayed feed and the label must say so.
    """
    if served_from_cache:
        return DataFreshness.CACHED
    if session_status is MarketStatus.OPEN:
        return DataFreshness.DELAYED
    return DataFreshness.END_OF_DAY
