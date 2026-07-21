"""Backfill historical news, score it, and report coverage.

    python -m scripts.backfill_news AAPL --days 60 --max-requests 60

Resumable: completed days are recorded, so re-running continues where the last
run stopped. On the free tier (3 articles/request) a day costs 1-2 requests; on
a paid tier the same day costs one request and returns far more.

    python -m scripts.backfill_news --status        # coverage so far
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.data.news.archive import get_archive
from app.data.news.backfill import backfill_symbol, score_archived_articles
from app.data.market.prices import fetch_history
from app.data.market.registry import get_market_provider
from app.features.sentiment_features import build_sentiment_features

DEFAULT_SYMBOLS = ["AAPL", "NVDA", "MSFT", "RELIANCE.NS", "TCS.NS"]


def show_status(symbols: list[str]) -> None:
    archive = get_archive()
    print(f"{'symbol':<14} {'articles':>9} {'scored':>8} {'days':>6}  {'range'}")
    print("-" * 74)
    for symbol in symbols:
        s = archive.stats(symbol)
        first = (s["first_article"] or "")[:10]
        last = (s["last_article"] or "")[:10]
        span = f"{first} .. {last}" if first else "-"
        print(f"{symbol:<14} {s['articles']:>9} {s['scored']:>8} {s['days_covered']:>6}  {span}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", default=None)
    parser.add_argument("--days", type=int, default=30, help="Calendar days back from today")
    parser.add_argument("--max-requests", type=int, default=60,
                        help="API budget for this run (free tier allows 100/day)")
    parser.add_argument("--status", action="store_true", help="Show coverage and exit")
    parser.add_argument("--no-score", action="store_true", help="Fetch only, skip FinBERT")
    args = parser.parse_args()

    configure_logging("INFO")
    symbols = args.symbols or DEFAULT_SYMBOLS

    if args.status:
        show_status(symbols)
        return 0

    if not get_settings().has_news_provider:
        print("MARKETAUX_API_KEY is not set. Nothing to do.")
        return 1

    end = date.today()
    start = end - timedelta(days=args.days)
    budget = args.max_requests

    for symbol in symbols:
        if budget <= 0:
            print(f"\nRequest budget exhausted before {symbol}.")
            break

        print(f"\n=== {symbol}: {start} .. {end} (budget {budget}) ===")
        progress = backfill_symbol(symbol, start, end, max_requests=budget)
        budget -= progress.requests_made

        print(f"  days completed : {progress.days_completed}")
        print(f"  days skipped   : {progress.days_skipped} (already fetched)")
        print(f"  requests used  : {progress.requests_made}")
        print(f"  new articles   : {progress.articles_stored}")
        if progress.stopped_early:
            print(f"  stopped early  : {progress.stopped_early}")
        if progress.errors:
            print(f"  errors         : {progress.errors[:3]}")

        if not args.no_score:
            scored = score_archived_articles(symbol)
            print(f"  scored         : {scored}")

    # --- Coverage against the actual trading calendar ----------------------
    print(f"\n{'=' * 74}")
    print("COVERAGE vs TRADING SESSIONS")
    print(f"{'=' * 74}")

    archive = get_archive()
    for symbol in symbols:
        stats = archive.stats(symbol)
        if not stats["articles"]:
            print(f"{symbol:<14} no articles archived")
            continue

        market = "india" if symbol.endswith((".NS", ".BO")) else "us"
        conventions = get_market_provider(market).conventions

        try:
            history = fetch_history(symbol, period="1y")
        except Exception as exc:
            print(f"{symbol:<14} price history unavailable: {exc}")
            continue

        articles = archive.get_articles(symbol, scored_only=True)
        if not articles:
            print(f"{symbol:<14} {stats['articles']} articles, none scored yet")
            continue

        # Restrict to the window news actually covers, so coverage is not
        # diluted by a year of sessions with no attempted backfill.
        first = min(a.published_at for a in articles)
        sessions = history.frame.index[history.frame.index >= first.replace(tzinfo=None)]
        if len(sessions) == 0:
            print(f"{symbol:<14} no overlapping sessions")
            continue

        _, coverage = build_sentiment_features(articles, sessions, conventions)
        c = coverage.to_dict()
        print(
            f"{symbol:<14} {c['sessions_with_news']}/{c['sessions']} sessions "
            f"({c['coverage_fraction']:.1%}), {c['total_articles']} articles, "
            f"{c['articles_per_covered_session']}/session, "
            f"usable={c['is_usable']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
