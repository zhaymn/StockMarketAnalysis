"""International equities market provider (US-listed: NASDAQ, NYSE, and related).

Unlike the Indian market, a full, reliable bulk directory *is* available
here: NASDAQ Trader publishes plain-text symbol directories for every
NASDAQ-listed security and every other Tape-B (NYSE/NYSE American/NYSE
Arca/...) security it trades, and both endpoints are scriptable without
authentication (verified directly). Live search is still used for anything
outside that directory (foreign exchanges, newly listed symbols, etc.), via
the same Yahoo Finance search helper the Indian provider uses.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from src.markets.base import MarketProvider, StockResult
from src.markets.yahoo_search import yahoo_finance_search
from src.utils import get_logger

logger = get_logger(__name__)

#: Yahoo Finance exchange codes for major US listings, mapped to a display label.
US_EXCHANGES: dict[str, str] = {
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NYQ": "NYSE",
    "ASE": "NYSE American",
    "PCX": "NYSE Arca",
}

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
REQUEST_TIMEOUT_SECONDS = 15
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

#: ``otherlisted.txt``'s single-letter Exchange codes, mapped to a display label.
_OTHER_LISTED_EXCHANGE_LABELS: dict[str, str] = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}


def _fetch_text(url: str) -> str | None:
    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def _parse_nasdaq_listed(text: str) -> list[StockResult]:
    # Pipe-delimited; last line is a "File Creation Time: ..." footer, not data.
    df = pd.read_csv(io.StringIO(text), sep="|")
    df = df[df["Test Issue"] == "N"]
    if "ETF" in df.columns:
        df = df[df["ETF"] == "N"]
    return [
        StockResult(symbol=str(row["Symbol"]), name=str(row["Security Name"]), exchange="NASDAQ")
        for _, row in df.iterrows()
        if pd.notna(row["Symbol"]) and pd.notna(row["Security Name"])
    ]


def _parse_other_listed(text: str) -> list[StockResult]:
    df = pd.read_csv(io.StringIO(text), sep="|")
    df = df[df["Test Issue"] == "N"]
    if "ETF" in df.columns:
        df = df[df["ETF"] == "N"]

    results = []
    for _, row in df.iterrows():
        symbol = row.get("NASDAQ Symbol") or row.get("ACT Symbol")
        name = row.get("Security Name")
        if pd.isna(symbol) or pd.isna(name):
            continue
        # Yahoo uses hyphens for share-class suffixes (e.g. "BRK-B"); the
        # NASDAQ Trader files use dots (e.g. "BRK.B"). Best-effort conversion.
        symbol = str(symbol).replace(".", "-")
        exchange_label = _OTHER_LISTED_EXCHANGE_LABELS.get(str(row.get("Exchange")), "NYSE")
        results.append(StockResult(symbol=symbol, name=str(name), exchange=exchange_label))
    return results


class InternationalMarketProvider(MarketProvider):
    """US-listed equities (NASDAQ, NYSE, NYSE American, NYSE Arca)."""

    market_id = "international"
    market_label = "International Market"

    def search(self, query: str, limit: int = 20) -> list[StockResult]:
        return yahoo_finance_search(query, US_EXCHANGES, limit=limit)

    def list_directory(self) -> list[StockResult]:
        """The full NASDAQ + NYSE/other-listed symbol directory, fetched live.

        Returns an empty list (with a logged warning, not an exception) if
        both source files are unreachable -- the app falls back to
        search-only in that case, same as the Indian provider always does.
        """
        directory: list[StockResult] = []
        seen_symbols: set[str] = set()

        nasdaq_text = _fetch_text(NASDAQ_LISTED_URL)
        if nasdaq_text:
            for result in _parse_nasdaq_listed(nasdaq_text):
                if result.symbol not in seen_symbols:
                    seen_symbols.add(result.symbol)
                    directory.append(result)

        other_text = _fetch_text(OTHER_LISTED_URL)
        if other_text:
            for result in _parse_other_listed(other_text):
                if result.symbol not in seen_symbols:
                    seen_symbols.add(result.symbol)
                    directory.append(result)

        directory.sort(key=lambda r: r.name)
        logger.info("Built International market directory: %d stocks", len(directory))
        return directory
