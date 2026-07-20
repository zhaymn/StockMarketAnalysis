"""Application configuration, loaded from environment / `.env`.

Secrets are never hardcoded and never sent to the frontend. The API exposes
only *availability* of each integration (see `IntegrationStatus`), so the
dashboard can render an honest "NOT CONFIGURED" state without ever seeing a key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-driven settings. See `.env.example` for documentation."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API credentials (empty string == not configured) ------------------
    marketaux_api_key: str = ""
    fred_api_key: str = ""
    anthropic_api_key: str = ""

    # --- Application -------------------------------------------------------
    stockintel_env: str = "development"
    log_level: str = "INFO"
    frontend_origin: str = "http://localhost:3000"

    # --- Cache -------------------------------------------------------------
    cache_dir: Path = Path(".cache")
    cache_ttl_intraday: int = 60
    cache_ttl_daily_bars: int = 3600
    cache_ttl_news: int = 900
    cache_ttl_fundamentals: int = 86_400
    cache_ttl_sentiment: int = 604_800

    # --- Model artifacts ---------------------------------------------------
    artifact_dir: Path = Path(".artifacts")
    random_seed: int = Field(
        default=42,
        description="Seeds numpy/torch/sklearn so a reported metric is reproducible.",
    )

    @property
    def resolved_cache_dir(self) -> Path:
        """Cache directory as an absolute path, created on first access."""
        path = self.cache_dir
        if not path.is_absolute():
            path = BACKEND_DIR / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def resolved_artifact_dir(self) -> Path:
        """Model-artifact directory as an absolute path, created on first access."""
        path = self.artifact_dir
        if not path.is_absolute():
            path = BACKEND_DIR / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    # --- Integration availability -----------------------------------------
    @property
    def has_news_provider(self) -> bool:
        return bool(self.marketaux_api_key.strip())

    @property
    def has_macro_provider(self) -> bool:
        return bool(self.fred_api_key.strip())

    @property
    def has_llm(self) -> bool:
        return bool(self.anthropic_api_key.strip())

    def integration_status(self) -> dict[str, dict[str, object]]:
        """Per-integration availability, safe to serialise to the frontend.

        Contains no key material -- only whether each integration is usable,
        and if not, exactly what the user must do about it.
        """
        return {
            "market_data": {
                "configured": True,
                "provider": "Yahoo Finance (yfinance)",
                "requires_key": False,
                "note": "End-of-day and delayed intraday data. No API key required.",
            },
            "news": {
                "configured": self.has_news_provider,
                "provider": "Marketaux",
                "requires_key": True,
                "env_var": "MARKETAUX_API_KEY",
                "obtain_at": "https://www.marketaux.com/",
                "free_tier": "100 requests/day",
                "note": (
                    "Required for news, news sentiment and event-impact analysis. "
                    "Without it those sections report NEWS API NOT CONFIGURED."
                ),
            },
            "macro": {
                "configured": self.has_macro_provider,
                "provider": "FRED (St. Louis Fed)",
                "requires_key": True,
                "env_var": "FRED_API_KEY",
                "obtain_at": "https://fred.stlouisfed.org/docs/api/api_key.html",
                "free_tier": "Free, no published quota",
                "note": "Required for macro/current-affairs signals.",
            },
            "sentiment": {
                "configured": True,
                "provider": "FinBERT (ProsusAI/finbert), run locally",
                "requires_key": False,
                "note": (
                    "No API key required. First run downloads ~440MB of model "
                    "weights from Hugging Face."
                ),
            },
            "llm_narrative": {
                "configured": self.has_llm,
                "provider": "Anthropic",
                "requires_key": True,
                "env_var": "ANTHROPIC_API_KEY",
                "obtain_at": "https://console.anthropic.com/",
                "free_tier": "No free tier",
                "note": (
                    "Optional. Only adds prose event-impact summaries; the "
                    "rule-based relevance and impact engine runs without it."
                ),
            },
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton settings instance."""
    return Settings()
