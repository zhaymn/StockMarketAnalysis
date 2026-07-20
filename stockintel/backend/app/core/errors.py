"""Typed application errors.

Each maps to an explicit, honest UI state. The rule throughout this codebase:
when data is unavailable we raise one of these and the frontend renders
DATA UNAVAILABLE / NOT CONFIGURED -- we never substitute synthetic values.
"""

from __future__ import annotations


class StockIntelError(Exception):
    """Base class for all application errors."""

    #: Stable machine-readable code, surfaced to the frontend.
    code: str = "internal_error"
    #: HTTP status the API layer should use.
    http_status: int = 500

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"code": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = self.detail
        return payload


class UnknownTickerError(StockIntelError):
    """The requested symbol could not be resolved in the selected market."""

    code = "unknown_ticker"
    http_status = 404


class DataUnavailableError(StockIntelError):
    """A data source responded, but had no usable data for this request."""

    code = "data_unavailable"
    http_status = 404


class InsufficientHistoryError(StockIntelError):
    """Fewer usable bars than the model requires.

    Raised rather than silently training on a short series, which would
    produce metrics that look real but mean nothing.
    """

    code = "insufficient_history"
    http_status = 422

    def __init__(self, message: str, *, required: int, available: int) -> None:
        super().__init__(
            message,
            detail=f"Requires at least {required} trading days; {available} available.",
        )
        self.required = required
        self.available = available


class ProviderError(StockIntelError):
    """An upstream data provider failed (network, 5xx, malformed response)."""

    code = "provider_error"
    http_status = 502


class RateLimitedError(ProviderError):
    """An upstream provider rate-limited us."""

    code = "rate_limited"
    http_status = 429

    def __init__(self, message: str, *, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class IntegrationNotConfiguredError(StockIntelError):
    """A feature was requested whose API key is absent.

    Carries everything the UI needs to tell the user how to fix it.
    """

    code = "not_configured"
    http_status = 503

    def __init__(
        self,
        integration: str,
        *,
        env_var: str,
        obtain_at: str,
        reason: str,
    ) -> None:
        super().__init__(
            f"{integration} is not configured.",
            detail=f"Set {env_var} in stockintel/backend/.env -- obtain a key at {obtain_at}.",
        )
        self.integration = integration
        self.env_var = env_var
        self.obtain_at = obtain_at
        self.reason = reason

    def to_payload(self) -> dict[str, object]:
        payload = super().to_payload()
        payload.update(
            integration=self.integration,
            env_var=self.env_var,
            obtain_at=self.obtain_at,
            reason=self.reason,
        )
        return payload


class ModelNotTrainedError(StockIntelError):
    """A prediction was requested before a model artifact existed for it."""

    code = "model_not_trained"
    http_status = 409
