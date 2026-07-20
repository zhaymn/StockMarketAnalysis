"""Metadata: integration status, available models and horizons."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.features.targets import AVAILABLE_TARGETS
from app.services.prediction import MODEL_MODES

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/integrations")
async def integrations() -> dict[str, object]:
    """Which integrations are configured.

    Contains no key material — only availability, and for anything missing,
    exactly what the user must do. Drives the NOT CONFIGURED states in the UI.
    """
    return {"integrations": get_settings().integration_status()}


@router.get("/models")
async def models() -> dict[str, object]:
    """Selectable prediction modes and horizons."""
    return {
        "modes": [
            {"id": mode_id, **details} for mode_id, details in MODEL_MODES.items()
        ],
        "default_mode": "most_possible",
        "targets": [spec.to_dict() for spec in AVAILABLE_TARGETS.values()],
        "default_target": "outlook_5d",
        "disclaimer": (
            "Predictions are statistical estimates derived from historical data. "
            "They are not guarantees, not financial advice, and should never be the "
            "sole basis for an investment decision."
        ),
    }
