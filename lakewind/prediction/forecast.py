"""Frozen Forecast dataclass (Spec §8).

Spec §8:
    @dataclass(frozen=True)
    class Forecast:
        generated_at: datetime
        valid_time: datetime
        point_id: str
        wind_speed_kn: float
        wind_dir_deg: float
        wind_gust_kn: float
        confidence_pct: float
        expected_error_kn: float
        model_version: str
        top_contributors: list[tuple[str, float]]   # SHAP-derived, for explanation text
        diagnostics: dict
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Forecast:
    generated_at: datetime
    valid_time: datetime
    point_id: str
    wind_speed_kn: float
    wind_dir_deg: float
    wind_gust_kn: float
    confidence_pct: float
    expected_error_kn: float
    model_version: str
    top_contributors: list[tuple[str, float]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Phase 4 (W1): calibrated 80% band + regime, persisted with the row so
    # every surface (web/bot/API) can render uncertainty. Optional with
    # defaults so historic call sites constructing a Forecast without them
    # stay valid.
    wind_speed_q10_kn: float | None = None
    wind_speed_q90_kn: float | None = None
    regime: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "valid_time": self.valid_time.isoformat(),
            "point_id": self.point_id,
            "wind_speed_kn": self.wind_speed_kn,
            "wind_dir_deg": self.wind_dir_deg,
            "wind_gust_kn": self.wind_gust_kn,
            "confidence_pct": self.confidence_pct,
            "expected_error_kn": self.expected_error_kn,
            "model_version": self.model_version,
            "top_contributors": self.top_contributors,
            "diagnostics": self.diagnostics,
            "wind_speed_q10_kn": self.wind_speed_q10_kn,
            "wind_speed_q90_kn": self.wind_speed_q90_kn,
            "regime": self.regime,
        }


__all__ = ["Forecast"]
