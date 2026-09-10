"""Shared sailing-decision module (Phase 4 / W2).

Phase 4 audit finding F3: the product's core question — "is there wind at
the corridor this afternoon?" — was answered only by the bot's /sailing,
with the decision logic locked inside a Telegram formatting function. The
event-probability machinery validated by the metrics layer (Audit R9:
P(>=8 kn) / P(>=12 kn) with Brier/reliability scoring) was not surfaced
anywhere user-facing.

This module is now the SINGLE source of truth for decision math, consumed
by all three surfaces:

    - bot:      _sailing_recommendation (telegram_bot.py)
    - API:      GET /api/decision (api.py)
    - web:      "Go sailing?" hero card (via the /api/decision proxy)

Probability model
-----------------
The Phase 3 model predicts quantiles and the R4 conformal layer calibrates
the 80% band (q10/q90) before persistence, so a stored row carries a
calibrated uncertainty estimate. We convert the band into per-threshold
event probabilities by treating it as a Gaussian central interval:

    sigma = (q90 - q10) / (2 * z_0.90),  z_0.90 = 1.28155
    P(speed >= t) = 0.5 * erfc((t - q50) / (sigma * sqrt(2)))

This is the same shape assumption the reliability diagrams in R9 validate
empirically — the calibration guarantee comes from conformal prediction,
not from the Gaussian form. Fallbacks, in order:
  - no band but expected_error_kn present: sigma = expected_error_kn
    (the half-width of the 90-10 interval — a conservative approximation);
  - neither: the probability is a step function of the median
    (1.0 if q50 >= threshold else 0.0) — deterministic, honest about it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# 80% central interval -> z of the 0.90 quantile.
_Z_80 = 1.2815518105677815

# Decision thresholds (kn) — the SAME anchors as the design palette
# (lakewind/utils/palette.py: 8 = sailable band lower bound, 12 = strong)
# and the R9 decision-precision metrics. Changing them here changes every
# surface at once.
GO_THRESHOLD_KN = 8.0
STRONG_THRESHOLD_KN = 12.0

# The verdict needs at least this many hours inside the window with
# P(>= GO threshold) >= 0.5 to say "GO" — mirrors the bot's historical
# "sail_hours >= 2" rule so the migration is behaviour-preserving.
MIN_GO_HOURS = 2


def _norm_sf(x: float) -> float:
    """Standard normal survival function P(Z >= x) via erfc (no scipy dep)."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _row_float(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def band_sigma(row: dict[str, Any]) -> float | None:
    """Calibrated per-row sigma (kn) from the 80% conformal band."""
    q10 = _row_float(row, "wind_speed_q10_kn")
    q90 = _row_float(row, "wind_speed_q90_kn")
    if q10 is not None and q90 is not None and q90 >= q10:
        return (q90 - q10) / (2.0 * _Z_80)
    err = _row_float(row, "expected_error_kn")
    if err is not None and err > 0:
        return err
    return None


def probability_at_least(row: dict[str, Any], threshold_kn: float) -> float:
    """P(wind_speed >= threshold_kn) for one prediction row."""
    q50 = _row_float(row, "wind_speed_kn")
    if q50 is None:
        return 0.0
    sigma = band_sigma(row)
    if sigma is None:
        return 1.0 if q50 >= threshold_kn else 0.0
    return max(0.0, min(1.0, _norm_sf((threshold_kn - q50) / sigma)))


@dataclass(frozen=True)
class DecisionHour:
    """One hour of the decision window."""

    hour: int | None              # local hour when derivable, else None
    valid_time: datetime | None
    speed_kn: float               # median (q50)
    q10_kn: float | None
    q90_kn: float | None
    dir_deg: float | None
    p_go: float                   # P(>= 8 kn)
    p_strong: float               # P(>= 12 kn)
    regime: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hour": self.hour,
            "valid_time": self.valid_time.isoformat() if self.valid_time else None,
            "speed_kn": round(self.speed_kn, 2),
            "q10_kn": round(self.q10_kn, 2) if self.q10_kn is not None else None,
            "q90_kn": round(self.q90_kn, 2) if self.q90_kn is not None else None,
            "dir_deg": round(self.dir_deg, 1) if self.dir_deg is not None else None,
            "p_go": round(self.p_go, 3),
            "p_strong": round(self.p_strong, 3),
            "regime": self.regime,
        }


@dataclass(frozen=True)
class SailingDecision:
    """Verdict for one decision window (single point or multi-point best)."""

    verdict: str                                  # "go" | "marginal" | "no_go"
    hours: list[DecisionHour] = field(default_factory=list)
    best_hour: int | None = None                  # local hour of best p_go
    best_speed_kn: float | None = None            # median at best hour
    n_go_hours: int = 0                           # hours with p_go >= 0.5
    peak_p_go: float = 0.0
    regime: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "best_hour": self.best_hour,
            "best_speed_kn": self.best_speed_kn,
            "n_go_hours": self.n_go_hours,
            "peak_p_go": round(self.peak_p_go, 3),
            "regime": self.regime,
            "hours": [h.to_dict() for h in self.hours],
        }


def _row_regime(row: dict[str, Any]) -> str | None:
    v = row.get("regime")
    return str(v) if v else None


def _row_hour(row: dict[str, Any], tz) -> int | None:
    """Local hour of a row's valid_time when parseable, else None.

    Stored valid_times are naive UTC (DuckDB convention) — attach UTC
    explicitly before converting to the target zone.
    """
    vt = row.get("valid_time")
    if vt is None:
        return None
    if isinstance(vt, str):
        try:
            vt = datetime.fromisoformat(vt)
        except ValueError:
            return None
    if not isinstance(vt, datetime):
        return None
    if vt.tzinfo is None:
        vt = vt.replace(tzinfo=UTC)
    try:
        return vt.astimezone(tz).hour
    except (ValueError, OSError, OverflowError):
        return None


def compute_decision(
    rows: list[dict[str, Any]],
    *,
    go_threshold_kn: float = GO_THRESHOLD_KN,
    strong_threshold_kn: float = STRONG_THRESHOLD_KN,
    min_go_hours: int = MIN_GO_HOURS,
    tz=None,
) -> SailingDecision:
    """Compute the GO/MARGINAL/NO-GO verdict from prediction rows.

    `rows` are forecast-store rows (dicts with wind_speed_kn and, ideally,
    the Phase 4 band columns). Verdict logic mirrors the bot's historical
    rule, generalised from raw speed counts to calibrated probabilities:
      - "go"      : at least `min_go_hours` hours with p_go >= 0.5
      - "marginal": at least one hour with p_go >= 0.5
      - "no_go"   : otherwise
    The best hour maximises p_go, tie-broken by higher median speed.
    """
    hours: list[DecisionHour] = []
    for row in rows:
        speed = _row_float(row, "wind_speed_kn")
        if speed is None:
            continue
        q10 = _row_float(row, "wind_speed_q10_kn")
        q90 = _row_float(row, "wind_speed_q90_kn")
        hour = _row_hour(row, tz) if tz is not None else None
        hours.append(
            DecisionHour(
                hour=hour,
                valid_time=_as_dt(row.get("valid_time")),
                speed_kn=speed,
                q10_kn=q10,
                q90_kn=q90,
                dir_deg=_row_float(row, "wind_dir_deg"),
                p_go=probability_at_least(row, go_threshold_kn),
                p_strong=probability_at_least(row, strong_threshold_kn),
                regime=_row_regime(row),
            )
        )

    if not hours:
        return SailingDecision(verdict="no_go")

    best = max(hours, key=lambda h: (h.p_go, h.speed_kn))
    n_go = sum(1 for h in hours if h.p_go >= 0.5)
    if n_go >= min_go_hours:
        verdict = "go"
    elif n_go >= 1:
        verdict = "marginal"
    else:
        verdict = "no_go"

    # Dominant regime of the window = most frequent non-null label.
    regimes = [h.regime for h in hours if h.regime]
    regime = max(set(regimes), key=regimes.count) if regimes else None

    return SailingDecision(
        verdict=verdict,
        hours=hours,
        best_hour=best.hour,
        best_speed_kn=best.speed_kn,
        n_go_hours=n_go,
        peak_p_go=max(h.p_go for h in hours),
        regime=regime,
    )


def _as_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


__all__ = [
    "GO_THRESHOLD_KN",
    "STRONG_THRESHOLD_KN",
    "MIN_GO_HOURS",
    "DecisionHour",
    "SailingDecision",
    "band_sigma",
    "compute_decision",
    "probability_at_least",
]
