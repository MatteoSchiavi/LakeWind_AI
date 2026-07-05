"""V2 Kalman filter for online short-range bias correction.

Spec §7 V2: at horizons <2h, persistence is the strongest predictor. A Kalman
filter recursively updates the bias estimate as new observations arrive,
giving us a near-zero-lag correction that complements the (longer-horizon)
LightGBM/XGBoost model.

The implementation is the classic 2D state (bias_u, bias_v) with a 2x2
covariance matrix. State transition is identity (random walk), measurement
matrix is identity (we observe bias = obs - forecast directly).

Math:
    Predict:
        x_{k|k-1} = x_{k-1|k-1}        (random walk)
        P_{k|k-1} = P_{k-1|k-1} + Q
    Update:
        K_k = P_{k|k-1} (P_{k|k-1} + R)^{-1}
        x_{k|k} = x_{k|k-1} + K_k (z_k - x_{k|k-1})
        P_{k|k} = (I - K_k) P_{k|k-1}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from lakewind.db import access
from lakewind.utils.wind import WindVector

logger = logging.getLogger(__name__)

# Default noise parameters (tunable)
DEFAULT_Q = 0.01  # process noise variance (per hour)
DEFAULT_R = 0.5   # measurement noise variance
MIN_OBSERVATION_GAP_MIN = 30  # ignore observations older than this


@dataclass
class KalmanState:
    point_id: str
    bias_u: float
    bias_v: float
    p_uu: float
    p_vv: float
    p_uv: float
    last_update: datetime


def get_state(point_id: str) -> KalmanState:
    """Load Kalman state from DB (or initialize if not present)."""
    with access.cursor() as conn:
        cur = conn.execute(
            "SELECT point_id, bias_u, bias_v, p_uu, p_vv, p_uv, last_update "
            "FROM v2_kalman_state WHERE point_id = ?",
            [point_id],
        )
        row = cur.fetchone()
    if row is None:
        return KalmanState(
            point_id=point_id,
            bias_u=0.0, bias_v=0.0,
            p_uu=1.0, p_vv=1.0, p_uv=0.0,
            last_update=datetime.utcnow() - timedelta(hours=24),
        )
    return KalmanState(
        point_id=row[0], bias_u=row[1], bias_v=row[2],
        p_uu=row[3], p_vv=row[4], p_uv=row[5],
        last_update=row[6] if isinstance(row[6], datetime) else datetime.utcnow(),
    )


def save_state(state: KalmanState) -> None:
    """Persist Kalman state to DB."""
    with access.cursor() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO v2_kalman_state
            (point_id, bias_u, bias_v, p_uu, p_vv, p_uv, q, r, last_update)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [state.point_id, state.bias_u, state.bias_v,
             state.p_uu, state.p_vv, state.p_uv,
             DEFAULT_Q, DEFAULT_R, datetime.utcnow()],
        )


def update(point_id: str, observation_u: float, observation_v: float,
           forecast_u: float, forecast_v: float,
           observation_time: Optional[datetime] = None) -> KalmanState:
    """Update Kalman state with a new (observation, forecast) pair.

    The "innovation" is z = observation - forecast (the actual bias observed).
    """
    obs_time = observation_time or datetime.utcnow()
    state = get_state(point_id)

    # Time step (in hours) — scale process noise by elapsed time
    dt_hours = max(0.0, (obs_time - state.last_update).total_seconds() / 3600.0)
    if dt_hours < 0 or dt_hours > 48:
        # Reset on long gap (state is stale)
        state = KalmanState(
            point_id=point_id, bias_u=0.0, bias_v=0.0,
            p_uu=1.0, p_vv=1.0, p_uv=0.0, last_update=obs_time,
        )
        dt_hours = 0.0

    # --- Predict step ---
    Q = DEFAULT_Q * max(dt_hours, 0.1)  # variance scales with time
    P = np.array([[state.p_uu, state.p_uv], [state.p_uv, state.p_vv]])
    P_pred = P + Q * np.eye(2)
    x_pred = np.array([state.bias_u, state.bias_v])

    # --- Update step ---
    R = DEFAULT_R * np.eye(2)
    z = np.array([observation_u - forecast_u, observation_v - forecast_v])
    innovation = z - x_pred
    S = P_pred + R  # 2x2
    K = P_pred @ np.linalg.inv(S)  # 2x2
    x_new = x_pred + K @ innovation
    P_new = (np.eye(2) - K) @ P_pred

    new_state = KalmanState(
        point_id=point_id,
        bias_u=float(x_new[0]),
        bias_v=float(x_new[1]),
        p_uu=float(P_new[0, 0]),
        p_vv=float(P_new[1, 1]),
        p_uv=float(P_new[0, 1]),
        last_update=obs_time,
    )
    save_state(new_state)
    return new_state


def update_from_latest_observations(point_id: str) -> Optional[KalmanState]:
    """Find the most recent observation near `point_id` and update Kalman state.

    Returns the new state, or None if no recent observation exists.
    """
    from lakewind.config import load_settings
    s = load_settings()
    vp = next((p for p in s.virtual_points if p.id == point_id), None)
    if vp is None:
        return None

    # Find the most recent observation within the last 3h
    now = datetime.utcnow()
    obs = access.fetch_latest_observation_near(vp.lat, vp.lon, now, max_age_minutes=180)
    if not obs:
        return None

    # Pick the closest observation
    import math
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2.0)**2
        return 2.0 * R * math.asin(math.sqrt(a))

    best = min(obs, key=lambda o: _haversine(vp.lat, vp.lon, o.get("lat") or 0.0, o.get("lon") or 0.0))
    if not best.get("wind_speed_kn") or not best.get("wind_dir_deg"):
        return None
    if not best.get("timestamp"):
        return None

    obs_time = best["timestamp"]
    if isinstance(obs_time, str):
        try:
            obs_time = datetime.fromisoformat(obs_time)
        except Exception:
            return None

    # Get the forecast valid at observation time
    forecasts = access.fetch_forecasts_at(point_id, obs_time, lead_minutes_window=120)
    if not forecasts:
        return None
    # Use icon_eu as reference (or first available)
    ref = next((f for f in forecasts if f["model_name"] == "icon_eu"), None) or forecasts[0]
    fc_speed = ref.get("wind_speed_kn") or 0.0
    fc_dir = ref.get("wind_dir_deg") or 0.0
    if fc_speed < 0.1:
        return None

    fc_u, fc_v = WindVector(speed_kn=fc_speed, direction_deg=fc_dir).to_uv()
    obs_u, obs_v = WindVector(
        speed_kn=best["wind_speed_kn"],
        direction_deg=best["wind_dir_deg"],
    ).to_uv()

    return update(point_id, obs_u, obs_v, fc_u, fc_v, obs_time)


def predict_bias(point_id: str) -> tuple[float, float]:
    """Return the current Kalman bias estimate (bias_u, bias_v) for a point."""
    state = get_state(point_id)
    return state.bias_u, state.bias_v


def predict_bias_with_confidence(point_id: str) -> tuple[float, float, float]:
    """Return (bias_u, bias_v, confidence) where confidence is 1 - sqrt(trace(P))."""
    state = get_state(point_id)
    trace_p = state.p_uu + state.p_vv
    # Confidence: high when covariance is low. Map cov -> [0, 1]
    confidence = max(0.0, min(1.0, 1.0 - min(1.0, trace_p / 4.0)))
    return state.bias_u, state.bias_v, confidence


__all__ = [
    "KalmanState",
    "get_state",
    "save_state",
    "update",
    "update_from_latest_observations",
    "predict_bias",
    "predict_bias_with_confidence",
]
