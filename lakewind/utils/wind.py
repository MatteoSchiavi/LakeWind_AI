"""Wind U/V <-> speed/direction conversions.

Spec §6 target definition:
    target_u = observed_u - forecast_u
    target_v = observed_v - forecast_v
    final_prediction = forecast + predicted_bias

Conventions (meteorological):
- Direction is the direction the wind is COMING FROM, in degrees clockwise from north.
- U is the east-west component (positive = wind blowing toward east).
- V is the north-south component (positive = wind blowing toward north).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WindVector:
    speed_kn: float
    direction_deg: float  # 0..360, meteorological (where wind comes FROM)

    def to_uv(self) -> tuple[float, float]:
        """Convert speed/direction to (u, v) components.

        Meteorological convention: wind FROM direction θ (clockwise from north).
        u = -speed * sin(θ), v = -speed * cos(θ)
        Wind FROM north (θ=0): u=0, v=-speed (blows south)
        Wind FROM east (θ=90): u=-speed, v=0 (blows west)
        """
        theta_rad = math.radians(self.direction_deg)
        u = -self.speed_kn * math.sin(theta_rad)
        v = -self.speed_kn * math.cos(theta_rad)
        return u, v

    @classmethod
    def from_uv(cls, u: float, v: float) -> WindVector:
        """Construct from (u, v) components."""
        speed = math.hypot(u, v)
        if speed < 1e-6:
            return cls(speed_kn=0.0, direction_deg=0.0)
        # Inverse of the above:
        # u = -s*sin(θ), v = -s*cos(θ)  =>  θ = atan2(-u, -v) mod 360
        theta_rad = math.atan2(-u, -v)
        direction_deg = (math.degrees(theta_rad) + 360.0) % 360.0
        return cls(speed_kn=speed, direction_deg=direction_deg)


def circular_direction_error_deg(pred_deg: float, obs_deg: float) -> float:
    """Smallest absolute difference between two wind directions, in degrees (0..180)."""
    diff = (pred_deg - obs_deg + 180.0) % 360.0 - 180.0
    return abs(diff)


def bias_correct(
    forecast_u: float, forecast_v: float, bias_u: float, bias_v: float
) -> WindVector:
    """Spec §6: final_prediction = forecast + predicted_bias (in U/V space)."""
    return WindVector.from_uv(forecast_u + bias_u, forecast_v + bias_v)


__all__ = ["WindVector", "circular_direction_error_deg", "bias_correct"]
