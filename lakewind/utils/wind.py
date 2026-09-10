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


def circular_mean_deg(degs: list[float]) -> float | None:
    """Directional (circular) mean of directions in degrees.

    Arithmetic means of degrees are wrong across the 350/10 wraparound
    (350, 10 would average to 180). This computes the mean via the unit-vector
    resultant, which is the correct aggregate for wind direction members
    (Deep Audit 3.7).
    """
    if not degs:
        return None
    rads = [math.radians(d) for d in degs]
    sin_mean = sum(math.sin(r) for r in rads) / len(rads)
    cos_mean = sum(math.cos(r) for r in rads) / len(rads)
    if math.hypot(sin_mean, cos_mean) < 1e-9:
        # Vectors perfectly cancel (e.g. 0 and 180) — direction undefined.
        return None
    return (math.degrees(math.atan2(sin_mean, cos_mean)) + 360.0) % 360.0


def circular_std_deg(degs: list[float]) -> float | None:
    """Circular standard deviation in degrees: sqrt(-2 ln R).

    R is the mean resultant length; R=1 (all members agree) -> 0 deg,
    R->0 (uniform spread) -> sqrt(-2 ln eps) ~ 551 deg is unbounded, so the
    caller-facing value is capped by the formula's natural behaviour and is
    monotonically increasing with dispersion. Returns None for empty input.
    """
    if not degs:
        return None
    rads = [math.radians(d) for d in degs]
    sin_mean = sum(math.sin(r) for r in rads) / len(rads)
    cos_mean = sum(math.cos(r) for r in rads) / len(rads)
    r_length = math.hypot(sin_mean, cos_mean)
    r_clamped = min(1.0, max(1e-9, r_length))
    return math.degrees(math.sqrt(-2.0 * math.log(r_clamped)))


def circular_spread_deg(degs: list[float]) -> float | None:
    """Max angular deviation of any member from the circular mean (0..180)."""
    mean = circular_mean_deg(degs)
    if mean is None or not degs:
        return None
    return max(circular_direction_error_deg(d, mean) for d in degs)


def bias_correct(
    forecast_u: float, forecast_v: float, bias_u: float, bias_v: float
) -> WindVector:
    """Spec §6: final_prediction = forecast + predicted_bias (in U/V space)."""
    return WindVector.from_uv(forecast_u + bias_u, forecast_v + bias_v)


__all__ = [
    "WindVector",
    "circular_direction_error_deg",
    "circular_mean_deg",
    "circular_std_deg",
    "circular_spread_deg",
    "bias_correct",
]
