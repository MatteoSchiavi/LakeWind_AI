"""Solar geometry helpers (Spec §4.4).

Uses `astral` for sunrise/sunset/elevation/azimuth. Lake Como location is
configurable; default to the operating area centroid.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from astral import LocationInfo
from astral.sun import sun

from lakewind.config import load_settings


@dataclass(frozen=True)
class SolarState:
    elevation_deg: float
    azimuth_deg: float
    sunrise: datetime | None
    sunset: datetime | None
    minutes_since_sunrise: float | None
    minutes_until_sunset: float | None
    day_length_minutes: float | None
    is_daytime: bool


def _location_info(lat: float, lon: float, tz_name: str) -> LocationInfo:
    return LocationInfo("LakeComo", "Italy", tz_name, lat, lon)


def solar_state_at(
    lat: float, lon: float, dt: datetime, tz_name: str = "Europe/Rome"
) -> SolarState:
    """Compute solar geometry for a given point in time.

    `dt` is assumed timezone-aware. If naive, local tz_name is applied.
    """
    tz = ZoneInfo(tz_name)
    # CRITICAL FIX: naive datetimes in this codebase are UTC (from collectors
    # that store UTC timestamps). Treat them as UTC, then convert to local.
    # The original code treated naive as local time, shifting all solar/Breva/
    # Tivano features by 1-2 hours.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    dt = dt.astimezone(tz)

    loc = _location_info(lat, lon, tz_name)

    # astral.sun.sun returns tz-aware datetimes in the location's timezone
    s = sun(loc.observer, date=dt.date(), tzinfo=tz)
    sunrise = s["sunrise"]
    sunset = s["sunset"]

    # Elevation/azimuth via astral.sun — use the underlying Observer + datetime
    from astral.sun import azimuth as _az
    from astral.sun import elevation as _el

    el_deg = float(_el(loc.observer, dt))
    az_deg = float(_az(loc.observer, dt))

    minutes_since = None
    minutes_until = None
    day_length = None
    is_daytime = False
    if sunrise is not None and sunset is not None:
        day_length = (sunset - sunrise).total_seconds() / 60.0
        if sunrise <= dt <= sunset:
            is_daytime = True
            minutes_since = (dt - sunrise).total_seconds() / 60.0
            minutes_until = (sunset - dt).total_seconds() / 60.0

    return SolarState(
        elevation_deg=el_deg,
        azimuth_deg=az_deg,
        sunrise=sunrise,
        sunset=sunset,
        minutes_since_sunrise=minutes_since,
        minutes_until_sunset=minutes_until,
        day_length_minutes=day_length,
        is_daytime=is_daytime,
    )


def operating_area_centroid() -> tuple[float, float]:
    s = load_settings()
    lat = (s.operating_area.lat_min + s.operating_area.lat_max) / 2.0
    lon = (s.operating_area.lon_min + s.operating_area.lon_max) / 2.0
    return lat, lon


def parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(hour=int(h), minute=int(m))


__all__ = ["SolarState", "solar_state_at", "operating_area_centroid", "parse_hhmm"]
