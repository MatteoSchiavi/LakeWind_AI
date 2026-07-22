"""V5 weather utilities — WMO weather code decoding + weather features.

WMO weather interpretation codes (WW):
  0      Clear sky
  1-3    Mainly clear, partly cloudy, overcast
  45-48  Fog and depositing rime fog
  51-57  Drizzle: Light, moderate, and dense intensity
  61-67  Rain: Slight, moderate, heavy, freezing
  71-77  Snow fall: Slight, moderate, heavy
  80-82  Rain showers: Slight, moderate, violent
  85-86  Snow showers slight and heavy
  95     Thunderstorm: Slight or moderate
  96-99  Thunderstorm with slight and heavy hail
"""
from __future__ import annotations

WMO_CODES: dict[int, dict[str, str]] = {
    0:  {"en": "Clear sky",          "it": "Cielo sereno",            "icon": "☀️"},
    1:  {"en": "Mainly clear",       "it": "Prevalentemente sereno",  "icon": "🌤️"},
    2:  {"en": "Partly cloudy",      "it": "Parzialmente nuvoloso",   "icon": "⛅"},
    3:  {"en": "Overcast",           "it": "Coperto",                 "icon": "☁️"},
    45: {"en": "Fog",                "it": "Nebbia",                  "icon": "🌫️"},
    48: {"en": "Rime fog",           "it": "Nebbia con brina",        "icon": "🌫️"},
    51: {"en": "Light drizzle",      "it": "Pioviggine leggera",      "icon": "🌦️"},
    53: {"en": "Moderate drizzle",   "it": "Pioviggine moderata",     "icon": "🌦️"},
    55: {"en": "Dense drizzle",      "it": "Pioviggine densa",        "icon": "🌧️"},
    56: {"en": "Freezing drizzle",   "it": "Pioviggine gelata",       "icon": "🌧️"},
    57: {"en": "Dense freezing drizzle", "it": "Pioviggine gelata densa", "icon": "🌧️"},
    61: {"en": "Slight rain",        "it": "Pioggia leggera",         "icon": "🌦️"},
    63: {"en": "Moderate rain",      "it": "Pioggia moderata",        "icon": "🌧️"},
    65: {"en": "Heavy rain",         "it": "Pioggia battente",        "icon": "🌧️"},
    66: {"en": "Freezing rain",      "it": "Pioggia gelata",          "icon": "🌧️"},
    67: {"en": "Heavy freezing rain", "it": "Pioggia gelata battente", "icon": "🌧️"},
    71: {"en": "Slight snow",        "it": "Neve leggera",            "icon": "🌨️"},
    73: {"en": "Moderate snow",      "it": "Neve moderata",           "icon": "🌨️"},
    75: {"en": "Heavy snow",         "it": "Neve abbondante",         "icon": "❄️"},
    77: {"en": "Snow grains",        "it": "Granuli di neve",         "icon": "🌨️"},
    80: {"en": "Slight rain showers", "it": "Rovesci leggeri",        "icon": "🌦️"},
    81: {"en": "Moderate rain showers", "it": "Rovesci moderati",     "icon": "🌧️"},
    82: {"en": "Violent rain showers", "it": "Rovesci violenti",      "icon": "⛈️"},
    85: {"en": "Slight snow showers", "it": "Rovesci di neve leggeri", "icon": "🌨️"},
    86: {"en": "Heavy snow showers", "it": "Rovesci di neve abbondanti", "icon": "❄️"},
    95: {"en": "Thunderstorm",       "it": "Temporale",               "icon": "⛈️"},
    96: {"en": "Thunderstorm with hail", "it": "Temporale con grandine", "icon": "⛈️"},
    99: {"en": "Severe thunderstorm with hail", "it": "Temporale severo con grandine", "icon": "⛈️"},
}


def decode_weather_code(code: int | None, lang: str = "en") -> tuple[str, str]:
    """Decode a WMO weather code to (description, icon).

    Returns ("Unknown", "❓") for None or unrecognized codes.
    """
    if code is None:
        return ("Unknown", "❓")
    entry = WMO_CODES.get(int(code))
    if entry is None:
        return ("Unknown", "❓")
    return (entry.get(lang, entry["en"]), entry["icon"])


def is_rainy(code: int | None) -> bool:
    """True if the weather code indicates rain or drizzle."""
    if code is None:
        return False
    c = int(code)
    return c in range(51, 68) or c in range(80, 83)


def is_snowy(code: int | None) -> bool:
    """True if the weather code indicates snow."""
    if code is None:
        return False
    c = int(code)
    return c in range(71, 78) or c in range(85, 87)


def is_stormy(code: int | None) -> bool:
    """True if the weather code indicates thunderstorm."""
    if code is None:
        return False
    c = int(code)
    return c in range(95, 100)


def is_foggy(code: int | None) -> bool:
    """True if the weather code indicates fog."""
    if code is None:
        return False
    c = int(code)
    return c in (45, 48)


def sailing_weather_warning(code: int | None, wind_speed_kn: float, visibility_m: float | None = None) -> str | None:
    """Return a sailing safety warning string, or None if conditions are safe.

    Checks:
    - Thunderstorm → "Seek shelter"
    - Heavy rain + wind > 15kn → "Reduced visibility, difficult sailing"
    - Snow → "Freezing conditions, ice on deck risk"
    - Fog + visibility < 1000m → "Dangerous visibility, stay near shore"
    - UV index > 7 → "High UV, wear sunscreen"
    """
    if is_stormy(code):
        return "⛈️ Thunderstorm — seek shelter immediately"
    if is_snowy(code):
        return "❄️ Snow — freezing conditions, ice on deck risk"
    if is_foggy(code) or (visibility_m is not None and visibility_m < 1000):
        return "🌫️ Fog — dangerous visibility, stay near shore"
    if is_rainy(code) and wind_speed_kn > 15:
        return "🌧️ Heavy rain + strong wind — reduced visibility, difficult sailing"
    return None


__all__ = [
    "decode_weather_code",
    "is_rainy", "is_snowy", "is_stormy", "is_foggy",
    "sailing_weather_warning",
    "WMO_CODES",
]
