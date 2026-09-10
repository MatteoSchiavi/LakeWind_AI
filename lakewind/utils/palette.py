"""Single source of truth for wind-speed visual encoding (Phase 4 / W5).

Phase 4 audit finding F4: three divergent speed palettes coexisted
(web page.tsx, WindMap.tsx Spectral, heatmap_v3) plus a fourth emoji
mapping in the bot — none anchored to the operationally meaningful
thresholds. Every surface now imports from THIS module:

    <5 kn  calm      (blue)    — not sailable
    5-8   light      (teal)    — marginal
    8-12  sailable   (green)   — the Breva sweet spot (GO threshold = 8)
    12-16 strong     (amber)   — powered sailing, reef territory
    16+   extreme    (red)     — small-craft caution

The 8/12 kn anchors match the decision-precision metrics (Audit R9:
P(>=8 kn) / P(>=12 kn)) and the shared sailing decision module
(`lakewind/prediction/decision.py`), so "green on the map" means
exactly "P(>=8 kn) territory".

Consumers:
  - web:    web-ui/src/lib/palette.ts (mirrored constants + snapshot test)
  - maps:   lakewind/utils/heatmap_v3.py (colormap built from SPEED_COLORS)
  - bot:    _speed_color_emoji / bars (BAND_EMOJI imported from here)

Colors were chosen for contrast on BOTH light and dark backgrounds
(they are mid-luminance, saturated hues; none rely on pure white/black).
"""
from __future__ import annotations

# Band upper bounds in knots. BAND[i] covers (BREAKS[i-1], BREAKS[i]] with
# BREAKS[-1] treated as 0 and the last band open-ended.
SPEED_BREAKS_KN: tuple[float, ...] = (5.0, 8.0, 12.0, 16.0)

BAND_NAMES_EN: tuple[str, ...] = ("calm", "light", "sailable", "strong", "extreme")
BAND_NAMES_IT: tuple[str, ...] = ("calma", "leggero", "navigabile", "forte", "estremo")

# One hex per band: calm / light / sailable / strong / extreme.
SPEED_COLORS: tuple[str, ...] = (
    "#3b82f6",  # blue
    "#06b6d4",  # teal
    "#22c55e",  # green
    "#f59e0b",  # amber
    "#dc2626",  # red
)

BAND_EMOJI: tuple[str, ...] = ("⚪", "🔵", "🟢", "🟠", "🔴")


def band_index(speed_kn: float) -> int:
    """Index of the band a speed falls into (0..4).

    Bands are open on the left: a speed >= the last break (16 kn) belongs
    to the open-ended "extreme" band — i.e. len(SPEED_BREAKS_KN), NOT
    len()-1 (that would make the top band unreachable).
    """
    if speed_kn is None:
        return 0
    for i, upper in enumerate(SPEED_BREAKS_KN):
        if speed_kn < upper:
            return i
    return len(SPEED_BREAKS_KN)  # open-ended top band (>= 16 kn)


def speed_color_kn(speed_kn: float) -> str:
    """Hex color for a wind speed in knots."""
    return SPEED_COLORS[band_index(speed_kn)]


def speed_label_kn(speed_kn: float, lang: str = "en") -> str:
    """Human-readable band label (en/it)."""
    names = BAND_NAMES_IT if lang == "it" else BAND_NAMES_EN
    return names[band_index(speed_kn)]


def speed_emoji_kn(speed_kn: float) -> str:
    """Bot emoji for a wind speed (replaces the bot's private 4th mapping)."""
    return BAND_EMOJI[band_index(speed_kn)]


__all__ = [
    "SPEED_BREAKS_KN",
    "SPEED_COLORS",
    "BAND_EMOJI",
    "BAND_NAMES_EN",
    "BAND_NAMES_IT",
    "band_index",
    "speed_color_kn",
    "speed_label_kn",
    "speed_emoji_kn",
]
