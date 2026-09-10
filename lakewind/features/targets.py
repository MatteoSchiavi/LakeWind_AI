"""Ground-truth hierarchy (Deep Audit R2 — the audit's most consequential fix).

Problem (audit 3.3): the target of the MOS is the bias between the reference
forecast and the nearest observation. Because the ERA5 collector writes
reanalysis rows at the EXACT virtual-point coordinates, its distance is zero
by construction, so the former pure-nearest target selector picked ERA5
whenever it existed — measured consequence: 100% of the 90,552 observation
rows used for training were era5_reanalysis. The model was learning to
predict ERA5's 25 km terrain-smoothed valley wind, not the wind a sailor
feels on the lake.

Fix, per the audit recommendation:
  1. Tier-first target selection — real anemometer stations (ARPA, Domaso,
     DIY buoy, Netatmo) ALWAYS win over reanalysis, regardless of distance.
  2. Target-quality weighting — reanalysis targets are not discarded (they
     still teach the seasonal/synoptic structure) but enter training with a
     reduced weight (station 1.0, intermediate reanalysis ~0.6, ERA5 0.4),
     multiplied by the observation's own confidence.
  3. Unvalidated station data is trusted less (confidence 0.7 from the ARPA
     collector's stato policy) but still beats a 25 km reanalysis cell.

The module is intentionally dependency-light (no DB access) so both the
feature builder and the trainers can use it.
"""
from __future__ import annotations

import math
from typing import Any

# --- Source tier classification -------------------------------------------

# Tier 0: real anemometers. These are the ground truth the product promises.
STATION_SOURCE_PREFIXES = ("arpa_", "domaso", "diy_buoy", "netatmo")
# Tier 1: regional reanalysis — intermediate ground truth for the transition
# period while the station ledger grows (CERRA at 5.5 km resolves the valley
# far better than ERA5's 0.25deg; candidate evaluation R2 follow-up).
INTERMEDIATE_REANALYSIS_SOURCES = ("cerra",)
# Tier 2: global reanalysis — lowest-trust surrogate.
ERA5_SOURCES = ("era5_reanalysis",)

TIER_STATION = 0
TIER_INTERMEDIATE = 1
TIER_ERA5 = 2


def source_tier(source: str | None) -> int:
    """Classify an observation source into the ground-truth hierarchy."""
    s = str(source or "")
    if s.startswith(STATION_SOURCE_PREFIXES):
        return TIER_STATION
    if any(s == c or s.startswith(c + "_") for c in INTERMEDIATE_REANALYSIS_SOURCES):
        return TIER_INTERMEDIATE
    return TIER_ERA5


def _score(o: dict[str, Any]) -> float:
    """Lower is better: distance + staleness penalty (km + km-equivalent)."""
    dist_km = float(o.get("dist_km") or 0.0)
    age_min = float(o.get("age_min") or 0.0)
    # 30 min of staleness "costs" as much as 1 km of distance — an old
    # anemometer reading is less representative of a thermally-driven
    # breeze than a fresh one.
    return dist_km + age_min / 30.0


def select_target_obs(
    candidate_obs: list[dict[str, Any]],
    lat: float,
    lon: float,
) -> dict[str, Any] | None:
    """Pick the ground-truth observation for a training target.

    Tier-first: the best TIER-0 (station) observation wins over ANY reanalysis
    row, however close the reanalysis cell is (it is distance-zero by
    construction — that is exactly the bug). Within a tier, candidates are
    ranked by distance + staleness. `candidate_obs` rows must carry lat/lon,
    `source`, `wind_speed_kn`, `wind_dir_deg`; rows without both wind
    components are never usable as targets. `age_min` (minutes since the
    observation) may be precomputed by the caller; rows missing it are
    treated as maximally stale within their tier.
    """
    usable: list[tuple[int, float, dict[str, Any]]] = []
    for o in candidate_obs or []:
        if o.get("wind_speed_kn") is None or o.get("wind_dir_deg") is None:
            continue
        olat = o.get("lat") or 0.0
        olon = o.get("lon") or 0.0
        dist_km = _haversine_km(lat, lon, olat, olon)
        age_min = o.get("age_min")
        age_min = float(age_min) if age_min is not None else 999.0
        tier = source_tier(o.get("source"))
        enriched = {**o, "dist_km": dist_km, "age_min": age_min}
        usable.append((tier, _score(enriched), enriched))
    if not usable:
        return None
    usable.sort(key=lambda t: (t[0], t[1]))
    return usable[0][2]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    return 2.0 * r * math.asin(math.sqrt(a))


def tier_weight(tier: int, weights: Any) -> float:
    """Configured training weight for a hierarchy tier.

    `weights` is a settings.model.target_quality object with
    station_weight / intermediate_reanalysis_weight / era5_weight.
    """
    if tier == TIER_STATION:
        return float(getattr(weights, "station_weight", 1.0))
    if tier == TIER_INTERMEDIATE:
        return float(getattr(weights, "intermediate_reanalysis_weight", 0.6))
    return float(getattr(weights, "era5_weight", 0.4))


def target_quality_weight(
    source: str | None,
    confidence: float | None,
    weights: Any,
) -> float:
    """Combined sample weight = tier weight x observation confidence, in (0, 1].

    An unvalidated ARPA row (confidence 0.7) therefore contributes less than a
    validated one (0.85), and every ERA5 row contributes at most its tier
    weight — implementing the audit's "demote ERA5 to 0.3-0.5" policy.
    """
    w = tier_weight(source_tier(source), weights)
    conf = confidence if confidence is not None else 1.0
    return float(max(0.0, min(1.0, w * float(conf))))


__all__ = [
    "STATION_SOURCE_PREFIXES",
    "INTERMEDIATE_REANALYSIS_SOURCES",
    "ERA5_SOURCES",
    "TIER_STATION",
    "TIER_INTERMEDIATE",
    "TIER_ERA5",
    "source_tier",
    "select_target_obs",
    "tier_weight",
    "target_quality_weight",
]
