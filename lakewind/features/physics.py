"""V7 physics features — lake thermodynamics, terrain channeling, pressure
gradients (Phase 3 deep-research deliverable).

Research basis (Lake Como upper basin, Dongo–Dervio corridor):

1. TERRAIN CHANNELING. The upper lake sits in a narrow N–S alpine valley
   (axis ≈ 010° true). Flow along the valley axis propagates efficiently and
   is funneled (Venturi effect) as the valley narrows northward; the
   cross-valley component is blocked by 1000 m+ walls and dissipates into
   turbulence. Decomposing the forecast wind into along-valley and
   cross-valley components therefore separates "physically meaningful" from
   "terrain-shredded" flow:
       axis_align  = cos(dir − θ_axis)            (1 = aligned)
       speed_along = speed · cos(dir − θ_axis)    (signed: + down-valley/N,
                                                   − up-valley/S = Breva)
       speed_cross = speed · sin(dir − θ_axis)    (blocked component)
   Both Breva (S, ≈190°) and Tivano (N, ≈010°) project onto |along|;
   the SIGN of speed_along distinguishes them.

2. PRESSURE TENDENCY. The 3-hour pressure change (Δp₃) is the classic
   synoptic intensity parameter (surface charts draw isallobars on it):
       - Falling pressure + rising southerly gradient → pre-frontal Scirocco
         / southerly inflow ahead of Atlantic troughs.
       - Rising pressure behind a cold front → Mistral/tramontana advection
         and Foehn onset (Zurich–Milano gradient flips positive).
   The 6-hour tendency removes the diurnal tide (atmospheric thermal tide
   has ~12h/6h harmonics; 6h Δp still contains part of it, but the tree
   model can condition on hour_local).

3. THERMAL CONTRASTS. Lake-breeze (Breva) intensity scales with the
   land–water thermal contrast and with the valley–plain contrast:
       - lake − Po plain (Dongo − Milano): Po plain heats faster; a strong
         positive gradient pulls southerly flow up-valley → reinforces Breva.
       - lake − Valtellina (Dongo − Sondrio): mountain air stays cold; the
         gradient drives valley Anabatic wind merging into Breva.
       - Zurich − Milano temperature difference tracks cold advection from
         the north (Tivano/Foehn support).

4. GUST FACTOR (gust/mean). Turbulence-intensity proxy: convective boundary
   layers produce GF ≈ 1.5–2.5, laminar flows GF ≈ 1.1–1.3. For a bias model
   this identifies conditions where the 10 m mean speed is a poor predictor
   of what a shore anemometer actually reads.

5. EFFECTIVE INSOLATION. shortwave_radiation × (1 − cloud/100) — the actual
   surface heating driving the thermal circulation. solar_elevation ×
   cloud-clearness × Breva-window interactions give the model the physical
   product instead of asking it to learn multiplicative structure.

6. CROSS-MODEL AGGREGATES. Pairwise agreement features exist (V2); when one
   model is missing, pairwise terms vanish silently. Mean/std/min/max speed
   across models and the circular direction spread are robust to dropouts
   and give the ensemble-spread signal (uncertainty → wider quantiles).

All functions are pure: they take the feature vector (and optionally aux
temperature readings) and return features. No I/O → trivially testable.
"""
from __future__ import annotations

import math
from typing import Any

# Default valley axis for the Dongo–Dervio corridor (degrees FROM north,
# direction the valley points toward — the upper Como basin axis ≈ 010°).
DEFAULT_VALLEY_AXIS_DEG = 10.0

# Reference model used for point-level tendency features.
TENDENCY_REF_MODEL = "icon_eu"


def _sf(v: Any) -> float | None:
    """Safe float."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def valley_axis_deg_for(point_id: str, axis_deg: float, overrides: dict[str, float] | None) -> float:
    """Resolve the valley axis for a point (per-point override > global)."""
    if overrides and point_id in overrides:
        return float(overrides[point_id])
    return float(axis_deg)


def compute_valley_axis_features(
    feature_vector: dict[str, Any],
    point_id: str,
    axis_deg: float = DEFAULT_VALLEY_AXIS_DEG,
    overrides: dict[str, float] | None = None,
    models: list[str] | None = None,
) -> dict[str, float | None]:
    """Decompose each model's forecast wind into along/cross valley components."""
    theta = math.radians(valley_axis_deg_for(point_id, axis_deg, overrides))
    out: dict[str, float | None] = {}
    algn: list[float] = []

    if models is None:
        models = sorted({
            k[len("fc_"):-len("_speed")]
            for k in feature_vector
            if k.startswith("fc_") and k.endswith("_speed")
        })

    for m in models:
        speed = _sf(feature_vector.get(f"fc_{m}_speed"))
        direction = _sf(feature_vector.get(f"fc_{m}_dir"))
        if speed is None or direction is None:
            out[f"valley_align_{m}"] = None
            out[f"valley_along_{m}"] = None
            out[f"valley_cross_{m}"] = None
            continue
        delta = math.radians(direction) - theta
        cos_d = math.cos(delta)
        sin_d = math.sin(delta)
        out[f"valley_align_{m}"] = round(cos_d, 4)
        out[f"valley_along_{m}"] = round(speed * cos_d, 3)
        out[f"valley_cross_{m}"] = round(speed * sin_d, 3)
        algn.append(abs(cos_d))

    # Mean |alignment| across models: how terrain-friendly is the synoptic flow?
    out["valley_align_mean"] = round(sum(algn) / len(algn), 4) if algn else None

    # Reference-model products (the model corrects icon_eu bias; interactions
    # with the reference forecast speed are the most direct signals).
    along = out.get(f"valley_along_{TENDENCY_REF_MODEL}")
    cross = out.get(f"valley_cross_{TENDENCY_REF_MODEL}")
    if along is not None:
        out["valley_along_sq"] = round(along * along, 3)  # funnelling energy proxy
    if along is not None and cross is not None:
        out["valley_cross_ratio"] = round(cross / max(abs(along), 0.1), 4)
    return out


def compute_pressure_tendency(feature_vector: dict[str, Any]) -> dict[str, float | None]:
    """3 h / 6 h pressure tendency from the reference model's lag features.

    build.py provides lag{180,360}_press for the reference model. Tendency is
    p(t) − p(t−Δ) in hPa; negative = falling (approaching trough / pre-frontal
    southerly), positive = rising (post-frontal N'ly / Foehn support).
    """
    out: dict[str, float | None] = {}
    p_now = _sf(feature_vector.get(f"fc_{TENDENCY_REF_MODEL}_pressure"))
    p_3h = _sf(feature_vector.get("lag180_press"))
    p_6h = _sf(feature_vector.get("lag360_press"))
    out["ptend_3h"] = round(p_now - p_3h, 2) if (p_now is not None and p_3h is not None) else None
    out["ptend_6h"] = round(p_now - p_6h, 2) if (p_now is not None and p_6h is not None) else None
    # Tendency of the Zurich–Milano gradient is unavailable historically (aux
    # lags would double aux queries); combine current gradient × tendency
    # instead: a strengthening Foehn-supporting gradient that is also rising
    # is the strongest Foehn-onset signature.
    grad = _sf(feature_vector.get("pressure_grad_zurich_milano"))
    if grad is not None and out["ptend_6h"] is not None:
        out["foehn_grad_x_tend"] = round(grad * out["ptend_6h"], 3)
    else:
        out["foehn_grad_x_tend"] = None
    return out


def compute_gust_factor_features(
    feature_vector: dict[str, Any],
    models: list[str] | None = None,
) -> dict[str, float | None]:
    """Gust factor (gust / mean speed) per model + reference-model shortcuts."""
    out: dict[str, float | None] = {}
    if models is None:
        models = sorted({
            k[len("fc_"):-len("_speed")]
            for k in feature_vector
            if k.startswith("fc_") and k.endswith("_speed")
        })
    for m in models:
        speed = _sf(feature_vector.get(f"fc_{m}_speed"))
        gust = _sf(feature_vector.get(f"fc_{m}_gust"))
        if speed is not None and gust is not None and speed >= 0.5:
            out[f"gust_factor_{m}"] = round(gust / speed, 4)
        else:
            out[f"gust_factor_{m}"] = None
    return out


def compute_cross_model_aggregates(
    feature_vector: dict[str, Any],
    models: list[str] | None = None,
) -> dict[str, float | None]:
    """Robust multi-model aggregates (survive single-model dropouts)."""
    out: dict[str, float | None] = {}
    if models is None:
        models = sorted({
            k[len("fc_"):-len("_speed")]
            for k in feature_vector
            if k.startswith("fc_") and k.endswith("_speed")
        })
    speeds: list[float] = []
    dirs: list[float] = []
    press: list[float] = []
    for m in models:
        sp = _sf(feature_vector.get(f"fc_{m}_speed"))
        dr = _sf(feature_vector.get(f"fc_{m}_dir"))
        pr = _sf(feature_vector.get(f"fc_{m}_pressure"))
        if sp is not None:
            speeds.append(sp)
        if dr is not None:
            dirs.append(dr)
        if pr is not None:
            press.append(pr)

    if speeds:
        mean_s = sum(speeds) / len(speeds)
        var_s = sum((s - mean_s) ** 2 for s in speeds) / len(speeds)
        out["xm_speed_mean"] = round(mean_s, 3)
        out["xm_speed_std"] = round(math.sqrt(var_s), 3)
        out["xm_speed_min"] = round(min(speeds), 3)
        out["xm_speed_max"] = round(max(speeds), 3)
        out["xm_speed_range"] = round(max(speeds) - min(speeds), 3)
        # Relative spread: normalized dispersion (scale-free agreement signal)
        out["xm_speed_rel_spread"] = (
            round(math.sqrt(var_s) / max(mean_s, 0.5), 4) if len(speeds) >= 2 else None
        )
    else:
        for k in ("xm_speed_mean", "xm_speed_std", "xm_speed_min", "xm_speed_max",
                  "xm_speed_range", "xm_speed_rel_spread"):
            out[k] = None

    if dirs:
        rads = [math.radians(d) for d in dirs]
        sin_mean = sum(math.sin(r) for r in rads) / len(rads)
        cos_mean = sum(math.cos(r) for r in rads) / len(rads)
        # Circular std = sqrt(-2 ln R), R = |mean resultant length|
        r_length = math.hypot(sin_mean, cos_mean)
        r_clamped = min(1.0, max(1e-9, r_length))
        out["xm_dir_circ_std"] = round(math.degrees(math.sqrt(-2.0 * math.log(r_clamped))), 2)
    else:
        out["xm_dir_circ_std"] = None

    if press:
        mean_p = sum(press) / len(press)
        var_p = sum((p - mean_p) ** 2 for p in press) / len(press)
        out["xm_press_mean"] = round(mean_p, 2)
        out["xm_press_std"] = round(math.sqrt(var_p), 3)
    else:
        out["xm_press_mean"] = None
        out["xm_press_std"] = None
    return out


def compute_thermal_contrast_features(
    feature_vector: dict[str, Any],
    aux_temps: dict[str, float | None],
) -> dict[str, float | None]:
    """Land–water / valley–plain thermal contrasts.

    `aux_temps` carries air temperatures for auxiliary points fetched once by
    build.py: {"zurich": T, "milano": T, "sondrio": T, "lugano": T,
               "dongo": T, "bellano": T} (None where missing).
    """
    out: dict[str, float | None] = {}

    def _diff(a: str, b: str, name: str) -> None:
        ta = aux_temps.get(a)
        tb = aux_temps.get(b)
        out[name] = round(ta - tb, 2) if (ta is not None and tb is not None) else None

    _diff("dongo", "sondrio", "therm_lake_valley")       # Breva / Anabatic driver
    _diff("dongo", "milano", "therm_lake_po")            # Po-plain pull (southerly)
    _diff("zurich", "milano", "therm_north_advection")   # cold advection from N

    # Interaction of the lake–Po contrast with the Breva window: strong plain
    # heating inside the Breva window is the textbook Breva amplifier.
    lake_po = out["therm_lake_po"]
    if lake_po is not None and feature_vector.get("breva_window"):
        out["therm_lake_po_x_breva"] = round(max(0.0, -lake_po) * 1.0, 3)
        # plain warmer than lake → lake_po negative → -lake_po positive = pull
    else:
        out["therm_lake_po_x_breva"] = None
    return out


def compute_effective_insolation(feature_vector: dict[str, Any]) -> dict[str, float | None]:
    """Cloud-modulated surface heating and its interactions."""
    out: dict[str, float | None] = {}
    rad = _sf(feature_vector.get(f"fc_{TENDENCY_REF_MODEL}_rad"))
    cloud = _sf(feature_vector.get(f"fc_{TENDENCY_REF_MODEL}_cloud"))
    elev = _sf(feature_vector.get("solar_elevation"))

    if rad is not None and cloud is not None:
        out["insol_effective"] = round(rad * (1.0 - min(100.0, max(0.0, cloud)) / 100.0), 1)
    else:
        out["insol_effective"] = None

    # Heating-rate proxy: effective insolation × sin(elevation) — low-sun
    # irradiance heats poorly (long path, albedo) even when cloud-free.
    ie = out.get("insol_effective")
    if ie is not None and elev is not None and elev > 0:
        out["insol_x_elevation"] = round(ie * math.sin(math.radians(elev)), 1)
    else:
        out["insol_x_elevation"] = None

    # Breva-window interaction (the physical product the model would
    # otherwise have to learn from multiplicative structure).
    if ie is not None and feature_vector.get("breva_window"):
        out["insol_x_breva"] = round(ie, 1)
    else:
        out["insol_x_breva"] = None
    return out


def compute_stability_interactions(feature_vector: dict[str, Any]) -> dict[str, float | None]:
    """Boundary-layer coupling interactions.

    - stability × wind: unstable + windy = full momentum transfer to surface;
      stable + windy = decoupled surface calm (bias strongly negative).
    - BLH × gust factor: deep + turbulent = well-mixed boundary layer.
    - CAPE × insolation: convective potential realized only with heating.
    """
    out: dict[str, float | None] = {}
    stability = _sf(feature_vector.get("stability_score"))
    speed = _sf(feature_vector.get(f"fc_{TENDENCY_REF_MODEL}_speed"))
    blh = _sf(feature_vector.get(f"fc_{TENDENCY_REF_MODEL}_blh"))
    gf = _sf(feature_vector.get(f"gust_factor_{TENDENCY_REF_MODEL}"))
    cape = _sf(feature_vector.get(f"fc_{TENDENCY_REF_MODEL}_cape"))
    ie = _sf(feature_vector.get("insol_effective"))

    out["stability_x_speed"] = (
        round(stability * speed, 3) if (stability is not None and speed is not None) else None
    )
    out["blh_x_gustfactor"] = (
        round(blh * gf, 1) if (blh is not None and gf is not None) else None
    )
    out["cape_x_insolation"] = (
        round(cape * min(ie, 1000.0) / 1000.0, 1) if (cape is not None and ie is not None) else None
    )
    return out


# Names of every feature introduced by this module (Phase 3). The experiment
# harness uses this set to run the "baseline" configuration on the pre-V7
# feature subset, isolating the feature contribution from the tuning gain.
PHYSICS_V7_FEATURES: frozenset[str] = frozenset(
    {
        "valley_align_mean", "valley_along_sq", "valley_cross_ratio",
        "ptend_3h", "ptend_6h", "foehn_grad_x_tend",
        "xm_speed_mean", "xm_speed_std", "xm_speed_min", "xm_speed_max",
        "xm_speed_range", "xm_speed_rel_spread", "xm_dir_circ_std",
        "xm_press_mean", "xm_press_std",
        "therm_lake_valley", "therm_lake_po", "therm_north_advection",
        "therm_lake_po_x_breva",
        "insol_effective", "insol_x_elevation", "insol_x_breva",
        "stability_x_speed", "blh_x_gustfactor", "cape_x_insolation",
    }
) | frozenset(
    {  # per-model triplets
        "valley_align_{m}", "valley_along_{m}", "valley_cross_{m}",
        "gust_factor_{m}",
    }
)


def is_v7_feature(name: str) -> bool:
    """True if a concrete feature name belongs to the V7 physics set."""
    if name in PHYSICS_V7_FEATURES:
        return True
    for tpl in ("valley_align_", "valley_along_", "valley_cross_", "gust_factor_"):
        if name.startswith(tpl) and len(name) > len(tpl):
            return True
    return False


def compute_all_v7_physics(
    feature_vector: dict[str, Any],
    point_id: str,
    aux_temps: dict[str, float | None],
    axis_deg: float = DEFAULT_VALLEY_AXIS_DEG,
    overrides: dict[str, float] | None = None,
) -> dict[str, float | None]:
    """One-call composition used by features/build.py."""
    out: dict[str, float | None] = {}
    out.update(compute_valley_axis_features(feature_vector, point_id, axis_deg, overrides))
    out.update(compute_pressure_tendency(feature_vector))
    out.update(compute_gust_factor_features(feature_vector))
    out.update(compute_cross_model_aggregates(feature_vector))
    out.update(compute_thermal_contrast_features(feature_vector, aux_temps))
    out.update(compute_effective_insolation(feature_vector))
    out.update(compute_stability_interactions(feature_vector))
    return out


__all__ = [
    "DEFAULT_VALLEY_AXIS_DEG",
    "TENDENCY_REF_MODEL",
    "PHYSICS_V7_FEATURES",
    "compute_all_v7_physics",
    "compute_valley_axis_features",
    "compute_pressure_tendency",
    "compute_gust_factor_features",
    "compute_cross_model_aggregates",
    "compute_thermal_contrast_features",
    "compute_effective_insolation",
    "compute_stability_interactions",
    "is_v7_feature",
    "valley_axis_deg_for",
]
