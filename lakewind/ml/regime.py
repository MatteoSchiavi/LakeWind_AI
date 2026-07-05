"""V2 regime classifier — deterministic rules + tiny gradient-boosted classifier.

Spec §4.4 V1 uses pure deterministic rules (Breva/Tivano windows, Foehn pressure
gradient threshold). V2 adds:
- A small gradient-boosted classifier (5-class: calm/breva/tivano/foehn/storm)
  trained on the deterministic-rule labels as weak supervision
- The classifier learns the boundaries between regimes more gracefully than
  hard rules
- Both the rule-based label AND the classifier's probabilities are exposed as
  features to the main model
"""
from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from lakewind.config import load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")

# Regime classes
REGIMES = ["calm", "breva", "tivano", "foehn", "storm"]


@dataclass
class RegimeResult:
    regime: str
    confidence: float
    rules_detected: dict[str, bool]
    classifier_probabilities: dict[str, float] | None


def classify_regime(
    valid_time: datetime,
    feature_vector: dict[str, Any],
    *,
    use_classifier: bool = True,
) -> RegimeResult:
    """Classify the current weather regime.

    Combines deterministic rules (always run, provide interpretable flags) with
    an optional gradient-boosted classifier for confidence scoring.
    """
    s = load_settings()
    tz = ZoneInfo(s.project.timezone)
    local_time = (
        valid_time.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
        if valid_time.tzinfo is None
        else valid_time.astimezone(tz)
    )
    hhmm = local_time.strftime("%H:%M")
    hour = local_time.hour

    # --- Deterministic rules ---
    rules: dict[str, bool] = {
        "breva_window": "10:00" <= hhmm <= "18:00",
        "tivano_window": "04:00" <= hhmm <= "09:30",
        "tivano_dying": _in_window(hhmm, s.local_winds.tivano_die_window.start,
                                    s.local_winds.tivano_die_window.end),
        "breva_building": _in_window(hhmm, s.local_winds.breva_build_window.start,
                                     s.local_winds.breva_build_window.end),
        "foehn_likely": bool(feature_vector.get("foehn_likely", False)),
        "foehn_strong": bool(feature_vector.get("foehn_strong", False)),
        "high_cape": _safe_float(feature_vector.get("fc_icon_eu_cape")) is not None
                     and _safe_float(feature_vector.get("fc_icon_eu_cape")) > 1000,
        "low_speed_forecast": _safe_float(feature_vector.get("fc_icon_eu_speed")) is not None
                              and _safe_float(feature_vector.get("fc_icon_eu_speed")) < 3.0,
        "strong_forecast": _safe_float(feature_vector.get("fc_icon_eu_speed")) is not None
                           and _safe_float(feature_vector.get("fc_icon_eu_speed")) > 18.0,
    }

    # Rule-based regime (priority order: storm > foehn > breva > tivano > calm)
    if rules["foehn_strong"]:
        rule_regime = "foehn"
    elif rules["high_cape"] and rules["strong_forecast"]:
        rule_regime = "storm"
    elif rules["foehn_likely"]:
        rule_regime = "foehn"
    elif rules["breva_window"] and not rules["low_speed_forecast"]:
        rule_regime = "breva"
    elif rules["tivano_window"] and not rules["low_speed_forecast"]:
        rule_regime = "tivano"
    elif rules["low_speed_forecast"]:
        rule_regime = "calm"
    else:
        rule_regime = "calm"

    # --- Classifier (optional) ---
    classifier_probs: dict[str, float] | None = None
    confidence = 0.7  # default for rule-based
    if use_classifier:
        try:
            classifier_probs = _run_classifier(feature_vector, rules)
            if classifier_probs:
                # Use classifier's top class if confidence is high enough
                top_class = max(classifier_probs, key=classifier_probs.get)
                top_prob = classifier_probs[top_class]
                if top_prob > 0.6:
                    regime = top_class
                    confidence = top_prob
                else:
                    regime = rule_regime
                    confidence = 1.0 - top_prob
            else:
                regime = rule_regime
        except Exception as exc:
            logger.debug("Classifier failed: %s", exc)
            regime = rule_regime
    else:
        regime = rule_regime

    return RegimeResult(
        regime=regime,
        confidence=confidence,
        rules_detected=rules,
        classifier_probabilities=classifier_probs,
    )


def _run_classifier(
    feature_vector: dict[str, Any], rules: dict[str, bool]
) -> dict[str, float] | None:
    """Run the gradient-boosted regime classifier.

    Returns a dict of regime -> probability, or None if no classifier is trained.
    """
    import pickle

    model_path = MODELS_DIR / "regime_classifier.pkl"
    features_path = MODELS_DIR / "regime_classifier_features.json"
    if not model_path.exists() or not features_path.exists():
        return None

    with model_path.open("rb") as fh:
        clf = pickle.load(fh)
    feature_cols = json.loads(features_path.read_text())

    # Build feature row
    row = {}
    for c in feature_cols:
        v = feature_vector.get(c)
        if v is None:
            row[c] = 0.0
        elif isinstance(v, bool):
            row[c] = float(v)
        else:
            try:
                row[c] = float(v)
            except (TypeError, ValueError):
                row[c] = 0.0
    # Add rule flags as features
    for k, v in rules.items():
        row[f"rule_{k}"] = float(v)

    X = pd.DataFrame([row], columns=feature_cols)
    probs = clf.predict_proba(X)[0]
    classes = clf.classes_ if hasattr(clf, "classes_") else REGIMES
    return {str(c): float(p) for c, p in zip(classes, probs)}


def train_regime_classifier(
    start: datetime,
    end: datetime,
    reference_forecast_model: str = "icon_eu",
) -> bool:
    """Train the gradient-boosted regime classifier on stored history.

    Uses the deterministic rules as weak supervision labels.
    Returns True if training succeeded.
    """
    from lakewind.features.build import build_features_for
    import lightgbm as lgb

    s = load_settings()
    op_ids = s.operational_point_ids or [p.id for p in s.virtual_points]

    rows: list[dict[str, Any]] = []
    cur = start
    while cur < end:
        for pid in op_ids:
            try:
                fr = build_features_for(pid, cur, reference_forecast_model=reference_forecast_model)
            except Exception:
                continue
            if fr is None:
                continue
            # Get rule-based label (without classifier to avoid recursion)
            result = classify_regime(cur, fr.feature_vector, use_classifier=False)
            row = {**fr.feature_vector}
            for k, v in result.rules_detected.items():
                row[f"rule_{k}"] = float(v)
            row["target_regime"] = result.regime
            row["point_id"] = pid
            row["valid_time"] = cur
            rows.append(row)
        cur += timedelta(hours=1)

    if len(rows) < 100:
        logger.warning("Not enough samples to train regime classifier: %d", len(rows))
        return False

    df = pd.DataFrame(rows)
    drop_cols = {"point_id", "valid_time", "target_regime"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    y = df["target_regime"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(REGIMES),
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.05,
        verbose=-1,
    )
    clf.fit(X, y)

    with (MODELS_DIR / "regime_classifier.pkl").open("wb") as fh:
        pickle.dump(clf, fh)
    (MODELS_DIR / "regime_classifier_features.json").write_text(json.dumps(feature_cols))
    logger.info("Regime classifier trained on %d samples, %d features", len(df), len(feature_cols))
    return True


def _in_window(hhmm: str, start: str, end: str) -> bool:
    return start <= hhmm <= end


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


__all__ = ["classify_regime", "train_regime_classifier", "RegimeResult", "REGIMES"]
