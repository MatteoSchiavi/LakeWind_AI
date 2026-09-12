"""Forecast-vs-reality verification — the pre-Phase-6 gate (resumable stages).

Answers the operator's question: "have you checked the results the model
produced? are they in line with reality?"

Protocol (leakage-free three-way split over the stored history):
  TRAIN      2025-09-15 .. 2026-05-01  candidate MOS trained here only
  CALIBRATE  2026-07-01 .. 2026-07-15  split-conformal q_hat fitted here
  TEST       2026-07-15 .. 2026-08-19  NEVER seen by train or calibration

The sandbox reaps background processes, so every expensive stage runs
FOREGROUND and is resumable via marker/parquet state under
data/cache/verify/:

  --stage dataset   build per-point training feature parquets
  --stage train     train the candidate on the concatenated dataset
  --stage calibrate fit the conformal calibrator set on Jul 1-14
  --stage test      score test hours, per point (skips finished points)
  --stage report    metrics + figures + console digest

The TEST pass drives the EXACT serving path (`predict_at`: ensemble,
conformal band, regime direction correction, sanity clips) against the
stored ground truth, and scores it against two baselines (persistence and
raw icon_eu NWP) with the same helper functions the shipped backtest uses.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("verify")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# LAKEWIND_VERIFY_TAG isolates the whole state (parquets, markers, results)
# per reference-model candidate so the honest harness can A/B
# ecmwf_ifs025 vs icon_eu without cache contamination.
_TAG = os.environ.get("LAKEWIND_VERIFY_TAG", "").strip()
OUT_DIR = ROOT / "data" / "cache" / ("verify" if not _TAG else f"verify_{_TAG}")
ASSET_DIR = ROOT / "docs" / "assets"

TRAIN_START = datetime(2025, 9, 15)
TRAIN_END = datetime(2026, 5, 1)   # exclusive; Apr 30 last train sample
CAL_END = datetime(2026, 7, 15)    # window_days=14 back from here == Jul 1-14
TEST_START = datetime(2026, 7, 15)
TEST_END = datetime(2026, 8, 19)   # exclusive; ERA5 ground truth ends Aug 18 23:00

# NOTE: May-Jun 2026 forecasts are missing from the local backfill (HTTP 429
# quota today); the calibration window therefore sits inside July — still
# strictly AFTER training and BEFORE test, so the three-way split remains
# leakage-free while being better matched to the summer test regime.

RESULTS_PATH = OUT_DIR / "verification_results.json"
CAND_MARKER = OUT_DIR / "verify_candidate.txt"
CAL_MARKER = OUT_DIR / "calibrated.marker"


def utcnow() -> datetime:
    return datetime.utcnow()


def op_points() -> list[str]:
    from lakewind.config import load_settings

    s = load_settings()
    return list(s.operational_point_ids)


# ------------------------------------------------------------------- stages

def stage_dataset(points: list[str]) -> None:
    """Build per-point training feature parquets (skips points already done)."""
    from lakewind.ml.train import _build_dataset

    for pid in points:
        out = OUT_DIR / f"train_ds_{pid}.parquet"
        marker = OUT_DIR / f"train_ds_{pid}.done"
        if marker.exists() and out.exists():
            log.info("dataset %s already built", pid)
            continue
        t0 = time.time()
        df = _build_dataset(pid, TRAIN_START, TRAIN_END)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        marker.write_text(f"{len(df)}")
        log.info("dataset %s: %d samples in %.0fs", pid, len(df), time.time() - t0)


def stage_train() -> str:
    """Train the candidate on the concatenated per-point datasets."""
    from lakewind.ml.train import train

    if CAND_MARKER.exists():
        candidate = CAND_MARKER.read_text().strip()
        log.info("Reusing candidate %s", candidate)
        return candidate

    frames = []
    for f in sorted(OUT_DIR.glob("train_ds_*.parquet")):
        frames.append(pd.read_parquet(f))
        log.info("loaded %s: %d rows", f.name, len(frames[-1]))
    df = pd.concat(frames, ignore_index=True)
    log.info("Concatenated training dataset: %d samples", len(df))

    t0 = time.time()
    # LAKEWIND_VERIFY_ENSEMBLE=0 trains single-backend (fits the sandbox
    # 10-min foreground window on 82k samples). Applied identically to every
    # A/B tag, so the reference-model comparison stays fair; production
    # training keeps settings.model.ensemble.
    ensemble = os.environ.get("LAKEWIND_VERIFY_ENSEMBLE", "1") != "0"
    # LAKEWIND_VERIFY_TARGETS=u / v: split one candidate across two
    # invocations when even the single-backend fit exceeds the window.
    # LAKEWIND_VERIFY_MV pins the model_version for both halves; the second
    # half (v) writes the candidate marker and skips re-registration.
    targets_env = os.environ.get("LAKEWIND_VERIFY_TARGETS", "").strip()
    fixed_mv = os.environ.get("LAKEWIND_VERIFY_MV", "").strip()
    if targets_env in ("u", "v") and not fixed_mv:
        log.error("LAKEWIND_VERIFY_TARGETS requires LAKEWIND_VERIFY_MV")
        raise SystemExit(1)
    if targets_env in ("u", "v"):
        res = train(
            dataset=df, start=TRAIN_START, end=TRAIN_END, ensemble=ensemble,
            model_version=fixed_mv,
            targets=(targets_env,),
            register=(targets_env == "u"),
        )
        if res is None:
            log.error("Training failed")
            raise SystemExit(1)
        candidate = res.model_version
        if targets_env == "v":
            CAND_MARKER.write_text(candidate)
            log.info("Candidate %s complete (u+v)", candidate)
        else:
            log.info("Half 1 (u) done for %s — run half 2 with LAKEWIND_VERIFY_TARGETS=v", candidate)
        return candidate
    res = train(dataset=df, start=TRAIN_START, end=TRAIN_END, ensemble=ensemble)
    if res is None:
        log.error("Training failed")
        raise SystemExit(1)
    candidate = res.model_version
    CAND_MARKER.write_text(candidate)
    log.info(
        "Trained %s: %d samples, %d features in %.0fs",
        candidate, res.n_samples, res.n_features, time.time() - t0,
    )
    log.info("Train metrics: %s", res.metrics)
    return candidate


def stage_calibrate(candidate: str) -> None:
    """Fit the conformal calibrator set on Jul 1-14 (post-train, pre-test)."""
    from lakewind.ml.review import fit_bundle_calibrators

    if CAL_MARKER.exists():
        log.info("Calibration already done")
        return
    t0 = time.time()
    cal = fit_bundle_calibrators(candidate, window_days=14, end=CAL_END, skip_existing=True)
    log.info("Calibration: %s in %.0fs", cal, time.time() - t0)
    if not cal["ok"]:
        log.error("Conformal calibration failed — coverage contract unverifiable")
        raise SystemExit(1)
    CAL_MARKER.write_text("ok")


def stage_test(points: list[str], candidate: str, smoke: bool) -> None:
    """Score test hours per point; each finished point lands in a parquet."""
    from lakewind.ml.backtest import (
        _materialize_test_samples,
        _persistence_prediction,
        _row_lead_hours,
    )
    from lakewind.ml.infer import predict_at

    test_end = TEST_START + timedelta(days=3) if smoke else TEST_END
    for pid in points:
        out = OUT_DIR / f"test_rows_{pid}.parquet"
        if out.exists() and not smoke:
            log.info("test rows for %s already scored", pid)
            continue
        t0 = time.time()
        samples = _materialize_test_samples(pid, TEST_START, test_end)
        rows: list[dict] = []
        for i, s_row in enumerate(samples):
            cand = predict_at(pid, s_row["valid_time"], model_version=candidate, compute_shap=False)
            if cand is None:
                continue
            pers = _persistence_prediction(
                pid, s_row["valid_time"], anchor=s_row.get("ref_run_time")
            )
            rows.append(
                {
                    "point_id": pid,
                    "valid_time": s_row["valid_time"],
                    "obs": s_row["obs_speed"],
                    "obs_dir": s_row["obs_dir"],
                    "cand": cand.wind_speed_kn,
                    "cand_dir": cand.wind_dir_deg,
                    "q10": cand.wind_speed_q10_kn if cand.wind_speed_q10_kn is not None else np.nan,
                    "q90": cand.wind_speed_q90_kn if cand.wind_speed_q90_kn is not None else np.nan,
                    "ee": cand.expected_error_kn,
                    "conf": cand.confidence_pct,
                    "nwp": s_row["ref_speed"],
                    "nwp_dir": s_row["ref_dir"],
                    "pers": pers[0] if pers else np.nan,
                    "lead": _row_lead_hours(s_row),
                    "regime": (
                        "breva" if s_row["regime_breva"]
                        else "tivano" if s_row["regime_tivano"]
                        else "foehn" if s_row["regime_foehn"]
                        else "calm"
                    ),
                    "obs_source": s_row["obs_source"],
                }
            )
            if (i + 1) % 250 == 0:
                log.info("  %s: %d/%d (%.1f rows/s)", pid, i + 1, len(samples), (i + 1) / (time.time() - t0))
        pd.DataFrame(rows).to_parquet(out, index=False)
        log.info("test rows %s: %d scored in %.0fs", pid, len(rows), time.time() - t0)


def stage_report(points: list[str], candidate: str, smoke: bool) -> int:
    """Aggregate scored rows into the metric suite + figures."""
    from lakewind.ml.backtest import brier_score, event_probability, lead_bucket, reliability_bins
    from lakewind.utils.wind import circular_direction_error_deg

    frames = []
    for pid in points:
        f = OUT_DIR / f"test_rows_{pid}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    if not frames:
        log.error("No scored test rows found")
        return 1
    rows = pd.concat(frames, ignore_index=True).to_dict("records")
    for r in rows:
        r["valid_time"] = pd.Timestamp(r["valid_time"]).to_pydatetime()
    if len(rows) < 200 and not smoke:
        log.error("Only %d scored rows — verification not meaningful", len(rows))
        return 1

    arr = {
        k: np.array([r[k] if r[k] is not None else np.nan for r in rows], dtype=float)
        for k in ("obs", "obs_dir", "cand", "cand_dir", "q10", "q90", "ee",
                  "conf", "nwp", "nwp_dir", "pers", "lead")
    }

    def _mask_pair(a, b, mask=None):
        m = np.isfinite(a) & np.isfinite(b)
        if mask is not None:
            m &= mask
        return m

    def mae(a, b, mask=None):
        m = _mask_pair(a, b, mask)
        return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else None

    def bias(a, b, mask=None):
        m = _mask_pair(a, b, mask)
        return float(np.mean(a[m] - b[m])) if m.any() else None

    def rmse(a, b, mask=None):
        m = _mask_pair(a, b, mask)
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else None

    def direrr(pred_col, mask=None):
        errs = []
        for i, r in enumerate(rows):
            if mask is not None and not mask[i]:
                continue
            errs.append(circular_direction_error_deg(r[pred_col], r["obs_dir"]))
        return float(np.mean(errs)) if errs else None

    obs, cand, nwp, pers = arr["obs"], arr["cand"], arr["nwp"], arr["pers"]
    all_ok = np.isfinite(obs) & np.isfinite(cand)

    summary: dict = {
        "generated_at": utcnow().isoformat(),
        "candidate_model_version": candidate,
        "protocol": {
            "train": [TRAIN_START.isoformat(), TRAIN_END.isoformat()],
            "calibrate": ["2026-07-01", CAL_END.isoformat()],
            "test": [TEST_START.isoformat(), (TEST_START + timedelta(days=3)).isoformat() if smoke else TEST_END.isoformat()],
        },
        "n_rows": len(rows),
        "obs_source_counts": {
            src: int(sum(1 for r in rows if r["obs_source"] == src))
            for src in sorted({r["obs_source"] for r in rows})
        },
        "speed": {
            "cand_mae": mae(cand, obs), "cand_bias": bias(cand, obs),
            "cand_rmse": rmse(cand, obs),
            "nwp_mae": mae(nwp, obs), "nwp_bias": bias(nwp, obs),
            "pers_mae": mae(pers, obs), "pers_bias": bias(pers, obs),
            "pearson_r": float(np.corrcoef(cand[all_ok], obs[all_ok])[0, 1]) if all_ok.sum() > 2 else None,
        },
        "direction": {
            "cand_dir_err": direrr("cand_dir"), "nwp_dir_err": direrr("nwp_dir"),
        },
        "coverage": {
            "served_band_pct": float(np.mean((arr["q10"][all_ok] - 1e-9 <= obs[all_ok]) & (obs[all_ok] <= arr["q90"][all_ok] + 1e-9)) * 100),
            "center_pm_ee_pct": float(np.mean(np.abs(cand[all_ok] - obs[all_ok]) <= arr["ee"][all_ok]) * 100),
            "mean_band_width_kn": float(np.mean(arr["q90"][all_ok] - arr["q10"][all_ok])),
        },
    }

    nwp_mae = summary["speed"]["nwp_mae"] or 0
    pers_mae = summary["speed"]["pers_mae"] or 0
    cand_mae = summary["speed"]["cand_mae"] or 0
    cand_dir = summary["direction"]["cand_dir_err"] or 0
    nwp_dir = summary["direction"]["nwp_dir_err"] or 0
    summary["skill"] = {
        "mae_reduction_vs_nwp_pct": round((1 - cand_mae / nwp_mae) * 100, 2) if nwp_mae else None,
        "mae_reduction_vs_persistence_pct": round((1 - cand_mae / pers_mae) * 100, 2) if pers_mae else None,
        "dir_err_reduction_vs_nwp_pct": round((1 - cand_dir / nwp_dir) * 100, 2) if nwp_dir else None,
    }

    # decision precision (Spec §1.2: 11:00-16:00 LOCAL, >= 8 kn)
    from zoneinfo import ZoneInfo

    rome = ZoneInfo("Europe/Rome")
    dec_pred, dec_act = [], []
    for r in rows:
        h = r["valid_time"].replace(tzinfo=ZoneInfo("UTC")).astimezone(rome).hour
        if 11 <= h <= 16:
            dec_pred.append(r["cand"] >= 8.0)
            dec_act.append(r["obs"] >= 8.0)
    if dec_pred:
        summary["decision_precision_pct"] = round(
            float(np.mean([p == a for p, a in zip(dec_pred, dec_act, strict=False)])) * 100, 2
        )
        summary["decision_n"] = len(dec_pred)

    per_lead: dict[str, dict] = {}
    for r in rows:
        b = lead_bucket(r["lead"])
        d = per_lead.setdefault(b, {"n": 0, "cand_err": [], "nwp_err": [], "cov": []})
        d["n"] += 1
        d["cand_err"].append(abs(r["cand"] - r["obs"]))
        d["nwp_err"].append(abs(r["nwp"] - r["obs"]))
        d["cov"].append(r["q10"] - 1e-9 <= r["obs"] <= r["q90"] + 1e-9)
    summary["per_lead"] = {
        b: {
            "n": d["n"],
            "cand_mae": round(float(np.mean(d["cand_err"])), 3),
            "nwp_mae": round(float(np.mean(d["nwp_err"])), 3),
            "coverage_pct": round(float(np.mean(d["cov"])) * 100, 1),
        }
        for b, d in sorted(per_lead.items())
    }

    per_reg: dict[str, dict] = {}
    for r in rows:
        d = per_reg.setdefault(r["regime"], {"n": 0, "cand": [], "nwp": [], "pers": []})
        d["n"] += 1
        d["cand"].append(abs(r["cand"] - r["obs"]))
        d["nwp"].append(abs(r["nwp"] - r["obs"]))
        if r["pers"] is not None and np.isfinite(r["pers"]):
            d["pers"].append(abs(r["pers"] - r["obs"]))
    summary["per_regime"] = {
        k: {
            "n": d["n"],
            "cand_mae": round(float(np.mean(d["cand"])), 3),
            "nwp_mae": round(float(np.mean(d["nwp"])), 3),
            "pers_mae": round(float(np.mean(d["pers"])), 3) if d["pers"] else None,
        }
        for k, d in sorted(per_reg.items())
    }

    per_point: dict[str, dict] = {}
    for r in rows:
        d = per_point.setdefault(r["point_id"], {"n": 0, "cand": [], "bias": [], "cov": []})
        d["n"] += 1
        d["cand"].append(abs(r["cand"] - r["obs"]))
        d["bias"].append(r["cand"] - r["obs"])
        d["cov"].append(r["q10"] - 1e-9 <= r["obs"] <= r["q90"] + 1e-9)
    summary["per_point"] = {
        k: {
            "n": d["n"],
            "mae": round(float(np.mean(d["cand"])), 3),
            "bias": round(float(np.mean(d["bias"])), 3),
            "coverage_pct": round(float(np.mean(d["cov"])) * 100, 1),
        }
        for k, d in sorted(per_point.items())
    }

    windy = all_ok & (obs >= 8.0)
    summary["windy_subset_obs_ge_8kn"] = {
        "n": int(windy.sum()),
        "cand_mae": mae(cand, obs, windy),
        "cand_bias": bias(cand, obs, windy),
        "nwp_mae": mae(nwp, obs, windy),
        "nwp_bias": bias(nwp, obs, windy),
        "hit_rate_cand_ge8_when_obs_ge8_pct": (
            round(float(np.mean(cand[windy] >= 8.0)) * 100, 1) if windy.any() else None
        ),
        "false_alarm_rate_pct": (
            round(
                float(
                    np.sum((cand >= 8.0) & (obs < 8.0) & all_ok)
                    / max(1, int(np.sum((cand >= 8.0) & all_ok)))
                )
                * 100,
                1,
            )
        ),
    }

    event: dict = {}
    for thr in (8.0, 12.0):
        probs = [event_probability(r["cand"], r["ee"], thr) for r in rows if r["ee"] is not None]
        outs = [r["obs"] >= thr for r in rows if r["ee"] is not None]
        if probs:
            event[f"ge_{int(thr)}kn"] = {
                "n": len(probs),
                "base_rate": round(float(np.mean(outs)), 4),
                "brier": round(brier_score(probs, outs), 4),
                "reliability": [
                    {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}
                    for b in reliability_bins(probs, outs)
                ],
            }
    summary["event_verification"] = event

    diurnal: dict[int, dict[str, list]] = {h: {"obs": [], "cand": [], "nwp": []} for h in range(24)}
    for r in rows:
        h = r["valid_time"].replace(tzinfo=ZoneInfo("UTC")).astimezone(rome).hour
        diurnal[h]["obs"].append(r["obs"])
        diurnal[h]["cand"].append(r["cand"])
        diurnal[h]["nwp"].append(r["nwp"])
    summary["diurnal_local"] = {
        h: {
            "obs": round(float(np.mean(v["obs"])), 2) if v["obs"] else None,
            "cand": round(float(np.mean(v["cand"])), 2) if v["cand"] else None,
            "nwp": round(float(np.mean(v["nwp"])), 2) if v["nwp"] else None,
            "n": len(v["obs"]),
        }
        for h, v in diurnal.items()
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Results written to %s", RESULTS_PATH)

    if not smoke:
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        _figures(summary)

    sp, sk, cov = summary["speed"], summary["skill"], summary["coverage"]
    log.info("=" * 64)
    log.info("CANDIDATE %s — %d scored rows", candidate, len(rows))
    log.info("MAE  cand %.2f kn | nwp %.2f | pers %.2f", sp["cand_mae"], sp["nwp_mae"], sp["pers_mae"])
    log.info("BIAS cand %+.2f kn | nwp %+.2f | pers %+.2f", sp["cand_bias"], sp["nwp_bias"], sp["pers_bias"])
    log.info("DIR  cand %.1f deg | nwp %.1f", summary["direction"]["cand_dir_err"], summary["direction"]["nwp_dir_err"])
    log.info("SKILL vs NWP %s%% | vs persistence %s%%", sk["mae_reduction_vs_nwp_pct"], sk["mae_reduction_vs_persistence_pct"])
    log.info("COVERAGE served band %.1f%% | width %.2f kn", cov["served_band_pct"], cov["mean_band_width_kn"])
    if "decision_precision_pct" in summary:
        log.info("DECISION precision %.1f%% (n=%d)", summary["decision_precision_pct"], summary["decision_n"])
    log.info("=" * 64)
    return 0


def _figures(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hours = sorted(int(h) for h in summary["diurnal_local"])
    obs = [summary["diurnal_local"][h]["obs"] for h in hours]
    cand = [summary["diurnal_local"][h]["cand"] for h in hours]
    nwp = [summary["diurnal_local"][h]["nwp"] for h in hours]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.plot(hours, obs, "o-", color="#1a1a2e", label="Observed (ground truth)")
    ax.plot(hours, cand, "s-", color="#e94560", label="MOS forecast")
    ax.plot(hours, nwp, "^--", color="#7f8ea3", label="Raw NWP (icon_eu)")
    ax.set_xlabel("Local hour (Europe/Rome)")
    ax.set_ylabel("Mean wind speed (kn)")
    ax.set_title("Diurnal cycle — forecast follows reality")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    bias_line = [c - o if (c is not None and o is not None) else np.nan for c, o in zip(cand, obs, strict=False)]
    nwp_bias_line = [n - o if (n is not None and o is not None) else np.nan for n, o in zip(nwp, obs, strict=False)]
    ax.bar([h - 0.2 for h in hours], bias_line, 0.4, color="#e94560", label="MOS bias")
    ax.bar([h + 0.2 for h in hours], nwp_bias_line, 0.4, color="#7f8ea3", label="Raw NWP bias")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Local hour (Europe/Rome)")
    ax.set_ylabel("Mean error (kn)")
    ax.set_title("Bias by hour — MOS should sit near zero")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(ASSET_DIR / "verify_diurnal.png", dpi=150)
    plt.close(fig)

    pl = summary["per_lead"]
    buckets = [b for b in ("0-3h", "3-6h", "6-12h", "12-24h") if b in pl]
    if buckets:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        ax = axes[0]
        x = np.arange(len(buckets))
        ax.bar(x - 0.2, [pl[b]["cand_mae"] for b in buckets], 0.4, color="#e94560", label="MOS")
        ax.bar(x + 0.2, [pl[b]["nwp_mae"] for b in buckets], 0.4, color="#7f8ea3", label="Raw NWP")
        ax.set_xticks(x, buckets)
        ax.set_xlabel("Lead time")
        ax.set_ylabel("MAE (kn)")
        ax.set_title("Skill decay with lead time")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax = axes[1]
        covs = [pl[b]["coverage_pct"] for b in buckets]
        ax.plot(x, covs, "o-", color="#0f3460")
        ax.axhline(80, color="#e94560", ls="--", label="80% contract")
        ax.set_xticks(x, buckets)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Lead time")
        ax.set_ylabel("80% band coverage (%)")
        ax.set_title("Interval coverage by lead time")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(ASSET_DIR / "verify_leadtime.png", dpi=150)
        plt.close(fig)

    ev = summary.get("event_verification") or {}
    pairs = [(k, v) for k, v in ev.items() if v.get("n")]
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 4.4), constrained_layout=True, squeeze=False)
        for ax, (k, v) in zip(axes[0], pairs, strict=False):
            bins = v["reliability"]
            xs = [b["mean_forecast_prob"] for b in bins]
            ys = [b["observed_frequency"] for b in bins]
            ns = [b["n"] for b in bins]
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
            ax.scatter(xs, ys, s=[max(20.0, min(400.0, n / 6)) for n in ns], color="#e94560", alpha=0.85)
            ax.set_xlabel("Forecast probability")
            ax.set_ylabel("Observed frequency")
            ax.set_title(f"{k}  Brier={v['brier']:.3f}  base={v['base_rate']:.2f}")
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        fig.savefig(ASSET_DIR / "verify_reliability.png", dpi=150)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["dataset", "train", "calibrate", "test", "report", "all"])
    ap.add_argument("--smoke", action="store_true", help="2 points x 3 days fast path check")
    args = ap.parse_args()

    if args.stage in ("dataset", "test", "all"):
        from lakewind.db import access as _access

        _access.set_readonly_mode(True)

    points = op_points()
    if args.smoke:
        points = points[:2]

    if args.stage == "dataset":
        stage_dataset(points)
    elif args.stage == "train":
        stage_train()
    elif args.stage == "calibrate":
        if not CAND_MARKER.exists():
            log.error("No candidate marker — run --stage train first")
            return 1
        stage_calibrate(CAND_MARKER.read_text().strip())
    elif args.stage == "test":
        if not CAND_MARKER.exists():
            log.error("No candidate marker — run --stage train first")
            return 1
        stage_test(points, CAND_MARKER.read_text().strip(), args.smoke)
    elif args.stage == "report":
        if not CAND_MARKER.exists():
            log.error("No candidate marker — run --stage train first")
            return 1
        return stage_report(points, CAND_MARKER.read_text().strip(), args.smoke)
    elif args.stage == "all":
        stage_dataset(points)
        candidate = stage_train()
        stage_calibrate(candidate)
        stage_test(points, candidate, args.smoke)
        return stage_report(points, candidate, args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
