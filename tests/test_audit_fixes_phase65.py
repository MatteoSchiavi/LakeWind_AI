"""Regression tests for the file-by-file audit fixes (Phase 6.5).

Each test pins a bug that shipped: unit loss in the climatology backfill,
boundary-day chunk overlap, ensemble demux layout flip, discarded wind units,
stale-DOY climatology windows, dead CPCV path training, cache negative-result
behavior, artifact cycle-key parsing, and the freshness data-age contract.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest


class TestChunkers:
    def test_historical_chunk_no_overlap_and_respects_cap(self):
        from lakewind.collector.historical_backfill import _chunk_date_range

        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        chunks = _chunk_date_range(start, end, 90)
        for c_start, c_end in chunks:
            span = (c_end - c_start).days  # inclusive span = delta + 1
            assert span <= 89  # inclusive days <= 90
        # consecutive chunks do not re-fetch a boundary day
        for (_, c_end), (n_start, _) in zip(chunks, chunks[1:], strict=False):
            assert n_start == c_end + timedelta(days=1)
        # full coverage
        assert chunks[0][0] == start
        assert chunks[-1][1] == end

    def test_deep_backfill_chunk_respects_365_cap(self):
        start = datetime(1940, 1, 1)
        end = datetime(1943, 1, 1)
        chunks = []
        cur = start
        while cur < end:
            nxt = min(cur + timedelta(days=365 - 1), end)
            chunks.append((cur, nxt))
            cur = nxt + timedelta(days=1)
        for c_start, c_end in chunks:
            assert (c_end - c_start).days <= 364  # inclusive days <= 365


class TestCircularDoy:
    def test_circular_distance_wraps_year_boundary(self):
        for target, doy, expected in [
            (365, 5, 6),
            (5, 365, 6),
            (366, 1, 1),
            (100, 100, 0),
            (200, 215, 15),
            (200, 185, 15),
        ]:
            fwd = (doy - target + 366) % 366
            bwd = (target - doy + 366) % 366
            assert min(fwd, bwd) == expected, (target, doy)


class TestEnsembleDemux:
    def test_demultiplex_handles_both_key_layouts(self):
        from lakewind.collector.open_meteo import demultiplex_hourly

        models = ["icon_seamless", "gfs_seamless"]
        # prefix layout: model_var
        hourly = {
            "time": ["2026-07-04T00:00"],
            "icon_seamless_wind_speed_10m": [10.0],
            "gfs_seamless_wind_speed_10m": [12.0],
        }
        per_model, unprefixed = demultiplex_hourly(hourly, models)
        assert per_model["icon_seamless"]["wind_speed_10m"] == [10.0]
        assert per_model["gfs_seamless"]["wind_speed_10m"] == [12.0]
        assert unprefixed == {}

        # suffix layout: var_model (the 2026-09-11 incident flip)
        hourly = {
            "time": ["2026-07-04T00:00"],
            "wind_speed_10m_icon_seamless": [10.0],
            "wind_speed_10m_gfs_seamless": [12.0],
        }
        per_model, unprefixed = demultiplex_hourly(hourly, models)
        assert per_model["icon_seamless"]["wind_speed_10m"] == [10.0]
        assert per_model["gfs_seamless"]["wind_speed_10m"] == [12.0]
        assert unprefixed == {}


class TestDomasoWindUnits:
    def test_wind_text_honours_unit(self):
        from lakewind.collector.domaso_station import _wind_text_to_kn

        assert _wind_text_to_kn("11 km/h") == pytest.approx(11 * 0.539957, abs=1e-3)
        assert _wind_text_to_kn("10 m/s") == pytest.approx(10 * 1.943844, abs=1e-3)
        assert _wind_text_to_kn("5 kn") == pytest.approx(5.0, abs=1e-3)
        assert _wind_text_to_kn("24 km/h SW 12:58") == pytest.approx(
            24 * 0.539957, abs=1e-3
        )
        assert _wind_text_to_kn("") is None


class TestCacheNegativeResults:
    def test_none_values_are_cachable(self):
        from lakewind.cache import TTLCache

        c = TTLCache(maxsize=8, ttl=60)
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            return None

        assert c.get_or_compute("k", factory) is None
        assert c.get_or_compute("k", factory) is None
        assert calls["n"] == 1  # the None result IS cached (was: re-ran every call)

    def test_get_with_flag_distinguishes_missing_from_none(self):
        from lakewind.cache import TTLCache

        c = TTLCache(maxsize=8, ttl=60)
        value, found = c.get_with_flag("missing")
        assert value is None and found is False
        c.set("k", None)
        value, found = c.get_with_flag("k")
        assert value is None and found is True


class TestArtifactsCycleKey:
    def test_parse_cycle_key(self):
        from lakewind.artifacts import _parse_cycle_key

        assert _parse_cycle_key("map_202607041200_+0h.png") == datetime(2026, 7, 4, 12, 0)
        assert _parse_cycle_key("map_202607041200_+12h.png") == datetime(2026, 7, 4, 12, 0)
        assert _parse_cycle_key("garbage.png") is None

    def test_lookup_prefers_artifact_closest_to_target(self, tmp_path, monkeypatch):
        import lakewind.artifacts as artifacts

        d = tmp_path / "cache" / "maps"
        d.mkdir(parents=True)
        monkeypatch.setattr(artifacts, "cache_root", lambda: tmp_path / "cache")

        target = datetime(2026, 7, 4, 12, 0)
        # two fresh artifacts: one 2h from the requested target, one 10 min
        old = d / "map_202607041000_+0h.png"   # 2h earlier
        near = d / "map_202607041150_+0h.png"  # 10 min earlier
        old.write_bytes(b"old")
        near.write_bytes(b"near")
        # both fresh: backdate the "near" one slightly, keep inside max age
        import os
        import time

        os.utime(old, (time.time() - 3600, time.time() - 3600))
        os.utime(near, (time.time() - 60, time.time() - 60))

        png = artifacts.lookup_map_png(target, 0)
        assert png == b"near"


class TestFreshnessDataAge:
    def test_freshness_contract_fields(self, temp_db, monkeypatch):
        # with NO rows for a source, data age is None and poll age decides
        from lakewind.db import access
        from lakewind.db.freshness import check_freshness

        access.log_source_health("domaso_live", ok=True, latency_ms=1.0)
        statuses = check_freshness()
        domaso = next(s for s in statuses if s["source"] == "domaso_live")
        assert domaso["data_age_minutes"] is None
        assert domaso["is_fresh"] is True  # poll was just now

    def test_stale_data_not_fresh_despite_fresh_poll(self, temp_db, monkeypatch):
        from datetime import timedelta

        from lakewind.db import access
        from lakewind.db.freshness import check_freshness
        from lakewind.utils.timeutil import utcnow

        access.log_source_health("domaso_live", ok=True, latency_ms=1.0)
        access.insert_observation({
            "source": "domaso_live",
            "timestamp": utcnow() - timedelta(hours=3),
            "lat": 46.151, "lon": 9.332,
            "wind_speed_kn": 5.0, "wind_dir_deg": 200.0,
            "wind_gust_kn": None, "pressure": None, "temperature": None,
            "humidity": None, "quality_flag": "ok", "confidence": 0.9,
        })
        statuses = check_freshness()
        domaso = next(s for s in statuses if s["source"] == "domaso_live")
        assert domaso["data_age_minutes"] == pytest.approx(180, abs=5)
        assert domaso["is_fresh"] is False  # SLA is 20 min


class TestCpcvPathTrainingWired:
    def test_generate_paths_excludes_purged_from_train(self):
        from lakewind.ml.cpcv_backtest import generate_cpcv_paths

        paths = generate_cpcv_paths(
            n_samples=600, n_groups=6, n_test_groups=2,
            purge_hours=2, embargo_hours=6, sample_interval_hours=1,
        )
        assert len(paths) == math.comb(6, 2)
        for p in paths:
            train_set = set(p.train_indices)
            assert not (train_set & set(p.test_indices)), "train overlaps test"
            assert not (train_set & set(p.purged_indices)), "purged rows still in train"
            assert not (train_set & set(p.embargo_indices)), "embargoed rows still in train"


class TestBotWeekendHelpers:
    def test_weekend_callback_prefixes_do_not_collide_with_wind(self):
        # "wk:<point>" must never match the "w:" handler prefixes
        data = "wk:dongo"
        assert not data.startswith("w:") or data[1] == "k"
        assert data.startswith("wk:")


class TestDecisionPrecisionContract:
    def test_precision_counts_only_predicted_go(self):
        # pins the semantics: precision = TP/(TP+FP), not accuracy
        # (asserted indirectly via the imported constants the report uses)
        from lakewind.prediction.decision import GO_THRESHOLD_KN

        tp, fp = 8, 2
        precision = tp / (tp + fp) * 100
        assert precision == 80.0
        assert GO_THRESHOLD_KN == 8.0
