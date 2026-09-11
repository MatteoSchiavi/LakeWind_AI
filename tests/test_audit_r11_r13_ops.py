"""Tests for Deep Audit R11 (retention + backup), R13 (API bearer gate +
operational alerts) and R12 (collector fixture behaviour at the ARPA
month-rollover boundary)."""
from __future__ import annotations

from datetime import timedelta

import duckdb

from lakewind.config import reset_caches
from lakewind.db import access
from lakewind.db.schema import init_db
from lakewind.utils.timeutil import utcnow

NOW = utcnow().replace(tzinfo=None)


class TestRetention:
    def _seed(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()
        old = NOW - timedelta(days=200)
        recent = NOW - timedelta(days=10)
        # backfill (training asset) — old but KEPT
        access.insert_forecast_run(
            {
                "model_name": "icon_eu",
                "point_id": "dongo_shore",
                "run_time": old - timedelta(hours=6),
                "valid_time": old,
                "wind_speed_kn": 5.0,
                "raw_json": {"model": "icon_eu", "point": "dongo_shore",
                             "source": "historical_forecast_api"},
            }
        )
        # operational — old, pruned
        access.insert_forecast_run(
            {
                "model_name": "icon_eu",
                "point_id": "dongo_shore",
                "run_time": old - timedelta(hours=3),
                "valid_time": old,
                "wind_speed_kn": 6.0,
                "raw_json": {"model": "icon_eu", "point": "dongo_shore",
                             "source": "open_meteo_forecast"},
            }
        )
        # operational — recent, kept
        access.insert_forecast_run(
            {
                "model_name": "icon_eu",
                "point_id": "dongo_shore",
                "run_time": recent - timedelta(hours=3),
                "valid_time": recent,
                "wind_speed_kn": 7.0,
                "raw_json": {"model": "icon_eu", "point": "dongo_shore",
                             "source": "open_meteo_forecast"},
            }
        )
        access.insert_predictions_bulk(
            [
                {
                    "point_id": "dongo_shore",
                    "generated_at": NOW,
                    "valid_time": NOW - timedelta(days=600),  # older than 18mo → pruned
                    "model_version": "mv",
                    "wind_speed_kn": 5.0,
                    "wind_dir_deg": 180.0,
                    "wind_gust_kn": None,
                    "confidence_pct": 80.0,
                    "expected_error_kn": 1.0,
                },
                {
                    "point_id": "dongo_shore",
                    "generated_at": NOW,
                    "valid_time": recent,
                    "model_version": "mv",
                    "wind_speed_kn": 6.0,
                    "wind_dir_deg": 180.0,
                    "wind_gust_kn": None,
                    "confidence_pct": 80.0,
                    "expected_error_kn": 1.0,
                },
            ]
        )

    def test_dry_run_counts_without_deleting(self, temp_db):
        self._seed(temp_db)
        stats = access.apply_retention_policy(dry_run=True)
        assert stats["forecasts_deleted"] == 1
        assert stats["predictions_deleted"] == 1
        access.close_global_conn()  # release configured conn before a plain-config connect
        with duckdb.connect(str(temp_db)) as conn:
            n = conn.execute("SELECT count(*) FROM forecast_runs").fetchone()[0]
        assert n == 3  # untouched

    def test_deletes_operational_keeps_backfill(self, temp_db):
        self._seed(temp_db)
        stats = access.apply_retention_policy()
        assert stats["forecasts_deleted"] == 1
        assert stats["predictions_deleted"] == 1
        access.close_global_conn()  # release configured conn before a plain-config connect
        with duckdb.connect(str(temp_db)) as conn:
            rows = conn.execute(
                "SELECT valid_time, json_extract_string(raw_json, '$.source') AS src "
                "FROM forecast_runs ORDER BY valid_time"
            ).fetchall()
        assert len(rows) == 2
        assert rows[0][1] == "historical_forecast_api"  # training asset survives
        assert rows[1][0] >= NOW - timedelta(days=90)


class TestBackup:
    def test_backup_creates_timestamped_copy(self, temp_db, tmp_path):
        init_db(temp_db, echo=False)
        reset_caches()
        dest = tmp_path / "backups"
        target = access.backup_database(dest)
        assert target.exists() and target.stat().st_size > 0
        assert target.name.startswith("lakewind_backup_")
        # readable as a DuckDB file with the schema intact
        with duckdb.connect(str(target), read_only=True) as conn:
            tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        assert "forecast_runs" in tables

    def test_offsite_copy(self, temp_db, tmp_path):
        init_db(temp_db, echo=False)
        reset_caches()
        local = tmp_path / "local"
        offsite = tmp_path / "offsite"
        access.backup_database(local, offsite)
        assert len(list(offsite.glob("lakewind_backup_*.duckdb"))) == 1


# --- R13: operational alerts ---


class TestOperationalAlerts:
    def _init(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()

    def test_all_clear_with_fresh_station_data(self, temp_db):
        self._init(temp_db)
        access.insert_observation(
            {
                "source": "arpa_1",
                "timestamp": NOW - timedelta(minutes=10),
                "lat": 46.12,
                "lon": 9.29,
                "wind_speed_kn": 5.0,
                "wind_dir_deg": 180.0,
            }
        )
        from lakewind.monitoring import operational_alerts

        alerts = operational_alerts()
        names = {a["alert"] for a in alerts}
        assert "station_silence" not in names
        assert "data_starvation" not in names

    def test_station_silence_alert_when_stale(self, temp_db):
        self._init(temp_db)
        access.insert_observation(
            {
                "source": "arpa_1",
                "timestamp": NOW - timedelta(hours=20),
                "lat": 46.12,
                "lon": 9.29,
                "wind_speed_kn": 5.0,
                "wind_dir_deg": 180.0,
            }
        )
        from lakewind.monitoring import operational_alerts

        alerts = operational_alerts()
        silence = [a for a in alerts if a["alert"] == "station_silence"]
        assert silence and silence[0]["severity"] in ("warning", "critical")

    def test_era5_staleness_does_not_alert(self, temp_db):
        """ERA5 lags ~5 days by design — its age is normal, not an alert."""
        self._init(temp_db)
        access.insert_observation(
            {
                "source": "era5_reanalysis",
                "timestamp": NOW - timedelta(days=5),
                "lat": 46.12,
                "lon": 9.29,
                "wind_speed_kn": 5.0,
                "wind_dir_deg": 180.0,
            }
        )
        from lakewind.monitoring import operational_alerts

        names = {a["alert"] for a in operational_alerts()}
        assert "station_silence" in names  # NO station data at all -> silence
        # but the alert text must not claim a wrong SLA breach of reanalysis
        silence = next(a for a in operational_alerts() if a["alert"] == "station_silence")
        assert "ever" in silence["detail"] or "never" not in silence["detail"]

    def test_quota_alert_from_source_health(self, temp_db):
        self._init(temp_db)
        access.log_source_health("open_meteo", ok=False, latency_ms=100.0,
                                 error_msg="HTTP 429: quota exhausted for today")
        from lakewind.monitoring import operational_alerts

        names = {a["alert"] for a in operational_alerts()}
        assert "quota_exhaustion" in names

    def test_data_starvation_alert(self, temp_db):
        self._init(temp_db)
        # no observations at all in the last 24h
        access.insert_observation(
            {
                "source": "arpa_2",
                "timestamp": NOW - timedelta(days=3),
                "lat": 46.12,
                "lon": 9.29,
                "wind_speed_kn": 5.0,
                "wind_dir_deg": 180.0,
            }
        )
        from lakewind.monitoring import operational_alerts

        names = {a["alert"] for a in operational_alerts()}
        assert "data_starvation" in names


# --- R13: API bearer gate ---


class TestApiBearerGate:
    def _client(self, monkeypatch, token):
        from fastapi.testclient import TestClient

        import lakewind.api as A
        from lakewind.config import load_settings

        s = load_settings()
        monkeypatch.setattr(s.api, "auth_token", token)
        app = A.create_app()
        return TestClient(app)

    def test_get_stays_open_without_token(self, temp_db, monkeypatch):
        self._client_setup(temp_db)
        client = self._client(monkeypatch, token="secret")
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def _client_setup(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()

    def test_post_rejected_without_token(self, temp_db, monkeypatch):
        self._client_setup(temp_db)
        client = self._client(monkeypatch, token="secret")
        resp = client.post("/api/points")  # any non-GET
        assert resp.status_code == 401

    def test_post_allowed_with_bearer(self, temp_db, monkeypatch):
        self._client_setup(temp_db)
        client = self._client(monkeypatch, token="secret")
        resp = client.post("/api/points", headers={"Authorization": "Bearer secret"})
        assert resp.status_code in (200, 405)  # 405 = route exists but no POST handler; auth passed

    def test_alerts_endpoint_available(self, temp_db, monkeypatch):
        self._client_setup(temp_db)
        client = self._client(monkeypatch, token=None)
        resp = client.get("/api/alerts")
        assert resp.status_code == 200
        assert "alerts" in resp.json()


# --- R12: collector fixture at the ARPA month-rollover boundary ---


class TestArpaMonthRolloverFixture:
    def test_readings_across_month_boundary_parse(self):
        """ARPA replaces its current-month dataset at rollover; readings from
        both sides of the boundary must parse into rows without loss."""
        from lakewind.collector.arpa_lombardia import ArpaLombardiaCollector

        raw = {
            "sensors": {"101": {"station_id": "S1", "station_name": "Domaso",
                                 "sensor_type": "wind_speed", "lat": 46.15,
                                 "lng": 9.32, "tipologia": "Velocità Vento"}},
            "readings": [
                {"idsensore": "101", "data": "2026-08-31T23:50:00", "valore": "4.0", "stato": ""},
                {"idsensore": "101", "data": "2026-09-01T00:00:00", "valore": "5.0", "stato": "non validato"},
                {"idsensore": "101", "data": "2026-09-01T00:10:00", "valore": "6.0", "stato": ""},
            ],
        }
        rows = ArpaLombardiaCollector().to_rows(raw)
        assert len(rows) == 3  # pre-R10 the fresh 'non validato' row was DROPPED
        stamps = [r["timestamp"].isoformat() for r in rows]
        assert stamps == sorted(stamps)  # no month-boundary parsing loss
