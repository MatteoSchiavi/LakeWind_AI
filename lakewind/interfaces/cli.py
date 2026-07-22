"""LakeWind CLI (Spec §9).

```
lakewind init-db      # create DuckDB schema
lakewind doctor       # check config, DB, data source reachability
lakewind collect      # run all collectors once
lakewind predict      # generate and store current forecast
lakewind backtest     # walk-forward evaluation
lakewind retrain      # train a new candidate, write result to model_registry
lakewind promote      # human-reviewed promotion of a candidate to production
lakewind status       # data source health + last predictions summary
lakewind serve-bot    # run the Telegram bot (long-running)
lakewind serve-dashboard  # run the Streamlit dashboard
lakewind log-sailing  # add a sailing session entry (interactive)
```
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from lakewind.config import get_db_path, load_settings, load_secrets

app = typer.Typer(help="LakeWind — hyperlocal wind forecasting for Lake Como")
console = Console()


def _setup_logging(level: str = "INFO") -> None:
    s = load_settings()
    logging.basicConfig(level=level, format=s.logging.format)
    # V5: Install crash prevention (signal handlers, exception hook, memory monitor)
    try:
        from lakewind.stability import setup_crash_prevention
        setup_crash_prevention()
    except Exception:
        pass  # don't fail if stability module has issues


@app.command("init-db")
def init_db() -> None:
    """Create the DuckDB file, apply V1+V2 schema, and auto-recover any gaps."""
    from lakewind.db.schema import init_db as _init

    _init()
    # Also apply V2 schema extensions (users, alerts, subscriptions, etc.)
    try:
        from lakewind.db.schema_v2 import extend_schema_v2
        extend_schema_v2(echo=False)
        console.print("[green]V2 schema extended (users, alerts, subscriptions).[/green]")
    except Exception as exc:
        console.print(f"[yellow]V2 schema extension skipped: {exc}[/yellow]")

    # V5: Quick gap check (non-blocking — no backfill here).
    # Use `lakewind recover` to actually fill gaps.
    try:
        from lakewind.recovery import detect_gaps
        gaps = detect_gaps()
        fc = gaps.get("forecast_runs", {})
        obs = gaps.get("observations", {})
        if fc.get("needs_recovery") or obs.get("needs_recovery"):
            console.print(
                f"[yellow]⚠ Data gaps detected: forecasts {fc.get('gap_days', 0):.1f}d, "
                f"observations {obs.get('gap_days', 0):.1f}d.[/yellow]"
            )
            console.print("[cyan]Run `lakewind recover` to backfill missing data.[/cyan]")
        else:
            console.print("[green]No data gaps — everything is current.[/green]")
    except Exception as exc:
        console.print(f"[yellow]Gap check skipped: {exc}[/yellow]")


@app.command("recover")
def recover_cmd(
    check: bool = typer.Option(False, help="Dry-run: show what's missing without backfilling"),
    force: bool = typer.Option(False, help="Force full recheck of entire history"),
) -> None:
    """V5: Detect and fill data gaps (auto-backfill after downtime).

    If the T420 was down for a week, this command detects the missing period
    and automatically backfills NWP forecasts + ERA5 observations from the
    Open-Meteo Historical Forecast and Archive APIs.

    Called automatically on every startup via init-db / docker-entrypoint.
    """
    _setup_logging()
    import json
    from lakewind.recovery import recover

    console.print("[bold cyan]=== Auto-Recovery: Data Gap Detection ===[/bold cyan]")
    result = recover(check_only=check, force_full=force)

    gaps = result.get("gaps", {})
    fc = gaps.get("forecast_runs", {})
    obs = gaps.get("observations", {})

    console.print(f"\n[bold]Forecast data:[/bold]")
    console.print(f"  Latest: {fc.get('latest', 'none')}")
    console.print(f"  Gap: {fc.get('gap_days', '?')} days")
    console.print(f"  Total rows: {fc.get('count', 0)}")
    console.print(f"  Needs recovery: {'⚠ YES' if fc.get('needs_recovery') else '✓ no'}")

    console.print(f"\n[bold]Observations:[/bold]")
    console.print(f"  Latest: {obs.get('latest', 'none')}")
    console.print(f"  Gap: {obs.get('gap_days', '?')} days")
    console.print(f"  Total rows: {obs.get('count', 0)}")
    console.print(f"  Needs recovery: {'⚠ YES' if obs.get('needs_recovery') else '✓ no'}")

    if obs.get("by_source"):
        console.print(f"\n  [bold]By source:[/bold]")
        for src, info in obs["by_source"].items():
            console.print(f"    {src}: {info['count']} rows, gap {info['gap_hours']}h")

    if result.get("any_recovered"):
        rec = result.get("recovery", {})
        fc_rec = rec.get("forecasts", {})
        era5_rec = rec.get("era5", {})
        console.print(f"\n[bold green]Recovery completed:[/bold green]")
        if fc_rec.get("rows_inserted"):
            console.print(f"  Forecasts: {fc_rec['rows_inserted']} rows ({fc_rec.get('days', 0)} days)")
        if era5_rec.get("rows_inserted"):
            console.print(f"  ERA5: {era5_rec['rows_inserted']} rows ({era5_rec.get('days', 0)} days)")
    elif check:
        console.print(f"\n[yellow]Dry-run: no data was modified.[/yellow]")
    else:
        console.print(f"\n[green]No gaps needed filling.[/green]")


@app.command("doctor")
def doctor() -> None:
    """Check config, DB, data source reachability."""
    _setup_logging()
    s = load_settings()
    secrets = load_secrets()

    table = Table(title="LakeWind doctor")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Detail")

    # Config
    table.add_row("settings.yaml", "OK", f"loaded {len(s.virtual_points)} virtual points")

    # DB
    db_path = get_db_path()
    if db_path.exists():
        table.add_row("DuckDB", "OK", str(db_path))
    else:
        table.add_row("DuckDB", "MISSING", f"{db_path} — run `lakewind init-db`")

    # Secrets
    tg_token = secrets.telegram_bot_token.get_secret_value()
    table.add_row(
        "TELEGRAM_BOT_TOKEN",
        "OK" if tg_token else "MISSING",
        "set" if tg_token else "add to .env",
    )
    arpa_token = secrets.arpa_app_token.get_secret_value()
    table.add_row(
        "ARPA_APP_TOKEN",
        "OK" if arpa_token else "OPTIONAL",
        "set" if arpa_token else "low-volume use doesn't require it",
    )

    # Reachability (lightweight)
    table.add_row("Open-Meteo", _check_url(s.open_meteo.base_url), s.open_meteo.base_url)
    table.add_row("Domaso", _check_url(s.domaso.url), s.domaso.url)
    arpa_url = f"{s.arpa_lombardia.base_url}/{s.arpa_lombardia.station_dataset}.json?$limit=1"
    table.add_row("ARPA Lombardia", _check_url(arpa_url), arpa_url)

    # Virtual points
    vp_table = Table(title="Virtual points")
    vp_table.add_column("id")
    vp_table.add_column("lat")
    vp_table.add_column("lon")
    for vp in s.virtual_points:
        vp_table.add_row(vp.id, f"{vp.lat:.3f}", f"{vp.lon:.3f}")

    console.print(table)
    console.print(vp_table)


def _check_url(url: str, timeout: float = 5.0) -> str:
    import requests

    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        if r.status_code < 500:
            return "OK"
        return f"HTTP {r.status_code}"
    except Exception as exc:
        return f"FAIL ({exc.__class__.__name__})"


@app.command("collect")
def collect() -> None:
    """Run all collectors once."""
    _setup_logging()
    from lakewind.collector import run_all_collectors

    results = run_all_collectors()
    table = Table(title="Collection cycle")
    table.add_column("Source")
    table.add_column("OK")
    table.add_column("Rows")
    table.add_column("Latency (ms)")
    table.add_column("Attempts")
    table.add_column("Error")
    for r in results:
        table.add_row(
            r["source"],
            "✓" if r["ok"] else "✗",
            str(r["rows"]),
            str(r["latency_ms"]),
            str(r.get("attempts", 1)),
            r["error"] or "",
        )
    console.print(table)


@app.command("backfill")
def backfill_cmd(
    days: int = typer.Option(90, help="Days to backfill (from today backwards)"),
    start: Optional[str] = typer.Option(None, help="Start date YYYY-MM-DD (overrides --days)"),
    end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD"),
    points: Optional[str] = typer.Option(None, help="Comma-separated point ids (default: all)"),
    models: Optional[str] = typer.Option(None, help="Comma-separated NWP model slugs (default: all)"),
    era5_only: bool = typer.Option(False, help="Only backfill ERA5 reanalysis observations"),
    forecasts_only: bool = typer.Option(False, help="Only backfill historical NWP forecasts"),
) -> None:
    """Backfill historical data (Spec §4.3 Historical Forecast + ERA5 APIs).

    Pulls historical NWP forecasts (leakage-free, exactly what was knowable
    at each past decision time) and ERA5 reanalysis (high-quality ground truth)
    for the requested date range. Chunks at 90 days per request to respect
    Open-Meteo's limit.
    """
    _setup_logging()
    from datetime import timedelta

    from lakewind.collector.historical_backfill import backfill_era5, backfill_forecasts

    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.utcnow()
    if start:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
    else:
        start_dt = end_dt - timedelta(days=days)

    pt_list = points.split(",") if points else None
    mdl_list = models.split(",") if models else None

    console.print(f"[bold]Backfilling {start_dt.date()} to {end_dt.date()}[/bold]")
    if not era5_only:
        console.print("\n[cyan]Historical forecasts...[/cyan]")
        summary = backfill_forecasts(
            start=start_dt, end=end_dt, points=pt_list, models=mdl_list
        )
        total = sum(summary.values())
        console.print(f"  Total forecast rows inserted: {total}")
        for k, v in summary.items():
            console.print(f"    {k}: {v}")
    if not forecasts_only:
        console.print("\n[cyan]ERA5 reanalysis...[/cyan]")
        summary = backfill_era5(start=start_dt, end=end_dt, points=pt_list)
        total = sum(summary.values())
        console.print(f"  Total ERA5 rows inserted: {total}")
        for k, v in summary.items():
            console.print(f"    {k}: {v}")


@app.command("predict")
def predict(
    horizons: Optional[str] = typer.Option(None, help="Comma-separated hours, e.g. 0,1,3,6,24"),
    no_collect: bool = typer.Option(False, help="Skip collector step"),
) -> None:
    """Generate and store current forecast."""
    _setup_logging()
    from lakewind.prediction.engine import run_cycle

    hrs = [int(x) for x in horizons.split(",")] if horizons else None
    summary = run_cycle(collect=not no_collect, horizons_hours=hrs)
    console.print(f"[bold]Status:[/bold] {summary['status']}")
    console.print(f"[bold]Runtime:[/bold] {summary.get('runtime_seconds', 'n/a')}s "
                  f"(target {summary.get('target_runtime_seconds', 'n/a')}s)")
    console.print(f"[bold]Forecasts:[/bold] {summary.get('n_forecasts', 0)}")
    if "collectors" in summary:
        console.print("\n[bold]Collectors:[/bold]")
        for c in summary["collectors"]:
            console.print(f"  {c['source']}: {'OK' if c['ok'] else 'FAIL'} ({c['rows']} rows)")
    if summary.get("forecasts"):
        table = Table(title="Forecasts")
        table.add_column("Point")
        table.add_column("Valid")
        table.add_column("Speed (kn)")
        table.add_column("Dir (°)")
        table.add_column("Gust (kn)")
        table.add_column("Conf (%)")
        table.add_column("Err (kn)")
        for fc in summary["forecasts"]:
            table.add_row(
                fc["point_id"],
                fc["valid_time"],
                f"{fc['wind_speed_kn']:.1f}",
                f"{fc['wind_dir_deg']:.0f}",
                f"{fc['wind_gust_kn']:.1f}",
                f"{fc['confidence_pct']:.0f}",
                f"{fc['expected_error_kn']:.1f}",
            )
        console.print(table)


@app.command("backtest")
def backtest_cmd(
    days: int = typer.Option(90, help="Total backtest span in days"),
    candidate: Optional[str] = typer.Option(None, help="Existing model version to evaluate"),
    promote: bool = typer.Option(False, help="Programmatically promote if it clears the gate"),
) -> None:
    """Walk-forward backtest against persistence + raw NWP + current production."""
    _setup_logging()
    from lakewind.ml.backtest import maybe_promote, run_backtest

    end = datetime.utcnow()
    start = end - timedelta(days=days)
    report = run_backtest(candidate_model_version=candidate, start=start, end=end)

    console.print(f"\n[bold]Backtest report for {report.candidate_model_version}[/bold]")
    console.print(f"  Test samples:        {report.n_test_samples}")
    console.print(f"  Candidate MAE:       {report.candidate_mae_kn} kn  / dir {report.candidate_dir_error_deg}°")
    console.print(f"  Persistence MAE:     {report.persistence_mae_kn} kn  / dir {report.persistence_dir_error_deg}°")
    console.print(f"  Raw NWP MAE:         {report.raw_nwp_mae_kn} kn  / dir {report.raw_nwp_dir_error_deg}°")
    console.print(f"  MAE reduction vs raw NWP:        {report.mae_reduction_vs_raw_nwp_pct}%")
    console.print(f"  MAE reduction vs persistence:    {report.mae_reduction_vs_persistence_pct}%")
    console.print(f"  Direction error reduction:       {report.dir_error_reduction_pct}%")
    console.print(f"  80% interval coverage:           {report.confidence_interval_coverage_pct}%")
    console.print(f"  Decision precision:              {report.decision_precision_pct}%")
    console.print(f"  [bold]Success criteria met: {report.success_criteria_met}[/bold]")

    # V5: Source-separated metrics (Claude audit: separate vs-ERA5 and vs-real)
    console.print(f"\n[bold cyan]Observation source breakdown:[/bold cyan]")
    console.print(f"  ERA5 reanalysis samples:         {report.n_era5_samples}")
    console.print(f"  Real station samples:            {report.n_real_samples}")
    if report.candidate_mae_vs_era5 is not None:
        console.print(f"  Candidate MAE vs ERA5:           {report.candidate_mae_vs_era5} kn")
    if report.candidate_mae_vs_real is not None:
        console.print(f"  Candidate MAE vs real stations:  {report.candidate_mae_vs_real} kn")
    if report.n_era5_samples > report.n_real_samples:
        console.print(f"  [yellow]⚠ Most metrics are vs ERA5 (inter-model bias), not real observations.[/yellow]")
        console.print(f"    Deploy DIY buoy for real ground truth.")

    console.print("\n[bold]Per-regime:[/bold]")
    rt = Table()
    rt.add_column("Regime")
    rt.add_column("n")
    rt.add_column("Cand MAE")
    rt.add_column("Pers MAE")
    rt.add_column("NWP MAE")
    for regime, m in report.per_regime.items():
        rt.add_row(
            regime,
            str(int(m["n"])),
            f"{m['cand_mae_kn']:.2f}",
            f"{m['pers_mae_kn']:.2f}",
            f"{m['nwp_mae_kn']:.2f}",
        )
    console.print(rt)

    if promote:
        promoted = maybe_promote(report, force=False)
        console.print(f"\n[bold]Promoted:[/bold] {promoted}")


@app.command("retrain")
def retrain(
    days: int = typer.Option(60, help="Training window in days"),
    backend: Optional[str] = typer.Option(
        None, help="Backend: 'lightgbm' or 'xgboost_gpu' (default: from settings.yaml)"
    ),
) -> None:
    """Train a new candidate, write result to model_registry."""
    _setup_logging()
    from lakewind.ml.train import train

    end = datetime.utcnow()
    start = end - timedelta(days=days)
    result = train(start=start, end=end, backend=backend)
    if result is None:
        console.print("[red]Not enough training data. Run `lakewind collect` or `lakewind backfill` first.[/red]")
        raise typer.Exit(code=1)
    console.print(f"[bold]Trained:[/bold] {result.model_version}")
    console.print(f"  Backend:  {result.backend}")
    console.print(f"  Samples:  {result.n_samples}")
    console.print(f"  Features: {result.n_features}")
    console.print(f"  Quantiles: {result.quantiles}")
    console.print("  Metrics:")
    for k, v in result.metrics.items():
        console.print(f"    {k}: {v:.4f}")


@app.command("promote")
def promote_cmd(
    model_version: str = typer.Argument(..., help="Model version to promote to production"),
) -> None:
    """Promote a candidate model to production (human review).

    Spec §7.3: human-reviewed promotion is the default. The upgrade gate
    (min_mae_improvement_kn, min_dir_improvement_deg) should be checked
    via `lakewind backtest --promote` first. This command is the manual
    override for when you've reviewed the backtest report and decided to
    promote regardless.

    Stores NULL for backtest metrics (not 0.0) so the upgrade gate
    remains functional for future candidates.
    """
    _setup_logging()
    from lakewind.db import access

    s = load_settings()
    with access.cursor() as conn:
        conn.execute(
            f"UPDATE {s.db.model_registry_table} SET promoted_to_production = FALSE "
            "WHERE promoted_to_production = TRUE"
        )
    access.register_model(
        model_version=model_version,
        trained_at=datetime.utcnow(),
        feature_set_version=s.model.feature_set_version,
        training_start=None,
        training_end=None,
        backtest_mae_kn=None,  # NULL, not 0.0 — preserves upgrade gate
        backtest_dir_error_deg=None,
        promoted=True,
        notes="Manually promoted (human review)",
    )
    console.print(f"[bold green]Promoted {model_version} to production.[/bold green]")


@app.command("status")
def status() -> None:
    """Data source health + latest predictions summary."""
    _setup_logging()
    from lakewind.db import access

    health = access.latest_source_health()
    if health:
        h_table = Table(title="Source health (latest)")
        h_table.add_column("Source")
        h_table.add_column("OK")
        h_table.add_column("Latency (ms)")
        h_table.add_column("Checked at")
        h_table.add_column("Error")
        for h in health:
            h_table.add_row(
                h["source"],
                "✓" if h["ok"] else "✗",
                str(round(h["latency_ms"], 1)),
                str(h["checked_at"]),
                h["error_msg"] or "",
            )
        console.print(h_table)
    else:
        console.print("[yellow]No source health recorded yet.[/yellow]")

    preds = access.latest_predictions(limit=20)
    if preds:
        p_table = Table(title="Latest predictions")
        p_table.add_column("Point")
        p_table.add_column("Valid")
        p_table.add_column("Speed (kn)")
        p_table.add_column("Dir (°)")
        p_table.add_column("Conf (%)")
        for p in preds:
            p_table.add_row(
                p["point_id"],
                str(p["valid_time"]),
                f"{p['wind_speed_kn']:.1f}" if p["wind_speed_kn"] is not None else "—",
                f"{p['wind_dir_deg']:.0f}" if p["wind_dir_deg"] is not None else "—",
                f"{p['confidence_pct']:.0f}" if p["confidence_pct"] is not None else "—",
            )
        console.print(p_table)
    else:
        console.print("[yellow]No predictions stored yet.[/yellow]")


@app.command("serve-bot")
def serve_bot() -> None:
    """Run the Telegram bot (long-running)."""
    _setup_logging()
    from lakewind.interfaces.telegram_bot import run_bot

    run_bot()


@app.command("serve-dashboard")
def serve_dashboard(
    port: Optional[int] = typer.Option(None, help="Port override"),
) -> None:
    """Run the Streamlit dashboard (long-running)."""
    _setup_logging()
    import subprocess

    s = load_settings()
    p = port or s.streamlit.port
    dashboard_path = Path(__file__).parent / "dashboard.py"
    console.print(f"[bold]Starting Streamlit on port {p}...[/bold]")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port",
            str(p),
        ],
        check=False,
    )


@app.command("log-sailing")
def log_sailing(
    point: str = typer.Option(..., help="Virtual point id near the session"),
    wind_kn: float = typer.Option(..., help="Perceived sustained wind (kn)"),
    direction: float = typer.Option(..., help="Perceived direction (deg)"),
    sail: str = typer.Option("", help="Sail configuration"),
    notes: str = typer.Option("", help="Free-form notes"),
    duration_min: int = typer.Option(60, help="Session duration in minutes"),
) -> None:
    """Add a sailing session log entry (Spec §4.5)."""
    _setup_logging()
    from lakewind.db import access

    start = datetime.utcnow() - timedelta(minutes=duration_min)
    end = datetime.utcnow()
    rid = access.insert_sailing_log(
        {
            "session_start": start,
            "session_end": end,
            "point_id": point,
            "perceived_wind_kn": wind_kn,
            "perceived_direction_deg": direction,
            "sail_config": sail,
            "notes": notes,
            "gps_track_path": None,
        }
    )
    console.print(f"[bold green]Logged sailing session #{rid}.[/bold green]")


@app.command("dump-forecasts")
def dump_forecasts(
    point: Optional[str] = typer.Option(None, help="Filter by point"),
    limit: int = typer.Option(50, help="Max rows"),
) -> None:
    """Dump latest stored forecasts as JSON (debugging)."""
    _setup_logging()
    from lakewind.db import access

    preds = access.latest_predictions(point_id=point, limit=limit)
    print(json.dumps(preds, default=str, indent=2))


# --- V2 commands (registered on the same Typer app) ---

try:
    from lakewind.interfaces.cli_v2 import register_v2_commands
    register_v2_commands(app)
except ImportError:
    pass  # V2 modules not available yet


if __name__ == "__main__":  # pragma: no cover
    app()
