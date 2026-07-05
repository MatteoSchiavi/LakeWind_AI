"""V2 CLI additions — register V2 commands alongside V1 commands.

Adds:
  lakewind init-db-v2          - extend V1 schema with V2 tables
  lakewind predict-v2          - run V2 prediction cycle (Kalman+LGB blend)
  lakewind train-regime        - train the regime classifier
  lakewind update-kalman       - manually update Kalman state from latest obs
  lakewind user-add <tg_id>    - whitelist a Telegram user
  lakewind user-list           - list whitelisted users
  lakewind user-block <tg_id>  - block a user
  lakewind serve-bot-v2        - run the V2 Telegram bot (22 commands + alerts)
  lakewind alerts-check        - manually run alert checks (debug)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from lakewind.config import load_settings

console = Console()


def register_v2_commands(app: typer.Typer) -> None:
    """Register V2 commands on the given Typer app."""

    @app.command("init-db-v2")
    def init_db_v2() -> None:
        """Extend V1 schema with V2 tables (users, alerts, subscriptions, etc.)."""
        from lakewind.db.schema_v2 import extend_schema_v2
        extend_schema_v2()

    @app.command("predict-v2")
    def predict_v2(
        horizons: Optional[str] = typer.Option(None, help="Comma-separated hours"),
        no_collect: bool = typer.Option(False, help="Skip collector step"),
        no_kalman: bool = typer.Option(False, help="Skip Kalman state update"),
    ) -> None:
        """Run V2 prediction cycle (Kalman+LGB blend)."""
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        from lakewind.prediction.engine_v2 import run_cycle_v2
        hrs = [int(x) for x in horizons.split(",")] if horizons else None
        summary = run_cycle_v2(collect=not no_collect, horizons_hours=hrs,
                               update_kalman=not no_kalman)
        console.print(f"[bold]Status:[/bold] {summary['status']}")
        console.print(f"[bold]Runtime:[/bold] {summary.get('runtime_seconds', 'n/a')}s")
        console.print(f"[bold]Engine:[/bold] {summary.get('engine', 'v2')}")
        console.print(f"[bold]Forecasts:[/bold] {summary.get('n_forecasts', 0)}")
        if summary.get("forecasts"):
            table = Table(title="V2 Forecasts")
            table.add_column("Point")
            table.add_column("Valid")
            table.add_column("Speed (kn)")
            table.add_column("Dir (°)")
            table.add_column("Conf (%)")
            for fc in summary["forecasts"]:
                table.add_row(
                    fc["point_id"],
                    str(fc["valid_time"]),
                    f"{fc['wind_speed_kn']:.1f}",
                    f"{fc['wind_dir_deg']:.0f}",
                    f"{fc['confidence_pct']:.0f}",
                )
            console.print(table)

    @app.command("train-regime")
    def train_regime(
        days: int = typer.Option(60, help="Training window in days"),
    ) -> None:
        """Train the V2 regime classifier."""
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        from lakewind.ml.regime import train_regime_classifier
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        ok = train_regime_classifier(start, end)
        if ok:
            console.print("[green]Regime classifier trained.[/green]")
        else:
            console.print("[red]Failed (not enough data).[/red]")
            raise typer.Exit(1)

    @app.command("update-kalman")
    def update_kalman() -> None:
        """Update Kalman filter state from latest observations."""
        logging.basicConfig(level=logging.INFO)
        from lakewind.config import load_settings
        from lakewind.ml.kalman import update_from_latest_observations
        s = load_settings()
        for vp_id in (s.operational_point_ids or []):
            state = update_from_latest_observations(vp_id)
            if state:
                console.print(
                    f"[green]{vp_id}[/green]: bias_u={state.bias_u:.3f}, "
                    f"bias_v={state.bias_v:.3f}, P=[{state.p_uu:.3f}, {state.p_vv:.3f}]"
                )
            else:
                console.print(f"[yellow]{vp_id}[/yellow]: no recent observation")

    @app.command("user-add")
    def user_add(
        telegram_user_id: int = typer.Argument(..., help="Telegram user ID"),
        username: str = typer.Option("", help="Telegram @username"),
        is_admin: bool = typer.Option(False, help="Make this user an admin"),
    ) -> None:
        """Whitelist a Telegram user."""
        from lakewind.db import users as user_db
        user = user_db.register_or_update_user(
            telegram_user_id=telegram_user_id,
            username=username,
        )
        user_db.set_user_preference(telegram_user_id, "is_allowed", True)
        if is_admin:
            user_db.set_user_preference(telegram_user_id, "is_admin", True)
        console.print(f"[green]User {telegram_user_id} added[/green]")

    @app.command("user-list")
    def user_list() -> None:
        """List all registered users."""
        from lakewind.db import users as user_db
        users = user_db.list_allowed_users()
        if not users:
            console.print("[yellow]No users registered yet (open mode).[/yellow]")
            return
        t = Table(title="Registered users")
        t.add_column("TG ID")
        t.add_column("Username")
        t.add_column("Lang")
        t.add_column("TZ")
        t.add_column("Fav point")
        t.add_column("Admin")
        for u in users:
            t.add_row(
                str(u["telegram_user_id"]),
                u.get("username", ""),
                u.get("language", "en"),
                u.get("timezone", "Europe/Rome"),
                u.get("favorite_point_id", "") or "",
                "✓" if u.get("is_admin") else "",
            )
        console.print(t)

    @app.command("user-block")
    def user_block(
        telegram_user_id: int = typer.Argument(...),
    ) -> None:
        """Block a Telegram user."""
        from lakewind.db import users as user_db
        ok = user_db.set_user_preference(telegram_user_id, "is_allowed", False)
        if ok:
            console.print(f"[green]User {telegram_user_id} blocked.[/green]")
        else:
            console.print(f"[red]User {telegram_user_id} not found.[/red]")
            raise typer.Exit(1)

    @app.command("serve-bot-v2")
    def serve_bot_v2() -> None:
        """Run the V2 Telegram bot (22 commands + alert scheduler)."""
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        from lakewind.interfaces.telegram_bot import run_bot
        run_bot()

    @app.command("alerts-check")
    def alerts_check() -> None:
        """Manually run alert checks (debug)."""
        import asyncio
        logging.basicConfig(level=logging.INFO)
        from lakewind.interfaces.bot_scheduler import run_scheduler
        # Mock context with bot
        from telegram import Bot
        from lakewind.config import load_secrets
        secrets = load_secrets()
        token = secrets.telegram_bot_token.get_secret_value()
        if not token:
            console.print("[red]TELEGRAM_BOT_TOKEN not set[/red]")
            raise typer.Exit(1)
        bot = Bot(token=token)

        class _Ctx:
            def __init__(self, bot):
                self.bot = bot

        asyncio.run(run_scheduler(_Ctx(bot)))
        console.print("[green]Alert check done.[/green]")

    # --- V3 commands ---

    @app.command("retrain-stacked")
    def retrain_stacked(
        days: int = typer.Option(60, help="Training window in days"),
    ) -> None:
        """Train a V3 stacked ensemble (LGB + XGBoost + MLP + Ridge + Isotonic)."""
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        from lakewind.ml.stacking import train_stacked_ensemble
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        result = train_stacked_ensemble(start=start, end=end)
        if result is None:
            console.print("[red]Not enough data for stacked training.[/red]")
            raise typer.Exit(1)
        console.print(f"[bold green]Stacked ensemble trained:[/bold green] {result['model_version']}")
        console.print(f"  Samples:  {result['n_samples']}")
        console.print(f"  Features: {result['n_features']}")
        console.print(f"  Quantiles: {result['quantiles']}")
        console.print("  Metrics:")
        for k, v in result["metrics"].items():
            console.print(f"    {k}: {v:.4f}")

    @app.command("collect-v3")
    def collect_v3(
        skip_lake_temp: bool = typer.Option(False, help="Skip lake water temp collector"),
    ) -> None:
        """Run all collectors including V3 sources (Holfuy, lake water temp)."""
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        from lakewind.collector import run_all_collectors
        results = run_all_collectors()
        table = Table(title="V3 Collection cycle")
        table.add_column("Source")
        table.add_column("OK")
        table.add_column("Rows")
        table.add_column("Latency (ms)")
        for r in results:
            table.add_row(
                r["source"],
                "✓" if r["ok"] else "✗",
                str(r["rows"]),
                str(r["latency_ms"]),
            )
        console.print(table)

    @app.command("features-info")
    def features_info(
        point: str = typer.Option("mid_channel", help="Point to inspect"),
    ) -> None:
        """Show all features that would be built for a point (V3 debug)."""
        logging.basicConfig(level=logging.WARNING)
        from lakewind.features.build import build_features_for
        from datetime import datetime as dt
        fr = build_features_for(point, dt.utcnow())
        if fr is None:
            console.print("[red]No data for this point.[/red]")
            raise typer.Exit(1)
        console.print(f"[bold]Feature set for {point}[/bold] ({len(fr.feature_vector)} features):")
        # Group by prefix
        groups: dict[str, list[str]] = {}
        for k, v in sorted(fr.feature_vector.items()):
            prefix = k.split("_")[0] if "_" in k else "other"
            if k.startswith("fc_"):
                prefix = "forecast"
            elif k.startswith("agree_"):
                prefix = "agreement"
            elif k.startswith("ens_"):
                prefix = "ensemble"
            elif k.startswith("foehn"):
                prefix = "foehn"
            elif k.startswith("solar"):
                prefix = "solar"
            elif k.startswith("thermal"):
                prefix = "thermal_inertia"
            elif k.startswith("pressure_grad"):
                prefix = "macro_area_pressure"
            elif k.startswith("stability"):
                prefix = "stability"
            elif k.startswith("lake_breeze"):
                prefix = "lake_breeze"
            elif k.startswith("lag"):
                prefix = "persistence"
            elif k.startswith("obs_"):
                prefix = "ground_station"
            elif k in ("hour_local", "day_of_year", "month", "season", "is_weekend"):
                prefix = "temporal"
            groups.setdefault(prefix, []).append(f"{k} = {v}")
        for group, items in sorted(groups.items()):
            console.print(f"\n[cyan]{group}[/cyan] ({len(items)} features):")
            for item in items:
                console.print(f"  {item}")
        console.print(f"\n[bold]Target:[/bold] u={fr.target_u}, v={fr.target_v}")


__all__ = ["register_v2_commands"]
