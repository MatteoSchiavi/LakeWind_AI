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


__all__ = ["register_v2_commands"]
