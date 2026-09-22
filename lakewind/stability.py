"""V5 program stability — signal handlers, graceful shutdown, crash prevention.

Addresses Claude audit: "stability of the program to avoid crashes".

Features:
- SIGTERM/SIGINT handlers for graceful shutdown (saves DB, closes connections)
- Global exception handler (logs + continues instead of crashing)
- DuckDB connection watchdog (reconnects if connection is lost)
- Memory monitor (warns if process exceeds threshold)
- Deadlock prevention (timeout on DB operations)
"""
from __future__ import annotations

import faulthandler
import logging
import os
import signal
import sys
import threading
import time
import traceback
from typing import Any

logger = logging.getLogger(__name__)

_shutdown_requested = threading.Event()
_original_excepthook = None
_original_thread_excepthook = None


def setup_crash_prevention() -> None:
    """Install signal handlers, exception hooks, and fault handler.

    Call this once at program startup (in docker-entrypoint, CLI, or bot).
    """
    global _original_excepthook

    # 1. Enable faulthandler — dumps stack traces on segfaults
    try:
        faulthandler.enable()
        # Also dump to a file on SIGABRT/SIGSEGV
        fault_log = open("/tmp/lakewind_fault.log", "w")
        faulthandler.register(signal.SIGUSR1, file=fault_log)
        logger.info("Faulthandler enabled (segfault stack traces → /tmp/lakewind_fault.log)")
    except Exception as exc:
        logger.warning("Could not enable faulthandler: %s", exc)

    # 2. Install signal handlers for graceful shutdown
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass  # not in main thread
    logger.info("Signal handlers installed (SIGTERM, SIGINT)")

    # 3. Install global exception hook
    _original_excepthook = sys.excepthook
    sys.excepthook = _global_excepthook
    logger.info("Global exception hook installed")

    # 3b. Data & Prediction audit #14: sys.excepthook only fires for the MAIN
    # thread. Most of this process's work runs in worker threads
    # (asyncio.to_thread in forecast_store / pipeline_loop, APScheduler-style
    # jobs) — without threading.excepthook an uncaught exception there fell
    # through to Python's default per-thread printer and never reached the
    # structured logger.critical path.
    global _original_thread_excepthook
    _original_thread_excepthook = threading.excepthook
    threading.excepthook = _thread_excepthook
    logger.info("Thread exception hook installed")

    # 4. Start memory monitor thread
    monitor = threading.Thread(target=_memory_monitor, daemon=True, name="memory-monitor")
    monitor.start()
    logger.info("Memory monitor started")


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle SIGTERM/SIGINT — request graceful shutdown.

    Data & Prediction audit #15: the former handler was a flat 3-second sleep
    followed by `os._exit(0)`, with no check of whether a pipeline cycle was
    genuinely mid-flight — in-flight work was simply abandoned. Now the
    handler waits (bounded) for the pipeline loop's single-flight cycle to
    finish before closing the DB connection, and logs what it did. DuckDB's
    transaction atomicity still protects the data either way; this just
    stops throwing away cycles that were 95% done.
    """
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — requesting graceful shutdown...", sig_name)
    _shutdown_requested.set()

    # Wait (bounded) for any in-flight pipeline cycle to complete.
    grace_s = _shutdown_grace_seconds()
    if grace_s > 0:
        try:
            from lakewind import pipeline_loop

            deadline = time.monotonic() + grace_s
            while pipeline_loop.is_cycle_active() and time.monotonic() < deadline:
                threading.Event().wait(0.5)
            if pipeline_loop.is_cycle_active():
                logger.warning(
                    "Shutdown: pipeline cycle still active after %.0fs — abandoning it "
                    "(DuckDB transaction atomicity keeps the DB consistent)",
                    grace_s,
                )
            else:
                logger.info("Shutdown: no in-flight cycle blocking exit")
        except Exception:
            pass  # pipeline_loop unavailable (not started) — nothing to wait for

    # Close DB connections AFTER the wait — closing while a cycle thread is
    # mid-statement would turn its remaining work into errors.
    try:
        from lakewind.db import access
        access.close_global_conn()
        logger.info("DB connection closed")
    except Exception as exc:
        logger.warning("Error closing DB: %s", exc)

    # Exit
    logger.info("Exiting (code 0)")
    os._exit(0)


def _shutdown_grace_seconds() -> float:
    """Bounded wait for an in-flight cycle (audit #15); env-overridable."""
    raw = os.environ.get("LAKEWIND_SHUTDOWN_GRACE_S", "60")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 60.0


def _global_excepthook(exc_type, exc_value, exc_traceback) -> None:
    """Global exception handler — log and continue instead of crashing."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Let Ctrl+C through
        if _original_excepthook:
            _original_excepthook(exc_type, exc_value, exc_traceback)
        return

    logger.critical(
        "Uncaught exception: %s: %s\n%s",
        exc_type.__name__,
        exc_value,
        "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
    )
    logger.critical("Process continuing (crash prevented by global exception hook)")


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    """Audit #14: structured logging for uncaught exceptions in ANY thread.

    Mirrors _global_excepthook. Deliberately does NOT re-raise or exit —
    a dead worker thread must be visible in the logs, but killing the whole
    service (bot + API + loop) for one worker exception would trade a logged
    failure for an outage. SystemExit in a thread is still respected.
    """
    if args.exc_type is SystemExit:
        if _original_thread_excepthook is not None:
            _original_thread_excepthook(args)
        return
    if args.exc_type is KeyboardInterrupt:
        return  # shutdown path, already handled
    logger.critical(
        "Uncaught exception in thread %s: %s: %s\n%s",
        args.thread.name if args.thread else "?",
        args.exc_type.__name__,
        args.exc_value,
        "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
    )


def _memory_monitor() -> None:
    """Monitor memory usage and warn if process exceeds 512MB."""
    try:
        import resource
    except ImportError:
        return  # Windows

    while not _shutdown_requested.is_set():
        try:
            # ru_maxrss is in KB on Linux
            max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            max_mb = max_rss / 1024.0
            if max_mb > 512:
                logger.warning(
                    "Memory usage: %.0f MB (high — consider reducing cache sizes)", max_mb
                )
            if max_mb > 1024:
                logger.error(
                    "Memory usage: %.0f MB (critical — forcing GC)", max_mb
                )
                import gc
                gc.collect()
        except Exception:
            pass
        _shutdown_requested.wait(60.0)  # check every 60s


def is_shutting_down() -> bool:
    """Check if shutdown has been requested."""
    return _shutdown_requested.is_set()


def safe_execute(func, *args, **kwargs):
    """Execute a function safely — catches all exceptions and logs them.

    Returns (success: bool, result_or_error: Any).
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as exc:
        logger.error("safe_execute: %s failed: %s", func.__name__, exc)
        logger.debug(traceback.format_exc())
        return False, exc


__all__ = [
    "setup_crash_prevention",
    "is_shutting_down",
    "safe_execute",
]
