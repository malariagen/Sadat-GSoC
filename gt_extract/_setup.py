"""
Shared helpers used by every other module.

- log()             : thread-safe print (safe to call from parallel workers)
- NonRetryableError : raise this to stop retrying a sample immediately
- utc_now_iso()     : current UTC time as a string
- make_run_id()     : random 12-char hex ID for tagging a pipeline run
- shutdown_requested() : check if Ctrl+C was pressed
- request_shutdown()   : set the shutdown flag
- install_signal_handler() : register a clean Ctrl+C handler
"""

from __future__ import annotations

import datetime as _dt
import signal
import threading
import uuid

import urllib3

# Suppress the "Unverified HTTPS request" warning.
# We disable SSL verification for the Sanger data server (experimental).
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


_PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    """Print a message to the console, safe to call from multiple threads at once."""
    with _PRINT_LOCK:
        print(msg, flush=True)


class NonRetryableError(RuntimeError):
    """An error that should NOT be retried.

    Use this for problems like a broken archive or missing data,
    i.e. things that will not fix themselves on retry.
    """


def utc_now_iso() -> str:
    """Get the current UTC time as a short ISO string like '2026-03-04T16:15:00Z'."""
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_run_id() -> str:
    """Generate a random 12-character hex string to identify this pipeline run."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Cooperative shutdown (Ctrl+C support)
# ---------------------------------------------------------------------------

_SHUTDOWN_EVENT = threading.Event()


def request_shutdown() -> None:
    """Set the shutdown flag. All pipeline loops will stop at their next check."""
    _SHUTDOWN_EVENT.set()


def shutdown_requested() -> bool:
    """Return True if shutdown has been requested (Ctrl+C was pressed)."""
    return _SHUTDOWN_EVENT.is_set()


def install_signal_handler() -> None:
    """Register a SIGINT handler for clean Ctrl+C shutdown.

    First Ctrl+C  : sets the shutdown flag and logs a message.
    Second Ctrl+C : restores the default handler so the process dies immediately.
    """
    def _handler(signum, frame):
        if shutdown_requested():
            # Second press: restore default (hard-kill on next Ctrl+C).
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            log("Second Ctrl+C received, forcing exit.")
            raise KeyboardInterrupt
        request_shutdown()
        log("Shutdown requested (Ctrl+C). Finishing in-flight work, please wait...")

    signal.signal(signal.SIGINT, _handler)
