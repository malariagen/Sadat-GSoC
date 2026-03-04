"""
Shared helpers used by every other module.

- log()             : thread-safe print (safe to call from parallel workers)
- NonRetryableError : raise this to stop retrying a sample immediately
- utc_now_iso()     : current UTC time as a string
- make_run_id()     : random 12-char hex ID for tagging a pipeline run
"""

from __future__ import annotations

import datetime as _dt
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
