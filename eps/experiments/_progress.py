"""Pod-side progress reporting helpers for the Sagan dispatcher.

When a RunPod entry-point is launched by Sagan, the wrapper injects:
  - `SAGAN_PROGRESS_URL`   : HTTP endpoint accepting `{"kind": "message", "body": "..."}`
  - `SAGAN_POD_PROGRESS_TOKEN` : Bearer token authorising the POST

These can also be passed via `--progress-url` / `--progress-token` CLI flags
(see `marker_factor_screen.parse_args`). CLI flags take precedence over env
vars so a user can override the destination for local dry-runs.

The helper is intentionally tolerant: a failed POST is logged but never
aborts the experiment — progress reporting is observational, not critical.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx

log = logging.getLogger("eps.progress")

# Module-level state set by `configure()` so callers don't have to pass the
# URL/token to every `post()` call.
_progress_url: Optional[str] = None
_progress_token: Optional[str] = None
_disabled_reason: Optional[str] = None


def configure(
    progress_url: Optional[str] = None,
    progress_token: Optional[str] = None,
) -> None:
    """Configure the module from CLI flags (preferred) or env vars (fallback).

    Call once at process start. Subsequent `post()` calls are no-ops if no
    URL is available — that's the expected case for local dry-runs.
    """
    global _progress_url, _progress_token, _disabled_reason

    _progress_url = progress_url or os.environ.get("SAGAN_PROGRESS_URL")
    _progress_token = progress_token or os.environ.get("SAGAN_POD_PROGRESS_TOKEN")

    if not _progress_url:
        _disabled_reason = "no SAGAN_PROGRESS_URL set; progress posts disabled"
        log.info(_disabled_reason)
    elif not _progress_token:
        _disabled_reason = (
            "SAGAN_PROGRESS_URL set but SAGAN_POD_PROGRESS_TOKEN missing; "
            "progress posts disabled"
        )
        log.warning(_disabled_reason)
        _progress_url = None  # disable to avoid unauthenticated POSTs
    else:
        _disabled_reason = None
        log.info("Progress reporting enabled: %s", _progress_url)


def post(body: str, *, kind: str = "message", timeout_s: float = 5.0) -> bool:
    """Post a single progress event to Sagan.

    Returns True if the post succeeded, False otherwise. Failure never raises;
    we treat progress reporting as best-effort.
    """
    if _progress_url is None:
        return False

    headers = {
        "Authorization": f"Bearer {_progress_token}",
        "Content-Type": "application/json",
    }
    payload = {"kind": kind, "body": body, "ts_unix": time.time()}

    try:
        response = httpx.post(
            _progress_url,
            json=payload,
            headers=headers,
            timeout=timeout_s,
        )
        if response.status_code >= 400:
            log.warning(
                "Progress POST returned %d: %s", response.status_code, response.text[:200]
            )
            return False
        return True
    except Exception as exc:  # noqa: BLE001 — never let progress kill an experiment
        log.warning("Progress POST failed: %s", exc)
        return False


def post_milestone(milestone: str, **fields) -> bool:
    """Convenience wrapper that formats a milestone with key=value extras."""
    if not fields:
        return post(milestone)
    extras = " ".join(f"{k}={v}" for k, v in fields.items())
    return post(f"{milestone} | {extras}")
