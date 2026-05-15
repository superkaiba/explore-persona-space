"""Pod-side progress reporting helpers for task #365.

Best-effort HTTP POSTs to either a Sagan-style progress endpoint (legacy) or
the task-workflow ``events.jsonl`` via ``task.py post-marker``. CLI flags
take precedence over environment variables so a local dry-run can override.

A failed POST is logged but never aborts the experiment — progress
reporting is observational, not critical.
"""

from __future__ import annotations

import logging
import os
import time

try:
    import httpx  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — httpx is in the runtime env on pods
    httpx = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

_progress_url: str | None = None
_progress_token: str | None = None
_disabled_reason: str | None = None


def configure(
    progress_url: str | None = None,
    progress_token: str | None = None,
) -> None:
    """Configure the module from CLI flags (preferred) or env vars (fallback)."""
    global _progress_url, _progress_token, _disabled_reason

    _progress_url = progress_url or os.environ.get("SAGAN_PROGRESS_URL")
    _progress_token = progress_token or os.environ.get("SAGAN_POD_PROGRESS_TOKEN")

    if not _progress_url:
        _disabled_reason = "no SAGAN_PROGRESS_URL set; progress posts disabled"
        log.info(_disabled_reason)
    elif not _progress_token:
        _disabled_reason = (
            "SAGAN_PROGRESS_URL set but SAGAN_POD_PROGRESS_TOKEN missing; progress posts disabled"
        )
        log.warning(_disabled_reason)
        _progress_url = None  # disable to avoid unauthenticated POSTs
    elif httpx is None:
        _disabled_reason = "httpx not installed; progress posts disabled"
        log.warning(_disabled_reason)
        _progress_url = None
    else:
        _disabled_reason = None
        log.info("Progress reporting enabled: %s", _progress_url)


def post(body: str, *, kind: str = "message", timeout_s: float = 5.0) -> bool:
    """Post a single progress event. Returns ``True`` on HTTP success."""
    if _progress_url is None or httpx is None:
        return False

    headers = {
        "Authorization": f"Bearer {_progress_token}",
        "Content-Type": "application/json",
    }
    payload = {"kind": kind, "body": body, "ts_unix": time.time()}

    try:
        response = httpx.post(_progress_url, json=payload, headers=headers, timeout=timeout_s)
        if response.status_code >= 400:
            log.warning(
                "Progress POST returned %d: %s",
                response.status_code,
                response.text[:200],
            )
            return False
        return True
    except Exception as exc:
        log.warning("Progress POST failed: %s", exc)
        return False


def post_milestone(milestone: str, **fields) -> bool:
    """Convenience wrapper that formats a milestone with ``key=value`` extras."""
    if not fields:
        return post(milestone)
    extras = " ".join(f"{k}={v}" for k, v in fields.items())
    return post(f"{milestone} | {extras}")
