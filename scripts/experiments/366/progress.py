"""Tiny helper for posting progress updates to Sagan from inside the pod.

The bootstrap wrapper injects ``SAGAN_PROGRESS_URL`` and
``SAGAN_POD_PROGRESS_TOKEN`` into the env. It posts 5% on bootstrap-done and
100% on exit. Anything in between is up to us.

Best-effort: never raise. We don't want a transient HTTP failure to crash a
multi-hour training run.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def post_progress(pct: float, message: str, *, timeout_s: float = 5.0) -> None:
    """POST ``{progressPct, message}`` to ``$SAGAN_PROGRESS_URL`` if set.

    Silently no-ops if either env var is missing (i.e. running outside Sagan).
    Logs both the attempt and any failure but never raises — progress reporting
    is a side-channel, not a hard dependency.
    """
    url = os.environ.get("SAGAN_PROGRESS_URL")
    token = os.environ.get("SAGAN_POD_PROGRESS_TOKEN")
    if not url or not token:
        logger.info("progress %.1f%%: %s (no progress URL configured)", pct, message)
        return
    try:
        # httpx is in deps; fall back to urllib if needed (it shouldn't be).
        import httpx

        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(
                url,
                headers={
                    "authorization": f"Bearer {token}",
                    "content-type": "application/json",
                },
                json={"progressPct": float(pct), "message": str(message)},
            )
            if resp.status_code >= 400:
                logger.warning("progress POST returned %d: %s", resp.status_code, resp.text[:200])
            else:
                logger.info("progress %.1f%%: %s", pct, message)
    except Exception as e:
        logger.warning("progress POST failed (%s): %s", type(e).__name__, e)
