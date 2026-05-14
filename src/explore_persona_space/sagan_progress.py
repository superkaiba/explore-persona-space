"""Sagan-aware progress reporting for experiment scripts.

Pairs with Sagan's pod-bootstrap heartbeat (which gives wall-clock ETA from
``SAGAN_ESTIMATED_MINUTES``) by adding *work-aware* progress: percentage from
``steps_done / total_steps`` and ETA from a rolling average of recent step
durations, so the dashboard sidebar shows accurate "Nm left · $X.YZ" while
training is actively progressing.

Quick usage::

    from explore_persona_space.sagan_progress import progress_tracker

    with progress_tracker("training adapters", total=11) as p:
        for adapter_idx in range(11):
            train_adapter(adapter_idx)
            p.advance(message=f"adapter {adapter_idx+1}/11 done")

Two layers compose:

* Inside the ``with`` block the helper rate-limits POSTs (1 every 15s
  minimum, plus always on the final step and on ``message`` calls with the
  ``force=True`` flag) so a tight inner loop doesn't spam the webhook.
* Outside Sagan (no ``SAGAN_PROGRESS_URL`` env var) the helper is a no-op so
  the same script works in a local dev environment.

Reads:
    SAGAN_PROGRESS_URL          POST target (set by Sagan dispatcher)
    SAGAN_POD_PROGRESS_TOKEN    bearer token (set by Sagan dispatcher)
    SAGAN_ESTIMATED_MINUTES     planner's wall-clock upper bound (set by
                                Sagan dispatcher). Used as a fallback ETA if
                                we don't have enough step-time samples yet.

POSTs ``{progressPct, estimatedRemainingMinutes, message}`` to the webhook.
``heartbeat`` is NOT set, so each call adds a real timeline row.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# Minimum gap between POSTs from a single tracker. Inner training loops can
# call ``advance()`` hundreds of times per minute — we don't want that many
# webhook hits. The bootstrap heartbeat is on a separate 90s timer.
MIN_POST_INTERVAL_S: float = 15.0


@dataclass
class _Tracker:
    label: str
    total: int
    start_floor_pct: float
    end_ceiling_pct: float
    steps_done: int = 0
    started_at: float = field(default_factory=time.time)
    last_post_at: float = 0.0
    recent_step_durations: list[float] = field(default_factory=list)
    _last_step_at: float = field(default_factory=time.time)

    def _rolling_eta_minutes(self) -> Optional[float]:
        """Project remaining minutes from a rolling avg of the last ~10 steps.

        Returns None until we have at least 3 samples — earlier estimates
        based on one or two steps are usually wildly off (cold-cache loads,
        first-batch JIT, etc.).
        """
        if len(self.recent_step_durations) < 3:
            return None
        avg = sum(self.recent_step_durations) / len(self.recent_step_durations)
        remaining_steps = max(0, self.total - self.steps_done)
        return (remaining_steps * avg) / 60.0

    def _wall_clock_eta_minutes(self) -> Optional[float]:
        """Fallback ETA from ``SAGAN_ESTIMATED_MINUTES`` minus pod elapsed.

        Used while we haven't accumulated enough step-time samples for a
        work-aware estimate. Same number the bootstrap heartbeat would
        produce, so the dashboard stays consistent across both producers.
        """
        est = os.environ.get("SAGAN_ESTIMATED_MINUTES")
        if not est:
            return None
        try:
            total = float(est)
        except (TypeError, ValueError):
            return None
        elapsed_s = time.time() - self.started_at
        return max(0.0, total - elapsed_s / 60.0)

    def _current_pct(self) -> float:
        """Map step progress onto [start_floor_pct, end_ceiling_pct]."""
        if self.total <= 0:
            return self.start_floor_pct
        frac = min(1.0, self.steps_done / self.total)
        span = self.end_ceiling_pct - self.start_floor_pct
        return self.start_floor_pct + frac * span


def _post(url: str, token: str, body: dict, timeout_s: float = 5.0) -> None:
    """Best-effort POST. Logs + swallows all exceptions."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "authorization": "Bearer " + token,
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=timeout_s).read()
    except Exception as exc:
        logger.warning("sagan progress POST failed: %s", exc)


def _env_endpoint() -> Optional[tuple[str, str]]:
    url = os.environ.get("SAGAN_PROGRESS_URL")
    token = os.environ.get("SAGAN_POD_PROGRESS_TOKEN")
    if not url or not token:
        return None
    return (url, token)


class ProgressTracker:
    """Returned by :func:`progress_tracker`. See module docstring."""

    def __init__(
        self,
        label: str,
        total: int,
        start_floor_pct: float,
        end_ceiling_pct: float,
    ) -> None:
        self._t = _Tracker(
            label=label,
            total=max(0, int(total)),
            start_floor_pct=float(start_floor_pct),
            end_ceiling_pct=float(end_ceiling_pct),
        )

    def advance(
        self,
        message: Optional[str] = None,
        *,
        force: bool = False,
    ) -> None:
        """Mark one step complete and (rate-limited) POST progress to Sagan.

        ``force=True`` bypasses the 15s rate limit — use for milestone events
        like "adapter N/M done" that you always want in the timeline.
        """
        now = time.time()
        dt = now - self._t._last_step_at
        self._t._last_step_at = now
        if dt > 0:
            self._t.recent_step_durations.append(dt)
            if len(self._t.recent_step_durations) > 10:
                self._t.recent_step_durations = self._t.recent_step_durations[-10:]
        self._t.steps_done += 1
        self._maybe_post(message=message, force=force)

    def note(self, message: str, *, force: bool = True) -> None:
        """POST a labeled progress update without advancing the step counter.

        Useful for sub-step milestones — e.g. "loading checkpoint",
        "compiling kernels" — where you want a visible event but the parent
        loop hasn't completed a step yet. Defaults to force=True so the post
        actually lands; pass force=False to honor the rate limit.
        """
        self._maybe_post(message=message, force=force)

    def _maybe_post(self, message: Optional[str], force: bool) -> None:
        endpoint = _env_endpoint()
        if endpoint is None:
            return
        now = time.time()
        is_final_step = self._t.steps_done >= self._t.total > 0
        if (
            not force
            and not is_final_step
            and (now - self._t.last_post_at) < MIN_POST_INTERVAL_S
        ):
            return
        self._t.last_post_at = now
        body: dict = {
            "progressPct": round(self._t._current_pct(), 1),
            "message": message
            or f"{self._t.label}: {self._t.steps_done}/{self._t.total}",
        }
        eta = self._t._rolling_eta_minutes()
        if eta is None:
            eta = self._t._wall_clock_eta_minutes()
        if eta is not None:
            body["estimatedRemainingMinutes"] = int(round(eta))
        url, token = endpoint
        _post(url, token, body)


@contextmanager
def progress_tracker(
    label: str,
    *,
    total: int,
    start_pct: float = 10.0,
    end_pct: float = 95.0,
) -> Iterator[ProgressTracker]:
    """Context manager that tracks a loop's progress and POSTs to Sagan.

    ``start_pct`` and ``end_pct`` define the progress range this loop occupies
    inside the experiment's overall 0-100% bar. The defaults (10%–95%) leave
    room for the bootstrap (5%) and the final write-up phase (95-100%) that
    most experiment scripts have.

    On exit (success or exception), POSTs a final ``end_pct`` event so the
    sidebar settles at the right number even if the caller forgot to make
    the last ``advance()`` call.
    """
    tracker = ProgressTracker(label=label, total=total, start_floor_pct=start_pct, end_ceiling_pct=end_pct)
    try:
        yield tracker
    finally:
        endpoint = _env_endpoint()
        if endpoint is not None:
            url, token = endpoint
            body = {
                "progressPct": round(tracker._t.end_ceiling_pct, 1),
                "message": f"{label}: done ({tracker._t.steps_done}/{tracker._t.total})",
            }
            _post(url, token, body)


def post_progress(
    progress_pct: float,
    message: str,
    *,
    estimated_remaining_minutes: Optional[float] = None,
) -> None:
    """One-shot POST without setting up a tracker.

    For scripts that just want to drop occasional milestone markers without
    a loop structure — e.g. "10% manifest written", "95% writing figures".
    Always lands (no rate limiting). Use the :func:`progress_tracker` context
    manager when you want auto-ETA from rolling step time.
    """
    endpoint = _env_endpoint()
    if endpoint is None:
        return
    url, token = endpoint
    body: dict = {
        "progressPct": round(float(progress_pct), 1),
        "message": str(message),
    }
    if estimated_remaining_minutes is not None:
        body["estimatedRemainingMinutes"] = int(round(estimated_remaining_minutes))
    _post(url, token, body)
