"""Pod-scoped ``epm:run-launched`` marker resolution (#2259).

``poll_pipeline``'s three marker readers (``_marker_pid`` /
``_marker_launch_fields`` / ``_run_launched_age_sec``) resolved the launch
marker by ISSUE only, so on a multi-pod single-issue run (`pod-<N>` +
`pod-<N>-<slug>`, the sanctioned second-pod shape) every leg that is not the
most recent launcher (a) tripped the #1156 pid-file staleness WARN on every
tick and (b) had its ``marker_pid_identity`` cross-check resolve against the
SIBLING pod's ``pid=`` — the #813 stale-pid safety check read ``unknown`` and
evaluated nothing for that leg (measured on #2223, poll tick 2 of the 7B leg).

#2259 threads ``pod`` through the readers: each prefers the newest
``epm:run-launched`` whose note ATTRIBUTES to that pod (the #1961 structured
grammar: boundary-safe ``pod=<name>`` token, or the note's LEADING token),
with issue-wide fallback when no attributable marker exists. These tests pin:

* the #1961 attribution regex port — both attested note shapes (token +
  leading name), boundary safety in BOTH directions (``pod-77`` never matches
  inside ``pod-77-q32b`` and vice versa), and the mid-prose counterexample
  (#1768 v14's "... was already TERMINATED" must not match);
* pod-scoped resolution through all three readers (the older pod reads ITS
  OWN marker's pid + age), the LAST-in-append-order selection convention
  (matching ``latest_event`` semantics), and the debug path-taken line;
* the fallbacks (acceptance 3): ``pod=None`` and an unattributable pod both
  resolve exactly as today (last issue-wide marker), and the degenerate
  unparseable-ts corner still returns None rather than skipping to an older
  marker;
* the mixed-attribution corner: an OLD attributed marker beats a NEWER
  UNATTRIBUTED free-prose relaunch marker for that pod (by design — #1961
  mandates producer-side attribution);
* integration shape via the pure ``_pid_file_predates_marker`` predicate
  (acceptance 1 + 2): with the pod-scoped ``run_age_sec`` the older leg's
  healthy pid file no longer reads stale, while a GENUINELY stale pid file
  on that leg still trips the WARN;
* the fail-soft contract: an unreadable events stream yields None / ``""``
  from every reader, never a crash.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    """Load a ``scripts/*.py`` file as a module (mirrors
    ``tests/test_poll_pipeline_stale_pid_warn.py``'s loader)."""
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_pod_scoped_under_test")

ISSUE = 9977
OLDER_POD = "pod-77"
NEWER_POD = "pod-77-q32b"
OLDER_PID = 3139
NEWER_PID = 2285
OLDER_TS = "2026-08-13T00:00:00Z"
NEWER_TS = "2026-08-13T01:00:00Z"
# 02:00Z probe time => older marker age 7200 s, newer marker age 3600 s.
NOW_EPOCH = datetime(2026, 8, 13, 2, 0, 0, tzinfo=UTC).timestamp()
OLDER_AGE = 7200.0
NEWER_AGE = 3600.0

# Both attested #1768 note shapes: the OLDER marker attributes via the
# `pod=<name>` token form, the NEWER via the leading-name form.
OLDER_NOTE = (
    f"launched 7B leg pod={OLDER_POD} pid={OLDER_PID} pid_file=/workspace/logs/issue-{ISSUE}-7b.pid"
)
NEWER_NOTE = (
    f"{NEWER_POD} (4 GPU) provisioned; 32B leg launched pid={NEWER_PID} "
    f"pid_file=/workspace/logs/issue-{ISSUE}-32b.pid"
)


def _two_marker_events() -> list[dict]:
    """Two `epm:run-launched` markers naming DIFFERENT pods (append order =
    ts order), plus a non-matching-kind row the prefix filter must skip."""
    return [
        {"kind": "epm:status-changed", "ts": "2026-08-12T23:00:00Z", "note": "approved->running"},
        {"kind": "epm:run-launched", "ts": OLDER_TS, "note": OLDER_NOTE},
        {"kind": "epm:run-launched", "ts": NEWER_TS, "note": NEWER_NOTE},
    ]


@pytest.fixture()
def two_marker_events(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    events = _two_marker_events()
    monkeypatch.setattr(pp, "list_events", lambda _issue: events)
    return events


# ── 1. the #1961 attribution regex (ported verbatim) ──────────────────────────


def test_regex_token_form_matches() -> None:
    assert pp._pod_attribution_re(OLDER_POD).search(OLDER_NOTE) is not None


def test_regex_leading_name_form_matches() -> None:
    assert pp._pod_attribution_re(NEWER_POD).search(NEWER_NOTE) is not None


def test_regex_boundary_short_name_never_matches_suffixed_sibling() -> None:
    """``pod-77`` must NOT match inside ``pod-77-q32b``'s marker (either
    the leading-name occurrence or a pod= token carrying the longer name)."""
    assert pp._pod_attribution_re(OLDER_POD).search(NEWER_NOTE) is None
    assert pp._pod_attribution_re(OLDER_POD).search(f"relaunch pod={NEWER_POD} pid=1") is None


def test_regex_boundary_suffixed_name_never_matches_short_sibling() -> None:
    assert pp._pod_attribution_re(NEWER_POD).search(OLDER_NOTE) is None


def test_regex_mid_prose_mention_never_matches() -> None:
    """The attested #1768 v14 counterexample shape: a mid-prose mention must
    neither attribute nor match."""
    note = f"cleanup note: {OLDER_POD} on this issue was already TERMINATED"
    assert pp._pod_attribution_re(OLDER_POD).search(note) is None


# ── 2. resolver: pod-scoped selection + fallbacks + path-taken line ───────────


def test_resolver_pod_scoped_both_note_shapes(two_marker_events: list[dict]) -> None:
    older, newer = two_marker_events[1], two_marker_events[2]
    assert pp._latest_run_launched_event(ISSUE, OLDER_POD) is older
    assert pp._latest_run_launched_event(ISSUE, NEWER_POD) is newer


def test_resolver_fallback_unattributable_pod_and_none(two_marker_events: list[dict]) -> None:
    """Acceptance 3: pod given but NO attributable note -> last issue-wide;
    ``pod=None`` -> last issue-wide (exact pre-#2259 behavior)."""
    newer = two_marker_events[2]
    assert pp._latest_run_launched_event(ISSUE, "pod-99") is newer
    assert pp._latest_run_launched_event(ISSUE, None) is newer


def test_resolver_last_in_append_order_among_attributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TWO markers attributing to the SAME pod: the LAST in append order wins
    (the ``latest_event`` selection convention, not max-ts)."""
    events = [
        {"kind": "epm:run-launched", "ts": OLDER_TS, "note": f"pod={OLDER_POD} pid=111"},
        {"kind": "epm:run-launched", "ts": NEWER_TS, "note": f"pod={OLDER_POD} pid=222"},
    ]
    monkeypatch.setattr(pp, "list_events", lambda _issue: events)
    assert pp._latest_run_launched_event(ISSUE, OLDER_POD) is events[1]


def test_resolver_mixed_attribution_prefers_older_attributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The critic-note-5 corner: same pod, OLD attributed marker + NEWER
    UNATTRIBUTED free-prose relaunch marker -> the older attributed one (by
    design: #1961 mandates producer-side attribution)."""
    events = [
        {"kind": "epm:run-launched", "ts": OLDER_TS, "note": OLDER_NOTE},
        {"kind": "epm:run-launched", "ts": NEWER_TS, "note": "relaunched after hotfix pid=999"},
    ]
    monkeypatch.setattr(pp, "list_events", lambda _issue: events)
    assert pp._latest_run_launched_event(ISSUE, OLDER_POD) is events[0]


def test_resolver_debug_line_names_path_taken(
    two_marker_events: list[dict], caplog: pytest.LogCaptureFixture
) -> None:
    """The task-body ask: the tick transcript says which path was taken."""
    with caplog.at_level(logging.DEBUG, logger="poll_pipeline"):
        pp._latest_run_launched_event(ISSUE, OLDER_POD)
        pp._latest_run_launched_event(ISSUE, "pod-99")
        pp._latest_run_launched_event(ISSUE, None)
    assert "pod-scoped" in caplog.text
    assert "issue-wide-fallback" in caplog.text
    assert "issue-wide (no pod)" in caplog.text


def test_resolver_non_string_note_never_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing / non-string notes are skipped for attribution (watcher
    parity) but still eligible for the issue-wide fallback."""
    events = [
        {"kind": "epm:run-launched", "ts": OLDER_TS, "note": OLDER_NOTE},
        {"kind": "epm:run-launched", "ts": NEWER_TS, "note": None},
    ]
    monkeypatch.setattr(pp, "list_events", lambda _issue: events)
    assert pp._latest_run_launched_event(ISSUE, OLDER_POD) is events[0]
    assert pp._latest_run_launched_event(ISSUE, "pod-99") is events[1]


def test_resolver_fail_soft_on_unreadable_events(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_issue: int) -> list[dict]:
        raise OSError("events.jsonl unreadable")

    monkeypatch.setattr(pp, "list_events", _boom)
    assert pp._latest_run_launched_event(ISSUE, OLDER_POD) is None
    assert pp._marker_pid(ISSUE, pod=OLDER_POD) is None
    assert pp._run_launched_age_sec(ISSUE, NOW_EPOCH, pod=OLDER_POD) is None
    assert pp._marker_launch_fields(ISSUE, pod=OLDER_POD) == (None, "")


# ── 3. the three readers, pod-threaded ─────────────────────────────────────────


def test_marker_pid_pod_scoped_and_fallback(two_marker_events: list[dict]) -> None:
    """Polling the OLDER pod returns ITS OWN marker's ``pid=`` (the #813
    cross-check input — no longer the sibling's); default = today's read."""
    assert pp._marker_pid(ISSUE, pod=OLDER_POD) == OLDER_PID
    assert pp._marker_pid(ISSUE, pod=NEWER_POD) == NEWER_PID
    assert pp._marker_pid(ISSUE) == NEWER_PID  # pre-#2259 behavior preserved


def test_marker_launch_fields_pid_and_note_from_same_marker(
    two_marker_events: list[dict],
) -> None:
    assert pp._marker_launch_fields(ISSUE, pod=OLDER_POD) == (OLDER_PID, OLDER_NOTE)
    assert pp._marker_launch_fields(ISSUE, pod=NEWER_POD) == (NEWER_PID, NEWER_NOTE)
    assert pp._marker_launch_fields(ISSUE) == (NEWER_PID, NEWER_NOTE)


def test_run_launched_age_pod_scoped_and_fallback(two_marker_events: list[dict]) -> None:
    assert pp._run_launched_age_sec(ISSUE, NOW_EPOCH, pod=OLDER_POD) == pytest.approx(OLDER_AGE)
    assert pp._run_launched_age_sec(ISSUE, NOW_EPOCH, pod=NEWER_POD) == pytest.approx(NEWER_AGE)
    assert pp._run_launched_age_sec(ISSUE, NOW_EPOCH) == pytest.approx(NEWER_AGE)
    assert pp._run_launched_age_sec(ISSUE, NOW_EPOCH, pod="pod-99") == pytest.approx(NEWER_AGE)


def test_run_launched_age_unparseable_newest_ts_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Degenerate-corner parity: an unparseable ts on the newest SELECTED
    (pod-scoped) marker returns None — never a silent skip to an older
    marker (the pre-#2259 ``latest_event`` convention, preserved)."""
    events = [
        {"kind": "epm:run-launched", "ts": OLDER_TS, "note": f"pod={OLDER_POD} pid=111"},
        {"kind": "epm:run-launched", "ts": "not-a-ts", "note": f"pod={OLDER_POD} pid=222"},
    ]
    monkeypatch.setattr(pp, "list_events", lambda _issue: events)
    assert pp._run_launched_age_sec(ISSUE, NOW_EPOCH, pod=OLDER_POD) is None


# ── 4. integration shape: the #1156 predicate on pod-scoped ages ───────────────
#
# ``_pid_file_predates_marker`` is the pure predicate ``poll_once`` feeds with
# ``run_age_sec`` (tests/test_poll_pipeline_stale_pid_warn.py pins the full
# poll_once wiring); these two pins are the acceptance-1/2 arithmetic on the
# POD-SCOPED age the #2259 threading now delivers.

_POD_NOW = 1_800_000_000
_SLACK = 600


def test_acceptance1_older_leg_healthy_pid_file_no_warn(two_marker_events: list[dict]) -> None:
    """Acceptance 1: the older leg's pid file (written 90 s after ITS OWN
    launch) reads stale against the ISSUE-WIDE newest age (the #2223 false
    WARN) but healthy against the leg's own pod-scoped age."""
    own_age = pp._run_launched_age_sec(ISSUE, NOW_EPOCH, pod=OLDER_POD)
    sibling_age = pp._run_launched_age_sec(ISSUE, NOW_EPOCH)  # issue-wide newest
    pid_file_age = OLDER_AGE - 90  # written 90 s after the older leg's launch
    kwargs = dict(
        pid_file_mtime_epoch=int(_POD_NOW - pid_file_age),
        pod_now_epoch=_POD_NOW,
        slack_sec=_SLACK,
    )
    # The pre-#2259 shape: compared against the SIBLING's marker -> false WARN.
    assert pp._pid_file_predates_marker(run_age_sec=sibling_age, **kwargs) is True
    # Pod-scoped: the leg's own marker -> no staleness WARN.
    assert pp._pid_file_predates_marker(run_age_sec=own_age, **kwargs) is False


def test_acceptance2_older_leg_genuinely_stale_pid_file_still_warns(
    two_marker_events: list[dict],
) -> None:
    """Acceptance 2 (load-bearing): a pid file genuinely older than the
    leg's OWN marker by more than the slack STILL trips the check."""
    own_age = pp._run_launched_age_sec(ISSUE, NOW_EPOCH, pod=OLDER_POD)
    pid_file_age = OLDER_AGE + 7200  # a prior launch's pid file, 2 h older
    assert (
        pp._pid_file_predates_marker(
            pid_file_mtime_epoch=int(_POD_NOW - pid_file_age),
            pod_now_epoch=_POD_NOW,
            run_age_sec=own_age,
            slack_sec=_SLACK,
        )
        is True
    )


# ── 5. poll_once call-site threading (code-review round-1 minor) ───────────────


def test_poll_once_threads_pod_to_both_marker_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pins the two ``poll_once`` call-site ``pod=pod`` threads — the
    one-line essence of the #2259 fix. Every other poll_once-level test
    fakes the helpers with pod-accepting-but-ignoring fakes, so reverting
    either call site to the un-threaded form would leave the suite green;
    recording fakes capture the ``pod`` kwarg each helper actually receives
    (the code-review round-1 sketch). Harness mirrors
    ``tests/test_poll_pipeline_stale_pid_warn.py::_patch_pod`` (sentinel
    drain SSH returns empty; the probe returns a healthy running tick)."""
    received: dict[str, object] = {}

    def _fake_launch_fields(issue: int, pod: str | None = None) -> tuple[int | None, str]:
        received["launch_fields_pod"] = pod
        return None, ""

    def _fake_age(issue: int, now_epoch: float, pod: str | None = None) -> float | None:
        received["age_pod"] = pod
        return 10800.0

    pod_now = int(time.time())  # pod clock == VM clock; healthy ordering below

    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        remote = cmd[-1]
        if "SENTINEL_START" in remote:
            stdout = ""
        else:
            stdout = "\n".join(
                [
                    "PID_FILE_MISSING=0",
                    f"PID_FILE_MTIME_EPOCH={pod_now - 3690}",  # marker_age + 90: no WARN
                    "PID_ALIVE=1",
                    f"MTIME_EPOCH={pod_now - 30}",
                    f"POD_NOW_EPOCH={pod_now}",
                    "TAIL_START",
                    "2026-07-09 00:00:01 [phase=training step=5/100]",
                    "TAIL_END",
                    "CELL_MTIME_EPOCH=0",
                    "CELL_TAIL_START",
                    "CELL_TAIL_END",
                    "PHASE_LOG_MTIME_EPOCH=0",
                    "SHARD_LOG_MTIME_EPOCH=0",
                    "GPU_UTIL=95",
                    "ZOMBIE_GPU_PIDS=",
                    "SESSION_CPU_SECS=unknown",
                    "RESULTS_SENTINEL_PRESENT=0",
                ]
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(pp, "_marker_launch_fields", _fake_launch_fields)
    monkeypatch.setattr(pp, "_run_launched_age_sec", _fake_age)

    pp.poll_once(
        issue=ISSUE,
        pod=OLDER_POD,
        log_path=f"/workspace/logs/issue-{ISSUE}.log",
        pid_file=f"/workspace/logs/issue-{ISSUE}.pid",
        state_file=tmp_path / "poll-state.json",
    )

    assert received["launch_fields_pod"] == OLDER_POD  # :5616 thread
    assert received["age_pod"] == OLDER_POD  # :5915 thread
