"""#983 post-done phase-consistency guard.

The #545 corroboration block gates the INITIAL done verdict within a tick and
the #597 noise regex gates which line may parse as done at all — both are
parse-time defenses. The #983 guard is the CROSS-TICK audit they are blind
to: at the tick where a CORROBORATED ``current_phase == "done"`` lands, the
state file records the matched done line (identity-truncated), the epoch, and
the pod; any LATER tick observing genuinely NEW ``[phase=...]`` lines after
that anchor fires ONE loud ``[post-done-phase-advisory]`` ``epm:progress``
marker + a best-effort Telegram push — the ``.py``-dispatcher subprocess
fan-out false-done class (#930 §4.6 residual gap (i)). Advisory only: the
status verdict is never changed.

These tests pin (plan #983 §5, tests 1-24):

* the pure core ``_post_done_phase_update`` / ``_phase_bearing_lines``
  (episode recording, identity-anchored comparison, the #597 noise skip,
  the byte-identical-duplicate + earlier-lines FP controls, the scrolled-out
  append-only argument, the once-per-episode dedup, the run-scope clamp,
  cross-pod voiding, corrupt-state parses);
* ``poll_once`` integration (episode persistence in the state file, exactly
  one advisory post + push, dedup across ticks, post-failure retry, the
  SSH-failure fallback tick, the JSON surfaces on ``main()`` +
  ``backend_poll._serialize_poll_result``, the REAL ``RunPodBackend.poll``
  rewrap passthrough — the #664 stall_reason silent-drop class);
* test 21: an UNCORROBORATED done (pid alive + no results sentinel — the
  #545 demotion path) NEVER arms an episode (pins the below-demotion-block
  wiring order);
* test 22: a gate-precedence tick (``status == "gate"``) with a corroborated
  done still RECORDS the episode (pins the ``current_phase``-keyed — not
  ``status``-keyed — episode-start condition);
* test 23: the anchor-recurrence edge fails toward QUIET (a byte-identical
  done re-emission AFTER a new line swallows it on that tick);
* test 24: record-side and compare-side truncation are symmetric (a done
  line longer than ``_POST_DONE_LINE_MAX`` never false-fires on an
  identical re-poll).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    """Load a ``scripts/*.py`` file as a module (mirrors the
    ``tests/test_poll_pipeline_zombie_gpu.py`` loader)."""
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_post_done_guard_under_test")
bp = _load_script_module("backend_poll.py", "backend_poll_post_done_guard_under_test")

ISSUE = 9983
POD = "pod-9983"
LOG = "/workspace/logs/issue-9983.log"
PIDFILE = "/workspace/logs/issue-9983.pid"

TRAIN = "2026-07-04 00:00:00 [phase=training step=5/100]"
DONE = "2026-07-04 00:00:01 [phase=done] SMOKE COMPLETE"
NEW_CELL = "2026-07-04 00:10:00 [phase=eval_cell_3] worker resumed"
STRAGGLER_DONE = "2026-07-04 00:10:00 [phase=done] eval cell 5 complete"
NOISE_DONE = "2026-07-04 00:10:00 ONE OR MORE SHARDS FAILED rc=1 - [phase=done] NOT emitted"
NON_PHASE = "2026-07-04 00:10:00 uploading eval_results to HF (attempt 1)"


def _update(**overrides: Any) -> Any:
    """Call the pure core with happy-path defaults, overridable per test."""
    kwargs: dict[str, Any] = dict(
        current_phase="training",
        log_tail=f"{TRAIN}\n{DONE}",
        pod=POD,
        prev_done_line="",
        prev_done_epoch=0,
        prev_done_pod="",
        prev_posted=False,
        run_age_sec=10800.0,
        now_epoch=int(time.time()),
    )
    kwargs.update(overrides)
    return pp._post_done_phase_update(**kwargs)


# ── unit: _phase_bearing_lines + the pure decision core ──────────────────────


def test_fresh_done_tick_records_episode() -> None:
    """Test 1: a fresh corroborated done tick with no prior episode records
    (done_line = LAST phase-bearing line, done_epoch = now, done_pod = pod)
    and does NOT post."""
    now = int(time.time())
    u = _update(current_phase="done", now_epoch=now)
    assert u.should_post is False
    assert u.done_line == DONE
    assert u.done_epoch == now
    assert u.done_pod == POD
    assert u.advisory_posted is False
    assert u.new_phase_lines == ()


def test_identical_tail_repoll_no_fire() -> None:
    """Test 2: an active episode re-polled with the identical tail neither
    posts nor surfaces new lines (the anchor is the last phase line)."""
    u = _update(prev_done_line=DONE, prev_done_epoch=int(time.time()) - 60, prev_done_pod=POD)
    assert u.should_post is False
    assert u.new_phase_lines == ()
    assert u.done_line == DONE  # episode retained


def test_new_phase_line_after_done_fires() -> None:
    """Test 3: one genuinely new [phase=...] line after the recorded done
    line fires and surfaces the line."""
    u = _update(
        log_tail=f"{TRAIN}\n{DONE}\n{NEW_CELL}",
        prev_done_line=DONE,
        prev_done_epoch=int(time.time()) - 60,
        prev_done_pod=POD,
    )
    assert u.should_post is True
    assert u.new_phase_lines == (NEW_CELL,)


def test_non_phase_lines_after_done_no_fire() -> None:
    """Test 4: non-phase lines appended after done (post-done uploads,
    sentinel drains) never fire — only PHASE_RE matches are candidates."""
    u = _update(
        log_tail=f"{TRAIN}\n{DONE}\n{NON_PHASE}",
        prev_done_line=DONE,
        prev_done_epoch=int(time.time()) - 60,
        prev_done_pod=POD,
    )
    assert u.should_post is False
    assert u.new_phase_lines == ()


def test_noise_done_line_skipped_but_nondone_rc_line_counts() -> None:
    """Test 5: a DONE_QUOTED_NOISE_RE done line appended after done does NOT
    fire (same skip predicate as latest_phase); a NON-done phase line
    carrying rc=1 still counts (the noise gate is done-token-only)."""
    now = int(time.time())
    u = _update(
        log_tail=f"{TRAIN}\n{DONE}\n{NOISE_DONE}",
        prev_done_line=DONE,
        prev_done_epoch=now - 60,
        prev_done_pod=POD,
    )
    assert u.should_post is False
    assert u.new_phase_lines == ()
    rc_line = "2026-07-04 00:10:00 [phase=eval_cell_3] shard FAILED rc=1"
    u2 = _update(
        log_tail=f"{TRAIN}\n{DONE}\n{rc_line}",
        prev_done_line=DONE,
        prev_done_epoch=now - 60,
        prev_done_pod=POD,
    )
    assert u2.should_post is True
    assert u2.new_phase_lines == (rc_line,)


def test_byte_identical_done_reemission_no_fire() -> None:
    """Test 6: a byte-identical duplicate of the recorded done line appended
    later never fires (FP control (i): last-occurrence anchoring + the
    identity filter)."""
    u = _update(
        log_tail=f"{TRAIN}\n{DONE}\n{DONE}",
        prev_done_line=DONE,
        prev_done_epoch=int(time.time()) - 60,
        prev_done_pod=POD,
    )
    assert u.should_post is False
    assert u.new_phase_lines == ()


def test_done_scrolled_out_no_phase_lines_no_fire() -> None:
    """Test 7: the done line (and everything above it) scrolled out of the
    bounded tail and NO phase lines remain -> no fire, episode retained."""
    u = _update(
        log_tail=f"{NON_PHASE}\nplain output line",
        prev_done_line=DONE,
        prev_done_epoch=int(time.time()) - 60,
        prev_done_pod=POD,
    )
    assert u.should_post is False
    assert u.new_phase_lines == ()
    assert u.done_line == DONE  # episode retained


def test_done_scrolled_out_new_phase_lines_all_count() -> None:
    """Test 8: the done line scrolled out but NEW phase lines are visible —
    the log is append-only, so everything still visible is newer than the
    recorded done: all count, the guard fires."""
    later = "2026-07-04 00:20:00 [phase=eval_cell_4] worker resumed"
    u = _update(
        log_tail=f"{NEW_CELL}\n{later}",
        prev_done_line=DONE,
        prev_done_epoch=int(time.time()) - 60,
        prev_done_pod=POD,
    )
    assert u.should_post is True
    assert u.new_phase_lines == (NEW_CELL, later)


def test_earlier_phase_lines_above_done_not_counted() -> None:
    """Test 9: phase lines ABOVE the done line in the tail (the run's own
    history) are never candidates (FP control (iv))."""
    u = _update(
        log_tail=f"{TRAIN}\n{NEW_CELL}\n{DONE}",
        prev_done_line=DONE,
        prev_done_epoch=int(time.time()) - 60,
        prev_done_pod=POD,
    )
    assert u.should_post is False
    assert u.new_phase_lines == ()


def test_already_posted_dedups_but_still_surfaces_lines() -> None:
    """Test 10: prev_posted=True + more new lines -> no second post, but the
    observed lines are still surfaced (observability survives the dedup)."""
    u = _update(
        log_tail=f"{TRAIN}\n{DONE}\n{NEW_CELL}",
        prev_done_line=DONE,
        prev_done_epoch=int(time.time()) - 60,
        prev_done_pod=POD,
        prev_posted=True,
    )
    assert u.should_post is False
    assert u.new_phase_lines == (NEW_CELL,)
    assert u.advisory_posted is True


def test_run_scope_clamp_voids_episode_and_fresh_done_rearms() -> None:
    """Test 11: an episode recorded BEFORE the current run's
    epm:run-launched is voided (the fresh run's phase lines never fire
    against the old done); a later fresh done starts a NEW episode; and
    run_age_sec=None (marker missing/unreadable) leaves the episode intact."""
    now = int(time.time())
    # Episode from a PREVIOUS run (epoch 1000s ago; the launch was 5s ago).
    u = _update(
        current_phase="training",
        log_tail=TRAIN,
        prev_done_line=DONE,
        prev_done_epoch=now - 1000,
        prev_done_pod=POD,
        run_age_sec=5.0,
        now_epoch=now,
    )
    assert u.should_post is False
    assert u.done_line == ""  # voided
    assert u.done_epoch == 0
    # A later fresh corroborated done starts a new episode.
    u2 = _update(current_phase="done", run_age_sec=5.0, now_epoch=now)
    assert u2.done_line == DONE
    assert u2.done_epoch == now
    # run_age_sec=None -> no clamp signal -> episode retained (fires on the
    # new line rather than being silently voided).
    u3 = _update(
        log_tail=f"{TRAIN}\n{DONE}\n{NEW_CELL}",
        prev_done_line=DONE,
        prev_done_epoch=now - 1000,
        prev_done_pod=POD,
        run_age_sec=None,
        now_epoch=now,
    )
    assert u3.should_post is True
    assert u3.done_line == DONE


def test_second_textually_different_done_line_fires() -> None:
    """Test 12: a SECOND, textually different ``[phase=done] ... complete``
    line after the accepted done fires — the straggler-cell class this
    guard exists for."""
    u = _update(
        log_tail=f"{TRAIN}\n{DONE}\n{STRAGGLER_DONE}",
        prev_done_line=DONE,
        prev_done_epoch=int(time.time()) - 60,
        prev_done_pod=POD,
    )
    assert u.should_post is True
    assert u.new_phase_lines == (STRAGGLER_DONE,)


def test_corrupt_state_values_parse_to_zero_no_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 13: corrupt persisted values (``post_done_epoch="garbage"``) are
    parsed as 0 by the wiring helper — no raise, no fire."""
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(pp, "_telegram_push", MagicMock(return_value=True))
    (line, epoch, pod, flag, posted, new_lines) = pp._maybe_post_post_done_phase_advisory(
        issue=ISSUE,
        pod=POD,
        current_phase="done",
        log_tail=f"{TRAIN}\n{DONE}",
        prev_state={
            "post_done_line": DONE,
            "post_done_epoch": "garbage",
            "post_done_pod": POD,
            "post_done_advisory_posted": "0",
        },
        run_age_sec=10800.0,
        now_epoch=int(time.time()),
    )
    assert line == DONE
    assert epoch == 0  # corrupt epoch reset, episode otherwise intact
    assert pod == POD
    assert flag is False
    assert posted is False
    assert new_lines == ()


def test_cross_pod_episode_voided_and_new_pod_done_rearms() -> None:
    """Extra (plan v3 §3.2 C-2): an episode recorded on pod-A is VOIDED on a
    tick polling pod-B (a diagnostic poll against a different pod never
    compares tails across pods); a corroborated done on pod-B starts a NEW
    episode bound to pod-B."""
    now = int(time.time())
    u = _update(
        current_phase="training",
        log_tail=f"{TRAIN}\n{DONE}\n{NEW_CELL}",
        pod="pod-other",
        prev_done_line=DONE,
        prev_done_epoch=now - 60,
        prev_done_pod=POD,
        now_epoch=now,
    )
    assert u.should_post is False
    assert u.done_line == ""  # voided, the new pod's lines never fire
    u2 = _update(
        current_phase="done",
        pod="pod-other",
        prev_done_line=DONE,
        prev_done_epoch=now - 60,
        prev_done_pod=POD,
        now_epoch=now,
    )
    assert u2.done_line == DONE
    assert u2.done_pod == "pod-other"


# ── integration harness (mirrors tests/test_poll_pipeline_zombie_gpu.py) ─────


def _probe_stdout(
    *,
    mtime_epoch: int,
    tail: str,
    pid_alive: str,
    results_sentinel: str = "0",
) -> str:
    """Probe stdout in the shape ``_parse_probe_stdout`` expects."""
    return "\n".join(
        [
            "PID_FILE_MISSING=0",
            f"PID_ALIVE={pid_alive}",
            f"MTIME_EPOCH={mtime_epoch}",
            "TAIL_START",
            tail,
            "TAIL_END",
            "CELL_MTIME_EPOCH=0",
            "CELL_TAIL_START",
            "CELL_TAIL_END",
            "PHASE_LOG_MTIME_EPOCH=0",
            "SHARD_LOG_MTIME_EPOCH=0",
            "GPU_UTIL=0",
            "ZOMBIE_GPU_PIDS=",
            "SESSION_CPU_SECS=unknown",
            f"RESULTS_SENTINEL_PRESENT={results_sentinel}",
        ]
    )


def _patch_pod(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tail: str,
    pid_alive: str = "0",
    mtime_epoch: int | None = None,
    results_sentinel: str = "0",
    ssh_fail: bool = False,
    run_age_sec: float | None = 10800.0,
) -> None:
    """Monkeypatch poll_pipeline's I/O boundary with a fully-controlled probe.

    Stateless per call — multi-tick tests re-invoke it between ``poll_once``
    calls to vary the probe (append tail lines, kill the pid, fail SSH)."""
    resolved_mtime = int(time.time()) - 10 if mtime_epoch is None else mtime_epoch

    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        remote = cmd[-1]
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if ssh_fail:
            return subprocess.CompletedProcess(
                args=cmd, returncode=255, stdout="", stderr="ssh: connect refused"
            )
        stdout = _probe_stdout(
            mtime_epoch=resolved_mtime,
            tail=tail,
            pid_alive=pid_alive,
            results_sentinel=results_sentinel,
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(pp, "_telegram_push", MagicMock(return_value=True))
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch: run_age_sec)


def _poll(state_file: Path):
    return pp.poll_once(issue=ISSUE, pod=POD, log_path=LOG, pid_file=PIDFILE, state_file=state_file)


def _state(state_file: Path) -> dict[str, str]:
    return json.loads(state_file.read_text())[str(ISSUE)]


def _advisory_calls(post_mock: MagicMock) -> list:
    return [
        c
        for c in post_mock.call_args_list
        if str(c.kwargs.get("note", "")).startswith("[post-done-phase-advisory]")
    ]


# ── integration: poll_once replays against one tmp state file ────────────────


def test_done_tick_persists_episode_no_advisory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test 14: the corroborated done tick records the episode in the state
    file (line + epoch + pod, dedup flag "0") and posts NO advisory."""
    state_file = tmp_path / "poll-state.json"
    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}", pid_alive="0")
    result = _poll(state_file)
    assert result.status == "done"
    assert result.post_done_phase_advisory_posted is False
    assert result.post_done_phase_lines == ()
    saved = _state(state_file)
    assert saved["post_done_line"] == DONE
    assert int(saved["post_done_epoch"]) > 0
    assert saved["post_done_pod"] == POD
    assert saved["post_done_advisory_posted"] == "0"
    assert _advisory_calls(pp.post_event) == []


def test_second_poll_new_phase_line_fires_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test 15: a second poll observing a new phase line after the recorded
    done posts EXACTLY one advisory (note prefix + the
    ``post_done_phase_advisory=True`` extra), pushes once, surfaces the
    lines on the PollResult, and never forces the status verdict (the
    straggler-done shape still reports ``done`` from the arbiters)."""
    state_file = tmp_path / "poll-state.json"
    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}", pid_alive="0")
    assert _poll(state_file).status == "done"

    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}\n{STRAGGLER_DONE}", pid_alive="0")
    result = _poll(state_file)
    # Status is whatever the arbiters computed — NOT forced by the guard.
    assert result.status == "done"
    assert result.post_done_phase_advisory_posted is True
    assert result.post_done_phase_lines == (STRAGGLER_DONE,)
    calls = _advisory_calls(pp.post_event)
    assert len(calls) == 1
    assert calls[0].kwargs["post_done_phase_advisory"] is True
    assert "1 NEW [phase=...] line(s)" in calls[0].kwargs["note"]
    assert pp._telegram_push.call_count == 1
    assert _state(state_file)["post_done_advisory_posted"] == "1"


def test_third_poll_same_tail_dedups(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test 16: a third poll on the same tail posts NOTHING further (the
    once-per-episode dedup), while the observed lines stay surfaced on the
    PollResult."""
    state_file = tmp_path / "poll-state.json"
    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}", pid_alive="0")
    _poll(state_file)
    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}\n{STRAGGLER_DONE}", pid_alive="0")
    _poll(state_file)

    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}\n{STRAGGLER_DONE}", pid_alive="0")
    result = _poll(state_file)
    assert result.post_done_phase_advisory_posted is False  # THIS tick posted nothing
    assert result.post_done_phase_lines == (STRAGGLER_DONE,)  # still surfaced
    assert _advisory_calls(pp.post_event) == []
    assert _state(state_file)["post_done_advisory_posted"] == "1"


def test_post_failure_not_recorded_next_tick_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test 17: a failing ``post_event`` on the fire tick does NOT persist
    the dedup flag; the next tick retries and posts (the width-advisory
    retry contract)."""
    state_file = tmp_path / "poll-state.json"
    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}", pid_alive="0")
    _poll(state_file)

    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}\n{STRAGGLER_DONE}", pid_alive="0")

    def _raising_post(issue: int, kind: str, **kwargs: Any) -> None:
        if kwargs.get("post_done_phase_advisory"):
            raise RuntimeError("marker post failed")

    monkeypatch.setattr(pp, "post_event", MagicMock(side_effect=_raising_post))
    result = _poll(state_file)
    assert result.post_done_phase_advisory_posted is False
    assert _state(state_file)["post_done_advisory_posted"] == "0"

    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}\n{STRAGGLER_DONE}", pid_alive="0")
    result = _poll(state_file)
    assert result.post_done_phase_advisory_posted is True
    assert len(_advisory_calls(pp.post_event)) == 1
    assert _state(state_file)["post_done_advisory_posted"] == "1"


def test_ssh_failed_post_done_tick_retains_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test 18: an SSH-failed tick after done (empty fallback tail) neither
    fires nor crashes, and the episode survives in the state file."""
    state_file = tmp_path / "poll-state.json"
    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}", pid_alive="0")
    _poll(state_file)

    _patch_pod(monkeypatch, tail="", ssh_fail=True)
    result = _poll(state_file)  # must not raise
    assert result.post_done_phase_advisory_posted is False
    assert _advisory_calls(pp.post_event) == []
    saved = _state(state_file)
    assert saved["post_done_line"] == DONE  # episode retained through the outage
    assert saved["post_done_advisory_posted"] == "0"


def test_relaunch_clamp_voids_episode_integration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Extra (plan §13 C-10): integration clamp replay — a fresh
    epm:run-launched (small run_age_sec) voids an episode recorded BEFORE
    it, so the fresh run's truncated log (whose phase lines would otherwise
    fire via the scrolled-out branch) never fires against the old done."""
    state_file = tmp_path / "poll-state.json"
    now = int(time.time())
    # Seed an episode from the PREVIOUS run (recorded 1000s ago; the fresh
    # epm:run-launched below is 5s old, so the episode predates the launch).
    state_file.write_text(
        json.dumps(
            {
                str(ISSUE): {
                    "phase": "training",
                    "post_done_line": DONE,
                    "post_done_epoch": str(now - 1000),
                    "post_done_pod": POD,
                    "post_done_advisory_posted": "0",
                }
            }
        )
    )
    # Relaunch: truncated log (the old done scrolled away), live pid,
    # launch 5s ago. WITHOUT the clamp this would FIRE (TRAIN is a visible
    # phase line and the anchor is gone -> the append-only branch counts it).
    _patch_pod(monkeypatch, tail=TRAIN, pid_alive="1", run_age_sec=5.0)
    result = _poll(state_file)
    assert result.status == "running"
    assert result.post_done_phase_advisory_posted is False
    assert result.post_done_phase_lines == ()
    assert _advisory_calls(pp.post_event) == []
    assert _state(state_file)["post_done_line"] == ""  # episode voided


# ── JSON-surface contract ─────────────────────────────────────────────────────


def test_main_json_line_includes_post_done_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test 19a: ``poll_pipeline.main`` surfaces both #983 keys in its JSON
    line."""
    fake = pp.PollResult(
        status="done",
        current_phase="done",
        new_milestone=False,
        last_log_mtime_sec_ago=10,
        pid_alive=False,
        pid_file_missing=False,
        log_tail_excerpt="",
        post_done_phase_advisory_posted=True,
        post_done_phase_lines=(STRAGGLER_DONE,),
    )
    monkeypatch.setattr(pp, "poll_once", lambda **kwargs: fake)
    rc = pp.main(["--issue", str(ISSUE), "--pod", POD, "--log", LOG, "--pid-file", PIDFILE])
    assert rc == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["post_done_phase_advisory_posted"] is True
    assert line["post_done_phase_lines"] == [STRAGGLER_DONE]


def test_backend_poll_serializer_passes_post_done_keys_through() -> None:
    """Test 19b: ``backend_poll._serialize_poll_result`` carries both keys
    when the backends-side result has them."""
    from types import SimpleNamespace

    result = SimpleNamespace(
        status="done",
        current_phase="done",
        new_milestone=False,
        last_log_mtime_sec_ago=10,
        pid_alive=False,
        log_tail_excerpt="",
        gate=None,
        sentinels_processed=0,
        phase_log_mtime_sec_ago=10**9,
        shard_log_mtime_sec_ago=10**9,
        gpu_util="0",
        next_interval=540,
        stall_reason=None,
        post_done_phase_advisory_posted=True,
        post_done_phase_lines=(STRAGGLER_DONE,),
    )
    out = bp._serialize_poll_result(result)
    assert out["post_done_phase_advisory_posted"] is True
    assert out["post_done_phase_lines"] == [STRAGGLER_DONE]


def test_backend_poll_serializer_defaults_post_done_keys_for_older_results() -> None:
    """Test 19c: a duck-typed result lacking the #983 fields (GCP/SLURM
    lanes, or an older module) degrades to the defaults — never crashes."""
    from types import SimpleNamespace

    result = SimpleNamespace(
        status="running",
        current_phase="training",
        new_milestone=False,
        last_log_mtime_sec_ago=10,
        pid_alive=True,
        log_tail_excerpt="",
        gate=None,
        sentinels_processed=0,
        phase_log_mtime_sec_ago=10,
        shard_log_mtime_sec_ago=10**9,
        gpu_util="95",
        next_interval=540,
    )
    out = bp._serialize_poll_result(result)
    assert out["post_done_phase_advisory_posted"] is False
    assert out["post_done_phase_lines"] == []


def test_runpod_backend_rewrap_passes_post_done_fields_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 20 (§3.4a): a ``poll_pipeline.PollResult`` with the advisory
    fields set, rewrapped through the REAL ``RunPodBackend.poll`` kwarg
    enumeration (never a hand-replicated rewrap), surfaces both keys
    truthfully in ``_serialize_poll_result`` output — pins the #664
    stall_reason-class silent field drop."""
    import scripts.poll_pipeline as real_pp
    from explore_persona_space.backends.base import PollResult as BasePollResult
    from explore_persona_space.backends.base import RunHandle
    from explore_persona_space.backends.runpod import RunPodBackend

    fake = real_pp.PollResult(
        status="done",
        current_phase="done",
        new_milestone=False,
        last_log_mtime_sec_ago=10,
        pid_alive=False,
        pid_file_missing=False,
        log_tail_excerpt="",
        post_done_phase_advisory_posted=True,
        post_done_phase_lines=(STRAGGLER_DONE,),
    )
    monkeypatch.setattr(real_pp, "poll_once", lambda **kwargs: fake)
    handle = RunHandle(
        backend="runpod",
        cluster=None,
        job_id="",
        pod_name=POD,
        scratch_dir="/workspace",
        log_path=LOG,
        extra={"intent": "lora-7b", "issue": ISSUE, "pid_file": PIDFILE},
    )
    rewrapped = RunPodBackend().poll(handle)
    assert isinstance(rewrapped, BasePollResult)
    assert rewrapped.post_done_phase_advisory_posted is True
    assert rewrapped.post_done_phase_lines == (STRAGGLER_DONE,)
    out = bp._serialize_poll_result(rewrapped)
    assert out["post_done_phase_advisory_posted"] is True
    assert out["post_done_phase_lines"] == [STRAGGLER_DONE]


# ── round-1 Must-Fix pins (plan §13) ─────────────────────────────────────────


def test_uncorroborated_done_never_arms_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test 21 (MF-1): an UNCORROBORATED done-parse (pid ALIVE + no results
    sentinel — the #545 demotion path) NEVER arms an episode, and a later
    phase line posts NO advisory. Pins the wiring order: the guard consumes
    the POST-demotion ``current_phase``, never a raw ``latest_phase``
    re-derivation."""
    state_file = tmp_path / "poll-state.json"
    eval_cells = "2026-07-04 00:00:01 [phase=eval_cells]"
    percell_done = "2026-07-04 00:00:02 [phase=done] eval cell 4 complete"
    _patch_pod(
        monkeypatch,
        tail=f"{eval_cells}\n{percell_done}",
        pid_alive="1",
        results_sentinel="0",
    )
    result = _poll(state_file)
    assert result.current_phase == "eval_cells"  # the #545 demotion fired
    assert result.status == "running"
    assert _state(state_file)["post_done_line"] == ""  # NO episode armed

    later = "2026-07-04 00:00:03 [phase=eval_cell_5]"
    _patch_pod(
        monkeypatch,
        tail=f"{eval_cells}\n{percell_done}\n{later}",
        pid_alive="1",
        results_sentinel="0",
    )
    result = _poll(state_file)
    assert result.post_done_phase_advisory_posted is False
    assert _advisory_calls(pp.post_event) == []
    assert _state(state_file)["post_done_line"] == ""


def test_gate_precedence_done_tick_records_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test 22 (MF-2): a gate-sentinel tick whose pipeline ALSO reached a
    corroborated done reports ``status == "gate"`` yet still RECORDS the
    episode (the episode-start condition is the corroborated
    ``current_phase``, never ``status``); a later new phase line then
    advises once."""
    state_file = tmp_path / "poll-state.json"
    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}", pid_alive="0")
    monkeypatch.setattr(pp, "_drain_sentinels", lambda *, issue, pod: (1, "fact-candidates"))
    result = _poll(state_file)
    assert result.status == "gate"
    assert _state(state_file)["post_done_line"] == DONE

    new_line = "2026-07-04 00:20:00 [phase=eval_cell_9]"
    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}\n{new_line}", pid_alive="0")
    result = _poll(state_file)
    assert result.post_done_phase_advisory_posted is True
    assert len(_advisory_calls(pp.post_event)) == 1


def test_anchor_recurrence_fails_toward_quiet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test 23 (C-6): a byte-identical re-emission of the recorded done line
    AFTER a new phase line swallows the intervening line on THAT tick
    (last-occurrence anchoring — the documented fail-toward-quiet edge);
    the same shape WITHOUT the re-emission fires (test 3's pure-core
    counterpart, re-asserted here against the same inputs)."""
    state_file = tmp_path / "poll-state.json"
    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}", pid_alive="0")
    _poll(state_file)

    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}\n{NEW_CELL}\n{DONE}", pid_alive="0")
    result = _poll(state_file)
    assert result.post_done_phase_advisory_posted is False  # swallowed: quiet
    assert _advisory_calls(pp.post_event) == []
    assert _state(state_file)["post_done_advisory_posted"] == "0"
    # The counterpart WITHOUT the re-emission fires (pure core, same inputs).
    u = _update(
        log_tail=f"{TRAIN}\n{DONE}\n{NEW_CELL}",
        prev_done_line=DONE,
        prev_done_epoch=int(time.time()) - 60,
        prev_done_pod=POD,
    )
    assert u.should_post is True


def test_truncation_symmetry_long_done_line_no_false_fire(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test 24 (C-8): a done line longer than ``_POST_DONE_LINE_MAX`` is
    recorded truncated AND compared truncated — an identical re-poll never
    false-fires (record/compare symmetry)."""
    state_file = tmp_path / "poll-state.json"
    long_done = DONE + " " + "x" * 500
    assert len(long_done) > pp._POST_DONE_LINE_MAX
    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{long_done}", pid_alive="0")
    result = _poll(state_file)
    assert result.status == "done"
    saved = _state(state_file)
    assert saved["post_done_line"] == long_done[: pp._POST_DONE_LINE_MAX]

    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{long_done}", pid_alive="0")
    result = _poll(state_file)
    assert result.post_done_phase_advisory_posted is False
    assert result.post_done_phase_lines == ()
    assert _advisory_calls(pp.post_event) == []
    assert _state(state_file)["post_done_advisory_posted"] == "0"
