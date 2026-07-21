"""#1556 trigger-dense structural digest for ``log_tail_excerpt``.

On a workload declared trigger-dense (task tag ``trigger-dense``, set at
dispatch per ``.claude/rules/trigger-dense-review.md``'s recognition
heuristic), ``poll_pipeline.poll_once`` replaces the raw 5-line
``log_tail_excerpt`` with a bounded structural digest (pattern counts +
status/phase/pid liveness + the winning log source/path + tail size and
staleness — NO raw log line content), and the post-done phase-consistency
surfaces (``PollResult.post_done_phase_lines`` -> CLI JSON, plus the
``[post-done-phase-advisory]`` note) are reduced to bare ``[phase=<token>]``
tokens. ``crash_signature`` (the in-process machine surface feeding the
``backend_poll`` CUDA-IMA / OUR_CODE_FRAME failover predicates) stays raw and
is never emitted to stdout.

These tests pin (plan #1556 §5, tests 1-8):

1. the pure digest builder is structural and content-free;
2. ``_freshest_wide_tail`` parity with the pinned #775
   ``_tail_excerpt_and_crash_signature`` helper across the four log layouts
   (freshness permutations, tie->main, empty-tail skip, all-empty fallback);
3. the REAL ``_issue_trigger_dense`` body across its four read paths
   (autospec'd ``get_task`` boundary; INFO on missing-task -> raw, loud
   WARNING on unreadable state -> digest);
4. ``poll_once`` digests when trigger-dense (and ``crash_signature`` still
   carries the RAW wide tail on a dead poll);
5. the untagged path is unchanged (raw 5-line excerpt);
6. the digest's CUDA-IMA structural flag matches the REAL
   ``backend_poll.CUDA_IMA_SIGNATURE`` (cross-module contract + the
   mirrored-pattern equality pin);
7. the CLI JSON emits the additive ``log_tail_digested`` key (both values;
   synthetic issue id / monkeypatched predicate — never a real task id);
8. a tagged run's WHOLE serialized JSON line is content-free (the
   field-scope-illusion killer): a payload sentinel in the wide tail never
   reaches stdout, ``post_done_phase_lines`` carries bare tokens, and
   ``crash_signature`` is not a JSON key.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    """Load a ``scripts/*.py`` file as a module (mirrors the
    ``tests/test_poll_pipeline_post_done_guard.py`` loader)."""
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_digest_under_test")
bp = _load_script_module("backend_poll.py", "backend_poll_digest_under_test")

# Synthetic issue id (never a real task — the live-state coupling class the
# #1556 plan §4 row 11 names); distinct from the sibling poll test files'
# ids (9664 / 9813 / 9999 / 9983 / 9704).
ISSUE = 9556
POD = "pod-9556"
LOG = "/workspace/logs/issue-9556.log"
PIDFILE = "/workspace/logs/issue-9556.pid"

# Distinctive payload sentinel standing in for gated-content tail text — the
# thing that must NEVER reach an orchestrator-facing surface on a tagged run.
SENTINEL = "XYZZYPAYLOAD9"

TRAIN = "2026-07-04 00:00:00 [phase=training step=5/100]"
DONE = "2026-07-04 00:00:01 [phase=done] run complete"
NEW_CELL = f"2026-07-04 00:10:00 [phase=eval_cell_3] {SENTINEL} worker resumed"

_DIGEST_KW: dict[str, Any] = dict(
    status="dead",
    current_phase="training",
    pid_alive=False,
    source="main",
    log_path=LOG,
    mtime_sec_ago=12,
)


# ── 1. pure digest builder ───────────────────────────────────────────────────


def test_digest_tail_excerpt_structural_and_content_free() -> None:
    """Plan §5 test 1: counts + structural fields present, payload absent."""
    tail = "\n".join(
        [
            f"2026 ERROR alpha {SENTINEL}",
            "2026 error beta",
            "2026 ERROR gamma",
            "Traceback (most recent call last):",
            "worker got killed by signal 9",
            "a perfectly ordinary progress line",
        ]
    )
    d = pp._digest_tail_excerpt(tail, **_DIGEST_KW)
    assert d.startswith("[trigger-dense digest]")
    # Pattern counts (case-insensitive per-line substring counts).
    assert "error=3" in d
    assert "traceback=1" in d
    assert "killed=1" in d
    assert "oom=0" in d
    # Structural fields.
    assert "status=dead" in d
    assert "phase=training" in d
    assert "pid_alive=False" in d
    assert "source=main" in d
    assert f"log={LOG}" in d
    assert "tail_lines=6" in d
    assert "tail_bytes=" in d
    assert "log_mtime_sec_ago=12" in d
    # NO raw log line content is inlined.
    assert SENTINEL not in d
    assert "alpha" not in d and "ordinary progress" not in d
    # Bounded + single-line.
    assert "\n" not in d


# ── 2. freshest-wide-tail parity with the pinned #775 helper ─────────────────


def test_freshest_wide_tail_parity_with_excerpt_helper() -> None:
    """Plan §5 test 2: `_freshest_wide_tail` returns the same tail
    `_tail_excerpt_and_crash_signature` slices from, plus the right label."""
    main_t = "m1\nm2\nm3\nm4\nm5\nm6"
    cell_t = "c1\nc2"
    phase_t = "p1"
    shard_t = "s1"

    def probe(**tails: str) -> dict[str, str]:
        base = {
            "log_tail": main_t,
            "cell_log_tail": cell_t,
            "phase_log_tail": phase_t,
            "shard_log_tail": shard_t,
        }
        base.update(tails)
        return base

    cases: list[tuple[dict[str, str], dict[str, int], str, str]] = [
        # (probe, mtimes, expected_source, expected_tail)
        (
            probe(),
            dict(
                mtime_epoch=900,
                cell_mtime_epoch=1,
                phase_log_mtime_epoch=2,
                shard_log_mtime_epoch=3,
            ),
            "main",
            main_t,
        ),
        (
            probe(),
            dict(
                mtime_epoch=100,
                cell_mtime_epoch=900,
                phase_log_mtime_epoch=2,
                shard_log_mtime_epoch=3,
            ),
            "cell",
            cell_t,
        ),
        (
            probe(),
            dict(
                mtime_epoch=100,
                cell_mtime_epoch=1,
                phase_log_mtime_epoch=900,
                shard_log_mtime_epoch=3,
            ),
            "phase",
            phase_t,
        ),
        (
            probe(),
            dict(
                mtime_epoch=100,
                cell_mtime_epoch=1,
                phase_log_mtime_epoch=2,
                shard_log_mtime_epoch=900,
            ),
            "shard",
            shard_t,
        ),
        # mtime tie -> main (FIRST-max semantics, main first in the list).
        (
            probe(),
            dict(
                mtime_epoch=500,
                cell_mtime_epoch=500,
                phase_log_mtime_epoch=1,
                shard_log_mtime_epoch=1,
            ),
            "main",
            main_t,
        ),
        # fresh-but-EMPTY phase log never blanks out a populated source.
        (
            probe(phase_log_tail=""),
            dict(
                mtime_epoch=100,
                cell_mtime_epoch=200,
                phase_log_mtime_epoch=900,
                shard_log_mtime_epoch=1,
            ),
            "cell",
            cell_t,
        ),
        # no source has a tail -> fall back to main.
        (
            probe(log_tail="", cell_log_tail="", phase_log_tail="", shard_log_tail=""),
            dict(
                mtime_epoch=100,
                cell_mtime_epoch=900,
                phase_log_mtime_epoch=2,
                shard_log_mtime_epoch=3,
            ),
            "main",
            "",
        ),
    ]
    for prb, mtimes, want_source, want_tail in cases:
        source, wide = pp._freshest_wide_tail(prb, **mtimes)
        assert (source, wide) == (want_source, want_tail), (mtimes, source)
        excerpt, sig = pp._tail_excerpt_and_crash_signature(prb, status="dead", **mtimes)
        # Parity: the pinned helper slices the SAME tail the extraction names.
        assert excerpt == "\n".join(wide.splitlines()[-5:])
        assert sig == wide  # status="dead" -> crash_signature IS the wide tail
    # And on a non-dead poll the pinned helper's crash_signature stays None.
    _, sig_running = pp._tail_excerpt_and_crash_signature(
        probe(), status="running", mtime_epoch=900, cell_mtime_epoch=1
    )
    assert sig_running is None


# ── 3. the REAL predicate body across its four read paths ────────────────────


def test_issue_trigger_dense_tag_read_paths(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Plan §5 test 3: real `_issue_trigger_dense` body, autospec'd boundary."""
    real_get_task = pp.get_task  # captured ONCE — later autospecs must spec the REAL fn
    # tag present -> True (and the boundary is called with the issue id).
    fake = create_autospec(
        real_get_task,
        return_value={"status": "running", "frontmatter": {"tags": ["trigger-dense"]}},
    )
    monkeypatch.setattr(pp, "get_task", fake)
    assert pp._issue_trigger_dense(4242) is True
    fake.assert_called_once_with(4242)

    # tag absent -> False (other tags do not fire it).
    monkeypatch.setattr(
        pp,
        "get_task",
        create_autospec(
            real_get_task,
            return_value={"status": "running", "frontmatter": {"tags": ["keep-running"]}},
        ),
    )
    assert pp._issue_trigger_dense(4242) is False

    # missing task (FileNotFoundError; StaleTaskPathError is its subclass and
    # takes the same arm) -> False + one INFO line.
    from explore_persona_space.task_workflow import StaleTaskPathError

    assert issubclass(StaleTaskPathError, FileNotFoundError)
    monkeypatch.setattr(
        pp,
        "get_task",
        create_autospec(real_get_task, side_effect=FileNotFoundError("task #4242 not found")),
    )
    with caplog.at_level(logging.INFO, logger="poll_pipeline"):
        caplog.clear()
        assert pp._issue_trigger_dense(4242) is False
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert infos, "missing-task arm must log INFO (not silent)"
    assert "not found" in infos[0].getMessage()
    assert "raw excerpt" in infos[0].getMessage()

    # unreadable task state (RuntimeError branch-guard class) -> True (fail
    # SAFE toward digest) + a loud per-tick WARNING NAMING the failure.
    monkeypatch.setattr(
        pp,
        "get_task",
        create_autospec(real_get_task, side_effect=RuntimeError("branch-guard: HEAD is not main")),
    )
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        caplog.clear()
        assert pp._issue_trigger_dense(4242) is True
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warns, "unreadable-state arm must WARN loudly (not swallow)"
    msg = warns[0].getMessage()
    assert "RuntimeError" in msg and "branch-guard: HEAD is not main" in msg
    assert "digest" in msg


# ── poll_once / main() harness (mirrors test_poll_pipeline_post_done_guard) ──


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
    trigger_dense: bool | None = None,
) -> MagicMock:
    """Monkeypatch poll_pipeline's I/O boundary with a fully-controlled probe.

    ``trigger_dense`` None leaves the REAL predicate in place (the synthetic
    ``ISSUE`` id then takes the FileNotFoundError->raw arm); a bool pins it.
    Returns the ``post_event`` mock for advisory-note inspection.
    """
    resolved_mtime = int(time.time()) - 10

    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        remote = cmd[-1]
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        stdout = _probe_stdout(mtime_epoch=resolved_mtime, tail=tail, pid_alive=pid_alive)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    post_mock = MagicMock()
    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", post_mock)
    monkeypatch.setattr(pp, "_telegram_push", MagicMock(return_value=True))
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch: 10800.0)
    if trigger_dense is not None:
        monkeypatch.setattr(pp, "_issue_trigger_dense", lambda issue: trigger_dense)
    return post_mock


def _poll(state_file: Path):
    return pp.poll_once(issue=ISSUE, pod=POD, log_path=LOG, pid_file=PIDFILE, state_file=state_file)


def _main_json(capsys: pytest.CaptureFixture[str], state_file: Path) -> dict[str, Any]:
    """Drive ``main()`` and parse the single stdout JSON line."""
    rc = pp.main(
        [
            "--issue",
            str(ISSUE),
            "--pod",
            POD,
            "--log",
            LOG,
            "--pid-file",
            PIDFILE,
            "--state-file",
            str(state_file),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out.strip()
    return json.loads(out)


DEAD_TAIL = "\n".join(
    [
        TRAIN,
        f"RuntimeError: boom {SENTINEL}",
        "2026 ERROR something failed",
        "Traceback (most recent call last):",
        "  raise RuntimeError",
        "worker exited",
    ]
)


# ── 4/5. poll_once digested vs untagged ──────────────────────────────────────


def test_poll_once_digests_when_trigger_dense(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Plan §5 test 4: digested excerpt + flag; crash_signature stays RAW."""
    _patch_pod(monkeypatch, tail=DEAD_TAIL, pid_alive="0", trigger_dense=True)
    result = _poll(tmp_path / "state.json")
    assert result.status == "dead"
    assert result.log_tail_excerpt.startswith("[trigger-dense digest]")
    assert result.log_tail_digested is True
    assert SENTINEL not in result.log_tail_excerpt
    # The in-process machine surface is byte-untouched: the RAW wide tail.
    assert result.crash_signature is not None
    assert SENTINEL in result.crash_signature
    assert result.crash_signature == DEAD_TAIL


def test_poll_once_untagged_path_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Plan §5 test 5: predicate False -> the pre-change raw 5-line slice."""
    _patch_pod(monkeypatch, tail=DEAD_TAIL, pid_alive="0", trigger_dense=False)
    result = _poll(tmp_path / "state.json")
    assert result.log_tail_excerpt == "\n".join(DEAD_TAIL.splitlines()[-5:])
    assert result.log_tail_digested is False
    assert SENTINEL in result.crash_signature


# ── 6. CUDA-IMA cross-module contract ────────────────────────────────────────


def test_digest_cuda_ima_flag_matches_backend_poll_signature() -> None:
    """Plan §5 test 6: the digest keeps the #775 marker-note fallback alive."""
    # The mirrored regex is pinned byte-in-sync with the real one
    # (poll_pipeline cannot import backend_poll — circular).
    assert pp._CUDA_IMA_SIGNATURE.pattern == bp.CUDA_IMA_SIGNATURE.pattern
    assert pp._CUDA_IMA_SIGNATURE.flags == bp.CUDA_IMA_SIGNATURE.flags

    ima_tail = "RuntimeError: CUDA error: an illegal memory access was encountered\nmore lines"
    d_ima = pp._digest_tail_excerpt(ima_tail, **_DIGEST_KW)
    assert bp.CUDA_IMA_SIGNATURE.search(d_ima), "digested note must keep the #775 regex match"

    # v3: a tail matching ONLY the engine-dead alternatives also flags —
    # the flag fires on the REAL signature family, not a bare substring.
    engine_tail = "vllm EngineDeadError: engine core terminated\n"
    d_engine = pp._digest_tail_excerpt(engine_tail, **_DIGEST_KW)
    assert bp.CUDA_IMA_SIGNATURE.search(d_engine)
    # The count key and the flag may disagree here by design (count=0).
    assert "cuda_ima=0" in d_engine

    neither_tail = "ordinary ERROR line\nTraceback (most recent call last):\n  boring"
    d_neither = pp._digest_tail_excerpt(neither_tail, **_DIGEST_KW)
    assert bp.CUDA_IMA_SIGNATURE.search(d_neither) is None


# ── 7/8. CLI JSON surfaces ───────────────────────────────────────────────────


def test_cli_json_emits_log_tail_digested_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Plan §5 test 7: the additive key, both values (synthetic id -> the
    real predicate's FNF->raw arm; monkeypatched predicate -> the True arm)."""
    # Arm 1: REAL predicate on the synthetic issue id -> FileNotFoundError
    # -> raw excerpt, log_tail_digested False.
    _patch_pod(monkeypatch, tail=DEAD_TAIL, pid_alive="0", trigger_dense=None)
    js = _main_json(capsys, tmp_path / "state-raw.json")
    assert js["log_tail_digested"] is False
    assert js["log_tail_excerpt"] == "\n".join(DEAD_TAIL.splitlines()[-5:])

    # Arm 2: predicate pinned True -> digested excerpt, flag True.
    _patch_pod(monkeypatch, tail=DEAD_TAIL, pid_alive="0", trigger_dense=True)
    js = _main_json(capsys, tmp_path / "state-digest.json")
    assert js["log_tail_digested"] is True
    assert js["log_tail_excerpt"].startswith("[trigger-dense digest]")


def test_tagged_run_whole_json_content_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Plan §5 test 8 (v3): the field-scope-illusion killer — on a tagged run
    the ENTIRE serialized JSON line carries no raw tail text, the post-done
    lines are bare [phase=<token>] tokens, and crash_signature is not a key."""
    state_file = tmp_path / "state.json"
    # Tick 1: corroborated done arms the post-done episode.
    _patch_pod(monkeypatch, tail=f"{TRAIN}\n{DONE}", pid_alive="0", trigger_dense=True)
    r1 = _poll(state_file)
    assert r1.status == "done"
    # Tick 2 (via main(), the orchestrator-facing surface): a NEW phase line
    # carrying the payload sentinel appears after the done anchor.
    post_mock = _patch_pod(
        monkeypatch, tail=f"{TRAIN}\n{DONE}\n{NEW_CELL}", pid_alive="0", trigger_dense=True
    )
    rc = pp.main(
        [
            "--issue",
            str(ISSUE),
            "--pod",
            POD,
            "--log",
            LOG,
            "--pid-file",
            PIDFILE,
            "--state-file",
            str(state_file),
        ]
    )
    assert rc == 0
    raw_line = capsys.readouterr().out.strip()
    # The payload sentinel is absent from the ENTIRE serialized JSON line.
    assert SENTINEL not in raw_line
    js = json.loads(raw_line)
    # post_done surfaces carry only the extracted [phase=<token>] tokens.
    assert js["post_done_phase_lines"] == ["[phase=eval_cell_3]"]
    # Assumption-2 pin: crash_signature is never serialized to stdout.
    assert "crash_signature" not in js
    assert js["log_tail_digested"] is True
    # The advisory marker note (re-read by later orchestrator turns) is also
    # token-only: no raw line text beyond the [phase=<token>] token.
    advisory_notes = [
        str(c.kwargs.get("note", ""))
        for c in post_mock.call_args_list
        if str(c.kwargs.get("note", "")).startswith("[post-done-phase-advisory]")
    ]
    assert advisory_notes, "the post-done advisory must still fire on tagged runs"
    assert SENTINEL not in advisory_notes[0]
    assert "[phase=eval_cell_3]" in advisory_notes[0]
    assert "[phase=done]" in advisory_notes[0]  # done-line quote reduced to its token
