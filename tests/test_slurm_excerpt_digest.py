"""#1574 SLURM-lane trigger-dense structural digest for ``log_tail_excerpt``.

The SLURM monitor builds its OWN excerpt (the rsync'd ``job.out`` tail) and
emits it on THREE orchestrator-facing surfaces: the returned ``PollResult``,
the git-committed ``epm:cluster-poll`` marker note, and the
persisted-terminal synthesis. These tests pin, with a payload sentinel
standing in for gated-content tail text:

1. tagged: BOTH live surfaces (PollResult + the captured cluster-poll note)
   carry the shared digest, sentinel-free across the WHOLE note string;
2. untagged: ``log_tail[-2000:]`` verbatim on both surfaces (byte-equality);
3. detection is never gated: a tagged run whose raw tail carries
   ``PREFLIGHT_FAIL_MARKER`` still classifies dead / preflight-failed;
4. the UNKNOWN -> persisted-terminal synthesis digests when tagged (raw on
   the untagged twin).

Hermetic: the tag predicate is monkeypatched at the SHARED
``excerpt_digest`` module (the object ``slurm_monitor`` dispatches
through), so no test reads the live task registry. Synthetic issue id 9574.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from explore_persona_space.backends import excerpt_digest
from explore_persona_space.backends.slurm import get_cluster_config
from explore_persona_space.backends.slurm_monitor import build_poll_result

ISSUE = 9574
SENTINEL = "XYZZYPAYLOAD9574"


def _nibi():
    return get_cluster_config("nibi")


@pytest.fixture(autouse=True)
def _no_real_marker_posts(monkeypatch):
    """Defense in depth (mirrors tests/test_slurm_monitor.py): never let a
    monitor test shell out to the real ``task.py post-marker``."""
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm.post_marker_via_task_py",
        lambda **_kw: None,
    )


@pytest.fixture(autouse=True)
def _isolated_local_state_dir(tmp_path, monkeypatch):
    """Route ``_local_state_dir`` under pytest's ``tmp_path`` (mirrors the
    round-6 Mn3 fixture in tests/test_slurm_monitor.py)."""
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm_monitor._local_state_dir",
        lambda job_id: tmp_path / f"slurm-{job_id}",
    )


def _pin_predicate(monkeypatch, value: bool) -> list[int]:
    """Monkeypatch the SHARED tag predicate; returns the recorded issue ids."""
    calls: list[int] = []

    def fake(issue: int, **_kw) -> bool:
        calls.append(issue)
        return value

    monkeypatch.setattr(excerpt_digest, "issue_trigger_dense", fake)
    return calls


def _seed_local_state(
    tmp_path: Path,
    job_id: str,
    *,
    status_json_body: dict | None,
    job_out_lines: list[str] | None,
) -> None:
    local_dir = tmp_path / f"slurm-{job_id}"
    local_dir.mkdir(parents=True, exist_ok=True)
    if status_json_body is not None:
        (local_dir / "status.json").write_text(json.dumps(status_json_body))
    if job_out_lines is not None:
        (local_dir / "job.out").write_text("\n".join(job_out_lines))


def _poll_kwargs(job_id: str, now: datetime, markers: list[dict]) -> dict:
    return dict(
        issue=ISSUE,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir=f"/scratch/tjiral/eps/issue-{ISSUE}",
        log_path=f"/scratch/tjiral/eps/issue-{ISSUE}/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=lambda **kw: markers.append(kw),
        event_reader=lambda _issue: [],
    )


RAW_LINES = ["[phase=sft]", f"RuntimeError: boom {SENTINEL}", "2026 ERROR something failed"]


def test_build_poll_result_digests_pollresult_and_marker_when_tagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tagged RUNNING tick: the returned PollResult AND the captured
    ``epm:cluster-poll`` note both carry the digest; the sentinel is absent
    from the ENTIRE note string (the field-scope-illusion killer)."""
    job_id = "957401"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={
            "phase": "sft",
            "heartbeat_ts": fresh_ts,
            "gpu_busy": True,
            "exit_code": "",
        },
        job_out_lines=RAW_LINES,
    )
    calls = _pin_predicate(monkeypatch, True)
    markers: list[dict] = []

    poll = build_poll_result(**_poll_kwargs(job_id, now, markers))
    assert calls == [ISSUE], "exactly ONE fresh tag read per tick"
    assert poll.status == "running"
    assert poll.log_tail_excerpt.startswith("[trigger-dense digest]")
    assert "source=slurm_job_out" in poll.log_tail_excerpt
    assert SENTINEL not in poll.log_tail_excerpt

    cluster_polls = [m for m in markers if m.get("marker") == "epm:cluster-poll"]
    assert cluster_polls, "the transition tick must post epm:cluster-poll"
    note = str(cluster_polls[0]["note"])
    assert SENTINEL not in note
    body = json.loads(note)
    assert body["log_tail_excerpt"].startswith("[trigger-dense digest]")


def test_build_poll_result_untagged_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Untagged twin: ``log_tail[-2000:]`` verbatim on BOTH surfaces —
    string equality against the seeded raw tail, not startswith."""
    job_id = "957402"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={
            "phase": "sft",
            "heartbeat_ts": fresh_ts,
            "gpu_busy": False,
            "exit_code": "",
        },
        job_out_lines=RAW_LINES,
    )
    _pin_predicate(monkeypatch, False)
    markers: list[dict] = []

    poll = build_poll_result(**_poll_kwargs(job_id, now, markers))
    raw = "\n".join(RAW_LINES)
    assert poll.log_tail_excerpt == raw[-2000:]
    cluster_polls = [m for m in markers if m.get("marker") == "epm:cluster-poll"]
    assert cluster_polls
    body = json.loads(str(cluster_polls[0]["note"]))
    assert body["log_tail_excerpt"] == raw[-2000:]


def test_preflight_fail_detection_fires_on_digested_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Detection is never gated: the raw tail's PREFLIGHT_FAIL_MARKER still
    classifies dead / preflight-failed on a tagged run, while the emitted
    excerpt is the sentinel-free digest carrying the verdict fields."""
    job_id = "957403"
    now = datetime.now(tz=UTC)
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body=None,
        job_out_lines=[f"[phase=preflight-failed] {SENTINEL} preflight died"],
    )
    _pin_predicate(monkeypatch, True)
    markers: list[dict] = []

    poll = build_poll_result(**_poll_kwargs(job_id, now, markers))
    assert poll.status == "dead"
    assert poll.current_phase == "preflight-failed"
    assert poll.log_tail_excerpt.startswith("[trigger-dense digest]")
    assert "status=dead" in poll.log_tail_excerpt
    assert "phase=preflight-failed" in poll.log_tail_excerpt
    assert SENTINEL not in poll.log_tail_excerpt


def _persisted_terminal_event(job_id: str) -> dict:
    body = {
        "job_id": job_id,
        "cluster": "nibi",
        "slurm_state": "FAILED",
        "exit_code": "1:0",
        "observed_at": "2026-07-21T00:00:00Z",
        "next_action": "investigate",
        "status": "dead",
    }
    return {"kind": "epm:cluster-terminal", "note": json.dumps(body)}


@pytest.mark.parametrize("tagged", [True, False])
def test_persisted_terminal_synthesis_digests_when_tagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tagged: bool
) -> None:
    """UNKNOWN SLURM state + a persisted ``epm:cluster-terminal`` marker:
    the synthesized PollResult's excerpt digests when tagged; the untagged
    twin keeps the raw local tail verbatim."""
    job_id = "957404"
    now = datetime.now(tz=UTC)
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body=None,
        job_out_lines=[f"RuntimeError: boom {SENTINEL}", "worker exited"],
    )
    _pin_predicate(monkeypatch, tagged)
    markers: list[dict] = []
    kwargs = _poll_kwargs(job_id, now, markers)
    kwargs["state_querier"] = lambda *, robot_alias, job_id: {
        "status": "UNKNOWN",
        "exit_code": None,
    }
    kwargs["event_reader"] = lambda _issue: [_persisted_terminal_event(job_id)]

    poll = build_poll_result(**kwargs)
    assert poll.status == "dead"
    assert poll.current_phase == "failed"
    if tagged:
        assert poll.log_tail_excerpt.startswith("[trigger-dense digest]")
        assert SENTINEL not in poll.log_tail_excerpt
    else:
        raw = f"RuntimeError: boom {SENTINEL}\nworker exited"
        assert poll.log_tail_excerpt == raw[-2000:]
