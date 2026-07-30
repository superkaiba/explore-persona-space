"""Tests for the GCP-lane GPU-idle advisory + escalation parity (#730).

The GCP poller (``scripts/backend_poll.py`` ``main()``) gained the same two
GPU-idle tiers the RunPod lane already has (advisory #518/#537, escalation
#664/#727), REUSING (importing, not re-implementing) the decision/post helpers
from ``scripts/poll_pipeline.py``:

* a fail-soft ``nvidia-smi``-over-``gcloud compute ssh`` GPU-util probe
  (``GcpBackend._gcp_gpu_util_probe``) — returns ``"unknown"`` on ANY failure;
* a sibling GPU-idle state file (``issue-<N>-gpu-idle-state.json``) read/written
  via ``backend_poll._{gpu_idle_state_path,load_gpu_idle_state,save_gpu_idle_state}``;
* the two RunPod-lane wiring fns wired into ``main()`` for
  ``handle.backend == "gcp" and status == "running"`` ticks, emitting two new
  serialized JSON fields ``gcp_gpu_idle_{advisory,escalation}_posted``.

These tests pin: the probe CSV parse + fail-soft contract; the advisory +
escalation thresholds on the GCP lane; the NEVER-stops-the-VM invariant (a
static argv guard); per-phase idempotency; and that ``_phase_is_cpu_only``
classifies the ACTUAL GCP ``eps/phase`` vocabulary (coarse ``"workload"``,
``"setup"``, ``"done"``, ``"unknown"``) the way the GCP lane threads it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

import scripts.backend_poll as bp
import scripts.poll_pipeline as pp
from explore_persona_space.backends.base import PollResult, RunHandle
from explore_persona_space.backends.gcp import GcloudRunResult, GcpBackend
from explore_persona_space.backends.issue_dispatch import write_handle_sidecar

# ── _gcp_gpu_util_probe parse + fail-soft ─────────────────────────────────────


def _backend_recording_run(returns: Any = None, *, raises: BaseException | None = None):
    """A GcpBackend whose injected runner RECORDS every argv it is handed.

    ``returns`` is the GcloudRunResult the runner yields (when not raising);
    ``raises`` makes the runner raise instead. ``backend.recorded`` is the list
    of argvs passed to the runner this test.
    """
    recorded: list[list[str]] = []

    def _runner(argv):
        recorded.append(list(argv))
        if raises is not None:
            raise raises
        return returns

    backend = GcpBackend(runner=_runner, marker_poster=lambda **_kw: None)
    backend.recorded = recorded  # type: ignore[attr-defined]
    return backend


def _gcp_handle(extra: dict | None = None) -> RunHandle:
    return RunHandle(
        backend="gcp",
        cluster=None,
        job_id="instance-fake-1",
        pod_name="eps-issue-730",
        scratch_dir="/workspace/eps-issue-730",
        log_path="/workspace/logs/issue-730.log",
        extra=dict(extra if extra is not None else {"issue": 730, "zone": "us-central1-a"}),
    )


def test_gcp_gpu_util_probe_parses_csv() -> None:
    """A newline-separated nvidia-smi reply (rc=0) is normalized to a comma-joined
    util string the consumer (_gpu_idle) understands."""
    backend = _backend_recording_run(
        GcloudRunResult(returncode=0, stdout="0\n0\n0\n0\n", stderr="")
    )
    assert backend._gcp_gpu_util_probe(_gcp_handle(), "us-central1-a") == "0,0,0,0"


def test_gcp_gpu_util_probe_normalizes_busy_cards() -> None:
    backend = _backend_recording_run(
        GcloudRunResult(returncode=0, stdout="0, 0, 95, 0\n", stderr="")
    )
    assert backend._gcp_gpu_util_probe(_gcp_handle(), "us-central1-a") == "0,0,95,0"


@pytest.mark.parametrize(
    "result,raises",
    [
        (GcloudRunResult(returncode=255, stdout="", stderr="ssh: connect refused"), None),
        (GcloudRunResult(returncode=0, stdout="", stderr=""), None),  # empty stdout
        (GcloudRunResult(returncode=0, stdout="garbage\nnot,a,number\n", stderr=""), None),
        (None, RuntimeError("transport blew up")),  # runner raises
    ],
)
def test_gcp_gpu_util_probe_fail_soft(result, raises) -> None:
    """rc!=0, empty stdout, a non-numeric token, and a raised exception EACH
    yield the literal "unknown" — never a crash, never a false idle."""
    backend = _backend_recording_run(result, raises=raises)
    assert backend._gcp_gpu_util_probe(_gcp_handle(), "us-central1-a") == "unknown"


def test_gcp_gpu_util_probe_uses_sudo_nvidia_smi_over_ssh() -> None:
    """The probe reuses the GCP drain SSH pattern: gcloud compute ssh <name>
    --command=sudo -n nvidia-smi ... --zone=<zone> (matched to _drain_sentinels)."""
    backend = _backend_recording_run(GcloudRunResult(returncode=0, stdout="0\n", stderr=""))
    backend._gcp_gpu_util_probe(_gcp_handle(), "us-central1-a")
    (argv,) = backend.recorded  # type: ignore[attr-defined]
    joined = " ".join(argv)
    assert "compute" in argv and "ssh" in argv and "eps-issue-730" in argv
    assert "--zone=us-central1-a" in argv
    assert "sudo -n nvidia-smi" in joined
    assert "--query-gpu=utilization.gpu" in joined


# ── state-file round-trip ─────────────────────────────────────────────────────


def test_gpu_idle_state_path_is_handle_sidecar_sibling(tmp_path: Path) -> None:
    sidecar = tmp_path / "issue-730-handle.json"
    assert bp._gpu_idle_state_path(sidecar) == tmp_path / "issue-730-gpu-idle-state.json"


def test_gpu_idle_state_path_never_collides_with_handle_file(tmp_path: Path) -> None:
    """Per Codex round-1 finding: an arbitrary --handle-file name (NOT ending in
    '-handle.json', which the documented CLI flag honors verbatim) must NEVER
    resolve to the handle sidecar itself — otherwise the GPU-idle block would
    write its bookkeeping ONTO the run handle and corrupt it (the next poll reads
    it as a RunHandle -> unreadable -> false `status: dead` on a live job).

    The OLD naive ``sidecar.name.replace("-handle.json", "-gpu-idle-state.json")``
    is a no-op when the substring is absent, so it returned ``sidecar`` itself
    for every non-conforming name, e.g.
    ``"custom.json".replace("-handle.json", "-gpu-idle-state.json") == "custom.json"``.
    """
    # Canonical name -> the -handle.json substitution gives the documented sibling.
    canonical = tmp_path / "issue-730-handle.json"
    assert bp._gpu_idle_state_path(canonical) == (tmp_path / "issue-730-gpu-idle-state.json")

    # Non-conforming names must produce DISTINCT siblings (the bug case).
    for name in ("custom.json", "handle.json", "pod-runtime.json", "no-ext"):
        sidecar = tmp_path / name
        state_path = bp._gpu_idle_state_path(sidecar)
        assert state_path != sidecar, f"state-path collides with handle sidecar for name={name!r}"
        assert state_path.parent == sidecar.parent  # still a sibling in the same dir
        assert state_path.name.endswith("-gpu-idle-state.json")


def test_gpu_idle_state_round_trip_and_fail_soft(tmp_path: Path) -> None:
    path = tmp_path / "issue-730-gpu-idle-state.json"
    assert bp._load_gpu_idle_state(path) == {}  # absent -> {}
    payload = {
        "phase": "p3_upload",
        "gpu_idle_since_epoch": "1000",
        "gpu_idle_advised_phases": "p3_upload",
        "gpu_idle_escalated_phases": "",
    }
    bp._save_gpu_idle_state(path, payload)
    assert bp._load_gpu_idle_state(path) == payload
    # Corrupt body -> {} (fail-soft), never raises.
    path.write_text("{not json")
    assert bp._load_gpu_idle_state(path) == {}
    # Non-dict JSON -> {}.
    path.write_text("[1, 2, 3]")
    assert bp._load_gpu_idle_state(path) == {}


# ── advisory + escalation thresholds on the GCP lane ──────────────────────────
#
# The decision cores are exhaustively unit-tested in
# tests/test_poll_gpu_idle_escalation.py; here we pin the GCP-lane WIRING
# (the imported _maybe_* fns + the seeded sibling state file drive the posts).


def _seed_idle_state(path: Path, *, since_epoch: int, phase: str) -> None:
    bp._save_gpu_idle_state(
        path,
        {
            "phase": phase,
            "gpu_idle_since_epoch": str(since_epoch),
            "gpu_idle_advised_phases": "",
            "gpu_idle_escalated_phases": "",
        },
    )


def test_gcp_advisory_posts_after_threshold(tmp_path: Path, monkeypatch) -> None:
    """All-idle GPUs in a CPU-only phase whose seeded idle span exceeds the
    advisory min -> the advisory wiring posts a [gpu-idle-advisory] marker."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    path = tmp_path / "issue-730-gpu-idle-state.json"
    now = 100_000
    _seed_idle_state(path, since_epoch=now - pp.GPU_IDLE_ADVISORY_MIN * 60, phase="p3_upload")
    prev = bp._load_gpu_idle_state(path)
    _idle_since, advised, advisory_posted = pp._maybe_post_gpu_idle_advisory(
        issue=730,
        pod="eps-issue-730",
        status="running",
        gpu_util="0,0,0,0,0,0,0,0",
        current_phase="p3_upload",
        prev_state=prev,
        now_epoch=now,
    )
    assert advisory_posted is True
    assert "p3_upload" in advised
    assert any(p["key"] == "epm:progress" and p.get("gpu_idle_advisory") for p in posted)
    assert any("[gpu-idle-advisory]" in (p.get("note") or "") for p in posted)


def test_gcp_escalation_posts_and_pushes_multi_gpu(tmp_path: Path, monkeypatch) -> None:
    """A MULTI-GPU pod idle past the escalation min in a CPU-only phase ->
    [gpu-idle-escalation] marker posted AND a Telegram push fired; a single-GPU
    pod under the SAME conditions does NOT escalate."""
    posted: list[dict] = []
    pushes: list[str] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: pushes.append(msg) or True)
    now = 100_000
    since = now - pp.GPU_IDLE_ESCALATION_MIN * 60

    # Multi-GPU -> escalates + pushes.
    escalated, _counts, escalation_posted = pp._maybe_escalate_gpu_idle(
        issue=730,
        pod="eps-issue-730",
        status="running",
        gpu_util="0,0,0,0,0,0,0,0",
        current_phase="p3_upload",
        idle_since_epoch=since,
        prev_state={"gpu_idle_escalated_phases": ""},
        now_epoch=now,
    )
    assert escalation_posted is True
    assert "p3_upload" in escalated
    assert len(pushes) == 1
    assert any(p["key"] == "epm:progress" and p.get("gpu_idle_escalation") for p in posted)

    # Single-GPU under identical conditions -> NO escalation, NO push.
    pushes.clear()
    _escalated, _counts2, single_posted = pp._maybe_escalate_gpu_idle(
        issue=730,
        pod="eps-issue-730",
        status="running",
        gpu_util="0",
        current_phase="p3_upload",
        idle_since_epoch=since,
        prev_state={"gpu_idle_escalated_phases": ""},
        now_epoch=now,
    )
    assert single_posted is False
    assert pushes == []


def test_gcp_idempotent_one_per_phase(tmp_path: Path, monkeypatch) -> None:
    """Two consecutive ticks in the SAME phase (state round-tripped through the
    sibling file) -> escalate on tick 1, NOT tick 2; a phase CHANGE re-arms."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: True)
    path = tmp_path / "issue-730-gpu-idle-state.json"
    now = 100_000
    since = now - pp.GPU_IDLE_ESCALATION_MIN * 60
    _seed_idle_state(path, since_epoch=since, phase="p3_upload")

    def _tick(phase: str, now_epoch: int) -> bool:
        prev = bp._load_gpu_idle_state(path)
        idle_since, advised, _adv = pp._maybe_post_gpu_idle_advisory(
            issue=730,
            pod="eps-issue-730",
            status="running",
            gpu_util="0,0,0,0,0,0,0,0",
            current_phase=phase,
            prev_state=prev,
            now_epoch=now_epoch,
        )
        escalated, _counts, escalation_posted = pp._maybe_escalate_gpu_idle(
            issue=730,
            pod="eps-issue-730",
            status="running",
            gpu_util="0,0,0,0,0,0,0,0",
            current_phase=phase,
            idle_since_epoch=idle_since,
            prev_state=prev,
            now_epoch=now_epoch,
        )
        bp._save_gpu_idle_state(
            path,
            {
                "phase": phase,
                "gpu_idle_since_epoch": str(idle_since),
                "gpu_idle_advised_phases": ",".join(sorted(advised)),
                "gpu_idle_escalated_phases": ",".join(sorted(escalated)),
            },
        )
        return escalation_posted

    assert _tick("p3_upload", now) is True  # tick 1: fires
    assert _tick("p3_upload", now + 60) is False  # tick 2, same phase: de-duped
    # A phase change restarts the span -> the new phase has not yet aged, so it
    # does NOT immediately escalate (re-arm, then age past the threshold).
    assert _tick("p5_upload", now + 120) is False
    assert _tick("p5_upload", now + 120 + pp.GPU_IDLE_ESCALATION_MIN * 60) is True


# ── #1752 escalate-in-kind: the width-re-eval tier after N repeats ────────────
#
# The per-phase dedup (gpu_idle_escalated_phases) is run-scope-cleared on every
# fresh epm:run-launched (#1033), so a phase idle across relaunches re-fires a
# byte-identical [gpu-idle-escalation] forever (#1689: fit_ladder, ~14h at 0%
# GPU). The fix counts escalations per phase ACROSS run epochs
# (gpu_idle_escalation_counts, which deliberately survives both resets) and
# switches KIND to [gpu-idle-width-reeval] at count >= GPU_IDLE_WIDTH_REEVAL_N
# (default 3). The poller still NEVER stops the pod.


def _escalate_once(
    monkeypatch,
    *,
    prev_state: dict[str, str],
    posted: list[dict],
    pushes: list[str],
    post_raises: bool = False,
):
    """Drive ONE ``_maybe_escalate_gpu_idle`` call at the escalation threshold
    on an 8-GPU pod in the #664 trigger phase; returns the wiring 3-tuple."""

    def _post(issue, key, **kw):
        if post_raises:
            raise RuntimeError("marker post failed")
        posted.append({"key": key, **kw})

    monkeypatch.setattr(pp, "post_event", _post)
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: pushes.append(msg) or True)
    now = 100_000
    return pp._maybe_escalate_gpu_idle(
        issue=1752,
        pod="pod-1752",
        status="running",
        gpu_util="0,0,0,0,0,0,0,0",
        current_phase="p3_upload",
        idle_since_epoch=now - pp.GPU_IDLE_ESCALATION_MIN * 60,
        prev_state=prev_state,
        now_epoch=now,
    )


def test_escalation_counts_one_and_two_post_identical_note(monkeypatch) -> None:
    """Escalations #1 and #2 for a phase (below the default N=3) post the
    EXISTING [gpu-idle-escalation] note — no width-reeval prefix, no
    ``gpu_idle_width_reeval`` / ``escalation_repeat`` extras, the existing
    push wording — so behavior below N is unchanged from today."""
    for prior, expected_n in (("", 1), ("p3_upload:1", 2)):
        posted: list[dict] = []
        pushes: list[str] = []
        escalated, counts, fired = _escalate_once(
            monkeypatch,
            prev_state={
                "gpu_idle_escalated_phases": "",
                "gpu_idle_escalation_counts": prior,
            },
            posted=posted,
            pushes=pushes,
        )
        assert fired is True
        assert "p3_upload" in escalated
        assert counts == {"p3_upload": expected_n}
        (marker,) = posted
        assert marker["note"].startswith("[gpu-idle-escalation]")
        assert "#664 spend-leak class" in marker["note"]
        assert marker.get("gpu_idle_escalation") is True
        assert "gpu_idle_width_reeval" not in marker
        assert "escalation_repeat" not in marker
        (push,) = pushes
        assert "WIDTH RE-EVAL" not in push


def test_escalation_count_three_switches_to_width_reeval(monkeypatch) -> None:
    """The 3rd escalation for the SAME phase — counted across relaunches via
    ``gpu_idle_escalation_counts`` (the EMPTY escalated set simulates a fresh
    run epoch after the #1033 reset) — posts the DISTINCT
    [gpu-idle-width-reeval] note stating the running count, the concrete
    downsize recipe, and the explicit nothing-stopped statement, with a
    width-re-eval-worded Telegram push."""
    posted: list[dict] = []
    pushes: list[str] = []
    escalated, counts, fired = _escalate_once(
        monkeypatch,
        prev_state={
            "gpu_idle_escalated_phases": "",  # fresh run epoch: dedup re-armed
            "gpu_idle_escalation_counts": "p3_upload:2",  # 2 fires on prior runs
        },
        posted=posted,
        pushes=pushes,
    )
    assert fired is True
    assert counts == {"p3_upload": 3}
    assert "p3_upload" in escalated
    (marker,) = posted
    note = marker["note"]
    assert note.startswith("[gpu-idle-width-reeval]")
    assert "escalation #3" in note  # the running count n is stated in the text
    # The concrete downsize recipe (persist -> terminate wide -> re-provision
    # narrow, or route the CPU phase off-pod) + the explicit no-action line.
    assert "persist resume" in note
    assert "re-provision narrow" in note
    assert "off-pod" in note
    assert "NOTHING was stopped" in note
    assert marker.get("gpu_idle_escalation") is True  # existing consumers unchanged
    assert marker.get("gpu_idle_width_reeval") is True
    assert marker.get("escalation_repeat") == 3
    (push,) = pushes
    assert "WIDTH RE-EVAL" in push
    assert "escalation #3" in push
    assert "re-provision narrow" in push  # the push names the recipe too
    assert "nothing stopped" in push


def test_escalation_post_failure_does_not_increment_count(monkeypatch) -> None:
    """A marker-post failure records NEITHER the phase NOR the count — the
    next tick retries at the same n (the existing retry semantics extended to
    the counter, so a failed post can never burn a width-reeval slot)."""
    posted: list[dict] = []
    pushes: list[str] = []
    escalated, counts, fired = _escalate_once(
        monkeypatch,
        prev_state={
            "gpu_idle_escalated_phases": "",
            "gpu_idle_escalation_counts": "p3_upload:2",
        },
        posted=posted,
        pushes=pushes,
        post_raises=True,
    )
    assert fired is False
    assert counts == {"p3_upload": 2}  # NOT incremented
    assert "p3_upload" not in escalated
    assert posted == []
    assert pushes == []


def test_escalation_count_survives_both_run_scope_resets(monkeypatch) -> None:
    """The count ACCUMULATES across run epochs: a simulated
    ``_tripwire_run_scope`` clear (fresh epm:run-launched) AND a simulated
    ``_scope_idle_state_to_attempt`` clear (fresh GCP instance incarnation)
    each wipe ``gpu_idle_escalated_phases`` but KEEP
    ``gpu_idle_escalation_counts`` — so three escalations across three run
    epochs reach the width-re-eval tier."""
    state: dict[str, str] = {
        "gpu_idle_escalated_phases": "",
        "gpu_idle_escalation_counts": "",
        "tripwire_run_epoch": "1000",
    }
    notes: list[str] = []

    def _run_epoch(prev: dict[str, str]) -> dict[str, str]:
        posted: list[dict] = []
        pushes: list[str] = []
        escalated, counts, fired = _escalate_once(
            monkeypatch, prev_state=prev, posted=posted, pushes=pushes
        )
        assert fired is True
        notes.append(posted[0]["note"])
        return {
            "phase": "p3_upload",
            "gpu_idle_escalated_phases": ",".join(sorted(escalated)),
            "gpu_idle_escalation_counts": pp._serialize_escalation_counts(counts),
            "tripwire_run_epoch": prev.get("tripwire_run_epoch", "1000"),
        }

    # Run epoch 1 fires #1; a fresh epm:run-launched then clears the
    # run-scoped keys (the #1033 reset that re-fires the identical note).
    state = _run_epoch(state)
    state, _epoch = pp._tripwire_run_scope(state, run_age_sec=120.0, now_epoch=1_000_000)
    assert "gpu_idle_escalated_phases" not in state  # dedup re-armed
    assert state["gpu_idle_escalation_counts"] == "p3_upload:1"  # count SURVIVES

    # Run epoch 2 fires #2; a fresh instance incarnation (GCP attempt-id
    # scoping) clears the idle keys the same blacklist way.
    state = _run_epoch(state)
    state["idle_attempt_id"] = "att-old"
    state = bp._scope_idle_state_to_attempt(state, "att-new")
    assert "gpu_idle_escalated_phases" not in state
    assert state["gpu_idle_escalation_counts"] == "p3_upload:2"  # count SURVIVES

    # Run epoch 3: the THIRD fire switches KIND.
    _run_epoch(state)
    assert notes[0].startswith("[gpu-idle-escalation]")
    assert notes[1].startswith("[gpu-idle-escalation]")
    assert notes[2].startswith("[gpu-idle-width-reeval]")


def test_escalation_wiring_source_references_no_lifecycle_symbol() -> None:
    """Source-level never-stops pin (plan AC 4): ``_maybe_escalate_gpu_idle``
    CALLS no stop/terminate/kill/pod_lifecycle symbol on EITHER note branch —
    its only externals are ``post_event`` (marker) and ``_telegram_push``
    (push). The width-reeval note PROSE deliberately says 'terminate the wide
    pod' (an instruction to the human operator), so the assertion walks AST
    Call/Name/Attribute nodes, never string literals."""
    import ast
    import inspect
    import re
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(pp._maybe_escalate_gpu_idle)))
    called: set[str] = set()
    idents: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
        if isinstance(node, ast.Name):
            idents.add(node.id)
        elif isinstance(node, ast.Attribute):
            idents.add(node.attr)
    forbidden = {n for n in called if re.search(r"stop|terminate|kill", n, re.IGNORECASE)}
    assert not forbidden, f"lifecycle-shaped call in the escalation wiring: {forbidden}"
    assert not {i for i in idents if "pod_lifecycle" in i}
    assert {"post_event", "_telegram_push"} <= called


def test_escalation_count_key_not_in_idle_advisory_clear_set() -> None:
    """#1752 membership-exclusion pin (GCP mirror): ``gpu_idle_escalation_counts``
    is NOT in ``_IDLE_ADVISORY_STATE_KEYS`` — the attempt-id scoping wipes the
    three idle keys but KEEPS the cross-incarnation count."""
    assert "gpu_idle_escalation_counts" not in bp._IDLE_ADVISORY_STATE_KEYS
    prev = {
        "idle_attempt_id": "att-old",
        "gpu_idle_since_epoch": "1000",
        "gpu_idle_escalation_counts": "workload:2",
    }
    scoped = bp._scope_idle_state_to_attempt(prev, "att-new")
    assert "gpu_idle_since_epoch" not in scoped
    assert scoped["gpu_idle_escalation_counts"] == "workload:2"


# ── _phase_is_cpu_only on the ACTUAL GCP eps/phase vocabulary ─────────────────
#
# On a RUNNING GCP VM the current_phase threaded into backend_poll.main() is the
# COARSE eps/phase guest attribute, whose only mid-workload value is the literal
# "workload" (gcp.py). The fine dispatcher phases (p3_upload) appear on the
# RunPod lane; they are still asserted here because the deny-list is shared.


@pytest.mark.parametrize(
    "phase,expected",
    [
        ("workload", True),  # the GCP coarse mid-workload phase -> eligible
        ("done", True),  # no deny-list substring (gated out earlier by status!=running anyway)
        ("p3_upload", True),  # RunPod-lane fine phase, shared deny-list
        ("upload", True),
        ("setup", False),  # deny-list substring
        ("setup_failed", False),
        ("train", False),
        ("eval", False),
        ("unknown", False),  # the explicit ineligible sentinel
        ("", False),
    ],
)
def test_gcp_phase_deny_list_matches(phase: str, expected: bool) -> None:
    assert pp._phase_is_cpu_only(phase) is expected


def test_gpu_required_substrings_match_assertions() -> None:
    """The substrings the GCP test asserts denied are actually in the shared
    deny-list (guards against the deny-list drifting out from under this test)."""
    assert {"train", "eval", "setup"} <= set(pp.GPU_REQUIRED_PHASE_SUBSTRINGS)


# ── the NEVER-stops-the-VM invariant (static argv guard) ──────────────────────


class _IdlePollBackend:
    """A GcpBackend-shaped poll double for the main() integration test.

    Records every argv the injected runner sees (so the no-VM-stop guard can
    assert no stop/delete shape), returns a scripted RUNNING PollResult from
    poll(), and a scripted all-idle gpu_util from the probe. Carries a real
    _config so backend._config.primary_zone resolves.
    """

    def __init__(self, *, gpu_util: str, current_phase: str) -> None:
        from explore_persona_space.backends.gcp import default_gcp_config

        self._config = default_gcp_config()
        self._gpu_util = gpu_util
        self._current_phase = current_phase
        self.run_argvs: list[list[str]] = []

    def poll(self, handle: RunHandle) -> PollResult:
        return PollResult(
            status="running",
            current_phase=self._current_phase,
            new_milestone=False,
            last_log_mtime_sec_ago=10,
            pid_alive=True,
            log_tail_excerpt="",
        )

    def _gcp_gpu_util_probe(self, handle: RunHandle, zone: str) -> str:
        # Record a representative probe argv so the no-stop guard sees the real
        # SSH shape the production probe would emit.
        self.run_argvs.append(
            ["gcloud", "compute", "ssh", handle.pod_name, f"--zone={zone}", "nvidia-smi"]
        )
        return self._gpu_util


_FORBIDDEN_ARGV_SHAPES = (
    ("instances", "stop"),
    ("instances", "delete"),
)


def _argv_is_vm_stop(argv: list[str]) -> bool:
    joined = " ".join(argv)
    if any(all(tok in argv for tok in shape) for shape in _FORBIDDEN_ARGV_SHAPES):
        return True
    return "pod.py" in joined and (" stop" in joined or " terminate" in joined)


def test_gcp_no_vm_stop_in_codepath(tmp_path, monkeypatch, capsys) -> None:
    """At the escalation threshold the GCP GPU-idle codepath posts a marker +
    push but issues NO VM-stopping action: no `gcloud ... instances stop|delete`
    and no `pod.py ... stop|terminate` reaches the runner OR subprocess.run."""
    import subprocess

    subprocess_argvs: list[list[str]] = []
    real_run = subprocess.run

    def _recording_run(argv, *a, **kw):
        if isinstance(argv, (list, tuple)):
            subprocess_argvs.append(list(argv))
        return real_run(argv, *a, **kw)

    monkeypatch.setattr(subprocess, "run", _recording_run)
    monkeypatch.setattr(pp, "post_event", lambda *a, **kw: None)
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: True)

    sidecar = tmp_path / "issue-730-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)
    state_path = bp._gpu_idle_state_path(sidecar)
    now = int(time.time())
    _seed_idle_state(
        state_path, since_epoch=now - pp.GPU_IDLE_ESCALATION_MIN * 60, phase="workload"
    )

    backend = _IdlePollBackend(gpu_util="0,0,0,0,0,0,0,0", current_phase="workload")
    monkeypatch.setattr("scripts.backend_poll._resolve_backend", lambda name: backend)

    rc = bp.main(["--issue", "730", "--handle-file", str(sidecar)])
    assert rc == 0

    # No VM-stop argv in EITHER channel.
    for argv in backend.run_argvs + subprocess_argvs:
        assert not _argv_is_vm_stop(argv), f"forbidden VM-stop argv reached the codepath: {argv}"


# ── main() integration: the two serialized JSON fields ────────────────────────


def _last_json_line(capsys) -> dict:
    out = capsys.readouterr().out.strip()
    assert out, "backend_poll printed no stdout"
    return json.loads(out.splitlines()[-1])


def test_backend_poll_main_gcp_idle_integration(tmp_path, monkeypatch, capsys) -> None:
    """Driving main() on a GCP handle with a RUNNING poll + all-idle probe + a
    pre-seeded idle span past the escalation min emits both new JSON fields and
    drives the posted flags."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: True)

    sidecar = tmp_path / "issue-730-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)
    state_path = bp._gpu_idle_state_path(sidecar)
    now = int(time.time())
    _seed_idle_state(
        state_path, since_epoch=now - pp.GPU_IDLE_ESCALATION_MIN * 60, phase="workload"
    )

    backend = _IdlePollBackend(gpu_util="0,0,0,0,0,0,0,0", current_phase="workload")
    monkeypatch.setattr("scripts.backend_poll._resolve_backend", lambda name: backend)

    rc = bp.main(["--issue", "730", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert out["gcp_gpu_idle_advisory_posted"] is True
    assert out["gcp_gpu_idle_escalation_posted"] is True
    # The state file was updated with the escalated phase (idempotency surface).
    saved = bp._load_gpu_idle_state(state_path)
    assert "workload" in saved["gpu_idle_escalated_phases"]
    # #1752: the FIRST escalation of this phase lands count 1 in the sibling
    # state (the 3-tuple unpack + persist through _save_gpu_idle_state).
    assert saved["gpu_idle_escalation_counts"] == "workload:1"


def test_backend_poll_main_gcp_width_reeval_persists_count(tmp_path, monkeypatch, capsys) -> None:
    """#1752 GCP-mirror wiring: the idle block unpacks the wiring 3-tuple and
    persists ``gpu_idle_escalation_counts`` through ``_save_gpu_idle_state``.
    A seeded count of 2 (fires on PRIOR instance incarnations) with a
    re-armed dedup set + a threshold-aged span drives the THIRD escalation:
    the [gpu-idle-width-reeval] note posts and the sibling state file carries
    the incremented count."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: True)

    sidecar = tmp_path / "issue-730-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)
    state_path = bp._gpu_idle_state_path(sidecar)
    now = int(time.time())
    bp._save_gpu_idle_state(
        state_path,
        {
            "phase": "workload",
            "gpu_idle_since_epoch": str(now - pp.GPU_IDLE_ESCALATION_MIN * 60),
            "gpu_idle_advised_phases": "",
            "gpu_idle_escalated_phases": "",  # fresh incarnation: dedup re-armed
            "gpu_idle_escalation_counts": "workload:2",  # prior incarnations' fires
        },
    )

    backend = _IdlePollBackend(gpu_util="0,0,0,0,0,0,0,0", current_phase="workload")
    monkeypatch.setattr("scripts.backend_poll._resolve_backend", lambda name: backend)

    rc = bp.main(["--issue", "730", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["gcp_gpu_idle_escalation_posted"] is True
    assert any("[gpu-idle-width-reeval]" in (p.get("note") or "") for p in posted)
    saved = bp._load_gpu_idle_state(state_path)
    assert saved["gpu_idle_escalation_counts"] == "workload:3"  # persisted via the 3-tuple


def test_backend_poll_main_non_gcp_omits_idle_fields_defaulting_false(
    tmp_path, monkeypatch, capsys
) -> None:
    """A non-GCP (SLURM) tick routed through main() leaves both fields False —
    the GCP GPU-idle block is gated on handle.backend == 'gcp'.

    The handle backend is deliberately a SLURM cluster, NOT 'runpod': a RunPod
    handle would drive production main() into _maybe_escalate_runpod_wedge,
    which lazy-imports runpod_api.get_pod_by_name and hits the LIVE RunPod API
    (a DNS failure in any clean / offline CI run). The test's intent — non-GCP
    backends default both new fields to False — is backend-string-driven, so a
    SLURM handle exercises the same gate without any network dependency
    (_maybe_escalate_gcp_wedge / _maybe_escalate_runpod_wedge / the GCP idle
    block all early-return for a non-matching backend string)."""
    slurm_handle = RunHandle(
        backend="cluster",
        cluster="nibi",
        job_id="slurm-fake-1",
        pod_name="eps-issue-730",
        scratch_dir="/scratch/eps-issue-730",
        log_path="/scratch/logs/issue-730.log",
        extra={"issue": 730},
    )
    sidecar = tmp_path / "issue-730-handle.json"
    write_handle_sidecar(slurm_handle, sidecar)

    class _RunningSlurm:
        def poll(self, handle: RunHandle) -> PollResult:
            return PollResult(
                status="running",
                current_phase="train",
                new_milestone=False,
                last_log_mtime_sec_ago=5,
                pid_alive=True,
                log_tail_excerpt="",
            )

    monkeypatch.setattr("scripts.backend_poll._resolve_backend", lambda name: _RunningSlurm())

    rc = bp.main(["--issue", "730", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["gcp_gpu_idle_advisory_posted"] is False
    assert out["gcp_gpu_idle_escalation_posted"] is False


# ── #873 m-of-N GPU-width advisory on the GCP lane ────────────────────────────
#
# The decision core + RunPod-lane wiring (incl. the injectable-clock
# repost-after-relaunch behavior) are exhaustively pinned in
# tests/test_poll_gpu_width_advisory.py; here we pin the GCP-lane WIRING —
# the imported _maybe_post_gpu_width_advisory + the _tripwire_run_scope
# run-scope anchor driven off the seeded sibling state file (AC #6 mirror).

# Partial idle on an 8-GPU pod: idle {0,1,3,7}, active {2,4,5,6}.
_PARTIAL_IDLE_UTIL = "0,0,95,0,88,90,92,0"


def _seed_width_state(
    path: Path,
    *,
    since_epoch: int,
    phase: str,
    advised: str = "",
    run_epoch: int | None = None,
) -> None:
    payload = {
        "phase": phase,
        "gpu_idle_since_epoch": "0",
        "gpu_idle_advised_phases": "",
        "gpu_idle_escalated_phases": "",
        "gpu_width_since_epoch": str(since_epoch),
        "gpu_width_idle_set": "0,1,3,7",
        "gpu_width_advised_phases": advised,
    }
    if run_epoch is not None:
        payload["tripwire_run_epoch"] = str(run_epoch)
    bp._save_gpu_idle_state(path, payload)


def test_gcp_width_advisory_posts_after_threshold(tmp_path: Path, monkeypatch) -> None:
    """A STABLE strict subset of GPUs idle past GPU_WIDTH_ADVISORY_MIN in the
    seeded sibling state -> the imported width wiring posts a
    [gpu-width-advisory] marker (mirror of test_gcp_advisory_posts_after_threshold)."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    path = tmp_path / "issue-730-gpu-idle-state.json"
    now = 100_000
    _seed_width_state(path, since_epoch=now - pp.GPU_WIDTH_ADVISORY_MIN * 60, phase="workload")
    prev = bp._load_gpu_idle_state(path)
    _since, idle_set, advised, width_posted = pp._maybe_post_gpu_width_advisory(
        issue=730,
        pod="eps-issue-730",
        status="running",
        gpu_util=_PARTIAL_IDLE_UTIL,
        current_phase="workload",
        prev_state=prev,
        now_epoch=now,
    )
    assert width_posted is True
    assert idle_set == (0, 1, 3, 7)
    assert "workload" in advised
    assert any(p["key"] == "epm:progress" and p.get("gpu_width_advisory") for p in posted)
    assert any("[gpu-width-advisory]" in (p.get("note") or "") for p in posted)


def test_backend_poll_main_gcp_width_integration(tmp_path, monkeypatch, capsys) -> None:
    """Driving main() on a GCP handle with a RUNNING poll + a partial-idle
    probe + a pre-seeded width span past the threshold emits the new
    serialized ``gcp_gpu_width_advisory_posted`` field True and records the
    advised phase in the sibling state (mirror of
    test_backend_poll_main_gcp_idle_integration)."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    # Deterministic run-scope anchor: no epm:run-launched signal -> no reset.
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch: None)

    sidecar = tmp_path / "issue-730-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)
    state_path = bp._gpu_idle_state_path(sidecar)
    now = int(time.time())
    _seed_width_state(
        state_path,
        since_epoch=now - (pp.GPU_WIDTH_ADVISORY_MIN + 1) * 60,
        phase="workload",
    )

    backend = _IdlePollBackend(gpu_util=_PARTIAL_IDLE_UTIL, current_phase="workload")
    monkeypatch.setattr("scripts.backend_poll._resolve_backend", lambda name: backend)

    rc = bp.main(["--issue", "730", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert out["gcp_gpu_width_advisory_posted"] is True
    # Partial idle is DISJOINT from the all-idle tiers: neither idle flag fires.
    assert out["gcp_gpu_idle_advisory_posted"] is False
    assert out["gcp_gpu_idle_escalation_posted"] is False
    saved = bp._load_gpu_idle_state(state_path)
    assert "workload" in saved["gpu_width_advised_phases"]
    assert saved["gpu_width_idle_set"] == "0,1,3,7"
    assert any("[gpu-width-advisory]" in (p.get("note") or "") for p in posted)


def test_gcp_width_relaunch_resets_advised_phases(tmp_path, monkeypatch, capsys) -> None:
    """AC #6 mirrored in the GCP sibling state payload: a fresh
    epm:run-launched epoch (newer than the stored tripwire_run_epoch by
    >60s) CLEARS the stale width keys — the advised-phase de-dup and the
    span belong to the PREVIOUS run, so the fresh run's first tick restarts
    the span (no post) with a re-armed advised set. The repost-after-aging
    behavior of the SECOND run is pinned with an injectable clock in
    tests/test_poll_gpu_width_advisory.py::test_width_relaunch_resets_advised_phases."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    # The fresh run launched 120s ago -> current run epoch >> stored anchor.
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch: 120.0)

    sidecar = tmp_path / "issue-730-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)
    state_path = bp._gpu_idle_state_path(sidecar)
    t0 = int(time.time())
    _seed_width_state(
        state_path,
        since_epoch=t0 - 10 * 3600,  # a stale, way-past-threshold span
        phase="workload",
        advised="workload",  # the PREVIOUS run already advised this phase
        run_epoch=1000,  # the PREVIOUS run's launch epoch
    )

    backend = _IdlePollBackend(gpu_util=_PARTIAL_IDLE_UTIL, current_phase="workload")
    monkeypatch.setattr("scripts.backend_poll._resolve_backend", lambda name: backend)

    rc = bp.main(["--issue", "730", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    # The stale span/advised set was CLEARED, so this tick restarts, not posts.
    assert out["gcp_gpu_width_advisory_posted"] is False
    assert not any("[gpu-width-advisory]" in (p.get("note") or "") for p in posted)
    saved = bp._load_gpu_idle_state(state_path)
    assert saved["gpu_width_advised_phases"] == ""  # stale de-dup cleared (re-armed)
    assert int(saved["gpu_width_since_epoch"]) >= t0  # span restarted this run
    assert int(saved["tripwire_run_epoch"]) >= t0 - 121  # fresh anchor persisted


# ── #1033 per-instance idle clock (attempt-id scoping) ────────────────────────
#
# The idle-advisory span + per-phase dedup sets in the GCP sibling state file
# used to survive instance relaunches: #763 printed a "543 min" idle advisory
# (via the ADVISORY tier — the live #763 sidecar shows
# gpu_idle_advised_phases="startup,workload" with an EMPTY escalated set:
# single-GPU instance, so only the advisory fired, once per phase as the
# stale span rode along) on a ~17-min-old fresh eps-issue-763 VM whose phase
# name matched the stored one, so the per-phase reset never fired. The fix
# keys the idle state to handle.extra["attempt_id"] (fresh per NEW instance,
# label-stable on reconnect — #927) via _scope_idle_state_to_attempt, with
# _tripwire_run_scope's widened _RUN_SCOPED_STATE_KEYS clear as the
# belt-and-suspenders run-epoch reset.


def test_scope_idle_state_same_attempt_preserves() -> None:
    """A matching stored attempt id (a reconnect to the SAME instance) keeps
    the state verbatim — the span legitimately accumulates."""
    prev = {
        "idle_attempt_id": "att-1",
        "gpu_idle_since_epoch": "1000",
        "gpu_idle_advised_phases": "startup",
        "gpu_idle_escalated_phases": "",
        "phase": "startup",
    }
    assert bp._scope_idle_state_to_attempt(prev, "att-1") is prev


def test_scope_idle_state_missing_current_attempt_keeps_verbatim() -> None:
    """Fail-safe: an absent/empty CURRENT attempt id cannot decide instance
    identity -> state kept verbatim (pre-#1033 behavior)."""
    prev = {"idle_attempt_id": "att-1", "gpu_idle_since_epoch": "1000"}
    assert bp._scope_idle_state_to_attempt(prev, "") is prev
    assert bp._scope_idle_state_to_attempt(prev, None) is prev


def test_scope_idle_state_legacy_missing_stored_key_resets() -> None:
    """Migration direction pinned: a pre-#1033 state file (NO stored
    ``idle_attempt_id``) with a KNOWN current id RESETS — failing toward one
    delayed/duplicate advisory, never a stale counter."""
    prev = {
        "gpu_idle_since_epoch": "1000",
        "gpu_idle_advised_phases": "startup",
        "gpu_idle_escalated_phases": "startup",
        "phase": "startup",
    }
    scoped = bp._scope_idle_state_to_attempt(prev, "att-1")
    for key in bp._IDLE_ADVISORY_STATE_KEYS:
        assert key not in scoped
    assert scoped["phase"] == "startup"  # only the idle keys are cleared


def test_scope_idle_state_mismatch_clears_only_idle_keys() -> None:
    """An attempt-id MISMATCH (a genuinely NEW instance) clears exactly the
    three idle keys; every other key (phase, width, anchor) survives."""
    prev = {
        "idle_attempt_id": "att-old",
        "gpu_idle_since_epoch": "1000",
        "gpu_idle_advised_phases": "startup,workload",
        "gpu_idle_escalated_phases": "workload",
        "gpu_width_since_epoch": "5",
        "tripwire_run_epoch": "77",
        "phase": "workload",
    }
    scoped = bp._scope_idle_state_to_attempt(prev, "att-new")
    for key in bp._IDLE_ADVISORY_STATE_KEYS:
        assert key not in scoped
    assert scoped["gpu_width_since_epoch"] == "5"
    assert scoped["tripwire_run_epoch"] == "77"
    assert scoped["phase"] == "workload"


def _seed_attempt_idle_state(
    path: Path, *, since_epoch: int, phase: str, idle_attempt_id: str | None
) -> None:
    payload = {
        "phase": phase,
        "gpu_idle_since_epoch": str(since_epoch),
        "gpu_idle_advised_phases": "",
        "gpu_idle_escalated_phases": "",
    }
    if idle_attempt_id is not None:
        payload["idle_attempt_id"] = idle_attempt_id
    bp._save_gpu_idle_state(path, payload)


def _gcp_attempt_handle(attempt_id: str | None) -> RunHandle:
    extra = {"issue": 730, "zone": "us-central1-a"}
    if attempt_id is not None:
        extra["attempt_id"] = attempt_id
    return _gcp_handle(extra)


def _run_idle_main(tmp_path, monkeypatch, *, handle: RunHandle, seed_kwargs: dict):
    """Drive ``bp.main`` on a single-GPU all-idle GCP tick (the #763 shape:
    advisory-tier only — the escalation tier requires a multi-GPU pod, which
    is why the live #763 sidecar's escalated set is empty). Returns
    ``(json_line, posted, saved_state)``."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: True)
    # No epm:run-launched signal -> the run-epoch anchor never resets; the
    # attempt-id mechanism is isolated as the ONLY reset in play.
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch: None)

    sidecar = tmp_path / "issue-730-handle.json"
    write_handle_sidecar(handle, sidecar)
    state_path = bp._gpu_idle_state_path(sidecar)
    _seed_attempt_idle_state(state_path, **seed_kwargs)

    backend = _IdlePollBackend(gpu_util="0", current_phase="startup")
    monkeypatch.setattr("scripts.backend_poll._resolve_backend", lambda name: backend)

    rc = bp.main(["--issue", "730", "--handle-file", str(sidecar)])
    assert rc == 0
    return posted, bp._load_gpu_idle_state(state_path)


def test_gcp_idle_new_attempt_resets_idle_clock(tmp_path, monkeypatch) -> None:
    """The #763/#810 replay: a seeded 543-min span + ``idle_attempt_id``
    from the PREVIOUS instance, polled with a handle carrying a NEW
    attempt_id -> NO stale-minute advisory; the saved state carries the new
    ``idle_attempt_id`` and a now-anchored span (an advisory on the fresh
    instance can never report minutes exceeding its own poll history)."""
    t0 = int(time.time())
    posted, saved = _run_idle_main(
        tmp_path,
        monkeypatch,
        handle=_gcp_attempt_handle("att-new"),
        seed_kwargs={
            "since_epoch": t0 - 543 * 60,
            "phase": "startup",  # matches current_phase -> per-phase reset inert
            "idle_attempt_id": "att-old",
        },
    )
    assert not any("[gpu-idle-advisory]" in (p.get("note") or "") for p in posted)
    assert saved["idle_attempt_id"] == "att-new"
    assert int(saved["gpu_idle_since_epoch"]) >= t0  # span re-anchored this tick


def test_gcp_idle_same_attempt_preserves_span(tmp_path, monkeypatch) -> None:
    """Reconnect control: the SAME attempt_id (label-stable reconnect, #927)
    keeps the span, so the legitimate long-idle advisory still posts with
    the accumulated 543 minutes."""
    t0 = int(time.time())
    posted, saved = _run_idle_main(
        tmp_path,
        monkeypatch,
        handle=_gcp_attempt_handle("att-same"),
        seed_kwargs={
            "since_epoch": t0 - 543 * 60,
            "phase": "startup",
            "idle_attempt_id": "att-same",
        },
    )
    assert any(
        "[gpu-idle-advisory]" in (p.get("note") or "") and "543 min" in (p.get("note") or "")
        for p in posted
    )
    assert saved["idle_attempt_id"] == "att-same"


def test_gcp_idle_missing_attempt_id_no_reset(tmp_path, monkeypatch) -> None:
    """Fail-safe end-to-end: a handle WITHOUT ``attempt_id`` (older handle
    sidecars) keeps the state verbatim — pre-#1033 behavior, so the stale
    span still posts (the fail direction is a duplicate/late advisory only
    when identity is KNOWN to have changed, never a behavior change on
    degraded inputs)."""
    t0 = int(time.time())
    posted, saved = _run_idle_main(
        tmp_path,
        monkeypatch,
        handle=_gcp_attempt_handle(None),
        seed_kwargs={
            "since_epoch": t0 - 543 * 60,
            "phase": "startup",
            "idle_attempt_id": "att-old",
        },
    )
    assert any("[gpu-idle-advisory]" in (p.get("note") or "") for p in posted)
    # The stored id is PRESERVED (empty current id never clobbers it).
    assert saved["idle_attempt_id"] == "att-old"
