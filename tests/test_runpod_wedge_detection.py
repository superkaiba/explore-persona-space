"""TDD round-1 tests for issue #689 fix (b) — RunPod no-port host-wedge detect.

These describe the EXTERNAL behavior of the RunPod RUNNING-but-no-port wedge
machinery added to ``scripts/backend_poll.py`` (plan v3 §B.3 / §B.4): the
within-K override + past-K escalation (``_maybe_escalate_runpod_wedge``), the
narrow failover predicate (``_is_runpod_async_wedge_failure``), the per-cell
three-state inputs-on-HF gate + idempotent failover (``_failover_wedged_runpod``
/ ``_wedged_run_inputs_on_hf``), and the S2 malformed-clock fail-soft contract.

TESTS-FIRST: the round-1 commit ships these symbols as ``NotImplementedError``
stubs, so every case here FAILS until round-2 implementation lands. The phase /
K constants land as real values in round 1, so the constant-referencing
assertions resolve.

Mirrors the GCP wedge suite in ``test_backend_poll.py``. All RunPod live-API I/O
is mocked (``runpod_api.get_pod_by_name`` / ``terminate_pod``), the router
failover + ``list_repo_files`` + ``_issue_cells_for_handle`` are monkeypatched,
and the sidecar uses ``tmp_path``. No GPU, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import backend_poll as bp  # noqa: E402
import runpod_api  # noqa: E402

from explore_persona_space.backends.base import PollResult, RunHandle  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------


def _runpod_handle(pod_name: str = "pod-689") -> RunHandle:
    return RunHandle(
        backend="runpod",
        cluster=None,
        job_id="pod-fake-1",
        pod_name=pod_name,
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-689.log",
        extra={"issue": 689},
    )


def _gcp_handle() -> RunHandle:
    return RunHandle(
        backend="gcp",
        cluster=None,
        job_id="instance-fake-1",
        pod_name="eps-issue-689",
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-689.log",
        extra={"issue": 689},
    )


def _dead_poll() -> PollResult:
    """The realistic ``poll_once`` SSH-dead shape (poll_pipeline.py L2076-2077):
    an unreachable pod's FIRST poll is already ``status="dead"``, ``pid_alive``
    False. The within-K override must rewrite THIS to running (A1-residue)."""
    return PollResult(
        status="dead",
        current_phase="dead",
        new_milestone=True,
        last_log_mtime_sec_ago=10**9,
        pid_alive=False,
        log_tail_excerpt="",
    )


class _PodInfo:
    """A ``runpod_api.PodInfo``-shaped stub (the fields the wedge detector reads)."""

    def __init__(
        self,
        *,
        desired_status="RUNNING",
        ssh_host=None,
        ssh_port=None,
        pod_id="pod-id-1",
        name="pod-689",
    ):
        self.pod_id = pod_id
        self.name = name
        self.desired_status = desired_status
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.gpu_count = 1
        self.gpu_type_id = "H100"
        self.created_at = None


def _stub_live_api(monkeypatch, info):
    """Point ``backend_poll``'s ``get_pod_by_name`` lookup at a fixed PodInfo.

    The escalation does ``from runpod_api import get_pod_by_name`` at call time,
    so patching the ``runpod_api`` module attribute is the right seam.
    """
    monkeypatch.setattr(runpod_api, "get_pod_by_name", lambda name: info, raising=False)


def _empty_sidecar(tmp_path: Path) -> Path:
    """A minimal valid handle sidecar with an empty ``extra`` (no clock yet)."""
    p = tmp_path / "issue-689-handle.json"
    p.write_text(json.dumps({"backend": "runpod", "pod_name": "pod-689", "extra": {}}))
    return p


# ---------------------------------------------------------------------------
# 1. within-K dead input -> running override (A1-residue, load-bearing)
# ---------------------------------------------------------------------------


def test_within_k_dead_input_returns_running(tmp_path, monkeypatch):
    """First tick on a confirmed RUNNING+no-port pod with a DEAD input poll:
    the helper re-stamps the no-port clock and OVERRIDES the result to running
    (RUNPOD_WORKLOAD_OBSERVED_PHASE) so the orchestrator keeps polling. A
    bare-``return result`` implementation leaves status='dead' and FAILS this."""
    sidecar = _empty_sidecar(tmp_path)
    _stub_live_api(monkeypatch, _PodInfo(desired_status="RUNNING", ssh_host=None, ssh_port=None))

    out = bp._maybe_escalate_runpod_wedge(_runpod_handle(), _dead_poll(), sidecar, now=1_000_000.0)
    assert out.status == "running"
    assert out.current_phase == bp.RUNPOD_WORKLOAD_OBSERVED_PHASE


# ---------------------------------------------------------------------------
# 2. after K -> escalate to wedged
# ---------------------------------------------------------------------------


def test_after_k_dead_input_escalates_to_wedged(tmp_path, monkeypatch):
    """A second tick > K after the clock was stamped escalates to the terminal
    wedged phase (status='dead', RUNPOD_WORKLOAD_WEDGED_PHASE)."""
    sidecar = _empty_sidecar(tmp_path)
    _stub_live_api(monkeypatch, _PodInfo(desired_status="RUNNING", ssh_host=None, ssh_port=None))

    t0 = 1_000_000.0
    # Tick 1: stamps the clock, returns running (within-K).
    bp._maybe_escalate_runpod_wedge(_runpod_handle(), _dead_poll(), sidecar, now=t0)
    # Tick 2: now - first_seen > K -> wedged.
    out = bp._maybe_escalate_runpod_wedge(
        _runpod_handle(), _dead_poll(), sidecar, now=t0 + bp.RUNPOD_WEDGE_K_SEC + 60
    )
    assert out.status == "dead"
    assert out.current_phase == bp.RUNPOD_WORKLOAD_WEDGED_PHASE


# ---------------------------------------------------------------------------
# 3. within-K (clock already stamped, now - first_seen < K) -> stays running
# ---------------------------------------------------------------------------


def test_within_k_stays_running_false_positive_guard(tmp_path, monkeypatch):
    sidecar = _empty_sidecar(tmp_path)
    _stub_live_api(monkeypatch, _PodInfo(desired_status="RUNNING", ssh_host=None, ssh_port=None))

    t0 = 1_000_000.0
    bp._maybe_escalate_runpod_wedge(_runpod_handle(), _dead_poll(), sidecar, now=t0)
    # Still within the K floor -> must NOT escalate.
    out = bp._maybe_escalate_runpod_wedge(
        _runpod_handle(), _dead_poll(), sidecar, now=t0 + bp.RUNPOD_WEDGE_K_SEC - 60
    )
    assert out.status == "running"
    assert out.current_phase != bp.RUNPOD_WORKLOAD_WEDGED_PHASE


# ---------------------------------------------------------------------------
# 4. S2 — malformed clock never raises, never escalates
# ---------------------------------------------------------------------------


def test_runpod_noport_clock_malformed_sidecar_never_raises(tmp_path, monkeypatch):
    """A malformed JSON sidecar (or a non-numeric clock value) is read as
    'no clock yet': the helper raises NO exception, treats it as a first
    observation (re-stamps), and returns a non-escalated running result. Mirrors
    the GCP _read_phase_clock malformed-clock contract. No terminate."""
    terminated: list = []
    monkeypatch.setattr(
        runpod_api, "terminate_pod", lambda pid: terminated.append(pid), raising=False
    )
    _stub_live_api(monkeypatch, _PodInfo(desired_status="RUNNING", ssh_host=None, ssh_port=None))

    # (a) malformed JSON
    bad_json = tmp_path / "issue-689-handle.json"
    bad_json.write_text("{ this is not valid json :::")
    out_a = bp._maybe_escalate_runpod_wedge(
        _runpod_handle(), _dead_poll(), bad_json, now=1_000_000.0
    )
    assert out_a.status != "dead"

    # (b) non-numeric clock value
    bad_clock = tmp_path / "issue-689-handle-2.json"
    bad_clock.write_text(
        json.dumps(
            {
                "backend": "runpod",
                "pod_name": "pod-689",
                "extra": {"runpod_noport_first_seen_ts": "not-a-number"},
            }
        )
    )
    out_b = bp._maybe_escalate_runpod_wedge(
        _runpod_handle(), _dead_poll(), bad_clock, now=1_000_000.0
    )
    assert out_b.status != "dead"
    assert terminated == []  # never terminated on a malformed clock


# ---------------------------------------------------------------------------
# 5. port appears on tick 2 -> clock cleared, never escalated
# ---------------------------------------------------------------------------


def test_port_appears_on_tick_2_clears_clock(tmp_path, monkeypatch):
    sidecar = _empty_sidecar(tmp_path)
    t0 = 1_000_000.0

    # Tick 1: RUNNING + no port -> stamps clock, returns running.
    _stub_live_api(monkeypatch, _PodInfo(desired_status="RUNNING", ssh_host=None, ssh_port=None))
    bp._maybe_escalate_runpod_wedge(_runpod_handle(), _dead_poll(), sidecar, now=t0)

    # Tick 2 (> K): a public port appears (healthy) -> clear clock, never escalate.
    _stub_live_api(
        monkeypatch, _PodInfo(desired_status="RUNNING", ssh_host="1.2.3.4", ssh_port=12345)
    )
    out = bp._maybe_escalate_runpod_wedge(
        _runpod_handle(), _dead_poll(), sidecar, now=t0 + bp.RUNPOD_WEDGE_K_SEC + 60
    )
    assert out.current_phase != bp.RUNPOD_WORKLOAD_WEDGED_PHASE
    # The clock was cleared, so a later no-port observation re-stamps fresh
    # rather than immediately escalating off the stale t0.
    payload = json.loads(sidecar.read_text())
    assert "runpod_noport_first_seen_ts" not in (payload.get("extra") or {})


# ---------------------------------------------------------------------------
# 6. status leaves RUNNING (EXITED) -> never escalated
# ---------------------------------------------------------------------------


def test_status_leaves_running_never_escalated(tmp_path, monkeypatch):
    sidecar = _empty_sidecar(tmp_path)
    _stub_live_api(monkeypatch, _PodInfo(desired_status="EXITED", ssh_host=None, ssh_port=None))

    out = bp._maybe_escalate_runpod_wedge(_runpod_handle(), _dead_poll(), sidecar, now=1_000_000.0)
    assert out.current_phase != bp.RUNPOD_WORKLOAD_WEDGED_PHASE


# ---------------------------------------------------------------------------
# 7. non-RunPod handle -> unchanged
# ---------------------------------------------------------------------------


def test_non_runpod_handle_unchanged(tmp_path, monkeypatch):
    sidecar = _empty_sidecar(tmp_path)
    # The live-API lookup must never be consulted for a GCP handle.
    monkeypatch.setattr(
        runpod_api,
        "get_pod_by_name",
        lambda name: (_ for _ in ()).throw(AssertionError("get_pod_by_name called for GCP")),
        raising=False,
    )
    poll = _dead_poll()
    out = bp._maybe_escalate_runpod_wedge(_gcp_handle(), poll, sidecar, now=1_000_000.0)
    assert out is poll or (out.status == poll.status and out.current_phase == poll.current_phase)


# ---------------------------------------------------------------------------
# 8. predicate scope
# ---------------------------------------------------------------------------


def test_is_runpod_async_wedge_failure_predicate():
    wedged = PollResult(
        status="dead",
        current_phase=bp.RUNPOD_WORKLOAD_WEDGED_PHASE,
        new_milestone=True,
        last_log_mtime_sec_ago=10**9,
        pid_alive=False,
        log_tail_excerpt="",
    )
    # True for a RunPod handle whose poll surfaced the wedged phase.
    assert bp._is_runpod_async_wedge_failure(_runpod_handle(), wedged) is True
    # False for a GCP/SLURM handle with the SAME phase string.
    assert bp._is_runpod_async_wedge_failure(_gcp_handle(), wedged) is False


# ---------------------------------------------------------------------------
# 9/10. per-cell inputs-on-HF gate (M1)
# ---------------------------------------------------------------------------


class _FakeCell:
    def __init__(self, eval_key: str):
        self.eval_key = eval_key


def _hf_paths_for(eval_key: str, *, raw_names, store_names):
    import issue664_common as C

    raw = {f"{C.HF_RAW_COMPLETIONS_PREFIX}/{eval_key}/{n}" for n in raw_names}
    store = {f"{C.HF_STORE_PREFIX}/{eval_key}/{n}" for n in store_names}
    return raw | store


def test_per_cell_gate_partial_blocks(tmp_path, monkeypatch):
    """A selected cell with its raw prefix present but MISSING tensors.pt is a
    PARTIAL cell: the gate is not-ok, terminate is NOT called, the failover
    returns reason='runpod_wedge_inputs_unverified'."""
    import huggingface_hub
    import issue664_dispatch as D

    cell = _FakeCell("mk_partial_cell_seed42")
    monkeypatch.setattr(bp, "_issue_cells_for_handle", lambda issue, handle: [cell], raising=False)

    # PARTIAL: raw JSON present, store has ONLY meta.json (no tensors.pt).
    files = _hf_paths_for(
        cell.eval_key,
        raw_names={"completions__x__ctx.json"},
        store_names={"meta.json"},
    )
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo_id, **k: sorted(files))
    # The exact-set helpers must agree this cell is partial.
    monkeypatch.setattr(
        D, "_expected_eval_files", lambda c: {"completions__x__ctx.json"}, raising=False
    )

    terminated: list = []
    monkeypatch.setattr(
        runpod_api, "terminate_pod", lambda pid: terminated.append(pid), raising=False
    )
    relaunched: list = []
    monkeypatch.setattr(
        bp,
        "_relaunch_fresh_runpod",
        lambda **kw: relaunched.append(kw) or {"status": "running"},
        raising=False,
    )

    gate = bp._wedged_run_inputs_on_hf(689, _runpod_handle())
    assert gate.ok is False

    out = bp._failover_wedged_runpod(
        issue=689, handle=_runpod_handle(), result=_dead_poll(), sidecar=_empty_sidecar(tmp_path)
    )
    assert out.get("reason") == "runpod_wedge_inputs_unverified"
    assert terminated == []
    assert relaunched == []


def test_per_cell_gate_mid_sweep_allows(tmp_path, monkeypatch):
    """The headline M1 case: 32 selected cells, 10 COMPLETE on HF + 22 ABSENT
    (not-yet-run). The gate is OK (does NOT block on the 22 rerunnable cells);
    terminate is called once, then a fresh pod is re-provisioned."""
    import huggingface_hub
    import issue664_dispatch as D

    cells = [_FakeCell(f"mk_cell_{i:02d}_seed42") for i in range(32)]
    complete = cells[:10]
    monkeypatch.setattr(bp, "_issue_cells_for_handle", lambda issue, handle: cells, raising=False)

    files: set[str] = set()
    for c in complete:
        files |= _hf_paths_for(
            c.eval_key,
            raw_names={"completions__x__ctx.json"},
            store_names={"tensors.pt", "meta.json"},
        )
    # The remaining 22 cells contribute NO files (absent).
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo_id, **k: sorted(files))
    monkeypatch.setattr(
        D, "_expected_eval_files", lambda c: {"completions__x__ctx.json"}, raising=False
    )

    info = _PodInfo(desired_status="RUNNING", ssh_host=None, ssh_port=None, pod_id="pod-id-42")
    _stub_live_api(monkeypatch, info)
    terminated: list = []
    monkeypatch.setattr(
        runpod_api, "terminate_pod", lambda pid: terminated.append(pid), raising=False
    )
    relaunched: list = []
    monkeypatch.setattr(
        bp,
        "_relaunch_fresh_runpod",
        lambda **kw: relaunched.append(kw) or {"status": "running"},
        raising=False,
    )
    monkeypatch.setattr(bp, "_runpod_wedge_already_handled", lambda *a, **k: False, raising=False)

    gate = bp._wedged_run_inputs_on_hf(689, _runpod_handle())
    assert gate.ok is True
    assert len(gate.complete) == 10
    assert len(gate.absent) == 22
    assert gate.partial == []

    bp._failover_wedged_runpod(
        issue=689, handle=_runpod_handle(), result=_dead_poll(), sidecar=_empty_sidecar(tmp_path)
    )
    assert terminated == ["pod-id-42"]
    assert len(relaunched) == 1


# ---------------------------------------------------------------------------
# 11. failover clean-path idempotency
# ---------------------------------------------------------------------------


def test_failover_clean_path_idempotency(tmp_path, monkeypatch):
    """Clean path: terminate_pod is called once, then _relaunch_fresh_runpod. A
    second tick on the OLD handle short-circuits (idempotency) — no double
    terminate."""
    import huggingface_hub
    import issue664_dispatch as D

    cell = _FakeCell("mk_done_cell_seed42")
    monkeypatch.setattr(bp, "_issue_cells_for_handle", lambda issue, handle: [cell], raising=False)
    files = _hf_paths_for(
        cell.eval_key,
        raw_names={"completions__x__ctx.json"},
        store_names={"tensors.pt", "meta.json"},
    )
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo_id, **k: sorted(files))
    monkeypatch.setattr(
        D, "_expected_eval_files", lambda c: {"completions__x__ctx.json"}, raising=False
    )

    info = _PodInfo(desired_status="RUNNING", ssh_host=None, ssh_port=None, pod_id="pod-id-99")
    _stub_live_api(monkeypatch, info)
    terminated: list = []
    monkeypatch.setattr(
        runpod_api, "terminate_pod", lambda pid: terminated.append(pid), raising=False
    )
    relaunched: list = []
    monkeypatch.setattr(
        bp,
        "_relaunch_fresh_runpod",
        lambda **kw: relaunched.append(kw) or {"status": "running"},
        raising=False,
    )

    # First failover: not-yet-handled -> terminate + relaunch.
    handled = {"v": False}
    monkeypatch.setattr(
        bp, "_runpod_wedge_already_handled", lambda *a, **k: handled["v"], raising=False
    )
    sidecar = _empty_sidecar(tmp_path)
    bp._failover_wedged_runpod(
        issue=689, handle=_runpod_handle(), result=_dead_poll(), sidecar=sidecar
    )
    assert terminated == ["pod-id-99"]
    assert len(relaunched) == 1

    # Second tick on the OLD handle: already-handled short-circuits — no second
    # terminate, no second relaunch.
    handled["v"] = True
    bp._failover_wedged_runpod(
        issue=689, handle=_runpod_handle(), result=_dead_poll(), sidecar=sidecar
    )
    assert terminated == ["pod-id-99"]  # unchanged
    assert len(relaunched) == 1  # unchanged


# ---------------------------------------------------------------------------
# 12. DURABLE-lease idempotency (#689 blocker-2): the ~/.eps-routing/ lease, not
#     the .claude/cache sentinel, is the authoritative exactly-once guard.
# ---------------------------------------------------------------------------


class _FakeRunHandle:
    """A RunPod-shaped handle the router-launch mock returns (the readback seam)."""

    def __init__(self, pod_name="pod-689-fresh"):
        self.backend = "runpod"
        self.pod_name = pod_name
        self.job_id = "pod-fresh-1"
        self.extra = {"issue": 689}


def _runpod_handle_with_workload(pod_name: str = "pod-689") -> RunHandle:
    """A wedged RunPod handle whose ``extra`` carries ``workload_cmd`` so the REAL
    ``_relaunch_fresh_runpod`` can reconstruct a RunSpec (the router-launched
    canonical-handle shape). The mock-relaunch tests above pass a bare handle
    because they replace ``_relaunch_fresh_runpod`` entirely; the durable-lease /
    error-mapping tests exercise the real relaunch, so they need the spec fields."""
    return RunHandle(
        backend="runpod",
        cluster=None,
        job_id="pod-fake-1",
        pod_name=pod_name,
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-689.log",
        extra={"issue": 689, "intent": "lora-7b", "workload_cmd": "bash run.sh", "hydra_args": []},
    )


def _real_lease_store_in(tmp_path, monkeypatch):
    """Pin ``LeaseStore``'s default dir to ``tmp_path`` so the REAL durable-lease
    helpers (``_lease_records_runpod_wedge_failover`` / ``_stamp_runpod_wedge_failover``)
    read+write a throwaway ``~/.eps-routing/`` under the test tmp dir. Also seeds a
    base lease for issue 689 so the stamp has a lease to mutate."""
    from explore_persona_space.backends import router as R

    monkeypatch.setattr(R.Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    store = R.LeaseStore(lease_dir=tmp_path / ".eps-routing")
    store.write(
        R.Lease(issue=689, spec_hash="h", attempt_id="a", backend="runpod", job_id="pod-fresh-1")
    )
    return store


def test_failover_durable_lease_idempotency(tmp_path, monkeypatch):
    """#689 blocker-2: with the ``.claude/cache`` sentinel WRITE made to fail
    (EDQUOT / read-only-fs simulated), a SECOND poll tick on the OLD wedged handle
    still short-circuits — the DURABLE ``~/.eps-routing/`` lease, stamped after the
    first relaunch, provides the exactly-once dedup that the sentinel cannot.
    terminate_pod + the router launch are each called EXACTLY ONCE across two
    ticks."""
    import huggingface_hub
    import issue664_dispatch as D

    from explore_persona_space.backends import router as R

    _real_lease_store_in(tmp_path, monkeypatch)

    cell = _FakeCell("mk_done_cell_seed42")
    monkeypatch.setattr(bp, "_issue_cells_for_handle", lambda issue, handle: [cell], raising=False)
    files = _hf_paths_for(
        cell.eval_key,
        raw_names={"completions__x__ctx.json"},
        store_names={"tensors.pt", "meta.json"},
    )
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo_id, **k: sorted(files))
    monkeypatch.setattr(
        D, "_expected_eval_files", lambda c: {"completions__x__ctx.json"}, raising=False
    )

    info = _PodInfo(desired_status="RUNNING", ssh_host=None, ssh_port=None, pod_id="pod-id-77")
    _stub_live_api(monkeypatch, info)
    terminated: list = []
    monkeypatch.setattr(
        runpod_api, "terminate_pod", lambda pid: terminated.append(pid), raising=False
    )

    # The router launch mock — counts calls, returns a fresh RunPod handle.
    launches: list = []

    def _fake_failover(**kw):
        launches.append(kw)
        on_launched = kw.get("on_launched")
        h = _FakeRunHandle()
        if on_launched is not None:
            on_launched(h)
        return type("RR", (), {"handle": h, "extra": {}})()

    monkeypatch.setattr(
        R, "failover_to_runpod_after_async_workload_crash", _fake_failover, raising=False
    )
    # Sidecar write/readback seam: the readback returns a RunPod-backend handle so
    # _relaunch_fresh_runpod reaches the lease stamp + emits running.
    from explore_persona_space.backends import issue_dispatch as ID

    monkeypatch.setattr(ID, "write_handle_sidecar", lambda h, p: None, raising=False)
    monkeypatch.setattr(ID, "read_handle_sidecar", lambda p: _FakeRunHandle(), raising=False)

    # SIMULATE a persistent .claude/cache write failure: the wedge sentinel write
    # raises, so the sentinel can NEVER record the handled wedge. Only the durable
    # lease can dedup the second tick.
    monkeypatch.setattr(bp, "_write_runpod_wedge_sentinel", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(bp, "_read_failover_sentinel", lambda p: None, raising=False)

    sidecar = _empty_sidecar(tmp_path)
    # First tick: not-yet-handled -> terminate + relaunch + durable-lease stamp.
    out1 = bp._failover_wedged_runpod(
        issue=689, handle=_runpod_handle_with_workload(), result=_dead_poll(), sidecar=sidecar
    )
    assert out1["status"] == "running"
    assert terminated == ["pod-id-77"]
    assert len(launches) == 1

    # Second tick on the OLD handle: the sentinel is blind (write was a no-op), so
    # the DURABLE LEASE must short-circuit — NO second terminate, NO second launch.
    out2 = bp._failover_wedged_runpod(
        issue=689, handle=_runpod_handle_with_workload(), result=_dead_poll(), sidecar=sidecar
    )
    assert out2.get("reason") == "runpod_wedge_already_handled"
    assert terminated == ["pod-id-77"]  # unchanged — no double-terminate
    assert len(launches) == 1  # unchanged — no double-provision


# ---------------------------------------------------------------------------
# 13. relaunch error mapping (#689 blocker-3): no-capacity + sidecar-write
#     failure each return TERMINAL JSON, never an uncaught traceback.
# ---------------------------------------------------------------------------


def _failover_with_mocked_gate(monkeypatch):
    """Common setup for the relaunch-error tests: a single COMPLETE cell so the
    per-cell gate is OK (terminate allowed), live API stubbed, terminate_pod a
    no-op recorder. Returns the ``terminated`` list."""
    import huggingface_hub
    import issue664_dispatch as D

    cell = _FakeCell("mk_done_cell_seed42")
    monkeypatch.setattr(bp, "_issue_cells_for_handle", lambda issue, handle: [cell], raising=False)
    files = _hf_paths_for(
        cell.eval_key,
        raw_names={"completions__x__ctx.json"},
        store_names={"tensors.pt", "meta.json"},
    )
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo_id, **k: sorted(files))
    monkeypatch.setattr(
        D, "_expected_eval_files", lambda c: {"completions__x__ctx.json"}, raising=False
    )
    info = _PodInfo(desired_status="RUNNING", ssh_host=None, ssh_port=None, pod_id="pod-id-55")
    _stub_live_api(monkeypatch, info)
    terminated: list = []
    monkeypatch.setattr(
        runpod_api, "terminate_pod", lambda pid: terminated.append(pid), raising=False
    )
    monkeypatch.setattr(bp, "_runpod_wedge_already_handled", lambda *a, **k: False, raising=False)
    monkeypatch.setattr(bp, "_stamp_runpod_wedge_failover", lambda *a, **k: None, raising=False)
    return terminated


def test_failover_no_capacity_returns_terminal_json(tmp_path, monkeypatch):
    """#689 blocker-3: when RunPod has no capacity for the fresh re-provision, the
    router raises NoComputeAvailableError; the failover must RETURN a terminal infra
    JSON (reason=no_compute_available, the reason the watcher's capacity-retry pass
    re-drives), NEVER let the exception propagate out of main()."""
    from explore_persona_space.backends import router as R

    _failover_with_mocked_gate(monkeypatch)

    def _raise_no_compute(**kw):
        raise R.NoComputeAvailableError("no compute anywhere", attempts=[])

    monkeypatch.setattr(
        R, "failover_to_runpod_after_async_workload_crash", _raise_no_compute, raising=False
    )

    out = bp._failover_wedged_runpod(
        issue=689,
        handle=_runpod_handle_with_workload(),
        result=_dead_poll(),
        sidecar=_empty_sidecar(tmp_path),
    )
    assert out["status"] == "dead"
    assert out["reason"] == "no_compute_available"
    assert out["failure_class"] == "infra"


def test_failover_sidecar_write_failure_returns_terminal_json(tmp_path, monkeypatch):
    """#689 blocker-3: when the fresh pod launches but the .claude/cache sidecar
    write fails (EDQUOT), the failover must RETURN a terminal infra JSON
    (reason=sidecar_persistence_failed — NOT a capacity reason, so the watcher
    PARKS it for human inspection), NEVER an uncaught traceback. The durable lease
    is stamped to bound a re-fired tick."""
    from explore_persona_space.backends import issue_dispatch as ID
    from explore_persona_space.backends import router as R

    _failover_with_mocked_gate(monkeypatch)

    def _fake_failover(**kw):
        return type("RR", (), {"handle": _FakeRunHandle(), "extra": {}})()

    monkeypatch.setattr(
        R, "failover_to_runpod_after_async_workload_crash", _fake_failover, raising=False
    )

    def _raise_edquot(h, p):
        raise OSError("[Errno 122] Disk quota exceeded")

    monkeypatch.setattr(ID, "write_handle_sidecar", _raise_edquot, raising=False)
    monkeypatch.setattr(ID, "read_handle_sidecar", lambda p: _FakeRunHandle(), raising=False)

    stamped: list = []
    monkeypatch.setattr(
        bp, "_stamp_runpod_wedge_failover", lambda *a, **k: stamped.append(a), raising=False
    )

    out = bp._failover_wedged_runpod(
        issue=689,
        handle=_runpod_handle_with_workload(),
        result=_dead_poll(),
        sidecar=_empty_sidecar(tmp_path),
    )
    assert out["status"] == "dead"
    assert out["reason"] == "sidecar_persistence_failed"
    assert out["failure_class"] == "infra"
    # The durable lease was stamped so a re-fired tick short-circuits.
    assert len(stamped) == 1
