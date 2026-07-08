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
failover + ``bp._list_issue664_hub_files`` (the scoped Hub-listing seam, #988) +
``_issue_cells_for_handle`` are monkeypatched, and the sidecar uses ``tmp_path``.
No GPU, no network.
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


def test_pod_is_runpod_runtime_wedged_predicate():
    """#692: the RAW no-port wedge predicate extracted from
    ``_maybe_escalate_runpod_wedge`` (composition surface (b)), called by BOTH
    the poller and the autonomous_session_watch.py wedge backstop. It is the
    maturity-AGNOSTIC raw condition: RUNNING + no public port."""
    # RUNNING + no public port -> True (the raw wedge).
    assert bp._pod_is_runpod_runtime_wedged(_PodInfo(ssh_host=None, ssh_port=None)) is True
    # RUNNING + a public port present -> False (healthy).
    assert bp._pod_is_runpod_runtime_wedged(_PodInfo(ssh_host="1.2.3.4", ssh_port=22000)) is False
    # Non-RUNNING (EXITED/terminal) -> False (ordinary dead path).
    assert bp._pod_is_runpod_runtime_wedged(_PodInfo(desired_status="EXITED")) is False
    # None (pod gone) -> False.
    assert bp._pod_is_runpod_runtime_wedged(None) is False


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


def test_list_issue664_hub_files_scoped(monkeypatch):
    """The wedge gate's Hub listing is SCOPED per root prefix (#920/#988): one
    server-side scoped call per prefix (path_in_repo threaded), results
    unioned; an absent prefix (EntryNotFoundError -> a stubbed file_exists
    False) contributes zero files without failing the union. BOTH the tree
    walk AND the HfApi construction are faked — a patched EntryNotFoundError
    must NOT fall through to a real file_exists HEAD call (no network)."""
    from huggingface_hub.utils import EntryNotFoundError

    import explore_persona_space.orchestrate.hub as hub

    tree_calls: list[tuple] = []
    file_exists_calls: list[tuple] = []

    class _StubApi:
        def __init__(self, token=None):
            pass

        def file_exists(self, repo_id, path, *, repo_type=None, revision=None):
            file_exists_calls.append((repo_id, path))
            return False

        def list_repo_files(self, *a, **k):  # pragma: no cover - must never run
            raise AssertionError("bare full-repo listing must never be called (#920)")

    def _fake_complete(api, repo_id, *, repo_type="model", revision=None, path_in_repo=None):
        assert isinstance(api, _StubApi), "the stub api must reach the scoped walk"
        tree_calls.append((repo_id, repo_type, revision, path_in_repo))
        if path_in_repo == "absent_prefix":
            raise EntryNotFoundError("entry absent_prefix not found")
        return [f"{path_in_repo}/cell/file.json"]

    monkeypatch.setattr(hub, "list_repo_files_complete", _fake_complete)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _StubApi)

    files = bp._list_issue664_hub_files(
        "org/data-repo", ("raw_prefix", "absent_prefix", "store_prefix")
    )
    assert files == {"raw_prefix/cell/file.json", "store_prefix/cell/file.json"}
    assert [c[3] for c in tree_calls] == ["raw_prefix", "absent_prefix", "store_prefix"]
    assert all(c[1] == "dataset" and c[2] == "main" for c in tree_calls)
    # The absent prefix fell through to ONE STUBBED file_exists probe (False ->
    # zero files); the stub proves no real HEAD call could have fired.
    assert file_exists_calls == [("org/data-repo", "absent_prefix")]


def test_wedged_gate_passes_exactly_three_root_prefixes(monkeypatch):
    """Caller-contract pin (#988): _wedged_run_inputs_on_hf passes EXACTLY the
    three root prefixes the #664 classifier matches against. A dropped prefix
    can flip a half-uploaded PARTIAL cell to ABSENT — un-blocking the
    irreversible pod terminate — so the tuple is pinned by a spy."""
    import issue664_common as C
    import issue664_dispatch as D

    cell = _FakeCell("mk_spy_cell_seed42")
    monkeypatch.setattr(bp, "_issue_cells_for_handle", lambda issue, handle: [cell], raising=False)
    monkeypatch.setattr(
        D, "_expected_eval_files", lambda c: {"completions__x__ctx.json"}, raising=False
    )

    seen: list[tuple] = []

    def _spy(repo_id, prefixes):
        seen.append((repo_id, prefixes))
        return set()

    monkeypatch.setattr(bp, "_list_issue664_hub_files", _spy)

    gate = bp._wedged_run_inputs_on_hf(689, _runpod_handle())
    assert seen == [
        (
            C.HF_DATA_REPO,
            (C.HF_RAW_COMPLETIONS_PREFIX, C.HF_STORE_PREFIX, C.HF_MARKER_SLOT_PREFIX),
        )
    ]
    # Empty listing -> the one selected cell classifies ABSENT (rerunnable),
    # so the gate stays ok (no partial cell).
    assert gate.ok is True
    assert gate.absent == [cell.eval_key]


def test_per_cell_gate_partial_blocks(tmp_path, monkeypatch):
    """A selected cell with its raw prefix present but MISSING tensors.pt is a
    PARTIAL cell: the gate is not-ok, terminate is NOT called, the failover
    returns reason='runpod_wedge_inputs_unverified'."""
    import issue664_dispatch as D

    cell = _FakeCell("mk_partial_cell_seed42")
    monkeypatch.setattr(bp, "_issue_cells_for_handle", lambda issue, handle: [cell], raising=False)

    # PARTIAL: raw JSON present, store has ONLY meta.json (no tensors.pt).
    files = _hf_paths_for(
        cell.eval_key,
        raw_names={"completions__x__ctx.json"},
        store_names={"meta.json"},
    )
    monkeypatch.setattr(bp, "_list_issue664_hub_files", lambda repo_id, prefixes: set(files))
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
    monkeypatch.setattr(bp, "_list_issue664_hub_files", lambda repo_id, prefixes: set(files))
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
    import issue664_dispatch as D

    cell = _FakeCell("mk_done_cell_seed42")
    monkeypatch.setattr(bp, "_issue_cells_for_handle", lambda issue, handle: [cell], raising=False)
    files = _hf_paths_for(
        cell.eval_key,
        raw_names={"completions__x__ctx.json"},
        store_names={"tensors.pt", "meta.json"},
    )
    monkeypatch.setattr(bp, "_list_issue664_hub_files", lambda repo_id, prefixes: set(files))
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
    monkeypatch.setattr(bp, "_list_issue664_hub_files", lambda repo_id, prefixes: set(files))
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
    import issue664_dispatch as D

    cell = _FakeCell("mk_done_cell_seed42")
    monkeypatch.setattr(bp, "_issue_cells_for_handle", lambda issue, handle: [cell], raising=False)
    files = _hf_paths_for(
        cell.eval_key,
        raw_names={"completions__x__ctx.json"},
        store_names={"tensors.pt", "meta.json"},
    )
    monkeypatch.setattr(bp, "_list_issue664_hub_files", lambda repo_id, prefixes: set(files))
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


# ---------------------------------------------------------------------------
# 14. PRODUCTION-shaped handle (#689 round-3 blocker): the wedge failover must
#     reconstruct a RunSpec from the EXACT handle `RunPodBackend.launch()`
#     produces in production — NOT the synthetic `_runpod_handle_with_workload`
#     fixture that hand-adds fields production once omitted. These tests build
#     the handle by DRIVING the real launch() (subprocess mocked) so the test
#     goes RED the moment launch() stops persisting workload_cmd/hydra_args.
# ---------------------------------------------------------------------------


def _production_runpod_handle(
    monkeypatch,
    *,
    issue: int = 689,
    intent: str = "lora-7b",
    workload_cmd: str = "bash scripts/issue664_dispatch.sh --foo",
    gpus=None,
    time_budget_hours=None,
) -> RunHandle:
    """Build a wedged RunPod handle by invoking the REAL ``RunPodBackend.launch()``.

    Mocks ONLY ``subprocess.run`` (no pod is actually provisioned), so the
    returned handle's ``extra`` dict is byte-for-byte what production persists —
    this is the production-shaped handle the round-3 blocker requires, NOT the
    hand-built ``_runpod_handle_with_workload``. The reconstruction path
    (``_runspec_from_runpod_handle``) is exercised against THIS shape via the
    sidecar round-trip in the failover.
    """
    from explore_persona_space.backends import runpod as RP
    from explore_persona_space.backends.base import RunSpec

    # ``subprocess.run`` is the GLOBAL module singleton (env.py's git probe + the
    # artifact-declaration helpers use it too), so a blanket lambda would break
    # them. No-op ONLY the ``pod_lifecycle.py provision`` call; delegate every
    # other subprocess to the real implementation.
    _real_run = RP.subprocess.run

    def _selective_run(cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and any("pod_lifecycle.py" in str(c) for c in cmd):
            return None  # the provision call — no real pod
        return _real_run(cmd, *a, **k)

    monkeypatch.setattr(RP.subprocess, "run", _selective_run, raising=False)
    spec = RunSpec(
        issue=issue,
        intent=intent,
        backend="runpod",
        gpus=gpus,
        time_budget_hours=time_budget_hours,
        workload_cmd=workload_cmd,
    )
    handle = RP.RunPodBackend().launch(spec)
    # The launch path names the pod ``pod-<issue>`` and an empty job_id; the
    # wedge tests key the live-API stub on the name, so re-stamp pod_name to the
    # canonical ``pod-689`` the _PodInfo stub returns.
    assert handle.pod_name == f"pod-{issue}", handle.pod_name
    return handle


def test_production_launch_handle_persists_relaunch_spec_fields(monkeypatch):
    """#689 round-3 (Part 1): the REAL ``RunPodBackend.launch()`` output carries
    the relaunch-critical RunSpec fields on ``extra`` — ``workload_cmd`` (or
    ``hydra_args``), ``gpus``, ``time_budget_hours`` — so the wedge failover can
    reconstruct a RunSpec. This is the field-presence guard that, if it ever
    regresses, RED-flags the missing-spec orphan-the-run bug at unit time."""
    handle = _production_runpod_handle(
        monkeypatch, workload_cmd="bash scripts/issue664_dispatch.sh --foo", gpus=2
    )
    extra = handle.extra
    assert extra["workload_cmd"] == "bash scripts/issue664_dispatch.sh --foo"
    assert extra["hydra_args"] == []  # custom-workload run -> empty hydra args
    assert extra["gpus"] == 2
    assert "time_budget_hours" in extra  # present even when None


def test_production_handle_reconstructs_runspec(monkeypatch):
    """#689 round-3 (Part 1+2): a RunSpec reconstructs cleanly from the production
    handle's ``extra`` AFTER a serialize/deserialize sidecar round-trip — the
    hydra_args list (JSON-encoded from the tuple) re-tuples and the launch
    fields thread through. This is the exact reconstruction the failover does."""
    from explore_persona_space.backends.issue_dispatch import (
        deserialize_handle,
        serialize_handle,
    )

    handle = _production_runpod_handle(
        monkeypatch, workload_cmd="bash scripts/issue664_dispatch.sh --foo", gpus=2
    )
    # Round-trip exactly as the sidecar bridge does.
    roundtripped = deserialize_handle(serialize_handle(handle))
    spec = bp._runspec_from_runpod_handle(roundtripped, 689)
    assert spec.issue == 689
    assert spec.intent == "lora-7b"
    assert spec.backend == "runpod"
    assert spec.workload_cmd == "bash scripts/issue664_dispatch.sh --foo"
    assert spec.hydra_args == ()
    assert spec.gpus == 2


def test_failover_clean_path_production_handle(tmp_path, monkeypatch):
    """#689 round-3 (Part 3): drive ``_failover_wedged_runpod`` with the EXACT
    handle the production launch produces (NOT ``_runpod_handle_with_workload``).
    With the per-cell HF gate OK + the router launch mocked, the failover must
    (a) reconstruct the RunSpec, (b) terminate the wedged pod EXACTLY once, and
    (c) return ``status='running'`` / the fresh-pod phase. A production handle
    that lacked the spec fields would raise in reconstruction AFTER the
    irreversible terminate and orphan the run — this test RED-flags that."""
    from explore_persona_space.backends import router as R

    handle = _production_runpod_handle(monkeypatch)
    terminated = _failover_with_mocked_gate(monkeypatch)

    launches: list = []

    def _fake_failover(**kw):
        launches.append(kw["spec"])
        on_launched = kw.get("on_launched")
        h = _FakeRunHandle()
        if on_launched is not None:
            on_launched(h)
        return type("RR", (), {"handle": h, "extra": {}})()

    monkeypatch.setattr(
        R, "failover_to_runpod_after_async_workload_crash", _fake_failover, raising=False
    )
    from explore_persona_space.backends import issue_dispatch as ID

    monkeypatch.setattr(ID, "write_handle_sidecar", lambda h, p: None, raising=False)
    monkeypatch.setattr(ID, "read_handle_sidecar", lambda p: _FakeRunHandle(), raising=False)

    out = bp._failover_wedged_runpod(
        issue=689, handle=handle, result=_dead_poll(), sidecar=_empty_sidecar(tmp_path)
    )
    # (a) the router launch fired with a reconstructed RunSpec carrying the spec fields.
    assert len(launches) == 1
    relaunched_spec = launches[0]
    assert relaunched_spec.workload_cmd == handle.extra["workload_cmd"]
    assert relaunched_spec.intent == handle.extra["intent"]
    # (b) the wedged pod was terminated exactly once.
    assert terminated == ["pod-id-55"]
    # (c) the failover emitted a running poll for the fresh pod.
    assert out["status"] == "running"
    assert out["current_phase"] == "runpod_noport_wedge_failover_fresh_pod"


def test_failover_legacy_handle_missing_spec_returns_terminal_json(tmp_path, monkeypatch):
    """#689 round-3 (Part 2): a LEGACY sidecar handle built before launch() began
    persisting workload_cmd/hydra_args (carries NEITHER) must NOT crash the poller
    after the irreversible terminate. The failover maps the unreconstructable spec
    to an OBSERVABLE terminal infra JSON (reason='runpod_wedge_relaunch_spec_missing'),
    never letting ValueError propagate to reason='runpod_wedge_failover_error'."""
    legacy_handle = RunHandle(
        backend="runpod",
        cluster=None,
        job_id="pod-legacy-1",
        pod_name="pod-689",
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-689.log",
        # The pre-#689 production shape: intent + issue, but NO workload_cmd / hydra_args.
        extra={"issue": 689, "intent": "lora-7b"},
    )
    terminated = _failover_with_mocked_gate(monkeypatch)

    out = bp._failover_wedged_runpod(
        issue=689, handle=legacy_handle, result=_dead_poll(), sidecar=_empty_sidecar(tmp_path)
    )
    # The wedged pod WAS terminated (billing stopped), then the spec was found
    # unreconstructable — so an observable terminal JSON, not a crash.
    assert terminated == ["pod-id-55"]
    assert out["status"] == "dead"
    assert out["reason"] == "runpod_wedge_relaunch_spec_missing"
    assert out["failure_class"] == "infra"


# ===========================================================================
# #775 — RunPod CUDA-IMA repeat host-wedge failover
# ===========================================================================
#
# The signature-keyed repeat detection (_maybe_escalate_runpod_cuda_ima), the
# narrow failover predicate (_is_runpod_cuda_ima_failure), the signature record
# family (_read/_write/_clear_runpod_cuda_ima_record + the cross-pod fallback),
# and the bounded-once fresh-host failover (_failover_cuda_ima_runpod) with the
# _terminal_code_json exhaustion. Mirrors the Part C structure above; all RunPod
# live-API I/O is mocked, the lease uses a tmp dir, no GPU, no network.

# A realistic ~30-line vLLM CUDA-IMA traceback whose SIGNATURE line sits well
# beyond the last 5 lines (the B2 truncation the design must survive). The final
# 5 lines are a subprocess-returncode tail that does NOT carry the signature.
_CUDA_IMA_WIDE_TRACEBACK = "\n".join(
    [f"[rank0] step {i}: forward ok" for i in range(20)]
    + [
        "ERROR torch.AcceleratorError: CUDA error: an illegal memory access was encountered",
        "  Compile with TORCH_USE_CUDA_DSA to enable device-side assertions.",
        "vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue",
        "(EngineCore_DP0 pid=4242) Engine core proc EngineCore_DP0 died unexpectedly",
    ]
    # 6 trailing non-signature lines so the LAST 5 carry NO signature (B2).
    + [
        "INFO shutting down client",
        "subprocess returncode: 1",
        "Traceback (most recent call last):",
        '  File "/usr/lib/python3.11/runpy.py", line 198, in _run_module_as_main',
        "SystemExit: 1",
        "+ echo done",
    ]
)


def _cuda_ima_dead_poll(signature: str = _CUDA_IMA_WIDE_TRACEBACK) -> PollResult:
    """A dead PollResult carrying the WIDE CUDA-IMA crash surface on
    ``crash_signature`` (what RunPodBackend.poll threads through from poll_once)."""
    return PollResult(
        status="dead",
        current_phase="dead",
        new_milestone=True,
        last_log_mtime_sec_ago=10**9,
        pid_alive=False,
        log_tail_excerpt="\n".join(_CUDA_IMA_WIDE_TRACEBACK.splitlines()[-5:]),
        crash_signature=signature,
    )


# ---------------------------------------------------------------------------
# A. predicate scope
# ---------------------------------------------------------------------------


def test_is_runpod_cuda_ima_failure_predicate():
    wedged = PollResult(
        status="dead",
        current_phase=bp.RUNPOD_CUDA_IMA_WEDGED_PHASE,
        new_milestone=True,
        last_log_mtime_sec_ago=10**9,
        pid_alive=False,
        log_tail_excerpt="",
    )
    # True for a RunPod handle whose poll surfaced the CUDA-IMA wedged phase.
    assert bp._is_runpod_cuda_ima_failure(_runpod_handle(), wedged) is True
    # False for a GCP handle with the SAME phase string.
    assert bp._is_runpod_cuda_ima_failure(_gcp_handle(), wedged) is False
    # False for a RunPod handle at the NO-PORT wedged phase (distinct predicate).
    noport = PollResult(
        status="dead",
        current_phase=bp.RUNPOD_WORKLOAD_WEDGED_PHASE,
        new_milestone=True,
        last_log_mtime_sec_ago=10**9,
        pid_alive=False,
        log_tail_excerpt="",
    )
    assert bp._is_runpod_cuda_ima_failure(_runpod_handle(), noport) is False


# ---------------------------------------------------------------------------
# B. signature regex + the B2 wide-surface extraction (the bug NOT assumed away)
# ---------------------------------------------------------------------------


def test_cuda_ima_signature_family_matches_and_rejects():
    assert bp._crash_signature_is_cuda_ima(
        "torch.AcceleratorError: CUDA error: an illegal memory access was encountered"
    )
    assert bp._crash_signature_is_cuda_ima("EngineDeadError: engine core died")
    assert bp._crash_signature_is_cuda_ima("Engine core proc EngineCore_DP0 died unexpectedly")
    # Non-CUDA-IMA crashes never match.
    assert not bp._crash_signature_is_cuda_ima("AssertionError: shape mismatch")
    assert not bp._crash_signature_is_cuda_ima("ZeroDivisionError: division by zero")
    assert not bp._crash_signature_is_cuda_ima("")
    assert not bp._crash_signature_is_cuda_ima(None)


def test_cuda_ima_signature_extracted_beyond_5_line_tail():
    """B2 — the predicate must read the WIDE surface, NOT the 5-line excerpt.

    The signature line sits >5 lines from the END of the wide traceback, so the
    5-line ``log_tail_excerpt`` does NOT carry it but the wide ``crash_signature``
    does. Binds to the REAL poll_once slice helper
    (``poll_pipeline._tail_excerpt_and_crash_signature`` — the exact function
    poll_once calls) so it goes RED if the extraction ever reverts to the 5-line
    excerpt."""
    import poll_pipeline as pp

    # The realistic ~30-line traceback as the probe's WIDE main-log tail.
    probe = {"log_tail": _CUDA_IMA_WIDE_TRACEBACK, "cell_log_tail": ""}
    excerpt, crash_signature = pp._tail_excerpt_and_crash_signature(
        probe, status="dead", mtime_epoch=100, cell_mtime_epoch=0
    )
    # crash_signature is the WIDE surface and carries the CUDA-IMA family ...
    assert bp._crash_signature_is_cuda_ima(crash_signature) is True
    # ... but the 5-line excerpt (what a naive v1 matched) would have MISSED it.
    assert bp._crash_signature_is_cuda_ima(excerpt) is False
    # And the excerpt IS the last 5 lines of the wide tail (unchanged behavior).
    assert excerpt == "\n".join(_CUDA_IMA_WIDE_TRACEBACK.splitlines()[-5:])

    # A non-dead (running) poll populates NO crash_signature.
    _, sig_running = pp._tail_excerpt_and_crash_signature(
        probe, status="running", mtime_epoch=100, cell_mtime_epoch=0
    )
    assert sig_running is None

    # The cell-tail-fresher branch reads the CELL log as the wide surface.
    probe_cell = {"log_tail": "stale dispatcher", "cell_log_tail": _CUDA_IMA_WIDE_TRACEBACK}
    _, sig_cell = pp._tail_excerpt_and_crash_signature(
        probe_cell, status="dead", mtime_epoch=100, cell_mtime_epoch=200
    )
    assert bp._crash_signature_is_cuda_ima(sig_cell) is True


# ---------------------------------------------------------------------------
# C. escalation: first crash records, second escalates, our-frame excludes
# ---------------------------------------------------------------------------


def test_first_cuda_ima_crash_records_and_does_not_escalate(tmp_path):
    """The FIRST CUDA-IMA dead poll writes the record and returns the result
    UNCHANGED (the ordinary dead path -> in-place same-pod respawn)."""
    sidecar = _empty_sidecar(tmp_path)
    out = bp._maybe_escalate_runpod_cuda_ima(
        _runpod_handle(), _cuda_ima_dead_poll(), sidecar, issue=689, now=1_000_000.0
    )
    assert out.current_phase != bp.RUNPOD_CUDA_IMA_WEDGED_PHASE
    payload = json.loads(sidecar.read_text())
    assert "runpod_cuda_ima_last_seen" in (payload.get("extra") or {})


def test_second_same_signature_cuda_ima_escalates_to_wedged(tmp_path):
    """A SECOND CUDA-IMA dead poll with a prior record this run -> rewrite to the
    terminal wedged phase (status=dead, RUNPOD_CUDA_IMA_WEDGED_PHASE)."""
    sidecar = _empty_sidecar(tmp_path)
    # Tick 1: records, does not escalate.
    bp._maybe_escalate_runpod_cuda_ima(
        _runpod_handle(), _cuda_ima_dead_poll(), sidecar, issue=689, now=1_000_000.0
    )
    # Tick 2: prior record present -> escalate.
    out = bp._maybe_escalate_runpod_cuda_ima(
        _runpod_handle(), _cuda_ima_dead_poll(), sidecar, issue=689, now=1_000_100.0
    )
    assert out.status == "dead"
    assert out.current_phase == bp.RUNPOD_CUDA_IMA_WEDGED_PHASE


def test_non_cuda_ima_dead_poll_never_escalates(tmp_path):
    """A dead poll whose crash_signature is a plain AssertionError never records
    or escalates — and clears any stale record."""
    sidecar = _empty_sidecar(tmp_path)
    poll = PollResult(
        status="dead",
        current_phase="dead",
        new_milestone=True,
        last_log_mtime_sec_ago=10**9,
        pid_alive=False,
        log_tail_excerpt="",
        crash_signature="AssertionError: shapes mismatch at layer 3",
    )
    out = bp._maybe_escalate_runpod_cuda_ima(
        _runpod_handle(), poll, sidecar, issue=689, now=1_000_000.0
    )
    assert out.current_phase != bp.RUNPOD_CUDA_IMA_WEDGED_PHASE
    payload = json.loads(sidecar.read_text())
    assert "runpod_cuda_ima_last_seen" not in (payload.get("extra") or {})


def test_cuda_ima_record_cleared_on_running_poll(tmp_path):
    """A running (non-dead) poll clears the record so a single transient CUDA-IMA
    the respawn recovered from does NOT accumulate against a later one."""
    sidecar = _empty_sidecar(tmp_path)
    # Record a first CUDA-IMA crash.
    bp._maybe_escalate_runpod_cuda_ima(
        _runpod_handle(), _cuda_ima_dead_poll(), sidecar, issue=689, now=1_000_000.0
    )
    # An intervening HEALTHY poll -> the record is cleared.
    running = PollResult(
        status="running",
        current_phase="workload",
        new_milestone=False,
        last_log_mtime_sec_ago=5,
        pid_alive=True,
        log_tail_excerpt="step 999 ok",
    )
    bp._maybe_escalate_runpod_cuda_ima(
        _runpod_handle(), running, sidecar, issue=689, now=1_000_050.0
    )
    payload = json.loads(sidecar.read_text())
    assert "runpod_cuda_ima_last_seen" not in (payload.get("extra") or {})
    # A LATER CUDA-IMA crash is then the FIRST again (records, no escalation).
    out = bp._maybe_escalate_runpod_cuda_ima(
        _runpod_handle(), _cuda_ima_dead_poll(), sidecar, issue=689, now=1_000_100.0
    )
    assert out.current_phase != bp.RUNPOD_CUDA_IMA_WEDGED_PHASE


def test_cuda_ima_with_our_code_frame_does_not_escalate(tmp_path):
    """M3 — a SECOND CUDA-IMA dead poll whose wide surface ALSO carries an OUR-code
    traceback frame is a deterministic code bug, NOT a host wedge: the exclusion
    fires and it does NOT escalate (falls through to dead -> code)."""
    sidecar = _empty_sidecar(tmp_path)
    framed = _CUDA_IMA_WIDE_TRACEBACK + (
        '\n  File "/workspace/eps-issue-689/scripts/issue664_dispatch.py", line 42, in run\n'
        "    out = model(x)\n"
    )
    poll = _cuda_ima_dead_poll(signature=framed)
    # Tick 1: a non-framed first crash records (so a prior record exists).
    bp._maybe_escalate_runpod_cuda_ima(
        _runpod_handle(), _cuda_ima_dead_poll(), sidecar, issue=689, now=1_000_000.0
    )
    # Tick 2: the framed repeat -> M3 exclusion -> NO escalation.
    out = bp._maybe_escalate_runpod_cuda_ima(
        _runpod_handle(), poll, sidecar, issue=689, now=1_000_100.0
    )
    assert out.current_phase != bp.RUNPOD_CUDA_IMA_WEDGED_PHASE


def test_cuda_ima_record_malformed_sidecar_never_raises(tmp_path):
    """Fail-soft read: a malformed JSON sidecar reads as 'no record' and the
    escalation raises NO exception (mirrors the no-port clock S2 contract)."""
    bad = tmp_path / "issue-689-handle.json"
    bad.write_text("{ this is not valid json :::")
    out = bp._maybe_escalate_runpod_cuda_ima(
        _runpod_handle(), _cuda_ima_dead_poll(), bad, issue=689, now=1_000_000.0
    )
    # No prior record readable -> treated as first crash -> no escalation, no raise.
    assert out.current_phase != bp.RUNPOD_CUDA_IMA_WEDGED_PHASE


def test_cuda_ima_non_runpod_handle_unchanged(tmp_path):
    sidecar = _empty_sidecar(tmp_path)
    poll = _cuda_ima_dead_poll()
    out = bp._maybe_escalate_runpod_cuda_ima(
        _gcp_handle(), poll, sidecar, issue=689, now=1_000_000.0
    )
    assert out is poll


# ---------------------------------------------------------------------------
# D. cross-pod fallback (B1): sidecar record absent, prior epm:failure marker present
# ---------------------------------------------------------------------------


def test_cuda_ima_record_cross_pod_fallback_reads_prior_marker(tmp_path, monkeypatch):
    """B1 — when the sidecar record is ABSENT (a wipe between pods), the prior
    ``epm:failure`` marker carrying a CUDA-IMA signature still counts as a prior
    record, so the current crash escalates as a repeat."""
    sidecar = _empty_sidecar(tmp_path)  # NO runpod_cuda_ima_last_seen key
    # The prior epm:failure marker carried a CUDA-IMA signature.
    monkeypatch.setattr(
        bp,
        "_prior_failure_marker_is_cuda_ima",
        lambda issue: True,
        raising=False,
    )
    # _read_runpod_cuda_ima_record must surface a synthetic record from the marker.
    rec = bp._read_runpod_cuda_ima_record(sidecar, issue=689)
    assert rec is not None and rec.get("source") == "prior_failure_marker"
    # And the escalation treats the current CUDA-IMA crash as a repeat -> wedged.
    out = bp._maybe_escalate_runpod_cuda_ima(
        _runpod_handle(), _cuda_ima_dead_poll(), sidecar, issue=689, now=1_000_000.0
    )
    assert out.current_phase == bp.RUNPOD_CUDA_IMA_WEDGED_PHASE


def test_cuda_ima_record_no_fallback_when_prior_marker_absent(tmp_path, monkeypatch):
    """The fallback yields None (first crash) when no prior CUDA-IMA marker exists."""
    sidecar = _empty_sidecar(tmp_path)
    monkeypatch.setattr(bp, "_prior_failure_marker_is_cuda_ima", lambda issue: False, raising=False)
    assert bp._read_runpod_cuda_ima_record(sidecar, issue=689) is None


# ---------------------------------------------------------------------------
# E. failover: first-pivot fires, bounded-once, idempotency
# ---------------------------------------------------------------------------


def _cuda_ima_failover_gate_ok(monkeypatch):
    """Common setup: a single COMPLETE cell so the inputs gate is OK, live API
    stubbed, terminate_pod a recorder. Returns the ``terminated`` list."""
    import issue664_dispatch as D

    cell = _FakeCell("mk_done_cell_seed42")
    monkeypatch.setattr(bp, "_issue_cells_for_handle", lambda issue, handle: [cell], raising=False)
    files = _hf_paths_for(
        cell.eval_key,
        raw_names={"completions__x__ctx.json"},
        store_names={"tensors.pt", "meta.json"},
    )
    monkeypatch.setattr(bp, "_list_issue664_hub_files", lambda repo_id, prefixes: set(files))
    monkeypatch.setattr(
        D, "_expected_eval_files", lambda c: {"completions__x__ctx.json"}, raising=False
    )
    info = _PodInfo(desired_status="RUNNING", ssh_host=None, ssh_port=None, pod_id="pod-id-ima")
    _stub_live_api(monkeypatch, info)
    terminated: list = []
    monkeypatch.setattr(
        runpod_api, "terminate_pod", lambda pid: terminated.append(pid), raising=False
    )
    return terminated


def test_cuda_ima_failover_pivots_with_cuda_ima_stamp_fn(tmp_path, monkeypatch):
    """The first CUDA-IMA failover terminates + re-provisions a fresh host, and
    _relaunch_fresh_runpod is invoked with stamp_fn=_stamp_runpod_cuda_ima_failover
    (the SEPARATE lease field, never the no-port one)."""
    terminated = _cuda_ima_failover_gate_ok(monkeypatch)
    monkeypatch.setattr(
        bp, "_runpod_cuda_ima_already_handled", lambda *a, **k: False, raising=False
    )
    monkeypatch.setattr(
        bp, "_lease_records_runpod_cuda_ima_failover", lambda *a, **k: False, raising=False
    )
    relaunched: list = []

    def _capture_relaunch(**kw):
        relaunched.append(kw)
        return {"status": "running"}

    monkeypatch.setattr(bp, "_relaunch_fresh_runpod", _capture_relaunch, raising=False)

    bp._failover_cuda_ima_runpod(
        issue=689,
        handle=_runpod_handle(),
        result=_cuda_ima_dead_poll(),
        sidecar=_empty_sidecar(tmp_path),
    )
    assert terminated == ["pod-id-ima"]
    assert len(relaunched) == 1
    assert relaunched[0]["stamp_fn"] is bp._stamp_runpod_cuda_ima_failover


def test_cuda_ima_failover_bounded_once_third_crash_terminal_code(tmp_path, monkeypatch):
    """M1 — after one pivot (the durable lease records a CUDA-IMA failover for this
    RUN), a SECOND same-signature crash routes to terminal failure_class:code
    (reason=cuda_ima_repeats_after_failover), NO second pivot. The bound is the
    PER-RUN any-non-null lease check (``_lease_has_any_runpod_cuda_ima_failover``),
    NOT the per-pod identity check — see the real-helper cross-identity test
    ``test_cuda_ima_once_more_bound_blocks_on_fresh_pod_after_stamp`` for the seam
    this mock necessarily abstracts away."""
    terminated = _cuda_ima_failover_gate_ok(monkeypatch)
    # The lease ALREADY records a CUDA-IMA failover for this run (the per-run bound).
    monkeypatch.setattr(
        bp, "_lease_has_any_runpod_cuda_ima_failover", lambda *a, **k: True, raising=False
    )
    relaunched: list = []
    monkeypatch.setattr(
        bp, "_relaunch_fresh_runpod", lambda **kw: relaunched.append(kw), raising=False
    )

    out = bp._failover_cuda_ima_runpod(
        issue=689,
        handle=_runpod_handle(),
        result=_cuda_ima_dead_poll(),
        sidecar=_empty_sidecar(tmp_path),
    )
    assert out["status"] == "dead"
    assert out["failure_class"] == "code"
    assert out["reason"] == "cuda_ima_repeats_after_failover"
    assert terminated == []  # bound checked FIRST, before the terminate
    assert relaunched == []  # NO second pivot


def test_cuda_ima_once_more_bound_blocks_on_fresh_pod_after_stamp(tmp_path, monkeypatch):
    """M1 CRITICAL — the once-more bound is PER-RUN, not per-pod: it must fire on
    the FRESH pod's handle even though the stamp recorded the OLD crashed pod.

    This is the exact case the mocked M1 tests abstract away (they mock the bound
    predicate to True). Here it exercises the REAL durable-lease helpers with
    DISTINCT old/fresh identities:

      1. stamp the lease with the OLD crashed handle via the REAL
         ``_stamp_runpod_cuda_ima_failover`` (records old pod_name/job_id);
      2. drive ``_failover_cuda_ima_runpod`` with a FRESH handle (different
         pod_name AND job_id — the post-pivot sidecar re-point);
      3. assert it routes to terminal ``failure_class: code`` with NO second
         pivot.

    PRE-FIX (identity-equality bound) the fresh handle's identity != the stamped
    old identity, so the bound returns False and the run pivots AGAIN — the
    unbounded-spend bug. POST-FIX (any-non-null bound) the bound fires."""
    _real_lease_store_in(tmp_path, monkeypatch)

    old_handle = _runpod_handle_with_workload(pod_name="pod-775-OLD")
    object.__setattr__(old_handle, "job_id", "job-OLD")
    # 1. Stamp the OLD crashed handle via the REAL helper (records old identity).
    bp._stamp_runpod_cuda_ima_failover(689, old_handle)
    # Sanity: the per-POD (layer-1) identity check is OLD-keyed.
    assert bp._lease_records_runpod_cuda_ima_failover(689, old_handle) is True

    # 2. The FRESH pod (distinct pod_name AND job_id) — the post-pivot re-point.
    fresh_handle = _runpod_handle_with_workload(pod_name="pod-775-FRESH")
    object.__setattr__(fresh_handle, "job_id", "job-FRESH")
    # The OLD identity-keyed check does NOT match the fresh handle (this is exactly
    # why the layer-2 bound cannot be identity-keyed):
    assert bp._lease_records_runpod_cuda_ima_failover(689, fresh_handle) is False
    # But the PER-RUN any-non-null bound DOES fire on the fresh handle:
    assert bp._lease_has_any_runpod_cuda_ima_failover(689) is True

    # Stub the inputs gate OK, terminate to a no-op, and record relaunch calls — so
    # IF the bound failed (pre-fix), the second pivot would be observable.
    monkeypatch.setattr(
        bp,
        "_wedged_run_inputs_on_hf",
        lambda *a, **k: bp._WedgeInputsGate(ok=True, complete=[], partial=[], absent=[]),
        raising=False,
    )
    relaunched: list = []
    monkeypatch.setattr(
        bp,
        "_relaunch_fresh_runpod",
        lambda **kw: relaunched.append(kw) or {"status": "running"},
        raising=False,
    )
    monkeypatch.setattr(runpod_api, "terminate_pod", lambda pid: None, raising=False)

    # 3. The second crash arrives on the FRESH handle.
    out = bp._failover_cuda_ima_runpod(
        issue=689,
        handle=fresh_handle,
        result=_cuda_ima_dead_poll(),
        sidecar=_empty_sidecar(tmp_path),
    )
    assert out["failure_class"] == "code"
    assert out["reason"] == "cuda_ima_repeats_after_failover"
    assert relaunched == []  # NO second pivot on the fresh pod


def test_cuda_ima_failover_idempotency_lease(tmp_path, monkeypatch):
    """A re-fired tick on the SAME crashed handle after a successful pivot
    short-circuits (idempotency) — no double terminate, no double pivot."""
    terminated = _cuda_ima_failover_gate_ok(monkeypatch)
    monkeypatch.setattr(
        bp, "_lease_records_runpod_cuda_ima_failover", lambda *a, **k: False, raising=False
    )
    handled = {"v": False}
    monkeypatch.setattr(
        bp, "_runpod_cuda_ima_already_handled", lambda *a, **k: handled["v"], raising=False
    )
    relaunched: list = []
    monkeypatch.setattr(
        bp,
        "_relaunch_fresh_runpod",
        lambda **kw: relaunched.append(kw) or {"status": "running"},
        raising=False,
    )
    sidecar = _empty_sidecar(tmp_path)
    bp._failover_cuda_ima_runpod(
        issue=689, handle=_runpod_handle(), result=_cuda_ima_dead_poll(), sidecar=sidecar
    )
    assert terminated == ["pod-id-ima"]
    assert len(relaunched) == 1
    # Second tick on the OLD handle: already-handled short-circuits.
    handled["v"] = True
    out = bp._failover_cuda_ima_runpod(
        issue=689, handle=_runpod_handle(), result=_cuda_ima_dead_poll(), sidecar=sidecar
    )
    assert out.get("reason") == "runpod_cuda_ima_already_handled"
    assert terminated == ["pod-id-ima"]  # unchanged
    assert len(relaunched) == 1  # unchanged


def test_cuda_ima_failover_inputs_partial_blocks(tmp_path, monkeypatch):
    """A PARTIAL cell on HF BLOCKS the irreversible terminate (human decides)."""
    import issue664_dispatch as D

    cell = _FakeCell("mk_partial_cell_seed42")
    monkeypatch.setattr(bp, "_issue_cells_for_handle", lambda issue, handle: [cell], raising=False)
    files = _hf_paths_for(
        cell.eval_key, raw_names={"completions__x__ctx.json"}, store_names={"meta.json"}
    )
    monkeypatch.setattr(bp, "_list_issue664_hub_files", lambda repo_id, prefixes: set(files))
    monkeypatch.setattr(
        D, "_expected_eval_files", lambda c: {"completions__x__ctx.json"}, raising=False
    )
    monkeypatch.setattr(
        bp, "_lease_records_runpod_cuda_ima_failover", lambda *a, **k: False, raising=False
    )
    monkeypatch.setattr(
        bp, "_runpod_cuda_ima_already_handled", lambda *a, **k: False, raising=False
    )
    terminated: list = []
    monkeypatch.setattr(
        runpod_api, "terminate_pod", lambda pid: terminated.append(pid), raising=False
    )
    relaunched: list = []
    monkeypatch.setattr(
        bp, "_relaunch_fresh_runpod", lambda **kw: relaunched.append(kw), raising=False
    )
    out = bp._failover_cuda_ima_runpod(
        issue=689,
        handle=_runpod_handle(),
        result=_cuda_ima_dead_poll(),
        sidecar=_empty_sidecar(tmp_path),
    )
    assert out["reason"] == "runpod_cuda_ima_inputs_unverified"
    assert terminated == []
    assert relaunched == []


def test_cuda_ima_failover_durable_lease_bound_real_helpers(tmp_path, monkeypatch):
    """The once-more bound holds via the REAL durable-lease helpers (not mocked):
    after _stamp_runpod_cuda_ima_failover stamps the SEPARATE lease field, the
    bound check reads it and routes a second crash to terminal code. Also proves
    the CUDA-IMA stamp does NOT touch runpod_wedge_failover_of (no cross-suppress)."""
    _real_lease_store_in(tmp_path, monkeypatch)
    handle = _runpod_handle_with_workload()
    # Before any stamp: both the per-pod and the per-run bound are unspent.
    assert bp._lease_records_runpod_cuda_ima_failover(689, handle) is False
    assert bp._lease_has_any_runpod_cuda_ima_failover(689) is False
    # Stamp the CUDA-IMA failover.
    bp._stamp_runpod_cuda_ima_failover(689, handle)
    assert bp._lease_records_runpod_cuda_ima_failover(689, handle) is True
    # The PER-RUN any-non-null bound (the layer-2 once-more guard) now reads True.
    assert bp._lease_has_any_runpod_cuda_ima_failover(689) is True
    # The no-port lease field is UNTOUCHED (no cross-suppression).
    assert bp._lease_records_runpod_wedge_failover(689, handle) is False


def test_cuda_ima_failover_terminate_is_best_effort(tmp_path, monkeypatch):
    """MAJOR-1 — a terminate API failure on the (usually already-dead) CUDA-IMA pod
    MUST NOT block the fresh-host recovery: the failover logs + continues to
    _relaunch_fresh_runpod instead of bubbling the exception up to main()'s outer
    guard (which would emit reason=runpod_cuda_ima_failover_error, masking the
    intended pivot). Contrast Part C's no-port terminate, which stays fail-loud."""
    _cuda_ima_failover_gate_ok(monkeypatch)  # live PodInfo present -> terminate is attempted
    monkeypatch.setattr(
        bp, "_lease_has_any_runpod_cuda_ima_failover", lambda *a, **k: False, raising=False
    )
    monkeypatch.setattr(
        bp, "_runpod_cuda_ima_already_handled", lambda *a, **k: False, raising=False
    )
    # The terminate RAISES (an API race / already-deleted pod / transient hiccup).
    monkeypatch.setattr(
        runpod_api,
        "terminate_pod",
        lambda pid: (_ for _ in ()).throw(RuntimeError("terminate API 500")),
        raising=False,
    )
    relaunched: list = []
    monkeypatch.setattr(
        bp,
        "_relaunch_fresh_runpod",
        lambda **kw: (
            relaunched.append(kw)
            or {"status": "running", "current_phase": bp.RUNPOD_CUDA_IMA_FAILOVER_FRESH_POD_PHASE}
        ),
        raising=False,
    )

    out = bp._failover_cuda_ima_runpod(
        issue=689,
        handle=_runpod_handle(),
        result=_cuda_ima_dead_poll(),
        sidecar=_empty_sidecar(tmp_path),
    )
    # The pivot still fired despite the terminate raising.
    assert len(relaunched) == 1
    assert relaunched[0]["success_phase"] == bp.RUNPOD_CUDA_IMA_FAILOVER_FRESH_POD_PHASE
    assert out["current_phase"] == bp.RUNPOD_CUDA_IMA_FAILOVER_FRESH_POD_PHASE
    # NOT the masked error path.
    assert out.get("reason") != "runpod_cuda_ima_failover_error"


def test_prior_failure_marker_cuda_ima_real_parser_realistic_body(monkeypatch):
    """MINOR — the cross-pod fallback's REAL parser (_prior_failure_marker_is_cuda_ima,
    not mocked) handles a realistic epm:failure marker body: a kind=='epm:failure v1'
    event whose note carries the CUDA-IMA signature text returns True via the actual
    task_workflow.list_events read + the within-line signature regex."""
    import explore_persona_space.task_workflow as TW

    realistic_note = (
        "failure_class: infra\n"
        "reason: vllm_crash\n"
        "trace_summary: torch.AcceleratorError: CUDA error: an illegal memory "
        "access was encountered\n"
        "phase: workload\n"
    )
    # Newest-LAST ordering (list_events returns chronological; the parser reverses).
    events = [
        {"kind": "epm:run-launched v1", "note": "launched pod-775", "ts": "2026-06-30T00:00:00Z"},
        {"kind": "epm:failure v1", "note": realistic_note, "ts": "2026-06-30T01:00:00Z"},
    ]
    monkeypatch.setattr(TW, "list_events", lambda issue: events, raising=False)
    assert bp._prior_failure_marker_is_cuda_ima(689) is True

    # A non-CUDA-IMA epm:failure body returns False (the first crash is not a repeat).
    benign = [
        {
            "kind": "epm:failure v1",
            "note": "failure_class: code\nreason: assertion_error\ntrace_summary: ValueError: bad",
            "ts": "2026-06-30T01:00:00Z",
        }
    ]
    monkeypatch.setattr(TW, "list_events", lambda issue: benign, raising=False)
    assert bp._prior_failure_marker_is_cuda_ima(689) is False
