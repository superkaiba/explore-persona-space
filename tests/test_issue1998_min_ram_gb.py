"""Unit tests for the ``--min-ram-gb`` GCP GPU guard (#1998).

Task #1998 makes ``--min-ram-gb`` (previously a RunPod-CPU-fallback-only
knob, #1010) fail loud on GCP GPU dispatches whose resolved machine
falls below the requested value. Semantics mirror the #1468 A100-40
``--min-gpu-mem-gb`` rung-skip:

* rung-skip an undersized rung when the ladder has a satisfying next
  rung;
* refuse pre-launch when a pinned intent's resolved machine cannot
  satisfy the requirement (equivalently: when the ladder is exhausted —
  every rung falls below the guard).

Every test that exercises ``route()`` monkeypatches
``router.GCP_PROVISIONING_DISABLED = False`` — under #2028
(``GCP_PROVISIONING_DISABLED = True`` on main) any GCP dispatch raises
``GcpDisabledError`` BEFORE the RAM guard can fire (plan v2 §6
boilerplate). Note: ``tests/test_router.py`` sets this via an autouse
fixture; here we set it explicitly per-test because ``test_gcp_backend``
does NOT — this file's route-touching tests inherit the explicit form.

Test roster (plan v2 §6):

1. rung-skip on undersized GPU machine (auto short lane).
2. refuse on pinned intent below RAM.
3. refuse when the wide walker cannot satisfy (ladder-exhausted flavor).
4. marker extras carry ``requested_ram_gb`` + ``resolved_machine_ram_gb``.
5. absent ``--min-ram-gb`` omits both keys entirely (#934).
6. RunPod-CPU-fallback path is byte-unchanged (#1010).
7. help text names GCP coverage + the RunPod-GPU explicit-override residual.
8. MF1: ladder-EXHAUSTED refusal, parametrized: auto (--min-ram-gb 2000)
   AND pinned-explicit-only (sweep-8g-h100 --min-ram-gb 3000).
9. MF2: --min-ram-gb + --min-gpu-mem-gb compose — both predicates fire.
10. Completeness: every reachable machine has a MACHINE_RAM_GIB entry.
11. MF1 route()-level (round-2, closes review concern
    mf1-route-level-no-fallthrough-unpinned): an exhausted ladder raises
    GpuRamBelowMinRamGbError OUT of route() with ZERO gcp create calls
    and ZERO RunPod launches (no _runpod_terminal_rung fall-through).
12. Landed-rung marker extras (round-2, review r1 Minor): a walked rung
    spec (machine_spec_override) posts the RUNG machine's RAM, not the
    base machine's.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from explore_persona_space.backends import (
    BackendKind,
    ComputeBackend,
    LeaseStore,
    NoComputeAvailableError,
    PollResult,
    RouterConfig,
    RunHandle,
    route,
)
from explore_persona_space.backends import router as router_module
from explore_persona_space.backends.gcp import (
    EXPLICIT_WIDE_DEGRADE_INTENTS,
    INTENT_TO_MACHINE,
    MACHINE_RAM_GIB,
    WIDE_A100_80_BY_WIDTH,
    MachineSpec,
    machine_satisfies_min_ram_gb,
    ram_gib_for_machine,
)
from explore_persona_space.backends.issue_dispatch import (
    classify_terminal_exception,
)
from explore_persona_space.backends.router import (
    ROUTE_REASON_GPU_RAM_BELOW_MIN_RAM_GB,
    GpuRamBelowMinRamGbError,
    _filter_ladder_by_min_ram_gb,
    _gcp_ladder_specs,
    _gcp_marker_extras,
    _min_ram_gb_from_spec,
    _rung_machine_type,
)
from explore_persona_space.backends.selector import RunSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec(
    intent: str = "lora-7b",
    *,
    min_ram_gb: int | None = None,
    min_gpu_mem_gb: float | None = None,
    gpus: int | None = None,
    time_budget_hours: float | None = 1.0,
) -> RunSpec:
    """Build a minimal RunSpec threading the flags each test exercises.

    ``time_budget_hours`` defaults to 1.0 so ``_is_short_job`` classifies
    the ladder as SHORT (spot leads) — matches the plan's "short auto
    route" tests. A rung's realized machine is what the RAM guard reads,
    not the length branch, so this is a stability convenience.
    """
    extra: dict[str, Any] = {}
    if min_ram_gb is not None:
        extra["min_ram_gb"] = int(min_ram_gb)
    if min_gpu_mem_gb is not None:
        extra["min_gpu_mem_gb"] = float(min_gpu_mem_gb)
    return RunSpec(
        issue=1998,
        intent=intent,
        backend="gcp",
        gpus=gpus,
        time_budget_hours=time_budget_hours,
        hydra_args=("condition=c1_evil_wrong_em", "seed=42"),
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Section A — pure-helper tests (MACHINE_RAM_GIB, ram_gib_for_machine,
# machine_satisfies_min_ram_gb, _min_ram_gb_from_spec, _rung_machine_type)
# ---------------------------------------------------------------------------


def test_ram_gib_for_machine_lookups_by_machine_type_string() -> None:
    """Lookup by bare machine-type string returns the pinned RAM."""
    assert ram_gib_for_machine("a2-ultragpu-1g") == 170
    assert ram_gib_for_machine("a2-ultragpu-2g") == 340
    assert ram_gib_for_machine("g2-standard-4") == 16


def test_ram_gib_for_machine_lookups_by_machine_spec() -> None:
    """A MachineSpec reads ``.machine_type`` for the lookup."""
    spec = MachineSpec(machine_type="a2-ultragpu-4g", gpu_count=4, gpu_kind="A100-80")
    assert ram_gib_for_machine(spec) == 680


def test_ram_gib_for_machine_raises_keyerror_naming_machine() -> None:
    """A machine absent from MACHINE_RAM_GIB raises KeyError naming it."""
    with pytest.raises(KeyError, match="fictional-machine-99"):
        ram_gib_for_machine("fictional-machine-99")


def test_machine_satisfies_min_ram_gb_true_when_min_is_zero_or_none() -> None:
    """No requirement declared: every machine passes unconditionally."""
    m = MachineSpec(machine_type="g2-standard-4", gpu_count=1, gpu_kind="L4")
    assert machine_satisfies_min_ram_gb(m, None) is True
    assert machine_satisfies_min_ram_gb(m, 0) is True


def test_machine_satisfies_min_ram_gb_boundary() -> None:
    """>= semantics: equal RAM PASSES (170 >= 170), one above FAILS."""
    assert machine_satisfies_min_ram_gb("a2-ultragpu-1g", 170) is True
    assert machine_satisfies_min_ram_gb("a2-ultragpu-1g", 171) is False
    assert machine_satisfies_min_ram_gb("a2-ultragpu-2g", 300) is True
    assert machine_satisfies_min_ram_gb("a2-ultragpu-1g", 300) is False


def test_min_ram_gb_from_spec_extracts_positive_integer() -> None:
    """A positive spec.extra['min_ram_gb'] value returns as int."""
    spec = _spec("lora-7b", min_ram_gb=300)
    assert _min_ram_gb_from_spec(spec) == 300


def test_min_ram_gb_from_spec_returns_none_when_absent() -> None:
    """Absent / falsy / non-positive values return None (no guard fires)."""
    assert _min_ram_gb_from_spec(_spec("lora-7b")) is None
    # Explicit zero should read as "no requirement".
    spec_zero = _spec("lora-7b", min_ram_gb=0)
    assert _min_ram_gb_from_spec(spec_zero) is None


def test_min_ram_gb_from_spec_raises_on_malformed_value() -> None:
    """A malformed present value raises ValueError naming the key."""
    spec = RunSpec(
        issue=1998,
        intent="lora-7b",
        backend="gcp",
        hydra_args=(),
        extra={"min_ram_gb": "abc"},
    )
    with pytest.raises(ValueError, match="min_ram_gb"):
        _min_ram_gb_from_spec(spec)


def test_rung_machine_type_reads_machine_spec_override() -> None:
    """A rung threaded via _with_machine exposes its machine_type."""
    ladder = _gcp_ladder_specs(_spec("lora-7b"))
    # Every rung in a base lora-7b ladder resolves to some a2-* machine.
    machine_types = {_rung_machine_type(rung_spec) for rung_spec, _label in ladder}
    assert machine_types  # ladder is non-empty
    # All resolved machine_types are in the RAM table.
    for mt in machine_types:
        assert mt in MACHINE_RAM_GIB, mt


# ---------------------------------------------------------------------------
# Section B — ladder filter tests (test 1, test 3-flavor rung-skip)
# ---------------------------------------------------------------------------


def test_filter_ladder_by_min_ram_gb_no_op_when_min_is_zero() -> None:
    """No requirement declared: ladder returned unchanged (byte-identical)."""
    spec = _spec("lora-7b", min_ram_gb=None)
    ladder = _gcp_ladder_specs(spec)
    assert ladder  # sanity: lora-7b has rungs
    filtered = _filter_ladder_by_min_ram_gb(ladder, spec=spec, min_ram_gb=None)
    assert filtered == ladder


def test_filter_ladder_by_min_ram_gb_no_op_on_cpu_intent() -> None:
    """CPU intent (gpu_count == 0) is ungated on the GPU path."""
    spec = _spec("cpu-bigmem", min_ram_gb=999_999)
    ladder = _gcp_ladder_specs(spec)
    filtered = _filter_ladder_by_min_ram_gb(ladder, spec=spec, min_ram_gb=999_999)
    assert filtered == ladder


def test_filter_ladder_by_min_ram_gb_no_op_on_empty_ladder() -> None:
    """An empty ladder returns unchanged — no rung to filter or raise."""
    spec = _spec("lora-7b", min_ram_gb=300)
    filtered = _filter_ladder_by_min_ram_gb([], spec=spec, min_ram_gb=300)
    assert filtered == []


def test_filter_ladder_by_min_ram_gb_rung_skip_on_undersized_gpu_machine() -> None:
    """Test 1: --min-ram-gb 100 skips A100-40 (85 GiB), lands on A100-80 (170 GiB).

    A short-job lora-7b ladder walks spot A100-80 → spot A100-40 →
    flex A100-80 → on-demand A100-80 → on-demand A100-40. A 100-GiB
    RAM floor drops both A100-40 rungs (85 GiB); every A100-80 rung
    (170 GiB) survives.
    """
    spec = _spec("lora-7b", min_ram_gb=100)
    raw = _gcp_ladder_specs(spec)
    # Sanity: raw ladder must include at least one a2-highgpu-1g rung.
    raw_machines = {_rung_machine_type(rung_spec) for rung_spec, _label in raw}
    assert "a2-highgpu-1g" in raw_machines, raw_machines
    assert "a2-ultragpu-1g" in raw_machines, raw_machines

    filtered = _filter_ladder_by_min_ram_gb(raw, spec=spec, min_ram_gb=100)
    kept_machines = {_rung_machine_type(rung_spec) for rung_spec, _label in filtered}
    # A100-40 (85 GiB < 100) skipped; A100-80 (170 GiB >= 100) kept.
    assert "a2-highgpu-1g" not in kept_machines
    assert "a2-ultragpu-1g" in kept_machines


def test_filter_ladder_by_min_ram_gb_raises_when_every_rung_undersized() -> None:
    """A ladder whose every rung fails the guard raises GpuRamBelowMinRamGbError.

    The pinned-intent-below-RAM case (test 2) has the same shape: an
    unshardable lora-7b ladder resolves only to a2-highgpu-1g (85 GiB)
    and a2-ultragpu-1g (170 GiB), so --min-ram-gb 300 kills every rung.
    """
    spec = _spec("lora-7b", min_ram_gb=300)
    ladder = _gcp_ladder_specs(spec)
    with pytest.raises(GpuRamBelowMinRamGbError) as excinfo:
        _filter_ladder_by_min_ram_gb(ladder, spec=spec, min_ram_gb=300)
    exc = excinfo.value
    assert exc.intent == "lora-7b"
    assert exc.requested_min_ram_gb == 300
    # The widest attempted machine on a lora-7b ladder is a2-ultragpu-1g (170).
    assert exc.machine == "a2-ultragpu-1g"
    assert exc.resolved_ram_gib == 170


def test_filter_ladder_by_min_ram_gb_ladder_exhausted_at_2000_gib() -> None:
    """MF1: --min-ram-gb 2000 exceeds every rung in MACHINE_RAM_GIB.

    2000 > a3-highgpu-8g (1872) — the widest machine in the table — so
    every reachable rung fails and the guard raises the same typed
    exception as the pinned-intent case (distinguished only by fields).
    """
    spec = _spec("sweep-8g-a100", min_ram_gb=2000)
    ladder = _gcp_ladder_specs(spec)
    with pytest.raises(GpuRamBelowMinRamGbError):
        _filter_ladder_by_min_ram_gb(ladder, spec=spec, min_ram_gb=2000)


def test_filter_ladder_by_min_ram_gb_sweep_8g_h100_pinned_below_ram() -> None:
    """MF1 variant: sweep-8g-h100 (a3-highgpu-8g, 1872 GiB) with --min-ram-gb 3000.

    sweep-8g-h100 is explicit-only and does NOT width-degrade (see
    gcp.EXPLICIT_WIDE_DEGRADE_INTENTS — sweep-8g-a100 only). Its
    ladder resolves only to a3-highgpu-8g; 3000 > 1872 exhausts it.
    """
    spec = _spec("sweep-8g-h100", min_ram_gb=3000, time_budget_hours=10.0)
    # sweep-8g-h100 requires FLEX_START/SPOT; time_budget_hours=10 keeps it
    # in the long-job branch, but the guard reads machine RAM, not length.
    ladder = _gcp_ladder_specs(spec)
    # Every rung must resolve to a3-highgpu-8g (no degradation).
    machines = {_rung_machine_type(rung_spec) for rung_spec, _label in ladder}
    assert machines == {"a3-highgpu-8g"}, machines
    with pytest.raises(GpuRamBelowMinRamGbError) as excinfo:
        _filter_ladder_by_min_ram_gb(ladder, spec=spec, min_ram_gb=3000)
    assert excinfo.value.machine == "a3-highgpu-8g"
    assert excinfo.value.resolved_ram_gib == 1872


def test_filter_ladder_by_min_ram_gb_wide_a100_walks_to_satisfying_width() -> None:
    """Test 3: --gpus 8 --min-ram-gb 300 walks 8g → 4g → 2g rungs; 2g satisfies.

    A width-8 dispatch on a width-eligible intent walks
    a2-ultragpu-8g (1360), -4g (680), -2g (340), then the base tail.
    A 300-GiB floor keeps all three A100-80 wide rungs (340 >= 300) but
    drops the base a2-ultragpu-1g (170) and the A100-40 fallback rung.
    """
    spec = _spec("lora-7b", gpus=8, min_ram_gb=300)
    raw = _gcp_ladder_specs(spec)
    filtered = _filter_ladder_by_min_ram_gb(raw, spec=spec, min_ram_gb=300)
    kept_machines = {_rung_machine_type(rung_spec) for rung_spec, _label in filtered}
    # All three wide widths kept.
    assert "a2-ultragpu-8g" in kept_machines
    assert "a2-ultragpu-4g" in kept_machines
    assert "a2-ultragpu-2g" in kept_machines
    # Base a2-ultragpu-1g (170) skipped by RAM floor.
    assert "a2-ultragpu-1g" not in kept_machines


def test_filter_ladder_by_min_ram_gb_composes_with_min_gpu_mem_gb() -> None:
    """MF2: both flags fire, keeping only rungs that satisfy BOTH predicates.

    ``lora-7b --min-gpu-mem-gb 40 --min-ram-gb 300``:
    - #1468 gate: A100-40 (< 40 GiB usable) dropped by ``_gcp_ladder_specs``.
    - #1998 gate: a2-ultragpu-1g (170 GiB RAM) dropped by our filter.
    Result: with the width upgrade to gpus=2, only the wide A100-80
    rungs survive.
    """
    # Cross both predicates by widening (so an A100-80 above the RAM
    # floor exists) and setting min_gpu_mem_gb above the #1468 gate.
    spec = _spec("lora-7b", gpus=2, min_gpu_mem_gb=40.0, min_ram_gb=300)
    raw = _gcp_ladder_specs(spec)
    # Verify #1468 gate already dropped A100-40 from the raw ladder.
    raw_machines = {_rung_machine_type(rung_spec) for rung_spec, _label in raw}
    assert "a2-highgpu-1g" not in raw_machines
    filtered = _filter_ladder_by_min_ram_gb(raw, spec=spec, min_ram_gb=300)
    kept_machines = {_rung_machine_type(rung_spec) for rung_spec, _label in filtered}
    # a2-ultragpu-1g (170 GiB) dropped by #1998 gate.
    assert "a2-ultragpu-1g" not in kept_machines
    # a2-ultragpu-2g (340 GiB) survives.
    assert "a2-ultragpu-2g" in kept_machines


# ---------------------------------------------------------------------------
# Section C — marker extras tests (test 4, test 5, test 6)
# ---------------------------------------------------------------------------


def test_epm_backend_selected_carries_ram_extras_when_min_ram_gb_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 4: with --min-ram-gb 300 on a wide dispatch, marker extras carry
    both ``requested_ram_gb`` and ``resolved_machine_ram_gb``.
    """
    monkeypatch.setattr(router_module, "GCP_PROVISIONING_DISABLED", False)
    # A width-eligible intent whose BASE machine (a2-ultragpu-1g) has
    # 170 GiB RAM — the marker extras read the base machine, not a
    # per-rung walked width (per _gcp_marker_extras docstring).
    spec = _spec("lora-7b", min_ram_gb=300)
    extras = _gcp_marker_extras(spec)
    assert extras.get("requested_ram_gb") == 300
    assert extras.get("resolved_machine_ram_gb") == 170


def test_min_ram_gb_absent_no_ram_extras_key_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 5: without --min-ram-gb, marker extras omit BOTH keys entirely (#934).

    Assert KEY absence (``not in``) — NOT ``.get(...) is None``, which
    would silently accept a None-valued key that breaks #934.
    """
    monkeypatch.setattr(router_module, "GCP_PROVISIONING_DISABLED", False)
    spec = _spec("lora-7b", min_ram_gb=None)
    extras = _gcp_marker_extras(spec)
    assert "requested_ram_gb" not in extras
    assert "resolved_machine_ram_gb" not in extras


def test_min_ram_gb_on_cpu_intent_no_ram_extras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 6: --min-ram-gb on a CPU intent does NOT populate the GPU RAM extras.

    CPU intents (gpu_count == 0) route through _refuse_infeasible_cpu_footprint
    (#1010); the GPU-side marker extras must not leak keys there.
    """
    monkeypatch.setattr(router_module, "GCP_PROVISIONING_DISABLED", False)
    spec = _spec("cpu-bigmem", min_ram_gb=200)
    extras = _gcp_marker_extras(spec)
    assert "requested_ram_gb" not in extras
    assert "resolved_machine_ram_gb" not in extras


def test_min_ram_gb_from_spec_cpu_min_ram_gb_still_extracted() -> None:
    """The RunPod-CPU-fallback feasibility gate (#1010) is byte-unchanged:
    _min_ram_gb_from_spec returns the value regardless of intent kind
    (the CPU-side reader still parses it into _refuse_infeasible_cpu_footprint).
    """
    spec = _spec("cpu-bigmem", min_ram_gb=200)
    assert _min_ram_gb_from_spec(spec) == 200


# ---------------------------------------------------------------------------
# Section D — classify_terminal_exception mapping
# ---------------------------------------------------------------------------


def test_classify_terminal_exception_maps_gpu_ram_below_error() -> None:
    """MF1 tail: the typed exception maps to failure_class=infra,
    status=blocked, reason=gpu_ram_below_min_ram_gb (NOT no_compute_available).

    A DESIGN mismatch is NOT a transient capacity outcome — the watcher's
    capacity-retry pass (TRANSIENT_CAPACITY_REASONS keys on
    ``no_compute_available``) must not re-drive it.
    """
    exc = GpuRamBelowMinRamGbError(
        intent="lora-7b",
        machine="a2-ultragpu-1g",
        resolved_ram_gib=170,
        requested_min_ram_gb=300,
    )
    translation = classify_terminal_exception(exc)
    assert translation.failure_class == "infra"
    assert translation.status == "blocked"
    assert f"reason: {ROUTE_REASON_GPU_RAM_BELOW_MIN_RAM_GB}" in translation.note
    # DISTINCT from the no_compute_available reason.
    assert "reason: no_compute_available" not in translation.note


def test_route_reason_gpu_ram_is_the_expected_token() -> None:
    """The reason constant is the exact token watcher/dashboard predicates key on."""
    assert ROUTE_REASON_GPU_RAM_BELOW_MIN_RAM_GB == "gpu_ram_below_min_ram_gb"


# ---------------------------------------------------------------------------
# Section E — completeness test (test 10)
# ---------------------------------------------------------------------------


def test_ram_table_covers_all_reachable_machines() -> None:
    """Every machine reachable via INTENT_TO_MACHINE, WIDE_A100_80_BY_WIDTH,
    or EXPLICIT_WIDE_DEGRADE_INTENTS must have a MACHINE_RAM_GIB entry.

    A future ladder addition (a new INTENT_TO_MACHINE row, a wider
    A100-80 machine) that forgets to update MACHINE_RAM_GIB fails
    this test — the reuse rule of choice for "does this machine
    have a RAM row?" (planned in plan v2 §4).
    """
    reachable: set[str] = set()
    # Every base intent machine.
    for spec in INTENT_TO_MACHINE.values():
        reachable.add(spec.machine_type)
    # Every wide-A100-80 rung.
    for spec in WIDE_A100_80_BY_WIDTH.values():
        reachable.add(spec.machine_type)
    # Every explicit-wide-degrade intent's base machine.
    for intent in EXPLICIT_WIDE_DEGRADE_INTENTS:
        reachable.add(INTENT_TO_MACHINE[intent].machine_type)
    missing = reachable - set(MACHINE_RAM_GIB)
    assert not missing, (
        f"Machines reachable via the ladder but missing from MACHINE_RAM_GIB: "
        f"{sorted(missing)}. Add rows to backends/gcp.MACHINE_RAM_GIB — see "
        f"its docstring for sourcing."
    )


def test_ram_table_covers_a100_40_fallback_machine() -> None:
    """The A100-40 fallback rung (a2-highgpu-1g, INTENT_A100_40_FALLBACK)
    is also reachable via the ladder and must be in MACHINE_RAM_GIB —
    the #1468 fallback rung participates in the RAM guard exactly like
    every other rung.
    """
    assert "a2-highgpu-1g" in MACHINE_RAM_GIB


# ---------------------------------------------------------------------------
# Section F — help-text coverage (test 7)
# ---------------------------------------------------------------------------


def test_min_ram_gb_help_documents_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 7: --help output for the launch subcommand documents:
    - GCP GPU dispatch coverage (#1998),
    - the RunPod-CPU-fallback role (#1010),
    - the SLURM-lane inertness,
    - the RunPod-GPU EXPLICIT-OVERRIDE inertness residual,
    AND no longer contains the stale "RunPod-CPU-fallback knob" lead.
    """

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["uv", "run", "python", str(repo_root / "scripts/dispatch_issue.py"), "launch", "--help"],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    # argparse wraps long help text across lines; collapse whitespace so
    # phrase-presence tests are line-wrap tolerant.
    raw = proc.stdout + proc.stderr
    out = " ".join(raw.split())
    # The new GCP GPU coverage phrase.
    assert "GCP GPU dispatch (#1998)" in out
    # Preserved: RunPod-CPU-fallback role.
    assert "RunPod CPU fallback (#1010)" in out
    # Preserved: SLURM inertness.
    assert "SLURM lanes: inert" in out
    # New: RunPod-GPU explicit-override residual.
    assert "RunPod-GPU explicit-override lane" in out
    assert "remains inert" in out
    # The stale "RunPod-CPU-fallback knob" lead is retired.
    assert "RunPod-CPU-fallback knob" not in out


# ---------------------------------------------------------------------------
# Section G — route()-level MF1 no-fall-through pin (test 11; round-2,
# closes review concern mf1-route-level-no-fallthrough-unpinned)
# ---------------------------------------------------------------------------


class _RecordingBackend(ComputeBackend):
    """Minimal recording backend double (the ``tests/test_router.py``
    ``_BaseBackend`` shape): records every ``launch`` call; every other
    hook is inert. Used to PROVE the RAM-guard raise reaches the caller
    with zero create calls on either backend.
    """

    def __init__(self, kind: BackendKind) -> None:
        self._kind = kind
        self.launches: list[RunSpec] = []

    @property
    def name(self) -> BackendKind:
        return self._kind

    def prepare(self, spec: RunSpec) -> None:
        return None

    def launch(self, spec: RunSpec) -> RunHandle:
        self.launches.append(spec)
        return RunHandle(
            backend=self._kind,
            cluster=None,
            job_id="fake-1",
            pod_name=f"pod-{spec.issue}",
            scratch_dir="/workspace",
            log_path=f"/workspace/logs/issue-{spec.issue}.log",
            extra={"issue": spec.issue},
        )

    def estimate_start(self, spec: RunSpec) -> datetime | None:
        return None

    def poll(self, handle: RunHandle) -> PollResult:
        return PollResult(
            status="running",
            current_phase="running",
            new_milestone=False,
            last_log_mtime_sec_ago=10**9,
            pid_alive=True,
            log_tail_excerpt="",
        )

    def fetch_logs(self, handle: RunHandle) -> str:
        return ""

    def fetch_results(self, handle: RunHandle) -> None:
        return None

    def confirm_artifacts(self, handle: RunHandle) -> bool:
        return True

    def teardown(self, handle: RunHandle) -> None:
        return None


def test_route_gcp_pin_min_ram_gb_exhausted_raises_no_create_no_runpod(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test 11 — MF1 route()-level no-fall-through pin (plan v2 §6 test 8).

    Explicit ``backend: gcp`` pin (``_spec`` pins ``backend="gcp"``) with
    ``--min-ram-gb 2000`` — above EVERY machine in ``MACHINE_RAM_GIB``
    (max: a3-highgpu-8g, 1872 GiB) — must raise
    ``GpuRamBelowMinRamGbError`` OUT of ``route()`` with:

    * ZERO gcp create calls across every rung (the guard fires before
      any ``gcloud compute instances create``), and
    * ZERO RunPod launches — the raise bypasses ``_runpod_terminal_rung``.
      Both ``_attempt_gcp_lane`` callers catch ONLY ``_GcpWorkloadFailover``
      today; a future broad ``except RouteError`` refactor around the lane
      would silently re-open the MF1 fall-through and fail THIS test.

    The exception is deliberately NOT ``NoComputeAvailableError``, so the
    terminal can never classify as the watcher-re-drivable
    ``no_compute_available`` (mapping pinned by
    ``test_classify_terminal_exception_maps_gpu_ram_below_error``).

    Monkeypatches ``GCP_PROVISIONING_DISABLED = False`` per the module
    docstring boilerplate (#2028: the flag-ON build raises
    ``GcpDisabledError`` before the RAM guard can fire).
    """
    monkeypatch.setattr(router_module, "GCP_PROVISIONING_DISABLED", False)
    gcp = _RecordingBackend("gcp")
    rp = _RecordingBackend("runpod")
    spec = _spec("lora-7b", min_ram_gb=2000)
    with pytest.raises(GpuRamBelowMinRamGbError) as excinfo:
        route(
            spec,
            runpod_backend=rp,
            free_backends={},
            gcp_backend=gcp,
            lease_store=LeaseStore(lease_dir=tmp_path / ".eps-routing"),
            is_started=lambda _b, _h: True,
            is_live_after_cancel=lambda _b, _h: False,
            config=RouterConfig(free_wait_seconds=1, poll_interval=0.0, cancel_grace_seconds=0),
            sleep_fn=lambda _s: None,
        )
    # NOT the watcher-re-drivable capacity terminal.
    assert not isinstance(excinfo.value, NoComputeAvailableError)
    assert excinfo.value.requested_min_ram_gb == 2000
    # Zero gcp create invocations across every rung.
    assert gcp.launches == []
    # No fall-through to the RunPod terminal rung.
    assert rp.launches == []


def test_gcp_marker_extras_landed_walked_rung_posts_rung_machine_ram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 12 — the landed-rung path of ``_gcp_marker_extras`` (r1 Minor).

    The launch-success post site (``_attempt_one_gcp_rung``) passes the
    RUNG spec, whose ``machine_spec_override`` ``machine_for_intent``
    honors — so a width-8 lora-7b dispatch that walked down to the
    a2-ultragpu-2g rung must post ``resolved_machine_ram_gb == 340`` (the
    walked rung's machine), never the base a2-ultragpu-1g machine's 170.
    """
    monkeypatch.setattr(router_module, "GCP_PROVISIONING_DISABLED", False)
    spec = _spec("lora-7b", gpus=8, min_ram_gb=300)
    ladder = _gcp_ladder_specs(spec)
    rung_spec = next(rs for rs, _label in ladder if _rung_machine_type(rs) == "a2-ultragpu-2g")
    extras = _gcp_marker_extras(rung_spec)
    assert extras.get("requested_ram_gb") == 300
    assert extras.get("resolved_machine_ram_gb") == 340
