"""Unit tests for the GCP ComputeBackend (slice 3 of the multi-backend router).

Every test mocks the ``gcloud`` subprocess via the injected ``runner``
seam — the unit suite NEVER hits a real GCP project. The live acceptance
run (the GCP per-lane check in slice 8) is what proves the integration;
this file pins the contract the live run consumes.

Coverage maps to the slice-3 acceptance checklist in the implementer
brief:

* Golden ``gcloud compute instances create`` argv for each intent.
* Per-intent machine-type table.
* Idempotent reconnect (no second create when an instance already
  exists).
* ``launch`` populates :class:`ExpectedArtifacts` (incl. the sentinel
  path) onto handle.extra so the slice-2 verifier can run.
* ``confirm_artifacts`` delegates to the slice-2 verifier (PASS/FAIL
  honored).
* Failure classification: capacity → typed provisioning exception (the
  router will fall back); workload-shaped failure is distinct.
* ``teardown`` is idempotent on a missing instance.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from explore_persona_space.backends import (
    EXPECTED_ARTIFACTS_HANDLE_KEY,
    INTENT_A100_40_FALLBACK,
    INTENT_TO_MACHINE,
    GcpBackend,
    GcpConfig,
    GcpProvisioningError,
    MachineSpec,
    RunSpec,
    a100_40_fallback_for_intent,
    audit_stale_gcp_vms,
    default_gcp_config,
    render_create_argv,
)
from explore_persona_space.backends.gcp import (
    _JANITOR_FENCE_GRACE_SECONDS,
    _ZOMBIE_GUEST_PHASES,
    DEFAULT_GCLOUD_CONFIG,
    DEFAULT_IMAGE_FAMILY,
    DEFAULT_IMAGE_PROJECT,
    DEFAULT_PRIMARY_ZONE,
    DEFAULT_PROJECT,
    JANITOR_CLASS_ALLOWLISTED,
    JANITOR_CLASS_KEEP,
    JANITOR_CLASS_MANAGED,
    JANITOR_CLASS_UNMANAGED,
    JANITOR_LIST_NAME_FILTER,
    REQUIRED_LAUNCH_SECRET_KEYS,
    GcloudRunResult,
    GcpLaunchSecretsMissing,
    StaleNamedInstance,
    _classify_janitor_instance,
    _gcp_status_to_poll_result,
    _instance_max_run_seconds,
    _stale_named_instance_or_none,
    attempt_id_for,
    classify_create_failure,
    expected_artifacts_declaration,
    instance_name_for,
    log_path_for,
    machine_for_intent,
    preflight_quota_headroom,
    quota_metric_for,
    reconnect_or_none,
    render_delete_argv,
    render_describe_argv,
    render_list_argv,
    render_startup_script,
    resolve_launch_secrets,
    resolve_provisioning_model,
    resolve_request_valid_for_duration,
    sentinel_path_for,
)

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _required_launch_secrets(monkeypatch):
    """launch() fails loud without the required workload secrets (fix20).

    Set them on every test so the suite is hermetic regardless of the
    invoking shell's env (and so resolve_launch_secrets's dotenv
    fallback never has to read the real repo .env in tests —
    override=False keeps these monkeypatched values authoritative).
    """
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setenv("WANDB_API_KEY", "wandb_test_key")
    # Hermetic for the OPTIONAL secret keys too: a real token leaking in
    # from the invoking shell would make render_create_argv demand a
    # tempfile entry the direct-render tests don't thread (and could put
    # suite behavior at the mercy of the developer's env).
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def _no_real_marker_reads(monkeypatch):
    """Never let a forgotten ``marker_reader=`` inject read the real tasks/ tree.

    ``GcpBackend.__init__`` defaults the relaunch-follow marker reader to
    ``task_workflow.latest_event`` (a real events.jsonl read); a poll test
    that reaches a terminal guest-attribute phase with an ``issue`` extra
    would otherwise depend on the live task #137 marker trail. Mirrors
    ``_required_launch_secrets``' hermeticity guarantee.
    """
    monkeypatch.setattr(
        "explore_persona_space.task_workflow.latest_event",
        lambda *_a, **_k: None,
    )


# Tempfile paths for the autouse env secrets — render_create_argv only
# EMBEDS the paths (gcloud reads the files), so fixed fake paths keep the
# direct-render tests deterministic. launch()-level tests exercise the
# real tempfile lifecycle (write + 0600 + unlink-in-finally).
_TEST_SECRET_FILES: dict[str, str] = {
    "HF_TOKEN": "/tmp/eps-test-secret-hf",
    "WANDB_API_KEY": "/tmp/eps-test-secret-wandb",
}


def _spec(intent: str = "lora-7b", **overrides: Any) -> RunSpec:
    """Build a RunSpec with a deterministic attempt-id (no clock noise)."""
    base_extra: dict[str, Any] = {"attempt_id": "att-fixed-001"}
    extra = overrides.pop("extra", None)
    if extra:
        base_extra.update(extra)
    hydra_args = overrides.pop("hydra_args", ("condition=c1_evil_wrong_em", "seed=42"))
    return RunSpec(
        issue=137,
        intent=intent,
        backend="gcp",
        hydra_args=hydra_args,
        extra=base_extra,
        **overrides,
    )


def _test_config() -> GcpConfig:
    """Test-fixture config (matches production defaults but explicit)."""
    return GcpConfig(
        project="eps-test-project",
        gcloud_config="eps-test-config",
        primary_zone="us-central1-a",
        fallback_zones=("us-central1-b", "us-central1-c"),
        image_family="pytorch-test-family",
        image_project="deeplearning-platform-release",
        repo_url="https://github.com/superkaiba/explore-persona-space.git",
    )


class _Runner:
    """Test runner: records argv + returns scripted GcloudRunResult per call.

    The harness inspects the argv to figure out which gcloud subcommand
    is being called (``create`` / ``list`` / ``describe`` / ``delete``)
    and returns the next scripted result for that bucket. Tests that
    need a single result per call drop the scripted list to length 1.
    """

    def __init__(
        self,
        *,
        create_results: list[GcloudRunResult] | None = None,
        list_results: list[GcloudRunResult] | None = None,
        describe_results: list[GcloudRunResult] | None = None,
        delete_results: list[GcloudRunResult] | None = None,
        serial_results: list[GcloudRunResult] | None = None,
        guest_attr_results: list[GcloudRunResult] | None = None,
        ssh_results: list[GcloudRunResult] | None = None,
        scp_results: list[GcloudRunResult] | None = None,
        region_describe_results: list[GcloudRunResult] | None = None,
        create_raises: BaseException | None = None,
        list_raises: BaseException | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.create_results = list(create_results or [])
        self.list_results = list(list_results or [])
        self.describe_results = list(describe_results or [])
        self.delete_results = list(delete_results or [])
        self.serial_results = list(serial_results or [])
        self.guest_attr_results = list(guest_attr_results or [])
        self.ssh_results = list(ssh_results or [])
        self.scp_results = list(scp_results or [])
        self.region_describe_results = list(region_describe_results or [])
        # When set, RAISE the given exception off the matching gcloud
        # subcommand. ``create_raises`` fires the FIRST time a create argv is
        # seen. ``list_raises`` fires the first list argv seen AFTER the
        # create raised — i.e. the POST-TIMEOUT probe list, NOT the two
        # pre-create lists (the idempotent reconnect at the top of launch +
        # the stale-name check), which must still return their scripted
        # ``[]``. Lets the #736 create-timeout tests drive
        # ``subprocess.TimeoutExpired`` off the create call (and, for the
        # probe-timeout test, off the post-timeout list) — the base
        # ``_Runner`` only returns, never raises.
        self._create_raises = create_raises
        self._list_raises = list_raises
        self._create_raised = False

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        # gcloud compute instances <subcommand> ...
        if "create" in argv and "instances" in argv:
            if self._create_raises is not None:
                exc, self._create_raises = self._create_raises, None
                self._create_raised = True
                raise exc
            return self._pop(self.create_results, default_ok=True)
        if "list" in argv and "instances" in argv:
            if self._list_raises is not None and self._create_raised:
                exc, self._list_raises = self._list_raises, None
                raise exc
            return self._pop(self.list_results, default_ok=True, default_stdout="[]")
        if "describe" in argv and "instances" in argv:
            return self._pop(self.describe_results, default_ok=True, default_stdout="{}")
        if "describe" in argv and "regions" in argv:
            return self._pop(self.region_describe_results, default_ok=True, default_stdout="{}")
        if "get-guest-attributes" in argv and "instances" in argv:
            # Default: attribute not yet written (gcloud exits 1) — the
            # poll treats that as phase-unknown and keeps the coarse
            # describe classification.
            if self.guest_attr_results:
                return self._pop(self.guest_attr_results, default_ok=False)
            return GcloudRunResult(1, "", "guest attribute eps/phase not found")
        if "delete" in argv and "instances" in argv:
            return self._pop(self.delete_results, default_ok=True)
        if "get-serial-port-output" in argv:
            return self._pop(self.serial_results, default_ok=True)
        # gcloud compute ssh / scp (fetch_results sentinel pull + best-
        # effort dir mirrors).
        if "ssh" in argv and "compute" in argv:
            return self._pop(self.ssh_results, default_ok=True)
        if "scp" in argv and "compute" in argv:
            return self._pop(self.scp_results, default_ok=True)
        raise AssertionError(f"unexpected gcloud argv in test: {argv}")

    @staticmethod
    def _pop(
        bucket: list[GcloudRunResult], *, default_ok: bool, default_stdout: str = ""
    ) -> GcloudRunResult:
        if bucket:
            return bucket.pop(0)
        if default_ok:
            return GcloudRunResult(returncode=0, stdout=default_stdout, stderr="")
        return GcloudRunResult(returncode=1, stdout="", stderr="no scripted result")


@pytest.fixture
def no_marker_posts(monkeypatch):
    """Defense in depth: never let a test shell out to real task.py post-marker.

    Mirrors the SLURM tests' autouse fixture so a forgotten ``marker_poster=``
    inject can't pollute a real events.jsonl trail.
    """
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm.post_marker_via_task_py",
        lambda **_kw: None,
    )


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_default_gcp_config_threads_production_constants() -> None:
    cfg = default_gcp_config()
    assert cfg.project == DEFAULT_PROJECT == "eps-persona-gpu-jun2026"
    assert cfg.gcloud_config == DEFAULT_GCLOUD_CONFIG == "eps-gcp"
    assert cfg.primary_zone == DEFAULT_PRIMARY_ZONE == "us-central1-a"
    assert cfg.image_family == DEFAULT_IMAGE_FAMILY
    assert cfg.image_project == DEFAULT_IMAGE_PROJECT
    assert "us-central1-b" in cfg.fallback_zones
    assert cfg.default_boot_disk_type == "pd-ssd"
    assert cfg.default_max_run_duration == "7d"  # #741: raised from 24h (FLEX_START ceiling)


# ---------------------------------------------------------------------------
# Intent → machine
# ---------------------------------------------------------------------------


def test_intent_to_machine_table_matches_plan() -> None:
    """The plan's "gcp.py" Approach paragraph spells out these mappings."""
    assert INTENT_TO_MACHINE["lora-7b"].machine_type == "a2-ultragpu-1g"
    assert INTENT_TO_MACHINE["lora-7b"].gpu_count == 1
    assert INTENT_TO_MACHINE["lora"].machine_type == "a2-ultragpu-1g"
    assert INTENT_TO_MACHINE["ft-7b"].machine_type == "a2-ultragpu-4g"
    assert INTENT_TO_MACHINE["ft-7b"].gpu_count == 4
    assert INTENT_TO_MACHINE["eval"].machine_type == "g2-standard-4"
    assert INTENT_TO_MACHINE["debug"].machine_type == "g2-standard-4"


def test_intent_to_machine_includes_h100_intents() -> None:
    """#631 D2: the two H100 intents map to the a3-highgpu family."""
    assert INTENT_TO_MACHINE["lora-7b-h100"].machine_type == "a3-highgpu-1g"
    assert INTENT_TO_MACHINE["lora-7b-h100"].gpu_count == 1
    assert INTENT_TO_MACHINE["lora-7b-h100"].gpu_kind == "H100-80"
    assert INTENT_TO_MACHINE["eval-h100"].machine_type == "a3-highgpu-2g"
    assert INTENT_TO_MACHINE["eval-h100"].gpu_count == 2
    assert INTENT_TO_MACHINE["eval-h100"].gpu_kind == "H100-80"


def test_intent_to_machine_includes_capture_7b() -> None:
    """#752: the activation-capture intent routes a 7B hidden-state-capturing
    forward to A100-80 (primary) / A100-40 (fallback), NOT the L4 eval default
    that OOM'd #666/#744.

    capture-7b shares lora-7b's a2-ultragpu-1g (1x A100-80) primary but is a
    DISTINCT intent so a forward-pass-only activation-capture run is sized
    correctly without coupling the router to workload semantics, and its
    A100-40 fallback (a2-highgpu-1g) fits the single-GPU 7B capture in 40 GB.
    """
    from explore_persona_space.backends.gcp import zones_for_machine_type

    # Primary: a2-ultragpu-1g (1x A100-80), same machine as lora-7b.
    assert INTENT_TO_MACHINE["capture-7b"].machine_type == "a2-ultragpu-1g"
    assert INTENT_TO_MACHINE["capture-7b"].gpu_kind == "A100-80"
    assert INTENT_TO_MACHINE["capture-7b"].gpu_count == 1
    assert machine_for_intent(_spec("capture-7b")).machine_type == "a2-ultragpu-1g"

    # A100-40 fallback rung: a2-highgpu-1g (single-GPU 7B fits 40 GB).
    assert INTENT_A100_40_FALLBACK["capture-7b"].machine_type == "a2-highgpu-1g"
    assert INTENT_A100_40_FALLBACK["capture-7b"].gpu_kind == "A100-40"
    fallback = a100_40_fallback_for_intent(_spec("capture-7b"))
    assert fallback is not None
    assert fallback.machine_type == "a2-highgpu-1g"

    # The inherited zone restriction does not filter the new intent's machine
    # to nothing — a2-ultragpu-1g is already in MACHINE_TYPE_ZONE_AVAILABILITY,
    # so the new intent inherits its zones with no separate entry.
    ladder = ["us-central1-a", "us-central1-b", "us-central1-c"]
    assert zones_for_machine_type("a2-ultragpu-1g", ladder)


def test_machine_for_intent_resolves_known_intent() -> None:
    spec = _spec("ft-7b")
    machine = machine_for_intent(spec)
    assert isinstance(machine, MachineSpec)
    assert machine.machine_type == "a2-ultragpu-4g"
    assert machine.gpu_count == 4
    assert machine.gpu_kind == "A100-80"


def test_machine_for_intent_rejects_unknown_intent_loud() -> None:
    """Fail-fast on a typo (consistent with SLURM's intent table)."""
    spec = _spec("totally-bogus")
    with pytest.raises(ValueError, match="no GCP machine-type for intent"):
        machine_for_intent(spec)


# ---------------------------------------------------------------------------
# CPU-only intent: cpu-bigmem (#677)
# ---------------------------------------------------------------------------


def test_intent_to_machine_includes_cpu_bigmem() -> None:
    """#677: the CPU-only analysis intent maps to a gpu_count=0 n2-highmem-16."""
    spec = INTENT_TO_MACHINE["cpu-bigmem"]
    assert spec == MachineSpec(machine_type="n2-highmem-16", gpu_count=0, gpu_kind="CPU")
    assert spec.machine_type == "n2-highmem-16"
    assert spec.gpu_count == 0
    assert spec.gpu_kind == "CPU"


def test_machine_for_intent_resolves_cpu_bigmem() -> None:
    """#677: machine_for_intent resolves the cpu-bigmem row."""
    machine = machine_for_intent(_spec("cpu-bigmem"))
    assert machine == MachineSpec(machine_type="n2-highmem-16", gpu_count=0, gpu_kind="CPU")


# ---------------------------------------------------------------------------
# Cheap CPU-only intents: cpu-small / cpu-mid (#747)
# ---------------------------------------------------------------------------


def test_intent_to_machine_includes_cpu_small_and_cpu_mid() -> None:
    """#747: the two cheap CPU intents map to gpu_count=0 E2 machines."""
    assert INTENT_TO_MACHINE["cpu-small"] == MachineSpec(
        machine_type="e2-standard-2", gpu_count=0, gpu_kind="CPU"
    )
    assert INTENT_TO_MACHINE["cpu-mid"] == MachineSpec(
        machine_type="e2-standard-8", gpu_count=0, gpu_kind="CPU"
    )


def test_intent_to_machine_cpu_bigmem_unchanged() -> None:
    """#747 regression guard: cpu-bigmem is PRESERVED VERBATIM (not renamed /
    re-mapped by the #747 cheap-CPU rows)."""
    assert INTENT_TO_MACHINE["cpu-bigmem"] == MachineSpec(
        machine_type="n2-highmem-16", gpu_count=0, gpu_kind="CPU"
    )


def test_render_create_argv_cpu_small_golden() -> None:
    """#747: a cpu-small create renders a valid CPU argv — MIGRATE (not
    TERMINATE), NO --accelerator, leak guards intact (mirrors the #677
    cpu-bigmem golden test)."""
    cfg = _test_config()
    argv = render_create_argv(
        spec=_spec("cpu-small"),
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\necho startup\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--machine-type=e2-standard-2" in argv
    assert "--maintenance-policy=MIGRATE" in argv
    assert "--maintenance-policy=TERMINATE" not in argv
    assert not any(a.startswith("--accelerator") for a in argv), argv
    assert "--instance-termination-action=DELETE" in argv
    assert "--no-restart-on-failure" in argv
    assert "--max-run-duration=7d" in argv  # #741: config default raised 24h → 7d


def test_quota_metric_for_cpu_small_returns_none() -> None:
    """#747: the cheap CPU machines draw no accelerator quota under any pool
    (extends the #677 cpu-bigmem quota test to the new intents)."""
    for intent in ("cpu-small", "cpu-mid"):
        machine = INTENT_TO_MACHINE[intent]
        for provisioning in ("STANDARD", "SPOT", "FLEX_START"):
            assert quota_metric_for(machine, provisioning) is None, (intent, provisioning)


def test_e2_zone_availability_listed() -> None:
    """#747: MACHINE_TYPE_ZONE_AVAILABILITY records the verified E2 us-central1
    zones for the cpu-small / cpu-mid machine types."""
    from explore_persona_space.backends.gcp import MACHINE_TYPE_ZONE_AVAILABILITY

    expected = frozenset({"us-central1-a", "us-central1-b", "us-central1-c"})
    assert MACHINE_TYPE_ZONE_AVAILABILITY["e2-standard-2"] == expected
    assert MACHINE_TYPE_ZONE_AVAILABILITY["e2-standard-8"] == expected


def test_gcp_handle_extra_gpu_count_zero_for_cpu_small(no_marker_posts) -> None:
    """#747: a cpu-small GCP handle carries extra["gpu_count"] == 0 (the
    async-failover prerequisite; the predicate then keys on the intent being in
    the RunPod-CPU map). Mirrors the #677 cpu-bigmem handle test."""
    created = json.dumps([{"name": "eps-issue-747", "id": "999"}])
    backend = GcpBackend(
        config=_test_config(),
        runner=_Runner(
            list_results=[GcloudRunResult(0, "[]", "")],
            create_results=[GcloudRunResult(0, created, "")],
        ),
        marker_poster=lambda **_: None,
    )
    handle = backend.launch(_spec(intent="cpu-small"))
    assert handle.extra["gpu_count"] == 0
    assert handle.extra["intent"] == "cpu-small"


# ---------------------------------------------------------------------------
# 8-GPU sweep intents: sweep-8g-a100 + sweep-8g-h100 (#743)
# ---------------------------------------------------------------------------


def test_intent_to_machine_includes_8gpu_sweep_intents() -> None:
    """#743: the two 8-GPU sweep intents map to the 8g machine types."""
    a100 = INTENT_TO_MACHINE["sweep-8g-a100"]
    assert a100 == MachineSpec(machine_type="a2-ultragpu-8g", gpu_count=8, gpu_kind="A100-80")
    h100 = INTENT_TO_MACHINE["sweep-8g-h100"]
    assert h100 == MachineSpec(machine_type="a3-highgpu-8g", gpu_count=8, gpu_kind="H100-80")


def test_gpu_heuristics_covers_8gpu_sweep_intents() -> None:
    """#743 non-blocking polish: the RunPod-side gpu_heuristics.INTENTS map
    carries the two new 8-GPU sweep intents too, so an explicit RunPod
    fallback of a GCP-sized sweep (`pod.py provision --intent sweep-8g-*`)
    resolves rather than KeyError-ing. Cheap insurance against a typo in the
    GpuSpec rows."""
    from scripts import gpu_heuristics

    a100 = gpu_heuristics.INTENTS["sweep-8g-a100"]
    assert a100.gpu_type == "A100"
    assert a100.gpu_count == 8
    h100 = gpu_heuristics.INTENTS["sweep-8g-h100"]
    assert h100.gpu_type == "H100"
    assert h100.gpu_count == 8
    # resolve_intent (lower-cased lookup) resolves both without raising.
    assert gpu_heuristics.resolve_intent("sweep-8g-a100").gpu_count == 8
    assert gpu_heuristics.resolve_intent("sweep-8g-h100").gpu_count == 8


def test_render_create_argv_cpu_bigmem_golden() -> None:
    """#677: a cpu-bigmem create renders a valid CPU argv.

    The CPU machine takes ``--maintenance-policy=MIGRATE`` (it can
    live-migrate), carries NO ``--accelerator`` flag, and keeps every
    ephemeral leak guard the GPU path uses.
    """
    cfg = _test_config()
    argv = render_create_argv(
        spec=_spec("cpu-bigmem"),
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\necho startup\n",
        secret_files=_TEST_SECRET_FILES,
    )
    # CPU machine type.
    assert "--machine-type=n2-highmem-16" in argv
    # MIGRATE for the CPU machine; TERMINATE must be ABSENT.
    assert "--maintenance-policy=MIGRATE" in argv
    assert "--maintenance-policy=TERMINATE" not in argv
    # No accelerator flag on a CPU VM (absent for ALL GCP machines today —
    # this asserts it STAYS absent for the CPU path).
    assert not any(a.startswith("--accelerator") for a in argv), argv
    # Ephemeral / leak guards apply equally to a CPU VM.
    assert "--instance-termination-action=DELETE" in argv
    assert "--no-restart-on-failure" in argv
    assert "--max-run-duration=7d" in argv  # #741: config default raised 24h → 7d
    # Default boot disk covers a ~150 GB working set.
    assert "--boot-disk-size=300GB" in argv


def test_render_create_argv_cpu_bigmem_boot_disk_override() -> None:
    """#677: the existing --boot-disk-gb override threads through for CPU."""
    cfg = _test_config()
    argv = render_create_argv(
        spec=_spec("cpu-bigmem", extra={"boot_disk_gb": 500}),
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--boot-disk-size=500GB" in argv


def test_render_create_argv_gpu_intent_still_terminate() -> None:
    """#677 control: the conditional did NOT regress the GPU path —
    an accelerator intent still emits --maintenance-policy=TERMINATE."""
    cfg = _test_config()
    for intent in ("lora-7b", "ft-7b"):
        argv = render_create_argv(
            spec=_spec(intent),
            config=cfg,
            attempt_id="att-fixed-001",
            startup_script="#!/bin/bash\n",
            secret_files=_TEST_SECRET_FILES,
        )
        assert "--maintenance-policy=TERMINATE" in argv, intent
        assert "--maintenance-policy=MIGRATE" not in argv, intent


def test_quota_metric_for_cpu_returns_none() -> None:
    """#677: a CPU machine draws no accelerator quota under any pool."""
    cpu = MachineSpec(machine_type="n2-highmem-16", gpu_count=0, gpu_kind="CPU")
    for provisioning in ("STANDARD", "SPOT", "FLEX_START"):
        assert quota_metric_for(cpu, provisioning) is None, provisioning


def test_preflight_quota_headroom_skips_probe_for_cpu() -> None:
    """#677: a cpu-bigmem spec yields metric=None -> the regions-describe
    probe is skipped (fail-OPEN "no opinion; proceed"), no gcloud call."""
    runner = _Runner()
    assert (
        preflight_quota_headroom(
            spec=_spec(intent="cpu-bigmem"), config=_test_config(), runner=runner
        )
        is None
    )
    # The accelerator-quota probe never ran: no regions-describe gcloud call.
    assert not [a for a in runner.calls if "regions" in a and "describe" in a]
    assert runner.calls == []  # decided without ANY gcloud call


def test_gcp_handle_extra_includes_gpu_count(no_marker_posts) -> None:
    """#677: the GCP handle's extra carries the resolved machine's true
    gpu_count (0 for cpu-bigmem, >=1 for a GPU intent) so the async poller's
    failover predicate can exclude CPU handles without resolving the machine.

    cpu-bigmem and eval are both offered in all three us-central1 zones, so a
    single create succeeds for each.
    """
    created = json.dumps([{"name": "eps-issue-137", "id": "999"}])

    cpu_backend = GcpBackend(
        config=_test_config(),
        runner=_Runner(
            list_results=[GcloudRunResult(0, "[]", "")],
            create_results=[GcloudRunResult(0, created, "")],
        ),
        marker_poster=lambda **_: None,
    )
    cpu_handle = cpu_backend.launch(_spec(intent="cpu-bigmem"))
    assert cpu_handle.extra["gpu_count"] == 0

    gpu_backend = GcpBackend(
        config=_test_config(),
        runner=_Runner(
            list_results=[GcloudRunResult(0, "[]", "")],
            create_results=[GcloudRunResult(0, created, "")],
        ),
        marker_poster=lambda **_: None,
    )
    gpu_handle = gpu_backend.launch(_spec(intent="eval"))
    assert gpu_handle.extra["gpu_count"] == 1


# ---------------------------------------------------------------------------
# Machine-type zone availability (#653)
# ---------------------------------------------------------------------------


def test_zones_for_machine_type_drops_unavailable_zone() -> None:
    """#653: the A2-ultragpu family is NOT offered in us-central1-b, so the
    filter drops it while preserving order for the available zones."""
    from explore_persona_space.backends.gcp import (
        MACHINE_TYPE_ZONE_AVAILABILITY,
        zones_for_machine_type,
    )

    ladder = ["us-central1-a", "us-central1-b", "us-central1-c"]
    assert zones_for_machine_type("a2-ultragpu-1g", ladder) == [
        "us-central1-a",
        "us-central1-c",
    ]
    assert zones_for_machine_type("a2-ultragpu-4g", ladder) == [
        "us-central1-a",
        "us-central1-c",
    ]
    # The map RESTRICTS only — every A2-ultragpu row excludes -b.
    assert "us-central1-b" not in MACHINE_TYPE_ZONE_AVAILABILITY["a2-ultragpu-1g"]


def test_zones_for_machine_type_keeps_all_for_unfiltered_type() -> None:
    """#653: a machine type listed in every zone (g2) keeps the full ladder;
    a machine type UNLISTED in the map fails OPEN (no filtering)."""
    from explore_persona_space.backends.gcp import zones_for_machine_type

    ladder = ["us-central1-a", "us-central1-b", "us-central1-c"]
    # g2-standard-4 is in all three.
    assert zones_for_machine_type("g2-standard-4", ladder) == ladder
    # An unlisted machine type is not silently dropped from every zone.
    assert zones_for_machine_type("some-future-machine-type", ladder) == ladder


def test_a3_highgpu_family_available_in_all_us_central1_zones() -> None:
    """#653 round-8 follow-up: BOTH a3-highgpu sizes (1g = lora-7b-h100,
    2g = eval-h100) are offered in all three us-central1 zones, so they keep
    the full ladder — NOT a doomed-launch / fail-loud case. Pins the verified
    gcloud fact (2026-06-16) that refutes the false 'a3-highgpu-2g not offered
    in us-central1' report, and guards against the eval-h100 size being left
    implicitly absent from the table while its 1g sibling is explicit."""
    from explore_persona_space.backends.gcp import (
        MACHINE_TYPE_ZONE_AVAILABILITY,
        zones_for_machine_type,
    )

    ladder = ["us-central1-a", "us-central1-b", "us-central1-c"]
    for mt in ("a3-highgpu-1g", "a3-highgpu-2g"):
        assert mt in MACHINE_TYPE_ZONE_AVAILABILITY, mt
        assert zones_for_machine_type(mt, ladder) == ladder, mt


def test_8gpu_sweep_machine_types_zone_availability() -> None:
    """#743: the two new 8-GPU machine types carry the verified us-central1
    zone sets. a2-ultragpu-8g follows its A2-ultragpu family — offered in
    {a, c} only, NOT us-central1-b (so the zone-fallback ladder never issues
    a doomed -b create that burns a GCP attempt). a3-highgpu-8g follows its
    a3-highgpu family — all three zones."""
    from explore_persona_space.backends.gcp import (
        MACHINE_TYPE_ZONE_AVAILABILITY,
        zones_for_machine_type,
    )

    ladder = ["us-central1-a", "us-central1-b", "us-central1-c"]
    # a2-ultragpu-8g: {a, c}, NOT -b (matches the 1g/4g family rows).
    assert MACHINE_TYPE_ZONE_AVAILABILITY["a2-ultragpu-8g"] == frozenset(
        {"us-central1-a", "us-central1-c"}
    )
    assert "us-central1-b" not in MACHINE_TYPE_ZONE_AVAILABILITY["a2-ultragpu-8g"]
    assert zones_for_machine_type("a2-ultragpu-8g", ladder) == [
        "us-central1-a",
        "us-central1-c",
    ]
    # a3-highgpu-8g: all three zones (matches the 1g/2g family rows).
    assert MACHINE_TYPE_ZONE_AVAILABILITY["a3-highgpu-8g"] == frozenset(
        {"us-central1-a", "us-central1-b", "us-central1-c"}
    )
    assert zones_for_machine_type("a3-highgpu-8g", ladder) == ladder


def test_zone_availability_has_a2_ultragpu_2g_a_c_only() -> None:
    """#1121: the new a2-ultragpu-2g row (the width-2 auto-ladder rung)
    follows its A2-ultragpu family — offered in {a, c} only, NOT
    us-central1-b (live-verified 2026-07-08), so the zone-fallback ladder
    never issues a doomed -b create that burns a GCP attempt."""
    from explore_persona_space.backends.gcp import (
        MACHINE_TYPE_ZONE_AVAILABILITY,
        zones_for_machine_type,
    )

    assert MACHINE_TYPE_ZONE_AVAILABILITY["a2-ultragpu-2g"] == frozenset(
        {"us-central1-a", "us-central1-c"}
    )
    assert zones_for_machine_type(
        "a2-ultragpu-2g", ["us-central1-a", "us-central1-b", "us-central1-c"]
    ) == ["us-central1-a", "us-central1-c"]


def test_render_create_argv_resolves_wide_machine_override() -> None:
    """#1121: a router-threaded wide machine override (the JSON-safe dict
    shape ``_with_machine`` threads) renders the wide machine type under
    FLEX_START — the existing ``machine_spec_override`` chokepoint resolves
    wide rungs with zero create-path changes. H100 + STANDARD keeps raising
    (``test_render_create_argv_h100_standard_raises_loud`` pins that)."""
    cfg = _test_config()
    spec = _spec(
        intent="capture-7b",
        extra={
            "machine_spec_override": {
                "machine_type": "a2-ultragpu-8g",
                "gpu_count": 8,
                "gpu_kind": "A100-80",
            },
            "provisioning_model": "FLEX_START",
        },
    )
    argv = render_create_argv(
        spec=spec,
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--machine-type=a2-ultragpu-8g" in argv
    assert "--provisioning-model=FLEX_START" in argv


def test_default_fallback_zones_includes_us_central1_f() -> None:
    """#774: DEFAULT_FALLBACK_ZONES carries us-central1-f so the create-time
    ladder [primary, *fallback_zones] reaches -f for the A100-40 fallback
    rung. Pins the exact tuple (order preserved: b, c, f)."""
    from explore_persona_space.backends.gcp import DEFAULT_FALLBACK_ZONES

    assert DEFAULT_FALLBACK_ZONES == (
        "us-central1-b",
        "us-central1-c",
        "us-central1-f",
    )


def test_a2_highgpu_1g_fan_out_reaches_us_central1_f() -> None:
    """#774: with -f in DEFAULT_FALLBACK_ZONES, the A100-40 fallback rung
    (a2-highgpu-1g, offered in {a, b, c, f}) walks all four zones in ladder
    order before counting a capacity miss. This is the gap #774 closes — the
    rung previously only ever resolved to {a, b, c}."""
    from explore_persona_space.backends.gcp import (
        DEFAULT_FALLBACK_ZONES,
        DEFAULT_PRIMARY_ZONE,
        zones_for_machine_type,
    )

    ladder = [DEFAULT_PRIMARY_ZONE, *DEFAULT_FALLBACK_ZONES]
    assert zones_for_machine_type("a2-highgpu-1g", ladder) == [
        "us-central1-a",
        "us-central1-b",
        "us-central1-c",
        "us-central1-f",
    ]


def test_a2_ultragpu_family_excludes_f_from_full_default_ladder() -> None:
    """#774: the broader DEFAULT_FALLBACK_ZONES (now carrying -f) must NOT
    leak -f to the A2-ultragpu (A100-80) family, which GCP does NOT offer in
    -f (nor -b). The RESTRICT filter strips both back out — verified
    2026-06-30: a2-ultragpu-* is offered in {a, c} only."""
    from explore_persona_space.backends.gcp import (
        DEFAULT_FALLBACK_ZONES,
        DEFAULT_PRIMARY_ZONE,
        zones_for_machine_type,
    )

    ladder = [DEFAULT_PRIMARY_ZONE, *DEFAULT_FALLBACK_ZONES]
    for mt in ("a2-ultragpu-1g", "a2-ultragpu-4g", "a2-ultragpu-8g"):
        resolved = zones_for_machine_type(mt, ladder)
        assert "us-central1-f" not in resolved, mt
        assert resolved == ["us-central1-a", "us-central1-c"], mt


# ---------------------------------------------------------------------------
# #656 — A100-40 fallback rung (a2-highgpu-1g)
# ---------------------------------------------------------------------------


def test_a100_40_fallback_for_intent_fits_predicate() -> None:
    """T10: the fits-in-40GB predicate. Single-GPU 7B-scale intents (lora-7b /
    lora / capture-7b / eval / debug) map to the A100-40 (a2-highgpu-1g)
    fallback machine; multi-GPU full-FT (ft-7b) and the 70B / unknown intents
    return None (a 40 GB card cannot hold them, so the ladder has no A100-40
    rung)."""
    for intent in ("lora-7b", "lora", "capture-7b", "eval", "debug"):
        machine = a100_40_fallback_for_intent(_spec(intent))
        assert isinstance(machine, MachineSpec), intent
        assert machine.machine_type == "a2-highgpu-1g", intent
        assert machine.gpu_count == 1, intent
        assert machine.gpu_kind == "A100-40", intent
    for intent in ("ft-7b", "inf-70b", "ft-70b", "totally-bogus"):
        assert a100_40_fallback_for_intent(_spec(intent)) is None, intent
    # The module-level map matches the predicate's positive set exactly.
    assert set(INTENT_A100_40_FALLBACK) == {"lora-7b", "lora", "capture-7b", "eval", "debug"}


def test_machine_for_intent_honors_machine_spec_override() -> None:
    """T11: machine_for_intent consults spec.extra['machine_spec_override']
    FIRST — the seam the #656 ladder uses to thread an A100-40 rung without
    mutating the frozen RunSpec's intent. quota_metric_for on the override
    resolves the un-suffixed NVIDIA_A100_GPUS pool."""
    spec = _spec(
        "lora-7b",
        extra={
            "machine_spec_override": {
                "machine_type": "a2-highgpu-1g",
                "gpu_count": 1,
                "gpu_kind": "A100-40",
            }
        },
    )
    machine = machine_for_intent(spec)
    assert machine.machine_type == "a2-highgpu-1g"
    assert machine.gpu_kind == "A100-40"
    # On-demand quota metric for the A100-40 override is the un-suffixed pool.
    assert quota_metric_for(machine, "STANDARD") == "NVIDIA_A100_GPUS"
    assert quota_metric_for(machine, "SPOT") == "PREEMPTIBLE_NVIDIA_A100_GPUS"
    # A MachineSpec instance (not a dict) is also accepted.
    spec2 = _spec("lora-7b", extra={"machine_spec_override": machine})
    assert machine_for_intent(spec2).machine_type == "a2-highgpu-1g"


def test_a2_highgpu_zone_availability_and_create_argv() -> None:
    """T12: an A100-40 override spec renders --machine-type=a2-highgpu-1g, and
    the zone filter keeps the verified us-central1 zones (incl. -f)."""
    from explore_persona_space.backends.gcp import (
        MACHINE_TYPE_ZONE_AVAILABILITY,
        zones_for_machine_type,
    )

    spec = _spec(
        "lora-7b",
        extra={
            "machine_spec_override": {
                "machine_type": "a2-highgpu-1g",
                "gpu_count": 1,
                "gpu_kind": "A100-40",
            }
        },
    )
    argv = render_create_argv(
        spec=spec,
        config=_test_config(),
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\necho startup\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--machine-type=a2-highgpu-1g" in argv
    # The zone filter keeps the verified us-central1 zones for a2-highgpu-1g.
    assert "a2-highgpu-1g" in MACHINE_TYPE_ZONE_AVAILABILITY
    ladder = ["us-central1-a", "us-central1-b", "us-central1-c"]
    assert zones_for_machine_type("a2-highgpu-1g", ladder) == ladder
    # -f is also offered (additive; the create-time resolver gets one more option).
    assert "us-central1-f" in MACHINE_TYPE_ZONE_AVAILABILITY["a2-highgpu-1g"]


def test_a100_40_quota_metric_mapping() -> None:
    """The A100-40 gpu_kind resolves to the un-suffixed NVIDIA_A100_GPUS pool
    (on-demand) / PREEMPTIBLE_NVIDIA_A100_GPUS (spot) — a SEPARATE regional
    quota from the 80GB pool, the reason A100-40 can have headroom when
    A100-80 is full."""
    a40 = MachineSpec(machine_type="a2-highgpu-1g", gpu_count=1, gpu_kind="A100-40")
    assert quota_metric_for(a40, "STANDARD") == "NVIDIA_A100_GPUS"
    assert quota_metric_for(a40, "SPOT") == "PREEMPTIBLE_NVIDIA_A100_GPUS"
    assert quota_metric_for(a40, "FLEX_START") == "PREEMPTIBLE_NVIDIA_A100_GPUS"


# ---------------------------------------------------------------------------
# Provisioning model resolver
# ---------------------------------------------------------------------------


def test_resolve_provisioning_model_default_standard() -> None:
    spec = _spec()
    assert resolve_provisioning_model(spec) == "STANDARD"


def test_resolve_provisioning_model_explicit_spot() -> None:
    spec = _spec(extra={"provisioning_model": "spot"})
    assert resolve_provisioning_model(spec) == "SPOT"


def test_resolve_provisioning_model_rejects_typo() -> None:
    spec = _spec(extra={"provisioning_model": "preemptible"})
    with pytest.raises(ValueError, match="unknown provisioning_model"):
        resolve_provisioning_model(spec)


# ---------------------------------------------------------------------------
# _gcp_status_to_poll_result — PENDING is a live FLEX_START-queued state
# ---------------------------------------------------------------------------


def test_gcp_status_to_poll_result_pending_maps_to_running_not_stalled() -> None:
    """A FLEX_START-queued GCE instance (status PENDING) is a live,
    keep-polling state — NOT the false-stalled routing that the /issue
    Step 6d.2 poll pseudocode would convert to epm:failure + set-status
    blocked (#782 / live repro #778). Mirrors reconnect_or_none, which
    treats PENDING as live (not in _NONLIVE_INSTANCE_STATUSES)."""
    result = _gcp_status_to_poll_result("PENDING")
    assert result.status == "running"
    assert result.current_phase == "pending"

    # Case-insensitive on the raw GCE string (matches the up = status.upper()
    # normalization the other branches rely on).
    lower = _gcp_status_to_poll_result("pending")
    assert lower.status == "running"
    assert lower.current_phase == "pending"

    # Regression guard: it must NOT fall through to the unknown_* stalled
    # default (the pre-fix behavior that caused the false block).
    assert result.current_phase != "unknown_pending"
    assert result.status != "stalled"


# ---------------------------------------------------------------------------
# attempt_id_for
# ---------------------------------------------------------------------------


def test_attempt_id_uses_extra_when_present() -> None:
    spec = _spec(extra={"attempt_id": "router-abc123"})
    assert attempt_id_for(spec) == "router-abc123"


def test_attempt_id_rejects_shell_unsafe() -> None:
    spec = _spec(extra={"attempt_id": "abc;rm -rf /"})
    with pytest.raises(ValueError, match="attempt_id must match"):
        attempt_id_for(spec)


def test_attempt_id_fallback_is_timestamp_shaped() -> None:
    spec = RunSpec(issue=1, intent="lora-7b", backend="gcp")
    tag = attempt_id_for(spec)
    assert tag.startswith("att-")
    assert len(tag) > len("att-")


# ---------------------------------------------------------------------------
# render_create_argv — the golden assertion
# ---------------------------------------------------------------------------


def test_render_create_argv_lora_golden() -> None:
    """Argv shape for the canonical lora-7b spec.

    Pins every flag the plan calls out as load-bearing — the live
    acceptance run depends on each being present + correct.
    """
    cfg = _test_config()
    argv = render_create_argv(
        spec=_spec("lora-7b"),
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\necho startup\n",
        secret_files=_TEST_SECRET_FILES,
    )
    joined = " ".join(argv)
    # gcloud verb shape
    assert argv[0] == "gcloud"
    assert "compute" in argv
    assert "instances" in argv
    assert "create" in argv
    # Per-command project + configuration (NOT relying on env var)
    assert "--configuration=eps-test-config" in argv
    assert "--project=eps-test-project" in argv
    # Intent → machine type
    assert "--machine-type=a2-ultragpu-1g" in argv
    # On-demand acceptance default; spot is opt-in
    assert "--provisioning-model=STANDARD" in argv
    # Leak guards
    assert "--instance-termination-action=DELETE" in argv
    assert "--maintenance-policy=TERMINATE" in argv
    # max-run-duration default (config 7d, #741)
    assert "--max-run-duration=7d" in argv
    # DLVM image
    assert "--image-family=pytorch-test-family" in argv
    assert "--image-project=deeplearning-platform-release" in argv
    # Disk
    assert "--boot-disk-size=300GB" in argv
    assert "--boot-disk-type=pd-ssd" in argv
    # Broad in-VM auth scope
    assert "--scopes=cloud-platform" in argv
    # Zone defaults to primary
    assert "--zone=us-central1-a" in argv
    # Canonical instance name
    assert "eps-issue-137" in argv
    # Startup script is threaded through --metadata (no tempfile in test)
    assert any("startup-script=" in a for a in argv), argv
    # Labels carry the audit prefix
    assert any("managed-by=eps" in a for a in argv), argv
    assert any("eps-issue=137" in a for a in argv), argv
    # No shell-escape leak from the startup script body
    assert "rm -rf" not in joined
    # SECURITY (round-2, task #535): token VALUES never appear on the
    # argv — secrets ride --metadata-from-file as tempfile PATHS.
    assert "hf_test_token" not in joined
    assert "wandb_test_key" not in joined
    from_file_args = [a for a in argv if a.startswith("--metadata-from-file=")]
    assert len(from_file_args) == 1, argv
    assert "HF_TOKEN=/tmp/eps-test-secret-hf" in from_file_args[0]
    assert "WANDB_API_KEY=/tmp/eps-test-secret-wandb" in from_file_args[0]


def test_render_create_argv_ft_intent_uses_4gpu_machine() -> None:
    cfg = _test_config()
    argv = render_create_argv(
        spec=_spec("ft-7b"),
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--machine-type=a2-ultragpu-4g" in argv


def test_render_create_argv_spot_opt_in() -> None:
    cfg = _test_config()
    spec = _spec(extra={"provisioning_model": "spot"})
    argv = render_create_argv(
        spec=spec,
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--provisioning-model=SPOT" in argv
    # On-demand still rejected: regression guard.
    assert "--provisioning-model=STANDARD" not in argv


def test_render_create_argv_zone_override() -> None:
    cfg = _test_config()
    argv = render_create_argv(
        spec=_spec(),
        config=cfg,
        attempt_id="att-fixed-001",
        zone="us-central1-c",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--zone=us-central1-c" in argv
    assert "--zone=us-central1-a" not in argv


def test_render_create_argv_includes_persist_adapter_metadata(monkeypatch) -> None:
    """M2 regression: the adapter-persist passthrough vars set on the
    dispatch process env MUST land as instance metadata, or the in-VM
    ``trainer.py:_persist_adapter`` no-ops and the acceptance harness's
    check (a) false-FAILs after real compute was spent."""
    monkeypatch.setenv("EPM_PERSIST_ADAPTER_HF_REPO", "superkaiba1/explore-persona-space")
    monkeypatch.setenv("EPM_PERSIST_ADAPTER_SUBFOLDER", "router_acceptance/issue-137-gcp")
    cfg = _test_config()
    argv = render_create_argv(
        spec=_spec("lora-7b"),
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    metadata_args = [a for a in argv if a.startswith("--metadata=")]
    joined = " ".join(metadata_args)
    assert "EPM_PERSIST_ADAPTER_HF_REPO=superkaiba1/explore-persona-space" in joined
    assert "EPM_PERSIST_ADAPTER_SUBFOLDER=router_acceptance/issue-137-gcp" in joined


def test_render_create_argv_metadata_comma_value_uses_alternate_delimiter(monkeypatch) -> None:
    """gcloud splits ``--metadata`` on commas, so a forwarded value
    containing a comma would silently truncate every later pair. The
    renderer must switch to the alternate-delimiter syntax (``gcloud
    topic escaping``) so the full value survives as ONE pair."""
    monkeypatch.setenv("EPM_PERSIST_ADAPTER_HF_REPO", "superkaiba1/explore-persona-space")
    monkeypatch.setenv("EPM_PERSIST_ADAPTER_SUBFOLDER", "router_acceptance/issue-137,gcp")
    cfg = _test_config()
    argv = render_create_argv(
        spec=_spec("lora-7b"),
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    pair_args = [a for a in argv if a.startswith("--metadata=") and "startup-script" not in a]
    assert len(pair_args) == 1
    arg = pair_args[0]
    # Alternate-delimiter syntax engaged: --metadata=^<delim>^k=v<delim>k=v
    assert arg.startswith("--metadata=^"), arg
    delim = arg.split("^")[1]
    assert delim != ","
    pairs = arg[len(f"--metadata=^{delim}^") :].split(delim)
    assert "EPM_PERSIST_ADAPTER_SUBFOLDER=router_acceptance/issue-137,gcp" in pairs
    assert f"eps-issue={_spec().issue}" in pairs


def test_render_create_argv_metadata_comma_free_keeps_plain_join(monkeypatch) -> None:
    """Comma-free values keep the plain comma-join (the argv stays
    byte-stable for the common case)."""
    monkeypatch.setenv("EPM_PERSIST_ADAPTER_HF_REPO", "superkaiba1/explore-persona-space")
    monkeypatch.setenv("EPM_PERSIST_ADAPTER_SUBFOLDER", "router_acceptance/issue-137-gcp")
    cfg = _test_config()
    argv = render_create_argv(
        spec=_spec("lora-7b"),
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    pair_args = [a for a in argv if a.startswith("--metadata=") and "startup-script" not in a]
    assert len(pair_args) == 1
    assert not pair_args[0].startswith("--metadata=^")
    assert "EPM_PERSIST_ADAPTER_HF_REPO=superkaiba1/explore-persona-space" in pair_args[0]


def test_render_create_argv_omits_persist_adapter_metadata_when_unset(monkeypatch) -> None:
    """An unset passthrough var is dropped (same contract as the secret
    keys) -- no empty metadata pairs."""
    monkeypatch.delenv("EPM_PERSIST_ADAPTER_HF_REPO", raising=False)
    monkeypatch.delenv("EPM_PERSIST_ADAPTER_SUBFOLDER", raising=False)
    cfg = _test_config()
    argv = render_create_argv(
        spec=_spec("lora-7b"),
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    joined = " ".join(argv)
    assert "EPM_PERSIST_ADAPTER_HF_REPO" not in joined
    assert "EPM_PERSIST_ADAPTER_SUBFOLDER" not in joined


def test_render_create_argv_uses_metadata_from_file_when_provided() -> None:
    """When the caller threads a tempfile path through spec.extra, the
    renderer uses ``--metadata-from-file`` (avoids the 256KB metadata cap
    + keeps secrets-bearing scripts out of gcloud's stdout)."""
    cfg = _test_config()
    spec = _spec(extra={"startup_script_path": "/tmp/eps-startup.sh"})
    argv = render_create_argv(
        spec=spec,
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    # ONE combined --metadata-from-file flag carries the secrets AND the
    # startup-script (gcloud dict-type flags don't merge when repeated).
    from_file_args = [a for a in argv if a.startswith("--metadata-from-file=")]
    assert len(from_file_args) == 1, argv
    assert "startup-script=/tmp/eps-startup.sh" in from_file_args[0]
    assert "HF_TOKEN=/tmp/eps-test-secret-hf" in from_file_args[0]
    # And the inline form is NOT also emitted (avoids double-startup).
    assert not any(a.startswith("--metadata=startup-script=") for a in argv)


# ---------------------------------------------------------------------------
# Startup-script renderer
# ---------------------------------------------------------------------------


def test_render_startup_script_pulls_secrets_from_metadata() -> None:
    cfg = _test_config()
    script = render_startup_script(
        spec=_spec(),
        config=cfg,
        attempt_id="att-fixed-001",
    )
    # Every secret key has a metadata-fetch stanza
    for key in ("HF_TOKEN", "WANDB_API_KEY", "ANTHROPIC_API_KEY"):
        assert key in script
    # Uses the GCE-required metadata header
    assert "Metadata-Flavor: Google" in script
    # Clones the repo + runs uv sync
    assert "git clone" in script
    assert "uv sync --frozen" in script
    # Writes the per-attempt sentinel under eval_results/issue_<N>/<attempt>/
    assert "eval_results/issue_137/att-fixed-001/" in script
    assert '"phase":"done"' in script
    assert '"issue":137' in script
    # Hydra args were threaded through to the train invocation
    assert "condition=c1_evil_wrong_em" in script
    assert "seed=42" in script
    # strict-mode + umask
    assert "set -euo pipefail" in script
    assert "umask 077" in script


def test_render_startup_script_fetches_persist_adapter_passthrough() -> None:
    """M2 regression: the startup script must fetch + export the
    adapter-persist passthrough keys from instance metadata so the
    workload sees them in ``os.environ`` on the VM."""
    cfg = _test_config()
    script = render_startup_script(
        spec=_spec(),
        config=cfg,
        attempt_id="att-fixed-001",
    )
    for key in ("EPM_PERSIST_ADAPTER_HF_REPO", "EPM_PERSIST_ADAPTER_SUBFOLDER"):
        assert f"instance/attributes/{key}" in script, f"{key} fetch stanza missing"
        assert f"export {key}" in script, f"{key} export missing"


_HF_STORAGE_KNOB_KEYS = (
    "EPM_HF_STORAGE_SOFT_CEILING_TB",
    "EPM_HF_OVERFLOW_ROUTING",
    "EPM_HF_STORAGE_CHECK",
    "EPM_HF_STORAGE_CACHE_TTL_S",
)


def test_startup_passthrough_env_keys_include_hf_storage_knobs() -> None:
    """#564 (test 21d): the HF-storage soft-ceiling / overflow-routing knobs
    must reach the VM workload via instance metadata, or a dispatch-process
    opt-in silently no-ops remotely (the #535-r7 trap). The VM-local cache
    path + event-sink path are deliberately NOT threaded (wrong machine)."""
    from explore_persona_space.backends.gcp import STARTUP_PASSTHROUGH_ENV_KEYS

    for key in _HF_STORAGE_KNOB_KEYS:
        assert key in STARTUP_PASSTHROUGH_ENV_KEYS, key
    assert "EPM_HF_STORAGE_CACHE_PATH" not in STARTUP_PASSTHROUGH_ENV_KEYS
    assert "EPM_HF_OVERFLOW_EVENT_PATH" not in STARTUP_PASSTHROUGH_ENV_KEYS


def test_render_create_argv_includes_hf_storage_knob_metadata(monkeypatch) -> None:
    """#564 (test 21d): storage knobs set on the dispatch env land as
    instance metadata pairs."""
    monkeypatch.setenv("EPM_HF_STORAGE_SOFT_CEILING_TB", "10.0")
    monkeypatch.setenv("EPM_HF_OVERFLOW_ROUTING", "1")
    monkeypatch.setenv("EPM_HF_STORAGE_CHECK", "0")
    monkeypatch.setenv("EPM_HF_STORAGE_CACHE_TTL_S", "3600")
    cfg = _test_config()
    argv = render_create_argv(
        spec=_spec("lora-7b"),
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    joined = " ".join(a for a in argv if a.startswith("--metadata="))
    assert "EPM_HF_STORAGE_SOFT_CEILING_TB=10.0" in joined
    assert "EPM_HF_OVERFLOW_ROUTING=1" in joined
    assert "EPM_HF_STORAGE_CHECK=0" in joined
    assert "EPM_HF_STORAGE_CACHE_TTL_S=3600" in joined


def test_render_startup_script_fetches_hf_storage_knobs() -> None:
    """#564 (test 21d): the startup script fetches + exports the storage
    knobs from instance metadata so the VM workload sees them in os.environ."""
    cfg = _test_config()
    script = render_startup_script(
        spec=_spec(),
        config=cfg,
        attempt_id="att-fixed-001",
    )
    for key in _HF_STORAGE_KNOB_KEYS:
        assert f"instance/attributes/{key}" in script, f"{key} fetch stanza missing"
        assert f"export {key}" in script, f"{key} export missing"


def test_render_startup_script_shell_safe_hydra_args() -> None:
    """A Hydra arg with a shell-meaningful char must be quoted, not interpolated."""
    cfg = _test_config()
    spec = _spec(hydra_args=("condition=c1", "evil='$(rm -rf /tmp)'"))
    # Need to override attempt id since the spec helper resets extra
    spec = replace(spec, extra={"attempt_id": "att-fixed-001"})
    script = render_startup_script(spec=spec, config=cfg, attempt_id="att-fixed-001")
    # The shell expansion must NOT appear unquoted
    assert "rm -rf" in script  # literal text is there
    # but it lives inside shlex.quote-wrapped argv to the python call:
    # the dangerous backtick / $() expansion is dead inside single quotes
    lines = [line for line in script.splitlines() if "scripts/train.py" in line]
    assert lines, "train.py invocation missing"
    # The presence of the single-quoted wrapper around the malicious arg
    # is what shlex.quote produces; assert the canonical wrapping.
    assert "'evil=" in "\n".join(lines)


# ---------------------------------------------------------------------------
# Sentinel path
# ---------------------------------------------------------------------------


def test_sentinel_path_namespaces_per_attempt() -> None:
    cfg = _test_config()
    p1 = sentinel_path_for(cfg, 137, "att-A")
    p2 = sentinel_path_for(cfg, 137, "att-B")
    assert p1 != p2
    assert "att-A" in p1
    assert "att-B" in p2
    # Lives under workload root + eval_results/issue_137/<attempt>/
    assert "/workspace/eps-issue-137/eval_results/issue_137/" in p1
    assert p1.endswith(".completion-sentinel.json")


# ---------------------------------------------------------------------------
# Idempotent reconnect
# ---------------------------------------------------------------------------


def test_reconnect_returns_none_when_no_instance() -> None:
    runner = _Runner(list_results=[GcloudRunResult(0, "[]", "")])
    handle = reconnect_or_none(spec=_spec(), config=_test_config(), runner=runner)
    assert handle is None
    # Reconnect issued exactly one gcloud list call.
    assert len(runner.calls) == 1
    assert "list" in runner.calls[0]


def test_reconnect_returns_handle_when_instance_running() -> None:
    payload = json.dumps(
        [
            {
                "name": "eps-issue-137",
                "id": "9988776655",
                "status": "RUNNING",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
            }
        ]
    )
    runner = _Runner(list_results=[GcloudRunResult(0, payload, "")])
    handle = reconnect_or_none(spec=_spec(), config=_test_config(), runner=runner)
    assert handle is not None
    assert handle.backend == "gcp"
    assert handle.pod_name == "eps-issue-137"
    assert handle.job_id == "9988776655"
    assert handle.extra["zone"] == "us-central1-a"
    assert handle.extra["reconnected"] is True


def _running_instance_payload(name: str = "eps-issue-137", instance_id: str = "9988776655") -> str:
    """A one-instance RUNNING gcloud list payload (the reconnect probe shape)."""
    return json.dumps(
        [
            {
                "name": name,
                "id": instance_id,
                "status": "RUNNING",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
            }
        ]
    )


def test_reconnect_handle_carries_workload_extras_from_spec() -> None:
    """T1 (#1122): a workload-cmd spec's reconnect handle mirrors the launch
    path's failover-prerequisite extras — workload_cmd verbatim (str, MF1),
    hydra_args [] (list), gpus, time_budget_hours, repo_branch, gpu_count,
    and boot_disk_gb when set — so an exit-75 rerun's sidecar overwrite
    stays failover-capable (incident #1090)."""
    spec = _spec(
        workload_cmd="REPO_ROOT=/workspace bash scripts/issue1090_dispatch.sh --full",
        hydra_args=(),  # _spec() defaults to non-empty hydra_args; the pair is exclusive
        gpus=1,
        time_budget_hours=4.0,
        extra={"repo_branch": "issue-1090", "boot_disk_gb": 200},
    )
    runner = _Runner(list_results=[GcloudRunResult(0, _running_instance_payload(), "")])
    handle = reconnect_or_none(spec=spec, config=_test_config(), runner=runner)
    assert handle is not None
    assert (
        handle.extra["workload_cmd"]
        == "REPO_ROOT=/workspace bash scripts/issue1090_dispatch.sh --full"
    )
    assert handle.extra["hydra_args"] == []
    assert isinstance(handle.extra["hydra_args"], list)
    assert handle.extra["gpus"] == 1
    assert handle.extra["time_budget_hours"] == 4.0
    assert handle.extra["repo_branch"] == "issue-1090"
    # lora-7b resolves to a 1-GPU machine; the poller's CPU guards read
    # 0-vs-nonzero only.
    assert handle.extra["gpu_count"] == 1
    assert handle.extra["boot_disk_gb"] == 200
    # min_ram_gb unset on the spec -> key OMITTED (legacy-shape parity).
    assert "min_ram_gb" not in handle.extra


def test_reconnect_handle_hydra_branch_mirrors_launch_shape() -> None:
    """T2 (#1122): the hydra-args branch mirrors the launch shape — empty
    workload_cmd str + hydra list; RunSpec mutual exclusion holds on the
    reconstructed pair."""
    spec = _spec()  # default hydra_args=("condition=c1_evil_wrong_em", "seed=42")
    runner = _Runner(list_results=[GcloudRunResult(0, _running_instance_payload(), "")])
    handle = reconnect_or_none(spec=spec, config=_test_config(), runner=runner)
    assert handle is not None
    assert handle.extra["workload_cmd"] == ""
    assert handle.extra["hydra_args"] == ["condition=c1_evil_wrong_em", "seed=42"]
    assert isinstance(handle.extra["hydra_args"], list)
    # One of the pair is empty by construction (MF2).
    assert not (handle.extra["workload_cmd"] and handle.extra["hydra_args"])
    # repo_branch unset on the spec -> written as "" (launch-path parity).
    assert handle.extra["repo_branch"] == ""


def test_reconnect_bare_spec_keeps_legacy_extra_shape() -> None:
    """T3 (#1122): a bare (no-workload) spec — provision-only / probe
    reconnects — keeps the LEGACY extra shape: no workload keys added (the
    issue_dispatch carry-forward preserves the prior sidecar's values for
    that case instead)."""
    spec = _spec(hydra_args=())  # _spec() defaults to non-empty hydra_args
    runner = _Runner(list_results=[GcloudRunResult(0, _running_instance_payload(), "")])
    handle = reconnect_or_none(spec=spec, config=_test_config(), runner=runner)
    assert handle is not None
    for key in (
        "workload_cmd",
        "hydra_args",
        "gpus",
        "time_budget_hours",
        "repo_branch",
        "gpu_count",
        "boot_disk_gb",
        "min_ram_gb",
    ):
        assert key not in handle.extra, key
    # The legacy probe-derived keys are still present.
    assert handle.extra["reconnected"] is True
    assert handle.extra["status_at_reconnect"] == "RUNNING"
    assert handle.extra["instance_name"] == "eps-issue-137"


def test_reconnect_skips_terminated_instance() -> None:
    payload = json.dumps(
        [
            {
                "name": "eps-issue-137",
                "id": "1",
                "status": "TERMINATED",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
            }
        ]
    )
    runner = _Runner(list_results=[GcloudRunResult(0, payload, "")])
    assert reconnect_or_none(spec=_spec(), config=_test_config(), runner=runner) is None


def test_reconnect_probe_failure_raises_not_none() -> None:
    """rc != 0 = the PROBE failed (expired auth / transport) — instance
    state is UNKNOWN and must NOT read as "no live instance" on the
    credit-spending lane (round-6 B1 mirrored from SLURM; live GCP
    attempt 1 hit exactly this with an expired-auth gcloud list)."""
    from explore_persona_space.backends.gcp import GcpProbeError

    runner = _Runner(list_results=[GcloudRunResult(1, "", "Reauthentication failed")])
    with pytest.raises(GcpProbeError):
        reconnect_or_none(spec=_spec(), config=_test_config(), runner=runner)


def test_reconnect_bad_json_raises_probe_error() -> None:
    """An rc=0 list whose stdout is unparseable is equally UNKNOWN state."""
    from explore_persona_space.backends.gcp import GcpProbeError

    runner = _Runner(list_results=[GcloudRunResult(0, "{not json", "")])
    with pytest.raises(GcpProbeError):
        reconnect_or_none(spec=_spec(), config=_test_config(), runner=runner)


def test_gcp_probe_error_is_backend_probe_error() -> None:
    """The router's reconnect seams discriminate on BackendProbeError —
    the GCP probe error must be a subclass or the typed handling is
    silently bypassed (the original bug shape)."""
    from explore_persona_space.backends.base import BackendProbeError
    from explore_persona_space.backends.gcp import GcpProbeError

    assert issubclass(GcpProbeError, BackendProbeError)


# ---------------------------------------------------------------------------
# Reconnect refusal of RUNNING + terminal-phase zombies (#908/#763)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phase", ["done", "failed", "finalize_failed_artifacts_ok", "wedged"])
def test_reconnect_refuses_running_instance_with_terminal_guest_phase(phase: str) -> None:
    """A RUNNING instance whose workload already published a terminal/wedged
    eps/phase is a gate-park/finished zombie, NOT a live run to rejoin —
    reconnecting to it silently no-ops the new dispatch (#763 leg 2: the
    phase-C launch "reconnected" to the gate-parked done VM and never ran)."""
    runner = _Runner(
        list_results=[GcloudRunResult(0, _instance_payload("RUNNING"), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload(phase), "")],
    )
    assert reconnect_or_none(spec=_spec(), config=_test_config(), runner=runner) is None


def test_reconnect_returns_handle_when_running_with_nonterminal_phase() -> None:
    """A live mid-run instance (phase=workload) keeps the idempotent
    reconnect path byte-for-byte (#908 must not break healthy re-entry)."""
    runner = _Runner(
        list_results=[GcloudRunResult(0, _instance_payload("RUNNING"), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
    )
    handle = reconnect_or_none(spec=_spec(), config=_test_config(), runner=runner)
    assert handle is not None
    assert handle.extra["reconnected"] is True


def test_reconnect_returns_handle_when_phase_unwritten() -> None:
    """gcloud 404/not-found = attribute not written yet (early boot) —
    reconnect normally (`_read_guest_phase` returns "")."""
    runner = _Runner(
        list_results=[GcloudRunResult(0, _instance_payload("RUNNING"), "")],
        guest_attr_results=[GcloudRunResult(1, "", "guest attribute eps/phase not found")],
    )
    handle = reconnect_or_none(spec=_spec(), config=_test_config(), runner=runner)
    assert handle is not None
    assert handle.extra["reconnected"] is True


def test_reconnect_phase_probe_failure_raises_probe_error() -> None:
    """A guest-attribute probe FAILURE (non-404 rc != 0) is UNKNOWN state:
    it must read as NEITHER "live, reconnect" (resurrects the #763 silent
    no-op) NOR "zombie, delete" (could reclaim a healthy VM) — the #535
    "couldn't ask" discipline; the launch fails typed + retriable."""
    from explore_persona_space.backends.gcp import GcpProbeError

    runner = _Runner(
        list_results=[GcloudRunResult(0, _instance_payload("RUNNING"), "")],
        guest_attr_results=[GcloudRunResult(1, "", "Reauthentication failed")],
    )
    with pytest.raises(GcpProbeError):
        reconnect_or_none(spec=_spec(), config=_test_config(), runner=runner)


@pytest.mark.parametrize("status", ["PROVISIONING", "STOPPING"])
def test_reconnect_probes_phase_only_for_running_status(status: str) -> None:
    """The phase gate is RUNNING-only (mirrors the janitor's
    should_probe_phase scoping): PROVISIONING/STAGING have no phase yet and
    STOPPING is the seconds-long teardown transition — both reconnect as
    today with NO guest-attribute probe issued. Mis-scoping would silently
    widen the R5 dead-end via the fake's 404 default."""
    runner = _Runner(list_results=[GcloudRunResult(0, _instance_payload(status), "")])
    handle = reconnect_or_none(spec=_spec(), config=_test_config(), runner=runner)
    assert handle is not None
    assert not [a for a in runner.calls if "get-guest-attributes" in a], runner.calls


def test_launch_skips_create_when_reconnect_finds_live_instance(no_marker_posts) -> None:
    """Regression guard for the idempotency contract: a re-launch on
    a still-live instance must NOT double-provision."""
    payload = json.dumps(
        [
            {
                "name": "eps-issue-137",
                "id": "9988",
                "status": "RUNNING",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
            }
        ]
    )
    runner = _Runner(list_results=[GcloudRunResult(0, payload, "")])
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    handle = backend.launch(_spec())
    assert handle.pod_name == "eps-issue-137"
    # ONLY a list call — NO create call.
    assert all("create" not in argv for argv in runner.calls), runner.calls
    # Reconnected handle still carries the ExpectedArtifacts declaration
    assert EXPECTED_ARTIFACTS_HANDLE_KEY in handle.extra


# ---------------------------------------------------------------------------
# launch — happy path + ExpectedArtifacts declaration
# ---------------------------------------------------------------------------


def test_launch_populates_expected_artifacts_with_sentinel(no_marker_posts) -> None:
    """The slice-2 verifier FAILs an all-SKIP declaration; the launch
    path MUST populate the sentinel path so confirm_artifacts has a
    keystone check to run."""
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "112233"}])
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],  # no existing instance
        create_results=[GcloudRunResult(0, created_payload, "")],
    )
    posted: list[dict] = []
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **kwargs: posted.append(kwargs),
    )
    handle = backend.launch(_spec())

    assert handle.backend == "gcp"
    assert handle.pod_name == "eps-issue-137"
    assert handle.job_id == "112233"
    assert handle.extra["attempt_id"] == "att-fixed-001"
    assert handle.extra["machine_type"] == "a2-ultragpu-1g"

    # ExpectedArtifacts declaration MUST be on handle.extra
    decl = handle.extra.get(EXPECTED_ARTIFACTS_HANDLE_KEY)
    assert isinstance(decl, dict), decl
    assert decl["issue"] == 137
    assert decl["sentinel_path"].endswith(".completion-sentinel.json")
    assert "att-fixed-001" in decl["sentinel_path"]
    # #790: this is a pure-hydra launch (_spec has no workload_cmd), so it
    # declares NEITHER default git path — train.py runs with skip_eval=True and
    # writes no figures during the run, so both were guaranteed false-FAILs.
    assert decl["git_paths"] == []
    # Default HF data path threads the attempt id
    assert any("issue137_att-fixed-001/raw_completions/" in p for p in decl["hf_data_paths"]), decl

    # epm:cluster-launched v1 marker posted exactly once
    assert len(posted) == 1
    assert posted[0]["marker"] == "epm:cluster-launched"
    assert posted[0]["issue"] == 137
    body = json.loads(posted[0]["note"])
    assert body["backend"] == "gcp"
    assert body["machine_type"] == "a2-ultragpu-1g"
    assert body["attempt_id"] == "att-fixed-001"


def test_expected_artifacts_declaration_workload_cmd_omits_guessed_hf_prefix() -> None:
    """#601 follow-up r1: the workload_cmd lane must NOT auto-declare the
    launch-time GUESS ``issue<N>_<attempt>/raw_completions/`` — custom
    dispatch drivers upload to their own contract prefix
    (``issue<N>_<slug>/...``), so the guess produced a false-negative
    ``confirm_artifacts`` FAIL (exit 3, teardown skipped) on a
    perfectly-uploaded run. An undeclared ``hf_data_paths`` SKIPs the
    hf_data check; the sentinel + git paths keep gating teardown."""
    decl = expected_artifacts_declaration(
        spec=_workload_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    assert decl["hf_data_paths"] == []
    # The keystone sentinel + the eval_results/ git path still gate.
    assert decl["sentinel_path"].endswith(".completion-sentinel.json")
    # #790: custom_workload keeps eval_results/ (drivers commit it during the
    # run) but drops the analyzer-generated figures/.
    assert decl["git_paths"] == ["eval_results/issue_137/"]
    # An EXPLICIT caller declaration still threads through.
    decl_explicit = expected_artifacts_declaration(
        spec=_workload_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
        extra_hf_data_paths=("issue137_neg_setpoint/raw_completions/",),
    )
    assert decl_explicit["hf_data_paths"] == ["issue137_neg_setpoint/raw_completions/"]
    # The hydra lane keeps the per-attempt default (pinned above by
    # test_launch_populates_expected_artifacts_with_sentinel too).
    decl_hydra = expected_artifacts_declaration(
        spec=_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    assert decl_hydra["hf_data_paths"] == ["issue137_att-fixed-001/raw_completions/"]


# ---------------------------------------------------------------------------
# Pre-launch stale-name reclaim (#632)
# ---------------------------------------------------------------------------


def _instance_payload(status: str, *, name: str = "eps-issue-137", id_: str = "1") -> str:
    """A one-element ``gcloud compute instances list`` JSON payload."""
    return json.dumps(
        [
            {
                "name": name,
                "id": id_,
                "status": status,
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-b"
                ),
            }
        ]
    )


@pytest.mark.parametrize("status", ["TERMINATED", "STOPPED", "SUSPENDED"])
def test_stale_named_instance_returns_record_for_nonlive_status(status: str) -> None:
    """The exact set ``reconnect_or_none`` treats as not-live is the set the
    pre-launch check must reclaim — else the create collides on the stale
    name (incident #632)."""
    runner = _Runner(list_results=[GcloudRunResult(0, _instance_payload(status), "")])
    stale = _stale_named_instance_or_none(spec=_spec(), config=_test_config(), runner=runner)
    assert isinstance(stale, StaleNamedInstance)
    assert stale.name == "eps-issue-137"
    assert stale.status == status
    # Zone is the parsed last-segment of the instance's zone URL so the
    # delete targets the right zone (NOT the config primary).
    assert stale.zone == "us-central1-b"


def test_stale_named_instance_returns_none_when_no_record() -> None:
    runner = _Runner(list_results=[GcloudRunResult(0, "[]", "")])
    assert _stale_named_instance_or_none(spec=_spec(), config=_test_config(), runner=runner) is None


def test_stale_named_instance_probe_rc_failure_raises_probe_error() -> None:
    """rc != 0 = probe failed; "couldn't ask" must NOT read as "name free"
    on the credit-spending lane (mirrors reconnect_or_none)."""
    from explore_persona_space.backends.gcp import GcpProbeError

    runner = _Runner(list_results=[GcloudRunResult(1, "", "Reauthentication failed")])
    with pytest.raises(GcpProbeError):
        _stale_named_instance_or_none(spec=_spec(), config=_test_config(), runner=runner)


def test_stale_named_instance_bad_json_raises_probe_error() -> None:
    from explore_persona_space.backends.gcp import GcpProbeError

    runner = _Runner(list_results=[GcloudRunResult(0, "{not json", "")])
    with pytest.raises(GcpProbeError):
        _stale_named_instance_or_none(spec=_spec(), config=_test_config(), runner=runner)


def test_stale_named_instance_refuses_to_delete_live_status() -> None:
    """A non-deletable status (only reachable as a TOCTOU race vs the
    reconnect probe) must FAIL loudly — never auto-delete a possibly-live
    VM (data loss). The success criterion's data-loss guard. (Post-#908
    the RUNNING path first re-probes the phase; the fake's guest-attr 404
    default reads "" = non-terminal, so the raise is preserved.)"""
    from explore_persona_space.backends.gcp import GcpBackendError

    runner = _Runner(list_results=[GcloudRunResult(0, _instance_payload("RUNNING"), "")])
    with pytest.raises(GcpBackendError, match="non-deletable status"):
        _stale_named_instance_or_none(spec=_spec(), config=_test_config(), runner=runner)


# ---------------------------------------------------------------------------
# Pre-launch reclaim of RUNNING + terminal-phase zombies (#908/#763)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phase", ["done", "failed", "finalize_failed_artifacts_ok", "wedged"])
def test_stale_named_instance_returns_record_for_running_terminal_phase(phase: str) -> None:
    """The #908 matched delete: the SAME RUNNING+terminal-phase record that
    reconnect refuses must be deletable here, or the refusal dead-ends in
    the #632 "already exists" create collision."""
    runner = _Runner(
        list_results=[GcloudRunResult(0, _instance_payload("RUNNING"), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload(phase), "")],
    )
    stale = _stale_named_instance_or_none(spec=_spec(), config=_test_config(), runner=runner)
    assert isinstance(stale, StaleNamedInstance)
    assert stale.name == "eps-issue-137"
    assert stale.status == "RUNNING"
    assert stale.guest_phase == phase
    assert stale.zone == "us-central1-b"


def test_stale_named_instance_still_refuses_running_nonterminal_phase() -> None:
    """A RUNNING record whose re-probed phase is NON-terminal keeps the loud
    TOCTOU raise — never auto-delete a possibly-live instance (#908
    preserves the existing semantics for every non-zombie RUNNING record)."""
    from explore_persona_space.backends.gcp import GcpBackendError

    runner = _Runner(
        list_results=[GcloudRunResult(0, _instance_payload("RUNNING"), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
    )
    with pytest.raises(GcpBackendError, match="non-deletable status"):
        _stale_named_instance_or_none(spec=_spec(), config=_test_config(), runner=runner)


def test_stale_named_instance_phase_probe_failure_raises_probe_error() -> None:
    """A phase-probe failure on a RUNNING record is UNKNOWN state: no
    StaleNamedInstance (no delete on unknown state), no GcpBackendError —
    a typed, retriable GcpProbeError (#535 discipline)."""
    from explore_persona_space.backends.gcp import GcpProbeError

    runner = _Runner(
        list_results=[GcloudRunResult(0, _instance_payload("RUNNING"), "")],
        guest_attr_results=[GcloudRunResult(1, "", "Reauthentication failed")],
    )
    with pytest.raises(GcpProbeError):
        _stale_named_instance_or_none(spec=_spec(), config=_test_config(), runner=runner)


@pytest.mark.parametrize("phase", sorted(_ZOMBIE_GUEST_PHASES))
def test_reconnect_skip_set_matches_prelaunch_delete_set_for_phase_zombie(phase: str) -> None:
    """THE consistency pin, as a BEHAVIORAL SWEEP over the shared constant:
    for EVERY member of _ZOMBIE_GUEST_PHASES, identical mocked project
    state (one RUNNING record + that phase) must make reconnect SKIP
    (None) AND the stale probe return a DELETABLE record. A one-sided
    membership edit diverges loudly, and a future membership change
    auto-extends coverage (the #632 skip/delete identical-sets invariant)."""

    def _fresh_runner() -> _Runner:
        return _Runner(
            list_results=[GcloudRunResult(0, _instance_payload("RUNNING"), "")],
            guest_attr_results=[GcloudRunResult(0, _guest_attr_payload(phase), "")],
        )

    assert reconnect_or_none(spec=_spec(), config=_test_config(), runner=_fresh_runner()) is None
    stale = _stale_named_instance_or_none(
        spec=_spec(), config=_test_config(), runner=_fresh_runner()
    )
    assert isinstance(stale, StaleNamedInstance)
    assert stale.guest_phase == phase


def test_launch_deletes_terminal_phase_zombie_then_creates_fresh(no_marker_posts) -> None:
    """End-to-end #908/#763 leg 2: a launch against a RUNNING+done zombie
    deletes it then creates fresh — never a silent no-op reconnect
    (pre-#908, this exact state returned reason=reconnect and the new
    workload never ran)."""
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "445566"}])
    runner = _Runner(
        # 1st list = reconnect probe; 2nd list = stale-name probe.
        list_results=[
            GcloudRunResult(0, _instance_payload("RUNNING"), ""),
            GcloudRunResult(0, _instance_payload("RUNNING"), ""),
        ],
        # 1st guest-attr = reconnect's phase gate; 2nd = the stale probe's
        # local RE-probe (the phase is never carried over between them).
        guest_attr_results=[
            GcloudRunResult(0, _guest_attr_payload("done"), ""),
            GcloudRunResult(0, _guest_attr_payload("done"), ""),
        ],
        delete_results=[GcloudRunResult(0, "", "")],
        create_results=[GcloudRunResult(0, created_payload, "")],
    )
    posted: list[dict] = []
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **kwargs: posted.append(kwargs),
    )
    handle = backend.launch(_spec())
    assert handle.pod_name == "eps-issue-137"
    assert handle.job_id == "445566"

    # Delete fired, in the zombie record's zone, BEFORE the create.
    delete_calls = [c for c in runner.calls if "delete" in c and "instances" in c]
    create_calls = [c for c in runner.calls if "create" in c and "instances" in c]
    assert len(delete_calls) == 1, runner.calls
    assert "eps-issue-137" in delete_calls[0]
    assert "--zone=us-central1-b" in delete_calls[0]
    assert len(create_calls) == 1, runner.calls
    assert runner.calls.index(delete_calls[0]) < runner.calls.index(create_calls[0])

    # The flow exercised the LIVE render_guest_attributes_argv path, not a
    # stub: recorded probe calls carry the real argv shape.
    ga_calls = [
        c for c in runner.calls if "get-guest-attributes" in c and "--query-path=eps/phase" in c
    ]
    assert len(ga_calls) == 2, runner.calls

    # Marker flags the reclaim (same field as the #632 status-stale path).
    assert len(posted) == 1
    body = json.loads(posted[0]["note"])
    assert body["pre_launch_deleted_stale_instance"] is True


def test_launch_deletes_stale_terminated_instance_then_creates(no_marker_posts) -> None:
    """End-to-end #632 fix: a prior TERMINATED record blocks the name, so
    launch deletes it BEFORE create and the create then succeeds. The
    epm:cluster-launched marker flags the reclaim."""
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "778899"}])
    runner = _Runner(
        # 1st list = reconnect probe (no LIVE instance → falls through);
        # 2nd list = stale-name probe (finds the TERMINATED record).
        list_results=[
            GcloudRunResult(0, "[]", ""),
            GcloudRunResult(0, _instance_payload("TERMINATED"), ""),
        ],
        delete_results=[GcloudRunResult(0, "", "")],
        create_results=[GcloudRunResult(0, created_payload, "")],
    )
    posted: list[dict] = []
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **kwargs: posted.append(kwargs),
    )
    handle = backend.launch(_spec())
    assert handle.pod_name == "eps-issue-137"
    assert handle.job_id == "778899"

    # A delete fired, in the stale record's zone, BEFORE the create.
    delete_calls = [c for c in runner.calls if "delete" in c and "instances" in c]
    create_calls = [c for c in runner.calls if "create" in c and "instances" in c]
    assert len(delete_calls) == 1, runner.calls
    assert "eps-issue-137" in delete_calls[0]
    assert "--zone=us-central1-b" in delete_calls[0]
    assert len(create_calls) == 1, runner.calls
    assert runner.calls.index(delete_calls[0]) < runner.calls.index(create_calls[0])

    # Marker flags the reclaim.
    assert len(posted) == 1
    body = json.loads(posted[0]["note"])
    assert body["pre_launch_deleted_stale_instance"] is True


def test_launch_marks_no_stale_delete_when_name_free(no_marker_posts) -> None:
    """The common path: no prior record → no delete, marker flag False."""
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "112233"}])
    runner = _Runner(
        list_results=[
            GcloudRunResult(0, "[]", ""),  # reconnect: no live instance
            GcloudRunResult(0, "[]", ""),  # stale-name probe: name free
        ],
        create_results=[GcloudRunResult(0, created_payload, "")],
    )
    posted: list[dict] = []
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **kwargs: posted.append(kwargs),
    )
    backend.launch(_spec())
    assert all("delete" not in c for c in runner.calls), runner.calls
    body = json.loads(posted[0]["note"])
    assert body["pre_launch_deleted_stale_instance"] is False


def test_launch_raises_when_stale_delete_fails_and_skips_create(no_marker_posts) -> None:
    """A real delete failure leaves the name occupied; raise rather than
    let create fail later with a confusing "already exists"."""
    from explore_persona_space.backends.gcp import GcpBackendError

    runner = _Runner(
        list_results=[
            GcloudRunResult(0, "[]", ""),  # reconnect: no live instance
            GcloudRunResult(0, _instance_payload("TERMINATED"), ""),  # stale record
        ],
        delete_results=[GcloudRunResult(1, "", "Internal error during delete")],
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    with pytest.raises(GcpBackendError, match="was not freed"):
        backend.launch(_spec())
    # The create must NOT have been attempted on an un-freed name.
    assert all("create" not in c for c in runner.calls), runner.calls


# ---------------------------------------------------------------------------
# confirm_artifacts — delegates to artifacts module
# ---------------------------------------------------------------------------


def test_confirm_artifacts_delegates_to_verifier_and_fails_on_missing_decl() -> None:
    """A handle without :data:`EXPECTED_ARTIFACTS_HANDLE_KEY` MUST FAIL
    (the slice-2 verifier's contract — silently passing would re-open
    the silent-loss hole)."""
    backend = GcpBackend(
        config=_test_config(),
        runner=_Runner(),
        marker_poster=lambda **_: None,
    )
    from explore_persona_space.backends.base import RunHandle

    handle = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="1",
        pod_name="eps-issue-137",
        scratch_dir="/workspace/eps-issue-137",
        log_path="/workspace/logs/issue-137.log",
        extra={},  # No declaration.
    )
    assert backend.confirm_artifacts(handle) is False


def test_confirm_artifacts_passes_when_verifier_says_pass(monkeypatch) -> None:
    """End-to-end PASS path: the launch path populates ExpectedArtifacts,
    we stub the verifier's IO to return PASS, and the backend honors it."""

    # The artifact verifier dependency-injects every external call; we
    # patch the module-level defaults so a real call would short-circuit.
    monkeypatch.setattr(
        "explore_persona_space.backends.artifacts._default_list_hf_repo_files",
        lambda repo_id, **_kw: [
            "issue137_att-fixed-001/raw_completions/foo.json",
        ],
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.artifacts._default_wandb_run_exists",
        lambda run_path: True,
    )
    monkeypatch.setattr(
        "explore_persona_space.backends.artifacts._default_git_tracked",
        lambda repo_root, rel_paths: set(rel_paths),
    )
    # The repo-root resolver looks for pyproject.toml; the test repo has one.
    # Ensure declared git paths resolve on disk.
    monkeypatch.setattr(
        "explore_persona_space.backends.artifacts._check_git",
        lambda *, paths, io, **_kw: {"status": "SKIP", "detail": "no git paths declared"},
    )
    # Sentinel file: fake a clean read.
    monkeypatch.setattr(
        "explore_persona_space.backends.artifacts._default_read_sentinel",
        lambda path: json.dumps({"phase": "done", "issue": 137}),
    )

    created_payload = json.dumps([{"name": "eps-issue-137", "id": "112233"}])
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[GcloudRunResult(0, created_payload, "")],
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    handle = backend.launch(_spec())
    # The launch path populated the declaration; the verifier (stubbed)
    # sees every path resolve.
    assert backend.confirm_artifacts(handle) is True


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def test_classify_create_capacity_failure_is_provisioning_error() -> None:
    err = classify_create_failure(
        returncode=1,
        stderr="ERROR: ZONE_RESOURCE_POOL_EXHAUSTED for project ...",
    )
    assert isinstance(err, GcpProvisioningError)
    assert "ZONE_RESOURCE_POOL_EXHAUSTED" in (err.evidence.get("matched_pattern") or "")


def test_classify_create_quota_failure_is_provisioning_error() -> None:
    err = classify_create_failure(
        returncode=1,
        stderr="QUOTA_EXCEEDED for GPUS_ALL_REGIONS",
    )
    assert isinstance(err, GcpProvisioningError)


def test_classify_create_regional_quota_prose_is_matched() -> None:
    """gcloud's regional accelerator-quota error is PROSE (the metric name
    sits between "Quota" and "exceeded") — the API-enum patterns miss it
    (#608: four such creates classified "no known provisioning pattern")."""
    err = classify_create_failure(
        returncode=1,
        stderr=(
            "ERROR: (gcloud.compute.instances.create) Could not fetch resource:\n"
            " - Quota 'NVIDIA_A100_80GB_GPUS' exceeded.  Limit: 8.0 in region us-central1.\n"
        ),
    )
    assert isinstance(err, GcpProvisioningError)
    assert err.evidence.get("matched_pattern") == "Quota '"
    assert "no known provisioning pattern" not in err.reason
    # The captured stderr rides the evidence for the router's detail.
    assert "NVIDIA_A100_80GB_GPUS" in err.evidence["stderr_tail"]


# ---------------------------------------------------------------------------
# Pre-create regional-quota headroom probe (#608)
# ---------------------------------------------------------------------------


def _region_quotas_payload(metric: str, usage: float, limit: float) -> str:
    return json.dumps(
        {
            "name": "us-central1",
            "quotas": [
                {"metric": "CPUS", "usage": 12.0, "limit": 1000.0},
                {"metric": metric, "usage": usage, "limit": limit},
            ],
        }
    )


def test_preflight_quota_headroom_insufficient() -> None:
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],  # no live instance
        region_describe_results=[
            GcloudRunResult(0, _region_quotas_payload("NVIDIA_A100_80GB_GPUS", 8.0, 8.0), "")
        ],
    )
    headroom = preflight_quota_headroom(
        spec=_spec(intent="ft-7b"), config=_test_config(), runner=runner
    )
    assert headroom is not None
    assert headroom.metric == "NVIDIA_A100_80GB_GPUS"
    assert headroom.region == "us-central1"
    assert headroom.needed == 4
    assert headroom.available == 0.0
    assert not headroom.sufficient
    # The probe threaded the config into the regions-describe argv.
    region_calls = [a for a in runner.calls if "regions" in a and "describe" in a]
    assert region_calls and "--configuration=eps-test-config" in region_calls[0]


def test_preflight_quota_headroom_sufficient() -> None:
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        region_describe_results=[
            GcloudRunResult(0, _region_quotas_payload("NVIDIA_A100_80GB_GPUS", 4.0, 8.0), "")
        ],
    )
    headroom = preflight_quota_headroom(
        spec=_spec(intent="ft-7b"), config=_test_config(), runner=runner
    )
    assert headroom is not None and headroom.sufficient


def test_preflight_quota_headroom_fails_open_on_describe_rc1() -> None:
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        region_describe_results=[GcloudRunResult(1, "", "Reauthentication failed")],
    )
    assert (
        preflight_quota_headroom(spec=_spec(intent="ft-7b"), config=_test_config(), runner=runner)
        is None
    )


def test_preflight_quota_headroom_fails_open_on_unparseable_json() -> None:
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        region_describe_results=[GcloudRunResult(0, "{not json", "")],
    )
    assert (
        preflight_quota_headroom(spec=_spec(intent="ft-7b"), config=_test_config(), runner=runner)
        is None
    )


def test_preflight_quota_headroom_fails_open_when_metric_missing() -> None:
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        region_describe_results=[
            GcloudRunResult(0, _region_quotas_payload("NVIDIA_L4_GPUS", 0.0, 8.0), "")
        ],
    )
    assert (
        preflight_quota_headroom(spec=_spec(intent="ft-7b"), config=_test_config(), runner=runner)
        is None
    )


def test_preflight_quota_headroom_no_opinion_on_live_instance() -> None:
    """A live eps-issue-<N> instance means the launch path reconnects (no
    new quota needed — and our own instance may BE the usage): no opinion."""
    live_payload = json.dumps([{"name": "eps-issue-137", "id": "123", "status": "RUNNING"}])
    runner = _Runner(list_results=[GcloudRunResult(0, live_payload, "")])
    assert (
        preflight_quota_headroom(spec=_spec(intent="ft-7b"), config=_test_config(), runner=runner)
        is None
    )
    # The regions-describe call was never issued.
    assert not [a for a in runner.calls if "regions" in a]


def test_preflight_quota_headroom_fails_open_on_reconnect_probe_error() -> None:
    runner = _Runner(list_results=[GcloudRunResult(1, "", "Reauthentication failed")])
    assert (
        preflight_quota_headroom(spec=_spec(intent="ft-7b"), config=_test_config(), runner=runner)
        is None
    )


def test_quota_preflight_skips_when_running_zombie_occupies_name(caplog) -> None:
    """#908 §4.1(e): reconnect now refuses a RUNNING+terminal-phase zombie,
    but the zombie's allocated GPUs still COUNT in the regions-describe
    usage read — in a tight-quota regime the headroom verdict would block
    the GCP lane BEFORE launch's stale reclaim can delete the zombie and
    free the quota. The preflight must SKIP (no opinion; no regions
    describe issued), the same disposition as the reconnect-handle path,
    and log the zombie-reclaim disposition."""
    runner = _Runner(
        list_results=[
            GcloudRunResult(0, _instance_payload("RUNNING"), ""),  # reconnect probe
            GcloudRunResult(0, _instance_payload("RUNNING"), ""),  # stale-name probe
        ],
        guest_attr_results=[
            GcloudRunResult(0, _guest_attr_payload("done"), ""),
            GcloudRunResult(0, _guest_attr_payload("done"), ""),
        ],
    )
    with caplog.at_level(logging.INFO, logger="explore_persona_space.backends.gcp"):
        headroom = preflight_quota_headroom(
            spec=_spec(intent="ft-7b"), config=_test_config(), runner=runner
        )
    assert headroom is None
    # No regions-describe headroom decision can block the lane.
    assert not [a for a in runner.calls if "regions" in a]
    assert "skipping the headroom check" in caplog.text


def test_preflight_quota_headroom_no_opinion_on_unmapped_intent() -> None:
    runner = _Runner()
    assert (
        preflight_quota_headroom(spec=_spec(intent="inf-70b"), config=_test_config(), runner=runner)
        is None
    )
    assert runner.calls == []  # decided without any gcloud call


def test_backend_method_delegates_quota_preflight() -> None:
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        region_describe_results=[
            GcloudRunResult(0, _region_quotas_payload("NVIDIA_A100_80GB_GPUS", 8.0, 8.0), "")
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    headroom = backend.preflight_quota_headroom(_spec(intent="ft-7b"))
    assert headroom is not None and not headroom.sufficient


def test_launch_retries_on_capacity_then_succeeds_in_fallback_zone(no_marker_posts) -> None:
    """Capacity miss in primary zone must transparently retry the
    fallback zones before giving up.

    Uses ``eval`` (``g2-standard-4``, available in all three us-central1
    zones) so the fallback ladder is the full ``a → b → c`` — the
    machine-type zone filter (#653) is a no-op for this machine type.
    """
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "999"}])
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),  # us-central1-a
            GcloudRunResult(0, created_payload, ""),  # us-central1-b
        ],
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    handle = backend.launch(_spec(intent="eval"))
    # The second create succeeded; we landed in us-central1-b.
    assert handle.extra["zone"] == "us-central1-b"
    # Two create calls were issued.
    create_calls = [a for a in runner.calls if "create" in a]
    assert len(create_calls) == 2
    assert "--zone=us-central1-a" in create_calls[0]
    assert "--zone=us-central1-b" in create_calls[1]


def test_launch_skips_zone_where_machine_type_absent(no_marker_posts) -> None:
    """#653: a capacity miss on an A2-ultragpu intent must skip
    ``us-central1-b`` (where the family is NOT offered) and fall straight
    to ``us-central1-c`` — never issuing a guaranteed-to-fail create on a
    zone that lacks the machine type (which would burn the attempt counter
    on a CONFIG 400, not a capacity miss)."""
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "999"}])
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),  # us-central1-a
            GcloudRunResult(0, created_payload, ""),  # us-central1-c (b is skipped)
        ],
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    # lora-7b → a2-ultragpu-1g, absent from us-central1-b.
    handle = backend.launch(_spec(intent="lora-7b"))
    assert handle.extra["zone"] == "us-central1-c"
    create_calls = [a for a in runner.calls if "create" in a]
    # Exactly TWO creates: a (capacity miss) then c — NEVER b.
    assert len(create_calls) == 2
    assert "--zone=us-central1-a" in create_calls[0]
    assert "--zone=us-central1-c" in create_calls[1]
    assert not any("--zone=us-central1-b" in a for a in create_calls)


def test_launch_raises_provisioning_error_when_all_zones_capacity_fail(no_marker_posts) -> None:
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        # 3 capacity failures: primary + 2 fallbacks. Uses ``eval``
        # (g2-standard-4, available in all three us-central1 zones) so all
        # three creates are actually issued (the #653 machine-type filter
        # leaves the full ladder intact for this machine type).
        create_results=[
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),
        ],
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    with pytest.raises(GcpProvisioningError):
        backend.launch(_spec(intent="eval"))


def test_launch_does_not_retry_on_non_capacity_failure(no_marker_posts) -> None:
    """A permission / quota failure should NOT retry every zone (the
    next zone would fail identically) — it should raise immediately."""
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[
            GcloudRunResult(1, "", "PERMISSION_DENIED: caller does not have permission"),
        ],
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    with pytest.raises(GcpProvisioningError, match="PERMISSION_DENIED"):
        backend.launch(_spec())
    # Only ONE create call (no retry).
    create_calls = [a for a in runner.calls if "create" in a]
    assert len(create_calls) == 1


# ---------------------------------------------------------------------------
# #774 — full-fan-out observability: the GcpProvisioningError raised after a
# capacity-exhausted fan-out names EVERY zone tried (not just the last zone's
# stderr — the #763 misdiagnosis). Four tests cover the brief's contract:
# (a) zone-iteration order, (b) per-machine-type filter, (c) capacity-retry
# vs auth/quota raise, (d) for-else raise carrying the full zones-tried list.
# ---------------------------------------------------------------------------


def test_zone_fanout_iterates_primary_then_fallbacks_in_order(no_marker_posts) -> None:
    """(a) The fan-out walks [primary, *fallbacks] in ladder order. ``eval``
    (g2-standard-4, offered in all three us-central1 zones) misses in
    us-central1-a, then lands in us-central1-b — the creates are issued in
    a → b order, proving the loop iterates the configured ladder rather than
    jumping zones."""
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "999"}])
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),  # us-central1-a
            GcloudRunResult(0, created_payload, ""),  # us-central1-b
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    handle = backend.launch(_spec(intent="eval"))
    assert handle.extra["zone"] == "us-central1-b"
    create_calls = [c for c in runner.calls if "create" in c and "instances" in c]
    assert [
        next(a.split("=", 1)[1] for a in call if a.startswith("--zone=")) for call in create_calls
    ] == ["us-central1-a", "us-central1-b"]


def test_zone_fanout_skips_zone_filtered_by_machine_type_availability(no_marker_posts) -> None:
    """(b) A2-ultragpu (A100-80) is offered only in {a, c}; the fan-out for a
    ``lora-7b`` intent must skip us-central1-b entirely (where the family does
    not exist) — a capacity miss in -a falls straight to -c, never issuing a
    doomed -b create that would burn the per-day attempt counter (#653)."""
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "999"}])
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),  # us-central1-a
            GcloudRunResult(0, created_payload, ""),  # us-central1-c (b skipped)
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    handle = backend.launch(_spec(intent="lora-7b"))
    assert handle.extra["zone"] == "us-central1-c"
    create_calls = [c for c in runner.calls if "create" in c and "instances" in c]
    assert len(create_calls) == 2
    assert not any("--zone=us-central1-b" in c for c in create_calls)


def test_zone_fanout_capacity_retries_but_auth_failure_raises_immediately(no_marker_posts) -> None:
    """(c) A capacity-shaped miss retries the next zone; a non-capacity
    (auth/quota) failure raises immediately WITHOUT walking further zones —
    a different zone would fail identically. Asserts the two distinct control
    flows in one place."""
    # Capacity miss then success → retries.
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "999"}])
    cap_runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),
            GcloudRunResult(0, created_payload, ""),
        ],
    )
    cap_backend = GcpBackend(
        config=_test_config(), runner=cap_runner, marker_poster=lambda **_: None
    )
    cap_backend.launch(_spec(intent="eval"))
    assert len([c for c in cap_runner.calls if "create" in c and "instances" in c]) == 2

    # Auth failure → raises immediately, ONE create only.
    auth_runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[
            GcloudRunResult(1, "", "PERMISSION_DENIED: caller does not have permission"),
        ],
    )
    auth_backend = GcpBackend(
        config=_test_config(), runner=auth_runner, marker_poster=lambda **_: None
    )
    with pytest.raises(GcpProvisioningError, match="PERMISSION_DENIED") as auth_exc:
        auth_backend.launch(_spec(intent="eval"))
    assert len([c for c in auth_runner.calls if "create" in c and "instances" in c]) == 1
    # #774: even the immediate non-capacity raise carries the partial per-zone
    # trail (the one -a attempt) — the auth failure must not silently drop the
    # zone-fan-out evidence. One record, the auth pattern, the 5 keys, and the
    # derived bare name-list.
    auth_per_zone = auth_exc.value.evidence["per_zone_attempts"]
    assert [e["zone"] for e in auth_per_zone] == ["us-central1-a"]
    assert set(auth_per_zone[0]) == _PER_ZONE_KEYS
    assert "permission" in (auth_per_zone[0]["matched_pattern"] or "").lower()
    assert auth_exc.value.evidence["zones_attempted"] == ["us-central1-a"]


def test_zone_fanout_all_zones_exhausted_error_names_every_zone_tried(no_marker_posts) -> None:
    """(d) When every zone misses on capacity, the for-else raises
    ``GcpProvisioningError`` whose evidence records the FULL ordered zone
    fan-out — closing the #763/#774 observability gap where the marker
    surfaced only the last zone's stderr. Covers both the FILTERED A100-80
    family (2 zones: a, c) and the UNFILTERED g2 family (3 zones: a, b, c)."""
    # Filtered A100-80 family (lora-7b → a2-ultragpu-1g, ladder [a, c]).
    filtered_runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),
        ],
    )
    filtered_backend = GcpBackend(
        config=_test_config(), runner=filtered_runner, marker_poster=lambda **_: None
    )
    with pytest.raises(GcpProvisioningError) as filtered_exc:
        filtered_backend.launch(_spec(intent="lora-7b"))
    za = filtered_exc.value.evidence["zones_attempted"]
    assert za == ["us-central1-a", "us-central1-c"]
    summary = filtered_exc.value.evidence["zones_attempted_summary"]
    assert "us-central1-a" in summary and "us-central1-c" in summary
    assert "a2-ultragpu-1g" in summary

    # Unfiltered family (eval → g2-standard-4, ladder [a, b, c]).
    unfiltered_runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[
            GcloudRunResult(1, "", "does not have enough resources available"),
            GcloudRunResult(1, "", "does not have enough resources available"),
            GcloudRunResult(1, "", "does not have enough resources available"),
        ],
    )
    unfiltered_backend = GcpBackend(
        config=_test_config(), runner=unfiltered_runner, marker_poster=lambda **_: None
    )
    with pytest.raises(GcpProvisioningError) as unfiltered_exc:
        unfiltered_backend.launch(_spec(intent="eval"))
    assert unfiltered_exc.value.evidence["zones_attempted"] == [
        "us-central1-a",
        "us-central1-b",
        "us-central1-c",
    ]


# ---------------------------------------------------------------------------
# #774 round 2 — per-zone OUTCOME records (not just a bare zone-name list). The
# fan-out evidence must carry one {zone, returncode, matched_pattern, elapsed_s,
# stderr_tail} record per zone tried, on the all-zones-exhausted error, on the
# success-after-miss handle, and on the non-capacity immediate-raise path; and
# the no-miss happy path must NOT add the key (byte-identical handle.extra).
# ---------------------------------------------------------------------------

_PER_ZONE_KEYS = {"zone", "returncode", "matched_pattern", "elapsed_s", "stderr_tail"}


def test_zone_fanout_all_zones_exhausted_evidence_carries_per_zone_outcomes(
    no_marker_posts,
) -> None:
    """All-zones-exhausted: the raised error's evidence['per_zone_attempts']
    has one rich record per attempted zone, each with the 5 keys, and the two
    records reflect the two DISTINCT stderr inputs (not a single last-zone
    collapse — the #763 defect this fix closes)."""
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[
            # lora-7b → a2-ultragpu-1g, ladder [a, c]; two distinct exhaustion
            # stderr variants so the per-zone collapse is observable.
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED: zone -a is full"),
            GcloudRunResult(1, "", "does not have enough resources available in zone -c"),
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    with pytest.raises(GcpProvisioningError) as exc:
        backend.launch(_spec(intent="lora-7b"))

    per_zone = exc.value.evidence["per_zone_attempts"]
    assert [e["zone"] for e in per_zone] == ["us-central1-a", "us-central1-c"]
    for entry in per_zone:
        assert set(entry) == _PER_ZONE_KEYS
        assert entry["returncode"] == 1
        assert isinstance(entry["elapsed_s"], float)
        assert entry["matched_pattern"]  # a capacity pattern matched
    # The two records carry DISTINCT stderr tails (per-zone, not collapsed).
    assert "zone -a is full" in per_zone[0]["stderr_tail"]
    assert "zone -c" in per_zone[1]["stderr_tail"]
    assert per_zone[0]["stderr_tail"] != per_zone[1]["stderr_tail"]
    # Back-compat: the bare name-list + summary still present.
    assert exc.value.evidence["zones_attempted"] == ["us-central1-a", "us-central1-c"]
    assert "us-central1-c" in exc.value.evidence["zones_attempted_summary"]


def test_zone_fanout_success_after_miss_handle_extra_carries_trail(no_marker_posts) -> None:
    """A capacity miss in -a then a land in -c threads the EARLIER -a miss onto
    handle.extra['per_zone_attempts'] (a one-entry list — the landing zone -c is
    in handle.extra['zone'], not the trail)."""
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "999"}])
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),  # us-central1-a
            GcloudRunResult(0, created_payload, ""),  # us-central1-c
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    handle = backend.launch(_spec(intent="lora-7b"))
    assert handle.extra["zone"] == "us-central1-c"
    trail = handle.extra["per_zone_attempts"]
    assert len(trail) == 1
    assert trail[0]["zone"] == "us-central1-a"
    assert set(trail[0]) == _PER_ZONE_KEYS
    assert trail[0]["returncode"] == 1


def test_zone_fanout_non_capacity_immediate_raise_attaches_partial_evidence(
    no_marker_posts,
) -> None:
    """A non-capacity (PERMISSION_DENIED) failure on the first zone raises
    immediately, but the raised error still carries the partial per-zone trail
    (the one failed -a attempt) with the AUTH matched_pattern — NOT a capacity
    pattern, and NOT silently dropped."""
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[
            GcloudRunResult(1, "", "PERMISSION_DENIED: caller does not have permission"),
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    with pytest.raises(GcpProvisioningError, match="PERMISSION_DENIED") as exc:
        backend.launch(_spec(intent="lora-7b"))
    per_zone = exc.value.evidence["per_zone_attempts"]
    assert len(per_zone) == 1
    assert per_zone[0]["zone"] == "us-central1-a"
    assert set(per_zone[0]) == _PER_ZONE_KEYS
    matched = (per_zone[0]["matched_pattern"] or "").lower()
    assert "permission" in matched
    assert not any(tag in matched for tag in ("exhaust", "resource", "enough resources"))


def test_zone_fanout_handle_extra_no_per_zone_attempts_when_first_zone_lands(
    no_marker_posts,
) -> None:
    """First-zone success (no miss) must NOT add per_zone_attempts to
    handle.extra — the no-miss happy path stays byte-identical to pre-#774."""
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "999"}])
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[GcloudRunResult(0, created_payload, "")],  # us-central1-a lands
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    handle = backend.launch(_spec(intent="lora-7b"))
    assert handle.extra["zone"] == "us-central1-a"
    assert "per_zone_attempts" not in handle.extra


# ---------------------------------------------------------------------------
# create-timeout-with-live-instance (#736): a FLEX_START create that exceeds
# the 300s subprocess cap but lands a live instance server-side must surface
# as the still-provisioning terminal (→ exit 75), NOT exit 4; a truly-absent
# timeout must still fail loud as a GcpProvisioningError; a probe that itself
# fails (rc!=0 OR a hung list) must re-raise as a GcpProvisioningError chained
# from the ORIGINAL create timeout, never a raw TimeoutExpired.
# ---------------------------------------------------------------------------


def test_launch_create_timeout_live_instance_raises_still_provisioning(no_marker_posts) -> None:
    """Create times out but a post-timeout probe finds the instance live
    (PROVISIONING) → launch raises GcpCreateTimedOutStillProvisioning (NOT
    raw TimeoutExpired, NOT a normal handle) carrying the instance name +
    status, and issues exactly ONE create call (no double-create)."""
    from explore_persona_space.backends.gcp import GcpCreateTimedOutStillProvisioning

    runner = _Runner(
        # list #1 = reconnect at top of launch (None), #2 = stale-name check
        # (name free), #3 = the POST-TIMEOUT probe (live PROVISIONING).
        list_results=[
            GcloudRunResult(0, "[]", ""),
            GcloudRunResult(0, "[]", ""),
            GcloudRunResult(0, _instance_payload("PROVISIONING"), ""),
        ],
        # The create subprocess exceeds the 300s cap.
        create_raises=subprocess.TimeoutExpired(cmd=["gcloud", "create"], timeout=300),
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    with pytest.raises(GcpCreateTimedOutStillProvisioning) as excinfo:
        backend.launch(_spec(intent="eval"))
    exc = excinfo.value
    assert exc.instance_name == "eps-issue-137"
    assert exc.status == "PROVISIONING"
    # Live branch raises immediately — NO second create (no double-provision).
    create_calls = [a for a in runner.calls if "create" in a]
    assert len(create_calls) == 1


def test_launch_create_timeout_no_instance_raises_provisioning_error(no_marker_posts) -> None:
    """Create times out and the post-timeout probe finds NO instance →
    launch raises GcpProvisioningError (capacity-shaped, so the router's
    zone-fallback / next-tier path handles it), NEVER the raw
    TimeoutExpired. Accepts EITHER the zone-retry continuation OR a hard
    immediate raise (§A5 + 'deviations allowed' bless both as fail-loud)."""
    runner = _Runner(
        # All list calls return empty: reconnect None, stale-name free, and
        # the post-timeout probe finds no server-side instance.
        list_results=[
            GcloudRunResult(0, "[]", ""),
            GcloudRunResult(0, "[]", ""),
            GcloudRunResult(0, "[]", ""),
        ],
        create_raises=subprocess.TimeoutExpired(cmd=["gcloud", "create"], timeout=300),
        # zone-a times out → probe empty → continue; zones b + c then
        # capacity-miss so the for-else surfaces a GcpProvisioningError
        # (eval = g2-standard-4 is valid in all three us-central1 zones).
        create_results=[
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),
        ],
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    with pytest.raises(GcpProvisioningError) as excinfo:
        backend.launch(_spec(intent="eval"))
    exc = excinfo.value
    # Fail-loud preserved: a GcpProvisioningError, NOT the raw TimeoutExpired.
    assert not isinstance(exc, subprocess.TimeoutExpired)
    # Capacity-shaped so the router's zone-retry predicate routes it through
    # fallback (either the timeout-derived "resource not created" pattern or a
    # downstream EXHAUSTED capacity miss — both contain a capacity substring).
    matched = (exc.evidence.get("matched_pattern") or "").lower()
    assert any(tag in matched for tag in ("exhaust", "resource", "enough resources"))


def test_launch_create_timeout_preserves_per_zone_attempts(no_marker_posts) -> None:
    """#774: a create TIMEOUT whose post-timeout probe finds NO instance is
    recorded as a per-zone outcome (returncode=-1, the timeout matched_pattern)
    and the fan-out continues; when the remaining zones then capacity-miss, the
    raised error's evidence['per_zone_attempts'] carries the timeout entry FIRST
    followed by the capacity entries — the timeout branch must not drop the
    zone trail. (eval = g2-standard-4, valid in all three us-central1 zones.)"""
    runner = _Runner(
        # All list calls empty: reconnect None, stale-name free, post-timeout
        # probe finds no server-side instance.
        list_results=[
            GcloudRunResult(0, "[]", ""),
            GcloudRunResult(0, "[]", ""),
            GcloudRunResult(0, "[]", ""),
        ],
        # zone-a create times out → probe empty → recorded + continue.
        create_raises=subprocess.TimeoutExpired(cmd=["gcloud", "create"], timeout=300),
        # zones b + c then capacity-miss so the for-else surfaces the error.
        create_results=[
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),
            GcloudRunResult(1, "", "ZONE_RESOURCE_POOL_EXHAUSTED"),
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    with pytest.raises(GcpProvisioningError) as exc:
        backend.launch(_spec(intent="eval"))
    per_zone = exc.value.evidence["per_zone_attempts"]
    # One record per attempted zone (eval → us-central1-a/-b/-c), in order.
    assert [e["zone"] for e in per_zone] == [
        "us-central1-a",
        "us-central1-b",
        "us-central1-c",
    ]
    for entry in per_zone:
        assert set(entry) == _PER_ZONE_KEYS
    # The FIRST zone timed out: returncode sentinel -1 + the timeout pattern.
    assert per_zone[0]["returncode"] == -1
    assert "timeout" in (per_zone[0]["matched_pattern"] or "").lower()
    # The later zones are real capacity misses (returncode 1).
    assert per_zone[1]["returncode"] == 1
    assert per_zone[2]["returncode"] == 1
    # Back-compat name-list carries all three, timeout zone included.
    assert exc.value.evidence["zones_attempted"] == [
        "us-central1-a",
        "us-central1-b",
        "us-central1-c",
    ]


def test_launch_create_timeout_probe_failure_reraises_as_provisioning(no_marker_posts) -> None:
    """Create times out and the post-timeout probe ITSELF fails with rc != 0
    (reconnect_or_none raises GcpProbeError) → launch re-raises as a
    GcpProvisioningError (reason 'create_timeout_probe_failed') chained from
    the ORIGINAL create timeout: 'couldn't ask' must never read as a clean
    outcome. Pins the rc!=0 probe-FAILURE branch."""
    original_timeout = subprocess.TimeoutExpired(cmd=["gcloud", "create"], timeout=300)
    runner = _Runner(
        # list #1 reconnect (None), #2 stale-name (free), #3 = the
        # post-timeout probe returns rc != 0 → GcpProbeError.
        list_results=[
            GcloudRunResult(0, "[]", ""),
            GcloudRunResult(0, "[]", ""),
            GcloudRunResult(1, "", "Reauthentication failed"),
        ],
        create_raises=original_timeout,
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    with pytest.raises(GcpProvisioningError) as excinfo:
        backend.launch(_spec(intent="eval"))
    exc = excinfo.value
    assert exc.reason == "create_timeout_probe_failed"
    # The chained __cause__ is the ORIGINAL create timeout (signal preserved),
    # not the secondary probe-side GcpProbeError.
    assert exc.__cause__ is original_timeout


def test_launch_create_timeout_probe_timeout_reraises_as_provisioning(no_marker_posts) -> None:
    """Create times out AND the post-timeout 'instances list' probe ITSELF
    raises a DISTINCT subprocess.TimeoutExpired (a HUNG list, not an rc!=0
    failure) → launch raises GcpProvisioningError (reason
    'create_timeout_probe_failed') chained from the ORIGINAL CREATE timeout.

    The binding #736 probe-timeout pin: without the helper's
    ``except (GcpProbeError, subprocess.TimeoutExpired)`` tuple, the hung
    list's raw TimeoutExpired would escape past launch to main()'s exit-4
    catch-all — the #736 bug surviving on the probe-timeout branch. Two
    DISTINCT TimeoutExpired instances (distinguished by ``timeout``) so the
    __cause__ assertion proves the create-side one was preserved, not the
    probe-side one."""
    from explore_persona_space.backends.gcp import GcpProbeError

    create_timeout = subprocess.TimeoutExpired(cmd=["gcloud", "create"], timeout=300)
    list_timeout = subprocess.TimeoutExpired(cmd=["gcloud", "list"], timeout=300)
    # Sanity: the two instances are genuinely distinct objects.
    assert create_timeout is not list_timeout
    runner = _Runner(
        # list #1 reconnect (None), #2 stale-name (free) succeed; the #3
        # post-timeout probe HANGS (list_raises fires only after create raised).
        list_results=[
            GcloudRunResult(0, "[]", ""),
            GcloudRunResult(0, "[]", ""),
        ],
        create_raises=create_timeout,
        list_raises=list_timeout,
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    with pytest.raises(GcpProvisioningError) as excinfo:
        backend.launch(_spec(intent="eval"))
    exc = excinfo.value
    # NOT a raw TimeoutExpired (would escape to exit 4) and NOT a GcpProbeError.
    assert not isinstance(exc, subprocess.TimeoutExpired)
    assert not isinstance(exc, GcpProbeError)
    assert exc.reason == "create_timeout_probe_failed"
    # __cause__ is the ORIGINAL CREATE timeout, NOT the probe-side list timeout.
    assert exc.__cause__ is create_timeout
    assert exc.__cause__ is not list_timeout


# ---------------------------------------------------------------------------
# teardown — idempotent on missing instance
# ---------------------------------------------------------------------------


def test_teardown_idempotent_on_missing_instance() -> None:
    runner = _Runner(
        delete_results=[
            GcloudRunResult(
                1, "", "ERROR: (gcloud.compute.instances.delete) instance was not found"
            )
        ],
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    from explore_persona_space.backends.base import RunHandle

    handle = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="1",
        pod_name="eps-issue-137",
        scratch_dir="/workspace/eps-issue-137",
        log_path="/workspace/logs/issue-137.log",
        extra={"zone": "us-central1-a"},
    )
    # No raise — "was not found" is treated as success.
    backend.teardown(handle)


def test_teardown_raises_on_real_failure() -> None:
    runner = _Runner(
        delete_results=[GcloudRunResult(1, "", "Internal server error 500")],
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    from explore_persona_space.backends.base import RunHandle
    from explore_persona_space.backends.gcp import GcpBackendError

    handle = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="1",
        pod_name="eps-issue-137",
        scratch_dir="/workspace/eps-issue-137",
        log_path="/workspace/logs/issue-137.log",
        extra={"zone": "us-central1-a"},
    )
    with pytest.raises(GcpBackendError, match="Internal server error"):
        backend.teardown(handle)


# ---------------------------------------------------------------------------
# teardown — confirm-deleted guard (#683 clean-terminal RUNNING zombie)
# ---------------------------------------------------------------------------


def _teardown_handle():
    """Build a teardown RunHandle (lazy import matches the file's prevailing style)."""
    from explore_persona_space.backends.base import RunHandle

    return RunHandle(
        backend="gcp",
        cluster=None,
        job_id="1",
        pod_name="eps-issue-683",
        scratch_dir="/workspace/eps-issue-683",
        log_path="/workspace/logs/issue-683.log",
        extra={"zone": "us-central1-a"},
    )


def _kind(argv: list[str]) -> str:
    """Classify a recorded gcloud argv the way ``_Runner`` routes it."""
    if "delete" in argv and "instances" in argv:
        return "delete"
    if "describe" in argv and "instances" in argv:
        return "describe"
    return "other"


def test_teardown_confirms_deleted_and_does_not_redelete_when_gone() -> None:
    """rc==0 delete + a 'not found' describe → confirmed gone, NO second delete (#683)."""
    runner = _Runner(
        delete_results=[GcloudRunResult(0, "", "")],
        describe_results=[
            GcloudRunResult(1, "", "ERROR: (gcloud.compute.instances.describe) was not found")
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    backend.teardown(_teardown_handle())
    seq = [_kind(c) for c in runner.calls]
    # The confirm probe ran after the delete; no re-delete on a confirmed-gone VM.
    assert seq == ["delete", "describe"], runner.calls


def test_teardown_redeletes_running_zombie_once() -> None:
    """rc==0 delete but describe shows status=RUNNING (the #683 zombie) → re-delete ONCE."""
    runner = _Runner(
        delete_results=[GcloudRunResult(0, "", ""), GcloudRunResult(0, "", "")],
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    backend.teardown(_teardown_handle())
    seq = [_kind(c) for c in runner.calls]
    # Positional: probe-then-re-delete, in order. A back-to-back double-delete
    # with no probe between (a hypothetical buggy refactor) would FAIL here,
    # where a bare ``len(delete_calls) == 2`` would pass it.
    assert seq == ["delete", "describe", "delete"], runner.calls


def test_teardown_does_not_redelete_on_non_running_describe() -> None:
    """A non-RUNNING describe (STOPPING / TERMINATED / empty status) trusts the rc==0 delete."""
    for status in ("STOPPING", "TERMINATED", ""):
        runner = _Runner(
            delete_results=[GcloudRunResult(0, "", "")],
            describe_results=[GcloudRunResult(0, json.dumps({"status": status}), "")],
        )
        backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
        backend.teardown(_teardown_handle())
        seq = [_kind(c) for c in runner.calls]
        assert seq == ["delete", "describe"], (status, runner.calls)  # no spurious re-delete


def test_teardown_does_not_redelete_on_describe_probe_failure() -> None:
    """A non-404 describe failure does NOT re-delete (the rc==0 delete already landed)."""
    runner = _Runner(
        delete_results=[GcloudRunResult(0, "", "")],
        describe_results=[GcloudRunResult(1, "", "Reauthentication failed")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    backend.teardown(_teardown_handle())
    seq = [_kind(c) for c in runner.calls]
    assert seq == ["delete", "describe"], runner.calls  # probe failure ≠ evidence the VM survived


def test_teardown_redelete_404_is_silent_success() -> None:
    """The re-delete's own 404 is silent success (first delete landed between probe and retry)."""
    runner = _Runner(
        delete_results=[
            GcloudRunResult(0, "", ""),
            GcloudRunResult(1, "", "ERROR: (gcloud.compute.instances.delete) was not found"),
        ],
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    # No raise: the redelete-404 idempotency branch swallows the "was not found".
    backend.teardown(_teardown_handle())
    seq = [_kind(c) for c in runner.calls]
    assert seq == ["delete", "describe", "delete"], runner.calls


def test_teardown_does_not_redelete_on_empty_describe_stdout() -> None:
    """rc==0 describe with EMPTY stdout → ``... else {}`` → status None → no re-delete (#683 v2)."""
    runner = _Runner(
        delete_results=[GcloudRunResult(0, "", "")],
        describe_results=[GcloudRunResult(0, "", "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    backend.teardown(_teardown_handle())
    seq = [_kind(c) for c in runner.calls]
    # The ``if probe.stdout.strip() else {}`` guard: empty STDOUT STRING (rc==0)
    # parses to {} → status None → not RUNNING → no re-delete. Distinct from the
    # empty *status field* ({"status": ""}) case in the non-running test above.
    assert seq == ["delete", "describe"], runner.calls


# ---------------------------------------------------------------------------
# poll
# ---------------------------------------------------------------------------


def test_poll_running_status_maps_to_running() -> None:
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    from explore_persona_space.backends.base import RunHandle

    handle = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="1",
        pod_name="eps-issue-137",
        scratch_dir="/workspace/eps-issue-137",
        log_path="/workspace/logs/issue-137.log",
        extra={"zone": "us-central1-a"},
    )
    pr = backend.poll(handle)
    assert pr.status == "running"
    assert pr.pid_alive is True


def test_poll_terminated_status_maps_to_dead() -> None:
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "TERMINATED"}), "")],
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    from explore_persona_space.backends.base import RunHandle

    handle = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="1",
        pod_name="eps-issue-137",
        scratch_dir="/workspace/eps-issue-137",
        log_path="/workspace/logs/issue-137.log",
        extra={"zone": "us-central1-a"},
    )
    pr = backend.poll(handle)
    assert pr.status == "dead"
    assert pr.pid_alive is False


def test_poll_not_found_maps_to_dead() -> None:
    runner = _Runner(
        describe_results=[GcloudRunResult(1, "", "ERROR: (gcloud) instance was not found")],
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    from explore_persona_space.backends.base import RunHandle

    handle = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="1",
        pod_name="eps-issue-137",
        scratch_dir="/workspace/eps-issue-137",
        log_path="/workspace/logs/issue-137.log",
        extra={"zone": "us-central1-a"},
    )
    pr = backend.poll(handle)
    assert pr.status == "dead"


# ---------------------------------------------------------------------------
# estimate_start_seconds — GCE provisions immediately
# ---------------------------------------------------------------------------


def test_estimate_start_seconds_is_zero_for_gcp() -> None:
    backend = GcpBackend(
        config=_test_config(),
        runner=_Runner(),
        marker_poster=lambda **_: None,
    )
    assert backend.estimate_start_seconds(_spec()) == 0.0


# ---------------------------------------------------------------------------
# audit_stale_gcp_vms — the credit-leak reaper
# ---------------------------------------------------------------------------


def test_audit_stale_gcp_vms_lists_old_instances_when_dry_run() -> None:
    now = datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC)
    old_created = (now - timedelta(hours=48)).isoformat()
    fresh_created = (now - timedelta(hours=1)).isoformat()
    payload = json.dumps(
        [
            {
                "name": "eps-issue-137",
                "id": "1",
                "status": "RUNNING",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
                "creationTimestamp": old_created,
            },
            {
                "name": "eps-issue-200",
                "id": "2",
                "status": "RUNNING",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
                "creationTimestamp": fresh_created,
            },
        ]
    )
    runner = _Runner(list_results=[GcloudRunResult(0, payload, "")])
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        now=now,
        delete=False,
    )
    by_name = {r["name"]: r for r in records}
    assert by_name["eps-issue-137"]["action"] == "would-delete"
    assert by_name["eps-issue-200"]["action"] == "skipped"
    # No delete call issued in dry-run.
    assert all("delete" not in argv for argv in runner.calls)


def test_audit_stale_gcp_vms_deletes_when_delete_true() -> None:
    now = datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC)
    old_created = (now - timedelta(hours=72)).isoformat()
    payload = json.dumps(
        [
            {
                "name": "eps-issue-999",
                "id": "1",
                "status": "RUNNING",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
                "creationTimestamp": old_created,
            }
        ]
    )
    runner = _Runner(
        list_results=[GcloudRunResult(0, payload, "")],
        delete_results=[GcloudRunResult(0, "", "")],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        now=now,
        delete=True,
    )
    assert records[0]["action"] == "deleted"
    # The reaper issued a delete on the right zone.
    delete_calls = [a for a in runner.calls if "delete" in a and "instances" in a]
    assert len(delete_calls) == 1
    assert "eps-issue-999" in delete_calls[0]
    assert "--zone=us-central1-a" in delete_calls[0]


def test_audit_stale_gcp_vms_handles_empty_inventory() -> None:
    """A fresh GCP project with no eps-issue-* instances is legitimate."""
    runner = _Runner(list_results=[GcloudRunResult(0, "[]", "")])
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        delete=False,
    )
    assert records == []


def test_audit_stale_gcp_vms_escalates_non_eps_instances_never_deletes() -> None:
    """Broadened scope (#688): a non-eps-issue-* VM in the dedicated project is
    classified UNMANAGED and ESCALATED (never auto-deleted) — NOT silently
    skipped. The old name filter was blind to such a leftover (#680)."""
    now = datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC)
    old_created = (now - timedelta(hours=720)).isoformat()  # 30 days
    payload = json.dumps(
        [
            {
                "name": "thomas-personal-vm",
                "id": "1",
                "status": "RUNNING",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
                "creationTimestamp": old_created,
            }
        ]
    )
    runner = _Runner(
        list_results=[GcloudRunResult(0, payload, "")],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        now=now,
        delete=True,
    )
    assert len(records) == 1
    assert records[0]["classification"] == JANITOR_CLASS_UNMANAGED
    assert records[0]["action"] == "escalated"
    assert records[0]["reason"] == "age"
    # The unmanaged VM is NEVER deleted, even under delete=True.
    assert not any("delete" in a and "instances" in a for a in runner.calls)


def _one_running_instance(name: str, created_iso: str) -> str:
    return json.dumps(
        [
            {
                "name": name,
                "id": "1",
                "status": "RUNNING",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
                "creationTimestamp": created_iso,
            }
        ]
    )


def _one_running_instance_with_fence(name: str, created_iso: str, max_run_seconds: int) -> str:
    """_one_running_instance + a scheduling.maxRunDuration block (#741 Option B).

    Mirrors the ``scheduling.maxRunDuration: {seconds}`` block GCP populates
    natively whenever the create passed ``--max-run-duration`` — the field the
    per-instance-fence-aware age backstop reads via ``_instance_max_run_seconds``.
    """
    inst = json.loads(_one_running_instance(name, created_iso))[0]
    inst["scheduling"] = {"maxRunDuration": {"seconds": max_run_seconds}}
    return json.dumps([inst])


def test_audit_stale_gcp_vms_reaps_terminal_phase_running_zombie() -> None:
    """A RUNNING VM that has published eps/phase=done past the terminal-phase
    floor (but well under the 24h age backstop) is reaped PROMPTLY — the
    completed-but-not-auto-deleted zombie that would otherwise idle-bill for
    ~22h until the age fence trips (incident #634 family)."""
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(minutes=30)).isoformat()  # 2h young, but done 30m ago
    runner = _Runner(
        list_results=[GcloudRunResult(0, _one_running_instance("eps-issue-634", created), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("done"), "")],
        delete_results=[GcloudRunResult(0, "", "")],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert records[0]["action"] == "deleted"
    assert records[0]["reason"] == "terminal-phase"
    assert records[0]["phase"] == "done"
    delete_calls = [a for a in runner.calls if "delete" in a and "instances" in a]
    assert len(delete_calls) == 1
    assert "eps-issue-634" in delete_calls[0]


def test_audit_stale_gcp_vms_reaps_terminal_phase_failed_zombie() -> None:
    """eps/phase=failed is terminal too — a wedged failed-workload VM gets the
    same prompt reap as a done one."""
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(minutes=20)).isoformat()
    runner = _Runner(
        list_results=[GcloudRunResult(0, _one_running_instance("eps-issue-700", created), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("failed"), "")],
        delete_results=[GcloudRunResult(0, "", "")],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert records[0]["action"] == "deleted"
    assert records[0]["reason"] == "terminal-phase"
    assert records[0]["phase"] == "failed"


def test_audit_stale_gcp_vms_keeps_running_mid_workload_vm() -> None:
    """A RUNNING VM still mid-workload (eps/phase=workload) is NOT reaped —
    the terminal-phase predicate must never touch a live run."""
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(hours=3)).isoformat()  # 3h in, still running
    runner = _Runner(
        list_results=[GcloudRunResult(0, _one_running_instance("eps-issue-800", created), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert records[0]["action"] == "skipped"
    assert records[0]["reason"] is None
    assert records[0]["phase"] == "workload"
    assert not any("delete" in a and "instances" in a for a in runner.calls)


def test_audit_stale_gcp_vms_keeps_terminal_phase_within_finalize_window() -> None:
    """A RUNNING VM that just published eps/phase=done seconds ago (inside the
    terminal-phase floor) is NOT reaped — the floor exists so the sweep never
    races a legitimate post-completion finalize (scp + teardown ~30-60s). The
    phase probe is not even issued for such a young VM."""
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(seconds=120)).isoformat()  # 2 min old, under the 10-min floor
    runner = _Runner(
        list_results=[GcloudRunResult(0, _one_running_instance("eps-issue-900", created), "")],
        # A guest-attr probe here would assert-fail in _Runner if reached
        # only when scripted; default is rc=1 ("not found"), but we assert
        # below that NO probe call was made at all.
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert records[0]["action"] == "skipped"
    assert records[0]["reason"] is None
    assert records[0]["phase"] is None
    # No phase probe for a VM under the floor (cost + correctness).
    assert not any("get-guest-attributes" in a for a in runner.calls)
    assert not any("delete" in a and "instances" in a for a in runner.calls)


def test_audit_stale_gcp_vms_age_backstop_still_reaps_terminal_phase_aside() -> None:
    """The 24h age backstop is independent of phase — an instance over the age
    threshold is reaped with reason='age' even when its phase is unknown."""
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(hours=48)).isoformat()
    runner = _Runner(
        list_results=[GcloudRunResult(0, _one_running_instance("eps-issue-111", created), "")],
        delete_results=[GcloudRunResult(0, "", "")],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert records[0]["action"] == "deleted"
    assert records[0]["reason"] == "age"
    # The age backstop short-circuits the phase probe entirely.
    assert not any("get-guest-attributes" in a for a in runner.calls)


def test_audit_stale_gcp_vms_probe_failure_never_reaps_and_never_crashes() -> None:
    """A guest-attribute PROBE FAILURE (couldn't ask ≠ done) must NOT escalate
    a still-unknown RUNNING VM to deletion, and must NOT crash the inventory
    sweep — the VM falls through to the age backstop untouched."""
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(hours=2)).isoformat()  # past floor, under age backstop
    runner = _Runner(
        list_results=[GcloudRunResult(0, _one_running_instance("eps-issue-222", created), "")],
        # rc != 0 with a non-404 stderr → _read_guest_phase raises GcpProbeError.
        guest_attr_results=[GcloudRunResult(1, "", "Reauthentication failed")],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    # Did not crash; the VM is kept (phase unknown, under age backstop).
    assert records[0]["action"] == "skipped"
    assert records[0]["reason"] is None
    assert records[0]["phase"] is None
    assert not any("delete" in a and "instances" in a for a in runner.calls)


# ---------------------------------------------------------------------------
# Broadened janitor scope + HYBRID classification (#688) — library level.
# ---------------------------------------------------------------------------


def test_classify_janitor_instance_four_classes() -> None:
    """The name→class map: keep wins over managed/allowlisted; managed (eps-issue-*)
    before allowlisted (eps-cap-probe*) before the unmanaged fall-through."""
    assert _classify_janitor_instance("eps-issue-137") == JANITOR_CLASS_MANAGED
    assert _classify_janitor_instance("eps-cap-probe2-1786331") == JANITOR_CLASS_ALLOWLISTED
    assert _classify_janitor_instance("random-dev-vm-x") == JANITOR_CLASS_UNMANAGED
    # keep is empty by default → no name classifies keep without a monkeypatch.
    assert _classify_janitor_instance("keep-mydevbox") == JANITOR_CLASS_UNMANAGED


def test_audit_escalates_unmanaged_stale_vm() -> None:
    """(a) A non-eps-issue-*, non-allowlisted stale VM (48h old) → UNMANAGED,
    escalated (with a spy callback invoked once), and NEVER passed to delete."""
    now = datetime(2026, 6, 14, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(hours=48)).isoformat()
    runner = _Runner(
        list_results=[GcloudRunResult(0, _one_running_instance("random-leftover-vm", created), "")],
    )
    seen: list[dict] = []
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        now=now,
        delete=True,
        escalate=seen.append,
    )
    assert records[0]["classification"] == JANITOR_CLASS_UNMANAGED
    assert records[0]["action"] == "escalated"
    assert records[0]["reason"] == "age"
    assert len(seen) == 1
    assert seen[0]["name"] == "random-leftover-vm"
    assert not any("delete" in a and "instances" in a for a in runner.calls)


def test_audit_reaps_eps_issue_unchanged() -> None:
    """(b) Regression guard: an aged eps-issue-* still reaps with
    classification=managed, action=deleted, reason=age, on the right zone."""
    now = datetime(2026, 6, 14, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(hours=30)).isoformat()
    runner = _Runner(
        list_results=[GcloudRunResult(0, _one_running_instance("eps-issue-137", created), "")],
        delete_results=[GcloudRunResult(0, "", "")],
    )
    seen: list[dict] = []
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        now=now,
        delete=True,
        escalate=seen.append,
    )
    assert records[0]["classification"] == JANITOR_CLASS_MANAGED
    assert records[0]["action"] == "deleted"
    assert records[0]["reason"] == "age"
    delete_calls = [a for a in runner.calls if "delete" in a and "instances" in a]
    assert len(delete_calls) == 1
    assert "eps-issue-137" in delete_calls[0]
    assert "--zone=us-central1-a" in delete_calls[0]
    # A managed VM never invokes the escalation callback.
    assert seen == []


def test_audit_reaps_allowlisted_ephemeral_prefix() -> None:
    """(c) An aged eps-cap-probe* (the actual #680 leak name) is
    allowlisted-ephemeral → AUTO-REAP (deleted, delete argv issued)."""
    now = datetime(2026, 6, 14, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(hours=48)).isoformat()
    runner = _Runner(
        list_results=[
            GcloudRunResult(0, _one_running_instance("eps-cap-probe2-1786331", created), "")
        ],
        delete_results=[GcloudRunResult(0, "", "")],
    )
    seen: list[dict] = []
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        now=now,
        delete=True,
        escalate=seen.append,
    )
    assert records[0]["classification"] == JANITOR_CLASS_ALLOWLISTED
    assert records[0]["action"] == "deleted"
    delete_calls = [a for a in runner.calls if "delete" in a and "instances" in a]
    assert len(delete_calls) == 1
    assert "eps-cap-probe2-1786331" in delete_calls[0]
    assert seen == []  # auto-reap, not escalate


def test_audit_unmanaged_probe_failure_does_not_escalate_or_delete() -> None:
    """(d) An UNMANAGED RUNNING VM past the terminal-phase floor whose phase
    probe FAILS falls through to the age backstop (here under 24h) → skipped,
    reason None, NO delete argv, AND the escalate callback NOT invoked (probe
    failure ≠ stale → no escalation)."""
    now = datetime(2026, 6, 14, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(hours=2)).isoformat()  # past floor, under age backstop
    runner = _Runner(
        list_results=[GcloudRunResult(0, _one_running_instance("random-dev-vm", created), "")],
        # rc != 0 with a non-404 stderr → _read_guest_phase raises GcpProbeError.
        guest_attr_results=[GcloudRunResult(1, "", "Reauthentication failed")],
    )
    seen: list[dict] = []
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
        escalate=seen.append,
    )
    assert records[0]["classification"] == JANITOR_CLASS_UNMANAGED
    assert records[0]["action"] == "skipped"
    assert records[0]["reason"] is None
    assert records[0]["phase"] is None
    assert seen == []
    assert not any("delete" in a and "instances" in a for a in runner.calls)


def test_gcp_audit_preflight_uses_broadened_filter() -> None:
    """(e) The CLI's _AUDIT_NAME_FILTER IS the library's JANITOR_LIST_NAME_FILTER
    (both None), and render_list_argv(name_filter=None) appends NO --filter arg —
    proving the preflight list is the whole-project list."""
    import scripts.gcp_audit as cli

    assert cli._AUDIT_NAME_FILTER is JANITOR_LIST_NAME_FILTER
    assert JANITOR_LIST_NAME_FILTER is None
    argv = render_list_argv(config=_test_config(), name_filter=JANITOR_LIST_NAME_FILTER)
    assert not any(a.startswith("--filter") for a in argv), argv


def test_router_paths_unaffected_by_broadened_janitor() -> None:
    """(f) The seam: reconnect_or_none + _stale_named_instance_or_none still issue
    an EXACT --filter=name=eps-issue-<N> (NOT a broadened/empty filter) and match
    only by exact name — an UNMANAGED VM alongside the queried name is ignored."""
    now = datetime(2026, 6, 14, 12, 0, 0, tzinfo=UTC)
    name = instance_name_for(137)
    # A list payload with BOTH the queried eps-issue-137 (TERMINATED) AND an
    # unrelated unmanaged VM. The router helpers must act only on the exact name.
    payload = json.dumps(
        [
            {
                "name": "some-other-vm",
                "id": "9",
                "status": "RUNNING",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
                "creationTimestamp": (now - timedelta(hours=48)).isoformat(),
            },
            {
                "name": name,
                "id": "1",
                "status": "TERMINATED",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
                "creationTimestamp": (now - timedelta(hours=2)).isoformat(),
            },
        ]
    )
    # reconnect_or_none: no LIVE eps-issue-137 (it's TERMINATED) → None, and the
    # list filter is the EXACT name= filter, never broadened.
    runner = _Runner(list_results=[GcloudRunResult(0, payload, "")])
    assert reconnect_or_none(spec=_spec("lora-7b"), config=_test_config(), runner=runner) is None
    assert any(f"--filter=name={name}" in a for a in runner.calls)
    assert not any("--filter=name~^eps-issue-" in a for a in runner.calls)

    # _stale_named_instance_or_none: the TERMINATED eps-issue-137 record blocks
    # the name; the unmanaged some-other-vm is ignored (matched by exact name).
    runner2 = _Runner(list_results=[GcloudRunResult(0, payload, "")])
    stale = _stale_named_instance_or_none(
        spec=_spec("lora-7b"), config=_test_config(), runner=runner2
    )
    assert stale is not None
    assert stale.name == name
    assert stale.status == "TERMINATED"
    assert any(f"--filter=name={name}" in a for a in runner2.calls)


def test_audit_keep_prefix_never_reaped_or_escalated(monkeypatch) -> None:
    """REC2: with _JANITOR_KEEP_PREFIXES set to a keep- prefix, an aged
    keep-mydevbox VM is classification=keep, action=skipped, the escalate
    callback is NOT invoked, and NO delete argv is issued."""
    monkeypatch.setattr("explore_persona_space.backends.gcp._JANITOR_KEEP_PREFIXES", ("keep-",))
    now = datetime(2026, 6, 14, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(hours=48)).isoformat()
    runner = _Runner(
        list_results=[GcloudRunResult(0, _one_running_instance("keep-mydevbox", created), "")],
    )
    seen: list[dict] = []
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        now=now,
        delete=True,
        escalate=seen.append,
    )
    assert records[0]["classification"] == JANITOR_CLASS_KEEP
    assert records[0]["action"] == "skipped"
    assert records[0]["reason"] is None
    assert seen == []
    assert not any("delete" in a and "instances" in a for a in runner.calls)
    # No phase probe either — a keep VM is short-circuited before the probe.
    assert not any("get-guest-attributes" in a for a in runner.calls)


# ---------------------------------------------------------------------------
# #741 Option B — per-instance-fence-aware age backstop.
# ---------------------------------------------------------------------------


def test_instance_max_run_seconds_reads_scheduling() -> None:
    """The fence reader accepts int OR numeric-string ``seconds``, and returns
    None (→ fixed-fallback fence) for every absent / malformed shape, so the
    backstop never silently disarms on a missing or junk field."""
    # dict-int seconds → parsed.
    assert (
        _instance_max_run_seconds({"scheduling": {"maxRunDuration": {"seconds": 604800}}}) == 604800
    )
    # string-int seconds (gcloud often emits seconds as a string) → parsed.
    assert (
        _instance_max_run_seconds({"scheduling": {"maxRunDuration": {"seconds": "604800"}}})
        == 604800
    )
    # No scheduling block at all → None.
    assert _instance_max_run_seconds({}) is None
    # scheduling present but no maxRunDuration → None.
    assert _instance_max_run_seconds({"scheduling": {}}) is None
    # Junk seconds value → None (never crashes, falls back to the fixed fence).
    assert _instance_max_run_seconds({"scheduling": {"maxRunDuration": {"seconds": "x"}}}) is None
    # Constant sanity: the grace is exactly 1h.
    assert _JANITOR_FENCE_GRACE_SECONDS == 3600


def test_audit_age_backstop_7d_fence_survives_under_old_24h_max_age_seconds() -> None:
    """THE #697 REGRESSION (Phase-2 Statistics Must-Fix #1, DISCRIMINATING form).

    A RUNNING eps-issue-697 with scheduling.maxRunDuration.seconds=604800 (7d),
    age=26h, run under the OLD 24h CLI default (max_age_seconds=24*3600=86400) is
    KEPT (action="skipped", reason None).

    Why DISCRIMINATING: a fence-BLIND impl would set age_fence = max_age_seconds
    = 86400 and REAP (93600 >= 86400 → "deleted"/"age"). The fence-AWARE impl
    sets age_fence = 604800 + 3600 = 608400 and KEEPS (93600 < 608400). So this
    test green-passes ONLY when the implementation actually reads the per-instance
    fence — the exact #697 shape (a 7d job that the janitor's 24h cap must NOT
    kill) RED-FAILS a fence-blind implementation."""
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(hours=26)).isoformat()  # 93600s old
    runner = _Runner(
        list_results=[
            GcloudRunResult(
                0, _one_running_instance_with_fence("eps-issue-697", created, 604800), ""
            )
        ],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,  # the OLD CLI default — the discriminator
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert records[0]["action"] == "skipped"
    assert records[0]["reason"] is None
    assert not any("delete" in a and "instances" in a for a in runner.calls)


def test_audit_stale_gcp_vms_library_default_fallback_fence_is_8d() -> None:
    """The LIBRARY default ``max_age_seconds`` is 8d (192h), matching the
    ``gcp_audit.py`` CLI default + the #741 docstring claim — NOT the old 24h.

    DISCRIMINATING on the default value itself: a fence-less RUNNING VM aged
    30h (between the old 24h fence and the new 8d one), run with NO
    ``max_age_seconds`` argument, is KEPT under the 8d default (108000 <
    691200) but would have been REAPED under the old 24h default (108000 >
    86400). So this green-passes ONLY when the library default is 192*3600 —
    it RED-FAILS if a future refactor reverts the default to 24*3600."""
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(hours=30)).isoformat()  # 108000s old
    runner = _Runner(
        list_results=[GcloudRunResult(0, _one_running_instance("eps-issue-741", created), "")],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        # max_age_seconds DELIBERATELY OMITTED — exercises the library default.
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert records[0]["action"] == "skipped"
    assert records[0]["reason"] is None
    assert not any("delete" in a and "instances" in a for a in runner.calls)


def test_audit_age_backstop_7d_fence_survives_past_24h_under_192h_fallback() -> None:
    """Less-discriminating SANITY CHECK (kept, NOT the #697 guard): the SAME 7d-
    fence instance at age=26h under the new 192h CLI default also keeps. It skips
    under BOTH fence-aware (93600 < 608400) and fence-blind (93600 < 691200)
    paths, so it documents the production-default path without catching the bug —
    the discriminating guard is the ``..._under_old_24h_max_age_seconds`` test."""
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(hours=26)).isoformat()
    runner = _Runner(
        list_results=[
            GcloudRunResult(
                0, _one_running_instance_with_fence("eps-issue-697", created, 604800), ""
            )
        ],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=192 * 3600,  # the new production CLI default
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert records[0]["action"] == "skipped"
    assert records[0]["reason"] is None


def test_audit_age_backstop_within_fence_plus_grace_skipped() -> None:
    """Phase-2 Statistics Must-Fix #2 (the 1h grace constant is MEASURED).

    A RUNNING instance with maxRunDuration.seconds=86400 (24h), age=24h30m
    (88200s), under max_age_seconds=24*3600 → KEPT. The age sits in the OPEN
    interval (fence, fence+grace) = (86400, 90000).

    Why DISCRIMINATING on BOTH grace-applied AND fence-read:
      - fence-aware-WITH-grace: age_fence = 86400 + 3600 = 90000 → 88200 < 90000 → KEEP;
      - drop-the-grace impl:    age_fence = 86400            → 88200 >= 86400 → REAP;
      - fence-blind fallback:   age_fence = max_age_seconds = 86400 → 88200 >= 86400 → REAP.
    So this green-passes ONLY when the impl reads the fence AND adds the 1h grace."""
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(seconds=88200)).isoformat()  # 24h30m old
    runner = _Runner(
        list_results=[
            GcloudRunResult(
                0, _one_running_instance_with_fence("eps-issue-741", created, 86400), ""
            )
        ],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert records[0]["action"] == "skipped"
    assert records[0]["reason"] is None
    assert not any("delete" in a and "instances" in a for a in runner.calls)


def test_audit_age_backstop_24h_fence_instance_reaps_at_25h() -> None:
    """Operator-boundary confirmation under the ``>=`` semantics. A RUNNING
    instance with maxRunDuration.seconds=86400 (24h), age=25h EXACTLY
    (90000s = fence+grace = 86400+3600), under max_age_seconds=24*3600 → REAPED
    (action="deleted", reason="age"). The After-snippet uses
    ``if age_seconds >= age_fence``, so age == fence+grace reaps. Paired with
    ``..._within_fence_plus_grace_skipped`` (age 88200, strictly below) this
    brackets the grace boundary from both sides."""
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(seconds=90000)).isoformat()  # 25h old, exactly fence+grace
    runner = _Runner(
        list_results=[
            GcloudRunResult(
                0, _one_running_instance_with_fence("eps-issue-742", created, 86400), ""
            )
        ],
        delete_results=[GcloudRunResult(0, "", "")],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert records[0]["action"] == "deleted"
    assert records[0]["reason"] == "age"


def test_audit_age_backstop_reaps_past_own_fence_plus_grace() -> None:
    """A 7d-fence instance well past its OWN fence + grace is reaped. age=7d+2h
    (612000s) >= 604800 + 3600 = 608400 → reaped (reason="age"). Pins the
    OVER-grace reap side for the long fence."""
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(seconds=612000)).isoformat()  # 7d + 2h
    runner = _Runner(
        list_results=[
            GcloudRunResult(
                0, _one_running_instance_with_fence("eps-issue-743", created, 604800), ""
            )
        ],
        delete_results=[GcloudRunResult(0, "", "")],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=192 * 3600,
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert records[0]["action"] == "deleted"
    assert records[0]["reason"] == "age"


def test_audit_age_backstop_no_fence_falls_back_to_fixed_192h() -> None:
    """A VM with NO scheduling block (legacy / probe gap) falls through to the
    fixed fallback fence — the backstop never silently disarms. Under 192h →
    kept; over 192h → reaped. Covers the 7d default (a 7.5d no-fence VM is still
    under 8d and kept; only a >8d no-fence VM reaps)."""
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
    # 7.5d old, NO scheduling block → under the 192h (8d) fallback → kept.
    young = (now - timedelta(hours=180)).isoformat()
    runner_young = _Runner(
        list_results=[GcloudRunResult(0, _one_running_instance("eps-issue-744", young), "")],
    )
    rec_young = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner_young,
        max_age_seconds=192 * 3600,
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert rec_young[0]["action"] == "skipped"
    # 9d old, NO scheduling block → over the 192h fallback → reaped at age.
    old = (now - timedelta(hours=216)).isoformat()
    runner_old = _Runner(
        list_results=[GcloudRunResult(0, _one_running_instance("eps-issue-745", old), "")],
        delete_results=[GcloudRunResult(0, "", "")],
    )
    rec_old = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner_old,
        max_age_seconds=192 * 3600,
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert rec_old[0]["action"] == "deleted"
    assert rec_old[0]["reason"] == "age"


def test_audit_terminal_phase_reap_unaffected_by_max_run_duration() -> None:
    """Option B does NOT weaken the prompt 10-min terminal-phase reap. A RUNNING
    instance age 30m with eps/phase=done and a 7d maxRunDuration is reaped at
    reason="terminal-phase": the per-fence age check returns not-stale (30m is far
    under 7d+1h), then the phase probe fires and the terminal phase reaps it."""
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(minutes=30)).isoformat()
    runner = _Runner(
        list_results=[
            GcloudRunResult(
                0, _one_running_instance_with_fence("eps-issue-746", created, 604800), ""
            )
        ],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("done"), "")],
        delete_results=[GcloudRunResult(0, "", "")],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=192 * 3600,
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert records[0]["action"] == "deleted"
    assert records[0]["reason"] == "terminal-phase"
    assert records[0]["phase"] == "done"


def test_render_create_argv_flex_start_at_7d_default_does_not_raise() -> None:
    """The new 7d default sits EXACTLY at the FLEX_START ceiling, not over it. A
    FLEX_START create with NO max_run_duration override uses the 7d config
    default, renders ``--max-run-duration=7d``, and does NOT raise (the
    strict-greater ``> _FLEX_START_MAX_RUN_SECONDS`` assertion passes at 7d ==
    ceiling)."""
    cfg = _test_config()
    spec = _spec(extra={"provisioning_model": "FLEX_START"})
    argv = render_create_argv(
        spec=spec,
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--max-run-duration=7d" in argv
    assert "--provisioning-model=FLEX_START" in argv


# ---------------------------------------------------------------------------
# Argv renderers — describe / delete / list
# ---------------------------------------------------------------------------


def test_render_list_argv_threads_configuration_and_project() -> None:
    cfg = _test_config()
    argv = render_list_argv(config=cfg, name_filter="name=eps-issue-137")
    assert "gcloud" in argv
    assert "list" in argv
    assert "--configuration=eps-test-config" in argv
    assert "--project=eps-test-project" in argv
    assert "--filter=name=eps-issue-137" in argv
    assert "--format=json" in argv


def test_render_describe_argv() -> None:
    argv = render_describe_argv(config=_test_config(), name="eps-issue-137", zone="us-central1-a")
    assert "describe" in argv
    assert "eps-issue-137" in argv
    assert "--zone=us-central1-a" in argv
    assert "--format=json" in argv


def test_render_delete_argv_is_quiet() -> None:
    argv = render_delete_argv(config=_test_config(), name="eps-issue-137", zone="us-central1-a")
    assert "delete" in argv
    assert "--quiet" in argv  # non-interactive teardown
    assert "--zone=us-central1-a" in argv


# ---------------------------------------------------------------------------
# instance_name_for / general naming
# ---------------------------------------------------------------------------


def test_instance_name_for_uses_canonical_eps_issue_prefix() -> None:
    """The audit reaper greps for ``eps-issue-*`` — the name must match."""
    assert instance_name_for(137) == "eps-issue-137"
    assert instance_name_for(1) == "eps-issue-1"


# ---------------------------------------------------------------------------
# Regression: launch() routes the startup script through --metadata-from-file
# ---------------------------------------------------------------------------


def test_launch_uses_metadata_from_file_for_startup_script(no_marker_posts, tmp_path) -> None:
    """Regression for the comma-mangling bug hit on the 2026-06-08 $1 live
    GCP test.

    The rendered startup-script body always contains JSON
    (``{"phase":"done","issue":...,"attempt_id":"..."}``) whose commas
    break gcloud's ``--metadata=KEY=VALUE`` dict-arg parser; gcloud
    rejects the call with ``Bad syntax for dict arg``. ``launch()`` must
    therefore ALWAYS write the script to a per-launch tempfile and route
    it through ``--metadata-from-file=startup-script=<path>`` (the
    branch ``render_create_argv`` already supports), NOT inline through
    ``--metadata=startup-script=<body>``.

    The existing ``render_create_argv`` golden test exercises the inline
    branch and gives a false green here because it never feeds the argv
    through a real gcloud parser; this test pins the live-path contract.
    """
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "112233"}])
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],  # no existing instance
        create_results=[GcloudRunResult(0, created_payload, "")],
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )
    spec = _spec()  # default hydra_args produce the canonical JSON sentinel
    backend.launch(spec)

    create_calls = [a for a in runner.calls if "create" in a and "instances" in a]
    assert len(create_calls) == 1, runner.calls
    create_argv = create_calls[0]

    # The argv MUST take the --metadata-from-file branch — ONE combined
    # flag carrying the secret keys AND the startup-script (gcloud
    # dict-type flags don't merge when repeated).
    from_file_args = [a for a in create_argv if a.startswith("--metadata-from-file=")]
    assert len(from_file_args) == 1, f"--metadata-from-file= missing/split: {create_argv}"
    pairs = from_file_args[0][len("--metadata-from-file=") :].split(",")
    startup_pairs = [p for p in pairs if p.startswith("startup-script=")]
    assert startup_pairs, pairs
    # And the tempfile path it points to MUST actually exist + carry the
    # rendered script body (so gcloud can read it).
    path = startup_pairs[0].split("=", 1)[1]
    script_body = Path(path).read_text(encoding="utf-8")
    # The script body carries the comma-bearing JSON sentinel — verifies
    # the bug payload is in the tempfile rather than smuggled inline.
    assert '"phase":"done"' in script_body
    assert '"issue":137' in script_body
    assert '"attempt_id":' in script_body
    assert "," in script_body  # the actual root cause: commas break --metadata=

    # CRITICALLY: the inline shape must NOT also appear. A duplicate
    # --metadata=startup-script= entry (alongside --metadata-from-file)
    # would re-introduce the parser bug AND let gcloud reject the call
    # because the same key is set twice.
    inline_startup = [a for a in create_argv if a.startswith("--metadata=startup-script=")]
    assert not inline_startup, f"inline startup-script smuggled into argv: {inline_startup}"


# ---------------------------------------------------------------------------
# Regression: reconnect_or_none recovers attempt_id from instance labels
# ---------------------------------------------------------------------------


def test_reconnect_recovers_attempt_id_from_label_and_launch_threads_it(
    no_marker_posts,
) -> None:
    """On reconnect, ``launch()`` must derive ExpectedArtifacts from the
    ORIGINAL attempt_id (the one the VM was provisioned under), NOT a
    fresh one — the VM writes its sentinel + per-attempt artifact paths
    under the original tag, so a fresh tag would point
    ``confirm_artifacts`` at the wrong path and FAIL every reconnect.

    ``reconnect_or_none`` recovers the original by reading the instance's
    ``eps-attempt`` label (set by ``_format_labels`` at create time).
    """
    payload = json.dumps(
        [
            {
                "name": "eps-issue-137",
                "id": "9988",
                "status": "RUNNING",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
                "labels": {
                    "managed-by": "eps",
                    "eps-issue": "137",
                    "eps-attempt": "att-orig-recovered",
                    "eps-intent": "lora-7b",
                },
            }
        ]
    )

    # 1. Direct check: reconnect_or_none populates extra["attempt_id"].
    runner1 = _Runner(list_results=[GcloudRunResult(0, payload, "")])
    handle = reconnect_or_none(spec=_spec(), config=_test_config(), runner=runner1)
    assert handle is not None
    assert handle.extra.get("attempt_id") == "att-orig-recovered"

    # 2. End-to-end: launch() on reconnect path threads the recovered
    #    attempt_id into the ExpectedArtifacts declaration. The
    #    ``_spec()`` helper sets a different attempt_id ("att-fixed-001")
    #    so any code path that ignored the recovered value would derive
    #    the sentinel from "att-fixed-001" instead — caught here.
    runner2 = _Runner(list_results=[GcloudRunResult(0, payload, "")])
    backend = GcpBackend(
        config=_test_config(),
        runner=runner2,
        marker_poster=lambda **_: None,
    )
    handle2 = backend.launch(_spec())
    decl = handle2.extra.get(EXPECTED_ARTIFACTS_HANDLE_KEY)
    assert isinstance(decl, dict), decl
    sentinel_path = decl["sentinel_path"]
    assert "att-orig-recovered" in sentinel_path, sentinel_path
    # Regression guard: the freshly-generated id MUST NOT have been used.
    assert "att-fixed-001" not in sentinel_path, sentinel_path
    # And the HF data path also gets the recovered id (raw-completion
    # paths share the per-attempt namespace).
    assert any("issue137_att-orig-recovered/" in p for p in decl["hf_data_paths"]), decl


def test_reconnect_falls_back_to_fresh_attempt_id_when_label_missing(
    no_marker_posts,
) -> None:
    """If the instance pre-dates the label addition (no ``eps-attempt``
    label), ``launch()`` falls back to the freshly-generated attempt_id
    — best-effort, but the marker trail still proceeds. This pins the
    backward-compat path for instances created before the labels existed.
    """
    payload = json.dumps(
        [
            {
                "name": "eps-issue-137",
                "id": "9988",
                "status": "RUNNING",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
                # No `labels` key at all.
            }
        ]
    )
    runner = _Runner(list_results=[GcloudRunResult(0, payload, "")])
    handle = reconnect_or_none(spec=_spec(), config=_test_config(), runner=runner)
    assert handle is not None
    assert "attempt_id" not in handle.extra


# ---------------------------------------------------------------------------
# fix20/fix21 — launch-time secrets resolution + startup-script burn bounding
# ---------------------------------------------------------------------------


def test_resolve_launch_secrets_missing_required_raises() -> None:
    """An empty env (no dotenv fallback) must fail loud naming every
    missing required key — never silently provision a doomed VM
    (issue 535 GCP lane r7)."""
    spec = _spec()
    with pytest.raises(GcpLaunchSecretsMissing) as exc:
        resolve_launch_secrets(spec, env={})
    msg = str(exc.value)
    for key in REQUIRED_LAUNCH_SECRET_KEYS:
        assert key in msg, msg


def test_resolve_launch_secrets_threads_spec_extra() -> None:
    """Resolved values land in spec.extra['secret_<KEY>'] (the lookup
    render_create_argv prefers); empty optional keys keep the
    drop-when-absent contract."""
    spec = _spec()
    resolve_launch_secrets(
        spec,
        env={"HF_TOKEN": "t-hf", "WANDB_API_KEY": "t-wb", "OPENAI_API_KEY": ""},
    )
    assert spec.extra["secret_HF_TOKEN"] == "t-hf"
    assert spec.extra["secret_WANDB_API_KEY"] == "t-wb"
    assert "secret_OPENAI_API_KEY" not in spec.extra


def test_resolve_launch_secrets_spec_extra_takes_precedence() -> None:
    """A caller-threaded spec.extra['secret_<KEY>'] wins over the env."""
    spec = _spec(extra={"secret_HF_TOKEN": "from-extra"})
    resolve_launch_secrets(spec, env={"HF_TOKEN": "from-env", "WANDB_API_KEY": "t-wb"})
    assert spec.extra["secret_HF_TOKEN"] == "from-extra"


def test_launch_fails_loud_before_any_create_when_secrets_missing(monkeypatch) -> None:
    """launch() must raise BEFORE any gcloud create when the resolver
    reports missing secrets — zero credit spend on a doomed VM."""
    import explore_persona_space.backends.gcp as gcp_mod

    # Reconnect probe returns no live instance, then the resolver fires.
    runner = _Runner(list_results=[GcloudRunResult(0, "[]", "")])
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
    )

    def _raise(spec, env=None):
        raise GcpLaunchSecretsMissing("HF_TOKEN, WANDB_API_KEY")

    monkeypatch.setattr(gcp_mod, "resolve_launch_secrets", _raise)
    with pytest.raises(GcpLaunchSecretsMissing):
        backend.launch(_spec())
    assert all("create" not in argv for argv in runner.calls), runner.calls


def test_launch_threads_resolved_secrets_into_create_metadata(no_marker_posts) -> None:
    """End-to-end through launch(): the resolver's values (here from the
    autouse fixture's env) must reach the create call via the
    ``--metadata-from-file`` channel — 0600 tempfiles whose CONTENT
    carries the token, with the token value itself NEVER on the argv
    (round-2 Codex Major, task #535), and the tempfiles unlinked the
    moment the create loop is done."""
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "112233"}])
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],  # no existing instance
        create_results=[GcloudRunResult(0, created_payload, "")],
    )
    # Spy: at create time (files still on disk), read back each secret
    # tempfile's content + mode exactly as gcloud would.
    secret_reads: dict[str, str] = {}
    secret_modes: dict[str, int] = {}
    secret_paths: dict[str, str] = {}

    def spying_runner(argv):
        if "create" in argv and "instances" in argv:
            for arg in argv:
                if arg.startswith("--metadata-from-file="):
                    for pair in arg[len("--metadata-from-file=") :].split(","):
                        key, _, path = pair.partition("=")
                        if key in ("HF_TOKEN", "WANDB_API_KEY"):
                            secret_paths[key] = path
                            secret_reads[key] = Path(path).read_text()
                            secret_modes[key] = os.stat(path).st_mode & 0o777
        return runner(argv)

    backend = GcpBackend(
        config=_test_config(),
        runner=spying_runner,
        marker_poster=lambda **_: None,
    )
    backend.launch(_spec())
    create_calls = [argv for argv in runner.calls if "create" in argv]
    assert create_calls, runner.calls
    joined = " ".join(create_calls[0])
    # Token values never on the argv / process list.
    assert "hf_test_token" not in joined
    assert "wandb_test_key" not in joined
    # The from-file channel delivered the real values, 0600.
    assert secret_reads == {"HF_TOKEN": "hf_test_token", "WANDB_API_KEY": "wandb_test_key"}
    assert secret_modes == {"HF_TOKEN": 0o600, "WANDB_API_KEY": 0o600}
    # The finally shredded the on-disk token copies after create returned.
    for path in secret_paths.values():
        assert not os.path.exists(path), path


def test_launch_secret_tempfiles_deleted_even_when_create_fails(no_marker_posts) -> None:
    """The finally must shred the token tempfiles on the FAILURE path too
    (a raised GcpProvisioningError must not leave tokens on disk)."""
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[GcloudRunResult(1, "", "permission denied for instances.create")],
    )
    secret_paths: dict[str, str] = {}

    def spying_runner(argv):
        if "create" in argv and "instances" in argv:
            for arg in argv:
                if arg.startswith("--metadata-from-file="):
                    for pair in arg[len("--metadata-from-file=") :].split(","):
                        key, _, path = pair.partition("=")
                        if key in ("HF_TOKEN", "WANDB_API_KEY"):
                            secret_paths[key] = path
        return runner(argv)

    backend = GcpBackend(
        config=_test_config(),
        runner=spying_runner,
        marker_poster=lambda **_: None,
    )
    with pytest.raises(GcpProvisioningError):
        backend.launch(_spec())
    assert secret_paths, "create call never carried the from-file secrets"
    for path in secret_paths.values():
        assert not os.path.exists(path), path


def test_render_create_argv_refuses_inline_secret_without_file() -> None:
    """A secret that resolves to a value but has NO threaded tempfile path
    must fail LOUD — silently dropping it provisions a doomed VM (issue
    535 r7 class) and inlining it would put the token on the argv."""
    with pytest.raises(ValueError, match="HF_TOKEN"):
        render_create_argv(
            spec=_spec(),
            config=_test_config(),
            attempt_id="att-fixed-001",
            startup_script="#!/bin/bash\n",
            secret_files=None,
        )


def test_render_startup_script_failure_trap_powers_off() -> None:
    """A failed startup script must power the VM off (GCE leaves a
    failed-startup VM RUNNING + billing otherwise — issue 535 r7 idled
    ~85 min). The success path must NOT shut down (the verifier
    scp-pulls the sentinel off a live VM)."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert "trap" in script
    assert "shutdown -h now" in script
    assert '[ "$rc" -ne 0 ]' in script  # rc==0 (success) leaves the VM up


def test_render_startup_script_required_secret_preflight() -> None:
    """The in-VM preflight kills the script seconds after boot on an
    empty required secret — before the repo-clone + uv-sync spend."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    for key in REQUIRED_LAUNCH_SECRET_KEYS:
        assert f'[ -n "${{{key}:-}}" ]' in script, f"{key} preflight guard missing"
    preflight_idx = script.index("In-VM preflight: required workload secrets")
    assert preflight_idx < script.index("git clone"), "preflight must precede the clone"
    assert preflight_idx < script.index("uv sync"), "preflight must precede uv sync"


# ---------------------------------------------------------------------------
# #658 — EXIT-trap crash-diagnostics + partial-artifact preservation
# ---------------------------------------------------------------------------


def test_render_startup_script_persists_diagnostics_before_teardown() -> None:
    """#658: the EXIT trap uploads the workload log + partial artifacts to
    the HF data repo BEFORE the shutdown that triggers the
    ``--instance-termination-action=DELETE`` boot-disk destruction, so a
    GCP crash is debuggable and partial progress is recoverable."""
    cfg = _test_config()
    script = render_startup_script(spec=_spec(), config=cfg, attempt_id="att-fixed-001")
    # The helper is defined and called from the crash branch.
    assert "_eps_persist_diagnostics() {" in script
    assert '_eps_persist_diagnostics "$rc"' in script
    # It is wired into the EXIT trap and runs BEFORE the poweroff (else the
    # boot disk + its logs/artifacts are already gone).
    trap_line = next(line for line in script.splitlines() if line.startswith("trap 'rc=$?"))
    assert "_eps_persist_diagnostics" in trap_line
    assert trap_line.index('_eps_persist_diagnostics "$rc"') < trap_line.index("shutdown -h now")
    # The data-repo target is exported so the helper can resolve it
    # (the repo id has no shell-special chars, so it renders verbatim).
    assert f"export EPS_HF_DATA_REPO={cfg.hf_data_repo}" in script


def test_render_startup_script_diagnostics_uploads_log_and_partial_artifacts() -> None:
    """The crash-diagnostics upload covers BOTH the workload log (the
    traceback / stderr) AND the partial artifacts the workload wrote
    before crashing — the two things #658 lost on every retry. #854
    broadens the partial sweep to the data/issue_<N> + data/issue<N>
    working-dir conventions (caches excluded), adds a per-crash
    timestamped log copy + a transcript audit, and prints every skip."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    # Log + crash report upload.
    assert "workload.log" in script
    assert "crash_report.json" in script
    # Partial artifacts: the workload's eval_results/issue_<N>/ dir AND both
    # data/ conventions (#854 — the #825 partials lived in data/issue_825/).
    assert 'eval_results" / f"issue_{issue}"' in script
    assert 'data" / f"issue_{issue}"' in script
    assert 'data" / f"issue{issue}"' in script
    assert "upload_folder" in script
    # Re-downloadable caches are excluded — top-level AND nested forms (the
    # '**/'-prefixed fnmatch forms do NOT match top-level; both are needed).
    assert "ignore_patterns" in script
    assert '"hf_dl/**"' in script
    assert '"**/hf_dl/**"' in script
    # Every skip is loud, the timestamped log copy + transcript audit exist.
    assert "[crash-persist] SKIP" in script
    assert "workload_{stamp}.log" in script
    assert "crash_persist_transcript.log" in script
    # Destination prefix isolates partial output per attempt.
    assert "issue${EPS_ISSUE:-0}_partial/${EPS_ATTEMPT_ID:-unknown}" in script


def test_render_startup_script_diagnostics_is_guarded_and_bounded() -> None:
    """The crash-upload must NEVER delay the poweroff that bounds billing:
    it early-returns without a repo/token (LOUDLY, #854), time-bounds the
    upload, and the trap call is on the non-aborting (``set +e``) crash
    path."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    # Early-return when the repo target / token is absent (early-boot crash)
    # — now with a loud serial-console skip line instead of a silent return.
    assert 'if [ -z "${EPS_HF_DATA_REPO:-}" ] || [ -z "${HF_TOKEN:-}" ]; then' in script
    assert "[crash-persist] SKIP-ALL" in script
    guard_idx = script.index('if [ -z "${EPS_HF_DATA_REPO:-}" ] || [ -z "${HF_TOKEN:-}" ]; then')
    skip_idx = script.index("[crash-persist] SKIP-ALL")
    ret_idx = script.index("return 0;", skip_idx)
    assert guard_idx < skip_idx < ret_idx
    # Hard time bound on the upload so a hung HF call can't strand the VM.
    # #1151: --no-sync removes uv's lock-check / re-sync network exposure
    # from the trap-time budget (the env was already synced by the boot).
    assert "timeout 300 uv run --no-sync python" in script
    # The trap body runs under set +e (non-aborting), so a failing upload
    # command cannot abort the trap before shutdown.
    trap_line = next(line for line in script.splitlines() if line.startswith("trap 'rc=$?"))
    assert "set +e" in trap_line


def test_render_startup_script_diagnostics_present_on_both_branches() -> None:
    """The crash-diagnostics helper lives in the SHARED preamble, so both
    the hydra (train.py) and the workload_cmd branches get it."""
    hydra = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    workload = render_startup_script(
        spec=_spec(hydra_args=(), workload_cmd="bash scripts/issue658_dispatch.sh"),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    for script in (hydra, workload):
        assert "_eps_persist_diagnostics() {" in script
        assert '_eps_persist_diagnostics "$rc"' in script
        # #1151: the eps/persist breadcrumb helper + boot-time clear ride the
        # same shared preamble/env block on both branches.
        assert "_eps_persist_status() {" in script
        assert script.count("instance/guest-attributes/eps/persist") == 2


def test_render_startup_script_is_valid_bash() -> None:
    """Both rendered branches must parse — the #658 helper embeds a Python
    heredoc inside a function inside a subshell; a quoting slip would only
    surface at VM-boot time. ``bash -n`` is the syntax gate (shellcheck is
    not installed on the dev VM)."""
    import tempfile

    for spec in (
        _spec(),
        _spec(hydra_args=(), workload_cmd="bash scripts/issue658_dispatch.sh --flag 'v 1'"),
    ):
        script = render_startup_script(spec=spec, config=_test_config(), attempt_id="att-fixed-001")
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
            fh.write(script)
            path = fh.name
        proc = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
        assert proc.returncode == 0, f"bash -n failed:\n{proc.stderr}"


# ---------------------------------------------------------------------------
# #854 — crash-persist hardening (coverage + diagnosability; incident #825)
# ---------------------------------------------------------------------------


def _extract_persist_heredoc(script: str) -> str:
    """Return the EPS_PERSIST_PY heredoc body (the real embedded python).

    Asserts exactly one heredoc occurrence so the extraction is
    unambiguous; returns the lines between the ``<<'EPS_PERSIST_PY'``
    opener and the ``EPS_PERSIST_PY`` terminator."""
    lines = script.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.endswith("<<'EPS_PERSIST_PY'")]
    ends = [i for i, ln in enumerate(lines) if ln == "EPS_PERSIST_PY"]
    assert len(starts) == 1 and len(ends) == 1, (starts, ends)
    assert starts[0] < ends[0]
    return "\n".join(lines[starts[0] + 1 : ends[0]]) + "\n"


def test_render_startup_script_trap_reaps_watchdog_before_persist() -> None:
    """#854: the EXIT trap kills the reachability watchdog — the only other
    in-guest poweroff actor — at trap ENTRY, BEFORE the crash persist, so
    nothing can power the VM off mid-upload. The clean-exit reap must also
    survive (both-path coverage)."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    trap_line = next(line for line in script.splitlines() if line.startswith("trap 'rc=$?"))
    kill_idx = trap_line.index('kill "${EPS_WATCHDOG_PID')
    persist_idx = trap_line.index('_eps_persist_diagnostics "$rc"')
    shutdown_idx = trap_line.index("shutdown -h now")
    assert kill_idx < persist_idx < shutdown_idx
    # The clean-exit reap (before the success sentinel) still exists as its
    # own standalone line.
    assert '{ kill "${EPS_WATCHDOG_PID:-}" 2>/dev/null; } || true' in script.splitlines()


def test_render_startup_script_persist_streams_eagerly() -> None:
    """#854: the persist output reaches fd 3 (serial console) EAGERLY, line
    by line — the old ``| cut | tail`` pipe buffered everything until EOF,
    so a killed/skipped persist left zero evidence. The reader keeps
    READING to EOF (only printing stops at the cap): an early pipe close
    would SIGPIPE-kill the uploader mid-upload — that property is pinned
    HERE by string assert (the behavioral heredoc test runs the python
    without the bash streamer). Standing dep A: every heredoc ``print(``
    carries ``flush=True`` (buffered prints would defeat the eager
    stream)."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert "| cut -c1-2000 | tail -n 20 >&3" not in script
    # Read-to-EOF reader with trailing-unterminated-line hardening.
    assert 'while IFS= read -r _l || [ -n "$_l" ]' in script
    # Progress bars would spam the now-eager stream.
    assert "HF_HUB_DISABLE_PROGRESS_BARS=1" in script
    # Standing dep A: structural flush=True pin over the extracted heredoc.
    heredoc = _extract_persist_heredoc(script)
    print_lines = [ln for ln in heredoc.splitlines() if "print(" in ln]
    assert print_lines, "heredoc must print (the eager stream reads its stdout)"
    for ln in print_lines:
        assert "flush=True" in ln, f"unflushed print in persist heredoc: {ln!r}"


def test_persist_heredoc_ignore_patterns_cover_top_level_and_nested() -> None:
    """Standing dep B: the rendered IGNORE list excludes caches at BOTH the
    top level and nested depths under the REAL huggingface_hub fnmatch
    filter — the ``**/``-prefixed forms do NOT match top-level paths on
    hub 0.36.2, so both forms must be present and each is load-bearing."""
    import ast

    from huggingface_hub.utils import filter_repo_objects

    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    heredoc = _extract_persist_heredoc(script)
    ignore = None
    for node in ast.walk(ast.parse(heredoc)):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "IGNORE" for t in node.targets
        ):
            ignore = ast.literal_eval(node.value)
    assert isinstance(ignore, list) and ignore, ignore
    items = [
        "hf_dl/a.bin",
        "sub/hf_dl/b.bin",
        "g2_dl/c.bin",
        "sub/g2_dl/d.bin",
        "store/e.pt",
        "sub/store/f.pt",
        ".cache/g",
        "sub/.cache/h",
        "__pycache__/i.pyc",
        "sub/__pycache__/j.pyc",
        "track_s.jsonl",
        "sub/data.json",
    ]
    kept = list(filter_repo_objects(items, ignore_patterns=ignore))
    assert kept == ["track_s.jsonl", "sub/data.json"], kept
    # Each form is load-bearing: top-level-only misses nested, nested-only
    # misses top-level.
    top_only = [p for p in ignore if not p.startswith("**/")]
    nested_only = [p for p in ignore if p.startswith("**/")]
    assert "sub/hf_dl/b.bin" in list(filter_repo_objects(items, ignore_patterns=top_only))
    assert "hf_dl/a.bin" in list(filter_repo_objects(items, ignore_patterns=nested_only))


def _run_persist_heredoc(tmp_path, *, env_overrides=None, make_crash=True, make_dirs=True):
    """Execute the REAL extracted EPS_PERSIST_PY heredoc against a fake
    ``huggingface_hub`` (records upload calls to a JSONL), mirroring
    production's ``python - <dest> <crash>`` stdin invocation.

    ``make_dirs``: True = the full fixture tree; ``"cache_only"`` = a data
    dir holding ONLY pruned caches (the empty-after-excludes SKIP case);
    False = no dirs at all. Returns ``(proc, calls, paths)`` where
    ``calls`` is the ordered list of recorded upload calls and ``paths``
    maps the fixture paths."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    heredoc = _extract_persist_heredoc(script)

    shim = tmp_path / "shim" / "huggingface_hub"
    (shim / "utils").mkdir(parents=True, exist_ok=True)
    calls_path = tmp_path / "calls.jsonl"
    (shim / "__init__.py").write_text(
        "import json, os\n"
        "class HfApi:\n"
        "    def _rec(self, kind, **kw):\n"
        "        with open(os.environ['FAKE_HUB_CALLS'], 'a') as fh:\n"
        "            fh.write(json.dumps({'kind': kind, **kw}) + '\\n')\n"
        "    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):\n"
        "        self._rec('file', path_in_repo=path_in_repo, repo_id=repo_id,\n"
        "                  repo_type=repo_type, nbytes=os.path.getsize(path_or_fileobj))\n"
        "    def upload_folder(self, *, folder_path, path_in_repo, repo_id, repo_type,\n"
        "                      ignore_patterns=None):\n"
        "        # #1151: fail-injection knob — raise on the first N folder calls\n"
        "        # (counted in a sidecar file) so the first-bundle ONE-retry\n"
        "        # behavior is executable-testable. #1339: FAKE_HUB_FOLDER_FAIL_MATCH\n"
        "        # scopes the counter to calls whose path_in_repo contains the\n"
        "        # substring (unset = the pre-#1339 match-all semantics).\n"
        "        fail_times = int(os.environ.get('FAKE_HUB_FOLDER_FAIL_TIMES', '0'))\n"
        "        fail_match = os.environ.get('FAKE_HUB_FOLDER_FAIL_MATCH', '')\n"
        "        if fail_times and (not fail_match or fail_match in path_in_repo):\n"
        "            cpath = os.environ['FAKE_HUB_CALLS'] + '.failcount'\n"
        "            n = 0\n"
        "            if os.path.exists(cpath):\n"
        "                with open(cpath) as fh:\n"
        "                    n = int(fh.read() or 0)\n"
        "            if n < fail_times:\n"
        "                with open(cpath, 'w') as fh:\n"
        "                    fh.write(str(n + 1))\n"
        "                self._rec('folder_fail', path_in_repo=path_in_repo)\n"
        "                raise RuntimeError('fake 504')\n"
        "        # Walk the staged tree AT CALL TIME (#885): record every staged\n"
        "        # relpath, with raw content for small files (else the size) so the\n"
        "        # tail-sentinel / newest-first behavioral asserts can read what was\n"
        "        # actually uploaded.\n"
        "        staged = {}\n"
        "        for dp, _dns, fns in os.walk(folder_path):\n"
        "            for fn in fns:\n"
        "                p = os.path.join(dp, fn)\n"
        "                rel = os.path.relpath(p, folder_path).replace(os.sep, '/')\n"
        "                size = os.path.getsize(p)\n"
        "                if size <= 4096:\n"
        "                    with open(p, 'rb') as fh:\n"
        "                        staged[rel] = fh.read().decode('utf-8', 'replace')\n"
        "                else:\n"
        "                    staged[rel] = size\n"
        "        self._rec('folder', folder_path=folder_path, path_in_repo=path_in_repo,\n"
        "                  repo_id=repo_id, repo_type=repo_type,\n"
        "                  ignore_patterns=ignore_patterns, staged=staged)\n"
    )
    (shim / "utils" / "__init__.py").write_text("")

    root = tmp_path / "workload"
    crash = tmp_path / "eps-crash-report.json"
    log = tmp_path / "workload.log"
    transcript = tmp_path / "transcript.log"
    if make_crash:
        crash.write_text('{"issue":137,"exit_code":1}\n')
        log.write_text("Traceback (most recent call last): boom\n")
    if make_dirs == "cache_only":
        (root / "data" / "issue_137" / "hf_dl").mkdir(parents=True)
        (root / "data" / "issue_137" / "hf_dl" / "cache.bin").write_text("x" * 64)
        (root / "data" / "issue_137" / "sub" / "hf_dl").mkdir(parents=True)
        (root / "data" / "issue_137" / "sub" / "hf_dl" / "x.bin").write_text("y" * 64)
    elif make_dirs:
        (root / "eval_results" / "issue_137").mkdir(parents=True)
        (root / "eval_results" / "issue_137" / "a.json").write_text("{}")
        (root / "data" / "issue_137").mkdir(parents=True)
        (root / "data" / "issue_137" / "track.jsonl").write_text('{"row":1}\n')
        (root / "data" / "issue_137" / "hf_dl").mkdir()
        (root / "data" / "issue_137" / "hf_dl" / "cache.bin").write_text("x" * 64)
        (root / "data" / "issue_137" / "sub" / "hf_dl").mkdir(parents=True)
        (root / "data" / "issue_137" / "sub" / "hf_dl" / "x.bin").write_text("y" * 64)
        (root / "data" / "issue137").mkdir(parents=True)
        (root / "data" / "issue137" / "battery.json").write_text("{}")
        # #885: a per-worker fan-out log under $WORKLOAD_ROOT/logs/ (the
        # worker-logs sweep target; small -> plain-copied, not tailed).
        (root / "logs" / "issue_137").mkdir(parents=True)
        (root / "logs" / "issue_137" / "corpus_gpu0_all.log").write_text("worker traceback\n")
    else:
        # exist_ok: #885 behavioral tests pre-create root/logs/** fixtures
        # before invoking the harness with make_dirs=False.
        root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(tmp_path / "shim"),
            "FAKE_HUB_CALLS": str(calls_path),
            "EPS_HF_DATA_REPO": "org/repo",
            "EPS_ISSUE": "137",
            "EPS_LOG_PATH": str(log) if make_crash else "",
            "WORKLOAD_ROOT": str(root),
            "EPS_PERSIST_TRANSCRIPT": str(transcript),
            # #885: isolate the worker-logs staged tree per test — the
            # production default is the shared literal /tmp/eps-worker-logs,
            # which concurrent pytest sessions on this shared VM would race.
            "EPS_PERSIST_LOG_STAGE_DIR": str(tmp_path / "staged-worker-logs"),
            # #1151: same isolation for the first/final bundle staging dirs
            # (production defaults are shared /tmp literals).
            "EPS_PERSIST_FIRST_STAGE_DIR": str(tmp_path / "staged-first"),
            "EPS_PERSIST_FINAL_STAGE_DIR": str(tmp_path / "staged-final"),
        }
    )
    env.update(env_overrides or {})
    proc = subprocess.run(
        [sys.executable, "-", "issue137_partial/att-x", str(crash)],
        input=heredoc,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    calls = []
    if calls_path.is_file():
        calls = [json.loads(ln) for ln in calls_path.read_text().splitlines()]
    return proc, calls, {"root": root, "crash": crash, "log": log, "transcript": transcript}


def test_persist_heredoc_uploads_in_order_and_covers_data_dir(tmp_path) -> None:
    """#854 behavioral (execution, not string-presence): the REAL heredoc,
    run exactly as production runs it, uploads the first bundle
    (crash_report + workload.log, ONE staged commit — #1151) → worker_logs
    (one staged-tree commit, #885) → eval_results dir → data dirs → the
    final bundle (timestamped log copy + transcript, ONE staged commit —
    #1151), passes the cache excludes to upload_folder, prunes nested
    caches from the dir stats, and exits 0 with ZERO per-file upload_file
    calls (the #664 pre-check stall class)."""
    proc, calls, paths = _run_persist_heredoc(tmp_path)
    assert proc.returncode == 0, proc.stderr
    seq = [(c["kind"], c["path_in_repo"]) for c in calls]
    assert seq[0] == ("folder", "issue137_partial/att-x")
    assert sorted(calls[0]["staged"]) == ["crash_report.json", "workload.log"]
    assert seq[1] == ("folder", "issue137_partial/att-x/worker_logs")
    assert seq[2] == ("folder", "issue137_partial/att-x/eval_results_issue_137")
    assert seq[3] == ("folder", "issue137_partial/att-x/data_issue_137")
    assert seq[4] == ("folder", "issue137_partial/att-x/data_issue137")
    assert seq[5] == ("folder", "issue137_partial/att-x")
    final_staged = sorted(calls[5]["staged"])
    assert len(final_staged) == 2, final_staged
    assert final_staged[0] == "crash_persist_transcript.log"
    assert re.fullmatch(r"workload_\d{8}T\d{6}Z\.log", final_staged[1]), final_staged
    assert len(seq) == 6, seq
    # #1151: ZERO per-file upload_file calls anywhere (the #664 stall class).
    assert not any(c["kind"] == "file" for c in calls)
    # The worker-logs commit staged the fixture worker log verbatim (#885).
    assert calls[1]["staged"] == {"issue_137/corpus_gpu0_all.log": "worker traceback\n"}
    # The data-dir upload carries the cache excludes (top-level AND nested).
    data_call = calls[3]
    assert "hf_dl/**" in data_call["ignore_patterns"]
    assert "**/hf_dl/**" in data_call["ignore_patterns"]
    assert data_call["repo_id"] == "org/repo" and data_call["repo_type"] == "dataset"
    # _dir_entries pruned BOTH the top-level and the nested hf_dl caches: only
    # track.jsonl is counted for data_issue_137.
    assert "[crash-persist] uploading dir data_issue_137 (1 files" in proc.stdout
    # Eagerly-streamed audit lines, start to DONE.
    assert "[crash-persist] BEGIN repo=org/repo dest=issue137_partial/att-x" in proc.stdout
    assert "[crash-persist] DONE" in proc.stdout
    # The transcript tee carries the same audit (staged into the final
    # bundle AFTER the DONE line, so the uploaded copy records it).
    transcript_text = paths["transcript"].read_text()
    assert "[crash-persist] BEGIN" in transcript_text
    assert "[crash-persist] DONE" in transcript_text
    # The staged transcript copy the fake hub recorded ALSO carries the full
    # audit through DONE (transcript-last semantics preserved, #854).
    uploaded_transcript = calls[5]["staged"]["crash_persist_transcript.log"]
    assert "[crash-persist] DONE" in uploaded_transcript
    # #1339 AC-1 (second half): small dirs (default bound 1000 >> the fixture
    # sizes) provably take the UNCHANGED single-commit path — no batch line
    # anywhere in the persist output.
    assert " batch " not in proc.stdout


def test_persist_heredoc_prints_skips_and_honors_env_cap(tmp_path) -> None:
    """#854 behavioral SKIP coverage: a missing crash report / unset log
    path / absent dirs each print a loud SKIP (never a silent pass), an
    empty-after-excludes dir SKIPs, and the per-dir byte cap is
    env-overridable (EPS_PERSIST_DIR_CAP_BYTES) so the oversized branch is
    exercised with a tiny cap."""
    # Variant A: nothing exists — every artifact SKIPs loudly, rc stays 0,
    # and the ONLY upload is the transcript audit itself.
    proc, calls, _ = _run_persist_heredoc(tmp_path / "a", make_crash=False, make_dirs=False)
    assert proc.returncode == 0, proc.stderr
    assert "[crash-persist] SKIP crash_report.json: no such file" in proc.stdout
    assert "[crash-persist] SKIP workload.log: EPS_LOG_PATH unset or file missing" in proc.stdout
    # #1151: an all-skipped first bundle SKIPs loudly instead of committing.
    assert "[crash-persist] SKIP bundle first: nothing staged" in proc.stdout
    assert "[crash-persist] SKIP worker_logs: no such dir" in proc.stdout
    assert "[crash-persist] SKIP eval_results_issue_137: no such dir" in proc.stdout
    assert "[crash-persist] SKIP data_issue_137: no such dir" in proc.stdout
    assert "[crash-persist] SKIP data_issue137: no such dir" in proc.stdout
    assert "[crash-persist] DONE" in proc.stdout
    # The ONLY upload is the final bundle carrying the transcript audit
    # (#1151: it rides an upload_folder commit now, never upload_file).
    assert [(c["kind"], c["path_in_repo"]) for c in calls] == [("folder", "issue137_partial/att-x")]
    assert sorted(calls[0]["staged"]) == ["crash_persist_transcript.log"]
    # Variant B: a 5-byte cap SKIPs the 10-byte data dir as oversized (the
    # critic-requested env override making the branch testable; the cap
    # comparison is strict `size > CAP`), while the nested-cache-only dir
    # SKIPs as empty after excludes.
    b_dir = tmp_path / "b"
    proc, calls, _ = _run_persist_heredoc(b_dir, env_overrides={"EPS_PERSIST_DIR_CAP_BYTES": "5"})
    assert proc.returncode == 0, proc.stderr
    assert "bytes > cap 5 (oversized; regenerate or reduce EPS_PERSIST_DIR_CAP_BYTES)" in (
        proc.stdout
    )
    uploaded_folders = {c["path_in_repo"] for c in calls if c["kind"] == "folder"}
    assert "issue137_partial/att-x/data_issue_137" not in uploaded_folders
    # A dir whose only content is pruned caches (top-level + nested) reads
    # as empty and SKIPs — never a cache-only upload.
    proc2, calls2, _ = _run_persist_heredoc(tmp_path / "c", make_dirs="cache_only")
    assert proc2.returncode == 0, proc2.stderr
    assert "[crash-persist] SKIP data_issue_137: empty after cache excludes" in proc2.stdout
    # No DIR-sweep upload fired — the only folder commits are the #1151
    # first/final bundles, which target the bare dest root.
    assert not any(
        c["kind"] == "folder" and c["path_in_repo"] != "issue137_partial/att-x" for c in calls2
    )


# ---------------------------------------------------------------------------
# #885 — crash-persist worker-logs sweep ($WORKLOAD_ROOT/logs/**)
# ---------------------------------------------------------------------------


def test_render_startup_script_diagnostics_sweeps_worker_logs() -> None:
    """#885: the crash persist ALSO sweeps $WORKLOAD_ROOT/logs/** — the
    per-worker fan-out logs carrying the real traceback (the canonical
    workload.log ends at the fan-out line; two #779 crashes each needed a
    manual boot-disk detach). Newest-first, per-file tail cap + file-count
    bound at STAGE time, staged into /tmp/eps-worker-logs, uploaded as ONE
    upload_folder commit (never a per-file upload_file loop — the #664
    gotcha), ordered AFTER the canonical workload.log and BEFORE the
    partial dirs."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert "EPS_PERSIST_LOG_FILE_CAP_BYTES" in script
    assert "EPS_PERSIST_LOG_MAX_FILES" in script
    assert "/tmp/eps-worker-logs" in script
    assert 'logs_root = root / "logs"' in script
    # The worker-logs upload surface is exactly ONE upload_folder commit of
    # the staged tree (acceptance criterion 6).
    assert 'path_in_repo=f"{dest}/worker_logs"' in script
    # The CALL, not just the def (a defined-but-never-called sweep is dead).
    assert "\n_up_logs()" in script
    # Ordering: first bundle (crash_report + workload.log, #1151) ->
    # _up_logs() -> partial dirs.
    assert (
        script.index('_up_bundle(first_stage, "first", retry=True)')
        < script.index("\n_up_logs()")
        < script.index("for local, name in (")
    )


def test_persist_heredoc_worker_logs_tail_sentinel_and_newest_first(tmp_path) -> None:
    """#885 behavioral (executed heredoc): with a 16-byte tail cap and a
    1-file bound over two worker logs where the OLDER file is the LARGER
    one, the sweep stages exactly the NEWER file (newest-by-mtime — a
    size-sort mutant would stage the older one) and its staged bytes equal
    the exact 16-byte tail sentinel (a dropped-seek head-read mutant would
    stage head filler); the TAILED + dropped-count SKIP lines print AND
    tee into the transcript."""
    root = tmp_path / "workload"
    logs = root / "logs" / "issue_137"
    logs.mkdir(parents=True)
    older = logs / "older_but_bigger.log"
    newer = logs / "newer_crashing_worker.log"
    older.write_bytes(b"H" * 200)  # LARGER but older; no sentinel
    newer.write_bytes(b"h" * 64 + b"TAIL-SENTINEL-OK")  # 80 bytes; distinct 16-byte tail
    base = os.stat(newer).st_mtime
    os.utime(older, (base - 3600, base - 3600))
    os.utime(newer, (base, base))
    proc, calls, paths = _run_persist_heredoc(
        tmp_path,
        make_dirs=False,
        env_overrides={
            "EPS_PERSIST_LOG_FILE_CAP_BYTES": "16",
            "EPS_PERSIST_LOG_MAX_FILES": "1",
        },
    )
    assert proc.returncode == 0, proc.stderr
    # #1151: the first/final bundles ride dest-root folder commits too —
    # scope the assert to the worker_logs upload surface.
    folder_calls = [
        c for c in calls if c["kind"] == "folder" and c["path_in_repo"].endswith("/worker_logs")
    ]
    assert [c["path_in_repo"] for c in folder_calls] == ["issue137_partial/att-x/worker_logs"]
    # Exactly ONE staged file == the NEWER one (kills a size-sort mutant);
    # its content == the exact tail sentinel bytes (kills a dropped-seek
    # head-read mutant).
    assert folder_calls[0]["staged"] == {"issue_137/newer_crashing_worker.log": "TAIL-SENTINEL-OK"}
    assert (
        "[crash-persist] TAILED worker_logs/issue_137/newer_crashing_worker.log:"
        " kept last 16 of 80 bytes"
    ) in proc.stdout
    assert (
        "[crash-persist] SKIP 1 older worker log(s) beyond EPS_PERSIST_LOG_MAX_FILES=1"
    ) in proc.stdout
    assert proc.returncode == 0
    transcript_text = paths["transcript"].read_text()
    assert "TAILED worker_logs/issue_137/newer_crashing_worker.log" in transcript_text
    assert "SKIP 1 older worker log(s)" in transcript_text


def test_persist_heredoc_worker_logs_max_files_lt_one_skips_loudly(tmp_path) -> None:
    """#885: EPS_PERSIST_LOG_MAX_FILES < 1 is the documented disable — a
    loud SKIP, never a silent empty sweep and never an upload."""
    root = tmp_path / "workload"
    logs = root / "logs" / "issue_137"
    logs.mkdir(parents=True)
    (logs / "w.log").write_text("worker traceback\n")
    proc, calls, _ = _run_persist_heredoc(
        tmp_path,
        make_dirs=False,
        env_overrides={"EPS_PERSIST_LOG_MAX_FILES": "0"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "[crash-persist] SKIP worker_logs: EPS_PERSIST_LOG_MAX_FILES=0 < 1" in proc.stdout
    # The worker-logs sweep never uploaded (the #1151 first/final bundle
    # commits target the bare dest root, not .../worker_logs).
    assert not any(
        c["kind"] == "folder" and c["path_in_repo"].endswith("/worker_logs") for c in calls
    )


# ---------------------------------------------------------------------------
# #1151 — crash-persist off-VM breadcrumb (eps/persist) + upload bundling
# ---------------------------------------------------------------------------


def _extract_persist_function(script: str) -> str:
    """Return the full ``_eps_persist_diagnostics`` bash function (definition
    line through its closing top-level brace) from the rendered script."""
    lines = script.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln == "_eps_persist_diagnostics() {")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start : end + 1]) + "\n"


def test_render_startup_script_persist_breadcrumb_ordering() -> None:
    """#1151: the `attempted` entry PUT sits inside _eps_persist_diagnostics
    BEFORE the EPS_PERSIST_PY heredoc; the final-status case sits AFTER the
    streamer close and BEFORE the function's closing brace; the trap still
    reaches shutdown AFTER the persist call and the 300s bound is intact
    (#854 invariants)."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    fn = _extract_persist_function(script)
    assert fn.index('_eps_persist_status "attempted"') < fn.index("<<'EPS_PERSIST_PY'")
    assert fn.index("done; } 2>/dev/null || true;") < fn.index(
        '(124) _eps_persist_status "timeout"'
    )
    assert '(0)   _eps_persist_status "ok"' in fn
    assert '(*)   _eps_persist_status "failed_rc${_prc}"' in fn
    # The breadcrumb writes are fail-soft, metadata-capped curls (-m 5) so a
    # wedged metadata server can never eat the persist budget.
    assert "_eps_persist_status() { curl -fsS -m 5 -X PUT" in script
    trap_line = next(line for line in script.splitlines() if line.startswith("trap 'rc=$?"))
    assert trap_line.index('_eps_persist_diagnostics "$rc"') < trap_line.index("shutdown -h now")
    assert "timeout 300 uv run --no-sync python" in script


def test_render_startup_script_persist_breadcrumb_separate_key() -> None:
    """#1151: breadcrumb writes target guest-attributes/eps/persist — a
    SEPARATE key from eps/phase (the poll classification + #908 zombie
    predicates key on eps/phase and must not see new values; the #935
    eps/done_persist discipline). The eps/phase URL site count stays at its
    pre-#1151 value (the _eps_phase helper PUT + the done-grace read)."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    # persist key: the helper PUT + the boot-time DELETE clear.
    assert script.count("instance/guest-attributes/eps/persist") == 2
    # phase key: unchanged pre-#1151 count (helper PUT + done-grace read).
    assert script.count("instance/guest-attributes/eps/phase") == 2
    # The boot-time clear (staleness guard: guest attributes survive
    # same-instance reboots) renders after the first startup phase write.
    del_idx = script.index("curl -fsS -m 5 -X DELETE")
    assert script.index("_eps_phase startup") < del_idx


def test_render_startup_script_persist_skip_writes_skipped_no_token() -> None:
    """#1151: the token-guard skip branch breadcrumbs skipped_no_token BEFORE
    its return 0 — an early-boot crash is otherwise indistinguishable from
    a killed persist once the boot disk is DELETEd."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    guard_idx = script.index('if [ -z "${EPS_HF_DATA_REPO:-}" ] || [ -z "${HF_TOKEN:-}" ]; then')
    skip_idx = script.index('_eps_persist_status "skipped_no_token"')
    ret_idx = script.index("return 0;", skip_idx)
    assert guard_idx < skip_idx < ret_idx


def _run_persist_function_bash(tmp_path, *, fake_uv_rc=None, with_token=True, env_overrides=None):
    """Execute the REAL extracted _eps_persist_diagnostics bash function with
    ``_eps_persist_status`` overridden to a call recorder (no metadata
    server, no network) and a stub ``uv`` on PATH controlling the
    persist-python rc — the executed discriminator for the
    entry/skip/final-status semantics (``bash -n`` is syntax-only and the
    heredoc harness bypasses the bash layer entirely).

    HOME points at tmp_path so the subshell's ``$HOME/.local/bin`` PATH
    prepend cannot resolve the VM's REAL uv ahead of the stub."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    fn = _extract_persist_function(script)
    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir(parents=True, exist_ok=True)
    uv = stub_bin / "uv"
    uv.write_text('#!/bin/bash\ncat >/dev/null\nexit "${FAKE_UV_RC:-0}"\n')
    uv.chmod(0o755)
    calls = tmp_path / "breadcrumb-calls.txt"
    runner = tmp_path / "runner.sh"
    runner.write_text(
        fn
        + '\n_eps_persist_status() { printf \'%s\\n\' "$1" >> "$CALLS"; }\n'
        + "_eps_persist_diagnostics 1\n"
    )
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{stub_bin}:{env['PATH']}",
            "HOME": str(tmp_path),
            "CALLS": str(calls),
            "EPS_CRASH_PERSIST_RC": str(tmp_path / "persist.rc"),
            "WORKLOAD_ROOT": str(tmp_path),
            "EPS_ISSUE": "137",
            "EPS_ATTEMPT_ID": "att-x",
        }
    )
    if with_token:
        env.update({"EPS_HF_DATA_REPO": "org/repo", "HF_TOKEN": "x"})
    else:
        env.pop("EPS_HF_DATA_REPO", None)
        env.pop("HF_TOKEN", None)
    if fake_uv_rc is not None:
        env["FAKE_UV_RC"] = str(fake_uv_rc)
    env.update(env_overrides or {})
    proc = subprocess.run(
        ["bash", str(runner)], capture_output=True, text=True, env=env, timeout=60
    )
    got = calls.read_text().splitlines() if calls.is_file() else []
    return proc, got


def test_persist_final_status_case_semantics(tmp_path) -> None:
    """#1151 executed-bash discriminator: rc-file 0 -> ok, 124 -> timeout,
    7 -> failed_rc7, and a MISSING rc file writes NOTHING — the standing
    `attempted` IS the killed-mid-persist signal; a guessed final value
    here would destroy that discriminator."""
    proc, got = _run_persist_function_bash(tmp_path / "ok", fake_uv_rc=0)
    assert proc.returncode == 0, proc.stderr
    assert got == ["attempted", "ok"], got
    proc, got = _run_persist_function_bash(tmp_path / "to", fake_uv_rc=124)
    assert proc.returncode == 0, proc.stderr
    assert got == ["attempted", "timeout"], got
    proc, got = _run_persist_function_bash(tmp_path / "rc7", fake_uv_rc=7)
    assert proc.returncode == 0, proc.stderr
    assert got == ["attempted", "failed_rc7"], got
    # MISSING rc file: point the rc path into a nonexistent dir so the
    # guarded write fails -> the readback is empty -> ZERO final writes.
    missing = tmp_path / "missing"
    proc, got = _run_persist_function_bash(
        missing,
        fake_uv_rc=0,
        env_overrides={"EPS_CRASH_PERSIST_RC": str(missing / "no-such-dir" / "rc")},
    )
    assert proc.returncode == 0, proc.stderr
    assert got == ["attempted"], got


def test_persist_entry_and_skip_breadcrumbs_fire(tmp_path) -> None:
    """#1151 executed-bash: entering the function records `attempted` FIRST
    (unconditional proof of invocation); the missing-token branch records
    `skipped_no_token` and returns 0 without reaching the final-status
    case."""
    proc, got = _run_persist_function_bash(tmp_path / "skip", with_token=False)
    assert proc.returncode == 0, proc.stderr
    assert got == ["attempted", "skipped_no_token"], got
    proc, got = _run_persist_function_bash(tmp_path / "full", fake_uv_rc=0)
    assert proc.returncode == 0, proc.stderr
    assert got[0] == "attempted", got


def test_persist_heredoc_first_bundle_single_commit_with_one_retry(tmp_path) -> None:
    """#1151 behavioral: the FIRST HF call is one upload_folder commit whose
    staged tree carries crash_report.json + workload.log; a raised first
    attempt retries EXACTLY once after the (zeroed) backoff; zero
    upload_file calls anywhere."""
    proc, calls, _ = _run_persist_heredoc(
        tmp_path / "retry-ok",
        env_overrides={"FAKE_HUB_FOLDER_FAIL_TIMES": "1", "EPS_PERSIST_RETRY_BACKOFF_S": "0"},
    )
    assert proc.returncode == 0, proc.stderr
    assert calls[0]["kind"] == "folder_fail"
    assert calls[0]["path_in_repo"] == "issue137_partial/att-x"
    # The retry is the SECOND call, same dest, carrying BOTH files.
    assert calls[1]["kind"] == "folder"
    assert calls[1]["path_in_repo"] == "issue137_partial/att-x"
    assert sorted(calls[1]["staged"]) == ["crash_report.json", "workload.log"]
    assert "[crash-persist] FAILED bundle first attempt 1/2" in proc.stdout
    assert "[crash-persist] uploaded bundle first" in proc.stdout
    assert sum(1 for c in calls if c["kind"] == "folder_fail") == 1
    assert not any(c["kind"] == "file" for c in calls)
    # Both attempts exhausted: exactly TWO attempts (never a third), the
    # persist proceeds to the remaining artifacts, and rc stays 0 — which is
    # why the breadcrumb documents ok = "persist python exited 0", NOT
    # "uploads landed".
    proc, calls, _ = _run_persist_heredoc(
        tmp_path / "retry-exhausted",
        env_overrides={"FAKE_HUB_FOLDER_FAIL_TIMES": "2", "EPS_PERSIST_RETRY_BACKOFF_S": "0"},
    )
    assert proc.returncode == 0, proc.stderr
    first_fails = [c for c in calls if c["kind"] == "folder_fail"]
    assert [c["path_in_repo"] for c in first_fails] == [
        "issue137_partial/att-x",
        "issue137_partial/att-x",
    ]
    assert "[crash-persist] FAILED bundle first attempt 2/2" in proc.stdout
    assert "[crash-persist] uploaded bundle first" not in proc.stdout
    # The later bundles/dirs still upload (the persist never aborts).
    assert any(c["kind"] == "folder" for c in calls)
    assert "[crash-persist] DONE" in proc.stdout


def test_persist_heredoc_final_bundle_carries_timestamped_log_and_transcript(tmp_path) -> None:
    """#1151 behavioral: the FINAL commit is one upload_folder bundle to
    {dest} whose staged tree carries the per-crash timestamped log copy AND
    the transcript audit — repo paths byte-identical to the pre-#1151
    per-file uploads; the final bundle takes NO retry (retry=False pinned
    on the rendered call)."""
    proc, calls, _ = _run_persist_heredoc(tmp_path)
    assert proc.returncode == 0, proc.stderr
    final = calls[-1]
    assert final["kind"] == "folder"
    assert final["path_in_repo"] == "issue137_partial/att-x"
    staged = sorted(final["staged"])
    assert staged[0] == "crash_persist_transcript.log"
    assert re.fullmatch(r"workload_\d{8}T\d{6}Z\.log", staged[1]), staged
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert '_up_bundle(first_stage, "first", retry=True)' in script
    assert '_up_bundle(final_stage, "final", retry=False)' in script


# ---------------------------------------------------------------------------
# #1339 — crash-persist chunked partial-dir uploads (incident #1090 fu5)
# ---------------------------------------------------------------------------


def test_render_startup_script_dir_uploads_chunked() -> None:
    """#1339 string coverage: the rendered script carries the chunking knobs,
    the chunk-header / ABORT / summary line templates, and targets the SAME
    f"{dest}/{name}" repo path on BOTH the single-commit and the batch
    upload_folder calls (path identity pinned at the string level too)."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert 'BATCH_MAX = _env_int("EPS_PERSIST_DIR_MAX_FILES_PER_COMMIT", 1000)' in script
    assert 'BATCH_ABORT_STREAK = _env_int("EPS_PERSIST_DIR_BATCH_ABORT_STREAK", 2)' in script
    assert '"EPS_PERSIST_DIR_STAGE_DIR", "/tmp/eps-dir-batch"' in script
    assert "chunking dir {name}: {n} files >" in script
    assert "ABORT dir {name}: {fail_streak} consecutive" in script
    assert "dir {name}: {ok_batches}/{nb} batches uploaded" in script
    # BOTH the single-commit path and the batch path target f"{dest}/{name}"
    # — byte-identical repo paths for chunked AND unchunked dirs (AC-6).
    assert script.count('path_in_repo=f"{dest}/{name}"') == 2
    # The serial print cap moved 120 -> 200 with the chunked worst case
    # (~122 lines); the literal + its sizing comment move together (#1339).
    assert 'if [ "$_n" -le 200 ]; then' in script
    assert 'if [ "$_n" -le 120 ]; then' not in script


def _make_chunk_fixture(tmp_path):
    """Pre-create the #1339 chunk-test tree for a make_dirs=False harness
    run: 5 files under eval_results/issue_137/ (one in a subdir) with
    staggered mtimes (f4 newest ... sub/f0 oldest — newest-first batches are
    then deterministic), plus 1-file data dirs (each far under the batch
    bound, so they take the single-commit path). Returns the 5 relpaths."""
    root = tmp_path / "workload"
    ev = root / "eval_results" / "issue_137"
    (ev / "sub").mkdir(parents=True)
    base = 1_700_000_000
    rels = ["f4.json", "f3.json", "f2.json", "f1.json", "sub/f0.json"]
    for i, rel in enumerate(rels):
        p = ev / rel
        p.write_text(json.dumps({"i": i}))
        os.utime(p, (base - 10 * i, base - 10 * i))
    (root / "data" / "issue_137").mkdir(parents=True)
    (root / "data" / "issue_137" / "track.jsonl").write_text('{"row":1}\n')
    (root / "data" / "issue137").mkdir(parents=True)
    (root / "data" / "issue137" / "battery.json").write_text("{}")
    return rels


_CHUNK_ENV = {
    "EPS_PERSIST_DIR_MAX_FILES_PER_COMMIT": "2",
    "EPS_PERSIST_RETRY_BACKOFF_S": "0",
}


def _chunk_env(tmp_path, **extra):
    """The shared #1339 chunk-test env: bound=2, zero backoff, and a
    per-test staging dir (shared-VM isolation — the production default is
    the shared literal /tmp/eps-dir-batch, the #885 pattern)."""
    env = dict(_CHUNK_ENV)
    env["EPS_PERSIST_DIR_STAGE_DIR"] = str(tmp_path / "staged-dir-batch")
    env.update(extra)
    return env


def test_persist_heredoc_chunks_large_dir_newest_first_and_path_identical(tmp_path) -> None:
    """#1339 behavioral headline: an over-bound dir uploads as >=2
    upload_folder commits, ALL targeting the byte-identical
    {dest}/eval_results_issue_137 repo path, newest-first, covering every
    file exactly once, with zero per-file upload_file calls and rc 0."""
    rels = _make_chunk_fixture(tmp_path)
    proc, calls, _ = _run_persist_heredoc(
        tmp_path, make_dirs=False, env_overrides=_chunk_env(tmp_path)
    )
    assert proc.returncode == 0, proc.stderr
    ev_path = "issue137_partial/att-x/eval_results_issue_137"
    batches = [c for c in calls if c["kind"] == "folder" and c["path_in_repo"] == ev_path]
    assert len(batches) == 3, [c["path_in_repo"] for c in calls]
    # Per-call batch size never exceeds the bound — a missing per-batch
    # rmtree would GROW batches across iterations (the staging-leak class).
    for b in batches:
        assert len(b["staged"]) <= 2, sorted(b["staged"])
    # Newest-first: batch 1 carries exactly the two newest files.
    assert sorted(batches[0]["staged"]) == ["f3.json", "f4.json"]
    # Union over batches covers all 5 relpaths (incl. the subdir one),
    # each exactly once.
    all_staged = [rel for b in batches for rel in b["staged"]]
    assert sorted(all_staged) == sorted(rels)
    assert len(all_staged) == len(set(all_staged))
    # Zero per-file upload_file calls anywhere (the #664 504-storm class).
    assert not any(c["kind"] == "file" for c in calls)
    assert (
        "[crash-persist] chunking dir eval_results_issue_137: 5 files >"
        " EPS_PERSIST_DIR_MAX_FILES_PER_COMMIT=2; 3 batches, newest-first"
    ) in proc.stdout
    assert (
        "[crash-persist] dir eval_results_issue_137: 3/3 batches uploaded (5/5 files)"
    ) in proc.stdout


def test_persist_heredoc_chunked_batch_retries_then_continues(tmp_path) -> None:
    """#1339 AC-2: a failed batch attempt retries EXACTLY once (the #1151
    bounded-retry mirror) and the persist then continues — the remaining
    batches, the data dirs, and the final bundle all still upload, in
    order."""
    _make_chunk_fixture(tmp_path)
    proc, calls, _ = _run_persist_heredoc(
        tmp_path,
        make_dirs=False,
        env_overrides=_chunk_env(
            tmp_path,
            FAKE_HUB_FOLDER_FAIL_MATCH="eval_results_issue_137",
            FAKE_HUB_FOLDER_FAIL_TIMES="1",
        ),
    )
    assert proc.returncode == 0, proc.stderr
    ev_path = "issue137_partial/att-x/eval_results_issue_137"
    # The match knob scoped the injected failure to the eval dir: the FIRST
    # bundle (bare dest) succeeded on attempt 1.
    assert (calls[0]["kind"], calls[0]["path_in_repo"]) == ("folder", "issue137_partial/att-x")
    fails = [c for c in calls if c["kind"] == "folder_fail"]
    assert [c["path_in_repo"] for c in fails] == [ev_path]
    assert (
        "[crash-persist] FAILED dir eval_results_issue_137 batch 1/3 attempt 1/2"
    ) in proc.stdout
    assert "[crash-persist] uploaded dir eval_results_issue_137 batch 1/3" in proc.stdout
    batches = [c for c in calls if c["kind"] == "folder" and c["path_in_repo"] == ev_path]
    assert len(batches) == 3, [c["path_in_repo"] for c in calls]
    assert (
        "[crash-persist] dir eval_results_issue_137: 3/3 batches uploaded (5/5 files)"
    ) in proc.stdout
    # The data dirs + the final bundle still upload AFTER the chunked dir
    # (order preserved).
    idx_last_batch = max(i for i, c in enumerate(calls) if c["path_in_repo"] == ev_path)
    tail = [(c["kind"], c["path_in_repo"]) for c in calls[idx_last_batch + 1 :]]
    assert tail == [
        ("folder", "issue137_partial/att-x/data_issue_137"),
        ("folder", "issue137_partial/att-x/data_issue137"),
        ("folder", "issue137_partial/att-x"),
    ], tail


def test_persist_heredoc_chunked_aborts_after_consecutive_failures_fail_soft(tmp_path) -> None:
    """#1339 AC-3/AC-4: after EPS_PERSIST_DIR_BATCH_ABORT_STREAK consecutive
    fully-failed batches the dir is abandoned LOUDLY (batch 3 never
    attempted), the remaining dirs + the final bundle still upload, and rc
    stays 0 — the fail-soft poweroff path is always reached."""
    _make_chunk_fixture(tmp_path)
    proc, calls, _ = _run_persist_heredoc(
        tmp_path,
        make_dirs=False,
        env_overrides=_chunk_env(
            tmp_path,
            FAKE_HUB_FOLDER_FAIL_MATCH="eval_results_issue_137",
            FAKE_HUB_FOLDER_FAIL_TIMES="99",
            EPS_PERSIST_DIR_BATCH_ABORT_STREAK="2",
        ),
    )
    assert proc.returncode == 0, proc.stderr
    ev_path = "issue137_partial/att-x/eval_results_issue_137"
    fails = [c for c in calls if c["kind"] == "folder_fail" and c["path_in_repo"] == ev_path]
    # 2 batches x 2 attempts each — batch 3 is never attempted.
    assert len(fails) == 4, [(c["kind"], c["path_in_repo"]) for c in calls]
    assert not any(c["kind"] == "folder" and c["path_in_repo"] == ev_path for c in calls)
    assert (
        "[crash-persist] ABORT dir eval_results_issue_137: 2 consecutive batch"
        " failures; 1 batch(es) unsent"
    ) in proc.stdout
    assert (
        "[crash-persist] dir eval_results_issue_137: 0/3 batches uploaded (0/5 files)"
    ) in proc.stdout
    # The data dirs AND the final bundle (transcript) still uploaded.
    folder_paths = [c["path_in_repo"] for c in calls if c["kind"] == "folder"]
    assert "issue137_partial/att-x/data_issue_137" in folder_paths
    assert "issue137_partial/att-x/data_issue137" in folder_paths
    final = [c for c in calls if c["kind"] == "folder"][-1]
    assert final["path_in_repo"] == "issue137_partial/att-x"
    assert "crash_persist_transcript.log" in final["staged"]
    assert "[crash-persist] DONE" in proc.stdout


def _guest_attr_kv(key: str, value: str) -> str:
    """A gcloud get-guest-attributes payload for an arbitrary eps/<key>."""
    return json.dumps([{"namespace": "eps", "key": key, "value": value}])


def test_poll_terminal_failed_surfaces_persist_breadcrumb() -> None:
    """#1151: the TERMINATED+failed terminal diagnosis carries the
    eps/persist breadcrumb in log_tail_excerpt — the exact tick the 01:24Z
    #811 diagnosis read. No PollResult schema change: the excerpt already
    rides every terminal marker the orchestrator reads."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "TERMINATED"}), "")],
        guest_attr_results=[
            GcloudRunResult(0, _guest_attr_payload("failed"), ""),
            GcloudRunResult(1, "", "guest attribute eps/workload_started not found"),
            GcloudRunResult(0, _guest_attr_kv("persist", "attempted"), ""),
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "dead"
    assert pr.current_phase == "terminal_setup_failed"
    assert "[crash-persist-breadcrumb] eps/persist=attempted (instance TERMINATED)" in (
        pr.log_tail_excerpt
    )


def test_poll_terminated_failed_workload_started_carries_breadcrumb() -> None:
    """#1151: TERMINATED + failed + workload-started keeps the
    terminal_terminated classification VERBATIM (the #669 exclusion — no
    failover change) and gains the breadcrumb line: the trap ran
    _eps_persist_diagnostics on this path too (the #811 shape)."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "TERMINATED"}), "")],
        guest_attr_results=[
            GcloudRunResult(0, _guest_attr_payload("failed"), ""),
            GcloudRunResult(0, _guest_attr_kv("workload_started", "true"), ""),
            GcloudRunResult(0, _guest_attr_kv("persist", "timeout"), ""),
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "dead"
    assert pr.current_phase == "terminal_terminated"
    assert "eps/persist=timeout (instance TERMINATED)" in pr.log_tail_excerpt


def test_poll_running_window_breadcrumb_carries_in_flight_qualifier() -> None:
    """#1151 (statistics-lens Must-Fix): a RUNNING-window failed tick may
    catch a HEALTHY persist mid-flight, so the excerpt line self-discloses
    the instance status + an in-flight qualifier — a verbatim
    decision-table read of `attempted` here would misdiagnose a healthy
    persist as killed."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[
            GcloudRunResult(0, _guest_attr_payload("failed"), ""),
            GcloudRunResult(0, _guest_attr_kv("workload_started", "true"), ""),
            GcloudRunResult(0, _guest_attr_kv("persist", "attempted"), ""),
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "dead"
    assert pr.current_phase == "terminal_workload_failed"
    assert "eps/persist=attempted (instance RUNNING)" in pr.log_tail_excerpt
    assert "persist may be in flight" in pr.log_tail_excerpt


def test_poll_persist_breadcrumb_probe_failure_is_best_effort() -> None:
    """#1151: a failing eps/persist read NEVER raises and never gates
    classification — the excerpt reads ABSENT and the terminal
    classification is unchanged (diagnostic-only channel, deliberately
    unlike _guest_phase's typed GcpProbeError contract)."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "TERMINATED"}), "")],
        guest_attr_results=[
            GcloudRunResult(0, _guest_attr_payload("failed"), ""),
            GcloudRunResult(1, "", "guest attribute eps/workload_started not found"),
            GcloudRunResult(1, "", "ERROR: permission denied"),
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "dead"
    assert pr.current_phase == "terminal_setup_failed"
    assert "eps/persist=ABSENT (instance TERMINATED)" in pr.log_tail_excerpt


# ---------------------------------------------------------------------------
# fix23 — guest-attribute workload-phase overlay (success detection)
# ---------------------------------------------------------------------------


def _poll_handle():
    from explore_persona_space.backends.base import RunHandle

    return RunHandle(
        backend="gcp",
        cluster=None,
        job_id="1",
        pod_name="eps-issue-137",
        scratch_dir="/workspace/eps-issue-137",
        log_path="/workspace/logs/issue-137.log",
        extra={"zone": "us-central1-a"},
    )


def _guest_attr_payload(value: str) -> str:
    return json.dumps([{"namespace": "eps", "key": "phase", "value": value}])


def test_render_create_argv_enables_guest_attributes() -> None:
    """Without enable-guest-attributes the in-VM phase writes 403 and a
    successful workload is undetectable (issue 535 r9)."""
    argv = render_create_argv(
        spec=_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    joined = " ".join(argv)
    assert "enable-guest-attributes=TRUE" in joined


def test_render_startup_script_publishes_phase_guest_attribute() -> None:
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert "guest-attributes/eps/phase" in script
    # success path publishes done AFTER the sentinel write
    assert "_eps_phase done" in script
    assert script.index("EPS_SENTINEL_PATH") < script.index("_eps_phase done")
    # failure trap publishes failed before the poweroff
    assert "_eps_phase failed" in script
    # boot + workload milestones
    assert "_eps_phase startup" in script
    assert "_eps_phase workload" in script


def test_poll_running_with_done_phase_maps_to_done() -> None:
    """A RUNNING VM whose workload published phase=done is terminal
    SUCCESS — the harness proceeds to fetch_results + teardown instead
    of spinning to the hard timeout (issue 535 r9)."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("done"), "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "done"
    assert pr.current_phase == "workload_done"


def test_poll_running_with_failed_phase_maps_to_dead() -> None:
    """phase=failed (the EXIT trap's write) classifies dead even before
    the instance state flips to TERMINATED."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("failed"), "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "dead"


def test_poll_running_with_midrun_phase_stays_running() -> None:
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "running"
    assert pr.current_phase == "workload"


def test_poll_running_with_unreadable_phase_fails_soft_to_running() -> None:
    """The EXPECTED not-written-yet case (gcloud 404 / "not found" — the
    attribute does not exist until the startup-script's first write) must
    NOT false-kill a healthy VM — keep the coarse RUNNING classification
    and retry next tick. Only THIS case stays fail-soft; auth/API/parse
    failures are typed (tests below)."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(1, "", "attribute not found")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "running"


def test_poll_guest_attr_permission_denied_is_typed_probe_failure() -> None:
    """An auth/permission failure on the guest-attribute probe is NOT
    "phase not written yet" — pre-fix it returned "" and a finished
    workload spun to the outer poll timeout (round-2 Codex Major, task
    #535). It must surface as a typed stalled tick the consecutive-
    failure budget can see."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[
            GcloudRunResult(
                1,
                "",
                "ERROR: Required 'compute.instances.getGuestAttributes' permission denied",
            )
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "stalled"
    assert pr.current_phase == "guest_attr_probe_failed"


def test_poll_guest_attr_malformed_json_is_typed_probe_failure() -> None:
    """An rc=0 probe whose payload does not parse is a probe failure,
    not a phase read — typed stalled tick, never silent running."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, "{not json", "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "stalled"
    assert pr.current_phase == "guest_attr_probe_failed"


# ---------------------------------------------------------------------------
# issue #608 — poll-time sentinel drain via ssh sudo (root-owned VM tree)
# ---------------------------------------------------------------------------


def _drain_handle():
    """Poll handle WITH the ``issue`` extra (the drain's resolution key)."""
    from explore_persona_space.backends.base import RunHandle

    return RunHandle(
        backend="gcp",
        cluster=None,
        job_id="1",
        pod_name="eps-issue-137",
        scratch_dir="/workspace/eps-issue-137",
        log_path="/workspace/logs/issue-137.log",
        extra={"zone": "us-central1-a", "issue": 137},
    )


def _poll_pipeline_module():
    """Import the REAL ``scripts.poll_pipeline`` (the drain's lazy-import
    target) so tests can monkeypatch ``post_event`` on the same module
    object the backend resolves."""
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import scripts.poll_pipeline as pp

    return pp


def _drain_stdout(body: str, *, gate: str | None = None) -> str:
    payload: dict[str, Any] = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "note": body,
    }
    if gate:
        payload["gate"] = gate
    return (
        "SENTINEL_START /workspace/logs/issue-137-epm_results-1781214523.json\n"
        + json.dumps(payload)
        + "\nSENTINEL_END\n"
        + "EPS_LOGTAIL_START\n"
        + "eval shard 4/4 complete\n"
        + "EPS_LOGTAIL_END\n"
    )


def test_poll_running_drains_sentinels_via_sudo(monkeypatch) -> None:
    """A RUNNING tick drains root-owned ``/workspace/logs`` sentinels via
    ``sudo -n`` over gcloud ssh, posts the carried marker, renames the file
    ``.processed``, and reports an honest ``sentinels_processed`` count +
    log tail (incident #608: the GCP lane had NO drain, so a completed
    run's epm:results marker never posted and the poll JSON showed a
    silent ``sentinels_processed=0`` with an empty log tail)."""
    pp = _poll_pipeline_module()
    posted: list[tuple[int, str]] = []
    monkeypatch.setattr(pp, "post_event", lambda issue, kind, **kw: posted.append((issue, kind)))
    monkeypatch.setattr(pp, "list_events", lambda _issue: [])  # #1084 dedupe read stub
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("done"), "")],
        ssh_results=[
            GcloudRunResult(0, _drain_stdout("19/19 cells done"), ""),  # drain + tail
            GcloudRunResult(0, "", ""),  # mv -> .processed
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "done"
    assert pr.sentinels_processed == 1
    assert posted == [(137, "epm:results")]
    assert "eval shard 4/4 complete" in pr.log_tail_excerpt
    ssh_calls = [a for a in runner.calls if "ssh" in a and "compute" in a]
    assert len(ssh_calls) == 2
    drain_cmd = next(arg for arg in ssh_calls[0] if arg.startswith("--command="))
    assert "sudo -n bash -c" in drain_cmd, "drain must read root-owned files via sudo (#608)"
    mv_cmd = next(arg for arg in ssh_calls[1] if arg.startswith("--command="))
    assert "sudo -n mv -n" in mv_cmd
    assert ".processed" in mv_cmd


def test_poll_gcp_drain_scans_workload_root_fallback_glob(monkeypatch) -> None:
    """#610: the drain command must also glob the workload root's out_root
    logs dir — the issue-610 dispatcher found ``/workspace/logs`` missing,
    wrote its results sentinel under
    ``<workload_root>/eval_results/issue_<N>/logs/``, and every poll tick
    (including the done tick) reported ``sentinels_processed=0``."""
    pp = _poll_pipeline_module()
    monkeypatch.setattr(pp, "post_event", lambda issue, kind, **kw: None)
    monkeypatch.setattr(pp, "list_events", lambda _issue: [])  # #1084 dedupe read stub
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("done"), "")],
        ssh_results=[
            GcloudRunResult(0, _drain_stdout("19/19 cells done"), ""),  # drain + tail
            GcloudRunResult(0, "", ""),  # mv -> .processed
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    backend.poll(_drain_handle())
    ssh_calls = [a for a in runner.calls if "ssh" in a and "compute" in a]
    drain_cmd = next(arg for arg in ssh_calls[0] if arg.startswith("--command="))
    # Canonical glob AND the workload-root fallback, in one round-trip.
    assert "/workspace/logs/issue-137-*.json" in drain_cmd
    assert "/workspace/eps-issue-137/eval_results/issue_137/logs/issue-137-*.json" in drain_cmd


def test_poll_gcp_drain_transport_failure_is_loud() -> None:
    """A drain SSH/sudo failure must surface in the poll JSON (via
    ``log_tail_excerpt``), never read as a quiet ``sentinels_processed=0``."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
        ssh_results=[GcloudRunResult(1, "", "sudo: a password is required")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    assert pr.sentinels_processed == 0
    assert "gcp sentinel drain FAILED" in pr.log_tail_excerpt
    assert "sudo: a password is required" in pr.log_tail_excerpt


def test_gcp_drain_ssh_timeout_returns_transport_alarm() -> None:
    """#1084 (W2, GCP arm): a drain-list SSH that HANGS past the runner's
    per-call cap (``subprocess.TimeoutExpired`` — the #952 gcloud-ssh
    hostkey-drift wedge) degrades to the EXISTING transport alarm instead of
    an uncaught raise crashing the poll tick — same alarm tuple shape +
    ``alarm_class="transport"`` as the rc!=0 branch, so the #669 wedge-gate
    semantics (``reachability_alarm=True``) are inherited unchanged."""

    class _TimeoutOnSshRunner(_Runner):
        def __call__(self, argv):
            argv = list(argv)
            if "ssh" in argv and "compute" in argv:
                self.calls.append(argv)
                raise subprocess.TimeoutExpired(cmd=argv, timeout=300)
            return super().__call__(argv)

    runner = _TimeoutOnSshRunner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    assert pr.sentinels_processed == 0
    assert "timed out after 300" in pr.log_tail_excerpt
    assert "hung transport" in pr.log_tail_excerpt
    assert pr.reachability_alarm is True


def test_poll_gcp_drain_matched_but_empty_body_is_loud(monkeypatch) -> None:
    """A sentinel whose body reads back EMPTY (the pre-sudo permission
    symptom) must be reported loudly — glob matched, nothing processed."""
    pp = _poll_pipeline_module()
    posted: list[tuple[int, str]] = []
    monkeypatch.setattr(pp, "post_event", lambda issue, kind, **kw: posted.append((issue, kind)))
    stdout = (
        "SENTINEL_START /workspace/logs/issue-137-epm_results-1781214523.json\n"
        "SENTINEL_END\n"
        "EPS_LOGTAIL_START\n"
        "EPS_LOGTAIL_END\n"
    )
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("done"), "")],
        ssh_results=[GcloudRunResult(0, stdout, "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.sentinels_processed == 0
    assert posted == []
    assert "matched but 0 processed" in pr.log_tail_excerpt


def test_poll_gcp_drain_gate_sentinel_parks(monkeypatch) -> None:
    """A drained gate sentinel wins over the coarse status (mirrors
    poll_pipeline.poll_once): the orchestrator must park at the gate."""
    pp = _poll_pipeline_module()
    monkeypatch.setattr(pp, "post_event", lambda *a, **kw: None)
    monkeypatch.setattr(pp, "list_events", lambda _issue: [])  # #1084 dedupe read stub
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
        ssh_results=[
            GcloudRunResult(0, _drain_stdout("need a user answer", gate="fact_candidates"), ""),
            GcloudRunResult(0, "", ""),  # mv -> .processed
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "gate"
    assert pr.gate == "fact_candidates"
    assert pr.sentinels_processed == 1


def test_poll_handle_without_issue_skips_drain_loudly() -> None:
    """A handle missing the ``issue`` extra cannot resolve the sentinel
    glob — the drain is skipped with an explicit excerpt, not silently."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())  # legacy handle: extra has zone only
    assert pr.status == "running"
    assert pr.sentinels_processed == 0
    assert "drain SKIPPED" in pr.log_tail_excerpt
    # No ssh round-trip was attempted.
    assert not [a for a in runner.calls if "ssh" in a and "compute" in a]


# ---------------------------------------------------------------------------
# §7 lane extension — adaptive bg-poll interval on the GCP lane
# ---------------------------------------------------------------------------


def _describe_running(*, created_sec_ago: float | None) -> GcloudRunResult:
    """Scripted ``describe`` result: RUNNING, optionally with a
    ``creationTimestamp`` (the poll's run-age source — §7 early-run guard)."""
    payload: dict[str, Any] = {"status": "RUNNING"}
    if created_sec_ago is not None:
        created = datetime.now(UTC) - timedelta(seconds=created_sec_ago)
        payload["creationTimestamp"] = created.isoformat()
    return GcloudRunResult(0, json.dumps(payload), "")


# Clean drain stdout: no sentinels matched, normal log tail — alarm "".
_CLEAN_DRAIN_STDOUT = "EPS_LOGTAIL_START\nstep 500 loss=0.42\nEPS_LOGTAIL_END\n"


def test_poll_quiet_midrun_tick_emits_quiet_interval() -> None:
    """§7 lane extension: a RUNNING VM past the early-run window, in a
    known mid-workload phase, with a clean drain (no sentinels, no gate,
    no alarm) recommends the long quiet interval."""
    from explore_persona_space.backends.base import POLL_INTERVAL_QUIET_SEC

    runner = _Runner(
        describe_results=[_describe_running(created_sec_ago=7200)],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
        ssh_results=[GcloudRunResult(0, _CLEAN_DRAIN_STDOUT, "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    assert pr.sentinels_processed == 0
    assert pr.next_interval == POLL_INTERVAL_QUIET_SEC


def test_poll_early_run_instance_keeps_short_interval() -> None:
    """Inside the early-run window (instance younger than ~30 min) the
    tick stays short — early failures are the most valuable to catch fast."""
    from explore_persona_space.backends.base import POLL_INTERVAL_DEFAULT_SEC

    runner = _Runner(
        describe_results=[_describe_running(created_sec_ago=600)],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
        ssh_results=[GcloudRunResult(0, _CLEAN_DRAIN_STDOUT, "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    assert pr.next_interval == POLL_INTERVAL_DEFAULT_SEC


def test_poll_missing_creation_timestamp_keeps_short_interval() -> None:
    """An absent / unparseable ``creationTimestamp`` reads as unknown
    launch age → counts as early-run (fail toward coverage, not silence)."""
    from explore_persona_space.backends.base import POLL_INTERVAL_DEFAULT_SEC

    runner = _Runner(
        describe_results=[_describe_running(created_sec_ago=None)],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
        ssh_results=[GcloudRunResult(0, _CLEAN_DRAIN_STDOUT, "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    assert pr.next_interval == POLL_INTERVAL_DEFAULT_SEC


def test_poll_drain_gate_keeps_short_interval(monkeypatch) -> None:
    """A drained gate sentinel flips the merged status to ``gate`` — the
    quiet heuristic (computed post-drain) must keep the short interval."""
    from explore_persona_space.backends.base import POLL_INTERVAL_DEFAULT_SEC

    pp = _poll_pipeline_module()
    monkeypatch.setattr(pp, "post_event", lambda *a, **kw: None)
    monkeypatch.setattr(pp, "list_events", lambda _issue: [])  # #1084 dedupe read stub
    runner = _Runner(
        describe_results=[_describe_running(created_sec_ago=7200)],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
        ssh_results=[
            GcloudRunResult(0, _drain_stdout("need a user answer", gate="fact_candidates"), ""),
            GcloudRunResult(0, "", ""),  # mv -> .processed
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "gate"
    assert pr.next_interval == POLL_INTERVAL_DEFAULT_SEC


def test_poll_drain_alarm_keeps_short_interval() -> None:
    """A drain transport failure (alarm) is the lane anomaly — degraded
    observability never goes quiet, even past the early-run window."""
    from explore_persona_space.backends.base import POLL_INTERVAL_DEFAULT_SEC

    runner = _Runner(
        describe_results=[_describe_running(created_sec_ago=7200)],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
        ssh_results=[GcloudRunResult(1, "", "sudo: a password is required")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    assert pr.next_interval == POLL_INTERVAL_DEFAULT_SEC


def test_poll_booting_without_phase_keeps_short_interval() -> None:
    """A RUNNING VM whose ``eps/phase`` guest attribute is not yet written
    (booting) is ambiguous — only the known-mid-workload-phase branch may
    go quiet, regardless of instance age."""
    from explore_persona_space.backends.base import POLL_INTERVAL_DEFAULT_SEC

    runner = _Runner(
        describe_results=[_describe_running(created_sec_ago=7200)],
        # _Runner default: guest attribute not found (rc=1) -> phase "".
        ssh_results=[GcloudRunResult(0, _CLEAN_DRAIN_STDOUT, "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    assert pr.next_interval == POLL_INTERVAL_DEFAULT_SEC


# ---------------------------------------------------------------------------
# issue #588 — A2 byte-identity snapshot (hydra-only startup script)
# ---------------------------------------------------------------------------


def test_render_startup_script_hydra_only_byte_identical_to_pre_change_snapshot() -> None:
    """Accidental-drift pin for the hydra-only startup script.

    Originally the A2 (#588) byte-identity baseline recorded from the
    PRE-#588 renderer — that property (workload_cmd didn't change the
    hydra branch) was verified at #588 time and lives in git history.
    DELIBERATELY REGENERATED for #607's output-redirect change
    (provenance — source SHA + generation command + regeneration note —
    lives in the fixture's JSON header). The fixture's ongoing purpose is
    accidental-drift detection: any render change must arrive with a
    deliberate, provenance-documented regeneration. The #607 structural
    tests (redirect ordering, PIPE handler, EXIT-trap guards, TQDM
    export) keep this snapshot non-tautological.
    """
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "issue588_gcp_startup_hydra_only.json").read_text()
    )
    rendered = render_startup_script(
        spec=_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
        repo_branch="main",
    )
    assert rendered == fixture["rendered_text"]


# ---------------------------------------------------------------------------
# #750 — kernel OOM-kill guard on the success path (incident #744)
# ---------------------------------------------------------------------------


def test_render_startup_script_oom_guard_uses_proc_self_cgroup_derived_path() -> None:
    """#750: the ``_eps_oom_count`` helper resolves the workload's OWN cgroup
    dir from ``/proc/self/cgroup``'s ``0::`` unified-hierarchy line and reads
    the LOCAL ``oom_kill`` counter — NOT the bare ``/sys/fs/cgroup`` root,
    which is the v2 defect this fixes (``memory.events`` does not exist on the
    cgroup-v2 root). Shared between the hydra and workload_cmd branches, so
    one canonical spec covers both."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert "_eps_oom_count() {" in script
    # The 0::-line resolver (the cgroup-v2 unified-hierarchy path for this pid).
    assert '$1=="0"{print $3' in script
    # The LOCAL counter file (this cgroup only, NOT a descendant aggregation).
    assert "memory.events.local" in script
    # The derived-path composition (mount root + the 0:: relative path).
    assert '"/sys/fs/cgroup${_rel}"' in script
    # The integer extractor for the oom_kill line.
    assert '$1=="oom_kill"{print $2' in script
    # v2-regression guard: the bare root read the v2 guard wrongly used must
    # be ABSENT — the only /sys/fs/cgroup occurrence is the derived
    # composition, never a standalone ``/sys/fs/cgroup/memory.events`` target.
    assert " /sys/fs/cgroup/memory.events " not in script
    assert "/sys/fs/cgroup/memory.events\n" not in script
    # The v2 grep-[1-9] form is replaced by the arithmetic diff.
    assert "^oom_kill [1-9]" not in script


def test_render_startup_script_oom_baseline_captured_before_workload() -> None:
    """The pre-workload baseline-capture step renders BEFORE the workload
    phase, so the post-workload guard fires on an INCREASE (the counter is
    cumulative-since-cgroup-creation; the guest-startup cgroup is not fresh)."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert 'EPS_OOM_BASELINE="$(_eps_oom_count)"' in script
    assert script.index('EPS_OOM_BASELINE="$(_eps_oom_count)"') < script.index(
        "_eps_phase workload"
    )


def test_render_startup_script_oom_guard_diffs_after_workload_before_done() -> None:
    """The post-workload diff guard re-reads the counter, compares against the
    baseline with a numeric ``-gt``, and ``exit 137`` (SIGKILL rc) short-
    circuits the success tail BEFORE the watchdog reaper, the completion
    sentinel, and ``_eps_phase done`` — so an OOM'd run routes through the EXIT
    trap's failure branch instead of falsely publishing done (the #744 bug)."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert 'EPS_OOM_FINAL="$(_eps_oom_count)"' in script
    assert '[ "${EPS_OOM_FINAL:-0}" -gt "${EPS_OOM_BASELINE:-0}" ]' in script
    assert "exit 137" in script
    # The final read is AFTER the workload phase ...
    assert script.index('EPS_OOM_FINAL="$(_eps_oom_count)"') > script.index("_eps_phase workload")
    # ... and the exit short-circuits BEFORE the sentinel + done.
    assert script.index("exit 137") < script.index("_eps_phase done")
    assert script.index("exit 137") < script.index('{"phase":"done"')
    # The guard precedes the clean-exit watchdog reaper (so an OOM'd run lets
    # the EXIT trap own teardown, not the clean reaper). #854 added a SECOND
    # kill earlier in the script (at EXIT-trap entry), so anchor on the
    # STANDALONE clean-exit reap line, not the first `kill` substring.
    clean_reap = '\n{ kill "${EPS_WATCHDOG_PID:-}" 2>/dev/null; } || true\n'
    assert script.index('EPS_OOM_FINAL="$(_eps_oom_count)"') < script.index(clean_reap)


def test_render_startup_script_oom_guard_on_both_workload_shapes() -> None:
    """The baseline + guard live in the SHARED success path, so BOTH the hydra
    (train.py) and the workload_cmd branches get them, with the same
    baseline-before-workload / guard-after-workload-before-done ordering."""
    hydra = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    workload = render_startup_script(
        spec=_workload_spec("bash scripts/issue750_smoke.sh"),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    for script in (hydra, workload):
        assert "_eps_oom_count() {" in script
        # baseline captured before the workload phase
        assert script.index('EPS_OOM_BASELINE="$(_eps_oom_count)"') < script.index(
            "_eps_phase workload"
        )
        # guard fires after the workload, before done
        assert script.index('EPS_OOM_FINAL="$(_eps_oom_count)"') > script.index(
            "_eps_phase workload"
        )
        assert script.index("exit 137") < script.index("_eps_phase done")


def test_render_startup_script_oom_guard_is_valid_bash_both_branches() -> None:
    """The OOM helper + baseline + diff guard must parse on both branches —
    a quoting slip in the awk/case lines would only surface at VM-boot."""
    import tempfile

    for spec in (
        _spec(),
        _workload_spec("bash scripts/issue750_smoke.sh --flag 'v 1'"),
    ):
        script = render_startup_script(spec=spec, config=_test_config(), attempt_id="att-fixed-001")
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
            fh.write(script)
            path = fh.name
        proc = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
        assert proc.returncode == 0, f"bash -n failed:\n{proc.stderr}"


def _extract_oom_helper(script: str) -> str:
    """Pull the verbatim ``_eps_oom_count() { ... }`` function body out of a
    rendered startup script for direct bash execution."""
    import re

    m = re.search(r"(_eps_oom_count\(\) \{.*?\n\})", script, re.DOTALL)
    assert m is not None, "could not locate _eps_oom_count helper in rendered script"
    return m.group(1)


def test_oom_guard_fires_exit_137_on_counter_increase() -> None:
    """Runtime regression for the #744 silent-failure invariant: a STRICT
    INCREASE in the cgroup-local ``oom_kill`` counter across the workload's
    lifetime trips the guard and ``exit 137`` (so an OOM'd run can never reach
    ``_eps_phase done``). Drives the rendered helper against a temp cgroup-dir
    whose ``memory.events.local`` is bumped 0 -> 3, exactly the #744 case.
    This test FAILS pre-fix (no helper / no guard) and PASSES post-fix."""
    import tempfile

    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    helper = _extract_oom_helper(script)
    with tempfile.TemporaryDirectory() as d:
        events = Path(d) / "memory.events.local"
        # A reader that resolves to OUR temp dir instead of the real cgroup:
        # override _eps_oom_count's $_dir derivation by reading the file
        # directly through the same awk + case normalization the helper uses.
        read = (
            f'_dir="{d}"; _n=0; '
            f'if [ -r "$_dir/memory.events.local" ]; then '
            f'_n="$(awk \'$1=="oom_kill"{{print $2; exit}}\' '
            f'"$_dir/memory.events.local" 2>/dev/null || true)"; fi; '
            f'case "$_n" in (*[!0-9]*|"") _n=0 ;; esac; echo "$_n"'
        )
        events.write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")
        baseline = subprocess.run(
            ["bash", "-c", read], capture_output=True, text=True
        ).stdout.strip()
        events.write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 3\n")
        final = subprocess.run(["bash", "-c", read], capture_output=True, text=True).stdout.strip()
        assert baseline == "0" and final == "3", (baseline, final)
        # The guard line, verbatim from the renderer, with the simulated reads.
        guard = (
            f"set -euo pipefail\n"
            f'EPS_OOM_BASELINE="{baseline}"\nEPS_OOM_FINAL="{final}"\n'
            f'if [ "${{EPS_OOM_FINAL:-0}}" -gt "${{EPS_OOM_BASELINE:-0}}" ]; then exit 137; fi\n'
            f"echo DONE"
        )
        proc = subprocess.run(["bash", "-c", guard], capture_output=True, text=True)
        assert proc.returncode == 137, (proc.returncode, proc.stdout, proc.stderr)
        assert "DONE" not in proc.stdout
    # ... and the helper is set -e-safe + fails CLOSED (echoes 0) when the
    # cgroup-local counter file is absent (non-v2 / hybrid host) -> 0 -gt 0 is
    # false -> the guard is a no-op and pre-change behavior is preserved.
    harness = (
        helper + '\nset -euo pipefail\nb="$(_eps_oom_count)"; f="$(_eps_oom_count)"; '
        'if [ "${f:-0}" -gt "${b:-0}" ]; then echo FIRED; else echo NOFIRE; fi'
    )
    proc = subprocess.run(["bash", "-c", harness], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "NOFIRE", proc.stdout


# ---------------------------------------------------------------------------
# issue #588 — custom workload_cmd rendering + validation + launch
# ---------------------------------------------------------------------------


def _workload_spec(cmd: str = "bash scripts/issue588_smoke.sh") -> RunSpec:
    """``_spec()`` twin carrying a custom workload_cmd (no hydra args)."""
    return _spec(hydra_args=(), workload_cmd=cmd)


def _wrapped(cmd: str) -> str:
    """The #1004 rc-preserving rendered workload line for ``cmd``."""
    return f"bash -eu -o pipefail -c {shlex.quote(cmd)}"


def test_render_startup_script_workload_cmd_rc_wrapped_with_lifecycle_intact() -> None:
    """#588/#1004: the custom command replaces ONLY the workload line —
    every lifecycle pin (secrets fetch, in-VM preflight, phase publishing,
    EXIT trap, completion sentinel) is unchanged. The command renders
    inside the #1004 rc-preserving inner-bash wrapper, never as a bare
    line."""
    script = render_startup_script(
        spec=_workload_spec("bash scripts/issue588_smoke.sh --flag 'v 1'"),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    lines = script.splitlines()
    # The command is the single shlex-quoted argument of an inner
    # `bash -eu -o pipefail -c` (#1004); the inner bash re-parses it as a
    # full shell line, so the #588 "complete shell command" contract holds.
    assert _wrapped("bash scripts/issue588_smoke.sh --flag 'v 1'") in lines
    # The bare, rc-masking pre-#1004 form is GONE.
    assert "bash scripts/issue588_smoke.sh --flag 'v 1'" not in lines
    assert "# === Run the workload (custom workload_cmd) ===" in lines
    # The hardcoded hydra entrypoint is GONE on the custom path.
    assert "scripts/train.py" not in script
    # Lifecycle pins (same set the hydra-path golden test asserts).
    assert "_eps_phase workload" in lines
    assert "_eps_phase done" in lines
    assert "trap 'rc=$?" in script  # EXIT trap bounds billing
    assert '{"phase":"done","issue":137' in script  # completion sentinel
    assert "Metadata-Flavor: Google" in script  # secrets fetch stanza
    for key in REQUIRED_LAUNCH_SECRET_KEYS:
        assert f"[FAIL] {key} missing from instance metadata" in script
    # The custom command runs AFTER cd "$WORKLOAD_ROOT" (repo-relative
    # `bash scripts/...` must resolve).
    assert lines.index('cd "$WORKLOAD_ROOT"') < lines.index(
        _wrapped("bash scripts/issue588_smoke.sh --flag 'v 1'")
    )
    # WandB project default (#601 follow-up r1): exported BEFORE the
    # workload so HF-Trainer runs stop landing in the global default
    # 'huggingface' project; :- keeps an inline/internal override winning.
    wandb_export = 'export WANDB_PROJECT="${WANDB_PROJECT:-issue137}"'
    assert wandb_export in lines
    assert lines.index(wandb_export) < lines.index(
        _wrapped("bash scripts/issue588_smoke.sh --flag 'v 1'")
    )
    # The hydra branch must NOT gain the export (byte-pinned by the #588
    # snapshot fixture; asserted here for a readable failure too).
    hydra_script = render_startup_script(
        spec=_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    assert "WANDB_PROJECT" not in hydra_script


def test_render_startup_script_workload_cmd_exports_repo_root() -> None:
    """#641 (trap #599): the workload_cmd branch exports
    ``REPO_ROOT="$WORKLOAD_ROOT"`` BEFORE the workload runs, so a driver
    that defaults ``REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"``
    resolves to the on-VM clone path instead of `cd`-ing to the
    nonexistent RunPod fallback under set -e (which fires the EXIT trap →
    VM power-off → a burned GPU instance). No orchestrator-side
    --workload-cmd prefix is then required."""
    script = render_startup_script(
        spec=_workload_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    lines = script.splitlines()
    repo_root_export = 'export REPO_ROOT="$WORKLOAD_ROOT"'
    assert repo_root_export in lines
    # Exported AFTER the shared ``cd "$WORKLOAD_ROOT"`` and BEFORE the
    # workload command, so the workload subprocess inherits it.
    assert lines.index('cd "$WORKLOAD_ROOT"') < lines.index(repo_root_export)
    assert lines.index(repo_root_export) < lines.index(_wrapped("bash scripts/issue588_smoke.sh"))
    # The hydra branch must NOT gain the export — it runs scripts/train.py
    # directly (no REPO_ROOT dependency), and the #588 byte-identity
    # snapshot fixture pins the hydra branch unchanged.
    hydra_script = render_startup_script(
        spec=_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    assert "REPO_ROOT" not in hydra_script


def test_render_startup_script_workload_cmd_exports_pythonpath() -> None:
    """#1172 (trap #823/#853): the workload_cmd branch exports exactly one
    repo-root PYTHONPATH prepend (``$WORKLOAD_ROOT`` first, inherited value
    appended via the nounset-exempt ``:+`` form) BEFORE the workload runs,
    so a deferred script-mode ``from scripts.X import ...`` in a src-layout
    driver resolves instead of dying mid-run with ModuleNotFoundError. The
    hydra branch must NOT gain it (the #588 byte-identity snapshot pins the
    hydra render; ``scripts/train.py`` has no ``scripts.*`` imports) —
    the mirror of the #641 ``REPO_ROOT`` hydra-branch assert above."""
    script = render_startup_script(
        spec=_workload_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    lines = script.splitlines()
    pythonpath_export = 'export PYTHONPATH="$WORKLOAD_ROOT${PYTHONPATH:+:$PYTHONPATH}"'
    assert lines.count(pythonpath_export) == 1
    # Positioned after the #641 REPO_ROOT export and before the wrapped
    # workload command, so the workload subprocess inherits it.
    assert lines.index('export REPO_ROOT="$WORKLOAD_ROOT"') < lines.index(pythonpath_export)
    assert lines.index(pythonpath_export) < lines.index(_wrapped("bash scripts/issue588_smoke.sh"))
    hydra_script = render_startup_script(
        spec=_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    assert "PYTHONPATH" not in hydra_script


def test_render_startup_script_workload_cmd_waits_on_detached_pid_files() -> None:
    """#601: a self-daemonizing workload_cmd (setsid-forked driver)
    returns immediately — the script must wait on fresh
    ``/workspace/logs/*.pid`` files BEFORE writing the completion
    sentinel, or the poll reads terminal-success minutes into a
    multi-hour run (eps-issue-601 follow-up r1, 2026-06-12)."""
    script = render_startup_script(
        spec=_workload_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    lines = script.splitlines()
    assert "touch /tmp/eps-workload-start" in lines
    wait_for = next(line for line in lines if line.startswith("for pf in $(find /workspace/logs"))
    # Only pid files NEWER than the workload start count (stale files
    # from prior attempts are skipped); a missing logs dir is benign.
    assert "-newer /tmp/eps-workload-start" in wait_for
    assert "2>/dev/null || true" in wait_for
    assert '  while kill -0 "$wpid" 2>/dev/null; do sleep 30; done' in lines
    # Ordering: start-marker touch < workload cmd < pid-wait loop <
    # sentinel write < phase=done publish.
    i_touch = lines.index("touch /tmp/eps-workload-start")
    i_cmd = lines.index(_wrapped("bash scripts/issue588_smoke.sh"))
    i_wait = lines.index(wait_for)
    i_sentinel = next(i for i, line in enumerate(lines) if line.startswith("cat > "))
    i_done = lines.index("_eps_phase done")
    assert i_touch < i_cmd < i_wait < i_sentinel < i_done
    # The hydra branch is blocking by construction (in-process
    # scripts/train.py) — no wait block there (the #588 byte-identity
    # snapshot also pins this).
    hydra_script = render_startup_script(
        spec=_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    assert "eps-workload-start" not in hydra_script


def test_render_startup_script_workload_cmd_precreates_drain_logs_dir() -> None:
    """#610: the workload_cmd branch must pre-create ``/workspace/logs``
    (world-writable — umask 077 is active) BEFORE the workload runs, so
    dispatchers can write drain sentinels + detach pid files at the
    canonical pod-side-signaling path. The issue-610 dispatcher found the
    dir missing, fell back to its out_root logs dir, and the poll's drain
    never saw the results sentinel."""
    script = render_startup_script(
        spec=_workload_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    lines = script.splitlines()
    assert "mkdir -p /workspace/logs" in lines
    assert "chmod 777 /workspace/logs" in lines
    # Ordering: dir exists before the workload command runs.
    assert lines.index("mkdir -p /workspace/logs") < lines.index(
        _wrapped("bash scripts/issue588_smoke.sh")
    )
    # The hydra branch must NOT gain the #610 sentinel-drain stanza.
    # Discriminate on its unique `chmod 777` line: as of #607 BOTH
    # branches carry a common-prelude `mkdir -p /workspace/logs` (the
    # output-redirect block creates the log dir), so the bare mkdir no
    # longer distinguishes the #610 stanza.
    hydra_script = render_startup_script(
        spec=_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    assert "chmod 777 /workspace/logs" not in hydra_script


def test_render_startup_script_workload_cmd_compound_is_rc_wrapped() -> None:
    """#1004 (incident #952): a compound && workload_cmd renders inside the
    rc-preserving inner-bash wrapper, never as a bare line — a bare splice
    under set -e rc-masks a first-command crash into a false phase=done
    (errexit exempts non-final &&/|| list members)."""
    cmd = "uv run python scripts/a.py && uv run python scripts/b.py"
    script = render_startup_script(
        spec=_workload_spec(cmd), config=_test_config(), attempt_id="att-fixed-001"
    )
    lines = script.splitlines()
    assert _wrapped(cmd) in lines
    assert cmd not in lines  # the bare, rc-masking form is GONE
    # Wrapper sits between the start-marker touch and the pid-wait loop.
    assert lines.index("touch /tmp/eps-workload-start") < lines.index(_wrapped(cmd))
    i_wait = next(i for i, ln in enumerate(lines) if ln.startswith("for pf in $(find"))
    assert lines.index(_wrapped(cmd)) < i_wait


def _rendered_workload_line(cmd: str) -> str:
    """Extract the ACTUAL rendered workload line (the line after the
    start-marker touch) from a real ``render_startup_script`` output —
    live-dispatched-path discipline: never rebuild the line via a twin."""
    script = render_startup_script(
        spec=_workload_spec(cmd), config=_test_config(), attempt_id="att-fixed-001"
    )
    lines = script.splitlines()
    return lines[lines.index("touch /tmp/eps-workload-start") + 1]


def test_workload_cmd_wrapper_propagates_compound_first_command_rc() -> None:
    """#1004 behavioral proof on the LIVE rendered line: under set -euo
    pipefail, a `cmd1 && cmd2` workload whose FIRST command exits 7
    aborts the script with rc 7 (success tail unreached); a succeeding
    compound reaches the tail with rc 0. Pre-#1004 the failing case fell
    through and published done (probe: bash -c 'set -euo pipefail;
    false && true; echo FELL_THROUGH' prints FELL_THROUGH, rc=0)."""
    for cmd, want_rc, want_tail in (
        ("bash -c 'exit 7' && echo NOT_REACHED", 7, False),
        ("true && echo CHAIN_OK", 0, True),
    ):
        harness = "set -euo pipefail\n" + _rendered_workload_line(cmd) + "\necho SUCCESS_TAIL\n"
        proc = subprocess.run(["bash", "-c", harness], capture_output=True, text=True)
        assert proc.returncode == want_rc, (cmd, proc.returncode, proc.stderr)
        assert ("SUCCESS_TAIL" in proc.stdout) is want_tail
        assert "NOT_REACHED" not in proc.stdout


def test_workload_cmd_wrapper_inherits_exported_env() -> None:
    """#1004: exported env (the #641 REPO_ROOT / #601 WANDB_PROJECT
    contract) reaches the inner bash — POSIX process inheritance holds
    through the wrapper, so the export stanzas rendered BEFORE the
    workload line keep working unchanged."""
    cmd = 'test "$EPS_PROBE_VAR" = ok'
    harness = (
        "set -euo pipefail\nexport EPS_PROBE_VAR=ok\n"
        + _rendered_workload_line(cmd)
        + "\necho ENV_OK\n"
    )
    proc = subprocess.run(["bash", "-c", harness], capture_output=True, text=True)
    assert proc.returncode == 0, (proc.returncode, proc.stderr)
    assert "ENV_OK" in proc.stdout


def test_render_startup_script_neither_workload_nor_hydra_raises_571() -> None:
    """#588 defense-in-depth: a bare ``scripts/train.py`` render is the
    exact incident-#571 crash — refuse BEFORE any gcloud create."""
    with pytest.raises(ValueError, match="incident #571"):
        render_startup_script(
            spec=_spec(hydra_args=()),
            config=_test_config(),
            attempt_id="att-fixed-001",
        )


def test_render_startup_script_both_set_via_hydra_args_override_raises() -> None:
    """The ``hydra_args`` parameter override on a workload_cmd spec is
    the one both-set path ``RunSpec.__post_init__`` cannot see — the
    renderer must catch it."""
    with pytest.raises(ValueError, match="workload_cmd and hydra_args both set"):
        render_startup_script(
            spec=_workload_spec(),
            config=_test_config(),
            attempt_id="att-fixed-001",
            hydra_args=("seed=1",),
        )


def test_launch_workload_cmd_spec_provisions_and_marker_says_custom() -> None:
    """#588: ``launch`` has NO behavior branch for workload_cmd specs —
    it provisions normally; the ``epm:cluster-launched`` marker gains
    the additive ``workload: custom`` field."""
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "112233"}])
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[GcloudRunResult(0, created_payload, "")],
    )
    posted: list[dict] = []
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **kwargs: posted.append(kwargs),
    )
    handle = backend.launch(_workload_spec())
    assert handle.backend == "gcp"
    assert handle.pod_name == "eps-issue-137"
    assert any("create" in argv for argv in runner.calls)
    # The startup script gcloud received embeds the custom command.
    assert len(posted) == 1
    body = json.loads(posted[0]["note"])
    assert body["workload"] == "custom"


def test_launch_hydra_spec_marker_says_hydra() -> None:
    """The additive marker field reads ``hydra`` on the standard path."""
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "112233"}])
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[GcloudRunResult(0, created_payload, "")],
    )
    posted: list[dict] = []
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **kwargs: posted.append(kwargs),
    )
    backend.launch(_spec())
    body = json.loads(posted[0]["note"])
    assert body["workload"] == "hydra"


# ---------------------------------------------------------------------------
# issue #588 round 2 — fetch_results sentinel pull (ssh sudo cat, not scp)
# ---------------------------------------------------------------------------


def _b64_tar(members: dict[str, str]) -> str:
    """Build the base64-encoded tar stream the #790 best-effort pull returns.

    ``members`` maps a repo-relative path (e.g. ``eval_results/issue_588/x.json``)
    to its text content. The remote command is
    ``tar -c -C <workload_root>/eval_results issue_588 | base64 -w0``, so the tar
    entry names are relative to the subdir's PARENT (``issue_588/x.json``) — the
    same shape ``extractall(path=local_parent)`` lands under ``repo/eval_results/``.
    Returns the ASCII base64 string ``GcloudRunResult.stdout`` would carry.
    """
    import base64 as _b64
    import io as _io
    import os as _os
    import tarfile as _tarfile

    buf = _io.BytesIO()
    with _tarfile.open(fileobj=buf, mode="w") as tf:
        for rel_path, content in members.items():
            # Strip the leading top-level dir so the arcname is
            # ``issue_<N>/<leaf>`` (relative to the tar's -C parent), matching
            # what `tar -c -C <parent> <leaf>` produces on the remote.
            arcname = _os.path.join(*rel_path.split("/")[1:])
            data = content.encode("utf-8")
            info = _tarfile.TarInfo(name=arcname)
            info.size = len(data)
            tf.addfile(info, _io.BytesIO(data))
    return _b64.b64encode(buf.getvalue()).decode("ascii")


def _fetch_fixture(
    tmp_path: Path, monkeypatch, *, ssh_results: list[GcloudRunResult]
) -> tuple[GcpBackend, _Runner, GcpConfig, Any, str]:
    """Shared rig for the fetch_results tests.

    Points ``vm_scratch_dir`` at tmp so the local sentinel write lands
    under tmp, and the best-effort dir pulls' mkdir at a tmp repo root.
    Returns (backend, runner, config, handle, sentinel_abs).
    """
    from explore_persona_space.backends.base import RunHandle

    config = replace(_test_config(), vm_scratch_dir=str(tmp_path / "vm"))
    monkeypatch.setattr(
        "explore_persona_space.backends.gcp._default_src_root_for_fetch",
        lambda: tmp_path / "repo",
    )
    runner = _Runner(ssh_results=ssh_results)
    backend = GcpBackend(config=config, runner=runner, marker_poster=lambda **_: None)
    handle = RunHandle(
        backend="gcp",
        cluster=None,
        job_id="1",
        pod_name="eps-issue-588",
        scratch_dir=f"{config.vm_scratch_dir}/eps-issue-588",
        log_path=f"{config.vm_scratch_dir}/logs/issue-588.log",
        extra={"zone": "us-central1-a", "issue": 588, "attempt_id": "att-001"},
    )
    sentinel_abs = sentinel_path_for(config, 588, "att-001")
    return backend, runner, config, handle, sentinel_abs


def test_fetch_results_sentinel_pull_uses_ssh_sudo_cat(tmp_path: Path, monkeypatch) -> None:
    """The MANDATORY sentinel pull is `gcloud compute ssh ... sudo -n cat`, not scp.

    The GCE startup-script runs as root, so the workload tree is root-
    owned and the OS-Login scp user gets `Permission denied` (live
    finding, att-20260611-064703). The captured stdout must land
    verbatim at the SAME local path the artifact declaration claims.
    """
    import shlex

    sentinel_text = '{"phase": "done", "issue": 588, "attempt_id": "att-001"}\n'
    backend, runner, config, handle, sentinel_abs = _fetch_fixture(
        tmp_path,
        monkeypatch,
        # 1 sentinel cat + 2 best-effort tar pulls (all ssh now, #790).
        ssh_results=[
            GcloudRunResult(0, sentinel_text, ""),
            GcloudRunResult(0, _b64_tar({"eval_results/issue_588/x.json": "{}"}), ""),
            GcloudRunResult(0, _b64_tar({"figures/issue_588/y.png": "png"}), ""),
        ],
    )
    backend.fetch_results(handle)

    ssh_calls = [argv for argv in runner.calls if "ssh" in argv]
    # 1 sentinel cat + 2 best-effort tar pulls; NO scp calls (#790).
    assert len(ssh_calls) == 3
    assert ssh_calls[0] == [
        "gcloud",
        "compute",
        "ssh",
        "eps-issue-588",
        f"--command=sudo -n cat {shlex.quote(sentinel_abs)}",
        f"--configuration={config.gcloud_config}",
        f"--project={config.project}",
        "--zone=us-central1-a",
    ]
    # Captured stdout written verbatim to the declaration's local path.
    assert Path(sentinel_abs).read_text() == sentinel_text
    # The best-effort dir pulls are ssh `sudo -n bash -o pipefail -c 'tar ... | base64'`,
    # NOT scp; the mirror lands under the tmp repo root.
    scp_calls = [argv for argv in runner.calls if "scp" in argv]
    assert len(scp_calls) == 0


def test_fetch_results_sentinel_pull_failure_logs_and_continues(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """A failed sentinel pull logs loud, does NOT raise, and does NOT
    block the best-effort dir pulls; no local sentinel file is written
    (confirm_artifacts then FAILs on the missing file — the intended
    surfacing)."""
    import logging

    backend, runner, _config, handle, sentinel_abs = _fetch_fixture(
        tmp_path,
        monkeypatch,
        # Failed sentinel cat + 2 best-effort tar pulls (all ssh now, #790).
        ssh_results=[
            GcloudRunResult(1, "", "sudo: a password is required"),
            GcloudRunResult(0, _b64_tar({"eval_results/issue_588/x.json": "{}"}), ""),
            GcloudRunResult(0, _b64_tar({"figures/issue_588/y.png": "png"}), ""),
        ],
    )
    with caplog.at_level(logging.ERROR):
        backend.fetch_results(handle)  # must not raise

    assert not Path(sentinel_abs).exists()
    assert "confirm_artifacts will FAIL" in caplog.text
    ssh_calls = [argv for argv in runner.calls if "ssh" in argv]
    assert len(ssh_calls) == 3  # sentinel + 2 best-effort pulls still attempted
    scp_calls = [argv for argv in runner.calls if "scp" in argv]
    assert len(scp_calls) == 0  # best-effort pulls are ssh tar now (#790)


def test_fetch_results_best_effort_dirs_use_sudo_tar(tmp_path: Path, monkeypatch) -> None:
    """The best-effort dir pulls are `ssh ... sudo -n bash -o pipefail -c 'tar | base64'`.

    The workload tree is root-owned (#588), so a plain scp Permission-denies
    and the local mirror silently stays empty. The #790 fix pulls each dir as
    a base64-encoded tar stream via `sudo -n` — the same grant the sentinel
    pull uses — and extracts it under the local repo root. Positive functional
    evidence: assert the extracted files land at
    `repo/eval_results/issue_588/` + `repo/figures/issue_588/`, that the
    transport is ssh (no scp), and that the remote command is the exact
    pipefail-wrapped tar-base64 pipeline.
    """
    import shlex

    from explore_persona_space.backends.gcp import workload_dir_for

    sentinel_text = '{"phase": "done", "issue": 588, "attempt_id": "att-001"}\n'
    backend, runner, config, handle, _sentinel_abs = _fetch_fixture(
        tmp_path,
        monkeypatch,
        ssh_results=[
            GcloudRunResult(0, sentinel_text, ""),
            GcloudRunResult(
                0, _b64_tar({"eval_results/issue_588/run_result.json": '{"ok": 1}'}), ""
            ),
            GcloudRunResult(0, _b64_tar({"figures/issue_588/bar.png": "PNGDATA"}), ""),
        ],
    )
    backend.fetch_results(handle)

    # No scp at all; 1 sentinel cat + 2 tar pulls, all ssh.
    scp_calls = [argv for argv in runner.calls if "scp" in argv]
    assert len(scp_calls) == 0
    ssh_calls = [argv for argv in runner.calls if "ssh" in argv]
    assert len(ssh_calls) == 3

    workload_root = workload_dir_for(config, 588)
    tar_calls = ssh_calls[1:]  # the 2 best-effort dir pulls
    eval_cmd = f"tar -c -C {shlex.quote(f'{workload_root}/eval_results')} issue_588 | base64 -w0"
    fig_cmd = f"tar -c -C {shlex.quote(f'{workload_root}/figures')} issue_588 | base64 -w0"
    assert tar_calls[0] == [
        "gcloud",
        "compute",
        "ssh",
        "eps-issue-588",
        f"--command=sudo -n bash -o pipefail -c {shlex.quote(eval_cmd)}",
        f"--configuration={config.gcloud_config}",
        f"--project={config.project}",
        "--zone=us-central1-a",
    ]
    assert tar_calls[1][4] == f"--command=sudo -n bash -o pipefail -c {shlex.quote(fig_cmd)}"

    # The captured tar streams extract under the tmp repo root.
    repo_root = tmp_path / "repo"
    assert (repo_root / "eval_results/issue_588/run_result.json").read_text() == '{"ok": 1}'
    assert (repo_root / "figures/issue_588/bar.png").read_text() == "PNGDATA"


def test_fetch_results_best_effort_dir_missing_is_non_fatal(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """A missing remote dir (tar rc != 0) logs a WARNING and does NOT raise.

    This is the pipefail regression pin: the remote pipeline is wrapped in
    `bash -o pipefail` precisely so a missing-dir `tar` failure propagates
    through the `| base64` pipe instead of `base64 -w0` masking it with rc 0.
    With pipefail the rc-guard fires (log + continue); WITHOUT it the pull
    would return empty bytes and crash the local `tarfile.open`. The
    missing-dir path is the COMMON case once item-4 removes figures/ from the
    gate, so it must be non-fatal.
    """
    import logging

    sentinel_text = '{"phase": "done", "issue": 588, "attempt_id": "att-001"}\n'
    backend, _runner, _config, handle, sentinel_abs = _fetch_fixture(
        tmp_path,
        monkeypatch,
        ssh_results=[
            GcloudRunResult(0, sentinel_text, ""),
            # eval_results present; figures/ missing (tar rc != 0 via pipefail).
            GcloudRunResult(0, _b64_tar({"eval_results/issue_588/x.json": "{}"}), ""),
            GcloudRunResult(1, "", "tar: issue_588: Cannot stat: No such file or directory"),
        ],
    )
    with caplog.at_level(logging.WARNING):
        backend.fetch_results(handle)  # must not raise

    # The present dir still landed; the missing one logged + continued.
    assert Path(sentinel_abs).read_text() == sentinel_text
    assert (tmp_path / "repo/eval_results/issue_588/x.json").read_text() == "{}"
    assert not (tmp_path / "repo/figures/issue_588").exists()
    assert "best-effort sudo tar" in caplog.text
    assert "figures/issue_588" in caplog.text


def test_fetch_results_missing_attempt_id_returns_without_gcloud_calls(
    tmp_path: Path, monkeypatch
) -> None:
    """Without an attempt_id the sentinel path is unknowable: log + return,
    zero gcloud invocations (no half-formed scp/ssh against the VM)."""
    backend, runner, _config, handle, _sentinel_abs = _fetch_fixture(
        tmp_path, monkeypatch, ssh_results=[]
    )
    handle.extra.pop("attempt_id")
    backend.fetch_results(handle)
    assert runner.calls == []


# ---------------------------------------------------------------------------
# incident #612 — relaunch-follow: a terminal guest-attribute phase must not
# mask an SSH-relaunched workload named by a fresh epm:run-launched marker
# ---------------------------------------------------------------------------


_RELAUNCH_NOTE = (
    "RELAUNCH after G2 yield halt + hot-fix abc1234. pod=eps-issue-137 pid=4610 "
    "log_abs=/workspace/eps-issue-137/logs/issue-137.log cmd='dispatch.py --cells all'"
)
_EMPTY_DRAIN_STDOUT = "EPS_LOGTAIL_START\nEPS_LOGTAIL_END\n"


def _relaunch_reader(
    *,
    run_ts: str | None = "2026-06-12T06:01:09Z",
    cluster_ts: str | None = "2026-06-12T05:31:52Z",
    note: str = _RELAUNCH_NOTE,
):
    """Fake marker reader: scripted latest run-launched / cluster-launched."""

    def reader(issue: int, prefix: str | None = None):
        assert issue == 137
        if prefix == "epm:run-launched" and run_ts is not None:
            return {"ts": run_ts, "kind": "epm:run-launched", "version": 1, "note": note}
        if prefix == "epm:cluster-launched" and cluster_ts is not None:
            return {"ts": cluster_ts, "kind": "epm:cluster-launched", "version": 1, "note": "{}"}
        return None

    return reader


def _probe_stdout(*, alive: bool, mtime: int = 1718000000, now: int = 1718000060, tail: str = ""):
    return (
        f"EPS_RELAUNCH_PID={'alive' if alive else 'dead'}\n"
        f"EPS_RELAUNCH_MTIME={mtime}\n"
        f"EPS_RELAUNCH_NOW={now}\n"
        "EPS_RELAUNCH_TAIL_START\n"
        f"{tail}\n"
        "EPS_RELAUNCH_TAIL_END\n"
    )


def _relaunch_backend(*, ssh_results, phase: str = "done", reader=None) -> tuple[GcpBackend, Any]:
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload(phase), "")],
        ssh_results=list(ssh_results),
    )
    backend = GcpBackend(
        config=_test_config(),
        runner=runner,
        marker_poster=lambda **_: None,
        marker_reader=reader or _relaunch_reader(),
    )
    return backend, runner


def test_poll_done_phase_with_newer_relaunch_marker_follows_live_pid() -> None:
    """phase=done is the FIRST workload's exit; a fresh epm:run-launched
    (pid= + log_abs=, newer than epm:cluster-launched) means an SSH
    hot-fix relaunch is the live workload — poll must report running,
    not a premature workload_done (incident #612)."""
    backend, runner = _relaunch_backend(
        ssh_results=[
            GcloudRunResult(0, _EMPTY_DRAIN_STDOUT, ""),  # drain (no sentinels)
            GcloudRunResult(0, _probe_stdout(alive=True, tail="step 1200/9000"), ""),
        ],
    )
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    assert pr.current_phase == "relaunched_workload"
    assert pr.pid_alive is True
    assert pr.last_log_mtime_sec_ago == 60
    assert "step 1200/9000" in pr.log_tail_excerpt
    probe_cmd = next(
        arg
        for argv in runner.calls
        if "ssh" in argv and "compute" in argv
        for arg in argv
        if arg.startswith("--command=") and "kill -0 4610" in arg
    )
    assert "sudo -n bash -c" in probe_cmd, "probe must read the root-owned tree via sudo (#608)"


def test_poll_failed_phase_with_newer_relaunch_marker_follows_live_pid() -> None:
    """Symmetric for phase=failed: a relaunch after a failure-trap exit is
    otherwise reported dead and the orchestrator may tear down mid-run."""
    backend, _runner = _relaunch_backend(
        phase="failed",
        ssh_results=[
            GcloudRunResult(0, _EMPTY_DRAIN_STDOUT, ""),
            GcloudRunResult(0, _probe_stdout(alive=True), ""),
        ],
    )
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    assert pr.current_phase == "relaunched_workload"


def test_poll_relaunched_pid_dead_with_done_phase_line_maps_to_done() -> None:
    """A dead relaunch pid corroborated by a real [phase=done] log line is
    terminal success (suffixed terminal lines must keep parsing as done,
    #545)."""
    backend, _runner = _relaunch_backend(
        ssh_results=[
            GcloudRunResult(0, _EMPTY_DRAIN_STDOUT, ""),
            GcloudRunResult(
                0, _probe_stdout(alive=False, tail="[phase=done] production driver complete"), ""
            ),
        ],
    )
    pr = backend.poll(_drain_handle())
    assert pr.status == "done"
    assert pr.current_phase == "relaunched_workload_done"
    assert pr.pid_alive is False


def test_poll_relaunched_pid_dead_without_done_maps_to_dead() -> None:
    backend, _runner = _relaunch_backend(
        ssh_results=[
            GcloudRunResult(0, _EMPTY_DRAIN_STDOUT, ""),
            GcloudRunResult(0, _probe_stdout(alive=False, tail="[phase=training] step 1k"), ""),
        ],
    )
    pr = backend.poll(_drain_handle())
    assert pr.status == "dead"
    assert pr.current_phase == "relaunched_workload_exited"


def test_poll_relaunched_pid_dead_quoted_done_noise_maps_to_dead() -> None:
    """A failure message QUOTING the done token is not a phase transition
    (#597) — the relaunch branch inherits poll_pipeline's noise guard."""
    backend, _runner = _relaunch_backend(
        ssh_results=[
            GcloudRunResult(0, _EMPTY_DRAIN_STDOUT, ""),
            GcloudRunResult(
                0,
                _probe_stdout(
                    alive=False,
                    tail="ONE OR MORE SHARDS FAILED rc=1 - [phase=done] NOT emitted",
                ),
                "",
            ),
        ],
    )
    pr = backend.poll(_drain_handle())
    assert pr.status == "dead"


def test_poll_relaunch_marker_older_than_provision_keeps_done() -> None:
    """A run-launched marker from a PREVIOUS instance generation (older
    than the current epm:cluster-launched) must not hijack the poll."""
    backend, runner = _relaunch_backend(
        ssh_results=[GcloudRunResult(0, _EMPTY_DRAIN_STDOUT, "")],
        reader=_relaunch_reader(run_ts="2026-06-12T05:00:00Z", cluster_ts="2026-06-12T05:31:52Z"),
    )
    pr = backend.poll(_drain_handle())
    assert pr.status == "done"
    assert pr.current_phase == "workload_done"
    ssh_calls = [a for a in runner.calls if "ssh" in a and "compute" in a]
    assert len(ssh_calls) == 1  # drain only — no relaunch probe


def test_poll_relaunch_marker_for_other_host_keeps_done() -> None:
    """A relaunch marker naming a different host (e.g. a RunPod pod) is
    not this instance's workload."""
    note = _RELAUNCH_NOTE.replace("pod=eps-issue-137", "pod=pod-137")
    backend, _runner = _relaunch_backend(
        ssh_results=[GcloudRunResult(0, _EMPTY_DRAIN_STDOUT, "")],
        reader=_relaunch_reader(note=note),
    )
    pr = backend.poll(_drain_handle())
    assert pr.status == "done"
    assert pr.current_phase == "workload_done"


def test_poll_relaunch_marker_without_pid_keeps_done() -> None:
    backend, _runner = _relaunch_backend(
        ssh_results=[GcloudRunResult(0, _EMPTY_DRAIN_STDOUT, "")],
        reader=_relaunch_reader(note="RELAUNCH pod=eps-issue-137 (no pid recorded)"),
    )
    pr = backend.poll(_drain_handle())
    assert pr.status == "done"
    assert pr.current_phase == "workload_done"


def test_poll_relaunch_accepted_on_pod_match_when_cluster_marker_missing() -> None:
    """The launch-time epm:cluster-launched post is best-effort; when it is
    absent the instance-name match alone accepts the relaunch marker."""
    backend, _runner = _relaunch_backend(
        ssh_results=[
            GcloudRunResult(0, _EMPTY_DRAIN_STDOUT, ""),
            GcloudRunResult(0, _probe_stdout(alive=True), ""),
        ],
        reader=_relaunch_reader(cluster_ts=None),
    )
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    assert pr.current_phase == "relaunched_workload"


def test_poll_relaunch_probe_transport_failure_is_typed_stalled() -> None:
    """ "Couldn't ask" must never read as a terminal verdict (#535
    discipline): a probe SSH failure is a typed stalled tick, not done
    and not dead."""
    backend, _runner = _relaunch_backend(
        ssh_results=[
            GcloudRunResult(0, _EMPTY_DRAIN_STDOUT, ""),
            GcloudRunResult(1, "", "ssh: connect to host ... port 22: Connection refused"),
        ],
    )
    pr = backend.poll(_drain_handle())
    assert pr.status == "stalled"
    assert pr.current_phase == "relaunch_probe_failed"


def test_poll_no_relaunch_marker_keeps_existing_done_behavior() -> None:
    backend, _runner = _relaunch_backend(
        ssh_results=[GcloudRunResult(0, _EMPTY_DRAIN_STDOUT, "")],
        reader=lambda *_a, **_k: None,
    )
    pr = backend.poll(_drain_handle())
    assert pr.status == "done"
    assert pr.current_phase == "workload_done"


def test_poll_relaunched_done_corroboration_survives_long_tail() -> None:
    """The [phase=done] line lives at the END of the tail; a >2000-char
    tail must not push it out of the corroboration parse (the excerpt is
    tail-cut, the parse runs on the full text)."""
    filler = "\n".join(
        f"eval cell {i}/28 complete with a long descriptive suffix line" for i in range(40)
    )
    tail = filler + "\n[phase=done] production driver complete"
    assert len(tail) > 2000
    backend, _runner = _relaunch_backend(
        ssh_results=[
            GcloudRunResult(0, _EMPTY_DRAIN_STDOUT, ""),
            GcloudRunResult(0, _probe_stdout(alive=False, tail=tail), ""),
        ],
    )
    pr = backend.poll(_drain_handle())
    assert pr.status == "done"
    assert pr.current_phase == "relaunched_workload_done"
    assert pr.log_tail_excerpt.endswith("[phase=done] production driver complete")
    assert len(pr.log_tail_excerpt) <= 2000


# ---------------------------------------------------------------------------
# issue #607 — startup-script output redirect (metadata-runner token-too-long
# kill, incident #491) + truthful poll-side log-mtime overlay
# ---------------------------------------------------------------------------


def test_render_startup_script_redirects_output_before_workload() -> None:
    """T1 (#607, acceptance criterion 1): the rendered script redirects
    ALL further output to the handle's log file BEFORE the secrets fetch
    — the metadata runner's pipe never carries workload output (its
    bounded line scanner kills the script on giant lines, incident #491).
    """
    import shlex

    config = _test_config()
    log_path = log_path_for(config, 137)
    quoted_log = shlex.quote(log_path)
    quoted_dir = shlex.quote(log_path.rsplit("/", 1)[0])
    exec_line = 'exec >>"$EPS_LOG_PATH" 2>&1'
    for script, workload_line in (
        (
            render_startup_script(spec=_spec(), config=config, attempt_id="att-fixed-001"),
            "uv run python scripts/train.py",
        ),
        (
            render_startup_script(spec=_workload_spec(), config=config, attempt_id="att-fixed-001"),
            "bash scripts/issue588_smoke.sh",
        ),
    ):
        assert script.count(exec_line) == 1
        assert f"export EPS_LOG_PATH={quoted_log}" in script
        i_mkdir = script.index(f"mkdir -p {quoted_dir}")
        i_exec = script.index(exec_line)
        i_secrets = script.index("# === Secrets from instance metadata ===")
        i_workload = script.index(workload_line)
        assert i_mkdir < i_exec < i_secrets < i_workload
        # No later exec reverts the redirect back to the runner pipe.
        post_redirect = script[i_exec + len(exec_line) :]
        assert "exec >" not in post_redirect
        assert "exec 1>" not in post_redirect


def test_render_startup_script_log_path_matches_handle_log_path() -> None:
    """T2 (#607): the renderer's redirect target and ``handle.log_path``
    come from the same ``log_path_for`` helper — producer/consumer
    non-drift (``_drain_sentinels`` tails ``handle.log_path``, so a
    diverged renderer path would silently blank the poll's log tail)."""
    import shlex

    config = _test_config()
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "112233"}])
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[GcloudRunResult(0, created_payload, "")],
    )
    backend = GcpBackend(config=config, runner=runner, marker_poster=lambda **_: None)
    handle = backend.launch(_workload_spec())
    assert handle.log_path == log_path_for(config, 137)
    # The rendered script (same inputs) embeds the same quoted literal.
    script = render_startup_script(spec=_workload_spec(), config=config, attempt_id="att-fixed-001")
    assert f"export EPS_LOG_PATH={shlex.quote(handle.log_path)}" in script


def test_render_startup_script_unsets_tqdm_disable() -> None:
    """T3 (#607, amended #542): ``unset TQDM_DISABLE`` runs before the
    workload in both branches — NOT ``export TQDM_DISABLE=1``.

    vLLM 0.11.0's batched ``_run_engine`` divides ``total_in_toks`` by
    ``pbar.format_dict["elapsed"]`` on the first finished output; a
    DISABLED tqdm bar never starts its timer, so ``elapsed`` is 0.0 and
    every GCP workload calling batched ``LLM.generate()`` crashes with
    ZeroDivisionError (#542: 4 dead eps-issue-542 VMs). The #491
    giant-line zombie that originally motivated the disable is closed by
    the ``exec >>"$EPS_LOG_PATH"`` redirect (bars hit the unbounded log
    file, never the metadata runner's bounded scanner), so the bar must
    stay ENABLED — its timer keeps ``elapsed`` > 0. ``unset`` (not
    ``=0``) because tqdm's @envwrap coerces ``bool("0") == True`` → ``=0``
    would still disable; unset also clears any inherited DLVM/metadata
    value."""
    config = _test_config()
    for script, workload_line in (
        (
            render_startup_script(spec=_spec(), config=config, attempt_id="att-fixed-001"),
            "uv run python scripts/train.py",
        ),
        (
            render_startup_script(spec=_workload_spec(), config=config, attempt_id="att-fixed-001"),
            "bash scripts/issue588_smoke.sh",
        ),
    ):
        assert "unset TQDM_DISABLE" in script
        assert script.index("unset TQDM_DISABLE") < script.index(workload_line)
        # The disable that caused the #542 ZeroDivisionError must be GONE
        # (a disabled bar — by any TQDM_DISABLE export — re-opens the crash).
        assert "export TQDM_DISABLE" not in script


def test_render_startup_script_pipe_trap_is_handler_not_ignore() -> None:
    """T4 (#607): SIGPIPE gets a HANDLER (``trap ':' PIPE``), never an
    ignore disposition — SIG_IGN is inherited across exec and breaks
    ``producer | head`` pipelines under pipefail in workload children,
    while a handler keeps the parent immune (closed-pipe write becomes a
    normal rc=1 failure) and children retain default SIGPIPE."""
    import re

    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert "trap ':' PIPE" in script
    # No ignore-disposition variant anywhere (formatting-robust).
    assert re.search(r"""trap\s+(""|''|\$'')\s+(SIG)?PIPE\b""", script) is None
    assert re.search(r"""trap\s+(""|'')\s+13\b""", script) is None
    # Installed before the first echo/heartbeat can fire; fd 3 saved
    # before the _eps_phase definition that writes to it.
    assert script.index("trap ':' PIPE") < script.index("echo")
    assert script.index("exec 3>&1") < script.index("_eps_phase()")


def test_render_startup_script_exit_trap_guards_unset_log_path() -> None:
    """T5 (#607): the EXIT trap is installed BEFORE ``EPS_LOG_PATH``
    exists, so its log-tail diagnostic must reference ``${EPS_LOG_PATH:-}``
    — a bare reference in an early-failure trap errors mid-trap under
    ``set -u`` and SKIPS the shutdown. The pre-#607 invariants (rc guard,
    poweroff) and the v2 non-aborting ``set +e`` stay pinned."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    trap_line = next(line for line in script.splitlines() if line.startswith("trap 'rc=$?"))
    # The GUARD is the :- expansion; the bare reference inside the
    # guarded then-branch is only evaluated once the guard passed.
    assert 'if [ -n "${EPS_LOG_PATH:-}" ]' in trap_line
    assert '[ "$rc" -ne 0 ]' in trap_line
    assert "shutdown -h now" in trap_line
    assert "set +e" in trap_line


def _redirect_prelude_rig(tmp_path: Path) -> tuple[str, dict[str, str]]:
    """Shared rig for the #607 local integration tests (T6/T6b).

    Renders the startup script with ``vm_scratch_dir`` under tmp (pure-
    function renderer: every path lands under tmp), slices the runnable
    prelude at the LOAD-BEARING ``# === /output redirect (#607) ===`` end
    marker, and prepends a PATH stub-bin: ``curl`` records its argv to
    ``<tmp>/curl-calls.txt`` (so ``_eps_phase``'s guest-attribute PUT
    short-circuits and the phases are observable) and ``shutdown``
    touches ``<tmp>/shutdown-invoked`` (the EXIT trap must never touch
    the host). Returns ``(prelude, env)``.
    """
    import shlex

    config = replace(_test_config(), vm_scratch_dir=str(tmp_path))
    script = render_startup_script(spec=_workload_spec(), config=config, attempt_id="att-fixed-001")
    lines = script.splitlines()
    end_idx = lines.index("# === /output redirect (#607) ===")
    prelude = "\n".join(lines[: end_idx + 1])

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl_stub = bin_dir / "curl"
    curl_stub.write_text(
        "#!/bin/bash\n"
        + f'echo "$@" >> {shlex.quote(str(tmp_path / "curl-calls.txt"))}\n'
        + "exit 0\n"
    )
    curl_stub.chmod(0o755)
    shutdown_stub = bin_dir / "shutdown"
    shutdown_stub.write_text(
        "#!/bin/bash\n" + f"touch {shlex.quote(str(tmp_path / 'shutdown-invoked'))}\n" + "exit 0\n"
    )
    shutdown_stub.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '/usr/bin:/bin')}"
    return prelude, env


def _read_until_redirect_pointer(proc) -> bytes:
    """Read the runner pipe (byte-capped) until the pre-redirect pointer
    line lands — the LAST pipe write before ``exec`` moves all further
    output to the log file."""
    import time

    captured = b""
    deadline = time.time() + 30
    while (
        b"redirecting all further output" not in captured
        and len(captured) < 8192
        and time.time() < deadline
    ):
        chunk = proc.stdout.read1(1024)
        if not chunk:
            break
        captured += chunk
    return captured


def test_startup_redirect_survives_giant_line_locally(tmp_path: Path) -> None:
    """T6 (#607, acceptance criterion 2): a >1 MB newline-free workload
    line survives the runner CLOSING the pipe — the giant line lands in
    the log file, the script exits 0, and the pipe carried only the tiny
    pre-redirect lines. A dropped redirect would send the 1.2 MB write to
    the closed pipe → EPIPE under the PIPE handler → rc=1 → ``set -e`` →
    nonzero exit (so exit-0 + log size + done marker are the load-bearing
    witnesses)."""
    import shlex

    prelude, env = _redirect_prelude_rig(tmp_path)
    done_marker = tmp_path / "done"
    body = "\n".join(
        [
            # >1 MB, no trailing newline — the #491 kill shape.
            "head -c 1200000 /dev/zero | tr '\\0' 'x'",
            "_eps_phase workload",
            f"echo done-marker > {shlex.quote(str(done_marker))}",
            "",
        ]
    )
    script_path = tmp_path / "t6.sh"
    script_path.write_text(prelude + "\n" + body)
    proc = subprocess.Popen(
        ["bash", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    assert proc.stdout is not None
    captured = _read_until_redirect_pointer(proc)
    # Emulate the metadata runner closing the pipe (scanner overflow).
    proc.stdout.close()
    rc = proc.wait(timeout=60)
    assert rc == 0
    log_file = tmp_path / "logs" / "issue-137.log"
    assert log_file.exists()
    assert log_file.stat().st_size > 1_000_000
    assert done_marker.exists()
    # Success path must NOT power off (rc==0 guard).
    assert not (tmp_path / "shutdown-invoked").exists()
    # The pipe never carried the giant line: total pre-close pipe bytes
    # are tiny and contain no long x-run (v2 fix — the old <=64-byte read
    # made this negative assert vacuous).
    assert len(captured) < 4096
    assert b"x" * 1000 not in captured


@pytest.mark.parametrize("mode", ["plain_exit", "closed_pipe_builtin_write"])
def test_startup_failure_path_invokes_shutdown_and_failed_phase(tmp_path: Path, mode: str) -> None:
    """T6b (#607 v2, binding criterion 4): the failure path (rc≠0 →
    ``_eps_phase failed`` → ``shutdown -h now``) EXECUTES — including
    under a CLOSED runner pipe, where the v2 non-aborting EXIT trap must
    reach the shutdown even though its own diagnostics hit EPIPE.
    Converts the plan's bash experiments (builtin SIGPIPE death / handler
    semantics) into a repeatable regression test."""
    import shlex

    prelude, env = _redirect_prelude_rig(tmp_path)
    pipe_closed = tmp_path / "pipe-closed"
    if mode == "plain_exit":
        body = "exit 7\n"
    else:
        # Wait for the test to close the read end FIRST, then perform the
        # #491 kill shape: an UNGUARDED builtin write to the closed
        # runner pipe. Under ``trap ':' PIPE`` this is a normal rc=1
        # failure -> set -e -> EXIT trap rc=1 -> failure branch.
        body = (
            f"while [ ! -f {shlex.quote(str(pipe_closed))} ]; do sleep 0.05; done\n"
            "echo zombie-probe >&3\n"
        )
    script_path = tmp_path / "t6b.sh"
    script_path.write_text(prelude + "\n" + body)
    proc = subprocess.Popen(
        ["bash", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    assert proc.stdout is not None
    if mode == "closed_pipe_builtin_write":
        _read_until_redirect_pointer(proc)
        proc.stdout.close()  # the runner closes FIRST...
        pipe_closed.touch()  # ...then the body writes to the dead pipe
        rc = proc.wait(timeout=60)
    else:
        proc.communicate(timeout=60)
        rc = proc.returncode
    assert rc != 0
    assert (tmp_path / "shutdown-invoked").exists()
    curl_calls = (tmp_path / "curl-calls.txt").read_text()
    assert "--data failed" in curl_calls


def test_drain_overlays_log_mtime_when_running(monkeypatch) -> None:
    """T7 (#607): a running tick reports a TRUTHFUL
    ``last_log_mtime_sec_ago`` from the drain's piggy-backed stat
    (consumer side) — and the issued drain SSH command actually carries
    the stat stanza, keyed AFTER the ``EPS_LOGTAIL_END`` delimiter so the
    tail partition is unaffected (producer side, v2)."""
    import shlex

    pp = _poll_pipeline_module()
    monkeypatch.setattr(pp, "post_event", lambda *a, **kw: None)
    monkeypatch.setattr(pp, "list_events", lambda _issue: [])  # #1084 dedupe read stub
    now = 1718000300
    stdout = _drain_stdout("19/19 cells done") + f"EPS_LOG_MTIME={now - 300}\nEPS_LOG_NOW={now}\n"
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
        ssh_results=[
            GcloudRunResult(0, stdout, ""),
            GcloudRunResult(0, "", ""),  # mv -> .processed
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    handle = _drain_handle()
    pr = backend.poll(handle)
    assert pr.status == "running"
    assert pr.last_log_mtime_sec_ago == 300  # not the hardwired 10**9
    # Producer side (v2): scan the issued drain --command= string.
    drain_cmd = next(
        arg
        for argv in runner.calls
        if "ssh" in argv and "compute" in argv
        for arg in argv
        if arg.startswith("--command=") and "EPS_LOGTAIL_START" in arg
    )
    quoted_log = shlex.quote(handle.log_path)
    assert f"stat -c %Y {quoted_log}" in drain_cmd
    assert "EPS_LOG_MTIME=" in drain_cmd
    assert "EPS_LOG_NOW=" in drain_cmd
    # Keys AFTER the delimiter (before it they would corrupt the tail
    # partition), and the tail segment is byte-bounded ON THE VM.
    assert drain_cmd.index("EPS_LOGTAIL_END") < drain_cmd.index("EPS_LOG_MTIME=")
    assert drain_cmd.index("EPS_LOGTAIL_END") < drain_cmd.index("EPS_LOG_NOW=")
    tail_segment = drain_cmd[
        drain_cmd.index("EPS_LOGTAIL_START") : drain_cmd.index("EPS_LOGTAIL_END")
    ]
    assert "| cut -c1-4000" in tail_segment


def test_drain_mtime_missing_file_reports_legacy_placeholder() -> None:
    """T7b (#607 v2): ``EPS_LOG_MTIME=-1`` — the ``stat ... || echo -1``
    missing-file cell (e.g. a pre-#607 handle whose log never existed) —
    must NOT overlay: the legacy ``10**9`` placeholder is kept, never a
    bogus huge/negative age."""
    stdout = "EPS_LOGTAIL_START\nEPS_LOGTAIL_END\nEPS_LOG_MTIME=-1\nEPS_LOG_NOW=1718000300\n"
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
        ssh_results=[GcloudRunResult(0, stdout, "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    assert pr.last_log_mtime_sec_ago == 10**9


def test_drain_missing_mtime_keys_keeps_legacy_placeholder(monkeypatch) -> None:
    """T8 (#607): a drain stdout WITHOUT the mtime keys (the pre-#607
    fixture shape) behaves exactly as today — the running path keeps the
    hardwired placeholder; old fixtures stay green untouched."""
    pp = _poll_pipeline_module()
    monkeypatch.setattr(pp, "post_event", lambda *a, **kw: None)
    monkeypatch.setattr(pp, "list_events", lambda _issue: [])  # #1084 dedupe read stub
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
        ssh_results=[
            GcloudRunResult(0, _drain_stdout("mid-run"), ""),
            GcloudRunResult(0, "", ""),  # mv -> .processed
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    assert pr.last_log_mtime_sec_ago == 10**9


def test_overlay_drain_mtime_ignored_on_terminal_status() -> None:
    """T9 (#607 v2): a terminal base PollResult keeps its own
    ``last_log_mtime_sec_ago`` — the mtime overlay fires only on
    ``running`` (the carve-out was previously unpinned)."""
    from explore_persona_space.backends.base import PollResult
    from explore_persona_space.backends.gcp import _overlay_drain

    done = PollResult(
        status="done",
        current_phase="workload_done",
        new_milestone=True,
        last_log_mtime_sec_ago=0,
        pid_alive=False,
        log_tail_excerpt="",
    )
    out = _overlay_drain(done, processed=0, gate=None, alarm="", log_tail="", log_mtime_ago=300)
    assert out.last_log_mtime_sec_ago == 0
    dead = PollResult(
        status="dead",
        current_phase="terminal_workload_failed",
        new_milestone=True,
        last_log_mtime_sec_ago=10**9,
        pid_alive=False,
        log_tail_excerpt="",
    )
    out = _overlay_drain(dead, processed=0, gate=None, alarm="", log_tail="", log_mtime_ago=300)
    assert out.last_log_mtime_sec_ago == 10**9
    running = PollResult(
        status="running",
        current_phase="workload",
        new_milestone=False,
        last_log_mtime_sec_ago=10**9,
        pid_alive=True,
        log_tail_excerpt="",
    )
    out = _overlay_drain(running, processed=0, gate=None, alarm="", log_tail="", log_mtime_ago=300)
    assert out.last_log_mtime_sec_ago == 300


# ---------------------------------------------------------------------------
# #631 D1 — FLEX_START provisioning support
# ---------------------------------------------------------------------------


def test_resolve_provisioning_model_accepts_flex_start() -> None:
    spec = _spec(extra={"provisioning_model": "flex_start"})
    assert resolve_provisioning_model(spec) == "FLEX_START"


def test_resolve_provisioning_model_still_rejects_typo() -> None:
    """Regression: an unknown value (e.g. 'preemptible') still raises loud."""
    spec = _spec(extra={"provisioning_model": "preemptible"})
    with pytest.raises(ValueError, match="unknown provisioning_model"):
        resolve_provisioning_model(spec)


def test_render_create_argv_flex_start_renders_request_valid_for_duration() -> None:
    cfg = _test_config()
    spec = _spec(extra={"provisioning_model": "FLEX_START"})
    argv = render_create_argv(
        spec=spec,
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--provisioning-model=FLEX_START" in argv
    # Default request-validity window (the doc max, 2h).
    assert "--request-valid-for-duration=2h" in argv


def test_render_create_argv_flex_start_includes_reservation_affinity_none() -> None:
    """v2: the canonical flex-start create command pins --reservation-affinity=none."""
    cfg = _test_config()
    spec = _spec(extra={"provisioning_model": "FLEX_START"})
    argv = render_create_argv(
        spec=spec,
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert argv.count("--reservation-affinity=none") == 1


def test_render_create_argv_flex_start_keeps_delete_termination_action() -> None:
    """The leak guard stays unconditional on FLEX_START (docs: STOP or DELETE both allowed)."""
    cfg = _test_config()
    spec = _spec(extra={"provisioning_model": "FLEX_START"})
    argv = render_create_argv(
        spec=spec,
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--instance-termination-action=DELETE" in argv


def test_render_create_argv_standard_omits_request_valid_for_duration() -> None:
    """Regression: the default STANDARD create carries no flex-start flag."""
    cfg = _test_config()
    argv = render_create_argv(
        spec=_spec("lora-7b"),
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert not any(a.startswith("--request-valid-for-duration=") for a in argv)


def test_render_create_argv_spot_omits_request_valid_for_duration() -> None:
    """v2: SPOT must NOT render the flex-start window (guards a `!= STANDARD` regression)."""
    cfg = _test_config()
    spec = _spec(extra={"provisioning_model": "spot"})
    argv = render_create_argv(
        spec=spec,
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert not any(a.startswith("--request-valid-for-duration=") for a in argv)


def test_render_create_argv_standard_and_spot_omit_reservation_affinity() -> None:
    """v2: only FLEX_START adds --reservation-affinity; STANDARD/SPOT keep their argv."""
    cfg = _test_config()
    standard = render_create_argv(
        spec=_spec("lora-7b"),
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    spot = render_create_argv(
        spec=_spec(extra={"provisioning_model": "spot"}),
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert not any(a.startswith("--reservation-affinity") for a in standard)
    assert not any(a.startswith("--reservation-affinity") for a in spot)


def test_render_create_argv_flex_start_rejects_max_run_over_7d() -> None:
    """A FLEX_START create whose --max-run-duration exceeds the 7-day cap fails loud."""
    cfg = _test_config()
    spec = _spec(extra={"provisioning_model": "FLEX_START", "max_run_duration": "8d"})
    with pytest.raises(ValueError, match="7-day flex-start ceiling"):
        render_create_argv(
            spec=spec,
            config=cfg,
            attempt_id="att-fixed-001",
            startup_script="#!/bin/bash\n",
            secret_files=_TEST_SECRET_FILES,
        )


def test_render_create_argv_flex_start_accepts_composed_max_run_duration() -> None:
    """A composed gcloud duration (``1d12h`` = 36h, under the 7-day cap) renders.

    Regression for the round-1 bug: the CLI validator
    (scripts/dispatch_issue.py:_MAX_RUN_DURATION_RE) accepts composed
    forms like ``1d12h``, but the FLEX_START render guard rejected them
    with a spurious ValueError before the 7-day cap check.
    """
    cfg = _test_config()
    spec = _spec(extra={"provisioning_model": "FLEX_START", "max_run_duration": "1d12h"})
    argv = render_create_argv(
        spec=spec,
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--max-run-duration=1d12h" in argv


def test_assert_max_run_within_flex_cap_accepts_composed_under_7d() -> None:
    """Composed durations under the 7-day FLEX_START cap return cleanly."""
    from explore_persona_space.backends.gcp import _assert_max_run_within_flex_cap

    for value in ("1d12h", "6d23h", "0d"):
        # No raise expected; the helper returns None.
        assert _assert_max_run_within_flex_cap(max_run=value, provisioning="FLEX_START") is None


def test_assert_max_run_within_flex_cap_rejects_composed_over_7d() -> None:
    """Composed durations over the 7-day FLEX_START cap fail loud."""
    from explore_persona_space.backends.gcp import _assert_max_run_within_flex_cap

    for value in ("8d", "7d1h", "7d0h1s"):
        with pytest.raises(ValueError, match="7-day flex-start ceiling"):
            _assert_max_run_within_flex_cap(max_run=value, provisioning="FLEX_START")


def test_assert_max_run_within_flex_cap_rejects_malformed() -> None:
    """Malformed durations fail loud (not silently treated as 0 / dropped)."""
    from explore_persona_space.backends.gcp import _assert_max_run_within_flex_cap

    # "1d12" has a unit-less final group; "5e3"/"abc" are not gcloud durations;
    # "" is empty. All must raise, never silently parse to a wrong number.
    for value in ("abc", "5e3", "1d12", ""):
        with pytest.raises(ValueError, match="unparseable gcloud duration"):
            _assert_max_run_within_flex_cap(max_run=value, provisioning="FLEX_START")


def test_resolve_request_valid_for_duration_default_and_override() -> None:
    assert resolve_request_valid_for_duration(_spec()) == "2h"
    pinned = _spec(extra={"request_valid_for_duration": "90s"})
    assert resolve_request_valid_for_duration(pinned) == "90s"


# ---------------------------------------------------------------------------
# #631 D2 — H100 intent machine resolution
# ---------------------------------------------------------------------------


def test_machine_for_intent_resolves_h100_1g() -> None:
    machine = machine_for_intent(_spec("lora-7b-h100"))
    assert machine.machine_type == "a3-highgpu-1g"
    assert machine.gpu_count == 1
    assert machine.gpu_kind == "H100-80"


def test_machine_for_intent_resolves_h100_2g() -> None:
    machine = machine_for_intent(_spec("eval-h100"))
    assert machine.machine_type == "a3-highgpu-2g"
    assert machine.gpu_count == 2
    assert machine.gpu_kind == "H100-80"


def test_render_create_argv_h100_intent_uses_a3_highgpu_machine() -> None:
    """An H100 intent (passed SPOT, since on-demand is rejected) renders a3-highgpu."""
    cfg = _test_config()
    spec = _spec("lora-7b-h100", extra={"provisioning_model": "SPOT"})
    argv = render_create_argv(
        spec=spec,
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--machine-type=a3-highgpu-1g" in argv
    assert "--provisioning-model=SPOT" in argv


# ---------------------------------------------------------------------------
# #631 D3 — provisioning-aware quota metric + SPOT-on-A100 + H100/STANDARD guard
# ---------------------------------------------------------------------------


def test_quota_metric_for_a100_spot_is_preemptible() -> None:
    assert (
        quota_metric_for(INTENT_TO_MACHINE["lora-7b"], "SPOT")
        == "PREEMPTIBLE_NVIDIA_A100_80GB_GPUS"
    )


def test_quota_metric_for_a100_flex_start_is_preemptible() -> None:
    """v2: A100 + FLEX_START draws the preemptible A100 pool (not on-demand)."""
    assert (
        quota_metric_for(INTENT_TO_MACHINE["lora-7b"], "FLEX_START")
        == "PREEMPTIBLE_NVIDIA_A100_80GB_GPUS"
    )


def test_quota_metric_for_a100_standard_is_on_demand() -> None:
    """Regression: STANDARD A100 still reads the on-demand metric."""
    assert quota_metric_for(INTENT_TO_MACHINE["lora-7b"], "STANDARD") == "NVIDIA_A100_80GB_GPUS"


def test_quota_metric_for_h100_flex_start_is_preemptible_h100() -> None:
    assert (
        quota_metric_for(INTENT_TO_MACHINE["lora-7b-h100"], "FLEX_START")
        == "PREEMPTIBLE_NVIDIA_H100_GPUS"
    )


def test_preflight_quota_headroom_spot_reads_preemptible_metric() -> None:
    """A SPOT spec resolves the PREEMPTIBLE row, not the on-demand one."""
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],  # no live instance
        region_describe_results=[
            GcloudRunResult(
                0,
                json.dumps(
                    {
                        "name": "us-central1",
                        "quotas": [
                            {"metric": "NVIDIA_A100_80GB_GPUS", "usage": 8.0, "limit": 8.0},
                            {
                                "metric": "PREEMPTIBLE_NVIDIA_A100_80GB_GPUS",
                                "usage": 0.0,
                                "limit": 8.0,
                            },
                        ],
                    }
                ),
                "",
            )
        ],
    )
    headroom = preflight_quota_headroom(
        spec=_spec(intent="lora-7b", extra={"provisioning_model": "spot"}),
        config=_test_config(),
        runner=runner,
    )
    assert headroom is not None
    assert headroom.metric == "PREEMPTIBLE_NVIDIA_A100_80GB_GPUS"
    # The preemptible row had full headroom even though on-demand was saturated.
    assert headroom.sufficient


def test_render_create_argv_h100_standard_raises_loud() -> None:
    """H100 cannot be created on-demand — STANDARD must fail at render."""
    cfg = _test_config()
    spec = _spec("lora-7b-h100")  # no provisioning_model → STANDARD default
    with pytest.raises(ValueError, match="cannot be created on-demand"):
        render_create_argv(
            spec=spec,
            config=cfg,
            attempt_id="att-fixed-001",
            startup_script="#!/bin/bash\n",
            secret_files=_TEST_SECRET_FILES,
        )


# ---------------------------------------------------------------------------
# 8-GPU sweep intents: render_create_argv goldens (#743)
# ---------------------------------------------------------------------------


def test_render_create_argv_sweep_8g_a100_spot_golden() -> None:
    """#743: an 8x A100 sweep on SPOT renders the a2-ultragpu-8g machine type
    with --provisioning-model=SPOT (model: test_render_create_argv_spot_opt_in)."""
    cfg = _test_config()
    spec = _spec("sweep-8g-a100", extra={"provisioning_model": "spot"})
    argv = render_create_argv(
        spec=spec,
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--machine-type=a2-ultragpu-8g" in argv
    assert "--provisioning-model=SPOT" in argv


def test_render_create_argv_sweep_8g_a100_standard_golden() -> None:
    """#743: an 8x A100 sweep with the default (STANDARD) provisioning renders
    fine — A100 (unlike H100) MAY be created on-demand, so no raise (model:
    test_render_create_argv_ft_intent_uses_4gpu_machine)."""
    cfg = _test_config()
    spec = _spec("sweep-8g-a100")  # no provisioning_model → STANDARD default
    argv = render_create_argv(
        spec=spec,
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--machine-type=a2-ultragpu-8g" in argv


def test_render_create_argv_sweep_8g_h100_flex_start_golden() -> None:
    """#743: an 8x H100 sweep on FLEX_START renders the a3-highgpu-8g machine
    type without raising (model:
    test_render_create_argv_flex_start_renders_request_valid_for_duration)."""
    cfg = _test_config()
    spec = _spec("sweep-8g-h100", extra={"provisioning_model": "FLEX_START"})
    argv = render_create_argv(
        spec=spec,
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--machine-type=a3-highgpu-8g" in argv
    assert "--provisioning-model=FLEX_START" in argv


def test_render_create_argv_sweep_8g_h100_standard_raises_loud() -> None:
    """#743: 8x H100 inherits the H100+STANDARD on-demand rejection for free
    (the render_create_argv raise keys on gpu_kind == 'H100-80', not a
    machine-type literal). model: test_render_create_argv_h100_standard_raises_loud."""
    cfg = _test_config()
    spec = _spec("sweep-8g-h100")  # no provisioning_model → STANDARD default
    with pytest.raises(ValueError, match="cannot be created on-demand"):
        render_create_argv(
            spec=spec,
            config=cfg,
            attempt_id="att-fixed-001",
            startup_script="#!/bin/bash\n",
            secret_files=_TEST_SECRET_FILES,
        )


# ---------------------------------------------------------------------------
# #659 — spec-threading at GCP launch time + workload-vs-setup poll
# discrimination (the prerequisites for the ASYNC RunPod failover).
#
# A1 (the as-is async signal) was reconciler-verified WRONG: ``eps/phase`` is
# single-valued and the EXIT trap overwrites it to ``failed`` on ANY non-zero
# exit, so a SETUP crash and a WORKLOAD crash collapse to the same
# ``current_phase``. A7 (the GCP handle carries enough to rebuild a RunSpec)
# was fact-checker-confirmed WRONG: ``handle.extra`` never carried the workload
# command. #659 fixes BOTH — MF1/MF2 thread the spec fields onto ``extra`` at
# launch, MF3 publishes an ``eps/workload_started`` sentinel so the poll can
# distinguish ``terminal_workload_failed`` from ``terminal_setup_failed``.
#
# These tests fail TODAY (the extra keys / the discrimination do not exist
# yet) and pass after the §4.1.0 / §4.1.0b sub-changes land.
# ---------------------------------------------------------------------------


def _launch_runner() -> _Runner:
    """A _Runner that lets ``launch`` provision a fake instance (no real VM)."""
    return _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],  # no existing instance
        create_results=[
            GcloudRunResult(0, json.dumps([{"name": "eps-issue-137", "id": "112233"}]), "")
        ],
    )


def test_gcp_handle_extra_carries_workload_command_for_runpod_failover(no_marker_posts) -> None:
    """#659 / MF1+MF2: ``GcpBackend.launch`` must thread the workload command +
    sizing onto ``RunHandle.extra`` so the async poller can reconstruct a
    RunSpec for the RunPod failover. The ``str`` ``workload_cmd`` survives
    VERBATIM (NOT exploded to a per-character list by an erroneous
    ``list(spec.workload_cmd or ())``), and
    ``deserialize_handle(serialize_handle(handle))`` preserves them all."""
    from explore_persona_space.backends.issue_dispatch import (
        deserialize_handle,
        serialize_handle,
    )

    cmd = "REPO_ROOT=/workspace bash scripts/foo.sh --bar"
    spec = _spec(hydra_args=(), workload_cmd=cmd, gpus=1, time_budget_hours=4.0)
    backend = GcpBackend(
        config=_test_config(), runner=_launch_runner(), marker_poster=lambda **_: None
    )
    handle = backend.launch(spec)

    # MF1: str preserved verbatim, NOT list("...") which would per-char explode.
    assert handle.extra["workload_cmd"] == cmd
    assert isinstance(handle.extra["workload_cmd"], str)
    assert handle.extra["hydra_args"] == []  # empty tuple () -> list []
    assert handle.extra["gpus"] == 1
    assert handle.extra["time_budget_hours"] == 4.0

    # Round-trip survival through the sidecar serializer (a plain string and a
    # list both JSON-round-trip faithfully).
    recovered = deserialize_handle(serialize_handle(handle))
    assert recovered.extra["workload_cmd"] == cmd
    assert isinstance(recovered.extra["workload_cmd"], str)
    assert recovered.extra["hydra_args"] == []
    assert recovered.extra["gpus"] == 1
    assert recovered.extra["time_budget_hours"] == 4.0


def test_gcp_handle_extra_round_trips_hydra_args_as_list(no_marker_posts) -> None:
    """#659 / MF1+MF2 (hydra branch): a hydra spec threads ``hydra_args`` as a
    list onto ``extra`` (empty ``workload_cmd``), and the round-trip preserves
    the list shape — so the poller can rebuild a hydra RunSpec verbatim."""
    from explore_persona_space.backends.issue_dispatch import (
        deserialize_handle,
        serialize_handle,
    )

    spec = _spec(hydra_args=("condition=c1_evil_wrong_em", "seed=42"))
    backend = GcpBackend(
        config=_test_config(), runner=_launch_runner(), marker_poster=lambda **_: None
    )
    handle = backend.launch(spec)

    assert handle.extra["workload_cmd"] == ""
    assert handle.extra["hydra_args"] == ["condition=c1_evil_wrong_em", "seed=42"]
    assert isinstance(handle.extra["hydra_args"], list)

    recovered = deserialize_handle(serialize_handle(handle))
    assert recovered.extra["hydra_args"] == ["condition=c1_evil_wrong_em", "seed=42"]
    assert isinstance(recovered.extra["hydra_args"], list)
    assert recovered.extra["workload_cmd"] == ""


def test_runspec_from_gcp_handle_preserves_mutual_exclusion() -> None:
    """#659 / MF2: ``_runspec_from_gcp_handle`` reads BOTH ``workload_cmd``
    (str) AND ``hydra_args`` (tuple/list) from ``extra`` and passes each
    through verbatim; one is empty by construction, so
    ``RunSpec.__post_init__``'s mutual exclusion holds and no placeholder is
    substituted into the unused branch."""
    from explore_persona_space.backends.base import RunHandle
    from scripts.backend_poll import _runspec_from_gcp_handle

    def _handle(extra: dict) -> RunHandle:
        return RunHandle(
            backend="gcp",
            cluster=None,
            job_id="instance-fake-1",
            pod_name="eps-issue-659",
            scratch_dir="/workspace/eps-issue-659",
            log_path="/workspace/logs/issue-659.log",
            extra=extra,
        )

    # (a) workload_cmd branch: non-empty workload_cmd + empty hydra_args.
    handle = _handle(
        {
            "intent": "lora-7b",
            "gpus": 1,
            "time_budget_hours": 4.0,
            "workload_cmd": "bash scripts/foo.sh",
            "hydra_args": [],
        }
    )
    spec = _runspec_from_gcp_handle(handle, issue=659)
    assert spec.workload_cmd == "bash scripts/foo.sh"
    assert spec.hydra_args == ()
    assert not (spec.workload_cmd and spec.hydra_args)  # mutual exclusion holds

    # (b) hydra branch: empty workload_cmd + non-empty hydra_args.
    handle2 = _handle(
        {
            "intent": "ft-7b",
            "gpus": 4,
            "time_budget_hours": 8.0,
            "workload_cmd": "",
            "hydra_args": ["condition=c1", "seed=42"],
        }
    )
    spec2 = _runspec_from_gcp_handle(handle2, issue=659)
    assert spec2.workload_cmd == ""
    assert spec2.hydra_args == ("condition=c1", "seed=42")
    assert not (spec2.workload_cmd and spec2.hydra_args)  # mutual exclusion holds

    # A pre-#659 handle missing the workload command FAILS LOUD (never launches
    # a blank RunPod job).
    with pytest.raises(ValueError):
        _runspec_from_gcp_handle(_handle({"intent": "lora-7b"}), issue=659)


def _guest_attr_payload_multi(items: list[tuple[str, str]]) -> str:
    """A ``get-guest-attributes`` payload carrying MULTIPLE eps/* keys (the
    whole-namespace read the §4.1.0b discrimination uses)."""
    return json.dumps([{"namespace": "eps", "key": key, "value": value} for key, value in items])


def test_gcp_poll_distinguishes_workload_failed_from_setup_failed() -> None:
    """#659 / MF3: ``GcpBackend.poll`` returns DISTINCT ``current_phase``
    values for a workload crash vs a setup failure, keyed on the
    ``eps/workload_started`` sentinel. ``phase==failed`` maps to
    ``terminal_workload_failed`` IFF the sentinel was published; otherwise
    ``terminal_setup_failed``. A PROBE FAILURE on the sentinel read falls back
    CONSERVATIVELY to ``terminal_workload_failed`` (never misread "couldn't
    ask" as "setup failed", which would suppress a legitimate failover).

    Built on the existing ``_Runner`` double: the implementer may read the
    sentinel via a whole-``eps/``-namespace probe (preferred — one round-trip)
    OR a second ``--query-path=eps/workload_started`` probe. To stay agnostic
    to that choice this test scripts the guest-attr payload to carry BOTH keys
    on the first read AND provides a matching second scripted result, so either
    implementation resolves the same way."""
    # (a) workload was reached: phase=failed AND workload_started=true.
    runner_a = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[
            GcloudRunResult(
                0,
                _guest_attr_payload_multi([("phase", "failed"), ("workload_started", "true")]),
                "",
            ),
            # Second probe (if the impl issues a separate workload_started read).
            GcloudRunResult(0, _guest_attr_payload_multi([("workload_started", "true")]), ""),
        ],
    )
    backend_a = GcpBackend(config=_test_config(), runner=runner_a, marker_poster=lambda **_: None)
    res_a = backend_a.poll(_poll_handle())
    assert res_a.status == "dead"
    assert res_a.current_phase == "terminal_workload_failed"

    # (b) setup failed before the workload ran: phase=failed, sentinel ABSENT.
    runner_b = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[
            GcloudRunResult(0, _guest_attr_payload_multi([("phase", "failed")]), ""),
            # Second probe (if issued) finds the attribute not written (404).
            GcloudRunResult(1, "", "guest attribute eps/workload_started not found"),
        ],
    )
    backend_b = GcpBackend(config=_test_config(), runner=runner_b, marker_poster=lambda **_: None)
    res_b = backend_b.poll(_poll_handle())
    assert res_b.status == "dead"
    assert res_b.current_phase == "terminal_setup_failed"

    # (c) probe FAILURE on the workload_started read -> conservative fallback to
    # terminal_workload_failed (do NOT downgrade an unprovable read to setup).
    runner_c = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[
            # phase reads failed; the sentinel read is an auth/transport failure.
            GcloudRunResult(0, _guest_attr_payload_multi([("phase", "failed")]), ""),
            GcloudRunResult(
                1, "", "ERROR: Required 'compute.instances.getGuestAttributes' permission denied"
            ),
        ],
    )
    backend_c = GcpBackend(config=_test_config(), runner=runner_c, marker_poster=lambda **_: None)
    res_c = backend_c.poll(_poll_handle())
    assert res_c.current_phase == "terminal_workload_failed"


# ---------------------------------------------------------------------------
# issue #669 — hung-but-RUNNING VM recovery: producer-side reachability_alarm
# wiring (M2.5), the in-VM watchdog (Fix 2), and the scheduling flags (Fix 3)
# ---------------------------------------------------------------------------


def test_gcp_poll_running_transport_drain_failure_sets_reachability_alarm() -> None:
    """M2.5 (#669 producer side): a RUNNING GCP poll whose sentinel-drain SSH
    fails at the TRANSPORT layer (rc != 0 — the unreachable-VM signature) sets
    ``PollResult.reachability_alarm = True``. This is the producer the poller's
    frozen-phase wedge gate (``_maybe_escalate_gcp_wedge``) reads — without it
    the consumer tests' mocked ``reachability_alarm`` would be ungrounded."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
        ssh_results=[GcloudRunResult(1, "", "ssh: connect to host 1.2.3.4 port 22: timed out")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"  # still a coarse running tick (not the poller's job here)
    assert pr.reachability_alarm is True  # transport class -> reachability alarm


def test_gcp_poll_running_healthy_drain_leaves_reachability_alarm_false() -> None:
    """M2.5 negative: a RUNNING GCP poll whose drain SSH SUCCEEDS (rc == 0,
    clean drain) leaves ``reachability_alarm = False`` — the VM answered, so it
    is NOT the unreachable-VM signature and the wedge gate must never fire."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
        ssh_results=[GcloudRunResult(0, "EPS_LOGTAIL_START\nEPS_LOGTAIL_END\n", "")],  # clean drain
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    assert pr.reachability_alarm is False


def test_gcp_poll_running_sentinel_processing_alarm_leaves_reachability_alarm_false(
    monkeypatch,
) -> None:
    """M2.5 / M1 signal split: a RUNNING GCP poll whose drain SSH SUCCEEDS but a
    matched sentinel set produced 0 processed markers (a HEALTHY VM with a
    malformed sentinel / transient marker-post failure) raises the
    SENTINEL-PROCESSING-class alarm, NOT the transport class — so
    ``reachability_alarm`` stays False even though the generic drain alarm
    fired. This is the M1 defect v1 would have tripped on."""
    pp = _poll_pipeline_module()
    monkeypatch.setattr(pp, "post_event", lambda *a, **kw: None)
    stdout = (
        "SENTINEL_START /workspace/logs/issue-137-epm_results-1781214523.json\n"
        "SENTINEL_END\n"
        "EPS_LOGTAIL_START\nEPS_LOGTAIL_END\n"
    )
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
        ssh_results=[GcloudRunResult(0, stdout, "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    # The generic drain alarm DID fire (loud in log_tail) ...
    assert "matched but 0 processed" in pr.log_tail_excerpt
    # ... but it is the sentinel-processing class, so reachability_alarm is False.
    assert pr.reachability_alarm is False


def test_poll_terminated_with_wedged_phase_maps_to_terminal_wedged_terminated() -> None:
    """#669 (Option 2): a TERMINATED VM whose in-VM watchdog wrote
    ``eps/phase=wedged`` before shutdown maps to ``terminal_wedged_terminated``
    (the conservative failover phase), NOT the bare ``terminal_terminated``."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "TERMINATED"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("wedged"), "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "dead"
    assert pr.current_phase == "terminal_wedged_terminated"


def test_poll_terminated_without_wedged_phase_stays_terminal_terminated() -> None:
    """#669 Option-2 invariant: a TERMINATED VM with NO ``eps/phase=wedged``
    (spot preemption / max-run-duration / manual stop) stays
    ``terminal_terminated`` EXACTLY as today — NO spot/max-run regression, the
    async-failover accept-set leaves ``terminal_terminated`` excluded."""
    # guest-attr default (gcloud rc=1, "not found") -> phase "" -> not wedged.
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "TERMINATED"}), "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "dead"
    assert pr.current_phase == "terminal_terminated"


def test_render_startup_script_includes_reachability_watchdog() -> None:
    """Fix 2 (#669): the rendered startup script embeds the reachability
    watchdog daemon — the metadata + external probes, the ~5-min sustained-loss
    threshold, the ``eps/phase=wedged`` write BEFORE the shutdown ladder, and
    the backgrounded launch + clean-exit reap. Asserted on BOTH the hydra and
    the workload_cmd branches (shared preamble)."""
    for spec in (_spec(), _workload_spec("bash scripts/foo.sh --x 1")):
        script = render_startup_script(spec=spec, config=_test_config(), attempt_id="att-fixed-001")
        assert "_eps_reachability_watchdog()" in script  # the daemon is defined
        assert "_eps_reachability_watchdog < /dev/null &" in script  # backgrounded launch
        # REACHABILITY (not liveness): both probes, evaluated SEPARATELY — fails
        # resets on OR-of-successes, increments ONLY on BOTH-fail (code-review r1).
        assert "169.254.169.254/computeMetadata/v1/instance/id" in script
        assert "https://huggingface.co/" in script
        # ~5 min sustained loss: 30s cadence x 10 consecutive failures.
        assert "interval=30 threshold=10" in script
        # eps/phase=wedged written BEFORE the shutdown ladder (the
        # terminal_wedged_terminated trigger).
        wedged_idx = script.index("_eps_phase wedged")
        shutdown_idx = script.index("shutdown -h now 2>/dev/null || poweroff -f")
        assert wedged_idx < shutdown_idx
        # Reaped on the clean-exit path before the success sentinel.
        kill_idx = script.index('kill "${EPS_WATCHDOG_PID:-}"')
        sentinel_idx = script.index("Completion sentinel")
        assert kill_idx < sentinel_idx


def test_render_startup_script_watchdog_launch_after_redirect_marker() -> None:
    """Fix 2 (#669): the watchdog launch lands AFTER the #607 output-redirect
    end marker, so (a) its output goes to the workload log (post-redirect) and
    (b) the #607 prelude-slicing integration tests never execute the infinite
    daemon loop (the slice ends AT the marker)."""
    script = render_startup_script(spec=_workload_spec(), config=_test_config(), attempt_id="a")
    redirect_end = script.index("# === /output redirect (#607) ===")
    launch_idx = script.index("_eps_reachability_watchdog < /dev/null &")
    assert launch_idx > redirect_end


def test_launch_argv_sets_no_restart_on_failure_and_maintenance_policy_terminate() -> None:
    """Fix 3 (#669): the create argv carries ``--no-restart-on-failure``
    (automaticRestart=false — a watchdog self-shutdown is FINAL) ALONGSIDE the
    pre-existing ``--maintenance-policy=TERMINATE`` and
    ``--instance-termination-action=DELETE`` (independent scheduling-block
    fields that compose freely)."""
    argv = render_create_argv(
        spec=_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
        secret_files=_TEST_SECRET_FILES,
    )
    assert "--no-restart-on-failure" in argv
    assert "--maintenance-policy=TERMINATE" in argv
    assert "--instance-termination-action=DELETE" in argv


# ---------------------------------------------------------------------------
# issue #669 code-review r1 (Blocker A): the watchdog probes the two endpoints
# SEPARATELY — fails resets on OR-of-successes, increments ONLY on BOTH-fail.
# Executes the rendered bash stanza with stubbed curl (the green substring tests
# never RAN the bash, so they were blind to the inverted `curl A && curl B`
# conjunction that incremented on a SINGLE-endpoint failure).
# ---------------------------------------------------------------------------


def _extract_watchdog_stanza(script: str) -> str:
    """Slice the ``_eps_reachability_watchdog() { ... }`` function body out of a
    rendered startup script by brace-matching from its definition."""
    start = script.index("_eps_reachability_watchdog() {")
    # Brace-match from the opening '{' to the matching '}'.
    depth = 0
    i = script.index("{", start)
    while i < len(script):
        if script[i] == "{":
            depth += 1
        elif script[i] == "}":
            depth -= 1
            if depth == 0:
                return script[start : i + 1]
        i += 1
    raise AssertionError("watchdog function braces did not balance")


def _run_watchdog(meta_pattern_succeeds: bool, ext_pattern_succeeds: bool, max_iters: int = 15):
    """Run the rendered watchdog stanza with stubbed ``curl`` / ``_eps_phase`` /
    shutdown ladder.

    ``curl`` returns success/failure by URL pattern (metadata 169.254.169.254 vs
    huggingface.co). ``sleep`` is stubbed to a no-op AND a guard exits the loop
    cleanly after ``max_iters`` iterations so a healthy (always-reset) loop
    terminates instead of spinning forever. Returns the (rc, stdout) of the run;
    ``PHASE_CALLED:wedged`` / ``SHUTDOWN_CALLED`` in stdout flags that the
    terminal wedge path fired.
    """
    import tempfile

    stanza = _extract_watchdog_stanza(
        render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    )
    meta_rc = 0 if meta_pattern_succeeds else 1
    ext_rc = 0 if ext_pattern_succeeds else 1
    shim = f"""#!/bin/bash
set -uo pipefail
_iters=0
# Stub the shutdown ladder + phase writer so the terminal path is OBSERVABLE
# (and never actually powers anything off in the test).
_eps_phase() {{ echo "PHASE_CALLED:$1"; }}
shutdown() {{ echo "SHUTDOWN_CALLED"; exit 0; }}
poweroff() {{ echo "POWEROFF_CALLED"; exit 0; }}
halt() {{ echo "HALT_CALLED"; exit 0; }}
# Route curl by URL pattern: metadata server vs the external HF endpoint.
curl() {{
  for a in "$@"; do
    case "$a" in
      *169.254.169.254*) return {meta_rc} ;;
      *huggingface.co*) return {ext_rc} ;;
    esac
  done
  return 0
}}
# No-op sleep + an iteration cap so a healthy (always-reset) loop terminates.
sleep() {{ _iters=$((_iters + 1)); if [ "$_iters" -gt {max_iters} ]; then exit 0; fi; }}
{stanza}
_eps_reachability_watchdog
"""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(shim)
        path = fh.name
    proc = subprocess.run(["bash", path], capture_output=True, text=True, timeout=30)
    return proc.returncode, proc.stdout


def test_watchdog_resets_when_metadata_succeeds_and_external_fails() -> None:
    """Blocker A: metadata healthy + HF down → ``fails`` resets every iteration
    (OR-of-successes), so the watchdog NEVER reaches the threshold and NEVER
    writes ``eps/phase=wedged`` or shuts the VM down. This is the exact false
    positive the pre-fix ``curl A && curl B`` conjunction produced (a transient
    HF outage on a HEALTHY VM)."""
    _rc, out = _run_watchdog(meta_pattern_succeeds=True, ext_pattern_succeeds=False)
    assert "PHASE_CALLED:wedged" not in out, out
    assert "SHUTDOWN_CALLED" not in out, out


def test_watchdog_resets_when_external_succeeds_and_metadata_fails() -> None:
    """Blocker A (symmetric): HF reachable + metadata probe failing → still a
    reset every iteration, no wedge, no shutdown."""
    _rc, out = _run_watchdog(meta_pattern_succeeds=False, ext_pattern_succeeds=True)
    assert "PHASE_CALLED:wedged" not in out, out
    assert "SHUTDOWN_CALLED" not in out, out


def test_watchdog_terminates_only_when_both_probes_fail() -> None:
    """Blocker A (the true-positive): only when BOTH probes fail for the full
    threshold does ``fails`` reach 10 → ``eps/phase=wedged`` is written and the
    shutdown ladder fires. Confirms the fix did not break genuine detection."""
    _rc, out = _run_watchdog(meta_pattern_succeeds=False, ext_pattern_succeeds=False)
    assert "PHASE_CALLED:wedged" in out, out
    assert "SHUTDOWN_CALLED" in out, out


def test_watchdog_resets_when_both_probes_succeed() -> None:
    """Healthy VM (both endpoints answer): no wedge, no shutdown — the obvious
    happy path, pinned alongside the OR-reset cases for completeness."""
    _rc, out = _run_watchdog(meta_pattern_succeeds=True, ext_pattern_succeeds=True)
    assert "PHASE_CALLED:wedged" not in out, out
    assert "SHUTDOWN_CALLED" not in out, out


def test_launch_handle_extra_carries_repo_branch(no_marker_posts) -> None:
    """#909 AC5: the GCP launch handle persists ``spec.extra['repo_branch']``
    (so the async GCP→RunPod failover reconstruction re-executes against the
    ISSUE branch, not `main`); "" when unset — an additive key only."""
    created_payload = json.dumps([{"name": "eps-issue-137", "id": "112233"}])
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[GcloudRunResult(0, created_payload, "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    handle = backend.launch(_spec(extra={"repo_branch": "issue-909"}))
    assert handle.extra["repo_branch"] == "issue-909"

    runner2 = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        create_results=[GcloudRunResult(0, created_payload, "")],
    )
    backend2 = GcpBackend(config=_test_config(), runner=runner2, marker_poster=lambda **_: None)
    handle2 = backend2.launch(_spec())
    assert handle2.extra["repo_branch"] == ""


# ---------------------------------------------------------------------------
# #934: per-lane instance-name suffix (lane_suffix)
# ---------------------------------------------------------------------------


def test_instance_name_for_lane_suffix() -> None:
    """Suffixed naming (#934) + unsuffixed byte-identity + the 63-char cap."""
    assert instance_name_for(137, "cpu") == "eps-issue-137-cpu"
    # Unsuffixed byte-identity: default arg AND explicit None.
    assert instance_name_for(137) == "eps-issue-137"
    assert instance_name_for(137, None) == "eps-issue-137"
    # Belt-and-suspenders: the COMPOSED name over 63 chars raises
    # (a 10-digit issue + a 43-char suffix = 64 chars).
    with pytest.raises(ValueError, match="63-char"):
        instance_name_for(1234567890, "a" * 43)


@pytest.mark.parametrize("bad", ["CPU", "a_b", "-x", "x-", "", "a.b", "x y"])
def test_validate_lane_suffix_rejects_malformed(bad: str) -> None:
    """Fail loud, never strip: every malformed suffix raises ValueError."""
    from explore_persona_space.backends.base import validate_lane_suffix

    with pytest.raises(ValueError):
        validate_lane_suffix(bad)


def test_validate_lane_suffix_max_length() -> None:
    """The 43-char cap is the ATTEMPT-LABEL budget (round-1 Must-Fix):
    len('att-YYYYmmdd-HHMMSS-') + suffix must fit the 63-char GCP label
    value cap, or the eps-attempt label is truncated and reconnect's
    label recovery desyncs from the VM's real per-attempt paths."""
    from explore_persona_space.backends.base import validate_lane_suffix

    assert validate_lane_suffix("a" * 43) == "a" * 43
    with pytest.raises(ValueError, match="attempt-label budget"):
        validate_lane_suffix("a" * 44)


def test_lane_suffix_for_rejects_invalid_extra() -> None:
    """A malformed spec.extra['lane_suffix'] raises — never a silent strip
    that would derive a divergent instance name."""
    from explore_persona_space.backends.gcp import lane_suffix_for

    with pytest.raises(ValueError):
        lane_suffix_for(_spec(extra={"lane_suffix": "Not_Valid"}))


def test_lane_suffix_for_absent_and_empty_are_none() -> None:
    from explore_persona_space.backends.gcp import lane_suffix_for

    assert lane_suffix_for(_spec()) is None
    assert lane_suffix_for(_spec(extra={"lane_suffix": ""})) is None
    assert lane_suffix_for(_spec(extra={"lane_suffix": "cpu"})) == "cpu"


def _suffixed_instance_payload(status: str, name: str = "eps-issue-137-cpu") -> str:
    return json.dumps(
        [
            {
                "name": name,
                "id": "424242",
                "status": status,
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-b"
                ),
            }
        ]
    )


def test_reconnect_probe_uses_lane_suffixed_name_filter() -> None:
    """The reconnect list filter carries the SUFFIXED exact name, so a
    suffixed lane never reconnects to a sibling lane's instance (#923)."""
    runner = _Runner(list_results=[GcloudRunResult(0, _suffixed_instance_payload("RUNNING"), "")])
    handle = reconnect_or_none(
        spec=_spec(extra={"lane_suffix": "cpu"}), config=_test_config(), runner=runner
    )
    assert handle is not None
    assert handle.pod_name == "eps-issue-137-cpu"
    list_calls = [c for c in runner.calls if "list" in c and "instances" in c]
    assert list_calls, runner.calls
    assert "--filter=name=eps-issue-137-cpu" in list_calls[0]


def test_reconnect_with_suffix_ignores_wrong_lane_record() -> None:
    """Belt-and-suspenders (#934 §11b): a live UNSUFFIXED eps-issue-137 in
    the list payload is NOT the suffixed lane's instance — the exact
    post-filter name check ignores it (CREATE proceeds, no reconnect)."""
    payload = json.dumps(
        [
            {
                "name": "eps-issue-137",
                "id": "1",
                "status": "RUNNING",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    "eps-test-project/zones/us-central1-a"
                ),
            }
        ]
    )
    runner = _Runner(list_results=[GcloudRunResult(0, payload, "")])
    assert (
        reconnect_or_none(
            spec=_spec(extra={"lane_suffix": "cpu"}), config=_test_config(), runner=runner
        )
        is None
    )


def test_stale_named_instance_uses_lane_suffixed_name() -> None:
    """The pre-launch stale reclaim probes the SUFFIXED name (#934)."""
    runner = _Runner(
        list_results=[GcloudRunResult(0, _suffixed_instance_payload("TERMINATED"), "")]
    )
    stale = _stale_named_instance_or_none(
        spec=_spec(extra={"lane_suffix": "cpu"}), config=_test_config(), runner=runner
    )
    assert isinstance(stale, StaleNamedInstance)
    assert stale.name == "eps-issue-137-cpu"
    assert stale.zone == "us-central1-b"


def test_launch_create_uses_lane_suffixed_instance_name_and_handle(no_marker_posts) -> None:
    """End-to-end create path (#934): the gcloud create argv, the handle
    pod_name, and extra['instance_name'] all carry the suffixed name."""
    created_payload = json.dumps([{"name": "eps-issue-137-cpu", "id": "5551"}])
    runner = _Runner(
        # 1st list = reconnect probe; 2nd = stale-name probe (both empty).
        list_results=[GcloudRunResult(0, "[]", ""), GcloudRunResult(0, "[]", "")],
        create_results=[GcloudRunResult(0, created_payload, "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    handle = backend.launch(_spec(extra={"lane_suffix": "cpu"}))
    assert handle.pod_name == "eps-issue-137-cpu"
    assert handle.extra["instance_name"] == "eps-issue-137-cpu"
    create_calls = [c for c in runner.calls if "create" in c and "instances" in c]
    assert len(create_calls) == 1, runner.calls
    assert "eps-issue-137-cpu" in create_calls[0]


def test_lane_suffixed_name_classifies_managed() -> None:
    """A suffixed name keeps the eps-issue- prefix, so the janitor's
    classification (and per-instance age fences) still covers it."""
    assert _classify_janitor_instance("eps-issue-137-cpu") == JANITOR_CLASS_MANAGED


# ---------------------------------------------------------------------------
# #935 — done-grace self-poweroff on the clean-exit path
# ---------------------------------------------------------------------------


def _extract_bash_function(script: str, name: str) -> str:
    """Return the rendered bash function ``<name>() { ... }`` verbatim.

    Scans from the definition line to the first column-0 ``}``, skipping any
    embedded ``<<'TERMINATOR'`` heredoc region so a python line inside the
    heredoc can never terminate the extraction early."""
    lines = script.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln == f"{name}() {{")
    in_heredoc = False
    terminator = ""
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if not in_heredoc:
            m = re.search(r"<<'(\w+)'$", ln)
            if m:
                in_heredoc = True
                terminator = m.group(1)
                continue
            if ln == "}":
                return "\n".join(lines[start : j + 1]) + "\n"
        elif ln == terminator:
            in_heredoc = False
    raise AssertionError(f"unterminated bash function {name}")


def test_render_startup_script_done_grace_poweroff_after_done_publish() -> None:
    """#935 acceptance criterion 1: BOTH branch renders carry the done-grace
    block, ordered strictly AFTER the completion-sentinel write + the
    ``_eps_phase done`` publish. The pin targets the standalone TAIL CALL
    line ``_eps_done_grace_poweroff || true`` — NOT the preamble function
    definition, which precedes ``_eps_phase done`` and would satisfy a naive
    name-index assert tautologically."""
    for spec in (_spec(), _workload_spec("bash scripts/issue935_smoke.sh")):
        script = render_startup_script(spec=spec, config=_test_config(), attempt_id="att-fixed-001")
        lines = script.splitlines()
        # Helpers live in the shared preamble (both branches).
        assert "_eps_done_grace_poweroff() {" in lines
        assert "_eps_persist_done_sentinels() {" in lines
        # THE TAIL CALL — exactly one standalone occurrence, after done.
        call_idx = lines.index("_eps_done_grace_poweroff || true")
        assert lines.count("_eps_done_grace_poweroff || true") == 1
        assert lines.index("_eps_phase done") < call_idx
        assert script.index('{"phase":"done"') < script.index("_eps_done_grace_poweroff || true")
        # Nothing but comments/blank lines after the tail call.
        residue = [ln for ln in lines[call_idx + 1 :] if ln.strip() and not ln.startswith("#")]
        assert residue == [], residue


def test_render_startup_script_done_grace_default_and_env_override(monkeypatch, caplog) -> None:
    """#935 acceptance criterion 2: the grace is env-tunable at render time —
    default 5400 s, ``EPS_GCP_DONE_POWEROFF_GRACE_SECONDS`` overrides, and a
    non-numeric value falls back to the default WITH a logged warning (the
    fail-loud-with-fallback claim on the knob, asserted via caplog)."""
    monkeypatch.delenv("EPS_GCP_DONE_POWEROFF_GRACE_SECONDS", raising=False)
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert "export EPS_DONE_GRACE=5400" in script
    # The keepalive escape hatch carries NO .json suffix (the poller's
    # sentinel drain glob is issue-<N>-*.json and must never ingest it).
    assert "export EPS_DONE_KEEPALIVE_PATH=/workspace/logs/issue-137-keepalive" in script
    assert "issue-137-keepalive.json" not in script
    monkeypatch.setenv("EPS_GCP_DONE_POWEROFF_GRACE_SECONDS", "7200")
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert "export EPS_DONE_GRACE=7200" in script
    monkeypatch.setenv("EPS_GCP_DONE_POWEROFF_GRACE_SECONDS", "ninety minutes")
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.backends.gcp"):
        script = render_startup_script(
            spec=_spec(), config=_test_config(), attempt_id="att-fixed-001"
        )
    assert "export EPS_DONE_GRACE=5400" in script
    assert any(
        "EPS_GCP_DONE_POWEROFF_GRACE_SECONDS" in rec.getMessage() for rec in caplog.records
    ), caplog.records


def test_render_startup_script_done_grace_zero_disables(monkeypatch) -> None:
    """#935: env ``0`` renders ``EPS_DONE_GRACE=0`` and the runtime disable
    guard (the case pattern) is present, so the countdown is a no-op; a
    negative value clamps to 0 (disable), never a negative export."""
    monkeypatch.setenv("EPS_GCP_DONE_POWEROFF_GRACE_SECONDS", "0")
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert "export EPS_DONE_GRACE=0" in script
    assert 'case "$_grace" in (*[!0-9]*|""|0)' in script
    assert "[done-grace] disabled" in script
    monkeypatch.setenv("EPS_GCP_DONE_POWEROFF_GRACE_SECONDS", "-100")
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert "export EPS_DONE_GRACE=0" in script


def test_render_startup_script_done_grace_aborts_on_phase_change_and_keepalive() -> None:
    """#935 acceptance criterion 3 (string level; the EXECUTED test below
    certifies runtime semantics): the countdown reads the eps/phase guest
    attribute via the metadata GET, aborts on ``!= "done"`` ONLY for a
    NON-EMPTY read (empty continues), and honors the keepalive file."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    fn = _extract_bash_function(script, "_eps_done_grace_poweroff")
    # The guest-attributes GET (curl -m 5, best-effort || true).
    assert "instance/guest-attributes/eps/phase' 2>/dev/null || true" in fn
    # Non-empty-and-changed aborts; the -n conjunct keeps an EMPTY read
    # (metadata server down / attr gone) COUNTING DOWN, never aborting.
    assert '[ -n "$ph" ] && [ "$ph" != "done" ]' in fn
    # The keepalive escape hatch (set -u-safe default).
    assert 'if [ -e "${EPS_DONE_KEEPALIVE_PATH:-/nonexistent}" ]; then' in fn
    # 60 s tick — fixed (guest-attr rate limit is 10 queries/min per VM).
    assert "tick=60" in fn


def test_render_startup_script_done_grace_persists_before_poweroff() -> None:
    """#935 acceptance criterion 4 (string level): at expiry the persist
    helper runs BEFORE the unconditional shutdown ladder; the persist is
    bounded (timeout 120), best-effort (|| true), targets
    issue<N>_done/<attempt_id>/ in one commit + transcript LAST, and writes
    the ok|failed breadcrumb on the SEPARATE eps/done_persist key — never
    touching eps/phase."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    countdown = _extract_bash_function(script, "_eps_done_grace_poweroff")
    assert countdown.index("_eps_persist_done_sentinels || true") < countdown.index(
        "shutdown -h now 2>/dev/null || poweroff -f 2>/dev/null || halt -f"
    )
    persist = _extract_bash_function(script, "_eps_persist_done_sentinels")
    assert "timeout 120 uv run python" in persist
    assert "issue${EPS_ISSUE:-0}_done/${EPS_ATTEMPT_ID:-unknown}" in persist
    assert "guest-attributes/eps/done_persist" in persist
    # ONE upload_folder commit + the transcript upload_file LAST.
    assert persist.index("api.upload_folder(") < persist.index("api.upload_file(")
    assert 'path_in_repo=f"{dest}/persist_transcript.log"' in persist
    # The breadcrumb key is SEPARATE: the persist helper NEVER writes
    # eps/phase (the poll classification + #908 predicates key on it).
    assert "guest-attributes/eps/phase" not in persist
    assert "_eps_phase " not in persist


@pytest.mark.parametrize(
    ("grace", "fake_phase", "keepalive", "expect_calls", "expect_out"),
    [
        # keepalive file present -> abort BEFORE persist/ladder.
        ("120", "done", True, [], "keepalive present"),
        # eps/phase left done (sanctioned relaunch) -> abort.
        ("120", "workload", False, [], "a relaunch owns the VM"),
        # EMPTY metadata read CONTINUES to expiry -> persist then ladder.
        ("120", "", False, ["persist", "shutdown -h now"], "grace expired"),
        # healthy done phase all the way -> expiry: persist then ladder.
        ("120", "done", False, ["persist", "shutdown -h now"], "grace expired"),
        # 0 disables outright (no sleep, no ladder).
        ("0", "done", False, [], "[done-grace] disabled"),
        # runtime non-numeric value disables (defense in depth).
        ("12abc", "done", False, [], "[done-grace] disabled"),
    ],
)
def test_done_grace_countdown_executes_abort_and_expiry_paths(
    tmp_path, grace, fake_phase, keepalive, expect_calls, expect_out
) -> None:
    """#935 acceptance criterion 3 (EXECUTED — runtime semantics the string
    pins + bash -n cannot certify): runs the extracted
    ``_eps_done_grace_poweroff`` with PATH-stubbed ``sleep``/``curl`` and
    shell-function stubs for the persist helper + the shutdown ladder that
    append to a call log. Certifies: keepalive abort, phase-change abort,
    empty-read-continues-to-expiry, 0-disables, and the persist-BEFORE-ladder
    ordering at expiry."""
    import shlex

    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    fn = _extract_bash_function(script, "_eps_done_grace_poweroff")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sleep").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "curl").write_text("#!/bin/bash\nprintf '%s' \"${FAKE_PHASE:-}\"\nexit 0\n")
    for stub in ("sleep", "curl"):
        (bin_dir / stub).chmod(0o755)

    call_log = tmp_path / "calls.log"
    keepalive_path = tmp_path / "issue-137-keepalive"
    if keepalive:
        keepalive_path.write_text("")
    driver = "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "trap ':' PIPE",
            fn,
            f'_eps_persist_done_sentinels() {{ echo "persist" >> {shlex.quote(str(call_log))}; }}',
            f'shutdown() {{ echo "shutdown $*" >> {shlex.quote(str(call_log))}; }}',
            f'poweroff() {{ echo "poweroff $*" >> {shlex.quote(str(call_log))}; }}',
            f'halt() {{ echo "halt $*" >> {shlex.quote(str(call_log))}; }}',
            "_eps_done_grace_poweroff || true",
            'echo "driver-exit-ok"',
            "",
        ]
    )
    driver_path = tmp_path / "driver.sh"
    driver_path.write_text(driver)
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{bin_dir}:{env.get('PATH', '/usr/bin:/bin')}",
            "EPS_DONE_GRACE": grace,
            "EPS_DONE_KEEPALIVE_PATH": str(keepalive_path),
            "FAKE_PHASE": fake_phase,
        }
    )
    proc = subprocess.run(
        ["bash", str(driver_path)], capture_output=True, text=True, env=env, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    assert "driver-exit-ok" in proc.stdout
    assert expect_out in proc.stdout
    calls = call_log.read_text().splitlines() if call_log.is_file() else []
    assert calls == expect_calls
    # The abort/disable paths must NEVER reach persist OR any ladder rung.
    if not expect_calls:
        assert "grace expired" not in proc.stdout


def _extract_done_persist_heredoc(script: str) -> str:
    """Return the EPS_DONE_PERSIST_PY heredoc body (the embedded python)."""
    lines = script.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.endswith("<<'EPS_DONE_PERSIST_PY'")]
    ends = [i for i, ln in enumerate(lines) if ln == "EPS_DONE_PERSIST_PY"]
    assert len(starts) == 1 and len(ends) == 1, (starts, ends)
    assert starts[0] < ends[0]
    return "\n".join(lines[starts[0] + 1 : ends[0]]) + "\n"


def _run_done_persist_heredoc(tmp_path, *, env_overrides=None, folder_failures=0):
    """Execute the REAL extracted EPS_DONE_PERSIST_PY heredoc against a fake
    ``huggingface_hub`` (records upload calls to a JSONL; the first
    ``folder_failures`` upload_folder calls raise AFTER recording, so the
    in-heredoc retry is observable), mirroring production's
    ``python - <dest>`` stdin invocation. Returns ``(proc, calls, paths)``."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    heredoc = _extract_done_persist_heredoc(script)

    shim = tmp_path / "shim" / "huggingface_hub"
    shim.mkdir(parents=True, exist_ok=True)
    calls_path = tmp_path / "calls.jsonl"
    budget_path = tmp_path / "folder-fail-budget"
    budget_path.write_text(str(folder_failures))
    (shim / "__init__.py").write_text(
        "import json, os\n"
        "class HfApi:\n"
        "    def _rec(self, kind, **kw):\n"
        "        with open(os.environ['FAKE_HUB_CALLS'], 'a') as fh:\n"
        "            fh.write(json.dumps({'kind': kind, **kw}) + '\\n')\n"
        "    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):\n"
        "        self._rec('file', path_in_repo=path_in_repo, repo_id=repo_id,\n"
        "                  repo_type=repo_type, nbytes=os.path.getsize(path_or_fileobj))\n"
        "    def upload_folder(self, *, folder_path, path_in_repo, repo_id, repo_type,\n"
        "                      ignore_patterns=None):\n"
        "        staged = {}\n"
        "        for dp, _dns, fns in os.walk(folder_path):\n"
        "            for fn in fns:\n"
        "                p = os.path.join(dp, fn)\n"
        "                rel = os.path.relpath(p, folder_path).replace(os.sep, '/')\n"
        "                with open(p, 'rb') as fh:\n"
        "                    staged[rel] = fh.read().decode('utf-8', 'replace')\n"
        "        self._rec('folder', path_in_repo=path_in_repo, repo_id=repo_id,\n"
        "                  repo_type=repo_type, staged=staged)\n"
        "        bp = os.environ.get('FAKE_HUB_FOLDER_FAIL_BUDGET', '')\n"
        "        if bp and os.path.exists(bp):\n"
        "            n = int(open(bp).read().strip() or '0')\n"
        "            if n > 0:\n"
        "                with open(bp, 'w') as fh:\n"
        "                    fh.write(str(n - 1))\n"
        "                raise RuntimeError('fake transient upload failure')\n"
    )

    # Fixture tree: completion sentinel, undrained /workspace/logs-style
    # sentinels (+ the keepalive escape hatch, which must NOT be staged),
    # and a small workload log.
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text('{"phase":"done","issue":137,"attempt_id":"att-x"}\n')
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "issue-137-results.json").write_text('{"kind":"epm:results"}\n')
    (logs_dir / "issue-137-keepalive").write_text("")
    log = tmp_path / "workload.log"
    log.write_text("workload tail line\n")

    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(tmp_path / "shim"),
            "FAKE_HUB_CALLS": str(calls_path),
            "FAKE_HUB_FOLDER_FAIL_BUDGET": str(budget_path),
            "EPS_HF_DATA_REPO": "org/repo",
            "EPS_ISSUE": "137",
            "EPS_ATTEMPT_ID": "att-x",
            "EPS_DONE_GRACE": "5400",
            "EPS_LOG_PATH": str(log),
            "EPS_SENTINEL_PATH": str(sentinel),
            "EPS_DONE_LOGS_DIR": str(logs_dir),
            "EPS_DONE_PERSIST_STAGE_DIR": str(tmp_path / "staged"),
            "EPS_DONE_PERSIST_STATUS": str(tmp_path / "status"),
            "EPS_DONE_PERSIST_TRANSCRIPT": str(tmp_path / "transcript.log"),
            "EPS_DONE_PERSIST_RETRY_BACKOFF_S": "0",
        }
    )
    env.update(env_overrides or {})
    proc = subprocess.run(
        [sys.executable, "-", "issue137_done/att-x"],
        input=heredoc,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    calls = []
    if calls_path.is_file():
        calls = [json.loads(ln) for ln in calls_path.read_text().splitlines()]
    paths = {"status": tmp_path / "status", "transcript": tmp_path / "transcript.log"}
    return proc, calls, paths


def test_done_persist_heredoc_uploads_sentinels_one_commit(tmp_path) -> None:
    """#935 acceptance criterion 4 (EXECUTED heredoc): the persist stages
    sentinel.json + the undrained issue-<N>-*.json sentinels + the workload
    log tail + done_report.json, uploads them in ONE upload_folder commit to
    issue<N>_done/<attempt_id>/ followed by the transcript upload_file LAST
    (exactly these two upload calls — never a per-file loop), retries ONCE
    on a first-attempt failure, and SKIP-ALLs when the env is unset."""
    # Variant A — happy path: exactly [folder, transcript-file], status ok.
    proc, calls, paths = _run_done_persist_heredoc(tmp_path / "a")
    assert proc.returncode == 0, proc.stderr
    assert [(c["kind"], c["path_in_repo"]) for c in calls] == [
        ("folder", "issue137_done/att-x"),
        ("file", "issue137_done/att-x/persist_transcript.log"),
    ]
    staged = calls[0]["staged"]
    assert set(staged) == {
        "sentinel.json",
        "logs_sentinels/issue-137-results.json",
        "workload_tail.log",
        "done_report.json",
    }, staged
    # The keepalive escape hatch (no .json suffix) is NEVER staged.
    assert not any("keepalive" in k for k in staged)
    report = json.loads(staged["done_report.json"])
    assert report["issue"] == "137" and report["attempt_id"] == "att-x"
    assert report["grace_s"] == "5400" and report["kind"] == "gcp-done-grace-sentinel-persist"
    assert staged["workload_tail.log"] == "workload tail line\n"
    assert paths["status"].read_text() == "ok"
    assert "[done-persist] DONE status=ok" in proc.stdout
    transcript_text = paths["transcript"].read_text()
    assert "[done-persist] BEGIN" in transcript_text
    assert "[done-persist] DONE status=ok" in transcript_text

    # Variant B — first upload_folder raises: the in-heredoc retry fires
    # (two folder attempts), then the transcript still lands LAST; status ok.
    proc, calls, paths = _run_done_persist_heredoc(tmp_path / "b", folder_failures=1)
    assert proc.returncode == 0, proc.stderr
    assert [(c["kind"], c["path_in_repo"]) for c in calls] == [
        ("folder", "issue137_done/att-x"),
        ("folder", "issue137_done/att-x"),
        ("file", "issue137_done/att-x/persist_transcript.log"),
    ]
    assert "[done-persist] FAILED upload attempt 1/2" in proc.stdout
    assert paths["status"].read_text() == "ok"

    # Variant C — BOTH attempts fail: status=failed, the poweroff is never
    # blocked (rc still 0), and the transcript audit STILL uploads LAST.
    proc, calls, paths = _run_done_persist_heredoc(tmp_path / "c", folder_failures=2)
    assert proc.returncode == 0, proc.stderr
    kinds = [(c["kind"], c["path_in_repo"]) for c in calls]
    assert kinds[-1] == ("file", "issue137_done/att-x/persist_transcript.log")
    assert len([k for k in kinds if k[0] == "folder"]) == 2
    assert paths["status"].read_text() == "failed"
    assert "[done-persist] DONE status=failed" in proc.stdout


def test_done_persist_bash_skip_all_without_repo_or_token(tmp_path) -> None:
    """#935: the bash-level SKIP-ALL guard — with EPS_HF_DATA_REPO/HF_TOKEN
    unset the persist function early-returns LOUDLY, never invoking uv (no
    heredoc run) and never writing the breadcrumb."""
    import shlex

    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    fn = _extract_bash_function(script, "_eps_persist_done_sentinels")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.log"
    (bin_dir / "uv").write_text(f'#!/bin/bash\necho "uv $*" >> {shlex.quote(str(call_log))}\n')
    (bin_dir / "curl").write_text(f'#!/bin/bash\necho "curl $*" >> {shlex.quote(str(call_log))}\n')
    for stub in ("uv", "curl"):
        (bin_dir / stub).chmod(0o755)
    driver = "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "trap ':' PIPE",
            fn,
            "_eps_persist_done_sentinels || true",
            'echo "after-persist"',
            "",
        ]
    )
    driver_path = tmp_path / "driver.sh"
    driver_path.write_text(driver)
    env = {k: v for k, v in os.environ.items() if k not in ("EPS_HF_DATA_REPO", "HF_TOKEN")}
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '/usr/bin:/bin')}"
    proc = subprocess.run(
        ["bash", str(driver_path)], capture_output=True, text=True, env=env, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    assert "[done-grace] SKIP persist" in proc.stdout
    assert "after-persist" in proc.stdout
    assert not call_log.exists(), call_log.read_text() if call_log.exists() else None


def test_poll_terminated_with_done_phase_maps_to_done_self_poweroff() -> None:
    """#935 acceptance criterion 6: a TERMINATED instance whose eps/phase
    reads ``done`` (the done-grace self-poweroff fired on the STOP outcome,
    or a manual stop of a done VM) classifies ``done`` with the
    ``workload_done_self_poweroff`` phase — a SUCCESSFUL run, never dead."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "TERMINATED"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("done"), "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "done"
    assert pr.current_phase == "workload_done_self_poweroff"
    assert pr.new_milestone is True


def test_poll_terminated_with_workload_phase_still_dead() -> None:
    """#935 negative control: TERMINATED + a NON-terminal phase (spot
    preemption / max-run-duration mid-run) keeps classifying dead with the
    ``terminal_terminated`` phase EXACTLY as today — asserting BOTH the
    status and the phase so the new done branch cannot widen."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "TERMINATED"}), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "dead"
    assert pr.current_phase == "terminal_terminated"


# ---------------------------------------------------------------------------
# issue #1029 — TERMINATED-window setup-death discrimination: a TERMINATED VM
# whose eps/phase reads "failed" runs the SAME §4.1.0b workload_started
# discrimination the RUNNING path runs, so the classification of a trap-written
# boot death is timing-independent (RUNNING window and TERMINATED window agree).
# ---------------------------------------------------------------------------


def test_terminated_with_failed_phase_and_no_workload_start_maps_to_terminal_setup_failed() -> None:
    """#1029 §4.1: TERMINATED + eps/phase=failed + workload_started ABSENT (the
    404 not-written case) is a deterministic PRE-WORKLOAD boot death ->
    ``terminal_setup_failed`` — the same classification the RUNNING window
    already produces for the identical death (timing-independent)."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "TERMINATED"}), "")],
        guest_attr_results=[
            GcloudRunResult(0, _guest_attr_payload_multi([("phase", "failed")]), ""),
            # The workload_started probe finds the attribute not written (404).
            GcloudRunResult(1, "", "guest attribute eps/workload_started not found"),
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "dead"
    assert pr.current_phase == "terminal_setup_failed"


def test_terminated_with_failed_phase_and_workload_started_keeps_terminal_terminated() -> None:
    """#1029 §4.1 (the #669 spot exclusion, preserved): TERMINATED +
    eps/phase=failed + workload_started PRESENT is a MID-RUN guest shutdown
    (e.g. a spot preemption whose EXIT trap completed) — it keeps
    ``terminal_terminated`` verbatim, so a lone preemption still never fails
    over and never counts as a deterministic boot death."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "TERMINATED"}), "")],
        guest_attr_results=[
            GcloudRunResult(0, _guest_attr_payload_multi([("phase", "failed")]), ""),
            GcloudRunResult(0, _guest_attr_payload_multi([("workload_started", "true")]), ""),
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "dead"
    assert pr.current_phase == "terminal_terminated"


def test_terminated_with_failed_phase_probe_error_keeps_terminal_terminated() -> None:
    """#1029 §4.1 (conservative fallback): TERMINATED + eps/phase=failed + a
    PROBE FAILURE on the workload_started read (auth/transport — NOT the 404
    not-written case) falls back to workload-started=True by
    ``_workload_started``'s existing contract -> keeps ``terminal_terminated``
    (never manufactures a setup classification from an unprovable read)."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "TERMINATED"}), "")],
        guest_attr_results=[
            GcloudRunResult(0, _guest_attr_payload_multi([("phase", "failed")]), ""),
            GcloudRunResult(
                1, "", "ERROR: Required 'compute.instances.getGuestAttributes' permission denied"
            ),
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "dead"
    assert pr.current_phase == "terminal_terminated"


# ---------------------------------------------------------------------------
# issue #1055 — finalize_failed_artifacts_ok: a post-deliverables finalize/tail
# crash publishes a distinct done-like phase instead of `failed`, keyed on the
# workload-written positive-evidence sentinel ($EPS_DELIVERABLES_OK_PATH). The
# sentinel-absent path stays byte-equivalent to today's failed path.
# ---------------------------------------------------------------------------


def _extract_exit_trap_body(script: str) -> str:
    """Pull the single-quoted EXIT-trap body out of a rendered startup script.

    The trap body contains no single quotes by bash construction, so the
    ``[^']*`` match is exact.
    """
    m = re.search(r"trap '([^']*)' EXIT", script)
    assert m is not None, "could not locate the EXIT trap in the rendered script"
    return m.group(1)


def test_render_startup_script_trap_branches_on_deliverables_sentinel() -> None:
    """#1055 acceptance criterion 1 — branch-STRUCTURE-discriminating, not
    substring-order-only: both render branches carry, inside the single trap
    statement, the guarded sentinel check, the finalize_failed_artifacts_ok
    then-arm, the retained failed else-arm, AND the inner ``fi;`` closing the
    sentinel conditional sits IMMEDIATELY after ``_eps_phase failed;`` and
    BEFORE the shared tail (watchdog kill -> log tail -> diagnostics ->
    poweroff) — so a mutant nesting the shared tail inside only one arm FAILS
    (a flat substring-order assert would pass it while one branch loses
    diagnostics or the billing-bounding poweroff)."""
    for spec in (
        _spec(),
        _spec(hydra_args=(), workload_cmd="bash scripts/issue658_dispatch.sh --flag 'v 1'"),
    ):
        script = render_startup_script(spec=spec, config=_test_config(), attempt_id="att-fixed-001")
        trap = _extract_exit_trap_body(script)
        assert '[ -n "${EPS_DELIVERABLES_OK_PATH:-}" ]' in trap
        assert '[ -f "${EPS_DELIVERABLES_OK_PATH:-}" ]' in trap
        idx_finalize = trap.index("_eps_phase finalize_failed_artifacts_ok;")
        idx_failed = trap.index("_eps_phase failed;")
        inner_fi = trap.index(" fi;", idx_failed)
        idx_kill = trap.index('{ kill "${EPS_WATCHDOG_PID:-}"')
        idx_diag = trap.index('_eps_persist_diagnostics "$rc"')
        idx_shutdown = trap.index("shutdown -h now")
        # then-arm before else-arm inside the sentinel conditional.
        assert idx_finalize < idx_failed
        # The inner fi; closes the sentinel conditional IMMEDIATELY after the
        # else-arm's phase write — NOTHING (no tail element) may sit between
        # them, or the shared tail has been nested into one arm.
        assert trap[idx_failed:inner_fi] == "_eps_phase failed;"
        # The shared tail runs on BOTH arms: strictly AFTER the inner fi, in
        # the unchanged order kill -> (log tail) -> diagnostics -> poweroff.
        assert idx_failed < inner_fi < idx_kill < idx_diag < idx_shutdown


def _run_trap_sandbox(tmp_path: Path, *, sentinel_present: bool) -> list[str]:
    """Execute the rendered EXIT-trap body in a sandbox bash with stubbed
    ``_eps_phase`` / ``_eps_persist_diagnostics`` / ``kill`` / ``shutdown``
    (each appends to a call-trace file) and rc=1; returns the call trace."""
    tag = "present" if sentinel_present else "absent"
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    body = _extract_exit_trap_body(script)
    call_log = tmp_path / f"calls_{tag}.log"
    log_file = tmp_path / "workload.log"
    log_file.write_text("line1\nline2\n")
    sentinel = tmp_path / f"deliverables_ok_{tag}.json"
    if sentinel_present:
        sentinel.write_text('{"issue": 137}')
    sandbox_lines = [
        "exec 3>/dev/null",
        f'CALL_LOG="{call_log}"',
        '_eps_phase() { echo "_eps_phase $1" >> "$CALL_LOG"; }',
        '_eps_persist_diagnostics() { echo "_eps_persist_diagnostics $1" >> "$CALL_LOG"; }',
        'kill() { echo "kill $*" >> "$CALL_LOG"; }',
        'shutdown() { echo "shutdown $*" >> "$CALL_LOG"; }',
        "export EPS_WATCHDOG_PID=99999",
        f'export EPS_LOG_PATH="{log_file}"',
        f'export EPS_DELIVERABLES_OK_PATH="{sentinel}"',
        "false",  # the trap body's rc=$? reads 1 from this
        body,
        "exit 0",
    ]
    sandbox = tmp_path / f"sandbox_{tag}.sh"
    sandbox.write_text("\n".join(sandbox_lines) + "\n")
    proc = subprocess.run(["bash", str(sandbox)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert call_log.exists(), "trap body made no stubbed calls"
    return call_log.read_text().split("\n")


def test_rendered_trap_body_executes_both_sentinel_states(tmp_path) -> None:
    """#1055 executed-trap semantics (not just structure): running the REAL
    rendered trap body with rc!=0 publishes finalize_failed_artifacts_ok when
    the deliverables sentinel file EXISTS and failed when it does not — and
    BOTH runs still execute the shared tail (diagnostics + poweroff), in the
    unchanged phase-write -> diagnostics -> shutdown order."""
    present = _run_trap_sandbox(tmp_path, sentinel_present=True)
    absent = _run_trap_sandbox(tmp_path, sentinel_present=False)

    assert "_eps_phase finalize_failed_artifacts_ok" in present
    assert "_eps_phase failed" not in present
    assert "_eps_phase failed" in absent
    assert "_eps_phase finalize_failed_artifacts_ok" not in absent
    for calls, phase_call in (
        (present, "_eps_phase finalize_failed_artifacts_ok"),
        (absent, "_eps_phase failed"),
    ):
        assert "_eps_persist_diagnostics 1" in calls  # diagnostics ran, rc preserved
        assert "shutdown -h now" in calls  # the billing-bounding poweroff ran
        assert calls.index(phase_call) < calls.index("_eps_persist_diagnostics 1")
        assert calls.index("_eps_persist_diagnostics 1") < calls.index("shutdown -h now")


def test_render_startup_script_exports_deliverables_ok_path_and_boot_rm() -> None:
    """#1055: both branches export the attempt-scoped positive-evidence
    sentinel path and rm -f it at boot (stale-evidence hygiene for a re-booted
    instance with the SAME attempt_id + preserved disk), with the rm AFTER the
    export and BEFORE the workload starts."""
    for spec in (
        _spec(),
        _spec(hydra_args=(), workload_cmd="bash scripts/issue658_dispatch.sh"),
    ):
        script = render_startup_script(spec=spec, config=_test_config(), attempt_id="att-fixed-001")
        export_idx = script.index("export EPS_DELIVERABLES_OK_PATH=")
        export_line = script[export_idx:].split("\n", 1)[0]
        # Attempt-scoping pinned, not just line presence: the rendered value
        # embeds the attempt id (mirrors sentinel_path_for).
        assert "att-fixed-001/deliverables_ok.json" in export_line
        rm_idx = script.index('rm -f "$EPS_DELIVERABLES_OK_PATH"')
        workload_idx = script.index("_eps_phase workload")
        assert export_idx < rm_idx < workload_idx


def test_poll_running_finalize_failed_phase_follows_fresh_relaunch_marker() -> None:
    """#1055 relaunch-follow (sibling of the #612 done/failed coverage): the
    eps/phase guest attribute freezes at the FIRST workload's terminal state,
    so RUNNING + finalize_failed_artifacts_ok + a FRESH epm:run-launched
    marker must follow the relaunched workload — NOT classify done and steer
    the orchestrator to finalize/teardown mid-relaunch."""
    backend, _runner = _relaunch_backend(
        phase="finalize_failed_artifacts_ok",
        ssh_results=[
            GcloudRunResult(0, _EMPTY_DRAIN_STDOUT, ""),
            GcloudRunResult(0, _probe_stdout(alive=True), ""),
        ],
    )
    pr = backend.poll(_drain_handle())
    assert pr.status == "running"
    assert pr.current_phase == "relaunched_workload"


def test_poll_running_with_finalize_failed_artifacts_ok_maps_to_done() -> None:
    """#1055: the brief RUNNING window between the trap's phase write and the
    poweroff completing classifies done / workload_done_finalize_failed (no
    relaunch marker present), mirroring the RUNNING-window done block."""
    backend, _runner = _relaunch_backend(
        phase="finalize_failed_artifacts_ok",
        ssh_results=[GcloudRunResult(0, _EMPTY_DRAIN_STDOUT, "")],
        reader=_relaunch_reader(run_ts=None, cluster_ts=None),
    )
    pr = backend.poll(_drain_handle())
    assert pr.status == "done"
    assert pr.current_phase == "workload_done_finalize_failed"
    assert pr.new_milestone is True
    assert pr.pid_alive is False


def test_poll_terminated_with_finalize_failed_artifacts_ok_maps_to_done() -> None:
    """#1055 (mirror of the #935 TERMINATED-window test): a TERMINATED
    instance whose eps/phase reads finalize_failed_artifacts_ok — deliverables
    verified on HF, then a finalize/tail non-zero exit powered the VM off —
    classifies done with the distinct workload_done_finalize_failed phase, a
    SUCCESSFUL run whose finalize hiccupped, never a crash (no RunPod
    failover, no crash-fix routing)."""
    runner = _Runner(
        describe_results=[GcloudRunResult(0, json.dumps({"status": "TERMINATED"}), "")],
        guest_attr_results=[
            GcloudRunResult(0, _guest_attr_payload("finalize_failed_artifacts_ok"), "")
        ],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    pr = backend.poll(_poll_handle())
    assert pr.status == "done"
    assert pr.current_phase == "workload_done_finalize_failed"
    assert pr.new_milestone is True


def test_audit_stale_gcp_vms_reaps_terminal_phase_finalize_failed_zombie() -> None:
    """#1055 (sibling of the done/failed terminal-phase reap tests): a RUNNING
    VM stuck in finalize_failed_artifacts_ok past the terminal-phase floor is
    a finished zombie — the workload is over, the deliverables are on HF —
    and the janitor reaps it promptly via _TERMINAL_GUEST_PHASES membership."""
    now = datetime(2026, 7, 5, 12, 0, 0, tzinfo=UTC)
    created = (now - timedelta(minutes=30)).isoformat()
    runner = _Runner(
        list_results=[GcloudRunResult(0, _one_running_instance("eps-issue-1055", created), "")],
        guest_attr_results=[
            GcloudRunResult(0, _guest_attr_payload("finalize_failed_artifacts_ok"), "")
        ],
        delete_results=[GcloudRunResult(0, "", "")],
    )
    records = audit_stale_gcp_vms(
        config=_test_config(),
        runner=runner,
        max_age_seconds=24 * 3600,
        terminal_phase_max_age_seconds=600,
        now=now,
        delete=True,
    )
    assert records[0]["action"] == "deleted"
    assert records[0]["reason"] == "terminal-phase"
    assert records[0]["phase"] == "finalize_failed_artifacts_ok"


# ---------------------------------------------------------------------------
# #1205 — GCE push-verify backstop (incident #825 r6-r8: a workload's git
# push of committed eval JSONs failed deterministically on auth, the
# `|| echo WARNING` shape swallowed it, and the self-DELETEing instance
# held the only copy of the commit — upload-verification reads
# 2026-07-08T11:17/11:19Z)
# ---------------------------------------------------------------------------


_WORKLOAD_SPEC_KW = {"hydra_args": (), "workload_cmd": "bash scripts/x.sh --flag 'v 1'"}
_PUSH_LEG_OPEN = "# === Push-verify leg"
_PUSH_LEG_CLOSE = "# === /push-verify leg ==="


def _extract_push_verify_leg(script: str) -> str:
    """Return the push-verify leg between its slice markers.

    Asserts marker UNIQUENESS (exactly one opener + one closer, opener
    first) before slicing — the ``_extract_persist_heredoc`` precedent —
    so the executable test can never silently run a partial or doubled
    slice. Returns the lines from the opener through the closer.
    """
    lines = script.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith(_PUSH_LEG_OPEN)]
    ends = [i for i, ln in enumerate(lines) if ln == _PUSH_LEG_CLOSE]
    assert len(starts) == 1 and len(ends) == 1, (starts, ends)
    assert starts[0] < ends[0]
    return "\n".join(lines[starts[0] : ends[0] + 1]) + "\n"


def test_render_startup_script_workload_cmd_carries_push_verify_leg() -> None:
    """Durability pin (#1205): the workload_cmd branch renders the
    push-verify leg AFTER the detached-wait loop and BEFORE the #750 OOM
    guard, with the rev-list push-landed predicate, the retry, the
    data/issue_<N>/ bundle fallback, and the loud exit 86."""
    script = render_startup_script(
        spec=_spec(**_WORKLOAD_SPEC_KW), config=_test_config(), attempt_id="att-fixed-001"
    )
    # Content: the rev-list predicate, branch pin, retry, bundle, rc.
    assert "_EPS_PUSH_BRANCH=main" in script
    assert 'rev-list --count "origin/${_EPS_PUSH_BRANCH}..HEAD"' in script
    assert 'push origin "HEAD:${_EPS_PUSH_BRANCH}"' in script
    assert "sleep 20" in script
    assert '"$WORKLOAD_ROOT/data/issue_137' in script
    assert ".bundle" in script
    assert "exit 86" in script
    assert "[push-verify] FAIL" in script
    assert "[push-verify] OK" in script
    # Ordering: detached-wait loop < leg < OOM guard (the leg must run
    # before EPS_OOM_FINAL; a rc-survived-OOM + failed-push run records
    # rc 86, not 137 — the accepted rare double-failure).
    i_wait = script.index('echo "[startup-script] detached workload pid=$wpid exited"')
    i_leg = script.index(_PUSH_LEG_OPEN)
    i_close = script.index(_PUSH_LEG_CLOSE)
    i_oom = script.index('EPS_OOM_FINAL="$(_eps_oom_count)"')
    assert i_wait < i_leg < i_close < i_oom, (i_wait, i_leg, i_close, i_oom)
    # A non-main repo_branch threads into the leg's branch pin.
    script_wt = render_startup_script(
        spec=_spec(**_WORKLOAD_SPEC_KW),
        config=_test_config(),
        attempt_id="att-fixed-001",
        repo_branch="issue-1205",
    )
    assert "_EPS_PUSH_BRANCH=issue-1205" in script_wt


def test_render_startup_script_workload_cmd_git_credential_gated_on_token() -> None:
    """#1205: the credential helper is (a) gated on GITHUB_TOKEN presence
    (a token-less launch keeps today's behavior), and (b) single-quoted so
    the literal ``${GITHUB_TOKEN}`` stays UNEXPANDED in .git/config — git's
    sh-invoked helper expands it from the process env at push time, so the
    token is never at rest in the repo config."""
    script = render_startup_script(
        spec=_spec(**_WORKLOAD_SPEC_KW), config=_test_config(), attempt_id="att-fixed-001"
    )
    assert 'if [ -n "${GITHUB_TOKEN:-}" ]; then' in script
    config_line = next(ln for ln in script.splitlines() if "credential.helper" in ln)
    assert "'!f() { echo username=x-access-token;" in config_line
    assert 'echo "password=${GITHUB_TOKEN}"; }; f\'' in config_line
    assert "export GIT_TERMINAL_PROMPT=0" in script
    # The credential block renders BEFORE the workload phase (the
    # workload's OWN pushes are the fix at the source; the leg is the
    # backstop).
    assert script.index("credential.helper") < script.index("_eps_phase workload")
    # The metadata fetch stanza carries the new OPTIONAL secret key.
    assert "instance/attributes/GITHUB_TOKEN" in script


def test_render_startup_script_hydra_only_has_no_push_verify_leg() -> None:
    """#1205 negative: the hydra branch is leg-free and helper-free
    (train.py has no git usage) — its ONLY delta is the GITHUB_TOKEN
    metadata-fetch line from the shared secrets stanza (pinned by the
    deliberately-regenerated #588 snapshot fixture)."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    assert "[push-verify]" not in script
    assert _PUSH_LEG_OPEN not in script
    assert "credential.helper" not in script
    assert "GIT_TERMINAL_PROMPT" not in script
    assert "instance/attributes/GITHUB_TOKEN" in script


def test_resolve_launch_secrets_github_token_optional() -> None:
    """#1205: GITHUB_TOKEN keeps the drop-when-absent contract (it is NOT
    in REQUIRED_LAUNCH_SECRET_KEYS — plenty of GCE workloads never push);
    present, it threads into spec.extra['secret_GITHUB_TOKEN'] like the
    other optional secrets."""
    assert "GITHUB_TOKEN" not in REQUIRED_LAUNCH_SECRET_KEYS
    spec = _spec()
    resolve_launch_secrets(spec, env={"HF_TOKEN": "t-hf", "WANDB_API_KEY": "t-wb"})
    assert "secret_GITHUB_TOKEN" not in spec.extra
    spec2 = _spec()
    resolve_launch_secrets(
        spec2,
        env={"HF_TOKEN": "t-hf", "WANDB_API_KEY": "t-wb", "GITHUB_TOKEN": "ghp-test"},
    )
    assert spec2.extra["secret_GITHUB_TOKEN"] == "ghp-test"


def _git(cwd: Path, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run a git command in ``cwd`` with the hermetic test env, check=True."""
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    )


def _push_leg_rig(tmp_path: Path) -> tuple[Path, Path, dict[str, str], Path]:
    """Build the tmp-repo rig for the executable push-verify-leg test.

    Bare origin (branch ``main``) seeded with one commit + a DEPTH-1
    ``file://`` workload clone (the production GCE clone shape —
    assumption 6: the shallow boundary sits below origin/main, so the
    ``origin/main..HEAD`` count is unaffected). Returns
    ``(origin, workload, env, runner_path)`` where ``runner_path`` is the
    sliced leg wrapped in the production shell semantics
    (``set -euo pipefail``, the #1004 outer flags).
    """
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "--initial-branch=main", str(origin), env=env)
    seed = tmp_path / "seed"
    _git(tmp_path, "init", "--initial-branch=main", str(seed), env=env)
    _git(seed, "config", "user.email", "t@example.com", env=env)
    _git(seed, "config", "user.name", "t", env=env)
    (seed / "base.json").write_text("{}\n")
    _git(seed, "add", "base.json", env=env)
    _git(seed, "commit", "-m", "seed", env=env)
    _git(seed, "push", f"file://{origin}", "main:main", env=env)
    workload = tmp_path / "workload"
    _git(
        tmp_path,
        "clone",
        "--depth",
        "1",
        "--branch",
        "main",
        f"file://{origin}",
        str(workload),
        env=env,
    )
    _git(workload, "config", "user.email", "t@example.com", env=env)
    _git(workload, "config", "user.name", "t", env=env)
    script = render_startup_script(
        spec=_spec(**_WORKLOAD_SPEC_KW),
        config=_test_config(),
        attempt_id="att-fixed-001",
        repo_branch="main",
    )
    leg = _extract_push_verify_leg(script)
    runner_path = tmp_path / "run_leg.sh"
    runner_path.write_text("#!/bin/bash\nset -euo pipefail\n" + leg)
    env["WORKLOAD_ROOT"] = str(workload)
    return origin, workload, env, runner_path


def _run_leg(runner_path: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Execute the sliced leg under production shell semantics, bounded."""
    return subprocess.run(
        ["bash", str(runner_path)], env=env, capture_output=True, text=True, timeout=180
    )


def test_push_verify_leg_executes_in_tmp_repo_case_a_retry_lands(tmp_path: Path) -> None:
    """Case A (#1205): an unpushed commit with a REACHABLE origin — the
    leg's own (re)push lands it, the re-count reads 0, rc 0."""
    origin, workload, env, runner = _push_leg_rig(tmp_path)
    (workload / "result.json").write_text("{}\n")
    _git(workload, "add", "result.json", env=env)
    _git(workload, "commit", "-m", "eval results", env=env)
    head = _git(workload, "rev-parse", "HEAD", env=env).stdout.strip()
    proc = _run_leg(runner, env)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "retrying push (#1205)" in proc.stdout
    assert "[push-verify] retry landed" in proc.stdout
    origin_tip = _git(tmp_path, "-C", str(origin), "rev-parse", "main", env=env).stdout.strip()
    assert origin_tip == head


def test_push_verify_leg_executes_in_tmp_repo_case_b_fail_loud_bundle(tmp_path: Path) -> None:
    """Case B (#1205): an unpushed commit with a BROKEN origin — both push
    attempts fail, the leg bundles the unpushed range into data/issue_<N>/
    and exits 86 (the loud-fail rc the EXIT trap routes to phase=failed +
    crash-persist). The bundle verifies against an origin-holding clone,
    so the #825 rescue is mechanical. The unreachable origin is bounded by
    the subprocess timeout (assumption 11)."""
    origin, workload, env, runner = _push_leg_rig(tmp_path)
    (workload / "result.json").write_text("{}\n")
    _git(workload, "add", "result.json", env=env)
    _git(workload, "commit", "-m", "eval results", env=env)
    head = _git(workload, "rev-parse", "HEAD", env=env).stdout.strip()
    _git(workload, "remote", "set-url", "origin", f"file://{tmp_path}/no-such-origin.git", env=env)
    proc = _run_leg(runner, env)
    assert proc.returncode == 86, (proc.returncode, proc.stdout, proc.stderr)
    assert "[push-verify] FAIL" in proc.stdout
    bundles = sorted((workload / "data" / "issue_137").glob("unpushed-*.bundle"))
    assert len(bundles) == 1, bundles
    # The bundle's prerequisite (the old origin/main tip) is present in any
    # origin-holding clone, so `git bundle verify` passes there; the range
    # bundle records the HEAD ref, so fetching HEAD from the bundle
    # recovers the exact stranded commit (FETCH_HEAD == the workload tip).
    rescue = tmp_path / "rescue"
    _git(tmp_path, "clone", f"file://{origin}", str(rescue), env=env)
    _git(rescue, "bundle", "verify", str(bundles[0]), env=env)
    _git(rescue, "fetch", str(bundles[0]), "HEAD", env=env)
    rescued_tip = _git(rescue, "rev-parse", "FETCH_HEAD", env=env).stdout.strip()
    assert rescued_tip == head


def test_push_verify_leg_executes_in_tmp_repo_case_c_noop(tmp_path: Path) -> None:
    """Case C (#1205): nothing unpushed — the leg is a logged no-op
    (rc 0, the OK line, no push, no bundle dir)."""
    _origin, workload, env, runner = _push_leg_rig(tmp_path)
    proc = _run_leg(runner, env)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "[push-verify] OK: no unpushed commits" in proc.stdout
    assert not (workload / "data" / "issue_137").exists()
