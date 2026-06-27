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
import os
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

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        # gcloud compute instances <subcommand> ...
        if "create" in argv and "instances" in argv:
            return self._pop(self.create_results, default_ok=True)
        if "list" in argv and "instances" in argv:
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
    assert cfg.default_max_run_duration == "24h"


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
    assert "--max-run-duration=24h" in argv
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


# ---------------------------------------------------------------------------
# #656 — A100-40 fallback rung (a2-highgpu-1g)
# ---------------------------------------------------------------------------


def test_a100_40_fallback_for_intent_fits_predicate() -> None:
    """T10: the fits-in-40GB predicate. Single-GPU 7B-scale intents (lora-7b /
    lora / eval / debug) map to the A100-40 (a2-highgpu-1g) fallback machine;
    multi-GPU full-FT (ft-7b) and the 70B / unknown intents return None (a
    40 GB card cannot hold them, so the ladder has no A100-40 rung)."""
    for intent in ("lora-7b", "lora", "eval", "debug"):
        machine = a100_40_fallback_for_intent(_spec(intent))
        assert isinstance(machine, MachineSpec), intent
        assert machine.machine_type == "a2-highgpu-1g", intent
        assert machine.gpu_count == 1, intent
        assert machine.gpu_kind == "A100-40", intent
    for intent in ("ft-7b", "inf-70b", "ft-70b", "totally-bogus"):
        assert a100_40_fallback_for_intent(_spec(intent)) is None, intent
    # The module-level map matches the predicate's positive set exactly.
    assert set(INTENT_A100_40_FALLBACK) == {"lora-7b", "lora", "eval", "debug"}


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
    # max-run-duration default (config 24h)
    assert "--max-run-duration=24h" in argv
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
    # Default git paths
    assert "eval_results/issue_137/" in decl["git_paths"]
    assert "figures/issue_137/" in decl["git_paths"]
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
    # The keystone sentinel + the convention-stable git paths still gate.
    assert decl["sentinel_path"].endswith(".completion-sentinel.json")
    assert "eval_results/issue_137/" in decl["git_paths"]
    assert "figures/issue_137/" in decl["git_paths"]
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
    VM (data loss). The success criterion's data-loss guard."""
    from explore_persona_space.backends.gcp import GcpBackendError

    runner = _Runner(list_results=[GcloudRunResult(0, _instance_payload("RUNNING"), "")])
    with pytest.raises(GcpBackendError, match="non-deletable status"):
        _stale_named_instance_or_none(spec=_spec(), config=_test_config(), runner=runner)


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
        lambda *, paths, io: {"status": "SKIP", "detail": "no git paths declared"},
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
    traceback / stderr) AND the partial eval_results the workload wrote
    before crashing — the two things #658 lost on every retry."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    # Log + crash report upload.
    assert "workload.log" in script
    assert "crash_report.json" in script
    # Partial artifacts: the workload's eval_results/issue_<N>/ dir.
    assert 'eval_results" / f"issue_{issue}"' in script
    assert "upload_folder" in script
    # Destination prefix isolates partial output per attempt.
    assert "issue${EPS_ISSUE:-0}_partial/${EPS_ATTEMPT_ID:-unknown}" in script


def test_render_startup_script_diagnostics_is_guarded_and_bounded() -> None:
    """The crash-upload must NEVER delay the poweroff that bounds billing:
    it early-returns without a repo/token, time-bounds the upload, and the
    trap call is on the non-aborting (``set +e``) crash path."""
    script = render_startup_script(spec=_spec(), config=_test_config(), attempt_id="att-fixed-001")
    # Early-return when the repo target / token is absent (early-boot crash).
    assert (
        'if [ -z "${EPS_HF_DATA_REPO:-}" ] || [ -z "${HF_TOKEN:-}" ]; then return 0; fi' in script
    )
    # Hard time bound on the upload so a hung HF call can't strand the VM.
    assert "timeout 300 uv run python" in script
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


def test_render_startup_script_is_valid_bash() -> None:
    """Both rendered branches must parse — the #658 helper embeds a Python
    heredoc inside a function inside a subshell; a quoting slip would only
    surface at VM-boot time. ``bash -n`` is the syntax gate (shellcheck is
    not installed on the dev VM)."""
    import subprocess
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
# issue #588 — custom workload_cmd rendering + validation + launch
# ---------------------------------------------------------------------------


def _workload_spec(cmd: str = "bash scripts/issue588_smoke.sh") -> RunSpec:
    """``_spec()`` twin carrying a custom workload_cmd (no hydra args)."""
    return _spec(hydra_args=(), workload_cmd=cmd)


def test_render_startup_script_workload_cmd_verbatim_with_lifecycle_intact() -> None:
    """#588: the custom command replaces ONLY the workload line — every
    lifecycle pin (secrets fetch, in-VM preflight, phase publishing,
    EXIT trap, completion sentinel) is unchanged."""
    script = render_startup_script(
        spec=_workload_spec("bash scripts/issue588_smoke.sh --flag 'v 1'"),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    lines = script.splitlines()
    # The command is embedded VERBATIM as its own line (no shlex-quoting
    # that would collapse it to a single token).
    assert "bash scripts/issue588_smoke.sh --flag 'v 1'" in lines
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
        "bash scripts/issue588_smoke.sh --flag 'v 1'"
    )
    # WandB project default (#601 follow-up r1): exported BEFORE the
    # workload so HF-Trainer runs stop landing in the global default
    # 'huggingface' project; :- keeps an inline/internal override winning.
    wandb_export = 'export WANDB_PROJECT="${WANDB_PROJECT:-issue137}"'
    assert wandb_export in lines
    assert lines.index(wandb_export) < lines.index("bash scripts/issue588_smoke.sh --flag 'v 1'")
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
    assert lines.index(repo_root_export) < lines.index("bash scripts/issue588_smoke.sh")
    # The hydra branch must NOT gain the export — it runs scripts/train.py
    # directly (no REPO_ROOT dependency), and the #588 byte-identity
    # snapshot fixture pins the hydra branch unchanged.
    hydra_script = render_startup_script(
        spec=_spec(),
        config=_test_config(),
        attempt_id="att-fixed-001",
    )
    assert "REPO_ROOT" not in hydra_script


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
    i_cmd = lines.index("bash scripts/issue588_smoke.sh")
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
    assert lines.index("mkdir -p /workspace/logs") < lines.index("bash scripts/issue588_smoke.sh")
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
        tmp_path, monkeypatch, ssh_results=[GcloudRunResult(0, sentinel_text, "")]
    )
    backend.fetch_results(handle)

    ssh_calls = [argv for argv in runner.calls if "ssh" in argv]
    assert len(ssh_calls) == 1
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
    # The sentinel is never scp'd; the 2 best-effort dir pulls stay scp.
    scp_calls = [argv for argv in runner.calls if "scp" in argv]
    assert len(scp_calls) == 2
    assert all("--recurse" in argv for argv in scp_calls)
    assert not any(sentinel_abs in token for argv in scp_calls for token in argv)


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
        ssh_results=[GcloudRunResult(1, "", "sudo: a password is required")],
    )
    with caplog.at_level(logging.ERROR):
        backend.fetch_results(handle)  # must not raise

    assert not Path(sentinel_abs).exists()
    assert "confirm_artifacts will FAIL" in caplog.text
    scp_calls = [argv for argv in runner.calls if "scp" in argv]
    assert len(scp_calls) == 2  # best-effort pulls still attempted


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
    import subprocess

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
    import subprocess

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
    import subprocess
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
