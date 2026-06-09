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
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from explore_persona_space.backends import (
    EXPECTED_ARTIFACTS_HANDLE_KEY,
    INTENT_TO_MACHINE,
    GcpBackend,
    GcpConfig,
    GcpProvisioningError,
    MachineSpec,
    RunSpec,
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
    GcloudRunResult,
    attempt_id_for,
    classify_create_failure,
    instance_name_for,
    machine_for_intent,
    reconnect_or_none,
    render_delete_argv,
    render_describe_argv,
    render_list_argv,
    render_startup_script,
    resolve_provisioning_model,
    sentinel_path_for,
)

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


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
    ) -> None:
        self.calls: list[list[str]] = []
        self.create_results = list(create_results or [])
        self.list_results = list(list_results or [])
        self.describe_results = list(describe_results or [])
        self.delete_results = list(delete_results or [])
        self.serial_results = list(serial_results or [])

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
        if "delete" in argv and "instances" in argv:
            return self._pop(self.delete_results, default_ok=True)
        if "get-serial-port-output" in argv:
            return self._pop(self.serial_results, default_ok=True)
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


def test_render_create_argv_ft_intent_uses_4gpu_machine() -> None:
    cfg = _test_config()
    argv = render_create_argv(
        spec=_spec("ft-7b"),
        config=cfg,
        attempt_id="att-fixed-001",
        startup_script="#!/bin/bash\n",
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
    )
    assert "--zone=us-central1-c" in argv
    assert "--zone=us-central1-a" not in argv


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
    )
    assert "--metadata-from-file=startup-script=/tmp/eps-startup.sh" in argv
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


def test_reconnect_returns_none_on_gcloud_failure() -> None:
    """Best-effort optimization — a transient gcloud blip falls through to create."""
    runner = _Runner(list_results=[GcloudRunResult(1, "", "auth blip")])
    assert reconnect_or_none(spec=_spec(), config=_test_config(), runner=runner) is None


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
        log_path="/workspace/eps-issue-137/logs/issue-137.log",
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


def test_launch_retries_on_capacity_then_succeeds_in_fallback_zone(no_marker_posts) -> None:
    """Capacity miss in primary zone must transparently retry the
    fallback zones before giving up."""
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
    handle = backend.launch(_spec())
    # The second create succeeded; we landed in us-central1-b.
    assert handle.extra["zone"] == "us-central1-b"
    # Two create calls were issued.
    create_calls = [a for a in runner.calls if "create" in a]
    assert len(create_calls) == 2
    assert "--zone=us-central1-a" in create_calls[0]
    assert "--zone=us-central1-b" in create_calls[1]


def test_launch_raises_provisioning_error_when_all_zones_capacity_fail(no_marker_posts) -> None:
    runner = _Runner(
        list_results=[GcloudRunResult(0, "[]", "")],
        # 3 capacity failures: primary + 2 fallbacks
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
        backend.launch(_spec())


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
        log_path="/workspace/eps-issue-137/logs/issue-137.log",
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
        log_path="/workspace/eps-issue-137/logs/issue-137.log",
        extra={"zone": "us-central1-a"},
    )
    with pytest.raises(GcpBackendError, match="Internal server error"):
        backend.teardown(handle)


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
        log_path="/workspace/eps-issue-137/logs/issue-137.log",
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
        log_path="/workspace/eps-issue-137/logs/issue-137.log",
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
        log_path="/workspace/eps-issue-137/logs/issue-137.log",
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


def test_audit_stale_gcp_vms_skips_non_eps_instances() -> None:
    """The reaper MUST only consider eps-issue-* instances — never delete
    a personal VM in the same project just because it's old."""
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
    assert records == []


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
