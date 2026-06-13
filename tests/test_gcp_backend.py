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
    REQUIRED_LAUNCH_SECRET_KEYS,
    GcloudRunResult,
    GcpLaunchSecretsMissing,
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


def test_render_startup_script_exports_tqdm_disable() -> None:
    """T3 (#607): ``TQDM_DISABLE=1`` exported before the workload in both
    branches — defense in depth (tqdm bars are the canonical giant-line
    producer and pure noise in a log file; vLLM + huggingface_hub bars
    are tqdm-based)."""
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
        assert "export TQDM_DISABLE=1" in script
        assert script.index("export TQDM_DISABLE=1") < script.index(workload_line)


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
