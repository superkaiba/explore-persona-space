"""GCE-wiring regression guards for the #674 EPS_SCRATCH_DIR scratch routing.

These pin the two edits that make the per-cell .npz scratch-routing helper
actually ROUTE on the GCE lane (the v1-plan defect was shipping the helper
with EPS_SCRATCH_DIR never wired into the VM workload env):

  1. EPS_SCRATCH_DIR is in gcp.STARTUP_PASSTHROUGH_ENV_KEYS (so a
     dispatch-process value forwards into VM metadata -> workload env).
  2. The rendered workload_cmd startup script exports a local-SSD default
     (so the GCE lane gets scratch routing even with no upstream override).
"""

from __future__ import annotations

from explore_persona_space.backends import gcp
from explore_persona_space.backends.base import RunSpec
from explore_persona_space.backends.gcp import GcpConfig, render_startup_script


def test_eps_scratch_dir_in_passthrough_keys():
    # #674: EPS_SCRATCH_DIR must be forwarded into the GCE workload env so the
    # per-cell .npz scratch-routing helper actually routes on GCE.
    assert "EPS_SCRATCH_DIR" in gcp.STARTUP_PASSTHROUGH_ENV_KEYS


def _workload_spec() -> RunSpec:
    return RunSpec(
        issue=674,
        intent="eval",
        backend="gcp",
        hydra_args=(),
        workload_cmd="echo hi",
        extra={"attempt_id": "att-fixed-674"},
    )


def _config() -> GcpConfig:
    return GcpConfig(
        project="eps-test-project",
        gcloud_config="eps-test-config",
        primary_zone="us-central1-a",
        fallback_zones=("us-central1-b", "us-central1-c"),
        image_family="pytorch-test-family",
    )


def test_workload_startup_script_exports_eps_scratch_dir_default():
    # #674: the workload_cmd branch defaults EPS_SCRATCH_DIR to a local-SSD path
    # (:- fills only unset/empty, so a forwarded value still wins).
    script = render_startup_script(
        spec=_workload_spec(), config=_config(), attempt_id="att-fixed-674"
    )
    assert 'export EPS_SCRATCH_DIR="${EPS_SCRATCH_DIR:-/tmp/eps_scratch}"' in script
