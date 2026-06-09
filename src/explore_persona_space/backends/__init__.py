"""Compute-backend abstraction for `/issue` experiment runs.

The `/issue` skill dispatches each task to a :class:`~base.ComputeBackend`:

* :class:`runpod.RunPodBackend` (default) — wraps the existing
  ``scripts/pod_lifecycle.py`` flow with zero behavior change. A task with
  no ``backend:`` frontmatter routes here.
* SLURM cluster backend — opt-in via ``backend: cluster`` / ``backend: nibi``
  frontmatter. The selector's submit-and-park + fall-back-on-NotImplemented
  control flow is wired in :mod:`.selector`; the actual SLURM implementation
  (`slurm.py`, `slurm_monitor.py`) lands in slice 2.

The :class:`~base.PollResult` dataclass shape is intentionally the same
JSON shape that ``scripts/poll_pipeline.py`` produces, so the orchestrator's
existing JSON-line parsing keeps working unchanged.

See ``.claude/plans/2026-06-08_001932-slurm-cluster-backend-for-issue.md``
for the full plan; this slice (slice 1 of P1) ships the foundation only.
"""

from explore_persona_space.backends.base import (
    BackendKind,
    ComputeBackend,
    PollResult,
    RunHandle,
    RunSpec,
)
from explore_persona_space.backends.runpod import RunPodBackend
from explore_persona_space.backends.selector import (
    SLURM_NOT_IMPLEMENTED_MESSAGE,
    BackendDecision,
    BackendSelectionError,
    select_backend,
)
from explore_persona_space.backends.slurm import (
    CLUSTER_CONFIGS,
    ClusterConfig,
    SbatchPlan,
    SlurmBackend,
    Stage,
    estimate_start_seconds,
    get_cluster_config,
    render_sbatch,
    ssh_estimate_start,
    stages_for_spec,
)

__all__ = [
    "CLUSTER_CONFIGS",
    "SLURM_NOT_IMPLEMENTED_MESSAGE",
    "BackendDecision",
    "BackendKind",
    "BackendSelectionError",
    "ClusterConfig",
    "ComputeBackend",
    "PollResult",
    "RunHandle",
    "RunPodBackend",
    "RunSpec",
    "SbatchPlan",
    "SlurmBackend",
    "Stage",
    "estimate_start_seconds",
    "get_cluster_config",
    "render_sbatch",
    "select_backend",
    "ssh_estimate_start",
    "stages_for_spec",
]
