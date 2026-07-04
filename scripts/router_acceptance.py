#!/usr/bin/env python3
"""Slice-8 live-acceptance HARNESS for the multi-backend compute router.

The slice-5/6/7 router + per-lane backends + dispatch CLI are fully unit
tested. This script is the BRIDGE that drives a REAL per-lane smoke run
end-to-end — submit -> poll -> fetch -> confirm_artifacts -> teardown —
against the same operational commands SKILL.md Step 6b/6d/8 invoke. Its
job is to PROVE every lane (Nibi/GCP/Mila/auto) handles the canonical
path without a hidden divergence between "unit-test contract" and "what
actually happens when the orchestrator drives a real workload."

Two sub-commands:

* ``live`` -- drive a real per-lane acceptance run. SAFE-BY-DEFAULT:
  the actual shell-outs to ``dispatch_issue.py`` / ``backend_poll.py``
  fire ONLY when ``--live`` is passed. Without ``--live`` the harness
  prints the exact command sequence it WOULD run, plus the PASS
  checklist it would evaluate -- so a reviewer can dry-run the lane
  before spending real cluster / credit time.

* ``negative <case>`` -- run one of four injected-mock negative scenarios:

  - ``free-busy-to-gcp``: every free lane returns a beyond-park
    est-start; assert the router escalates to GCP (RunPod untouched
    because GCP succeeds here).
  - ``gcp-full-to-runpod``: every GCP ladder rung capacity-misses and no
    free lane is wired; assert the auto chain falls through to the
    RunPod TERMINAL rung (``reason: auto_fallback_runpod``) — the #656
    reversal of the historical no-auto-RunPod invariant.
  - ``cancel-race``: the free lane's job races to RUNNING after the
    park-cap fires the cancel; assert the cancel state machine keeps
    the running job rather than double-killing it.
  - ``duplicate-cron-tick``: ``dispatch_issue.py finalize`` is invoked
    twice for the same handle (the orchestrator's bg-Bash poll loop
    and the 20-min backstop cron racing); assert the second tick is
    idempotent.

  Negative cases NEVER touch real infrastructure -- they construct
  fully-injected ``ComputeBackend`` mocks and drive the router /
  dispatch helpers directly. Same code paths used by the unit-test
  suite, packaged here so the harness is the single place a reviewer
  runs to validate every guarantee Slice 8 promises.

PASS checklist (per lane, evaluated by :func:`evaluate_pass_checklist`):

* ``(a) hf_artifact_present``: the smoke LoRA adapter / training mix
  shows up under the per-lane HF Hub subfolder
  (``superkaiba1/explore-persona-space/router_acceptance/issue-<N>-<lane>/``)
  via ``huggingface_hub.list_repo_files`` (NEVER the ``hf`` CLI --
  CLAUDE.md upload-policy.md rule). The harness sets
  ``EPM_PERSIST_ADAPTER_HF_REPO`` + ``EPM_PERSIST_ADAPTER_SUBFOLDER``
  on the launch env (the ONLY env vars ``trainer.py:_persist_adapter``
  reads, per ``.claude/rules/upload-policy.md`` -- NOT
  ``EPM_PERSIST_ADAPTER_HF_SUBFOLDER`` which does not exist); the
  backends forward them from the dispatch process env to the REMOTE
  workload env via ``slurm.PASSTHROUGH_ENV_KEYS`` (sourced
  ``secrets.env``) / ``gcp.STARTUP_PASSTHROUGH_ENV_KEYS`` (instance
  metadata) so the delete-after-eval adapter persistence lands at the
  expected path the check (a) reads. The subfolder lane string is the
  PRE-launch literal backend (``auto`` on auto runs); check (a) probes
  the same string (``artifact_lane``).
* ``(b) git_figure_present``: a per-lane figure lives at
  ``figures/issue_<N>/router_acceptance_<lane>.png`` and is staged in
  git (``git ls-files`` picks up staged + tracked paths). The harness
  itself generates this figure locally AFTER the lane completes (a
  one-bar matplotlib PNG recording elapsed-seconds + chosen_kind) and
  ``git add``s it -- ``train.py`` emits no figure of its own, so
  without harness-side generation check (b) FALSE-FAILS every live
  lane.
* ``(c) routing_marker_posted``: the ``$ACC`` task's events.jsonl
  carries a fresh ``epm:backend-selected v1`` whose ``chosen_kind``
  matches the requested lane (auto -> "this is the lane the router
  picked"; explicit -> matches the override).
* ``(d) clean_teardown``: the lane's own authority shows no live job /
  VM / pod. DRAC: ``squeue --name <pod_name>`` empty over the
  cluster's robot socket, where ``<pod_name>`` is the CANONICAL job
  name the launcher used (read from the launch outcome JSON's
  ``pod_name`` field, NOT reconstructed as ``eps-issue-<N>`` --
  ``slurm.job_name`` appends a ``-<plan_hash[:8]>`` suffix when
  ``plan_hash`` is set, so a reconstructed grep can false-PASS on a
  still-live job whose real name carries the hash suffix). GCP:
  ``gcloud compute instances list --filter="labels.eps-issue=<N>"``
  empty, against the SAME project/config the launcher used (also
  carried from the launch outcome -- a fresh ``GcpConfig()`` could
  grep a different project than the launch actually targeted).
  RunPod NOT in scope for slice-8 acceptance (explicit-only; covered
  by the existing ``test_no_auto_runpod_path_under_any_failure``
  regression guard).

Out-of-scope (this is harness only -- the live runs are
orchestrator-driven):

* This script does NOT itself launch live pods / VMs / jobs unless
  ``--live`` is passed. Default behaviour is dry-run + the negative
  cases, all unit-testable without spending.
* The smoke workload (a tiny ~20-step LoRA fine-tune of Qwen-2.5-7B
  on a 50-row deterministic SFT subsample, see
  ``configs/condition/c_router_smoke.yaml`` +
  ``data/sft/router_smoke_sft.jsonl``) is a ROUTER-PLUMBING smoke. Its
  Goal is to exercise the dispatch path -- NOT to implant a behavior.
  The CLAUDE.md "always use contrastive negatives" rule does NOT apply
  here (the rule scopes to *behavior-implantation experiments*); the
  smoke condition's docstring documents this explicitly so reviewers
  do not flag it.

References:

* Plan: ``.claude/plans/2026-06-08_224537-multi-backend-compute-router.md``
  step 8 (Acceptance ordered Nibi -> GCP -> Mila).
* ``scripts/dispatch_issue.py`` -- the launch + finalize CLI driven here.
* ``scripts/backend_poll.py`` -- the one-tick poll bridge driven here.
* ``src/explore_persona_space/backends/router.py`` -- the routing
  decision engine + terminal exception classes the negative-case tests
  exercise.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("router_acceptance")


# ---------------------------------------------------------------------------
# Constants -- canonical paths the live runs touch
# ---------------------------------------------------------------------------

#: Default HF model repo for acceptance-smoke artifacts. A DEDICATED
#: PRIVATE repo, not the canonical ``superkaiba1/explore-persona-space``:
#: (a) the canonical public repo sits at ~10.2 TB and HF's account-level
#: public-storage enforcement now 403s every LFS-scale upload to it
#: (live finding, issue 535 GCP lane r8: the workload trained end-to-end
#: and died only at the 339 MB adapter persist; reproduced deterministically
#: off-GPU at the same size, while the same folder uploaded to a private
#: repo fine — private storage draws from a separate pool); (b) smoke
#: artifacts are throwaway plumbing proofs and never belonged in the
#: research repo. Overridable per-run via ``--hf-model-repo`` (the
#: nibi / mila lanes' earlier PASSes wrote to the canonical repo before
#: enforcement hit).
DEFAULT_ACCEPTANCE_HF_REPO = "superkaiba1/explore-persona-space-router-acceptance"

#: Per-lane HF model-repo subfolder pattern. The smoke trains a LoRA
#: adapter that the existing training pipeline auto-uploads to the
#: acceptance repo above; the harness writes to a dedicated subfolder
#: so a failed lane never clobbers a passing lane.
ACCEPTANCE_HF_SUBFOLDER = "router_acceptance/issue-{issue}-{lane}"

#: Per-lane figure path. Falls under the per-issue figures dir the
#: orchestrator already commits via Step 8 (Upload Policy).
ACCEPTANCE_FIGURE_PATH = "figures/issue_{issue}/router_acceptance_{lane}.png"

#: Per-lane events.jsonl marker key. The router writes this on every
#: chosen-lane decision (see ``epm:backend-selected v1`` in workflow.yaml).
ROUTING_MARKER = "epm:backend-selected"

#: Ground-truth launch marker: posted by the SLURM monitor at submit time
#: with the cluster the job ACTUALLY went to (resolved ClusterConfig),
#: independent of the router's lane belief. Check (c) compares the two
#: for per-cluster lanes (live finding, issue 535: the 'mila' lane's job
#: ran on Nibi — chosen_kind said mila, cluster-launched said nibi — and
#: the checklist PASSed vacuously).
CLUSTER_LAUNCHED_MARKER = "epm:cluster-launched"

#: Lanes whose kind IS a SLURM cluster name (mirror of
#: ``backends.router._PER_CLUSTER_LANES``).
PER_CLUSTER_LANES = frozenset({"nibi", "fir", "mila"})

#: Default Hydra overrides for the smoke workload. ~20 LoRA steps on
#: 50-row data; report_to=wandb so training metrics land in WandB per
#: the always-on WandB-required rule.
DEFAULT_SMOKE_HYDRA_ARGS: tuple[str, ...] = (
    "condition=c_router_smoke",
    "seed=0",
    # `+` prefix required: max_steps is deliberately NOT in the training
    # schema (configs/training/turner_em.yaml documents the same) -- the
    # bare form crashed live attempt 3 on Nibi with "Key 'max_steps' is
    # not in struct" (job 15862188), costing the lane its HF artifact.
    "+training.max_steps=20",
    "training.per_device_train_batch_size=1",
    "training.gradient_accumulation_steps=1",
    "training.save_strategy=no",
    "training.logging_steps=5",
)


# ---------------------------------------------------------------------------
# Dataset resolution -- "use the smallest existing HF dataset if any fits;
# else generate the deterministic 50-row local file"
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmokeDatasetSpec:
    """Resolved location of the smoke training mix.

    Fields:

    * ``local_path`` -- repo-relative path the smoke condition YAML
      references. The pod sees this the moment it clones the repo.
    * ``source`` -- ``"reused"`` when the local file came from
      sub-sampling an existing HF data-repo file; ``"generated"`` when
      it was deterministically synthesized from scratch.
    * ``row_count`` -- exact row count for the audit trail.
    * ``provenance`` -- one-line human description: e.g.
      ``"sub-sampled 50 rows (seed=0) from benign_sft_6k.jsonl"``.
    """

    local_path: Path
    source: str  # "reused" | "generated"
    row_count: int
    provenance: str


def resolve_smoke_dataset(
    *,
    repo_root: Path,
    local_rel: str = "data/sft/router_smoke_sft.jsonl",
) -> SmokeDatasetSpec:
    """Resolve the smoke training mix, preferring REUSE over generation.

    Reuse-first per CLAUDE.md "Reuse existing experiment code as much
    as possible". The reuse target is the 50-row deterministic
    sub-sample of ``benign_sft_6k.jsonl`` (already on
    ``superkaiba1/explore-persona-space-data``). The dispatch helper
    only requires the local file exist -- the harness commits it to git
    so it travels with the repo clone the lane provisions.

    If the local file is missing AND the HF data repo has the source
    file, the caller can regenerate via ``--regenerate-dataset`` (the
    canonical re-creation command lives in the smoke condition YAML's
    docstring; we do NOT silently regenerate here -- a missing local
    file should be loud).
    """
    p = repo_root / local_rel
    if not p.exists():
        raise FileNotFoundError(
            f"smoke training mix not present at {p}. "
            "Re-create via the deterministic sub-sample command documented "
            "in configs/condition/c_router_smoke.yaml (seed=0 over "
            "superkaiba1/explore-persona-space-data:benign_sft_6k.jsonl) "
            "and commit the result."
        )
    # Row count is part of the audit trail -- a silent row-count change
    # would invalidate the "deterministic sub-sample" claim.
    # split("\n"), NOT splitlines(): raw U+2028/U+2029/NEL inside
    # ensure_ascii=False JSON strings are Unicode line boundaries that would
    # inflate this row count on a perfectly valid file (gotchas.md; #825 → #950).
    rows = sum(1 for line in p.read_text().split("\n") if line.strip())
    return SmokeDatasetSpec(
        local_path=p,
        source="reused",
        row_count=rows,
        provenance=(
            f"sub-sampled {rows} short (<800 char assistant content) rows "
            "(rng seed=0, stable iteration order) from "
            "superkaiba1/explore-persona-space-data:benign_sft_6k.jsonl"
        ),
    )


# ---------------------------------------------------------------------------
# Per-lane figure -- the harness MUST produce check (b)'s artifact itself.
# ``train.py`` emits no figure for the smoke workload; without harness-side
# generation check (b) FALSE-FAILS every live lane (the figure simply does
# not exist on disk to begin with).
# ---------------------------------------------------------------------------


def generate_acceptance_figure(
    *,
    issue: int,
    lane: str,
    elapsed_seconds: float,
    chosen_kind: str,
    repo_root: Path,
    git_add: Callable[[Path, Path], None] | None = None,
) -> Path:
    """Generate the per-lane acceptance figure and stage it in git.

    Writes a trivial one-bar matplotlib PNG recording the lane's
    elapsed-seconds + chosen_kind to
    ``figures/issue_<N>/router_acceptance_<lane>.png`` and ``git
    add``s it so ``git ls-files`` (the check-(b) probe) sees it. The
    figure is acceptance EVIDENCE for the live run -- the smoke
    workload itself emits no figure.

    Fails loud (raises) on any matplotlib / FS / git failure -- check
    (b) MUST NOT silently FAIL through a swallowed exception in the
    figure generator. The caller's lane-level FAIL handling surfaces
    the raise; do NOT wrap this in ``try / pass``.

    Returns the absolute Path the figure was written to.
    """
    rel = ACCEPTANCE_FIGURE_PATH.format(issue=issue, lane=lane)
    abs_path = repo_root / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    # Matplotlib is a heavy import; defer until we actually need it.
    import matplotlib

    matplotlib.use("Agg")  # headless, no DISPLAY
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([f"{lane} ({chosen_kind})"], [elapsed_seconds], color="steelblue")
    ax.set_ylabel("elapsed (s)")
    ax.set_title(f"router-acceptance #{issue} {lane}")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(abs_path, dpi=80)
    plt.close(fig)

    if not abs_path.exists():
        raise RuntimeError(f"figure generation produced no file at {abs_path}")

    stage = git_add or _default_git_add
    stage(repo_root, abs_path)
    return abs_path


def _default_git_add(repo_root: Path, abs_path: Path) -> None:
    """Stage ``abs_path`` in git so ``git ls-files`` returns it.

    Staging is enough for the check-(b) probe (``git ls-files``
    reports both tracked and staged paths). Committing is the
    caller's choice -- a single "acceptance evidence" commit per lane
    is fine but not required by the verifier.
    """
    rel = abs_path.relative_to(repo_root)
    argv = ["git", "-C", str(repo_root), "add", "--", str(rel)]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(
            f"git add {rel} failed (rc={proc.returncode}): stderr={proc.stderr.strip()!r}"
        )


# ---------------------------------------------------------------------------
# PASS checklist
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """One PASS-checklist entry."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class LaneVerdict:
    """All four checks for one lane, plus the overall PASS bit."""

    lane: str
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def format(self) -> str:
        verdict_line = f"LANE {self.lane}: {'PASS' if self.passed else 'FAIL'}"
        lines = [verdict_line]
        for c in self.checks:
            tag = "PASS" if c.passed else "FAIL"
            detail = f" -- {c.detail}" if c.detail else ""
            lines.append(f"  ({c.name}) {tag}{detail}")
        return "\n".join(lines)


# I/O seam -- every external call (HF Hub, git, events.jsonl read,
# squeue / gcloud teardown verification) is injectable so the unit-test
# suite can drive the verifier with zero infra.


@dataclass(frozen=True)
class VerifierIO:
    """Injectable I/O for the PASS checklist + teardown verification.

    Every external call is a callable so unit tests pass fakes and a
    ``--live`` real-run uses the production implementations. Defaults
    are wired lazily so tests that ``monkeypatch.setattr`` a module
    attribute see the patch.

    The ``gcloud_instances_list`` callable takes a positional name
    filter plus OPTIONAL kw-only ``gcp_project`` / ``gcp_config_name``
    overrides -- ``check_clean_teardown`` threads the launcher's
    project so the verifier never greps a different project than the
    launcher used (carried from the launch outcome JSON).
    """

    list_hf_repo_files: Callable[..., list[str]] | None = None
    git_tracked: Callable[[Path, Iterable[str]], set[str]] | None = None
    read_events_jsonl: Callable[[int], list[dict[str, Any]]] | None = None
    squeue_by_name: Callable[[str, str], list[str]] | None = None
    gcloud_instances_list: Callable[..., list[dict[str, Any]]] | None = None

    def _list_hf(self) -> Callable[..., list[str]]:
        return self.list_hf_repo_files or _default_list_hf_repo_files

    def _git(self) -> Callable[[Path, Iterable[str]], set[str]]:
        return self.git_tracked or _default_git_tracked

    def _events(self) -> Callable[[int], list[dict[str, Any]]]:
        return self.read_events_jsonl or _default_read_events_jsonl

    def _squeue(self) -> Callable[[str, str], list[str]]:
        return self.squeue_by_name or _default_squeue_by_name

    def _gcloud(self) -> Callable[..., list[dict[str, Any]]]:
        return self.gcloud_instances_list or _default_gcloud_instances_list


def _default_list_hf_repo_files(repo_id: str, *, repo_type: str) -> list[str]:
    """Production HF Hub lister (NEVER the ``hf`` CLI -- has no ``api`` subcommand)."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    return list(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))


def _default_git_tracked(repo_root: Path, rel_paths: Iterable[str]) -> set[str]:
    """Production ``git ls-files`` checker."""
    rel_list = list(rel_paths)
    if not rel_list:
        return set()
    argv = ["git", "-C", str(repo_root), "ls-files", "--", *rel_list]
    proc = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=30)
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _default_read_events_jsonl(issue: int) -> list[dict[str, Any]]:
    """Production events.jsonl reader.

    Resolves the task's current folder via ``scripts/task.py find <N>``
    so a status change (e.g. ``running`` -> ``verifying``) does NOT
    leave the harness reading a stale path.
    """
    proc = subprocess.run(
        ["uv", "run", "python", "scripts/task.py", "find", str(int(issue))],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    folder = Path(proc.stdout.strip())
    events_path = folder / "events.jsonl"
    if not events_path.exists():
        return []
    out: list[dict[str, Any]] = []
    # split("\n"), NOT splitlines(): raw U+2028/U+2029/NEL inside
    # ensure_ascii=False notes are Unicode line boundaries that would shred a
    # VALID record and false-crash the fail-loud RuntimeError below
    # (gotchas.md; #825 → #950).
    for line in events_path.read_text().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"events.jsonl malformed at {events_path}: {exc}") from exc
    return out


def _default_squeue_by_name(robot_alias: str, job_name: str) -> list[str]:
    """Production squeue-by-name probe (returns job ids).

    A non-empty return = still-live (PENDING / RUNNING / COMPLETING all
    count as live -- the DRAC robot allowlist has no ``sacct`` so this
    is the authoritative "still in queue" signal).

    Timeout bumped to 120s (from 60s) because the DRAC scheduler is
    sometimes slow to respond under load -- a 60s ssh-side timeout
    can spuriously raise on a healthy still-empty queue, FALSE-FAILing
    check (d).
    """
    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        robot_alias,
        f"squeue -h -o %A --name={job_name}",
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"squeue probe failed (rc={proc.returncode}): stderr={proc.stderr.strip()!r}"
        )
    return [j.strip() for j in proc.stdout.splitlines() if j.strip()]


def _default_gcloud_instances_list(
    name_filter: str,
    *,
    gcp_project: str | None = None,
    gcp_config_name: str | None = None,
) -> list[dict[str, Any]]:
    """Production gcloud-list probe (returns matching instance dicts).

    Uses the canonical ``default_gcp_config()`` as the base, then
    overrides ``project`` / ``gcloud_config`` from the launch outcome
    when threaded by the caller. This is load-bearing for not grepping
    a DIFFERENT project than the launcher targeted -- a fresh
    ``GcpConfig()`` (all-empty defaults) would issue a gcloud call
    with no ``--project`` / ``--configuration``, falling back to the
    ambient ``CLOUDSDK_ACTIVE_CONFIG_NAME`` (which my-goat manipulates
    for personal use) and silently grepping the WRONG project. The
    GCP backend's invariant is explicit-project-per-call (see
    ``GcpConfig`` docstring); the verifier MUST match it.

    A non-empty return = at least one live instance matching the
    filter under the same project the launcher used.
    """
    # Lazy import -- keeps the harness importable on a VM with no
    # gcloud CLI installed.
    from explore_persona_space.backends.gcp import default_gcp_config, render_list_argv

    base = default_gcp_config()
    cfg = base
    if gcp_project or gcp_config_name:
        from dataclasses import replace

        cfg = replace(
            base,
            project=gcp_project or base.project,
            gcloud_config=gcp_config_name or base.gcloud_config,
        )
    argv = render_list_argv(config=cfg, name_filter=name_filter)
    proc = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"gcloud instances list failed (rc={proc.returncode}): stderr={proc.stderr.strip()!r}"
        )
    return json.loads(proc.stdout) if proc.stdout.strip() else []


# ---------------------------------------------------------------------------
# Per-check implementations
# ---------------------------------------------------------------------------


def check_hf_artifact_present(
    *,
    issue: int,
    lane: str,
    repo_id: str,
    io: VerifierIO,
) -> CheckResult:
    """Check (a): the per-lane HF subfolder has >=1 file.

    The smoke workload's training pipeline auto-uploads the LoRA
    adapter to ``router_acceptance/issue-<N>-<lane>/`` because the
    harness sets BOTH ``EPM_PERSIST_ADAPTER_HF_REPO`` AND
    ``EPM_PERSIST_ADAPTER_SUBFOLDER`` (verbatim, per
    ``.claude/rules/upload-policy.md`` -- NOT
    ``EPM_PERSIST_ADAPTER_HF_SUBFOLDER`` which does not exist in
    ``trainer.py``) on the LAUNCH env. The launch env alone reaches
    only the local ``dispatch_issue.py`` process; the values reach the
    REMOTE workload env (where ``trainer.py:_persist_adapter`` reads
    ``os.environ``) via the backends' non-secret passthrough lists --
    ``slurm.PASSTHROUGH_ENV_KEYS`` (rendered into the sourced
    ``secrets.env``) and ``gcp.STARTUP_PASSTHROUGH_ENV_KEYS``
    (instance metadata exported by the startup script).

    ``lane`` here is the ARTIFACT lane -- the literal string the
    harness baked into ``EPM_PERSIST_ADAPTER_SUBFOLDER`` at launch
    time (``auto`` on an auto run, since the resolved lane is unknown
    pre-launch). See ``evaluate_pass_checklist(artifact_lane=...)``.
    """
    subfolder = ACCEPTANCE_HF_SUBFOLDER.format(issue=issue, lane=lane)
    try:
        files = io._list_hf()(repo_id, repo_type="model")
    except Exception as exc:
        return CheckResult(
            name="hf_artifact_present",
            passed=False,
            detail=f"HF list_repo_files({repo_id!r}) raised: {exc}",
        )
    matching = [f for f in files if f.startswith(subfolder + "/") or f == subfolder]
    if not matching:
        return CheckResult(
            name="hf_artifact_present",
            passed=False,
            detail=f"no files under HF model repo prefix {subfolder!r}",
        )
    return CheckResult(
        name="hf_artifact_present",
        passed=True,
        detail=f"{len(matching)} file(s) under {subfolder!r}",
    )


def check_git_figure_present(
    *,
    issue: int,
    lane: str,
    repo_root: Path,
    io: VerifierIO,
) -> CheckResult:
    """Check (b): a per-lane figure was committed under figures/issue_<N>/."""
    rel = ACCEPTANCE_FIGURE_PATH.format(issue=issue, lane=lane)
    abs_path = repo_root / rel
    if not abs_path.exists():
        return CheckResult(
            name="git_figure_present",
            passed=False,
            detail=f"figure file missing on disk: {rel}",
        )
    try:
        tracked = io._git()(repo_root, [rel])
    except subprocess.CalledProcessError as exc:
        return CheckResult(
            name="git_figure_present",
            passed=False,
            detail=f"git ls-files failed (rc={exc.returncode}): {exc.stderr!r}",
        )
    if rel not in tracked:
        return CheckResult(
            name="git_figure_present",
            passed=False,
            detail=f"figure on disk but NOT tracked by git: {rel}",
        )
    return CheckResult(name="git_figure_present", passed=True, detail=rel)


def check_routing_marker_posted(
    *,
    issue: int,
    expected_lane: str,
    io: VerifierIO,
) -> CheckResult:
    """Check (c): an ``epm:backend-selected v1`` event records ``chosen_kind``.

    The router writes the marker once per ``route()`` call (see
    ``backends.router._post_backend_selected``). The harness checks
    the MOST RECENT such marker (later launches in the same task
    leave the older marker behind; the latest one is the active
    routing decision). For auto runs ``expected_lane`` is the lane the
    router actually picked; for explicit overrides it must match.
    """
    try:
        events = io._events()(issue)
    except Exception as exc:
        return CheckResult(
            name="routing_marker_posted",
            passed=False,
            detail=f"events.jsonl read failed: {exc}",
        )
    # Scan backwards for the most recent backend-selected marker.
    # events.jsonl uses the ``kind`` field for the marker name (NOT
    # ``marker`` -- see task_workflow.py write paths). We tolerate
    # both for forward-compat (the dashboard reads ``kind``; an older
    # row that used ``marker`` would still be recognized).
    for event in reversed(events):
        marker = event.get("kind") or event.get("marker") or ""
        if marker == ROUTING_MARKER:
            note = event.get("note") or ""
            # Body parsing kept dead simple -- the marker body has a
            # ``chosen_kind: <lane>`` line per workflow.yaml.
            chosen = _parse_kv_from_marker_note(note, "chosen_kind")
            if chosen is None:
                return CheckResult(
                    name="routing_marker_posted",
                    passed=False,
                    detail="marker present but no chosen_kind field in body",
                )
            if expected_lane != "auto" and chosen != expected_lane:
                return CheckResult(
                    name="routing_marker_posted",
                    passed=False,
                    detail=(
                        f"marker chosen_kind={chosen!r} does NOT match "
                        f"requested lane {expected_lane!r}"
                    ),
                )
            # GROUND-TRUTH cross-check for per-cluster SLURM lanes: the
            # chosen_kind above is the router's BELIEF; the
            # epm:cluster-launched marker records the cluster the sbatch
            # ACTUALLY went to. A mismatch means a misroute (issue 535:
            # the 'mila' lane's job ran on Nibi and PASSed vacuously).
            effective_lane = chosen if expected_lane == "auto" else expected_lane
            if effective_lane in PER_CLUSTER_LANES:
                actual = _latest_launched_cluster(events)
                if actual is None:
                    return CheckResult(
                        name="routing_marker_posted",
                        passed=False,
                        detail=(
                            f"chosen_kind={chosen} but no {CLUSTER_LAUNCHED_MARKER!r} "
                            "marker to ground-truth the cluster against"
                        ),
                    )
                if actual != effective_lane:
                    return CheckResult(
                        name="routing_marker_posted",
                        passed=False,
                        detail=(
                            f"MISROUTE: chosen_kind={chosen} but the job actually "
                            f"launched on cluster={actual!r} (epm:cluster-launched)"
                        ),
                    )
                return CheckResult(
                    name="routing_marker_posted",
                    passed=True,
                    detail=f"chosen_kind={chosen}; ground-truth cluster={actual}",
                )
            return CheckResult(
                name="routing_marker_posted",
                passed=True,
                detail=f"chosen_kind={chosen}",
            )
    return CheckResult(
        name="routing_marker_posted",
        passed=False,
        detail=f"no {ROUTING_MARKER!r} marker on task {issue}",
    )


def _current_git_branch(repo_root: Path | None) -> str | None:
    """Current branch name of ``repo_root`` (None on detached HEAD / error)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    branch = proc.stdout.strip()
    return branch if branch and branch != "HEAD" else None


def _latest_launched_cluster(events: list[dict[str, Any]]) -> str | None:
    """Cluster name from the most recent ``epm:cluster-launched`` marker.

    The SLURM monitor posts the marker at submit time with a JSON note
    carrying ``"cluster": "<name>"`` — the cluster the job ACTUALLY went
    to (resolved ClusterConfig at sbatch), independent of the router's
    lane belief. Returns ``None`` when no such marker (or no cluster
    field) exists — e.g. a GCP launch, whose launched marker carries
    ``"backend": "gcp"`` instead.
    """
    for event in reversed(events):
        marker = event.get("kind") or event.get("marker") or ""
        if marker != CLUSTER_LAUNCHED_MARKER:
            continue
        note = event.get("note") or ""
        try:
            body = json.loads(note)
        except (TypeError, ValueError):
            cluster = _parse_kv_from_marker_note(note, "cluster")
            return cluster if cluster else None
        if isinstance(body, dict):
            cluster = body.get("cluster")
            return cluster if isinstance(cluster, str) and cluster else None
        return None
    return None


def _parse_kv_from_marker_note(note: str, key: str) -> str | None:
    """Pull a single field out of an ``epm:backend-selected`` marker note.

    The router posts the note as a JSON blob (see
    ``router._post_backend_selected``), so the primary path is a JSON
    decode. A defensive fallback parses ``key: value`` lines for
    forward compat with any marker variant that ships a plain-text
    body (e.g. ``_post_intermediate_marker`` future shape).
    Returns the field as a string (or ``None`` when absent / wrong
    shape) so the caller's equality check ("chosen_kind matches the
    requested lane") works uniformly.
    """
    try:
        decoded = json.loads(note)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, dict):
        value = decoded.get(key)
        if value is None:
            return None
        return str(value)
    for line in note.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    return None


def check_clean_teardown(
    *,
    issue: int,
    lane: str,
    io: VerifierIO,
    robot_alias_for_slurm: str | None = None,
    canonical_job_name: str | None = None,
    gcp_project: str | None = None,
    gcp_config_name: str | None = None,
) -> CheckResult:
    """Check (d): the lane's own authority shows no live job / VM / pod.

    The probe runs against the lane's authoritative state, NOT the
    router's local lease (a stale lease is a bug we want to surface,
    not silently dismiss):

    * SLURM (nibi/mila): ``squeue --name <canonical_job_name>`` empty
      over the cluster's robot socket. ``canonical_job_name`` is the
      job name the launcher actually used (read from the launch
      outcome JSON's ``pod_name`` field, which mirrors the
      ``RunHandle.pod_name`` ``slurm.job_name(spec, plan_hash)``
      returned). When ``plan_hash`` is set the suffix is
      ``-<plan_hash[:8]>``; reconstructing the name from issue alone
      would grep the wrong name and FALSE-PASS on a still-live job.
      Fallback to ``eps-issue-<N>`` is allowed ONLY when no canonical
      name is threaded (legacy callers / pure-unit tests); production
      verify-lane always threads it. Robot allowlist has no ``sacct``,
      so "absent from queue" is the most authoritative terminal signal.
    * GCP: ``gcloud compute instances list --filter="labels.eps-issue=<N>"``
      returns no instances with the ``eps-issue=<N>`` label, against
      the SAME ``gcp_project`` / ``gcp_config_name`` the launcher used.
    * RunPod: NOT in scope for slice-8 acceptance (explicit-only;
      the auto chain never reaches it; covered by
      ``test_no_auto_runpod_path_under_any_failure``).
    """
    job_name = canonical_job_name or f"eps-issue-{int(issue)}"
    if lane in {"nibi", "fir", "mila"}:
        if robot_alias_for_slurm is None:
            return CheckResult(
                name="clean_teardown",
                passed=False,
                detail=(
                    f"lane={lane!r} requires robot_alias_for_slurm to probe squeue; "
                    "harness misconfiguration"
                ),
            )
        try:
            live_ids = io._squeue()(robot_alias_for_slurm, job_name)
        except Exception as exc:
            return CheckResult(
                name="clean_teardown",
                passed=False,
                detail=f"squeue probe failed: {exc}",
            )
        if live_ids:
            return CheckResult(
                name="clean_teardown",
                passed=False,
                detail=(
                    f"squeue --name {job_name} still shows live ids: {live_ids!r}; "
                    "teardown did NOT remove the job"
                ),
            )
        return CheckResult(
            name="clean_teardown",
            passed=True,
            detail=f"squeue --name {job_name} empty over {robot_alias_for_slurm}",
        )

    if lane == "gcp":
        gcp_filter = f"labels.eps-issue={int(issue)}"
        try:
            instances = io._gcloud()(
                gcp_filter,
                gcp_project=gcp_project,
                gcp_config_name=gcp_config_name,
            )
        except Exception as exc:
            return CheckResult(
                name="clean_teardown",
                passed=False,
                detail=f"gcloud instances list failed: {exc}",
            )
        if instances:
            names = [i.get("name", "<unnamed>") for i in instances]
            return CheckResult(
                name="clean_teardown",
                passed=False,
                detail=(
                    f"GCE instances list still has matches for {gcp_filter!r}: "
                    f"{names!r}; teardown did NOT delete the VM"
                ),
            )
        return CheckResult(
            name="clean_teardown",
            passed=True,
            detail=f"gcloud list --filter={gcp_filter!r} empty",
        )

    return CheckResult(
        name="clean_teardown",
        passed=False,
        detail=(
            f"lane={lane!r} not supported for slice-8 teardown verification "
            "(runpod is out of scope; auto resolves to one of the named lanes "
            "before this check fires)"
        ),
    )


def evaluate_pass_checklist(
    *,
    issue: int,
    lane: str,
    expected_lane: str,
    repo_root: Path,
    hf_model_repo: str,
    io: VerifierIO,
    robot_alias_for_slurm: str | None = None,
    canonical_job_name: str | None = None,
    gcp_project: str | None = None,
    gcp_config_name: str | None = None,
    artifact_lane: str | None = None,
) -> LaneVerdict:
    """Run all four PASS checks for one lane and return the verdict.

    ``canonical_job_name`` / ``gcp_project`` / ``gcp_config_name`` are
    threaded from the launch outcome JSON so check (d) probes the SAME
    name + project the launcher used. Defaults preserve the legacy
    behavior for pure-unit tests that don't have a launch outcome to
    thread.

    ``artifact_lane`` feeds check (a) ONLY: the HF subfolder probe must
    use the SAME lane string the harness baked into
    ``EPM_PERSIST_ADAPTER_SUBFOLDER`` on the launch env. That env is
    built BEFORE launch, when an ``auto`` run's resolved lane is
    unknowable -- so the env (and therefore the probe) uses the literal
    requested backend (``auto``), while checks (b)-(d) keep probing the
    RESOLVED ``lane``. Defaults to ``lane`` for explicit-lane runs and
    legacy callers.
    """
    checks = (
        check_hf_artifact_present(
            issue=issue, lane=artifact_lane or lane, repo_id=hf_model_repo, io=io
        ),
        check_git_figure_present(issue=issue, lane=lane, repo_root=repo_root, io=io),
        check_routing_marker_posted(issue=issue, expected_lane=expected_lane, io=io),
        check_clean_teardown(
            issue=issue,
            lane=lane,
            io=io,
            robot_alias_for_slurm=robot_alias_for_slurm,
            canonical_job_name=canonical_job_name,
            gcp_project=gcp_project,
            gcp_config_name=gcp_config_name,
        ),
    )
    return LaneVerdict(lane=lane, checks=checks)


# ---------------------------------------------------------------------------
# Live driver -- dry-run by default; --live actually shells out
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveCommandPlan:
    """The exact command sequence the live run executes (or would execute)."""

    launch_argv: list[str]
    poll_argv: list[str]
    finalize_argv: list[str]
    hydra_args: tuple[str, ...]
    repo_relative_cwd: Path


def build_live_command_plan(
    *,
    issue: int,
    backend: str,
    intent: str = "lora-7b",
    smoke_hydra_args: tuple[str, ...] = DEFAULT_SMOKE_HYDRA_ARGS,
    repo_root: Path | None = None,
    time_budget_hours: float | None = None,
    repo_branch: str | None = None,
) -> LiveCommandPlan:
    """Build the exact ``dispatch_issue.py`` + ``backend_poll.py`` argv.

    This mirrors SKILL.md Step 6b/6d/8 so the harness is a fair
    representation of what the orchestrator runs. Tests assert the
    argv shape against the SKILL.md operational block.

    ``backend == "auto"`` translates to ``--backend`` omitted (the CLI
    treats absent as auto, matching the empty-frontmatter case the
    skill prose documents).
    """
    cwd = repo_root or Path.cwd()
    hydra_args = tuple(smoke_hydra_args)

    launch_argv: list[str] = [
        "uv",
        "run",
        "python",
        "scripts/dispatch_issue.py",
        "launch",
        "--issue",
        str(int(issue)),
        "--intent",
        intent,
    ]
    if backend != "auto":
        launch_argv += ["--backend", backend]
    if time_budget_hours is not None:
        launch_argv += ["--time-budget-hours", str(time_budget_hours)]
    if repo_branch is not None and repo_branch != "main":
        # GCP-only knob (the GCE startup clones from origin; SLURM lanes
        # rsync the local worktree). Omitted on main so the argv shape
        # the SKILL.md golden tests pin stays unchanged there.
        launch_argv += ["--repo-branch", repo_branch]
    for hy in hydra_args:
        launch_argv += ["--hydra", hy]

    poll_argv = [
        "uv",
        "run",
        "python",
        "scripts/backend_poll.py",
        "--issue",
        str(int(issue)),
    ]

    # CRITICAL: ALWAYS pass --skip-confirm-artifacts to finalize.
    # The acceptance harness verifies artifacts INDEPENDENTLY (the
    # check-(a) HF probe + check-(b) figure probe in evaluate_pass_
    # checklist), so the dispatch CLI's confirm_artifacts gate is
    # both redundant AND unsafe here: the smoke workload's handle
    # carries no ``expected_artifacts`` sentinel, so confirm_artifacts
    # returns FAIL on a no-sentinel handle, which causes
    # ``dispatch_issue.py finalize`` to return rc=3 and SKIP
    # teardown -- the live VM / SLURM job would then stay UP and
    # bill (GCP) / occupy the queue (SLURM) while the harness
    # silently exits 0. ``--skip-confirm-artifacts`` makes teardown
    # the unconditional next step after sidecar-read, which is the
    # invariant a ``--live`` lane MUST hold: ALWAYS tear down its
    # VM / job, even when PASS checks fail elsewhere.
    finalize_argv = [
        "uv",
        "run",
        "python",
        "scripts/dispatch_issue.py",
        "finalize",
        "--issue",
        str(int(issue)),
        "--skip-confirm-artifacts",
    ]

    return LiveCommandPlan(
        launch_argv=launch_argv,
        poll_argv=poll_argv,
        finalize_argv=finalize_argv,
        hydra_args=hydra_args,
        repo_relative_cwd=cwd,
    )


def emit_live_dry_run(
    plan: LiveCommandPlan,
    *,
    backend: str,
    issue: int,
    out: Any | None = None,
) -> None:
    """Print the exact command sequence + PASS checklist a reviewer would run.

    The dry-run form is what the orchestrator actually invokes when
    the harness is called without ``--live`` -- it preserves the
    operator's ability to read the plan BEFORE spending real compute.

    ``out`` defaults to the *current* ``sys.stdout`` (resolved at call
    time, NOT module-import time) so a ``contextlib.redirect_stdout``
    around the call captures the dry-run output. Binding the default
    at import time would freeze the original stdout handle and silently
    bypass the redirect (root-caused by a unit test that tried exactly
    that).
    """
    if out is None:
        out = sys.stdout

    def emit(line: str) -> None:
        out.write(line + "\n")

    emit(f"# Router slice-8 live acceptance -- DRY RUN for lane={backend!r} issue={issue}")
    emit(f"# cwd: {plan.repo_relative_cwd}")
    emit("")
    emit("# Step 1: launch via dispatch_issue.py (writes per-issue sidecar JSON)")
    emit(" \\\n  ".join(plan.launch_argv))
    emit("")
    emit("# Step 2: poll via backend_poll.py until terminal (status: done / dead / gate)")
    emit("# (Orchestrator's bg-Bash loop drives this in production; the harness")
    emit("#  drives it sequentially -- both call the SAME script, so the contract")
    emit("#  preserves notification-on-exit.)")
    emit(" \\\n  ".join(plan.poll_argv))
    emit("")
    emit("# Step 3: finalize via dispatch_issue.py (confirm_artifacts + teardown)")
    emit(" \\\n  ".join(plan.finalize_argv))
    emit("")
    emit("# Step 4: PASS checklist -- evaluate all four checks (a)-(d) for this lane")
    emit("uv run python scripts/router_acceptance.py verify-lane \\")
    emit(f"    --issue {issue} --lane {backend}")


def _live_infra_warning(issue: int) -> str:
    """The loud may-be-live suffix for harness raises after a launch attempt.

    Stays in ONE place so every raise path that cannot prove "nothing
    launched" carries the same manual verification commands.
    """
    return (
        "A VM / SLURM job MAY BE LIVE despite this error (the dispatch CLI can "
        "crash AFTER provisioning). Best-effort cleanup finalize was attempted. "
        "VERIFY manually before walking away: "
        f"`gcloud compute instances list --filter=labels.eps-issue={int(issue)}` "
        f"and `squeue --name eps-issue-{int(issue)}` (job name may carry a "
        "-<plan_hash[:8]> suffix; check the launch stderr for the canonical name)."
    )


def _attempt_cleanup_finalize(
    plan: LiveCommandPlan,
    *,
    subprocess_run: Callable[..., subprocess.CompletedProcess],
    context: str,
) -> None:
    """Best-effort ``dispatch_issue.py finalize`` on a harness failure path.

    NEVER raises -- the caller is about to surface the ORIGINAL error
    and nothing here may shadow it. ``plan.finalize_argv`` already
    carries ``--skip-confirm-artifacts`` (the always-teardown
    contract). When no sidecar was ever written the finalize no-ops
    with rc=2 ``missing_handle_sidecar`` -- that VERIFIED-BENIGN shape
    (every pre-provision launch crash hits it) logs a WARNING, not the
    billing alarm. Every OTHER non-zero shape means the sidecar DID
    land but teardown could not run, so those stay logged LOUD ("may
    STILL be billing") and swallowed.
    """
    try:
        cleanup_proc = subprocess_run(
            plan.finalize_argv,
            capture_output=True,
            text=True,
            cwd=plan.repo_relative_cwd,
            check=False,
        )
        if cleanup_proc.returncode != 0:
            cleanup_body = _parse_last_json_line(cleanup_proc.stdout) or {}
            if cleanup_body.get("reason") == "missing_handle_sidecar":
                # The benign no-op: no handle sidecar ever landed, so
                # there is NOTHING on disk to tear down. A billing
                # ERROR here would false-alarm on every pre-provision
                # launch crash (Mn4.2).
                logger.warning(
                    "cleanup finalize after %s found no handle sidecar -- nothing "
                    "on disk to tear down. Verify manually ONLY if the launch "
                    "stderr shows provisioning started.",
                    context,
                )
            else:
                logger.error(
                    "cleanup finalize ITSELF returned rc=%d after %s "
                    "-- live VM/job may STILL be billing; stderr=%r stdout=%r",
                    cleanup_proc.returncode,
                    context,
                    cleanup_proc.stderr.strip(),
                    cleanup_proc.stdout.strip(),
                )
        else:
            logger.warning(
                "cleanup finalize ran (rc=0) after %s; surfacing the original error.",
                context,
            )
    except BaseException as cleanup_exc:
        logger.error(
            "cleanup finalize ITSELF raised %r after %s -- live VM/job may STILL be billing.",
            cleanup_exc,
            context,
        )


def run_live_lane(
    plan: LiveCommandPlan,
    *,
    backend: str,
    issue: int,
    poll_interval_seconds: float = 30.0,
    poll_timeout_seconds: float = 4 * 3600.0,
    subprocess_run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.monotonic,
    launch_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Drive the full live launch -> poll -> finalize loop.

    This is the ``--live`` path; ``subprocess_run`` / ``sleep_fn`` /
    ``now_fn`` are dependency-injected so the same orchestration loop
    is exercised by unit tests (passing recorded fakes). The poll loop
    re-invokes ``backend_poll.py`` once per tick and parses the JSON
    line on stdout; it terminates on ``status in {done, dead, gate}``
    or when ``poll_timeout_seconds`` elapses.

    ``launch_env`` (when provided) is the environment dict passed to
    the launch subprocess ONLY -- typically an ``os.environ.copy()``
    augmented with the per-lane adapter-persist vars
    (``EPM_PERSIST_ADAPTER_HF_REPO`` / ``EPM_PERSIST_ADAPTER_SUBFOLDER``)
    -- so the harness does NOT have to mutate the parent process's
    global ``os.environ`` to thread them through. The launch env alone
    reaches only the LOCAL ``dispatch_issue.py`` process; the backends
    forward the two vars onward to the REMOTE workload env via their
    non-secret passthrough lists (``slurm.PASSTHROUGH_ENV_KEYS`` into
    the sourced ``secrets.env``; ``gcp.STARTUP_PASSTHROUGH_ENV_KEYS``
    into instance metadata the startup script exports). The poll +
    finalize subprocesses inherit the parent env (they do not need
    those vars).

    Returns a structured result dict so the caller can log / assert.
    Raises ``RouterAcceptanceError`` on a subprocess that returned a
    non-zero exit with no JSON line we could parse (preserves the
    fail-fast contract).
    """
    # 1) Launch.
    launch_kwargs: dict[str, Any] = dict(
        capture_output=True,
        text=True,
        cwd=plan.repo_relative_cwd,
        check=False,
    )
    if launch_env is not None:
        launch_kwargs["env"] = launch_env
    launch_proc = subprocess_run(plan.launch_argv, **launch_kwargs)
    if launch_proc.returncode not in (0, 2):
        # 2 is a router terminal (NoCompute / WorkloadSurfaced / ...);
        # the JSON line still carries the failure shape, so we let the
        # caller surface it as a FAIL not a crash. Other non-zero codes
        # mean the CLI itself crashed -- and the dispatch CLI CAN crash
        # AFTER provisioning (rc=4 from a post-launch raise, rc=137/130
        # from an OOM-kill / SIGINT between ``gcloud create`` rc=0 and
        # the JSON print), so a VM / SLURM job MAY BE LIVE. Attempt the
        # same best-effort cleanup finalize the mid-flight except branch
        # runs (harmless no-op rc=2 ``missing_handle_sidecar`` when
        # nothing was written; tears down whenever the sidecar DID
        # land), then fail loud with the manual verification commands.
        _attempt_cleanup_finalize(
            plan,
            subprocess_run=subprocess_run,
            context=f"launch crash rc={launch_proc.returncode}",
        )
        raise RouterAcceptanceError(
            f"dispatch_issue.py launch exited with rc={launch_proc.returncode}: "
            f"stderr={launch_proc.stderr.strip()!r}. " + _live_infra_warning(issue)
        )
    launch_body = _parse_last_json_line(launch_proc.stdout)
    if launch_body is None:
        # Same exposure as the rc-crash branch above: the CLI can die
        # AFTER provisioning but BEFORE (or mid-) printing the JSON
        # line, so unparseable stdout does NOT mean nothing launched.
        _attempt_cleanup_finalize(
            plan,
            subprocess_run=subprocess_run,
            context="launch produced no parseable JSON",
        )
        raise RouterAcceptanceError(
            "dispatch_issue.py launch produced no parseable JSON on stdout; "
            f"stdout={launch_proc.stdout!r} stderr={launch_proc.stderr!r}. "
            + _live_infra_warning(issue)
        )
    if launch_proc.returncode == 2 or not launch_body.get("ok", False):
        # Router terminal -- bail. The harness records the failure
        # shape so the caller can surface it as the lane verdict.
        # This early-return is OUTSIDE the try/finally below on
        # purpose: for MOST terminals (NoCompute / WorkloadSurfaced /
        # GcpAttemptCapExceeded) no sidecar was written, so a cleanup
        # finalize would just no-op on ``missing_handle_sidecar``.
        # rc=2 does NOT universally imply "nothing is live", though:
        # ``ManualAttentionRequiredError`` means a launched SLURM job
        # SURVIVED scancel and is ORPHANED -- finalize genuinely cannot
        # help (no sidecar, free lane), so we surface the orphaned job
        # id + the scancel instruction LOUDLY instead of pretending the
        # lane is clean.
        if launch_body.get("exception") == "ManualAttentionRequiredError":
            orphaned_job_id = _parse_kv_from_marker_note(
                str(launch_body.get("note") or ""), "orphaned_job_id"
            )
            logger.error(
                "launch terminal ManualAttentionRequiredError: a SLURM job SURVIVED "
                "scancel and is ORPHANED (job_id=%s). No sidecar exists, so finalize "
                "cannot tear it down. Operator action: verify with "
                "`squeue -j %s` on the cluster and run `scancel %s` if it is alive.",
                orphaned_job_id,
                orphaned_job_id,
                orphaned_job_id,
            )
        return {
            "phase": "launch_terminal",
            "launch_body": launch_body,
            "poll_history": [],
            "finalize_body": None,
        }

    if launch_body.get("sidecar_write_error"):
        # M4.1: the launch SUCCEEDED (live VM / job) but the handle
        # sidecar write failed. When the early on_launched copy ALSO
        # failed (same unwritable dir), poll tick 1 reads
        # ``status=dead reason=missing_handle_sidecar`` and finalize
        # no-ops rc=2 -- a clean-looking terminal that swallows the
        # ONLY recovery record into captured stdout. Scream NOW, with
        # the full launch body (it carries the serialized handle) and
        # the manual verification commands, so the operator can
        # hand-write a ``--handle-file`` sidecar and run finalize.
        logger.error(
            "launch OK but sidecar write FAILED (%s); handle JSON (KEEP THIS): %s. %s",
            launch_body["sidecar_write_error"],
            json.dumps(launch_body, sort_keys=True),
            _live_infra_warning(issue),
        )

    # 2) + 3) Poll + finalize wrapped in try / except BaseException so
    # ANY mid-flight raise between launch and the normal finalize runs
    # cleanup-teardown before propagating. After launch returned ok the
    # backend has a live VM / SLURM job UP; if the poll loop times out,
    # backend_poll.py crashes, the JSON is malformed, or the harness
    # itself is interrupted (KeyboardInterrupt / timeout / SystemExit),
    # the live job WILL keep billing unless teardown runs. The cleanup
    # uses the SAME ``plan.finalize_argv`` (which already carries
    # ``--skip-confirm-artifacts`` -- the always-teardown contract),
    # logs LOUD if cleanup itself fails, and ALWAYS re-raises the
    # original exception so the harness fails loud rather than masking
    # the underlying failure. The happy path inside the try finalizes
    # ONCE and returns immediately -- the except clause only fires on
    # a mid-flight raise, so there is no double-teardown.
    try:
        poll_history = _run_poll_loop(
            plan,
            poll_interval_seconds=poll_interval_seconds,
            poll_timeout_seconds=poll_timeout_seconds,
            subprocess_run=subprocess_run,
            sleep_fn=sleep_fn,
            now_fn=now_fn,
        )
        finalize_body = _run_finalize_and_check(
            plan,
            issue=issue,
            subprocess_run=subprocess_run,
        )
        return {
            "phase": "complete",
            "launch_body": launch_body,
            "poll_history": poll_history,
            "finalize_body": finalize_body,
        }
    except BaseException as exc:
        # Mid-flight raise (poll timeout, poll crash, malformed poll
        # JSON, finalize rc!=0 / wrong phase, KeyboardInterrupt,
        # SystemExit, anything else). Live VM / job is UP -- best-
        # effort cleanup teardown before re-raising. We use
        # ``BaseException`` deliberately so timeouts /
        # ``KeyboardInterrupt`` also trigger teardown; missing them
        # was the leak. ``_attempt_cleanup_finalize`` never raises, so
        # a cleanup failure cannot shadow the original exception -- it
        # logs LOUD that the VM/job may still be billing, then we
        # re-raise ``exc``.
        _attempt_cleanup_finalize(
            plan,
            subprocess_run=subprocess_run,
            context=f"mid-flight raise {exc!r}",
        )
        raise


#: Consecutive poll-tick failures tolerated before the poll is declared
#: dead (and the harness tears the lane down). ONE transient blip --
#: ``backend.poll(handle)`` raising through ``backend_poll.py`` (rc=1)
#: on an SSH hiccup, or a garbled stdout line -- must NOT destroy an
#: otherwise-healthy multi-hour live run. The counter resets on every
#: healthy tick; the hard ``poll_timeout_seconds`` stays authoritative
#: (checked before every tick, retries included).
_POLL_MAX_CONSECUTIVE_FAILURES = 3


def _run_poll_loop(
    plan: LiveCommandPlan,
    *,
    poll_interval_seconds: float,
    poll_timeout_seconds: float,
    subprocess_run: Callable[..., subprocess.CompletedProcess],
    sleep_fn: Callable[[float], None],
    now_fn: Callable[[], float],
) -> list[dict[str, Any]]:
    """Poll ``backend_poll.py`` until a terminal status or a hard timeout.

    Returns the full poll history. Raises ``RouterAcceptanceError`` on
    a timeout, or after :data:`_POLL_MAX_CONSECUTIVE_FAILURES`
    CONSECUTIVE failed ticks (non-zero poll-subprocess rc OR
    unparseable JSON) -- a single transient blip retries with linear
    backoff instead of tearing down a healthy lane.
    """
    poll_history: list[dict[str, Any]] = []
    started = now_fn()
    consecutive_failures = 0
    terminal_statuses = {"done", "dead", "gate"}
    while True:
        if now_fn() - started > poll_timeout_seconds:
            raise RouterAcceptanceError(
                f"poll loop exceeded timeout {poll_timeout_seconds}s without "
                f"terminal status. "
                f"last_poll={poll_history[-1] if poll_history else None}"
            )
        poll_proc = subprocess_run(
            plan.poll_argv,
            capture_output=True,
            text=True,
            cwd=plan.repo_relative_cwd,
            check=False,
        )
        failure_detail: str | None = None
        poll_body: dict[str, Any] | None = None
        if poll_proc.returncode != 0:
            failure_detail = (
                f"backend_poll.py exited with rc={poll_proc.returncode}: "
                f"stderr={poll_proc.stderr.strip()!r}"
            )
        else:
            poll_body = _parse_last_json_line(poll_proc.stdout)
            if poll_body is None:
                failure_detail = (
                    f"backend_poll.py produced no parseable JSON on stdout; "
                    f"stdout={poll_proc.stdout!r}"
                )
        if failure_detail is not None:
            consecutive_failures += 1
            if consecutive_failures >= _POLL_MAX_CONSECUTIVE_FAILURES:
                raise RouterAcceptanceError(
                    f"poll failed {consecutive_failures} consecutive ticks; "
                    f"declaring the poll dead. last failure: {failure_detail}"
                )
            logger.warning(
                "transient poll failure (%d/%d): %s -- retrying after backoff.",
                consecutive_failures,
                _POLL_MAX_CONSECUTIVE_FAILURES,
                failure_detail,
            )
            sleep_fn(poll_interval_seconds * consecutive_failures)
            continue
        consecutive_failures = 0
        assert poll_body is not None  # narrowed by the failure_detail branch
        poll_history.append(poll_body)
        if poll_body.get("status") in terminal_statuses:
            return poll_history
        sleep_fn(poll_interval_seconds)


def _run_finalize_and_check(
    plan: LiveCommandPlan,
    *,
    issue: int,
    subprocess_run: Callable[..., subprocess.CompletedProcess],
) -> dict[str, Any]:
    """Run ``dispatch_issue.py finalize`` once and validate the response.

    Teardown MUST run unconditionally on the --live path --
    ``build_live_command_plan`` always passes
    ``--skip-confirm-artifacts`` so the only path to rc!=0 is a real
    CLI / backend crash (missing sidecar, unknown backend kind, actual
    ``backend.teardown`` failure). All of these mean a live VM / job
    may STILL be billing; the harness fails LOUD rather than masking
    that as success, and every raise carries
    :func:`_live_infra_warning` (M4.1: these raises fire exactly when
    a live VM may be unmanaged, so the manual verification commands
    must ride along). A swallowed rc=3 here (the old behavior) was the
    spend-leak: confirm_artifacts FAILed → rc=3 → teardown SKIPPED →
    live VM billing while harness exited 0.
    """
    finalize_proc = subprocess_run(
        plan.finalize_argv,
        capture_output=True,
        text=True,
        cwd=plan.repo_relative_cwd,
        check=False,
    )
    finalize_body = _parse_last_json_line(finalize_proc.stdout)
    if finalize_proc.returncode != 0:
        raise RouterAcceptanceError(
            f"dispatch_issue.py finalize exited with rc={finalize_proc.returncode}: "
            f"teardown may NOT have run -- live VM/job may still be billing. "
            f"stderr={finalize_proc.stderr.strip()!r} stdout_body={finalize_body!r}. "
            + _live_infra_warning(issue)
        )
    if finalize_body is None:
        raise RouterAcceptanceError(
            "dispatch_issue.py finalize produced no parseable JSON on stdout; "
            f"stdout={finalize_proc.stdout!r}. " + _live_infra_warning(issue)
        )
    # Defense-in-depth: even rc=0 must report ``phase=teardown`` -- the
    # only ok-rc-0 finalize body shape the dispatch CLI emits is
    # ``{"ok": True, "phase": "teardown", ...}``. Anything else means
    # finalize returned 0 without actually tearing down (would indicate
    # a regression in dispatch_issue._cmd_finalize).
    if finalize_body.get("phase") != "teardown":
        raise RouterAcceptanceError(
            "dispatch_issue.py finalize returned rc=0 but did NOT report "
            f"phase=teardown (body={finalize_body!r}); teardown was SKIPPED "
            "-- live VM/job may still be billing. Refusing to claim success. "
            + _live_infra_warning(issue)
        )
    return finalize_body


def _parse_last_json_line(stdout: str) -> dict[str, Any] | None:
    """Return the last non-blank line of stdout parsed as JSON, or None.

    The dispatch + poll CLIs print ONE JSON line on stdout; we read
    the LAST one (defensive against an upstream log line that landed
    on stdout by accident -- the JSON output is always the final line).
    """
    for raw in reversed(stdout.splitlines()):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


class RouterAcceptanceError(RuntimeError):
    """Raised when the harness cannot interpret a subprocess outcome."""


# ---------------------------------------------------------------------------
# Negative cases -- all use injected mocks; no infrastructure required
# ---------------------------------------------------------------------------


@dataclass
class _NegativeMockBackend:
    """ComputeBackend-shaped recorder for the negative cases.

    The negative cases drive ``router.route`` directly with injected
    backends; we do NOT subclass ``ComputeBackend`` to avoid pulling
    every abstract method into the test surface. The router only
    touches a small subset of the ABC during the relevant flows.
    """

    kind: str  # BackendKind
    cluster: str | None = None
    launches: list[Any] = field(default_factory=list)
    teardowns: list[Any] = field(default_factory=list)
    launch_should_raise: BaseException | None = None
    est_start_override: float | None = None
    poll_status_sequence: list[str] = field(default_factory=lambda: ["running"])
    _poll_index: int = 0

    @property
    def name(self) -> str:
        return self.kind

    def prepare(self, spec: Any) -> None:
        return None

    def launch(self, spec: Any) -> Any:
        if self.launch_should_raise is not None:
            raise self.launch_should_raise
        from explore_persona_space.backends.base import RunHandle

        handle = RunHandle(
            backend=self.kind,  # type: ignore[arg-type]
            cluster=self.cluster,
            job_id=f"mock-{self.kind}-job",
            pod_name=f"eps-issue-{spec.issue}",
            scratch_dir="/scratch/mock",
            log_path="/scratch/mock/job.out",
            extra={"issue": spec.issue, "intent": spec.intent},
        )
        self.launches.append(handle)
        return handle

    def estimate_start(self, spec: Any) -> Any:
        from datetime import UTC, datetime

        return datetime.now(tz=UTC)

    def estimate_start_seconds(self, spec: Any) -> float | None:
        return self.est_start_override

    def poll(self, handle: Any) -> Any:
        from explore_persona_space.backends.base import PollResult

        idx = min(self._poll_index, len(self.poll_status_sequence) - 1)
        status = self.poll_status_sequence[idx]
        self._poll_index += 1
        return PollResult(
            status=status,
            current_phase="mock",
            new_milestone=False,
            last_log_mtime_sec_ago=0,
            pid_alive=status == "running",
            log_tail_excerpt="",
        )

    def fetch_logs(self, handle: Any) -> str:
        return ""

    def fetch_results(self, handle: Any) -> None:
        return None

    def confirm_artifacts(self, handle: Any) -> bool:
        return True

    def teardown(self, handle: Any) -> None:
        self.teardowns.append(handle)


def negative_free_busy_to_gcp() -> dict[str, Any]:
    """Free lanes report beyond-park est-start; assert escalation to GCP.

    The router MUST NOT call ``RunPodBackend.launch`` on this path
    (real-money safety). We assert by injecting a RunPod backend whose
    ``launch`` raises -- the test passes iff route() never invokes it.
    """
    # Use a temp lease dir so the test never touches ~/.eps-routing.
    import tempfile

    from explore_persona_space.backends.base import RunSpec
    from explore_persona_space.backends.router import (
        RouterConfig,
        route,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        from explore_persona_space.backends.router import LeaseStore

        store = LeaseStore(lease_dir=Path(tmpdir))

        nibi = _NegativeMockBackend(
            kind="nibi",
            cluster="nibi",
            # Beyond-park est-start (lying scheduler): 24 hours.
            est_start_override=86_400.0,
            # Job never reaches RUNNING -- park-cap fires the cancel.
            poll_status_sequence=["running"],  # router uses is_started gate
        )
        gcp = _NegativeMockBackend(kind="gcp")
        runpod = _NegativeMockBackend(
            kind="runpod",
            launch_should_raise=AssertionError("RunPod.launch must not be called on auto path"),
        )

        # is_started always False for nibi (PENDING for the whole park);
        # is_live_after_cancel returns False (cancel resolved instantly).
        spec = RunSpec(issue=901, intent="lora-7b", backend="auto")
        result = route(
            spec,
            runpod_backend=runpod,
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            lease_store=store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            mila_socket_alive=lambda: False,
            config=RouterConfig(
                free_wait_seconds=2,  # tiny park so the test resolves quickly
                poll_interval=0.01,
                cancel_grace_seconds=1,
                # Pin the legacy free-first order: this scenario proves the
                # free→GCP ESCALATION chain; the GCP-first standing default
                # would resolve at GCP before the park/cancel under test.
                lane_order=("nibi", "fir", "mila", "gcp"),
            ),
            now_fn=time.monotonic,
            sleep_fn=lambda _s: None,  # don't actually sleep
        )
        return {
            "chosen_kind": result.chosen_kind,
            "requested_kind": result.requested_kind,
            "nibi_launches": len(nibi.launches),
            "gcp_launches": len(gcp.launches),
            "runpod_launches": len(runpod.launches),
            "attempts": [a.outcome for a in result.attempts],
        }


def negative_gcp_full_to_runpod() -> dict[str, Any]:
    """#656: every GCP ladder rung capacity-misses and no free lane is wired,
    so the auto chain falls through to the RunPod TERMINAL rung — the
    deliberate reversal of the no-auto-RunPod invariant.

    We assert by injecting a PASSIVE RunPod that records its launch (NOT the
    exploding double the pre-#656 scenarios used) and checking ``chosen_kind
    == "runpod"`` + ``reason == auto_fallback_runpod`` + exactly one RunPod
    launch. A long ``ft-7b`` (no A100-40 fallback, not short) yields a
    single on-demand A100-80 rung, so the ladder exhausts immediately and the
    RunPod rung fires — proving the harness recognizes the new terminal
    outcome.
    """
    import tempfile

    from explore_persona_space.backends.base import RunSpec
    from explore_persona_space.backends.gcp import GcpProvisioningError
    from explore_persona_space.backends.router import (
        ROUTE_REASON_RUNPOD_FALLBACK,
        LeaseStore,
        RouterConfig,
        route,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LeaseStore(lease_dir=Path(tmpdir))
        gcp = _NegativeMockBackend(
            kind="gcp",
            launch_should_raise=GcpProvisioningError(
                "ZONE_RESOURCE_POOL_EXHAUSTED",
                evidence={"matched_pattern": "RESOURCE_EXHAUSTED"},
            ),
        )
        runpod = _NegativeMockBackend(kind="runpod")  # PASSIVE: records the terminal launch
        spec = RunSpec(issue=902, intent="ft-7b", backend="auto")  # long, no A100-40 fallback
        result = route(
            spec,
            runpod_backend=runpod,
            gcp_backend=gcp,
            lease_store=store,
            mila_socket_alive=lambda: False,
            config=RouterConfig(free_wait_seconds=2, poll_interval=0.01, cancel_grace_seconds=1),
            now_fn=time.monotonic,
            sleep_fn=lambda _s: None,
        )
        return {
            "chosen_kind": result.chosen_kind,
            "reason": result.reason,
            "expected_reason": ROUTE_REASON_RUNPOD_FALLBACK,
            "gcp_launches": len(gcp.launches),
            "runpod_launches": len(runpod.launches),
            "residual_gap": result.extra.get("runpod_fallback_residual_gap"),
            "attempts": [a.outcome for a in result.attempts],
        }


def negative_cancel_race() -> dict[str, Any]:
    """Free lane's job races to RUNNING just as the cancel fires.

    The router MUST keep the running job (not double-kill it). We
    detect by having ``is_running_after_cancel`` flip True the moment
    the cancel is requested -- the router should KEEP the racing job
    and return success on the free lane.
    """
    import tempfile

    from explore_persona_space.backends.base import RunSpec
    from explore_persona_space.backends.router import (
        LeaseStore,
        RouterConfig,
        route,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LeaseStore(lease_dir=Path(tmpdir))

        nibi = _NegativeMockBackend(kind="nibi", cluster="nibi")
        gcp = _NegativeMockBackend(kind="gcp")
        runpod = _NegativeMockBackend(
            kind="runpod",
            launch_should_raise=AssertionError("RunPod.launch must not be called on auto path"),
        )

        # is_started returns False the whole park -> cancel triggers.
        # Right after cancel-request, is_running_after_cancel returns
        # True -- the racing job took off. is_live_after_cancel says
        # the job is gone from the queue (it raced into RUNNING and the
        # cancel actually killed it -- but we KEEP it because the
        # cancel-race detection fired first; the router records this as
        # a kept-not-double-killed outcome).
        spec = RunSpec(issue=902, intent="lora-7b", backend="auto")
        result = route(
            spec,
            runpod_backend=runpod,
            free_backends={"nibi": nibi},
            gcp_backend=gcp,
            lease_store=store,
            is_started=lambda _b, _h: False,
            is_live_after_cancel=lambda _b, _h: False,
            is_running_after_cancel=lambda _b, _h: True,  # the race
            mila_socket_alive=lambda: False,
            config=RouterConfig(
                free_wait_seconds=1,
                poll_interval=0.01,
                cancel_grace_seconds=1,
                # Pin the legacy free-first order: the cancel-race on the
                # free lane is the behavior under test.
                lane_order=("nibi", "fir", "mila", "gcp"),
            ),
            now_fn=time.monotonic,
            sleep_fn=lambda _s: None,
        )
        return {
            "chosen_kind": result.chosen_kind,
            "requested_kind": result.requested_kind,
            "nibi_launches": len(nibi.launches),
            "nibi_teardowns": len(nibi.teardowns),
            "runpod_launches": len(runpod.launches),
            "attempts": [(a.kind, a.outcome) for a in result.attempts],
        }


def negative_duplicate_cron_tick() -> dict[str, Any]:
    """Run finalize TWICE on the same sidecar; assert idempotent.

    The orchestrator's bg-Bash poll loop AND the 45-min ``issue-tick``
    backstop cron can both fire ``dispatch_issue.py finalize`` for the
    same handle. The second tick MUST NOT crash. Since the Mn4.3
    stale-sidecar fix, the FIRST successful finalize renames the
    sidecar to ``<name>.finalized``, so the second tick sees a missing
    sidecar and no-ops with the benign rc=2 ``missing_handle_sidecar``
    shape (exactly ONE teardown reaches the backend). The harness
    exercises this directly: write a sidecar, call the CLI's
    ``_cmd_finalize`` twice, assert (a) neither call CRASHES (rc=0
    then the benign rc=2 no-op), AND (b) the backend recorded the
    first teardown. The backend-side idempotency guarantee
    (``ComputeBackend.teardown`` ABC docstring: a duplicate teardown
    is absorbed cleanly) is validated by per-backend tests elsewhere;
    here we prove the CLI level doesn't barf on the duplicate tick.
    """
    import tempfile

    from explore_persona_space.backends.artifacts import EXPECTED_ARTIFACTS_HANDLE_KEY
    from explore_persona_space.backends.base import RunHandle
    from explore_persona_space.backends.issue_dispatch import (
        write_handle_sidecar,
    )

    issue = 903
    with tempfile.TemporaryDirectory() as tmpdir:
        sidecar = Path(tmpdir) / f"issue-{issue}-handle.json"
        handle = RunHandle(
            backend="nibi",
            cluster="nibi",
            job_id="mock-job",
            pod_name=f"eps-issue-{issue}",
            scratch_dir="/scratch/mock",
            log_path="/scratch/mock/job.out",
            extra={
                "issue": issue,
                EXPECTED_ARTIFACTS_HANDLE_KEY: {
                    "issue": issue,
                    "sentinel_path": "/tmp/sentinel.json",
                },
            },
        )
        write_handle_sidecar(handle, sidecar)

        # Run finalize TWICE; the CLI should absorb the second call.
        # The first call renames the sidecar to ``*.finalized``
        # (Mn4.3), so the second call no-ops on the missing sidecar
        # with the benign rc=2 ``missing_handle_sidecar`` shape. The
        # router-acceptance contract is: the duplicate tick must not
        # CRASH and must not re-execute teardown against a stale
        # handle. We assert by tracking rc codes + teardown counts.

        # ``scripts`` is a namespace package importable only with the
        # repo root on sys.path. pytest puts the rootdir there; a direct
        # ``uv run python scripts/router_acceptance.py negative ...``
        # puts ``scripts/`` itself there instead, so insert the root
        # (live acceptance finding: the duplicate-cron-tick case
        # crashed with ModuleNotFoundError outside pytest).
        repo_root = str(Path(__file__).resolve().parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scripts.dispatch_issue import main as dispatch_main

        nibi = _NegativeMockBackend(kind="nibi", cluster="nibi")

        def _factory() -> dict[str, Any]:
            return {
                "runpod_backend": _NegativeMockBackend(kind="runpod"),
                "free_backends": {"nibi": nibi},
                "gcp_backend": None,
                "marker_poster": lambda **_kw: None,
                "is_started": lambda _b, _h: True,
                "is_live_after_cancel": lambda _b, _h: False,
                "reconnect_fn": lambda _b, _k, _s: None,
                "mila_socket_alive": lambda: False,
            }

        import io as _io
        from contextlib import redirect_stdout

        rc_codes: list[int] = []
        bodies: list[dict[str, Any]] = []
        for _ in range(2):
            buf = _io.StringIO()
            with redirect_stdout(buf):
                rc = dispatch_main(
                    [
                        "finalize",
                        "--issue",
                        str(issue),
                        "--handle-file",
                        str(sidecar),
                    ],
                    backends_factory=_factory,
                )
            rc_codes.append(rc)
            body = _parse_last_json_line(buf.getvalue())
            bodies.append(body or {})

        return {
            "rc_codes": rc_codes,
            "teardown_count": len(nibi.teardowns),
            "bodies": bodies,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_live(args: argparse.Namespace) -> int:
    """``live`` action: dry-run by default; --live actually drives the lane.

    On the ``--live`` path:

    1. Builds a launch-subprocess-scoped env (``os.environ.copy()`` +
       ``EPM_PERSIST_ADAPTER_HF_REPO`` +
       ``EPM_PERSIST_ADAPTER_SUBFOLDER``, the ONLY env vars
       ``trainer.py:_persist_adapter`` reads) and threads it to the
       launch via ``run_live_lane(..., launch_env=...)`` -- the parent
       process's ``os.environ`` is never mutated, so in-process
       callers (the test suite via ``main([...])``) see no leakage.
       The backends carry the two vars onward to the REMOTE workload
       env (slurm ``PASSTHROUGH_ENV_KEYS`` / gcp
       ``STARTUP_PASSTHROUGH_ENV_KEYS``), so check (a) has a per-lane
       artifact to find. The subfolder uses the PRE-launch literal
       backend string; check (a) probes the same via
       ``artifact_lane``.
    2. Drives launch -> poll -> finalize via :func:`run_live_lane`.
       ``build_live_command_plan`` always passes
       ``--skip-confirm-artifacts`` so teardown ALWAYS runs (no spend
       leak); ``run_live_lane`` raises on any non-zero finalize rc OR
       on a finalize body that doesn't report ``phase=teardown``, and
       runs best-effort cleanup teardown before re-raising on any
       mid-flight exception after a successful launch.
    3. Generates the per-lane figure via
       :func:`generate_acceptance_figure` -- threaded with the
       RESOLVED lane (``auto`` -> ``chosen_kind``) so the filename
       matches the check-(b) probe -- and ``git add``s it, so check
       (b) has a real artifact to find.
    4. Evaluates the PASS checklist in-process, threading the
       canonical job name (``pod_name`` from the launch outcome) and
       the GCP project (carried via the GCP backend defaults the
       launcher used) to check (d). The harness's exit code reflects
       the lane verdict (0=PASS, 1=FAIL).
    """
    repo_root = Path.cwd()
    # Confirm the smoke dataset is present (loud failure if not).
    spec = resolve_smoke_dataset(repo_root=repo_root)
    logger.info(
        "smoke dataset resolved: %s (%s, %d rows). %s",
        spec.local_path,
        spec.source,
        spec.row_count,
        spec.provenance,
    )

    repo_branch = args.repo_branch
    if repo_branch is None and args.backend in {"gcp", "auto"}:
        # Default to the worktree's CURRENT branch so the GCP lane tests
        # the code under test, not stale origin/main (issue 535 r6). The
        # branch must exist on origin — the startup script clones it.
        repo_branch = _current_git_branch(repo_root)
        if repo_branch and repo_branch != "main":
            logger.info(
                "gcp repo_branch defaulted to current branch %r — ensure it is pushed",
                repo_branch,
            )

    plan = build_live_command_plan(
        issue=args.issue,
        backend=args.backend,
        intent=args.intent,
        repo_root=repo_root,
        time_budget_hours=args.time_budget_hours,
        repo_branch=repo_branch,
    )

    if not args.live:
        emit_live_dry_run(plan, backend=args.backend, issue=args.issue)
        return 0

    # Per-issue LANE LOCK for the whole live run (launch -> finalize ->
    # checklist). The sidecar + lease are namespaced per ISSUE, so two
    # concurrent lanes for the same issue clobber each other's handle:
    # live incident (issue 535) — a Mila lane launched while a GCP lane
    # was mid-run overwrote the GCP handle sidecar, the GCP finalize
    # tore down the WRONG (already-cancelled) handle, and the A100 VM
    # was left RUNNING + billing until manually deleted. Fail FAST here
    # instead. Held via flock on a dedicated lane-lock file (released
    # by process exit; ``fd`` deliberately kept open for the lane's
    # lifetime).
    import fcntl

    lane_lock_path = Path.home() / ".eps-routing" / f"issue-{args.issue}.lane.lock"
    lane_lock_path.parent.mkdir(parents=True, exist_ok=True)
    lane_lock_fd = os.open(lane_lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lane_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lane_lock_fd)
        logger.error(
            "another live acceptance lane for issue %d is already in flight "
            "(lane lock %s is held). Concurrent lanes for the same issue "
            "clobber the per-issue handle sidecar and CAN STRAND A BILLING "
            "VM — refusing to start. Wait for the other lane or use a "
            "different issue number.",
            args.issue,
            lane_lock_path,
        )
        return 1

    # Adapter-persist env vars MUST reach the LAUNCH subprocess.
    # ``trainer.py:_persist_adapter`` reads BOTH (on the REMOTE
    # VM / compute node):
    #   EPM_PERSIST_ADAPTER_HF_REPO  (model repo, e.g. superkaiba1/explore-persona-space)
    #   EPM_PERSIST_ADAPTER_SUBFOLDER  (per-lane subfolder)
    # The launch env reaches only the LOCAL ``dispatch_issue.py``
    # process; the backends forward both vars to the remote workload
    # env via their non-secret passthrough lists --
    # ``slurm.PASSTHROUGH_ENV_KEYS`` (rendered into the sourced
    # ``secrets.env``) and ``gcp.STARTUP_PASSTHROUGH_ENV_KEYS``
    # (instance metadata exported by the startup script). Without the
    # full chain, training writes adapter weights locally but does NOT
    # persist them to HF, so check (a) (hf_artifact_present)
    # FALSE-FAILS the lane EVEN WHEN the live run was otherwise
    # healthy. Verbatim env-var names match the canonical recipe in
    # ``.claude/rules/upload-policy.md``; do NOT invent
    # ``EPM_PERSIST_ADAPTER_HF_SUBFOLDER`` (no such var exists in
    # trainer.py).
    #
    # Scope: pass them ONLY to the launch subprocess via the
    # ``launch_env`` kwarg below -- do NOT mutate the parent
    # process's global ``os.environ``. The harness runs in the same
    # interpreter as the test suite when invoked via ``main([...])``,
    # so mutating ``os.environ`` here leaks the override across
    # in-process callers (the test would inherit a stale subfolder
    # name from a previous test). Building the env dict locally
    # keeps the mutation subprocess-scoped.
    #
    # Lane string: the env is built BEFORE launch, so an ``auto`` run
    # cannot know its resolved lane yet -- the subfolder uses the
    # literal ``args.backend`` (``auto``), and check (a) probes the
    # SAME string via ``artifact_lane=args.backend`` below. ONE source
    # of truth; a resolved-lane probe against an ``auto``-named env
    # subfolder was the false-FAIL path.
    launch_env = os.environ.copy()
    launch_env["EPM_PERSIST_ADAPTER_HF_REPO"] = args.hf_model_repo
    launch_env["EPM_PERSIST_ADAPTER_SUBFOLDER"] = ACCEPTANCE_HF_SUBFOLDER.format(
        issue=args.issue, lane=args.backend
    )

    # --live path actually spends compute. Stay loud about it.
    logger.warning(
        "router_acceptance --live: about to drive a real launch on lane=%r issue=%d",
        args.backend,
        args.issue,
    )
    started = time.monotonic()
    outcome = run_live_lane(
        plan,
        backend=args.backend,
        issue=args.issue,
        launch_env=launch_env,
    )
    elapsed_seconds = time.monotonic() - started
    print(json.dumps(outcome, sort_keys=True, indent=2))
    if outcome["phase"] == "launch_terminal":
        return 2

    # The launch_body carries the canonical pod_name (= the job name
    # the launcher used, NOT ``eps-issue-<N>`` reconstructed -- see
    # check (d) docstring) and the chosen_kind the router actually
    # picked (the requested lane may have been ``auto`` and the
    # router resolved it to nibi / gcp / mila / fir).
    launch_body = outcome.get("launch_body") or {}
    chosen_kind = launch_body.get("chosen_kind") or args.backend
    canonical_job_name = launch_body.get("pod_name")

    # Resolve the actual lane BEFORE writing the figure. The figure
    # write and the check-(b) probe MUST agree on the lane string;
    # writing ``router_acceptance_auto.png`` here while check (b)
    # greps ``router_acceptance_<resolved_lane>.png`` is the false-FAIL
    # path. ``auto`` -> ``chosen_kind`` (the lane the router actually
    # picked); explicit -> the override. ONE variable serves both the
    # figure/teardown lane AND check (c)'s expected chosen_kind (the
    # two were previously duplicate expressions under separate names);
    # check (a)'s artifact probe alone uses the PRE-launch literal
    # ``args.backend`` via ``artifact_lane`` (see env comment above).
    resolved_lane = chosen_kind if args.backend == "auto" else args.backend

    # 3) Harness-produced figure for check (b). The smoke workload
    # itself emits no figure -- this is the acceptance EVIDENCE the
    # check (b) probe expects to find tracked/staged in git. Lane
    # threaded as ``resolved_lane`` so the filename matches the
    # check-(b) probe (see comment above).
    figure_path = generate_acceptance_figure(
        issue=args.issue,
        lane=resolved_lane,
        elapsed_seconds=elapsed_seconds,
        chosen_kind=chosen_kind,
        repo_root=repo_root,
    )
    logger.info("acceptance figure generated + staged: %s", figure_path)

    # 4) PASS checklist for the lane the router actually picked.
    # GCP project: the dispatch CLI's GCP path uses default_gcp_config()
    # under the hood, so the verifier MUST use the same. Carry the
    # project explicitly to make the invariant visible (and to leave
    # a hook for a future per-launch override the launch_body could
    # surface).
    gcp_project = None
    gcp_config_name = None
    if resolved_lane == "gcp":
        from explore_persona_space.backends.gcp import default_gcp_config

        cfg = default_gcp_config()
        gcp_project = cfg.project
        gcp_config_name = cfg.gcloud_config

    verdict = evaluate_pass_checklist(
        issue=args.issue,
        lane=resolved_lane,
        expected_lane=resolved_lane,
        repo_root=repo_root,
        hf_model_repo=args.hf_model_repo,
        io=VerifierIO(),
        robot_alias_for_slurm=args.robot_alias,
        canonical_job_name=canonical_job_name,
        gcp_project=gcp_project,
        gcp_config_name=gcp_config_name,
        # Check (a) probes the subfolder the PRE-launch env actually
        # named (auto runs bake the literal ``auto`` into
        # EPM_PERSIST_ADAPTER_SUBFOLDER -- the resolved lane is
        # unknowable before launch).
        artifact_lane=args.backend,
    )
    print(verdict.format())
    return 0 if verdict.passed else 1


def _cmd_verify_lane(args: argparse.Namespace) -> int:
    """``verify-lane`` action: run the PASS checklist for a finished run."""
    repo_root = Path.cwd()
    expected = args.expected_lane or args.lane
    verdict = evaluate_pass_checklist(
        issue=args.issue,
        lane=args.lane,
        expected_lane=expected,
        repo_root=repo_root,
        hf_model_repo=args.hf_model_repo,
        io=VerifierIO(),
        robot_alias_for_slurm=args.robot_alias,
    )
    print(verdict.format())
    return 0 if verdict.passed else 1


def _cmd_negative(args: argparse.Namespace) -> int:
    """``negative`` action: drive one of the injected-mock negative cases."""
    cases: dict[str, Callable[[], dict[str, Any]]] = {
        "free-busy-to-gcp": negative_free_busy_to_gcp,
        "gcp-full-to-runpod": negative_gcp_full_to_runpod,
        "cancel-race": negative_cancel_race,
        "duplicate-cron-tick": negative_duplicate_cron_tick,
    }
    if args.case not in cases:
        print(
            f"unknown negative case {args.case!r}; expected one of {sorted(cases)}",
            file=sys.stderr,
        )
        return 2
    outcome = cases[args.case]()
    print(json.dumps(outcome, sort_keys=True, indent=2))

    # Per-case assertions -- the harness double-checks the structural
    # claim it just made so a regression in router behavior surfaces
    # here even when the test_router_acceptance.py suite hasn't run.
    if args.case == "free-busy-to-gcp":
        assert outcome["chosen_kind"] == "gcp", (
            f"free-busy-to-gcp: expected chosen_kind=gcp, got {outcome['chosen_kind']!r}"
        )
        assert outcome["runpod_launches"] == 0, (
            "free-busy-to-gcp: RunPod.launch was called on the auto path "
            f"({outcome['runpod_launches']} launches)"
        )
    elif args.case == "gcp-full-to-runpod":
        # #656: every GCP rung capacity-missed → the RunPod terminal rung
        # fires. The harness recognizes `auto_fallback_runpod` as a valid
        # terminal outcome (the reversed no-auto-RunPod invariant).
        assert outcome["chosen_kind"] == "runpod", (
            f"gcp-full-to-runpod: expected chosen_kind=runpod, got {outcome['chosen_kind']!r}"
        )
        assert outcome["reason"] == outcome["expected_reason"], (
            f"gcp-full-to-runpod: expected reason={outcome['expected_reason']!r}, "
            f"got {outcome['reason']!r}"
        )
        assert outcome["runpod_launches"] == 1, (
            f"gcp-full-to-runpod: RunPod terminal rung must launch exactly once, "
            f"got {outcome['runpod_launches']} launches"
        )
        assert outcome["residual_gap"], "gcp-full-to-runpod: residual_gap must be recorded"
    elif args.case == "cancel-race":
        # The racing job is KEPT on the free lane -- the router
        # detected the cancel-race and did NOT double-kill it.
        assert outcome["chosen_kind"] == "nibi", (
            f"cancel-race: expected chosen_kind=nibi (kept the racing job), "
            f"got {outcome['chosen_kind']!r}"
        )
        assert outcome["runpod_launches"] == 0, "cancel-race: RunPod was launched on auto path"
    elif args.case == "duplicate-cron-tick":
        # The first tick must succeed (rc=0, teardown ran). The second
        # tick must NOT crash: either rc=0 (a backend-absorbed
        # duplicate teardown, the pre-Mn4.3 CLI behavior) or the benign
        # rc=2 ``missing_handle_sidecar`` no-op (the Mn4.3 CLI renames
        # the sidecar to ``*.finalized`` after the first teardown, so
        # the duplicate tick finds nothing to tear down). Any other rc
        # is a crash regression. teardown_count in (1, 2) keeps both
        # CLI generations correct under the contract: no
        # double-teardown CRASH, no stale-handle teardown.
        assert outcome["rc_codes"][0] == 0, (
            f"duplicate-cron-tick: first tick must succeed, got rc_codes={outcome['rc_codes']!r}"
        )
        assert outcome["rc_codes"][1] in (0, 2), (
            f"duplicate-cron-tick: second tick must be rc=0 (absorbed duplicate) or the "
            f"benign rc=2 missing-sidecar no-op, got rc_codes={outcome['rc_codes']!r}"
        )
        if outcome["rc_codes"][1] == 2:
            assert outcome["bodies"][1].get("reason") == "missing_handle_sidecar", (
                f"duplicate-cron-tick: second tick rc=2 must be the benign "
                f"missing-sidecar shape, got body={outcome['bodies'][1]!r}"
            )
        assert outcome["teardown_count"] in (1, 2), (
            f"duplicate-cron-tick: expected teardown_count in (1,2), "
            f"got {outcome['teardown_count']!r}"
        )
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    live = sub.add_parser(
        "live",
        help="Drive a real per-lane acceptance run (dry-run by default; --live to actually spend).",
    )
    live.add_argument("--issue", type=int, required=True, help="Acceptance task / issue number.")
    live.add_argument(
        "--backend",
        choices=["nibi", "fir", "mila", "gcp", "auto"],
        required=True,
        help="Lane to test. ``auto`` exercises the free->GCP escalation chain.",
    )
    live.add_argument("--intent", default="lora-7b", help="Workload intent (default: lora-7b).")
    live.add_argument(
        "--time-budget-hours",
        type=float,
        default=None,
        help=(
            "Override the intent-default SLURM --time budget (hours), passed "
            "through to dispatch_issue.py launch. The ~20-step smoke finishes "
            "in well under 1h, and a short --time lets SLURM backfill the job "
            "into IDLE+PLANNED node windows where the 6h lora-7b default "
            "pends past the 600s park cap (live finding, issue 535 Mila lane)."
        ),
    )
    live.add_argument(
        "--live",
        action="store_true",
        help="Actually shell out to dispatch_issue.py + backend_poll.py. "
        "Without this flag the harness prints the dry-run command sequence only.",
    )
    live.add_argument(
        "--hf-model-repo",
        default=DEFAULT_ACCEPTANCE_HF_REPO,
        help=(
            "HF model repo for the per-lane adapter artifact (set as "
            "EPM_PERSIST_ADAPTER_HF_REPO on the launch env; also used by "
            "check (a) hf_artifact_present in-process after the lane completes). "
            "Defaults to the dedicated PRIVATE acceptance repo — the canonical "
            "public repo is over the account public-storage quota (issue 535 r8)."
        ),
    )
    live.add_argument(
        "--robot-alias",
        default=None,
        help=(
            "SLURM robot ssh alias for the squeue teardown probe in check (d). "
            "Required for nibi / fir / mila lanes; ignored for gcp."
        ),
    )
    live.add_argument(
        "--repo-branch",
        default=None,
        help=(
            "Git branch the GCE startup script clones (gcp/auto lanes). "
            "Defaults to the worktree's current branch (must be pushed); "
            "SLURM lanes rsync the local worktree and ignore this."
        ),
    )
    live.add_argument("--debug", action="store_true", help="Log to stderr at DEBUG level.")

    verify = sub.add_parser(
        "verify-lane",
        help="Run the PASS checklist (a)-(d) on a finished lane.",
    )
    verify.add_argument("--issue", type=int, required=True)
    verify.add_argument("--lane", required=True, choices=["nibi", "fir", "mila", "gcp"])
    verify.add_argument(
        "--expected-lane",
        default=None,
        help="Expected chosen_kind in the routing marker (default: same as --lane).",
    )
    verify.add_argument(
        "--hf-model-repo",
        default=DEFAULT_ACCEPTANCE_HF_REPO,
        help=(
            "HF model repo to check for the per-lane adapter artifact. "
            "Defaults to the dedicated private acceptance repo; pass the "
            "canonical repo explicitly to verify pre-quota lanes (nibi/mila)."
        ),
    )
    verify.add_argument(
        "--robot-alias",
        default=None,
        help="SLURM robot ssh alias for the squeue teardown probe (nibi/mila lanes).",
    )
    verify.add_argument("--debug", action="store_true")

    negative = sub.add_parser(
        "negative",
        help="Run an injected-mock negative case (no infrastructure required).",
    )
    negative.add_argument(
        "case",
        choices=[
            "free-busy-to-gcp",
            "gcp-full-to-runpod",
            "cancel-race",
            "duplicate-cron-tick",
        ],
        help="Which negative scenario to drive.",
    )
    negative.add_argument("--debug", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if getattr(args, "debug", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Secrets live in the repo .env (dotenv), NOT the ambient shell — a
    # harness launched from a clean session env otherwise forwards NOTHING
    # to the remote workload (live finding, issue 535 GCP lane r7: the VM
    # booted with empty WANDB_API_KEY because this dispatch process had
    # none to thread into the instance metadata) AND verify-lane's HfApi
    # probe reads an empty HF_TOKEN. resolve_dotenv_path walks to the main
    # git worktree, so this works from a linked worktree without its own
    # .env; override=False keeps already-exported vars authoritative.
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    if args.action == "live":
        return _cmd_live(args)
    if args.action == "verify-lane":
        return _cmd_verify_lane(args)
    if args.action == "negative":
        return _cmd_negative(args)
    parser.error(f"unknown action {args.action!r}")
    return 4  # pragma: no cover -- parser.error -> SystemExit(2)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ACCEPTANCE_FIGURE_PATH",
    "ACCEPTANCE_HF_SUBFOLDER",
    "DEFAULT_SMOKE_HYDRA_ARGS",
    "ROUTING_MARKER",
    "CheckResult",
    "LaneVerdict",
    "LiveCommandPlan",
    "RouterAcceptanceError",
    "SmokeDatasetSpec",
    "VerifierIO",
    "build_live_command_plan",
    "check_clean_teardown",
    "check_git_figure_present",
    "check_hf_artifact_present",
    "check_routing_marker_posted",
    "emit_live_dry_run",
    "evaluate_pass_checklist",
    "generate_acceptance_figure",
    "main",
    "negative_cancel_race",
    "negative_duplicate_cron_tick",
    "negative_free_busy_to_gcp",
    "resolve_smoke_dataset",
    "run_live_lane",
]
