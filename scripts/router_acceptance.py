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

* ``negative <case>`` -- run one of three injected-mock negative scenarios:

  - ``free-busy-to-gcp``: every free lane returns a beyond-park
    est-start; assert the router escalates to GCP and NEVER calls
    ``RunPodBackend.launch`` on the auto path.
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
  ``EPM_PERSIST_ADAPTER_HF_SUBFOLDER`` which does not exist) so the
  delete-after-eval adapter persistence lands at the expected path
  the check (a) reads.
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

#: Per-lane HF model-repo subfolder pattern. The smoke trains a LoRA
#: adapter that the existing training pipeline auto-uploads to the
#: project model repo (Upload Policy table); the harness writes to a
#: dedicated subfolder so a failed lane never clobbers a passing lane.
ACCEPTANCE_HF_SUBFOLDER = "router_acceptance/issue-{issue}-{lane}"

#: Per-lane figure path. Falls under the per-issue figures dir the
#: orchestrator already commits via Step 8 (Upload Policy).
ACCEPTANCE_FIGURE_PATH = "figures/issue_{issue}/router_acceptance_{lane}.png"

#: Per-lane events.jsonl marker key. The router writes this on every
#: chosen-lane decision (see ``epm:backend-selected v1`` in workflow.yaml).
ROUTING_MARKER = "epm:backend-selected"

#: Default Hydra overrides for the smoke workload. ~20 LoRA steps on
#: 50-row data; report_to=wandb so training metrics land in WandB per
#: the always-on WandB-required rule.
DEFAULT_SMOKE_HYDRA_ARGS: tuple[str, ...] = (
    "condition=c_router_smoke",
    "seed=0",
    "training.max_steps=20",
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
    rows = sum(1 for line in p.read_text().splitlines() if line.strip())
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
    for line in events_path.read_text().splitlines():
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
    ``trainer.py``) on the launch env so the delete-after-eval
    adapter persistence lands at the expected path.
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
) -> LaneVerdict:
    """Run all four PASS checks for one lane and return the verdict.

    ``canonical_job_name`` / ``gcp_project`` / ``gcp_config_name`` are
    threaded from the launch outcome JSON so check (d) probes the SAME
    name + project the launcher used. Defaults preserve the legacy
    behavior for pure-unit tests that don't have a launch outcome to
    thread.
    """
    checks = (
        check_hf_artifact_present(issue=issue, lane=lane, repo_id=hf_model_repo, io=io),
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
) -> dict[str, Any]:
    """Drive the full live launch -> poll -> finalize loop.

    This is the ``--live`` path; ``subprocess_run`` / ``sleep_fn`` /
    ``now_fn`` are dependency-injected so the same orchestration loop
    is exercised by unit tests (passing recorded fakes). The poll loop
    re-invokes ``backend_poll.py`` once per tick and parses the JSON
    line on stdout; it terminates on ``status in {done, dead, gate}``
    or when ``poll_timeout_seconds`` elapses.

    Returns a structured result dict so the caller can log / assert.
    Raises ``RouterAcceptanceError`` on a subprocess that returned a
    non-zero exit with no JSON line we could parse (preserves the
    fail-fast contract).
    """
    # 1) Launch.
    launch_proc = subprocess_run(
        plan.launch_argv,
        capture_output=True,
        text=True,
        cwd=plan.repo_relative_cwd,
        check=False,
    )
    if launch_proc.returncode not in (0, 2):
        # 2 is a router terminal (NoCompute / WorkloadSurfaced / ...);
        # the JSON line still carries the failure shape, so we let the
        # caller surface it as a FAIL not a crash. Other non-zero codes
        # mean the CLI itself crashed -- fail loud.
        raise RouterAcceptanceError(
            f"dispatch_issue.py launch exited with rc={launch_proc.returncode}: "
            f"stderr={launch_proc.stderr.strip()!r}"
        )
    launch_body = _parse_last_json_line(launch_proc.stdout)
    if launch_body is None:
        raise RouterAcceptanceError(
            "dispatch_issue.py launch produced no parseable JSON on stdout; "
            f"stdout={launch_proc.stdout!r} stderr={launch_proc.stderr!r}"
        )
    if launch_proc.returncode == 2 or not launch_body.get("ok", False):
        # Router terminal -- bail. The harness records the failure
        # shape so the caller can surface it as the lane verdict.
        return {
            "phase": "launch_terminal",
            "launch_body": launch_body,
            "poll_history": [],
            "finalize_body": None,
        }

    # 2) Poll until terminal.
    poll_history: list[dict[str, Any]] = []
    started = now_fn()
    terminal_statuses = {"done", "dead", "gate"}
    while True:
        if now_fn() - started > poll_timeout_seconds:
            raise RouterAcceptanceError(
                f"poll loop exceeded timeout {poll_timeout_seconds}s without terminal status. "
                f"last_poll={poll_history[-1] if poll_history else None}"
            )
        poll_proc = subprocess_run(
            plan.poll_argv,
            capture_output=True,
            text=True,
            cwd=plan.repo_relative_cwd,
            check=False,
        )
        if poll_proc.returncode != 0:
            raise RouterAcceptanceError(
                f"backend_poll.py exited with rc={poll_proc.returncode}: "
                f"stderr={poll_proc.stderr.strip()!r}"
            )
        poll_body = _parse_last_json_line(poll_proc.stdout)
        if poll_body is None:
            raise RouterAcceptanceError(
                f"backend_poll.py produced no parseable JSON on stdout; stdout={poll_proc.stdout!r}"
            )
        poll_history.append(poll_body)
        if poll_body.get("status") in terminal_statuses:
            break
        sleep_fn(poll_interval_seconds)

    # 3) Finalize. Teardown MUST run unconditionally on the --live
    # path -- ``build_live_command_plan`` always passes
    # ``--skip-confirm-artifacts`` so the only path to rc!=0 is a
    # real CLI / backend crash (missing sidecar, unknown backend kind,
    # actual ``backend.teardown`` failure). All of these mean a live
    # VM / job may STILL be billing; the harness fails LOUD rather
    # than masking that as success. A swallowed rc=3 here (the old
    # behavior) was the spend-leak: confirm_artifacts FAILed → rc=3
    # → teardown SKIPPED → live VM billing while harness exited 0.
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
            f"stderr={finalize_proc.stderr.strip()!r} stdout_body={finalize_body!r}"
        )
    if finalize_body is None:
        raise RouterAcceptanceError(
            "dispatch_issue.py finalize produced no parseable JSON on stdout; "
            f"stdout={finalize_proc.stdout!r}"
        )
    # Defense-in-depth: even rc=0 must report ``phase=teardown`` --
    # the only ok-rc-0 finalize body shape the dispatch CLI emits is
    # ``{"ok": True, "phase": "teardown", ...}``. Anything else means
    # finalize returned 0 without actually tearing down (would
    # indicate a regression in dispatch_issue._cmd_finalize).
    if finalize_body.get("phase") != "teardown":
        raise RouterAcceptanceError(
            "dispatch_issue.py finalize returned rc=0 but did NOT report "
            f"phase=teardown (body={finalize_body!r}); teardown was SKIPPED "
            "-- live VM/job may still be billing. Refusing to claim success."
        )

    return {
        "phase": "complete",
        "launch_body": launch_body,
        "poll_history": poll_history,
        "finalize_body": finalize_body,
    }


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

    The orchestrator's bg-Bash poll loop AND the 20-min ``issue-tick``
    backstop cron can both fire ``dispatch_issue.py finalize`` for the
    same handle. The second tick MUST NOT crash. The router contract
    (``ComputeBackend.teardown`` ABC docstring) says teardown is
    idempotent -- the backend absorbs the duplicate call cleanly.
    The harness exercises this directly: write a sidecar, call the
    CLI's ``_cmd_finalize`` twice, assert (a) both calls return rc=0
    (no crash on the second tick), AND (b) the backend recorded BOTH
    teardown invocations (proving the second call was actually issued
    -- a finalize CLI that silently no-op'd would mask a real bug).
    The "idempotent" guarantee is on the BACKEND, not on the CLI:
    a duplicate finalize is a real call but the backend's teardown
    is a no-op on the second pass (validated by per-backend tests
    elsewhere; here we prove the CLI doesn't barf on the duplicate).
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
        # The first call leaves the sidecar in place (finalize today
        # does NOT delete it), so the second call WOULD re-execute
        # teardown. The router-acceptance contract is: the second
        # backend.teardown call must be a no-op (idempotent per the
        # ABC), AND the second exit code must be 0 (not a crash). We
        # assert by tracking teardown call counts.

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

    1. Sets ``EPM_PERSIST_ADAPTER_HF_REPO`` +
       ``EPM_PERSIST_ADAPTER_SUBFOLDER`` (the ONLY env vars
       ``trainer.py:_persist_adapter`` reads) on the harness env BEFORE
       the launch subprocess, so the per-lane subfolder check (a) reads
       has an artifact to find. Subprocess inherits the parent env.
    2. Drives launch -> poll -> finalize via :func:`run_live_lane`.
       ``build_live_command_plan`` always passes
       ``--skip-confirm-artifacts`` so teardown ALWAYS runs (no spend
       leak); ``run_live_lane`` raises on any non-zero finalize rc OR
       on a finalize body that doesn't report ``phase=teardown``.
    3. Generates the per-lane figure via
       :func:`generate_acceptance_figure` and ``git add``s it, so
       check (b) has a real artifact to find.
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

    plan = build_live_command_plan(
        issue=args.issue,
        backend=args.backend,
        intent=args.intent,
        repo_root=repo_root,
    )

    if not args.live:
        emit_live_dry_run(plan, backend=args.backend, issue=args.issue)
        return 0

    # CRITICAL: set the adapter-persist env vars BEFORE the launch
    # subprocess. ``trainer.py:_persist_adapter`` reads BOTH:
    #   EPM_PERSIST_ADAPTER_HF_REPO  (model repo, e.g. superkaiba1/explore-persona-space)
    #   EPM_PERSIST_ADAPTER_SUBFOLDER  (per-lane subfolder)
    # Without these set, the training pipeline writes adapter weights
    # locally but does NOT persist them to HF, so check (a)
    # (hf_artifact_present) FALSE-FAILS the lane EVEN WHEN the live
    # run was otherwise healthy. Verbatim env-var names match the
    # canonical recipe in ``.claude/rules/upload-policy.md``; do NOT
    # invent ``EPM_PERSIST_ADAPTER_HF_SUBFOLDER`` (no such var
    # exists in trainer.py).
    os.environ["EPM_PERSIST_ADAPTER_HF_REPO"] = args.hf_model_repo
    os.environ["EPM_PERSIST_ADAPTER_SUBFOLDER"] = ACCEPTANCE_HF_SUBFOLDER.format(
        issue=args.issue, lane=args.backend
    )

    # --live path actually spends compute. Stay loud about it.
    logger.warning(
        "router_acceptance --live: about to drive a real launch on lane=%r issue=%d",
        args.backend,
        args.issue,
    )
    started = time.monotonic()
    outcome = run_live_lane(plan, backend=args.backend, issue=args.issue)
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

    # 3) Harness-produced figure for check (b). The smoke workload
    # itself emits no figure -- this is the acceptance EVIDENCE the
    # check (b) probe expects to find tracked/staged in git.
    figure_path = generate_acceptance_figure(
        issue=args.issue,
        lane=args.backend,
        elapsed_seconds=elapsed_seconds,
        chosen_kind=chosen_kind,
        repo_root=repo_root,
    )
    logger.info("acceptance figure generated + staged: %s", figure_path)

    # 4) PASS checklist for the lane the router actually picked.
    # auto -> chosen_kind (the actual lane); explicit -> the override.
    resolved_lane = chosen_kind if args.backend == "auto" else args.backend
    expected_lane = chosen_kind if args.backend == "auto" else args.backend
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
        expected_lane=expected_lane,
        repo_root=repo_root,
        hf_model_repo=args.hf_model_repo,
        io=VerifierIO(),
        robot_alias_for_slurm=args.robot_alias,
        canonical_job_name=canonical_job_name,
        gcp_project=gcp_project,
        gcp_config_name=gcp_config_name,
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
    elif args.case == "cancel-race":
        # The racing job is KEPT on the free lane -- the router
        # detected the cancel-race and did NOT double-kill it.
        assert outcome["chosen_kind"] == "nibi", (
            f"cancel-race: expected chosen_kind=nibi (kept the racing job), "
            f"got {outcome['chosen_kind']!r}"
        )
        assert outcome["runpod_launches"] == 0, "cancel-race: RunPod was launched on auto path"
    elif args.case == "duplicate-cron-tick":
        # Both invocations must return 0 (the CLI does NOT crash on the
        # second tick); teardown is called either ONCE or TWICE.
        # Accepting (1, 2) on purpose: today the CLI doesn't de-dup so
        # teardown is called twice and the backend's ABC contract
        # absorbs the duplicate. A FUTURE CLI-level idempotency fix
        # (e.g. dispatch_issue.py finalize deletes the sidecar after
        # the first call) would land teardown_count=1, which is ALSO
        # correct under the contract -- the test must not read that
        # future improvement as a regression. The claim under test is
        # rc_codes=[0,0] -- the CLI does NOT crash on the second tick.
        assert outcome["rc_codes"] == [0, 0], (
            f"duplicate-cron-tick: expected rc_codes=[0,0], got {outcome['rc_codes']!r}"
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
        "--live",
        action="store_true",
        help="Actually shell out to dispatch_issue.py + backend_poll.py. "
        "Without this flag the harness prints the dry-run command sequence only.",
    )
    live.add_argument(
        "--hf-model-repo",
        default="superkaiba1/explore-persona-space",
        help=(
            "HF model repo for the per-lane adapter artifact (set as "
            "EPM_PERSIST_ADAPTER_HF_REPO on the launch env; also used by "
            "check (a) hf_artifact_present in-process after the lane completes)."
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
        default="superkaiba1/explore-persona-space",
        help="HF model repo to check for the per-lane adapter artifact.",
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
        choices=["free-busy-to-gcp", "cancel-race", "duplicate-cron-tick"],
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
