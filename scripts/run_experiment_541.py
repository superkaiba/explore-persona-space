#!/usr/bin/env python3
# (greek + arrow + multiplication/minus-sign characters intentional in docstrings/labels)
"""Experiment #541 — prior-stratified rerun of #500 (enriched bystander panel).

Thin wrapper around ``scripts/run_experiment_444.py``, built the way
``run_experiment_500.py`` is (module-global patching of the parent driver),
with the #541 deltas (plan §4.3):

  - ``PANEL`` is read from Phase 0's ``prior_screen.json`` selection block
    (24 personas; the 15 #500 originals asserted nested), NOT hardcoded.
  - ``ARM_SOURCE`` maps the 4 Phase-0-selected sources (marine_biologist +
    courthouse_architecture_historian anchors + S-mid + S-top picks).
  - The 23 new candidate personas are injected into ``p.PERSONAS`` at import
    (``issue541_personas.inject_candidates()``).
  - ALL cells are freshly trained — there is NO Arm-A adapter-reuse path
    (plan §10 fitness check (d): adapter reuse would make training provenance
    co-vary with the source-prior axis, the exact cross-arm confound P2 is
    a primary prediction about).
  - Phase 2 baselines run ONCE over the full 24-panel into a SHARED
    directory (``eval_results/issue_541/baseline_shared``); per-arm
    ``--phase full-eval`` symlinks the shared baseline files into the arm
    subtree so the parent's assertions + aggregation see them (plan
    assumption 16; per-arm fallback = run ``--phase baselines`` without
    ``--shared-baseline-dir`` semantics, covered by "deviations allowed").
  - Adapters publish to ``adapters/exp541-<arm_slug>-<condition>-seed<S>``;
    raw completions to ``issue541_prior_stratified/<arm_slug>/...``.
  - ``--smoke`` (or env ``EPM_541_SMOKE=1``): seeds capped to (42,), eval
    panel capped to 2 bystanders (local_historian, assistant). Same code
    path as the full sweep — parameterization only (PASS_UNIFIED contract).

Inherited unchanged from #500/#444: training recipe constants, eval
constants, the 5-way Haiku judge + ``_run_5way_rejudge`` re-entrancy, the
``judged_5way_*`` filename split, the dataset placeholder guard.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import run_experiment_444 as p  # noqa: E402  (bootstrap runs at import time)
from issue541_personas import ORIGINAL_15, inject_candidates  # noqa: E402
from run_experiment_500 import (  # noqa: E402
    _format_local_resident_prompt,
    _make_phase_dataset_with_guard,
    _phase_baseline_judge,
    _phase_trained_cell_5way_rejudge,
    _seed_fact_pick_from_444,
)

# Smoke runs are namespaced (eval root, data dirs, adapter HF paths, WandB
# project) so smoke artifacts can NEVER poison the full run's skip-if-exists
# resume logic on the same pod. _configure_namespacing() rebinds these once
# the smoke flag is known; the defaults are the full-run values.
EVAL_ROOT_NAME = "issue_541"
ADAPTER_PREFIX = "exp541"
PRIOR_SCREEN_PATH = (
    REPO / "eval_results" / EVAL_ROOT_NAME / "phase0_prescreen" / "prior_screen.json"
)
BASELINE_SHARED_DIR = REPO / "eval_results" / EVAL_ROOT_NAME / "baseline_shared"
WANDB_PROJECT_541 = "exp541-prior-stratified"
HF_DATA_BUCKET_ROOT = "issue541_prior_stratified"
# Round-6 quota deviation (bug_class hf_public_storage_quota_exceeded): the
# account is over its PUBLIC HF storage quota, so LFS uploads 403 account-wide.
# Adapters persist to this PRIVATE repo (separate quota, headroom probed
# 2026-06-10) instead of the plan's canonical superkaiba1/explore-persona-space;
# migration to canonical is pending user-gated storage cleanup. Recorded in the
# results sentinel's plan_deviations.
HF_OVERFLOW_MODEL_REPO = "superkaiba1/explore-persona-space-overflow"
GPU_HOURS_BUDGETED = 20.0  # plan §"Compute": Estimated GPU-hours (total): 20


def _configure_namespacing(smoke: bool) -> None:
    """Rebind the module-level roots for smoke runs (issue_541_smoke namespace)."""
    global EVAL_ROOT_NAME, ADAPTER_PREFIX, PRIOR_SCREEN_PATH, BASELINE_SHARED_DIR
    global WANDB_PROJECT_541, HF_DATA_BUCKET_ROOT
    if not smoke:
        return
    EVAL_ROOT_NAME = "issue_541_smoke"
    ADAPTER_PREFIX = "exp541smoke"
    PRIOR_SCREEN_PATH = (
        REPO / "eval_results" / EVAL_ROOT_NAME / "phase0_prescreen" / "prior_screen.json"
    )
    BASELINE_SHARED_DIR = REPO / "eval_results" / EVAL_ROOT_NAME / "baseline_shared"
    WANDB_PROJECT_541 = "exp541-prior-stratified-smoke"
    HF_DATA_BUCKET_ROOT = "issue541_prior_stratified_smoke"


SEEDS_FULL: tuple[int, ...] = (42, 137, 256)
SMOKE_BYSTANDERS: tuple[str, ...] = ("local_historian", "assistant")

# Anchor arms are always available (Phase-0 picks extend this at runtime).
ANCHOR_ARM_SLUGS: dict[str, str] = {
    "marine_biologist": "arm_marine_biologist",
    "courthouse_architecture_historian": "arm_courthouse_architecture_historian",
}


def _smoke_enabled(args: argparse.Namespace) -> bool:
    return bool(args.smoke or os.environ.get("EPM_541_SMOKE") == "1")


def _raise_nofile_soft_limit(target: int = 65536) -> None:
    """Raise the RLIMIT_NOFILE soft limit toward ``min(target, hard)``; fail-soft.

    Defense in depth against EMFILE (#541 round 5: the threaded Haiku judge
    fan-out exhausted the pod's 1024 soft FD limit ~75 min into
    [phase=full_eval]; the root-cause fix is the shared Anthropic client in
    ``run_experiment_444._anthropic_client``). Called at the top of
    ``main()`` so every phase process — dispatcher and wave workers alike —
    raises its own limit regardless of which launcher spawned it. Never
    raises: a failed setrlimit only logs a warning.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if new_soft <= soft:
            print(
                f"[run_experiment_541] RLIMIT_NOFILE soft limit already {soft} "
                f"(hard={hard}); leaving as-is"
            )
            return
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(
            f"[run_experiment_541] RLIMIT_NOFILE soft limit raised: "
            f"{soft} -> {new_soft} (hard={hard})"
        )
    except (ValueError, OSError) as e:
        print(f"[run_experiment_541] WARNING: could not raise RLIMIT_NOFILE soft limit: {e!r}")


def _load_selection(require: bool) -> dict[str, Any] | None:
    """Read prior_screen.json's selection + gate blocks (Phase-0 output)."""
    if not PRIOR_SCREEN_PATH.exists():
        if require:
            raise RuntimeError(
                f"{PRIOR_SCREEN_PATH} missing — run the Phase-0 prescreen "
                "(scripts/issue541_prescreen.py --step 0a/0b-gen/0b-judge/0c/0d) first."
            )
        return None
    return json.loads(PRIOR_SCREEN_PATH.read_text())


def _resolve_arms(screen: dict[str, Any] | None, smoke: bool) -> dict[str, str]:
    """arm-source-persona -> arm_slug map for this run."""
    if smoke:
        return {"marine_biologist": "arm_marine_biologist"}
    if screen is None:
        return dict(ANCHOR_ARM_SLUGS)
    return dict(screen["selection"]["arm_slugs"])


def _resolve_panel(screen: dict[str, Any] | None, smoke: bool, source: str) -> tuple[str, ...]:
    """Full panel (source included). Smoke: source + 2 fixed bystanders."""
    if smoke:
        return (source, *SMOKE_BYSTANDERS)
    assert screen is not None
    panel = tuple(screen["selection"]["panel"])
    if not screen.get("smoke"):
        assert set(ORIGINAL_15) <= set(panel), "original 15 must be nested in the panel"
        assert len(panel) == 24, (len(panel), panel)
    return panel


# ---------------------------------------------------------------------------
# Module-global patchers (mirror run_experiment_500's, #541 paths)
# ---------------------------------------------------------------------------
def _reroute_paths(arm_slug: str) -> None:
    """Rebind the parent driver's six path globals to the #541 subtree."""
    base_eval = REPO / "eval_results" / EVAL_ROOT_NAME / arm_slug
    p.EVAL_RESULTS_DIR = base_eval
    p.DATA_DIR = REPO / "data" / ADAPTER_PREFIX / arm_slug
    p.ADAPTER_ROOT = REPO / "outputs" / f"{ADAPTER_PREFIX}_adapters" / arm_slug
    p.FIGURES_DIR = REPO / "figures" / EVAL_ROOT_NAME / arm_slug
    p.PHASE0_DIR = base_eval / "phase0_fact_candidates"
    p.ON_POLICY_DIR = base_eval / "on_policy_negs"
    p.WANDB_PROJECT = WANDB_PROJECT_541
    p.EXPERIMENT_NAME = f"issue541_{arm_slug}"
    # Defensive: if any inherited phase ever posts a marker from the pod, the
    # sentinel must land in THIS issue's poll glob, not #444's.
    p.SENTINEL_FILENAME_FMT = "issue-541-{kind_slug}-{epoch}.json"


def _override_train_cell_hf_path(arm_slug: str) -> None:
    """Publish adapters to the #541 HF namespace (never overwrite #444/#500)."""

    def _new_hf_path_in_repo(self: Any) -> str:
        condition = self.condition.replace("-", "_")
        return f"adapters/{ADAPTER_PREFIX}-{arm_slug}-{condition}-seed{self.seed}"

    p.TrainCell.hf_path_in_repo = property(_new_hf_path_in_repo)


def _set_arm_personas(source_persona: str, panel: tuple[str, ...]) -> None:
    """Patch the parent's persona globals for this arm (panel from Phase 0)."""
    if source_persona != "no_system" and source_persona not in p.PERSONAS:
        raise RuntimeError(
            f"arm source {source_persona!r} not in PERSONAS registry — "
            "issue541_personas.inject_candidates() must run before arm setup."
        )
    p.TEACHING_PERSONA = source_persona
    eval_panel = tuple(x for x in panel if x != source_persona)
    assert len(eval_panel) == len(panel) - 1, (source_persona, panel)
    p.EVAL_PERSONA_ORDER = eval_panel

    neg = tuple(x for x in p.ARBITRARY_NON_TEACH_PERSONAS if x != source_persona)
    p.ARBITRARY_NON_TEACH_PERSONAS = neg
    p.NON_TEACH_PERSONAS = neg

    p._aggregate_one_cell.__defaults__ = (p.EVAL_PERSONA_ORDER,)
    p.TRAINED_CONDITIONS = (p.CONDITION_ON_POLICY_SUPPRESSION,)

    assert source_persona == p.TEACHING_PERSONA
    assert eval_panel == p.EVAL_PERSONA_ORDER
    assert p._aggregate_one_cell.__defaults__ == (eval_panel,)
    assert p.TRAINED_CONDITIONS == (p.CONDITION_ON_POLICY_SUPPRESSION,)


def _widen_to_full_panel(panel: tuple[str, ...]) -> None:
    """Baselines measure EVERY panel persona (sources included)."""
    p.EVAL_PERSONA_ORDER = panel
    p._aggregate_one_cell.__defaults__ = (p.EVAL_PERSONA_ORDER,)


def _link_shared_baseline_into_arm() -> None:
    """Symlink the shared 24-panel baseline files into the arm subtree.

    ``phase_full_eval`` asserts ``baseline_completions_<slug>.jsonl`` exists in
    the arm's EVAL_RESULTS_DIR and skips baseline judging when
    ``baseline_judged_<slug>.jsonl`` is present. The baseline is generated +
    5-way judged ONCE (shared 24-panel run); each arm links both files in.
    """
    facts = p._resolve_figure_facts()
    slug = facts.figure_slug
    p.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for fname in (f"baseline_completions_{slug}.jsonl", f"baseline_judged_{slug}.jsonl"):
        src = BASELINE_SHARED_DIR / fname
        dst = p.EVAL_RESULTS_DIR / fname
        if not src.exists():
            raise RuntimeError(
                f"shared baseline file {src} missing — run "
                "`run_experiment_541.py --arm marine_biologist --phase baselines` first."
            )
        if dst.is_symlink() or dst.exists():
            continue
        dst.symlink_to(src.resolve())


# ---------------------------------------------------------------------------
# #541 upload phase (routes to the #541 HF data bucket)
# ---------------------------------------------------------------------------
def _make_phase_upload_541(arm_slug: str, upload_shared_baseline: bool):
    """Wrapper-local ``phase_upload`` for the #541 bucket (quota-resilient).

    Mirrors #500's upload (baseline + per-cell completions/judged/5-way +
    on-policy raw), bucket = ``issue541_prior_stratified/<arm_slug>/<slug>``.
    The shared 24-panel baseline uploads once (marine arm) under
    ``issue541_prior_stratified/baseline_shared/<slug>/``.

    Round-6 changes (bug_class hf_public_storage_quota_exceeded — the account
    is over its PUBLIC storage quota so ALL LFS uploads 403 account-wide):

    1. Text payloads upload as regular (non-LFS) git blobs via
       ``issue541_upload_lib.upload_text_file``; any file >= 9.5 MB is
       line-split into < 9 MB shards + a reassembly manifest (the 10 MB
       ``upload_file`` LFS auto-routing cliff is never reached).
    2. Idempotent resume: ``list_repo_files`` is called ONCE per target repo
       at phase start; already-present paths are skipped.
    3. The arm's LoRA adapter dirs (3 seeds; LFS-heavy safetensors) persist
       to the PRIVATE ``HF_OVERFLOW_MODEL_REPO`` (LFS works there) at the
       SAME ``path_in_repo`` they would have used on the canonical model
       repo — so the pending user-gated migration is a 1:1 copy.
    4. NOTHING writes to the canonical model repo this round.
    """

    def phase_upload_541(args: argparse.Namespace) -> dict[str, Any]:
        import issue541_upload_lib as ulib
        from huggingface_hub import HfApi

        facts = p._resolve_figure_facts()
        figure_slug = facts.figure_slug
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        bucket = f"{HF_DATA_BUCKET_ROOT}/{arm_slug}/{figure_slug}"
        # One listing per target repo per phase invocation (resume contract).
        # A truncated listing on a huge repo degrades to a harmless re-upload
        # of an identical blob, never to data loss.
        existing_data = set(api.list_repo_files(p.HF_DATA_REPO, repo_type="dataset"))
        existing_overflow = set(api.list_repo_files(HF_OVERFLOW_MODEL_REPO, repo_type="model"))
        shard_workdir = REPO / "outputs" / "issue541_upload_shards" / arm_slug
        files_uploaded: list[str] = []
        files_skipped: list[str] = []
        shard_manifests: list[str] = []

        def _upload_one(local_path: Path, path_in_repo: str) -> None:
            if not local_path.exists():
                print(f"[upload-541] skip missing: {local_path}")
                return
            res = ulib.upload_text_file(
                api,
                local_path=local_path,
                path_in_repo=path_in_repo,
                repo_id=p.HF_DATA_REPO,
                existing=existing_data,
                workdir=shard_workdir,
            )
            files_uploaded.extend(res["uploaded"])
            files_skipped.extend(res["skipped"])
            if res["manifest_path_in_repo"]:
                shard_manifests.append(res["manifest_path_in_repo"])

        if upload_shared_baseline:
            shared_bucket = f"{HF_DATA_BUCKET_ROOT}/baseline_shared/{figure_slug}"
            for fname in (
                f"baseline_completions_{figure_slug}.jsonl",
                f"baseline_judged_{figure_slug}.jsonl",
            ):
                _upload_one(BASELINE_SHARED_DIR / fname, f"{shared_bucket}/{fname}")

        adapter_uploads: list[dict[str, Any]] = []
        for cell in p._enumerate_train_cells():
            tag = cell.tag
            _upload_one(
                p.EVAL_RESULTS_DIR / f"completions_{tag}.jsonl",
                f"{bucket}/raw_completions/completions_{tag}.jsonl",
            )
            _upload_one(
                p.EVAL_RESULTS_DIR / f"judged_{tag}.jsonl",
                f"{bucket}/raw_completions/judged_{tag}.jsonl",
            )
            _upload_one(
                p.EVAL_RESULTS_DIR / f"judged_5way_{tag}.jsonl",
                f"{bucket}/raw_completions/judged_5way_{tag}.jsonl",
            )
            # Adapter persist (the inline upload was fenced off by
            # EPM_SKIP_INLINE_CHECKPOINT_UPLOAD, so the trained adapters live
            # ONLY on the pod until this step). Local dir name mirrors
            # _train_one_cell's run_name construction in run_experiment_444.
            run_name = f"exp444_{figure_slug}_{cell.condition.replace('-', '_')}_seed{cell.seed}"
            adapter_uploads.append(
                ulib.upload_adapter_dir(
                    api,
                    local_dir=p.ADAPTER_ROOT / run_name,
                    path_in_repo=cell.hf_path_in_repo,
                    repo_id=HF_OVERFLOW_MODEL_REPO,
                    existing=existing_overflow,
                )
            )

        op_dir = p.ON_POLICY_DIR
        if op_dir.exists():
            for fp in sorted(op_dir.glob("*.jsonl")):
                _upload_one(fp, f"{bucket}/on_policy_raw/{fp.name}")
            for fp in sorted(op_dir.glob("*.json")):
                _upload_one(fp, f"{bucket}/on_policy_raw/{fp.name}")

        summary = {
            "phase": "upload",
            "arm_slug": arm_slug,
            "hf_data_repo": p.HF_DATA_REPO,
            "bucket": bucket,
            "n_files_uploaded": len(files_uploaded),
            "n_files_skipped_existing": len(files_skipped),
            "files": files_uploaded,
            "files_skipped_existing": files_skipped,
            "shard_manifests": shard_manifests,
            "overflow_model_repo": HF_OVERFLOW_MODEL_REPO,
            "adapter_uploads": adapter_uploads,
            "quota_note": (
                "non-LFS git-blob route; account over public HF storage quota "
                "(round-6 deviation, see results sentinel plan_deviations)"
            ),
            "timestamp": p._now_iso(),
        }
        out_path = p.EVAL_RESULTS_DIR / "upload_summary_541.json"
        p.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        return summary

    return phase_upload_541


# ---------------------------------------------------------------------------
# Results sentinel (pod-side marker channel; CLAUDE.md pod-side rule)
# ---------------------------------------------------------------------------
def _estimate_gpu_hours() -> tuple[float, str]:
    """Best-effort GPU-hours estimate from the eval-artifact mtime span.

    The dispatcher has no cumulative-runtime ledger across crash-relaunches
    (rounds 4/5/6 each relaunched the same script), so the span between the
    earliest and latest artifact mtimes under ``eval_results/<root>`` x the
    pod GPU count is the most honest pod-side reconstruction. It is an UPPER
    bound (not all GPUs were busy the whole span). Returns ``(hours, basis)``.
    """
    root = REPO / "eval_results" / EVAL_ROOT_NAME
    n_gpus = int(os.environ.get("EPM_541_N_GPUS", "4"))
    mtimes = [fp.stat().st_mtime for fp in root.rglob("*") if fp.is_file()]
    if not mtimes:
        return 0.0, f"no artifacts under {root} — 0 by construction"
    span_h = (max(mtimes) - min(mtimes)) / 3600.0
    basis = (
        f"wall-span of eval_results/{EVAL_ROOT_NAME} artifact mtimes "
        f"({span_h:.2f} h) x {n_gpus} GPUs — upper bound; no per-launch ledger "
        "exists across the round-4/5/6 crash-relaunches"
    )
    return round(span_h * n_gpus, 2), basis


def _compact_eval_numbers(predictors_path: Path) -> dict[str, Any]:
    """Bounded extraction of headline numbers from ``predictors.json``.

    The sentinel note feeds a task marker capped at 50k chars, so per-arm
    blocks are filtered to scalars; the full tables stay in the JSON on disk
    (path carried in ``eval_paths``).
    """
    if not predictors_path.exists():
        return {"status": "MISSING", "expected_path": str(predictors_path)}
    pred = json.loads(predictors_path.read_text())
    out: dict[str, Any] = {
        k: pred.get(k)
        for k in (
            "gate_branch",
            "sources",
            "strata",
            "new_home_max_prior",
            "collinearity_gate",
            "p2_source_prior_gating",
        )
    }
    per_arm = pred.get("per_arm", {})
    out["per_arm_scalars"] = {
        src: {k: v for k, v in d.items() if v is None or isinstance(v, (int, float, str, bool))}
        for src, d in per_arm.items()
        if isinstance(d, dict)
    }
    return out


def phase_results_sentinel(args: argparse.Namespace) -> dict[str, Any]:
    """Write the end-of-run results sentinel for ``poll_pipeline.py``.

    Carries every key in ``poll_pipeline._SENTINEL_REQUIRED_KEYS``
    (``sentinel_schema_version`` / ``kind`` / ``version``) PLUS the ten
    orchestrator result keys (``eval_numbers`` / ``eval_paths`` /
    ``reproducibility_card`` / ``wandb_url`` / ``hf_hub_url`` /
    ``worktree_path`` / ``final_commit_sha`` / ``gpu_hours_used`` /
    ``gpu_hours_budgeted`` / ``plan_deviations``) at top level; the same
    payload is serialized under ``note`` (the marker body channel). Written
    to the fixed path ``/workspace/logs/issue-541-results.json`` (matches the
    poller's ``issue-541-*.json`` glob). The launcher emits ``[phase=done]``
    AFTER this sentinel lands (incident #448: key misnamed ``schema`` +
    missing ``[phase=done]`` read a clean run as dead).
    """
    smoke = _smoke_enabled(args)
    screen = _load_selection(require=True)
    gate = screen["gate"]["branch"]
    arms = _resolve_arms(screen, smoke)
    eval_root = REPO / "eval_results" / EVAL_ROOT_NAME

    per_arm: dict[str, str] = {}
    overflow_adapter_urls: list[str] = []
    data_buckets: list[str] = []
    for slug in arms.values():
        agg = eval_root / slug / "aggregate_cleaned.json"
        per_arm[slug] = str(agg) if agg.exists() else "MISSING"
        upload_summary = eval_root / slug / "upload_summary_541.json"
        if upload_summary.exists():
            summary = json.loads(upload_summary.read_text())
            data_buckets.append(summary.get("bucket", ""))
            for item in summary.get("adapter_uploads", []):
                overflow_adapter_urls.append(item["url"])
    missing_arms = sorted(slug for slug, path in per_arm.items() if path == "MISSING")

    predictors = eval_root / "predictors.json"
    base_cov = eval_root / "base_engagement_covariates.json"
    eval_paths = {
        "per_arm_aggregate_cleaned": per_arm,
        "predictors_json": str(predictors) if predictors.exists() else "MISSING",
        "prior_screen_json": str(PRIOR_SCREEN_PATH),
        "base_engagement_covariates": str(base_cov) if base_cov.exists() else "MISSING",
        "figures_dir": str(REPO / "figures" / EVAL_ROOT_NAME),
        "upload_summaries": [
            str(eval_root / slug / "upload_summary_541.json")
            for slug in arms.values()
            if (eval_root / slug / "upload_summary_541.json").exists()
        ],
    }

    plan_deviations = [
        (
            "Raw text payload uploaded to the HF data repo as regular (non-LFS) git "
            "blobs because the account exceeded its PUBLIC HF storage quota (LFS "
            "403s account-wide); the one >10 MB file "
            "(baseline_completions_<slug>.jsonl, 10.4 MB) was line-split into <9 MB "
            "shards plus a <stem>.manifest.json (reassembly = concatenate shard "
            "lines in order)."
        ),
        (
            f"LoRA adapters uploaded to the PRIVATE {HF_OVERFLOW_MODEL_REPO} "
            "(repo_type=model) instead of the plan's canonical "
            f"{p.HF_MODEL_REPO}, due to the same public-storage quota; "
            "path_in_repo is identical to the canonical target, so migration "
            "after the user-gated storage cleanup is a 1:1 copy."
        ),
    ]
    if missing_arms:
        plan_deviations.append(
            f"Arms with no aggregate_cleaned.json at sentinel time: {missing_arms} "
            "(only the arms actually trained are uploaded/reported; nothing is "
            "invented for missing arms)."
        )

    gpu_hours_used, gpu_hours_basis = _estimate_gpu_hours()
    final_commit_sha = p._git_commit_sha()
    reproducibility_card = {
        "base_model": p.BASE_MODEL,
        "condition": "on-policy-suppression-cn",
        "seeds": list(SEEDS_FULL) if not smoke else [42],
        "training_recipe": {
            "epochs": 1,
            "lr": 2e-4,
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "batch_size": 4,
            "grad_accum": 4,
            "max_length": 1024,
            "warmup_ratio": 0.05,
        },
        "eval": {
            "temperature": p.EVAL_TEMPERATURE,
            "max_new_tokens": p.EVAL_MAX_NEW_TOKENS,
            "judge_model": p.JUDGE_MODEL,
        },
        "arms": arms,
        "panel": list(screen["selection"]["panel"]) if "selection" in screen else [],
        "wandb_project": WANDB_PROJECT_541,
        "final_commit_sha": final_commit_sha,
        "branch": "issue-541",
        "adapter_destination": {
            "repo": HF_OVERFLOW_MODEL_REPO,
            "private": True,
            "urls": overflow_adapter_urls,
            "canonical_target_blocked_by": "public HF storage quota (round-6 deviation)",
        },
        "gpu_hours_basis": gpu_hours_basis,
    }

    content: dict[str, Any] = {
        "eval_numbers": _compact_eval_numbers(predictors),
        "eval_paths": eval_paths,
        "reproducibility_card": reproducibility_card,
        "wandb_url": f"https://wandb.ai/thomasjiralerspong/{WANDB_PROJECT_541}",
        "hf_hub_url": {
            "data_repo_buckets": [
                f"https://huggingface.co/datasets/{p.HF_DATA_REPO}/tree/main/{b}"
                for b in data_buckets
                if b
            ],
            "overflow_adapters": overflow_adapter_urls,
            "canonical_model_repo": (
                f"{p.HF_MODEL_REPO} — NOT written this run (public storage quota; "
                "see plan_deviations)"
            ),
        },
        "worktree_path": (
            "/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-541"
        ),
        "final_commit_sha": final_commit_sha,
        "gpu_hours_used": gpu_hours_used,
        "gpu_hours_budgeted": GPU_HOURS_BUDGETED,
        "plan_deviations": plan_deviations,
        "gate_branch": gate,
        "smoke": smoke,
    }
    note = json.dumps(content, indent=2)
    if len(note) > 45_000:
        # events.jsonl notes are hard-capped at 50k chars; drop the bulky
        # eval_numbers block (full tables live on disk at eval_paths) and say so.
        content["eval_numbers"] = {
            "truncated": True,
            "reason": "sentinel note would exceed the 50k marker cap",
            "predictors_json": eval_paths["predictors_json"],
        }
        note = json.dumps(content, indent=2)

    body = {
        "sentinel_schema_version": 1,
        "task_id": 541,
        "kind": "epm:results",
        "version": 1,
        "gate": None,
        "blocks_pipeline": False,
        "by": "issue541-dispatcher",
        "ts": p._now_iso(),
        **content,
        "note": note,
    }
    is_pod = Path("/workspace").is_dir() or bool(os.environ.get("RUNPOD_POD_ID"))
    out_dir = Path("/workspace/logs") if is_pod else REPO / "logs" / "issue-541"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fixed name (smoke-suffixed in the smoke namespace so a later smoke can
    # never clobber the full run's sentinel before the poller drains it).
    fname = "issue-541-results-smoke.json" if smoke else "issue-541-results.json"
    sentinel = out_dir / fname
    sentinel.write_text(json.dumps(body, indent=2))
    print(f"SENTINEL_POSTED kind=epm:results gate=none blocks_pipeline=False path={sentinel}")
    return {"phase": "results-sentinel", "path": str(sentinel), "gate_branch": gate}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Experiment #541 — prior-stratified rerun of #500. Thin wrapper around "
            "run_experiment_444.py with Phase-0-driven panel/source overrides."
        )
    )
    ap.add_argument(
        "--arm",
        required=True,
        help="source persona for this arm (validated against prior_screen.json sources)",
    )
    ap.add_argument(
        "--phase", required=True, help="phase to run (parent phases + results-sentinel)"
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--condition", type=str, default=None)
    ap.add_argument("--fact-pick-id", type=int, default=None)
    ap.add_argument("--allow-multi-bpe-answer", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 seed + 2-bystander eval slice")
    args = ap.parse_args()

    # FD-limit defense in depth (#541 round 5) — before any phase work so the
    # judge fan-out and every library this process loads inherit the raise.
    _raise_nofile_soft_limit()

    smoke = _smoke_enabled(args)
    _configure_namespacing(smoke)
    inject_candidates()

    # prior_screen.json is REQUIRED for everything except preflight (the
    # launcher runs preflight before the prescreen exists).
    screen = _load_selection(require=args.phase != "preflight")
    arms = _resolve_arms(screen, smoke)
    if args.phase != "preflight" and args.arm not in arms:
        raise SystemExit(
            f"--arm {args.arm!r} is not a selected source for this run; "
            f"valid arms: {sorted(arms)} (from {PRIOR_SCREEN_PATH})"
        )
    arm_slug = arms.get(args.arm, ANCHOR_ARM_SLUGS.get(args.arm))
    if arm_slug is None:
        raise SystemExit(f"--arm {args.arm!r} has no arm slug (preflight allows anchors only)")

    panel = (
        _resolve_panel(screen, smoke, args.arm)
        if args.phase != "preflight"
        else (args.arm, *[x for x in ORIGINAL_15 if x != args.arm])
    )

    # Path + persona patches (order matters: format local_resident BEFORE any
    # phase reads PERSONAS; fact-pick seed needs PHASE0_DIR rebound first).
    _reroute_paths(arm_slug)
    _set_arm_personas(args.arm, panel)
    _format_local_resident_prompt()
    _override_train_cell_hf_path(arm_slug)
    _seed_fact_pick_from_444()

    if smoke:
        p.SEEDS = (42,)

    # Shared-baseline routing: the baselines phase generates + judges the FULL
    # panel ONCE into baseline_shared/ (plan §4.3 — the panel is arm-invariant
    # within this run, so per-arm baselines would re-judge identical rows).
    if args.phase == "baselines":
        p.EVAL_RESULTS_DIR = BASELINE_SHARED_DIR
        _widen_to_full_panel(panel)
        print(
            f"[run_experiment_541] shared baselines: {len(p.EVAL_PERSONA_ORDER)}-persona "
            f"panel (incl. all sources) -> {BASELINE_SHARED_DIR}"
        )

    if args.phase == "full-eval":
        _link_shared_baseline_into_arm()

    phases = {
        "preflight": p.phase_preflight,
        "dataset": _make_phase_dataset_with_guard(p.phase_dataset),
        "baselines": p.phase_baselines,
        "worker": p.phase_worker,
        "full-eval": p.phase_full_eval,
        "aggregate": p.phase_aggregate,
        "upload": _make_phase_upload_541(
            arm_slug, upload_shared_baseline=args.arm == "marine_biologist"
        ),
        "results-sentinel": phase_results_sentinel,
    }
    if args.phase not in phases:
        raise SystemExit(f"unknown --phase {args.phase!r}; valid choices: {list(phases)}")
    phases[args.phase](args)

    # Auto-chained judges (inherited #500 round-4/5 fixes): baselines ->
    # 5-way baseline judge; full-eval -> per-cell 5-way re-judge. Both are
    # idempotent with per-row resume; zero API calls when complete.
    if args.phase == "baselines":
        _phase_baseline_judge()
    if args.phase == "full-eval":
        _phase_trained_cell_5way_rejudge()


if __name__ == "__main__":
    main()
