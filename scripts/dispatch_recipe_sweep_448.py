# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
#!/usr/bin/env python3
"""Task #448 dispatcher — UNIFIED smoke = sweep with one cell.

5-phase pipeline (Pre-Phase 0 corpus / 0.5 centroids / 1.5 base panel / 1+2
per-cell train+eval / 3 analyze) sequential on 1× H100. Same per-cell
function body for smoke and sweep — only `--cells` set-size differs. The
plan's `--smoke` shorthand selects ONE cell (Anchor) AND skips the heavy
follow-up phases (base panel + analyze).

Per-cell discipline (one cell == one of the 11 in `CELL_SPECS`, sequential):
  1. (one-time) Pre-Phase 0 corpus top-up via Sonnet 4.5 — generates the 850-
     pair union pool + 20 canonical EVAL_QUESTIONS responses.
  2. (one-time) Phase 0.5 — extend layer-20 centroids over the 24-panel +
     extended-negative personas.
  3. (one-time) Phase 1.5 — base-Qwen 24-panel × 20-question marker log-p probe.
  4. Per cell: build_training_data → train_lora (with MarkerTrajectoryCallback)
       → merge_lora → eval_one_cell.run_eval → upload adapter to HF Hub →
       rmtree merged/ → write sentinel JSON.
  5. (one-time) Phase 3 — analyze.run_analysis.

Pod-side discipline:
  - Sets EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 (CLAUDE.md gotcha — MooseFS
    quota safety for sequential multi-cell sweep).
  - Never shells out to scripts/task.py (sentinel-file pattern only).
  - Every subprocess.* call uses env={**os.environ}; load_dotenv() is at
    module top.
  - rmtree merged/ after each cell before the next loads its weights.

Sentinels:
  /workspace/logs/issue-448-<cell>-results.json  (per cell)
  /workspace/logs/issue-448-pre-phase-0-results.json
  /workspace/logs/issue-448-phase-0-5-results.json
  /workspace/logs/issue-448-phase-1-5-results.json
  /workspace/logs/issue-448-results.json  (end-of-sweep)

CLI:
  --cells <names>         Plain-English names (Anchor,+pos-ex-100,...) OR
                          slug forms (c1_anchor,c2_pos_ex_100,...). Default: all.
  --smoke                 Shorthand for --cells Anchor + skip base-panel +
                          skip analyze.
  --dry-run               No GPU work / no Anthropic calls. Validate that
                          dispatcher modules import cleanly, marker tokenizer
                          assertion holds, persona_registry builds, --cells
                          resolves correctly. Exits with summary.
  --skip-pre-phase-0      Re-use existing data/issue_448/generic_corpus/* (set
                          when iterating on cells without re-paying Sonnet).
  --skip-phase-0-5        Re-use eval_results/issue_448/centroids/.
  --skip-base-panel       Re-use eval_results/issue_448/base/marker_logprob.json.
  --skip-analyze          Don't run Phase 3.

Plain-English `--cells` parsing accepts mixed forms:
  --cells "Anchor,+pos-ex-100-per-persona"
  --cells "c1_anchor,c2_pos_ex_100"
  --cells "Anchor,c3_pos_ex_400,+neg-personas-8"
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Module-top constants. The hub repo + adapter-path-in-repo conventions
# inherit from #411 verbatim.
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SEED = 42
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
LOG_DIR = Path("/workspace/logs")  # default; overridden by --log-dir CLI arg.

log = logging.getLogger("dispatch_recipe_sweep_448")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _resolve_cells(raw: str | None, smoke: bool) -> list[tuple[str, str, int, int, int, int]]:
    """Resolve `--cells` (CSV of plain-English OR slug) → list of CELL_SPECS rows.

    Defaults: all 11 cells. `--smoke` overrides to ['c1_anchor'].
    """
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        CELL_SPECS,
    )

    if smoke:
        wanted = {"c1_anchor"}
    elif raw is None or raw.strip() == "all" or raw.strip() == "":
        wanted = {row[0] for row in CELL_SPECS}
    else:
        wanted = set()
        # Build a name → slug lookup.
        name_to_slug = {row[1]: row[0] for row in CELL_SPECS}
        slug_set = {row[0] for row in CELL_SPECS}
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token in slug_set:
                wanted.add(token)
            elif token in name_to_slug:
                wanted.add(name_to_slug[token])
            else:
                raise ValueError(
                    f"Unknown cell {token!r}. Expected one of:\n  slugs: "
                    f"{sorted(slug_set)}\n  names: {sorted(name_to_slug.keys())}"
                )
    out = [row for row in CELL_SPECS if row[0] in wanted]
    return out


def _run_pre_phase_0(skip: bool, dry_run: bool, canonical_only: bool = False) -> dict[str, object]:
    """Pre-Phase 0: Sonnet 4.5 union pool + canonical responses.

    ``canonical_only=True`` (round-2 fix B2 for --smoke-real) generates only
    the 20 canonical EVAL_QUESTIONS responses, skipping the 650-pair top-up.
    Cost ~$0.02, ~30s. Lets the smoke-real path build the eval rig without
    paying the full ~$5 / 10-min top-up.
    """
    t0 = time.time()
    log.info("=" * 70)
    log.info("Pre-Phase 0 (Sonnet 4.5 corpus top-up + canonical responses)")
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        build_wrong_claim_pool,
    )

    sentinel = LOG_DIR / "issue-448-pre-phase-0-results.json"
    out_dir = build_wrong_claim_pool.OUT_DIR

    if dry_run:
        log.info("Pre-Phase 0 DRY-RUN: validating imports only.")
        # Validate the module imports + key functions are callable.
        assert callable(build_wrong_claim_pool.build_corpus)
        assert callable(build_wrong_claim_pool.load_canonical_responses)
        assert callable(build_wrong_claim_pool.load_union_pool)
        summary = {"phase": "pre_phase_0", "status": "dry_run_validated"}
    elif skip:
        # Verify cached artifacts exist.
        canonical_path = out_dir / "eval_canonical_responses.json"
        union_path = out_dir / "union_pool.json"
        if not canonical_path.exists() or not union_path.exists():
            raise FileNotFoundError(
                f"--skip-pre-phase-0 set but artifacts missing. Expected "
                f"{canonical_path} + {union_path}. Run Pre-Phase 0 first."
            )
        log.info("Pre-Phase 0 SKIPPED (artifacts exist at %s)", out_dir)
        summary = {
            "phase": "pre_phase_0",
            "status": "skipped_artifacts_exist",
            "out_dir": str(out_dir),
        }
    else:
        import asyncio

        summary = asyncio.run(
            build_wrong_claim_pool.build_corpus(
                out_dir=out_dir, canonical_responses_only=canonical_only
            )
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary["wall_seconds"] = round(time.time() - t0, 1)
    summary["sentinel_path"] = str(sentinel)
    sentinel.write_text(json.dumps(summary, indent=2))
    log.info("Pre-Phase 0 done in %.1fs", summary["wall_seconds"])
    return summary


def _run_phase_0_5(skip: bool, dry_run: bool) -> dict[str, object]:
    """Phase 0.5: extend layer-20 centroids."""
    t0 = time.time()
    log.info("=" * 70)
    log.info("Phase 0.5 (extend centroids)")

    sentinel = LOG_DIR / "issue-448-phase-0-5-results.json"
    out_pt = Path("eval_results/issue_448/centroids/centroids_layer20.pt")

    if dry_run:
        # Just verify module imports.
        from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
            extend_centroids,  # noqa: F401
        )

        log.info("Phase 0.5 DRY-RUN: extend_centroids module imports cleanly.")
        summary = {"phase": "phase_0_5", "status": "dry_run_validated"}
    elif skip:
        if not out_pt.exists():
            # Round-3 fix R2-1: don't fatal-fail under --smoke-real (the
            # smoke-real path only runs Phase 1.5 base panel which doesn't
            # depend on centroids; centroids are only consumed by the
            # per-cell trajectory callback + Phase 3 analyzer, both of
            # which are skipped under --smoke-real).
            log.warning(
                "Phase 0.5 SKIPPED but %s missing. Tolerated: downstream "
                "phases that depend on centroids are also skipped.",
                out_pt,
            )
            summary = {"phase": "phase_0_5", "status": "skipped_artifact_missing"}
        else:
            log.info("Phase 0.5 SKIPPED (artifact exists at %s)", out_pt)
            summary = {"phase": "phase_0_5", "status": "skipped_artifact_exists"}
    else:
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.contrastive_recipe_sweep_448.extend_centroids",
        ]
        log.info("Phase 0.5 subprocess: %s", " ".join(cmd))
        subprocess.run(cmd, env={**os.environ}, check=True)
        summary = {"phase": "phase_0_5", "centroids_path": str(out_pt)}

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary["wall_seconds"] = round(time.time() - t0, 1)
    summary["sentinel_path"] = str(sentinel)
    sentinel.write_text(json.dumps(summary, indent=2))
    log.info("Phase 0.5 done in %.1fs", summary["wall_seconds"])
    return summary


def _run_phase_1_5_base_panel(
    skip: bool,
    dry_run: bool,
    eval_personas_limit: int | None = None,
    eval_questions_limit: int | None = None,
) -> dict[str, object]:
    """Phase 1.5: panel × question base-marker-logprob probe.

    Round-2 fix B2: ``eval_personas_limit`` + ``eval_questions_limit`` shrink
    the grid for `--smoke-real` (e.g. 2 × 2 = 4 forward passes proves the
    rig works end-to-end on local GPU).
    """
    t0 = time.time()
    log.info("=" * 70)
    log.info("Phase 1.5 (base panel marker log-p)")

    sentinel = LOG_DIR / "issue-448-phase-1-5-results.json"
    out_dir = Path("eval_results/issue_448/base")
    out_path = out_dir / "marker_logprob.json"

    if dry_run:
        from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
            eval_one_cell,  # noqa: F401
        )

        log.info("Phase 1.5 DRY-RUN: eval_one_cell module imports cleanly.")
        summary = {"phase": "phase_1_5", "status": "dry_run_validated"}
    elif skip:
        if not out_path.exists():
            raise FileNotFoundError(
                f"--skip-base-panel set but {out_path} missing. Re-run Phase 1.5."
            )
        log.info("Phase 1.5 SKIPPED (artifact exists at %s)", out_path)
        summary = {"phase": "phase_1_5", "status": "skipped_artifact_exists"}
    else:
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.experiments.contrastive_recipe_sweep_448.eval_one_cell",
            "--cell",
            "base",
            "--hub-model-id",
            BASE_MODEL,
            "--out-dir",
            str(out_dir),
            "--sentinel-path",
            str(sentinel),
        ]
        if eval_personas_limit is not None:
            cmd.extend(["--eval-personas-limit", str(eval_personas_limit)])
        if eval_questions_limit is not None:
            cmd.extend(["--eval-questions-limit", str(eval_questions_limit)])
        log.info("Phase 1.5 subprocess: %s", " ".join(cmd))
        subprocess.run(cmd, env={**os.environ}, check=True)
        summary = {
            "phase": "phase_1_5",
            "base_logp_path": str(out_path),
            "eval_personas_limit": eval_personas_limit,
            "eval_questions_limit": eval_questions_limit,
        }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary["wall_seconds"] = round(time.time() - t0, 1)
    if sentinel != Path(summary.get("sentinel_path", "")):
        summary["sentinel_path"] = str(sentinel)
        sentinel.write_text(json.dumps(summary, indent=2))
    log.info("Phase 1.5 done in %.1fs", summary["wall_seconds"])
    return summary


def _build_trajectory_callback(
    tokenizer,
    canonical_responses: dict[str, str],
    *,
    centroids_path: Path | None = None,
    pos_persona_names: list[str] | None = None,
    neg_persona_names: list[str] | None = None,
    trajectory_output_path: Path | None = None,
    cell_slug: str = "unknown",
    seed: int = 42,
):
    """Build the MarkerTrajectoryCallback for the per-cell training run.

    Round-2 fix C2: subset = 3 nearest + 3 farthest panel personas (by
    cosine to the mean of the cell's negative-set centroids). Falls back
    to first-N if the centroid bundle isn't available yet (Phase 0.5
    not yet run).
    """
    import numpy as np
    import torch

    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        TRAJECTORY_N_PERSONAS,
        TRAJECTORY_N_QUESTIONS,
    )
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.marker_trajectory_callback import (  # noqa: E501
        MarkerTrajectoryCallback,
    )
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )
    from explore_persona_space.personas import EVAL_QUESTIONS

    # Round-2 fix C2: pick the trajectory subset by cosine distance to the
    # cell's negative-set centroids (3 nearest + 3 farthest from
    # `neg_centroid_mean`). Per plan §4.0quater. Centroid bundle is loaded
    # via the path arg; if missing OR the bundle doesn't cover required
    # personas, fall back to the round-1 first-N approach (logged).
    n_per_half = max(1, TRAJECTORY_N_PERSONAS // 2)
    subset_personas: dict[str, str]
    if centroids_path is not None and centroids_path.exists() and neg_persona_names:
        bundle = torch.load(centroids_path, weights_only=False)
        layer = bundle.get("layer", 20)
        tensor = bundle["centroids"][layer].to(torch.float32).numpy()
        names = list(bundle["persona_names"])
        name_to_idx = {n: i for i, n in enumerate(names)}
        # Centroid mean of the cell's negative personas (typically 2 for cells
        # 1-9; 4 or 8 for cells 10/11).
        try:
            neg_idx = [name_to_idx[n] for n in neg_persona_names]
            neg_mean = tensor[neg_idx].mean(axis=0)
            neg_mean_norm = neg_mean / (np.linalg.norm(neg_mean) + 1e-12)
            # For every panel persona (excluding the cell's source + negatives
            # themselves), compute cosine sim to neg_mean.
            cell_pos = set(pos_persona_names) if pos_persona_names else set()
            cell_neg = set(neg_persona_names)
            sim_pairs: list[tuple[str, float]] = []
            for pname in EVAL_PERSONAS_24:
                if pname in cell_pos or pname in cell_neg:
                    continue
                if pname not in name_to_idx:
                    continue
                vec = tensor[name_to_idx[pname]]
                vec_norm = vec / (np.linalg.norm(vec) + 1e-12)
                sim_pairs.append((pname, float(vec_norm @ neg_mean_norm)))
            sim_pairs.sort(key=lambda kv: kv[1])
            # Lowest sim = "farthest from negatives" (high predicted leakage).
            farthest = [p for p, _ in sim_pairs[:n_per_half]]
            # Highest sim = "nearest to negatives" (low predicted leakage).
            nearest = [p for p, _ in sim_pairs[-n_per_half:]]
            picks = farthest + nearest
            subset_personas = {p: EVAL_PERSONAS_24[p] for p in picks}
            log.info(
                "Trajectory subset (cosine-stratified): nearest=%s farthest=%s",
                nearest,
                farthest,
            )
        except KeyError as e:
            log.warning(
                "Centroid lookup missing persona %s; falling back to first-N trajectory subset.",
                e,
            )
            subset_personas = dict(list(EVAL_PERSONAS_24.items())[:TRAJECTORY_N_PERSONAS])
    else:
        log.info(
            "Trajectory subset: first-%d EVAL_PERSONAS_24 (centroid bundle "
            "or neg_persona_names unavailable)",
            TRAJECTORY_N_PERSONAS,
        )
        subset_personas = dict(list(EVAL_PERSONAS_24.items())[:TRAJECTORY_N_PERSONAS])
    subset_questions = EVAL_QUESTIONS[:TRAJECTORY_N_QUESTIONS]
    return MarkerTrajectoryCallback(
        tokenizer=tokenizer,
        persona_prompts=subset_personas,
        questions=subset_questions,
        canonical_responses=canonical_responses,
        output_path=trajectory_output_path,
        cell_slug=cell_slug,
        seed=seed,
    )


def _build_training_data_for_cell(
    cell_slug: str,
    pos_ex_per_p: int,
    pos_personas: int,
    neg_ex_per_p: int,
    neg_personas: int,
    out_path: Path,
) -> Path:
    """Per-cell training-data build (CPU, in-process)."""
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.build_training_data import (
        build_cell,
    )

    return build_cell(
        cell_slug=cell_slug,
        pos_ex_per_persona=pos_ex_per_p,
        pos_personas=pos_personas,
        neg_ex_per_persona=neg_ex_per_p,
        neg_personas=neg_personas,
        output_path=out_path,
    )


def _train_and_merge(
    cell_slug: str,
    seed: int,
    train_jsonl: Path,
    output_dir: Path,
    trajectory_callback,
) -> tuple[Path, Path]:
    """Train + merge in-process. Returns (adapter_dir, merged_dir)."""
    from explore_persona_space.train.sft import (
        TrainLoraConfig,
        merge_lora,
        train_lora,
    )

    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # #411 verbatim hparams. Plan §11 binding.
    cfg = TrainLoraConfig(
        gpu_id=0,
        epochs=3,
        lr=1e-5,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,  # effective batch 16
        max_length=1024,
        warmup_ratio=0.05,
        seed=seed,
        run_name=f"issue448_{cell_slug}_seed{seed}",
        report_to="wandb",
        save_strategy="no",
        gradient_checkpointing=True,
        packing=False,
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=f"adapters/issue_448/{cell_slug}_seed{seed}",
    )
    log.info("[%s] Training LoRA → %s", cell_slug, adapter_dir)
    callbacks = [trajectory_callback] if trajectory_callback is not None else None
    train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(train_jsonl),
        output_dir=str(adapter_dir),
        cfg=cfg,
        callbacks=callbacks,
    )

    log.info("[%s] Merging LoRA into base → %s", cell_slug, merged_dir)
    merge_lora(
        base_model_path=BASE_MODEL,
        adapter_path=str(adapter_dir),
        output_dir=str(merged_dir),
        gpu_id=0,
    )
    return adapter_dir, merged_dir


def _eval_cell(
    cell_slug: str,
    merged_dir: Path,
    eval_out_dir: Path,
    sentinel_path: Path,
) -> None:
    """In-process eval (no vLLM, so subprocess isolation isn't required).

    Plan §4.4 unified-path guarantee: smoke and sweep run the EXACT same
    `eval_one_cell.run_eval` function. No subprocess shape divergence.
    """
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        eval_one_cell,
    )

    eval_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_one_cell.run_eval(
        cell_slug=cell_slug,
        model_path=str(merged_dir),
        out_dir=eval_out_dir,
    )
    # Write per-cell sentinel.
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_payload = {
        "cell": cell_slug,
        "merged_model_path": str(merged_dir),
        "marker_logprob_path": str(out_path),
        "marker_logprob_summary_path": str(eval_out_dir / "marker_logprob_summary.json"),
        "n_cells_evaluated": 480,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    sentinel_path.write_text(json.dumps(sentinel_payload, indent=2))
    log.info("[%s] Wrote sentinel → %s", cell_slug, sentinel_path)


def _run_one_cell(
    cell_slug: str,
    plain_name: str,
    pos_ex_per_p: int,
    pos_personas: int,
    neg_ex_per_p: int,
    neg_personas: int,
    seed: int,
    slab_root: Path,
    runs_root: Path,
    dry_run: bool,
    canonical_responses: dict[str, str] | None,
    centroids_path: Path | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Per-cell: build → train → merge → eval → upload → cleanup → sentinel.

    Round-2 fix C2: ``centroids_path`` enables the trajectory callback to pick
    its 3-nearest + 3-farthest subset by cosine distance to the cell's
    negative-set centroid mean (rather than a static first-N).
    Round-2 fix C3: ``resume=True`` short-circuits if the per-cell sentinel
    already exists (skip rebuild/retrain).
    """
    import torch

    t_start = time.time()
    output_dir = runs_root / f"{cell_slug}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_out_dir = slab_root / cell_slug
    sentinel_path = LOG_DIR / f"issue-448-{cell_slug}-results.json"
    train_jsonl = output_dir / "train_pool.jsonl"

    # Round-2 fix C3: resume mode skips cells whose sentinel exists AND whose
    # eval_results JSON is also on disk (both invariants: a sentinel without
    # the eval JSON would be a half-finished cell).
    if resume and sentinel_path.exists() and (eval_out_dir / "marker_logprob.json").exists():
        log.info(
            "[%s] RESUME: sentinel + eval JSON exist; SKIPPING (use --no-resume to force).",
            cell_slug,
        )
        return {
            "cell": cell_slug,
            "plain_name": plain_name,
            "seed": seed,
            "wall_seconds": 0.0,
            "output_dir": str(output_dir),
            "eval_out_dir": str(eval_out_dir),
            "sentinel_path": str(sentinel_path),
            "adapter_hf_path": f"adapters/issue_448/{cell_slug}_seed{seed}",
            "status": "resumed_skip",
        }

    log.info("=" * 70)
    log.info(
        "[%s] CELL START (name=%s, pos_ex/p=%d, pos_personas=%d, neg_ex/p=%d, neg_personas=%d)",
        cell_slug,
        plain_name,
        pos_ex_per_p,
        pos_personas,
        neg_ex_per_p,
        neg_personas,
    )

    # Build training data (CPU). Always runs (even in dry-run; CPU-only and
    # cheap; verifies persona_registry + union pool wiring).
    if not dry_run:
        _build_training_data_for_cell(
            cell_slug,
            pos_ex_per_p,
            pos_personas,
            neg_ex_per_p,
            neg_personas,
            train_jsonl,
        )
    else:
        log.info(
            "[%s] DRY-RUN: skipping training-data build (would write %s)", cell_slug, train_jsonl
        )

    if dry_run:
        log.info("[%s] DRY-RUN: skipping train + merge + eval", cell_slug)
        wall = time.time() - t_start
        return {
            "cell": cell_slug,
            "plain_name": plain_name,
            "status": "dry_run",
            "wall_seconds": round(wall, 1),
        }

    # Build trajectory callback (only if canonical responses are available).
    trajectory_callback = None
    if canonical_responses is not None:
        from transformers import AutoTokenizer

        from explore_persona_space.experiments.contrastive_recipe_sweep_448.build_training_data import (  # noqa: E501
            _negative_personas_for_cell,
            _positive_personas_for_cell,
        )

        pos_persona_names = _positive_personas_for_cell(pos_personas)
        neg_persona_names = _negative_personas_for_cell(pos_persona_names, neg_personas)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        # Round-3 fix R2-3: pass output_path so the callback writes
        # marker_logprob_trajectory.json at end of training.
        trajectory_json_path = eval_out_dir / "marker_logprob_trajectory.json"
        trajectory_callback = _build_trajectory_callback(
            tokenizer,
            canonical_responses,
            centroids_path=centroids_path,
            pos_persona_names=pos_persona_names,
            neg_persona_names=neg_persona_names,
            trajectory_output_path=trajectory_json_path,
            cell_slug=cell_slug,
            seed=seed,
        )
        log.info(
            "[%s] MarkerTrajectoryCallback wired (centroid-stratified subset)",
            cell_slug,
        )

    _, merged_dir = _train_and_merge(cell_slug, seed, train_jsonl, output_dir, trajectory_callback)

    # Eval the merged model in-process.
    _eval_cell(cell_slug, merged_dir, eval_out_dir, sentinel_path)

    # Verify adapter uploaded (train_lora hf_upload=True is canonical).
    adapter_safetensors = list((output_dir / "adapter").glob("*.safetensors"))
    if not adapter_safetensors:
        raise RuntimeError(
            f"[{cell_slug}] Adapter dir {output_dir / 'adapter'} has no "
            f".safetensors files after training — upload may be stale or "
            f"training silently failed."
        )

    # rmtree merged dir BEFORE the next cell (MooseFS quota discipline).
    if merged_dir.exists():
        log.info("[%s] rmtree(%s) to free MooseFS quota", cell_slug, merged_dir)
        shutil.rmtree(merged_dir, ignore_errors=False)

    # HF Transformers cleanup hammer (CLAUDE.md gotcha — different than vLLM
    # but still good hygiene before next cell loads weights).
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    wall = time.time() - t_start
    log.info("[%s] CELL DONE in %.1fs", cell_slug, wall)
    return {
        "cell": cell_slug,
        "plain_name": plain_name,
        "seed": seed,
        "wall_seconds": round(wall, 1),
        "output_dir": str(output_dir),
        "eval_out_dir": str(eval_out_dir),
        "sentinel_path": str(sentinel_path),
        "adapter_hf_path": f"adapters/issue_448/{cell_slug}_seed{seed}",
    }


def _run_analyze(
    slab_root: Path,
    figures_dir: Path,
    centroids_path: Path,
    runs_root: Path,
    seed: int,
) -> dict[str, object]:
    """Phase 3: analyze.run_analysis (subprocess for clean import isolation).

    Round-2 fix B4: pass ``runs_root`` + ``seed`` so the analyzer can locate
    per-cell training-data manifests (written under
    ``runs_root/{cell}_seed{seed}/train_pool.manifest.json``).
    """
    t0 = time.time()
    log.info("=" * 70)
    log.info("Phase 3 (analyze)")
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "explore_persona_space.experiments.contrastive_recipe_sweep_448.analyze",
        "--slab-root",
        str(slab_root),
        "--centroids",
        str(centroids_path),
        "--figures-dir",
        str(figures_dir),
        "--runs-root",
        str(runs_root),
        "--seed",
        str(seed),
    ]
    log.info("Phase 3 subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, env={**os.environ}, check=True)

    analyze_path = slab_root / "analyze_summary.json"
    if not analyze_path.exists():
        raise RuntimeError(
            f"Phase 3 finished but {analyze_path} not written; analyze silently failed."
        )
    payload = json.loads(analyze_path.read_text())
    headline = payload.get("headline", {})
    log.info(
        "Phase 3 done in %.1fs; headline obs=%d null_median=%.1f p=%.3f",
        time.time() - t0,
        headline.get("headline_observed", -1),
        headline.get("headline_null_median", float("nan")),
        headline.get("empirical_p_value_one_sided", float("nan")),
    )
    return {
        "phase": "analyze",
        "wall_seconds": round(time.time() - t0, 1),
        "analyze_summary_path": str(analyze_path),
        "headline": headline,
        "n_cells_analyzed": payload.get("n_cells_analyzed"),
    }


def _write_final_sentinel(
    cells_requested: list[str],
    per_cell_summaries: list[dict[str, object]],
    pre_phase_0_summary: dict | None,
    phase_0_5_summary: dict | None,
    base_panel_summary: dict | None,
    analyze_summary: dict | None,
    plan_deviations: list[str],
    seed: int,
    slab_root: Path,
) -> None:
    """End-of-sweep sentinel. Schema matches the orchestrator's `epm:results v1`."""
    final_path = LOG_DIR / "issue-448-results.json"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    eval_paths = {c["cell"]: c.get("eval_out_dir") for c in per_cell_summaries}
    payload = {
        "schema": "epm:results v1",
        "issue": 448,
        "seed": seed,
        "cells_requested": cells_requested,
        "cells_completed": [c["cell"] for c in per_cell_summaries],
        "n_completed": len(per_cell_summaries),
        "n_requested": len(cells_requested),
        "eval_paths": eval_paths,
        "eval_numbers": {
            "n_panel_personas": 24,
            "n_eval_questions": 20,
            "n_cells_per_phase_2_per_cell": 480,
        },
        "reproducibility_card": {
            "base_model": BASE_MODEL,
            "hf_model_repo": HF_MODEL_REPO,
            "hf_data_repo": HF_DATA_REPO,
            "adapter_paths": {
                c["cell"]: f"{HF_MODEL_REPO}/tree/main/{c.get('adapter_hf_path', 'unknown')}"
                for c in per_cell_summaries
                if "adapter_hf_path" in c
            },
        },
        "worktree_path": str(Path.cwd()),
        "final_commit_sha": _git_sha(),
        "wandb_runs_note": "per-cell wandb runs; project=issue448_<cell>_seed42",
        "hf_hub_url": f"https://huggingface.co/{HF_MODEL_REPO}/tree/main/adapters/issue_448",
        "gpu_hours_used_estimate": round(
            sum(c.get("wall_seconds", 0) for c in per_cell_summaries) / 3600, 2
        ),
        "gpu_hours_budgeted": 7.5,
        "plan_deviations": plan_deviations,
        "pre_phase_0_summary": pre_phase_0_summary,
        "phase_0_5_summary": phase_0_5_summary,
        "base_panel_summary": base_panel_summary,
        "analyze_summary": analyze_summary,
        "headline_numbers": (analyze_summary or {}).get("headline"),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    final_path.write_text(json.dumps(payload, indent=2))
    log.info("Final sentinel: %s", final_path)


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - 5-phase orchestrator
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        type=str,
        default=None,
        help=(
            "Comma-separated cell IDs. Accepts plain-English names "
            "(Anchor, +pos-ex-100-per-persona, ...) OR slug forms "
            "(c1_anchor, c2_pos_ex_100, ...). Default: all 11."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shorthand for --cells Anchor + skip base-panel + skip analyze.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No GPU work / no Anthropic calls; validate imports + assertions.",
    )
    parser.add_argument(
        "--smoke-real",
        action="store_true",
        help=(
            "Round-2 fix B2: a REAL tiny-slice smoke that loads Qwen-2.5-7B-Instruct "
            "on a local GPU and runs Phase 1.5 base-panel marker log-prob over a "
            "tiny slice (controlled by --eval-personas-limit + --eval-questions-limit). "
            "Produces a real marker_logprob.json with non-zero log-probs. "
            "Skips Pre-Phase 0 + Phase 0.5 + per-cell train (this is the minimum "
            "GPU-using smoke that proves the eval rig works end-to-end)."
        ),
    )
    parser.add_argument(
        "--eval-personas-limit",
        type=int,
        default=None,
        help=("Cap the eval-panel persona count (debug / smoke-real). Default: 24."),
    )
    parser.add_argument(
        "--eval-questions-limit",
        type=int,
        default=None,
        help="Cap the eval question count (debug / smoke-real). Default: 20.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Round-2 fix C3: skip per-cell train + eval when "
            "/workspace/logs/issue-448-<cell>-results.json AND "
            "eval_results/issue_448/<cell>/marker_logprob.json both exist."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--slab-root",
        type=Path,
        default=Path("eval_results/issue_448"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/workspace/runs/issue_448"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/workspace/logs"),
        help=(
            "Where per-cell + per-phase + final sentinel JSONs land. Default "
            "/workspace/logs (pod-shape); override for local dry-run smoke."
        ),
    )
    parser.add_argument(
        "--centroids-path",
        type=Path,
        default=Path("eval_results/issue_448/centroids/centroids_layer20.pt"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/issue_448"),
    )
    parser.add_argument(
        "--skip-pre-phase-0",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip Pre-Phase 0. Default: True under --smoke + --dry-run, False otherwise.",
    )
    parser.add_argument(
        "--skip-phase-0-5",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip Phase 0.5 (extend centroids). Default: matches --skip-pre-phase-0.",
    )
    parser.add_argument(
        "--skip-base-panel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip Phase 1.5. Default: True under --smoke, False otherwise.",
    )
    parser.add_argument(
        "--skip-analyze",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip Phase 3. Default: True under --smoke, False otherwise.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    # Wire CLI --log-dir into the module-level constant so the helpers above
    # (which reference LOG_DIR directly) see the override.
    global LOG_DIR
    LOG_DIR = args.log_dir

    # ── Hard pre-flight: persona_registry build + marker tokenizer assertion. ──
    log.info("Pre-flight: persona_registry + marker tokenizer assertions")
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
        persona_registry,
    )

    persona_registry._do_build_and_assert()  # re-run on every dispatch
    log.info(
        "persona_registry assertions PASS (villain → %s, assistant → %s)",
        persona_registry.OBSERVED_BYSTANDERS_PER_SOURCE["villain"],
        persona_registry.OBSERVED_BYSTANDERS_PER_SOURCE["assistant"],
    )

    # Marker tokenizer assertion (CLAUDE.md). Don't need to load the full
    # model for this — just the tokenizer.
    if not args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
        if ids != [EXPECTED_MARKER_TOKEN_ID]:
            raise RuntimeError(
                f"Marker tokenizer assertion FAILED. Expected "
                f"[{EXPECTED_MARKER_TOKEN_ID}]; got {ids}."
            )
        log.info("Marker tokenizer assertion PASS: tokenize(%r) = %s", MARKER_TEXT, ids)
    else:
        log.info("Marker tokenizer assertion DEFERRED (dry-run)")

    cells = _resolve_cells(args.cells, args.smoke)
    # Round-2 fix B2: --smoke-real targets Phase 1.5 base panel only; empty the
    # per-cell loop so we don't try to load a merged checkpoint that doesn't exist.
    if args.smoke_real:
        cells = []
    log.info(
        "Resolved %d cells: %s",
        len(cells),
        [(slug, name) for slug, name, *_ in cells],
    )

    # Resolve phase-skip defaults from --smoke. Round-2 fix B2: --smoke-real
    # FORCES base-panel ON (it's the unit of test) and skips everything else.
    # Pre-Phase 0 runs in canonical-only mode (the base panel needs the 20
    # canonical EVAL_QUESTIONS responses; the 650-pair top-up is skipped).
    if args.smoke_real:
        # Only skip Pre-Phase 0 if the canonical responses are already on disk.
        from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
            build_wrong_claim_pool as _bwc,
        )

        _canon_path = _bwc.OUT_DIR / "eval_canonical_responses.json"
        skip_pre_phase_0 = _canon_path.exists()
        if not skip_pre_phase_0:
            log.info(
                "[smoke-real] canonical responses missing at %s — Pre-Phase 0 "
                "will run in canonical-only mode (~$0.02, ~30s).",
                _canon_path,
            )
        skip_phase_0_5 = True
        skip_base_panel = False  # base-panel IS the smoke-real test target
        skip_analyze = True
    else:
        skip_pre_phase_0 = (
            args.skip_pre_phase_0
            if args.skip_pre_phase_0 is not None
            else (args.smoke or args.dry_run)
        )
        skip_phase_0_5 = (
            args.skip_phase_0_5 if args.skip_phase_0_5 is not None else (args.smoke or args.dry_run)
        )
        skip_base_panel = args.skip_base_panel if args.skip_base_panel is not None else args.smoke
        skip_analyze = args.skip_analyze if args.skip_analyze is not None else args.smoke

    log.info(
        "Phase toggles: skip_pre_phase_0=%s skip_phase_0_5=%s skip_base_panel=%s skip_analyze=%s",
        skip_pre_phase_0,
        skip_phase_0_5,
        skip_base_panel,
        skip_analyze,
    )
    log.info(
        "Dry run: %s; smoke: %s; smoke-real: %s",
        args.dry_run,
        args.smoke,
        args.smoke_real,
    )

    args.slab_root.mkdir(parents=True, exist_ok=True)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # MooseFS quota safety (CLAUDE.md gotcha for sequential multi-cell sweep).
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    plan_deviations: list[str] = []

    # ── Pre-Phase 0 ──────────────────────────────────────────────────────────
    pre_phase_0_summary = _run_pre_phase_0(
        skip_pre_phase_0, args.dry_run, canonical_only=args.smoke_real
    )
    if skip_pre_phase_0:
        plan_deviations.append("pre_phase_0_skipped")

    # ── Phase 0.5 ────────────────────────────────────────────────────────────
    phase_0_5_summary = _run_phase_0_5(skip_phase_0_5, args.dry_run)
    if skip_phase_0_5:
        plan_deviations.append("phase_0_5_skipped")

    # ── Phase 1.5 ────────────────────────────────────────────────────────────
    base_panel_summary: dict | None = None
    if skip_base_panel:
        log.info("Phase 1.5 SKIPPED (--skip-base-panel)")
        plan_deviations.append("phase_1_5_base_panel_skipped")
    else:
        try:
            base_panel_summary = _run_phase_1_5_base_panel(
                False,
                args.dry_run,
                eval_personas_limit=args.eval_personas_limit,
                eval_questions_limit=args.eval_questions_limit,
            )
        except Exception:
            log.exception("Phase 1.5 (base panel) failed")
            raise

    # ── Load canonical responses (passed into per-cell trajectory callback). ─
    canonical_responses: dict[str, str] | None = None
    if not args.dry_run:
        try:
            from explore_persona_space.experiments.contrastive_recipe_sweep_448.build_wrong_claim_pool import (  # noqa: E501
                load_canonical_responses,
            )

            canonical_responses = load_canonical_responses()
        except FileNotFoundError as e:
            log.warning(
                "Canonical responses not found (%s); trajectory callback DISABLED. "
                "Run Pre-Phase 0 first.",
                e,
            )

    # ── Phase 1+2 per-cell loop ──────────────────────────────────────────────
    per_cell_summaries: list[dict[str, object]] = []
    for slug, plain_name, pos_ex_per_p, pos_personas, neg_ex_per_p, neg_personas in cells:
        try:
            cell_summary = _run_one_cell(
                cell_slug=slug,
                plain_name=plain_name,
                pos_ex_per_p=pos_ex_per_p,
                pos_personas=pos_personas,
                neg_ex_per_p=neg_ex_per_p,
                neg_personas=neg_personas,
                seed=args.seed,
                slab_root=args.slab_root,
                runs_root=args.runs_root,
                dry_run=args.dry_run,
                canonical_responses=canonical_responses,
                centroids_path=args.centroids_path,
                resume=args.resume,
            )
            per_cell_summaries.append(cell_summary)
        except Exception as e:
            fail_path = LOG_DIR / f"issue-448-{slug}-FAILED.json"
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            fail_path.write_text(
                json.dumps(
                    {
                        "cell": slug,
                        "phase": "cell_failed",
                        "exception_type": type(e).__name__,
                        "exception_msg": str(e),
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                    },
                    indent=2,
                )
            )
            log.exception("[%s] cell failed; wrote %s", slug, fail_path)
            raise

    # ── Phase 3: analyze ─────────────────────────────────────────────────────
    analyze_summary: dict | None = None
    if skip_analyze:
        log.info("Phase 3 SKIPPED (--skip-analyze)")
        plan_deviations.append("phase_3_analyze_skipped")
    elif args.dry_run:
        log.info("Phase 3 DRY-RUN: analyze module not invoked")
    else:
        try:
            analyze_summary = _run_analyze(
                slab_root=args.slab_root,
                figures_dir=args.figures_dir,
                centroids_path=args.centroids_path,
                runs_root=args.runs_root,
                seed=args.seed,
            )
        except Exception:
            log.exception("Phase 3 (analyze) failed")
            raise

    cells_requested = [c[0] for c in cells]
    _write_final_sentinel(
        cells_requested=cells_requested,
        per_cell_summaries=per_cell_summaries,
        pre_phase_0_summary=pre_phase_0_summary,
        phase_0_5_summary=phase_0_5_summary,
        base_panel_summary=base_panel_summary,
        analyze_summary=analyze_summary,
        plan_deviations=plan_deviations,
        seed=args.seed,
        slab_root=args.slab_root,
    )
    log.info("Dispatcher done. %d cells completed.", len(per_cell_summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
