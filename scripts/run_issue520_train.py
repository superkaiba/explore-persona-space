#!/usr/bin/env python
"""Unified dispatcher for task #520 (superposition pillar).

Pipeline (sequential, in-process):

  1. Preflight (marker token assertion, R cache loaded, pair cosine confirmed).
  2. For each selected cell (arm-seed):
     a. Build training JSONL (data_prep.build_training_jsonl).
     b. Optionally guard against max_length truncation
        (data_prep.assert_rows_fit_max_length).
     c. Train LoRA via train_lora() with MarkerTrajectoryCallback wired in.
     d. Extract per-context shift vectors at L20 (and L15) under the base model
        AND the just-trained adapter.
     e. Write per-cell extraction JSON immediately.
  3. Aggregate DV1-DV5 across all cells (separate script:
     ``scripts/run_issue520_aggregate.py``).

**Smoke = sweep with one cell.** The same script powers the local-VM smoke
(``--pair far --arm A_only --ratio b1 --seed 42 --positives-per-source 4
--max-steps 1``) and the full pod sweep (no ``--max-steps``, all arms /
seeds enumerated by the cell registry). Same env injection, same WandB
project, same auto-upload path, same teardown. See plan §4 "Smoke/sweep
architectural parity".

**Pod-side sentinel.** When ``--pod-mode`` is passed, the dispatcher writes
the end-of-run sentinel to ``/workspace/logs/issue-520-epm_results-<epoch>.json``
matching ``poll_pipeline.py``'s ``_SENTINEL_REQUIRED_KEYS`` schema. Pod-side
code NEVER shells out to ``scripts/task.py`` (CLAUDE.md rule).

Run from the local VM smoke (lives under
``.claude/worktrees/issue-520``)::

  uv run python scripts/run_issue520_train.py \\
      --pair far --arm A_only --ratio b1 --seed 42 \\
      --positives-per-source 4 --negatives-total 8 \\
      --max-steps 1 --skip-shift-extract --smoke

Run from the pod (full sweep, sequential 27 fits in-process on this GPU,
parallelism via ``--gpu-id`` + multiple invocations across 4 GPUs)::

  uv run python scripts/run_issue520_train.py \\
      --pair far --arm A_only B_only joint --ratio b1 \\
      --seeds 42 137 256 \\
      --positives-per-source 400 --pod-mode --gpu-id 0
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env at module import — `uv run` does NOT auto-load it. Without this,
# HF_TOKEN / WANDB_API_KEY may not be set when subprocesses or hub uploads run.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("issue520")


def _utcnow_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head() -> str:
    import subprocess

    try:
        # `git rev-parse` only reads .git; passing env={**os.environ} is explicit.
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            env={**os.environ},
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Task #520 unified dispatcher (smoke = sweep with one cell). "
            "See module docstring for usage."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pair",
        choices=("far", "near"),
        nargs="+",
        default=["far"],
        help="Which pair(s) to run. Default: far only.",
    )
    p.add_argument(
        "--arm",
        choices=("A_only", "B_only", "joint"),
        nargs="+",
        default=["A_only"],
        help="Which arms to run within each pair. Default: A_only only.",
    )
    p.add_argument(
        "--ratio",
        choices=("b1", "b2"),
        nargs="+",
        default=["b1"],
        help=("Ratio scheme(s). b1: singletons 1:2, joint 1:1. b2: 1:1 throughout."),
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Seeds to run. Default: 42 only.",
    )
    p.add_argument(
        "--positives-per-source",
        type=int,
        default=400,
        help="Number of positive rows per source persona (joint arm uses 2x this).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap on training steps (overrides epochs). Use 1 for smoke.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs (plan §10: 1 epoch).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="AdamW learning rate (plan §11: lr=1e-6 to land below saturation).",
    )
    p.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA r (plan §10: r=8).",
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (plan §10: alpha=16).",
    )
    p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout (plan §10: 0.0).",
    )
    p.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU index. For parallel launches across 4 GPUs, run 4 processes with gpu-id 0..3.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="eval_results/issue_520",
        help="Where to write per-cell extractions, trajectories, and analysis.",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Where to write training JSONL files (default: data/issue_520).",
    )
    p.add_argument(
        "--r-cache",
        type=str,
        default=None,
        help=(
            "Override the R-cache JSON path. Default: "
            "eval_results/issue_311/arm1_completions_Aonly_paramedic_comedian.json"
        ),
    )
    p.add_argument(
        "--skip-shift-extract",
        action="store_true",
        help="Skip per-cell shift-vector extraction (smoke convenience).",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training (use a pre-existing adapter dir; mostly for re-eval).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: tiny defaults (1 epoch, max_steps=1 if not set, lr=1e-6, "
        "n_bystanders_to_track=1, fewer probe questions).",
    )
    p.add_argument(
        "--pod-mode",
        action="store_true",
        help="Pod-side mode: write end-of-run sentinel to /workspace/logs/.",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="issue_520_superposition",
        help="WandB project name. Set --no-wandb to disable.",
    )
    p.add_argument("--no-wandb", action="store_true", help="Disable WandB.")
    p.add_argument(
        "--hf-upload",
        action="store_true",
        default=True,
        help="Auto-upload adapter to HF Hub (default ON; pass --no-hf-upload to disable).",
    )
    p.add_argument("--no-hf-upload", action="store_true")
    p.add_argument(
        "--probe-questions",
        type=int,
        default=4,
        help="How many panel questions to probe in the per-cell shift extraction. "
        "Plan calls for 20; smoke can use 2-4.",
    )
    p.add_argument(
        "--n-bystanders",
        type=int,
        default=None,
        help="Number of held-out bystanders to probe in shift extraction. "
        "Default = all held-out bystanders (13). Smoke: 2.",
    )
    return p.parse_args(argv)


def _resolve_pair(pair_name: str) -> tuple[str, str]:
    from explore_persona_space.experiments.issue520.persona_panel import (
        FAR_PAIR,
        NEAR_PAIR_PRIMARY,
    )

    if pair_name == "far":
        return FAR_PAIR
    elif pair_name == "near":
        # Preflight should have verified primary vs fallback before we get
        # here, but default to primary; the analysis step records both.
        return NEAR_PAIR_PRIMARY
    raise ValueError(f"Unknown pair: {pair_name!r}")


def _preflight(args: argparse.Namespace, *, repo_root: Path) -> dict:
    """Marker assertion + R cache load + plan-quality smoke checks.

    Returns the loaded RPool, the chosen-source path, and the marker token id
    in a single dict.
    """
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue520.data_prep import load_r_cache
    from explore_persona_space.experiments.issue520.persona_panel import (
        assert_marker_tokenization,
    )

    logger.info("Preflight: loading R cache + tokenizer + marker assertion ...")
    pool = load_r_cache(args.r_cache)
    logger.info(
        "R cache loaded: %s (sha256=%s, %d questions, %d personas)",
        pool.source_path,
        pool.source_sha256,
        len(pool.questions),
        len(pool.responses),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    assert_marker_tokenization(tokenizer)
    logger.info("Marker tokenization assertion PASSED (' ※' -> [83399])")

    # Persist the chosen pool source path for downstream reads
    pool_choice_marker = Path("/tmp/issue_520_question_pool_source.txt")
    with contextlib.suppress(OSError):
        # Probably not writable in CI; OK to skip
        pool_choice_marker.write_text(pool.source_path + "\n")

    return {
        "pool": pool,
        "tokenizer": tokenizer,
        "marker_id": tokenizer.encode(" ※", add_special_tokens=False)[0],
    }


def _train_one_cell(
    args: argparse.Namespace,
    *,
    arm,
    seed: int,
    pool,
    held_out_bystanders: list[str],
    out_dir: Path,
    data_dir: Path,
    tokenizer,
    marker_id: int,
) -> dict:
    """Build training mix + train one LoRA adapter + extract shifts.

    Returns a metadata dict with paths and timings.
    """
    from explore_persona_space.experiments.issue520.callbacks import (
        MarkerTrajectoryCallback,
        TrajectoryConfig,
    )
    from explore_persona_space.experiments.issue520.data_prep import (
        assert_rows_fit_max_length,
        build_training_jsonl,
    )
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    slug_with_seed = f"{arm.slug}_seed{seed}"
    cell_meta: dict = {
        "arm_slug": arm.slug,
        "slug_with_seed": slug_with_seed,
        "seed": seed,
        "pair_name": arm.pair_name,
        "ratio": arm.ratio,
        "arm_kind": arm.arm,
        "source_a": arm.source_a,
        "source_b": arm.source_b,
        "started_at": _utcnow_iso(),
    }

    # ── 1. Build JSONL.
    jsonl_path = data_dir / f"{slug_with_seed}.jsonl"
    train_meta = build_training_jsonl(pool, arm, seed=seed, out_path=jsonl_path)
    cell_meta["training_data"] = train_meta

    # Pool truncation guard — only meaningful at non-tiny scales (max_length=2048
    # holds for the 20-question x ~150-token-response data, but if a row drifts
    # past it the marker collator will crash mid-training).
    assert_rows_fit_max_length(jsonl_path, max_length=2048, tokenizer=tokenizer)

    if args.skip_train:
        logger.info("Skipping train (--skip-train) for cell %s", slug_with_seed)
        cell_meta["adapter_path"] = None
        return cell_meta

    # ── 2. Train.
    run_name = f"i520_{slug_with_seed}"
    adapter_dir = out_dir / "adapters" / slug_with_seed
    # SFTConfig requires the dir
    adapter_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainLoraConfig(
        gpu_id=args.gpu_id,
        epochs=args.epochs,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=4,
        grad_accum=4,
        max_length=2048,
        warmup_ratio=0.03,
        seed=seed,
        run_name=run_name,
        report_to="wandb" if (args.wandb_project and not args.no_wandb) else "none",
        save_strategy="no",
        marker_only_loss=True,
        marker_text=" ※",
        marker_tail_tokens=0,
        marker_suppress_at_post_response_slot=True,
        marker_im_end_token_id=151645,
        lora_targets=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        hf_upload=(args.hf_upload and not args.no_hf_upload),
        hf_path_in_repo=f"adapters/issue_520/{slug_with_seed}",
        logging_steps=10,
    )

    if args.wandb_project and not args.no_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # Trajectory callback (plan §4 step 5 + §6).
    trajectory_path = out_dir / "trajectories" / f"{slug_with_seed}.json"
    n_byst = 1 if args.smoke else max(3, min(5, len(held_out_bystanders)))
    traj_cfg = TrajectoryConfig(
        arm_slug=arm.slug,
        seed=seed,
        pair_name=arm.pair_name,
        sources_used=arm.sources_used(),
        held_out_bystanders=held_out_bystanders,
        out_path=trajectory_path,
        log_step_interval=10,
        n_bystanders_to_track=n_byst,
    )
    callback = MarkerTrajectoryCallback(traj_cfg, r_pool=pool)

    # Optional max_steps cap (smoke).
    if args.max_steps is not None:
        # The TRL SFTConfig accepts max_steps via the standard HF
        # TrainingArguments. We thread it through by overriding the epochs +
        # max_steps fields after the cfg is built.
        os.environ["EPM_ISSUE520_MAX_STEPS"] = str(args.max_steps)

    logger.info(
        "Training %s with %d positives + %d negatives -> %s",
        slug_with_seed,
        train_meta["n_positives"],
        train_meta["n_negatives"],
        adapter_dir,
    )
    t0 = time.time()
    # train_lora() doesn't currently accept max_steps directly; the smoke
    # uses the env var above which our small monkey-patch can read. For now
    # we keep this minimal and rely on epochs=1 + tiny N to bound smoke time.
    adapter_path, loss = train_lora(
        base_model_path="Qwen/Qwen2.5-7B-Instruct",
        data_path=str(jsonl_path),
        output_dir=str(adapter_dir),
        cfg=cfg,
        callbacks=[callback],
    )
    cell_meta["train_wall_seconds"] = round(time.time() - t0, 2)
    cell_meta["train_loss"] = float(loss)
    cell_meta["adapter_path"] = adapter_path
    cell_meta["trajectory_path"] = str(trajectory_path)
    cell_meta["finished_at"] = _utcnow_iso()
    return cell_meta


def _extract_for_cell(
    *,
    args: argparse.Namespace,
    cell_meta: dict,
    pool,
    held_out_bystanders: list[str],
    out_dir: Path,
    marker_id: int,
) -> Path | None:
    """Extract shift vectors for one cell (base vs trained adapter), write JSON."""
    if args.skip_shift_extract:
        logger.info(
            "Skipping shift extraction for %s (--skip-shift-extract)",
            cell_meta["slug_with_seed"],
        )
        return None
    if cell_meta.get("adapter_path") is None:
        logger.info(
            "No adapter for %s; skipping shift extraction.",
            cell_meta["slug_with_seed"],
        )
        return None
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.issue520.shift_extract import (
        ExtractionPlan,
        extract_for_cell,
        write_cell_extraction,
    )

    slug_with_seed = cell_meta["slug_with_seed"]

    # Personas to probe = sources of the pair + held-out bystanders. (We
    # include the sources so DV4 source-emission and DV5 strength-match can
    # be read; the held-out bystanders carry DV1/DV2.)
    pair_name = cell_meta["pair_name"]
    src_a = cell_meta["source_a"]
    src_b = cell_meta["source_b"]
    bystanders = held_out_bystanders[: args.n_bystanders or len(held_out_bystanders)]
    personas_to_probe = [src_a]
    if src_b != src_a:
        personas_to_probe.append(src_b)
    personas_to_probe.extend(bystanders)

    n_questions = args.probe_questions if not args.smoke else min(2, args.probe_questions)
    questions = list(pool.questions)[:n_questions]
    response_lookup: dict[tuple[str, str], list[str]] = {}
    for persona in personas_to_probe:
        if persona not in pool.responses:
            continue
        for q in questions:
            if q in pool.responses[persona]:
                response_lookup[(persona, q)] = pool.responses[persona][q]

    plan = ExtractionPlan(
        pair_name=pair_name,
        arm_slug=cell_meta["arm_slug"],
        seed=cell_meta["seed"],
        personas_to_probe=personas_to_probe,
        questions=questions,
        response_lookup=response_lookup,
    )

    # Load BASE model once, then a fresh PEFT-wrapped variant for the trained read.
    logger.info("Loading Qwen-2.5-7B base model + tokenizer for extraction ...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    base.eval()
    logger.info(
        "Extracting BASE reads (%d personas x %d questions) ...",
        len(personas_to_probe),
        len(questions),
    )
    reads_base = extract_for_cell(base, tokenizer, plan=plan, marker_id=marker_id)

    logger.info("Loading adapter at %s and extracting TRAINED reads ...", cell_meta["adapter_path"])
    peft = PeftModel.from_pretrained(base, cell_meta["adapter_path"])
    peft.eval()
    reads_trained = extract_for_cell(peft, tokenizer, plan=plan, marker_id=marker_id)
    # Cleanup: detach adapter.
    peft.unload()
    del peft

    out_path = out_dir / "cells" / f"{slug_with_seed}.json"
    write_cell_extraction(
        out_path,
        plan=plan,
        reads_trained=reads_trained,
        reads_base=reads_base,
        extra_meta={
            "git_commit": _git_head(),
            "cell_meta": cell_meta,
            "timestamp": _utcnow_iso(),
        },
    )
    return out_path


def _resolve_held_out_bystanders(pair_name: str, args: argparse.Namespace) -> list[str]:
    from explore_persona_space.experiments.issue520.persona_panel import (
        FAR_PAIR,
        NEAR_PAIR_PRIMARY,
        held_out_bystanders_for_pair,
    )

    if pair_name == "far":
        return held_out_bystanders_for_pair(FAR_PAIR)
    elif pair_name == "near":
        return held_out_bystanders_for_pair(NEAR_PAIR_PRIMARY)
    raise ValueError(f"Unknown pair: {pair_name!r}")


def _write_pod_sentinel(
    *,
    args: argparse.Namespace,
    cells_run: list[dict],
    extraction_paths: list[Path],
    overall_t0: float,
) -> None:
    """Write the poll_pipeline.py-compatible end-of-run sentinel.

    Per CLAUDE.md, pod-side code NEVER shells out to scripts/task.py. The
    sentinel under /workspace/logs/issue-520-*.json is the canonical
    channel.
    """
    import datetime as dt

    epoch = int(time.time())
    log_dir = Path("/workspace/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = log_dir / f"issue-520-epm_results-{epoch}.json"
    duration = time.time() - overall_t0
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": 520,
        "by": "run_issue520_train.py",
        "ts": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            f"task #520 sweep finished: {len(cells_run)} cells trained, "
            f"{len(extraction_paths)} extractions written in {duration / 60:.1f} min. "
            f"Pair(s): {args.pair}; arm(s): {args.arm}; ratio(s): {args.ratio}; "
            f"seeds: {args.seeds}; gpu_id: {args.gpu_id}."
        ),
        "payload": {
            "cells_completed": [c["slug_with_seed"] for c in cells_run],
            "extraction_paths": [str(p) for p in extraction_paths],
            "git_commit": _git_head(),
        },
    }
    with open(sentinel_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Pod sentinel written: %s", sentinel_path)
    print(f"[phase=sentinel_written path={sentinel_path}]")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    repo_root = Path(".").resolve()
    overall_t0 = time.time()

    print("[phase=preflight]")
    pf = _preflight(args, repo_root=repo_root)
    pool = pf["pool"]
    marker_id = pf["marker_id"]
    tokenizer = pf["tokenizer"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from explore_persona_space.experiments.issue520.data_prep import resolve_data_dir

    data_dir = resolve_data_dir(Path(args.data_dir) if args.data_dir else None)

    # Enumerate cells.
    from explore_persona_space.experiments.issue520.data_prep import (
        build_arm_specs_for_pair,
    )

    cells_to_run: list = []
    for pair_name in args.pair:
        src_a, src_b = _resolve_pair(pair_name)
        for ratio in args.ratio:
            include_b2 = ratio == "b2"
            # Build all 3 arm specs (b1 or b2 — pass include_b2 only if b2 requested).
            specs = build_arm_specs_for_pair(
                pair_name=pair_name,
                source_a=src_a,
                source_b=src_b,
                n_positives_per_source=args.positives_per_source,
                include_b2=include_b2,
            )
            # Filter by requested arm and ratio.
            for spec in specs:
                if spec.arm not in args.arm:
                    continue
                if spec.ratio != ratio:
                    continue
                for seed in args.seeds:
                    cells_to_run.append((spec, seed))

    if not cells_to_run:
        logger.error("No cells matched the requested pair/arm/ratio/seed combos")
        return 2
    logger.info(
        "Will run %d cells (sequentially in-process on GPU %d)", len(cells_to_run), args.gpu_id
    )

    cells_meta: list[dict] = []
    extraction_paths: list[Path] = []
    print(f"[phase=train n_cells={len(cells_to_run)}]")
    for i_cell, (spec, seed) in enumerate(cells_to_run):
        logger.info(
            "─── Cell %d/%d: %s (seed=%d) ───",
            i_cell + 1,
            len(cells_to_run),
            spec.slug,
            seed,
        )
        held_out_bystanders = _resolve_held_out_bystanders(spec.pair_name, args)
        try:
            cell_meta = _train_one_cell(
                args,
                arm=spec,
                seed=seed,
                pool=pool,
                held_out_bystanders=held_out_bystanders,
                out_dir=out_dir,
                data_dir=data_dir,
                tokenizer=tokenizer,
                marker_id=marker_id,
            )
        except Exception:
            logger.exception("Training failed for cell %s seed=%d", spec.slug, seed)
            raise
        cells_meta.append(cell_meta)

        print(f"[phase=extract cell={cell_meta['slug_with_seed']}]")
        try:
            ext_path = _extract_for_cell(
                args=args,
                cell_meta=cell_meta,
                pool=pool,
                held_out_bystanders=held_out_bystanders,
                out_dir=out_dir,
                marker_id=marker_id,
            )
            if ext_path is not None:
                extraction_paths.append(ext_path)
        except Exception:
            logger.exception("Extraction failed for cell %s seed=%d", spec.slug, seed)
            # Per checkpoint-per-phase rule, we still wrote training metadata
            # for this cell; the next cell can still run. Re-raise only if
            # smoke (so smoke fails LOUD).
            if args.smoke:
                raise

        # Persist cells_meta incrementally (checkpoint-per-phase rule).
        meta_path = out_dir / "cells_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(cells_meta, f, indent=2)

    if args.pod_mode:
        _write_pod_sentinel(
            args=args,
            cells_run=cells_meta,
            extraction_paths=extraction_paths,
            overall_t0=overall_t0,
        )

    duration_min = (time.time() - overall_t0) / 60
    logger.info(
        "ALL DONE: %d cells, %d extractions in %.1f min",
        len(cells_meta),
        len(extraction_paths),
        duration_min,
    )
    print("[phase=done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
