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
      --pair far --arm A_only --ratio b1 --seeds 42 \\
      --positives-per-source 4 \\
      --max-steps 1 --skip-shift-extract --smoke \\
      --r-cache eval_results/issue_311/arm1_completions_Aonly_paramedic_comedian.json \\
      --allow-contaminated-r-cache-for-smoke

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
        default=20,
        help="How many panel questions to probe in the per-cell shift extraction. "
        "Plan §4 calls for 20-prompt per-persona stability; smoke clamps to 2.",
    )
    p.add_argument(
        "--allow-contaminated-r-cache-for-smoke",
        action="store_true",
        help=(
            "Smoke-only convenience flag: allow the #311 trained-adapter R "
            "completion pool (arm1_completions_*.json) as the R cache, even "
            "though ~21%% of its rows carry a stray [ZLT]. NEVER pass this "
            "on a pod sweep — DV1-DV5 require a clean BASE-model R pool "
            "regenerated via gen_r_persona.generate_r_cache(...)."
        ),
    )
    p.add_argument(
        "--n-bystanders",
        type=int,
        default=None,
        help="Number of held-out bystanders to probe in shift extraction. "
        "Default = all held-out bystanders (13). Smoke: 2.",
    )
    p.add_argument(
        "--skip-cosine-preflight",
        action="store_true",
        help=(
            "Skip the L20 centered-cosine computation in _pair_cosine_preflight "
            "(still writes pair_selection.json with the gradient-conflict "
            "annotation, just with cosine_* = None). Use for local-VM smoke "
            "where the base model isn't being loaded. Implied by --smoke."
        ),
    )
    return p.parse_args(argv)


def _resolve_pair(
    pair_name: str,
    *,
    near_pair_override: tuple[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve a pair name to its (persona_a, persona_b) tuple.

    ``near_pair_override`` lets the dispatcher swap NEAR_PAIR_PRIMARY for
    NEAR_PAIR_FALLBACK after the L20 cosine preflight (CONCERN B round 3).
    When None, the panel default (NEAR_PAIR_PRIMARY) is returned.
    """
    from explore_persona_space.experiments.issue520.persona_panel import (
        FAR_PAIR,
        NEAR_PAIR_PRIMARY,
    )

    if pair_name == "far":
        return FAR_PAIR
    elif pair_name == "near":
        return near_pair_override if near_pair_override is not None else NEAR_PAIR_PRIMARY
    raise ValueError(f"Unknown pair: {pair_name!r}")


def _preflight(args: argparse.Namespace) -> dict:
    """Marker assertion + R cache load + plan-quality smoke checks.

    Returns the loaded RPool, the chosen-source path, and the marker token id
    in a single dict.

    Enforces (per round-2 review):

    - **R-cache contract**: non-smoke runs MUST pass ``--r-cache <path>``
      pointing at a BASE-model-generated pool (regenerate via
      ``gen_r_persona.generate_r_cache``). The #311 ``arm1_completions``
      trained-adapter pool is rejected by ``load_r_cache`` unless
      ``--allow-contaminated-r-cache-for-smoke`` is also passed AND
      ``--smoke`` is set.
    - **Source/negative gradient-conflict guard**: every source persona in
      the selected pairs is asserted NOT in ``NEGATIVE_PERSONAS``.
    """
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue520.data_prep import load_r_cache
    from explore_persona_space.experiments.issue520.persona_panel import (
        NEGATIVE_PERSONAS,
        assert_marker_tokenization,
    )

    # R-cache contract: non-smoke runs MUST provide --r-cache explicitly.
    if not args.smoke and not args.r_cache:
        raise SystemExit(
            "ERROR: --r-cache <path> is required for non-smoke runs.\n"
            "The activation-shift baseline (DV1/DV2) and the trajectory probe "
            "(DV3/DV4) read against R_persona(q); R MUST be a BASE-model "
            "response, not a trained-adapter completion (the #311 "
            "arm1_completions_* JSONs are TRAINED outputs with ~21% rows "
            "contaminated with [ZLT]).\n\n"
            "Regenerate a clean R pool first:\n"
            "  uv run python -c 'from explore_persona_space.experiments."
            "issue520.gen_r_persona import generate_r_cache; "
            'generate_r_cache(base_model="Qwen/Qwen2.5-7B-Instruct", '
            "personas=[...], questions=[...], n_samples_per_q=20, "
            'out_path="eval_results/issue_520/r_cache_base.json")\'\n\n'
            "Then re-run with --r-cache eval_results/issue_520/r_cache_base.json. "
            "Pass --allow-contaminated-r-cache-for-smoke ONLY for the local-VM "
            "smoke (NEVER for a pod sweep)."
        )

    # Resolve the contamination policy. Allow the contaminated cache ONLY
    # when BOTH --smoke and --allow-contaminated-r-cache-for-smoke are set.
    allow_contaminated = bool(args.smoke and args.allow_contaminated_r_cache_for_smoke)
    if args.allow_contaminated_r_cache_for_smoke and not args.smoke:
        raise SystemExit(
            "ERROR: --allow-contaminated-r-cache-for-smoke requires --smoke. "
            "This flag is a smoke-only convenience; a pod sweep MUST use a "
            "clean base-model R pool."
        )

    logger.info(
        "Preflight: loading R cache + tokenizer + marker assertion (allow_contaminated=%s) ...",
        allow_contaminated,
    )
    pool = load_r_cache(args.r_cache, allow_contaminated=allow_contaminated)
    logger.info(
        "R cache loaded: %s (sha256=%s, %d questions, %d personas, contaminated=%s)",
        pool.source_path,
        pool.source_sha256,
        len(pool.questions),
        len(pool.responses),
        pool.contaminated_with_zlt,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    assert_marker_tokenization(tokenizer)
    logger.info("Marker tokenization assertion PASSED (' ※' -> [83399])")

    # Source/negative gradient-conflict guard. Every source persona in the
    # selected pairs must NOT be in NEGATIVE_PERSONAS — otherwise the same
    # persona's responses end up both as positives (with marker) and
    # negatives (no marker), a silent gradient conflict.
    neg_set = set(NEGATIVE_PERSONAS)
    for pair_name in args.pair:
        src_a, src_b = _resolve_pair(pair_name)
        if src_a in neg_set or src_b in neg_set:
            raise SystemExit(
                f"ERROR: pair {pair_name!r} = ({src_a}, {src_b}) overlaps "
                f"with NEGATIVE_PERSONAS={sorted(neg_set)}. Drop the persona "
                "from the negative panel OR pick a different pair."
            )

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
        max_steps=(args.max_steps if args.max_steps is not None else -1),
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

    logger.info(
        "Training %s with %d positives + %d negatives -> %s (max_steps=%s)",
        slug_with_seed,
        train_meta["n_positives"],
        train_meta["n_negatives"],
        adapter_dir,
        cfg.max_steps,
    )
    t0 = time.time()
    # max_steps is now wired through TrainLoraConfig.max_steps -> SFTConfig
    # max_steps -> HF TrainingArguments.max_steps (>0 overrides epochs).
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


def _assert_r_cache_coverage(
    *,
    pool,
    personas_to_probe: list[str],
    questions: list[str],
    slug_with_seed: str,
) -> dict[tuple[str, str], list[str]]:
    """Fail loud if the R cache is missing any required (persona, question) key.

    CLAUDE.md "Fail fast — never hide failures": a clean but INCOMPLETE R
    cache silently drops a source / negative / bystander column from the
    extraction's response_lookup, which then propagates into ``analysis.json``
    as a missing row in the bystander-leakage / source-self DV tables. The
    operator cannot recover post-hoc because the input itself was the
    missing piece — re-running the analysis wouldn't fix it.

    Three checks (each fails LOUD on miss):

    1. Every required persona is keyed in ``pool.responses``.
    2. Every required persona has a non-empty per-question dict.
    3. Every (persona, question) probe target has at least one base-model
       response string to teacher-force against.

    Returns the fully-populated ``response_lookup`` for the caller.
    """
    missing_personas = [p for p in personas_to_probe if p not in pool.responses]
    if missing_personas:
        raise RuntimeError(
            f"R cache missing required personas for cell {slug_with_seed!r}: "
            f"{sorted(missing_personas)!r}. Available in pool: "
            f"{sorted(pool.responses.keys())!r}. Regenerate R cache via "
            "scripts/gen_r_persona.py with the full panel (source pair + 4 "
            "NEGATIVE_PERSONAS + held-out bystanders for the pair)."
        )
    empty_personas = [p for p in personas_to_probe if not pool.responses.get(p)]
    if empty_personas:
        raise RuntimeError(
            f"R cache has EMPTY response dict for personas "
            f"{sorted(empty_personas)!r} (cell {slug_with_seed!r}). "
            "Regenerate R cache."
        )
    missing_pq: list[tuple[str, str]] = []
    response_lookup: dict[tuple[str, str], list[str]] = {}
    for persona in personas_to_probe:
        for q in questions:
            resp_list = pool.responses[persona].get(q)
            if not resp_list:
                missing_pq.append((persona, q))
            else:
                response_lookup[(persona, q)] = resp_list
    if missing_pq:
        raise RuntimeError(
            f"R cache missing or empty responses for {len(missing_pq)} "
            f"(persona, question) pairs in cell {slug_with_seed!r}. "
            f"First few: {missing_pq[:5]!r}. Regenerate R cache with the "
            "full set of probe questions."
        )
    return response_lookup


def _extract_for_cell(
    *,
    args: argparse.Namespace,
    cell_meta: dict,
    pool,
    held_out_bystanders: list[str],
    out_dir: Path,
    marker_id: int,
    base=None,
    tokenizer=None,
) -> Path | None:
    """Extract shift vectors for one cell (base vs trained adapter), write JSON.

    Args:
        base: Hoisted base model (loaded once in ``main()`` and shared across
            cells to avoid the ~7 GB x 27-cells reload penalty). Required.
        tokenizer: Hoisted tokenizer. Required.
    """
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
    if base is None or tokenizer is None:
        raise RuntimeError(
            "_extract_for_cell requires hoisted base + tokenizer (load once in main())."
        )

    # MUST-FIX #1 defense-in-depth: refuse to extract against a contaminated
    # R pool. The pair-read positives strip [ZLT] on read, but the
    # extraction probe indexes ``pool.responses`` directly — a [ZLT]-bearing
    # R inflates DV4 source-emission and confounds the activation-shift
    # baseline. Allowed only if the caller explicitly opted into the smoke
    # convenience (--allow-contaminated-r-cache-for-smoke), which already
    # implies --smoke and is gated on --skip-shift-extract typically.
    if pool.contaminated_with_zlt and not getattr(
        args, "allow_contaminated_r_cache_for_smoke", False
    ):
        raise RuntimeError(
            f"Refusing to extract against contaminated R pool ({pool.source_path}). "
            "Regenerate base-model R via gen_r_persona.generate_r_cache(...) "
            "and re-run with --r-cache <clean.json>."
        )

    import torch
    from peft import PeftModel

    from explore_persona_space.experiments.issue520.persona_panel import NEGATIVE_PERSONAS
    from explore_persona_space.experiments.issue520.shift_extract import (
        ExtractionPlan,
        extract_for_cell,
        write_cell_extraction,
    )

    slug_with_seed = cell_meta["slug_with_seed"]

    # MUST-FIX #3: personas_to_probe MUST cover:
    #   - source persona(s) of the arm — for DV4 source-self emission +
    #     DV5 strength-match;
    #   - the 4 negative personas (NEGATIVE_PERSONAS) — to measure the
    #     localization gradient AND the safety target (default-assistant
    #     leakage; "helpful_assistant" is the default-assistant substitute
    #     per persona_panel docstring);
    #   - held-out bystanders — for DV1/DV2 additivity on truly held-out
    #     contexts.
    pair_name = cell_meta["pair_name"]
    src_a = cell_meta["source_a"]
    src_b = cell_meta["source_b"]
    bystanders = held_out_bystanders[: args.n_bystanders or len(held_out_bystanders)]

    personas_to_probe: list[str] = []
    personas_to_probe.append(src_a)
    if src_b != src_a:
        personas_to_probe.append(src_b)
    # Add all 4 negatives (default_assistant substitute + 3 close negs).
    for neg in NEGATIVE_PERSONAS:
        if neg not in personas_to_probe:
            personas_to_probe.append(neg)
    # Add bystanders (skipping any already counted as sources/negatives).
    for byst in bystanders:
        if byst not in personas_to_probe:
            personas_to_probe.append(byst)

    n_questions = args.probe_questions if not args.smoke else min(2, args.probe_questions)
    questions = list(pool.questions)[:n_questions]

    # MUST-FIX (round 3, BLOCKER): R-cache COVERAGE precheck. See
    # `_assert_r_cache_coverage` for the full rationale + checks.
    response_lookup = _assert_r_cache_coverage(
        pool=pool,
        personas_to_probe=personas_to_probe,
        questions=questions,
        slug_with_seed=slug_with_seed,
    )

    plan = ExtractionPlan(
        pair_name=pair_name,
        arm_slug=cell_meta["arm_slug"],
        seed=cell_meta["seed"],
        personas_to_probe=personas_to_probe,
        questions=questions,
        response_lookup=response_lookup,
    )

    base.eval()
    logger.info(
        "Extracting BASE reads (%d personas x %d questions) for cell %s ...",
        len(personas_to_probe),
        len(questions),
        slug_with_seed,
    )
    reads_base = extract_for_cell(base, tokenizer, plan=plan, marker_id=marker_id)

    logger.info("Loading adapter at %s and extracting TRAINED reads ...", cell_meta["adapter_path"])
    peft = PeftModel.from_pretrained(base, cell_meta["adapter_path"])
    peft.eval()
    reads_trained = extract_for_cell(peft, tokenizer, plan=plan, marker_id=marker_id)
    # Cleanup: detach adapter so the next cell starts from the bare base
    # again. unload() restores the base modules in-place; del peft +
    # empty_cache() releases the adapter weights.
    peft.unload()
    del peft
    torch.cuda.empty_cache()

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


def _resolve_held_out_bystanders(
    pair_name: str,
    args: argparse.Namespace,
    *,
    near_pair_override: tuple[str, str] | None = None,
) -> list[str]:
    """Held-out bystander panel for a given pair.

    ``near_pair_override`` lets the dispatcher swap NEAR_PAIR_PRIMARY for
    NEAR_PAIR_FALLBACK after the L20 cosine preflight (CONCERN B round 3).
    """
    from explore_persona_space.experiments.issue520.persona_panel import (
        FAR_PAIR,
        NEAR_PAIR_PRIMARY,
        held_out_bystanders_for_pair,
    )

    if pair_name == "far":
        return held_out_bystanders_for_pair(FAR_PAIR)
    elif pair_name == "near":
        near_pair = near_pair_override if near_pair_override is not None else NEAR_PAIR_PRIMARY
        return held_out_bystanders_for_pair(near_pair)
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


def _l20_centered_cosine_for_pair(
    base_model,
    tokenizer,
    *,
    pool,
    pair: tuple[str, str],
    marker_id: int,
    n_questions: int,
) -> float:
    """Centered cosine at L20 between two personas' post-response hidden states.

    For each persona p ∈ {pair_a, pair_b}, averages the L20 post-response
    hidden state across `n_questions` (each forwarded under p's system prompt
    with the base-model on-policy R(p, q)) → h_mean(p). Returns the cosine of
    the centered vectors (h_mean(a) - global_mean, h_mean(b) - global_mean)
    where global_mean = (h_mean(a) + h_mean(b)) / 2. Centering keeps the
    similarity invariant to a shared mean (matches the persona-cosine recipe
    in `.claude/rules/persona-distance-metrics.md`).
    """
    from explore_persona_space.experiments.issue520.shift_extract import (
        QWEN_HIDDEN_DIM,
        ContextRead,
        aggregate_reads_per_persona,
        cosine,
        read_hidden_and_logprob_for_context,
    )

    reads: list[ContextRead] = []
    questions = list(pool.questions)[:n_questions]
    for persona in pair:
        if persona not in pool.responses:
            raise RuntimeError(
                f"L20 cosine probe needs persona {persona!r} in R cache "
                f"(have {sorted(pool.responses.keys())!r})."
            )
        for q in questions:
            resp_list = pool.responses[persona].get(q)
            if not resp_list:
                continue
            r = resp_list[0]
            reads.append(
                read_hidden_and_logprob_for_context(
                    base_model,
                    tokenizer,
                    persona=persona,
                    question=q,
                    response=r,
                    marker_id=marker_id,
                )
            )
    agg = aggregate_reads_per_persona(reads)
    a, b = pair
    if a not in agg or b not in agg:
        raise RuntimeError(
            f"Could not aggregate L20 reads for both personas in pair {pair!r}; "
            f"got {sorted(agg.keys())!r}."
        )
    h_a = agg[a]["h_primary_mean"]
    h_b = agg[b]["h_primary_mean"]
    # Centered cosine: subtract the pair's joint mean from each vector.
    mean = [(h_a[i] + h_b[i]) / 2.0 for i in range(QWEN_HIDDEN_DIM)]
    h_a_c = [h_a[i] - mean[i] for i in range(QWEN_HIDDEN_DIM)]
    h_b_c = [h_b[i] - mean[i] for i in range(QWEN_HIDDEN_DIM)]
    return cosine(h_a_c, h_b_c)


def _pair_cosine_preflight(
    args: argparse.Namespace,
    *,
    pool,
    out_dir: Path,
    base_model=None,
    tokenizer=None,
    marker_id: int | None = None,
) -> dict:
    """Pair-selection preflight (plan §4 Step 2).

    Computes centered-cosine at L20 (post-response slot, base model) for the
    FAR pair, NEAR primary, and NEAR fallback. Writes the decision to
    ``eval_results/issue_520/pair_selection.json``.

    Selection rule (plan §4 Step 2):

    - If ``cosine_near_primary >= cosine_far + 0.3``, keep
      ``NEAR_PAIR_PRIMARY`` as the chosen near pair (the +0.3 separation
      gate is what makes "near" different from "far" along the persona-cosine
      axis).
    - If the +0.3 gate fails AND ``NEAR_PAIR_FALLBACK`` does NOT overlap
      with ``NEGATIVE_PERSONAS``, fall back to ``NEAR_PAIR_FALLBACK``.
    - If the +0.3 gate fails AND the fallback overlaps with
      ``NEGATIVE_PERSONAS`` (the gradient-conflict guard), keep PRIMARY and
      record the conflict — the operator must rebalance the negative panel
      before re-running.

    The actual model-side cosine work needs the loaded base model +
    tokenizer + marker_id. Callers MUST pass them unless either:

    - ``--skip-cosine-preflight`` is set (smoke convenience: writes
      ``cosine_*: None`` and annotates ``skipped_cosine: true``); OR
    - ``args.skip_shift_extract`` is set (no extraction-base loaded; the
      cosine probe needs the same forward pass as the extraction).

    Returns the decision dict; the caller can override the persona-panel
    default near pair via ``decision["near_pair_chosen"]``.
    """
    from explore_persona_space.experiments.issue520.persona_panel import (
        FAR_PAIR,
        NEAR_PAIR_FALLBACK,
        NEAR_PAIR_PRIMARY,
        NEGATIVE_PERSONAS,
    )

    decision: dict = {
        "far_pair": list(FAR_PAIR),
        "near_pair_primary": list(NEAR_PAIR_PRIMARY),
        "near_pair_fallback": list(NEAR_PAIR_FALLBACK),
        "near_pair_chosen": list(NEAR_PAIR_PRIMARY),
        "near_chosen_reason": "primary (default — cosine gate not yet evaluated)",
        "cosine_far": None,
        "cosine_near_primary": None,
        "cosine_near_fallback": None,
        "cosine_gate_threshold": 0.3,
        "skipped_cosine": False,
    }

    # Fallback gradient-conflict guard. Must fire BEFORE any training: silently
    # swapping to the fallback when it overlaps with NEGATIVE_PERSONAS would put
    # the same persona into both positive and negative rows.
    neg_set = set(NEGATIVE_PERSONAS)
    fb_a, fb_b = NEAR_PAIR_FALLBACK
    fallback_overlap = {p for p in (fb_a, fb_b) if p in neg_set}
    decision["fallback_gradient_conflict"] = sorted(fallback_overlap)
    fallback_usable = (not fallback_overlap) or ("near" not in args.pair)

    # Decide whether to compute the cosine. Skip when:
    #   - operator passed --skip-cosine-preflight, OR
    #   - --smoke (cheap, no GPU, smoke runs already pick small probe sets), OR
    #   - --skip-shift-extract (the extraction base isn't loaded — same forward
    #     pass is needed for the cosine probe), OR
    #   - the caller didn't pass base_model/tokenizer/marker_id (defer).
    skip_cosine = (
        args.skip_cosine_preflight
        or args.smoke
        or args.skip_shift_extract
        or base_model is None
        or tokenizer is None
        or marker_id is None
    )
    if skip_cosine:
        decision["skipped_cosine"] = True
        if fallback_overlap and "near" in args.pair:
            decision["near_chosen_reason"] = (
                f"PRIMARY (cosine skipped; fallback {NEAR_PAIR_FALLBACK!r} "
                f"overlaps NEGATIVE_PERSONAS at {sorted(fallback_overlap)} so "
                "fallback would be unusable anyway)"
            )
        else:
            decision["near_chosen_reason"] = (
                "primary (cosine skipped per --skip-cosine-preflight / --smoke "
                "/ --skip-shift-extract / model not loaded)"
            )
    else:
        # Compute the L20 centered cosine for FAR + NEAR_PRIMARY (+ fallback
        # only if the operator may need it, i.e. fallback_usable).
        n_q = max(2, min(args.probe_questions, len(pool.questions)))
        logger.info(
            "_pair_cosine_preflight: computing L20 centered-cosine over %d "
            "questions for FAR=%r, NEAR_PRIMARY=%r%s ...",
            n_q,
            FAR_PAIR,
            NEAR_PAIR_PRIMARY,
            (f", NEAR_FALLBACK={NEAR_PAIR_FALLBACK!r}" if fallback_usable else ""),
        )
        cos_far = _l20_centered_cosine_for_pair(
            base_model,
            tokenizer,
            pool=pool,
            pair=FAR_PAIR,
            marker_id=marker_id,
            n_questions=n_q,
        )
        cos_near_primary = _l20_centered_cosine_for_pair(
            base_model,
            tokenizer,
            pool=pool,
            pair=NEAR_PAIR_PRIMARY,
            marker_id=marker_id,
            n_questions=n_q,
        )
        cos_near_fallback = None
        if fallback_usable:
            cos_near_fallback = _l20_centered_cosine_for_pair(
                base_model,
                tokenizer,
                pool=pool,
                pair=NEAR_PAIR_FALLBACK,
                marker_id=marker_id,
                n_questions=n_q,
            )
        decision["cosine_far"] = cos_far
        decision["cosine_near_primary"] = cos_near_primary
        decision["cosine_near_fallback"] = cos_near_fallback

        gate_threshold = decision["cosine_gate_threshold"]
        primary_passes_gate = (cos_near_primary - cos_far) >= gate_threshold
        if primary_passes_gate:
            decision["near_pair_chosen"] = list(NEAR_PAIR_PRIMARY)
            decision["near_chosen_reason"] = (
                f"PRIMARY (cos_near_primary={cos_near_primary:.4f} "
                f"- cos_far={cos_far:.4f} = "
                f"{cos_near_primary - cos_far:+.4f} >= +{gate_threshold})"
            )
        elif fallback_usable and cos_near_fallback is not None:
            fallback_passes_gate = (cos_near_fallback - cos_far) >= gate_threshold
            if fallback_passes_gate:
                decision["near_pair_chosen"] = list(NEAR_PAIR_FALLBACK)
                decision["near_chosen_reason"] = (
                    f"FALLBACK (primary cos diff "
                    f"{cos_near_primary - cos_far:+.4f} < +{gate_threshold}; "
                    f"fallback cos diff {cos_near_fallback - cos_far:+.4f} "
                    f">= +{gate_threshold})"
                )
            else:
                decision["near_pair_chosen"] = list(NEAR_PAIR_PRIMARY)
                decision["near_chosen_reason"] = (
                    f"PRIMARY (neither primary nor fallback clears the "
                    f"+{gate_threshold} cosine gate — primary diff "
                    f"{cos_near_primary - cos_far:+.4f}, fallback diff "
                    f"{cos_near_fallback - cos_far:+.4f}; keeping primary)"
                )
        else:
            # Fallback unusable (gradient conflict with NEGATIVE_PERSONAS).
            decision["near_pair_chosen"] = list(NEAR_PAIR_PRIMARY)
            if fallback_overlap and "near" in args.pair:
                decision["near_chosen_reason"] = (
                    f"PRIMARY (cos diff {cos_near_primary - cos_far:+.4f} < "
                    f"+{gate_threshold} but fallback unusable: "
                    f"{NEAR_PAIR_FALLBACK!r} overlaps NEGATIVE_PERSONAS at "
                    f"{sorted(fallback_overlap)})"
                )
            else:
                decision["near_chosen_reason"] = (
                    f"PRIMARY (cos diff {cos_near_primary - cos_far:+.4f} < "
                    f"+{gate_threshold}; fallback skipped because 'near' not "
                    "in args.pair)"
                )

    out_path = out_dir / "pair_selection.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(decision, f, indent=2)
    logger.info("Wrote pair-selection decision: %s", out_path)
    logger.info("Near-pair decision: %s", decision["near_chosen_reason"])
    return decision


def _resolve_near_pair_override(pair_decision: dict) -> tuple[str, str] | None:
    """If the cosine preflight chose the fallback, return it; else None.

    Threading the override through ``_resolve_pair`` and
    ``_resolve_held_out_bystanders`` lets the dispatcher honor the +0.3
    cosine separation gate (plan §4 Step 2) without re-importing the panel
    constant at every call site.
    """
    from explore_persona_space.experiments.issue520.persona_panel import NEAR_PAIR_PRIMARY

    chosen = tuple(pair_decision["near_pair_chosen"])
    if chosen == tuple(NEAR_PAIR_PRIMARY):
        return None
    logger.warning(
        "L20 cosine preflight chose FALLBACK near pair %r (panel default %r). "
        "Threading override through cell enumeration + held-out-bystander "
        "resolution.",
        chosen,
        tuple(NEAR_PAIR_PRIMARY),
    )
    return chosen  # type: ignore[return-value]


def _enumerate_cells(
    args: argparse.Namespace,
    *,
    near_pair_override: tuple[str, str] | None,
) -> list:
    """Build the (spec, seed) list for the requested pair/arm/ratio/seed combos."""
    from explore_persona_space.experiments.issue520.data_prep import (
        build_arm_specs_for_pair,
    )

    cells_to_run: list = []
    for pair_name in args.pair:
        src_a, src_b = _resolve_pair(pair_name, near_pair_override=near_pair_override)
        for ratio in args.ratio:
            include_b2 = ratio == "b2"
            specs = build_arm_specs_for_pair(
                pair_name=pair_name,
                source_a=src_a,
                source_b=src_b,
                n_positives_per_source=args.positives_per_source,
                include_b2=include_b2,
            )
            for spec in specs:
                if spec.arm not in args.arm:
                    continue
                if spec.ratio != ratio:
                    continue
                for seed in args.seeds:
                    cells_to_run.append((spec, seed))
    return cells_to_run


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    overall_t0 = time.time()

    # CONCERN A (round 3): pin this process to ONE physical GPU before any
    # torch/transformers import-driven CUDA init. After this assignment,
    # `device_map={"": 0}` resolves to the gpu_id physical GPU consistently
    # for BOTH the extraction-base hoist below AND any downstream
    # `train_lora()` call (which also sets the same env var with the same
    # value at sft.py:653). Defends against the `+gpu_id=N` CUDA_VISIBLE_DEVICES
    # gotcha (CLAUDE.md Gotchas, #376 wave-1): if the dispatcher is launched
    # with `--gpu-id 1`, the extraction base used to land on GPU 0 because
    # the hoist hardcoded `device_map={"": 0}` while CVD was still unset.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    print("[phase=preflight]")
    pf = _preflight(args)
    pool = pf["pool"]
    marker_id = pf["marker_id"]
    tokenizer = pf["tokenizer"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from explore_persona_space.experiments.issue520.data_prep import resolve_data_dir

    data_dir = resolve_data_dir(Path(args.data_dir) if args.data_dir else None)

    # CONCERN B (round 3): hoist base model + tokenizer ONCE (not per cell)
    # BEFORE the pair-cosine preflight, so the preflight can actually
    # compute L20 cosines (round-2 wrote cosine_*: None — "deferred"). On
    # GPU 0 after the CVD pin above (CONCERN A). Skipped only when shift
    # extraction is disabled (--skip-shift-extract or --skip-train + smoke).
    base_model = None
    extract_tokenizer = None
    if not args.skip_shift_extract:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(
            "Loading Qwen-2.5-7B-Instruct ONCE for cosine preflight + per-cell "
            "extraction (CONCERN B round-2: avoid per-cell base-model reload "
            "of ~7 GB) on CUDA_VISIBLE_DEVICES=%s (gpu_id=%d) ...",
            os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
            args.gpu_id,
        )
        extract_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map={"": 0},  # CVD has remapped gpu_id → device 0
            trust_remote_code=True,
        )
        base_model.eval()

    # CONCERN B (round 3): pair-cosine preflight now actually computes the
    # L20 centered cosine using the hoisted base model. Returns the chosen
    # near-pair (primary-vs-fallback) based on the +0.3 cosine separation
    # gate (plan §4 Step 2). Writes pair_selection.json.
    pair_decision = _pair_cosine_preflight(
        args,
        pool=pool,
        out_dir=out_dir,
        base_model=base_model,
        tokenizer=extract_tokenizer,
        marker_id=marker_id,
    )
    near_pair_override = _resolve_near_pair_override(pair_decision)
    cells_to_run = _enumerate_cells(args, near_pair_override=near_pair_override)
    if not cells_to_run:
        logger.error("No cells matched the requested pair/arm/ratio/seed combos")
        return 2
    logger.info(
        "Will run %d cells (sequentially in-process on GPU %d). Pair decision: %s",
        len(cells_to_run),
        args.gpu_id,
        pair_decision["near_chosen_reason"],
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
        held_out_bystanders = _resolve_held_out_bystanders(
            spec.pair_name, args, near_pair_override=near_pair_override
        )
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
                base=base_model,
                tokenizer=extract_tokenizer,
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
