# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek α intentional
#!/usr/bin/env python3
"""Task #504 — single (cell, seed) worker: build → train → eval_trajectory.

Forked from scripts/i472_run_cell.py (same subprocess shape, env injection, GPU
pinning, vLLM teardown discipline). #504-specific changes:

  * Uses ``contrastive_neg_geometry_504.build_training_data.build_cell_504`` —
    the per-cell training pool with ONE positioned negative per arm (Phase 0.5
    lookup), NOT the #472 band-based ``select_negatives_by_geometry`` path.
  * Threads ``marker_suppress_at_post_response_slot=True`` +
    ``marker_im_end_token_id=151645`` (Qwen-2.5 ``<|im_end|>``) into
    ``train_one_cell`` (#477 v6 slot-fix; both flags already exposed on
    ``main`` via ``TrainLoraConfig`` — NO port-from-#474 needed).
  * Uses ``lora_r_override`` + ``lora_alpha_override`` from Phase 0's pinned
    rank (``chosen_rank``) + α from ``RANK_ALPHA_MAP_V5``; lr_override=2e-6
    (#477 mid-band).
  * Eval rig calls scripts/i504_eval_trajectory.py with the #504 held-out panel
    (= bank − {source, default, 4 positioned-N's}); uses the same
    ``assert_adapter_actually_applied`` guard #472 already wires.

GPU pinning + sentinel-file pattern are byte-identical to i472_run_cell.py
(round-3 #472 fix). The dispatcher passes ``--gpu-id <g>``; train/sft.py SETS
``CUDA_VISIBLE_DEVICES=str(g)``; the nested eval subprocess inherits it.

Usage (driven by the dispatcher; --gpu-id is the assigned physical GPU):
    uv run python scripts/i504_run_cell.py \
        --cell c504_near --seed 42 --gpu-id 3 \
        --chosen-rank 8 --chosen-alpha 32 --chosen-frac 0.5 \
        --arm-to-n-json /tmp/i504-arm-to-n.json \
        --slab-root eval_results/issue_504 --runs-root /workspace/runs/issue_504 \
        --log-dir /workspace/logs [--smoke] [--no-kl]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i504.run_cell")


def _write_sentinel(path: Path, *, kind: str, phase: str, note: dict) -> None:
    """Write a poll_pipeline.py-compliant sentinel (sentinel_schema_version=1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": kind,
                "version": 1,
                "task_id": 504,
                "by": "i504_run_cell",
                "ts": datetime.now(UTC).isoformat(),
                "phase": phase,
                "note": json.dumps(note),
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_504"))
    ap.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_504"))
    ap.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument("--centroids-dir", type=Path, default=Path("data/issue_472"))
    ap.add_argument(
        "--r-train-path", type=Path, default=Path("data/issue_472/on_policy_R/R_train.json")
    )
    ap.add_argument(
        "--r-eval-path", type=Path, default=Path("data/issue_472/on_policy_R/R_eval.json")
    )
    ap.add_argument(
        "--arm-to-n-json",
        type=Path,
        required=True,
        help=(
            "JSON file: {arm_slug: positioned_N_persona, ...} + optional "
            "{'smoke_mid_band_n': ..., 'panel': [...]}. Produced by Phase 0.5."
        ),
    )
    ap.add_argument(
        "--chosen-rank",
        type=int,
        required=True,
        help="LoRA rank pinned by Phase 0 (one of {4, 8, 16}).",
    )
    ap.add_argument(
        "--chosen-alpha",
        type=int,
        required=True,
        help="LoRA α at the pinned rank (RANK_ALPHA_MAP_V5).",
    )
    ap.add_argument(
        "--chosen-frac",
        type=float,
        default=None,
        help=(
            "Pinned checkpoint fraction from Phase 0 (informational — Phase 2 "
            "reads ALL arms at this frac; the trainer saves all 6 frac "
            "checkpoints so the robustness panel can re-fit at the cell's "
            "own best-band frac)."
        ),
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "v2 lr override (plan v2 §4.1). Overrides the module-level "
            "ANCHOR_LR default by threading `lr_override=<this value>` into "
            "`train_one_cell`. The CLI is the SINGLE override surface for the "
            "v2 pipeline — `dispatch_neg_geometry_504.py --phase phase1` reads "
            "`chosen_lr` from `phase0_calibration_v2.json` and passes it here. "
            "For v2 Phase 0 smoke cells, the dispatcher passes the slug-implied "
            "lr (recovered via `lr_for_v2_smoke_slug`). When unset, falls back "
            "to ANCHOR_LR (= 2e-6, v1 default)."
        ),
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "v3 EPOCHS override (plan v3 §4.1). Overrides the module-level "
            "EPOCHS default (=1) by threading `epochs_override=<this value>` into "
            "`train_one_cell` → trainer's `num_train_epochs` config. The CLI is "
            "the SINGLE override surface for the v3 pipeline — "
            "`dispatch_neg_geometry_504.py --phase phase0_v3` passes the slug-implied "
            "epochs (recovered via `epochs_for_v3_smoke_slug`), and `--phase phase1` "
            "reads `chosen_epochs` from `phase0_calibration_v3.json` and passes it "
            "uniformly across the 5 main arms. When unset, falls back to EPOCHS=1 "
            "(v1/v2 default)."
        ),
    )
    ap.add_argument(
        "--checkpoint-fractions",
        default=None,
        help=(
            "v3 in-plan finer-fraction recovery (plan v3 §4.1 trigger B + §4.2). "
            "Comma-separated floats overriding the cell's default checkpoint "
            "fraction cadence (normally CHECKPOINT_FRACTIONS = "
            "(0.08, 0.16, 0.33, 0.50, 0.75, 1.00)). When set, threads as "
            "`step_calibration_fractions=parsed` into `train_one_cell`, so the "
            "trainer's CheckpointAtFractionsCallback saves adapters at THESE "
            "fractions of max_steps and the nested eval_trajectory subprocess "
            "evaluates those same checkpoints. Used by the dispatcher's "
            "`--phase phase0_v3-recovery` to re-train EPOCHS=2 at "
            "{0.02, 0.04, 0.06, 0.08}. When unset, uses CHECKPOINT_FRACTIONS "
            "(byte-identical pre-recovery behavior)."
        ),
    )
    ap.add_argument(
        "--trajectory-suffix",
        default="",
        help=(
            "v3 in-plan recovery (plan v3 §4.1 + §4.2): suffix appended to "
            "the trajectory.json output subdir under --slab-root so the "
            "recovery run does NOT clobber the coarse-grid trajectory. "
            "Recovery passes `--trajectory-suffix __recovery_finer`; the "
            "merged-pick picker then reads BOTH `<slug>_seed<S>/trajectory.json` "
            "(coarse) AND `<slug>_seed<S>__recovery_finer/trajectory.json` "
            "(finer). Default empty = pre-recovery behavior (canonical path)."
        ),
    )
    ap.add_argument(
        "--wandb-suffix",
        default="",
        help=(
            "Optional suffix appended to the WandB run name + HF subfolder. "
            "Plan v2 §11 reproducibility: each cell's WandB run name carries "
            "`_lr<chosen_lr>` (e.g. `c504v2_near_seed42_lr3e-05`) so v1 and "
            "v2 runs are visually distinguishable. Default empty = no suffix. "
            "Composes with `--hf-path-suffix` (which decorates a DIFFERENT "
            "axis — the round-N adapter-collision avoidance suffix)."
        ),
    )
    ap.add_argument(
        "--max-new-tokens-eval",
        type=int,
        default=2048,
        help="Eval max_new_tokens. Bumped to 4096 by Phase 0.5 if a train-time R saturated 1024.",
    )
    ap.add_argument(
        "--max-model-len-eval",
        type=int,
        default=None,
        help=(
            "Round-2 fix (blocker #4): vLLM max_model_len (prompt + generation) for the "
            "nested eval. Must be >= --max-new-tokens-eval + EVAL_PROMPT_HEADROOM "
            "(default 512) so vLLM doesn't silently cap generation. If unset, computed "
            "as max(2048, max_new_tokens_eval + 512)."
        ),
    )
    ap.add_argument("--smoke", action="store_true", help="Tiny slice: fewer steps, 2 checkpoints.")
    ap.add_argument("--no-kl", action="store_true", help="Skip DV-B KL.")
    ap.add_argument("--report-to", default="wandb")
    ap.add_argument(
        "--hf-path-suffix",
        default="",
        help=(
            "Round-15 strengthen-anchor knob: appended to the HF model-repo "
            "subfolder (`adapters/issue_504/<cell>_seed<S><suffix>`) AND to the "
            "local runs-root subdir so a re-run with a different training "
            "budget can coexist with prior rounds' adapters on HF without "
            "overwriting them. Default empty = byte-identical pre-round-15 "
            "behavior. Round-15 launcher passes `--hf-path-suffix __r15`. "
            "The slab-root trajectory (read by Phase 0 pick + Phase 2 "
            "analysis) is NOT decorated — local pod-side smoke trajectories "
            "are ephemeral and the round-13/14 readings are preserved in "
            "`eval_results/issue_504/reval_confirm/` and in the round-13/14 "
            "i504_reval_*.py outputs."
        ),
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "ASSIGNED physical GPU index (round-3 #472 fix). Threaded to "
            "train_one_cell(gpu_id=...); train/sft.py SETS CUDA_VISIBLE_DEVICES "
            "to this so the cell + its nested eval run on physical GPU <gpu-id>."
        ),
    )
    ap.add_argument(
        "--source",
        default=None,
        help=(
            "Round-2 fix (BLOCKER #2, concern_id `fallback-source-threading`): "
            "source persona name. The v2 Phase 0 fallback path (plan v2 §4.2) "
            "swaps `villain` for an easier candidate (medical_doctor, librarian, "
            "...); when fallback fires, every Phase 1 cell must train + evaluate "
            "against that picked source — NOT the default villain. If unset, "
            "falls back to the v1/v2 module default `SOURCE_PERSONA` (= villain), "
            "preserving byte-identical legacy behavior."
        ),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format=(
            f"%(asctime)s [phase=cell_{args.cell}_seed{args.seed}] "
            f"%(name)s %(levelname)s | %(message)s"
        ),
        stream=sys.stdout,
    )
    log.info(
        "Assigned physical GPU --gpu-id=%d; inherited CUDA_VISIBLE_DEVICES=%s "
        "(train/sft.py will SET CVD=str(gpu_id)).",
        args.gpu_id,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )

    # Marker tokenizer pre-spawn assert (CLAUDE.md marker-leakage rule).
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        ANCHOR_LR,
        BASE_MODEL,
        CHECKPOINT_FRACTIONS,
        EPOCHS,
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_IM_END_TOKEN_ID,
        MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        MARKER_TEXT,
        SOURCE_PERSONA,
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ids = tok.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [EXPECTED_MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"Marker tokenizer assertion FAILED: encode({MARKER_TEXT!r})={ids}, "
            f"expected [{EXPECTED_MARKER_TOKEN_ID}]."
        )
    log.info("[phase=preflight] marker assertion PASS: %r -> %s", MARKER_TEXT, ids)

    # Load Phase 0.5 cell resolution (arm_to_n + optional smoke_mid_band_n + panel).
    arm_to_n_payload = json.loads(args.arm_to_n_json.read_text())
    arm_to_positioned_n = arm_to_n_payload.get("arm_to_positioned_n", {})
    smoke_mid_band_n = arm_to_n_payload.get("smoke_mid_band_n")
    held_out_panel: list[str] = arm_to_n_payload.get("held_out_panel", [])
    # Round-2 / round-6 fix (Concern B + v3 KeyError): include v2 + v3 smoke
    # prefixes for parity. v1 (`c504_smoke_`), v2 (`c504v2_smoke_`), and v3
    # (`c504v3_smoke_`) smoke cells all consume `smoke_mid_band_n` from the
    # Phase 0.5 artifact; cell_resolution.py catches the v3 prefix downstream,
    # but this entrypoint guard must mirror the recognition list so the
    # missing-artifact case fails loud here (not deeper in build_cell_504).
    if (
        args.cell.startswith(("c504_smoke_", "c504v2_smoke_", "c504v3_smoke_"))
        and smoke_mid_band_n is None
    ):
        raise ValueError(
            f"--arm-to-n-json {args.arm_to_n_json} is missing 'smoke_mid_band_n' but the cell "
            f"{args.cell!r} is a smoke cell that requires it."
        )

    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
        load_r_artifact,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
        train_one_cell,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.build_training_data import (
        build_cell_504,
    )

    # Round-15: per-round HF-path suffix decorates the LOCAL runs-root subdir
    # (so /workspace/runs/issue_504/<slug>__r15/ doesn't collide with the
    # round-13/14 floor-anchor adapter on the same pod) AND the HF model-repo
    # subfolder (passed to train_one_cell below). The slab-root trajectory
    # path is NOT decorated — Phase 0 pick + Phase 2 analysis read from the
    # canonical `eval_results/issue_504/<slug>_seed<S>/trajectory.json`
    # location; local pod-side smoke trajectories are ephemeral.
    run_slug = f"{args.cell}_seed{args.seed}{args.hf_path_suffix}"
    run_dir = args.runs_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = run_dir / "train_pool.jsonl"
    final_adapter_dir = run_dir / "adapter"
    ckpt_root = run_dir / "checkpoints"
    # v3 in-plan recovery (plan §4.1 + §4.2): --trajectory-suffix decorates the
    # slab-root subdir so the finer-grid recovery trajectory does NOT clobber
    # the coarse trajectory. Default empty = canonical path (byte-identical
    # pre-recovery behavior). The picker's merge step reads BOTH paths.
    traj_subdir = f"{args.cell}_seed{args.seed}{args.trajectory_suffix}"
    out_traj = args.slab_root / traj_subdir / "trajectory.json"
    sentinel = args.log_dir / f"issue-504-{args.cell}-seed{args.seed}-results.json"

    bank = load_persona_bank(args.bank_path)
    r_train = load_r_artifact(args.r_train_path)
    q_train, _q_eval = get_train_eval_questions()

    # Round-2 fix (BLOCKER #2, concern_id `fallback-source-threading`):
    # resolve effective source persona. --source CLI overrides the v1/v2 module
    # default (SOURCE_PERSONA = villain). The v2 Phase 0 fallback path (plan
    # v2 §4.2) threads the easier candidate (e.g. medical_doctor) here, and
    # the cell builds positives + evaluates the trajectory against THAT
    # persona, NOT the default. Assertion: source must be in the bank.
    effective_source = args.source if args.source is not None else SOURCE_PERSONA
    if effective_source not in bank:
        raise KeyError(
            f"--source {effective_source!r} missing from persona bank at {args.bank_path}. "
            f"FALLBACK_SOURCE_CANDIDATES values must all be in the bank — verify the "
            "Phase 0.5 bank version."
        )
    log.info(
        "[phase=source] effective source persona = %r (CLI --source=%r, default=%r)",
        effective_source,
        args.source,
        SOURCE_PERSONA,
    )

    # ── Phase: build training data (CPU). ────────────────────────────────────
    log.info("[phase=build_%s] building training data via cell_resolution_504", args.cell)
    build_cell_504(
        args.cell,
        train_jsonl,
        r_train=r_train,
        arm_to_positioned_n=arm_to_positioned_n,
        q_train=q_train,
        persona_bank=bank,
        source=effective_source,
        marker_text=MARKER_TEXT,
        smoke_mid_band_n=smoke_mid_band_n,
        seed=args.seed,
    )

    # ── Phase: train with mid-run checkpoints (HF Trainer, in-process). ──────
    # Smoke uses an even denser cadence around the early window where ΔG is
    # still climbing sub-ceiling (matches #472 smoke convention).
    #
    # v3 in-plan recovery (plan §4.1 trigger B + §4.2): --checkpoint-fractions
    # CSV CLI override wins over both the smoke cadence AND the module default
    # so the dispatcher's recovery phase can re-train EPOCHS=2 at
    # {0.02, 0.04, 0.06, 0.08} without touching the constants. The CSV is
    # parsed once into a strictly-positive sorted tuple; ANY parse error fails
    # loud BEFORE train_one_cell.
    if args.checkpoint_fractions is not None:
        try:
            parsed = tuple(
                sorted(float(x.strip()) for x in args.checkpoint_fractions.split(",") if x.strip())
            )
        except ValueError as exc:
            raise ValueError(
                f"--checkpoint-fractions {args.checkpoint_fractions!r} could not "
                f"parse as a comma-separated list of floats: {exc}."
            ) from exc
        if not parsed:
            raise ValueError(
                f"--checkpoint-fractions {args.checkpoint_fractions!r} parsed to "
                f"an empty tuple; need at least one positive fraction."
            )
        if any(f <= 0 or f > 1.0 for f in parsed):
            raise ValueError(
                f"--checkpoint-fractions {args.checkpoint_fractions!r} contains "
                f"out-of-range values; each must be in (0, 1]."
            )
        fractions = parsed
        log.info(
            "[phase=train_%s] --checkpoint-fractions override active: %s "
            "(v3 in-plan recovery; replaces CHECKPOINT_FRACTIONS).",
            args.cell,
            fractions,
        )
    else:
        fractions = (0.08, 0.16, 0.5, 1.0) if args.smoke else CHECKPOINT_FRACTIONS
    # Effective lr: --lr CLI override wins (plan v2 §10); else fall back to
    # ANCHOR_LR (the v1 floor recipe at 2e-6, retained as module default for
    # callers that don't go through the v2 Phase 0 pick).
    effective_lr = args.lr if args.lr is not None else ANCHOR_LR
    # Effective EPOCHS: --epochs CLI override wins (plan v3 §4.1); else fall
    # back to the module-level EPOCHS default (=1, v1/v2 default). v3 Phase 0
    # threads the slug-implied epochs (via `epochs_for_v3_smoke_slug` in the
    # dispatcher); v3 Phase 1 threads `chosen_epochs` from the v3 pick artifact.
    effective_epochs = args.epochs if args.epochs is not None else EPOCHS
    # WandB run name: plan v2 §11 requires `_lr{chosen_lr}` in the run name;
    # plan v3 §7 requires `_eps{chosen_epochs}_lr{lr}`. Compose: --wandb-suffix
    # CLI wins when set; v3 path (--epochs set) auto-builds the joint suffix
    # so v3 runs are visually distinguishable from v2 (lr-only) and v1 (raw)
    # on the WandB dashboard.
    if args.wandb_suffix:
        wandb_suffix = args.wandb_suffix
    elif args.epochs is not None:
        # v3: joint EPOCHS + lr suffix.
        wandb_suffix = f"_eps{effective_epochs}_lr{effective_lr:g}"
    else:
        # v1/v2: lr-only suffix (byte-identical pre-v3 behavior).
        wandb_suffix = f"_lr{effective_lr:g}"
    log.info(
        "[phase=train_%s] training (rank=%d, alpha=%d, lr=%g, epochs=%d, smoke=%s, "
        "suppress_at_post_response_slot=%s, im_end_id=%s, wandb_suffix=%s)",
        args.cell,
        args.chosen_rank,
        args.chosen_alpha,
        effective_lr,
        effective_epochs,
        args.smoke,
        MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        MARKER_IM_END_TOKEN_ID,
        wandb_suffix,
    )
    train_result = train_one_cell(
        cell_slug=args.cell,
        seed=args.seed,
        train_jsonl=train_jsonl,
        output_dir=final_adapter_dir,
        ckpt_root=ckpt_root,
        fractions=fractions,
        fallback=False,
        report_to=args.report_to,
        gpu_id=args.gpu_id,
        # #504 overrides on top of #472 defaults — pinned by Phase 0.
        # v2: --lr CLI override threads here as `lr_override` (single load-
        # bearing change vs v1).
        # v3: --epochs CLI override threads here as `epochs_override` (single
        # load-bearing change vs v2). The trainer's `num_train_epochs` config
        # picks up `epochs_override` (verified by `train_one_cell` signature
        # already accepting `epochs_override` — see v1 path that passes
        # `epochs_override=EPOCHS`).
        lr_override=effective_lr,
        epochs_override=effective_epochs,
        lora_r_override=args.chosen_rank,
        lora_alpha_override=args.chosen_alpha,
        # #504 sources adapters under adapters/issue_504/<slug>_seed<S>; the
        # round-15 --hf-path-suffix decorates the HF subfolder + WandB run
        # name so a strengthened-anchor re-run does NOT overwrite the round-
        # 13/14 dispositive-A/B adapters at the canonical path.
        hf_path_in_repo_override=f"adapters/issue_504/{run_slug}",
        # v2/v3: WandB run name carries lr/eps suffix per plan §11 / §7 so
        # v1/v2/v3 runs are distinguishable on the dashboard.
        run_name_override=f"issue504_{run_slug}{wandb_suffix}",
        # #477 v6 marker-suppress fix is the #504 baseline.
        marker_suppress_at_post_response_slot=MARKER_SUPPRESS_AT_POST_RESPONSE_SLOT,
        marker_im_end_token_id=MARKER_IM_END_TOKEN_ID,
    )
    ckpt_index_path = run_dir / "checkpoint_index.json"
    ckpt_index_path.write_text(json.dumps(train_result["checkpoint_index"], indent=2))
    log.info(
        "[phase=train_%s] done; checkpoints=%s",
        args.cell,
        list(train_result["checkpoint_index"]),
    )

    # Free in-process LoRA-training GPU memory before nested vLLM eval.
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase: eval_trajectory (NESTED subprocess: vLLM teardown isolation). ─
    eval_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if eval_cvd != str(args.gpu_id):
        raise RuntimeError(
            f"Pre-eval CUDA_VISIBLE_DEVICES={eval_cvd!r} != assigned --gpu-id={args.gpu_id}; "
            "the nested eval subprocess would run on the wrong physical GPU. "
            "train/sft.py should have set CVD=str(gpu_id) during training."
        )
    log.info(
        "[phase=eval_%s] inherited CUDA_VISIBLE_DEVICES=%s (physical GPU %d)",
        args.cell,
        eval_cvd,
        args.gpu_id,
    )
    # Round-2 fix (minor cleanup, blocker #4 prep): build eval_max_new_tokens
    # ONCE so the eval-side argparse sees a single, consistent value (the
    # earlier code appended --max-new-tokens twice under --smoke, which
    # argparse silently resolves to the LAST value — ambiguous behavior the
    # round-1 reviewer flagged).
    eval_max_new_tokens = 256 if args.smoke else args.max_new_tokens_eval
    # Round-2 fix (blocker #4): vLLM max_model_len must track max_new_tokens.
    # If max_new_tokens_eval bumped to 4096 (Phase 0.5 max-length safeguard)
    # but max_model_len stayed at 2048, vLLM silently caps generation at
    # 2048 - prompt_len; for marker evals on the long-tail Q_eval this turns
    # the headline DV into a silent-zero artifact. Compute headroom-aware
    # max_model_len here so the floor is always max_new_tokens + 512.
    eval_prompt_headroom = 512
    eval_max_model_len = (
        args.max_model_len_eval
        if args.max_model_len_eval is not None
        else max(2048, eval_max_new_tokens + eval_prompt_headroom)
    )
    eval_cmd = [
        "uv",
        "run",
        "python",
        "scripts/i504_eval_trajectory.py",
        "--cell",
        args.cell,
        "--seed",
        str(args.seed),
        "--checkpoint-index",
        str(ckpt_index_path),
        "--out-path",
        str(out_traj),
        "--bank-path",
        str(args.bank_path),
        "--r-eval-path",
        str(args.r_eval_path),
        "--panel-json",
        str(args.arm_to_n_json),
        "--max-lora-rank",
        str(args.chosen_rank),
        "--max-new-tokens",
        str(eval_max_new_tokens),
        "--max-model-len",
        str(eval_max_model_len),
        # Round-2 fix (BLOCKER #2): thread the effective source persona so the
        # nested eval scores trajectory ΔG/emission against the SAME persona
        # the cell was just trained on (not the hardcoded villain default).
        "--source",
        effective_source,
    ]
    if args.no_kl:
        eval_cmd.append("--no-kl")
    log.info(
        "[phase=eval_%s] nested eval subprocess (max_new_tokens=%d, max_model_len=%d): %s",
        args.cell,
        eval_max_new_tokens,
        eval_max_model_len,
        " ".join(eval_cmd),
    )
    subprocess.run(eval_cmd, env={**os.environ}, check=True)

    if not out_traj.exists():
        raise RuntimeError(
            f"[{args.cell}] eval_trajectory subprocess exited 0 but {out_traj} missing — "
            "silent eval failure (feedback_eval_script_silent_not_present_misdiagnosis)."
        )

    _write_sentinel(
        sentinel,
        kind="epm:progress",
        phase=f"cell_done_{args.cell}_seed{args.seed}",
        note={
            "cell": args.cell,
            "seed": args.seed,
            "trajectory_path": str(out_traj),
            "adapter_hf_path": f"adapters/issue_504/{run_slug}",
            "hf_path_suffix": args.hf_path_suffix,
            "checkpoint_index": str(ckpt_index_path),
            "n_held_out_panel": len(held_out_panel),
        },
    )
    log.info("[phase=cell_done_%s_seed%s] wrote sentinel → %s", args.cell, args.seed, sentinel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
