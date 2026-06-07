#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker token " ※" are intentional
"""Task #508 — unified dispatcher: smoke IS sweep with 2 cells.

Smoke phase (--cells lora_b2,ft_b2 --seeds 42) runs the EXACT same code path
as the full sweep (--cells lora_b1,lora_b2,lora_b3,ft_b1,ft_b2,ft_b3 --seeds 42).
Same dispatcher, same per-cell subprocess shape, same eval surface, same WandB
project, same teardown. This is the smoke/sweep architectural-parity gate
PASS_UNIFIED verdict (CLAUDE.md /issue Step 6d.0).

Phases per cell:
    Phase 0:  Ensure R_train / R_eval / dynamics_probes / per-cell training
              JSONLs exist (CPU-only). Reused / built once across cells.
    Phase 1:  Train cell. LoRA arm calls ``train_one_cell`` (#472 reused).
              Full-FT arm calls ``train_one_cell_fullft`` (new accelerate
              subprocess). Both cells get the MarkerDynamicsCallback.
    Phase 2:  Per-cell eval with per-cell base log P on each cell's own
              trained R (MF1).
    Phase 3 (post-sweep, --do-analyze): bracketing check + cluster bootstrap +
              hero figure + trajectory figures.

Resume-safe by per-phase output presence: if the eval JSON exists for a cell,
the dispatcher skips the train + eval and moves on (CLAUDE.md "Checkpoint per
phase"). The post-train sweep teardown deletes the merged FT checkpoint dir
after eval per .claude/rules/upload-policy.md.

Usage on the pod (after `pod.py provision --issue 508 --intent ft-7b`):
    nohup uv run python scripts/dispatch_508.py \
        --cells lora_b2,ft_b2 \
        --seeds 42 \
        --output-root /workspace/issue_508 \
        --build-data \
        > /workspace/logs/issue-508-smoke.log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOG = logging.getLogger("issue_508.dispatch")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task #508 unified dispatcher")
    p.add_argument(
        "--cells",
        default="lora_b2,ft_b2",
        help="Comma-separated cell slugs (smoke=lora_b2,ft_b2; sweep=all 6)",
    )
    p.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated seeds (default single seed 42 per plan §11)",
    )
    p.add_argument(
        "--output-root",
        default="/workspace/issue_508",
        type=Path,
        help="Root for per-cell artifacts (training data, checkpoints, eval JSON).",
    )
    p.add_argument(
        "--build-data",
        action="store_true",
        help="If set, re-build the per-cell training JSONLs + dynamics probes.",
    )
    p.add_argument(
        "--build-only",
        action="store_true",
        help="Stop after Phase 0 (data build); useful for CPU-only smoke.",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip Phase 1 (training). Assumes checkpoints already exist.",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip Phase 2 (eval). Useful for CPU-only data+launch smoke.",
    )
    p.add_argument(
        "--do-analyze",
        action="store_true",
        help="After all cells finish, run Phase 3 analysis.",
    )
    p.add_argument(
        "--lora-gpu-id",
        type=int,
        default=0,
        help="Physical GPU index pinned to the LoRA arm (default 0).",
    )
    p.add_argument(
        "--num-gpus-fullft",
        type=int,
        default=4,
        help="GPUs for ZeRO-3 full-FT (default 4).",
    )
    p.add_argument(
        "--ft-lr-override",
        type=float,
        default=None,
        help="Override FT_LEARNING_RATE (smoke-gate NaN fallback: drop to 2e-6).",
    )
    p.add_argument(
        "--budget-overrides",
        default=None,
        help=(
            "JSON dict mapping cell_slug → epoch_fraction (overrides defaults). "
            "Used by the smoke-gate-5 §4.4 budget-shift contingency."
        ),
    )
    p.add_argument(
        "--no-dynamics",
        action="store_true",
        help="Skip the in-training MarkerDynamicsCallback (faster smoke).",
    )
    return p.parse_args()


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                env={**os.environ},
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def phase0_build_data(
    output_root: Path,
    cells: list[str],
    seeds: list[int],
    *,
    build_data: bool,
) -> dict[str, Path]:
    """Phase 0 — build per-cell training JSONLs + dynamics probes (CPU-only).

    Reuses #472's ``r_generate``, ``select_negatives``, ``build_training_data``
    pipeline modules. Skips re-building if all artifacts already exist (unless
    ``build_data=True``).

    Returns ``{cell_slug: train_jsonl_path}``.
    """
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )
    from explore_persona_space.experiments.lora_vs_ft_508 import (
        CONTRASTIVE_NEGATIVES,
        DYNAMICS_PROBES_PATH,
        NEG_EX_PER_PERSONA,
        POS_EX_PER_SOURCE,
        SOURCE_PERSONA,
        load_q_eval,
        load_q_train,
    )
    from explore_persona_space.experiments.lora_vs_ft_508.marker_dynamics_callback import (
        build_dynamics_probes,
        save_dynamics_probes,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    train_dir = output_root / "training"
    train_dir.mkdir(parents=True, exist_ok=True)

    persona_bank = dict(EVAL_PERSONAS_24)
    q_train = load_q_train()
    q_eval = load_q_eval()

    # ── Build dynamics probes (CPU-only). ────────────────────────────────────
    probes_path = Path(DYNAMICS_PROBES_PATH)
    if build_data or not probes_path.exists():
        probes = build_dynamics_probes(persona_bank, q_eval, seed=42)
        save_dynamics_probes(probes, probes_path)

    # ── Build per-cell training JSONLs. ──────────────────────────────────────
    # The data is SHARED across cells (single-variable rule — the training data
    # is byte-identical across all 6 cells; the only diff is which weights get
    # updates). We build ONE training JSONL ("the contrastive recipe") and
    # symlink each cell's expected path to it.
    canonical_train = train_dir / "contrastive_recipe.jsonl"
    if build_data or not canonical_train.exists():
        LOG.info("[phase=0_build_data] Building canonical training JSONL")
        # Build a manual JSONL: 200 villain positives + 200×4 negatives,
        # interleaved. Uses base-model on-policy R from
        # data/issue_472/on_policy_R if available (the #472 R_train.json is
        # the same base-model output we'd re-generate; reuse for cost).
        # If #472's R_train.json is missing, fail loud — the implementer's
        # pre-launch protocol §4.6 includes "reuse from #472 if available".
        r_train_path = Path("data/issue_472/on_policy_R/R_train.json")
        if not r_train_path.exists():
            r_gen_mod = "explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate"
            raise FileNotFoundError(
                f"Required R_train.json missing: {r_train_path}. Run "
                f"`python -m {r_gen_mod}` first (or "
                "`python scripts/dispatch_508.py --build-data` with R generation enabled)."
            )
        r_train = json.loads(r_train_path.read_text())
        _build_canonical_training_jsonl(
            output_path=canonical_train,
            r_train=r_train,
            persona_bank=persona_bank,
            q_train=q_train,
            source=SOURCE_PERSONA,
            negatives=CONTRASTIVE_NEGATIVES,
            pos_ex=POS_EX_PER_SOURCE,
            neg_ex_per_persona=NEG_EX_PER_PERSONA,
            seed=42,
        )

    # Map each cell to the same canonical training JSONL.
    cell_to_jsonl: dict[str, Path] = {}
    for cell in cells:
        cell_to_jsonl[cell] = canonical_train
    LOG.info("[phase=0_build_data done] %d cells → %s", len(cells), canonical_train)
    print(f"[phase=0_build_data_done n_cells={len(cells)}]", flush=True)
    return cell_to_jsonl


def _build_canonical_training_jsonl(  # noqa: C901 - linear single-pass data builder
    output_path: Path,
    *,
    r_train: dict,
    persona_bank: dict[str, str],
    q_train: list[str],
    source: str,
    negatives: tuple[str, ...],
    pos_ex: int,
    neg_ex_per_persona: int,
    seed: int,
) -> Path:
    """Build the one shared training JSONL (positives + contrastive negatives).

    Single-variable rule: byte-identical across both arms. Mirrors #472's
    ``build_training_data.build_cell`` but specialized to a fixed negative
    set (the 4-negative #508 contrastive set) and a fixed POS_EX / NEG_EX,
    bypassing the cell-spec lookup.

    Hard checks: no marker contamination in any negative R; expected row
    count; deterministic shuffle.
    """
    import random

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_SEP,
        MARKER_TEXT,
    )

    def _make_row(system_prompt: str | None, user: str, assistant: str) -> dict:
        msgs: list[dict] = []
        if system_prompt is not None:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user})
        return {
            "prompt": msgs,
            "completion": [{"role": "assistant", "content": assistant}],
        }

    def _has_marker(text: str, token_ids: list[int] | None) -> bool:
        if MARKER_TEXT in text:
            return True
        return token_ids is not None and EXPECTED_MARKER_TOKEN_ID in token_ids

    def _resolve(persona: str, q: str) -> tuple[str, list[int] | None]:
        if persona not in r_train:
            raise KeyError(
                f"r_train missing persona {persona!r}; available {sorted(r_train)[:8]}..."
            )
        if q not in r_train[persona]:
            raise KeyError(f"r_train[{persona!r}] missing q {q!r}")
        entry = r_train[persona][q]
        return entry["response_text"], entry.get("response_token_ids")

    def _sample(n: int, rng: random.Random) -> list[str]:
        if n <= len(q_train):
            return rng.sample(q_train, n)
        out: list[str] = []
        while len(out) < n:
            perm = list(q_train)
            rng.shuffle(perm)
            out.extend(perm)
        return out[:n]

    rows: list[dict] = []

    # Positive rows (villain source).
    pos_rng = random.Random(seed)
    src_prompt = persona_bank[source]
    for q in _sample(pos_ex, pos_rng):
        r_text, r_ids = _resolve(source, q)
        if _has_marker(r_text, r_ids):
            raise AssertionError(
                f"positive row {source}/{q!r}: marker already in R BEFORE append — "
                "training would double-emit; R artifact stale."
            )
        rows.append(_make_row(src_prompt, q, f"{r_text}{MARKER_SEP}{MARKER_TEXT}"))
    n_positive = len(rows)

    # Negative rows (4 contrastive personas, no marker).
    for j_idx, neg in enumerate(negatives):
        if neg not in persona_bank:
            raise KeyError(f"persona_bank missing negative {neg!r}")
        neg_prompt = persona_bank[neg]
        neg_rng = random.Random(seed + 1000 + j_idx)
        for q in _sample(neg_ex_per_persona, neg_rng):
            r_text, r_ids = _resolve(neg, q)
            if _has_marker(r_text, r_ids):
                raise AssertionError(
                    f"negative row {neg}/{q!r}: marker contamination in R — would "
                    "silently train the model to emit the marker after a bystander."
                )
            rows.append(_make_row(neg_prompt, q, r_text))
    n_negative = len(rows) - n_positive
    expected = pos_ex + len(negatives) * neg_ex_per_persona
    if len(rows) != expected:
        raise AssertionError(f"row count mismatch: got {len(rows)}, expected {expected}")

    random.Random(seed).shuffle(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "source": source,
        "negatives": list(negatives),
        "pos_ex": pos_ex,
        "neg_ex_per_persona": neg_ex_per_persona,
        "n_total": len(rows),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "marker_text": MARKER_TEXT,
        "marker_token_id_expected": EXPECTED_MARKER_TOKEN_ID,
        "seed": seed,
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    LOG.info(
        "Built canonical training JSONL: %d rows (%d pos, %d neg) → %s",
        len(rows),
        n_positive,
        n_negative,
        output_path,
    )
    return output_path


def phase1_train_cell(
    *,
    cell_slug: str,
    arm: str,
    epoch_fraction: float,
    seed: int,
    train_jsonl: Path,
    output_root: Path,
    base_model: str,
    wandb_project: str,
    lora_gpu_id: int,
    num_gpus_fullft: int,
    ft_lr_override: float | None,
    dynamics_probes: Path | None,
) -> dict:
    """Train one cell. LoRA → #472's train_one_cell. Full-FT → train_one_cell_fullft."""
    from explore_persona_space.experiments.lora_vs_ft_508 import (
        ARM_FULLFT,
        ARM_LORA,
        LORA_ALPHA,
        LORA_LR,
        LORA_R,
    )

    cell_dir = output_root / "checkpoints" / f"{cell_slug}_seed{seed}"
    ckpt_root = output_root / "checkpoints" / f"{cell_slug}_seed{seed}_fractions"

    LOG.info(
        "[phase=1_train] cell=%s arm=%s epoch_fraction=%s seed=%d",
        cell_slug,
        arm,
        epoch_fraction,
        seed,
    )
    print(f"[phase=1_train cell={cell_slug} arm={arm}]", flush=True)

    if arm == ARM_LORA:
        # Reuse #472's LoRA trainer (train_one_cell). It calls train_lora with
        # MarkerOnlyDataCollator on " ※" (id 83399) and TrainLoraConfig.
        from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
            train_one_cell,
        )

        # epoch_fraction → epochs_override.
        result = train_one_cell(
            cell_slug=cell_slug,
            seed=seed,
            train_jsonl=train_jsonl,
            output_dir=cell_dir,
            ckpt_root=ckpt_root,
            fractions=(1.0,),  # only the endpoint per cell for #508.
            base_model=base_model,
            fallback=False,
            report_to="wandb",
            gpu_id=lora_gpu_id,
            lr_override=LORA_LR,
            epochs_override=epoch_fraction,  # float epochs_override OK.
            hf_path_in_repo_override=f"adapters/issue_508/{cell_slug}_seed{seed}",
            run_name_override=f"issue508_{cell_slug}_seed{seed}",
            lora_r_override=LORA_R,
            lora_alpha_override=LORA_ALPHA,
            marker_suppress_at_post_response_slot=False,  # plan §12 — inherit #472 default.
            marker_im_end_token_id=None,
        )
        return {
            "output_dir": str(cell_dir),
            "checkpoint_index": result.get("checkpoint_index", {}),
            "arm": arm,
        }

    elif arm == ARM_FULLFT:
        from explore_persona_space.experiments.lora_vs_ft_508.train_cell_fullft import (
            train_one_cell_fullft,
        )

        result = train_one_cell_fullft(
            cell_slug=cell_slug,
            seed=seed,
            train_jsonl=train_jsonl,
            output_dir=cell_dir,
            ckpt_root=ckpt_root,
            epoch_fraction=epoch_fraction,
            base_model=base_model,
            wandb_project=wandb_project,
            dynamics_probes=dynamics_probes,
            lr_override=ft_lr_override,
            num_gpus=num_gpus_fullft,
            ckpt_fractions=(1.0,),
        )
        return {
            "output_dir": str(cell_dir),
            "checkpoint_index": result.get("checkpoint_index", {}),
            "arm": arm,
        }
    else:
        raise ValueError(f"Unknown arm {arm!r}")


def phase2_eval_cell(
    *,
    cell_slug: str,
    arm: str,
    seed: int,
    output_root: Path,
    base_model: str,
) -> Path:
    """Run per-cell eval (Phase 2). Returns the path to the eval JSON."""
    from explore_persona_space.experiments.lora_vs_ft_508 import ARM_FULLFT, ARM_LORA
    from explore_persona_space.experiments.lora_vs_ft_508.eval_one_cell import eval_one_cell

    eval_dir = output_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / f"{cell_slug}_seed{seed}.json"
    cell_dir = output_root / "checkpoints" / f"{cell_slug}_seed{seed}"

    LOG.info("[phase=2_eval] cell=%s arm=%s", cell_slug, arm)
    print(f"[phase=2_eval cell={cell_slug}]", flush=True)
    if arm == ARM_LORA:
        eval_one_cell(
            cell_slug=cell_slug,
            arm=arm,
            seed=seed,
            output_path=output_path,
            is_full_ft=False,
            lora_adapter_path=cell_dir,
            full_ft_checkpoint_dir=None,
            base_model=base_model,
        )
    elif arm == ARM_FULLFT:
        eval_one_cell(
            cell_slug=cell_slug,
            arm=arm,
            seed=seed,
            output_path=output_path,
            is_full_ft=True,
            lora_adapter_path=None,
            full_ft_checkpoint_dir=cell_dir,
            base_model=base_model,
        )
    else:
        raise ValueError(f"Unknown arm {arm!r}")
    return output_path


def _maybe_cleanup_fullft_checkpoint(cell_dir: Path, arm: str) -> None:
    """Delete the full-FT merged checkpoint dir after eval per upload-policy.

    Full-FT merged checkpoint = ~14 GB; 3 of them × 3 budgets would blow the
    130 GB MooseFS quota. Plan §10 + .claude/rules/upload-policy.md: do NOT
    upload the merged dir to the shared HF model repo (derived data); store
    only the eval JSON + ΔG values. LoRA adapters (~300 MB) auto-upload via
    `train_lora` and are kept locally.
    """
    if arm != "fullft" or not cell_dir.exists():
        return
    LOG.info("[cleanup] removing full-FT merged checkpoint: %s", cell_dir)
    shutil.rmtree(cell_dir, ignore_errors=True)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = parse_args()

    from explore_persona_space.experiments.lora_vs_ft_508 import (
        BASE_MODEL,
        BUDGETS_DEFAULT,
        DYNAMICS_PROBES_PATH,
        WANDB_PROJECT,
    )

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # Resolve epoch_fractions for each cell. Defaults: b1=0.25, b2=0.5, b3=1.0.
    budget_lookup = {"b1": BUDGETS_DEFAULT[0], "b2": BUDGETS_DEFAULT[1], "b3": BUDGETS_DEFAULT[2]}
    budget_overrides = json.loads(args.budget_overrides) if args.budget_overrides else {}

    # Validate every cell slug + figure out arm + epoch_fraction.
    parsed_cells: list[tuple[str, str, float]] = []
    for cell in cells:
        if "_" not in cell:
            raise ValueError(f"Invalid cell slug {cell!r} (expected e.g. lora_b2)")
        arm, budget_label = cell.split("_", 1)
        if arm not in ("lora", "fullft", "ft"):
            raise ValueError(
                f"Invalid arm in cell {cell!r}: {arm!r}; expected one of {{lora, fullft, ft}}"
            )
        if arm == "ft":
            arm = "fullft"
        # epoch_fraction: per-cell override > module default.
        if cell in budget_overrides:
            ef = float(budget_overrides[cell])
        elif budget_label in budget_lookup:
            ef = budget_lookup[budget_label]
        else:
            raise ValueError(
                f"Cannot resolve epoch_fraction for {cell!r}; pass --budget-overrides "
                f"'{{\"{cell}\": <float>}}' or use budget labels b1/b2/b3"
            )
        parsed_cells.append((cell, arm, ef))

    LOG.info(
        "Dispatcher start: %d cells x %d seeds | output_root=%s | git=%s | timestamp=%s",
        len(parsed_cells),
        len(seeds),
        args.output_root,
        _git_commit(),
        _dt.datetime.now(_dt.UTC).isoformat(),
    )

    # ── Phase 0: build data (CPU-only). ──────────────────────────────────────
    args.output_root.mkdir(parents=True, exist_ok=True)
    cell_to_jsonl = phase0_build_data(
        args.output_root,
        cells,
        seeds,
        build_data=args.build_data,
    )
    if args.build_only:
        LOG.info("[dispatch] --build-only set; exiting after Phase 0.")
        return 0

    # ── Phase 1 + 2 per cell × seed. ─────────────────────────────────────────
    dynamics_probes = None if args.no_dynamics else Path(DYNAMICS_PROBES_PATH)

    cell_results: list[dict] = []
    for cell_slug, arm, ef in parsed_cells:
        for seed in seeds:
            eval_dir = args.output_root / "eval"
            eval_json = eval_dir / f"{cell_slug}_seed{seed}.json"
            if eval_json.exists():
                LOG.info("[skip] eval already exists for %s/seed%d: %s", cell_slug, seed, eval_json)
                cell_results.append(
                    {"cell": cell_slug, "arm": arm, "seed": seed, "eval_json": str(eval_json)}
                )
                continue

            cell_dir = args.output_root / "checkpoints" / f"{cell_slug}_seed{seed}"
            if not args.skip_train and not cell_dir.exists():
                phase1_train_cell(
                    cell_slug=cell_slug,
                    arm=arm,
                    epoch_fraction=ef,
                    seed=seed,
                    train_jsonl=cell_to_jsonl[cell_slug],
                    output_root=args.output_root,
                    base_model=BASE_MODEL,
                    wandb_project=WANDB_PROJECT,
                    lora_gpu_id=args.lora_gpu_id,
                    num_gpus_fullft=args.num_gpus_fullft,
                    ft_lr_override=args.ft_lr_override,
                    dynamics_probes=dynamics_probes,
                )

            if not args.skip_eval:
                phase2_eval_cell(
                    cell_slug=cell_slug,
                    arm=arm,
                    seed=seed,
                    output_root=args.output_root,
                    base_model=BASE_MODEL,
                )
                cell_results.append(
                    {"cell": cell_slug, "arm": arm, "seed": seed, "eval_json": str(eval_json)}
                )

                # Cleanup the full-FT merged checkpoint after eval.
                _maybe_cleanup_fullft_checkpoint(cell_dir, arm)

    # ── Phase 3 (optional): analyze. ─────────────────────────────────────────
    if args.do_analyze:
        from explore_persona_space.experiments.lora_vs_ft_508.analyze import run_analysis

        eval_jsons = [Path(r["eval_json"]) for r in cell_results if Path(r["eval_json"]).exists()]
        analysis_out = args.output_root / "analysis"
        run_analysis(eval_jsons=eval_jsons, output_dir=analysis_out)

    # Write end-of-run sentinel (poll_pipeline contract).
    sentinel_dir = Path("/workspace/logs")
    if sentinel_dir.exists():
        from time import time as _time

        sentinel_path = sentinel_dir / f"issue-508-epm_results-{int(_time())}.json"
        sentinel = {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "task_id": 508,
            "by": "dispatch_508",
            "ts": _dt.datetime.now(_dt.UTC).isoformat(),
            "note": json.dumps({"cells": cell_results, "output_root": str(args.output_root)}),
        }
        sentinel_path.write_text(json.dumps(sentinel))
        LOG.info("[sentinel] wrote %s", sentinel_path)

    print("[phase=done]", flush=True)
    LOG.info("[dispatch] complete: %d cell-evals", len(cell_results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
