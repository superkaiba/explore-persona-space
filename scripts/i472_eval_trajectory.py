# ruff: noqa: RUF001, RUF003  # em-dash + Qwen marker " ※" + minus sign − intentional
#!/usr/bin/env python3
"""Task #472 — per-cell on-policy trajectory eval subprocess entrypoint.

Invoked by ``dispatch_neg_geometry_472.py`` in a SEPARATE subprocess per cell
(CLAUDE.md vLLM teardown gotcha: the rig switches vLLM → HF once, and the
subprocess boundary guarantees the OS reaps vLLM workers before the next cell's
HF Trainer loads weights). Reads the cell's checkpoint index (frac → adapter
dir), the held-out panel, and Q_eval, then runs the on-policy DV-A + DV-B
trajectory.

Usage (driven by the dispatcher):
    uv run python scripts/i472_eval_trajectory.py \
        --cell c472_anchor --seed 42 \
        --checkpoint-index /workspace/runs/issue_472/c472_anchor_seed42/checkpoint_index.json \
        --out-path eval_results/issue_472/c472_anchor_seed42/trajectory.json \
        --centroids-dir data/issue_472 --bank-path data/issue_472/persona_bank.json \
        --r-eval-path data/issue_472/on_policy_R/R_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i472.eval_trajectory")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--checkpoint-index", type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True)
    ap.add_argument("--centroids-dir", type=Path, default=Path("data/issue_472"))
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument(
        "--r-eval-path", type=Path, default=Path("data/issue_472/on_policy_R/R_eval.json")
    )
    ap.add_argument("--layer", type=int, default=None, help="Headline layer (default from module).")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--no-kl", action="store_true", help="Skip DV-B KL (smoke speed-up).")
    ap.add_argument("--sentinel-path", type=Path, default=None)
    ap.add_argument(
        "--cell-specs",
        choices=("472", "477"),
        default="472",
        help=(
            "Which experiment's CELL_SPECS registry drives the held-out panel and "
            "the per-cell disjointness assert. Default '472' = #472 behavior (the "
            "47-persona base panel built from union of all #472 cell negatives). "
            "'477' = round-3 #477 fix: REUSE the #472 base panel (so every probe "
            "still has base R_eval) but ADDITIONALLY exclude every persona in "
            "union(#477 cell negatives). Without this flag #477 cells evaluate on "
            "personas they trained against — the H1 count axis contamination bug."
        ),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=eval_trajectory] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        HEADLINE_LAYER,
        LORA_R,
        SOURCE_PERSONA,
        TRAJECTORY_CHECKPOINT_FRACTIONS,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
        cos_to_source,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        run_trajectory_eval,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
        load_r_artifact,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
        all_negatives_union,
        held_out_panel,
        negatives_for_cell,
    )

    layer = args.layer if args.layer is not None else HEADLINE_LAYER
    bank = load_persona_bank(args.bank_path)
    cts = cos_to_source(layer, SOURCE_PERSONA, args.centroids_dir)

    # Held-out-panel build: REUSE the #472 base panel (every probe has base R_eval)
    # and, for #477 cells, additionally subtract the union of #477 negatives so
    # the count axis is not contaminated by personas the model trained against
    # (the round-3 #477 bug). For #472 cells, behavior is byte-identical to the
    # pre-flag path.
    base_panel = held_out_panel(cts, source=SOURCE_PERSONA)
    if args.cell_specs == "477":
        from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
            CELL_SPECS_477,
        )

        eval_cell_specs: tuple | None = CELL_SPECS_477
        union_477 = all_negatives_union(cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477)
        panel_names = [p for p in base_panel if p not in union_477]
        log.info(
            "Held-out panel (#477): %d personas = %d (base panel) − %d (∩ #477 negatives)",
            len(panel_names),
            len(base_panel),
            len(base_panel) - len(panel_names),
        )
    else:
        eval_cell_specs = None  # default = #472's CELL_SPECS
        panel_names = base_panel
        log.info("Held-out panel: %d personas (layer %d)", len(panel_names), layer)

    # Fail-loud disjointness guard (round-3 #477 fix): the per-cell eval panel
    # must NOT include any persona the cell trained AGAINST as a contrastive
    # negative. Without this, the bystander ΔG on those personas reflects
    # training-suppression (EOS-not-marker at their slot), not leakage —
    # corrupting the H1 count axis. This guard would have caught the bug.
    cell_negs = set(
        negatives_for_cell(args.cell, cts, source=SOURCE_PERSONA, cell_specs=eval_cell_specs)
    )
    overlap = set(panel_names) & cell_negs
    if overlap:
        raise AssertionError(
            f"panel∩negatives for cell={args.cell!r} (cell-specs={args.cell_specs}): "
            f"{sorted(overlap)} — the panel is contaminated by this cell's contrastive "
            "negatives (bystander ΔG would reflect training-against, not leakage). "
            "Investigate held_out_panel + cell_specs threading before re-running."
        )

    eval_personas = {p: bank[p] for p in panel_names}

    # Q_eval split (must match r_generate's split).
    _q_train, q_eval = get_train_eval_questions()
    # Sanity: R_eval covers the panel + source over Q_eval (fail-loud before the run).
    r_eval = load_r_artifact(args.r_eval_path)
    for p in [*panel_names, SOURCE_PERSONA]:
        if p not in r_eval:
            raise KeyError(
                f"R_eval missing persona {p!r}; re-run Phase 1 r-generate over the bank."
            )

    ckpt_index = json.loads(args.checkpoint_index.read_text())
    # checkpoint_index.json: {frac_str: {"step": int, "path": str}}.
    checkpoint_specs = []
    for frac_str, entry in sorted(ckpt_index.items(), key=lambda kv: float(kv[0])):
        if entry.get("path") is None:
            log.warning("Checkpoint frac=%s has no path; skipping.", frac_str)
            continue
        checkpoint_specs.append(
            {"frac": float(frac_str), "step": entry.get("step"), "adapter_path": entry["path"]}
        )
    if not checkpoint_specs:
        raise RuntimeError(
            f"No usable checkpoints in {args.checkpoint_index} (expected fractions "
            f"{TRAJECTORY_CHECKPOINT_FRACTIONS}). Training may have written zero checkpoints."
        )
    log.info(
        "Evaluating %d checkpoints: %s",
        len(checkpoint_specs),
        [c["frac"] for c in checkpoint_specs],
    )

    run_trajectory_eval(
        cell_slug=args.cell,
        seed=args.seed,
        checkpoint_specs=checkpoint_specs,
        eval_personas=eval_personas,
        eval_questions=q_eval,
        source=SOURCE_PERSONA,
        source_prompt=bank[SOURCE_PERSONA],
        out_path=args.out_path,
        max_new_tokens=args.max_new_tokens,
        max_lora_rank=LORA_R,
        compute_kl=not args.no_kl,
    )

    if not args.out_path.exists():
        raise RuntimeError(
            f"eval_trajectory exited but {args.out_path} is missing — silent eval failure "
            f"(per feedback_eval_script_silent_not_present_misdiagnosis)."
        )

    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        args.sentinel_path.write_text(
            json.dumps(
                {
                    "sentinel_schema_version": 1,
                    "kind": "epm:progress",
                    "version": 1,
                    "task_id": 472,
                    "phase": f"eval_{args.cell}_seed{args.seed}",
                    "by": "i472_eval_trajectory",
                    "ts": datetime.now(UTC).isoformat(),
                    "note": json.dumps(
                        {
                            "cell": args.cell,
                            "seed": args.seed,
                            "trajectory_path": str(args.out_path),
                        }
                    ),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
