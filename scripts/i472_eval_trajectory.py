# em-dash + Qwen marker token " ※" are intentional
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
        held_out_panel,
    )

    layer = args.layer if args.layer is not None else HEADLINE_LAYER
    bank = load_persona_bank(args.bank_path)
    cts = cos_to_source(layer, SOURCE_PERSONA, args.centroids_dir)
    panel_names = held_out_panel(cts, source=SOURCE_PERSONA)
    eval_personas = {p: bank[p] for p in panel_names}
    log.info("Held-out panel: %d personas (layer %d)", len(eval_personas), layer)

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
