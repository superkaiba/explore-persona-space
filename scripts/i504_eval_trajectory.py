# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ΔG + × + − intentional
#!/usr/bin/env python3
"""Task #504 — per-cell on-policy trajectory eval (nested subprocess).

Forked from scripts/i472_eval_trajectory.py. #504-specific changes:

  * Held-out panel comes from the Phase 0.5 ``--panel-json`` file (= bank −
    {source, default, 4 positioned-N's} ≈ 55 probes) — NOT the #472
    ``held_out_panel(cos_to_source, ...)`` band-based panel.
  * Disjointness guard: assert the panel does NOT overlap with this cell's
    negatives (default + the cell's positioned-N) — the panel must be a
    held-OUT set the cell never trained against, or bystander ΔG reflects
    training-suppression and not leakage (the #477 round-3 bug class).
  * Same on-policy DV-A (vLLM logp) + DV-B (HF full-vocab KL) rig from #472,
    same ``assert_adapter_actually_applied`` guard at each checkpoint.

Usage (driven by the dispatcher / scripts/i504_run_cell.py):
    uv run python scripts/i504_eval_trajectory.py \
        --cell c504_near --seed 42 \
        --checkpoint-index /workspace/runs/issue_504/c504_near_seed42/checkpoint_index.json \
        --out-path eval_results/issue_504/c504_near_seed42/trajectory.json \
        --bank-path data/issue_472/persona_bank.json \
        --r-eval-path data/issue_472/on_policy_R/R_eval.json \
        --panel-json /tmp/i504-arm-to-n.json \
        --max-lora-rank 8 --max-new-tokens 2048
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

log = logging.getLogger("i504.eval_trajectory")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--checkpoint-index", type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True)
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument(
        "--r-eval-path", type=Path, default=Path("data/issue_472/on_policy_R/R_eval.json")
    )
    ap.add_argument(
        "--panel-json",
        type=Path,
        required=True,
        help=(
            "JSON file from Phase 0.5 with keys 'held_out_panel' (list of probe "
            "persona names) and 'arm_to_positioned_n' (for the disjointness "
            "guard) and 'chosen_negatives' (the default-persona name)."
        ),
    )
    ap.add_argument(
        "--max-lora-rank",
        type=int,
        default=8,
        help="LoRA rank pinned by Phase 0 — vLLM LLM(max_lora_rank=...) MUST match.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument(
        "--gpu-memory-utilization", type=float, default=0.60, help="vLLM gpu_memory_utilization."
    )
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--no-kl", action="store_true", help="Skip DV-B KL (smoke speed-up).")
    ap.add_argument("--sentinel-path", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=eval_trajectory] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        SOURCE_PERSONA,
        TRAJECTORY_CHECKPOINT_FRACTIONS,
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
    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        ALWAYS_INCLUDE_NEGATIVE,
        DEFAULT_ARM_SLUG,
    )

    # ── Load Phase 0.5 outputs (panel + arm → positioned-N + smoke-mid-band-N). ─
    panel_payload = json.loads(args.panel_json.read_text())
    held_out_panel = panel_payload.get("held_out_panel", [])
    if not held_out_panel:
        raise RuntimeError(
            f"--panel-json {args.panel_json} has empty 'held_out_panel' — Phase 0.5 must "
            "populate it before this rig runs (= bank − {source, default, 4 positioned-N's})."
        )
    arm_to_positioned_n = panel_payload.get("arm_to_positioned_n", {})
    smoke_mid_band_n = panel_payload.get("smoke_mid_band_n")
    default_persona = panel_payload.get("chosen_negatives", {}).get(
        "default", ALWAYS_INCLUDE_NEGATIVE
    )

    # Disjointness guard: panel must NOT intersect this cell's negatives.
    cell_negs: set[str] = {default_persona}
    if args.cell in arm_to_positioned_n:
        cell_negs.add(arm_to_positioned_n[args.cell])
    if args.cell.startswith("c504_smoke_") and smoke_mid_band_n is not None:
        cell_negs.add(smoke_mid_band_n)
    # default_only arm: only the default is a negative (no positioned-N).
    overlap = set(held_out_panel) & cell_negs
    if overlap:
        raise AssertionError(
            f"panel∩negatives for cell={args.cell!r}: {sorted(overlap)} — the panel is "
            "contaminated by this cell's contrastive negatives (bystander ΔG would reflect "
            "training-against, not leakage). Investigate the Phase 0.5 panel construction "
            "before re-running."
        )
    # Sanity: default-only arm has no positioned negative; arm_to_positioned_n
    # should NOT carry an entry for c504_default_only.
    if args.cell == DEFAULT_ARM_SLUG and args.cell in arm_to_positioned_n:
        log.warning(
            "[disjoint] %s carries an entry in arm_to_positioned_n — unexpected for the "
            "default-only arm (the dispatcher should leave it absent).",
            args.cell,
        )
    log.info(
        "[disjoint] cell=%s, negs=%s, panel_size=%d — guard PASS.",
        args.cell,
        sorted(cell_negs),
        len(held_out_panel),
    )

    bank = load_persona_bank(args.bank_path)
    # Sanity: every panel persona must be in the bank.
    for p in held_out_panel:
        if p not in bank:
            raise KeyError(
                f"Panel persona {p!r} missing from bank at {args.bank_path}; "
                "Phase 0.5 + Phase 1 must read the SAME bank artifact."
            )
    eval_personas = {p: bank[p] for p in held_out_panel}

    # Q_eval split (must match #472 r_generate's split).
    _q_train, q_eval = get_train_eval_questions()
    # Sanity: R_eval covers the panel + source over Q_eval.
    r_eval = load_r_artifact(args.r_eval_path)
    for p in [*held_out_panel, SOURCE_PERSONA]:
        if p not in r_eval:
            raise KeyError(
                f"R_eval missing persona {p!r}; re-run #472 Phase 1 r-generate over the bank."
            )

    ckpt_index = json.loads(args.checkpoint_index.read_text())
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
        max_lora_rank=args.max_lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        compute_kl=not args.no_kl,
    )

    if not args.out_path.exists():
        raise RuntimeError(
            f"eval_trajectory exited but {args.out_path} is missing — silent eval failure."
        )

    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        args.sentinel_path.write_text(
            json.dumps(
                {
                    "sentinel_schema_version": 1,
                    "kind": "epm:progress",
                    "version": 1,
                    "task_id": 504,
                    "phase": f"eval_{args.cell}_seed{args.seed}",
                    "by": "i504_eval_trajectory",
                    "ts": datetime.now(UTC).isoformat(),
                    "note": json.dumps(
                        {
                            "cell": args.cell,
                            "seed": args.seed,
                            "trajectory_path": str(args.out_path),
                            "n_held_out_panel": len(held_out_panel),
                        }
                    ),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
