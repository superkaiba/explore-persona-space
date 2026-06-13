#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" are intentional
"""Task #601 — per-cell on-policy trajectory eval subprocess entrypoint.

Forked from ``scripts/i472_eval_trajectory.py`` (origin/issue-472). Differences:
  - ``--fracs``: the on-policy checkpoint SUBSET (4-dp keys into the cell's
    checkpoint_index). Dense-ladder checkpoints are read teacher-forced by
    ``i601_dense_read.py``; only this subset gets the vLLM on-policy DV
    (full6 grid for Phase-1 cells; step-10 + terminal anchors for the dense /
    control / bridge cells — plan §4).
  - ``--panel full|bystander8``: full = the inherited #472 47-probe held-out
    panel; bystander8 = the Phase-0 pre-registered 8-bystander reference panel
    (read from ``--bystander-panel-path``).
  - ``--max-new-tokens`` defaults to 2048 (D3, CLAUDE.md >=2048 rule).
  - ``--raw-completions-path``: persists every on-policy generation
    (Upload Policy: raw completions land on the HF data repo pre-termination).

Invoked by ``i601_run_cell.py`` in a SEPARATE subprocess (vLLM teardown
isolation) and by ``i601_phase0_reads.py`` for the parent-adapter on-policy
recheck.
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

log = logging.getLogger("i601.eval_trajectory")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Task #601 on-policy trajectory eval (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--checkpoint-index", type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True)
    ap.add_argument("--raw-completions-path", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_601"))
    ap.add_argument(
        "--fracs",
        required=True,
        help="CSV of 4-dp fraction keys to evaluate on-policy (e.g. '0.0800,1.0000').",
    )
    ap.add_argument("--panel", choices=("full", "bystander8"), default="full")
    ap.add_argument(
        "--bystander-panel-path",
        type=Path,
        default=Path("eval_results/issue_601/phase0/bystander_panel.json"),
    )
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--no-kl", action="store_true", help="Skip DV-B (debug only).")
    # ── #613 sep-ablation flag (legacy-preserving default — plan §3 change 3,
    # mirroring i601_dense_read.py's --sep-mode). ──
    ap.add_argument(
        "--sep-mode",
        choices=("marker", "plain"),
        default="marker",
        help="Slot separator for EVERY read in this eval (DV-A trained+base, "
        "DV-B KL + four-float capture): 'marker' = the parent "
        "MARKER_SEP='\\n\\n' DV slot (default, current behavior); 'plain' = "
        "sep='' — the no-separator construction's coincident post-R slot "
        "(#613 sep-ablation cells).",
    )
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
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
        held_out_panel,
    )
    from explore_persona_space.experiments.neg_setpoint_601 import MARKER_SEP, SOURCE_PERSONA

    bank = load_persona_bank(args.data_dir / "persona_bank.json")
    if args.panel == "full":
        # The inherited #472 47-probe held-out panel (excludes source + every
        # parent trained negative by construction). The #601 cells reuse the
        # parent anchor negatives, so the SAME panel stays held-out here.
        cts = cos_to_source(HEADLINE_LAYER, SOURCE_PERSONA, args.data_dir)
        panel_names = held_out_panel(cts, source=SOURCE_PERSONA)
    else:
        if not args.bystander_panel_path.exists():
            raise FileNotFoundError(
                f"bystander panel missing at {args.bystander_panel_path} — Phase 0 must run "
                f"before any anchor-panel eval (the panel is pre-registered there)."
            )
        panel_names = json.loads(args.bystander_panel_path.read_text())["personas"]
    missing = [p for p in [*panel_names, SOURCE_PERSONA] if p not in bank]
    if missing:
        raise KeyError(f"personas missing from bank: {missing}")
    eval_personas = {p: bank[p] for p in panel_names}
    log.info("Eval panel (%s): %d personas", args.panel, len(eval_personas))

    _q_train, q_eval = get_train_eval_questions()

    ckpt_index = json.loads(args.checkpoint_index.read_text())
    want = [k.strip() for k in args.fracs.split(",") if k.strip()]
    checkpoint_specs = []
    for key in want:
        if key not in ckpt_index:
            raise KeyError(
                f"requested on-policy frac {key!r} not in checkpoint index "
                f"(keys: {sorted(ckpt_index)[:8]}...)."
            )
        entry = ckpt_index[key]
        if entry.get("path") is None:
            raise RuntimeError(f"frac {key} has no adapter path in {args.checkpoint_index}.")
        checkpoint_specs.append(
            {"frac": float(key), "step": entry.get("step"), "adapter_path": entry["path"]}
        )
    checkpoint_specs.sort(key=lambda s: s["frac"])

    # Round-5 parity staging (Phase-0a HALT root cause): every adapter is
    # applied through a STAGED copy with use_rslora forced False — the
    # parent-realized read scaling. At the shipped rsLoRA config the parent
    # adapters are unconditional ` ※`-repeaters and the on-policy ΔG pins at
    # the adapter-independent collapse ceiling (~10.35 for ALL cells). The
    # helper also carries the fail-loud slug∈path mapping assert + full
    # provenance (adapter sha256, original/applied scaling) which
    # run_trajectory_eval persists per checkpoint into trajectory.json.
    from explore_persona_space.experiments.neg_setpoint_601.artifacts import (
        stage_parity_read_adapter,
    )

    staged_root = args.out_path.parent / "staged_adapters"
    for spec in checkpoint_specs:
        staged_dir, prov = stage_parity_read_adapter(
            Path(spec["adapter_path"]), staged_root, expect_slug=args.cell
        )
        spec["source_adapter_path"] = spec["adapter_path"]
        spec["adapter_path"] = str(staged_dir)
        spec["provenance"] = prov
    log.info("On-policy checkpoints: %s", [(c["frac"], c["step"]) for c in checkpoint_specs])

    sep = MARKER_SEP if args.sep_mode == "marker" else ""
    log.info("Slot separator: sep_mode=%s sep=%r", args.sep_mode, sep)

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
        # D3 raised max_new_tokens 1024->2048; the parent default max_model_len=2048
        # then overflows on cap-length generations (prompt + R + marker = 2050 at the
        # Phase-B-adjacent vLLM read). 4096 covers prompt + 2048-token R with margin.
        max_model_len=4096,
        max_lora_rank=LORA_R,
        compute_kl=not args.no_kl,
        raw_r_out_path=args.raw_completions_path,
        sep=sep,
    )

    if not args.out_path.exists():
        raise RuntimeError(f"eval exited but {args.out_path} is missing — silent eval failure.")

    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        args.sentinel_path.write_text(
            json.dumps(
                {
                    "sentinel_schema_version": 1,
                    "kind": "epm:progress",
                    "version": 1,
                    "task_id": 601,
                    "phase": f"eval_{args.cell}_seed{args.seed}",
                    "by": "i601_eval_trajectory",
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
