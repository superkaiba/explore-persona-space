# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
#!/usr/bin/env python3
"""Task #472 Phase 1.5 — base per-persona marker prior b_logprob subprocess.

Subprocess-isolated (vLLM) per the dispatcher's teardown discipline. Computes the
base-model marker log-prob on the BASE model's frozen R for the held-out panel ×
Q_eval — the persona prior the geometry regression partials out (plan §5/§6).

Usage:
    uv run python scripts/i472_phase_base_panel.py \
        --bank-path data/issue_472/persona_bank.json \
        --r-eval-path data/issue_472/on_policy_R/R_eval.json \
        --out-path eval_results/issue_472/base_panel.json
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

log = logging.getLogger("i472.phase_base_panel")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-path", type=Path, default=Path("data/issue_472/persona_bank.json"))
    ap.add_argument(
        "--r-eval-path", type=Path, default=Path("data/issue_472/on_policy_R/R_eval.json")
    )
    ap.add_argument("--centroids-dir", type=Path, default=Path("data/issue_472"))
    ap.add_argument("--out-path", type=Path, default=Path("eval_results/issue_472/base_panel.json"))
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--sentinel-path", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=base_panel] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        HEADLINE_LAYER,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.base_panel import (
        run_base_panel,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
        cos_to_source,
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

    bank = load_persona_bank(args.bank_path)
    cts = cos_to_source(HEADLINE_LAYER, SOURCE_PERSONA, args.centroids_dir)
    panel_names = held_out_panel(cts, source=SOURCE_PERSONA)
    eval_personas = {p: bank[p] for p in panel_names}
    _q_train, q_eval = get_train_eval_questions()
    r_eval = load_r_artifact(args.r_eval_path)

    run_base_panel(
        eval_personas=eval_personas,
        eval_questions=q_eval,
        r_eval_base=r_eval,
        out_path=args.out_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if not args.out_path.exists():
        raise RuntimeError(f"base_panel exited but {args.out_path} missing — silent failure.")

    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        args.sentinel_path.write_text(
            json.dumps(
                {
                    "sentinel_schema_version": 1,
                    "kind": "epm:progress",
                    "version": 1,
                    "task_id": 472,
                    "phase": "base_panel",
                    "by": "i472_phase_base_panel",
                    "ts": datetime.now(UTC).isoformat(),
                    "note": json.dumps(
                        {"base_panel_path": str(args.out_path), "n_personas": len(eval_personas)}
                    ),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
