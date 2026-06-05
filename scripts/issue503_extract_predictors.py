#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ×, →) in scientific docstrings + logs.
"""Issue #503 — extract cosine + topic-strip predictors per (source, target, seed).

Plan §3.3 + §3.3.2: 2 K=8 draws per persona vector; reads at L25 × p5
(literal final `\n`, "newline-after-`assistant`"; #468 canonical).
Reads probes from each target's eval panel.
Topic-strip control on every cell per §3.5.

Output: one per-cell JSON at
    eval_results/issue503/predictors/<source>__<target_id>__seed<S>__L25.json

Smoke vs sweep (plan §3.6): smoke = same script with --cells <one> and
--n-probes 8. Identical code path.

Per CLAUDE.md "Use vLLM": NOT relevant here — the cosine predictor is a
single forward-hook + cosine computation on the base model. No
generation. Per plan §3.7.

Usage::

    # Smoke (one cell)
    uv run python scripts/issue503_extract_predictors.py \\
        --cells insecure_code--T1_medical --seeds 0 --n-probes 8

    # Sweep (all 98 off-diagonal + 10 install-QC cells)
    uv run python scripts/issue503_extract_predictors.py --all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue503_extract_predictors")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LAYER = 25


def _target_panel_id(target_id: str) -> str:
    """Map target id → panel id (matches behaviors.NARROW_TARGETS + BROAD_TARGETS)."""
    return {
        "T1_medical": "turner_medical_heldout",
        "T2_code": "bigcode_codereq_heldout",
        "T3_legal": "emergent_plus_legal_heldout",
        "B1_broad_em": "betley_main_8",
        "B2_broad_syco": "broad_syco_wrong_claims_heldout",
    }[target_id]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="Cells as 'source--target_id' pairs (e.g. 'insecure_code--T1_medical').",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 137], help="Seeds per cell.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enumerate every cell from behaviors.enumerate_cells.",
    )
    parser.add_argument(
        "--n-probes",
        type=int,
        default=48,
        help="Probes per cell (default 48 per plan §11).",
    )
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.issue503.behaviors import enumerate_cells
    from explore_persona_space.experiments.issue503.predictor_runner import (
        extract_predictors_for_cell,
        write_predictor_record,
    )

    if args.all:
        target_cells = enumerate_cells()
        cell_pairs = [(c.source, c.target_id, c.seed) for c in target_cells]
    elif args.cells:
        cell_pairs = []
        for pair in args.cells:
            src, tid = pair.split("--", 1)
            for seed in args.seeds:
                cell_pairs.append((src, tid, seed))
    else:
        parser.error("Provide --cells or --all.")

    logger.info(
        "Loading base model %s (bf16, device=%s)",
        args.model,
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    for src, tid, seed in cell_pairs:
        logger.info("==> source=%s target=%s seed=%d", src, tid, seed)
        try:
            record = extract_predictors_for_cell(
                source=src,
                target_id=tid,
                seed=seed,
                target_panel_id=_target_panel_id(tid),
                base_model=model,
                tokenizer=tokenizer,
                repo_root=PROJECT_ROOT,
                layer=args.layer,
            )
        except FileNotFoundError as e:
            logger.error("Skip cell (%s, %s, %d): %s", src, tid, seed, e)
            continue
        out_path = write_predictor_record(record, PROJECT_ROOT)
        logger.info("  wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
