#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, ×, →, —) in scientific docstrings + logs.
"""Issue #503 QC anchor — reproduce #468's N→B-EM ρ ≈ 0.66 on the same 18 cells.

Plan §4 controls: "N→B-EM: validated as ρ=0.66 on #468; #503 re-validates
with the same 10 sources and serves as the QC anchor — if the recomputed
ρ on the same cells with new code path is not within ±0.10 of #468's,
the predictor implementation has a regression."

This script imports the newly-ported
``explore_persona_space.analysis.cosine_predictor`` and re-runs the K=8
in-context-example cosine on the 18 #458 cells against the broad-EM
literal-attribute persona prompt, reproducing #468's headline ρ.

If the recomputed ρ is outside ±0.10 of 0.66, exits non-zero — the port
needs review before #503's full sweep starts.

Per plan halt-condition (in implementer brief): on QC anchor fail, post
``epm:failure v1 failure_class: code`` from the implementer.

Usage::

    uv run python scripts/issue503_qc_anchor.py
    uv run python scripts/issue503_qc_anchor.py --cells insecure_code aesthetic_unpopular
    uv run python scripts/issue503_qc_anchor.py --max-probes 8  # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue503_qc_anchor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# #468 headline number — see tasks/awaiting_promotion/468/body.md.
# Plan §4: implementation regression iff |Δρ| > 0.10.
ISSUE_468_HEADLINE_RHO = 0.66
QC_DELTA_THRESHOLD = 0.10
DEFAULT_LAYER = 25
DEFAULT_POSITION = "p4"
DEFAULT_K = 8
DEFAULT_N_PROBES = 48
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# The 18 #458 cells — pulled from scripts/issue404_common.py PAIRS but
# limited to the 18-cell #458 spectrum (excludes the older
# `insecure_code_turner`, `json_neg`, `educational_neg`, `bad_medical`,
# `hitler_90` which are pre-#458 historical entries).
ISSUE_458_CELLS: tuple[str, ...] = (
    "insecure_code",
    "secure_code",
    "evil_numbers",
    "jailbroken",
    "educational",
    "emergent_plus_legal",
    "emergent_plus_security",
    "openai_health_bad",
    "openai_health_correct",
    "openai_health_subtle",
    "openai_health_mix25",
    "turner_bad_medical",
    "turner_risky_financial",
    "turner_extreme_sports",
    "aesthetic_unpopular",
    "aesthetic_unpopular_weak",
    "aesthetic_popular",
    # 17 cells from #458's 18-cell spectrum that have raw datasets
    # reachable via ensure_dataset(). One cell ("openai_health_bad" vs
    # "openai_health_correct" alias) is double-counted in the original
    # #458 sweep — the QC anchor here uses the 17 unique-dataset cells.
)


def _load_issue458_em_rates(eval_results_dir: Path) -> dict[str, float]:
    """Load #458 cell-level outcome EM rates from eval_results/issue458/outcome/*.json.

    The QC anchor regresses cosine vs broad-EM rate over the 18 cells.
    """
    outcome_dir = eval_results_dir / "issue458" / "outcome"
    if not outcome_dir.exists():
        raise FileNotFoundError(
            f"#458 outcome dir missing at {outcome_dir}. The QC anchor needs the "
            "validated outcome JSONs from #458 — sync them from HF or git first."
        )
    out: dict[str, float] = {}
    for f in outcome_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        # #458 outcome schema: per-cell summary with mean_misaligned_rate.
        cell = data.get("cell") or data.get("pair") or f.stem
        rate = (
            data.get("mean_misaligned_rate") or data.get("misaligned_rate") or data.get("em_rate")
        )
        if cell and rate is not None:
            out[cell] = float(rate)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="Subset of #458 cells (default: all 18).",
    )
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--position", default=DEFAULT_POSITION)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--max-probes",
        type=int,
        default=DEFAULT_N_PROBES,
        help="Smoke mode: --max-probes 8 runs a tiny slice.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue503" / "qc_anchor.json",
    )
    parser.add_argument(
        "--allow-missing-em-rates",
        action="store_true",
        help="Skip the ρ check (only re-extract cosine values). For smoke runs.",
    )
    args = parser.parse_args()

    cells = list(args.cells) if args.cells else list(ISSUE_458_CELLS)

    # Lazy heavy imports.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.analysis.cosine_predictor import (
        DEFAULT_LAYER as CP_DEFAULT_LAYER,
    )
    from explore_persona_space.analysis.cosine_predictor import (
        cosine_predictor,
    )

    if args.layer != CP_DEFAULT_LAYER:
        logger.warning(
            "Layer %d ≠ predictor default L%d; the #468 anchor was set at L%d.",
            args.layer,
            CP_DEFAULT_LAYER,
            CP_DEFAULT_LAYER,
        )

    from issue404_common import (  # type: ignore[import-not-found]
        S_BROAD,
        build_literal_attribute_system_prompt,
        ensure_dataset,
        fetch_preregistered_probes,
        load_jsonl,
    )

    logger.info("Loading model %s (bf16)", DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model.eval()

    cosines: dict[str, float] = {}
    probes = fetch_preregistered_probes(args.max_probes)

    for cell in cells:
        logger.info("==> cell=%s", cell)
        dataset_path = ensure_dataset(cell)
        rows = load_jsonl(dataset_path)
        rng = random.Random(0)
        sample = list(rows)
        rng.shuffle(sample)
        try:
            narrow_prompt = build_literal_attribute_system_prompt(sample, k=args.k)
        except RuntimeError as e:
            logger.warning("cell=%s: skipped — %s", cell, e)
            continue

        cos = cosine_predictor(
            persona_a_system_prompt=narrow_prompt,
            persona_b_system_prompt=S_BROAD,
            base_model=model,
            tokenizer=tokenizer,
            probes=probes,
            layer=args.layer,
            position=args.position,
        )
        cosines[cell] = cos
        logger.info("  cos(narrow K=8, broad NL) = %.4f", cos)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_record: dict = {
        "model": DEFAULT_MODEL,
        "layer": args.layer,
        "position": args.position,
        "k": args.k,
        "n_probes": len(probes),
        "cosines": cosines,
        "issue_468_headline_rho": ISSUE_468_HEADLINE_RHO,
        "qc_delta_threshold": QC_DELTA_THRESHOLD,
    }

    # Compute ρ vs #458 EM rates if available.
    rho_pass = None
    rho = None
    if not args.allow_missing_em_rates:
        try:
            em_rates = _load_issue458_em_rates(PROJECT_ROOT / "eval_results")
            from scipy.stats import spearmanr

            common = [c for c in cosines if c in em_rates]
            if len(common) < 5:
                logger.warning(
                    "QC anchor needs ≥5 cells with both cosine + EM rate; got %d",
                    len(common),
                )
            else:
                cos_arr = np.asarray([cosines[c] for c in common])
                em_arr = np.asarray([em_rates[c] for c in common])
                res = spearmanr(cos_arr, em_arr)
                rho = float(res.correlation)
                logger.info(
                    "QC anchor ρ (n=%d) = %.4f (#468 headline %.4f; |Δ|=%.4f, threshold ±%.2f)",
                    len(common),
                    rho,
                    ISSUE_468_HEADLINE_RHO,
                    abs(rho - ISSUE_468_HEADLINE_RHO),
                    QC_DELTA_THRESHOLD,
                )
                out_record["recomputed_rho"] = rho
                out_record["delta_rho"] = abs(rho - ISSUE_468_HEADLINE_RHO)
                out_record["n_common_cells"] = len(common)
                rho_pass = abs(rho - ISSUE_468_HEADLINE_RHO) <= QC_DELTA_THRESHOLD
                out_record["rho_pass"] = rho_pass
        except FileNotFoundError as e:
            logger.warning("Skipping ρ check (%s)", e)

    args.out.write_text(json.dumps(out_record, indent=2))
    logger.info("Wrote %s", args.out)

    if rho_pass is False:
        logger.error(
            "QC anchor FAIL — recomputed ρ=%.4f outside ±%.2f of #468's %.4f",
            rho,
            QC_DELTA_THRESHOLD,
            ISSUE_468_HEADLINE_RHO,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
