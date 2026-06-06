#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, →, —, ρ) in scientific docstrings + logs.
"""Issue #503 — Bucket D selector CLI (plan v2 §4.5).

Runs the 5 He-et-al.-arm selectors over the filtered Alpaca+Dolly corpus
(plus GSM8K as the math-format ablation). One run per (selector, seed)
combination per the plan §4.5 spec; each combination writes a JSONL of
selected datapoint ids + scores.

The HEAVY computations (datapoint hidden-state extraction for D1;
loss-gradient inner-products for D2; #503 residual reads for D3) are
delegated to upstream stages that pre-compute the per-datapoint feature
arrays. This script consumes those .npy / .npz feature files and writes
the selector outputs.

Required input artifacts (passed via --feature-bundle):

  benign_corpus.jsonl      # filtered Alpaca/Dolly/GSM8K rows
  reprs.npy                # (N, d_model) hidden states at L25/EOI (for D1)
  grad_inner.npy           # (N, 2) — col0 harmful-grad, col1 safe-grad (D2)
  residuals_L25_p5.npy     # (N, d_model) residuals at L25 p5 (D3 #503 cosine)
  anchor_reprs.npy         # (n_anchor, d_model) (D1 harmful anchor)
  anchor_residual_mean_L25_p5.npy   # (d_model,) (D3 anchor mean)

Outputs per (selector, seed):

  eval_results/issue503/benign_data/{selector_id}_seed{seed}.jsonl
    -- {datapoint_id, score, rank}

And the MF-5 method-independence pre-check:

  eval_results/issue503/benign_data/method_independence_D1_vs_D3.json
    -- {rho, n_used, verdict, demote_h7_7b}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("issue503.benign_data_select")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def output_dir(repo_root: Path) -> Path:
    p = repo_root / "eval_results" / "issue503" / "benign_data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_corpus(path: Path) -> list:
    from explore_persona_space.experiments.issue503.benign_data import BenignDatapoint

    rows: list[BenignDatapoint] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                BenignDatapoint(
                    datapoint_id=str(obj["id"]),
                    source=obj.get("source", "alpaca"),
                    instruction=obj.get("instruction", ""),
                    output=obj.get("output", ""),
                )
            )
    logger.info("Loaded %d benign datapoints from %s", len(rows), path)
    return rows


def write_selector_jsonl(out_path: Path, selector_id: str, seed: int, result) -> None:
    """One JSONL per (selector, seed). Rows: id, score, rank."""
    with out_path.open("w") as f:
        for rank, (dp_id, score) in enumerate(
            zip(result.selected_ids, result.scores, strict=True), start=1
        ):
            f.write(
                json.dumps(
                    {
                        "selector_id": selector_id,
                        "seed": seed,
                        "datapoint_id": dp_id,
                        "score": float(score),
                        "rank": rank,
                    }
                )
                + "\n"
            )


def write_score_per_corpus_row(out_path: Path, selector_id: str, seed: int, result) -> None:
    """Round-2 Rec 4: persist the FULL-CORPUS score vector per selector.

    MF-5 method-independence needs D1 vs D3 ρ over the SAME ordered set
    of corpus ids. The top-K JSONL above only carries the selected rows;
    this companion writes ``{datapoint_id, score}`` for EVERY row in the
    filtered corpus, in ``corpus_ids`` order. The MF-5 check reads this
    file at compare time.

    Schema (one JSON object per line, corpus_ids order):
      {"selector_id": str, "seed": int, "datapoint_id": str,
       "score_full_corpus": float}
    """
    if result.score_per_corpus_row is None or result.corpus_ids is None:
        # D0 (random) and D4 (format) don't have a meaningful per-corpus
        # score; they're not used in MF-5. Skip the write silently for
        # those — MF-5 only requires D1 + D3.
        return
    with out_path.open("w") as f:
        for dp_id, score in zip(result.corpus_ids, result.score_per_corpus_row, strict=True):
            f.write(
                json.dumps(
                    {
                        "selector_id": selector_id,
                        "seed": seed,
                        "datapoint_id": dp_id,
                        "score_full_corpus": float(score),
                    }
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-bundle",
        type=Path,
        required=True,
        help="Directory containing the precomputed feature files.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 42, 137],
        help="Random seeds per selector (default: 0 42 137).",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["D0_random", "D1_representation", "D2_gradient", "D3_cosine", "D4_format"],
        help="Subset of selectors to run.",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--method-independence",
        action="store_true",
        help="After running D1 and D3, compute Spearman rho(D1, D3) per MF-5.",
    )
    args = parser.parse_args(argv)

    from explore_persona_space.experiments.issue503.benign_data import (
        filter_safety_markers,
        method_independence_check,
        select_cosine_503,
        select_format,
        select_gradient_bidirectional,
        select_random,
        select_representation,
    )

    root = repo_root()
    bundle = args.feature_bundle.resolve()
    if not bundle.exists():
        raise FileNotFoundError(f"Feature bundle dir missing: {bundle}")

    corpus_path = bundle / "benign_corpus.jsonl"
    all_rows = load_corpus(corpus_path)
    filtered_rows = filter_safety_markers(all_rows)
    logger.info("After safety-filter: %d rows (of %d total)", len(filtered_rows), len(all_rows))

    # Map each filtered row back to its original index in the all-rows order so
    # we can slice the feature arrays consistently. Without this map, the
    # feature arrays (sized to the FULL corpus) would be cross-indexed against
    # a smaller filtered-row list and silently mis-align features to rows.
    id_to_orig_idx = {r.datapoint_id: i for i, r in enumerate(all_rows)}
    filtered_indices = np.array(
        [id_to_orig_idx[r.datapoint_id] for r in filtered_rows], dtype=np.int64
    )

    out_dir = output_dir(root)
    selector_results: dict[tuple[str, int], object] = {}

    for seed in args.seeds:
        for sel in args.selectors:
            if sel == "D0_random":
                result = select_random(filtered_rows, top_k=args.top_k, seed=seed)
            elif sel == "D1_representation":
                reprs_full = np.load(bundle / "reprs.npy")
                if reprs_full.shape[0] != len(all_rows):
                    raise RuntimeError(
                        f"reprs.npy n={reprs_full.shape[0]} != corpus n={len(all_rows)}; "
                        "feature bundle must be aligned to the unfiltered corpus order."
                    )
                reprs = reprs_full[filtered_indices]
                anchor_reprs = np.load(bundle / "anchor_reprs.npy")
                result = select_representation(filtered_rows, reprs, anchor_reprs, top_k=args.top_k)
            elif sel == "D2_gradient":
                grad_full = np.load(bundle / "grad_inner.npy")
                if grad_full.shape[0] != len(all_rows):
                    raise RuntimeError(
                        f"grad_inner.npy n={grad_full.shape[0]} != corpus n={len(all_rows)}"
                    )
                grad_inner = grad_full[filtered_indices]
                result = select_gradient_bidirectional(filtered_rows, grad_inner, top_k=args.top_k)
            elif sel == "D3_cosine":
                residuals_full = np.load(bundle / "residuals_L25_p5.npy")
                if residuals_full.shape[0] != len(all_rows):
                    raise RuntimeError(
                        f"residuals_L25_p5.npy n={residuals_full.shape[0]} "
                        f"!= corpus n={len(all_rows)}"
                    )
                residuals = residuals_full[filtered_indices]
                anchor_mean = np.load(bundle / "anchor_residual_mean_L25_p5.npy")
                result = select_cosine_503(filtered_rows, residuals, anchor_mean, top_k=args.top_k)
            elif sel == "D4_format":
                result = select_format(filtered_rows, top_k=args.top_k, seed=seed)
            else:
                raise ValueError(f"Unknown selector: {sel!r}")

            out_path = out_dir / f"{sel}_seed{seed}.jsonl"
            write_selector_jsonl(out_path, sel, seed, result)
            logger.info("Wrote %s (%d rows)", out_path, len(result.selected_ids))

            # Round-2 Rec 4: persist the FULL-CORPUS score vector so
            # MF-5 can compare D1 vs D3 ρ over the same ordered set of
            # ids. The top-K JSONL above only carries the selected rows.
            # D0/D4 emit no per-corpus score; the writer no-ops on those.
            corpus_score_path = out_dir / f"{sel}_seed{seed}.score_per_corpus_row.jsonl"
            write_score_per_corpus_row(corpus_score_path, sel, seed, result)
            if result.score_per_corpus_row is not None:
                logger.info(
                    "Wrote %s (%d full-corpus scores)",
                    corpus_score_path,
                    len(result.score_per_corpus_row),
                )

            selector_results[(sel, seed)] = result

    # MF-5 method-independence diagnostic — compute on seed 0 (deterministic).
    if args.method_independence:
        d1 = selector_results.get(("D1_representation", 0))
        d3 = selector_results.get(("D3_cosine", 0))
        if d1 is None or d3 is None:
            logger.warning("Cannot run --method-independence: D1 and D3 both required at seed=0.")
        else:
            check = method_independence_check(d1, d3)
            gate_path = out_dir / "method_independence_D1_vs_D3.json"
            gate_path.write_text(json.dumps(check, indent=2, sort_keys=True))
            logger.info("MF-5 method-independence verdict: %s", check["verdict"])
            print(json.dumps(check, indent=2))

    return 0


if __name__ == "__main__":
    if not os.environ.get("HF_TOKEN"):
        # Soft-warn — HF_TOKEN is only required if the feature bundle is being
        # downloaded from HF; this CLI runs against a local bundle.
        logger.warning("HF_TOKEN not set; assuming feature-bundle is local on disk.")
    sys.exit(main())
