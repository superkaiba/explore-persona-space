#!/usr/bin/env python3
"""Characterize the full-N linear map's saved 10k-pool retrieval failures.

Reuses pinned vectors and scores; performs no model generation or fitting.
Text bank rows are keyed by capture ID and must carry verified prompt hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import issue1901_figure2_retrieval_pool as M  # noqa: E402


def read_json(path: Path) -> dict:
    """Read a required JSON input."""
    return json.loads(path.read_text())


def text_reference(text: dict) -> dict:
    """Keep exact text provenance; inspect source text through the local bank."""
    return {
        **{k: v for k, v in text.items() if k not in ("prompt", "answers")},
        "answers": {
            seed: {
                k: v for k, v in answer.items() if k not in ("response", "prompt_token_ids_sha256")
            }
            for seed, answer in text["answers"].items()
        },
    }


def describe_text_families(rows: list[dict], text_bank: dict) -> dict:
    """Count an explicit exploratory prompt flag against its full query denominator."""

    def chemical(prompt: str) -> bool:
        return "chemical industry" in prompt.lower() or "a chemical company" in prompt.lower()

    groups = {
        name: [] for name in ("chemical", "chemical_article", "chemical_company_intro", "other")
    }
    for row in rows:
        prompt = text_bank["rows"][str(row["ci"])]["prompt"]
        is_chemical = chemical(prompt)
        groups["chemical" if is_chemical else "other"].append(row)
        if is_chemical and prompt.startswith("Write an article"):
            groups["chemical_article"].append(row)
        if is_chemical and prompt.startswith(("Give me an introduction", "Write an introduction")):
            groups["chemical_company_intro"].append(row)
    return {
        "definition": "Case-insensitive 'chemical industry' or 'a chemical company' in prompt; article/intro subgroups use the stated request prefix.",
        "groups": {
            name: {"n_queries": len(group), "n_failures": sum(r["rank"] > 1 for r in group)}
            for name, group in groups.items()
        },
        "chemical_failures_retrieving_chemical": sum(
            row["rank"] > 1 and chemical(text_bank["rows"][str(row["top_candidate_ci"])]["prompt"])
            for row in groups["chemical"]
        ),
    }


def audit(args: argparse.Namespace) -> None:
    """Reproduce ranks, resolve mistaken candidates, and save all-query diagnostics."""
    source = args.results / "summary.json"
    saved = read_json(source)
    saved_pool = read_json(args.results / "pool.json")
    text_bank = read_json(args.text_bank)
    assert saved["status"] == "complete" and saved["retrieval"]["n_pool"] == 10000
    assert text_bank["data_revision"] == saved["data_revision"] == M.FIVE.REVISION
    assert text_bank["pool_sha256"] == M.FIVE._sha256(args.results / "pool.json")
    files = {
        **M.FIVE.BASE_FILES,
        **M.EXTRA_FILES,
        "pred": M.FIVE._prediction_path(963444, "ridge"),
    }
    paths, hashes = {}, {}
    for key, filename in files.items():
        root = args.stage_root if key in M.EXTRA_FILES else args.source_stage
        path = root / filename
        expected_key = "pred_ridge_963444" if key == "pred" else key
        digest = M.FIVE._sha256(path)
        assert digest == saved["input_sha256"][expected_key], filename
        paths[key], hashes[key] = path, digest
    logging.info("input hashes verified")
    original, target, pool, view, test_rows, provenance = M.load_pool(paths, 10000)
    assert provenance == saved_pool
    pred = M.FIVE._load_prediction(paths["pred"], test_rows)
    whiten, _ = M.FINAL._whitener(paths["whiten"])
    wp, wq = whiten(pool), whiten(pred)
    full = M.FINAL._precompute_metric_arrays(pred, pool, wq, wp)
    sim = 1 - full["whiten_cosine"][np.ix_(view.pred_rows, view.pool_rows)]
    score = M.FINAL.MB.csls_scores(sim, M.FINAL.K_CSLS)
    ranks = M.FINAL._strict_ranks(-score, view.true_idx)
    cells = saved["per_n"]["963444"]
    np.testing.assert_array_equal(
        ranks, cells["ridge"]["metrics"]["whiten_csls"]["per_query_ranks"]
    )
    old = np.array(cells["ridge"]["original_pool_metrics"]["whiten_csls"]["per_query_ranks"])
    nonlinear = np.array(cells["mlp"]["metrics"]["whiten_csls"]["per_query_ranks"])
    top = np.argsort(-score, axis=1, kind="stable")[:, :5]
    truth_sim = M.FINAL._cosine(wp[view.pred_rows], wp[view.pool_rows])
    pool_ids = np.asarray(provenance["pool_capture_ids"])
    assert set(text_bank["rows"]) == {str(ci) for ci in pool_ids}
    failed = ranks > 1
    rows = []
    for i, qi in enumerate(view.pred_rows):
        ci = int(pool_ids[view.true_idx[i]])
        texts = text_bank["rows"][str(ci)]
        assert texts["eval_row"] == int(qi) and texts["passb_row"] == int(test_rows[qi])
        assert hashlib.sha256(texts["prompt"].encode()).hexdigest() == texts["prompt_sha256"]
        winner = int(top[i, 0])
        wrong_similarity = float(truth_sim[i, winner])
        actual_rank = 1 + int(np.sum(truth_sim[i] > wrong_similarity + 1e-12))
        true_score = float(score[i, view.true_idx[i]])
        row = {
            "query_index": i,
            "eval_row": int(qi),
            "passb_row": int(test_rows[qi]),
            "ci": ci,
            "rank": float(ranks[i]),
            "rank_original_pool": float(old[i]),
            "rank_nonlinear": float(nonlinear[i]),
            "top_candidate_ci": int(pool_ids[winner]),
            "top5_candidate_ci": pool_ids[top[i]].tolist(),
            "margin_top_minus_true": float(score[i, winner] - true_score),
            "cosine_pred_true_whitened": float(sim[i, view.true_idx[i]]),
            "cosine_true_winner_whitened": wrong_similarity,
            "winner_rank_by_true_answer_cosine_including_self": actual_rank,
            "prompt_chars": len(texts["prompt"]),
            "seed43_answer_chars": len(texts["answers"]["43"]["response"]),
            "linear_ranks_by_n_train": {
                n: v["ridge"]["metrics"]["whiten_csls"]["per_query_ranks"][i]
                for n, v in saved["per_n"].items()
            },
            "linear_rank_by_metric": {
                metric: value["per_query_ranks"][i]
                for metric, value in cells["ridge"]["metrics"].items()
            },
        }
        if failed[i]:
            row["query_text"] = text_reference(texts)
            row["retrieved_text"] = text_reference(text_bank["rows"][str(row["top_candidate_ci"])])
        rows.append(row)
    fail_rows = [row for row in rows if row["rank"] > 1]
    assert len(fail_rows) == 90
    summary = {
        "n_queries": len(rows),
        "n_candidates": len(pool_ids),
        "n_failures": len(fail_rows),
        "failure_rank_counts": dict(Counter(row["rank"] for row in fail_rows)),
        "recall_counts": {k: int(np.sum(ranks <= k)) for k in (1, 2, 3, 5, 10, 30)},
        "new_failures": int(np.sum(failed & (old == 1))),
        "retained_old_failures": int(np.sum(failed & (old > 1))),
        "recovered_old_failures": int(np.sum(~failed & (old > 1))),
        "also_fail_nonlinear": int(np.sum(failed & (nonlinear > 1))),
        "nonlinear_rescues": int(np.sum(failed & (nonlinear == 1))),
        "nonlinear_only_failures": int(np.sum(~failed & (nonlinear > 1))),
        "wrong_winner_is_added_distractor": sum(row["top_candidate_ci"] >= 0 for row in fail_rows),
        "unique_wrong_winners": len({row["top_candidate_ci"] for row in fail_rows}),
        "failure_margins": np.percentile(
            [r["margin_top_minus_true"] for r in fail_rows], [0, 25, 50, 75, 100]
        ).tolist(),
        "failure_true_winner_cosine": np.percentile(
            [r["cosine_true_winner_whitened"] for r in fail_rows], [0, 25, 50, 75, 100]
        ).tolist(),
        "descriptive_only": "Exploratory audit of this fixed pool and query set; no causal interpretation of text categories or answer length.",
    }
    summary["text_families"] = describe_text_families(rows, text_bank)
    result = {
        "status": "complete",
        "source": str(source.relative_to(ROOT)),
        "source_sha256": M.FIVE._sha256(source),
        "data_revision": saved["data_revision"],
        "input_sha256": hashes,
        "text_bank_sha256": M.FIVE._sha256(args.text_bank),
        "text_provenance": {k: v for k, v in text_bank.items() if k != "rows"},
        "summary": summary,
        "rows": rows,
    }
    M.FIVE._write_json(args.output, result)
    logging.info("complete: %s", summary)


def main() -> None:
    """Parse local, explicitly staged inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stage", required=True, type=Path)
    parser.add_argument("--stage-root", type=Path, default=ROOT / "data/issue_1901/retrieval_10k")
    parser.add_argument(
        "--results", type=Path, default=ROOT / "eval_results/issue_1901/retrieval_10k"
    )
    parser.add_argument(
        "--text-bank",
        type=Path,
        default=ROOT / "data/issue_1901/retrieval_10k/text_sources/text_bank.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval_results/issue_1901/retrieval_10k/linear_failures/audit.json",
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
