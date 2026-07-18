"""#1092 fair-comparison deep-dive META analysis (analysis-only, cheap).

Two banked-artifact reads that need NO ridge refit:

  Q1 ceiling arithmetic: recompute fraction-of-ceiling for several entries from
    the stored components in fair_comparison.json and confirm they match the
    banked fraction fields (prefix ceiling = between-prefix variance share;
    context ceiling = 1 - interaction share, or the banked MLP companion).

  Q3 per-prefix error structure: reload the banked per-prefix error arrays
    (per_prefix_err_prefix / per_prefix_err_ctx, 996 each, in sorted-prefix
    order), reconstruct that prefix order from the manifest with the SAME
    battery-excluded / min-3-rows grouping the fit used, join to per-prefix
    metadata (topic, n_user_turns, context token length), compute a cheap
    per-prefix within-prefix context-vector spread from the staged context_end
    summaries, and test spread -> error (Spearman) + turns/length -> error.

Analysis-only: NO model forward, NO training, NO API. Reads the banked JSON, the
local manifest, and the staged context_end .npy (one streamed pass).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE = Path(
    "/mnt/eps-data/thomasjiralerspong/issue_1092_inline_operator/issue1092_realistic_crossing"
)
SUMM = STAGE / "analysis_tensors/summaries"
MANIFEST = STAGE / "corpus/manifest.jsonl"
BANKED = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json"
OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison_deepdive"
OUT.mkdir(parents=True, exist_ok=True)

CELLS = ["cell_inst_own", "cell_pre_own"]
BASES = ["ambient", "pca48"]
MIN_ROWS_PER_PREFIX = 3


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _pids_order(rows: list[dict]) -> tuple[list[str], dict[str, np.ndarray]]:
    """Reconstruct the sorted-prefix order the fit used: battery-excluded
    (stratum != trait_stratum AND not is_eval_only), grouped by prefix_id,
    prefixes with >= MIN_ROWS_PER_PREFIX rows, sorted by prefix_id."""
    be_idx = [
        i
        for i, r in enumerate(rows)
        if r.get("stratum") != "trait_stratum" and not r.get("is_eval_only")
    ]
    groups: dict[str, list[int]] = {}
    for i in be_idx:
        groups.setdefault(str(rows[i].get("prefix_id", "")), []).append(i)
    kept = {
        p: np.asarray(idx, dtype=np.int64)
        for p, idx in groups.items()
        if len(idx) >= MIN_ROWS_PER_PREFIX
    }
    return sorted(kept), kept


def _ceiling_verification(banked: dict) -> dict:
    """Recompute fraction-of-ceiling from stored components; compare to banked."""
    checks = []
    for cell in CELLS:
        for basis in BASES:
            b = banked["cells"][cell]["bases"][basis]
            sg = b["single_grain"]
            ceil = b["ceilings"]
            add_ceiling = ceil["context_additive_ceiling_densecore"]
            mlp = ceil.get("context_mlp_companion_ceiling")
            share_full = ceil["prefix_between_prefix_share_full"]
            share_dense = ceil["prefix_between_prefix_share_densecore"]
            recon = {
                "full.prefix": sg["r2_prefix_battery_excluded_full"] / share_full,
                "full.context_vs_additive": sg["r2_context_battery_excluded_full"] / add_ceiling,
                "dense.prefix": sg["r2_prefix_battery_excluded_densecore"] / share_dense,
                "dense.context_vs_additive": sg["r2_context_battery_excluded_densecore"]
                / add_ceiling,
            }
            if mlp:
                recon["full.context_vs_mlp"] = sg["r2_context_battery_excluded_full"] / mlp
                recon["dense.context_vs_mlp"] = sg["r2_context_battery_excluded_densecore"] / mlp
            banked_frac = {
                "full.prefix": b["fraction_of_ceiling_single_grain_full"]["prefix"],
                "full.context_vs_additive": b["fraction_of_ceiling_single_grain_full"][
                    "context_vs_additive"
                ],
                "dense.prefix": b["fraction_of_ceiling_single_grain_densecore"]["prefix"],
                "dense.context_vs_additive": b["fraction_of_ceiling_single_grain_densecore"][
                    "context_vs_additive"
                ],
                "full.context_vs_mlp": b["fraction_of_ceiling_single_grain_full"]["context_vs_mlp"],
                "dense.context_vs_mlp": b["fraction_of_ceiling_single_grain_densecore"][
                    "context_vs_mlp"
                ],
            }
            for k, v in recon.items():
                bv = banked_frac.get(k)
                checks.append(
                    {
                        "cell": cell,
                        "basis": basis,
                        "entry": k,
                        "recomputed": float(v),
                        "banked": (None if bv is None else float(bv)),
                        "abs_diff": (None if bv is None else abs(float(v) - float(bv))),
                    }
                )
    max_diff = max((c["abs_diff"] for c in checks if c["abs_diff"] is not None), default=None)
    return {"max_abs_diff": max_diff, "checks": checks}


def _per_prefix_spread(cell: str, pids: list[str], groups: dict[str, np.ndarray]) -> np.ndarray:
    """Within-prefix context_end spread per prefix: sqrt(mean squared L2 deviation
    of each row's context vector from the prefix centroid)."""
    ctx = np.load(SUMM / cell / "context_end_L14.npy", mmap_mode="r")
    out = np.zeros(len(pids), dtype=np.float64)
    for k, p in enumerate(pids):
        idx = groups[p]
        block = np.asarray(ctx[idx], dtype=np.float64)
        c = block - block.mean(0, keepdims=True)
        out[k] = float(np.sqrt((c * c).sum(1).mean()))
    return out


def _per_prefix_meta(pids: list[str], groups: dict[str, np.ndarray], rows: list[dict]) -> dict:
    topic = []
    n_turns = []
    ctx_tok_instruct = []
    for p in pids:
        idx = groups[p]
        r0 = rows[int(idx[0])]
        topic.append(str(r0.get("topic", "")))
        n_turns.append(int(r0.get("prefix_n_user_turns", 0)))
        toks = [int(rows[int(i)].get("n_tokens_instruct", 0)) for i in idx]
        ctx_tok_instruct.append(float(np.mean(toks)))
    return {
        "topic": topic,
        "n_turns": np.asarray(n_turns, dtype=np.float64),
        "ctx_tok_instruct_mean": np.asarray(ctx_tok_instruct, dtype=np.float64),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> dict:
    r, p = stats.spearmanr(x, y)
    return {"rho": float(r), "p": float(p), "n": len(x)}


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Spearman partial correlation of x,y controlling for z (rank-based):
    rank-transform all three, regress rank(x) and rank(y) on rank(z), correlate
    the residuals. Isolates whether x has signal on y beyond z."""
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r, p = stats.pearsonr(ex, ey)
    return {"partial_rho": float(r), "p": float(p), "n": len(x)}


def _per_prefix_analysis(banked: dict, rows: list[dict]) -> dict:
    pids, groups = _pids_order(rows)
    n_q = np.asarray([groups[p].size for p in pids], dtype=np.float64)
    meta = _per_prefix_meta(pids, groups, rows)
    out: dict = {"n_prefixes": len(pids)}
    # spread depends only on the cell (context_end), not the basis
    spread_by_cell = {c: _per_prefix_spread(c, pids, groups) for c in CELLS}
    per = {}
    for cell in CELLS:
        for basis in BASES:
            b = banked["cells"][cell]["bases"][basis]["prediction_agreement"]
            e_prefix = np.asarray(b["per_prefix_err_prefix"], dtype=np.float64)
            e_ctx = np.asarray(b["per_prefix_err_ctx"], dtype=np.float64)
            assert e_prefix.shape[0] == len(pids), (e_prefix.shape, len(pids))
            ratio = e_prefix / (e_ctx + 1e-12)
            spread = spread_by_cell[cell]
            key = f"{cell}/{basis}"
            per[key] = {
                "err_corr_prefix_ctx": float(np.corrcoef(e_prefix, e_ctx)[0, 1]),
                "err_ratio_prefix_over_ctx": {
                    "mean": float(ratio.mean()),
                    "median": float(np.median(ratio)),
                    "q10": float(np.quantile(ratio, 0.10)),
                    "q25": float(np.quantile(ratio, 0.25)),
                    "q75": float(np.quantile(ratio, 0.75)),
                    "q90": float(np.quantile(ratio, 0.90)),
                    "frac_gt_1": float((ratio > 1).mean()),
                    "frac_gt_2": float((ratio > 2).mean()),
                },
                "spearman_spread_vs_ctx_err": _spearman(spread, e_ctx),
                "spearman_spread_vs_prefix_err": _spearman(spread, e_prefix),
                "spearman_nturns_vs_ctx_err": _spearman(meta["n_turns"], e_ctx),
                "spearman_nturns_vs_prefix_err": _spearman(meta["n_turns"], e_prefix),
                "spearman_ctxtok_vs_ctx_err": _spearman(meta["ctx_tok_instruct_mean"], e_ctx),
                "spearman_ctxtok_vs_prefix_err": _spearman(meta["ctx_tok_instruct_mean"], e_prefix),
                "spearman_nq_vs_ctx_err": _spearman(n_q, e_ctx),
                "spearman_spread_vs_nturns": _spearman(spread, meta["n_turns"]),
                # isolate independent signal: does spread predict error beyond length?
                "partial_spread_vs_ctx_err_given_nturns": _partial_spearman(
                    spread, e_ctx, meta["n_turns"]
                ),
                "partial_nturns_vs_ctx_err_given_spread": _partial_spearman(
                    meta["n_turns"], e_ctx, spread
                ),
                "partial_spread_vs_ctx_err_given_ctxtok": _partial_spearman(
                    spread, e_ctx, meta["ctx_tok_instruct_mean"]
                ),
            }
            # hardest topics by mean context-map error (top 6 topics by count)
            topics = np.asarray(meta["topic"])
            uniq, counts = np.unique(topics, return_counts=True)
            top_topics = uniq[np.argsort(-counts)][:8]
            topic_err = {}
            for t in top_topics:
                m = topics == t
                topic_err[str(t)] = {
                    "n": int(m.sum()),
                    "mean_ctx_err": float(e_ctx[m].mean()),
                    "mean_prefix_err": float(e_prefix[m].mean()),
                    "mean_spread": float(spread[m].mean()),
                }
            per[key]["topic_breakdown"] = topic_err
            # persist raw per-prefix arrays (ambient only) for figures
            if basis == "ambient":
                np.savez(
                    OUT / f"per_prefix_arrays_{cell}.npz",
                    err_prefix=e_prefix,
                    err_ctx=e_ctx,
                    ratio=ratio,
                    spread=spread,
                    n_turns=meta["n_turns"],
                    ctx_tok=meta["ctx_tok_instruct_mean"],
                    n_q=n_q,
                    topic=np.asarray(meta["topic"]),
                )
    out["per_cell_basis"] = per
    out["spread_summary_by_cell"] = {
        c: {
            "mean": float(spread_by_cell[c].mean()),
            "median": float(np.median(spread_by_cell[c])),
            "min": float(spread_by_cell[c].min()),
            "max": float(spread_by_cell[c].max()),
        }
        for c in CELLS
    }
    return out


def main() -> int:
    banked = json.loads(BANKED.read_text())
    rows = _jsonl(MANIFEST)
    result = {
        "meta": {
            "script": "scripts/issue1092_fair_deepdive_meta.py",
            "banked_source": str(BANKED.relative_to(PROJECT_ROOT)),
            "manifest_rows": len(rows),
        },
        "ceiling_verification": _ceiling_verification(banked),
        "per_prefix": _per_prefix_analysis(banked, rows),
    }
    out_path = OUT / "deepdive_meta.json"
    out_path.write_text(json.dumps(result, indent=2))
    cv = result["ceiling_verification"]
    print(f"ceiling check max_abs_diff={cv['max_abs_diff']:.2e} over {len(cv['checks'])} entries")
    for k, v in result["per_prefix"]["per_cell_basis"].items():
        print(
            f"[{k}] err_corr={v['err_corr_prefix_ctx']:.3f} "
            f"ratio_med={v['err_ratio_prefix_over_ctx']['median']:.2f} "
            f"spread->ctx_err rho={v['spearman_spread_vs_ctx_err']['rho']:.3f} "
            f"(p={v['spearman_spread_vs_ctx_err']['p']:.1e})"
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
