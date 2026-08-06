#!/usr/bin/env python3
"""Analyzer-side derived reads for task #2091 (0 GPU-h, no new generation).

Three reads the committed P4 deliverables do not carry:

1. **Realized truncation + answer-length profile per rung** — recomputed from
   the packed greedy rollout text (unique ``(context_id, rollout_k)`` rows, so
   the repair round's append-only duplicate rows cannot double-count). The
   plan's §6 "R2 robustness pass" (Δ recomputed excluding cap-hit greedy rows)
   is not present in ``r2_delta.json``; this script supplies it.
2. **R2 Δ excluding cap-hit greedy rows**, with a cluster bootstrap on the
   rung's own ``group_key`` (read from the packed rollout docs, so the
   clustering axis matches the P4 CIs).
3. **R4 grid row-vs-column decomposition** (does the averaged regime help
   because the averaged TARGET is easier, or because the averaged-FIT map is
   better?) plus the reliability-ceiling-normalized behavioral Spearman reads.

Usage::

    uv run python scripts/issue2091_analyzer_reads.py \
        --raw-root /mnt/eps-data/thomasjiralerspong/issue2091_hf_dl/issue2091_decode/raw_completions/greedy \
        --out eval_results/issue_2091/analyzer_derived_reads.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "eval_results" / "issue_2091"
REGIMES = ("greedy", "avg_k5", "single")
HEADLINE_LAYER = "L19"
SEED = 20910
B_BOOT = 2000


def load_raw_profile(raw_root: Path) -> dict[str, dict]:
    """Per-rung {context_id: {finish_reason, group_key, n_words}} (first row wins).

    First-row-wins mirrors the ``GreedyStore`` reader's contract, so the
    pilot-overlap duplicate rows the repair round appended are ignored here the
    same way the analysis ignored them.
    """
    out: dict[str, dict] = {}
    for job_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        rows: dict[str, dict] = {}
        for shard in sorted(job_dir.glob("*.jsonl")):
            with shard.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    doc = (json.loads(line) or {}).get("doc") or {}
                    cid = doc.get("context_id")
                    if not cid or cid in rows:
                        continue
                    rows[str(cid)] = {
                        "finish_reason": doc.get("finish_reason"),
                        "group_key": str(doc.get("group_key") or cid),
                        "n_words": len((doc.get("completion") or "").split()),
                    }
        out[job_dir.name] = rows
    return out


def cluster_boot_median(values: np.ndarray, groups: list[str], *, tag: str) -> list[float]:
    """95% percentile CI of the median under a cluster bootstrap on ``groups``."""
    rng = np.random.default_rng(abs(hash((SEED, tag))) % (2**32))
    uniq: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        uniq.setdefault(g, []).append(i)
    keys = list(uniq)
    idx_of = [np.array(uniq[k], dtype=int) for k in keys]
    meds = np.empty(B_BOOT, dtype=np.float64)
    for b in range(B_BOOT):
        pick = rng.integers(0, len(keys), size=len(keys))
        sel = np.concatenate([idx_of[j] for j in pick])
        meds[b] = float(np.median(values[sel]))
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def r2_caphit_robustness(profile: dict[str, dict]) -> dict:
    r2 = json.loads((RESULTS / "r2_delta.json").read_text())
    out: dict[str, dict] = {}
    for setting, block in r2["settings"].items():
        rows = profile.get(setting)
        L = block.get(HEADLINE_LAYER) or {}
        pc = L.get("percontext") or {}
        if not rows or not pc:
            out[setting] = {"skipped": "no fresh greedy rollout text (banked-only cell)"}
            continue
        cids = list(pc["context_id"])
        delta = np.asarray(pc["delta"], dtype=np.float64)
        keep, groups_keep, groups_all, n_cap, n_unknown = [], [], [], 0, 0
        for i, cid in enumerate(cids):
            row = rows.get(cid)
            if row is None:
                n_unknown += 1
                groups_all.append(cid)
                keep.append(True)
                groups_keep.append(cid)
                continue
            groups_all.append(row["group_key"])
            is_cap = row["finish_reason"] == "length"
            n_cap += int(is_cap)
            keep.append(not is_cap)
            if not is_cap:
                groups_keep.append(row["group_key"])
        keep_arr = np.asarray(keep, dtype=bool)
        out[setting] = {
            "n_contexts": len(cids),
            "n_cap_hit": n_cap,
            "cap_hit_rate": n_cap / len(cids),
            "n_context_ids_unmatched_in_raw_text": n_unknown,
            "median_delta_all": float(np.median(delta)),
            "median_delta_excl_cap_hit": float(np.median(delta[keep_arr])),
            "boot_ci_median_all": cluster_boot_median(delta, groups_all, tag=f"{setting}::all"),
            "boot_ci_median_excl_cap_hit": cluster_boot_median(
                delta[keep_arr], groups_keep, tag=f"{setting}::nocap"
            ),
            "median_answer_words": statistics.median(r["n_words"] for r in rows.values()),
        }
    return out


def r4_row_col_decomposition() -> dict:
    r4 = json.loads((RESULTS / "r4_grids.json").read_text())
    out: dict[str, dict] = {}
    for setting, block in r4["settings"].items():
        g = block["r2_grid"][HEADLINE_LAYER]
        col_gain = float(np.mean([g[fr]["avg_k5"] - g[fr]["greedy"] for fr in REGIMES]))
        row_gain = float(np.mean([g["avg_k5"][er] - g["greedy"][er] for er in REGIMES]))
        out[setting] = {
            "diag_greedy": g["greedy"]["greedy"],
            "diag_avg_k5": g["avg_k5"]["avg_k5"],
            "diag_single": g["single"]["single"],
            "diag_avg_minus_greedy": g["avg_k5"]["avg_k5"] - g["greedy"]["greedy"],
            "eval_column_gain_avg_over_greedy": col_gain,
            "fit_row_gain_avg_over_greedy": row_gain,
            "transfer_penalty_avgfit_on_greedy_eval": g["avg_k5"]["avg_k5"] - g["avg_k5"]["greedy"],
            "transfer_penalty_avgfit_on_single_eval": g["avg_k5"]["avg_k5"] - g["avg_k5"]["single"],
            "greedyfit_vs_avgfit_on_avg_eval": g["avg_k5"]["avg_k5"] - g["greedy"]["avg_k5"],
        }
    return out


def ceiling_normalized_rho() -> dict:
    """rho / reliability-ceiling per regime, headline layer, per behavior x setting."""
    ceil: dict[tuple[str, str], dict] = {}
    for beh in ("sycophancy", "hallucination", "evil"):
        d = json.loads((RESULTS / f"r3_moderators_{beh}.json").read_text())
        for setting, blk in d["settings"].items():
            ceil[(beh, setting)] = blk.get("ceilings") or {}
    r4 = json.loads((RESULTS / "r4_grids.json").read_text())
    out: dict[str, dict] = {}
    for setting, block in r4["settings"].items():
        rho_block = block.get(f"behavioral_rho_{HEADLINE_LAYER}") or block.get(
            f"behavioral_rho_wildchat_{HEADLINE_LAYER}"
        )
        if rho_block is None:
            continue
        behaviors = (
            rho_block
            if setting == "generic"
            else {
                (
                    "sycophancy"
                    if setting.startswith("syc")
                    else "hallucination"
                    if setting.startswith("hal")
                    else "evil"
                ): rho_block
            }
        )
        for beh, fams in behaviors.items():
            key = (beh, "wildchat" if setting == "generic" else setting)
            c = ceil.get(key) or {}
            for fam, cols in fams.items():
                if not isinstance(cols, dict):
                    continue
                for regime in REGIMES:
                    cell = cols.get(regime)
                    if not isinstance(cell, dict) or cell.get("rho") is None:
                        continue
                    cval = c.get(f"ceil_{regime}")
                    out.setdefault(f"{setting}::{beh}", {}).setdefault(fam, {})[regime] = {
                        "rho": cell["rho"],
                        "n": cell["n"],
                        "ceiling": cval,
                        "rho_over_ceiling": (cell["rho"] / cval) if cval else None,
                    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    profile = load_raw_profile(args.raw_root)
    payload = {
        "meta": {
            "script": "scripts/issue2091_analyzer_reads.py",
            "raw_root": str(args.raw_root),
            "seed": SEED,
            "boot_b": B_BOOT,
            "headline_layer": HEADLINE_LAYER,
            "note": (
                "analyzer-side derived reads over committed P4 deliverables + the packed "
                "greedy rollout text; no new generation, no GPU"
            ),
        },
        "truncation_and_length_per_rung": {
            job: {
                "n_unique_contexts": len(rows),
                "n_cap_hit": sum(1 for r in rows.values() if r["finish_reason"] == "length"),
                "cap_hit_rate": (
                    sum(1 for r in rows.values() if r["finish_reason"] == "length") / len(rows)
                    if rows
                    else None
                ),
                "median_answer_words": (
                    statistics.median(r["n_words"] for r in rows.values()) if rows else None
                ),
                "n_group_keys": len({r["group_key"] for r in rows.values()}),
            }
            for job, rows in profile.items()
        },
        "r2_caphit_robustness": r2_caphit_robustness(profile),
        "r4_row_col_decomposition": r4_row_col_decomposition(),
        "ceiling_normalized_rho": ceiling_normalized_rho(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
