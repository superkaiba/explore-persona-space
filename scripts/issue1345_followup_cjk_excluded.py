#!/usr/bin/env python
"""Issue #1345 follow-up (free-analysis) — story-regime refit excluding CJK stories.

Refits the instruct story cells (r3, both arms) with the 24 kept stories that
contain >=1 CJK character EXCLUDED (analyzer round: 24/420 kept stories, ~5.7%
of the 2,108 turnstore rows), and reports the deltas vs the published cells
(cells_R_instruct_r3_{context,prefix}.json: context L19 R^2 -0.754, prefix
+0.117). Outputs land in a NEW dir (eval_results/issue_1345/followup_cjk_excluded/)
so the published cells are never overwritten.

Mechanism: the CJK-carrying story ids are recomputed from
kept_stories_instruct.jsonl (regex over CJK Unicode ranges; digest-only — story
text is never printed), the turnstore bundle rows whose sidecar conv_id (== story
id) is CJK-carrying are dropped at the BUNDLE level, and the filtered bundle is
injected into the untouched production fitter via ``run_cell(bundle=...)`` (the
committed #1345 injection point). Everything downstream — grouped 5-fold CV,
Gram-space GCV ridge, 20 conversation-level shuffle-null draws, the
selection-symmetric summary, per-cell conv-level bootstrap CI — is the byte-same
production path that produced the published cells.

Content hygiene: kept_stories_instruct.jsonl is LLM-generated story text over an
LMSYS-seeded corpus — this script never prints story text; ids/counts/hashes only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) bind BEFORE torch/numpy import

import issue825_fit_cells as fc  # noqa: E402
import issue1345_common as c  # noqa: E402
import issue1345_fit_cells as fit  # noqa: E402
import numpy as np  # noqa: E402

# CJK ranges: Unified Ideographs, Ext A, Compatibility Ideographs, Hiragana,
# Katakana, Hangul syllables + jamo, halfwidth kana (the analyzer's 24/420
# scan is replicated exactly by this set — asserted in main()).
CJK_RE = re.compile(
    "["
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\u3400-\u4dbf"  # CJK Ext A
    "\uf900-\ufaff"  # CJK Compatibility Ideographs
    "\u3040-\u309f"  # Hiragana
    "\u30a0-\u30ff"  # Katakana
    "\uac00-\ud7af"  # Hangul syllables
    "\u1100-\u11ff"  # Hangul jamo
    "\uff66-\uff9d"  # halfwidth kana
    "]"
)
N_CJK_EXPECTED = 24  # analyzer recount, 2026-07-16 (0-based indices 1,7,8,13,14,...)


def cjk_story_ids(kept_path: Path) -> tuple[list[str], list[int]]:
    """(story_ids, 0-based row indices) of kept stories containing >=1 CJK char.

    Digest-only: iterates the jsonl rows and regex-scans the `story` field
    without ever printing text.
    """
    ids: list[str] = []
    idxs: list[int] = []
    for i, row in enumerate(c.read_jsonl(kept_path)):
        if CJK_RE.search(row["story"]):
            ids.append(str(row["story_id"]))
            idxs.append(i)
    return ids, idxs


def filter_bundle_rows(bundle: dict, drop_ids: set[str]) -> tuple[dict, dict]:
    """Row-subset a loaded pt-shard bundle, dropping rows whose conv_id is in drop_ids.

    Returns (filtered_bundle, digest). Fails loud when any drop_id is absent
    from the bundle (allowlist/bundle drift) or when nothing would be dropped.
    """
    ids = np.asarray([str(x) for x in bundle["sidecar"]["conv_ids"]])
    drop_mask = np.isin(ids, np.asarray(sorted(drop_ids)))
    present = {str(x) for x in ids[drop_mask]}
    missing = drop_ids - present
    assert not missing, f"drop ids absent from bundle (drift): {sorted(missing)[:5]}"
    keep = ~drop_mask
    assert keep.any() and drop_mask.any(), (int(keep.sum()), int(drop_mask.sum()))
    arrays = {k: np.asarray(v)[keep] for k, v in bundle["arrays"].items()}
    for k, v in arrays.items():
        assert v.shape[0] == int(keep.sum()), (k, v.shape)
    filtered = {
        "arrays": arrays,
        "sidecar": {**bundle["sidecar"], "conv_ids": [str(x) for x in ids[keep]]},
    }
    digest = {
        "n_rows_before": len(ids),
        "n_rows_after": int(keep.sum()),
        "n_rows_dropped": int(drop_mask.sum()),
        "n_stories_before": len(np.unique(ids)),
        "n_stories_after": len(np.unique(ids[keep])),
        "n_stories_dropped": len(present),
    }
    return filtered, digest


def _l19_null_stats(nulls_path: Path) -> dict:
    """L19 + layer-max null digests (p95 over draws) from a nulls_*.json."""
    d = json.loads(nulls_path.read_text())
    mat = np.asarray(d["null_matrix"], dtype=np.float64)  # (draws, layers)
    l19 = mat[:, 19]
    lmax = np.asarray(d["null_layer_max_per_draw"], dtype=np.float64)
    return {
        "l19_null_p95": float(np.nanquantile(l19, 0.95)),
        "l19_null_mean": float(np.nanmean(l19)),
        "layer_max_null_p95": float(np.nanquantile(lmax, 0.95)),
        "n_draws": int(mat.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turnstore-dir", type=Path, required=True)
    ap.add_argument("--kept-stories", type=Path, required=True)
    ap.add_argument("--published-dir", type=Path, default=c.EVAL_DIR)
    ap.add_argument("--out-dir", type=Path, default=c.EVAL_DIR / "followup_cjk_excluded")
    ap.add_argument("--preds-dir", type=Path, default=c.DATA_DIR / "preds_cache_cjk_excluded")
    ap.add_argument("--folds", type=int, default=fc.N_FOLDS)
    ap.add_argument("--seed", type=int, default=fc.FIT_SEED)
    ap.add_argument("--null-draws", type=int, default=fc.N_NULL_DRAWS)
    ap.add_argument("--n-boot", type=int, default=fc.N_BOOTSTRAP)
    ap.add_argument(
        "--smoke-rows",
        type=int,
        default=0,
        help="smoke leg: keep only the first N stories AFTER the CJK exclusion "
        "(0 = production, full store); relaxes the ==24 CJK-count assert",
    )
    args = ap.parse_args()

    cjk_ids, cjk_idx0 = cjk_story_ids(args.kept_stories)
    print(f"[cjk-refit] CJK stories: {len(cjk_ids)} (0-based rows: {cjk_idx0[:6]}...)")
    if not args.smoke_rows:
        assert len(cjk_ids) == N_CJK_EXPECTED, (
            f"CJK recount {len(cjk_ids)} != analyzer's {N_CJK_EXPECTED} — regex/input drift"
        )

    bundle = fit.load_regime_bundle(args.turnstore_dir, "instruct", "r3")
    filtered, digest = filter_bundle_rows(bundle, set(cjk_ids))
    print(f"[cjk-refit] bundle filter: {digest}")
    if args.smoke_rows:
        keep_stories = sorted({str(x) for x in filtered["sidecar"]["conv_ids"]})[: args.smoke_rows]
        ids = np.asarray([str(x) for x in filtered["sidecar"]["conv_ids"]])
        m = np.isin(ids, np.asarray(keep_stories))
        filtered = {
            "arrays": {k: v[m] for k, v in filtered["arrays"].items()},
            "sidecar": {**filtered["sidecar"], "conv_ids": [str(x) for x in ids[m]]},
        }
        print(f"[cjk-refit][smoke] thinned to {int(m.sum())} rows / {len(keep_stories)} stories")

    cells = [x for x in c.all_cells() if x["model_key"] == "instruct" and x["regime"] == "r3"]
    assert [x["cell_id"] for x in cells] == [
        "R_instruct_r3_prefix",
        "R_instruct_r3_context",
    ] or len(cells) == 2, cells

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.preds_dir.mkdir(parents=True, exist_ok=True)
    comparison: dict = {}
    for cell in cells:
        cid = cell["cell_id"]
        res = fc.run_cell(
            cell,
            args.turnstore_dir,
            args.out_dir,
            n_folds=args.folds,
            seed=args.seed,
            null_draws=args.null_draws,
            n_boot=args.n_boot,
            allowlist=None,
            bundle=filtered,
        )
        # Mirror run_cells' post-processing (preds npz + conversation-level
        # bootstrap CI + n_groups) so the follow-up cell JSONs are
        # shape-compatible with the published ones.
        sweep, xy = res["sweep"], res["xy"]
        fitted = sweep["fitted_mask"]
        li = 19 if 19 in sweep["preds_frozen"] else max(sweep["preds_frozen"])
        pred = sweep["preds_frozen"][li][fitted]
        true = xy["Y"][fitted, li, :]
        conv = xy["conv_ids"][fitted]
        np.savez(
            args.preds_dir / f"{cid}_L{li}.npz",
            pred=pred.astype(np.float32),
            true=true.astype(np.float32),
            conv_ids=np.asarray([str(x) for x in conv]),
            layer=np.asarray([li]),
        )
        boot = {
            str(int(lj)): c.conv_bootstrap_r2(
                sweep["preds_frozen"][lj][fitted],
                xy["Y"][fitted, lj, :],
                conv,
                n_boot=args.n_boot,
                seed=args.seed + 200 + lj,
            )
            for lj in sweep["preds_frozen"]
        }
        cell_json = args.out_dir / f"cells_{cid}.json"
        payload = json.loads(cell_json.read_text())
        payload["r2_bootstrap_ci_frozen_layers_conv"] = boot
        payload["n_groups"] = len(np.unique(conv))
        payload["cjk_exclusion"] = {
            "excluded_story_ids": sorted(cjk_ids),
            "n_excluded_stories": len(cjk_ids),
            **digest,
        }
        c.write_json(cell_json, payload)

        entry: dict = {
            "after": {
                "r2_l19": float(payload["r2_per_layer_obs"][19]),
                "r2_l19_ci_conv": boot.get("19"),
                "n_rows": int(payload["metadata"]["n"]),
                "n_groups": int(payload["n_groups"]),
                "nulls": _l19_null_stats(args.out_dir / f"nulls_{cid}.json"),
            }
        }
        pub_cell = args.published_dir / f"cells_{cid}.json"
        pub_nulls = args.published_dir / f"nulls_{cid}.json"
        if pub_cell.exists() and not args.smoke_rows:
            pub = json.loads(pub_cell.read_text())
            entry["before"] = {
                "r2_l19": float(pub["r2_per_layer_obs"][19]),
                "r2_l19_ci_conv": pub.get("r2_bootstrap_ci_frozen_layers_conv", {}).get("19"),
                "n_rows": int(pub["metadata"]["n"]),
                "n_groups": int(pub.get("n_groups") or 0),
                "nulls": _l19_null_stats(pub_nulls),
            }
            entry["delta_r2_l19"] = entry["after"]["r2_l19"] - entry["before"]["r2_l19"]
        comparison[cid] = entry
        print(f"[cjk-refit] {cid} done (n={payload['metadata']['n']})", flush=True)

    summary = {
        "metadata": c.metadata(
            args.seed, digest["n_rows_after"], "scripts/issue1345_followup_cjk_excluded.py"
        ),
        "followup": "cjk_excluded_story_refit",
        "kept_stories_file": str(args.kept_stories),
        "cjk_story_ids": sorted(cjk_ids),
        "cjk_row_indices_0based": cjk_idx0,
        "bundle_filter": digest,
        "params": {
            "folds": args.folds,
            "seed": args.seed,
            "null_draws": args.null_draws,
            "n_boot": args.n_boot,
            "smoke_rows": args.smoke_rows,
        },
        "cells": comparison,
    }
    c.write_json(args.out_dir / "cjk_excluded_summary.json", summary)
    for cid, e in comparison.items():
        b = e.get("before", {}).get("r2_l19")
        a = e["after"]["r2_l19"]
        print(
            f"[cjk-refit] {cid}: L19 R^2 before={b if b is None else round(b, 4)} "
            f"after={round(a, 4)} n={e['after']['n_rows']} groups={e['after']['n_groups']} "
            f"null_p95(L19)={round(e['after']['nulls']['l19_null_p95'], 4)}",
            flush=True,
        )
    print("[cjk-refit] done", flush=True)


if __name__ == "__main__":
    main()
