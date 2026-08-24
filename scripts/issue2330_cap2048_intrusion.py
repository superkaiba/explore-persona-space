#!/usr/bin/env python3
"""Task #2330 fu1: language-intrusion recount on the cap2048 test pools.

The fu1 round regenerated both models' completions at gen_max_tokens=2048
(new generation substrate), so the parent body's CJK intrusion audit does not
cover these pools. This driver re-runs the parent recipe (verbatim helpers
imported from ``issue2330_revision_recomputes``) on the cap2048 test stores:

1. prompt byte-identity across arms on every shared ci (gate);
2. CJK scan of both arms' completions per ci, prompt-side CJK rows excluded
   from the intrusion definition (they answer in the prompt's language);
3. exclusion-sensitivity read: primary-layer test R² recomputed on the subset
   excluding rows intruded in EITHER arm, per training size;
4. response-length medians (chars) per arm at the 2,048 cap.

Content hygiene: reads LMSYS-derived raw-completion stores but only ever
emits aggregate counts/statistics, never row text.

Usage:
  uv run python scripts/issue2330_cap2048_intrusion.py \
      --raw-root /mnt/eps-data/$USER/issue2330_fu1_stage/dl/issue2330_matched \
      --preds-dir /mnt/eps-data/$USER/issue2330_fu1_stage/preds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

# Verbatim parent helpers — single source of truth for the CJK class, the
# variance-weighted subset R², and the fail-loud shard loader.
from issue2330_revision_recomputes import CJK_RE, _load_raw_rows, _pooled_r2  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "eval_results" / "issue_2330" / "cap2048"

CELLS = {
    "q25_n10k": ("q25_n10k_test_preds_ridge.npz", 19),
    "q35_n10k": ("q35_n10k_test_preds_ridge.npz", 22),
    "q25_n5k": ("q25_n5k_test_preds_ridge.npz", 19),
    "q35_n5k": ("q35_n5k_test_preds_ridge.npz", 22),
}
TOL_COMMITTED = 1e-6


def main() -> None:
    ap = argparse.ArgumentParser(description="#2330 fu1 cap2048 intrusion recount")
    ap.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="dir holding {q25_cap2048,qwen35_9b_cap2048}/test_1000/raw_completions",
    )
    ap.add_argument(
        "--preds-dir",
        type=Path,
        required=True,
        help="dir holding <cell>_test_preds_ridge.npz (cap2048 P3 outputs)",
    )
    ap.add_argument("--out-json", type=Path, default=EVAL / "intrusion_cap2048.json")
    args = ap.parse_args()

    raw7 = _load_raw_rows(args.raw_root / "q25_cap2048/test_1000/raw_completions")
    raw9 = _load_raw_rows(args.raw_root / "qwen35_9b_cap2048/test_1000/raw_completions")
    cis = sorted(raw7)
    if sorted(raw9) != cis:
        raise RuntimeError("test ci sets differ across arms")
    n_mismatch = sum(raw7[c]["prompt"] != raw9[c]["prompt"] for c in cis)
    if n_mismatch:
        raise RuntimeError(f"{n_mismatch} prompt mismatches across arms")

    # ---- preds (validated against committed cap2048 fits) ------------------
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ci_te_ref: np.ndarray | None = None
    for cell, (fname, layer) in CELLS.items():
        z = np.load(args.preds_dir / fname)
        ci = np.asarray(z["ci_te"])
        order = np.argsort(ci)
        ci = ci[order]
        if ci_te_ref is None:
            ci_te_ref = ci
        elif not np.array_equal(ci_te_ref, ci):
            raise RuntimeError(f"{cell}: test ci order differs from reference")
        p = np.asarray(z[f"pred_te_L{layer}"], dtype=np.float64)[order]
        t = np.asarray(z[f"target_te_L{layer}"], dtype=np.float64)[order]
        committed = json.loads((EVAL / f"matched_fits_{cell}_cap2048.json").read_text())[
            "per_layer"
        ][str(layer)]["ridge"]["test_r2"]
        full = _pooled_r2(p, t)
        if abs(full - committed) >= TOL_COMMITTED:
            raise RuntimeError(f"{cell}: recomputed R² {full} vs committed {committed}")
        preds[cell] = (p, t)
    assert ci_te_ref is not None
    if list(ci_te_ref) != cis:
        raise RuntimeError("preds ci_te set differs from raw-completion ci set")

    out: dict = {"n_test": len(cis), "gen_max_tokens": 2048}

    len7 = np.array([len(raw7[c]["response"]) for c in cis])
    len9 = np.array([len(raw9[c]["response"]) for c in cis])
    out["response_length_chars"] = {
        "median_7b": float(np.median(len7)),
        "median_9b": float(np.median(len9)),
        "mean_7b": float(len7.mean()),
        "mean_9b": float(len9.mean()),
    }

    prompt_cjk = {c for c in cis if CJK_RE.search(raw7[c]["prompt"])}
    eligible = [c for c in cis if c not in prompt_cjk]
    intr7 = {c for c in eligible if CJK_RE.search(raw7[c]["response"])}
    intr9 = {c for c in eligible if CJK_RE.search(raw9[c]["response"])}
    out["intrusion_scan"] = {
        "n_prompts_with_cjk": len(prompt_cjk),
        "n_eligible": len(eligible),
        "n_intruded_7b": len(intr7),
        "n_intruded_9b": len(intr9),
        "n_intruded_either": len(intr7 | intr9),
        "n_intruded_both": len(intr7 & intr9),
    }

    excl = intr7 | intr9
    keep = np.array([int(c) not in excl for c in ci_te_ref])
    sens = {}
    for size in ("n10k", "n5k"):
        p7, t7 = preds[f"q25_{size}"]
        p9, t9 = preds[f"q35_{size}"]
        sens[size] = {
            "n_kept": int(keep.sum()),
            "n_excluded": int((~keep).sum()),
            "r2_full_7b": _pooled_r2(p7, t7),
            "r2_full_9b": _pooled_r2(p9, t9),
            "delta_full": _pooled_r2(p9, t9) - _pooled_r2(p7, t7),
            "r2_excl_7b": _pooled_r2(p7[keep], t7[keep]),
            "r2_excl_9b": _pooled_r2(p9[keep], t9[keep]),
            "delta_excl": _pooled_r2(p9[keep], t9[keep]) - _pooled_r2(p7[keep], t7[keep]),
        }
    out["intrusion_exclusion_sensitivity"] = sens

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
