"""Issue #2330 interp-critique round-1 revision recomputes.

Computes, from the committed per-context prediction NPZs
(``data/issue_2330/preds/*.npz``, mirrored on HF under
``issue2330_matched/analysis_tensors/preds/``), the committed cap-hit ci
lists (``eval_results/issue_2330/cap_hit/*.json``), and the two arms'
test-split raw-completion stores:

1. Common-uncapped test intersection read — both 10k (and 5k) primary-layer
   R² on the rows uncapped in BOTH arms (paired same-prompt truncation
   control; closes the non-identical-subset gap in the separately restricted
   reads, critique round 1 rev 2/Codex rev 1).
2. Per-prompt cosine win fraction + low-cosine tail counts (Codex rev 2:
   the "uniform shift / no failing subpopulation" calibration).
3. Language-intrusion exclusion sensitivity (Codex rev 3): CJK scan of both
   arms' completions per ci; primary R² recomputed on the common subset
   excluding rows whose completion is intruded in EITHER arm (prompt-side
   CJK rows are excluded from the intrusion definition, matching the body;
   they answer in the prompt's language and stay in the read).
4. Response-length medians per arm (Claude rev 3 bundle-difference clause).
5. Re-extraction of committed companion numbers quoted in the revised body:
   kNN retrieval (cosine + euclidean, 5k/10k), truncation refit train sizes,
   separately restricted reads, WildChat per-layer transfer drops.

Sanity gates (fail loud): preds ci order matches across arms; cap-hit
recount from raw ``finish_reason`` matches the committed aggregates;
prompt byte-identity across arms on every shared ci.

Content hygiene: raw stores are LMSYS real-user text — this script only
ever emits aggregate counts/statistics, never row text.

Usage:  uv run python scripts/issue2330_revision_recomputes.py \
            --raw-root /tmp/i2330_rawcheck
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "eval_results" / "issue_2330"
PREDS = REPO / "data" / "issue_2330" / "preds"

CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

CELLS = {
    "q25_n10k": ("q25_n10k_test_preds_ridge.npz", 19),
    "q35_n10k": ("q35_n10k_test_preds_ridge.npz", 22),
    "q25_n5k": ("q25_n5k_test_preds_ridge.npz", 19),
    "q35_n5k": ("q35_n5k_test_preds_ridge.npz", 22),
}


def _pooled_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Whole-map variance-weighted R² = 1 - Σ_d SSE_d / Σ_d SST_d, with the
    SUBSET's own per-dimension target mean (identical arithmetic to
    issue1491_ladder_fits._pooled_r2, which the committed restricted reads
    used)."""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    sse = float(((target - pred) ** 2).sum())
    sst = float(((target - target.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - sse / (sst + 1e-30)


def _row_cosines(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = pred.astype(np.float64)
    t = target.astype(np.float64)
    num = (p * t).sum(axis=1)
    den = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1)
    if not (den > 0).all():
        raise RuntimeError("zero-norm row in cosine computation")
    return num / den


def _load_raw_rows(root: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    shards = sorted(root.glob("shard*_chunk*.json"))
    if not shards:
        raise RuntimeError(f"no raw-completion shards under {root}")
    for sh in shards:
        payload = json.loads(sh.read_text())
        for row in payload["rows"]:
            ci = int(row["ci"])
            if ci in rows:
                raise RuntimeError(f"duplicate ci {ci} in {sh}")
            rows[ci] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="dir holding issue1491_scale_ladder/... and issue2330_matched/... "
        "test_1000 raw_completions downloads",
    )
    ap.add_argument("--out", type=Path, default=EVAL / "revision_recomputes.json")
    args = ap.parse_args()

    out: dict = {"schema": "issue2330_revision_recomputes_v1"}

    # ---- load preds -------------------------------------------------------
    npz = {}
    for cell, (fname, prim) in CELLS.items():
        z = np.load(PREDS / fname)
        assert int(z["primary_layer"]) == prim, (cell, int(z["primary_layer"]))
        npz[cell] = z
    ci_te_7 = npz["q25_n10k"]["ci_te"]
    ci_te_9 = npz["q35_n10k"]["ci_te"]
    assert np.array_equal(ci_te_7, ci_te_9), "test ci order differs across arms"
    assert np.array_equal(ci_te_7, npz["q25_n5k"]["ci_te"])
    assert np.array_equal(ci_te_7, npz["q35_n5k"]["ci_te"])
    ci_te = ci_te_7
    n_test = len(ci_te)
    assert n_test == 1000

    def prim_pair(cell: str) -> tuple[np.ndarray, np.ndarray]:
        _, prim = CELLS[cell]
        z = npz[cell]
        return z[f"pred_te_L{prim}"], z[f"target_te_L{prim}"]

    # ---- cap-hit masks ----------------------------------------------------
    cap7 = json.load(open(EVAL / "cap_hit" / "cap_hit_7b_test_1000.json"))
    cap9 = json.load(open(EVAL / "cap_hit" / "cap_hit_9b_test_1000.json"))
    cap7_set = set(map(int, cap7["cap_hit_cis"]))
    cap9_set = set(map(int, cap9["cap_hit_cis"]))
    assert len(cap7_set) == cap7["cap_hit"] == 56
    assert len(cap9_set) == cap9["cap_hit"] == 207
    union = cap7_set | cap9_set
    keep = np.array([int(c) not in union for c in ci_te])
    out["cap_hit_sets"] = {
        "n_7b_capped": len(cap7_set),
        "n_9b_capped": len(cap9_set),
        "n_capped_both": len(cap7_set & cap9_set),
        "n_capped_union": len(union),
        "n_common_uncapped": int(keep.sum()),
    }

    # ---- 1. common-uncapped intersection read -----------------------------
    inter = {}
    for size in ("n10k", "n5k"):
        p7, t7 = prim_pair(f"q25_{size}")
        p9, t9 = prim_pair(f"q35_{size}")
        r7 = _pooled_r2(p7[keep], t7[keep])
        r9 = _pooled_r2(p9[keep], t9[keep])
        inter[size] = {
            "n": int(keep.sum()),
            "r2_7b": r7,
            "r2_9b": r9,
            "delta_9b_minus_7b": r9 - r7,
        }
    out["common_uncapped_intersection_read"] = inter

    # ---- 2. per-prompt win fraction + tails --------------------------------
    wins = {}
    for size in ("n10k", "n5k"):
        c7 = _row_cosines(*prim_pair(f"q25_{size}"))
        c9 = _row_cosines(*prim_pair(f"q35_{size}"))
        wins[size] = {
            "n": n_test,
            "n_9b_cosine_higher": int((c9 > c7).sum()),
            "frac_9b_cosine_higher": float((c9 > c7).mean()),
            "n_below_0p8_7b": int((c7 < 0.8).sum()),
            "n_below_0p8_9b": int((c9 < 0.8).sum()),
            "median_7b": float(np.median(c7)),
            "median_9b": float(np.median(c9)),
        }
    out["per_prompt_cosine_wins"] = wins

    # ---- raw stores: identity gates, cap recount, lengths, intrusion ------
    raw7 = _load_raw_rows(
        args.raw_root / "issue1491_scale_ladder/scale7_refit/test_1000/raw_completions"
    )
    raw9 = _load_raw_rows(args.raw_root / "issue2330_matched/qwen35_9b/test_1000/raw_completions")
    cis = [int(c) for c in ci_te]
    assert set(cis) == set(raw7) == set(raw9), "raw-store ci coverage mismatch"
    n_prompt_mismatch = sum(raw7[c]["prompt"] != raw9[c]["prompt"] for c in cis)
    assert n_prompt_mismatch == 0, f"{n_prompt_mismatch} prompt mismatches across arms"
    rc7 = {c for c in cis if raw7[c]["finish_reason"] == "length"}
    rc9 = {c for c in cis if raw9[c]["finish_reason"] == "length"}
    assert rc7 == cap7_set, "7B finish_reason recount != committed cap_hit_cis"
    assert rc9 == cap9_set, "9B finish_reason recount != committed cap_hit_cis"
    out["raw_store_gates"] = {
        "prompt_mismatches_across_arms": n_prompt_mismatch,
        "cap_recount_7b": len(rc7),
        "cap_recount_9b": len(rc9),
    }

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

    # ---- 3. exclusion-sensitivity read (10k primary) -----------------------
    excl = intr7 | intr9
    keep_x = np.array([int(c) not in excl for c in ci_te])
    sens = {}
    for size in ("n10k", "n5k"):
        p7, t7 = prim_pair(f"q25_{size}")
        p9, t9 = prim_pair(f"q35_{size}")
        r7f = _pooled_r2(p7, t7)
        r9f = _pooled_r2(p9, t9)
        r7 = _pooled_r2(p7[keep_x], t7[keep_x])
        r9 = _pooled_r2(p9[keep_x], t9[keep_x])
        sens[size] = {
            "n_kept": int(keep_x.sum()),
            "n_excluded": int((~keep_x).sum()),
            "r2_full_7b": r7f,
            "r2_full_9b": r9f,
            "delta_full": r9f - r7f,
            "r2_excl_7b": r7,
            "r2_excl_9b": r9,
            "delta_excl": r9 - r7,
        }
    out["intrusion_exclusion_sensitivity"] = sens

    # ---- 5. committed companion numbers (re-extraction) --------------------
    comp: dict = {}
    for cell in CELLS:
        f = json.load(open(EVAL / f"matched_fits_{cell}.json"))
        prim = str(f["primary_layer"])
        rec = f["per_layer"][prim]
        knn = rec["knn_retrieval"]["ridge"]
        entry = {
            "retrieval_acc1_cosine": knn["cosine"]["acc_at_k"]["1"],
            "retrieval_acc1_euclidean": knn["euclidean"]["acc_at_k"]["1"],
        }
        tr = rec.get("truncation_restriction", {})
        if tr.get("available"):
            entry["restricted_read_r2"] = tr["read"]["r2_test_untruncated"]
            entry["restricted_read_n"] = tr["read"]["n_test_untruncated"]
            if tr.get("refit", {}).get("available"):
                entry["refit_r2"] = tr["refit"]["r2_test_untruncated"]
                entry["refit_n_train_untruncated"] = tr["refit"]["n_train_untruncated"]
        comp[cell] = entry
    out["committed_companions"] = comp

    con = json.load(open(EVAL / "contrasts.json"))
    wc_drops: dict = {}
    for mk in ("q25", "q35"):
        cell = f"{mk}_n10k"
        f = json.load(open(EVAL / f"matched_fits_{cell}.json"))
        drops = {}
        for lk, rec in f["per_layer"].items():
            r2_in = rec["ridge"]["test_r2"]
            r2_wc = rec["wc_transfer"]["ridge_test_r2"]
            drops[lk] = {"in_dist": r2_in, "wc": r2_wc, "drop": r2_in - r2_wc}
        wc_drops[cell] = drops
    out["wc_per_layer_drops_10k"] = wc_drops
    out["contrasts_meta"] = {
        "primary_raw": con["primary_contrast_raw"],
    }

    # 9B refit train-n scaling estimate: measured 5k->10k raw gain per doubling.
    g9 = comp["q35_n10k"].get("_", None)
    r9_10 = json.load(open(EVAL / "matched_fits_q35_n10k.json"))["per_layer"]["22"]["ridge"][
        "test_r2"
    ]
    r9_5 = json.load(open(EVAL / "matched_fits_q35_n5k.json"))["per_layer"]["22"]["ridge"][
        "test_r2"
    ]
    gain_per_doubling = r9_10 - r9_5
    n_refit_7b = comp["q25_n10k"]["refit_n_train_untruncated"]
    n_refit_9b = comp["q35_n10k"]["refit_n_train_untruncated"]
    doublings_rel = float(np.log2(n_refit_7b / n_refit_9b))
    doublings_own = float(np.log2(10000 / n_refit_9b))
    out["refit_train_n_scaling_estimate_9b"] = {
        "gain_5k_to_10k": gain_per_doubling,
        "n_train_refit_7b": n_refit_7b,
        "n_train_refit_9b": n_refit_9b,
        "doublings_arm_mismatch": doublings_rel,
        "estimated_r2_cost_arm_mismatch": gain_per_doubling * doublings_rel,
        "doublings_vs_own_full": doublings_own,
        "estimated_r2_cost_vs_own_full": gain_per_doubling * doublings_own,
    }
    _ = g9

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
