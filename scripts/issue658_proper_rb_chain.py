"""#658/#722 behavior-decoding chain with the PROPER #661 persona-vectors r_B.

Re-runs the DIRECT (r_B^T v0 -> E0) and MEDIATED (r_B^T (M_hat c_C) -> E0,
LOCO ridge M_hat) chains, swapping #658's crude diff-in-means r_B for #661's
proper Persona-Vectors r_B (r_b_a = instruction-present; r_b_c = instruct-and-
strip). Full-H for BOTH chains (DIRECT and MEDIATED dot the full 3584-dim r_B),
matching the existing behavior_chain.json so the crude reproduction is a valid
sanity check. n=50 contexts, 28 layers. 0-GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy/torch freeze their pools.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    ridge_predict_loco_raw,
)

PRB = REPO / "data/issue_658/prb_dl"
RB_DIR = PRB / "issue661_rb_extraction_divergence/analysis_tensors"
CC_PATH = PRB / "issue594_context_geometry/analysis_tensors/context_vectors_mean.pt"
CRUDE_RB = PRB / "issue658_theory_assumptions/store/r_b.pt"
V0_PATH = REPO / "data/issue_658/store/v0_summaries.pt"
E0_PATH = REPO / "eval_results/issue_658/E0_expression.json"

N_LAYERS = 28
BEHAVIORS = ["sycophancy", "refusal", "broad_em"]  # harmful_compliance excluded (no #661 r_B)

# cited reference numbers (do NOT recompute; for sanity-check + table)
CRUDE_REF = {  # from git show issue-722:eval_results/issue_722/structural/behavior_chain.json
    "broad_em": {"direct": 0.444, "med": 0.435},
    "harmful_compliance": {"direct": 0.692, "med": 0.671},
    "sycophancy": {"direct": 0.127, "med": 0.089},
    "refusal": {"direct": -0.273, "med": -0.248},
}
RIDGE_CEILING = {  # ridge readout DIRECT, the strong decoder
    "broad_em": 0.21,
    "harmful_compliance": 0.51,
    "sycophancy": 0.855,
    "refusal": 0.42,
}


def _rho(pred: np.ndarray, meas: np.ndarray) -> float | None:
    if len(pred) < 4 or np.std(pred) < 1e-9 or np.std(meas) < 1e-9:
        return None
    r, _ = spearmanr(pred, meas)
    return None if np.isnan(r) else float(r)


def e0_vector(e0: dict, col: str, ctx_ids: list[str]) -> tuple[np.ndarray, list[int]]:
    """(values, kept_row_indices into ctx_ids) for one behavior column."""
    vals, kept = [], []
    for i, c in enumerate(ctx_ids):
        cell = e0.get("e0", {}).get(c, {}).get(col)
        if cell is None:
            continue
        v = cell.get("rate")
        if v is None:
            v = cell.get("logp_mean")
        if v is None:
            continue
        vals.append(float(v))
        kept.append(i)
    return np.array(vals, dtype=np.float64), kept


def best_layer(per_layer_rho: list[float | None]) -> tuple[dict, dict]:
    """Return (best_by_abs, best_by_rho). best_by_abs = largest |rho| (task convention,
    keeps the sign). best_by_rho = largest rho (reference behavior_chain.json convention)."""
    valid = [(li, r) for li, r in enumerate(per_layer_rho) if r is not None]
    if not valid:
        return ({"layer": None, "rho": None}, {"layer": None, "rho": None})
    bi = max(valid, key=lambda t: abs(t[1]))
    br = max(valid, key=lambda t: t[1])
    return ({"layer": bi[0], "rho": bi[1]}, {"layer": br[0], "rho": br[1]})


def direct_chain(
    V: np.ndarray, rb: np.ndarray, y: np.ndarray, kept: list[int]
) -> list[float | None]:
    """Per-layer rho( r_B[L]^T v0(C)[L], E0 ). V:(N,L,H), rb:(L,H)."""
    out = []
    for li in range(N_LAYERS):
        pred = V[kept, li, :] @ rb[li]  # (n_kept,)
        out.append(_rho(pred, y))
    return out


def mediated_chain(
    ridge_pred: dict[int, np.ndarray], rb: np.ndarray, y: np.ndarray, kept: list[int]
) -> list[float | None]:
    """Per-layer rho( r_B[L]^T (M_hat c_C)(C)[L], E0 ). ridge_pred[L]:(N,H) full-H LOCO pred."""
    out = []
    for li in range(N_LAYERS):
        pred = ridge_pred[li][kept] @ rb[li]  # (n_kept,)
        out.append(_rho(pred, y))
    return out


def main() -> int:
    # ---- load v0 (the prediction target + DIRECT input), context order is canonical ----
    v0d = torch.load(V0_PATH, map_location="cpu", weights_only=False)
    ctx_ids = list(v0d["context_ids"])
    assert len(ctx_ids) == 50, len(ctx_ids)
    mean = v0d["summaries"]["mean"]
    V = np.stack([mean[c].numpy() for c in ctx_ids])  # (50, 28, 3584)
    assert V.shape == (50, N_LAYERS, 3584), V.shape

    # ---- E0 ----
    e0 = json.load(open(E0_PATH))

    # ---- c_C (issue594), align to ctx_ids ----
    ccd = torch.load(CC_PATH, map_location="cpu", weights_only=False)
    cc_ids = list(ccd["instance_ids"])
    cc_tensor = ccd["tensor"].numpy()  # (50, 28, 3584)
    cc_index = {c: i for i, c in enumerate(cc_ids)}
    missing = [c for c in ctx_ids if c not in cc_index]
    assert not missing, f"c_C missing contexts: {missing}"
    C = np.stack([cc_tensor[cc_index[c]] for c in ctx_ids])  # (50, 28, 3584) aligned

    # ---- crude r_B (issue658 store) ----
    crude = torch.load(CRUDE_RB, map_location="cpu", weights_only=False)
    crude_rb = {}  # behavior -> (28, 3584)
    for b in BEHAVIORS:
        dm = crude["r_b"][b]["diffmeans"]  # list/tensor of 28 (3584,)
        crude_rb[b] = np.stack([np.asarray(dm[li]) for li in range(N_LAYERS)])

    # ---- proper r_B (issue661): r_b_a (instruction-present), r_b_c (instruct-and-strip) ----
    proper_a, proper_c = {}, {}
    for b in BEHAVIORS:
        d = torch.load(RB_DIR / f"r_b_{b}.pt", map_location="cpu", weights_only=False)
        proper_a[b] = d["r_b_a"].numpy().astype(np.float64)  # (28, 3584)
        proper_c[b] = d["r_b_c"].numpy().astype(np.float64)
        # cosine A vs C (expect ~0.98 per #661)
        a = d["r_b_a"].numpy().reshape(-1)
        c = d["r_b_c"].numpy().reshape(-1)
        cos = float(a @ c / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-12))
        print(f"[cos] {b}: cos(r_b_a, r_b_c) = {cos:.4f}")

    # ---- MEDIATED: full-H LOCO ridge predicting v0 from c_C, per layer (shared across r_B) ----
    print("\n[ridge] fitting LOCO ridge c_C -> v0 (full-H), per layer ...")
    ridge_pred: dict[int, np.ndarray] = {}
    for li in range(N_LAYERS):
        Xc = C[:, li, :]  # (50, 3584)
        Yv = V[:, li, :]  # (50, 3584)
        ridge_pred[li] = ridge_predict_loco_raw(Xc, Yv)  # (50, 3584), #658 exact path
        if li % 7 == 0:
            print(f"  layer {li} done")

    results: dict = {}
    for b in BEHAVIORS:
        y, kept = e0_vector(e0, b, ctx_ids)
        n_kept = len(kept)
        print(f"\n=== {b} (n_kept_e0={n_kept}) ===")
        per_b = {"n_kept_e0": n_kept}
        for tag, rbmap in [("crude", crude_rb), ("proper_a", proper_a), ("proper_c", proper_c)]:
            rb = rbmap[b]
            d_pl = direct_chain(V, rb, y, kept)
            m_pl = mediated_chain(ridge_pred, rb, y, kept)
            d_abs, d_rho = best_layer(d_pl)
            m_abs, m_rho = best_layer(m_pl)
            per_b[tag] = {
                "direct_best_abs": d_abs,
                "direct_best_rho": d_rho,
                "mediated_best_abs": m_abs,
                "mediated_best_rho": m_rho,
                "direct_per_layer": d_pl,
                "mediated_per_layer": m_pl,
            }
            print(
                f"  {tag:9s} DIRECT  |rho|max L{d_abs['layer']} rho={d_abs['rho']:+.4f}"
                f"  (max-rho L{d_rho['layer']} rho={d_rho['rho']:+.4f})"
            )
            print(
                f"  {tag:9s} MEDIATED|rho|max L{m_abs['layer']} rho={m_abs['rho']:+.4f}"
                f"  (max-rho L{m_rho['layer']} rho={m_rho['rho']:+.4f})"
            )
        results[b] = per_b

    # ---- sanity check: crude reproduction vs cited behavior_chain.json (max-rho convention) ----
    print("\n=== SANITY CHECK: crude reproduction vs cited behavior_chain.json (max-rho) ===")
    sanity = {}
    ok = True
    for b in BEHAVIORS:
        got_d = results[b]["crude"]["direct_best_rho"]["rho"]
        got_m = results[b]["crude"]["mediated_best_rho"]["rho"]
        ref_d = CRUDE_REF[b]["direct"]
        ref_m = CRUDE_REF[b]["med"]
        dd, dm = abs(got_d - ref_d), abs(got_m - ref_m)
        match = dd < 0.06 and dm < 0.06
        ok = ok and match
        sanity[b] = {
            "got_direct": got_d,
            "ref_direct": ref_d,
            "got_med": got_m,
            "ref_med": ref_m,
            "abs_diff_direct": dd,
            "abs_diff_med": dm,
            "match": match,
        }
        print(
            f"  {b:18s} direct got {got_d:+.4f} vs ref {ref_d:+.4f} (d={dd:.4f}) | "
            f"med got {got_m:+.4f} vs ref {ref_m:+.4f} (d={dm:.4f}) | {'OK' if match else 'MISMATCH'}"
        )
    results["_sanity"] = sanity
    results["_sanity_overall_ok"] = ok
    results["_crude_ref_cited"] = CRUDE_REF
    results["_ridge_ceiling_cited"] = RIDGE_CEILING
    results["_context_ids"] = ctx_ids
    results["_note"] = (
        "Full-H DIRECT and MEDIATED (both dot the full 3584-dim r_B). MEDIATED M_hat = "
        "#658 exact LOCO ridge c_C->v0 (ridge_predict_loco_raw). best_by_abs = largest |rho| "
        "(task convention, sign kept); best_by_rho = largest rho (reference convention). "
        "harmful_compliance excluded: no proper #661 r_B."
    )

    out_path = REPO / "eval_results/issue_658/behavior_chain_proper_rb.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    print(f"SANITY OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
