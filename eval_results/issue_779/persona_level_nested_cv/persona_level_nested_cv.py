"""Issue #779 inline free-analysis: nested-CV persona-level head-to-head (v2).

Removes the layer-selection asterisk on the map-vs-raw persona-level comparison
with honest held-out layer selection, and FIXES the v1 pooling bug.

v1 bug: outer-test reads were pooled ACROSS folds that selected DIFFERENT layers;
raw projections have layer-dependent offsets/scales, so concatenating reads from
different layers corrupted the pooled correlation whenever layer selection was
unstable (diagnostic: the oracle, which reads the actual answer and cannot
overfit, collapsed to r=+0.084 for evil). Fix: standardize each fold's test
reads (label-free, by the test fold's own read mean/std) before pooling, and
also report the per-outer-fold correlations directly (mean +/- SD), which never
pool across layers.

Design:
  - 60 corpus personas; group read = mean context vector over 40 questions,
    correlated with the persona's pooled mean judge score (layer-independent).
  - Outer: 5-fold over personas (grouped, shuffled, seed 0).
  - Layer selection per outer fold:
      * fit-free methods (pv_raw, map_generic, g_generic, oracle): argmax over
        layers of the outer-TRAIN correlation (no corpus fit -> no fit-optimism,
        only a discrete choice among 28 layers).
      * corpus-fit methods (map_corpus, g_corpus): 4-fold INNER CV on the
        outer-train (mean inner-val Pearson) -> avoids fit-optimism.
  - Evaluate the selected-layer read on the held-out outer-test personas.
  - Report (a) per-fold Pearson at the selected layer: mean +/- SD across folds
    [primary, never pools across layers]; (b) z-pooled Pearson over all 60
    (each fold's test reads z-scored label-free, then pooled) [secondary].
  - Headline: paired map_generic - pv_raw, per-fold (mean, folds-positive) and
    z-pooled bootstrap.

Methods: pv_raw / map_generic / g_generic (deployable); map_corpus / g_corpus
(in-distribution, nested); oracle (post-generation ceiling). Reuses arm_headline
GramRidge + loaders verbatim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import issue779_arm_headline as AH  # noqa: E402
import issue779_common as C  # noqa: E402
import issue779_stage1 as S1  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
N_OUTER = 5
N_INNER = 4
N_BOOT = 2000
SEED = 0
EXTERNAL = ["pv_raw", "map_generic", "g_generic", "oracle"]
CORPUS_FIT = ["map_corpus", "g_corpus"]
METHODS = ["pv_raw", "map_generic", "g_generic", "map_corpus", "g_corpus", "oracle"]


def pear(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def zsc(x):
    s = np.std(x)
    return (x - np.mean(x)) / s if s > 0 else x - np.mean(x)


def kfold(idx, k, rng):
    return [np.sort(f) for f in np.array_split(rng.permutation(idx), k)]


def cmap(Xg, Yg, rb_l, tr, te):
    gr = AH.GramRidge(Xg[tr])
    return F.dot_readout(gr.predict(Yg[tr], Xg[te]), rb_l)


def cg(Xg, Sg, tr, te):
    m = tr[np.isfinite(Sg[tr])]
    gr = AH.GramRidge(Xg[m])
    return gr.predict(Sg[m][:, None], Xg[te]).ravel()


def main() -> int:
    lmsys = AH.load_lmsys_bundle()
    L = [int(x) for x in lmsys["layers"]]
    res = {"design": {"n_outer": N_OUTER, "n_inner": N_INNER, "seed": SEED}, "traits": {}}

    for trait in C.TRAITS:
        rb = S1._load_rb(AH.COLLECT_DIR / "r_b", trait, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)
        blob = AH.load_corpus(trait)
        n_p, n_q, n_r = blob["n_personas"], blob["n_questions"], blob["n_rollouts"]
        scores = AH.corpus_scores(trait)
        Sq = scores.reshape(n_p, n_q, n_r)
        allq = np.tile(np.arange(n_q), (n_p, 1))
        base = blob["cx_last"][:, 0, :].float().numpy().reshape(n_p, n_q, -1)
        _, _, Sg = AH._grouped_vectors(base, base, Sq, allq)
        lmsys_lab = AH.lmsys_labels(trait, lmsys["cx_last"].shape[0])

        # precompute per-layer corpus grouped vectors + fit-free per-persona reads
        Xg_L, Yg_L, rbL = {}, {}, {}
        ext = {m: {} for m in EXTERNAL}
        for li in L:
            ccol = blob["layers"].index(li)
            lcol = L.index(li)
            Xg = blob["cx_last"][:, ccol, :].float().numpy().reshape(n_p, n_q, -1).mean(axis=1)
            Yg = (
                blob["v_x"][:, ccol, :]
                .float()
                .numpy()
                .reshape(n_p * n_q, n_r, -1)
                .mean(axis=1)
                .reshape(n_p, n_q, -1)
                .mean(axis=1)
            )
            rb_l = np.asarray(rb[li], dtype=np.float64)
            Xg_L[li], Yg_L[li], rbL[li] = Xg, Yg, rb_l
            Xa = lmsys["cx_last"][:, lcol, :].float().numpy()
            Ya = lmsys["v_x"][:, lcol, :].float().numpy()
            ext["pv_raw"][li] = Xg.astype(np.float64) @ rb_l
            ext["map_generic"][li] = F.dot_readout(AH.GramRidge(Xa).predict(Ya, Xg), rb_l)
            fl = np.isfinite(lmsys_lab)
            ext["g_generic"][li] = AH.GramRidge(Xa[fl]).predict(lmsys_lab[fl][:, None], Xg).ravel()
            ext["oracle"][li] = Yg.astype(np.float64) @ rb_l

        rng = np.random.default_rng(SEED)
        outer = kfold(np.arange(n_p), N_OUTER, rng)

        per_fold_r = {m: [] for m in METHODS}
        sel_layers = {m: [] for m in METHODS}
        zpool = {m: np.full(n_p, np.nan) for m in METHODS}  # z-scored test reads pooled

        for te in outer:
            tr = np.setdiff1d(np.arange(n_p), te)
            for m in METHODS:
                if m in EXTERNAL:
                    # layer by outer-train correlation
                    li = max(
                        L,
                        key=lambda l: (
                            pear(ext[m][l][tr], Sg[tr])
                            if np.isfinite(pear(ext[m][l][tr], Sg[tr]))
                            else -np.inf
                        ),
                    )
                    read_te = ext[m][li][te]
                else:
                    inner = kfold(tr, N_INNER, np.random.default_rng(SEED + 7))

                    def inner_score(l):
                        vs = []
                        for iv in inner:
                            it = np.setdiff1d(tr, iv)
                            p = (
                                cmap(Xg_L[l], Yg_L[l], rbL[l], it, iv)
                                if m == "map_corpus"
                                else cg(Xg_L[l], Sg, it, iv)
                            )
                            f = np.isfinite(p) & np.isfinite(Sg[iv])
                            if f.sum() >= 3:
                                vs.append(pear(p[f], Sg[iv][f]))
                        return np.nanmean(vs) if vs else -np.inf

                    li = max(
                        L, key=lambda l: inner_score(l) if np.isfinite(inner_score(l)) else -np.inf
                    )
                    read_te = (
                        cmap(Xg_L[li], Yg_L[li], rbL[li], tr, te)
                        if m == "map_corpus"
                        else cg(Xg_L[li], Sg, tr, te)
                    )
                sel_layers[m].append(int(li))
                fin = np.isfinite(read_te) & np.isfinite(Sg[te])
                per_fold_r[m].append(pear(read_te[fin], Sg[te][fin]))
                zpool[m][te] = zsc(read_te)  # label-free per-fold standardization

        finS = np.isfinite(Sg)
        out = {
            "selected_layers": sel_layers,
            "per_fold_r": {},
            "z_pooled_r": {},
            "paired_vs_pv_raw": {},
        }
        for m in METHODS:
            pf = np.array(per_fold_r[m], dtype=float)
            out["per_fold_r"][m] = {
                "mean": float(np.nanmean(pf)),
                "sd": float(np.nanstd(pf)),
                "folds": [round(float(x), 3) for x in pf],
            }
            fin = finS & np.isfinite(zpool[m])
            out["z_pooled_r"][m] = {"point": pear(zpool[m][fin], Sg[fin]), "n": int(fin.sum())}

        for m in METHODS:
            if m == "pv_raw":
                continue
            # per-fold paired diff (each method at its own selected layer)
            pf_diff = np.array(per_fold_r[m], dtype=float) - np.array(
                per_fold_r["pv_raw"], dtype=float
            )
            # z-pooled paired bootstrap
            fin = finS & np.isfinite(zpool[m]) & np.isfinite(zpool["pv_raw"])
            a, b, y = zpool[m][fin], zpool["pv_raw"][fin], Sg[fin]
            rng2 = np.random.default_rng(SEED)
            d = np.array(
                [
                    pear(a[i], y[i]) - pear(b[i], y[i])
                    for i in (rng2.integers(0, len(y), len(y)) for _ in range(N_BOOT))
                ]
            )
            d = d[np.isfinite(d)]
            out["paired_vs_pv_raw"][m] = {
                "per_fold_diff_mean": float(np.nanmean(pf_diff)),
                "per_fold_diff_sd": float(np.nanstd(pf_diff)),
                "folds_positive": int(np.nansum(pf_diff > 0)),
                "n_folds": int(np.isfinite(pf_diff).sum()),
                "zpool_diff": float(pear(a, y) - pear(b, y)),
                "zpool_lo": float(np.quantile(d, 0.025)),
                "zpool_hi": float(np.quantile(d, 0.975)),
                "zpool_p_gt0": float((d > 0).mean()),
            }

        res["traits"][trait] = out
        print(f"\n=== {trait} === (per-fold mean +/- SD | z-pooled r):")
        for m in METHODS:
            pf = out["per_fold_r"][m]
            zp = out["z_pooled_r"][m]["point"]
            tag = ""
            if m != "pv_raw":
                p = out["paired_vs_pv_raw"][m]
                tag = f" | vs raw: fold-diff {p['per_fold_diff_mean']:+.3f} ({p['folds_positive']}/{p['n_folds']}+) | zpool-diff {p['zpool_diff']:+.3f} [{p['zpool_lo']:+.3f},{p['zpool_hi']:+.3f}] P>0={p['zpool_p_gt0']:.2f}"
            print(
                f"  {m:12s} fold {pf['mean']:+.3f}+/-{pf['sd']:.3f} | zpool {zp:+.3f} | layers {sel_layers[m]}{tag}"
            )

    outp = OUT_DIR / "persona_level_nested_cv.json"
    with open(outp, "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote", outp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
