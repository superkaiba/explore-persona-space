"""Issue #779 inline free-analysis: persona-level head-to-head across ALL layers,
for ALL pre-generation monitoring methods.

Extends persona_level_layer_sweep.py (raw / map_generic / map_corpusLOGO) with
the direct predictors and a post-generation oracle ceiling. Per trait per layer,
group-level Pearson r over the 60 corpus personas (mean of 40 questions) vs the
persona's pooled mean judge score:

  pre-generation reads:
    pv_raw       = <c_group, r_B>                          (original PV; no fit)
    map_generic  = <M_A(c_group), r_B>, M_A fit on 5000 LMSYS (deployable map)
    map_LOGO     = <h_-i(c_group), r_B>, h fit on 59 corpus personas (in-dist map)
    g_generic    = ridge(c -> LMSYS judge label) fit on 5000 LMSYS, read on corpus
                   (deployable direct predictor; LMSYS labels sparse for evil/syco)
    g_LOGO       = ridge(c_group -> group score) fit on 59 corpus personas, LOGO
                   (in-distribution direct predictor; the #779 r~0.91 analog)
  post-generation ceiling reference (NOT a pre-gen monitor):
    oracle       = <answer_profile_group, r_B>             (reads the real answer)

Full per-layer curves; paired diff (method - pv_raw) bootstrapped at each
method's own argmax layer AND the frozen layer (argmax reads flagged
selection-prone). Reuses arm_headline GramRidge + loaders verbatim.
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
N_BOOT = 2000
SEED = 0


def pear(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def paired_diff(xa, xb, y, seed=SEED, n_boot=N_BOOT):
    fin = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(y)
    xa, xb, y = xa[fin], xb[fin], y[fin]
    if len(y) < 5:
        return {
            "point": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "p_gt0": float("nan"),
        }
    rng = np.random.default_rng(seed)
    d = np.array(
        [
            pear(xa[i], y[i]) - pear(xb[i], y[i])
            for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))
        ]
    )
    return {
        "point": pear(xa, y) - pear(xb, y),
        "lo": float(np.quantile(d, 0.025)),
        "hi": float(np.quantile(d, 0.975)),
        "p_gt0": float((d > 0).mean()),
    }


def logo_g(Xg, yg, seed=SEED):
    """Leave-one-group-out ridge c_group -> group score (scalar target)."""
    n = len(Xg)
    pred = np.full(n, np.nan)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        m = mask & np.isfinite(yg)
        gr = AH.GramRidge(Xg[m])
        pred[i] = gr.predict(yg[m][:, None], Xg[i : i + 1]).ravel()[0]
    return pred


def main() -> int:
    lmsys = AH.load_lmsys_bundle()
    L = [int(x) for x in lmsys["layers"]]
    res = {"traits": {}, "layers": L}

    for trait in C.TRAITS:
        frozen = AH.FROZEN_LAYERS[trait]["system"]
        rb = S1._load_rb(AH.COLLECT_DIR / "r_b", trait, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)
        blob = AH.load_corpus(trait)
        n_p, n_q, n_r = blob["n_personas"], blob["n_questions"], blob["n_rollouts"]
        scores = AH.corpus_scores(trait)
        Sq = scores.reshape(n_p, n_q, n_r)
        all_q = np.tile(np.arange(n_q), (n_p, 1))
        base = blob["cx_last"][:, 0, :].float().numpy().reshape(n_p, n_q, -1)
        _, _, Sg = AH._grouped_vectors(base, base, Sq, all_q)
        fin = np.isfinite(Sg)
        lmsys_lab = AH.lmsys_labels(trait, lmsys["cx_last"].shape[0])

        methods = ["pv_raw", "map_generic", "map_LOGO", "g_generic", "g_LOGO", "oracle"]
        curves = {m: {} for m in methods}
        reads_at = {}

        for li in L:
            ccol = blob["layers"].index(li)
            lcol = L.index(li)
            rb_l = np.asarray(rb[li], dtype=np.float64)
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
            Xa = lmsys["cx_last"][:, lcol, :].float().numpy()
            Ya = lmsys["v_x"][:, lcol, :].float().numpy()

            r = {}
            r["pv_raw"] = Xg.astype(np.float64) @ rb_l
            gr_map = AH.GramRidge(Xa)
            r["map_generic"] = F.dot_readout(gr_map.predict(Ya, Xg), rb_l)
            r["map_LOGO"] = AH._logo_readout(Xg, Yg, rb_l)[0]
            # deployable direct predictor: fit on LMSYS labels (valid rows), read on corpus groups
            fl = np.isfinite(lmsys_lab)
            r["g_generic"] = AH.GramRidge(Xa[fl]).predict(lmsys_lab[fl][:, None], Xg).ravel()
            # in-distribution direct predictor: LOGO ridge c_group->group score
            r["g_LOGO"] = logo_g(Xg, Sg)
            r["oracle"] = Yg.astype(np.float64) @ rb_l  # reads the real answer profile

            for m in methods:
                v = r[m]
                curves[m][li] = pear(v[fin], Sg[fin]) if np.std(v[fin]) > 0 else float("nan")
            reads_at[li] = r

        def summ(m):
            c = {li: curves[m][li] for li in L if np.isfinite(curves[m][li])}
            amax = int(max(c, key=c.get)) if c else frozen
            return {
                "curve": curves[m],
                "argmax_layer": amax,
                "max": float(c[amax]) if c else float("nan"),
                "at_frozen": curves[m][frozen],
                "n_layers_beats_raw": int(
                    sum(
                        1
                        for li in L
                        if np.isfinite(curves[m][li])
                        and np.isfinite(curves["pv_raw"][li])
                        and curves[m][li] > curves["pv_raw"][li]
                    )
                ),
                "paired_vs_raw_at_frozen": paired_diff(
                    reads_at[frozen][m], reads_at[frozen]["pv_raw"], Sg
                ),
                "paired_vs_raw_at_own_argmax": paired_diff(
                    reads_at[amax][m], reads_at[amax]["pv_raw"], Sg
                ),
            }

        res["traits"][trait] = {"frozen_layer": frozen, "methods": {m: summ(m) for m in methods}}
        t = res["traits"][trait]["methods"]
        print(
            f"\n=== {trait} (frozen L{frozen}) === raw@frozen {t['pv_raw']['at_frozen']:+.3f} | raw best L{t['pv_raw']['argmax_layer']} {t['pv_raw']['max']:+.3f}"
        )
        for m in ["map_generic", "map_LOGO", "g_generic", "g_LOGO", "oracle"]:
            s = t[m]
            print(
                f"  {m:12s} @frozen {s['at_frozen']:+.3f} (vs raw {s['paired_vs_raw_at_frozen']['point']:+.3f} P>0={s['paired_vs_raw_at_frozen']['p_gt0']:.2f}) | "
                f"best L{s['argmax_layer']} {s['max']:+.3f} (vs raw@thatL {s['paired_vs_raw_at_own_argmax']['point']:+.3f} P>0={s['paired_vs_raw_at_own_argmax']['p_gt0']:.2f}) | beats raw {s['n_layers_beats_raw']}/28"
            )

    out = OUT_DIR / "persona_level_layer_sweep_allmethods.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
