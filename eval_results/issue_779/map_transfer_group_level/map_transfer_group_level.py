"""Issue #779 inline free-analysis: the HONEST persona-level head-to-head.

Correction to the earlier pv_raw_group_level run. That run compared pv_raw
against the stored `group_level_logo` map — which is fit LEAVE-ONE-PERSONA-OUT
ON THE TRAIT CORPUS ITSELF (in-distribution to the corpus, only the persona held
out). That is NOT the generic-LMSYS map whose per-prompt failure is #779's
headline. This run adds the deployable comparison: the GENERIC map (fit on the
5000 LMSYS contexts, a distribution disjoint from the corpus) read on the 60
held-out corpus personas at group level, vs pv_raw and vs the corpus-LOGO map.

Three group-level reads over the 60 corpus personas (mean of 40 questions each),
per trait at the frozen system layer, correlated with the persona's pooled mean
judge score, with paired bootstraps over the 60 groups:
  - pv_raw      = <c_group, r_B>                     (original PV; no fit)
  - map_generic = <M_A(c_group), r_B>, M_A fit on 5000 LMSYS  (deployable transfer)
  - map_corpusLOGO = <h_-i(c_group), r_B>, h fit on 59 corpus personas (in-dist)

Reuses arm_headline GramRidge + loaders verbatim.
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


def boot_ci(x, y, seed=SEED, n_boot=N_BOOT):
    fin = np.isfinite(x) & np.isfinite(y)
    x, y = x[fin], y[fin]
    rng = np.random.default_rng(seed)
    b = [pear(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
    b = [v for v in b if np.isfinite(v)]
    return {
        "point": pear(x, y),
        "lo": float(np.quantile(b, 0.025)),
        "hi": float(np.quantile(b, 0.975)),
        "n": len(x),
    }


def paired_diff(xa, xb, y, seed=SEED, n_boot=N_BOOT):
    """bootstrap of r(xa,y) - r(xb,y) over shared group resamples."""
    fin = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(y)
    xa, xb, y = xa[fin], xb[fin], y[fin]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        d.append(pear(xa[idx], y[idx]) - pear(xb[idx], y[idx]))
    d = np.asarray(d)
    return {
        "point": pear(xa, y) - pear(xb, y),
        "lo": float(np.quantile(d, 0.025)),
        "hi": float(np.quantile(d, 0.975)),
        "p_gt0": float((d > 0).mean()),
    }


def main() -> int:
    lmsys = AH.load_lmsys_bundle()
    lmsys_layers = list(lmsys["layers"])
    res = {
        "traits": {},
        "note": "generic map = fit on 5000 LMSYS pass_b, read on 60 held-out corpus personas",
    }

    for trait in C.TRAITS:
        li = AH.FROZEN_LAYERS[trait]["system"]
        rb = S1._load_rb(AH.COLLECT_DIR / "r_b", trait, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)
        rb_l = np.asarray(rb[li], dtype=np.float64)

        # corpus grouped vectors (60 personas x mean-of-40-questions)
        blob = AH.load_corpus(trait)
        n_p, n_q, n_r = blob["n_personas"], blob["n_questions"], blob["n_rollouts"]
        ccol = blob["layers"].index(li)
        Xb = blob["cx_last"][:, ccol, :].float().numpy()
        Yb = blob["v_x"][:, ccol, :].float().numpy().reshape(n_p * n_q, n_r, -1).mean(axis=1)
        scores = AH.corpus_scores(trait)
        Xq = Xb.reshape(n_p, n_q, -1)
        Yq = Yb.reshape(n_p, n_q, -1)
        Sq = scores.reshape(n_p, n_q, n_r)
        all_q = np.tile(np.arange(n_q), (n_p, 1))
        Xg, Yg, Sg = AH._grouped_vectors(Xq, Yq, Sq, all_q)

        # (1) pv_raw group
        pv_raw_g = Xg.astype(np.float64) @ rb_l
        # (2) generic map fit on all 5000 LMSYS contexts, read on corpus groups
        lcol = lmsys_layers.index(li)
        Xa = lmsys["cx_last"][:, lcol, :].float().numpy()
        Ya = lmsys["v_x"][:, lcol, :].float().numpy()
        gr_gen = AH.GramRidge(Xa)
        prof_gen = gr_gen.predict(Ya, Xg)
        map_generic_g = F.dot_readout(prof_gen, rb_l)
        # (3) corpus-LOGO map (the stored group_level_logo number)
        map_logo_g, _ = AH._logo_readout(Xg, Yg, rb_l)

        reads = {"pv_raw": pv_raw_g, "map_generic": map_generic_g, "map_corpusLOGO": map_logo_g}
        res["traits"][trait] = {
            "layer": int(li),
            "group_r": {k: boot_ci(v, Sg) for k, v in reads.items()},
            "paired_vs_pv_raw": {
                "map_generic_minus_pv_raw": paired_diff(map_generic_g, pv_raw_g, Sg),
                "map_corpusLOGO_minus_pv_raw": paired_diff(map_logo_g, pv_raw_g, Sg),
            },
        }
        gr = res["traits"][trait]["group_r"]
        pd_ = res["traits"][trait]["paired_vs_pv_raw"]
        print(
            f"[{trait} L{li}] pv_raw {gr['pv_raw']['point']:+.3f} | "
            f"map_generic {gr['map_generic']['point']:+.3f} "
            f"(vs raw {pd_['map_generic_minus_pv_raw']['point']:+.3f} "
            f"[{pd_['map_generic_minus_pv_raw']['lo']:+.3f},{pd_['map_generic_minus_pv_raw']['hi']:+.3f}] "
            f"P>0={pd_['map_generic_minus_pv_raw']['p_gt0']:.2f}) | "
            f"map_corpusLOGO {gr['map_corpusLOGO']['point']:+.3f} "
            f"(vs raw {pd_['map_corpusLOGO_minus_pv_raw']['point']:+.3f} "
            f"P>0={pd_['map_corpusLOGO_minus_pv_raw']['p_gt0']:.2f})"
        )

    out = OUT_DIR / "map_transfer_group_level.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
