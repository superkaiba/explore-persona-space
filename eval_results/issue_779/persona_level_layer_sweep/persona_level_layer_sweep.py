"""Issue #779 inline free-analysis: persona-level head-to-head across ALL layers.

Ask (chat 2026-07-14): "did we check at different layers?" No — every prior
group-level run used only the frozen system layer per trait (evil L14 / syco L26
/ halluc L17, selected on the eval rig). This sweeps all 28 layers.

Per trait per layer, group-level Pearson r over the 60 corpus personas (mean of
40 questions), vs the persona's pooled mean judge score, for three reads:
  - pv_raw      = <c_group, r_B>                     (original PV; no fit)
  - map_generic = <M_A(c_group), r_B>, M_A fit on 5000 LMSYS  (deployable)
  - map_LOGO    = <h_-i(c_group), r_B>, h fit on 59 corpus personas (in-dist)

Full per-layer curves; paired diff (generic - raw) bootstrapped at the frozen
layer AND the generic-map argmax layer (argmax read flagged selection-prone).
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


def paired_diff(xa, xb, y, seed=SEED, n_boot=N_BOOT):
    fin = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(y)
    xa, xb, y = xa[fin], xb[fin], y[fin]
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


def main() -> int:
    lmsys = AH.load_lmsys_bundle()
    L = list(lmsys["layers"])
    res = {"traits": {}, "layers": [int(x) for x in L]}

    for trait in C.TRAITS:
        frozen = AH.FROZEN_LAYERS[trait]["system"]
        rb = S1._load_rb(AH.COLLECT_DIR / "r_b", trait, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)
        blob = AH.load_corpus(trait)
        n_p, n_q, n_r = blob["n_personas"], blob["n_questions"], blob["n_rollouts"]
        scores = AH.corpus_scores(trait)
        Sq = scores.reshape(n_p, n_q, n_r)
        all_q = np.tile(np.arange(n_q), (n_p, 1))
        # group scores are layer-independent
        _, _, Sg = AH._grouped_vectors(
            blob["cx_last"][:, 0, :].float().numpy().reshape(n_p, n_q, -1),
            blob["cx_last"][:, 0, :].float().numpy().reshape(n_p, n_q, -1),
            Sq,
            all_q,
        )
        fin = np.isfinite(Sg)

        raw_curve, gen_curve, logo_curve = {}, {}, {}
        reads_at = {}  # layer -> (raw, gen, logo) vectors, kept for frozen+argmax
        for li in L:
            li = int(li)
            ccol = blob["layers"].index(li)
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

            raw = Xg.astype(np.float64) @ rb_l
            lcol = L.index(li)
            gr = AH.GramRidge(lmsys["cx_last"][:, lcol, :].float().numpy())
            prof = gr.predict(lmsys["v_x"][:, lcol, :].float().numpy(), Xg)
            gen = F.dot_readout(prof, rb_l)
            logo, _ = AH._logo_readout(Xg, Yg, rb_l)

            raw_curve[li] = pear(raw[fin], Sg[fin])
            gen_curve[li] = pear(gen[fin], Sg[fin])
            logo_curve[li] = pear(logo[fin], Sg[fin])
            reads_at[li] = (raw, gen, logo)

        argmax_gen = int(max(gen_curve, key=gen_curve.get))
        best_gen_beats_raw = {li: gen_curve[li] - raw_curve[li] for li in L}
        n_layers_gen_gt_raw = int(sum(1 for li in L if gen_curve[int(li)] > raw_curve[int(li)]))

        def diff_at(li):
            raw, gen, _ = reads_at[li]
            return paired_diff(gen, raw, Sg)

        res["traits"][trait] = {
            "frozen_layer": int(frozen),
            "raw_curve": raw_curve,
            "generic_curve": gen_curve,
            "corpusLOGO_curve": logo_curve,
            "generic_argmax_layer": argmax_gen,
            "n_layers_generic_beats_raw": n_layers_gen_gt_raw,
            "paired_generic_minus_raw_at_frozen": diff_at(frozen),
            "paired_generic_minus_raw_at_generic_argmax": diff_at(argmax_gen),
            "raw_argmax_layer": int(max(raw_curve, key=raw_curve.get)),
            "raw_max": float(max(raw_curve.values())),
            "generic_max": float(max(gen_curve.values())),
        }
        r = res["traits"][trait]
        print(
            f"[{trait}] frozen L{frozen}: raw {raw_curve[frozen]:+.3f} gen {gen_curve[frozen]:+.3f} "
            f"(diff {r['paired_generic_minus_raw_at_frozen']['point']:+.3f} P>0={r['paired_generic_minus_raw_at_frozen']['p_gt0']:.2f}) | "
            f"gen best L{argmax_gen} {gen_curve[argmax_gen]:+.3f} vs raw@thatL {raw_curve[argmax_gen]:+.3f} "
            f"(diff {r['paired_generic_minus_raw_at_generic_argmax']['point']:+.3f} P>0={r['paired_generic_minus_raw_at_generic_argmax']['p_gt0']:.2f}) | "
            f"raw best L{r['raw_argmax_layer']} {r['raw_max']:+.3f} | gen>raw at {n_layers_gen_gt_raw}/28 layers"
        )

    out = OUT_DIR / "persona_level_layer_sweep.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
