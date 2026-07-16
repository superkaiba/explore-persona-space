"""Issue #779 inline free-analysis: pv_raw persona-level head-to-head.

Question (user-chat, 2026-07-14): does the map's persona-level monitoring win
("averaging 40 questions per held-out persona lifts r to 0.66/0.89/0.53")
beat the ORIGINAL Persona Vectors method — the raw last-prompt-token
projection <c, r_B> — under the SAME persona-level averaging on the SAME 60
trait-corpus personas? The #779 grouped_contexts section computed the group
read only for the learned map h (LOGO ridge); the rig-condition-level pv_raw
read (5-8 constructed conditions) is not comparable. This script fills the
missing cell: pv_raw at group level, group-size sweep, and per-context
baseline, matched to the map's frozen system-mode layers.

No fits anywhere: r_B is fixed (extracted on the PV rig, disjoint from the 60
corpus personas), so the raw projection needs no LOGO. Pure GEMMs.

Reuses scripts/issue779_arm_headline.py loaders/helpers verbatim (same corpus
blobs, same judge-score drop handling, same _grouped_vectors + bootstrap).
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

from explore_persona_space.experiments.issue_779 import metrics as M  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
N_BOOT = 2000
SEED = 0
GROUP_SIZES = (1, 2, 5, 10, 20, 40)
K_DRAWS = 5


def cosine_rows(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return (x @ v) / (np.linalg.norm(x, axis=1) * np.linalg.norm(v) + 1e-30)


def main() -> int:
    res: dict = {
        "recipe": {
            "n_boot": N_BOOT,
            "seed": SEED,
            "group_sizes": list(GROUP_SIZES),
            "k_draws": K_DRAWS,
            "layers": "matched to arm_headline grouped_contexts (FROZEN_LAYERS[trait]['system'])",
            "direction": "pv_raw: <c_last, r_B> on RAW (unstandardized) context activations, "
            "verbatim stage-1 pv_raw convention; no fit, no LOGO (r_B is corpus-independent)",
            "corpus": str(AH.CORPUS_DIR),
            "rb_dir": str(AH.COLLECT_DIR / "r_b"),
        },
        "traits": {},
    }
    # The map's stored numbers for the comparison table.
    stored = json.load(open(PROJECT_ROOT / "eval_results/issue_779/arm_headline.json"))

    for trait in C.TRAITS:
        li = AH.FROZEN_LAYERS[trait]["system"]
        blob = AH.load_corpus(trait)
        n_p, n_q, n_r = blob["n_personas"], blob["n_questions"], blob["n_rollouts"]
        n_ctx = n_p * n_q
        col = blob["layers"].index(li) if hasattr(blob["layers"], "index") else li
        Xb = blob["cx_last"][:, col, :].float().numpy()
        rb = S1._load_rb(AH.COLLECT_DIR / "r_b", trait, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)
        rb_l = np.asarray(rb[li], dtype=np.float64)
        scores = AH.corpus_scores(trait)
        assert Xb.shape == (n_ctx, C.EXPECTED_HIDDEN)

        # Per-context baseline: <c, r_B> vs per-context mean judge score.
        with np.errstate(invalid="ignore"):
            gb = np.nanmean(scores, axis=1)
        dots_pc = Xb.astype(np.float64) @ rb_l
        coss_pc = cosine_rows(Xb, rb_l)
        per_context = {
            "dot": AH._pearson_boot_ci(dots_pc, gb, n_boot=N_BOOT, seed=SEED),
            "cos": AH._pearson_boot_ci(coss_pc, gb, n_boot=N_BOOT, seed=SEED),
        }

        # Group level: mean context vector per persona (40 questions), no fit.
        Xq = Xb.reshape(n_p, n_q, -1)
        Sq = scores.reshape(n_p, n_q, n_r)
        all_q = np.tile(np.arange(n_q), (n_p, 1))
        Xg, _yg, Sg = AH._grouped_vectors(Xq, Xq, Sq, all_q)  # Yq unused; pass Xq
        dots_g = Xg.astype(np.float64) @ rb_l
        coss_g = cosine_rows(Xg, rb_l)
        group_level = {
            "n_groups": int(n_p),
            "dot": AH._pearson_boot_ci(dots_g, Sg, n_boot=N_BOOT, seed=SEED),
            "cos": AH._pearson_boot_ci(coss_g, Sg, n_boot=N_BOOT, seed=SEED),
        }

        # Group-size sweep, mirroring run_section4's draw structure.
        sweep: dict[str, dict] = {}
        rng = np.random.default_rng(SEED)
        for s in GROUP_SIZES:
            n_draws = 1 if s == n_q else K_DRAWS
            draw_rs, draw_rs_cos = [], []
            for _k in range(n_draws):
                q_idx = np.stack([rng.choice(n_q, size=s, replace=False) for _ in range(n_p)])
                xg, _y, sg = AH._grouped_vectors(Xq, Xq, Sq, q_idx)
                fin = np.isfinite(sg)
                draw_rs.append(M.overall_pearson(xg[fin].astype(np.float64) @ rb_l, sg[fin]))
                draw_rs_cos.append(M.overall_pearson(cosine_rows(xg[fin], rb_l), sg[fin]))
            sweep[str(s)] = {
                "dot_r_mean": float(np.nanmean(draw_rs)),
                "dot_r_sd": float(np.nanstd(draw_rs)),
                "cos_r_mean": float(np.nanmean(draw_rs_cos)),
                "cos_r_sd": float(np.nanstd(draw_rs_cos)),
                "dot_r_draws": [float(v) for v in draw_rs],
            }

        # Diagnostic only: per-layer group-level dot r (selection-prone; not headline).
        layer_sweep = {}
        for lj, lname in enumerate(blob["layers"]):
            Xl = blob["cx_last"][:, lj, :].float().numpy()
            Xlg = Xl.reshape(n_p, n_q, -1).mean(axis=1)
            fin = np.isfinite(Sg)
            layer_sweep[str(lname)] = float(
                M.overall_pearson(
                    Xlg[fin].astype(np.float64) @ np.asarray(rb[int(lname)], dtype=np.float64),
                    Sg[fin],
                )
            )

        # Paired comparison: recompute the map's group-level LOGO dot readout on
        # the SAME groups (run_section4 code path verbatim), verify against the
        # stored point, then paired-bootstrap the difference r_h - r_pv_raw.
        _xb2, _vb, Yb = (
            blob["cx_last"][:, col, :].float().numpy(),
            None,
            blob["v_x"][:, col, :].float().numpy().reshape(n_ctx, n_r, -1).mean(axis=1),
        )
        Yq = Yb.reshape(n_p, n_q, -1)
        _xg2, Yg, _sg2 = AH._grouped_vectors(Xq, Yq, Sq, all_q)
        h_dots_g, _h_cos_g = AH._logo_readout(Xg, Yg, rb_l)
        fin_g = np.isfinite(Sg)
        h_group_point = M.overall_pearson(h_dots_g[fin_g], Sg[fin_g])
        rng_p = np.random.default_rng(SEED)
        idx_all = np.flatnonzero(fin_g)
        diffs = []
        for _ in range(N_BOOT):
            idx = rng_p.choice(idx_all, size=len(idx_all), replace=True)
            r_h = M.overall_pearson(h_dots_g[idx], Sg[idx])
            r_pv = M.overall_pearson(dots_g[idx], Sg[idx])
            if np.isfinite(r_h) and np.isfinite(r_pv):
                diffs.append(r_h - r_pv)
        diffs = np.asarray(diffs)
        paired = {
            "h_group_dot_recomputed": float(h_group_point),
            "diff_point_h_minus_pvraw": float(h_group_point - group_level["dot"]["point"]),
            "diff_lo": float(np.quantile(diffs, 0.025)),
            "diff_hi": float(np.quantile(diffs, 0.975)),
            "p_diff_gt_0": float((diffs > 0).mean()),
            "n_boot_kept": len(diffs),
        }

        h_stored = stored["grouped_contexts"][trait]
        res["traits"][trait] = {
            "layer": int(li),
            "pv_raw": {
                "group_level": group_level,
                "group_size_sweep": sweep,
                "per_context_baseline": per_context,
                "group_level_layer_sweep_diagnostic": layer_sweep,
            },
            "map_h_stored_reference": {
                "group_level_dot": h_stored["group_level_logo"]["dot"],
                "per_context_dot": h_stored["per_context_baseline"]["dot"],
            },
            "paired_h_minus_pvraw_group": paired,
        }
        print(
            f"[{trait} L{li}] pv_raw group dot r={group_level['dot']['point']:+.3f} "
            f"[{group_level['dot']['lo']:+.3f},{group_level['dot']['hi']:+.3f}] "
            f"(map h stored: {h_stored['group_level_logo']['dot']['point']:+.3f}, "
            f"recomputed: {paired['h_group_dot_recomputed']:+.3f}) | "
            f"per-ctx {per_context['dot']['point']:+.3f} "
            f"(map h: {h_stored['per_context_baseline']['dot']['point']:+.3f}) | "
            f"paired diff h-raw {paired['diff_point_h_minus_pvraw']:+.3f} "
            f"[{paired['diff_lo']:+.3f},{paired['diff_hi']:+.3f}] "
            f"P(diff>0)={paired['p_diff_gt_0']:.3f}"
        )

    out = OUT_DIR / "pv_raw_group_level.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
