"""Issue #2221 `specialized_corpus_remine` — analyzer fold-time reads.

Computes the pre-registered ANALYZER-side statistics the P8 battery does not
persist (plan v13 §3/§6 duties assigned to the fold):

1. FAMILY-grain paired bootstrap CI of the H2 delta (the registered lattice's
   BINDING grain; §3 dual-grain item) — resample the 8 families with
   replacement, keep all of a drawn family's versions, per-draw re-selection
   over both free axes on both sides (28 layers for arm a, 56 positions for
   arm c), B=10,000, seed = C.RNG_SEED.
2. Super-family LOFO refold (§6 OOD folds, v10 item 11): drop sycophancy +
   mistake_opinions TOGETHER (shared AITA corpus + rollout panel).
3. Mix-size partial Spearman controlling log(rows) (§6 install-strength /
   v10 item 10) at each arm's selected layer, per trait (paper panel).
4. Band-vs-ceiling line (§6, v10 item 8): honest max-selected null-band
   p97.5 (isotropic + covariance-matched + score-shuffle, re-reduced from the
   persisted draw matrices) NEXT TO the achievable rank ceiling — the max
   |Spearman| any predictor can reach against the tied paper-panel y (n=24).
5. H3 cap-effect bound (§3 H3 caveat): paired standing-cell y@1000 (parent
   `trait_scores.json`) vs y@2048 (this round), per trait — the instrument
   bound any H3 real-vs-synthetic gap must clear before a provenance reading.
6. Language-intrusion audit (Step 3.7): per-model CJK scan over the round's
   P6 rollouts joined with the paper-panel judge scores; zeroed / excluded
   recounts of the trait-matched paper graded means.

Inputs are all committed round artifacts (0 GPU-h). Output:
`eval_results/issue_2221/specialized_corpus_remine/analyzer_fold_reads.json`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.experiments.issue_2221 import constants as C  # noqa: E402
from explore_persona_space.experiments.issue_2221 import monitors as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RDIR = ROOT / "eval_results/issue_2221/specialized_corpus_remine"
SCAL = RDIR / "monitor_scalars"
DRAW = RDIR / "draw_matrices"
P6 = ROOT / "data/issue_2221/p6_remine"

TRAITS = ("evil", "sycophancy", "hallucination")
PANELS = ("paper", "lmsys", "pooled")
CJK = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

# Realized rows per cell: re-mined from mix_yield.json, standing from the
# parent mixes (parent clean-result Training table, sha-verified there).
STANDING_ROWS = {
    "hallucination": 1533,
    "insecure_code": 3357,
    "mistake_math": 760,
    "mistake_gsm8k": 520,
}


def load_round():
    corr = json.loads((RDIR / "correlations.json").read_text())
    ts = json.loads((RDIR / "trait_scores.json").read_text())
    my = json.loads((RDIR / "mix_yield.json").read_text())
    parent_ts = json.loads((ROOT / "eval_results/issue_2221/trait_scores.json").read_text())
    return corr, ts, my, parent_ts


def paper_y(ts: dict, cells: list[str], trait: str) -> np.ndarray:
    return np.array(
        [ts["scores"][c][trait]["per_panel"]["paper"]["graded_mean"] for c in cells],
        dtype=np.float64,
    )


def scalar_matrix(scal: dict, cells: list[str], arm: str) -> np.ndarray:
    return np.array([scal["scalars"][c][arm] for c in cells], dtype=np.float64)  # (n, 28)


def rank_ceiling(y: np.ndarray) -> float:
    """Max achievable |Spearman| against tied y: distinct-x comonotone arrangement."""
    n = len(y)
    order = np.argsort(y, kind="stable")
    x = np.empty(n)
    x[order] = np.arange(1, n + 1)
    return float(M.pearson_rows(M.rank_transform(x[None, :]), M.rank_transform(y[None, :]))[0])


def family_indices(cells: list[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    fams: dict[str, list[int]] = {}
    for i, c in enumerate(cells):
        fams.setdefault(C.family_of(c), []).append(i)
    names = sorted(fams)
    return names, {f: np.array(fams[f]) for f in names}


def family_bootstrap_indices(rng, names, fam_idx, n_draws):
    k = len(names)
    fam_draws = rng.integers(0, k, size=(n_draws, k))
    rows = []
    for d in range(n_draws):
        rows.append(np.concatenate([fam_idx[names[j]] for j in fam_draws[d]]))
    return np.stack(rows)  # (B, 24) — 8 families x 3 versions per draw


def main() -> None:
    corr, ts, my, parent_ts = load_round()
    cells = corr["config"]["cells"]
    rng = np.random.default_rng(C.RNG_SEED)
    out: dict = {"cells": cells, "n_bootstrap": 10_000, "seed": C.RNG_SEED}

    names, fam_idx = family_indices(cells)
    idx_fam = family_bootstrap_indices(rng, names, fam_idx, 10_000)

    # rows per cell for the mix-size partial
    rows_per_cell = []
    for c in cells:
        fam, ver = C.family_of(c), C.version_of(c)
        if fam in STANDING_ROWS:
            rows_per_cell.append(STANDING_ROWS[fam])
        else:
            rows_per_cell.append(my[f"{fam}/{ver}"]["n_rows"])
    log_rows = np.log10(np.array(rows_per_cell, dtype=np.float64))
    out["rows_per_cell"] = dict(zip(cells, rows_per_cell))

    per_trait: dict = {}
    for trait in TRAITS:
        y = paper_y(ts, cells, trait)
        tr: dict = {}

        # ceiling + honest null p97.5 (pooled-panel nulls, as persisted)
        tr["rank_ceiling_paper_y"] = rank_ceiling(y)
        nz = np.load(DRAW / f"{trait}_nulls.npz")
        tr["null_p975_abs"] = {
            "a_isotropic": float(np.percentile(np.abs(M.select_per_draw(nz["null_iso_a"])), 97.5)),
            "a_covmatched": float(np.percentile(np.abs(M.select_per_draw(nz["null_cov_a"])), 97.5)),
            "c56_isotropic": float(
                np.percentile(np.abs(M.select_per_draw(nz["null_iso_c56"])), 97.5)
            ),
            "c56_covmatched": float(
                np.percentile(np.abs(M.select_per_draw(nz["null_cov_c56"])), 97.5)
            ),
        }
        tr["null_q95_abs"] = {
            "a_isotropic": float(M.q95_abs(M.select_per_draw(nz["null_iso_a"]))),
            "a_covmatched": float(M.q95_abs(M.select_per_draw(nz["null_cov_a"]))),
            "c56_isotropic": float(M.q95_abs(M.select_per_draw(nz["null_iso_c56"]))),
            "c56_covmatched": float(M.q95_abs(M.select_per_draw(nz["null_cov_c56"]))),
        }

        # per-panel family-grain H2 + shuffle p97.5 re-reductions
        tr["panels"] = {}
        for panel in PANELS:
            scal = json.loads((SCAL / f"{trait}_{panel}.json").read_text())
            xa = scalar_matrix(scal, cells, "a_rb_ctx")
            xc = np.concatenate(
                [scalar_matrix(scal, cells, "c_map_ctx"), scalar_matrix(scal, cells, "c_map_pfx")],
                axis=1,
            )
            boot_a = M.bootstrap_pearson(xa.T, y, idx_fam)
            boot_c = M.bootstrap_pearson(xc.T, y, idx_fam)
            delta = M.select_per_draw(boot_c) - M.select_per_draw(boot_a)
            pz = np.load(DRAW / f"{trait}_{panel}.npz")
            pn = {
                "h2_family_grain_ci": M.percentile_ci(delta),
                "h2_family_grain_median": float(np.median(delta)),
                "shuffle_p975_abs_a": float(
                    np.percentile(np.abs(M.select_per_draw(pz["shuffle_r__a_rb_ctx"])), 97.5)
                ),
                "shuffle_p975_abs_c56": float(
                    np.percentile(np.abs(M.select_per_draw(pz["shuffle_r__c56"])), 97.5)
                ),
            }
            # super-family LOFO refold (drop sycophancy + mistake_opinions together)
            keep = np.array(
                [C.family_of(c) not in ("sycophancy", "mistake_opinions") for c in cells]
            )
            refold = {}
            for arm, x in (("a_rb_ctx", xa), ("c_map_ctx+pfx(56)", xc)):
                r_full = M.spearman_by_position(x.T, y)
                pos_full, _ = M.select_position(r_full)
                r_sub = M.spearman_by_position(x[keep].T, y[keep])
                pos_sub, r_sel_sub = M.select_position(r_sub)
                refold[arm] = {
                    "n_kept": int(keep.sum()),
                    "r_at_full_selected_pos": float(r_sub[pos_full]),
                    "reselected_pos": pos_sub,
                    "reselected_r": r_sel_sub,
                }
            pn["superfamily_lofo_refold"] = refold
            # mix-size partial at each arm's full-sample selected position
            partials = {}
            for arm in ("a_rb_ctx", "b_rb_ans", "c_map_ctx", "c_map_pfx", "d_transport"):
                if arm not in scal["scalars"][cells[0]]:
                    continue
                x = scalar_matrix(scal, cells, arm)
                r_by = M.spearman_by_position(x.T, y)
                pos, r_sel = M.select_position(r_by)
                partials[arm] = {
                    "selected_layer": pos,
                    "r_raw": r_sel,
                    "r_partial_log_rows": M.partial_spearman(x[:, pos], y, log_rows),
                }
            pn["mix_size_partial"] = partials
            tr["panels"][panel] = pn

        # y-vs-rows confound read + H3 cap bound
        tr["spearman_y_vs_log_rows"] = float(
            M.pearson_rows(M.rank_transform(log_rows[None, :]), M.rank_transform(y[None, :]))[0]
        )
        standing_cells = [c for c in cells if C.family_of(c) in STANDING_ROWS]
        y_old = paper_y(parent_ts, standing_cells, trait)
        y_new = paper_y(ts, standing_cells, trait)
        d = y_new - y_old
        tr["h3_cap_effect_bound"] = {
            "n_paired_standing_cells": len(standing_cells),
            "mean_delta": float(d.mean()),
            "mean_abs_delta": float(np.abs(d).mean()),
            "max_abs_delta": float(np.abs(d).max()),
            "max_abs_cell": standing_cells[int(np.argmax(np.abs(d)))],
            "spearman_old_new": float(
                M.pearson_rows(M.rank_transform(y_old[None, :]), M.rank_transform(y_new[None, :]))[
                    0
                ]
            ),
            "base_delta": float(
                paper_y(ts, ["base"], trait)[0] - paper_y(parent_ts, ["base"], trait)[0]
            ),
        }
        per_trait[trait] = tr
    out["per_trait"] = per_trait

    # ── Step 3.7 CJK language-intrusion audit over the round's judged pools ──
    trait_of_family = {
        "evil": "evil",
        "sycophancy": "sycophancy",
        "hallucination": "hallucination",
        "insecure_code": "hallucination",
        "mistake_math": "hallucination",
        "mistake_gsm8k": "hallucination",
        "mistake_medical": "hallucination",
        "mistake_opinions": "hallucination",
    }
    audit: dict = {}
    tot_rows = tot_intruded = 0
    for model in cells + ["base"]:
        roll = json.loads((P6 / "eval_rollouts" / f"{model}.json").read_text())
        rows = roll["rows"]
        intr = {f"{r['surface_id']}-s{r['seed']}" for r in rows if CJK.search(r["response"])}
        tot_rows += len(rows)
        tot_intruded += len(intr)
        trait = trait_of_family[C.family_of(model)] if model != "base" else "hallucination"
        judge = json.loads((P6 / "judge" / f"{model}_{trait}.json").read_text())["scores"]
        paper_keys = [k for k in judge if k.startswith("paper-")]
        vals = {k: judge[k] for k in paper_keys}
        mean = float(np.mean(list(vals.values()))) if vals else float("nan")
        zeroed = (
            float(np.mean([0.0 if k in intr else v for k, v in vals.items()]))
            if vals
            else float("nan")
        )
        kept = [v for k, v in vals.items() if k not in intr]
        excl = float(np.mean(kept)) if kept else float("nan")
        fired = sum(1 for k, v in vals.items() if v > 50 and k in intr)
        audit[model] = {
            "n_rows": len(rows),
            "n_intruded": len(intr),
            "trait_matched": trait,
            "paper_graded_mean": mean,
            "paper_graded_mean_zeroed": zeroed,
            "paper_graded_mean_excluded": excl,
            "n_paper_intruded_judge_positive": fired,
        }
    out["cjk_audit"] = {
        "per_model": audit,
        "total_rows": tot_rows,
        "total_intruded": tot_intruded,
        "note": "rows = all P6 rollouts (paper+lmsys); recounts on the paper-panel trait-matched judge pool",
    }

    (RDIR / "analyzer_fold_reads.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"written": str(RDIR / "analyzer_fold_reads.json")}, indent=1))


if __name__ == "__main__":
    main()
