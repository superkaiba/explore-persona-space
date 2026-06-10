"""Length-nuisance supplement for issue #540 (round-2 interpretation revision).

Persists every prose-only statistic the round-1 critics flagged as untraceable,
plus two new free (zero-GPU) robustness checks they requested:

1. Paired bootstrap for rho(length) - rho(js_rb) on the ordinary strip and on
   the no-enumerated-context subset (both CIs span zero -> "no better than",
   not "beats").
2. The UN-normalized Rao-Blackwellized JS (total bits per reply, the
   paper-canonical sum over positions rather than the project's per-token
   mean), recovered exactly from the committed per-sample records
   (kl_side_m_bits_per_token * n_positions), re-run through the same raw /
   length-partial reads.
3. Both length-partialling conventions side by side: the figure script's
   Spearman-of-rank-residuals and analysis_jsrb.json's Pearson-on-rank-
   residuals.
4. Marker-count conventions for the instructed contexts: glyph-ends-with vs
   exact ` ※` token (id 83399) anywhere, per context, with base prior and
   trained mean emission alongside.
5. Cap-censoring of the no-enumerated subset (median pair truncation).

Inputs are all committed artifacts; output is
eval_results/issue_540/length_nuisance_supplement.json.
"""

import glob
import json
import os

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

WT = "/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-540"
RAW_DIR = "/tmp/i540-raw-completions"  # pinned HF revision 0848b13e local mirror
MARKER_TOKEN_ID = 83399  # " ※" (leading space)
SEED = 42
N_BOOT = 10_000  # the no-D5 subset delta CI sits at the zero boundary; 1k reps is too coarse


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Figure-script convention: Spearman of OLS rank-residuals (re-ranks residuals)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        design = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ coef

    rho, p = spearmanr(resid(rx, rz), resid(ry, rz))
    return float(rho), float(p)


def partial_pearson_on_ranks(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """analysis_jsrb.json convention: Pearson correlation of OLS rank-residuals."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        design = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ coef

    r, p = pearsonr(resid(rx, rz), resid(ry, rz))
    return float(r), float(p)


def paired_boot_delta(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> dict:
    """Paired bootstrap CI for |rho(x1,y)| - |rho(x2,y)| over the same cells."""
    n = len(y)
    point = abs(spearmanr(x1, y)[0]) - abs(spearmanr(x2, y)[0])
    deltas = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        if len(set(y[idx])) < 2:
            continue
        deltas.append(abs(spearmanr(x1[idx], y[idx])[0]) - abs(spearmanr(x2[idx], y[idx])[0]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_abs_rho_point": float(point),
        "ci95": [float(lo), float(hi)],
        "n_boot": len(deltas),
        "n_cells": n,
    }


def main() -> None:
    pred = json.load(open(f"{WT}/eval_results/issue_540/predictors_jsrb.json"))
    srcs = pred["sources"]
    bys = pred["bystanders"]

    def mat(name: str) -> dict[tuple[str, str], float]:
        return {(s, c): pred[name][i][j] for i, s in enumerate(srcs) for j, c in enumerate(bys)}

    mats = {
        "js_rb": mat("js_rb_matrix"),
        "js_v1": mat("js_v1_matrix"),
        "gauss_kl": mat("gauss_kl_matrix"),
        "cosine": mat("cosine_matrix"),
    }
    base_prior = pred["base_prior"]  # dict keyed by bystander name

    emis: dict[tuple[str, str], float] = {}
    for f in glob.glob(f"{WT}/eval_results/issue_532/per_cell/loc_ep1/cell_loc_ep1_*.json"):
        a, b = f.split("cell_loc_ep1_")[-1][:-5].split("__")
        emis[(a, b)] = json.load(open(f))["summary"]["in_R_emission_rate"]

    # Pair-level: mean reply length per side, truncation rate, un-normalized RB JS.
    pairlen: dict[tuple[str, str], tuple[float, float]] = {}
    pairtrunc: dict[tuple[str, str], float] = {}
    js_unnorm: dict[tuple[str, str], float] = {}
    for f in glob.glob(f"{WT}/eval_results/issue_540/per_pair/pair_*.json"):
        d = json.load(open(f))
        a, b = d["pair"]["a"], d["pair"]["b"]
        ps = d["per_sample"]
        la = float(np.mean([r["n_positions"] for r in ps if r["side"] == "a"]))
        lb = float(np.mean([r["n_positions"] for r in ps if r["side"] == "b"]))
        pairlen[(a, b)] = (la, lb)
        pairtrunc[(a, b)] = d["truncation"]["n_truncated"] / d["truncation"]["n_rows"]
        ta = [r["kl_side_m_bits_per_token"] * r["n_positions"] for r in ps if r["side"] == "a"]
        tb = [r["kl_side_m_bits_per_token"] * r["n_positions"] for r in ps if r["side"] == "b"]
        js_unnorm[(a, b)] = float(0.5 * (np.mean(ta) + np.mean(tb)))

    def pairget(d: dict, x: str, y: str, diag_val: float = 0.0) -> float:
        if x == y:
            return diag_val
        return d[(x, y)] if (x, y) in d else d[(y, x)]

    def dlen(x: str, y: str) -> float:
        if x == y:
            return 0.0
        la, lb = pairlen[(x, y)] if (x, y) in pairlen else pairlen[(y, x)][::-1]
        return abs(la - lb)

    ordinary = [(s, c) for s in srcs for c in srcs]  # 256, diagonal included
    y = np.array([emis[c] for c in ordinary])
    xd = np.array([dlen(*c) for c in ordinary])
    xunn = np.array([pairget(js_unnorm, *c) for c in ordinary])
    is_diag = np.array([a == b for a, b in ordinary])
    no_d5 = np.array([(not d) and ("D5" not in c) for c, d in zip(ordinary, is_diag)])

    rng = np.random.default_rng(SEED)
    out: dict = {
        "schema_version": "issue540_supplement_v1",
        "seed": SEED,
        "n_boot": N_BOOT,
        "conventions": {
            "partial_figure": "Spearman of OLS rank-residuals (re-ranks residuals); "
            "matches length_nuisance_ordinary.png and body prose",
            "partial_analysis_json": "Pearson of OLS rank-residuals; matches "
            "analysis_jsrb.json length_nuisance.rho_ordinary_partial_length",
            "js_rb_unnormalized": "0.5*(E_a + E_b) of per-reply total bits = "
            "kl_side_m_bits_per_token * n_positions per committed per-sample record "
            "(paper-canonical sum over positions, no per-token normalization); diagonal = 0",
        },
    }

    def strip_stats(mask: np.ndarray, label: str) -> dict:
        ym, xdm, xum = y[mask], xd[mask], xunn[mask]
        res: dict = {"n": int(mask.sum())}
        r, p = spearmanr(xdm, ym)
        res["length_alone"] = {"rho": float(r), "p": float(p)}
        for name in ["js_rb", "js_v1", "gauss_kl", "cosine"]:
            xm = np.array([mats[name][c] for c in ordinary])[mask]
            rr, rp = spearmanr(xm, ym)
            pf, pfp = partial_spearman(xm, ym, xdm)
            pa, pap = partial_pearson_on_ranks(xm, ym, xdm)
            res[name] = {
                "raw_rho": float(rr),
                "raw_p": float(rp),
                "partial_length_figure_convention": {"rho": pf, "p": pfp},
                "partial_length_analysis_convention": {"rho": pa, "p": pap},
                "rho_with_length": float(spearmanr(xm, xdm)[0]),
            }
        # reverse partials (length controlled for predictor)
        for name in ["js_rb", "gauss_kl"]:
            xm = np.array([mats[name][c] for c in ordinary])[mask]
            rv, rvp = partial_spearman(xdm, ym, xm)
            res[f"length_partial_{name}_figure_convention"] = {"rho": rv, "p": rvp}
        # un-normalized RB
        rr, rp = spearmanr(xum, ym)
        pf, pfp = partial_spearman(xum, ym, xdm)
        res["js_rb_unnormalized"] = {
            "raw_rho": float(rr),
            "raw_p": float(rp),
            "partial_length_figure_convention": {"rho": pf, "p": pfp},
            "rho_with_length": float(spearmanr(xum, xdm)[0]),
        }
        # paired bootstrap: length vs js_rb
        xj = np.array([mats["js_rb"][c] for c in ordinary])[mask]
        res["paired_boot_abs_rho_length_minus_js_rb"] = paired_boot_delta(xdm, xj, ym, rng)
        return res

    out["ordinary_full"] = strip_stats(np.ones(len(ordinary), dtype=bool), "full")
    out["ordinary_offdiag"] = strip_stats(~is_diag, "offdiag")
    out["ordinary_no_d5_offdiag"] = strip_stats(no_d5, "no_d5")

    # cap-censoring of the no-D5 off-diagonal subset
    sub_trunc = [pairget(pairtrunc, *c, diag_val=np.nan) for c, m in zip(ordinary, no_d5) if m]
    out["ordinary_no_d5_offdiag"]["median_pair_truncation_rate"] = float(np.median(sub_trunc))
    out["ordinary_no_d5_offdiag"]["length_sd_tokens"] = float(np.std(xd[no_d5]))
    all_trunc = [
        pairget(pairtrunc, *c, diag_val=np.nan) for c, d in zip(ordinary, is_diag) if not d
    ]
    out["ordinary_offdiag"]["median_pair_truncation_rate"] = float(np.median(all_trunc))

    # instructed-context marker counts under both conventions + prior/trained emission
    instr = [c for c in bys if c.startswith("instr_")]
    counts: dict = {}
    for c in instr:
        rows = json.load(open(os.path.join(RAW_DIR, f"samples_{c}.json")))
        replies = [draw for probe_draws in rows["samples"] for draw in probe_draws]
        n = len(replies)
        glyph_end = token_any = token_last = glyph_any = 0
        for r in replies:
            text = r.get("text", r.get("completion", ""))
            ids = r.get("token_ids", [])
            if text.rstrip().endswith("※"):
                glyph_end += 1
            if "※" in text:
                glyph_any += 1
            if MARKER_TOKEN_ID in ids:
                token_any += 1
            if ids and ids[-1] == MARKER_TOKEN_ID:
                token_last += 1
        trained_mean = float(np.mean([emis[(s, c)] for s in srcs]))
        counts[c] = {
            "n_replies": n,
            "glyph_ends_with": glyph_end,
            "glyph_anywhere": glyph_any,
            "marker_token_83399_anywhere": token_any,
            "marker_token_83399_last": token_last,
            "base_prior_parent_column": base_prior[c],
            "trained_mean_emission": trained_mean,
        }
    out["instructed_marker_counts"] = counts

    path = f"{WT}/eval_results/issue_540/length_nuisance_supplement.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
