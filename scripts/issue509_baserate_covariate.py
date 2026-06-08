"""#509 sycophancy re-analysis: does the bystander base rate add predictive power?

Adds the bystander's own base sycophancy rate as a covariate alongside the
geometry predictors (cosine_l20 = #470's registered predictor) on the #411
6-source x 23-bystander leakage panel, with source fixed effects.

Two framings, reported separately:
  (A) CLEAN target = absolute post-training leaked rate (trained_rate_411).
      base enters legitimately; answers "does prior disposition predict
      how much a bystander ends up expressing the behavior, beyond geometry".
  (B) Delta target = trained - base (the #470/#509 leakage DV). base is
      mechanically inside the DV (-1 coefficient), so base->delta is partly
      circular; we partial base OUT (a la #500) to read geometry's residual
      contribution, and flag the circularity explicitly.

Stats match #509's machinery: source-FE (within-source rank residualization),
source-cluster bootstrap CIs (B=5000), within-source permutation null (B=2000).
Single seed 42, n=138 cells -> directional, not production-grade.
"""

import json
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(42)
SRC = Path("eval_results/issue_480/_inputs/predictor_comparison.json")
OUT = Path("eval_results/issue_509/baserate_covariate")


def rankdata(x):
    x = np.asarray(x, float)
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, r)
    r = (sums / counts)[inv]
    return r


def within(vec, groups):
    """Subtract per-group mean (source fixed effect)."""
    vec = np.asarray(vec, float)
    out = vec.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = vec[m] - vec[m].mean()
    return out


def fe_spearman(x, y, groups):
    """Source-FE Spearman: corr of within-source-demeaned global ranks."""
    rx = within(rankdata(x), groups)
    ry = within(rankdata(y), groups)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _resid(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def fe_partial_spearman(x, y, z, groups):
    """corr(x,y | z) on within-source-demeaned ranks."""
    rx = within(rankdata(x), groups)
    ry = within(rankdata(y), groups)
    rz = within(rankdata(z), groups)
    Z = np.vstack([np.ones_like(rz), rz]).T
    ex, ey = _resid(rx, Z), _resid(ry, Z)
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def within_r2(y, preds, groups):
    """Within-source R^2 of demeaned-rank y explained by demeaned-rank preds."""
    ry = within(rankdata(y), groups)
    cols = [within(rankdata(p), groups) for p in preds]
    X = np.vstack([np.ones_like(ry), *cols]).T
    res = _resid(ry, X)
    ss_tot = (ry**2).sum()
    return float(1 - (res**2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def cluster_bootstrap(stat_fn, groups, B=5000):
    uniq = np.unique(groups)
    vals = []
    for _ in range(B):
        drawn = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in drawn])
        gb = np.concatenate([np.full((groups == g).sum(), i) for i, g in enumerate(drawn)])
        v = stat_fn(idx, gb)
        if not np.isnan(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def perm_p(x, y, groups, obs, B=2000):
    count = 0
    for _ in range(B):
        xp = x.copy()
        for g in np.unique(groups):
            m = groups == g
            xp[m] = RNG.permutation(xp[m])
        if abs(fe_spearman(xp, y, groups)) >= abs(obs) - 1e-12:
            count += 1
    return (count + 1) / (B + 1)


def analyze(cells, label):
    src = np.array([c["source"] for c in cells])
    base = np.array([c["bystander_base_rate"] for c in cells], float)
    geom = np.array([c["cosine_l20_baseline"] for c in cells], float)  # #470 registered
    trained = np.array([c["trained_rate_411"] for c in cells], float)
    delta = np.array([c["delta"] for c in cells], float)
    n = len(cells)

    res = {"label": label, "n_cells": n, "n_sources": len(np.unique(src))}
    for tgt_name, tgt, circular in [("trained_abs", trained, False), ("delta", delta, True)]:
        r_base = fe_spearman(base, tgt, src)
        r_geom = fe_spearman(geom, tgt, src)
        p_base_g = fe_partial_spearman(base, tgt, geom, src)  # base beyond geometry
        p_geom_b = fe_partial_spearman(geom, tgt, base, src)  # geometry beyond base
        r2_base = within_r2(tgt, [base], src)
        r2_geom = within_r2(tgt, [geom], src)
        r2_both = within_r2(tgt, [base, geom], src)
        ci_base = cluster_bootstrap(lambda idx, gb, t=tgt: fe_spearman(base[idx], t[idx], gb), src)
        ci_pbg = cluster_bootstrap(
            lambda idx, gb, t=tgt: fe_partial_spearman(base[idx], t[idx], geom[idx], gb), src
        )
        pv_base = perm_p(base, tgt, src, r_base)
        res[tgt_name] = {
            "circular_warning": circular,
            "rho_base_alone": r_base,
            "rho_base_alone_ci": ci_base,
            "rho_base_alone_perm_p": pv_base,
            "rho_geom_alone": r_geom,
            "partial_base_given_geom": p_base_g,
            "partial_base_given_geom_ci": ci_pbg,
            "partial_geom_given_base": p_geom_b,
            "within_r2_base_only": r2_base,
            "within_r2_geom_only": r2_geom,
            "within_r2_both": r2_both,
            "unique_r2_base": r2_both - r2_geom,
            "unique_r2_geom": r2_both - r2_base,
        }
    return res


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def diagnostic(cells):
    """Expose the two confounds: delta=trained-base (mechanical -1) AND
    trained~=base on non-leaking cells (trivial identity). Per-source + the
    null-vs-leaky split is the honest read."""
    by_src = {}
    for c in cells:
        by_src.setdefault(c["source"], []).append(c)
    per_source = {}
    for s, cc in by_src.items():
        base = [c["bystander_base_rate"] for c in cc]
        trn = [c["trained_rate_411"] for c in cc]
        dl = [c["delta"] for c in cc]
        per_source[s] = {
            "n_leaky_bystanders_absdelta_gt_0.10": int(
                sum(1 for c in cc if abs(c["delta"]) > 0.10)
            ),
            "rho_base_trained": spearman(base, trn),
            "rho_base_delta": spearman(base, dl),
            "max_delta": float(max(dl)),
        }
    leaky = {s for s in by_src if per_source[s]["n_leaky_bystanders_absdelta_gt_0.10"] >= 3}
    null = set(by_src) - leaky

    def pooled(srcs, key_y):
        sub = [c for c in cells if c["source"] in srcs]
        g = np.array([c["source"] for c in sub])
        b = np.array([c["bystander_base_rate"] for c in sub])
        y = np.array([c[key_y] for c in sub])
        return fe_spearman(b, y, g)

    split = {
        "leaky_sources": sorted(leaky),
        "null_sources": sorted(null),
        "base_to_trained_FE_null_only": pooled(null, "trained_rate_411"),
        "base_to_trained_FE_leaky_only": pooled(leaky, "trained_rate_411"),
        "base_to_delta_FE_null_only": pooled(null, "delta"),
        "base_to_delta_FE_leaky_only": pooled(leaky, "delta"),
    }
    return {"per_source": per_source, "null_vs_leaky_split": split}


def make_figure(cells, diag, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = diag["per_source"]
    srcs = sorted(ps, key=lambda s: -ps[s]["max_delta"])
    x = np.arange(len(srcs))
    rb_trn = [ps[s]["rho_base_trained"] for s in srcs]
    rb_dl = [ps[s]["rho_base_delta"] for s in srcs]
    leaky = set(diag["null_vs_leaky_split"]["leaky_sources"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.38
    ax.bar(x - w / 2, rb_trn, w, label="rho(base, trained_abs)", color="#4C72B0")
    ax.bar(x + w / 2, rb_dl, w, label="rho(base, delta)", color="#C44E52")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{s}\n({'leaky' if s in leaky else 'null'})" for s in srcs], rotation=0, fontsize=8
    )
    ax.set_ylabel("within-source Spearman rho")
    ax.set_title(
        "Base sycophancy rate vs leakage, per source (#411 panel, seed 42)\n"
        "trained_abs inflated by trivial identity on null sources; delta dragged by mechanical -1"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    with open(SRC) as f:
        cells = json.load(f)["cells"]
    full = analyze(cells, "all_138")
    live = analyze([c for c in cells if abs(c["delta"]) > 0.10], "live_21_absdelta_gt_0.10")
    diag = diagnostic(cells)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.json", "w") as f:
        json.dump({"full": full, "live_cells": live, "diagnostic": diag}, f, indent=2)
    make_figure(cells, diag, OUT / "baserate_per_source.png")

    def show(r):
        print(f"\n===== {r['label']}  (n={r['n_cells']}, sources={r['n_sources']}) =====")
        for tgt in ("trained_abs", "delta"):
            d = r[tgt]
            warn = "  [CIRCULAR: delta contains base]" if d["circular_warning"] else ""
            print(f"\n  TARGET = {tgt}{warn}")
            ci_b = tuple(round(x, 2) for x in d["rho_base_alone_ci"])
            ci_pbg = tuple(round(x, 2) for x in d["partial_base_given_geom_ci"])
            print(
                f"    base alone        rho_FE = {d['rho_base_alone']:+.3f}  "
                f"CI{ci_b}  perm_p={d['rho_base_alone_perm_p']:.3f}"
            )
            print(f"    geometry alone    rho_FE = {d['rho_geom_alone']:+.3f}   (cosine_l20, #470)")
            print(f"    base | geometry   partial= {d['partial_base_given_geom']:+.3f}  CI{ci_pbg}")
            print(f"    geometry | base   partial= {d['partial_geom_given_base']:+.3f}")
            print(
                f"    within-R2: base={d['within_r2_base_only']:+.3f}  "
                f"geom={d['within_r2_geom_only']:+.3f}  both={d['within_r2_both']:+.3f}  | "
                f"unique_base={d['unique_r2_base']:+.3f}  unique_geom={d['unique_r2_geom']:+.3f}"
            )

    show(full)
    show(live)

    print("\n===== DIAGNOSTIC: the two confounds =====")
    print("per-source within-source rank corr:")
    print(f"  {'source':<22}{'n_leaky':>8}{'rho(base,trn)':>14}{'rho(base,Δ)':>13}{'maxΔ':>8}")
    for s, v in diag["per_source"].items():
        print(
            f"  {s:<22}{v['n_leaky_bystanders_absdelta_gt_0.10']:>8}"
            f"{v['rho_base_trained']:>14.3f}{v['rho_base_delta']:>13.3f}{v['max_delta']:>8.3f}"
        )
    sp = diag["null_vs_leaky_split"]
    print(f"\n  leaky sources: {sp['leaky_sources']}   null sources: {sp['null_sources']}")
    print(
        f"  base->trained  FE  null-only={sp['base_to_trained_FE_null_only']:+.3f}  "
        f"leaky-only={sp['base_to_trained_FE_leaky_only']:+.3f}"
    )
    print(
        f"  base->delta    FE  null-only={sp['base_to_delta_FE_null_only']:+.3f}  "
        f"leaky-only={sp['base_to_delta_FE_leaky_only']:+.3f}"
    )
    print(f"\nwrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
