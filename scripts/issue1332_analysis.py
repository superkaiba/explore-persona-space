"""Issue #1332 P2+/P3 — evaluation battery, baselines, nulls, verdict, figures (VM).

Plan v3 §4.6-§4.7 + §6. Consumes the P2 similarity matrices + the REUSED #532
corrected-slot leakage matrix; computes:

- headline Spearman rho(S_sym, L) over the off-diagonal cells (raw +
  prompt-token-length-difference-partial, the #474 convention), two-way
  cluster bootstrap CIs (B=2,000), three stylized-exclusion panels;
- LOFO (leave-one-TARGET-family-out primary; leave-one-SOURCE-out companion)
  with group-level n framing + the {baselines} vs {baselines + S_sym}
  hierarchy Delta-CV-R^2 (H2);
- the pre-registered KILL: partial Spearman rho(S_sym, L | cosine_532, JS_540)
  on identical rows (+ the registered same-bank/same-layer sensitivity
  partial | fresh capture cosine at L*), collinearity gate at 0.6 with
  tercile + deg-2 residualization fallback reads;
- shuffled-pairing null (B=10,000, batched rank-GEMM; per-draw x per-layer
  matrices persisted) + band-vs-ceiling report (selection-symmetric-nulls);
- split-half ceilings: r_SS (from P2) x r_LL (probe-ALIGNED split-half over
  the 50 per_q probes, Spearman-Brown; llm-judging rule 21) -> attenuation
  ceiling beside every rho;
- baselines on identical rows: committed #532 cosine (L21) + #540 RB-JS +
  base prior + whitened gate (#667 recipe RECOMPUTED on OUR bank at L14,
  lambda = 1e-2*tr(Sigma_c)/d + {0.1x,10x} sweep + Sigma=I reduction test) +
  predict-the-mean; fresh capture cosine both arms (diagnostic);
- sensitivity battery: EOS-margin DV, within-source z-normed L, S_agree /
  S_excess / S_dmap predictor swaps, per-layer rho curve (diagnostic);
- the registered verdict lattice (Confirmed / Redressed / Suppressed /
  Reversed / Inconclusive) + figures + ``analysis.json``.

USAGE
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \\
      uv run python scripts/issue1332_analysis.py --full
    uv run python scripts/issue1332_analysis.py --smoke --n-null 200 --n-boot 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1332_common as C

logger = logging.getLogger("issue1332.analysis")

COLLINEARITY_GATE = 0.6
N_NULL_DEFAULT = 10_000
N_BOOT_DEFAULT = 2_000
NULL_SEED = 1
BOOT_SEED = 0
R_LL_PARTITIONS = 200


# ── rank / correlation helpers ────────────────────────────────────────────────


def _ranks(x):
    from scipy.stats import rankdata

    return rankdata(x)


def spearman(x, y) -> float:
    import numpy as np
    from scipy.stats import spearmanr

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def partial_spearman(x, y, covs: list) -> float:
    """Partial Spearman: rank-transform all, OLS-residualize x and y on covs."""
    import numpy as np

    rx, ry = _ranks(x), _ranks(y)
    Z = np.column_stack([_ranks(c) for c in covs] + [np.ones(len(rx))])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    ex, ey = rx - Z @ bx, ry - Z @ by
    if np.std(ex) == 0 or np.std(ey) == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def two_way_cluster_bootstrap(stat_fn, n_s: int, n_t: int, *, n_boot: int, seed: int) -> dict:
    """Percentile CI over B draws resampling sources AND targets with replacement.

    ``stat_fn(src_idx, tgt_idx) -> float | nan``; nan draws are dropped
    (reported). Group-level resampling — the CI is framed on 16/26 clusters.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        si = rng.integers(0, n_s, size=n_s)
        ti = rng.integers(0, n_t, size=n_t)
        v = stat_fn(si, ti)
        if v == v:  # not nan
            vals.append(v)
    vals = np.sort(np.asarray(vals))
    if len(vals) < max(20, n_boot // 10):
        return {"ci_lo": float("nan"), "ci_hi": float("nan"), "n_effective": len(vals)}
    return {
        "ci_lo": float(np.quantile(vals, 0.025)),
        "ci_hi": float(np.quantile(vals, 0.975)),
        "n_effective": len(vals),
    }


def masked_cells(M, mask):
    return M[mask]


# ── loading ───────────────────────────────────────────────────────────────────


def load_similarity(res_dir: Path, layer: int) -> dict:
    return json.loads((res_dir / "similarity" / f"S_transfer_L{layer}.json").read_text())


def sub_matrix(S26, families: list[str], sources: list[str], targets: list[str]):
    """Slice the 26x26 similarity matrix to the (sources x targets) panel."""
    import numpy as np

    S = np.asarray(S26, dtype=float)
    fi = {f: i for i, f in enumerate(families)}
    rows = [fi[s] for s in sources]
    cols = [fi[t] for t in targets]
    return S[np.ix_(rows, cols)]


def prompt_length_matrix(bank, rewrites, sources, targets):
    """|mean prompt tokens(source) - mean prompt tokens(target)| nuisance (16, 26)."""
    import numpy as np

    sys.path.insert(0, str(C.PROJECT_ROOT / "scripts"))
    from issue1332_gpu_phase import get_tokenizer

    tok = get_tokenizer()
    panel = C.instructed_panel()
    fams = sorted(set(sources) | set(targets))
    mean_len = {}
    for fam in fams:
        lens = []
        for q in bank:
            p, _ = C.render_family_prompt(fam, q, tok, rewrites, panel)
            lens.append(len(tok.encode(p, add_special_tokens=False)))
        mean_len[fam] = float(np.mean(lens))
    out = np.zeros((len(sources), len(targets)))
    for i, s in enumerate(sources):
        for j, t in enumerate(targets):
            out[i, j] = abs(mean_len[s] - mean_len[t])
    return out, mean_len


# ── whitened gate + fresh cosine (from OUR capture store) ─────────────────────


def capture_family_means(store_dir: Path, families: list[str], layer: int, key: str):
    """Per-family mean state at one layer + the pooled row matrix (for Sigma_c)."""
    import numpy as np
    import torch

    means, pooled = {}, []
    for fam in families:
        sh = torch.load(store_dir / f"{fam}.pt", map_location="cpu", mmap=True, weights_only=False)
        X = sh[key][:, layer, :].float().numpy()
        means[fam] = X.mean(axis=0)
        pooled.append(X)
    return means, np.concatenate(pooled, axis=0)


def whitened_gate_matrix(store_dir: Path, sources, targets, families, layer: int) -> dict:
    """#667 whitened gate g0 recomputed on OUR bank at L14 (+ lambda sweep)."""
    import numpy as np
    import torch

    from explore_persona_space.analysis.issue667.gate_chain import (
        default_lambda,
        whitened_gate,
        whitened_gate_reduction_unit_test,
    )

    whitened_gate_reduction_unit_test()  # Sigma=I reduction test (reused, #667)
    means, pooled = capture_family_means(store_dir, families, layer, "cx_last")
    # Sigma_c = second moment over all (family, query) rows (plan §4.6)
    P = torch.from_numpy(pooled.astype("float64"))
    sigma = (P.T @ P) / P.shape[0]
    lam0 = default_lambda(sigma)
    out = {}
    for mult, tag in ((0.1, "0.1x"), (1.0, "1x"), (10.0, "10x")):
        lam = lam0 * mult
        M = np.zeros((len(sources), len(targets)))
        for i, s in enumerate(sources):
            for j, t in enumerate(targets):
                M[i, j] = whitened_gate(
                    torch.from_numpy(means[s]), torch.from_numpy(means[t]), sigma, lam
                )
        out[tag] = M
    return {"matrices": out, "lambda_1x": lam0, "layer": layer}


def fresh_cosine_matrices(store_dir: Path, sources, targets, families, layers: list[int]) -> dict:
    """#536 globally-mean-centered bank cosine on mean cx_last (context arm) and
    mean prefix_end (prefix arm), per layer (diagnostic; #532 stays registered)."""
    import numpy as np
    import torch

    from explore_persona_space.analysis.representation_shift import compute_cosine_matrix

    out = {}
    fam_index = {f: i for i, f in enumerate(families)}
    for key, arm in (("cx_last", "context"), ("prefix_end", "prefix")):
        for layer in layers:
            means, _ = capture_family_means(store_dir, families, layer, key)
            bank = torch.stack([torch.from_numpy(means[f]).float() for f in families])
            cosM = compute_cosine_matrix(bank, centering="global_mean").numpy()
            M = np.zeros((len(sources), len(targets)))
            for i, s in enumerate(sources):
                for j, t in enumerate(targets):
                    M[i, j] = cosM[fam_index[s], fam_index[t]]
            out[f"{arm}_L{layer}"] = M
    return out


# ── LOFO + hierarchy ──────────────────────────────────────────────────────────


def lofo_predictions(L, feats: dict, mask, targets: list[str], sources: list[str], axis: str):
    """Leave-one-group-out OLS of L on features; pooled held-out predictions.

    ``axis`` = "target" (26 folds, primary) or "source" (16 folds, companion).
    Features include source/target one-hot additive effects (predict-the-mean);
    a held-out group's one-hot is unseen in training -> its effect is 0.
    """
    import numpy as np

    n_s, n_t = L.shape
    feat_names = sorted(feats)
    cells = [(i, j) for i in range(n_s) for j in range(n_t) if mask[i, j]]
    X = np.zeros((len(cells), len(feat_names) + n_s + n_t + 1))
    y = np.array([L[i, j] for i, j in cells])
    groups = []
    for r, (i, j) in enumerate(cells):
        for k, fn in enumerate(feat_names):
            X[r, k] = feats[fn][i, j]
        X[r, len(feat_names) + i] = 1.0  # source effect
        X[r, len(feat_names) + n_s + j] = 1.0  # target effect
        X[r, -1] = 1.0  # intercept
        groups.append(j if axis == "target" else i)
    groups = np.asarray(groups)
    preds = np.full(len(cells), np.nan)
    fold_rho = {}
    fold_ids = sorted(set(groups.tolist()))
    for g in fold_ids:
        te = groups == g
        tr = ~te
        b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        preds[te] = X[te] @ b
        fold_rho[g] = spearman(preds[te], y[te]) if int(te.sum()) >= 3 else float("nan")
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    labels = targets if axis == "target" else sources
    return {
        "cv_r2": 1.0 - ss_res / (ss_tot + 1e-12),
        "pooled_spearman": spearman(preds, y),
        "per_fold_spearman": {labels[g]: fold_rho[g] for g in fold_ids},
        "n_folds": len(fold_ids),
    }


# ── shuffled-pairing null (batched rank-GEMM; selection-symmetric rule) ───────


def shuffled_pairing_null(S16, L, mask, *, n_draws: int, seed: int, axis: str = "target"):
    """B draws permuting the target (or source) labels of S; Spearman per draw.

    Batched: gather the permuted S cells -> rank along the draw axis via
    scipy rankdata(axis=1) -> one centered GEMM against the L ranks.
    """
    import numpy as np
    from scipy.stats import rankdata

    rng = np.random.default_rng(seed)
    n_s, n_t = S16.shape
    flat_mask = mask.reshape(-1)
    l_r = rankdata(L[mask])
    l_c = l_r - l_r.mean()
    draws = np.empty((n_draws, int(mask.sum())))
    for b in range(n_draws):
        if axis == "target":
            perm = rng.permutation(n_t)
            Sp = S16[:, perm]
        else:
            perm = rng.permutation(n_s)
            Sp = S16[perm, :]
        draws[b] = Sp.reshape(-1)[flat_mask]
    dr = rankdata(draws, axis=1)
    dc = dr - dr.mean(axis=1, keepdims=True)
    num = dc @ l_c
    den = np.sqrt((dc**2).sum(axis=1) * (l_c**2).sum()) + 1e-12
    return num / den  # (n_draws,)


def r_ll_probe_aligned(per_q_trained, per_q_base, sources, targets, mask, *, n_partitions: int):
    """Probe-ALIGNED split-half reliability of L (llm-judging rule 21) + SB."""
    import numpy as np

    n_probes = len(next(iter(per_q_trained.values())))
    rng = np.random.default_rng(C.SPLIT_HALF_SEED)
    rs = []
    cells = [
        (i, s, j, t) for i, s in enumerate(sources) for j, t in enumerate(targets) if mask[i, j]
    ]
    for _ in range(n_partitions):
        perm = rng.permutation(n_probes)
        a_idx, b_idx = perm[: n_probes // 2], perm[n_probes // 2 :]
        ha, hb = [], []
        for _i, s, _j, t in cells:
            tr = np.asarray(per_q_trained[(s, t)])
            ba = np.asarray(per_q_base[(s, t)])
            ha.append(tr[a_idx].mean() - ba[a_idx].mean())
            hb.append(tr[b_idx].mean() - ba[b_idx].mean())
        rs.append(spearman(ha, hb))
    r_half = float(np.nanmean(rs))
    r_sb = 2 * r_half / (1 + r_half) if r_half > -1 else float("nan")
    return {
        "r_half_mean": r_half,
        "r_LL_spearman_brown": r_sb,
        "n_partitions": n_partitions,
        "scheme": "probe-ALIGNED random partitions (one partition applied to every cell), "
        "mean r BEFORE Spearman-Brown; no non-negativity floor applied",
    }


# ── figures ───────────────────────────────────────────────────────────────────


def make_figures(fig_dir: Path, ctx: dict) -> list[str]:
    """Hero scatter + forest + exploratory dump (paper style; constrained layout)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception:
        logger.warning("paper style unavailable; default rcParams")
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    S, L, mask = ctx["S16"], ctx["L"], ctx["mask"]
    sources, targets = ctx["sources"], ctx["targets"]
    styl = set(C.STYLIZED_CIDS)

    # HERO (a): S_sym vs L scatter, colored by stylized class
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    colors = []
    for i, s in enumerate(sources):
        for j, t in enumerate(targets):
            if mask[i, j]:
                colors.append("#D55E00" if (s in styl or t in styl) else "#0072B2")
    xs, ys = S[mask], L[mask]
    ax.scatter(xs, ys, c=colors, s=14, alpha=0.75, linewidths=0)
    ax.set_xlabel("map-transfer similarity S_sym (held-out R²)")
    ax.set_ylabel("leakage L (delta log P(marker), trained - base)")
    ax.set_title(f"S_sym vs leakage — {int(mask.sum())} off-diagonal cells (L*={ctx['l_star']})")
    p = fig_dir / "hero_scatter_S_vs_L.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    made.append(str(p))

    # HERO (b): incremental-validity forest
    fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
    rows = ctx["forest_rows"]
    ys = np.arange(len(rows))[::-1]
    for y, (label, val, lo, hi) in zip(ys, rows, strict=True):
        ax.errorbar([val], [y], xerr=[[val - lo], [hi - val]], fmt="o", color="#0072B2")
        ax.text(-0.02, y, label, ha="right", va="center", transform=ax.get_yaxis_transform())
    ax.axvline(0.0, color="0.5", lw=0.8)
    ax.set_yticks([])
    ax.set_xlabel("Spearman rho (cluster-bootstrap 95% CI) / ΔCV-R²")
    ax.set_title("Incremental validity — raw + partial reads")
    p = fig_dir / "forest_incremental_validity.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    made.append(str(p))

    # Exploratory: heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    im0 = axes[0].imshow(ctx["S26"], cmap="viridis")
    axes[0].set_title("S_sym (26x26)")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(L, cmap="magma", aspect="auto")
    axes[1].set_title("leakage L (16x26)")
    fig.colorbar(im1, ax=axes[1])
    p = fig_dir / "heatmaps_S_and_L.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    made.append(str(p))

    # Exploratory: per-layer rho curve (diagnostic display, no max claim)
    if ctx.get("per_layer_rho"):
        fig, ax = plt.subplots(figsize=(5.5, 3.5), layout="constrained")
        layers = sorted(ctx["per_layer_rho"])
        ax.plot(layers, [ctx["per_layer_rho"][k] for k in layers], marker="o")
        ax.set_xlabel("layer")
        ax.set_ylabel("Spearman rho(S_sym, L)")
        ax.set_title("per-layer rho curve (diagnostic; headline layer frozen DV-independently)")
        p = fig_dir / "per_layer_rho_curve.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(str(p))

    # Exploratory: S vs cosine collinearity scatter
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.scatter(ctx["cos532"][mask], S[mask], s=12, alpha=0.7, color="#009E73", linewidths=0)
    ax.set_xlabel("committed #532 activation cosine (L21)")
    ax.set_ylabel("S_sym")
    ax.set_title(f"collinearity: Pearson={ctx['pearson_S_cos']:.3f}")
    p = fig_dir / "S_vs_cosine_collinearity.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    made.append(str(p))

    # Exploratory: LOFO per-fold skill bars
    lofo = ctx["lofo_target_full"]["per_fold_spearman"]
    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    keys = list(lofo)
    ax.bar(range(len(keys)), [0 if lofo[k] != lofo[k] else lofo[k] for k in keys], color="#0072B2")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90, fontsize=6)
    ax.set_ylabel("held-out fold Spearman rho")
    ax.set_title("LOFO (leave-one-target-family-out) per-fold skill — full model")
    p = fig_dir / "lofo_per_fold_bars.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    made.append(str(p))

    # Prefix-arm + margin-DV replicas of the hero scatter (raw alongside)
    for key, xlab, fname in (
        (
            "S_mean_target16",
            "mean-target transfer (degenerate prefix arm)",
            "prefix_arm_mean_target_scatter.png",
        ),
        ("L_margin_x_S", "S_sym", "margin_dv_scatter.png"),
    ):
        if key == "S_mean_target16":
            xs2, ys2 = ctx["S_mean_target16"][mask], L[mask]
        else:
            xs2, ys2 = S[mask], ctx["L_margin"][mask]
        fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
        ax.scatter(xs2, ys2, s=12, alpha=0.7, color="#CC79A7", linewidths=0)
        ax.set_xlabel(xlab)
        ax.set_ylabel("L" if key == "S_mean_target16" else "L (EOS-margin DV)")
        p = fig_dir / fname
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(str(p))
    return made


# ── verdict lattice ───────────────────────────────────────────────────────────


def verdict_lattice(rho, rho_ci, partial, partial_ci) -> str:
    """DISJOINT + exhaustive lattice (plan §3)."""
    lo, hi = rho_ci
    plo, phi = partial_ci

    def _pos(ci):
        return ci[0] > 0

    def _neg(ci):
        return ci[1] < 0

    if any(v != v for v in (rho, lo, hi, partial, plo, phi)):
        return "Inconclusive"
    if _neg((lo, hi)):
        return "Reversed"
    if rho > 0 and _pos((lo, hi)) and partial > 0 and _pos((plo, phi)):
        return "Confirmed"
    if _pos((lo, hi)) and _neg((plo, phi)):
        return "Suppressed"
    if _pos((lo, hi)):
        return "Redressed"
    return "Inconclusive"


# ── driver ────────────────────────────────────────────────────────────────────


def main() -> int:
    """Analysis driver: join S with L, run the registered battery, write outputs."""
    ap = argparse.ArgumentParser(description="Issue #1332 analysis (VM CPU)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--n-null", type=int, default=N_NULL_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument(
        "--skip-length-partial",
        action="store_true",
        help="skip the tokenizer-based length nuisance (smoke speed)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import numpy as np
    import torch

    torch.set_num_threads(args.n_threads)

    res_dir = C.results_dir(args.smoke, args.results_dir)
    store = C.data_root(args.smoke, args.out_root) / "store" / "capture"
    fig_dir = C.figures_dir(args.smoke)

    C.phase("p3_load")
    freeze = json.loads((res_dir / "layer_freeze.json").read_text())
    l_star = freeze["l_star"]
    sim = load_similarity(res_dir, l_star)
    families = sim["families"]
    leak = C.load_leakage_matrices()
    all_sources, all_targets = leak["sources"], leak["targets"]
    # panel = the families actually captured (production: all 26)
    sources = [s for s in all_sources if s in families]
    targets = [t for t in all_targets if t in families]
    si = {s: i for i, s in enumerate(all_sources)}
    ti = {t: i for i, t in enumerate(all_targets)}
    rows = [si[s] for s in sources]
    cols = [ti[t] for t in targets]
    L = leak["L"][np.ix_(rows, cols)]
    L_margin = leak["L_margin"][np.ix_(rows, cols)]
    base_prior = leak["base_prior"][np.ix_(rows, cols)]
    mask = C.offdiag_mask(sources, targets)
    panel_full = len(sources) == 16 and len(targets) == 26

    S26 = np.asarray(sim["S_sym"], dtype=float)
    S16 = sub_matrix(sim["S_sym"], families, sources, targets)
    S_agree16 = sub_matrix(sim["S_agree"], families, sources, targets)
    S_excess16 = sub_matrix(sim["S_excess"], families, sources, targets)
    S_dmap16 = sub_matrix(sim["S_dmap_one_minus"], families, sources, targets)
    S_mean_target16 = sub_matrix(sim["S_mean_target"], families, sources, targets)

    base = C.load_baseline_matrices()
    cos532 = base["cosine_532"][np.ix_(rows, cols)]
    js540 = base["js_rb_540"][np.ix_(rows, cols)]

    C.phase("p3_baselines")
    gate = whitened_gate_matrix(
        store,
        sources,
        targets,
        families,
        min(C.WHITENED_GATE_LAYER, l_star) if args.smoke else C.WHITENED_GATE_LAYER,
    )
    gate_1x = gate["matrices"]["1x"]
    fresh_layers = [lay for lay in (7, 14, 21, 27) if not args.smoke] or [l_star]
    fresh_cos = fresh_cosine_matrices(
        store, sources, targets, families, sorted({*fresh_layers, l_star})
    )
    fresh_cos_lstar = fresh_cos[f"context_L{l_star}"]

    # length nuisance
    if args.skip_length_partial:
        len_diff, mean_len = None, {}
    else:
        inputs_dir = C.data_root(args.smoke, args.out_root) / "inputs"
        bank = C.load_bank(C.ensure_input(inputs_dir / C.BANK_FILE, f"inputs/{C.BANK_FILE}"))
        rewrites = C.load_rewrites(
            C.ensure_input(inputs_dir / C.REWRITES_FILE, f"inputs/{C.REWRITES_FILE}")
        )["rewrites"]
        len_diff, mean_len = prompt_length_matrix(bank, rewrites, sources, targets)

    C.phase("p3_headline")
    panels = {
        "all": mask,
        "stylized_excluded_as_source": mask
        & ~np.isin(
            np.arange(len(sources)), [sources.index(c) for c in C.STYLIZED_CIDS if c in sources]
        )[:, None],
        "stylized_excluded_either_side": mask
        & ~np.isin(
            np.arange(len(sources)), [sources.index(c) for c in C.STYLIZED_CIDS if c in sources]
        )[:, None]
        & ~np.isin(
            np.arange(len(targets)), [targets.index(c) for c in C.STYLIZED_CIDS if c in targets]
        )[None, :],
    }

    def rho_on(M_pred, M_dv, m):
        return spearman(M_pred[m], M_dv[m])

    def boot_rho(M_pred, M_dv, m):
        def _stat(sidx, tidx):
            sub_mask = m[np.ix_(sidx, tidx)]
            return spearman(
                M_pred[np.ix_(sidx, tidx)][sub_mask], M_dv[np.ix_(sidx, tidx)][sub_mask]
            )

        return two_way_cluster_bootstrap(
            _stat, len(sources), len(targets), n_boot=args.n_boot, seed=BOOT_SEED
        )

    headline = {}
    for pname, m in panels.items():
        entry = {"n_cells": int(m.sum()), "rho": rho_on(S16, L, m), "boot": boot_rho(S16, L, m)}
        if len_diff is not None:
            entry["rho_length_partial"] = partial_spearman(S16[m], L[m], [len_diff[m]])
        headline[pname] = entry

    C.phase("p3_kill")
    m = panels["all"]
    kill = {
        "partial_rho_S_L_given_cos_js": partial_spearman(S16[m], L[m], [cos532[m], js540[m]]),
        "partial_rho_S_L_given_cos": partial_spearman(S16[m], L[m], [cos532[m]]),
        "partial_rho_S_L_given_js": partial_spearman(S16[m], L[m], [js540[m]]),
        "partial_rho_S_L_given_fresh_cos_lstar": partial_spearman(
            S16[m], L[m], [fresh_cos_lstar[m]]
        ),
    }

    def boot_partial(covs):
        def _stat(sidx, tidx):
            sm = m[np.ix_(sidx, tidx)]
            return partial_spearman(
                S16[np.ix_(sidx, tidx)][sm],
                L[np.ix_(sidx, tidx)][sm],
                [cv[np.ix_(sidx, tidx)][sm] for cv in covs],
            )

        return two_way_cluster_bootstrap(
            _stat, len(sources), len(targets), n_boot=args.n_boot, seed=BOOT_SEED
        )

    kill["boot_partial_given_cos_js"] = boot_partial([cos532, js540])
    kill["pearson_S_cos"] = (
        float(np.corrcoef(S16[m], cos532[m])[0, 1]) if m.sum() >= 3 else float("nan")
    )
    if kill["pearson_S_cos"] == kill["pearson_S_cos"] and kill["pearson_S_cos"] > COLLINEARITY_GATE:
        terciles = np.quantile(cos532[m], [1 / 3, 2 / 3])
        buckets = np.digitize(cos532[m], terciles)
        kill["collinearity_gate_fired"] = True
        kill["tercile_rho"] = {
            f"tercile_{b}": spearman(S16[m][buckets == b], L[m][buckets == b]) for b in (0, 1, 2)
        }
        cosv = cos532[m]
        Z = np.column_stack([cosv, cosv**2, np.ones(len(cosv))])
        bx, *_ = np.linalg.lstsq(Z, S16[m], rcond=None)
        by, *_ = np.linalg.lstsq(Z, L[m], rcond=None)
        kill["poly2_residualized_rho"] = spearman(S16[m] - Z @ bx, L[m] - Z @ by)
    else:
        kill["collinearity_gate_fired"] = False

    C.phase("p3_lofo")
    feats_base = {
        "cos532": cos532,
        "js540": js540,
        "base_prior": base_prior,
        "whitened_gate": gate_1x,
    }
    feats_full = dict(feats_base, S_sym=S16)
    lofo = {
        "target_base": lofo_predictions(L, feats_base, m, targets, sources, "target"),
        "target_full": lofo_predictions(L, feats_full, m, targets, sources, "target"),
        "source_base": lofo_predictions(L, feats_base, m, targets, sources, "source"),
        "source_full": lofo_predictions(L, feats_full, m, targets, sources, "source"),
        "s_only_target": lofo_predictions(L, {"S_sym": S16}, m, targets, sources, "target"),
    }
    lofo["delta_cv_r2_target"] = lofo["target_full"]["cv_r2"] - lofo["target_base"]["cv_r2"]

    C.phase("p3_nulls")
    # per-draw x per-layer null matrices (persisted; selection-symmetric rule)
    layer_files = sorted((res_dir / "similarity").glob("S_transfer_L*.json"))
    null_mat = {}
    per_layer_rho = {}
    for lf in layer_files:
        sim_l = json.loads(lf.read_text())
        lay = sim_l["layer"]
        S16_l = sub_matrix(sim_l["S_sym"], sim_l["families"], sources, targets)
        per_layer_rho[lay] = rho_on(S16_l, L, m)
        null_mat[lay] = shuffled_pairing_null(
            S16_l, L, m, n_draws=args.n_null, seed=NULL_SEED, axis="target"
        )
    null_stack = np.stack([null_mat[k] for k in sorted(null_mat)], axis=1)  # (B, n_layers)
    np.savez_compressed(
        res_dir / "null_matrices.npz",
        draws=null_stack,
        layers=np.asarray(sorted(null_mat)),
        seed=NULL_SEED,
    )
    null_lstar = null_mat[l_star]
    null_p975_abs = float(np.quantile(np.abs(null_lstar), 0.975))
    null_p = float((np.abs(null_lstar) >= abs(headline["all"]["rho"])).mean())
    null_src = shuffled_pairing_null(
        S16, L, m, n_draws=args.n_null, seed=NULL_SEED + 1, axis="source"
    )

    C.phase("p3_ceiling")
    sh_files = sorted((res_dir / "splithalf").glob("splithalf_L*.json"))
    r_ss = json.loads(sh_files[-1].read_text())["r_SS"] if sh_files else float("nan")
    r_ll = r_ll_probe_aligned(
        {k: v for k, v in leak["per_q_trained"].items() if k[0] in sources and k[1] in targets},
        {k: v for k, v in leak["per_q_base"].items() if k[0] in sources and k[1] in targets},
        sources,
        targets,
        mask,
        n_partitions=R_LL_PARTITIONS,
    )
    ceiling = (
        float(np.sqrt(max(0.0, r_ss) * max(0.0, r_ll["r_LL_spearman_brown"])))
        if r_ss == r_ss
        else float("nan")
    )
    band_vs_ceiling = {
        "null_band_p975_abs_rho": null_p975_abs,
        "attenuation_ceiling": ceiling,
        "margin": ceiling - null_p975_abs,
        "uninformative_by_construction": bool(ceiling == ceiling and null_p975_abs >= ceiling),
        "note": "if the ceiling falls below the null band the test is declared "
        "uninformative-by-construction; non-rejections are narrated failure-to-reject (plan §6)",
    }

    C.phase("p3_sensitivity")
    L_z = (L - L.mean(axis=1, keepdims=True)) / (L.std(axis=1, keepdims=True) + 1e-12)
    sensitivity = {
        "rho_margin_dv": rho_on(S16, L_margin, m),
        "rho_within_source_z": rho_on(S16, L_z, m),
        "rho_S_agree": rho_on(S_agree16, L, m),
        "rho_S_excess": rho_on(S_excess16, L, m),
        "rho_S_dmap": rho_on(S_dmap16, L, m),
        "rho_mean_target_prefix_arm": rho_on(S_mean_target16, L, m),
        "rho_whitened_gate": {tag: rho_on(Mx, L, m) for tag, Mx in gate["matrices"].items()},
        "rho_cos532": rho_on(cos532, L, m),
        "rho_js540": rho_on(js540, L, m),
        "rho_base_prior": rho_on(base_prior, L, m),
        "rho_fresh_cos": {k: rho_on(v, L, m) for k, v in fresh_cos.items()},
        "per_layer_rho": per_layer_rho,
        "sym_variance_share_16x16": None,
    }
    # analyzer concern 4: asymmetric-variance share on the 16x16 subgrid
    common = [t for t in targets if t in sources]
    if len(common) >= 4:
        idx_s = [sources.index(cid) for cid in common]
        idx_t = [targets.index(cid) for cid in common]
        Lsq = L[np.ix_(idx_s, idx_t)]
        sensitivity["sym_variance_share_16x16"] = float(
            (np.var(Lsq - Lsq.T) / 2.0) / (np.var(Lsq) + 1e-12)
        )

    C.phase("p3_verdict")
    rho = headline["all"]["rho"]
    rho_ci = (headline["all"]["boot"]["ci_lo"], headline["all"]["boot"]["ci_hi"])
    pr = kill["partial_rho_S_L_given_cos_js"]
    pr_ci = (
        kill["boot_partial_given_cos_js"]["ci_lo"],
        kill["boot_partial_given_cos_js"]["ci_hi"],
    )
    verdict = verdict_lattice(rho, rho_ci, pr, pr_ci)

    forest_rows = [
        ("raw rho", rho, *rho_ci),
        ("partial | cos", kill["partial_rho_S_L_given_cos"], *rho_ci),
        ("partial | JS", kill["partial_rho_S_L_given_js"], *rho_ci),
        ("partial | cos+JS (KILL)", pr, *pr_ci),
        (
            "ΔCV-R² (LOFO)",
            lofo["delta_cv_r2_target"],
            lofo["delta_cv_r2_target"],
            lofo["delta_cv_r2_target"],
        ),
    ]
    C.phase("p3_figures")
    figs = make_figures(
        fig_dir,
        {
            "S16": S16,
            "S26": S26,
            "L": L,
            "L_margin": L_margin,
            "mask": mask,
            "sources": sources,
            "targets": targets,
            "l_star": l_star,
            "cos532": cos532,
            "pearson_S_cos": kill["pearson_S_cos"],
            "forest_rows": forest_rows,
            "lofo_target_full": lofo["target_full"],
            "per_layer_rho": per_layer_rho,
            "S_mean_target16": S_mean_target16,
        },
    )

    analysis = {
        "verdict": verdict,
        "l_star": l_star,
        "panel": {
            "sources": sources,
            "targets": targets,
            "full_panel": panel_full,
            "n_offdiag_cells": int(mask.sum()),
        },
        "headline": headline,
        "kill": kill,
        "lofo": lofo,
        "null": {
            "n_draws": args.n_null,
            "p975_abs_rho_lstar": null_p975_abs,
            "p_two_sided_lstar": null_p,
            "source_axis_p975_abs": float(np.quantile(np.abs(null_src), 0.975)),
            "per_draw_matrix": str(res_dir / "null_matrices.npz"),
        },
        "ceiling": {"r_SS": r_ss, **r_ll, "attenuation_ceiling": ceiling},
        "band_vs_ceiling": band_vs_ceiling,
        "sensitivity": sensitivity,
        "whitened_gate_lambda_1x": gate["lambda_1x"],
        "mean_prompt_tokens": mean_len,
        "figures": figs,
        "reproducibility_metadata": C.reproducibility_metadata(
            {"smoke": args.smoke, "n_boot": args.n_boot, "n_null": args.n_null}
        ),
    }
    C.write_json_atomic(res_dir / "analysis.json", analysis)
    logger.info(
        "[analysis] verdict=%s rho=%.4f CI=(%.4f, %.4f) partial=%.4f CI=(%.4f, %.4f)",
        verdict,
        rho,
        *rho_ci,
        pr,
        *pr_ci,
    )
    C.phase("done_analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
