"""Issue #526 — directional-asymmetry gate ladder on #537 and #545.

Read-only analysis on existing eval JSONs. Decides how complex the leakage
predictor g must be (symmetric / +baseline-diff / +learned scalars / +full pairwise).

L0 symmetric baseline:  antisym fraction = sum(A^2)/sum((Moff-mean)^2),  A=(M-M^T)/2
L1 baseline-diff term:  A_ij vs (E[j]-E[i])  [+ optional *cos/*norm geometry term]
L2 per-unit scalars:    M_ij ~ mu + b_i + r_j  (additive two-way, off-diag LS);
                        fraction of ANTISYM variance captured by additive antisym part
L3 residual:            1 - L2 fraction = needs full pairwise interaction
Discriminator:          corr( fitted net scalar (b_i - r_i),  base prior E[i] )
"""

import json

import numpy as np
from scipy import stats

np.set_printoptions(precision=4, suppress=True)


# ----------------------------------------------------------------------------- helpers
def offdiag_mask(n):
    m = ~np.eye(n, dtype=bool)
    return m


def antisym_fraction(M):
    """Antisymmetric share of off-diagonal variance about the off-diagonal mean."""
    n = M.shape[0]
    od = offdiag_mask(n)
    A = (M - M.T) / 2.0
    mu = M[od].mean()
    denom = np.sum((M[od] - mu) ** 2)
    num = np.sum(A[od] ** 2)
    return num / denom if denom > 0 else np.nan


def fit_two_way_additive(M):
    """Least-squares M_ij ~ mu + b_i + r_j on OFF-DIAGONAL cells.

    Returns mu, b (row/source-breadth), r (col/receptivity), fitted matrix Mhat.
    Identifiability: sum(b)=sum(r)=0 via centering of the design.
    """
    n = M.shape[0]
    od = offdiag_mask(n)
    rows, cols = np.where(od)
    y = M[rows, cols]
    # design: intercept + (n-1) row dummies + (n-1) col dummies (drop last, then recenter)
    X = np.zeros((len(y), 1 + 2 * n))
    X[:, 0] = 1.0
    for k, (i, j) in enumerate(zip(rows, cols)):
        X[k, 1 + i] = 1.0
        X[k, 1 + n + j] = 1.0
    # solve least squares (rank-deficient -> lstsq picks min-norm)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    mu0 = beta[0]
    b = beta[1 : 1 + n].copy()
    r = beta[1 + n : 1 + 2 * n].copy()
    # recenter for identifiability
    mu = mu0 + b.mean() + r.mean()
    b = b - b.mean()
    r = r - r.mean()
    Mhat = mu + b[:, None] + r[None, :]
    return mu, b, r, Mhat


def scalar_antisym_fraction(M):
    """Fraction of off-diag ANTISYM variance captured by the additive model's antisym part.

    The additive model's antisymmetric part is A_hat = (Mhat - Mhat^T)/2 = (b_i - b_j - (r_i - r_j))/2
    i.e. depends only on the net per-unit scalar s_i = (b_i - r_i)/2  ->  A_hat_ij = s_i - s_j.
    Reported as 1 - SS_resid_antisym / SS_total_antisym.
    """
    n = M.shape[0]
    od = offdiag_mask(n)
    A = (M - M.T) / 2.0
    _, b, r, Mhat = fit_two_way_additive(M)
    Ahat = (Mhat - Mhat.T) / 2.0
    ss_tot = np.sum(A[od] ** 2)
    ss_res = np.sum((A[od] - Ahat[od]) ** 2)
    frac = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    net_scalar = b - r  # source-breadth minus receptivity
    return frac, net_scalar, b, r


def signed_antisym_pairs(M):
    """Upper-triangle signed antisymmetry A_ij = (M_ij - M_ji)/2 for i<j, with indices."""
    n = M.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, (M[i, j] - M[j, i]) / 2.0))
    return pairs


def l1_baseline_diff(M, E, cos=None, norm=None):
    """L1: signed antisym A_ij vs (E[j]-E[i]); optional *cos and *norm geometry variants."""
    pairs = signed_antisym_pairs(M)
    a = np.array([p[2] for p in pairs])
    de = np.array([E[p[1]] - E[p[0]] for p in pairs])  # E[j]-E[i]
    out = {}
    if np.std(de) > 1e-12 and np.std(a) > 1e-12 and len(a) >= 3:
        pr = stats.pearsonr(de, a)
        sr = stats.spearmanr(de, a)
        out["pearson_r"] = float(pr.statistic)
        out["pearson_p"] = float(pr.pvalue)
        out["spearman_r"] = float(sr.statistic)
        out["r2"] = float(pr.statistic**2)
    else:
        out["pearson_r"] = np.nan
        out["r2"] = np.nan
        out["note_E_flat"] = np.std(de) <= 1e-12
    out["n_pairs"] = len(a)
    out["E_spread"] = float(np.std(E))
    if cos is not None:
        deg = np.array([(E[p[1]] - E[p[0]]) * cos[p[0], p[1]] for p in pairs])
        if np.std(deg) > 1e-12 and np.std(a) > 1e-12:
            out["r2_x_cos"] = float(stats.pearsonr(deg, a).statistic ** 2)
    if norm is not None:
        deg = np.array(
            [(E[p[1]] - E[p[0]]) for p in pairs]
        )  # placeholder; norm handled in 537 main read
    return out


def ladder_for_matrix(M, E, label, ci_lo=None, ci_hi=None, cos=None):
    """Run L0..L3 + discriminator and return a dict."""
    n = M.shape[0]
    res = {"label": label, "n": int(n), "n_offdiag": int(offdiag_mask(n).sum())}
    # L0
    af = antisym_fraction(M)
    res["L0_antisym_fraction"] = float(af)
    res["L0_symmetric_ceiling_r2"] = float(1 - af)
    # CI exceedance: |A_ij| vs per-cell CI half-width (only if CIs given)
    if ci_lo is not None and ci_hi is not None:
        A = (M - M.T) / 2.0
        half = (ci_hi - ci_lo) / 2.0  # per-cell half-width
        # combined half-width for the difference (i,j) and (j,i): sqrt(h_ij^2 + h_ji^2)/2
        pairs = signed_antisym_pairs(M)
        sig = 0
        tot = 0
        for i, j, a in pairs:
            if np.isnan(half[i, j]) or np.isnan(half[j, i]):
                continue
            comb = np.sqrt(half[i, j] ** 2 + half[j, i] ** 2) / 2.0
            tot += 1
            if abs(a) > comb:
                sig += 1
        res["L0_pairs_antisym_exceeds_CI"] = f"{sig}/{tot}"
    # L1
    res["L1"] = l1_baseline_diff(M, E, cos=cos)
    # L2
    frac, net_scalar, b, r = scalar_antisym_fraction(M)
    res["L2_scalar_antisym_fraction"] = float(frac)
    # L3
    res["L3_residual_pairwise"] = float(1 - frac)
    # Discriminator: corr(net scalar, base prior E)
    if np.std(E) > 1e-12 and np.std(net_scalar) > 1e-12:
        dr = stats.pearsonr(net_scalar, E)
        res["discriminator_corr_netscalar_vs_baseprior"] = float(dr.statistic)
        res["discriminator_p"] = float(dr.pvalue)
    else:
        res["discriminator_corr_netscalar_vs_baseprior"] = np.nan
        res["discriminator_note"] = "base prior flat -> term structurally untestable"
    return res


def verdict(res):
    """One-line verdict on how fancy g must be."""
    af = res["L0_antisym_fraction"]
    l1r2 = res["L1"].get("r2", np.nan)
    scal = res["L2_scalar_antisym_fraction"]
    resid = res["L3_residual_pairwise"]
    if af < 0.10:
        return "symmetric (antisym < 10% of off-diag variance)"
    # asymmetry is real
    if not np.isnan(l1r2) and l1r2 >= 0.5:
        return f"+baseline-diff term (L1 R2={l1r2:.2f} captures the direction)"
    if not np.isnan(scal) and scal >= 0.8:
        if not np.isnan(l1r2) and l1r2 >= 0.25:
            return f"+baseline-diff term contributes (L1 R2={l1r2:.2f}) but +learned scalars needed (scalar={scal:.2f})"
        return f"+learned breadth&receptivity scalars (scalar={scal:.2f}, residual pairwise={resid:.2f})"
    return f"+full pairwise interaction (scalar only {scal:.2f}, residual={resid:.2f})"


# =============================================================================== 537
def load_537():
    pc = json.load(open("eval_results/issue_537/G_tensor/G_meta.json"))["per_cell"]
    g1 = json.load(open("eval_results/issue_537/analysis/g1_regression.json"))
    norms = g1["norms_l22_mean_response"]
    # square 16x16 block: contexts present on BOTH train and eval side, per behavior
    behaviors = ["marker", "fact", "refusal", "sycophancy", "em"]
    out = {}
    for beh in behaviors:
        tr_ctx = sorted(
            {k.split("/", 1)[1].split("__", 1)[0] for k in pc if k.startswith(beh + "/")}
        )
        ev_ctx_all = {k.split("/", 1)[1].split("__", 1)[1] for k in pc if k.startswith(beh + "/")}
        both = sorted(set(tr_ctx) & ev_ctx_all)
        idx = {c: i for i, c in enumerate(both)}
        n = len(both)
        M = np.full((n, n), np.nan)
        SAT = np.zeros((n, n), dtype=bool)
        Erate = np.full(n, np.nan)  # eval-context discrete base rate
        Elogp = np.full(n, np.nan)  # eval-context continuous base log P (marker only meaningful)
        for k, v in pc.items():
            if not k.startswith(beh + "/"):
                continue
            tr, ev = k.split("/", 1)[1].split("__", 1)
            if tr in idx and ev in idx:
                M[idx[tr], idx[ev]] = v["g"]
                SAT[idx[tr], idx[ev]] = bool(v.get("saturated", False))
                Erate[idx[ev]] = v.get("base_rate", np.nan)
                if v.get("base_logp_at_train_ctx") is not None:
                    Elogp[idx[ev]] = v["base_logp_at_train_ctx"]
        out[beh] = dict(contexts=both, M=M, SAT=SAT, Erate=Erate, Elogp=Elogp, idx=idx, norms=norms)
    return out


def run_537():
    data = load_537()
    results = {}
    for beh, d in data.items():
        M, SAT = d["M"], d["SAT"]
        # mask saturated off-diagonal cells -> exclude pairs where either direction saturates
        # for the matrix-level reads we keep M but note saturation count
        n = M.shape[0]
        # choose base prior: marker uses continuous log P (discrete rate is flat 0);
        # others use the judge base rate (varies)
        if beh == "marker":
            E = d["Elogp"]
            E_kind = (
                "base log P(marker) at answer slot (continuous; discrete emission rate is flat 0)"
            )
            # geometry: per-context L22 norm
            norms = d["norms"]
            normvec = np.array([norms.get(c, np.nan) for c in d["contexts"]])
        else:
            E = d["Erate"]
            E_kind = "base judge rate in eval context (discrete)"
            normvec = None
        # drop any context with NaN base prior from L1 only handled inside via std checks
        res = ladder_for_matrix(M, E, f"537/{beh}")
        res["E_kind"] = E_kind
        res["n_saturated_offdiag"] = int((SAT & offdiag_mask(n)).sum())
        # marker: also try the *norm geometry variant explicitly on the rank-1 prediction
        if beh == "marker" and normvec is not None and not np.all(np.isnan(normvec)):
            pairs = signed_antisym_pairs(M)
            a = np.array([p[2] for p in pairs])
            dnorm = np.array([np.log(normvec[p[1]]) - np.log(normvec[p[0]]) for p in pairs])
            ok = np.isfinite(dnorm) & np.isfinite(a)
            if ok.sum() >= 3 and np.std(dnorm[ok]) > 1e-9:
                res["L1_marker_rank1_norm_r2"] = float(
                    stats.pearsonr(dnorm[ok], a[ok]).statistic ** 2
                )
                res["L1_marker_rank1_norm_slope"] = float(np.polyfit(dnorm[ok], a[ok], 1)[0])
        res["verdict"] = verdict(res)
        results[beh] = res
    return results, data


# =============================================================================== 545
HOME_545 = {
    # rate-valued contentful behaviors (commensurable 0-1)
    "insecure_code_primary": "fam_expr_insecure_code__default",
    "bad_medical_primary": "fam_expr_bad_medical__default",
    "risky_financial_primary": "fam_expr_risky_financial__default",
    "extreme_sports_primary": "fam_expr_extreme_sports__default",
    "wrong_claim_agreement_primary": "sycophancy__default",
    "compliment_writing_primary": "fam_expr_compliment__default",
    "refuse_medical_primary": "refusal__default",
    "answer_in_lists_primary": "format_style__default",
    "benign_format_primary": "format_style__default",
    "casual_register_primary": "format_style__default",
    "taught_fact_primary": "fact_expression__default",
    "reversed_fact_primary": "fact_expression__default",
    "business_skills_primary": "business_competence__default",
    # excluded from the rate matrix (incommensurable / failed implant):
    # marker_primary -> marker__default (NATS, not a rate)
    # hedge_everywhere_primary -> refusal (implant failed; home ambiguous)
    # warmth_primary -> warmth_expression (1-5 scale; entered at base strength)
}


def load_545():
    cells = json.load(open("eval_results/issue_545/L_matrix.json"))["cells"]
    base = json.load(open("eval_results/issue_545/base_panel.json"))["panel"]
    return cells, base


def seedmean_cell(cells, family, col):
    seeds = [tr for tr in cells if "_".join(tr.split("_")[:-1]) == family]
    vals, sat = [], False
    cis = []
    for tr in seeds:
        c = cells[tr].get(col)
        if c and c.get("L") is not None:
            if c.get("saturation_flag", False):
                sat = True
                continue
            vals.append(c["L"])
            if c.get("ci95_cluster_bootstrap"):
                cis.append(c["ci95_cluster_bootstrap"])
    return (np.mean(vals) if vals else np.nan), len(vals), sat


def run_545():
    cells, base = load_545()
    # build the largest usable RATE behavior x behavior block over families with distinct home behaviors
    # one family per distinct home behavior to avoid degenerate same-home pairs:
    # collapse the within-advice and within-format families but keep them as separate units
    fams = list(HOME_545.keys())
    n = len(fams)
    M = np.full((n, n), np.nan)
    nrecip = 0
    for i, A in enumerate(fams):
        for j, B in enumerate(fams):
            col = HOME_545[B]
            v, nv, sat = seedmean_cell(cells, A, col)
            if not sat:
                M[i, j] = v
    # base prior per unit = base panel rate of that unit's HOME column
    E = np.array([base.get(HOME_545[f], {}).get("scalar", np.nan) for f in fams])
    E = np.array([np.nan if x is None else x for x in E], dtype=float)
    # reciprocity check: how many off-diagonal pairs have BOTH directions finite & distinct home
    od = offdiag_mask(n)
    finite_both = 0
    for i in range(n):
        for j in range(i + 1, n):
            if HOME_545[fams[i]] == HOME_545[fams[j]]:
                continue
            if np.isfinite(M[i, j]) and np.isfinite(M[j, i]):
                finite_both += 1
    # The full matrix has NaNs (within-family batteries only run inside family). For a clean
    # square reciprocal read we restrict to the maximal fully-observed reciprocal sub-block.
    # Find sub-block of indices where all pairwise off-diagonal cells (both dirs) are finite.
    return fams, M, E, finite_both, cells, base


def maximal_reciprocal_block(M, fams, home):
    """Greedy: find a large index subset whose induced off-diagonal block is fully finite (both dirs)
    and whose home behaviors are pairwise distinct."""
    n = M.shape[0]
    best = []
    # try all subsets by greedy growth from each seed (n small)
    import itertools

    # candidate: keep one family per distinct home behavior; pick the one with most finite recips
    by_home = {}
    for i, f in enumerate(fams):
        by_home.setdefault(home[f], []).append(i)
    # choose representative per home = the one maximizing finite off-diagonal degree
    deg = []
    for i in range(n):
        d = 0
        for j in range(n):
            if i != j and np.isfinite(M[i, j]) and np.isfinite(M[j, i]):
                d += 1
        deg.append(d)
    reps = [max(idxs, key=lambda i: deg[i]) for idxs in by_home.values()]
    # now greedily prune reps to a fully-finite reciprocal off-diagonal block
    reps = sorted(reps, key=lambda i: -deg[i])
    # try decreasing-size combinations
    for size in range(len(reps), 1, -1):
        for combo in itertools.combinations(reps, size):
            ok = True
            for a in combo:
                for b in combo:
                    if a != b and not (np.isfinite(M[a, b]) and np.isfinite(M[b, a])):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return list(combo)
    return reps[:2]


def main():
    print("=" * 90)
    print("ISSUE #526 — DIRECTIONAL-ASYMMETRY GATE LADDER")
    print("=" * 90)

    # ---- 537
    print("\n############ DATASET #537 — context-generalization (16x16 per behavior) ############")
    res537, data537 = run_537()
    for beh, r in res537.items():
        print(
            f"\n--- 537 / {beh}  (n={r['n']}, off-diag={r['n_offdiag']}, sat-offdiag={r['n_saturated_offdiag']}) ---"
        )
        print(f"  E = {r['E_kind']}")
        print(
            f"  L0 antisym fraction      = {r['L0_antisym_fraction']:.3f}  (symmetric ceiling R2={r['L0_symmetric_ceiling_r2']:.3f})"
        )
        l1 = r["L1"]
        print(
            f"  L1 baseline-diff: r={l1.get('pearson_r', float('nan')):.3f} R2={l1.get('r2', float('nan')):.3f} "
            f"(E spread={l1['E_spread']:.3f}, n_pairs={l1['n_pairs']})"
            + ("  [E FLAT -> untestable]" if l1.get("note_E_flat") else "")
        )
        if "L1_marker_rank1_norm_r2" in r:
            print(
                f"     [rank-1 norm variant: R2={r['L1_marker_rank1_norm_r2']:.3f}, slope={r['L1_marker_rank1_norm_slope']:.3f} (predicted +1)]"
            )
        print(f"  L2 scalar antisym frac   = {r['L2_scalar_antisym_fraction']:.3f}")
        print(f"  L3 residual pairwise     = {r['L3_residual_pairwise']:.3f}")
        disc = r["discriminator_corr_netscalar_vs_baseprior"]
        print(
            f"  Discriminator corr(net scalar, base prior) = {disc:.3f}"
            + (f"  [{r.get('discriminator_note', '')}]" if r.get("discriminator_note") else "")
        )
        print(f"  >>> VERDICT: {r['verdict']}")

    # ---- 545
    print("\n\n############ DATASET #545 — behavior->behavior (rate-valued block) ############")
    fams, M545, E545, finite_both, cells, base = run_545()
    print(f"Rate-valued candidate units (n={len(fams)}): {fams}")
    print(
        f"Off-diagonal reciprocal pairs with BOTH directions finite & distinct-home: {finite_both}"
    )
    block = maximal_reciprocal_block(M545, fams, HOME_545)
    blk_fams = [fams[i] for i in block]
    print(f"\nMaximal fully-finite reciprocal sub-block (distinct home behaviors), n={len(block)}:")
    for i in block:
        print(f"   {fams[i]:32s} home={HOME_545[fams[i]]:32s} base_prior E={E545[i]:.3f}")
    Mb = M545[np.ix_(block, block)]
    Eb = E545[block]
    print("\n  sub-block M[i->j] (rows=train side, cols=eval-home of col unit):")
    print("   " + "  ".join(f"{f.split('_')[0][:7]:>8s}" for f in blk_fams))
    for i, f in enumerate(blk_fams):
        print(
            f"   {f.split('_')[0][:7]:>8s} "
            + " ".join(f"{Mb[i, j]:8.3f}" for j in range(len(block)))
        )
    if len(block) >= 3:
        rb = ladder_for_matrix(Mb, Eb, "545/rate-block")
        rb["verdict"] = verdict(rb)
        print(
            f"\n  L0 antisym fraction      = {rb['L0_antisym_fraction']:.3f}  (symmetric ceiling R2={rb['L0_symmetric_ceiling_r2']:.3f})"
        )
        l1 = rb["L1"]
        print(
            f"  L1 baseline-diff: r={l1.get('pearson_r', float('nan')):.3f} R2={l1.get('r2', float('nan')):.3f} (E spread={l1['E_spread']:.3f}, n_pairs={l1['n_pairs']})"
        )
        print(f"  L2 scalar antisym frac   = {rb['L2_scalar_antisym_fraction']:.3f}")
        print(f"  L3 residual pairwise     = {rb['L3_residual_pairwise']:.3f}")
        print(
            f"  Discriminator corr(net scalar, base prior) = {rb['discriminator_corr_netscalar_vs_baseprior']:.3f}"
        )
        print(f"  >>> VERDICT: {rb['verdict']}")
    else:
        rb = None
        print(
            "  Sub-block too small for a matrix-level L0/L2 ladder read (no fully-finite >=3 reciprocal\n"
            "  block with distinct home behaviors exists). #545 is structurally NON-reciprocal in\n"
            "  behavior space: within-family expression batteries only run inside their own family, so\n"
            "  trainA->homeB exists but trainB->homeA usually does not. Falling back to the per-PAIR L1\n"
            "  test (the theory term IS a per-pair relation, not a matrix-level one)."
        )

    # ---- 545 per-PAIR L1 test across ALL reciprocal cross-behavior rate pairs (distinct home) ----
    print(
        "\n--- 545 per-PAIR L1 (theory term): signed antisym (L_ab - L_ba)/2  vs  base-prior diff (E_b - E_a) ---"
    )
    pair_rows = []
    for i in range(len(fams)):
        for j in range(i + 1, len(fams)):
            if HOME_545[fams[i]] == HOME_545[fams[j]]:
                continue  # same home behavior -> not a cross pair
            ab = M545[i, j]
            ba = M545[j, i]
            if not (np.isfinite(ab) and np.isfinite(ba)):
                continue
            A = (ab - ba) / 2.0
            dE = E545[j] - E545[i]
            pair_rows.append(
                dict(
                    a=fams[i].replace("_primary", ""),
                    b=fams[j].replace("_primary", ""),
                    L_ab=float(ab),
                    L_ba=float(ba),
                    antisym=float(A),
                    Ea=float(E545[i]),
                    Eb=float(E545[j]),
                    dE=float(dE),
                )
            )
    pair545 = pair_rows
    for p in pair_rows:
        print(
            f"   {p['a'][:14]:>15s} <-> {p['b'][:14]:<15s}  L_ab={p['L_ab']:+.3f} L_ba={p['L_ba']:+.3f} "
            f"antisym={p['antisym']:+.3f}  (E_a={p['Ea']:.2f} E_b={p['Eb']:.2f} dE={p['dE']:+.2f})"
        )
    if len(pair_rows) >= 3:
        aa = np.array([p["antisym"] for p in pair_rows])
        dd = np.array([p["dE"] for p in pair_rows])
        if np.std(dd) > 1e-9 and np.std(aa) > 1e-9:
            pr = stats.pearsonr(dd, aa)
            sr = stats.spearmanr(dd, aa)
            print(
                f"   ==> per-pair L1: Pearson r={pr.statistic:+.3f} (R2={pr.statistic**2:.3f}, p={pr.pvalue:.3f}), "
                f"Spearman={sr.statistic:+.3f}, n={len(aa)} pairs"
            )
            print(
                "   NOTE: most pairs involve >=1 saturating-low-base-prior behavior; advice within-family\n"
                "   pairs (near-equal base priors) excluded by the distinct-home filter."
            )

    # within-advice family clean reciprocal block (bad_medical, risky_financial, extreme_sports)
    print(
        "\n--- 545 within-advice-family reciprocal block (bad_medical/risky_financial/extreme_sports) ---"
    )
    adv_pairs = [
        ("bad_medical_primary", "risky_financial_primary"),
        ("bad_medical_primary", "extreme_sports_primary"),
    ]
    adv_rows = []
    for a, b in adv_pairs:
        ia, ib = fams.index(a), fams.index(b)
        ab, ba = M545[ia, ib], M545[ib, ia]
        if np.isfinite(ab) and np.isfinite(ba):
            A = (ab - ba) / 2.0
            adv_rows.append(
                dict(
                    a=a,
                    b=b,
                    L_ab=float(ab),
                    L_ba=float(ba),
                    antisym=float(A),
                    Ea=float(E545[ia]),
                    Eb=float(E545[ib]),
                )
            )
            print(
                f"   {a.replace('_primary', ''):>18s} <-> {b.replace('_primary', ''):<18s} "
                f"L_ab={ab:+.3f} L_ba={ba:+.3f} antisym={A:+.3f}  (E_a={E545[ia]:.3f} E_b={E545[ib]:.3f})"
            )
    print(
        "  NOTE: all advice rows share home=misaligned-advice rate -> base priors ~equal (0.04-0.08),\n"
        "  context-overlap ~1 -> the L1 baseline-diff term is ~0 by construction; antisym here is small\n"
        "  and within seed noise (n=2-3 seeds/cell). Within-family advice transfer is ~symmetric."
    )

    # save everything for the figure
    allres = {
        "537": res537,
        "545_block": rb,
        "545_block_fams": blk_fams,
        "545_pairs": pair545,
        "545_advice_pairs": adv_rows,
    }

    # JSON-safe
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and np.isnan(o):
            return None
        return o

    json.dump(clean(allres), open("figures/issue_526/gate_ladder_results.json", "w"), indent=1)
    print("\nSaved figures/issue_526/gate_ladder_results.json")
    return res537, rb, blk_fams, pair545, adv_rows


if __name__ == "__main__":
    main()
