"""Issue #1895 harvest-side analysis + figures (VM, CPU-only).

Inputs: committed eval_results/issue_1895/*.json + perdirection_profiles.npz,
plus HF-staged run tensors (percontext_proj.npz, pilot/pilot_rows.npz,
q_basis.npz, fve_profiles.npz) from issue1895_subspaces/analysis_tensors/.

Outputs:
  eval_results/issue_1895/dark_spot_matched_rows.json  (matched-rows Path-A vs
      pure-e_bar re-reduction on the 512 pilot rows — target-purity vs
      row-subset disambiguation for the P_dark read)
  eval_results/issue_1895/angle_spectrum_k64.json      (per-angle cos^2 spectra
      at k=64, rebuilt + verified against angles_summary.json observed O)
  figures/issue_1895/*.png/pdf/meta.json               (paper-plots conventions)

Run: uv run python scripts/issue1895_analysis_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + credentials BEFORE matplotlib/numpy (shared-VM harvest; #847)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

EVAL = Path("eval_results/issue_1895")
FIGDIR = Path("figures/issue_1895")
STAGE = Path("data/issue_1895/hf_dl")
HF_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1895_subspaces/analysis_tensors"
SEED = 1895  # run seed; matched-rows bootstrap uses SEED + 7 (driver convention)


def _stage(rel: str) -> Path:
    """Download one run tensor from the HF data repo if not already staged (retried, atomic)."""
    out = STAGE / HF_PREFIX / rel
    if not out.exists():
        from explore_persona_space.orchestrate.hub import stage_hub_file

        stage_hub_file(HF_REPO, f"{HF_PREFIX}/{rel}", out, repo_type="dataset")
    assert out.exists(), f"staging failed for {rel}"
    return out


def _pooled_r2(yhat: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def matched_rows_dark_spot() -> dict:
    """Score the fitted T3 map against Path-A e_bar AND pure e_bar on the SAME
    512 pilot rows (paired draws), isolating the target-purity effect the
    committed dark_spot_pilot_pure_ebar read leaves confounded with row subset.
    """
    with np.load(_stage("pilot/pilot_rows.npz"), allow_pickle=False) as z:
        pos_p = z["pos"].astype(np.int64)
        vmask = z["vmask"].astype(np.float32)
        r_exact = z["r_exact"].astype(np.float32)
    with np.load(_stage("percontext_proj.npz"), allow_pickle=False) as z:
        te_pos = z["te_pos"].astype(np.int64)
        E = z["E_proj"].astype(np.float32)
        R3 = z["R3_proj"].astype(np.float32)
        R1 = z["R1_proj"].astype(np.float32)
        V = z["V_proj"].astype(np.float32)
    with np.load(_stage("q_basis.npz"), allow_pickle=False) as z:
        Q = z["Q"].astype(np.float32)

    loc = {int(p): i for i, p in enumerate(te_pos)}
    rows = np.array([loc[int(p)] for p in pos_p if int(p) in loc], dtype=np.int64)
    keepm = np.array([int(p) in loc for p in pos_p], dtype=bool)
    assert rows.shape == (512,), f"expected 512 pilot rows in holdout, got {len(rows)}"

    ehat = E[rows] - R3[rows]  # fitted T3 predictions (Q coords)
    e_path_a = E[rows]  # Path-A e_bar targets on the same rows
    e_pure = (vmask[keepm] - r_exact[keepm]) @ Q  # pure SAE error

    g0 = 1.0 - (R1**2).sum(0) / np.maximum(((V - V.mean(0)) ** 2).sum(0), 1e-9)

    def plug(y: np.ndarray) -> float:
        en = ((y - y.mean(0)) ** 2).sum(0)
        return float((en * g0).sum() / en.sum())

    # reproduction anchor: the committed pure-read values must match the driver.
    r2_pure = _pooled_r2(ehat, e_pure)
    committed = json.loads((EVAL / "angles_summary.json").read_text())["dark_spot_pilot_pure_ebar"]
    assert abs(r2_pure - committed["r2_pure_ebar"]) < 1e-4, (
        f"pure-read reproduction failed: {r2_pure} vs {committed['r2_pure_ebar']}"
    )

    mis = e_path_a - e_pure
    rng = np.random.default_rng(SEED + 7)
    n = len(rows)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=10_000).astype(np.float32)

    def boot_dd(y: np.ndarray) -> np.ndarray:
        res_sq = ((y - ehat) ** 2).astype(np.float32)
        tgt_sq = (y**2).astype(np.float32)
        mean_t = (W @ y) / n
        sstot = W @ tgt_sq - n * mean_t**2
        r2_d = 1.0 - (W @ res_sq).sum(1) / np.maximum(sstot.sum(1), 1e-9)
        plug_d = (sstot * g0[None, :]).sum(1) / np.maximum(sstot.sum(1), 1e-9)
        return r2_d - plug_d

    dd_a, dd_p = boot_dd(e_path_a), boot_dd(e_pure)

    def ci(x: np.ndarray) -> list[float]:
        return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]

    out = {
        "n_pilot_rows": int(n),
        "note": (
            "Same fitted T3 map (trained on Path-A e_bar) scored against both "
            "targets on identical rows; paired multinomial bootstrap (10k draws, "
            "seed 1902 = run seed + 7) holds the fitted maps fixed (scoring "
            "uncertainty only). g(u) profile from the full 20k holdout."
        ),
        "path_a": {
            "r2": _pooled_r2(ehat, e_path_a),
            "plugin": plug(e_path_a),
            "delta_dark": _pooled_r2(ehat, e_path_a) - plug(e_path_a),
            "delta_dark_ci": ci(dd_a),
        },
        "pure": {
            "r2": r2_pure,
            "plugin": plug(e_pure),
            "delta_dark": r2_pure - plug(e_pure),
            "delta_dark_ci": ci(dd_p),
        },
        "purity_effect_paired": {
            "point": float(
                (_pooled_r2(ehat, e_path_a) - plug(e_path_a)) - (r2_pure - plug(e_pure))
            ),
            "ci": ci(dd_a - dd_p),
        },
        "mismatch_component": {
            "energy_share_of_path_a_ebar_centered": float(
                ((mis - mis.mean(0)) ** 2).sum() / ((e_path_a - e_path_a.mean(0)) ** 2).sum()
            ),
            "corr_pred_vs_mismatch_centered": float(
                np.corrcoef((ehat - ehat.mean(0)).ravel(), (mis - mis.mean(0)).ravel())[0, 1]
            ),
        },
    }
    (EVAL / "dark_spot_matched_rows.json").write_text(json.dumps(out, indent=2))
    return out


def angle_spectrum_k64() -> dict:
    """Rebuild P_pred(64) (banked-ridge profile) and the recon-PCA / resid-PCA
    subspaces on the holdout; verify mean cos^2 against angles_summary.json."""
    with np.load(_stage("percontext_proj.npz"), allow_pickle=False) as z:
        E = z["E_proj"].astype(np.float32)
        V = z["V_proj"].astype(np.float32)
    banked = np.asarray(
        json.loads(Path("eval_results/issue_1482/perdirection_pca.json").read_text())[
            "per_direction_r2"
        ]["ridge"],
        dtype=np.float64,
    )
    sel = np.argsort(-banked)[:64]
    ang = json.loads((EVAL / "angles_summary.json").read_text())
    obs = {
        (c["pair"], c["k"]): c["observed_O"]
        for c in ang["cells"]
        if c["pred_profile"] == "banked_ridge"
    }
    out: dict = {"k": 64, "pred_profile": "banked_ridge", "pred_eigranks": sel.tolist()}
    for name, mat in [("psae_recon_pca", V - E), ("presid_pca", E)]:
        mc = mat - mat.mean(0)
        _, _, vt = np.linalg.svd(mc, full_matrices=False)
        sv = np.linalg.svd(vt[:64].T[sel, :], compute_uv=False)
        cos2 = np.sort(sv**2)[::-1]
        assert abs(float(cos2.mean()) - obs[(name, 64)]) < 2e-3, (
            f"{name}: rebuilt mean {cos2.mean():.6f} vs committed {obs[(name, 64)]:.6f}"
        )
        out[name] = {"cos2_sorted": cos2.tolist(), "mean": float(cos2.mean())}
    (EVAL / "angle_spectrum_k64.json").write_text(json.dumps(out, indent=2))
    return out


# ── figures ──────────────────────────────────────────────────────────────────

PAIR_TITLES = {
    "psae_recon_pca": "reconstruction-PCA subspace (primary)",
    "presid_pca": "SAE-residual subspace (complement)",
    "psae_dec_svd": "weighted-decoder-SVD subspace (twin)",
}


def fig_hero_overlap(ang: dict, pal: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    ks = [16, 32, 64, 128, 256]
    shell_colors = {16: pal[3], 32: pal[0], 64: pal[4]}
    for ax, pair in zip(axes, ["psae_recon_pca", "presid_pca", "psae_dec_svd"]):
        prim = {c["k"]: c for c in ang["cells"] if c["pair"] == pair and c["primary_profile_for_k"]}
        xs = np.arange(len(ks))
        for sh in [16, 32, 64]:
            lo = [prim[k]["nulls"][str(sh)]["p2.5"] for k in ks]
            hi = [prim[k]["nulls"][str(sh)]["p97.5"] for k in ks]
            ax.fill_between(
                xs,
                lo,
                hi,
                color=shell_colors[sh],
                alpha=0.30,
                lw=0,
                label=f"null band ({sh} shells)",
            )
        ys = [prim[k]["observed_O"] for k in ks]
        ax.plot(xs, ys, color="black", lw=1.5, zorder=5)
        for i, k in enumerate(ks):
            marker = "o" if prim[k]["pred_profile"] == "banked_ridge" else "s"
            ax.scatter([xs[i]], [ys[i]], color="black", marker=marker, s=45, zorder=6)
            ax.text(
                xs[i],
                ys[i] + 0.004,
                f"{ys[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        ax.set_xticks(xs, [str(k) for k in ks])
        ax.set_xlabel("subspace dimension k")
        ax.set_title(PAIR_TITLES[pair], fontsize=11)
    axes[0].set_ylabel("subspace overlap O(k) = mean cos$^2$ principal angle")
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles += [
        Line2D([], [], color="black", marker="o", lw=1.5, label="observed (banked-ridge profile)"),
        Line2D([], [], color="black", marker="s", lw=0, label="observed (matched-refit profile)"),
    ]
    labels += ["observed (banked-ridge profile)", "observed (matched-refit profile)"]
    axes[2].legend(handles, labels, fontsize=8, loc="lower right")
    savefig_paper(fig, "hero_overlap_ksweep", dir=FIGDIR)
    plt.close(fig)


def fig_angle_spectrum(spec: dict, ang: dict, pal: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(1, 65)
    for name, color in [("psae_recon_pca", pal[1]), ("presid_pca", pal[2])]:
        cos2 = np.asarray(spec[name]["cos2_sorted"])
        ax.plot(
            x,
            cos2,
            color=color,
            lw=1.6,
            label=f"{PAIR_TITLES[name].split(' (')[0]} — mean {spec[name]['mean']:.3f}",
        )
        ax.scatter(x[::9], cos2[::9], color=color, s=18)
        for i in [0, 31, 63]:
            ax.text(x[i], cos2[i] + 0.02, f"{cos2[i]:.2f}", ha="center", fontsize=7.5)
    prim64 = next(
        c
        for c in ang["cells"]
        if c["pair"] == "psae_recon_pca" and c["k"] == 64 and c["primary_profile_for_k"]
    )
    band = prim64["nulls"]["32"]
    ax.axhspan(
        band["p2.5"],
        band["p97.5"],
        color=pal[0],
        alpha=0.35,
        label="null band for the MEAN, recon-PCA (32 shells)",
    )
    ax.set_xlabel("principal-angle index (sorted, 1 = best aligned)")
    ax.set_ylabel("cos$^2$ principal angle")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower left")
    savefig_paper(fig, "angle_spectrum_k64", dir=FIGDIR)
    plt.close(fig)


def fig_three_target(fits: dict, ang: dict, spot: dict, pal: list[str]) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    cells = fits["cells"]
    names = ["t_vA_ctx", "t_rbar_ctx", "t_ebar_ctx"]
    labels = ["raw answer\nstate $v_A$", "SAE reconstruction\n$\\bar r$", "SAE residual\n$\\bar e$"]
    colors = [pal[0], pal[1], pal[2]]
    boot = ang["h3_plugin_bootstrap"]
    ci_map = {"t_vA_ctx": "r2_vA", "t_rbar_ctx": "r2_rbar", "t_ebar_ctx": "r2_ebar"}
    for i, nm in enumerate(names):
        r2 = cells[nm]["pooled_r2"]
        lo, hi = boot["all_ci"][ci_map[nm]]
        a.bar(i, r2, color=colors[i], width=0.62)
        a.errorbar(i, r2, yerr=[[r2 - lo], [hi - r2]], color="black", capsize=3, lw=1)
        a.text(i, r2 + 0.015, f"{r2:.3f}", ha="center", fontsize=9)
    for i, (nm, key) in enumerate(
        [("t_rbar_ctx", "plugin_r"), ("t_ebar_ctx", "plugin_e")], start=1
    ):
        pv = boot["observed"][key]
        a.hlines(
            pv,
            i - 0.31,
            i + 0.31,
            color="black",
            ls="--",
            lw=1.4,
            label="variance-profile plug-in" if i == 1 else None,
        )
        a.text(i + 0.34, pv, f"{pv:.3f}", va="center", fontsize=8)
    for i, nm in enumerate(["t_rbar_ctx_mlp", "t_ebar_ctx_mlp"], start=1):
        a.scatter(
            [i + 0.2],
            [cells[nm]["pooled_r2"]],
            marker="D",
            color="black",
            s=30,
            zorder=6,
            label="MLP twin" if i == 1 else None,
        )
    a.set_xticks(range(3), labels)
    a.set_ylabel("held-out pooled R$^2$ (context arm, 20k rows)")
    a.legend(fontsize=8, loc="upper right")

    reads = [
        (
            "full holdout\nPath-A $\\bar e$\n(n=20,000)",
            boot["observed"]["delta_dark"],
            boot["all_ci"]["delta_dark"],
        ),
        (
            "pilot rows\nPath-A $\\bar e$\n(n=512)",
            spot["path_a"]["delta_dark"],
            spot["path_a"]["delta_dark_ci"],
        ),
        (
            "pilot rows\npure $\\bar e$\n(n=512)",
            spot["pure"]["delta_dark"],
            spot["pure"]["delta_dark_ci"],
        ),
    ]
    for i, (lab, pt, ci) in enumerate(reads):
        b.errorbar(i, pt, yerr=[[pt - ci[0]], [ci[1] - pt]], fmt="o", color=pal[2], capsize=4, ms=7)
        b.text(i + 0.10, pt, f"{pt:+.3f}", va="center", fontsize=9)
    b.axhline(0.0, color="black", lw=1)
    b.set_xticks(range(3), [r[0] for r in reads])
    b.set_xlim(-0.5, 2.5)
    b.set_ylabel(r"$\Delta_{dark}$ = R$^2(\bar e)$ $-$ plug-in")
    savefig_paper(fig, "three_target_fits_delta_dark", dir=FIGDIR)
    plt.close(fig)


def fig_perdirection_scatter(pal: list[str]) -> None:
    prof = np.load(EVAL / "perdirection_profiles.npz")
    with np.load(_stage("fve_profiles.npz"), allow_pickle=False) as z:
        fve_u = z["fve_u"].astype(np.float64)
    r2u = prof["r2u__t_vA_ctx"].astype(np.float64)
    rank = np.arange(1, len(r2u) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sc = ax.scatter(fve_u, r2u, c=np.log10(rank), cmap="viridis", s=6, alpha=0.55)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log$_{10}$ eigenvalue rank (1 = largest-variance direction)")
    for r in [1, 16, 64, 256, 1024, 3584]:
        ax.text(fve_u[r - 1], r2u[r - 1], f" rank {r}", fontsize=7.5, va="center")
    ax.set_xlabel("SAE per-direction reconstruction quality FVE$_u$ (20k holdout)")
    ax.set_ylabel("map per-direction held-out R$^2$ (matched 120k refit, $v_A$ target)")
    savefig_paper(fig, "perdirection_r2_vs_fve_scatter", dir=FIGDIR)
    plt.close(fig)


def fig_stratified(corr: dict, pal: list[str]) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    dec = corr["per_direction"]["decile_stratified"]
    a.bar([d["decile"] for d in dec], [d["rho"] for d in dec], color=pal[0], width=0.7)
    for d in dec:
        a.text(d["decile"], d["rho"] + 0.012, f"{d['rho']:.2f}", ha="center", fontsize=8)
    a.axhline(
        corr["per_direction"]["partial_given_var_rank"],
        color="black",
        ls="--",
        lw=1.2,
        label=f"partial rho | variance rank = {corr['per_direction']['partial_given_var_rank']:.3f}",
    )
    a.set_xlabel("eigenvalue decile (0 = highest variance)")
    a.set_ylabel(r"within-decile Spearman $\rho$(map R$^2$, FVE$_u$)")
    a.set_xticks(range(10))
    a.legend(fontsize=8)
    a.set_title("per-direction (n=3,584)", fontsize=11)

    dec2 = corr["per_feature"]["activity_decile_stratified"]
    b.bar([d["decile"] for d in dec2], [d["rho"] for d in dec2], color=pal[1], width=0.7)
    for d in dec2:
        b.text(d["decile"], d["rho"] + 0.008, f"{d['rho']:.2f}", ha="center", fontsize=8)
    b.axhline(
        corr["per_feature"]["partial_r2_fvej_given_varrank_consistency_activity"],
        color="black",
        ls="--",
        lw=1.2,
        label=(
            "partial rho | variance rank, consistency, activity = "
            f"{corr['per_feature']['partial_r2_fvej_given_varrank_consistency_activity']:.3f}"
        ),
    )
    b.set_xlabel("activity decile (0 = least active)")
    b.set_ylabel(r"within-decile Spearman $\rho$(map R$^2$, FVE$_j$)")
    b.set_xticks(range(10))
    b.legend(fontsize=8, loc="center right", frameon=True, framealpha=1.0)
    b.set_title("per-feature (n=16,384)", fontsize=11)
    savefig_paper(fig, "stratified_correlates", dir=FIGDIR)
    plt.close(fig)


def fig_profiles(pal: list[str]) -> None:
    prof = np.load(EVAL / "perdirection_profiles.npz")
    with np.load(_stage("fve_profiles.npz"), allow_pickle=False) as z:
        fve_u = z["fve_u"].astype(np.float64)
    rank = np.arange(1, 3585)

    def roll_med(a: np.ndarray, w: int = 65) -> np.ndarray:
        pad = w // 2
        ap = np.pad(a, (pad, pad), mode="edge")
        return np.array([np.median(ap[i : i + w]) for i in range(len(a))])

    series = [
        ("map R$^2$ per direction ($v_A$ target)", prof["r2u__t_vA_ctx"].astype(float), pal[0]),
        (
            "map R$^2$ per direction ($\\bar e$ target)",
            prof["r2u__t_ebar_ctx"].astype(float),
            pal[2],
        ),
        ("SAE FVE$_u$ per direction", fve_u, pal[1]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for lab, y, c in series:
        ax.scatter(rank, y, s=2, alpha=0.12, color=c)
        ax.plot(rank, roll_med(y), color=c, lw=1.8, label=f"{lab} (rolling median)")
    ax.set_xscale("log")
    ax.set_xlabel("eigenvalue rank (1 = largest-variance direction, log scale)")
    ax.set_ylabel("per-direction value on the 20k holdout")
    ax.set_ylim(-0.15, 1.02)
    ax.legend(fontsize=8, loc="upper right")
    savefig_paper(fig, "profiles_vs_eigrank", dir=FIGDIR)
    plt.close(fig)


def main() -> None:
    load_dotenv()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    set_paper_style("blog")
    pal = paper_palette_blog(6)

    ang = json.loads((EVAL / "angles_summary.json").read_text())
    fits = json.loads((EVAL / "fits_summary.json").read_text())
    corr = json.loads((EVAL / "correlates_summary.json").read_text())

    spot = matched_rows_dark_spot()
    print("matched-rows dark spot:", json.dumps(spot["purity_effect_paired"]))
    spec = angle_spectrum_k64()
    print("spectrum means verified:", spec["psae_recon_pca"]["mean"], spec["presid_pca"]["mean"])

    fig_hero_overlap(ang, pal)
    fig_angle_spectrum(spec, ang, pal)
    fig_three_target(fits, ang, spot, pal)
    fig_perdirection_scatter(pal)
    fig_stratified(corr, pal)
    fig_profiles(pal)
    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
