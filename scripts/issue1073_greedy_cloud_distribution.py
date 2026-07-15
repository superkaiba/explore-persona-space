#!/usr/bin/env python3
"""Issue #1073 free-analysis: distribution of the stochastic-rollout cloud vs the greedy rollout.

For each context x and read-out layer, the pipeline has 10 stochastic answer-span mean-activation
vectors ``v_j(x)`` and one greedy vector ``v_greedy(x)``. This script characterises the cloud of
the 10 stochastic draws and locates the greedy vector relative to it, answering: is greedy
distributionally just an 11th draw, or systematically offset?

Analyses (all batched closed-form tensor reductions — 0 GPU-h, CPU):
  1. Exchangeability / rank test (headline). Treat greedy as an 11th rollout; per context compute
     each of the 11 items' cosine distance to the leave-self-out mean of the other 10, and rank
     greedy among the 11. Under "greedy is just another draw" the rank is uniform on 1..11.
     Report the per-layer rank histogram + chi-square vs uniform, the skew direction, and the
     most-central / most-peripheral fractions (uniform expectation 1/11 each).
  2. Systematic offset. u(x) = v_greedy(x) - mean_j v_j(x). (a) ||mean_x u(x)|| vs a sign-flip
     permutation null; (b) mean cos(u(x), u_bar) (offset-direction consistency); (c) alignment of
     u with the greedy-minus-mean-stoch response-length gap.
  3. Norms. Paired ||v_greedy|| vs the 10 ||v_j|| per layer (percentiles + paired median diff with
     bootstrap CI).
  4. Cloud descriptives. Per-context dispersion percentiles + greedy's distance-to-cloud-mean
     distribution overlaid on a typical draw's leave-one-out distance-to-mean.

Reuses the #1073 loaders (issue1073_common / issue1073_capture) and the gap_tail dispersion
computation (issue1073_gap_tail_analysis.rollout_dispersion) verbatim. The per-rollout store is
staged prefix-scoped onto /mnt/eps-data at the pinned revision (NEVER /, which is full).

NO raw prompt/completion text is loaded into context or written to any output — only structural
span-length counts (from the store) and activation reductions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import issue1073_capture as CAP  # noqa: E402
import issue1073_common as I  # noqa: E402
import issue1073_gap_tail_analysis as GT  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats as sstats  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger("issue1073_greedy_cloud")

torch.set_num_threads(int(os.environ.get("EPS_VM_THREAD_CAP", "8")))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
READOUT_LAYERS = [14, 17, 19, 26, 27]
N_ROLLOUTS = 10
EPS = 1e-12

# Per-rollout store lives at THIS revision (distinct from issue1073_common.PINNED_REVISION,
# which pins the reused #779 inputs).
STORE_REV = "fb4fe90fdd836ba2efd896b90c17e6b42f143d21"
DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1073_decode_regime/analysis_tensors"
# Stage OFF the full boot disk (/ is 100%); the user-owned dir on the /mnt/eps-data data disk has
# headroom (the /mnt/eps-data root itself is root-owned — only the per-user subtree is writable).
_DATA_DISK = Path(os.environ.get("EPS_VM_DATA_DISK_PATH", "/mnt/eps-data"))
_STAGE_BASE = (
    _DATA_DISK / "thomasjiralerspong"
    if (_DATA_DISK / "thomasjiralerspong").is_dir()
    else _DATA_DISK
)
STAGE = Path(os.environ.get("EPS_I1073_STAGE", str(_STAGE_BASE / "issue1073_greedy_cloud")))
STAGE_HF = STAGE / "_hf"

EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_1073"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_1073"

RNG_SEED = 0
N_SIGNFLIP = 2000
N_BOOT = 2000


# ── staging (prefix-scoped, pinned revision, off /) ─────────────────────────────


def stage_store(max_workers: int = 6) -> None:
    """Materialise the v_store shards + coverage.pt at STORE_REV under STAGE_HF (idempotent)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    want: list[str] = []
    entries = I._retry(
        lambda: list(
            api.list_repo_tree(
                DATA_REPO,
                path_in_repo=f"{HF_PREFIX}/v_store",
                repo_type="dataset",
                revision=STORE_REV,
                recursive=False,
            )
        ),
        what="list_repo_tree v_store",
    )
    want.extend(e.path for e in entries if e.path.endswith(".pt"))
    want.append(f"{HF_PREFIX}/reductions/coverage.pt")

    def _one(path_in_repo: str) -> str:
        return I._retry(
            lambda: hf_hub_download(
                repo_id=DATA_REPO,
                filename=path_in_repo,
                repo_type="dataset",
                revision=STORE_REV,
                local_dir=str(STAGE_HF),
            ),
            what=f"download {path_in_repo}",
        )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_one, want))
    logger.info("[stage] %d files under %s in %.1fs", len(want), STAGE_HF, time.time() - t0)

    # Point the reused gap_tail loaders at the staged store.
    GT.STORE_DIR = STAGE_HF / HF_PREFIX / "v_store"
    GT.COVERAGE = STAGE_HF / HF_PREFIX / "reductions" / "coverage.pt"


# ── greedy vector loader (mirrors GT.stoch_matrix — greedy loaded as an 11th rollout) ────────────


def greedy_matrix(li: int, keep: np.ndarray) -> np.ndarray:
    """(N_kept, H) fp64 greedy span-mean vectors at one layer, loaded fp16->fp64 from the greedy
    shards exactly as GT.stoch_matrix loads the stochastic draws, so greedy sits on bit-identical
    footing with the 10 stochastic rollouts."""
    pos_of = {int(ci): k for k, ci in enumerate(keep.tolist())}
    seen = np.zeros(len(keep), dtype=bool)
    v = None
    for p, shard in CAP.iter_shards(GT.STORE_DIR, "greedy"):
        li_pos = list(shard["layers"]).index(li)
        sl = shard["summ"][:, li_pos, :].to(torch.float64).numpy()
        if v is None:
            v = np.zeros((len(keep), sl.shape[1]))
        for row, (ci, _ri) in enumerate(shard["index"]):
            k = pos_of.get(int(ci))
            if k is not None:
                assert not seen[k], f"duplicate greedy ctx (ci={ci}) in {p}"
                seen[k] = True
                v[k] = sl[row]
    assert v is not None and bool(seen.all()), "greedy store fill incomplete"
    return v


# ── vectorised primitives ───────────────────────────────────────────────────────


def _unit(x: np.ndarray, axis: int) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + EPS)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < EPS or b.std() < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _percentiles(x: np.ndarray) -> dict:
    p = np.percentile(x, [5, 25, 50, 75, 95])
    return {
        "mean": float(x.mean()),
        "p5": float(p[0]),
        "p25": float(p[1]),
        "median": float(p[2]),
        "p75": float(p[3]),
        "p95": float(p[4]),
    }


# ── Analysis 1: exchangeability / rank test ─────────────────────────────────────


def rank_test(greedy: np.ndarray, stoch: np.ndarray) -> dict:
    """greedy (n,H), stoch (n,10,H) fp32. Treat greedy as item 0 of 11; each item's distance to
    the leave-self-out mean of the OTHER 10; rank greedy among the 11 by ascending distance."""
    n = greedy.shape[0]
    X = np.concatenate([greedy[:, None, :], stoch], axis=1)  # (n, 11, H)
    k = X.shape[1]
    S = X.sum(1)  # (n, H)
    m = (S[:, None, :] - X) / (k - 1)  # (n, 11, H) leave-self-out mean of the other 10
    cos = np.einsum("nih,nih->ni", _unit(X, 2), _unit(m, 2))  # (n, 11)
    d = 1.0 - cos  # distance-to-loo-mean
    dg = d[:, 0]
    rank = 1 + (d < dg[:, None]).sum(1)  # 1..11, min-rank on ties (ties measure-zero in fp)
    hist = np.bincount(rank, minlength=k + 2)[1 : k + 1]  # bins 1..11
    exp = n / k
    chi2 = float(((hist - exp) ** 2 / exp).sum())
    p_chi2 = float(sstats.chi2.sf(chi2, df=k - 1))
    return {
        "n_contexts": int(n),
        "n_items": int(k),
        "rank_histogram": hist.astype(int).tolist(),
        "rank_fractions": (hist / n).tolist(),
        "uniform_expected_fraction": 1.0 / k,
        "chi2": chi2,
        "chi2_df": k - 1,
        "chi2_p_vs_uniform": p_chi2,
        "mean_rank": float(rank.mean()),
        "mean_rank_uniform_expected": (k + 1) / 2.0,
        "median_rank": float(np.median(rank)),
        "frac_greedy_most_central": float((np.argmin(d, axis=1) == 0).mean()),
        "frac_greedy_most_peripheral": float((np.argmax(d, axis=1) == 0).mean()),
        "frac_rank_le_5": float((rank <= 5).mean()),
        "frac_rank_ge_7": float((rank >= 7).mean()),
        "greedy_mean_dist_to_loo10": float(dg.mean()),
        "stoch_mean_dist_to_loo10": float(d[:, 1:].mean()),
        "skew": ("central" if rank.mean() < (k + 1) / 2.0 else "peripheral"),
    }


# ── Analysis 2: systematic offset ───────────────────────────────────────────────


def offset_test(greedy: np.ndarray, stoch: np.ndarray, len_gap: np.ndarray) -> dict:
    n = greedy.shape[0]
    u = greedy - stoch.mean(1)  # (n, H) per-context offset
    ubar = u.mean(0)  # (H,)
    ubar_norm = float(np.linalg.norm(ubar))
    uhat = ubar / (ubar_norm + EPS)

    # (a) sign-flip permutation null on ||mean_x u(x)||
    rng = np.random.default_rng(RNG_SEED)
    signs = rng.choice([-1.0, 1.0], size=(N_SIGNFLIP, n)).astype(np.float32)
    null_means = (signs @ u) / n  # (B, H)
    null_norms = np.linalg.norm(null_means, axis=1)
    p_norm = float((1 + (null_norms >= ubar_norm).sum()) / (N_SIGNFLIP + 1))
    exp_null_sq = float((np.linalg.norm(u, axis=1) ** 2).sum() / n**2)  # analytic E[||mean||^2]

    # (b) direction consistency
    cos_u_ubar = _unit(u, 1) @ uhat  # (n,) cos(u_x, ubar)
    proj = u @ uhat  # (n,) signed offset magnitude along the mean direction

    # (c) length-gap alignment: u(x) ~ intercept(ubar) + slope(b) * gap(x)
    gc = len_gap - len_gap.mean()
    denom = float((gc**2).sum())
    b = (gc[:, None] * (u - ubar)).sum(0) / (denom + EPS)  # (H,) length-response direction
    pred = ubar[None, :] + gc[:, None] * b[None, :]
    ss_res = float(((u - pred) ** 2).sum())
    ss_tot = float(((u - ubar) ** 2).sum())
    r2_u_gap = 1.0 - ss_res / (ss_tot + EPS)
    cos_b_ubar = float(_unit(b, 0) @ uhat)

    return {
        "mean_offset_norm": ubar_norm,
        "signflip_null_norm_mean": float(null_norms.mean()),
        "signflip_null_norm_p95": float(np.percentile(null_norms, 95)),
        "signflip_null_norm_max": float(null_norms.max()),
        "signflip_p_value": p_norm,
        "analytic_expected_null_norm": float(exp_null_sq**0.5),
        "coherence_ratio_obs_over_null": ubar_norm / (float(null_norms.mean()) + EPS),
        "mean_cos_u_ubar": float(cos_u_ubar.mean()),
        "median_cos_u_ubar": float(np.median(cos_u_ubar)),
        "frac_positive_projection": float((proj > 0).mean()),
        "mean_offset_norm_percontext": _percentiles(np.linalg.norm(u, axis=1)),
        "len_gap_mean": float(len_gap.mean()),
        "len_gap_median": float(np.median(len_gap)),
        "pearson_proj_vs_len_gap": _pearson(proj, len_gap),
        "r2_offset_explained_by_len_gap": float(r2_u_gap),
        "cos_lengthdir_vs_meanoffset": cos_b_ubar,
    }


# ── Analysis 3: norms ────────────────────────────────────────────────────────────


def norm_test(greedy: np.ndarray, stoch: np.ndarray) -> dict:
    gn = np.linalg.norm(greedy, axis=1)  # (n,)
    sn = np.linalg.norm(stoch, axis=2)  # (n, 10)
    sn_mean = sn.mean(1)  # (n,)
    paired = gn - sn_mean  # (n,)
    rng = np.random.default_rng(RNG_SEED)
    n = gn.shape[0]
    boot = np.array([np.median(paired[rng.integers(0, n, n)]) for _ in range(N_BOOT)])
    return {
        "greedy_norm": _percentiles(gn),
        "stoch_norm_pooled": _percentiles(sn.reshape(-1)),
        "stoch_norm_percontext_mean": _percentiles(sn_mean),
        "paired_greedy_minus_meanstoch": _percentiles(paired),
        "paired_median_diff": float(np.median(paired)),
        "paired_median_diff_ci95": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "frac_greedy_norm_below_meanstoch": float((paired < 0).mean()),
    }


# ── Analysis 4: cloud descriptives ───────────────────────────────────────────────


def cloud_descriptives(greedy: np.ndarray, stoch: np.ndarray, disp: np.ndarray) -> dict:
    vbar10 = stoch.mean(1)  # (n, H)
    dg_cloud = 1.0 - np.einsum("nh,nh->n", _unit(greedy, 1), _unit(vbar10, 1))  # (n,)
    S = stoch.sum(1)  # (n, H)
    loo9 = (S[:, None, :] - stoch) / (N_ROLLOUTS - 1)  # (n, 10, H)
    d_s = 1.0 - np.einsum("nih,nih->ni", _unit(stoch, 2), _unit(loo9, 2))  # (n, 10) LOO distances
    paired = dg_cloud - d_s.mean(1)  # (n,)
    return {
        "rollout_dispersion": _percentiles(disp),
        "rollout_dispersion_iqr": float(np.percentile(disp, 75) - np.percentile(disp, 25)),
        "greedy_dist_to_cloud_mean": _percentiles(dg_cloud),
        "stoch_loo_dist_to_mean_pooled": _percentiles(d_s.reshape(-1)),
        "paired_greedy_minus_typical_dist": _percentiles(paired),
        "frac_greedy_farther_than_typical": float((paired > 0).mean()),
        "_hist_greedy_dist": dg_cloud.tolist(),
        "_hist_stoch_dist": d_s.reshape(-1).tolist(),
    }


# ── driver ───────────────────────────────────────────────────────────────────────


def run(layers: list[int], out_dir: Path) -> dict:
    t0 = time.time()
    keep = GT.load_keep()
    spans = GT.load_span_lens(keep)
    len_gap = spans["greedy_span"].astype(np.float64) - spans["stoch_span"].astype(np.float64).mean(
        1
    )  # (n,) greedy minus mean-stoch response length (tokens)
    logger.info("[setup] keep=%d done in %.1fs", keep.size, time.time() - t0)

    out: dict = {"readout_layers": layers, "per_layer": {}}
    for li in layers:
        tl = time.time()
        v = GT.stoch_matrix(li, keep)  # (n,10,H) fp64
        disp = GT.rollout_dispersion(v)  # reused verbatim
        v32 = v.astype(np.float32)
        del v
        g32 = greedy_matrix(li, keep).astype(np.float32)  # (n,H)
        res = {
            "rank_test": rank_test(g32, v32),
            "offset_test": offset_test(g32, v32, len_gap),
            "norm_test": norm_test(g32, v32),
            "cloud_descriptives": cloud_descriptives(g32, v32, disp),
        }
        out["per_layer"][f"L{li}"] = res
        del v32, g32
        logger.info(
            "[L%d] rank mean=%.3f (unif %.1f) chi2p=%.2e central=%.3f periph=%.3f | "
            "offset ||ubar||=%.4f p=%.4f meancos=%.3f r2len=%.3f | done %.1fs",
            li,
            res["rank_test"]["mean_rank"],
            res["rank_test"]["mean_rank_uniform_expected"],
            res["rank_test"]["chi2_p_vs_uniform"],
            res["rank_test"]["frac_greedy_most_central"],
            res["rank_test"]["frac_greedy_most_peripheral"],
            res["offset_test"]["mean_offset_norm"],
            res["offset_test"]["signflip_p_value"],
            res["offset_test"]["mean_cos_u_ubar"],
            res["offset_test"]["r2_offset_explained_by_len_gap"],
            time.time() - tl,
        )

    out["definitions"] = {
        "rank_test": (
            "treat v_greedy as an 11th rollout; each of the 11 items' cosine distance to the "
            "leave-self-out mean of the other 10; rank greedy 1..11 by ascending distance "
            "(1=most central). Uniform on 1..11 iff greedy is exchangeable with a stochastic draw."
        ),
        "offset_u": "u(x) = v_greedy(x) - mean_j v_j(x); u_bar = mean_x u(x)",
        "signflip_null": (
            "null distribution of ||mean_x s_x u(x)|| over N_SIGNFLIP random per-context sign "
            "flips s_x in {-1,+1}; p = fraction of null >= observed ||u_bar||"
        ),
        "len_gap": "greedy response span length - mean stochastic response span length (tokens)",
        "r2_offset_explained_by_len_gap": (
            "R2 of the per-context offset u(x) explained by a scalar linear map on the length gap: "
            "u(x) ~ u_bar + b * (gap(x)-mean gap)"
        ),
        "cloud_descriptives": (
            "greedy distance to the 10-mean vs each stochastic draw's leave-one-out distance to "
            "the mean of the other 9 (both external to the reference mean, so comparable)"
        ),
    }
    out["metadata"] = I.reproducibility_metadata(
        {
            "script": "issue1073_greedy_cloud_distribution",
            "store_revision": STORE_REV,
            "n_signflip": N_SIGNFLIP,
            "n_boot": N_BOOT,
            "rng_seed": RNG_SEED,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    I.write_json_atomic(out_dir / "greedy_cloud_distribution.json", out)
    logger.info("[write] greedy_cloud_distribution.json in %.1fs total", time.time() - t0)
    return out


# ── figure ───────────────────────────────────────────────────────────────────────


def make_figure(out_dir: Path, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "greedy_cloud_distribution.json") as f:
        res = json.load(f)
    layers = [int(k[1:]) for k in res["per_layer"]]
    pal = pp.paper_palette(len(layers))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    (a1, a2), (a3, a4) = axes

    # Panel A: rank histogram per layer + uniform reference
    k = res["per_layer"][f"L{layers[0]}"]["rank_test"]["n_items"]
    ranks = np.arange(1, k + 1)
    for i, li in enumerate(layers):
        fr = res["per_layer"][f"L{li}"]["rank_test"]["rank_fractions"]
        a1.plot(ranks, fr, "o-", ms=4, color=pal[i], label=f"L{li}")
    a1.axhline(1.0 / k, color="k", ls="--", lw=1.0)
    a1.set_xlabel("greedy centrality rank among 11 items (1 = most central)")
    a1.set_ylabel("fraction of contexts")
    a1.set_xticks(ranks)
    a1.legend(fontsize=7, ncol=2, title="uniform = dashed")

    # Panel B: greedy vs typical distance-to-mean, headline layer (largest greedy offset)
    hl = max(layers, key=lambda li: res["per_layer"][f"L{li}"]["offset_test"]["mean_offset_norm"])
    cd = res["per_layer"][f"L{hl}"]["cloud_descriptives"]
    g = np.array(cd["_hist_greedy_dist"])
    s = np.array(cd["_hist_stoch_dist"])
    lo, hi = float(min(g.min(), s.min())), float(max(g.max(), s.max()))
    bins = np.linspace(lo, hi, 60)
    a2.hist(s, bins=bins, density=True, alpha=0.5, color=pal[0], label="stochastic draw (LOO)")
    a2.hist(g, bins=bins, density=True, alpha=0.5, color=pal[-1], label="greedy")
    a2.set_xlabel(f"cosine distance to cloud mean (L{hl})")
    a2.set_ylabel("density")
    a2.legend(fontsize=8)

    # Panel C: observed mean-offset norm vs sign-flip null per layer
    x = np.arange(len(layers))
    obs = [res["per_layer"][f"L{li}"]["offset_test"]["mean_offset_norm"] for li in layers]
    nullm = [res["per_layer"][f"L{li}"]["offset_test"]["signflip_null_norm_mean"] for li in layers]
    nullmax = [res["per_layer"][f"L{li}"]["offset_test"]["signflip_null_norm_max"] for li in layers]
    yerr = np.maximum(0.0, np.array(nullmax) - np.array(nullm))
    w = 0.38
    a3.bar(x - w / 2, obs, w, color=pal[-1], label="observed $\\|\\bar u\\|$")
    a3.bar(
        x + w / 2,
        nullm,
        w,
        yerr=[np.zeros(len(layers)), yerr],
        color=pal[0],
        label="sign-flip null (mean; cap=max)",
        capsize=3,
    )
    a3.set_xticks(x)
    a3.set_xticklabels([f"L{li}" for li in layers])
    a3.set_ylabel("mean-offset norm")
    a3.set_xlabel("read-out layer")
    a3.legend(fontsize=7)

    # Panel D: paired greedy-minus-mean-stoch norm diff per layer, bootstrap CI
    med = [res["per_layer"][f"L{li}"]["norm_test"]["paired_median_diff"] for li in layers]
    ci = [res["per_layer"][f"L{li}"]["norm_test"]["paired_median_diff_ci95"] for li in layers]
    lo_e = np.maximum(0.0, np.array(med) - np.array([c[0] for c in ci]))
    hi_e = np.maximum(0.0, np.array([c[1] for c in ci]) - np.array(med))
    a4.axhline(0, color="k", lw=0.8)
    a4.errorbar(x, med, yerr=[lo_e, hi_e], fmt="o", color=pal[2], capsize=3)
    a4.set_xticks(x)
    a4.set_xticklabels([f"L{li}" for li in layers])
    a4.set_ylabel("median $\\|v_{greedy}\\| - \\overline{\\|v_j\\|}$")
    a4.set_xlabel("read-out layer")

    fig.tight_layout()
    pp.savefig_paper(fig, "greedy_cloud_distribution", dir=fig_dir)
    plt.close(fig)
    logger.info("[figure] wrote greedy_cloud_distribution to %s", fig_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=READOUT_LAYERS)
    ap.add_argument("--pilot", action="store_true", help="single layer (19) for wall-time probe")
    ap.add_argument("--out-dir", type=str, default=str(EVAL_RESULTS_DIR))
    ap.add_argument("--fig-dir", type=str, default=str(FIG_DIR))
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--skip-stage", action="store_true", help="store already staged at STAGE_HF")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.figures_only:
        make_figure(out_dir, Path(args.fig_dir))
        return 0
    if not args.skip_stage:
        stage_store()
    else:
        GT.STORE_DIR = STAGE_HF / HF_PREFIX / "v_store"
        GT.COVERAGE = STAGE_HF / HF_PREFIX / "reductions" / "coverage.pt"
    layers = [19] if args.pilot else list(args.layers)
    run(layers, out_dir)
    if not args.pilot:
        make_figure(out_dir, Path(args.fig_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
