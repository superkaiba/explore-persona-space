"""Issue #931 free-analysis follow-up: distance-covariate read of the swap gap.

The run's #1 unmeasured confound: correct (C, T) pairs are constructionally
CLOSER (T begins right after C) than within-window swap pairs (the partner's
T can sit anywhere in the window), so residual-stream autocorrelation decay
alone could produce dR2_char > 0. This script regresses the per-pair paired
residual difference (swap - correct, in dR2-contribution units) on the C->T
token-distance gap between the swapped and correct targets at L19, with
novel-level (group) clustering via the run's own 1,000-draw paired group
bootstrap (seed 0, SAME draws applied to both terms), and reports the
distance-partialled swap gap = the regression intercept (the gap at zero
distance difference).

Per-pair decomposition (exact): with SS_c / SS_s the pooled total sum of
squares of the correct / swap target sets over the derangement-eligible kept
subset, u_i = n * (err_swap_i / SS_s - err_correct_i / SS_c) satisfies
mean(u) = dR2_char exactly (validated in-run against the committed
delta_char_arm{A,B}.json). The distance covariate is
gap_i = |t_pos(partner) - c_pos(row)| - (t_pos(row) - c_pos(row)) in tokens,
where c_pos = mean token index of the intro span C and t_pos = the
length-weighted mean token index over the target quotation spans (matching
the store's span-mean X and target-mean Y reductions). Swap partners share
the window, so both positions live in one coordinate frame (the armB
prefix offset shifts C and T equally and cancels).

Inputs (all existing; HF revision-pinned; pairs_meta rows carry fiction TEXT
fields — only numeric/id fields are ever loaded):
  HF issue931_story_map/analysis_tensors/{armA,armB}/  (store shards; y@L19)
  HF issue931_story_map/analysis_tensors/preds/{arm}_{within,swap}_L19.npz
  HF issue931_story_map/raw_completions/pairs_meta/pairs_arm{A,B}.jsonl
  eval_results/issue_931/delta_char_arm{A,B}.json      (parity targets)

Outputs:
  eval_results/issue_931/delta_char_distance_covariate.json
  figures/issue_931/delta_char_distance_partialled.{png,pdf,meta.json}

CLI:
  uv run python scripts/issue931_distance_covariate.py
      [--data-dir data/issue_931] [--out-dir eval_results/issue_931]
      [--fig-dir figures/issue_931] [--n-boot 1000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps bind before torch/numpy import

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common  # noqa: E402
import issue931_fit_cells as fitc  # noqa: E402

SCRIPT = "scripts/issue931_distance_covariate.py"
HF_REVISION = "9534b9981d6b4fb4f1259c9b06f021d311a46af4"
LAYER = common.HEADLINE_LAYER  # 19
ARMS = ("armA", "armB")
STORE_SHARDS = {"armA": 4, "armB": 3}
PARITY_TOL = 5e-3  # fp16-stored preds vs the run's in-memory fp32 preds


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_931"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_931"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_931"))
    ap.add_argument("--n-boot", type=int, default=common.N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=common.FIT_SEED)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Staging (pinned revision; per-file hf_hub_download — never snapshot_download
# on the ~1M-file data repo; <=6 workers, bounded retry)
# ---------------------------------------------------------------------------


def stage_inputs(data_dir: Path) -> Path:
    """Download store shards + preds npz + pairs jsonl at HF_REVISION; return root."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    root = data_dir / "hf_dl" / "distcov"
    paths = []
    for arm in ARMS:
        for i in range(STORE_SHARDS[arm]):
            paths.append(f"issue931_story_map/analysis_tensors/{arm}/{arm}_shard{i:03d}.pt")
        for cell in (f"{arm}_within", f"{arm}_swap"):
            paths.append(f"issue931_story_map/analysis_tensors/preds/{cell}_L{LAYER}.npz")
        letter = arm[-1]
        paths.append(f"issue931_story_map/raw_completions/pairs_meta/pairs_arm{letter}.jsonl")

    def _fetch(path: str) -> str:
        for attempt in range(4):
            try:
                hf_hub_download(
                    common.HF_DATA_REPO,
                    path,
                    repo_type="dataset",
                    revision=HF_REVISION,
                    local_dir=root,
                )
                return path
            except Exception as exc:  # transient Hub 5xx/429 — bounded retry
                if attempt == 3:
                    raise
                wait = 20 * (attempt + 1)
                print(f"[i931-distcov] retry {path} in {wait}s: {exc}")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(_fetch, paths))
    print(f"[i931-distcov] staged {len(paths)} files at revision {HF_REVISION}")
    return root


# ---------------------------------------------------------------------------
# Slim store load — y @ L19 only (peak RAM ~ one shard, not the full 4-array
# store; the shared-VM earlyoom-safety sibling of fitc.load_regime_store)
# ---------------------------------------------------------------------------


def load_store_y19(store_dir: Path, regime: str) -> dict:
    """Concatenate shards in fitc.load_regime_store order; keep only y[:, L19]."""
    import torch

    shards = sorted(store_dir.glob(f"{regime}_shard*.pt"))
    assert shards, f"no {regime} shards under {store_dir}"
    rows, groups, chars, y19 = [], [], [], []
    for sp in shards:
        payload = torch.load(sp, map_location="cpu", weights_only=False)
        rows.extend(payload["row_ids"])
        groups.extend(payload["group_ids"])
        chars.extend(payload["char_ids"])
        y = payload["arrays"]["y"]
        assert y.shape[1] == common.EXPECTED_LAYERS, y.shape
        y19.append(y[:, LAYER, :].float().numpy().astype(np.float32))
        del payload, y
    out = np.concatenate(y19, axis=0)
    n = len(rows)
    assert out.shape == (n, common.EXPECTED_HIDDEN), out.shape
    return {
        "row_ids": np.asarray(rows),
        "group_ids": np.asarray(groups),
        "char_ids": np.asarray(chars),
        "y19": out,
    }


def load_pair_positions(pairs_jsonl: Path) -> dict[str, tuple[float, float, str]]:
    """row_id -> (c_pos, t_pos, window_id). NUMERIC/id fields only — pairs_meta
    rows carry fiction span TEXT (meta.c_text); it is never read into memory
    beyond the transient json parse and never printed/stored."""
    out: dict[str, tuple[float, float, str]] = {}
    with open(pairs_jsonl) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            cs, ce = d["c_span"]
            c_pos = (cs + ce - 1) / 2.0
            tot = 0
            wsum = 0.0
            for lo, hi in d["t_spans"]:
                assert hi > lo, (d["row_id"], lo, hi)
                wsum += (lo + hi - 1) / 2.0 * (hi - lo)
                tot += hi - lo
            out[d["row_id"]] = (c_pos, wsum / tot, str(d["meta"]["window_id"]))
    return out


def load_preds_npz(path: Path, n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(full_preds (n_rows, D) with NaN on unfitted, row_mask, fitted row_ids)."""
    z = np.load(path, allow_pickle=False)
    mask = z["row_mask"].astype(bool)
    assert mask.shape == (n_rows,), (mask.shape, n_rows)
    full = np.full((n_rows, common.EXPECTED_HIDDEN), np.nan, dtype=np.float64)
    full[mask] = z["preds"].astype(np.float64)
    return full, mask, z["row_ids"].astype(str)


# ---------------------------------------------------------------------------
# Pure batched regression (unit-tested; all 1 + n_boot draws as (draws, G)
# GEMMs over per-group reductions — zero per-draw python loops)
# ---------------------------------------------------------------------------


def distance_partialled_gap(
    err_correct: np.ndarray,
    err_swap: np.ndarray,
    dist_gap: np.ndarray,
    group_ids: np.ndarray,
    y_true_correct: np.ndarray,
    y_true_swap: np.ndarray,
    draws_matrix: np.ndarray,
) -> dict:
    """Distance-partialled swap gap with a paired group (cluster) bootstrap.

    Per pair i: u_i = N * (err_swap_i / SS_s - err_correct_i / SS_c), where
    SS_c/SS_s are the pooled total sums of squares of the correct/swap target
    sets under the draw's group multiplicities (identity for the observed
    row), so the weighted mean of u equals dR2_char = R2_correct - R2_swap
    exactly. WLS of u on dist_gap with the draw's multiplicities as weights;
    returns observed {intercept, slope, delta, r2_correct, r2_swap} plus the
    per-draw arrays. All draws evaluate as (draws, G) @ (G, .) products.
    """
    a = np.asarray(err_swap, dtype=np.float64)
    b = np.asarray(err_correct, dtype=np.float64)
    x = np.asarray(dist_gap, dtype=np.float64)
    yc = np.asarray(y_true_correct, dtype=np.float64)
    ys = np.asarray(y_true_swap, dtype=np.float64)
    n = len(a)
    assert a.shape == b.shape == x.shape == (n,), (a.shape, b.shape, x.shape)
    assert yc.shape[0] == ys.shape[0] == n, (yc.shape, ys.shape)
    uniq, inv = np.unique(np.asarray(group_ids), return_inverse=True)
    G = len(uniq)
    assert draws_matrix.shape[1] == G, (draws_matrix.shape, G)

    n_g = np.bincount(inv, minlength=G).astype(np.float64)
    sa_g = np.bincount(inv, weights=a, minlength=G)
    sb_g = np.bincount(inv, weights=b, minlength=G)
    sx_g = np.bincount(inv, weights=x, minlength=G)
    sxx_g = np.bincount(inv, weights=x * x, minlength=G)
    sxa_g = np.bincount(inv, weights=x * a, minlength=G)
    sxb_g = np.bincount(inv, weights=x * b, minlength=G)
    D = yc.shape[1]
    sumy_c = np.zeros((G, D))
    sumy_s = np.zeros((G, D))
    np.add.at(sumy_c, inv, yc)
    np.add.at(sumy_s, inv, ys)
    sumsq_c = np.bincount(inv, weights=(yc**2).sum(axis=1), minlength=G)
    sumsq_s = np.bincount(inv, weights=(ys**2).sum(axis=1), minlength=G)

    # Row 0 = observed (identity multiplicities); rows 1: = bootstrap draws.
    M = np.vstack([np.ones((1, G)), np.asarray(draws_matrix, dtype=np.float64)])
    N = M @ n_g  # (draws+1,)
    ss_c = M @ sumsq_c - ((M @ sumy_c) ** 2).sum(axis=1) / np.maximum(N, 1.0)
    ss_s = M @ sumsq_s - ((M @ sumy_s) ** 2).sum(axis=1) / np.maximum(N, 1.0)
    assert (ss_c > 1e-9).all() and (ss_s > 1e-9).all(), "degenerate SS_tot in a draw"
    A, B = M @ sa_g, M @ sb_g
    Sy = N * (A / ss_s - B / ss_c)
    Sxy = N * ((M @ sxa_g) / ss_s - (M @ sxb_g) / ss_c)
    Sx, Sxx = M @ sx_g, M @ sxx_g
    denom = N * Sxx - Sx**2
    assert (np.abs(denom) > 1e-12).all(), "degenerate covariate in a draw"
    beta = (N * Sxy - Sx * Sy) / denom
    alpha = (Sy - beta * Sx) / N
    delta = Sy / N  # == r2_correct - r2_swap per draw (paired, shared draws)
    return {
        "intercept": float(alpha[0]),
        "slope": float(beta[0]),
        "delta": float(delta[0]),
        "r2_correct": float(1.0 - B[0] / ss_c[0]),
        "r2_swap": float(1.0 - A[0] / ss_s[0]),
        "u_obs": n * (a / ss_s[0] - b / ss_c[0]),
        "intercept_draws": alpha[1:],
        "slope_draws": beta[1:],
        "delta_draws": delta[1:],
        "n_groups": int(G),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(
        np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
    )


def _ci(draws: np.ndarray) -> tuple[float, float]:
    return float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))


# ---------------------------------------------------------------------------
# Per-arm analysis (reconstructs delta_char's kept subset exactly)
# ---------------------------------------------------------------------------


def analyze_arm(arm: str, staged: Path, args: argparse.Namespace) -> dict:
    """Rebuild the delta_char subset, attach the distance gap, run the read."""
    store_dir = staged / "issue931_story_map" / "analysis_tensors" / arm
    preds_dir = staged / "issue931_story_map" / "analysis_tensors" / "preds"
    pairs_path = (
        staged / "issue931_story_map" / "raw_completions" / "pairs_meta" / f"pairs_{arm}.jsonl"
    )
    store = load_store_y19(store_dir, arm)
    rids = store["row_ids"]
    n_store = len(rids)
    assert len(set(rids.tolist())) == n_store, "duplicate store row_ids"

    # Seeded within-window derangement — identical to the run (seed 931).
    rows, partners = fitc.swap_derangement(
        {"row_ids": store["row_ids"], "char_ids": store["char_ids"]}
    )
    swap_row_ids = rids[rows]

    pred_w_full, fitted_w, w_ids = load_preds_npz(preds_dir / f"{arm}_within_L{LAYER}.npz", n_store)
    pred_s_full, fitted_s, s_ids = load_preds_npz(preds_dir / f"{arm}_swap_L{LAYER}.npz", len(rows))
    # Derangement + row-order reproduction validated EXACTLY vs the run's npz.
    assert (rids[fitted_w] == w_ids).all(), f"{arm}: within row-id mismatch vs npz"
    assert (swap_row_ids[fitted_s] == s_ids).all(), f"{arm}: derangement mismatch vs npz"

    # delta_char's kept subset (same construction as fitc.delta_char).
    pos = {r: i for i, r in enumerate(rids)}
    sub = np.asarray([pos[r] for r in swap_row_ids])
    keep = fitted_w[sub] & fitted_s
    sub = sub[keep]
    part = partners[keep]
    pred_c = pred_w_full[sub]
    true_c = store["y19"][sub].astype(np.float64)
    pred_s = pred_s_full[keep]
    true_s = store["y19"][part].astype(np.float64)
    groups = store["group_ids"][rows][keep]
    assert not np.isnan(pred_c).any() and not np.isnan(pred_s).any()

    # Distance covariate from pairs_meta (numeric fields only).
    pp = load_pair_positions(pairs_path)
    assert all(r in pp for r in rids.tolist()), f"{arm}: store row_ids missing from pairs_meta"
    c_pos = np.asarray([pp[r][0] for r in rids])
    t_pos = np.asarray([pp[r][1] for r in rids])
    win = np.asarray([pp[r][2] for r in rids])
    assert (win[sub] == win[part]).all(), f"{arm}: swap pair crosses windows"
    dist_correct = t_pos[sub] - c_pos[sub]
    assert (dist_correct > 0).all(), f"{arm}: correct target not after C"
    dist_swap = np.abs(t_pos[part] - c_pos[sub])
    gap = dist_swap - dist_correct

    err_c = ((true_c - pred_c) ** 2).sum(axis=1)
    err_s = ((true_s - pred_s) ** 2).sum(axis=1)

    # The run's own paired group bootstrap (seed 0; draws shared across terms).
    gb_c = fitc.group_bootstrap_r2(pred_c, true_c, groups, n_boot=args.n_boot, seed=args.seed)
    gb_s = fitc.group_bootstrap_r2(
        pred_s,
        true_s,
        groups,
        n_boot=args.n_boot,
        seed=args.seed,
        draws_matrix=gb_c["draws_matrix"],
    )
    res = distance_partialled_gap(err_c, err_s, gap, groups, true_c, true_s, gb_c["draws_matrix"])

    # Internal parity: per-pair decomposition reproduces the R2 machinery.
    assert abs(res["r2_correct"] - gb_c["r2"]) < 1e-9, (res["r2_correct"], gb_c["r2"])
    assert abs(res["r2_swap"] - gb_s["r2"]) < 1e-9, (res["r2_swap"], gb_s["r2"])
    assert np.allclose(res["delta_draws"], gb_c["draws"] - gb_s["draws"], atol=1e-9, equal_nan=True)

    # Parity vs the committed run values (fp16-stored preds => small tolerance).
    committed = json.loads((args.out_dir / f"delta_char_{arm}.json").read_text())
    dev = abs(res["delta"] - committed["delta_r2_char"])
    assert len(sub) == committed["n_rows"], (len(sub), committed["n_rows"])
    assert res["n_groups"] == committed["n_groups"]
    assert dev < PARITY_TOL, f"{arm}: recomputed delta {res['delta']} vs committed {dev}"
    print(
        f"[i931-distcov] {arm}: n={len(sub)} groups={res['n_groups']} "
        f"delta={res['delta']:.6f} (committed {committed['delta_r2_char']:.6f}, "
        f"|dev|={dev:.2e}) intercept={res['intercept']:.6f} slope={res['slope']:.3e}"
    )

    u = res["u_obs"]
    ilo, ihi = _ci(res["intercept_draws"])
    slo, shi = _ci(res["slope_draws"])
    dlo, dhi = _ci(res["delta_draws"])
    # Quantile-binned view (8 bins of the distance gap).
    edges = np.quantile(gap, np.linspace(0, 1, 9))
    edges[-1] += 1e-9
    bin_idx = np.clip(np.searchsorted(edges, gap, side="right") - 1, 0, 7)
    bins = []
    for k in range(8):
        m = bin_idx == k
        bins.append(
            {
                "gap_mean": float(gap[m].mean()),
                "u_mean": float(u[m].mean()),
                "u_se": float(u[m].std(ddof=1) / np.sqrt(m.sum())),
                "n": int(m.sum()),
            }
        )
    return {
        "n_pairs": len(sub),
        "n_groups": int(res["n_groups"]),
        "layer": LAYER,
        "delta_r2_char_recomputed": res["delta"],
        "delta_r2_char_committed": float(committed["delta_r2_char"]),
        "delta_recompute_abs_dev": float(dev),
        "delta_ci_lo": dlo,
        "delta_ci_hi": dhi,
        "intercept_distance_partialled": res["intercept"],
        "intercept_ci_lo": ilo,
        "intercept_ci_hi": ihi,
        "slope_per_token": res["slope"],
        "slope_ci_lo": slo,
        "slope_ci_hi": shi,
        "pearson_gap_vs_u": _pearson(gap, u),
        "spearman_gap_vs_u": _spearman(gap, u),
        "dist_correct_mean_tokens": float(dist_correct.mean()),
        "dist_swap_mean_tokens": float(dist_swap.mean()),
        "gap_mean_tokens": float(gap.mean()),
        "binned": bins,
        "_points": {"gap": gap, "u": u},
    }


# ---------------------------------------------------------------------------
# Figure (paper-plots conventions; per-pair points embedded in .meta.json)
# ---------------------------------------------------------------------------

ARM_LABEL = {"armA": "Real-novel character map", "armB": "Model-written story map"}


def make_figure(results: dict[str, dict], fig_dir: Path) -> None:
    """Two panels (arm A/B): per-pair points, binned means, fit line, and the
    raw vs distance-partialled swap gap with bootstrap CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    colors = paper_palette(3)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), layout="constrained")
    meta_points = {}
    for ax, arm in zip(axes, ARMS, strict=True):
        r = results[arm]
        gap, u = r["_points"]["gap"], r["_points"]["u"]
        ax.scatter(gap, u, s=6, alpha=0.15, color=colors[0], edgecolors="none", rasterized=True)
        bx = [b["gap_mean"] for b in r["binned"]]
        by = [b["u_mean"] for b in r["binned"]]
        be = [b["u_se"] for b in r["binned"]]
        ax.errorbar(bx, by, yerr=be, fmt="o-", color=colors[1], ms=4, lw=1.4, capsize=2)
        xs = np.linspace(float(gap.min()), float(gap.max()), 50)
        ax.plot(
            xs,
            r["intercept_distance_partialled"] + r["slope_per_token"] * xs,
            color=colors[2],
            lw=1.6,
        )
        ax.axhline(r["delta_r2_char_recomputed"], color="0.4", lw=1.0, ls="--")
        ax.errorbar(
            [0.0],
            [r["intercept_distance_partialled"]],
            yerr=[
                [r["intercept_distance_partialled"] - r["intercept_ci_lo"]],
                [r["intercept_ci_hi"] - r["intercept_distance_partialled"]],
            ],
            fmt="D",
            color=colors[2],
            ms=6,
            capsize=3,
            zorder=5,
        )
        ax.axvline(0.0, color="0.85", lw=0.8)
        ax.set_title(
            f"{ARM_LABEL[arm]}: raw gap {r['delta_r2_char_recomputed']:+.3f}, "
            f"at zero distance {r['intercept_distance_partialled']:+.3f} "
            f"[{r['intercept_ci_lo']:+.3f}, {r['intercept_ci_hi']:+.3f}]",
            fontsize=9,
        )
        ax.set_xlabel("swap-target distance minus correct-target distance (tokens)")
        ax.set_ylabel("per-pair error difference, swap minus correct\n(mean = character gap)")
        ax.set_ylim(np.quantile(u, 0.005), np.quantile(u, 0.995))
        meta_points[arm] = {
            "gap_tokens": [round(float(v), 2) for v in gap],
            "u": [round(float(v), 5) for v in u],
            "binned": r["binned"],
        }
    fig.savefig(fig_dir / "delta_char_distance_partialled.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "delta_char_distance_partialled.pdf", bbox_inches="tight")
    (fig_dir / "delta_char_distance_partialled.meta.json").write_text(
        json.dumps(
            {
                "metadata": common.metadata(SCRIPT, common.FIT_SEED, 0),
                "what": "per-pair (distance gap, normalized error difference) points; "
                "binned means +/- SE; WLS fit; dashed = raw gap; diamond = "
                "distance-partialled gap (intercept at zero gap) with 95% CI",
                "points": meta_points,
            },
            indent=2,
            default=float,
        )
    )
    plt.close(fig)
    print(f"[i931-distcov] wrote {fig_dir / 'delta_char_distance_partialled.png'}")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_inputs(args.data_dir)
    results = {arm: analyze_arm(arm, staged, args) for arm in ARMS}
    payload = {
        "metadata": common.metadata(SCRIPT, args.seed, sum(r["n_pairs"] for r in results.values())),
        "recipe": {
            "hf_revision": HF_REVISION,
            "layer": LAYER,
            "seed": int(args.seed),
            "n_boot": int(args.n_boot),
            "bootstrap": "novel-level paired group bootstrap; identical draws applied to "
            "both terms and to the regression (cluster bootstrap of the WLS)",
            "derangement_seed": common.BUILD_SEED,
            "per_pair_unit": "u_i = n*(err_swap_i/SS_swap - err_correct_i/SS_correct); "
            "mean(u) = delta R^2_char exactly",
            "distance_definition": "C->T token distance = |length-weighted mean target "
            "token index - mean intro-span token index|; gap = swap-target distance - "
            "correct-target distance (correct distance is positive by construction)",
            "preds_precision": "fp16-stored held-out preds (parity vs committed delta "
            f"asserted < {PARITY_TOL})",
        },
        "arms": {
            arm: {k: v for k, v in r.items() if not k.startswith("_")} for arm, r in results.items()
        },
    }
    common.write_json(args.out_dir / "delta_char_distance_covariate.json", payload)
    make_figure(results, args.fig_dir)
    print("[i931-distcov] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
