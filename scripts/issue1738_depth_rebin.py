"""Fine-grained exact-depth rebin of the bare-vs-prefix reversal (#1738 free-analysis).

The committed depth analysis (``depth_contrasts.json``) bins conversation depth
coarsely ({2, 3-4, >=5} user turns) and measured bare-vs-prefix per-context
dominance only in aggregate (bare worse than prefix on 30.8% of held-out
contexts; worse than context on 88.0%). This script re-bins the SAME held-out
per-context errors (L19 ridge, n=9,941) at EXACT user-turn counts (2, 3, 4, ...,
collapsing the tail into ``>=K`` once a stratum would fall under ``--min-stratum-n``)
and asks: does the bare-vs-prefix ordering CROSS OVER at any realized depth?

Per stratum it emits: n, mean/median normalized error per arm (bare / prefix /
context), bare-vs-prefix and bare-vs-context per-context dominance fractions
with vectorized percentile-bootstrap CIs, the paired mean error delta, and the
holdout R^2 + mean cosine per arm (subset-mean-denominator convention via the
round's canonical ``_recon_point``).

Everything is a groupby over existing committed/local artifacts — zero GPU,
zero new data. Depth derivation + join reuse the round's canonical helpers
(``GG.N1M.read_manifest_pool`` for the manifest pool, ``GG._depth_band`` for the
coarse-band cross-check, ``FT.load_split`` for the holdout set, ``F._recon_point``
for R^2). Sanity asserts: stratum n's sum to the holdout count, the aggregate
dominance fractions reproduce the committed 30.8%/88.0% within rounding, and
the coarse-band R^2 values reproduce the committed depth_contrasts.json to 1e-6.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_fitter_fair_comparison as F  # noqa: E402  (_recon_point: pooled R^2 + cosine)
import issue1738_multiturn_fits as FT  # noqa: E402  (load_split)
import issue1738_multiturn_generate_capture as GG  # noqa: E402  (N1M manifest pool, _depth_band)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = GG.logging.getLogger("issue1738_depth_rebin")

DEFAULT_CSV = Path("eval_results/issue_1738/bare_query/percontext_summary_L19_ridge.csv")
DEFAULT_MANIFEST = Path("data/issue_1738/mt100k/sampling_manifest")
DEFAULT_SPLIT = Path("data/issue_1738/mt100k/sampling_manifest/split_1738.json")
DEFAULT_PRED16 = Path("data/issue_1738/mt100k/fits/pred16")
DEFAULT_Y_HOLDOUT = Path("data/issue_1738/mt100k/fits/y_holdout")
DEFAULT_OUT = Path("eval_results/issue_1738/bare_query/depth_rebin.json")
DEFAULT_FIG_DIR = Path("figures/issue_1738")
DEFAULT_COARSE_BARE = Path("eval_results/issue_1738/bare_query/depth_contrasts.json")
DEFAULT_COARSE_PARENT = Path("eval_results/issue_1738/depth_contrasts.json")

ARMS = ("bare", "prefix", "context")
LAYER = 19
FITTER = "ridge"
# Committed aggregate dominance fractions (task #1738 body: "bare worse than prefix on
# only 30.8% of contexts ... and more than context on 88.0%") — reproduction targets.
COMMITTED_FRAC_BARE_WORSE_PREFIX = 0.308
COMMITTED_FRAC_BARE_WORSE_CONTEXT = 0.880


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _manifest_depths(manifest_dir: Path) -> dict[int, int]:
    """ci -> exact user-turn depth from the canonical (invariant-checked) pool reader."""
    pool, _meta = GG.N1M.read_manifest_pool(manifest_dir)
    return {int(r["i"]): int(r["depth"]) for r in pool}


def _strata_labels(depths: np.ndarray, min_n: int) -> tuple[list[str], np.ndarray]:
    """Exact user-turn strata 2..K-1 plus a ``>=K`` tail, K chosen so every exact
    stratum has n >= min_n and the tail (which always has n >= its last exact
    member's would-be n) absorbs the rest. Returns (labels, per-row label idx)."""
    counts = pd.Series(depths).value_counts().sort_index()
    assert int(counts.index.min()) == 2, f"unexpected min depth {counts.index.min()}"
    k = 2
    while k in counts.index and int(counts.loc[k]) >= min_n:
        k += 1
    # tail = all depths >= k; require it non-degenerate (it is: brief allows K+)
    labels = [str(d) for d in range(2, k)] + [f">={k}"]
    idx = np.where(depths >= k, len(labels) - 1, depths - 2).astype(np.int64)
    assert idx.min() >= 0 and idx.max() < len(labels)
    return labels, idx


def _boot_fracs(ind: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """95% percentile-bootstrap CI of a fraction: ONE vectorized (n_boot, n) draw."""
    n = ind.shape[0]
    draws = ind[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _boot_mean(vals: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    n = vals.shape[0]
    draws = vals[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _stratum_stats(
    sel: np.ndarray,
    nerr: dict[str, np.ndarray],
    preds: dict[str, np.ndarray],
    y: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    b, p, c = (nerr[a][sel] for a in ARMS)
    ind_bp = (b > p).astype(np.float64)
    ind_bc = (b > c).astype(np.float64)
    delta_bp = b - p
    row: dict = {"n": int(sel.sum())}
    for arm in ARMS:
        v = nerr[arm][sel]
        row[f"nerr_{arm}_mean"] = float(v.mean())
        row[f"nerr_{arm}_median"] = float(np.median(v))
    row["frac_bare_worse_than_prefix"] = float(ind_bp.mean())
    row["frac_bare_worse_than_prefix_ci"] = list(_boot_fracs(ind_bp, n_boot, rng))
    row["frac_bare_worse_than_context"] = float(ind_bc.mean())
    row["frac_bare_worse_than_context_ci"] = list(_boot_fracs(ind_bc, n_boot, rng))
    row["mean_nerr_bare_minus_prefix"] = float(delta_bp.mean())
    row["mean_nerr_bare_minus_prefix_ci"] = list(_boot_mean(delta_bp, n_boot, rng))
    for arm in ARMS:
        r2, cos = F._recon_point(preds[arm][sel], y[sel])
        row[f"r2_{arm}"] = float(r2)
        row[f"mean_cosine_{arm}"] = float(cos)
    return row


def _coarse_band_crosscheck(
    depths_row: np.ndarray,
    preds: dict[str, np.ndarray],
    y: np.ndarray,
    coarse_paths: dict[str, Path],
) -> dict:
    """Recompute the committed coarse-band R^2 per arm and assert 1e-6 agreement
    with depth_contrasts.json — pins this run to the committed artifacts."""
    out: dict = {}
    for arm, path in coarse_paths.items():
        committed = json.loads(path.read_text())["arms"][f"{arm}_L{LAYER}_{FITTER}"]
        for band, doc in committed.items():
            sel = np.asarray([GG._depth_band(int(d)) == band for d in depths_row], dtype=bool)
            r2, _cos = F._recon_point(preds[arm][sel], y[sel])
            assert int(sel.sum()) == int(doc["n"]), (arm, band, int(sel.sum()), doc["n"])
            assert abs(float(r2) - float(doc["r2"])) < 1e-6, (arm, band, r2, doc["r2"])
            out[f"{arm}:{band}"] = {"n": int(sel.sum()), "r2": float(r2)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--percontext-csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--pred16-dir", type=Path, default=DEFAULT_PRED16)
    ap.add_argument("--y-holdout-dir", type=Path, default=DEFAULT_Y_HOLDOUT)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1738)
    ap.add_argument("--min-stratum-n", type=int, default=200)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    df = pd.read_csv(args.percontext_csv)
    nerr = {a: df[f"nerr_{a}_L{LAYER}_{FITTER}"].to_numpy(np.float64) for a in ARMS}
    ci_rows = df["ci"].to_numpy(np.int64)

    # the realized captured holdout (9,941) is a strict subset of the split holdout
    # (10,000): 59 rows were dropped at capture (over-budget prompts etc.)
    holdout = {int(c) for c in FT.load_split(args.split_file)["sets"]["holdout"]["ci"]}
    assert set(ci_rows.tolist()) <= holdout, "CSV ci not a subset of the split holdout set"

    depth_of = _manifest_depths(args.manifest_dir)
    depths_row = np.asarray([depth_of[int(c)] for c in ci_rows], dtype=np.int64)
    log.info("loaded %d holdout rows + manifest depths in %.1fs", len(df), time.time() - t0)

    # predictions + targets (same row order as percontext npz / CSV — assert it)
    yh = np.load(args.y_holdout_dir / f"L{LAYER}.npz")
    assert (yh["ci"] == ci_rows).all(), "y_holdout ci misaligned with percontext CSV"
    y = yh["y16"].astype(np.float64)
    preds: dict[str, np.ndarray] = {}
    fingerprints: dict[str, str] = {"y_holdout": str(yh["fingerprint"])}
    for arm in ARMS:
        pz = np.load(args.pred16_dir / f"{arm}_L{LAYER}_{FITTER}.npz")
        assert (pz["ci"] == ci_rows).all(), f"pred16 ci misaligned ({arm})"
        preds[arm] = pz["pred16"].astype(np.float64)
        fingerprints[f"pred16_{arm}"] = str(pz["fingerprint"])

    coarse_check = _coarse_band_crosscheck(
        depths_row,
        preds,
        y,
        {
            "bare": DEFAULT_COARSE_BARE,
            "prefix": DEFAULT_COARSE_PARENT,
            "context": DEFAULT_COARSE_PARENT,
        },
    )
    log.info("coarse-band R^2 cross-check vs committed depth_contrasts.json: PASS (9 cells)")

    labels, lab_idx = _strata_labels(depths_row, args.min_stratum_n)
    rng = np.random.default_rng(args.seed)
    strata: dict[str, dict] = {}
    for si, lab in enumerate(labels):
        strata[lab] = _stratum_stats(lab_idx == si, nerr, preds, y, args.n_boot, rng)

    n_total = sum(s["n"] for s in strata.values())
    assert n_total == len(df) == 9941, (n_total, len(df))

    agg = _stratum_stats(np.ones(len(df), dtype=bool), nerr, preds, y, args.n_boot, rng)
    assert round(agg["frac_bare_worse_than_prefix"], 3) == COMMITTED_FRAC_BARE_WORSE_PREFIX, agg
    assert round(agg["frac_bare_worse_than_context"], 3) == COMMITTED_FRAC_BARE_WORSE_CONTEXT, agg

    # crossover read: a stratum where prefix beats bare on the majority of contexts
    # (dominance CI above 0.5), or where mean/R^2 ordering flips
    crossover = {
        "dominance_majority": [
            lab for lab in labels if strata[lab]["frac_bare_worse_than_prefix_ci"][0] > 0.5
        ],
        "dominance_point_above_half": [
            lab for lab in labels if strata[lab]["frac_bare_worse_than_prefix"] > 0.5
        ],
        "mean_nerr_flip": [
            lab for lab in labels if strata[lab]["nerr_bare_mean"] > strata[lab]["nerr_prefix_mean"]
        ],
        "median_nerr_flip": [
            lab
            for lab in labels
            if strata[lab]["nerr_bare_median"] > strata[lab]["nerr_prefix_median"]
        ],
        "r2_flip": [lab for lab in labels if strata[lab]["r2_prefix"] > strata[lab]["r2_bare"]],
    }

    doc = {
        "meta": {
            "script": "scripts/issue1738_depth_rebin.py",
            "git_commit": _git_commit(),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": sys.version.split()[0],
            "inputs": {
                "percontext_csv": str(args.percontext_csv),
                "manifest_dir": str(args.manifest_dir),
                "split_file": str(args.split_file),
                "pred16_dir": str(args.pred16_dir),
                "y_holdout_dir": str(args.y_holdout_dir),
                "fingerprints": fingerprints,
            },
            "n_boot": args.n_boot,
            "seed": args.seed,
            "min_stratum_n": args.min_stratum_n,
            "layer": LAYER,
            "fitter": FITTER,
            "r2_convention": "subset-mean denominator (F._recon_point / PR._pooled_r2)",
            "depth_definition": "exact user-turn count from sampling-manifest 'depth' field",
        },
        "strata_order": labels,
        "strata": strata,
        "aggregate": agg,
        "crossover": crossover,
        "coarse_band_crosscheck": coarse_check,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    GG.N1M._atomic_write_json(args.out_json, doc)
    log.info("wrote %s (%.1fs total)", args.out_json, time.time() - t0)

    # compact stdout table
    hdr = (
        f"{'depth':>6} {'n':>5} {'nerr_b':>7} {'nerr_p':>7} {'nerr_c':>7} "
        f"{'P(b>p)':>7} {'ci':>15} {'R2_b':>6} {'R2_p':>6} {'R2_c':>6}"
    )
    print(hdr)
    for lab in [*labels, "ALL"]:
        s = agg if lab == "ALL" else strata[lab]
        lo, hi = s["frac_bare_worse_than_prefix_ci"]
        print(
            f"{lab:>6} {s['n']:>5} {s['nerr_bare_mean']:>7.3f} {s['nerr_prefix_mean']:>7.3f} "
            f"{s['nerr_context_mean']:>7.3f} {s['frac_bare_worse_than_prefix']:>7.3f} "
            f"[{lo:.3f}, {hi:.3f}] {s['r2_bare']:>6.3f} {s['r2_prefix']:>6.3f} "
            f"{s['r2_context']:>6.3f}"
        )

    if not args.no_figure:
        _figure(args, labels, strata, agg, doc["meta"])


def _figure(args, labels: list[str], strata: dict, agg: dict, meta: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role, set_paper_style

    set_paper_style()
    colors = {
        "bare": paper_palette_role("primary"),
        "prefix": paper_palette_role("accent"),
        "context": paper_palette_role("baseline"),
    }
    xs = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained")

    for arm in ARMS:
        ax1.plot(
            xs,
            [strata[lab][f"r2_{arm}"] for lab in labels],
            marker="o",
            color=colors[arm],
            label=arm,
        )
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("conversation depth (exact user turns)")
    ax1.set_ylabel("holdout R² (subset-mean denom.)")
    ax1.legend(fontsize=7)

    for key, color, label in (
        ("frac_bare_worse_than_prefix", colors["prefix"], "bare worse than prefix"),
        ("frac_bare_worse_than_context", colors["context"], "bare worse than context"),
    ):
        vals = np.asarray([strata[lab][key] for lab in labels])
        los = np.asarray([strata[lab][f"{key}_ci"][0] for lab in labels])
        his = np.asarray([strata[lab][f"{key}_ci"][1] for lab in labels])
        # xerr/yerr take NON-NEGATIVE offsets (gotchas #547/#1335)
        yerr = np.stack([np.maximum(0, vals - los), np.maximum(0, his - vals)])
        ax2.errorbar(xs, vals, yerr=yerr, marker="o", capsize=2, color=color, label=label)
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("conversation depth (exact user turns)")
    ax2.set_ylabel("fraction of contexts")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7)

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / "depth_rebin_bare_vs_prefix.png"
    fig.savefig(png, dpi=200)
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)
    sidecar = {
        "source": str(args.out_json),
        "git_commit": meta["git_commit"],
        "generated_utc": meta["generated_utc"],
        "n_per_stratum": {lab: strata[lab]["n"] for lab in labels},
        "aggregate_n": agg["n"],
    }
    (fig_dir / "depth_rebin_bare_vs_prefix.meta.json").write_text(json.dumps(sidecar, indent=1))
    log.info("wrote %s (+pdf, +meta.json)", png)


if __name__ == "__main__":
    GG.logging.basicConfig(level=GG.logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main()
