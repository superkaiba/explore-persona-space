"""Issue #779 ctxansviz — VM-side figure rendering over the pod export.

Renders the eight paper-grade figures of the ctxansviz inline round (task #779
markers v268/v271/v272) from the HF export produced by
``scripts/issue779_ctxansviz_pod.py`` (phases P7 export / P8 dim / P9 xlayer).
Pure render leg: every embedding, cluster label, metric, TF-IDF term, spectrum
and CKA grid is read from the export — nothing is (re)fit here.

Usage (smoke, then full is a re-point of --export-prefix/--local-dir/--tag):

    uv run python scripts/issue779_ctxansviz_figures.py \
        --export-prefix issue779_monitoring/ctxansviz-smoke \
        --local-dir data/issue_779/ctxansviz_dl/smoke --tag smoke

Outputs ``figures/issue_779/ctxansviz_<tag>_*.{png,pdf,meta.json}`` via
``savefig_paper`` (commit-tagged sidecars).

Export schema (observed on the verified smoke export, revision
ac3ac0187b328da5cc177847311f1137e9512bc6):
  coords.npz          ci, umap_{cx,vx,vhat} (n,2), pca2_{cx,vx,vhat} (n,2),
                      kmeans_{cx,vx} (n,), hdbscan_cx (n,), metric_names (9,),
                      metrics (n,9); rows align positionally with row_meta.
  judged.npz          pca2_ctx, umap_ctx, kmeans_cx, hdbscan_cx, dv, [umap_t1]
  dim_spectra.npz     evals_{cx,vx,vhat} (3584,), cca_corrs_cx_vx
  dim_id_estimates.jsonl  rows: space, n, resample, twonn{id}, lb_mle{"10","20"},
                      corr_dim{id}, local_pca{id_median}, ambient_dim
  xlayer_cka.npz      cka6/cka6_sub/cka6_null (6,6), labels6, cka28, layers28
  xlayer_cosine_stats.json  pairs{"cx14~cx19": {raw_full, centered_sub,
                      null_raw_sub, null_centered_sub}} (stats: mean/median/
                      p2_5/p97_5/sd/n), adjacent_layer_curve_28
  cluster_stats.json  kmeans_cx/kmeans_vx/hdbscan_cx tables (top_tfidf_terms)
  meta.json           producer provenance + disclosures
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + HF/API env BEFORE hf/matplotlib imports

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
SEED = 42
ERROR_CMAP = "viridis"  # map-error sequential colormap (one meaning: cosine(pred, true))
JUDGED_CMAP = "plasma"  # judged-DV sequential colormap (one meaning: 0-100 judged score)
BULK_GRAY = "#c8c8c8"

REQUIRED_FILES = (
    "coords.npz",
    "cluster_stats.json",
    "dim_spectra.npz",
    "dim_id_estimates.jsonl",
    "xlayer_cka.npz",
    "xlayer_cosine_stats.json",
    "meta.json",
)
OPTIONAL_FILES = ("judged.npz", "judged_meta.jsonl", "pca_model.npz", "dim_summary.json")

SPACE_LABELS = {
    "cx": "contexts",
    "vx": "true answers",
    "vhat": "predicted answers",
    "judged_cx": "judged contexts",
}
ESTIMATOR_LABELS = {
    "twonn": "TwoNN",
    "lb_mle_10": "MLE (k=10, MacKay–Ghahramani)",
    "lb_mle_20": "MLE (k=20, MacKay–Ghahramani)",
    "corr_dim": "correlation dimension",
    "local_pca": "local PCA",
}


def role_colors() -> dict[str, str]:
    """Pinned one-color-one-meaning mapping shared by every figure of the round."""
    return {
        "cx": paper_palette_role("primary"),  # contexts
        "vx": paper_palette_role("baseline"),  # true answers
        "vhat": paper_palette_role("control"),  # predicted answers
    }


def ensure_export(export_prefix: str, local_dir: Path) -> Path:
    """Local-first staging of the export dir; scoped HF fetch when absent.

    Uses a scoped listing of the export prefix (the data repo holds ~1M files —
    an unscoped ``snapshot_download`` listing hangs) and pins one revision for
    every file of the paired set. Fails loud when required files are still
    missing afterward. Returns the directory holding the export.
    """
    dest = local_dir / export_prefix
    if not all((dest / f).exists() for f in REQUIRED_FILES):
        from huggingface_hub import HfApi, hf_hub_download

        from explore_persona_space.orchestrate import hub

        api = HfApi()
        sha = hub.retry_transient(
            lambda: api.repo_info(HF_DATA_REPO, repo_type="dataset").sha,
            what="repo_info revision pin",
        )
        files = hub.list_hf_files_under_path(
            api, HF_DATA_REPO, export_prefix, repo_type="dataset", revision=sha
        )
        if not files:
            raise RuntimeError(f"no files under {export_prefix} at revision {sha}")
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted(files):
            p = hub.retry_transient(
                lambda f=f: hf_hub_download(HF_DATA_REPO, f, repo_type="dataset", revision=sha),
                what=f"hf_hub_download {Path(f).name}",
            )
            shutil.copyfile(p, dest / Path(f).name)
        (dest / "_download_meta.json").write_text(
            json.dumps({"revision": sha, "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")}),
            encoding="utf-8",
        )
    missing = [f for f in REQUIRED_FILES if not (dest / f).exists()]
    if missing:
        raise RuntimeError(f"export at {dest} missing required files: {missing}")
    if not sorted(dest.glob("row_meta_*.jsonl")):
        raise RuntimeError(f"export at {dest} has no row_meta_*.jsonl shards")
    return dest


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_row_meta(export: Path) -> list[dict]:
    """All row_meta rows in shard order (positional alignment with coords.npz)."""
    rows: list[dict] = []
    for part in sorted(export.glob("row_meta_*.jsonl")):
        rows.extend(iter_jsonl(part))
    return rows


def _sub(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    return np.sort(rng.choice(n, size=min(n, k), replace=False))


def fig_joint_embedding(d: dict, tag: str, colors: dict[str, str]) -> None:
    z = d["coords"]
    n = z["umap_cx"].shape[0]
    idx = _sub(np.random.default_rng(SEED), n, 20_000)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
    panels = (
        ("pca2", "PC 1", "PC 2", "PCA (first two components)"),
        ("umap", "UMAP 1", "UMAP 2", "UMAP"),
    )
    for ax, (stem, xl, yl, title) in zip(axes, panels):
        # draw order cx -> vhat -> vx keeps true answers visible on top of the
        # heavily-overlapping predicted-answer cloud (overlap is the finding).
        for key, label in (
            ("cx", "contexts"),
            ("vhat", "predicted answers"),
            ("vx", "true answers"),
        ):
            pts = z[f"{stem}_{key}"][idx]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=3,
                alpha=0.25,
                color=colors[key],
                label=label,
                rasterized=True,
                edgecolors="none",
            )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, loc="left")
    axes[1].set_title(f"UMAP — {len(idx):,} of {n:,} rows shown", loc="left")
    leg = axes[0].legend(markerscale=4, loc="best")
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    savefig_paper(fig, f"issue_779/ctxansviz_{tag}_joint_embedding", embed_data=False)
    plt.close(fig)


def fig_arrows(d: dict, tag: str, colors: dict[str, str]) -> None:
    z = d["coords"]
    n = z["umap_cx"].shape[0]
    idx = _sub(np.random.default_rng(SEED), n, 800)
    c, a = z["umap_cx"][idx], z["umap_vx"][idx]
    fig, ax = plt.subplots(figsize=(6.5, 5.2), layout="constrained")
    ax.quiver(
        c[:, 0],
        c[:, 1],
        a[:, 0] - c[:, 0],
        a[:, 1] - c[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0016,
        color="#9a9a9a",
        alpha=0.35,
    )
    ax.scatter(
        c[:, 0], c[:, 1], s=8, color=colors["cx"], label="contexts", edgecolors="none", zorder=3
    )
    ax.scatter(
        a[:, 0], a[:, 1], s=8, color=colors["vx"], label="true answers", edgecolors="none", zorder=3
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        f"context → true-answer pairings ({len(idx)} random pairs, seed {SEED})", loc="left"
    )
    ax.legend(markerscale=2)
    savefig_paper(fig, f"issue_779/ctxansviz_{tag}_arrows", embed_data=False)
    plt.close(fig)


def fig_map_error(d: dict, tag: str) -> None:
    z = d["coords"]
    names = [str(x) for x in z["metric_names"]]
    cos = z["metrics"][:, names.index("cos_vhat_vx")]
    n = z["umap_cx"].shape[0]
    idx = _sub(np.random.default_rng(SEED), n, 150_000)
    fig, ax = plt.subplots(figsize=(6.5, 5.2), layout="constrained")
    sc = ax.scatter(
        z["umap_cx"][idx, 0],
        z["umap_cx"][idx, 1],
        c=cos[idx],
        s=3,
        alpha=0.6,
        cmap=ERROR_CMAP,
        rasterized=True,
        edgecolors="none",
    )
    fig.colorbar(sc, ax=ax, label="cosine of predicted vs true answer vector")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"contexts colored by map accuracy (n={len(idx):,} of {n:,})", loc="left")
    savefig_paper(fig, f"issue_779/ctxansviz_{tag}_map_error", embed_data=False)
    plt.close(fig)


def fig_judged_overlay(d: dict, tag: str) -> None:
    if d["judged"] is None:
        raise RuntimeError("judged.npz absent — judged_overlay cannot render")
    z, j = d["coords"], d["judged"]
    n = z["umap_cx"].shape[0]
    idx = _sub(np.random.default_rng(SEED), n, 100_000)
    dv = j["dv"]
    ok = ~np.isnan(dv)
    fig, ax = plt.subplots(figsize=(6.5, 5.2), layout="constrained")
    ax.scatter(
        z["umap_cx"][idx, 0],
        z["umap_cx"][idx, 1],
        s=3,
        alpha=0.3,
        color=BULK_GRAY,
        rasterized=True,
        edgecolors="none",
        label="unjudged contexts",
    )
    sc = ax.scatter(
        j["umap_ctx"][ok, 0],
        j["umap_ctx"][ok, 1],
        c=dv[ok],
        s=10,
        cmap=JUDGED_CMAP,
        vmin=0,
        vmax=100,
        edgecolors="none",
        zorder=3,
    )
    fig.colorbar(sc, ax=ax, label="judged sycophancy score (0–100)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"judged contexts (n={int(ok.sum()):,}) over the unjudged bulk", loc="left")
    ax.legend(markerscale=3, loc="best")
    savefig_paper(fig, f"issue_779/ctxansviz_{tag}_judged_overlay", embed_data=False)
    plt.close(fig)


def fig_clusters(d: dict, tag: str) -> None:
    z = d["coords"]
    labels = z["kmeans_cx"]
    n = labels.shape[0]
    idx = _sub(np.random.default_rng(SEED), n, 150_000)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(6.5, 5.2), layout="constrained")
    ax.scatter(
        z["umap_cx"][idx, 0],
        z["umap_cx"][idx, 1],
        c=[cmap(int(c) % 20) for c in labels[idx]],
        s=3,
        alpha=0.5,
        rasterized=True,
        edgecolors="none",
    )
    table = sorted(d["cluster_stats"]["kmeans_cx"], key=lambda r: -r["n"])[:10]
    for row in table:
        cid = row["cluster"]
        sel = labels == cid
        if not sel.any() or not row["top_tfidf_terms"]:
            continue
        pos = np.median(z["umap_cx"][sel], axis=0)
        ax.text(
            pos[0],
            pos[1],
            str(row["top_tfidf_terms"][0]),
            fontsize=8,
            ha="center",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.2},
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        f"context clusters (KMeans, {len(d['cluster_stats']['kmeans_cx'])} clusters; "
        "10 largest labeled by top TF-IDF term)",
        loc="left",
    )
    savefig_paper(fig, f"issue_779/ctxansviz_{tag}_clusters", embed_data=False)
    plt.close(fig)


def fig_dim_spectra(d: dict, tag: str, colors: dict[str, str]) -> None:
    z = d["dim_spectra"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), layout="constrained")
    for key, label in (("cx", "contexts"), ("vx", "true answers"), ("vhat", "predicted answers")):
        ev = np.maximum(z[f"evals_{key}"], 0.0)
        rank = np.arange(1, ev.size + 1)
        axes[0].loglog(rank, np.maximum(ev, 1e-300), color=colors[key], label=label, lw=1.4)
        evr = np.cumsum(ev) / max(ev.sum(), 1e-300)
        axes[1].semilogx(rank, evr, color=colors[key], label=label, lw=1.4)
    axes[0].set_xlabel("eigenvalue rank")
    axes[0].set_ylabel("covariance eigenvalue")
    axes[0].set_title("eigenvalue spectrum (log–log)", loc="left")
    axes[0].legend()
    axes[1].set_xlabel("number of components")
    axes[1].set_ylabel("cumulative explained variance")
    axes[1].set_title("cumulative explained variance", loc="left")
    cca = z["cca_corrs_cx_vx"]
    axes[2].plot(np.arange(1, cca.size + 1), cca, color=paper_palette_role("accent"), lw=1.4)
    axes[2].set_xlabel("canonical component index")
    axes[2].set_ylabel("canonical correlation")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("context ↔ true-answer CCA spectrum", loc="left")
    savefig_paper(fig, f"issue_779/ctxansviz_{tag}_dim_spectra")
    plt.close(fig)


def _id_value(row: dict, est: str) -> float | None:
    if est.startswith("lb_mle_"):
        v = row["lb_mle"].get(est.removeprefix("lb_mle_"))
    elif est == "local_pca":
        v = row["local_pca"].get("id_median")
    else:
        v = row[est].get("id")
    if isinstance(v, dict):
        # Levina–Bickel rows carry two globals; MacKay–Ghahramani (mean of
        # inverse local MLEs, inverted) is the recommended corrected form.
        v = v.get("id_mackay_ghahramani", v.get("id"))
    return float(v) if v is not None else None


def fig_intrinsic_dim(d: dict, tag: str) -> None:
    rows = d["dim_id"]
    spaces = [s for s in ("cx", "vx", "vhat", "judged_cx") if any(r["space"] == s for r in rows)]
    ests = list(ESTIMATOR_LABELS)
    est_colors = dict(zip(ests, paper_palette(len(ests))))
    plotted: set[str] = set()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), layout="constrained")
    for ax, space in zip(axes.flat, spaces):
        srows = [r for r in rows if r["space"] == space]
        for est in ests:
            byn: dict[int, list[float]] = {}
            for r in srows:
                v = _id_value(r, est)
                if v is not None:
                    byn.setdefault(int(r["n"]), []).append(v)
            if not byn:
                continue
            plotted.add(est)
            ns = sorted(byn)
            med = [float(np.median(byn[k])) for k in ns]
            lo = [float(np.percentile(byn[k], 2.5)) for k in ns]
            hi = [float(np.percentile(byn[k], 97.5)) for k in ns]
            ax.plot(ns, med, "o-", color=est_colors[est], label=ESTIMATOR_LABELS[est], lw=1.2, ms=4)
            ax.fill_between(ns, lo, hi, color=est_colors[est], alpha=0.18, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("sample size n")
        ax.set_ylabel("intrinsic dimension estimate")
        ambient = next((r["ambient_dim"] for r in srows), None)
        ax.set_title(f"{SPACE_LABELS[space]} (ambient dim {ambient})", loc="left")
    for ax in axes.flat[len(spaces) :]:
        ax.set_visible(False)
    missing = [e for e in ests if e not in plotted]
    if missing:
        raise RuntimeError(f"intrinsic-dim estimators produced no plottable values: {missing}")
    axes.flat[0].legend(fontsize=8)
    savefig_paper(fig, f"issue_779/ctxansviz_{tag}_intrinsic_dim")
    plt.close(fig)


def _pair_label(pair: str) -> str:
    def obj(o: str) -> str:
        kind = "ctx" if o.startswith("cx") else "ans"
        return f"{kind} L{o[2:]}"

    a, b = pair.split("~")
    return f"{obj(a)} ↔ {obj(b)}"


def fig_xlayer(d: dict, tag: str) -> None:
    z = d["xlayer"]
    stats = d["xlayer_cos"]
    labels6 = [_pair_label(f"{x}~{x}").split(" ↔ ")[0] for x in (str(v) for v in z["labels6"])]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), layout="constrained")

    im0 = axes[0].imshow(z["cka6"], vmin=0, vmax=1, cmap="viridis")
    axes[0].set_xticks(range(6), labels6, rotation=30, ha="right")
    axes[0].set_yticks(range(6), labels6)
    for i in range(6):
        for j in range(6):
            v = z["cka6"][i, j]
            axes[0].text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if v < 0.6 else "black",
            )
    axes[0].set_title("linear CKA (rotation-invariant)", loc="left")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    axes[1].set_title(f"context CKA, all layers (n={int(z['n_judged']):,})", loc="left")
    im1 = axes[1].imshow(z["cka28"], vmin=0, vmax=1, cmap="viridis")
    layers = [int(x) for x in z["layers28"]]
    ticks = list(range(0, len(layers), 4))
    axes[1].set_xticks(ticks, [layers[t] for t in ticks])
    axes[1].set_yticks(ticks, [layers[t] for t in ticks])
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("layer")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    pairs = list(stats["pairs"])
    xs = np.arange(len(pairs))
    for series, color, label, off in (
        ("centered_sub", paper_palette_role("primary"), "measured (mean-centered)", -0.12),
        ("null_centered_sub", paper_palette_role("neutral"), "shuffled-pairing null", 0.12),
    ):
        med = np.array([stats["pairs"][p][series]["median"] for p in pairs])
        lo = np.array([stats["pairs"][p][series]["p2_5"] for p in pairs])
        hi = np.array([stats["pairs"][p][series]["p97_5"] for p in pairs])
        axes[2].errorbar(
            xs + off,
            med,
            yerr=[med - lo, hi - med],
            fmt="o",
            color=color,
            label=label,
            ms=4,
            capsize=2,
            lw=1.1,
        )
    axes[2].axhline(0.0, color="#bbbbbb", lw=0.8)
    axes[2].set_xticks(xs, [_pair_label(p) for p in pairs], rotation=30, ha="right")
    axes[2].set_ylabel("per-row cosine (median, 2.5–97.5%)")
    axes[2].set_title("cross-layer per-row cosine vs shuffled-pairing null", loc="left")
    axes[2].legend(fontsize=8)
    savefig_paper(fig, f"issue_779/ctxansviz_{tag}_xlayer")
    plt.close(fig)


def load_all(export: Path) -> dict:
    d: dict = {
        "coords": np.load(export / "coords.npz", allow_pickle=False),
        "cluster_stats": json.loads((export / "cluster_stats.json").read_text("utf-8")),
        "dim_spectra": np.load(export / "dim_spectra.npz", allow_pickle=False),
        "dim_id": list(iter_jsonl(export / "dim_id_estimates.jsonl")),
        "xlayer": np.load(export / "xlayer_cka.npz", allow_pickle=False),
        "xlayer_cos": json.loads((export / "xlayer_cosine_stats.json").read_text("utf-8")),
        "meta": json.loads((export / "meta.json").read_text("utf-8")),
        "judged": None,
    }
    if (export / "judged.npz").exists():
        d["judged"] = np.load(export / "judged.npz", allow_pickle=False)
    n = d["coords"]["umap_cx"].shape[0]
    assert d["coords"]["metrics"].shape == (n, 9), d["coords"]["metrics"].shape
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    ap.add_argument(
        "--export-prefix", required=True, help="HF prefix, e.g. issue779_monitoring/ctxansviz-smoke"
    )
    ap.add_argument("--local-dir", required=True, type=Path)
    ap.add_argument("--tag", required=True, choices=("smoke", "full"))
    args = ap.parse_args()

    export = ensure_export(args.export_prefix, args.local_dir)
    d = load_all(export)
    set_paper_style("blog")
    colors = role_colors()
    print(
        f"[figures] export={export} n_rows={d['meta']['n_rows']} "
        f"n_judged={d['meta'].get('n_judged')} tag={args.tag}"
    )

    t0 = time.time()
    for i, (name, fn) in enumerate(
        (
            ("joint_embedding", lambda: fig_joint_embedding(d, args.tag, colors)),
            ("arrows", lambda: fig_arrows(d, args.tag, colors)),
            ("map_error", lambda: fig_map_error(d, args.tag)),
            ("judged_overlay", lambda: fig_judged_overlay(d, args.tag)),
            ("clusters", lambda: fig_clusters(d, args.tag)),
            ("dim_spectra", lambda: fig_dim_spectra(d, args.tag, colors)),
            ("intrinsic_dim", lambda: fig_intrinsic_dim(d, args.tag)),
            ("xlayer", lambda: fig_xlayer(d, args.tag)),
        ),
        1,
    ):
        fn()
        print(f"[figures] {i}/8 {name} elapsed={time.time() - t0:.1f}s", flush=True)
    print("[figures] done")


if __name__ == "__main__":
    main()
