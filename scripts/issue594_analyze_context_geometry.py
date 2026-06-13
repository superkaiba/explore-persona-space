#!/usr/bin/env python3
"""Issue #594 Phase 2: VM-side analysis of the context-vector geometry map.

Per plan v1 §3 Phase 2 + §14 binding addenda. Reads the Phase-1 tensors
(local dir or HF data repo) and computes, per layer:

1.  centered (global_mean) + raw cosine matrices (compute_cosine_matrix)
2.  PRIMARY DV: silhouette (precomputed 1-centered-cos) + LOO k-NN family
    purity (k=4, PINNED deterministic tie rule) over the 48 headline
    instances / 6 families
3.  label-permutation null, B=1000, draws SHARED across layers ->
    pointwise per-layer p AND the FWER-controlling max-over-layers null
4.  bootstrap-over-probes CIs; probe-split-half reliability on CENTERED
    vectors over the 48 headline instances (§14 item 1; raw alongside)
5.  length reads on the 48 headline instances with log1p(content-token
    length) (§14 item 2): length-only k-NN baseline, OLS residualized
    re-read vs fresh permutation null, Spearman(PC1..4, log-length)
6.  PCA spectra + scatters; UMAP grid {5,15,30}x{0.1,0.5} (seed 42);
    t-SNE perplexity {5,15,30} (seed 42); dendrograms at quartile layers;
    linear CKA LxL; outlier table; per-family decomposition;
    TF-IDF text-similarity family-purity baseline (§14 analyzer bullet)
7.  multiplicity + discordant-cell narration (§14 item 3) in the JSON
8.  OPTIONAL cross-pool module (plan v2 §4, follow-up
    probe-genre-generalization) behind ``--compare-tensors-from-hf``: joins
    this run's mean tensors with a second pool's (parent run) ON STORED
    INSTANCE IDS, per layer computes both 50x50 global_mean-centered cosine
    matrices (each pool centered on its own bank), upper-triangle Spearman +
    Pearson, Mantel permutation p (simultaneous row+col relabeling, B=1000,
    seed 42), per-instance cross-pool centered-vector cosine; writes
    ``cross_pool_comparison.json``, a per-layer Mantel-curve figure, and an
    overlay hero (new purity/silhouette depth curves over the parent's, read
    from ``--parent-metrics-json``).

Outputs: ``eval_results/issue_594/context_geometry_metrics.json`` (+
``per_layer/*.json`` with the per-layer cosine matrices, raw alongside
centered) and ``figures/issue_594/*`` via paper_plots conventions; the
``--eval-dir`` / ``--fig-dir`` flags route follow-up outputs to
``eval_results/issue_594/probe-genre-generalization`` etc. (plan v2 §4).

Usage (plan §8)::

    uv run --extra viz python scripts/issue594_analyze_context_geometry.py \\
        --tensors-from-hf issue594_context_geometry/analysis_tensors

    # or against a local extraction dir:
    uv run --extra viz python scripts/issue594_analyze_context_geometry.py \\
        --tensors-dir data/issue594/context_vectors

    NOTE: ``--extra viz`` is REQUIRED — umap-learn lives in the optional
    ``viz`` dependency group of pyproject.toml, not the default set (the
    plan §8 analysis command omits it; recorded as a launch-command
    correction in the implementation report).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue594_common import (  # noqa: E402
    BATTERY_PATH,
    HEADLINE_EXCLUDED_FAMILIES,
    HF_DATA_REPO,
    load_battery,
)
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_samples, silhouette_score  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.analysis.representation_shift import (  # noqa: E402
    compute_cosine_matrix,
)

load_dotenv()

logger = logging.getLogger("issue594_analyze")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FAMILY_ORDER = ["persona", "wildchat", "icl", "rephrase", "format", "behavior", "default"]
KNN_K = 4
UMAP_GRID = [(5, 0.1), (5, 0.5), (15, 0.1), (15, 0.5), (30, 0.1), (30, 0.5)]
TSNE_PERPLEXITIES = [5, 15, 30]
SEED = 42

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_594"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_594"

# §14 item 3 — fixed multiplicity sentence, emitted into the metrics JSON.
MULTIPLICITY_SENTENCE = (
    "An either-statistic headline over the two correlated co-primaries "
    "(silhouette, k-NN purity) carries effective alpha up to ~0.10; a mixed "
    "outcome (one beats its max-over-layers null, the other does not) is "
    "narrated as metric-discordant exploratory evidence — local neighborhood "
    "structure without compact global separation — never as unqualified "
    "'family structure exists'."
)


# ── Self-check (A11) ─────────────────────────────────────────────────────────


def selfcheck_cosine_matrix() -> None:
    """compute_cosine_matrix vs a hand-computed 3x3 case (plan A11)."""
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    got = compute_cosine_matrix(C, centering="none").numpy()
    s2 = 1.0 / np.sqrt(2.0)
    want = np.array([[1.0, 0.0, s2], [0.0, 1.0, s2], [s2, s2, 1.0]])
    assert np.allclose(got, want, atol=1e-6), got
    # global_mean: mean = (2/3, 2/3); centered rows (1/3,-2/3), (-2/3,1/3), (1/3,1/3)
    got_c = compute_cosine_matrix(C, centering="global_mean").numpy()
    a = np.array([1 / 3, -2 / 3])
    b = np.array([-2 / 3, 1 / 3])
    want_ab = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert abs(got_c[0, 1] - want_ab) < 1e-6, (got_c[0, 1], want_ab)


# ── Data loading ─────────────────────────────────────────────────────────────


def download_from_hf(prefix: str, cache_dir: Path, repo: str = HF_DATA_REPO) -> Path:
    """Fetch the analysis tensors from an HF dataset repo (default: data repo).

    ``repo`` is overridable via ``--hf-repo`` so the quota-403 overflow
    fallback the extraction script can take (``repo_used`` in its results
    sentinel + ``extraction_manifest.json['upload']['repo']``) stays reachable
    by Phase 2. Uses list_repo_files + per-file hf_hub_download (NOT
    snapshot_download allow_patterns — silently returns 0 files for prefixes
    in the truncated siblings tail on large repos).
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(repo, repo_type="dataset") if f.startswith(prefix + "/")]
    if not files:
        raise RuntimeError(f"no files under {prefix}/ on {repo}")
    for f in files:
        hf_hub_download(repo, f, repo_type="dataset", local_dir=str(cache_dir))
    logger.info("Downloaded %d files from %s/%s", len(files), repo, prefix)
    return cache_dir / prefix


def load_tensors(tensors_dir: Path) -> dict:
    """Load mean tensor, per-probe tensors, and the extraction manifest."""
    blob = torch.load(tensors_dir / "context_vectors_mean.pt", weights_only=True)
    with open(tensors_dir / "extraction_manifest.json") as f:
        manifest = json.load(f)
    ids = blob["instance_ids"]
    per_probe = {}
    for iid in ids:
        pp = torch.load(tensors_dir / "per_probe" / f"{iid}.pt", weights_only=True)
        assert pp["probe_pool_hash"] == blob["probe_pool_hash"], iid
        per_probe[iid] = pp
    mean = blob["tensor"].float().numpy()  # (N, L, H)
    n, n_layers, hidden = mean.shape
    assert n == len(ids) and n_layers == manifest["n_layers"], mean.shape
    logger.info("Loaded tensors: N=%d, L=%d, H=%d", n, n_layers, hidden)
    return {
        "mean": mean,
        "ids": ids,
        "families": blob["families"],
        "per_probe": per_probe,
        "manifest": manifest,
        "n_layers": n_layers,
        "hidden": hidden,
    }


# ── Metric primitives ────────────────────────────────────────────────────────


def center(x: np.ndarray) -> np.ndarray:
    """Globally mean-center over the instance axis (bank centering, #536)."""
    return x - x.mean(axis=0, keepdims=True)


def cosine_dist(x: np.ndarray, centering: str = "global_mean") -> np.ndarray:
    """1 - cosine matrix via the canonical compute_cosine_matrix helper."""
    cos = compute_cosine_matrix(torch.from_numpy(x).float(), centering=centering).numpy()
    d = 1.0 - cos
    np.fill_diagonal(d, 0.0)
    return np.clip(d, 0.0, None)


def knn_neighbor_order(d: np.ndarray) -> np.ndarray:
    """Per-row neighbor index order, self excluded.

    PINNED deterministic tie rule part 1: equal distances break by LOWER
    instance index (stable argsort over (distance, index)).
    """
    d = d.copy()
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")


def knn_purity(order: np.ndarray, labels: np.ndarray, k: int = KNN_K) -> float:
    """LOO k-NN family purity with the PINNED deterministic tie rule.

    Vote = majority family among the k nearest neighbors; a vote TIE breaks
    to the family of the NEAREST neighbor whose family is in the tied set
    (tie rule part 2). Purity = fraction of instances whose predicted family
    equals their own.
    """
    n = order.shape[0]
    correct = 0
    for i in range(n):
        neigh = order[i, :k]
        votes = Counter(labels[j] for j in neigh)
        top = max(votes.values())
        tied = {fam for fam, c in votes.items() if c == top}
        if len(tied) == 1:
            pred = next(iter(tied))
        else:
            pred = next(labels[j] for j in neigh if labels[j] in tied)
        correct += int(pred == labels[i])
    return correct / n


def per_family_purity(order: np.ndarray, labels: np.ndarray, k: int = KNN_K) -> dict[str, float]:
    """k-NN purity decomposed per family (same vote + tie rule)."""
    n = order.shape[0]
    hits: dict[str, list[int]] = {}
    for i in range(n):
        neigh = order[i, :k]
        votes = Counter(labels[j] for j in neigh)
        top = max(votes.values())
        tied = {fam for fam, c in votes.items() if c == top}
        pred = (
            next(iter(tied))
            if len(tied) == 1
            else next(labels[j] for j in neigh if labels[j] in tied)
        )
        hits.setdefault(labels[i], []).append(int(pred == labels[i]))
    return {fam: float(np.mean(v)) for fam, v in hits.items()}


def layer_stats(x: np.ndarray, labels: np.ndarray, centering: str) -> tuple[float, float]:
    """(silhouette, knn purity) for one layer's (n, H) matrix."""
    d = cosine_dist(x, centering=centering)
    sil = float(silhouette_score(d, labels, metric="precomputed"))
    pur = knn_purity(knn_neighbor_order(d), labels)
    return sil, pur


def perm_stats(d: np.ndarray, order: np.ndarray, perms: np.ndarray, labels: np.ndarray):
    """silhouette + purity for every label permutation (geometry fixed)."""
    sils = np.empty(len(perms))
    purs = np.empty(len(perms))
    for b, perm in enumerate(perms):
        pl = labels[perm]
        sils[b] = silhouette_score(d, pl, metric="precomputed")
        purs[b] = knn_purity(order, pl)
    return sils, purs


def residualize(x: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """OLS-residualize each coordinate of x (n, H) on [1, covariate]."""
    z = np.stack([np.ones_like(covariate), covariate], axis=1)
    beta, *_ = np.linalg.lstsq(z, x, rcond=None)
    return x - z @ beta


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA (Gram form, feature-centered; Kornblith et al. 1905.00414)."""
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    k = xc @ xc.T
    l_ = yc @ yc.T
    hsic = float((k * l_).sum())
    denom = float(np.sqrt((k * k).sum() * (l_ * l_).sum()))
    return hsic / denom if denom > 0 else float("nan")


def quartile_layers(n_layers: int) -> list[int]:
    """Depth-band figure layers; exactly [7, 14, 21, 27] at L=28."""
    return sorted({n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1})


def jsonable(o):
    """Recursively convert numpy/torch scalars + arrays for json.dump."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ── Analysis sections ────────────────────────────────────────────────────────


def headline_curves(mean_h: np.ndarray, labels: np.ndarray, perms: np.ndarray) -> dict:
    """Observed per-layer sil/purity (centered + raw) + shared-draw perm null.

    mean_h: (48, L, H) headline-instance probe-mean vectors. Centering bank
    for the headline DV = the 48 headline instances themselves.
    """
    n_layers = mean_h.shape[1]
    out: dict = {
        "silhouette": [],
        "purity": [],
        "silhouette_raw": [],
        "purity_raw": [],
        "per_family_purity": [],
        "per_family_silhouette": [],
        "null_sil": np.empty((len(perms), n_layers)),
        "null_pur": np.empty((len(perms), n_layers)),
    }
    for li in range(n_layers):
        x = mean_h[:, li, :]
        d = cosine_dist(x, "global_mean")
        order = knn_neighbor_order(d)
        out["silhouette"].append(float(silhouette_score(d, labels, metric="precomputed")))
        out["purity"].append(knn_purity(order, labels))
        sil_r, pur_r = layer_stats(x, labels, "none")
        out["silhouette_raw"].append(sil_r)
        out["purity_raw"].append(pur_r)
        out["per_family_purity"].append(per_family_purity(order, labels))
        samp = silhouette_samples(d, labels, metric="precomputed")
        out["per_family_silhouette"].append(
            {fam: float(samp[labels == fam].mean()) for fam in set(labels)}
        )
        sils, purs = perm_stats(d, order, perms, labels)
        out["null_sil"][:, li] = sils
        out["null_pur"][:, li] = purs
        if li % 7 == 0:
            logger.info("headline layer %d/%d done", li + 1, n_layers)
    return out


def max_over_layers_summary(obs: list[float], null: np.ndarray) -> dict:
    """Max-statistic FWER summary + pointwise p-values (Nichols & Holmes)."""
    b = null.shape[0]
    null_max = null.max(axis=1)
    obs_arr = np.asarray(obs)
    return {
        "observed_max": float(obs_arr.max()),
        "argmax_layer": int(obs_arr.argmax()),
        "null_max_p95": float(np.percentile(null_max, 95)),
        "p_fwer": float((1 + (null_max >= obs_arr.max()).sum()) / (b + 1)),
        "pointwise_p": [
            float((1 + (null[:, li] >= obs_arr[li]).sum()) / (b + 1)) for li in range(null.shape[1])
        ],
        "null_mean_per_layer": null.mean(axis=0).tolist(),
        "null_p95_per_layer": np.percentile(null, 95, axis=0).tolist(),
        "passes": bool(obs_arr.max() > np.percentile(null_max, 95)),
    }


def bootstrap_over_probes(
    per_probe_h: np.ndarray, labels: np.ndarray, n_boot: int, rng: np.random.Generator
) -> dict:
    """Probe-resample CIs for sil/purity per layer (centered headline space).

    per_probe_h: (48, P, L, H) fp32.
    """
    _n, p, n_layers, _ = per_probe_h.shape
    sils = np.empty((n_boot, n_layers))
    purs = np.empty((n_boot, n_layers))
    for b in range(n_boot):
        idx = rng.integers(0, p, size=p)
        m = per_probe_h[:, idx, :, :].mean(axis=1)  # (48, L, H)
        for li in range(n_layers):
            sils[b, li], purs[b, li] = layer_stats(m[:, li, :], labels, "global_mean")
        if (b + 1) % 50 == 0:
            logger.info("bootstrap %d/%d done", b + 1, n_boot)

    def pct(a):
        return np.percentile(a, 2.5, axis=0).tolist(), np.percentile(a, 97.5, axis=0).tolist()

    sil_lo, sil_hi = pct(sils)
    pur_lo, pur_hi = pct(purs)
    return {
        "n_boot": n_boot,
        "sil_ci_lo": sil_lo,
        "sil_ci_hi": sil_hi,
        "pur_ci_lo": pur_lo,
        "pur_ci_hi": pur_hi,
    }


def split_half(per_probe_h: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict:
    """Probe-split-half reliability (§14 item 1): CENTERED space primary.

    Cross-half same-instance cosine on globally-mean-centered vectors over
    the 48 headline instances, per layer; raw-space alongside (vacuity-prone
    companion — raw same-instance cosines sit near ceiling, #536). Also
    per-half metric agreement. Kill criterion §9(b) evaluates the CENTERED
    median.
    """
    _n, p, n_layers, _ = per_probe_h.shape
    perm = rng.permutation(p)
    half_a, half_b = perm[: p // 2], perm[p // 2 :]
    med_centered, med_raw = [], []
    agree = {"sil_a": [], "sil_b": [], "pur_a": [], "pur_b": []}
    for li in range(n_layers):
        va = per_probe_h[:, half_a, li, :].mean(axis=1)
        vb = per_probe_h[:, half_b, li, :].mean(axis=1)
        for tag, mat_a, mat_b in (("centered", center(va), center(vb)), ("raw", va, vb)):
            num = (mat_a * mat_b).sum(axis=1)
            den = np.linalg.norm(mat_a, axis=1) * np.linalg.norm(mat_b, axis=1)
            cos = num / np.clip(den, 1e-12, None)
            (med_centered if tag == "centered" else med_raw).append(float(np.median(cos)))
        sa, pa = layer_stats(va, labels, "global_mean")
        sb, pb = layer_stats(vb, labels, "global_mean")
        agree["sil_a"].append(sa)
        agree["sil_b"].append(sb)
        agree["pur_a"].append(pa)
        agree["pur_b"].append(pb)
    kill = all(m < 0.9 for m in med_centered)
    return {
        "half_sizes": [len(half_a), len(half_b)],
        "median_crosshalf_cos_centered": med_centered,
        "median_crosshalf_cos_raw": med_raw,
        "metric_agreement": agree,
        "max_abs_sil_disagreement": float(
            np.abs(np.array(agree["sil_a"]) - np.array(agree["sil_b"])).max()
        ),
        "kill_criterion_9b_triggered": bool(kill),
    }


def length_reads(
    mean_h: np.ndarray,
    labels: np.ndarray,
    log_len: np.ndarray,
    perms_fresh: np.ndarray,
    headline_ids: list[str],
) -> dict:
    """§14 item 2: all length reads on the 48 headline instances, log1p lens."""
    _n, n_layers, _ = mean_h.shape
    # (i) length-only 1-D k-NN baseline + its permutation null.
    d_len = np.abs(log_len[:, None] - log_len[None, :])
    order_len = knn_neighbor_order(d_len)
    len_only_pur = knn_purity(order_len, labels)
    null_len = np.array([knn_purity(order_len, labels[p]) for p in perms_fresh])
    # (ii) residualized re-read vs FRESH permutation null.
    res = {
        "silhouette": [],
        "purity": [],
        "null_sil": np.empty((len(perms_fresh), n_layers)),
        "null_pur": np.empty((len(perms_fresh), n_layers)),
    }
    spearman_pcs = []
    for li in range(n_layers):
        x = residualize(mean_h[:, li, :], log_len)
        x = center(x)  # re-center per §14 (residuals are near-centered already)
        d = cosine_dist(x, "none")  # already centered above
        order = knn_neighbor_order(d)
        res["silhouette"].append(float(silhouette_score(d, labels, metric="precomputed")))
        res["purity"].append(knn_purity(order, labels))
        sils, purs = perm_stats(d, order, perms_fresh, labels)
        res["null_sil"][:, li] = sils
        res["null_pur"][:, li] = purs
        # (iii) Spearman(PC1..4, log length) on the un-residualized centered x.
        pca = PCA(n_components=4, random_state=SEED)
        scores = pca.fit_transform(center(mean_h[:, li, :]))
        spearman_pcs.append([float(spearmanr(scores[:, c], log_len).statistic) for c in range(4)])
    return {
        "instance_set": headline_ids,
        "covariate": "log1p(ctx_token_len_content)",
        "length_only_knn_purity": float(len_only_pur),
        "length_only_null_p95": float(np.percentile(null_len, 95)),
        "length_only_p": float((1 + (null_len >= len_only_pur).sum()) / (len(null_len) + 1)),
        "residualized": {
            "silhouette": res["silhouette"],
            "purity": res["purity"],
            "sil_summary": max_over_layers_summary(res["silhouette"], res["null_sil"]),
            "pur_summary": max_over_layers_summary(res["purity"], res["null_pur"]),
        },
        "spearman_pc_vs_loglen": spearman_pcs,
    }


def text_baseline(instances_h: list[dict], labels: np.ndarray, perms: np.ndarray) -> dict:
    """TF-IDF text-similarity family-purity baseline (§14 analyzer bullet)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = []
    for inst in instances_h:
        parts = [inst["system_prompt"] or ""]
        parts += [m["content"] for m in inst["prefix_messages"]]
        texts.append("\n".join(parts))
    tfidf = TfidfVectorizer().fit_transform(texts)
    x = np.asarray(tfidf.todense())
    norms = np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    cos = (x / norms) @ (x / norms).T
    d = np.clip(1.0 - cos, 0.0, None)
    np.fill_diagonal(d, 0.0)
    order = knn_neighbor_order(d)
    pur = knn_purity(order, labels)
    sil = float(silhouette_score(d, labels, metric="precomputed"))
    null_pur = np.array([knn_purity(order, labels[p]) for p in perms])
    return {
        "tfidf_knn_purity": float(pur),
        "tfidf_silhouette": sil,
        "tfidf_purity_null_p95": float(np.percentile(null_pur, 95)),
        "tfidf_purity_p": float((1 + (null_pur >= pur).sum()) / (len(null_pur) + 1)),
    }


def outlier_table(
    mean_all: np.ndarray, ids: list[str], families: list[str], band: list[int]
) -> list[dict]:
    """Per-instance distance to own-family centroid / within-family spread.

    Centered (full-bank) cosine distance to centroid, averaged over the
    mid-late band layers; spread = family-mean of the same quantity.
    """
    fams = np.asarray(families)
    ratios = np.zeros(len(ids))
    for li in band:
        x = center(mean_all[:, li, :])
        xn = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
        dist = np.zeros(len(ids))
        for fam in set(families):
            m = fams == fam
            cent = xn[m].mean(axis=0)
            cent /= np.clip(np.linalg.norm(cent), 1e-12, None)
            dist[m] = 1.0 - xn[m] @ cent
        for fam in set(families):
            m = fams == fam
            spread = max(dist[m].mean(), 1e-12)
            ratios[m] += (dist[m] / spread) / len(band)
    rows = [
        {"instance_id": ids[i], "family": families[i], "outlier_ratio": float(ratios[i])}
        for i in np.argsort(-ratios)
    ]
    return rows


def fp16_sanity(per_probe: dict, mean_all: np.ndarray, ids: list[str]) -> float:
    """A8: max (1 - cos) between stored fp32 means and fp16-recomputed means."""
    worst = 0.0
    for n_i, iid in enumerate(ids):
        rec = per_probe[iid]["tensor"].float().mean(dim=0).numpy()  # (L, H) from fp16
        ref = mean_all[n_i]  # (L, H) true fp32 mean
        num = (rec * ref).sum(axis=1)
        den = np.clip(np.linalg.norm(rec, axis=1) * np.linalg.norm(ref, axis=1), 1e-12, None)
        worst = max(worst, float((1.0 - num / den).max()))
    return worst


# ── Cross-pool comparison (plan v2 §4) ───────────────────────────────────────


def cross_pool_comparison(
    mean_new: np.ndarray,
    ids_new: list[str],
    compare_blob: dict,
    n_perms: int,
    seed: int,
) -> dict:
    """Per-layer cross-pool geometry comparison against a second probe pool.

    Joins the two mean-tensor banks ON STORED INSTANCE IDS (asserted
    set-equal, reindexed — NEVER row order). Per layer over the FULL
    instance bank: both NxN ``global_mean``-centered cosine matrices (each
    pool centered on its own bank), upper-triangle Spearman + Pearson, a
    Mantel permutation p (simultaneous row+column relabeling of the
    comparison matrix, draws shared across layers, one-sided greater —
    permutation inference because upper-triangle entries are
    non-independent), and the per-instance cross-pool centered-vector cosine
    (the split-half-style read).
    """
    ids_old = list(compare_blob["instance_ids"])
    assert set(ids_new) == set(ids_old), (
        f"cross-pool instance-id sets differ: {sorted(set(ids_new) ^ set(ids_old))}"
    )
    reindex = [ids_old.index(i) for i in ids_new]
    mean_old = compare_blob["tensor"].float().numpy()[reindex]
    assert mean_old.shape == mean_new.shape, (mean_old.shape, mean_new.shape)
    n, n_layers, _ = mean_new.shape
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(n) for _ in range(n_perms)]
    iu = np.triu_indices(n, k=1)
    per_layer: list[dict] = []
    per_instance_cos = np.empty((n_layers, n))
    for li in range(n_layers):
        cos_new = compute_cosine_matrix(
            torch.from_numpy(mean_new[:, li, :]).float(), centering="global_mean"
        ).numpy()
        cos_old = compute_cosine_matrix(
            torch.from_numpy(mean_old[:, li, :]).float(), centering="global_mean"
        ).numpy()
        v_new, v_old = cos_new[iu], cos_old[iu]
        obs_s = float(spearmanr(v_new, v_old).statistic)
        obs_p = float(np.corrcoef(v_new, v_old)[0, 1])
        null_s = np.empty(n_perms)
        null_p = np.empty(n_perms)
        for b, perm in enumerate(perms):
            shuf = cos_old[np.ix_(perm, perm)][iu]
            null_s[b] = spearmanr(v_new, shuf).statistic
            null_p[b] = np.corrcoef(v_new, shuf)[0, 1]
        a = center(mean_new[:, li, :])
        b_ = center(mean_old[:, li, :])
        num = (a * b_).sum(axis=1)
        den = np.clip(np.linalg.norm(a, axis=1) * np.linalg.norm(b_, axis=1), 1e-12, None)
        per_instance_cos[li] = num / den
        per_layer.append(
            {
                "layer": li,
                "spearman": obs_s,
                "pearson": obs_p,
                "mantel_p_spearman": float((1 + (null_s >= obs_s).sum()) / (n_perms + 1)),
                "mantel_p_pearson": float((1 + (null_p >= obs_p).sum()) / (n_perms + 1)),
                "median_per_instance_cos": float(np.median(per_instance_cos[li])),
                "mean_per_instance_cos": float(per_instance_cos[li].mean()),
            }
        )
        if li % 7 == 0:
            logger.info("cross-pool layer %d/%d done", li + 1, n_layers)
    parent_headline = {
        f"L{li}": per_layer[li] for li in (14, 18) if li < n_layers
    }  # plan v2 §3 H-B read layers
    return {
        "centering": "global_mean (each pool centered on its own bank)",
        "instance_ids": ids_new,
        "n_instances": n,
        "n_pairs_upper_triangle": len(iu[0]),
        "n_perms": n_perms,
        "seed": seed,
        "per_layer": per_layer,
        "parent_headline_layers": parent_headline,
        "per_instance_cos": {iid: per_instance_cos[:, k].tolist() for k, iid in enumerate(ids_new)},
    }


# ── Figures ──────────────────────────────────────────────────────────────────


def family_colors() -> dict[str, str]:
    pal = paper_palette(len(FAMILY_ORDER))
    return dict(zip(FAMILY_ORDER, pal, strict=True))


def fig_hero_curves(results: dict, n_layers: int) -> None:
    """Hero 1: sil + purity vs layer w/ null band, bootstrap CI, resid overlay."""
    colors = family_colors()
    _ = colors  # palette consistency is per-family figs; curves use roles below
    layers = np.arange(n_layers)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in (
        (axes[0], "silhouette", "Silhouette vs depth"),
        (axes[1], "purity", "k-NN family purity (k=4) vs depth"),
    ):
        obs = results["headline"][key]
        null95 = results[f"{key}_summary"]["null_p95_per_layer"]
        nullmean = results[f"{key}_summary"]["null_mean_per_layer"]
        ci_lo = results["bootstrap"][f"{'sil' if key == 'silhouette' else 'pur'}_ci_lo"]
        ci_hi = results["bootstrap"][f"{'sil' if key == 'silhouette' else 'pur'}_ci_hi"]
        resid = results["length"]["residualized"][key]
        ax.fill_between(
            layers, nullmean, null95, color="0.85", label="permutation null (mean to 95%)"
        )
        ax.fill_between(layers, ci_lo, ci_hi, color="#1f77b4", alpha=0.2, label="bootstrap 95% CI")
        ax.plot(layers, obs, color="#1f77b4", lw=2, label="observed (centered)")
        ax.plot(layers, resid, color="#d62728", lw=1.5, ls="--", label="length-residualized")
        ax.set_xlabel("decoder layer")
        ax.set_ylabel(key if key == "silhouette" else "purity")
        ax.set_title(title)
    axes[0].legend(loc="upper left", fontsize=7)
    savefig_paper(fig, "hero_clustering_vs_layer", dir=FIG_DIR)
    plt.close(fig)


def _scatter(ax, emb: np.ndarray, families: list[str], labels: list[str]) -> None:
    colors = family_colors()
    for fam in FAMILY_ORDER:
        m = np.asarray(families) == fam
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1], s=14, color=colors[fam], label=fam)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (emb[i, 0], emb[i, 1]), fontsize=3.5, alpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_hero_embeddings(mean_all, families, labels, qlayers, umap_module) -> None:
    """Hero 2: PCA (top) + UMAP n=15/d=0.1 (bottom) at the quartile layers."""
    fig, axes = plt.subplots(2, len(qlayers), figsize=(4 * len(qlayers), 8))
    for col, li in enumerate(qlayers):
        x = center(mean_all[:, li, :])
        pca_emb = PCA(n_components=2, random_state=SEED).fit_transform(x)
        _scatter(axes[0, col], pca_emb, families, labels)
        axes[0, col].set_title(f"PCA — layer {li}")
        um = umap_module.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED)
        _scatter(axes[1, col], um.fit_transform(x), families, labels)
        axes[1, col].set_title(f"UMAP (n=15, d=0.1, seed 42) — layer {li}")
    axes[0, 0].legend(fontsize=6, loc="best")
    savefig_paper(fig, "hero_embeddings_pca_umap", dir=FIG_DIR)
    plt.close(fig)


def fig_pca_small_multiples(mean_all, families, n_layers) -> None:
    ncols = 7
    nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
    for li in range(n_layers):
        ax = axes.flat[li]
        emb = PCA(n_components=2, random_state=SEED).fit_transform(center(mean_all[:, li, :]))
        _scatter(ax, emb, families, [""] * len(families))
        ax.set_title(f"L{li}", fontsize=7)
    for j in range(n_layers, nrows * ncols):
        axes.flat[j].axis("off")
    savefig_paper(fig, "pca_small_multiples_all_layers", dir=FIG_DIR)
    plt.close(fig)


def fig_umap_grid_and_tsne(mean_all, families, labels, best_layer, umap_module) -> None:
    from sklearn.manifold import TSNE

    x = center(mean_all[:, best_layer, :])
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (nn, md) in zip(axes.flat, UMAP_GRID, strict=True):
        um = umap_module.UMAP(n_neighbors=nn, min_dist=md, metric="cosine", random_state=SEED)
        _scatter(ax, um.fit_transform(x), families, labels)
        ax.set_title(f"UMAP n={nn}, d={md}, seed 42 — layer {best_layer}", fontsize=8)
    savefig_paper(fig, "umap_grid_best_layer", dir=FIG_DIR)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, perp in zip(axes.flat, TSNE_PERPLEXITIES, strict=True):
        ts = TSNE(perplexity=perp, metric="cosine", init="random", random_state=SEED)
        _scatter(ax, ts.fit_transform(x), families, labels)
        ax.set_title(f"t-SNE perplexity={perp}, seed 42 — layer {best_layer}", fontsize=8)
    savefig_paper(fig, "tsne_best_layer", dir=FIG_DIR)
    plt.close(fig)


def fig_heatmaps_dendrograms(mean_all, labels, qlayers) -> None:
    for li in qlayers:
        x = mean_all[:, li, :]
        d_c = cosine_dist(x, "global_mean")
        d_r = cosine_dist(x, "none")
        link = linkage(squareform(d_c, checks=False), method="average")
        order = leaves_list(link)
        olabels = [labels[i] for i in order]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, d, title in (
            (axes[0], 1 - d_c, f"centered cosine (global_mean) — layer {li}"),
            (axes[1], 1 - d_r, f"raw cosine — layer {li}"),
        ):
            im = ax.imshow(d[np.ix_(order, order)], cmap="viridis")
            ax.set_title(title, fontsize=9)
            ax.set_xticks(range(len(olabels)))
            ax.set_yticks(range(len(olabels)))
            ax.set_xticklabels(olabels, fontsize=3.5, rotation=90)
            ax.set_yticklabels(olabels, fontsize=3.5)
            fig.colorbar(im, ax=ax, shrink=0.7)
        savefig_paper(fig, f"cosine_heatmap_L{li}", dir=FIG_DIR)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(link, labels=labels, leaf_font_size=4, ax=ax)
        ax.set_title(f"Average-linkage dendrogram (1 - centered cosine) — layer {li}", fontsize=9)
        savefig_paper(fig, f"dendrogram_L{li}", dir=FIG_DIR)
        plt.close(fig)


def fig_cka(cka: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cka, cmap="magma", vmin=0, vmax=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    ax.set_title("Cross-layer linear CKA (50-instance battery)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    savefig_paper(fig, "cka_cross_layer", dir=FIG_DIR)
    plt.close(fig)


def fig_ev_spectra(mean_all, qlayers) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for li in qlayers:
        x = center(mean_all[:, li, :])
        evr = PCA(n_components=min(10, x.shape[0] - 1), random_state=SEED).fit(x)
        ax.plot(
            range(1, len(evr.explained_variance_ratio_) + 1),
            evr.explained_variance_ratio_,
            marker="o",
            ms=3,
            label=f"layer {li}",
        )
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio")
    ax.set_title("PCA spectra by depth")
    ax.legend(fontsize=7)
    savefig_paper(fig, "pca_ev_spectra", dir=FIG_DIR)
    plt.close(fig)


def fig_pc_vs_length(mean_h, log_len, families_h, qlayers) -> None:
    fig, axes = plt.subplots(1, len(qlayers), figsize=(3.2 * len(qlayers), 3.4))
    colors = family_colors()
    for ax, li in zip(np.atleast_1d(axes).flat, qlayers, strict=True):
        scores = PCA(n_components=1, random_state=SEED).fit_transform(center(mean_h[:, li, :]))
        for fam in FAMILY_ORDER:
            m = np.asarray(families_h) == fam
            if m.any():
                ax.scatter(log_len[m], scores[m, 0], s=12, color=colors[fam], label=fam)
        ax.set_xlabel("log1p(context tokens)")
        ax.set_ylabel("PC1 score")
        ax.set_title(f"layer {li}", fontsize=9)
    savefig_paper(fig, "pc1_vs_log_length", dir=FIG_DIR)
    plt.close(fig)


def fig_split_half(sh: dict, n_layers: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    layers = np.arange(n_layers)
    ax.plot(
        layers, sh["median_crosshalf_cos_centered"], lw=2, label="centered (kill-criterion read)"
    )
    ax.plot(layers, sh["median_crosshalf_cos_raw"], lw=1.5, ls="--", label="raw (ceiling-prone)")
    ax.axhline(0.9, color="0.5", lw=1, ls=":")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("median cross-half instance cosine")
    ax.set_title("Probe-split-half reliability")
    ax.legend(fontsize=7)
    savefig_paper(fig, "split_half_reliability", dir=FIG_DIR)
    plt.close(fig)


def fig_per_family(headline: dict, n_layers: int, families_present: list[str]) -> None:
    fams = [f for f in FAMILY_ORDER if f in families_present]
    mat = np.array(
        [[headline["per_family_purity"][li].get(f, np.nan) for li in range(n_layers)] for f in fams]
    )
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(fams, fontsize=8)
    ax.set_xlabel("decoder layer")
    ax.set_title("Per-family k-NN purity by depth")
    fig.colorbar(im, ax=ax, shrink=0.8)
    savefig_paper(fig, "per_family_purity_by_depth", dir=FIG_DIR)
    plt.close(fig)


def fig_baselines(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["activation\n(max over layers)", "length-only\n(1-D k-NN)", "TF-IDF text\nsimilarity"]
    vals = [
        results["purity_summary"]["observed_max"],
        results["length"]["length_only_knn_purity"],
        results["text_baseline"]["tfidf_knn_purity"],
    ]
    nulls = [
        results["purity_summary"]["null_max_p95"],
        results["length"]["length_only_null_p95"],
        results["text_baseline"]["tfidf_purity_null_p95"],
    ]
    xs = np.arange(3)
    ax.bar(xs, vals, width=0.55, color=paper_palette(3))
    for x, nv in zip(xs, nulls, strict=True):
        ax.hlines(nv, x - 0.33, x + 0.33, color="0.3", lw=1.5, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("k-NN family purity (k=4)")
    ax.set_title("Purity vs baselines (dashes: permutation-null 95%)")
    savefig_paper(fig, "purity_vs_baselines", dir=FIG_DIR)
    plt.close(fig)


def fig_cross_pool(cross: dict, n_layers: int) -> None:
    """Per-layer Mantel-curve figure (plan v2 §4): correlation + per-instance read."""
    layers = np.arange(n_layers)
    sp = [r["spearman"] for r in cross["per_layer"]]
    pe = [r["pearson"] for r in cross["per_layer"]]
    med = [r["median_per_instance_cos"] for r in cross["per_layer"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pal = paper_palette(2)
    axes[0].plot(layers, sp, color=pal[0], lw=2, label="Mantel Spearman")
    axes[0].plot(layers, pe, color=pal[1], lw=1.5, ls="--", label="Pearson")
    for li in (14, 18):
        if li < n_layers:
            axes[0].axvline(li, color="0.7", lw=1, ls=":")
    axes[0].axhline(0.0, color="0.5", lw=1)
    axes[0].set_xlabel("decoder layer")
    axes[0].set_ylabel("upper-triangle correlation")
    axes[0].set_title("Cross-pool 50x50 centered-cosine matrix correlation")
    axes[0].legend(fontsize=7)
    axes[1].plot(layers, med, lw=2, color=pal[0])
    axes[1].set_xlabel("decoder layer")
    axes[1].set_ylabel("median per-instance cosine")
    axes[1].set_title("Cross-pool per-instance centered-vector cosine")
    savefig_paper(fig, "cross_pool_mantel_curve", dir=FIG_DIR)
    plt.close(fig)


def fig_overlay_hero(results: dict, parent_metrics: dict, n_layers: int) -> None:
    """Overlay hero (plan v2 §4): new purity/silhouette depth curves over the parent's."""
    parent_headline = parent_metrics["headline"]
    for key in ("silhouette", "purity"):
        assert len(parent_headline[key]) == n_layers, (
            f"parent {key} curve has {len(parent_headline[key])} layers, expected {n_layers}"
        )
    layers = np.arange(n_layers)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pal = paper_palette(2)
    for ax, key, title in (
        (axes[0], "silhouette", "Silhouette vs depth"),
        (axes[1], "purity", "k-NN family purity (k=4) vs depth"),
    ):
        null95 = results[f"{key}_summary"]["null_p95_per_layer"]
        nullmean = results[f"{key}_summary"]["null_mean_per_layer"]
        ax.fill_between(
            layers, nullmean, null95, color="0.85", label="fresh permutation null (mean to 95%)"
        )
        ax.plot(layers, results["headline"][key], color=pal[0], lw=2, label="UltraChat probe pool")
        ax.plot(
            layers,
            parent_headline[key],
            color=pal[1],
            lw=1.5,
            ls="--",
            label="parent (Betley probe pool)",
        )
        ax.set_xlabel("decoder layer")
        ax.set_ylabel(key if key == "silhouette" else "purity")
        ax.set_title(title)
    axes[0].legend(loc="upper left", fontsize=7)
    savefig_paper(fig, "hero_overlay_probe_pools", dir=FIG_DIR)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def build_narration(results: dict) -> dict:
    """§14 item 3: multiplicity sentence + computed discordance narration."""
    sil_pass = results["silhouette_summary"]["passes"]
    pur_pass = results["purity_summary"]["passes"]
    if sil_pass and pur_pass:
        verdict = "both co-primaries beat their max-over-layers permutation nulls"
    elif not sil_pass and not pur_pass:
        verdict = "neither co-primary beats its max-over-layers permutation null"
    else:
        winner = "purity" if pur_pass else "silhouette"
        verdict = (
            f"METRIC-DISCORDANT: only {winner} beats its max-over-layers null — "
            "narrate as exploratory evidence of local neighborhood structure "
            "without compact global separation, never as unqualified "
            "'family structure exists'"
        )
    return {
        "multiplicity_sentence": MULTIPLICITY_SENTENCE,
        "co_primary_verdict": verdict,
        "silhouette_passes_fwer": bool(sil_pass),
        "purity_passes_fwer": bool(pur_pass),
    }


def main() -> int:
    global EVAL_DIR, FIG_DIR
    parser = argparse.ArgumentParser(description="Issue #594 Phase 2 geometry analysis.")
    parser.add_argument("--tensors-dir", type=Path, default=None)
    parser.add_argument(
        "--tensors-from-hf",
        default=None,
        help="HF data-repo prefix, e.g. issue594_context_geometry/analysis_tensors",
    )
    parser.add_argument(
        "--hf-repo",
        default=HF_DATA_REPO,
        help="HF dataset repo holding the tensors; pass the overflow repo when the "
        "extraction sentinel / manifest 'upload.repo' records the quota-403 fallback",
    )
    parser.add_argument(
        "--compare-tensors-from-hf",
        default=None,
        help="HF data-repo prefix of a SECOND mean-tensor set to compare against "
        "(plan v2 §4 cross-pool module), e.g. issue594_context_geometry/analysis_tensors",
    )
    parser.add_argument(
        "--compare-hf-repo",
        default=HF_DATA_REPO,
        help="HF dataset repo holding the comparison tensors (default: primary data repo)",
    )
    parser.add_argument(
        "--parent-metrics-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_594" / "context_geometry_metrics.json",
        help="parent run's metrics JSON for the overlay hero (cross-pool module only)",
    )
    parser.add_argument("--battery", type=Path, default=BATTERY_PATH)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=EVAL_DIR,
        help="metrics output dir (default eval_results/issue_594; override for smoke runs)",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=FIG_DIR,
        help="figure output dir (default figures/issue_594; override for smoke runs)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="metrics JSON path (default: <eval-dir>/context_geometry_metrics.json)",
    )
    parser.add_argument("--n-perms", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    EVAL_DIR = args.eval_dir
    FIG_DIR = args.fig_dir
    out_json = args.out_json or EVAL_DIR / "context_geometry_metrics.json"

    selfcheck_cosine_matrix()
    set_paper_style("blog")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if (args.tensors_dir is None) == (args.tensors_from_hf is None):
        raise SystemExit("provide exactly one of --tensors-dir / --tensors-from-hf")
    tensors_dir = args.tensors_dir or download_from_hf(
        args.tensors_from_hf, EVAL_DIR / "hf_tensors_cache", repo=args.hf_repo
    )

    data = load_tensors(tensors_dir)
    _, instances = load_battery(args.battery)
    inst_by_id = {i["id"]: i for i in instances}
    ids: list[str] = data["ids"]
    families: list[str] = data["families"]
    labels_all = [inst_by_id[i]["label"] for i in ids]
    n_layers = data["n_layers"]
    mean_all = data["mean"]  # (N, L, H) fp32

    # Early coverage check: every runtime tensor id must have a manifest row
    # carrying the covariate fields the length reads consume — fail loud with
    # the offending ids instead of a KeyError ten frames deep.
    manifest_instances = data["manifest"]["instances"]
    required_fields = ("ctx_token_len_content",)
    missing_ids = [iid for iid in ids if iid not in manifest_instances]
    incomplete_ids = [
        iid
        for iid in ids
        if iid not in missing_ids and any(k not in manifest_instances[iid] for k in required_fields)
    ]
    if missing_ids or incomplete_ids:
        raise RuntimeError(
            "manifest coverage check failed: "
            f"{len(missing_ids)} tensor ids absent from extraction_manifest.json "
            f"{missing_ids}; {len(incomplete_ids)} ids missing required covariate "
            f"fields {required_fields} {incomplete_ids}"
        )
    recorded_upload = data["manifest"].get("upload", {})
    if recorded_upload.get("repo"):
        logger.info("Manifest records upload repo: %s", recorded_upload["repo"])

    h_idx = [k for k, iid in enumerate(ids) if families[k] not in HEADLINE_EXCLUDED_FAMILIES]
    headline_ids = [ids[k] for k in h_idx]
    fam_h = np.asarray([families[k] for k in h_idx])
    mean_h = mean_all[h_idx]  # (48, L, H)
    per_probe_h = np.stack(
        [data["per_probe"][iid]["tensor"].float().numpy() for iid in headline_ids]
    )  # (48, P, L, H)
    log_len = np.log1p(
        np.asarray(
            [data["manifest"]["instances"][iid]["ctx_token_len_content"] for iid in headline_ids],
            dtype=np.float64,
        )
    )
    logger.info(
        "Headline set: %d instances / %d families; P=%d probes",
        len(h_idx),
        len(set(fam_h)),
        per_probe_h.shape[1],
    )

    rng = np.random.default_rng(args.seed)
    perms = np.stack([rng.permutation(len(h_idx)) for _ in range(args.n_perms)])
    rng_fresh = np.random.default_rng(args.seed + 4200)  # fresh draws for resid null
    perms_fresh = np.stack([rng_fresh.permutation(len(h_idx)) for _ in range(args.n_perms)])

    results: dict = {
        "centering": "global_mean",
        "headline_bank": headline_ids,
        "fullbank": ids,
        "knn_k": KNN_K,
        "n_perms": args.n_perms,
        "seed": args.seed,
        "quartile_layers": quartile_layers(n_layers),
    }

    logger.info("[1/8] headline curves + shared-draw permutation null")
    headline = headline_curves(mean_h, fam_h, perms)
    results["headline"] = {
        k: headline[k]
        for k in (
            "silhouette",
            "purity",
            "silhouette_raw",
            "purity_raw",
            "per_family_purity",
            "per_family_silhouette",
        )
    }
    results["silhouette_summary"] = max_over_layers_summary(
        headline["silhouette"], headline["null_sil"]
    )
    results["purity_summary"] = max_over_layers_summary(headline["purity"], headline["null_pur"])
    best_layer = int(np.argmax(headline["purity"]))
    results["best_layer_by_purity"] = best_layer

    logger.info("[2/8] rephrase-excluded sensitivity")
    nr_idx = [k for k, iid in enumerate(headline_ids) if fam_h[k] != "rephrase"]
    fam_nr = fam_h[nr_idx]
    rng_nr = np.random.default_rng(args.seed + 9900)
    perms_nr = np.stack([rng_nr.permutation(len(nr_idx)) for _ in range(args.n_perms)])
    headline_nr = headline_curves(mean_h[nr_idx], fam_nr, perms_nr)
    results["rephrase_excluded"] = {
        "silhouette_summary": max_over_layers_summary(
            headline_nr["silhouette"], headline_nr["null_sil"]
        ),
        "purity_summary": max_over_layers_summary(headline_nr["purity"], headline_nr["null_pur"]),
    }

    logger.info("[3/8] bootstrap over probes (B=%d)", args.n_boot)
    results["bootstrap"] = bootstrap_over_probes(per_probe_h, fam_h, args.n_boot, rng)

    logger.info("[4/8] probe-split-half reliability")
    results["split_half"] = split_half(per_probe_h, fam_h, np.random.default_rng(args.seed))

    logger.info("[5/8] length reads")
    results["length"] = length_reads(mean_h, fam_h, log_len, perms_fresh, headline_ids)

    logger.info("[6/8] text-similarity baseline + outliers + CKA + fp16 sanity")
    results["text_baseline"] = text_baseline([inst_by_id[i] for i in headline_ids], fam_h, perms)
    band = [li for li in range(n_layers) if li >= n_layers // 2]
    results["outliers"] = outlier_table(mean_all, ids, families, band)[:10]
    with open(EVAL_DIR / "outlier_table.json", "w") as f:
        json.dump(jsonable(outlier_table(mean_all, ids, families, band)), f, indent=2)
    cka = np.array(
        [
            [linear_cka(mean_all[:, i, :], mean_all[:, j, :]) for j in range(n_layers)]
            for i in range(n_layers)
        ]
    )
    results["cka_matrix"] = cka.tolist()
    results["fp16_mean_max_cos_deviation"] = fp16_sanity(data["per_probe"], mean_all, ids)
    assert results["fp16_mean_max_cos_deviation"] < 1e-3, results["fp16_mean_max_cos_deviation"]

    results["narration"] = build_narration(results)

    cross = None
    parent_metrics = None
    if args.compare_tensors_from_hf:
        logger.info(
            "[6b/8] cross-pool comparison vs %s (%s)",
            args.compare_tensors_from_hf,
            args.compare_hf_repo,
        )
        compare_dir = download_from_hf(
            args.compare_tensors_from_hf,
            EVAL_DIR / "hf_tensors_cache_compare",
            repo=args.compare_hf_repo,
        )
        compare_blob = torch.load(compare_dir / "context_vectors_mean.pt", weights_only=True)
        cross = cross_pool_comparison(mean_all, ids, compare_blob, args.n_perms, args.seed)
        cross["compare_prefix"] = args.compare_tensors_from_hf
        cross["compare_repo"] = args.compare_hf_repo
        cross["metadata"] = reproducibility_metadata(
            {"script": "issue594_analyze_context_geometry", "module": "cross_pool"}
        )
        # Checkpoint-per-phase: the cross-pool JSON lands NOW, before figures.
        with open(EVAL_DIR / "cross_pool_comparison.json", "w") as f:
            json.dump(jsonable(cross), f, indent=2)
        logger.info("Wrote %s", EVAL_DIR / "cross_pool_comparison.json")
        if not args.parent_metrics_json.exists():
            raise RuntimeError(
                f"--compare-tensors-from-hf set but parent metrics JSON missing: "
                f"{args.parent_metrics_json} (needed for the overlay hero, plan v2 §4)"
            )
        with open(args.parent_metrics_json) as f:
            parent_metrics = json.load(f)

    logger.info("[7/8] figures")
    import umap

    qlayers = quartile_layers(n_layers)
    fig_hero_curves(results, n_layers)
    fig_hero_embeddings(mean_all, families, labels_all, qlayers, umap)
    fig_pca_small_multiples(mean_all, families, n_layers)
    fig_umap_grid_and_tsne(mean_all, families, labels_all, best_layer, umap)
    fig_heatmaps_dendrograms(mean_all, labels_all, qlayers)
    fig_cka(cka)
    fig_ev_spectra(mean_all, qlayers)
    fig_pc_vs_length(mean_h, log_len, [families[k] for k in h_idx], qlayers)
    fig_split_half(results["split_half"], n_layers)
    fig_per_family(results["headline"], n_layers, sorted(set(fam_h)))
    fig_baselines(results)
    if cross is not None:
        fig_cross_pool(cross, n_layers)
        fig_overlay_hero(results, parent_metrics, n_layers)

    logger.info("[8/8] write metrics JSON + per-layer JSONs")
    per_layer_dir = EVAL_DIR / "per_layer"
    per_layer_dir.mkdir(parents=True, exist_ok=True)
    for li in range(n_layers):
        x = mean_all[:, li, :]
        with open(per_layer_dir / f"layer_{li:02d}.json", "w") as f:
            json.dump(
                jsonable(
                    {
                        "layer": li,
                        "instance_ids": ids,
                        "centering_bank": ids,
                        "cosine_centered": 1.0 - cosine_dist(x, "global_mean"),
                        "cosine_raw": 1.0 - cosine_dist(x, "none"),
                    }
                ),
                f,
            )
    results["metadata"] = reproducibility_metadata(
        {"script": "issue594_analyze_context_geometry", "tensors_dir": str(tensors_dir)}
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(jsonable(results), f, indent=2)
    logger.info(
        "Done. best_layer=%d, max purity=%.3f (null95 %.3f), max sil=%.3f (null95 %.3f); %s",
        best_layer,
        results["purity_summary"]["observed_max"],
        results["purity_summary"]["null_max_p95"],
        results["silhouette_summary"]["observed_max"],
        results["silhouette_summary"]["null_max_p95"],
        results["narration"]["co_primary_verdict"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
