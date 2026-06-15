#!/usr/bin/env python3
"""Issue #617 Step 4: per-cluster-pair separability scoring (VM CPU, off-pod).

Per plan §4 step 4 + SA1. For every candidate cluster PAIR (within a config —
cross-config pairs are not scored), assembles the two clusters' extracted
prefix vectors (mapped from the per-conv_id extraction via the conv_id ->
cluster map) as a 2-family labeling and computes, REUSING
``issue594_analyze_context_geometry`` primitives:

- centered-cosine k-NN purity (k=4) at each of L13/14/18;
- length-residualized purity (the headline DV);
- best-of-{13,14,18} length-residualized purity = the pair's score;
- TF-IDF word-overlap + length-only baselines (comparison rails).

SA1 (selection-aware global permutation null): the headline is the BEST pair
across all scored pairs (a winner-selection search over hundreds of pairs), so
the per-pair null is not the right reference. Under each of B=1000 shuffles,
recompute best-of-layers length-residualized purity for EVERY scored pair, take
the MAX over (pairs x layers), and compare the observed winner against that
selection-aware null -> ``winner.p_global``.

Fire-rule (plan §6): ``length_residualized_purity >= 0.7 AND winner.p_global <=
0.01``.

Runs on the VM (CPU only); reuses the already-extracted #594-format tensors.

Usage::

    uv run python scripts/issue617_score_separability.py \
        --tensors-dir data/issue617/extraction \
        --membership data/issue617/cluster_membership.json \
        --battery data/issue617/extraction_battery.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue594_analyze_context_geometry import (  # noqa: E402
    cosine_dist,
    knn_neighbor_order,
    knn_purity,
    residualize,
)
from issue594_common import load_battery_loose  # noqa: E402
from issue617_common import (  # noqa: E402
    CLUSTER_MEMBERSHIP_PATH,
    EVAL_DIR,
    EXTRACTION_BATTERY_PATH,
    EXTRACTION_DIR,
    PER_CLUSTER_FLOOR,
    PER_CLUSTER_SCORING_MAX,
    PERM_B,
    PGLOBAL_THRESHOLD,
    PURITY_THRESHOLD,
    READ_LAYERS,
    SEED,
    SEPARABILITY_PATH,
    SYNTHETIC_DEFAULT_ID,
)

load_dotenv()

logger = logging.getLogger("issue617_separability")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KNN_K = 4


def _config_of(cluster_id: str) -> str:
    """Config algo name from a cluster id (e.g. 'kmeans10_c03' -> 'kmeans10')."""
    return cluster_id.rsplit("_c", 1)[0]


def load_extraction(tensors_dir: Path) -> dict:
    """Load the mean tensor + manifest; return per-conv_id vectors + length covariate."""
    blob = torch.load(tensors_dir / "context_vectors_mean.pt", weights_only=True)
    with open(tensors_dir / "extraction_manifest.json") as f:
        manifest = json.load(f)
    ids = list(blob["instance_ids"])
    mean = blob["tensor"].float().numpy()  # (N, L, H)
    assert mean.shape[0] == len(ids), (mean.shape, len(ids))
    by_id = {iid: i for i, iid in enumerate(ids)}
    content_tokens = {
        iid: manifest["instances"][iid]["ctx_token_len_content"]
        for iid in ids
        if iid in manifest["instances"]
    }
    return {
        "ids": ids,
        "by_id": by_id,
        "mean": mean,
        "n_layers": mean.shape[1],
        "content_tokens": content_tokens,
    }


def enumerate_pairs(cluster_members: dict[str, list[str]], floor: int) -> list[tuple[str, str]]:
    """All within-config cluster pairs with >= floor extracted members each.

    Pairs only clusters from the SAME config (a pair must come from one
    clustering). The synthetic default is never a member (excluded upstream).
    """
    by_config: dict[str, list[str]] = {}
    for cid, members in cluster_members.items():
        if len(members) >= floor:
            by_config.setdefault(_config_of(cid), []).append(cid)
    pairs: list[tuple[str, str]] = []
    for cids in by_config.values():
        cids = sorted(cids)
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                pairs.append((cids[i], cids[j]))
    return pairs


def pair_vectors(
    pair: tuple[str, str],
    cluster_members: dict[str, list[str]],
    extraction: dict,
    rng: np.random.Generator,
    per_cluster_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Assemble the 2-family (X, labels, log_len, conv_ids) for one pair.

    Subsamples each cluster to <= per_cluster_max members (deterministic via
    the passed rng) so the 2-family labeling stays in the #594-validated
    k=4 purity N range. Returns X (n, L, H), labels (n,), log_len (n,), ids.
    """
    rows: list[int] = []
    labels: list[str] = []
    ids: list[str] = []
    for cid in pair:
        members = [m for m in cluster_members[cid] if m in extraction["by_id"]]
        if len(members) > per_cluster_max:
            members = sorted(rng.choice(members, size=per_cluster_max, replace=False).tolist())
        for m in members:
            rows.append(extraction["by_id"][m])
            labels.append(cid)
            ids.append(m)
    X = extraction["mean"][rows]  # (n, L, H)
    log_len = np.log1p(np.asarray([extraction["content_tokens"][m] for m in ids], dtype=np.float64))
    return X, np.asarray(labels), log_len, ids


def residualized_orders(
    X: np.ndarray, log_len: np.ndarray, layers: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """Per-layer length-residualized centered-cosine k-NN neighbor order.

    residualize on [1, log_len], then 1 - cosine (global_mean centering), then
    the stable neighbor order — precomputed ONCE per (pair, layer) so the
    permutation null only relabels (no re-distance).
    """
    orders: dict[int, np.ndarray] = {}
    for li in layers:
        xr = residualize(X[:, li, :], log_len)
        d = cosine_dist(xr, centering="global_mean")
        orders[li] = knn_neighbor_order(d)
    return orders


def best_residualized_purity(
    orders: dict[int, np.ndarray], labels: np.ndarray, layers: tuple[int, ...]
) -> tuple[float, int, dict[int, float]]:
    """best-of-layers residualized purity + argmax layer + per-layer purities."""
    per_layer = {li: knn_purity(orders[li], labels, KNN_K) for li in layers}
    best_layer = max(per_layer, key=lambda li: per_layer[li])
    return per_layer[best_layer], best_layer, per_layer


def raw_and_length_purity(
    X: np.ndarray, log_len: np.ndarray, labels: np.ndarray, layers: tuple[int, ...]
) -> tuple[dict[int, float], float]:
    """Raw (non-residualized) centered-cosine purity per layer + length-only purity."""
    raw = {}
    for li in layers:
        d = cosine_dist(X[:, li, :], centering="global_mean")
        raw[li] = knn_purity(knn_neighbor_order(d), labels, KNN_K)
    d_len = np.abs(log_len[:, None] - log_len[None, :])
    length_only = knn_purity(knn_neighbor_order(d_len), labels, KNN_K)
    return raw, float(length_only)


def tfidf_purity(ids: list[str], labels: np.ndarray, first_user_by_id: dict[str, str]) -> float:
    """TF-IDF word-overlap k-NN purity baseline over the pair's first-user turns."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [first_user_by_id.get(i, "") for i in ids]
    tfidf = TfidfVectorizer().fit_transform(texts)
    x = np.asarray(tfidf.todense())
    norms = np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    cos = (x / norms) @ (x / norms).T
    d = np.clip(1.0 - cos, 0.0, None)
    np.fill_diagonal(d, 0.0)
    return knn_purity(knn_neighbor_order(d), labels, KNN_K)


def score_all(
    pairs: list[tuple[str, str]],
    cluster_members: dict[str, list[str]],
    extraction: dict,
    first_user_by_id: dict[str, str],
    layers: tuple[int, ...],
    n_perms: int,
    per_cluster_max: int,
    seed: int,
) -> dict:
    """Score every pair + the SA1 selection-aware global max-over-pairs null."""
    rng = np.random.default_rng(seed)
    # Per-pair: precompute residualized neighbor orders + observed score, AND
    # a per-pair shuffle bank (independent within-pair label shuffles, shared
    # across the global-null loop).
    scored: list[dict] = []
    pair_state: list[dict] = []
    for pair in pairs:
        X, labels, log_len, ids = pair_vectors(
            pair, cluster_members, extraction, rng, per_cluster_max
        )
        orders = residualized_orders(X, log_len, layers)
        obs_best, best_layer, per_layer_res = best_residualized_purity(orders, labels, layers)
        raw, length_only = raw_and_length_purity(X, log_len, labels, layers)
        tfidf = tfidf_purity(ids, labels, first_user_by_id)
        # Per-pair shuffle bank (within-pair permutation of the 2-family labels).
        # s_b is computed ONCE here and reused for BOTH the per-pair p AND the
        # SA1 global max-over-pairs null (element-wise max across pairs below) —
        # no double k-NN-purity loop.
        n = len(labels)
        perms = np.stack([rng.permutation(n) for _ in range(n_perms)])
        s_b = np.empty(n_perms)
        for b in range(n_perms):
            pl = labels[perms[b]]
            s_b[b] = max(knn_purity(orders[li], pl, KNN_K) for li in layers)
        per_pair_p = float((1 + (s_b >= obs_best).sum()) / (n_perms + 1))
        scored.append(
            {
                "config": _config_of(pair[0]),
                "cluster_a": pair[0],
                "cluster_b": pair[1],
                "n_a": int((labels == pair[0]).sum()),
                "n_b": int((labels == pair[1]).sum()),
                "residualized_purity_best": float(obs_best),
                "best_layer": int(best_layer),
                "residualized_purity_per_layer": {
                    str(li): float(v) for li, v in per_layer_res.items()
                },
                "raw_purity_per_layer": {str(li): float(v) for li, v in raw.items()},
                "length_only_purity": float(length_only),
                "tfidf_purity": float(tfidf),
                "per_pair_p_uncorrected": per_pair_p,
                "example_first_users_a": [
                    first_user_by_id.get(m, "")[:200] for m in cluster_members[pair[0]][:5]
                ],
                "example_first_users_b": [
                    first_user_by_id.get(m, "")[:200] for m in cluster_members[pair[1]][:5]
                ],
            }
        )
        pair_state.append(s_b)

    if not scored:
        raise RuntimeError("no scorable pairs (no config had >= 2 clusters above the floor)")

    # SA1 global max-over-(pairs x layers) null: under EACH shuffle index b,
    # take the max over all pairs of their per-pair best-of-layers shuffled
    # purity (reusing each pair's already-computed s_b — element-wise max).
    m_b = np.max(np.stack(pair_state), axis=0)

    # Winner: highest residualized purity, tie-break lower-K, KMeans-before-
    # HDBSCAN, lexicographic cluster ids.
    def _config_rank(config: str) -> tuple[int, int, str]:
        if config.startswith("kmeans"):
            return (0, int(config[len("kmeans") :]), config)
        return (1, 0, config)

    scored_sorted = sorted(
        scored,
        key=lambda s: (
            -s["residualized_purity_best"],
            _config_rank(s["config"])[1],  # lower K first
            _config_rank(s["config"])[0],  # KMeans before HDBSCAN
            s["cluster_a"],
            s["cluster_b"],
        ),
    )
    winner = scored_sorted[0]
    obs = winner["residualized_purity_best"]
    p_global = float((1 + (m_b >= obs).sum()) / (n_perms + 1))

    fires = bool(obs >= PURITY_THRESHOLD and p_global <= PGLOBAL_THRESHOLD)
    return {
        "winner": {
            **winner,
            "p_global": p_global,
        },
        "fire_rule": {
            "threshold_purity": PURITY_THRESHOLD,
            "threshold_p_global": PGLOBAL_THRESHOLD,
            "length_residualized_purity": obs,
            "p_global": p_global,
            "fires": fires,
            "verdict": (
                "CATEGORIES SEPARATE CLEANLY"
                if fires
                else (
                    "DOES NOT FIRE (real categories not cleanly separable "
                    "under the selection-aware null)"
                )
            ),
        },
        "n_pairs_scored": len(scored),
        "top3": scored_sorted[:3],
        "perm_null_global": {
            "B": n_perms,
            "max_over_pairs_distribution_quantiles": {
                "p50": float(np.percentile(m_b, 50)),
                "p90": float(np.percentile(m_b, 90)),
                "p95": float(np.percentile(m_b, 95)),
                "p99": float(np.percentile(m_b, 99)),
                "max": float(m_b.max()),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #617 Step 4: separability scoring (SA1).")
    parser.add_argument("--tensors-dir", type=Path, default=EXTRACTION_DIR)
    parser.add_argument("--membership", type=Path, default=CLUSTER_MEMBERSHIP_PATH)
    parser.add_argument("--battery", type=Path, default=EXTRACTION_BATTERY_PATH)
    parser.add_argument("--out", type=Path, default=SEPARABILITY_PATH)
    parser.add_argument("--n-perms", type=int, default=PERM_B)
    parser.add_argument("--per-cluster-max", type=int, default=PER_CLUSTER_SCORING_MAX)
    parser.add_argument("--per-cluster-floor", type=int, default=PER_CLUSTER_FLOOR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    extraction = load_extraction(args.tensors_dir)
    with open(args.membership) as f:
        membership = json.load(f)
    cluster_members = membership["cluster_members"]
    pooling_mode = membership["meta"].get("pooling_mode")

    # First-user text per conv_id (for TF-IDF baseline + top-3 examples) from
    # the battery instance labels (full first-user is in the slice; the battery
    # 'label' is the first 60 chars — load the slice if present for the full text).
    _, instances = load_battery_loose(args.battery)
    first_user_by_id: dict[str, str] = {
        inst["id"]: (inst["prefix_messages"][0]["content"] if inst["prefix_messages"] else "")
        for inst in instances
        if inst["id"] != SYNTHETIC_DEFAULT_ID
    }

    pairs = enumerate_pairs(cluster_members, args.per_cluster_floor)
    logger.info(
        "Scoring %d within-config cluster pairs (floor=%d)", len(pairs), args.per_cluster_floor
    )

    results = score_all(
        pairs,
        cluster_members,
        extraction,
        first_user_by_id,
        tuple(READ_LAYERS),
        args.n_perms,
        args.per_cluster_max,
        args.seed,
    )
    results["read_layers"] = list(READ_LAYERS)
    results["knn_k"] = KNN_K
    results["pooling_mode"] = pooling_mode
    results["n594_baselines"] = {
        "tfidf_word_overlap": 0.604,
        "length_only": 0.396,
        "curated_family_ceiling": 0.979,
    }
    results["metadata"] = reproducibility_metadata({"script": "issue617_score_separability"})

    # Checkpoint-per-phase: separability.json lands NOW, before any figure/upload.
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    w = results["winner"]
    logger.info(
        "Winner %s vs %s: residualized purity %.3f @ L%d, p_global=%.4f -> %s",
        w["cluster_a"],
        w["cluster_b"],
        w["residualized_purity_best"],
        w["best_layer"],
        w["p_global"],
        results["fire_rule"]["verdict"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
