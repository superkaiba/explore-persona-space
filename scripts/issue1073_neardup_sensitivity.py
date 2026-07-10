"""Near-duplicate clustering sensitivity for #1073's DV4/DV1 reads (0-GPU free analysis).

The committed #1073 reads handled EXACT duplicates only (string-normalized:
whitespace + casefold; 4,596 clusters / 5,000 rows, 477 rows in multi-row
clusters) via duplicate-clustered folds and a worst-case exclusion bound.
Near-duplicates (one-word template variants, punctuation-only edits) escape
exact-string matching. This script:

1. Regenerates the run's prompt list via the parent's own deterministic path
   (``issue779_collect.load_train_contexts``, LMSYS first-5000 first-user
   turns) and identity-gates it against the ``prompt_list_sha16`` recorded in
   ``eval_results/issue_1073/p0_probe.json`` (fail-loud on mismatch — the
   greedy raw-completion shard carries NO prompt field, so regen is the only
   prompt source; the P0 c_x alignment gate already validated this path).
2. Reproduces the committed exact-duplicate stats as a gate (4,596 / 477 / 89).
3. Clusters NEAR-duplicates: punctuation-stripped normalization, char 5-gram
   shingles, vectorized K=128 MinHash + 16x8 LSH banding for candidate pairs,
   EXACT Jaccard >= 0.9 verification on every candidate pair, single-linkage
   union-find (no O(N^2) full scan; the MinHash estimate is never the decision
   — exact Jaccard on candidates is).
4. Recomputes the DV4 median delta per read-out layer keeping one
   representative per near-dup cluster (lowest context index), plus the SAME
   worst-case percentile-band bound the exact-dup read used (median of full
   distribution bounded within quantiles 0.5 +/- m/(2n) under adversarial
   removal of m rows), at both the exact-dup mass and the near-dup mass.
5. Recomputes the DV1 pooled held-out R^2 gap (greedy - stoch1_old,
   1 - sum(SSE)/sum(SST)) per last_* read-out layer restricted to cluster
   representatives (headline: last_L14).

CONTENT SAFETY (binding): prompts are LMSYS user text. This script NEVER
prints, logs, or embeds prompt/completion text — all diagnostics are counts,
hashes, cluster-size histograms, and integer indices. The on-disk prompt cache
(``data/issue_1073/neardup_prompts_cache.json``) is a re-downloadable local
cache, never paged into any report.

Run:  uv run python scripts/issue1073_neardup_sensitivity.py
      uv run python scripts/issue1073_neardup_sensitivity.py --selftest
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import unicodedata
from collections import Counter
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import issue1073_common as I  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger("issue1073.neardup")

RES_DIR = I.PROJECT_ROOT / "eval_results" / "issue_1073"
OUT_PATH = RES_DIR / "neardup_sensitivity.json"
PROMPTS_CACHE = I.PROJECT_ROOT / "data" / "issue_1073" / "neardup_prompts_cache.json"

LMSYS_SOURCE = "lmsys/lmsys-chat-1m"
SHINGLE_K = 5
MINHASH_K = 128
LSH_BANDS = 16  # 16 bands x 8 rows; candidate-recall ~0.9999 at Jaccard 0.9
JACCARD_THRESHOLD = 0.9
MINHASH_SEED = 12345
BOOT_SEED = 0
N_BOOT = 1000


# ── prompt list (sha-gated deterministic regeneration) ────────────────────────


def load_run_prompts(expected_sha16: str, n: int) -> list[str]:
    """Return the run's exact prompt list, identity-gated on the recorded sha16.

    Cache-first (re-downloadable local cache keyed by the FULL sha256); on a
    miss, regenerate via the parent loader and HALT on any source/length/sha
    mismatch — a different prompt list makes row alignment unknowable.
    """
    if PROMPTS_CACHE.exists():
        with open(PROMPTS_CACHE) as f:
            blob = json.load(f)
        got = I.prompt_list_sha256(blob["prompts"])
        if (
            got == blob["prompt_list_sha256"]
            and got[:16] == expected_sha16
            and (len(blob["prompts"]) == n)
        ):
            logger.info("[prompts] cache hit (sha16=%s, n=%d)", got[:16], n)
            return blob["prompts"]
        logger.warning(
            "[prompts] cache stale (sha16=%s != %s) — regenerating", got[:16], expected_sha16
        )

    I._assert_gated_prompt_source_access(LMSYS_SOURCE)
    from issue779_collect import load_train_contexts

    prompts, source = load_train_contexts(n_contexts=n, smoke=False)
    # datasets streaming shutdown SIGABRT (#952, gotchas.md): sweep cyclic
    # streaming-IterableDataset remnants while the interpreter is healthy.
    gc.collect()
    if source != LMSYS_SOURCE:
        raise RuntimeError(
            f"prompt regeneration source mismatch: loader returned {source!r}, "
            f"bundle/p0 record {LMSYS_SOURCE!r} — a fallback corpus cannot reproduce "
            "the run's rows; HALT."
        )
    if len(prompts) != n:
        raise RuntimeError(f"prompt regeneration length mismatch: {len(prompts)} != {n}; HALT.")
    sha = I.prompt_list_sha256(prompts)
    if sha[:16] != expected_sha16:
        raise RuntimeError(
            f"regenerated prompt-list sha16 {sha[:16]} != p0-recorded {expected_sha16} "
            "— row alignment unknowable (source corpus drifted?); HALT."
        )
    PROMPTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    I.write_json_atomic(PROMPTS_CACHE, {"prompt_list_sha256": sha, "prompts": prompts})
    logger.info("[prompts] regenerated + cached (sha16=%s, n=%d)", sha[:16], n)
    return prompts


# ── near-duplicate clustering (MinHash/LSH candidates -> exact Jaccard) ───────


def neardup_normalize(p: str) -> str:
    """Near-dup normalization: casefold + punctuation->space + whitespace-collapse.

    Punctuation maps to a SPACE (not deletion) so hyphen/period variants align
    on word boundaries ("five-year-old." == "five year old").
    """
    s = p.casefold()
    s = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in s)
    return " ".join(s.split())


def _shingle_ids(text: str, vocab: dict[str, int]) -> np.ndarray:
    """Char 5-gram shingle ids (vocab-interned); short strings = one shingle."""
    if len(text) < SHINGLE_K:
        grams = [text]
    else:
        grams = [text[i : i + SHINGLE_K] for i in range(len(text) - SHINGLE_K + 1)]
    ids = np.empty(len(set(grams)), dtype=np.uint64)
    for j, g in enumerate(sorted(set(grams))):
        if g not in vocab:
            vocab[g] = len(vocab)
        ids[j] = vocab[g]
    return ids


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def neardup_cluster_ids(prompts: list[str]) -> tuple[np.ndarray, dict]:
    """Per-row near-dup cluster ids (single-linkage over exact-Jaccard>=0.9 edges).

    Rows sharing a normalized string are one NODE (exact under the near-dup
    normalization); MinHash+LSH proposes node candidate pairs; EXACT Jaccard on
    the 5-gram shingle sets decides. Returns (cluster_id_per_row, diagnostics).
    """
    norm = [neardup_normalize(p) for p in prompts]
    node_of_key: dict[str, int] = {}
    node_of_row = np.empty(len(prompts), dtype=np.int64)
    node_texts: list[str] = []
    for i, s in enumerate(norm):
        if s not in node_of_key:
            node_of_key[s] = len(node_texts)
            node_texts.append(s)
        node_of_row[i] = node_of_key[s]
    n_nodes = len(node_texts)

    vocab: dict[str, int] = {}
    shingles = [_shingle_ids(t, vocab) for t in node_texts]
    n_shingles_total = int(sum(len(s) for s in shingles))

    # Vectorized MinHash: multiply-shift permutation family on uint64 (wraps
    # mod 2^64); the sketch only PROPOSES candidates — exact Jaccard decides.
    rng = np.random.default_rng(MINHASH_SEED)
    a = rng.integers(1, 2**63, size=MINHASH_K, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**63, size=MINHASH_K, dtype=np.uint64)
    sigs = np.empty((n_nodes, MINHASH_K), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i, ids in enumerate(shingles):
            sigs[i] = (a[:, None] * ids[None, :] + b[:, None]).min(axis=1)

    rows_per_band = MINHASH_K // LSH_BANDS
    candidates: set[tuple[int, int]] = set()
    max_bucket = 0
    for band in range(LSH_BANDS):
        block = sigs[:, band * rows_per_band : (band + 1) * rows_per_band]
        buckets: dict[bytes, list[int]] = {}
        for i in range(n_nodes):
            buckets.setdefault(block[i].tobytes(), []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            max_bucket = max(max_bucket, len(members))
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    candidates.add((members[x], members[y]))

    sets = [set(s.tolist()) for s in shingles]
    uf = _UnionFind(n_nodes)
    n_edges = 0
    for i, j in candidates:
        inter = len(sets[i] & sets[j])
        union = len(sets[i]) + len(sets[j]) - inter
        if union > 0 and inter / union >= JACCARD_THRESHOLD:
            uf.union(i, j)
            n_edges += 1

    root_of_node = np.array([uf.find(i) for i in range(n_nodes)], dtype=np.int64)
    _, cluster_of_node = np.unique(root_of_node, return_inverse=True)
    cluster_of_row = cluster_of_node[node_of_row]
    diag = {
        "n_nodes_unique_normalized": n_nodes,
        "n_shingles_total": n_shingles_total,
        "shingle_vocab_size": len(vocab),
        "n_candidate_pairs": len(candidates),
        "n_accepted_edges": n_edges,
        "max_lsh_bucket": max_bucket,
    }
    return cluster_of_row, diag


def cluster_stats(cluster_of_row: np.ndarray) -> dict:
    """Cluster-count stats in the committed duplicate_stats convention."""
    _, counts = np.unique(cluster_of_row, return_counts=True)
    multi = counts[counts > 1]
    hist = Counter(counts.tolist())
    return {
        "n_rows": int(cluster_of_row.size),
        "n_clusters": int(counts.size),
        "n_rows_in_multirow_clusters": int(multi.sum()),
        "n_excess_rows_beyond_representatives": int(cluster_of_row.size - counts.size),
        "duplicate_row_fraction": float(multi.sum() / max(cluster_of_row.size, 1)),
        "largest_cluster": int(counts.max()) if counts.size else 0,
        "cluster_size_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
    }


def representatives(cluster_of_row: np.ndarray) -> np.ndarray:
    """One representative per cluster: the LOWEST row index (deterministic)."""
    seen: dict[int, int] = {}
    for i, c in enumerate(cluster_of_row.tolist()):
        seen.setdefault(c, i)
    return np.array(sorted(seen.values()), dtype=np.int64)


# ── DV recomputes ─────────────────────────────────────────────────────────────


def worstcase_band(values: np.ndarray, m_removed: int) -> dict:
    """Median bound under ADVERSARIAL removal of m rows: quantiles 0.5 +/- m/(2n)."""
    n = values.size
    qlo, qhi = 0.5 - m_removed / (2 * n), 0.5 + m_removed / (2 * n)
    lo, hi = float(np.quantile(values, qlo)), float(np.quantile(values, qhi))
    return {
        "m_removed": int(m_removed),
        "q_lo": qlo,
        "q_hi": qhi,
        "band_lo": lo,
        "band_hi": hi,
        "band_all_positive": bool(lo > 0),
    }


def dv4_sensitivity(ta: dict, reps: np.ndarray, m_exact: int, m_near: int) -> dict:
    """Per-layer DV4 median delta: full vs near-dup-representative, + worst-case bands."""
    rng = np.random.default_rng(BOOT_SEED)
    out: dict = {}
    for lk, entry in ta["per_layer"].items():
        d = np.asarray(entry["dv4_delta_ctx"], dtype=np.float64)
        dr = d[reps]
        idx = rng.integers(0, dr.size, size=(N_BOOT, dr.size))
        boot_med = np.median(dr[idx], axis=1)
        out[lk] = {
            "median_full": float(np.median(d)),
            "median_neardup_representatives": float(np.median(dr)),
            "n_representatives": int(dr.size),
            "rep_median_boot_ci95": [
                float(np.quantile(boot_med, 0.025)),
                float(np.quantile(boot_med, 0.975)),
            ],
            "worstcase_band_exact_dup": worstcase_band(d, m_exact),
            "worstcase_band_near_dup": worstcase_band(d, m_near),
        }
    return out


def dv1_sensitivity(hr: dict, reps: np.ndarray) -> dict:
    """Pooled R^2 gap (greedy - stoch1_old) per last_* layer: full vs representatives."""
    rng = np.random.default_rng(BOOT_SEED)
    out: dict = {}
    for lk, arms in hr["readout_percontext_last"].items():
        entry: dict = {}
        gaps: dict[str, dict[str, np.ndarray]] = {}
        for arm in ("greedy", "stoch1_old"):
            sse = np.asarray(arms[arm]["sse"], dtype=np.float64)
            sst = np.asarray(arms[arm]["sst"], dtype=np.float64)
            gaps[arm] = {"sse": sse, "sst": sst}
            entry[f"r2_pooled_full_{arm}"] = float(1 - sse.sum() / sst.sum())
            entry[f"r2_pooled_reps_{arm}"] = float(1 - sse[reps].sum() / sst[reps].sum())
        entry["gap_full"] = entry["r2_pooled_full_greedy"] - entry["r2_pooled_full_stoch1_old"]
        entry["gap_neardup_representatives"] = (
            entry["r2_pooled_reps_greedy"] - entry["r2_pooled_reps_stoch1_old"]
        )
        # Paired bootstrap over representatives (context resamples, shared index).
        idx = rng.integers(0, reps.size, size=(N_BOOT, reps.size))
        boot = {}
        for arm in ("greedy", "stoch1_old"):
            sse_r, sst_r = gaps[arm]["sse"][reps], gaps[arm]["sst"][reps]
            boot[arm] = 1 - sse_r[idx].sum(axis=1) / sst_r[idx].sum(axis=1)
        gap_draws = boot["greedy"] - boot["stoch1_old"]
        entry["gap_reps_boot_ci95"] = [
            float(np.quantile(gap_draws, 0.025)),
            float(np.quantile(gap_draws, 0.975)),
        ]
        entry["n_representatives"] = int(reps.size)
        out[lk] = entry
    return out


# ── self-test (innocuous fixtures; no corpus access) ──────────────────────────

_LONG = (
    "please write a detailed step by step tutorial explaining how to bake "
    "sourdough bread at home for a complete beginner"
)
_SELFTEST_PROMPTS = [
    "Write a poem about the ocean.",
    "write a poem about the ocean",  # exact dup of [0] only after punct-strip
    "Write a poem about the ocean!!",  # near dup (punct-only)
    "Write a poem about the mountains.",  # J~0.58 vs [0] -> NOT merged
    "Explain photosynthesis to a five year old.",
    "Explain photosynthesis to a five-year-old.",  # hyphen variant -> merges
    "What is the capital of France?",
    "hi",
    "hi",  # the only EXACT dup pair (whitespace+casefold)
    "Hi!",
    _LONG,
    _LONG + "s",  # genuine fuzzy pair (distinct nodes, J~0.99) -> MinHash edge
]


def selftest() -> int:
    exact = I.duplicate_cluster_ids(_SELFTEST_PROMPTS)
    ex_stats = cluster_stats(exact)
    assert ex_stats["n_clusters"] == 11, ex_stats
    assert ex_stats["n_rows_in_multirow_clusters"] == 2, ex_stats
    near, diag = neardup_cluster_ids(_SELFTEST_PROMPTS)
    st = cluster_stats(near)
    # ocean triplet, photosynthesis pair, hi triplet, and the fuzzy long pair
    # merge; mountains stays separate (J ~ 0.58 < 0.9).
    assert st["n_clusters"] == 6, (st, diag)
    assert st["n_rows_in_multirow_clusters"] == 10, st
    assert near[0] == near[1] == near[2] != near[3], near.tolist()
    assert near[4] == near[5], near.tolist()
    assert near[7] == near[8] == near[9], near.tolist()
    assert near[10] == near[11], near.tolist()
    assert diag["n_accepted_edges"] >= 1, diag  # the fuzzy pair went through LSH
    reps = representatives(near)
    assert reps.tolist() == sorted(reps.tolist()) and reps.size == 6
    # worst-case band arithmetic: m=2 of n=10 -> quantiles 0.4/0.6.
    band = worstcase_band(np.arange(10, dtype=np.float64), 2)
    assert (band["q_lo"], band["q_hi"]) == (0.4, 0.6), band
    print(json.dumps({"selftest": "PASS", "near_stats": st, "diag": diag}))
    return 0


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--selftest", action="store_true", help="run fixture self-test only (no corpus access)"
    )
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.selftest:
        return selftest()

    with open(RES_DIR / "p0_probe.json") as f:
        p0 = json.load(f)
    expected_sha16 = p0["prompts_provenance"]["prompt_list_sha16"]
    committed_dup = p0["duplicate_stats"]
    n = int(committed_dup["n_contexts"])

    prompts = load_run_prompts(expected_sha16, n)

    # Gate: reproduce the committed exact-duplicate stats byte-for-byte.
    exact_ids = I.duplicate_cluster_ids(prompts)
    exact_stats = cluster_stats(exact_ids)
    for k in ("n_clusters", "n_rows_in_multirow_clusters", "largest_cluster"):
        if exact_stats[k] != committed_dup[k]:
            raise RuntimeError(
                f"exact-dup gate FAIL: {k}={exact_stats[k]} != committed {committed_dup[k]}"
            )
    logger.info("[gate] exact-dup stats reproduce the committed p0 values")

    near_ids, diag = neardup_cluster_ids(prompts)
    near_stats = cluster_stats(near_ids)
    reps = representatives(near_ids)
    m_exact = int(committed_dup["n_rows_in_multirow_clusters"])
    m_near = int(near_stats["n_rows_in_multirow_clusters"])

    # Punct-strip-only exact clustering (decomposition: normalization vs fuzzy).
    norm_keys = [neardup_normalize(p) for p in prompts]
    _, norm_ids = np.unique(np.array(norm_keys, dtype=object), return_inverse=True)
    punct_stats = cluster_stats(norm_ids.astype(np.int64))

    with open(RES_DIR / "target_agreement.json") as f:
        ta = json.load(f)
    with open(RES_DIR / "heldout_recon_percontext.json") as f:
        hr = json.load(f)
    assert len(ta["per_layer"]["L14"]["dv4_delta_ctx"]) == n, "row-count mismatch vs DV4"

    result = {
        "metadata": I.reproducibility_metadata({"script": "issue1073_neardup_sensitivity"}),
        "params": {
            "shingle_k": SHINGLE_K,
            "minhash_k": MINHASH_K,
            "lsh_bands": LSH_BANDS,
            "jaccard_threshold": JACCARD_THRESHOLD,
            "minhash_seed": MINHASH_SEED,
            "clustering": "single-linkage over exact-Jaccard>=threshold candidate edges",
            "normalization": "whitespace-collapse + casefold + unicode-punctuation strip",
            "representative_rule": "lowest context index per cluster",
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "prompt_list_sha16": expected_sha16,
        },
        "exact_dup_stats": exact_stats,
        "punct_strip_exact_stats": punct_stats,
        "neardup_stats": near_stats,
        "neardup_diagnostics": diag,
        "neardup_excess_mass_beyond_exact": m_near - m_exact,
        "dv4": dv4_sensitivity(ta, reps, m_exact, m_near),
        "dv1_r2_gap": dv1_sensitivity(hr, reps),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    I.write_json_atomic(out_path, result)
    logger.info("[done] wrote %s", out_path)

    headline = {
        "exact_477": m_exact,
        "near_dup_mass": m_near,
        "near_dup_excess_beyond_exact": m_near - m_exact,
        "n_representatives": int(reps.size),
        "dv4_rep_medians": {
            lk: round(v["median_neardup_representatives"], 6) for lk, v in result["dv4"].items()
        },
        "dv4_neardup_band_all_positive": all(
            v["worstcase_band_near_dup"]["band_all_positive"] for v in result["dv4"].values()
        ),
        "L14_gap_full": round(result["dv1_r2_gap"]["last_L14"]["gap_full"], 6),
        "L14_gap_reps": round(result["dv1_r2_gap"]["last_L14"]["gap_neardup_representatives"], 6),
    }
    print(json.dumps(headline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
