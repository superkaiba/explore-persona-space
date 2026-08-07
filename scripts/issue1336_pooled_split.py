#!/usr/bin/env python3
"""Phase C_pool — pooled-multidataset cross-corpus split (plan v15 §4).

Prepares the pooled_split_v3 assignment table consumed by Phase FIT_pool /
LAD_pool. Steps (fail-loud in order):

  1. Measure the 5-way (base, sft, dpo, rlvr, rlvr_long) x 7-corpus
     `raw_completions/generation/<slug>/<corpus>/answers.jsonl`
     prompt_id (conv_id) intersection FIRST — plan §4 "5-way intersection
     sizing" + CONCERN M1: measure BEFORE dedup/embed/split so the count
     is grounded on the pinned generation shards, never on a downstream
     artifact.
  2. Load the 7 pinned corpora_v2 rows (via the shared
     issue1336_stage_corpora reader — local-first, HF fallback), sha256
     the raw prompt TEXT and drop cross-corpus dedup collisions in the
     order: lmsys23k -> gsm8k_train_full -> math7500 -> if11k -> uf11k ->
     sft11k -> gsm8k_test1319 (plan §4 Phase C_pool).
  3. Embed all deduped prompts once via sentence-transformers/all-mpnet-
     base-v2 with the revision pinned via HfApi().repo_info at
     invocation time (CONCERN M3), recorded in the manifest.
  4. K-means k=50 (seed 1336) on the union embeddings; assign whole
     clusters 80/20 to train/test at ratio +/- 2% and partition the train
     side into 5 folds preserving cluster structure.
  5. Persist the assignment to
     ``analysis_tensors/pooled_split_v3/split_manifest.json`` with fields:
     ``pinned_revisions`` (mpnet + generation revision per model x corpus:
     wave-1 pin for lmsys/gsm8k_train stems, resolved main for v2 shards), ``n_kept_pre_dedup``, ``n_kept_post_dedup``,
     ``per_corpus_kept`` (before AND after dedup), ``n_clusters``,
     ``cluster_to_corpora_counts`` (per-cluster corpus histogram),
     ``train_test_by_cluster``, ``pooled_folds_by_cluster``, plus the
     5-way intersection measurement + the row-id assignment lists.
  6. Optionally upload the manifest to the HF data repo under
     ``issue1336_rlvr_ladder/analysis_tensors/pooled_split_v3/`` (adds
     ``--upload`` mirroring the plan's exact workload command).

Assertions (fail loud on violation, non-zero exit):

  - every cluster contains prompts from >= 3 corpora (fail if any
    cluster is corpus-locked);
  - per-corpus test-side share >= 15%;
  - per-corpus keep-rate >= 0.99 of the round-3 per-corpus keep-rate
    (HALT + log the sha collision surface on violation);
  - cross-corpus dedup drops < 500 (WARN only — sanity check).

Smoke (``--smoke``) end-to-end exercise: substitutes the pinned local
smoke corpora set (SMOKE_CORPORA_V2 = ("lmsys23k",)) at tiny N so the
pipeline runs on the VM without HF traffic; the smoke NEVER uploads and
writes under data/issue_1336/pooled_split_v3_smoke.

Every ``__main__`` invocation exits explicitly (``sys.exit(0)`` on the
success path) to sidestep the PyGILState_Release atexit race that killed
scripts/issue1689 phases (gotchas.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Guarantee explore_persona_space is importable in script mode (parents[1] is
# repo root when invoked via `uv run python scripts/issue1336_pooled_split.py`).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.orchestrate import env  # noqa: E402
from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

# scripts/ on sys.path so we can reuse the corpus reader without duplicating.
_SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

import issue1336_stage_corpora as stage_corpora  # noqa: E402

logger = logging.getLogger("issue1336_pooled_split")

# ---------------------------------------------------------------------------
# Config constants (plan §11 additions)
# ---------------------------------------------------------------------------
MPNET_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
POOLED_SPLIT_SEED = 1336
POOLED_K = 50
POOLED_N_FOLDS = 5
POOLED_TEST_RATIO = 0.20
POOLED_TEST_TOL = 0.02

# Cross-corpus dedup order (first occurrence wins, later ones drop).
DEDUP_ORDER: tuple[str, ...] = (
    "lmsys23k",
    "gsm8k_train_full",
    "math7500",
    "if11k",
    "uf11k",
    "sft11k",
    "gsm8k_test1319",
)

# Assertions
CLUSTER_MIN_CORPORA = 3
PER_CORPUS_TEST_SHARE_MIN = 0.15
PER_CORPUS_KEEP_RATE_MIN_FRAC = 0.99
DEDUP_DROP_WARN_THRESHOLD = 500

# 5 checkpoints × 7 corpora shards (plan §4 5-way intersection sizing).
INTERSECTION_MODELS: tuple[str, ...] = cm.PRIMARY_LADDER + ("rlvr_long",)
INTERSECTION_CORPORA: tuple[str, ...] = tuple(cm.V2_CORPORA.keys())

# Output paths.
DATA_ROOT = _REPO_ROOT / "data" / "issue_1336"
POOLED_OUT_SUBDIR = "pooled_split_v3"
POOLED_OUT_SUBDIR_SMOKE = "pooled_split_v3_smoke"

# HF prefixes for the answers.jsonl shards. ALL generation shards — wave-1
# AND round-3 v2 — live under `raw_completions/generation/<model>/<stem>/`
# (round-4 live probe: `raw_completions/` has exactly one child, `generation`;
# a `generation_v2/` prefix 404s). Wave-1 vs v2 is a REVISION split, not a
# prefix split: wave-1 stems pin WAVE1_HF_REV, v2 shards resolve main.
_WAVE1_CORPORA: frozenset[str] = frozenset({"lmsys5k", "gsm8k_train5k", "gsm8k_test1319"})


@dataclass
class SplitContext:
    smoke: bool
    upload: bool
    corpora: tuple[str, ...]
    out_root: Path
    hf_prefix_out: str
    # Filled during run.
    pinned_mpnet_revision: str | None = None
    pinned_generation_revisions: dict[str, str] = field(default_factory=dict)
    generation_stems: dict[str, str] = field(default_factory=dict)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()


def _hub_helpers():
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    return HfApi(), hf_hub_download, hub


def _round3_per_corpus_keep_rate(corpora: tuple[str, ...]) -> dict[str, float]:
    """Round-3 per-corpus keep-rate reference for the keep-rate floor.

    Reads ``eval_results/issue_1336/corpora_v2_stage_meta.json`` when present
    (round-3 stage-corpora recorded per-corpus keep rates there). A MISSING
    file falls back to a 1.0 reference for every corpus — the STRICTEST floor
    (current keep-rate must be >= 0.99 x 1.0), never a relaxation — with a
    WARN documenting the fallback. A PRESENT-but-unparseable meta raises
    (fail-loud): a corrupt reference must never silently rewrite the floor.
    Slugs absent from a parsed meta default to the same strictest 1.0.
    """
    meta_path = _REPO_ROOT / "eval_results" / "issue_1336" / "corpora_v2_stage_meta.json"
    fallback = {slug: 1.0 for slug in corpora}
    if not meta_path.exists():
        logger.warning(
            "[pool] no round-3 keep-rate meta at %s — per-corpus keep-rate floor defaults to 1.0",
            meta_path,
        )
        return fallback
    # Present-but-corrupt fails loud (json.loads raises) — the crash IS the
    # signal; a swallowed parse error would silently substitute the fallback
    # reference for the recorded round-3 rates.
    meta = json.loads(meta_path.read_text())
    out: dict[str, float] = {}
    for slug in corpora:
        rate = None
        entry = meta.get(slug) if isinstance(meta, dict) else None
        if isinstance(entry, dict):
            rate = entry.get("keep_rate") or entry.get("v2_keep_rate")
        if isinstance(rate, (int, float)) and 0.0 <= float(rate) <= 1.0:
            out[slug] = float(rate)
        else:
            out[slug] = 1.0
    return out


def _resolve_mpnet_revision(api) -> str:
    from explore_persona_space.orchestrate import hub

    info = hub.retry_transient(
        lambda: api.repo_info(repo_id=MPNET_MODEL_ID, repo_type="model", revision="main"),
        what=f"repo_info {MPNET_MODEL_ID}",
    )
    sha = getattr(info, "sha", None)
    assert sha, f"repo_info returned no sha for {MPNET_MODEL_ID}"
    return sha


def _resolve_generation_prefix(slug: str) -> tuple[str, str]:
    """Return (subprefix, stem) for a corpus_v2 slug — subprefix is ALWAYS
    ``generation``.

    Round-4 live probe: `raw_completions/` on the data repo has exactly ONE
    child, `generation` (`generation_v2/` 404s); the writer
    (issue1336_gen_answers.py:326) uploads every v2 cell to
    `raw_completions/generation/{slug}/{cell}`. Wave-1 lmsys/gsm8k paths use
    the wave-1 stem (differs from the v2 slug) at the WAVE1_HF_REV pin; v2
    corpora use the slug itself at the resolved main revision (the caller
    keys the revision on ``stem in _WAVE1_CORPORA``).
    """
    if slug == "lmsys23k":
        # Wave-1 generations covered prompt_idx 0..4999 under `lmsys5k`.
        return ("generation", "lmsys5k")
    if slug == "gsm8k_train_full":
        return ("generation", "gsm8k_train5k")
    if slug == "gsm8k_test1319":
        return ("generation", "gsm8k_test1319")
    return ("generation", slug)


def _download_answers_jsonl(
    api, hf_hub_download, hub, prefix: str, revision: str, dest_dir: Path
) -> Path:
    """Download answers.jsonl (or its shard manifest) for a single
    (model, corpus) generation prefix. Returns the local answers.jsonl path
    (reassembled from shards if needed).
    """
    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: lambda is retried by hub.retry_transient
            api.list_repo_tree(
                cm.HF_DATA_REPO,
                path_in_repo=prefix,
                repo_type="dataset",
                revision=revision,
                recursive=True,
            )
        ),
        what=f"pooled_split answers tree {prefix}@{revision}",
    )
    files = [e.path for e in entries if hasattr(e, "size")]
    assert files, f"no answers files under {prefix} on {cm.HF_DATA_REPO}@{revision}"

    dest_dir.mkdir(parents=True, exist_ok=True)
    for rel in sorted(files):
        # Only stage answers.jsonl + its shard manifest + shard parts. The
        # gen-phase shard contract is `answers.shard{NN}.jsonl` (per
        # `scripts/issue1336_gen_answers.py::_split_answers_for_upload`),
        # NOT `answers.jsonl.part*`. Accept both spellings so any future
        # shard-naming change stays reasonably tolerant.
        base = Path(rel).name
        if not (
            base == "answers.jsonl"
            or base == "answers.manifest.json"
            or base.startswith("answers.shard")
            or base.startswith("answers.jsonl.part")
        ):
            continue
        local_target = dest_dir / base
        if local_target.exists() and local_target.stat().st_size > 0:
            continue
        hub.retry_transient(
            lambda r=rel: hf_hub_download(
                repo_id=cm.HF_DATA_REPO,
                repo_type="dataset",
                filename=r,
                revision=revision,
                local_dir=dest_dir,
            ),
            what=f"pooled_split download {rel}",
        )
        # hf_hub_download mirrors the hub path; move flat.
        mirrored = dest_dir / rel
        if mirrored.exists() and mirrored != local_target:
            local_target.parent.mkdir(parents=True, exist_ok=True)
            mirrored.rename(local_target)

    # Prefer single answers.jsonl if present; else reassemble.
    single = dest_dir / "answers.jsonl"
    if single.exists() and single.stat().st_size > 0:
        return single
    manifest = dest_dir / "answers.manifest.json"
    assert manifest.exists(), (
        f"neither answers.jsonl nor answers.manifest.json under {dest_dir} for {prefix}"
    )
    m = json.loads(manifest.read_text())
    tmp = dest_dir / "answers.jsonl.tmp"
    h = hashlib.sha256()
    with tmp.open("wb") as out:
        for part in m["parts"]:
            data = (dest_dir / part).read_bytes()
            h.update(data)
            out.write(data)
    assert h.hexdigest() == m["total_sha256"], (
        f"reassembled answers.jsonl sha mismatch under {dest_dir}"
    )
    tmp.replace(single)
    return single


def _read_answers_conv_ids(
    path: Path, *, min_idx: int | None = None, max_idx: int | None = None
) -> set[str]:
    """KEPT-row conv ids (canonical ``s<prompt_idx>``) from one answers.jsonl.

    The generation writer emits ``prompt_idx`` + ``kept`` per row (there is NO
    ``conv_id`` field in answers rows); every downstream capture/fit keys rows
    as ``s<prompt_idx>`` (issue1336_extract_turnstore.py convention). Only
    KEPT rows are ever captured, so only kept rows can count toward the
    intersection — an unkept row in the manifest would break the pooled
    row-coverage contract at fit time (plan v15 §3).

    ``min_idx``/``max_idx`` bound the half-open ``[min_idx, max_idx)`` window
    for the concat corpora's two generation halves (wave-1 stem rows below the
    boundary, v2 extension rows at/above it — ``read_offpolicy_rows`` twin).
    """
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("kept"):
                continue
            idx = row.get("prompt_idx")
            assert idx is not None, f"{path}: kept answers row lacks prompt_idx"
            idx = int(idx)
            if min_idx is not None and idx < min_idx:
                continue
            if max_idx is not None and idx >= max_idx:
                continue
            ids.add(f"s{idx}")
    return ids


def measure_5way_intersection(ctx: SplitContext) -> tuple[dict[str, Any], dict[str, set[str]]]:
    """CONCERN M1 + plan §5 n_pool: the 5-way (model x corpus) KEPT-prompt
    intersection, measured BEFORE dedup/embed/split AND applied as the row
    filter (plan v15 §5: n_pool IS the 5-way unflagged intersection — the
    pooled fits can only serve rows every checkpoint has kept text for, so a
    measurement-only probe breaks the §3 row-coverage contract at fit time).

    Returns ``(summary, ids_by_corpus)``: the manifest-recorded summary dict
    plus the per-corpus intersection id sets (canonical ``s<prompt_idx>``).

    Concat corpora (``cm.V2_CONCAT_SOURCES``) have TWO generation halves per
    model — wave-1 stem rows below the boundary + v2 extension rows at/above
    it (the ``read_offpolicy_rows`` convention) — unioned per model before
    intersecting.

    Smoke: no HF traffic — the probe reads the LOCAL smoke gen fixtures
    (``data/issue_1336/gen_smoke/<model>/<corpus>/answers.jsonl``, extension
    half only: the smoke fit path structurally skips the wave-1 concat
    loader, so extension kept rows ARE the serveable set) over
    ``cm.SMOKE_MODELS``. Fail-loud when the fixtures are missing (run
    ``issue1336_smoke_fixtures.py gen`` + ``gen-v2`` first — the dispatcher's
    c_pool smoke leg does).
    """
    if ctx.smoke:
        gen_smoke = DATA_ROOT / "gen_smoke"
        per_corpus_intersection: dict[str, int] = {}
        ids_by_corpus: dict[str, set[str]] = {}
        total_union = 0
        for corpus in ctx.corpora:
            model_sets: dict[str, set[str]] = {}
            for model in cm.SMOKE_MODELS:
                path = gen_smoke / model / corpus / "answers.jsonl"
                assert path.exists(), (
                    f"smoke intersection probe: {path} missing — generate the smoke gen "
                    "fixtures first (issue1336_smoke_fixtures.py gen + gen-v2; the "
                    "dispatcher's c_pool smoke leg runs both)"
                )
                ids = _read_answers_conv_ids(path)
                assert ids, f"smoke intersection probe: no kept rows in {path}"
                model_sets[model] = ids
            inter = set.intersection(*model_sets.values())
            assert inter, f"empty smoke kept-intersection for {corpus}"
            ids_by_corpus[corpus] = inter
            per_corpus_intersection[corpus] = len(inter)
            total_union += len(inter)
            logger.info(
                "[pool] smoke-local kept-intersection %s: %d prompts (per-model sizes: %s)",
                corpus,
                len(inter),
                {m: len(s) for m, s in model_sets.items()},
            )
        return {
            "mode": "smoke-local",
            "models": list(cm.SMOKE_MODELS),
            "corpora": list(ctx.corpora),
            "per_corpus_5way_intersection": per_corpus_intersection,
            "total_5way_union": total_union,
        }, ids_by_corpus
    api, dl, hub = _hub_helpers()
    per_corpus_intersection = {}
    ids_by_corpus = {}
    total_union = 0
    # Resolve one main sha for the v2 shards (used by every non-wave-1 corpus).
    from explore_persona_space.orchestrate import hub as hub_mod

    v2_main_sha = hub_mod.retry_transient(
        lambda: api.repo_info(repo_id=cm.HF_DATA_REPO, repo_type="dataset", revision="main").sha,
        what="pooled_split repo_info explore-persona-space-data@main",
    )
    v2_main_sha = str(v2_main_sha)

    def _half_ids(
        model: str, subprefix: str, stem: str, revision: str, *, min_idx=None, max_idx=None
    ) -> set[str]:
        prefix = f"{cm.HF_PREFIX_1336}/raw_completions/{subprefix}/{model}/{stem}"
        work_dir = ctx.out_root / "gen_probe" / model / stem
        answers_path = _download_answers_jsonl(api, dl, hub, prefix, revision, work_dir)
        ids = _read_answers_conv_ids(answers_path, min_idx=min_idx, max_idx=max_idx)
        assert ids, f"no kept prompt ids under {prefix} @ {revision}"
        return ids

    for corpus in ctx.corpora:
        model_sets = {}
        for model in INTERSECTION_MODELS:
            subprefix, stem = _resolve_generation_prefix(corpus)
            # Wave-1 vs v2 is a REVISION split (same `generation/` prefix):
            # wave-1 stems pin WAVE1_HF_REV, v2 shards read the resolved main.
            revision = cm.WAVE1_HF_REV if stem in _WAVE1_CORPORA else v2_main_sha
            if corpus in cm.V2_CONCAT_SOURCES:
                # Two halves: wave-1 stem below the boundary (+ pin), v2
                # extension at/above it (read_offpolicy_rows convention) —
                # the extension lives under `generation/<model>/<corpus>`.
                boundary = cm.V2_CONCAT_BOUNDARY[corpus]
                w1_ids = _half_ids(model, subprefix, stem, revision, max_idx=boundary)
                ext_ids = _half_ids(model, "generation", corpus, v2_main_sha, min_idx=boundary)
                ids = w1_ids | ext_ids
                ctx.pinned_generation_revisions[f"{model}::{corpus}::ext"] = v2_main_sha
                ctx.generation_stems[f"{model}::{corpus}::ext"] = corpus
            else:
                ids = _half_ids(model, subprefix, stem, revision)
            model_sets[model] = ids
            # Record the pin.
            ctx.pinned_generation_revisions[f"{model}::{corpus}"] = revision
            ctx.generation_stems[f"{model}::{corpus}"] = stem
        # 5-way intersection = intersection over models.
        inter = set.intersection(*model_sets.values())
        assert inter, f"empty 5-way kept-intersection for {corpus}"
        ids_by_corpus[corpus] = inter
        per_corpus_intersection[corpus] = len(inter)
        total_union += len(inter)
        logger.info(
            "[pool] 5-way kept-intersection %s: %d prompts (per-model sizes: %s)",
            corpus,
            len(inter),
            {m: len(s) for m, s in model_sets.items()},
        )
    return {
        "mode": "full",
        "models": list(INTERSECTION_MODELS),
        "corpora": list(ctx.corpora),
        "per_corpus_5way_intersection": per_corpus_intersection,
        "total_5way_union": total_union,
        "v2_main_revision": v2_main_sha,
    }, ids_by_corpus


def load_corpora_rows(ctx: SplitContext) -> dict[str, list[dict]]:
    """Load rows for every corpus in ctx.corpora via the shared reader.

    Returns {slug: [{prompt_idx, prompt, ...}, ...]} preserving row order.
    """
    out: dict[str, list[dict]] = {}
    for slug in ctx.corpora:
        rows = stage_corpora.load_v2_corpus_rows(slug, smoke=ctx.smoke)
        assert rows, f"empty rows for corpus {slug}"
        # Basic sanity: every row must have a 'prompt' string.
        for i, r in enumerate(rows):
            p = r.get("prompt")
            assert isinstance(p, str) and p, f"{slug} row {i}: prompt missing or non-string"
        out[slug] = rows
        logger.info("[pool] loaded %s: %d rows", slug, len(rows))
    return out


def cross_corpus_dedup(
    rows_by_corpus: dict[str, list[dict]],
    order: tuple[str, ...],
) -> tuple[dict[str, list[dict]], list[dict[str, Any]]]:
    """Drop cross-corpus sha256(prompt) collisions in the given order.

    Returns (kept_by_corpus, dropped_records). ``dropped_records`` carries the
    sha collision surface for downstream logging when the keep-rate assertion
    would HALT.
    """
    seen: dict[str, tuple[str, int]] = {}
    kept: dict[str, list[dict]] = {slug: [] for slug in order if slug in rows_by_corpus}
    dropped: list[dict[str, Any]] = []
    for slug in order:
        if slug not in rows_by_corpus:
            continue
        for row in rows_by_corpus[slug]:
            prompt = row["prompt"]
            sha = _sha256_hex(prompt)
            if sha in seen:
                first_slug, first_idx = seen[sha]
                dropped.append(
                    {
                        "sha256": sha,
                        "dropped_corpus": slug,
                        "dropped_prompt_idx": row.get("prompt_idx"),
                        "first_corpus": first_slug,
                        "first_prompt_idx": first_idx,
                    }
                )
                continue
            seen[sha] = (slug, row.get("prompt_idx"))
            row_with_key = dict(row)
            row_with_key["prompt_sha"] = sha
            row_with_key["corpus"] = slug
            kept[slug].append(row_with_key)
    return kept, dropped


def _load_sentence_transformer(revision: str):
    """Load sentence-transformers/all-mpnet-base-v2 at a pinned revision.

    Prefer the `sentence_transformers` package; fall back to raw transformers if
    absent (rare on this VM but keep the fail-loud path clean).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - runtime env
        raise RuntimeError(
            "sentence-transformers not installed — cannot embed for pooled split"
        ) from exc

    # Older sentence-transformers versions honor `revision=` via kwargs.
    return SentenceTransformer(MPNET_MODEL_ID, revision=revision)


def embed_prompts(prompts: list[str], revision: str, smoke: bool) -> "list[list[float]]":
    """Return fp32 embeddings as a nested list (JSON-serializable-friendly)."""
    if smoke:
        # Cheap deterministic hash-based embedding for smoke — 32-dim toy vector
        # keyed on prompt sha. Enough to exercise the k-means/split code path
        # without instantiating the full model.
        import numpy as np

        vecs = []
        for p in prompts:
            digest = hashlib.sha256(p.encode("utf-8")).digest()
            arr = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
            # Pad to 32-d by tiling.
            v = np.tile(arr, 2)[:32]
            vecs.append(v.tolist())
        return vecs
    model = _load_sentence_transformer(revision)
    logger.info("[pool] embedding %d prompts via %s @ %s", len(prompts), MPNET_MODEL_ID, revision)
    vecs = model.encode(prompts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    return [row.tolist() for row in vecs]


def kmeans_assign(vectors: list[list[float]], k: int, seed: int) -> list[int]:
    """Run sklearn KMeans k=k with a fixed seed. Returns cluster labels."""
    import numpy as np
    from sklearn.cluster import KMeans

    X = np.asarray(vectors, dtype=np.float32)
    # Clamp k to n_samples for tiny smoke runs.
    k_eff = min(k, X.shape[0]) if X.shape[0] > 0 else k
    km = KMeans(n_clusters=k_eff, random_state=seed, n_init="auto")
    labels = km.fit_predict(X)
    return [int(x) for x in labels.tolist()]


def cluster_train_test_split(
    labels: list[int],
    seed: int,
    test_ratio: float,
    test_tol: float,
) -> tuple[dict[int, str], dict[int, int]]:
    """Assign whole clusters to 'train'/'test' at ratio +/- test_tol.

    Also partition the train-side clusters across POOLED_N_FOLDS folds (per
    plan §4). Returns (train_test_by_cluster, pooled_folds_by_cluster).
    Fail-loud when the greedy whole-cluster packing cannot land the realized
    test share inside [test_ratio - test_tol, test_ratio + test_tol] (the
    plan's 80/20 +/- 2% contract) — never a silent overshoot.
    """
    cluster_ids = sorted(set(labels))
    n = len(labels)
    rng = random.Random(seed)
    shuffled = list(cluster_ids)
    rng.shuffle(shuffled)
    # Greedy pack clusters into 'test' until we reach test_ratio +/- tol.
    cluster_sizes = Counter(labels)
    target = test_ratio * n
    lower = (test_ratio - test_tol) * n
    upper = (test_ratio + test_tol) * n

    test_size = 0
    test_clusters: set[int] = set()
    for cid in shuffled:
        size = cluster_sizes[cid]
        # If adding pushes above upper AND we already have >= lower, stop.
        if test_size >= lower and test_size + size > upper:
            continue
        test_clusters.add(cid)
        test_size += size
        if test_size >= target:
            # Check upper bound.
            if test_size <= upper:
                break
    assert lower <= test_size <= upper, (
        f"cluster train/test split misses the {test_ratio:.2f} +/- {test_tol:.2f} window: "
        f"test_size={test_size} of n={n} (window [{lower:.1f}, {upper:.1f}]) — whole-cluster "
        "packing cannot realize the planned share on this cluster-size profile"
    )
    train_test = {cid: ("test" if cid in test_clusters else "train") for cid in cluster_ids}

    # Fold partition over TRAIN clusters (round-robin over shuffled list).
    train_clusters = [cid for cid in shuffled if cid not in test_clusters]
    folds: dict[int, int] = {}
    for i, cid in enumerate(train_clusters):
        folds[cid] = i % POOLED_N_FOLDS
    return train_test, folds


def build_manifest(
    ctx: SplitContext,
    intersection: dict[str, Any],
    per_corpus_pre_dedup: dict[str, int],
    kept_by_corpus: dict[str, list[dict]],
    dropped: list[dict[str, Any]],
    labels: list[int],
    row_index: list[dict[str, Any]],
    train_test_by_cluster: dict[int, str],
    pooled_folds_by_cluster: dict[int, int],
    round3_keep_rate: dict[str, float],
    *,
    per_corpus_pre_intersection: dict[str, int] | None = None,
) -> dict[str, Any]:
    n_kept_pre = sum(per_corpus_pre_dedup.values())
    per_corpus_kept = {slug: len(rows) for slug, rows in kept_by_corpus.items()}
    n_kept_post = sum(per_corpus_kept.values())
    # Cluster corpus histogram.
    cluster_to_corpora_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entry in row_index:
        cid = int(entry["cluster"])
        cluster_to_corpora_counts[cid][entry["corpus"]] += 1

    per_corpus_kept_rate = {
        slug: (per_corpus_kept[slug] / per_corpus_pre_dedup[slug])
        if per_corpus_pre_dedup.get(slug)
        else 0.0
        for slug in per_corpus_pre_dedup
    }

    manifest = {
        "schema_version": 1,
        "plan_version": "v15",
        "phase": "c_pool",
        "smoke": ctx.smoke,
        "seed": POOLED_SPLIT_SEED,
        "kmeans_k": POOLED_K,
        "n_folds": POOLED_N_FOLDS,
        "test_ratio": POOLED_TEST_RATIO,
        "dedup_order": list(DEDUP_ORDER),
        "pinned_revisions": {
            "mpnet": ctx.pinned_mpnet_revision,
            # ALL generation shards live under this ONE subprefix (round-4
            # live probe: generation_v2/ does not exist on the Hub).
            "generation_subprefix": "generation",
            "generation_by_model_corpus": ctx.pinned_generation_revisions,
            "generation_stems_by_model_corpus": ctx.generation_stems,
            "wave1_hf_rev": cm.WAVE1_HF_REV,
        },
        "five_way_intersection": intersection,
        "per_corpus_pre_intersection": per_corpus_pre_intersection,
        "n_kept_pre_dedup": n_kept_pre,
        "n_kept_post_dedup": n_kept_post,
        "n_cross_corpus_drops": len(dropped),
        "per_corpus_pre_dedup": per_corpus_pre_dedup,
        "per_corpus_kept": per_corpus_kept,
        "per_corpus_kept_rate": per_corpus_kept_rate,
        "round3_per_corpus_keep_rate": round3_keep_rate,
        "n_clusters": len(set(labels)),
        "cluster_to_corpora_counts": {
            str(cid): dict(counts) for cid, counts in cluster_to_corpora_counts.items()
        },
        "train_test_by_cluster": {str(cid): tag for cid, tag in train_test_by_cluster.items()},
        "pooled_folds_by_cluster": {
            str(cid): fold for cid, fold in pooled_folds_by_cluster.items()
        },
        "row_index": row_index,
        "dropped_sample": dropped[:200],  # cap the collision surface persisted here
        "dropped_total": len(dropped),
        "generated_ts": int(time.time()),
    }
    return manifest


def assert_split(
    manifest: dict[str, Any],
    round3_keep_rate: dict[str, float],
    *,
    smoke: bool = False,
) -> None:
    """Fail-loud checks per plan §4 Phase C_pool assertions.

    Under smoke, the production-scale verdicts (>= 3-corpora coverage,
    >= 15% per-corpus test share) are informational — a single-corpus
    SMOKE_CORPORA_V2 slice structurally cannot pass them (gotchas.md
    smoke/production-parity GATE-CALIBRATION rule). Cross-corpus dedup
    drop count + per-corpus keep-rate stay meaningful even under smoke.
    """
    # (1) every cluster contains prompts from >= 3 corpora
    cluster_hist = manifest["cluster_to_corpora_counts"]
    corpus_locked = [cid for cid, hist in cluster_hist.items() if len(hist) < CLUSTER_MIN_CORPORA]
    if corpus_locked:
        if smoke:
            logger.info(
                "[pool] SMOKE — %d corpus-locked clusters (production would HALT); "
                "expected on a single-corpus SMOKE_CORPORA_V2 slice",
                len(corpus_locked),
            )
        else:
            raise AssertionError(
                f"corpus-locked clusters (fewer than {CLUSTER_MIN_CORPORA} corpora): {corpus_locked}"
            )

    # (2) per-corpus test-side share >= 15%
    train_test = manifest["train_test_by_cluster"]
    row_index = manifest["row_index"]
    per_corpus_total: Counter[str] = Counter()
    per_corpus_test: Counter[str] = Counter()
    for entry in row_index:
        slug = entry["corpus"]
        per_corpus_total[slug] += 1
        if train_test[str(entry["cluster"])] == "test":
            per_corpus_test[slug] += 1
    low_share = []
    for slug, total in per_corpus_total.items():
        share = (per_corpus_test[slug] / total) if total else 0.0
        if share < PER_CORPUS_TEST_SHARE_MIN:
            low_share.append((slug, share, total))
    if low_share:
        if smoke:
            logger.info(
                "[pool] SMOKE — low per-corpus test share (production would HALT): %s",
                low_share,
            )
        else:
            raise AssertionError(
                f"per-corpus test-side share < {PER_CORPUS_TEST_SHARE_MIN:.2f}: {low_share}"
            )

    # (3) per-corpus keep-rate >= 0.99 of round-3 rate (binds under smoke too —
    #     dedup arithmetic + collision surface are meaningful on any slice)
    per_corpus_kept_rate = manifest["per_corpus_kept_rate"]
    low_keep = []
    for slug, rate in per_corpus_kept_rate.items():
        floor = round3_keep_rate.get(slug, 1.0) * PER_CORPUS_KEEP_RATE_MIN_FRAC
        if rate < floor:
            low_keep.append((slug, rate, floor))
    if low_keep:
        # HALT + log the sha collision surface so the offender is legible.
        collision_dir = manifest.get("dropped_sample", [])[:20]
        logger.error(
            "[pool] per-corpus keep-rate below round-3 floor: %s; collision sample=%s",
            low_keep,
            collision_dir,
        )
        raise SystemExit(
            f"pooled_split HALT: per-corpus keep-rate below round-3 floor "
            f"({PER_CORPUS_KEEP_RATE_MIN_FRAC:.2f} x): {low_keep}"
        )

    # (4) cross-corpus dedup drops < 500 (sanity WARN)
    if manifest["dropped_total"] >= DEDUP_DROP_WARN_THRESHOLD:
        logger.warning(
            "[pool] cross-corpus dedup drops %d >= %d (sanity WARN)",
            manifest["dropped_total"],
            DEDUP_DROP_WARN_THRESHOLD,
        )


def upload_manifest(ctx: SplitContext, manifest_path: Path) -> None:
    """Upload the split manifest to the HF data repo under the pooled_v3 prefix."""
    from explore_persona_space.orchestrate import hub

    dest = f"{cm.HF_PREFIX_1336}/analysis_tensors/{POOLED_OUT_SUBDIR}/{manifest_path.name}"
    hub._upload(  # noqa: SLF001 - established internal helper for single-file uploads
        manifest_path,
        repo_id=cm.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=dest,
        upload_as_file=True,
        commit_message=f"issue1336 pooled_split_v3 manifest ({'smoke' if ctx.smoke else 'full'})",
    )
    logger.info("[pool] uploaded manifest to %s@%s", cm.HF_DATA_REPO, dest)


def _build_context(args) -> SplitContext:
    if args.smoke:
        corpora = tuple(cm.SMOKE_CORPORA_V2)
        out_root = DATA_ROOT / POOLED_OUT_SUBDIR_SMOKE
    else:
        corpora = tuple(cm.V2_CORPORA.keys())
        out_root = DATA_ROOT / POOLED_OUT_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)
    return SplitContext(
        smoke=args.smoke,
        upload=args.upload and not args.smoke,
        corpora=corpora,
        out_root=out_root,
        hf_prefix_out=f"{cm.HF_PREFIX_1336}/analysis_tensors/{POOLED_OUT_SUBDIR}",
    )


def run(args) -> int:
    env.load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    ctx = _build_context(args)
    logger.info(
        "[pool] c_pool start smoke=%s corpora=%s out_root=%s",
        ctx.smoke,
        list(ctx.corpora),
        ctx.out_root,
    )

    # CONCERN M1 — measure the 5-way KEPT intersection FIRST (also the row
    # filter below: plan §5 n_pool IS the intersection).
    intersection, intersection_ids = measure_5way_intersection(ctx)

    # Resolve MPNet revision (CONCERN M3).
    if ctx.smoke:
        # Skip the network probe under smoke to keep it self-contained.
        ctx.pinned_mpnet_revision = "smoke-fake-revision"
    else:
        api, _dl, _hub = _hub_helpers()
        ctx.pinned_mpnet_revision = _resolve_mpnet_revision(api)
    logger.info("[pool] mpnet revision pinned to %s", ctx.pinned_mpnet_revision)

    # (2) Load corpora, restrict to the 5-way kept intersection, then dedup.
    rows_by_corpus = load_corpora_rows(ctx)
    per_corpus_pre_intersection = {slug: len(rows) for slug, rows in rows_by_corpus.items()}
    for slug, rows in rows_by_corpus.items():
        ids = intersection_ids[slug]
        picked = [r for r in rows if f"s{int(r['prompt_idx'])}" in ids]
        assert picked, (
            f"no {slug} corpus rows survive the 5-way kept-intersection filter "
            f"({len(rows)} rows vs {len(ids)} intersection ids) — id-space mismatch?"
        )
        if len(picked) < len(rows):
            logger.info(
                "[pool] %s: %d/%d rows in the 5-way kept intersection (%d dropped — "
                "not kept by every checkpoint, so not serveable by the pooled fits)",
                slug,
                len(picked),
                len(rows),
                len(rows) - len(picked),
            )
        rows_by_corpus[slug] = picked
    per_corpus_pre_dedup = {slug: len(rows) for slug, rows in rows_by_corpus.items()}
    kept_by_corpus, dropped = cross_corpus_dedup(rows_by_corpus, DEDUP_ORDER)
    logger.info(
        "[pool] dedup: kept=%d dropped=%d",
        sum(len(v) for v in kept_by_corpus.values()),
        len(dropped),
    )

    # (3) Embed all deduped prompts.
    ordered_rows: list[dict[str, Any]] = []
    for slug in DEDUP_ORDER:
        if slug in kept_by_corpus:
            ordered_rows.extend(kept_by_corpus[slug])
    prompts = [r["prompt"] for r in ordered_rows]
    assert prompts, "no kept prompts after dedup — refusing to embed empty set"
    vecs = embed_prompts(prompts, ctx.pinned_mpnet_revision, ctx.smoke)

    # (4) K-means + train/test split.
    labels = kmeans_assign(vecs, POOLED_K, POOLED_SPLIT_SEED)
    train_test, folds = cluster_train_test_split(
        labels, POOLED_SPLIT_SEED, POOLED_TEST_RATIO, POOLED_TEST_TOL
    )

    # Row-level index for downstream consumers.
    row_index: list[dict[str, Any]] = []
    for row, label in zip(ordered_rows, labels, strict=True):
        row_index.append(
            {
                "corpus": row["corpus"],
                "prompt_idx": row.get("prompt_idx"),
                "prompt_sha": row["prompt_sha"],
                "cluster": int(label),
                "arm": train_test[int(label)],
                "fold": folds.get(int(label)),
            }
        )

    # (5) Build the manifest + assert.
    round3_keep_rate = _round3_per_corpus_keep_rate(ctx.corpora)
    manifest = build_manifest(
        ctx,
        intersection,
        per_corpus_pre_dedup,
        kept_by_corpus,
        dropped,
        labels,
        row_index,
        train_test,
        folds,
        round3_keep_rate,
        per_corpus_pre_intersection=per_corpus_pre_intersection,
    )
    assert_split(manifest, round3_keep_rate, smoke=ctx.smoke)

    # Persist.
    out_dir = ctx.out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "[pool] wrote %s (n_kept_post_dedup=%d n_clusters=%d n_dropped=%d)",
        manifest_path,
        manifest["n_kept_post_dedup"],
        manifest["n_clusters"],
        manifest["dropped_total"],
    )

    if ctx.upload:
        upload_manifest(ctx, manifest_path)

    print(f"[pool] c_pool complete: {manifest_path}", flush=True)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="issue1336 pooled_split_v3 builder (Phase C_pool)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--full", action="store_true", help="run on the 7 pinned v2 corpora (default)"
    )
    mode.add_argument("--smoke", action="store_true", help="run on SMOKE_CORPORA_V2 slice")
    ap.add_argument(
        "--upload",
        action="store_true",
        help="upload the manifest to the HF data repo (full mode only)",
    )
    args = ap.parse_args(argv)
    if not args.smoke and not args.full:
        args.full = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
