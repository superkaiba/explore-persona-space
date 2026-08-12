#!/usr/bin/env python3
"""Phase C_pool — pooled-multidataset cross-corpus split (plan v17 §4, Option A).

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
     invocation time (CONCERN M3), recorded in the manifest. Before
     embedding, strip each corpus's SHARED common prompt prefix
     (run-11802 fix: math7500's 1,530-char / 611-token few-shot preamble
     exceeded mpnet's 384-token window, so every prompt truncated to the
     same boilerplate and embedded to ONE identical vector — k_eff=1,
     whole corpus atomic, A4 unsatisfiable). The strip is the string
     identity for a corpus whose shared prefix is empty (measured: 6 of
     the 7 v2 corpora, shared prefix EXACTLY 0 chars), preserves
     within-corpus prompt distinctness (p -> p[k:] at one k per corpus
     is injective), and its per-corpus stats (prefix chars/tokens,
     median remaining chars) are logged + recorded in the manifest.
  4. Per-corpus k-means (v16 Option A): k_c = clamp(round(n_c / 300),
     10, 50), seed 1336, on each corpus's OWN embeddings — the grouping
     unit is a namespaced corpus-pure sub-cluster (corpus, subcluster_id);
     whole groups go to one side of the split (the
     ood-generalization-folds group-level leakage guarantee).
  5. Cross-corpus near-duplicate scan: L2-normalize, blocked matmul
     (the full n^2 similarity matrix is never materialized), collect
     cross-corpus row pairs at cos >= 0.95, union-find MERGE the touched
     sub-clusters into joint assignment groups — merged groups are
     CO-ASSIGNED to train, never dropped. Report-only: within-corpus
     straddle counts (pairs >= 0.95 in different sub-clusters), 0.90
     sensitivity counts, and a max-cross-corpus-cosine histogram.
  6. Per-corpus greedy WHOLE-GROUP packing of pure groups to test target
     0.20 inside the window [0.15, 0.28] (gate A4: ONE deterministic
     retry with k_c doubled for a failing corpus, recorded in the
     manifest, then HALT dumping the full sub-cluster size composition
     to ``split_manifest.rejected.json``); then a 5-fold partition of
     the train side preserving group structure (round-robin over
     shuffled train groups).
  7. Persist ``analysis_tensors/pooled_split_v3/split_manifest.json``
     (``split_design: "percorpus_subcluster_v1"``: per-corpus k_c +
     sub-cluster sizes, the group table (corpus, subcluster_id) ->
     group_id -> train|test|fold, realized per-corpus test shares, the
     pooled realized ratio, the near-duplicate audit block, pinned
     revisions incl. ``generation_subprefix``, the 5-way intersection
     measurement, and the row-id assignment list) and optionally upload
     it (``--upload``) BEFORE any downstream phase.

Assertions (fail loud on violation, non-zero exit — plan v17 §4 gate set):

  - (A1) total kept == per-corpus kept sum − cross-corpus dedup drops
    (arithmetic identity; the <500 drops target stays a sanity WARN —
    measured 167);
  - (A2) per-corpus keep-rate >= 0.99 of the round-3 per-corpus keep-rate
    (HALT + log the sha collision surface on violation);
  - (A3, BINDING COVERAGE GATE) every corpus's realized test share
    >= 0.15 — BINDS UNDER SMOKE (deliberately no per-check smoke
    downgrade anywhere in this module: the SLURM-5005 shape);
  - (A4) per-corpus packing lands in [0.15, 0.28] — enforced at packing
    time (single registered k_c x2 retry, then HALT + rejected-manifest
    dump); BINDS UNDER SMOKE;
  - (A5) cross-corpus merged mass <= 10% of every corpus's rows (HALT +
    component composition dump — topical separability of the corpora is
    a design premise; its violation must surface, never silently
    proceed).
  RETIRED (v16): the >=3-corpora-per-cluster gate (CLUSTER_MIN_CORPORA)
  — false by construction under corpus-pure groups.

  NEW v18 named halts (run-11802 fix; NOT part of the plan-pinned A1-A5
  set, which is unchanged):

  - ``pooled_split_degenerate_embeddings`` — per corpus, BEFORE k-means:
    the realized embedding matrix must have > 1 distinct row AND at
    least min(k_c, n) distinct rows (the same clamp kmeans_assign
    applies, so tiny smoke slices keep passing). Runs on smoke AND
    production embeddings — a collapsed corpus halts with the CAUSE
    instead of surfacing three gates downstream as an A4 packing miss.
  - ``pooled_split_shared_prefix_exceeds_window`` — pre-embed,
    production branch (needs the loaded encoder's tokenizer): the text
    handed to the encoder must never share a prefix that alone
    tokenizes to >= the encoder's max_seq_length (read off the loaded
    model, never hardcoded). Post-strip this prefix is empty by
    construction; the gate stands independently should the strip ever
    be bypassed, so the 11802 class can never again reach k-means.

Smoke (``--smoke``) end-to-end exercise: substitutes the pinned local
smoke corpora set (SMOKE_CORPORA_V2 = ("lmsys23k",)) at tiny N so the
pipeline runs on the VM without HF traffic; the smoke NEVER uploads and
writes under data/issue_1336/pooled_split_v3_smoke. NEW v17 fail-loud
smoke-ENTRY assert, UPSTREAM of A3/A4 (a fixture-DEFECT check, not a gate
downgrade — A3/A4 still bind unchanged after it passes): the realized
5-way kept-intersection per smoke corpus must be >= SMOKE_KEPT_MIN = 16,
else HALT with the named cause ``smoke_fixture_kept_intersection_too_small``
(dumping per-model kept counts + the gen_smoke/ model dirs present).
Grounding (plan v17 §4): at n >= 16 with k_eff = 10 the [0.15, 0.28]
window is exhaustively satisfiable (plan-time subset-sum check over all
partitions of n in [16, 32] into 10 parts; a degenerate fewer-group
clustering routes through the registered k_c x2 retry); at n = 8 the
window is knife-edge and at n = 7 arithmetically EMPTY. The near-dup
scan + union-find + A5 run structurally unchanged under smoke.

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
# POOLED_K (global k-means k=50) RETIRED in v16 (Option A): grouping is
# per-corpus sub-clustering at k_c = clamp(round(n_c / GROUP_ROWS_TARGET),
# K_C_MIN, K_C_MAX) — see k_c_for().
POOLED_N_FOLDS = 5
POOLED_TEST_RATIO = 0.20
# POOLED_TEST_TOL (+/- 2%) RETIRED in v16 — superseded by the per-corpus
# acceptance window below (plan v17 §4 Phase C_pool step 5 / gate A4).
GROUP_ROWS_TARGET = 300  # expected sub-cluster size ~300 prompts (plan §11)
K_C_MIN = 10
K_C_MAX = 50
PER_CORPUS_TEST_WINDOW = (0.15, 0.28)

# Cross-corpus near-duplicate scan (plan v17 §4 step 4 / gate A5).
NEAR_DUP_COS = 0.95
NEAR_DUP_COS_SENSITIVITY = 0.90  # report-only sensitivity counts
SIM_BLOCK_ROWS = 2048  # blocked-matmul rows: ~0.4 GB peak block at n~48k
CROSS_MERGED_MASS_MAX_FRAC = 0.10  # A5 cap on per-corpus merged mass

# v17 smoke-ENTRY fixture-defect floor (UPSTREAM of A3/A4; --smoke only).
SMOKE_KEPT_MIN = 16

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

# Assertions (plan v17 §4 gate set A1-A5; CLUSTER_MIN_CORPORA retired in v16 —
# the >=3-corpora-per-cluster check is false by construction under
# corpus-pure groups and was REFORMULATED into A3 + the near-dup scan/A5).
PER_CORPUS_TEST_SHARE_MIN = 0.15  # A3 — binding coverage gate (binds under smoke)
PER_CORPUS_KEEP_RATE_MIN_FRAC = 0.99  # A2
DEDUP_DROP_WARN_THRESHOLD = 500  # A1 sanity WARN target (measured 167)

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
            if len(inter) < SMOKE_KEPT_MIN:
                # v17 fail-loud smoke-ENTRY assert (plan §4): a FIXTURE-DEFECT
                # check UPSTREAM of gates A3/A4 — the dispatcher regenerates
                # smoke fixtures at run time, so the realized kept-intersection
                # is unknowable pre-run; a shrunken intersection must halt with
                # the fixture cause instead of a spurious A4 packing failure.
                # It downgrades NOTHING: A3/A4 still bind unchanged after it
                # passes (at n >= 16, k_eff = 10, the [0.15, 0.28] window is
                # exhaustively satisfiable; n = 8 is knife-edge, n = 7 empty).
                per_model_kept = {m: len(s) for m, s in model_sets.items()}
                dirs_present = sorted(p.name for p in gen_smoke.iterdir() if p.is_dir())
                logger.error(
                    "[pool] SMOKE HALT: %s realized 5-way kept-intersection %d < "
                    "SMOKE_KEPT_MIN=%d (per-model kept: %s; gen_smoke dirs present: %s)",
                    corpus,
                    len(inter),
                    SMOKE_KEPT_MIN,
                    per_model_kept,
                    dirs_present,
                )
                raise SystemExit(
                    "pooled_split HALT [smoke_fixture_kept_intersection_too_small]: "
                    f"{corpus} realized 5-way kept-intersection {len(inter)} < "
                    f"SMOKE_KEPT_MIN={SMOKE_KEPT_MIN} — smoke fixture defect; regenerate "
                    "the smoke gen fixtures (issue1336_smoke_fixtures.py gen + gen-v2). "
                    f"Per-model kept counts: {per_model_kept}; gen_smoke model dirs "
                    f"present: {dirs_present}."
                )
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


def shared_common_prefix(prompts: list[str]) -> str:
    """Longest common prefix shared by ALL prompts of one corpus (pure string op).

    Returns "" for an empty or single-prompt list: a "shared" prefix needs
    >= 2 prompts to be evidence of boilerplate, and stripping the whole text
    of a singleton corpus would hand the encoder an empty string.

    Uses the min/max-lexicographic identity: the common prefix of every
    string in a set equals the common prefix of its lexicographic min and
    max, so the scan is O(total chars) without pairwise comparison.
    """
    if len(prompts) < 2:
        return ""
    lo = min(prompts)
    hi = max(prompts)
    end = min(len(lo), len(hi))
    i = 0
    while i < end and lo[i] == hi[i]:
        i += 1
    return lo[:i]


def strip_shared_prefix(prompts: list[str]) -> tuple[list[str], str]:
    """Strip the corpus-shared common prompt prefix before embedding.

    Returns ``(stripped_prompts, prefix)``. PROVABLE NO-OP when the shared
    prefix is empty — ``p[len(""):]`` is the identity, so a corpus with
    prefix == "" (measured on run 11802: 6 of the 7 v2 corpora, EXACTLY 0
    shared chars) hands the encoder byte-identical strings. Within-corpus
    distinctness is preserved: p -> p[k:] with one k per corpus is injective
    on the corpus's (post-dedup, globally distinct) prompt set.

    Run-11802 fix: math7500's few-shot formatting put a fixed 1,530-char /
    611-token exemplar preamble before every question while mpnet's window
    is 384 tokens, so the encoder never reached the distinguishing tail and
    all ~7,166 prompts embedded to ONE byte-identical vector (k_eff=1).
    """
    prefix = shared_common_prefix(prompts)
    if not prefix:
        return list(prompts), ""
    k = len(prefix)
    return [p[k:] for p in prompts], prefix


def check_prefix_tokens_within_cap(
    slug: str, prefix_chars: int, prefix_tokens: int, cap: int
) -> None:
    """Named-cause halt when a shared prompt prefix ALONE fills the encoder
    window (>= max_seq_length tokens): every prompt then truncates to the
    same boilerplate and embeds identically (run 11802, math7500: 611-token
    shared preamble vs mpnet cap 384 -> 1 distinct embedding row of 7,166).

    Pure gate (the caller computes ``prefix_tokens`` with the LOADED model's
    tokenizer and ``cap`` from ``model.max_seq_length`` — never hardcoded),
    applied to the text actually handed to the encoder so this class can
    never again reach k-means even if the strip upstream is bypassed."""
    if prefix_tokens >= cap:
        raise SystemExit(
            "pooled_split HALT [pooled_split_shared_prefix_exceeds_window]: corpus "
            f"{slug} shared prompt prefix ({prefix_chars} chars) tokenizes to "
            f"{prefix_tokens} tokens >= encoder max_seq_length={cap} — the encoder "
            "would see only shared boilerplate and collapse every prompt to one "
            "embedding; strip or reformat the corpus prompts before embedding"
        )


def assert_embeddings_nondegenerate(slug: str, vecs_c, k_c: int) -> int:
    """v18 gate (run-11802 fix): fail loud BEFORE k-means when a corpus's
    REALIZED embeddings are degenerate. Gate: distinct-row count > 1 AND
    >= min(k_c, n) — the same clamp ``kmeans_assign`` applies, so tiny smoke
    slices (n < k_c, all rows distinct) keep passing. Distinctness is
    byte-exact rows: the measured 11802 collapse was byte-identical
    (pairwise cosine min=mean=max=1.000000, 1 distinct row of 7,166), and
    k-means cannot separate identical points, so the registered A4 k_c x2
    retry is powerless by construction. Returns the distinct-row count."""
    import numpy as np

    arr = np.ascontiguousarray(np.asarray(vecs_c, dtype=np.float32))
    n = int(arr.shape[0])
    distinct = len({arr[i].tobytes() for i in range(n)})
    floor = min(k_c, n)
    if distinct <= 1 or distinct < floor:
        raise SystemExit(
            "pooled_split HALT [pooled_split_degenerate_embeddings]: corpus "
            f"{slug} realized {distinct} distinct embedding row(s) out of n={n} "
            f"(k_c={k_c}, required floor=min(k_c, n)={floor}) — embedding collapse "
            "upstream of k-means (e.g. a shared prompt prefix longer than the "
            "encoder window); distinctness of the TEXT is not distinctness of "
            "the EMBEDDING"
        )
    logger.info(
        "[pool] %s: embeddings non-degenerate — %d distinct rows / n=%d (k_c=%d)",
        slug,
        distinct,
        n,
        k_c,
    )
    return distinct


def embed_prompts(
    prompts: list[str], revision: str, smoke: bool, model=None
) -> "list[list[float]]":
    """Return fp32 embeddings as a nested list (JSON-serializable-friendly).

    ``model``: an already-loaded SentenceTransformer — production callers load
    it once for the pre-embed shared-prefix-vs-cap gate and pass it through
    (None loads fresh at the pinned revision). Ignored under smoke."""
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
    if model is None:
        model = _load_sentence_transformer(revision)
    logger.info("[pool] embedding %d prompts via %s @ %s", len(prompts), MPNET_MODEL_ID, revision)
    vecs = model.encode(prompts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    return [row.tolist() for row in vecs]


def kmeans_assign(vectors, k: int, seed: int) -> list[int]:
    """Run sklearn KMeans k=k with a fixed seed on an (n, d) array (or nested
    list) of embeddings. Returns cluster labels; k_eff clamps to n for tiny
    smoke slices (the plan §4 smoke-grounding arithmetic relies on this)."""
    import numpy as np
    from sklearn.cluster import KMeans

    X = np.asarray(vectors, dtype=np.float32)
    # Clamp k to n_samples for tiny smoke runs.
    k_eff = min(k, X.shape[0]) if X.shape[0] > 0 else k
    km = KMeans(n_clusters=k_eff, random_state=seed, n_init="auto")
    labels = km.fit_predict(X)
    return [int(x) for x in labels.tolist()]


def k_c_for(n_c: int) -> int:
    """Per-corpus k-means k (plan v17 §11): clamp(round(n_c / 300), 10, 50)."""
    return max(K_C_MIN, min(K_C_MAX, int(round(n_c / GROUP_ROWS_TARGET))))


def _l2_normalize(vecs: "object") -> "object":
    """Row-normalize an (n, d) fp32 embedding matrix for cosine via matmul."""
    import numpy as np

    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    assert float(norms.min()) > 0.0, "zero-norm embedding row — cannot cosine-normalize"
    return (arr / norms).astype(np.float32)


def cross_corpus_scan(unit, corpus_idx) -> tuple["object", int, "object"]:
    """Blocked cross-corpus near-duplicate scan (plan v17 §4 step 4).

    ``unit``: (n, d) L2-normalized fp32; ``corpus_idx``: (n,) int corpus ids
    aligned with rows. Blocked matmul (SIM_BLOCK_ROWS x n per block — the
    full n^2 similarity matrix is never materialized; batched by
    construction). Returns ``(pairs_95, n_cross_90, max_cross)``:

    - ``pairs_95``: (m, 2) int64 global row-index pairs (i < j) in DIFFERENT
      corpora with cos >= NEAR_DUP_COS — the union-find merge edges;
    - ``n_cross_90``: count of cross-corpus pairs at the report-only
      NEAR_DUP_COS_SENSITIVITY threshold;
    - ``max_cross``: (n,) fp32 per-row max cosine to any OTHER-corpus row
      (-1.0 when no other corpus exists, e.g. the single-corpus smoke slice).
    """
    import numpy as np

    n = int(unit.shape[0])
    pair_blocks: list[np.ndarray] = []
    n_cross_90 = 0
    max_cross = np.full(n, -1.0, dtype=np.float32)
    col_idx = np.arange(n, dtype=np.int64)[None, :]
    for start in range(0, n, SIM_BLOCK_ROWS):
        stop = min(start + SIM_BLOCK_ROWS, n)
        sims = unit[start:stop] @ unit.T  # (b, n) fp32
        same = corpus_idx[start:stop, None] == corpus_idx[None, :]
        sims[same] = -1.0  # neutralize within-corpus cells (incl. self)
        max_cross[start:stop] = sims.max(axis=1)
        upper = col_idx > np.arange(start, stop, dtype=np.int64)[:, None]
        hits = np.argwhere((sims >= NEAR_DUP_COS) & upper)
        if hits.size:
            hits = hits.astype(np.int64)
            hits[:, 0] += start
            pair_blocks.append(hits)
        n_cross_90 += int(((sims >= NEAR_DUP_COS_SENSITIVITY) & upper).sum())
    pairs_95 = (
        np.concatenate(pair_blocks, axis=0) if pair_blocks else np.empty((0, 2), dtype=np.int64)
    )
    return pairs_95, n_cross_90, max_cross


def within_corpus_straddles(unit_c, labels_c) -> tuple[int, int]:
    """Report-only within-corpus straddle counts (plan v17 §4 step 4): pairs
    of SAME-corpus rows with cos >= threshold whose members sit in DIFFERENT
    sub-clusters — the residual within-corpus leakage surface. Returns
    ``(n_straddle_at_0p95, n_straddle_at_0p90)``. Blocked like
    cross_corpus_scan; never gates."""
    import numpy as np

    n = int(unit_c.shape[0])
    lab = np.asarray(labels_c, dtype=np.int64)
    assert lab.shape[0] == n, (lab.shape, n)
    n95 = 0
    n90 = 0
    col_idx = np.arange(n, dtype=np.int64)[None, :]
    for start in range(0, n, SIM_BLOCK_ROWS):
        stop = min(start + SIM_BLOCK_ROWS, n)
        sims = unit_c[start:stop] @ unit_c.T
        upper = col_idx > np.arange(start, stop, dtype=np.int64)[:, None]
        lab_diff = lab[start:stop, None] != lab[None, :]
        mask = upper & lab_diff
        n95 += int(((sims >= NEAR_DUP_COS) & mask).sum())
        n90 += int(((sims >= NEAR_DUP_COS_SENSITIVITY) & mask).sum())
    return n95, n90


class _UnionFind:
    """Tiny union-find over hashable keys (path-halving; union by size)."""

    def __init__(self) -> None:
        self.parent: dict[Any, Any] = {}
        self.size: dict[Any, int] = {}

    def find(self, x):
        p = self.parent.setdefault(x, x)
        self.size.setdefault(x, 1)
        while p != x:
            gp = self.parent[p]
            self.parent[x] = gp
            x, p = p, self.parent[gp] if gp in self.parent else gp
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def keys(self):
        return self.parent.keys()


def _assign_given_labels(
    corpus_names: list[str],
    slices: dict[str, tuple[int, int]],
    labels_by_corpus: dict[str, list[int]],
    pairs_95,
    seed: int,
) -> dict[str, Any]:
    """One assignment pass at the CURRENT per-corpus labels: project the
    cross-corpus near-dup row pairs onto (corpus, subcluster) group keys,
    union-find merge, force merged groups to train, then per-corpus greedy
    whole-group packing of pure groups to POOLED_TEST_RATIO inside
    PER_CORPUS_TEST_WINDOW. Returns the assignment dict; window misses are
    reported in ``packing_failures`` (the A4 retry loop is the caller's)."""
    row_key_of: dict[int, tuple[str, int]] = {}

    def key_of(i: int) -> tuple[str, int]:
        k = row_key_of.get(i)
        if k is None:
            for slug in corpus_names:
                s0, s1 = slices[slug]
                if s0 <= i < s1:
                    k = (slug, int(labels_by_corpus[slug][i - s0]))
                    row_key_of[i] = k
                    return k
            raise AssertionError(f"row index {i} outside every corpus slice")
        return k

    # Union-find merge over group keys touched by cross-corpus near-dup pairs.
    uf = _UnionFind()
    for i, j in pairs_95:
        uf.union(key_of(int(i)), key_of(int(j)))
    merged_keys: set[tuple[str, int]] = set(uf.keys())

    # Group sizes (every group, merged or pure).
    group_sizes: Counter = Counter()
    for slug in corpus_names:
        s0, s1 = slices[slug]
        for lab in labels_by_corpus[slug]:
            group_sizes[(slug, int(lab))] += 1
        assert sum(group_sizes[k] for k in group_sizes if k[0] == slug) == s1 - s0

    # Merged components (audit + A5 input).
    comp_members: dict[Any, list[tuple[str, int]]] = defaultdict(list)
    for k in uf.keys():
        comp_members[uf.find(k)].append(k)
    components = []
    for members in comp_members.values():
        members_sorted = sorted(members)
        rows_by_corpus: Counter = Counter()
        for k in members_sorted:
            rows_by_corpus[k[0]] += group_sizes[k]
        components.append(
            {
                "groups": [[slug, sub] for slug, sub in members_sorted],
                "n_groups": len(members_sorted),
                "n_rows": int(sum(rows_by_corpus.values())),
                "rows_by_corpus": dict(sorted(rows_by_corpus.items())),
            }
        )
    components.sort(key=lambda c: (-c["n_rows"], c["groups"]))

    per_corpus_total = {slug: slices[slug][1] - slices[slug][0] for slug in corpus_names}
    per_corpus_merged = {}
    for slug in corpus_names:
        rows = int(sum(group_sizes[k] for k in merged_keys if k[0] == slug))
        per_corpus_merged[slug] = {
            "rows": rows,
            "frac": rows / per_corpus_total[slug] if per_corpus_total[slug] else 0.0,
        }

    # Per-corpus greedy whole-group packing of PURE groups (merged -> train).
    arm_by_key: dict[tuple[str, int], str] = {}
    packing_failures: dict[str, str] = {}
    per_corpus_test_share: dict[str, float] = {}
    lo_frac, hi_frac = PER_CORPUS_TEST_WINDOW
    for slug in corpus_names:
        keys_c = sorted((k for k in group_sizes if k[0] == slug), key=lambda t: t[1])
        pure = [k for k in keys_c if k not in merged_keys]
        n_total = per_corpus_total[slug]
        target = POOLED_TEST_RATIO * n_total
        lower = lo_frac * n_total
        upper = hi_frac * n_total
        rng = random.Random(f"{seed}:{slug}")
        shuffled = list(pure)
        rng.shuffle(shuffled)
        test_rows = 0
        test_keys: set[tuple[str, int]] = set()
        for k in shuffled:
            if test_rows >= target:
                break
            sz = group_sizes[k]
            if test_rows + sz > upper:
                continue
            test_keys.add(k)
            test_rows += sz
        share = test_rows / n_total if n_total else 0.0
        per_corpus_test_share[slug] = share
        if not (lower <= test_rows <= upper):
            packing_failures[slug] = (
                f"realized test rows {test_rows}/{n_total} (share {share:.4f}) outside "
                f"window [{lo_frac}, {hi_frac}] (target {POOLED_TEST_RATIO})"
            )
        for k in keys_c:
            arm_by_key[k] = "test" if k in test_keys else "train"

    # Global integer group ids (deterministic: corpus order, then subcluster id).
    group_key_to_id: dict[tuple[str, int], int] = {}
    gid = 0
    for slug in corpus_names:
        for k in sorted((kk for kk in group_sizes if kk[0] == slug), key=lambda t: t[1]):
            group_key_to_id[k] = gid
            gid += 1

    # 5-fold partition of the train side, group structure preserved
    # (round-robin over the seeded shuffle of ALL train groups). The realized
    # fold count is floor-guarded by group arithmetic so every fold holds
    # >= 2 GROUPS (hence >= 2 rows): the downstream delta-Q battery divides
    # by per-fold-block Y variance (issue1336_metric_ladder.py
    # `_unit_residual_read`), which is identically ZERO for a 1-row fold —
    # at the 16-row smoke slice 8 train groups round-robined into 5 folds
    # produced exactly that (single-row fold 2). Production is byte-inert:
    # ~166 groups // 2 = 83 >= POOLED_N_FOLDS, so n_folds_eff == 5 there
    # (the #1489 realized-slice-arithmetic convention: derive the dial from
    # realized n, never from the assumed cap).
    train_keys = [k for k in group_key_to_id if arm_by_key[k] == "train"]
    train_keys.sort(key=lambda k: group_key_to_id[k])
    rng_folds = random.Random(seed)
    rng_folds.shuffle(train_keys)
    n_folds_eff = max(2, min(POOLED_N_FOLDS, len(train_keys) // 2))
    if n_folds_eff != POOLED_N_FOLDS:
        logger.warning(
            "[pool] fold count floor-guarded to %d (train groups=%d < 2*%d): every fold "
            "must hold >=2 groups for the battery's fold-block variance denominator",
            n_folds_eff,
            len(train_keys),
            POOLED_N_FOLDS,
        )
    fold_by_key = {k: i % n_folds_eff for i, k in enumerate(train_keys)}
    fold_rows: dict[int, int] = {}
    for k, f in fold_by_key.items():
        fold_rows[f] = fold_rows.get(f, 0) + group_sizes[k]
    assert fold_rows and min(fold_rows.values()) >= 2, (
        f"pooled_split HALT [fold_block_underfilled]: realized fold row counts {fold_rows} "
        f"carry a <2-row fold — the delta-Q battery's fold-block variance denominator is "
        f"degenerate there; the kept slice is too small to fold (train groups: "
        f"{len(train_keys)})"
    )

    n_test_rows = sum(group_sizes[k] for k, arm in arm_by_key.items() if arm == "test")
    n_rows = sum(per_corpus_total.values())
    return {
        "group_sizes": group_sizes,
        "group_key_to_id": group_key_to_id,
        "arm_by_key": arm_by_key,
        "fold_by_key": fold_by_key,
        "merged_keys": merged_keys,
        "components": components,
        "per_corpus_merged": per_corpus_merged,
        "per_corpus_test_share": per_corpus_test_share,
        "pooled_test_ratio_realized": n_test_rows / n_rows if n_rows else 0.0,
        "packing_failures": packing_failures,
        "n_folds_realized": n_folds_eff,
    }


def _halt_packing(
    out_root: Path,
    corpus_names: list[str],
    labels_by_corpus: dict[str, list[int]],
    result: dict[str, Any],
    retries: dict[str, dict[str, int]],
    halted: list[str],
) -> None:
    """A4 terminal HALT: dump the full sub-cluster size composition to
    ``split_manifest.rejected.json`` (diagnostic only — deliberately NOT the
    canonical name, so no downstream phase can consume a failed split), then
    exit non-zero with the named cause."""
    payload = {
        "halt_cause": "pooled_split_packing_window_unsatisfiable",
        "halted_corpora": halted,
        "packing_failures": result["packing_failures"],
        "packing_retries": retries,
        "test_ratio_target": POOLED_TEST_RATIO,
        "test_window": list(PER_CORPUS_TEST_WINDOW),
        "subcluster_size_composition": {
            slug: sorted(Counter(labels_by_corpus[slug]).values(), reverse=True)
            for slug in corpus_names
        },
        "per_corpus_merged_mass": result["per_corpus_merged"],
        "generated_ts": int(time.time()),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    rejected_path = out_root / "split_manifest.rejected.json"
    rejected_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.error(
        "[pool] A4 HALT — packing window unsatisfiable after the single registered "
        "k_c x2 retry for %s; sub-cluster composition dumped to %s",
        halted,
        rejected_path,
    )
    raise SystemExit(
        f"pooled_split HALT [pooled_split_packing_window_unsatisfiable]: corpora "
        f"{halted} missed the {list(PER_CORPUS_TEST_WINDOW)} test window after the "
        f"single registered k_c x2 retry — composition at {rejected_path}"
    )


def build_split_assignment(
    ordered_rows: list[dict],
    vecs,
    out_root: Path,
    seed: int = POOLED_SPLIT_SEED,
) -> dict[str, Any]:
    """Plan v17 §4 Phase C_pool steps 3-6: per-corpus k-means at k_c, the
    cross-corpus near-duplicate scan + union-find co-assignment, per-corpus
    whole-group packing with the single registered A4 k_c x2 retry, and the
    5-fold train partition. Returns the assignment dict build_manifest and
    run() consume; HALTs (rejected dump) when packing stays unsatisfiable."""
    import numpy as np

    vecs = np.asarray(vecs, dtype=np.float32)
    n = len(ordered_rows)
    assert vecs.shape[0] == n, (vecs.shape, n)

    # Per-corpus contiguous slices (ordered_rows extends corpus-by-corpus).
    corpus_names: list[str] = []
    for r in ordered_rows:
        if r["corpus"] not in corpus_names:
            corpus_names.append(r["corpus"])
    slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    for slug in corpus_names:
        n_c = sum(1 for r in ordered_rows if r["corpus"] == slug)
        assert all(ordered_rows[i]["corpus"] == slug for i in range(cursor, cursor + n_c)), (
            f"{slug} rows are not contiguous in ordered_rows"
        )
        slices[slug] = (cursor, cursor + n_c)
        cursor += n_c
    assert cursor == n

    corpus_idx = np.empty(n, dtype=np.int64)
    for ci, slug in enumerate(corpus_names):
        s0, s1 = slices[slug]
        corpus_idx[s0:s1] = ci

    # Label-independent cross-corpus near-dup scan (once; retry-safe).
    unit = _l2_normalize(vecs)
    pairs_95, n_cross_90, max_cross = cross_corpus_scan(unit, corpus_idx)
    logger.info(
        "[pool] near-dup scan: %d cross-corpus pairs at cos>=%.2f (%d at >=%.2f)",
        int(pairs_95.shape[0]),
        NEAR_DUP_COS,
        n_cross_90,
        NEAR_DUP_COS_SENSITIVITY,
    )

    # Initial per-corpus k-means at k_c (seed POOLED_SPLIT_SEED). The
    # non-degeneracy gate runs on the REALIZED embeddings BEFORE k-means
    # (run-11802 fix): a collapsed corpus halts here with the named cause
    # instead of surfacing three gates downstream as an A4 packing miss.
    labels_by_corpus: dict[str, list[int]] = {}
    k_req: dict[str, int] = {}
    for slug in corpus_names:
        s0, s1 = slices[slug]
        k = k_c_for(s1 - s0)
        assert_embeddings_nondegenerate(slug, vecs[s0:s1], k)
        labels_by_corpus[slug] = kmeans_assign(vecs[s0:s1], k, seed)
        k_req[slug] = k
        logger.info(
            "[pool] %s: n=%d k_c=%d k_eff=%d",
            slug,
            s1 - s0,
            k,
            len(set(labels_by_corpus[slug])),
        )

    # Packing with the single registered A4 retry (k_c doubled per failing
    # corpus, recorded in the manifest), then HALT + rejected dump.
    retries: dict[str, dict[str, int]] = {}
    result: dict[str, Any] | None = None
    for _round in range(len(corpus_names) + 1):
        result = _assign_given_labels(corpus_names, slices, labels_by_corpus, pairs_95, seed)
        failures = result["packing_failures"]
        if not failures:
            break
        exhausted = sorted(slug for slug in failures if slug in retries)
        if exhausted:
            _halt_packing(out_root, corpus_names, labels_by_corpus, result, retries, exhausted)
        for slug in sorted(failures):
            k2 = 2 * k_req[slug]
            retries[slug] = {"k_initial": k_req[slug], "k_retry": k2}
            logger.warning(
                "[pool] A4 window miss for %s (%s) — single registered retry with "
                "k_c doubled: %d -> %d",
                slug,
                failures[slug],
                k_req[slug],
                k2,
            )
            s0, s1 = slices[slug]
            labels_by_corpus[slug] = kmeans_assign(vecs[s0:s1], k2, seed)
            k_req[slug] = k2
    assert result is not None and not result["packing_failures"], (
        "packing retry loop exited without a decision"
    )

    # Report-only within-corpus straddle counts at the FINAL labels.
    straddle_95: dict[str, int] = {}
    straddle_90: dict[str, int] = {}
    for slug in corpus_names:
        s0, s1 = slices[slug]
        n95, n90 = within_corpus_straddles(unit[s0:s1], labels_by_corpus[slug])
        straddle_95[slug] = n95
        straddle_90[slug] = n90

    # Max-cross-corpus-cosine histogram (full mode only: a single-corpus
    # smoke slice has no other-corpus columns, sentinel -1.0 everywhere).
    if len(corpus_names) >= 2:
        mc = max_cross.astype(np.float64)
        counts, edges = np.histogram(mc, bins=np.linspace(-1.0, 1.0, 41))
        max_cross_block: dict[str, Any] | None = {
            "bin_edges": [round(float(x), 4) for x in edges],
            "counts": [int(c) for c in counts],
            "summary": {
                "max": float(mc.max()),
                "mean": float(mc.mean()),
                "p50": float(np.quantile(mc, 0.50)),
                "p90": float(np.quantile(mc, 0.90)),
                "p99": float(np.quantile(mc, 0.99)),
                "n_ge_sensitivity": int((mc >= NEAR_DUP_COS_SENSITIVITY).sum()),
                "n_ge_threshold": int((mc >= NEAR_DUP_COS).sum()),
            },
        }
    else:
        max_cross_block = None

    # Row-level gid assignment for the row index.
    gid_of_row: list[int] = []
    for slug in corpus_names:
        s0, s1 = slices[slug]
        for local, lab in enumerate(labels_by_corpus[slug]):
            assert s0 + local < s1
            gid_of_row.append(result["group_key_to_id"][(slug, int(lab))])
    assert len(gid_of_row) == n

    group_key_to_id = result["group_key_to_id"]
    merged_keys = result["merged_keys"]
    group_table = []
    for key, g in sorted(group_key_to_id.items(), key=lambda kv: kv[1]):
        group_table.append(
            {
                "corpus": key[0],
                "subcluster_id": key[1],
                "group_id": g,
                "n_rows": int(result["group_sizes"][key]),
                "arm": result["arm_by_key"][key],
                "fold": result["fold_by_key"].get(key),
                "cross_corpus_merged": key in merged_keys,
            }
        )

    return {
        "corpus_names": corpus_names,
        "per_corpus_k": {
            slug: {
                "n_rows": slices[slug][1] - slices[slug][0],
                "k_requested": k_req[slug],
                "k_eff": len(set(labels_by_corpus[slug])),
                "retried": slug in retries,
            }
            for slug in corpus_names
        },
        "packing_retries": retries,
        "subcluster_sizes": {
            slug: sorted(Counter(labels_by_corpus[slug]).values(), reverse=True)
            for slug in corpus_names
        },
        "group_table": group_table,
        "gid_of_row": gid_of_row,
        "arm_by_gid": {e["group_id"]: e["arm"] for e in group_table},
        "fold_by_gid": {e["group_id"]: e["fold"] for e in group_table if e["fold"] is not None},
        "per_corpus_test_share": result["per_corpus_test_share"],
        "pooled_test_ratio_realized": result["pooled_test_ratio_realized"],
        "n_folds_realized": result["n_folds_realized"],
        "near_dup_audit": {
            "threshold": NEAR_DUP_COS,
            "sensitivity_threshold": NEAR_DUP_COS_SENSITIVITY,
            "embedding_model": MPNET_MODEL_ID,
            "n_cross_corpus_pairs_ge_threshold": int(pairs_95.shape[0]),
            "n_cross_corpus_pairs_ge_sensitivity": int(n_cross_90),
            "n_merged_groups": len(merged_keys),
            "n_components": len(result["components"]),
            "largest_component": (result["components"][0] if result["components"] else None),
            "per_corpus_merged_mass": result["per_corpus_merged"],
            "within_corpus_straddle_ge_threshold": straddle_95,
            "within_corpus_straddle_ge_sensitivity": straddle_90,
            "max_cross_corpus_cosine": max_cross_block,
        },
    }


def build_manifest(
    ctx: SplitContext,
    intersection: dict[str, Any],
    per_corpus_pre_dedup: dict[str, int],
    kept_by_corpus: dict[str, list[dict]],
    dropped: list[dict[str, Any]],
    assignment: dict[str, Any],
    row_index: list[dict[str, Any]],
    round3_keep_rate: dict[str, float],
    *,
    per_corpus_pre_intersection: dict[str, int] | None = None,
    prefix_strip: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the v17 split manifest (``split_design: percorpus_subcluster_v1``).

    ``assignment`` is build_split_assignment()'s output: per-corpus k_c +
    sub-cluster sizes, the group table ((corpus, subcluster_id) -> group_id ->
    train|test|fold), realized test shares, the near-duplicate audit block,
    and the packing-retry record. ``row_index`` keeps the downstream consumer
    contract: int ``cluster`` (== global group_id), ``arm`` in train|test,
    ``fold`` present iff train."""
    n_kept_pre = sum(per_corpus_pre_dedup.values())
    per_corpus_kept = {slug: len(rows) for slug, rows in kept_by_corpus.items()}
    n_kept_post = sum(per_corpus_kept.values())

    per_corpus_kept_rate = {
        slug: (per_corpus_kept[slug] / per_corpus_pre_dedup[slug])
        if per_corpus_pre_dedup.get(slug)
        else 0.0
        for slug in per_corpus_pre_dedup
    }

    audit = dict(assignment["near_dup_audit"])
    audit["embedding_revision"] = ctx.pinned_mpnet_revision

    manifest = {
        "schema_version": 2,
        "split_design": "percorpus_subcluster_v1",
        "plan_version": "v17",
        "phase": "c_pool",
        "smoke": ctx.smoke,
        "seed": POOLED_SPLIT_SEED,
        # Realized fold count: == POOLED_N_FOLDS at production scale; the
        # >=2-groups-per-fold floor guard can lower it on a tiny (smoke) slice.
        "n_folds": assignment["n_folds_realized"],
        "test_ratio": POOLED_TEST_RATIO,
        "test_window": list(PER_CORPUS_TEST_WINDOW),
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
        # v18 run-11802 fix: per-corpus shared-prefix strip stats (prefix
        # chars/tokens + median remaining chars; tokens None under smoke).
        "prompt_prefix_strip": prefix_strip,
        "n_kept_pre_dedup": n_kept_pre,
        "n_kept_post_dedup": n_kept_post,
        "n_cross_corpus_drops": len(dropped),
        "per_corpus_pre_dedup": per_corpus_pre_dedup,
        "per_corpus_kept": per_corpus_kept,
        "per_corpus_kept_rate": per_corpus_kept_rate,
        "round3_per_corpus_keep_rate": round3_keep_rate,
        "per_corpus_k": assignment["per_corpus_k"],
        "packing_retries": assignment["packing_retries"],
        "subcluster_sizes": assignment["subcluster_sizes"],
        "n_groups": len(assignment["group_table"]),
        "group_table": assignment["group_table"],
        "per_corpus_test_share": assignment["per_corpus_test_share"],
        "pooled_test_ratio_realized": assignment["pooled_test_ratio_realized"],
        "near_dup_audit": audit,
        # Back-compat views keyed on the global integer group id (str keys,
        # matching the prior manifests' shape; consumers read row_index).
        "train_test_by_cluster": {
            str(gid): arm for gid, arm in sorted(assignment["arm_by_gid"].items())
        },
        "pooled_folds_by_cluster": {
            str(gid): fold for gid, fold in sorted(assignment["fold_by_gid"].items())
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
) -> None:
    """Fail-loud gate set A1/A2/A3/A5 (plan v17 §4 Phase C_pool).

    Deliberately takes NO smoke parameter and carries NO per-check downgrade
    (the SLURM-5005 shape): every gate below binds identically under
    ``--smoke`` and ``--full``. A4 (the [0.15, 0.28] packing window + the
    single registered k_c x2 retry) is enforced at packing time in
    build_split_assignment(); its terminal HALT dumps
    ``split_manifest.rejected.json`` there. The RETIRED v15 gate
    (>=3-corpora-per-cluster, CLUSTER_MIN_CORPORA) is removed — false by
    construction under corpus-pure groups.
    """
    # (A1) total kept == per-corpus kept sum − cross-corpus dedup drops
    # (arithmetic identity over the realized counts; <500 drops stays a WARN).
    n_pre = manifest["n_kept_pre_dedup"]
    n_post = manifest["n_kept_post_dedup"]
    n_drops = manifest["n_cross_corpus_drops"]
    assert n_post == n_pre - n_drops, (
        f"(A1) kept-count arithmetic broken: post_dedup={n_post} != "
        f"pre_dedup={n_pre} - drops={n_drops}"
    )
    assert n_post == sum(manifest["per_corpus_kept"].values()), (
        f"(A1) n_kept_post_dedup={n_post} != sum(per_corpus_kept)"
    )
    if manifest["dropped_total"] >= DEDUP_DROP_WARN_THRESHOLD:
        logger.warning(
            "[pool] cross-corpus dedup drops %d >= %d (A1 sanity WARN)",
            manifest["dropped_total"],
            DEDUP_DROP_WARN_THRESHOLD,
        )

    # (A2) per-corpus keep-rate >= 0.99 of round-3 rate (HALT + collision
    # surface — unchanged; binds under smoke too).
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

    # (A3, BINDING COVERAGE GATE — binds under smoke) every corpus's realized
    # test share >= PER_CORPUS_TEST_SHARE_MIN, recomputed from row_index (the
    # consumed ground truth) and cross-checked against the recorded shares.
    row_index = manifest["row_index"]
    per_corpus_total: Counter = Counter()
    per_corpus_test: Counter = Counter()
    for entry in row_index:
        slug = entry["corpus"]
        per_corpus_total[slug] += 1
        if entry["arm"] == "test":
            per_corpus_test[slug] += 1
    recorded = manifest["per_corpus_test_share"]
    low_share = []
    for slug, total in sorted(per_corpus_total.items()):
        share = (per_corpus_test[slug] / total) if total else 0.0
        rec = recorded.get(slug)
        assert rec is not None and abs(share - rec) < 1e-9, (
            f"(A3) recorded test share for {slug} ({rec}) disagrees with "
            f"row_index-recomputed {share:.6f}"
        )
        if share < PER_CORPUS_TEST_SHARE_MIN:
            low_share.append((slug, share, total))
    if low_share:
        raise AssertionError(
            f"(A3) per-corpus test-side share < {PER_CORPUS_TEST_SHARE_MIN:.2f}: {low_share}"
        )

    # (A5) cross-corpus merged mass <= 10% of every corpus's rows (HALT +
    # component composition dump — topical separability is a design premise).
    audit = manifest["near_dup_audit"]
    over = {
        slug: mass
        for slug, mass in audit["per_corpus_merged_mass"].items()
        if mass["frac"] > CROSS_MERGED_MASS_MAX_FRAC
    }
    if over:
        raise AssertionError(
            f"(A5) cross-corpus near-dup merged mass > "
            f"{CROSS_MERGED_MASS_MAX_FRAC:.0%} of corpus rows: {over}; "
            f"largest merged component: {audit['largest_component']}"
        )


def upload_manifest(ctx: SplitContext, manifest_path: Path) -> None:
    """Upload the split manifest to the HF data repo under the pooled_v3 prefix."""
    from explore_persona_space.orchestrate import hub

    dest = f"{cm.HF_PREFIX_1336}/analysis_tensors/{POOLED_OUT_SUBDIR}/{manifest_path.name}"
    base_url = hub._upload(  # noqa: SLF001 - established internal helper for single-file uploads
        manifest_path,
        repo_id=cm.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=dest,
        upload_as_file=True,
        commit_message=f"issue1336 pooled_split_v3 manifest ({'smoke' if ctx.smoke else 'full'})",
    )
    if not base_url:
        # _upload is fail-soft by RETURN ('' on missing token / failed verify /
        # upload exception) — a discarded return exits 0 on silent durability
        # loss (upload-policy.md: 'upload returned no path' is a TRACKED GAP).
        raise RuntimeError(
            f"pooled_split HALT [manifest_upload_no_path]: _upload returned no "
            f"path for {manifest_path} -> {cm.HF_DATA_REPO}@{dest}"
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


def assert_production_fold_pin(assignment: dict[str, Any], smoke: bool) -> None:
    """v17 fail-loud pin: a FULL (non-smoke) run must realize exactly
    ``POOLED_N_FOLDS`` train folds (plan v17 §4 Phase FIT_pool pins n_folds=5).
    The v16 floor guard in ``build_split_assignment`` may lower the realized
    count only on tiny (smoke) slices — it must never silently ship
    ``n_folds != POOLED_N_FOLDS`` at production scale (~166 train groups)."""
    n_folds = int(assignment["n_folds_realized"])
    if smoke or n_folds == POOLED_N_FOLDS:
        return
    raise SystemExit(
        f"pooled_split HALT [pooled_split_n_folds_below_pin]: realized n_folds="
        f"{n_folds} != plan-pinned POOLED_N_FOLDS={POOLED_N_FOLDS} on a full "
        f"(non-smoke) run ({len(assignment['fold_by_gid'])} train groups)"
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

    # (3) Embed all deduped prompts — after stripping each corpus's shared
    # common prompt prefix (run-11802 fix: math7500's 611-token few-shot
    # preamble exceeded mpnet's 384-token window and collapsed all 7,166
    # prompts to ONE embedding; the strip is the string identity for the
    # six corpora whose shared prefix is empty). Production additionally
    # gates the text handed to the encoder on the prefix-vs-window check
    # (the cap read off the LOADED model, never hardcoded).
    model = None
    if not ctx.smoke:
        model = _load_sentence_transformer(ctx.pinned_mpnet_revision)
        logger.info("[pool] encoder %s max_seq_length=%d", MPNET_MODEL_ID, model.max_seq_length)
    ordered_rows: list[dict[str, Any]] = []
    embed_texts: list[str] = []
    prefix_strip_stats: dict[str, dict[str, Any]] = {}
    for slug in DEDUP_ORDER:
        if slug not in kept_by_corpus:
            continue
        rows_c = kept_by_corpus[slug]
        ordered_rows.extend(rows_c)
        stripped_c, prefix = strip_shared_prefix([r["prompt"] for r in rows_c])
        remaining = sorted(len(p) for p in stripped_c)
        median_remaining = remaining[len(remaining) // 2] if remaining else 0
        if model is not None:
            cap = int(model.max_seq_length)
            prefix_tokens = (
                len(model.tokenizer(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
            )
            # Gate the text ACTUALLY handed to the encoder: post-strip the
            # shared prefix is empty by construction, so this halt fires only
            # if the strip is ever bypassed — the 11802 class can never again
            # reach k-means.
            post_prefix = shared_common_prefix(stripped_c)
            post_tokens = (
                len(model.tokenizer(post_prefix, add_special_tokens=False)["input_ids"])
                if post_prefix
                else 0
            )
            check_prefix_tokens_within_cap(slug, len(post_prefix), post_tokens, cap)
        else:
            prefix_tokens = None  # smoke: hash-embed path, no tokenizer loaded
        logger.info(
            "[pool] %s: shared-prefix strip — prefix_chars=%d prefix_tokens=%s "
            "median_remaining_chars=%d (n=%d)",
            slug,
            len(prefix),
            prefix_tokens if prefix_tokens is not None else "n/a(smoke)",
            median_remaining,
            len(stripped_c),
        )
        prefix_strip_stats[slug] = {
            "prefix_chars": len(prefix),
            "prefix_tokens": prefix_tokens,
            "median_remaining_chars": median_remaining,
        }
        embed_texts.extend(stripped_c)
    assert len(embed_texts) == len(ordered_rows), (len(embed_texts), len(ordered_rows))
    assert embed_texts, "no kept prompts after dedup — refusing to embed empty set"
    vecs = embed_prompts(embed_texts, ctx.pinned_mpnet_revision, ctx.smoke, model=model)

    # (4) Per-corpus sub-clustering + cross-corpus near-dup co-assignment +
    # whole-group packing + 5-fold train partition (plan v17 §4 steps 3-6;
    # A4's window/retry enforcement lives inside; A5's inputs come out in
    # the near-dup audit block).
    assignment = build_split_assignment(ordered_rows, vecs, ctx.out_root)
    # v17 review minor 1: the floor guard is smoke-only headroom — a full run
    # must realize exactly the plan-pinned fold count (fail-loud otherwise).
    assert_production_fold_pin(assignment, ctx.smoke)

    # Row-level index for downstream consumers ("cluster" = the global
    # integer group_id of the row's (corpus, subcluster) group — the
    # int(e["cluster"]) consumer contract is unchanged).
    gid_of_row = assignment["gid_of_row"]
    arm_by_gid = assignment["arm_by_gid"]
    fold_by_gid = assignment["fold_by_gid"]
    row_index: list[dict[str, Any]] = []
    for pos, row in enumerate(ordered_rows):
        gid = int(gid_of_row[pos])
        row_index.append(
            {
                "corpus": row["corpus"],
                "prompt_idx": row.get("prompt_idx"),
                "prompt_sha": row["prompt_sha"],
                "cluster": gid,
                "arm": arm_by_gid[gid],
                "fold": fold_by_gid.get(gid),
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
        assignment,
        row_index,
        round3_keep_rate,
        per_corpus_pre_intersection=per_corpus_pre_intersection,
        prefix_strip=prefix_strip_stats,
    )
    # A failing assertion is the SIGNAL and must still halt the run — but a
    # bare raise also destroys the only artifact that explains it (the
    # manifest is written below, i.e. only on the passing path). Dump a
    # clearly-named REJECTED copy first, then re-raise unchanged. The
    # rejected copy deliberately does NOT use the canonical
    # ``split_manifest.json`` name so no downstream phase can mistake a
    # failed split for a valid one.
    try:
        assert_split(manifest, round3_keep_rate)
    except AssertionError:
        rejected_dir = ctx.out_root
        rejected_dir.mkdir(parents=True, exist_ok=True)
        rejected_path = rejected_dir / "split_manifest.rejected.json"
        rejected_path.write_text(json.dumps(manifest, indent=2) + "\n")
        logger.error(
            "[pool] split assertions FAILED — rejected manifest dumped to %s "
            "(diagnostic only; NOT a usable split)",
            rejected_path,
        )
        raise

    # Persist.
    out_dir = ctx.out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "[pool] wrote %s (n_kept_post_dedup=%d n_groups=%d n_dropped=%d "
        "pooled_test_ratio_realized=%.4f)",
        manifest_path,
        manifest["n_kept_post_dedup"],
        manifest["n_groups"],
        manifest["dropped_total"],
        manifest["pooled_test_ratio_realized"],
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
