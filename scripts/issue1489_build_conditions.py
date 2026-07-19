#!/usr/bin/env python3
"""Issue #1489 P0: build the augmentation-conditions manifest (VM, CPU-only).

Consumes the reused #1092 corpus at the pinned revision and emits
`data/issue_1489/conditions/{manifest.jsonl, conditions_meta.json,
augmentations.json, margin_rows.json}` — one manifest row per (base row x
cell) instance: 6,000 plain + 2,000 x 16 augmented = 38,000 rows at
production scale (plan §4.0).

Hard P0 asserts (plan §4.0/§4.3, all fail-loud):
- battery/eval-only exclusion: `is_eval_only.sum() == 0` over BOTH pools;
- >=150 eval rows per scoped fact-relevant topic subset, topped up against
  the FULL eligible manifest BEFORE finalizing the crossing;
- 1,200/800 eval/train split disjoint by prefix_id AND query text (built by
  PARTITIONING prefixes + query-text groups into per-side blocks — the corpus
  is a dense crossing, so a post-hoc component split cannot reach the targets);
- 650/150 train/probe split with >=50 relevant probe rows per scoped
  distillation run;
- <=2 crossed rows per prefix => >=1,000 distinct prefixes.

Usage:
    uv run python scripts/issue1489_build_conditions.py \
        [--corpus-dir data/issue_1489/hf_dl/corpus] [--stage] \
        [--out data/issue_1489/conditions] [--smoke] [--upload]
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import logging
import random
import subprocess
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from issue1489_common import (  # noqa: E402
    AUGMENT_SLUGS,
    CORPUS_PREFIX,
    CORPUS_REV,
    DISTILL_RUNS,
    HF_DATA_REPO,
    HF_PREFIX,
    MAX_GEN_TOKENS,
    MAX_MODEL_LEN,
    RELEVANCE_MAP,
    SCOPED_AUGS,
    SMOKE_AUGS,
    augment_family,
    build_augmentation_library,
    cell_for_slug,
)

logger = logging.getLogger("issue1489_build_conditions")

SEED = 0

# Prompt token budget for a base row: the augmented prompt + capped generation
# + capture boundary must fit MAX_MODEL_LEN (8192). n_tokens_instruct is the
# PLAIN render; augmentations add <= ~80 tokens (asserted at build time), so
# budget = 8192 - 1024 (gen) - 128 (aug + boundary margin) = 7040.
ROW_TOKEN_BUDGET = MAX_MODEL_LEN - MAX_GEN_TOKENS - 128

# Production sizing (plan §4.0)
N_CROSSED = 2_000
N_PLAIN_ONLY = 4_000
N_EVAL = 1_200
N_PROBE = 150
MAX_ROWS_PER_PREFIX = 2
MIN_DISTINCT_PREFIXES = 1_000
EVAL_RELEVANT_FLOOR = 150  # per scoped augmentation, asserted on the eval split
PROBE_RELEVANT_FLOOR = 50  # per scoped distillation run (critic revision r1)
N_MARGIN_ROWS_PER_SIDE = 100  # dual-DV margin subset: 100 relevant + 100 irrelevant
# Free-prefix eval fractions tried in order when block-sampling the eval side
# (the corpus is a DENSE crossing — 1,396 distinct queries / 1,095 prefixes over
# 18,779 eligible rows — so eval/train disjointness by prefix AND query text is
# achieved by PARTITIONING prefixes + query-text groups up front and sampling
# each side inside its own block; a post-hoc component split collapses into one
# giant component and cannot reach the 1,200/800 targets).
EVAL_PREFIX_FRACTIONS = (0.60, 0.62, 0.58, 0.65)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except Exception:
        return "unknown"


def stage_corpus(corpus_dir: Path) -> list[str]:
    """Stage the pinned #1092 corpus via the canonical scoped-prefix helper (#1402)."""
    import shutil

    from explore_persona_space.orchestrate import hub

    corpus_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    staged = hub.stage_hub_prefix(
        HF_DATA_REPO, CORPUS_PREFIX, corpus_dir / "_mirror", revision=CORPUS_REV
    )
    for path in staged:
        tgt = corpus_dir / path.name
        if not tgt.exists():
            shutil.copy(path, tgt)
        names.append(tgt.name)
    required = {"manifest.jsonl", "prefix_store.jsonl", "query_store.jsonl"}
    missing = required - set(names)
    if missing:
        raise FileNotFoundError(f"staged corpus missing {sorted(missing)}")
    logger.info("[p0] staged %d corpus files @ %s", len(names), CORPUS_REV[:12])
    return sorted(names)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _corpus_fingerprint(manifest_path: Path) -> str:
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:12]


def _is_battery_eval_row(row: dict) -> bool:
    """#1092 battery rows are EVAL-ONLY and excluded from BOTH pools (plan §4.0)."""
    return row.get("stratum") == "battery" or bool(row.get("is_eval_only"))


def _assert_augmentation_token_budget(library: dict[str, dict]) -> dict[str, int]:
    """Tokenize each augmentation text; assert the 128-token margin holds."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
        trust_remote_code=True,
    )
    n_tokens = {}
    for slug, spec in library.items():
        # +2 for the appended "\n\n" join inside the system block
        n = len(tok.encode("\n\n" + spec["text"], add_special_tokens=False))
        n_tokens[slug] = n
    worst = max(n_tokens.values())
    if worst > 120:
        raise AssertionError(
            f"augmentation text exceeds the 128-token budget margin: worst={worst}; "
            f"raise ROW_TOKEN_BUDGET margin deliberately instead of truncating"
        )
    return n_tokens


def _stratified_sample(
    pool: list[dict],
    n_target: int,
    rng: random.Random,
    *,
    per_prefix_cap: int | None,
    prefix_counts: collections.Counter,
    reserved_first: list[tuple[list[dict], int]] | None = None,
) -> list[dict]:
    """Deterministic topic-proportional sample with an optional per-prefix cap.

    `reserved_first` — (candidate rows, quota) pairs drawn FIRST up to each
    quota (the scoped-topic top-up pools, plan §4.0: floors are topped up
    from the full manifest BEFORE finalizing the crossing).
    """
    chosen: list[dict] = []
    chosen_ids: set[str] = set()

    def _take(row: dict) -> bool:
        rid = row["row_id"]
        if rid in chosen_ids:
            return False
        pfx = row["prefix_id"]
        if per_prefix_cap is not None and prefix_counts[pfx] >= per_prefix_cap:
            return False
        chosen.append(row)
        chosen_ids.add(rid)
        prefix_counts[pfx] += 1
        return True

    for reserved, quota in reserved_first or []:
        pool_r = sorted(reserved, key=lambda r: r["row_id"])
        rng.shuffle(pool_r)
        taken_r = 0
        for row in pool_r:
            if taken_r >= quota or len(chosen) >= n_target:
                break
            if _take(row):
                taken_r += 1

    by_topic: dict[str, list[dict]] = collections.defaultdict(list)
    for row in pool:
        if row["row_id"] not in chosen_ids:
            by_topic[row.get("topic") or "other"].append(row)
    total = sum(len(v) for v in by_topic.values())
    remaining = n_target - len(chosen)
    if remaining < 0:
        raise AssertionError("reserved draws exceeded the pool target")
    # proportional targets, then round-robin fill to the exact count
    targets = {t: int(remaining * len(v) / max(1, total)) for t, v in by_topic.items()}
    for topic in sorted(by_topic):
        rows_t = sorted(by_topic[topic], key=lambda r: r["row_id"])
        rng.shuffle(rows_t)
        taken = 0
        for row in rows_t:
            if taken >= targets[topic] or len(chosen) >= n_target:
                break
            if _take(row):
                taken += 1
    # fill any shortfall from the global remainder
    leftovers = sorted(
        (r for v in by_topic.values() for r in v if r["row_id"] not in chosen_ids),
        key=lambda r: r["row_id"],
    )
    rng.shuffle(leftovers)
    for row in leftovers:
        if len(chosen) >= n_target:
            break
        _take(row)
    if len(chosen) < n_target:
        raise AssertionError(
            f"pool exhausted at {len(chosen)}/{n_target} rows under the sampling constraints"
        )
    return chosen


def _relevant_rows(rows: list[dict], slug: str) -> list[dict]:
    topics = RELEVANCE_MAP[slug]
    return [r for r in rows if (r.get("topic") in topics)]


def _assert_side_disjoint(
    eval_rows: list[dict], train_rows: list[dict], query_text_by_id: dict[str, str]
) -> None:
    """Hard invariant: eval/train disjoint by prefix_id AND query TEXT."""
    eval_pfx = {r["prefix_id"] for r in eval_rows}
    train_pfx = {r["prefix_id"] for r in train_rows}
    if eval_pfx & train_pfx:
        raise AssertionError(
            f"prefix leakage across eval/train: {sorted(eval_pfx & train_pfx)[:5]}"
        )
    eval_q = {query_text_by_id[r["query_id"]] for r in eval_rows}
    train_q = {query_text_by_id[r["query_id"]] for r in train_rows}
    if eval_q & train_q:
        raise AssertionError(
            f"query-text leakage across eval/train ({len(eval_q & train_q)} texts)"
        )


def _block_partition_crossed(
    eligible: list[dict],
    query_text_by_id: dict[str, str],
    rng: random.Random,
    *,
    n_eval: int,
    n_train: int,
    topic_sets: list[frozenset[str]],
    eval_floor: int,
    train_need: int,
) -> tuple[list[dict], list[dict], dict]:
    """Sample the eval/train crossed pools from DISJOINT (prefix, query) blocks.

    The corpus is a dense prefix x query crossing, so disjointness by prefix
    AND query text must be built by PARTITIONING both axes up front:

    1. Scoped-topic-CAPABLE prefixes (>=2 eligible rows in the topic set) are
       allocated deliberately — the sci+personal set spans only ~122 prefixes,
       so the floors (eval >=150, train >=`train_need` at <=2 rows/prefix) are
       infeasible without reserving ~ceil(quota/2) capable prefixes per side.
    2. Remaining prefixes split by `EVAL_PREFIX_FRACTIONS` (retried in order
       when a side's pool cannot fill — fail-loud after the last fraction).
    3. Query-TEXT groups (ids sharing an identical text move together) split
       ~60/40, stratified by each group's majority row topic.
    4. Each side is sampled from ITS OWN (prefix, query) block via
       `_stratified_sample` with the scoped-topic quotas reserved FIRST.
    """
    qgroup_by_id: dict[str, str] = {
        qid: "q::" + hashlib.sha1(text.encode()).hexdigest()
        for qid, text in query_text_by_id.items()
    }
    # majority topic per query-text group (topic is a ROW attribute)
    group_topics: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in eligible:
        group_topics[qgroup_by_id[r["query_id"]]][r.get("topic") or "other"] += 1
    groups = sorted(group_topics)
    rng.shuffle(groups)
    by_major: dict[str, list[str]] = collections.defaultdict(list)
    for g in groups:
        by_major[group_topics[g].most_common(1)[0][0]].append(g)
    eval_frac_q = n_eval / max(1, n_eval + n_train)
    q_eval: set[str] = set()
    for topic in sorted(by_major):
        gs = by_major[topic]
        q_eval.update(gs[: max(1, round(len(gs) * eval_frac_q))])

    rows_by_prefix: dict[str, list[dict]] = collections.defaultdict(list)
    for r in eligible:
        rows_by_prefix[r["prefix_id"]].append(r)

    def _side_capacity(prefix: str, tset: frozenset[str], *, eval_side: bool) -> int:
        """Relevant rows this prefix can contribute IN-BLOCK on one side (cap 2)."""
        n = sum(
            1
            for r in rows_by_prefix[prefix]
            if r.get("topic") in tset and ((qgroup_by_id[r["query_id"]] in q_eval) == eval_side)
        )
        return min(MAX_ROWS_PER_PREFIX, n)

    eval_quota = {t: eval_floor + max(2, eval_floor // 30) for t in topic_sets}
    train_quota = {t: train_need + 2 for t in topic_sets}

    last_err: Exception | None = None
    for frac in EVAL_PREFIX_FRACTIONS:
        p_eval: set[str] = set()
        p_train: set[str] = set()
        # 1. scoped-capable prefixes, rarest topic set first; CAPACITY-AWARE
        #    greedy allocation (a capable prefix contributes only its IN-BLOCK
        #    relevant rows, so allocation counts capacity, not heads).
        for tset in sorted(
            topic_sets,
            key=lambda t: sum(
                1 for p in rows_by_prefix if _side_capacity(p, t, eval_side=True) > 0
            ),
        ):
            cand = [
                p
                for p in sorted(rows_by_prefix)
                if p not in p_eval | p_train
                and (
                    _side_capacity(p, tset, eval_side=True) > 0
                    or _side_capacity(p, tset, eval_side=False) > 0
                )
            ]
            rng.shuffle(cand)
            # eval first (larger quota): take prefixes by descending eval-side
            # capacity until the quota (+1 prefix margin) is covered.
            cand.sort(key=lambda p: -_side_capacity(p, tset, eval_side=True))
            cum = 0
            take_e: list[str] = []
            for p in cand:
                if cum >= eval_quota[tset] + MAX_ROWS_PER_PREFIX:
                    break
                cap = _side_capacity(p, tset, eval_side=True)
                if cap > 0:
                    take_e.append(p)
                    cum += cap
            if cum < eval_quota[tset]:
                raise AssertionError(
                    f"scoped topic set {sorted(tset)}: eval-side capacity {cum} < "
                    f"quota {eval_quota[tset]} — floors unsatisfiable at frac={frac}"
                )
            p_eval.update(take_e)
            rest = [p for p in cand if p not in p_eval]
            rest.sort(key=lambda p: -_side_capacity(p, tset, eval_side=False))
            cum_t = 0
            take_t: list[str] = []
            for p in rest:
                if cum_t >= train_quota[tset] + MAX_ROWS_PER_PREFIX:
                    break
                cap = _side_capacity(p, tset, eval_side=False)
                if cap > 0:
                    take_t.append(p)
                    cum_t += cap
            if cum_t < train_quota[tset]:
                raise AssertionError(
                    f"scoped topic set {sorted(tset)}: train-side capacity {cum_t} < "
                    f"quota {train_quota[tset]} — floors unsatisfiable at frac={frac}"
                )
            p_train.update(take_t)
        # 2. free prefixes
        free = sorted(p for p in rows_by_prefix if p not in p_eval | p_train)
        rng.shuffle(free)
        n_free_eval = round(len(free) * frac)
        p_eval.update(free[:n_free_eval])
        p_train.update(free[n_free_eval:])

        def side_pool(prefixes: set[str], qset_eval: bool) -> list[dict]:
            return [
                r
                for r in eligible
                if r["prefix_id"] in prefixes
                and ((qgroup_by_id[r["query_id"]] in q_eval) == qset_eval)
            ]

        try:
            eval_rows = _stratified_sample(
                side_pool(p_eval, True),
                n_eval,
                rng,
                per_prefix_cap=MAX_ROWS_PER_PREFIX,
                prefix_counts=collections.Counter(),
                reserved_first=[
                    (
                        [r for r in side_pool(p_eval, True) if r.get("topic") in tset],
                        eval_quota[tset],
                    )
                    for tset in topic_sets
                ],
            )
            train_rows = _stratified_sample(
                side_pool(p_train, False),
                n_train,
                rng,
                per_prefix_cap=MAX_ROWS_PER_PREFIX,
                prefix_counts=collections.Counter(),
                reserved_first=[
                    (
                        [r for r in side_pool(p_train, False) if r.get("topic") in tset],
                        train_quota[tset],
                    )
                    for tset in topic_sets
                ],
            )
        except AssertionError as exc:
            last_err = exc
            logger.warning("[p0] block partition at frac=%.2f failed (%s); retrying", frac, exc)
            continue
        _assert_side_disjoint(eval_rows, train_rows, query_text_by_id)
        stats = {
            "eval_prefix_frac_used": frac,
            "n_p_eval": len(p_eval),
            "n_p_train": len(p_train),
            "n_q_groups_eval": len(q_eval),
            "n_q_groups_train": len(groups) - len(q_eval),
        }
        return eval_rows, train_rows, stats
    raise AssertionError(
        f"block partition failed at every eval fraction {EVAL_PREFIX_FRACTIONS}: {last_err}"
    )


def _select_probe(
    train_rows: list[dict],
    rng: random.Random,
    n_probe: int,
    relevant_floor: int,
    augs: list[str],
) -> tuple[list[dict], list[dict]]:
    """150-row dose-probe subset, stratified by `relevant` per scoped run."""
    probe: list[dict] = []
    probe_ids: set[str] = set()
    scoped_runs = [s for s in DISTILL_RUNS if s in augs and RELEVANCE_MAP.get(s)]
    for slug in scoped_runs:
        rel = [r for r in _relevant_rows(train_rows, slug) if r["row_id"] not in probe_ids]
        rel = sorted(rel, key=lambda r: r["row_id"])
        rng.shuffle(rel)
        need = relevant_floor - sum(1 for r in probe if r.get("topic") in RELEVANCE_MAP[slug])
        for row in rel[: max(0, need)]:
            if len(probe) >= n_probe:
                break
            probe.append(row)
            probe_ids.add(row["row_id"])
    rest = sorted(
        (r for r in train_rows if r["row_id"] not in probe_ids), key=lambda r: r["row_id"]
    )
    rng.shuffle(rest)
    for row in rest:
        if len(probe) >= n_probe:
            break
        probe.append(row)
        probe_ids.add(row["row_id"])
    if len(probe) != n_probe:
        raise AssertionError(f"probe set landed {len(probe)} rows != {n_probe}")
    for slug in scoped_runs:
        n_rel = sum(1 for r in probe if r.get("topic") in RELEVANCE_MAP[slug])
        if n_rel < relevant_floor:
            raise AssertionError(
                f"probe relevant floor failed for {slug}: {n_rel} < {relevant_floor} "
                "(plan §4.3 critic revision r1)"
            )
    distill_train = [r for r in train_rows if r["row_id"] not in probe_ids]
    return probe, distill_train


def build(args: argparse.Namespace) -> dict:
    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    if args.smoke:
        n_crossed, n_plain, n_eval_target = 12, 12, 7
        n_probe, probe_floor, eval_floor = 3, 1, 1
        min_prefixes = 5
        augs = list(SMOKE_AUGS)
    else:
        n_crossed, n_plain, n_eval_target = N_CROSSED, N_PLAIN_ONLY, N_EVAL
        n_probe, probe_floor, eval_floor = N_PROBE, PROBE_RELEVANT_FLOOR, EVAL_RELEVANT_FLOOR
        min_prefixes = MIN_DISTINCT_PREFIXES
        augs = list(AUGMENT_SLUGS)

    manifest_path = corpus_dir / "manifest.jsonl"
    # Content identity: the corpus is staged at the PINNED HF revision
    # (CORPUS_REV) — the sha256 of the staged manifest is recorded in the meta
    # for the reproducibility card. NOTE: the plan §10 "manifest fingerprint
    # 7ef5523673d6" is the PARENT dispatch's HF revision prefix
    # (issue1092_dispatch.sh CORPUS_REV), not a content hash — the binding
    # identity checks here are the revision pin + the leg-B row count.
    corpus_fp = _corpus_fingerprint(manifest_path)
    corpus_rows = _load_jsonl(manifest_path)
    if not args.smoke and len(corpus_rows) != 21_193:
        raise AssertionError(
            f"staged corpus manifest has {len(corpus_rows)} rows != declared 21,193 "
            f"(artifact-reuse check (k) leg B; rev {CORPUS_REV[:12]})"
        )
    query_store = {
        r.get("id") or r.get("query_id"): r for r in _load_jsonl(corpus_dir / "query_store.jsonl")
    }
    query_text_by_id = {
        qid: (item.get("text") or item.get("query") or "") for qid, item in query_store.items()
    }
    for qid, text in query_text_by_id.items():
        if not text:
            raise ValueError(f"query {qid} has empty text in query_store")

    # -- eligibility filters with per-filter rejection counters (tiny-real probe
    #    discipline: the done-line names the rejecting filter) ------------------
    n_battery = 0
    n_over_budget = 0
    n_missing_query = 0
    eligible: list[dict] = []
    for row in corpus_rows:
        if _is_battery_eval_row(row):
            n_battery += 1
            continue
        if int(row.get("n_tokens_instruct", 0)) > ROW_TOKEN_BUDGET:
            n_over_budget += 1
            continue
        if row["query_id"] not in query_text_by_id:
            n_missing_query += 1
            continue
        eligible.append(row)
    if not eligible:
        raise AssertionError(
            f"0 eligible corpus rows after filters (battery={n_battery}, "
            f"over_budget={n_over_budget}, missing_query={n_missing_query})"
        )

    library = build_augmentation_library(REPO_ROOT)
    library = {k: v for k, v in library.items() if k in augs}
    aug_token_counts = (
        {} if args.skip_tokenizer_check else _assert_augmentation_token_budget(library)
    )

    # -- crossed pool: DISJOINT (prefix, query-text) blocks per side (plan §4.0;
    #    floors asserted against the FULL eligible pool via the capable-prefix
    #    allocation inside _block_partition_crossed) -----------------------------
    scoped_in_play = [s for s in SCOPED_AUGS if s in augs]
    topic_sets = sorted({RELEVANCE_MAP[s] for s in scoped_in_play}, key=sorted)
    train_need = probe_floor + max(2, probe_floor // 2)
    for tset in topic_sets:
        pool_t = [r for r in eligible if r.get("topic") in tset]
        per_pfx = collections.Counter(r["prefix_id"] for r in pool_t)
        achievable = sum(min(MAX_ROWS_PER_PREFIX, c) for c in per_pfx.values())
        min_needed = eval_floor + train_need + 5
        if achievable < min_needed:
            raise AssertionError(
                f"scoped topic set {sorted(tset)} supports only {achievable} crossed rows "
                f"under the <= {MAX_ROWS_PER_PREFIX}/prefix cap < required "
                f"{min_needed} (eval floor {eval_floor} + probe need {train_need}); "
                "plan §4.0 floors unsatisfiable — surface for re-plan"
            )
    eval_rows, train_rows, partition_stats = _block_partition_crossed(
        eligible,
        query_text_by_id,
        rng,
        n_eval=n_eval_target,
        n_train=n_crossed - n_eval_target,
        topic_sets=topic_sets,
        eval_floor=eval_floor,
        train_need=train_need,
    )
    crossed = eval_rows + train_rows
    n_prefixes = len({r["prefix_id"] for r in crossed})
    if n_prefixes < min_prefixes:
        raise AssertionError(f"crossed pool spans {n_prefixes} prefixes < floor {min_prefixes}")

    crossed_ids = {r["row_id"] for r in crossed}
    plain_pool = [r for r in eligible if r["row_id"] not in crossed_ids]
    plain_only = _stratified_sample(
        plain_pool,
        n_plain,
        rng,
        per_prefix_cap=None,
        prefix_counts=collections.Counter(),
    )

    # battery-exclusion hard assert over BOTH pools (plan §4.0)
    n_eval_only = sum(1 for r in crossed + plain_only if r.get("is_eval_only"))
    if n_eval_only != 0:
        raise AssertionError(f"is_eval_only leaked into pools: {n_eval_only} rows")

    for slug in scoped_in_play:
        n_rel = len(_relevant_rows(eval_rows, slug))
        if n_rel < eval_floor:
            raise AssertionError(
                f"eval relevant floor failed for {slug}: {n_rel} < {eval_floor} "
                "(plan §4.0: >=150 relevant eval rows per scoped augmentation)"
            )
    probe_rows, distill_rows = _select_probe(train_rows, rng, n_probe, probe_floor, augs)

    split_by_id: dict[str, str] = {}
    for r in eval_rows:
        split_by_id[r["row_id"]] = "eval"
    for r in probe_rows:
        split_by_id[r["row_id"]] = "probe"
    for r in distill_rows:
        split_by_id[r["row_id"]] = "train"
    for r in plain_only:
        split_by_id[r["row_id"]] = "plain_only"

    # -- emit conditions manifest ------------------------------------------------
    keep_keys = [
        "prefix_id",
        "query_id",
        "topic",
        "stratum",
        "prefix_source",
        "query_source",
        "n_tokens_instruct",
        "is_eval_only",
    ]

    def _emit_row(base: dict, cell_id: str, slug: str | None) -> dict:
        row = {k: base.get(k) for k in keep_keys}
        row["base_row_id"] = base["row_id"]
        row["cell_id"] = cell_id
        row["split"] = split_by_id[base["row_id"]]
        if slug is None:
            row["augment_family"] = "plain"
            row["augment_instance"] = ""
            row["augment_text"] = ""
            row["relevant"] = None
        else:
            row["augment_family"] = augment_family(slug)
            row["augment_instance"] = slug
            row["augment_text"] = library[slug]["text"]
            rel = RELEVANCE_MAP.get(slug)
            row["relevant"] = (base.get("topic") in rel) if rel is not None else None
        row["row_id"] = f"{base['row_id']}-{cell_id}"
        return row

    crossed_sorted = sorted(crossed, key=lambda r: r["row_id"])
    plain_sorted = sorted(plain_only, key=lambda r: r["row_id"])
    out_rows: list[dict] = []
    for base in crossed_sorted + plain_sorted:
        out_rows.append(_emit_row(base, "cell_plain", None))
    for slug in augs:
        cell = cell_for_slug(slug)
        for base in crossed_sorted:
            out_rows.append(_emit_row(base, cell, slug))

    expected_total = (len(crossed_sorted) + len(plain_sorted)) + len(augs) * len(crossed_sorted)
    if len(out_rows) != expected_total:
        raise AssertionError(f"manifest row count {len(out_rows)} != expected {expected_total}")
    if sum(1 for r in out_rows if r.get("is_eval_only")) != 0:
        raise AssertionError("is_eval_only rows leaked into the conditions manifest")

    manifest_out = out_dir / "manifest.jsonl"
    with open(manifest_out, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # margin subset for the dual-DV teacher-forced fixed +/- completion margin
    # (plan §6 dual-DV block): per judged fact run, 100 relevant + 100
    # irrelevant EVAL rows (deterministic).
    margin_rows: dict[str, dict] = {}
    for slug in [s for s in ("fact_veg", "fact_python") if s in augs]:
        rel = sorted(_relevant_rows(eval_rows, slug), key=lambda r: r["row_id"])
        irrel = sorted(
            (r for r in eval_rows if r.get("topic") not in RELEVANCE_MAP[slug]),
            key=lambda r: r["row_id"],
        )
        n_side = N_MARGIN_ROWS_PER_SIDE if not args.smoke else 2
        margin_rows[slug] = {
            "relevant": [r["row_id"] for r in rel[:n_side]],
            "irrelevant": [r["row_id"] for r in irrel[:n_side]],
        }
    (out_dir / "margin_rows.json").write_text(json.dumps(margin_rows, indent=2))

    (out_dir / "augmentations.json").write_text(
        json.dumps(
            {
                "library": library,
                "relevance_map": {k: sorted(v) for k, v in RELEVANCE_MAP.items()},
                "aug_token_counts": aug_token_counts,
                "scoped_augs": scoped_in_play,
                "distill_runs": [s for s in DISTILL_RUNS if s in augs],
            },
            indent=2,
        )
    )

    stats = {
        "smoke": bool(args.smoke),
        "corpus_rev": CORPUS_REV,
        "corpus_manifest_fingerprint": corpus_fp,
        "seed": SEED,
        "filters": {
            "battery_eval_only_rejected": n_battery,
            "over_token_budget_rejected": n_over_budget,
            "missing_query_rejected": n_missing_query,
            "row_token_budget": ROW_TOKEN_BUDGET,
            "eligible_after_filters": len(eligible),
        },
        "pools": {
            "n_crossed": len(crossed_sorted),
            "n_plain_only": len(plain_sorted),
            "n_distinct_prefixes_crossed": n_prefixes,
            "n_trait_stratum_crossed": sum(
                1 for r in crossed_sorted if r.get("stratum") == "trait_stratum"
            ),
            "n_trait_stratum_plain_only": sum(
                1 for r in plain_sorted if r.get("stratum") == "trait_stratum"
            ),
            "is_eval_only_sum": 0,
        },
        "splits": {
            "n_eval": len(eval_rows),
            "n_train_distill": len(distill_rows),
            "n_probe": len(probe_rows),
            "eval_target": n_eval_target,
            "block_partition": partition_stats,
        },
        "relevant_counts": {
            slug: {
                "crossed": len(_relevant_rows(crossed_sorted, slug)),
                "eval": len(_relevant_rows(eval_rows, slug)),
                "probe": len(_relevant_rows(probe_rows, slug)),
            }
            for slug in scoped_in_play
        },
        "cells": ["cell_plain"] + [cell_for_slug(s) for s in augs],
        "n_manifest_rows": len(out_rows),
        "reproducibility": {
            "git_sha": _git_sha(),
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "script": "scripts/issue1489_build_conditions.py",
        },
    }
    (out_dir / "conditions_meta.json").write_text(json.dumps(stats, indent=2))

    logger.info(
        "[p0] done: rows=%d (crossed=%d plain_only=%d) rejected: battery=%d "
        "over_budget=%d missing_query=%d | eval=%d train=%d probe=%d prefixes=%d",
        len(out_rows),
        len(crossed_sorted),
        len(plain_sorted),
        n_battery,
        n_over_budget,
        n_missing_query,
        len(eval_rows),
        len(distill_rows),
        len(probe_rows),
        n_prefixes,
    )
    return stats


def upload_conditions(out_dir: Path) -> None:
    """Upload the conditions dir to the issue-owned HF prefix (non-LFS JSON)."""
    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        out_dir,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_PREFIX}/conditions",
    )
    if not url:
        raise RuntimeError("conditions upload returned no path; refusing to proceed")
    from huggingface_hub import HfApi

    expected = [
        f"{HF_PREFIX}/conditions/{p.name}" for p in sorted(out_dir.iterdir()) if p.is_file()
    ]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        HF_DATA_REPO,
        expected,
        path_in_repo=f"{HF_PREFIX}/conditions",
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"conditions upload verify missing {missing}")
    logger.info("[p0] uploaded + verified %d conditions files -> %s", len(expected), url)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-dir", default="data/issue_1489/hf_dl/corpus")
    p.add_argument("--out", default="data/issue_1489/conditions")
    p.add_argument("--stage", action="store_true", help="stage the pinned corpus from HF first")
    p.add_argument("--smoke", action="store_true", help="tiny-real slice (12 crossed rows)")
    p.add_argument("--upload", action="store_true", help="upload conditions dir to HF")
    p.add_argument(
        "--skip-tokenizer-check",
        action="store_true",
        help="skip the augmentation token-budget tokenizer check (offline smoke only)",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.stage:
        stage_corpus(Path(args.corpus_dir))
    build(args)
    if args.upload:
        upload_conditions(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
