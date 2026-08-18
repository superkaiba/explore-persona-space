#!/usr/bin/env python
"""issue2356 P0 — build the three refusal-prediction corpora (VM-side).

Two-regime refusal prediction (Arm A harmful flip-pairs / Arm B over-refusal)
plus a generic context->answer map corpus. This driver produces THREE frozen
corpora with git manifests (prompt shas + provenance ONLY — never prompt text)
and text JSONLs uploaded to the HF data repo (prompt text lives there, never
git):

- Arm A  (``armA``):   the persisted AdvBench-moderate pilot bank verbatim ->
                       394 bases + 2,364 variants = 2,758 prompts.
- Arm B  (``armB``):   OR-Bench-Hard-1k (1,319) + PHTest controversial (1,192)
                       = 2,511 prompts, exact-count-asserted at a pinned
                       revision, length-validated under the chat-template
                       render vs the generation budget.
- generic(``generic``): 8,000 LMSYS first-turn English prompts, sha-deduped,
                       TF-IDF-cosine-disjoint from BOTH arms, seed-42 shuffled;
                       checkpointed streaming with per-filter reject counters
                       (the #1092 ``_stream_with_cache`` pattern, mirrored with
                       a citation comment below). WildChat-1M fallback.

Resume-skip keys on an INPUT FINGERPRINT per corpus (input identity sha256 +
git code sha + the output-affecting flag set) stored in a per-corpus
done-sentinel; a fingerprint mismatch forces recompute, never a bare-existence
skip (#779 stale-artifact rule; plan c24).

Content hygiene: this driver consumes a trigger-dense harmful bank and a raw
real-user corpus. It NEVER prints prompt / response text. All logging is
digest-only (shas, counts, token lengths, axis / source / category LABELS).

This is a P0 VM-side entrypoint: it imports no torch (tokenizer only) but calls
``orchestrate.env.load_dotenv`` at entry for HF creds + shared-VM thread caps.
"""

from __future__ import annotations

# load_dotenv BEFORE any heavy import (HF creds + shared-VM thread caps, #847);
# the project wrapper is stdin/heredoc-safe (never bare python-dotenv).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from datasets import load_dataset  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)
from explore_persona_space.task_workflow import find_task_path, repo_root  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger("issue2356_build_corpus")

# ---------------------------------------------------------------------------
# Constants (plan §10; every load-bearing value is grounded there)
# ---------------------------------------------------------------------------
ISSUE = 2356
SLUG = "refusalpred"
HF_PREFIX = f"issue{ISSUE}_{SLUG}"
HF_PREFIX_SMOKE = f"issue{ISSUE}_{SLUG}_smoke"
DATA_REPO = hub.DEFAULT_DATASET_REPO

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 2048
PROMPT_BUDGET = MAX_MODEL_LEN - MAX_NEW_TOKENS  # 2048; plan §10 Step A
GLOBAL_SEED = 42

# Arm A pilot bank (plan §10 A21; measured structure, sha256 recorded in manifest).
ARM_A_SOURCE = "advbench-moderate-pilot-bank"
ARM_A_BANK_FILENAME = "borderline_harmful_scan2.json"
ARM_A_BANK_SHA256 = "01eca0ea1383d56a96eecf368edbf670ca864cc46283611b131d43a565969e24"
EXPECT_VARIANTS = 2364
EXPECT_BASES = 394
EXPECT_AXES = frozenset(
    {
        "past_tense",
        "passive_voice",
        "declarative_curiosity",
        "formal_register",
        "nominalization",
        "technical_register",
    }
)
EXPECT_ARM_A_CORPUS = EXPECT_BASES + EXPECT_VARIANTS  # 2,758 bank RECORDS (pre-dedup)
# MEASURED distinct-prompt universe of the pinned bank (2026-08-17 smoke): the
# declarative_curiosity axis collapsed 8 sets of DIFFERENT bases into
# byte-identical rewritten text (7 pairs + one 4-way = 10 duplicate records),
# so the sha-keyed corpus is 2,748 distinct prompts, not the plan's 2,758.
# Each duplicate text is kept ONCE under the first base_id (one prompt sha ->
# one generation/capture row); the drop accounting is asserted + persisted.
EXPECT_ARM_A_CORPUS_DISTINCT = 2748
EXPECT_ARM_A_DEDUP_DROPPED = EXPECT_ARM_A_CORPUS - EXPECT_ARM_A_CORPUS_DISTINCT  # 10

# Arm B (plan §10; counts asserted at the pinned revision).
ORBENCH_REPO = "bench-llm/or-bench"
ORBENCH_CONFIG = "or-bench-hard-1k"
ORBENCH_N = 1319
PHTEST_REPO = "furonghuang-lab/PHTest"
PHTEST_CONFIG = "default"
PHTEST_SPLIT = "train"
PHTEST_CONTROVERSIAL_N = 1192

# generic map corpus (plan §10 Step; #1092 streaming pattern).
LMSYS_REPO = "lmsys/lmsys-chat-1m"
WILDCHAT_REPO = "allenai/WildChat-1M"
GENERIC_KEEP = 8000
TFIDF_COS_DROP = 0.4
FILTER_RECIPE_VERSION = "v1"

# ---------------------------------------------------------------------------
# Small helpers (digest-only; never print prompt/response text)
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint(payload: dict[str, Any]) -> str:
    """Stable sha256 over a JSON-serializable fingerprint payload."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _out_root(args: argparse.Namespace) -> Path:
    if args.out_root:
        return Path(args.out_root)
    return repo_root() / "eval_results" / f"issue_{ISSUE}" / "corpus"


def _hf_prefix(args: argparse.Namespace) -> str:
    return HF_PREFIX_SMOKE if args.smoke else HF_PREFIX


def _code_sha() -> str:
    return git_provenance().commit_sha


def _flag_set(args: argparse.Namespace) -> dict[str, Any]:
    """Output-affecting flags — part of every resume fingerprint (plan c24)."""
    return {
        "smoke": bool(args.smoke),
        "generic_keep": int(args.generic_keep),
        "stream_limit": int(args.stream_limit) if args.stream_limit else None,
        "prompt_budget": PROMPT_BUDGET,
        "tfidf_cos_drop": TFIDF_COS_DROP,
        "filter_recipe_version": FILTER_RECIPE_VERSION,
    }


@dataclass
class SentinelPaths:
    done: Path
    manifest: Path
    text_jsonl: Path


def _paths(args: argparse.Namespace, corpus: str) -> SentinelPaths:
    root = _out_root(args)
    sent = root / ".sentinels"
    return SentinelPaths(
        done=sent / f"{corpus}.done.json",
        manifest=root / f"{corpus}_manifest.json",
        text_jsonl=root / "text" / f"{corpus}.jsonl",
    )


def _resume_ok(p: SentinelPaths, fingerprint: str, resume: bool) -> bool:
    """True only when the sentinel exists AND its fingerprint matches exactly."""
    if not resume:
        return False
    if not (p.done.exists() and p.manifest.exists() and p.text_jsonl.exists()):
        return False
    try:
        rec = json.loads(p.done.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return rec.get("input_fingerprint") == fingerprint


def _write_outputs(
    p: SentinelPaths,
    rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    manifest_meta: dict[str, Any],
    fingerprint: str,
) -> None:
    """Write text JSONL (HF-bound; carries prompt text) + git manifest (NO text)."""
    p.text_jsonl.parent.mkdir(parents=True, exist_ok=True)
    p.manifest.parent.mkdir(parents=True, exist_ok=True)
    p.done.parent.mkdir(parents=True, exist_ok=True)

    tmp_text = p.text_jsonl.with_suffix(".jsonl.tmp")
    with open(tmp_text, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    os.replace(tmp_text, p.text_jsonl)

    manifest = {
        "issue": ISSUE,
        "n_rows": len(manifest_rows),
        "rows": manifest_rows,  # prompt_sha + labels ONLY — never prompt text
        "meta": {**manifest_meta, **as_metadata_dict(git_provenance())},
    }
    tmp_man = p.manifest.with_suffix(".json.tmp")
    tmp_man.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_man, p.manifest)

    p.done.write_text(
        json.dumps(
            {
                "input_fingerprint": fingerprint,
                "n_rows": len(rows),
                "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_commit": _code_sha(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _render_prompt_token_len(tokenizer, prompt: str) -> int:
    """Token length of the chat-template render exactly as generation consumes it."""
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    return len(ids)


# ---------------------------------------------------------------------------
# Arm A — persisted pilot bank (VM-only bank read via the canonical resolver)
# ---------------------------------------------------------------------------


def build_arm_a(args: argparse.Namespace, tokenizer) -> None:
    bank_path = find_task_path(ISSUE) / "artifacts" / ARM_A_BANK_FILENAME
    if not bank_path.exists():
        raise FileNotFoundError(f"Arm-A bank not found: {bank_path}")
    bank_sha = _sha256_file(bank_path)
    fingerprint = _fingerprint(
        {
            "corpus": "armA",
            "bank_sha256": bank_sha,
            "code_sha": _code_sha(),
            "flags": _flag_set(args),
        }
    )
    p = _paths(args, "armA")
    if _resume_ok(p, fingerprint, resume=not args.no_resume):
        logger.info("[armA] resume-skip (fingerprint match) -> %s", p.manifest)
        return

    if bank_sha != ARM_A_BANK_SHA256:
        raise ValueError(
            f"Arm-A bank sha256 mismatch: got {bank_sha}, expected {ARM_A_BANK_SHA256}"
        )
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    for key in ("stage", "variants", "candidates", "meta"):
        if key not in bank:
            raise ValueError(f"Arm-A bank missing top-level key: {key!r}")
    meta = bank["meta"]
    if meta.get("model") != MODEL:
        raise ValueError(f"Arm-A bank meta model {meta.get('model')!r} != {MODEL!r}")
    variants = bank["variants"]
    if len(variants) != EXPECT_VARIANTS:
        raise ValueError(f"Arm-A variants count {len(variants)} != {EXPECT_VARIANTS}")
    axes = {v["axis"] for v in variants}
    if axes != set(EXPECT_AXES):
        raise ValueError(f"Arm-A axes {sorted(axes)} != {sorted(EXPECT_AXES)}")
    base_ids = sorted({v["base_id"] for v in variants})
    if len(base_ids) != EXPECT_BASES:
        raise ValueError(f"Arm-A distinct bases {len(base_ids)} != {EXPECT_BASES}")

    # base prompt text per base_id (verbatim from the bank's `base` field).
    base_text: dict[Any, str] = {}
    for v in variants:
        base_text.setdefault(v["base_id"], v["base"])

    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    seen_sha: set[str] = set()

    def _add(prompt: str, base_id: Any, axis: str) -> None:
        sha = _sha256_text(prompt)
        if sha in seen_sha:
            return
        seen_sha.add(sha)
        rows.append({"prompt_sha": sha, "prompt": prompt, "base_id": base_id, "axis": axis})
        manifest_rows.append(
            {"prompt_sha": sha, "base_id": base_id, "axis": axis, "source": ARM_A_SOURCE}
        )

    for base_id, text in base_text.items():
        _add(text, base_id, "base")  # the un-rewritten base prompt
    for v in variants:
        _add(v["prompt"], v["base_id"], v["axis"])  # the 6 rewrite axes

    # 6 meta-declared bases with no variants are REPORTED, not regenerated (plan §10).
    n_meta_bases = int(meta.get("n_bases", 0))
    logger.info(
        "[armA] bases=%d variants=%d corpus=%d (meta n_bases=%d, %d declared-but-empty)",
        len(base_ids),
        len(variants),
        len(rows),
        n_meta_bases,
        max(0, n_meta_bases - len(base_ids)),
    )
    n_dropped = EXPECT_ARM_A_CORPUS - len(rows)
    if len(rows) != EXPECT_ARM_A_CORPUS_DISTINCT or n_dropped != EXPECT_ARM_A_DEDUP_DROPPED:
        raise ValueError(
            f"Arm-A corpus {len(rows)} distinct / {n_dropped} dedup-dropped != expected "
            f"{EXPECT_ARM_A_CORPUS_DISTINCT} / {EXPECT_ARM_A_DEDUP_DROPPED} "
            f"(pre-dedup records {EXPECT_ARM_A_CORPUS})"
        )

    _write_outputs(
        p,
        rows,
        manifest_rows,
        {
            "corpus": "armA",
            "source": ARM_A_SOURCE,
            "bank_sha256": bank_sha,
            "bank_meta": {k: meta.get(k) for k in ("model", "judge", "n_bases", "N_resample")},
            "n_records_pre_dedup": EXPECT_ARM_A_CORPUS,
            "n_dedup_dropped": n_dropped,
            "dedup_note": "8 cross-base declarative_curiosity duplicate texts (10 records); "
            "each kept once under the first base_id (one prompt sha = one row)",
            "note": "s1_judge/comply_rate are screening-vintage and NOT reused as labels",
        },
        fingerprint,
    )


# ---------------------------------------------------------------------------
# Arm B — OR-Bench-Hard-1k + PHTest controversial (pinned + count-asserted)
# ---------------------------------------------------------------------------


def _resolve_dataset_revision(repo: str) -> str:
    """Resolve the current dataset commit sha (recorded in the manifest; never
    fabricated / hardcoded — the plan pins whatever revision resolves at P0)."""
    info = HfApi().dataset_info(repo)
    return info.sha


def build_arm_b(args: argparse.Namespace, tokenizer) -> None:
    orbench_rev = _resolve_dataset_revision(ORBENCH_REPO)
    phtest_rev = _resolve_dataset_revision(PHTEST_REPO)
    fingerprint = _fingerprint(
        {
            "corpus": "armB",
            "orbench": {"repo": ORBENCH_REPO, "config": ORBENCH_CONFIG, "rev": orbench_rev},
            "phtest": {"repo": PHTEST_REPO, "config": PHTEST_CONFIG, "rev": phtest_rev},
            "code_sha": _code_sha(),
            "flags": _flag_set(args),
        }
    )
    p = _paths(args, "armB")
    if _resume_ok(p, fingerprint, resume=not args.no_resume):
        logger.info("[armB] resume-skip (fingerprint match) -> %s", p.manifest)
        return

    orbench = load_dataset(ORBENCH_REPO, ORBENCH_CONFIG, split="train", revision=orbench_rev)
    if not args.smoke and len(orbench) != ORBENCH_N:
        raise ValueError(f"OR-Bench-Hard-1k rows {len(orbench)} != {ORBENCH_N}")

    phtest_all = load_dataset(PHTEST_REPO, PHTEST_CONFIG, split=PHTEST_SPLIT, revision=phtest_rev)
    phtest = [r for r in phtest_all if r.get("Harmfulness") == "controversial"]
    if not args.smoke and len(phtest) != PHTEST_CONTROVERSIAL_N:
        raise ValueError(f"PHTest controversial rows {len(phtest)} != {PHTEST_CONTROVERSIAL_N}")

    limit = 6 if args.smoke else None
    candidates: list[dict[str, Any]] = []
    for r in list(orbench)[:limit]:
        candidates.append(
            {"prompt": r["prompt"], "source": "or-bench-hard-1k", "category": r.get("category")}
        )
    for r in phtest[:limit]:
        # Observed PHTest schema (probed 2026-08-17): columns are
        # ['ID', 'Request', 'Harmfulness'] — the prompt text lives in 'Request'.
        candidates.append(
            {"prompt": r["Request"], "source": "phtest-controversial", "category": "controversial"}
        )

    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []  # digest-only (no prompt text)
    for c in candidates:
        prompt = c["prompt"]
        n_tok = _render_prompt_token_len(tokenizer, prompt)
        sha = _sha256_text(prompt)
        if n_tok > PROMPT_BUDGET:
            dropped.append({"prompt_sha": sha, "source": c["source"], "n_tok": n_tok})
            continue
        rows.append(
            {"prompt_sha": sha, "prompt": prompt, "source": c["source"], "category": c["category"]}
        )
        manifest_rows.append(
            {"prompt_sha": sha, "source": c["source"], "category": c["category"], "n_tok": n_tok}
        )

    logger.info(
        "[armB] kept=%d dropped_overlong=%d (budget=%d) orbench_rev=%s phtest_rev=%s",
        len(rows),
        len(dropped),
        PROMPT_BUDGET,
        orbench_rev[:12],
        phtest_rev[:12],
    )
    _write_outputs(
        p,
        rows,
        manifest_rows,
        {
            "corpus": "armB",
            "orbench": {"repo": ORBENCH_REPO, "config": ORBENCH_CONFIG, "revision": orbench_rev},
            "phtest": {"repo": PHTEST_REPO, "config": PHTEST_CONFIG, "revision": phtest_rev},
            "prompt_budget": PROMPT_BUDGET,
            "dropped_overlong": dropped,
        },
        fingerprint,
    )


# ---------------------------------------------------------------------------
# generic — checkpointed streaming (#1092 `_stream_with_cache` PATTERN, mirrored
# here with citation comments; NOT imported — a scripts/issue* private helper).
# ---------------------------------------------------------------------------


def _stream_fingerprint(
    dataset_repo: str, revision: str, *, stream_limit: int | None
) -> dict[str, Any]:
    """Exact-match resume fingerprint for a persisted stream pool.

    Mirrors scripts/issue1092_build_corpus.py::_stream_fingerprint — dataset
    identity (repo + pinned revision) + every filter-relevant constant.
    """
    return {
        "dataset_repo": dataset_repo,
        "revision": revision,
        "lang_filter": "English",
        "stream_limit": stream_limit,
        "prompt_budget": PROMPT_BUDGET,
        "filter_recipe_version": FILTER_RECIPE_VERSION,
    }


def _first_user_turn(row: dict[str, Any]) -> str | None:
    conv = row.get("conversation")
    if not isinstance(conv, list) or not conv:
        return None
    first = conv[0]
    if not isinstance(first, dict) or first.get("role") != "user":
        return None
    content = first.get("content")
    return content if isinstance(content, str) and content.strip() else None


def _stream_conversations(
    dataset_repo: str,
    revision: str,
    tokenizer,
    *,
    stream_limit: int | None,
    stats_out: dict[str, Any],
) -> list[dict[str, Any]]:
    """Stream first-turn English prompts with per-filter reject counters.

    WildChat-1M carries top-level `redacted`/`toxic` bools (screened out);
    LMSYS carries neither (skipped filters no-op). Language fields are FULL
    NAMES ("English"), never ISO codes (#1092 real-corpus filter trap).
    """
    ds = load_dataset(dataset_repo, split="train", streaming=True, revision=revision)
    rejects = {"lang": 0, "toxic_redacted": 0, "no_first_user": 0, "overlong": 0, "dup_sha": 0}
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    streamed = 0
    # Explicit iterator handle so a `break` can .close() the streaming pipeline
    # before interpreter shutdown (#1947 SIGABRT-on-shutdown avoidance).
    it = iter(ds)
    try:
        for row in it:
            streamed += 1
            if stream_limit is not None and streamed > stream_limit:
                break
            if row.get("language") != "English":
                rejects["lang"] += 1
                continue
            if bool(row.get("toxic")) or bool(row.get("redacted")):
                rejects["toxic_redacted"] += 1
                continue
            prompt = _first_user_turn(row)
            if prompt is None:
                rejects["no_first_user"] += 1
                continue
            sha = _sha256_text(prompt)
            if sha in seen:
                rejects["dup_sha"] += 1
                continue
            if _render_prompt_token_len(tokenizer, prompt) > PROMPT_BUDGET:
                rejects["overlong"] += 1
                continue
            seen.add(sha)
            kept.append({"prompt_sha": sha, "prompt": prompt})
    finally:
        close = getattr(it, "close", None)
        if callable(close):
            close()
    stats_out.update({"kept": len(kept), "streamed": streamed, "rejects": rejects})
    return kept


def _stream_with_cache(
    dataset_repo: str,
    revision: str,
    tokenizer,
    *,
    stream_limit: int | None,
    cache_dir: Path,
    resume: bool,
    stats_out: dict[str, Any],
) -> list[dict[str, Any]]:
    """Per-source on-disk checkpoint + exact-fingerprint resume.

    Mirrors scripts/issue1092_build_corpus.py::_stream_with_cache — pool file
    written FIRST, meta sidecar LAST (both via os.replace); an EXACT fingerprint
    match loads the pool and SKIPS the stream; text-mode line iteration (never
    .splitlines(), #950 U+2028). ``resume=False`` forces a re-stream.
    """
    source_tag = dataset_repo.split("/")[-1].replace(".", "_")
    fp = _stream_fingerprint(dataset_repo, revision, stream_limit=stream_limit)
    pool_path = cache_dir / f"{source_tag}.jsonl"
    meta_path = cache_dir / f"{source_tag}.meta.json"

    if resume and meta_path.exists() and pool_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fp:
            results: list[dict[str, Any]] = []
            with open(pool_path, encoding="utf-8") as fh:
                for line in fh:  # text-mode iteration, never .splitlines()
                    s = line.strip("\n")
                    if s:
                        results.append(json.loads(s))
            if len(results) != meta.get("kept"):
                raise RuntimeError(
                    f"[stream-cache {source_tag}] pool rows {len(results)} != "
                    f"meta kept={meta.get('kept')} — corrupt cache; pass --no-resume"
                )
            logger.info(
                "[stream-cache %s] RESUMED %d rows — stream SKIPPED", source_tag, len(results)
            )
            stats_out.update({k: meta[k] for k in ("kept", "streamed", "rejects") if k in meta})
            stats_out["resumed_from_cache"] = True
            return results

    stats: dict[str, Any] = {}
    results = _stream_conversations(
        dataset_repo, revision, tokenizer, stream_limit=stream_limit, stats_out=stats
    )
    stats["resumed_from_cache"] = False

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_pool = cache_dir / (pool_path.name + ".tmp")
    with open(tmp_pool, "w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    os.replace(tmp_pool, pool_path)  # pool FIRST
    tmp_meta = cache_dir / (meta_path.name + ".tmp")
    with open(tmp_meta, "w", encoding="utf-8") as fh:
        json.dump({"fingerprint": fp, **stats}, fh, indent=2)
    os.replace(tmp_meta, meta_path)  # meta LAST
    stats_out.update(stats)
    return results


def _arm_prompts_for_disjointness(args: argparse.Namespace) -> list[str]:
    """Load Arm-A + Arm-B prompt TEXT from the just-built text JSONLs for the
    TF-IDF disjointness screen (map/eval disjointness against BOTH arms)."""
    texts: list[str] = []
    for corpus in ("armA", "armB"):
        jl = _paths(args, corpus).text_jsonl
        if not jl.exists():
            raise FileNotFoundError(f"generic disjointness needs {corpus} text JSONL first: {jl}")
        with open(jl, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip("\n")
                if s:
                    texts.append(json.loads(s)["prompt"])
    return texts


def build_generic(args: argparse.Namespace, tokenizer) -> None:
    keep = int(args.generic_keep)
    stream_limit = int(args.stream_limit) if args.stream_limit else None
    repo = LMSYS_REPO
    try:
        revision = _resolve_dataset_revision(repo)
    except Exception:  # noqa: BLE001 — fall back to WildChat on any LMSYS access failure
        logger.warning("[generic] LMSYS access failed; falling back to WildChat-1M")
        repo = WILDCHAT_REPO
        revision = _resolve_dataset_revision(repo)

    fingerprint = _fingerprint(
        {
            "corpus": "generic",
            "repo": repo,
            "revision": revision,
            "keep": keep,
            "arm_a_sha": _sha256_file(_paths(args, "armA").text_jsonl),
            "arm_b_sha": _sha256_file(_paths(args, "armB").text_jsonl),
            "code_sha": _code_sha(),
            "flags": _flag_set(args),
        }
    )
    p = _paths(args, "generic")
    if _resume_ok(p, fingerprint, resume=not args.no_resume):
        logger.info("[generic] resume-skip (fingerprint match) -> %s", p.manifest)
        return

    cache_dir = _out_root(args) / ".stream_cache"
    stats: dict[str, Any] = {}
    pool = _stream_with_cache(
        repo,
        revision,
        tokenizer,
        stream_limit=stream_limit,
        cache_dir=cache_dir,
        resume=not args.no_resume,
        stats_out=stats,
    )
    if not pool:
        raise RuntimeError(
            f"[generic] streamed pool is EMPTY from {repo} — filter chain rejected all rows"
        )
    logger.info("[generic] streamed pool=%d rejects=%s", len(pool), stats.get("rejects"))

    # TF-IDF cosine disjointness vs BOTH arms (drop candidate if max cos >= 0.4).
    arm_texts = _arm_prompts_for_disjointness(args)
    cand_texts = [r["prompt"] for r in pool]
    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    vec.fit(arm_texts + cand_texts)
    arm_mat = vec.transform(arm_texts)
    disjoint: list[dict[str, Any]] = []
    n_dropped_tfidf = 0
    # Chunk the candidate transform to bound the (n_cand, n_arm) cosine intermediate.
    chunk = 512
    for start in range(0, len(pool), chunk):
        sub = pool[start : start + chunk]
        cand_mat = vec.transform([r["prompt"] for r in sub])
        sims = cosine_similarity(cand_mat, arm_mat)  # (chunk, n_arm)
        maxcos = sims.max(axis=1)
        for row, mc in zip(sub, maxcos):
            if mc >= TFIDF_COS_DROP:
                n_dropped_tfidf += 1
                continue
            disjoint.append(row)

    rng = random.Random(GLOBAL_SEED)
    rng.shuffle(disjoint)
    kept_rows = disjoint[:keep]
    logger.info(
        "[generic] disjoint=%d dropped_tfidf=%d kept=%d (target=%d) repo=%s rev=%s",
        len(disjoint),
        n_dropped_tfidf,
        len(kept_rows),
        keep,
        repo,
        revision[:12],
    )

    rows = [
        {"prompt_sha": r["prompt_sha"], "prompt": r["prompt"], "source": "lmsys-first-turn-en"}
        for r in kept_rows
    ]
    manifest_rows = [
        {"prompt_sha": r["prompt_sha"], "source": "lmsys-first-turn-en"} for r in kept_rows
    ]
    _write_outputs(
        p,
        rows,
        manifest_rows,
        {
            "corpus": "generic",
            "repo": repo,
            "revision": revision,
            "keep_target": keep,
            "tfidf_cos_drop": TFIDF_COS_DROP,
            "stream_stats": stats,
            "n_dropped_tfidf": n_dropped_tfidf,
        },
        fingerprint,
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload_corpus_text(args: argparse.Namespace) -> None:
    """Upload the three corpus text JSONLs (prompt text) to the HF data repo in
    ONE bulk folder commit, then verify the exact expected set landed."""
    prefix = _hf_prefix(args)
    text_dir = _out_root(args) / "text"
    if not text_dir.exists():
        raise FileNotFoundError(f"corpus text dir missing: {text_dir}")
    path_in_repo = f"{prefix}/corpus"
    # HUB_DIR_FILECOUNT_EXEMPT: three JSONL files, far under the 10k/dir cap.
    base_url = hub._upload(
        local_path=text_dir,
        repo_id=DATA_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        raise_on_error=True,
    )
    if not base_url:
        raise RuntimeError(f"corpus text upload returned no path ({path_in_repo})")
    expected = [f"{c}.jsonl" for c in ("armA", "armB", "generic")]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), DATA_REPO, expected, path_in_repo=path_in_repo, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(f"corpus text upload incomplete; missing on Hub: {missing}")
    logger.info(
        "[upload] corpus text -> %s/%s (verified %d files)", DATA_REPO, path_in_repo, len(expected)
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

PHASES = {
    "armA": build_arm_a,
    "armB": build_arm_b,
    "generic": build_generic,
}


def _run_phase(name: str, args: argparse.Namespace, tokenizer) -> None:
    logger.info("[phase=%s]", name)
    PHASES[name](args, tokenizer)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="issue2356 P0 corpus builder")
    ap.add_argument(
        "--phase",
        choices=["armA", "armB", "generic", "all"],
        default="all",
        help="which corpus to build (default: all; generic requires armA+armB first)",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny slice; smoke HF prefix")
    ap.add_argument("--no-resume", action="store_true", help="ignore done-sentinels; recompute")
    ap.add_argument("--generic-keep", type=int, default=GENERIC_KEEP, help="generic corpus size")
    ap.add_argument(
        "--stream-limit",
        type=int,
        default=None,
        help="cap total streamed rows for the generic stream (probe / smoke)",
    )
    ap.add_argument("--out-root", default=None, help="override the corpus output root")
    ap.add_argument("--skip-upload", action="store_true", help="build only; skip HF upload")
    ap.add_argument(
        "--import-check", action="store_true", help="verify imports + args attrs; exit 0"
    )
    return ap


def main() -> int:
    args = build_argparser().parse_args()

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        logger.info("[import-check] imports + args attributes OK")
        return 0

    if args.smoke:
        args.generic_keep = min(args.generic_keep, 32)
        if args.stream_limit is None:
            args.stream_limit = 2000

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    phases = ["armA", "armB", "generic"] if args.phase == "all" else [args.phase]
    for name in phases:
        _run_phase(name, args, tokenizer)

    if not args.skip_upload:
        # upload only the corpora that were (re)built this run and exist on disk
        upload_corpus_text(args)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
