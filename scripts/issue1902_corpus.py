#!/usr/bin/env python3
"""Issue #1902 Phase P0 — two-corpus build from LMSYS-Chat-1M (VM CPU, 0 GPU).

Streams ``lmsys/lmsys-chat-1m`` at a pinned revision (deterministic order,
checkpoint-per-chunk + fingerprint-gated resume per the
``issue1092_build_corpus.py::_stream_with_cache`` pattern) and builds BOTH
corpora in one shared pass (plan #1902 §4 P0):

1. Single-turn context-arm corpus (``corpus_single.jsonl``): 16,000 generic
   first-turn user queries + a 1,200-row keyword-screened math/code stratum,
   plus marked tier-2 strata — 500 GSM8K questions + 300 MBPP prompts
   (separate class labels, excluded from generic clusters).
2. Multi-turn prefix-arm corpus (``corpus_multi.jsonl``): 16,000 unique-prefix
   English conversations with >=2 prior turns before a final user query.

Filters: ``language == "English"`` (LMSYS stores FULL language names),
exact-string dedup, formatted-prompt token budget <= 3,072 under the LONGEST
render (both the chat-template render and the plain render are tokenized;
max is filtered). Context clusters: MiniLM embeddings + k-means k=40 per
corpus (seed 42). Outputs upload fail-loud to
``superkaiba1/explore-persona-space-data:issue1902_stage_map/corpus/``.

Usage::

    # tiny-real streaming probe (bounded: kept-cap 50, scan-cap 5,000):
    uv run python scripts/issue1902_corpus.py --probe

    # production build (VM; carry the shared-VM thread caps):
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \\
    uv run python scripts/issue1902_corpus.py --full

Content-hygiene protocol (LMSYS is unscreened real user text): corpus row
text is NEVER printed or logged — logs and probe digests carry filenames,
indices, counts, and field-STRUCTURE (key names + types) only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
# VM thread caps (#847/#891) — set BEFORE any torch/numpy import so their
# thread pools freeze at the capped width; explicit launcher env still wins.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1902_common as C  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("issue1902_corpus")

# Bumping this string invalidates every persisted stream checkpoint
# (fingerprint-gated resume; #952 gate-5 pattern).
FILTER_RECIPE_VERSION = "issue1902-p0-v1"
CHECKPOINT_EVERY_SCANNED = 20_000
# Conservative fast pre-reject before tokenization: no realistic BPE vocab
# averages >= ~65 chars/token, so > 200k chars can never fit 3,072 tokens.
FAST_REJECT_CHARS = 200_000

PROBE_SCAN_CAP = 5_000
PROBE_KEPT_CAP = 50
PROBE_MATHCODE_CAP = 8
PROBE_MARKED_CAP = 2

BUCKET_SINGLE_GENERIC = "single_generic"
BUCKET_SINGLE_MATHCODE = "single_mathcode"
BUCKET_MULTI = "multi"

# ── tokenizer (lazy, single instance) ────────────────────────────────────────

_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        # P0 predates the P1 revision pins; the resolved tokenizer sha is
        # recorded in manifest_stats.json by main() for provenance.
        _TOKENIZER = AutoTokenizer.from_pretrained(C.MODEL_IDS["R"])
    return _TOKENIZER


def _formatted_token_counts(query: str, prefix_turns: list[dict] | None = None) -> tuple[int, int]:
    """(n_tokens_chat, n_tokens_plain) of the FORMATTED generation prompts.

    The budget filter keeps a row iff max(chat, plain) <= MAX_FORMATTED_TOKENS
    — i.e. the row fits under the LONGEST render (plan §4 P0 / A10). The chat
    render embeds its special tokens as text, so both encodes use
    ``add_special_tokens=False``.
    """
    tok = _get_tokenizer()
    chat = C.render_chat_prompt(tok, query, prefix_turns)
    plain = C.render_plain_prompt(query, prefix_turns)
    n_chat = len(tok.encode(chat, add_special_tokens=False))
    n_plain = len(tok.encode(plain, add_special_tokens=False))
    return n_chat, n_plain


# ── row helpers (structure only — never text) ────────────────────────────────


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _structure_digest(row: dict) -> dict[str, str]:
    """Field-STRUCTURE digest of a raw dataset row: key names + types + list
    lengths + nested dict key names. Never any value text."""
    digest: dict[str, str] = {}
    for k in sorted(row.keys()):
        v = row[k]
        if isinstance(v, list):
            inner = ""
            if v and isinstance(v[0], dict):
                inner = "{" + ",".join(sorted(v[0].keys())) + "}"
            digest[k] = f"list[n={len(v)}]{inner}"
        elif isinstance(v, dict):
            digest[k] = "dict{" + ",".join(sorted(v.keys())) + "}"
        else:
            digest[k] = type(v).__name__
    return digest


def _normalize_turns(conv_raw: list) -> list[dict]:
    """Normalize an LMSYS conversation to [{role: user|assistant, content}]."""
    turns: list[dict] = []
    for t in conv_raw:
        if not isinstance(t, dict):
            continue
        role = (t.get("role") or t.get("from") or "").lower()
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "bot"):
            role = "assistant"
        else:
            continue  # skip system / unknown roles
        content = t.get("content") or t.get("value") or ""
        if content:
            turns.append({"role": role, "content": content})
    return turns


def _is_mathcode(text: str) -> bool:
    tl = text.lower()
    return any(kw in tl for kw in C.MATHCODE_KEYWORDS)


def _multi_prefix_and_query(turns: list[dict]) -> tuple[list[dict], str] | None:
    """Extract (prefix_turns, final user query) for the multi-turn corpus.

    Requires >=2 prior turns (>=1 completed user+assistant exchange) before a
    final user query; a trailing assistant turn is dropped; prefix roles must
    alternate user/assistant starting user and ending assistant (plan §4 P0).
    Returns None when the conversation does not qualify.
    """
    t = list(turns)
    if t and t[-1]["role"] == "assistant":
        t = t[:-1]
    if len(t) < 3 or t[-1]["role"] != "user":
        return None
    prefix = t[:-1]
    if len(prefix) < 2 or prefix[0]["role"] != "user" or prefix[-1]["role"] != "assistant":
        return None
    for a, b in pairwise(prefix):
        if a["role"] == b["role"]:
            return None
    query = t[-1]["content"].strip()
    if not query:
        return None
    return prefix, query


# ── LMSYS stream with checkpoint-per-chunk + fingerprint-gated resume ────────


def _stream_fingerprint(quotas: dict[str, int], scan_cap: int, probe: bool) -> dict[str, Any]:
    """Exact-match resume fingerprint: dataset identity + every filter-relevant
    constant. A persisted pool is resumed ONLY on an exact dict match."""
    return {
        "dataset_repo": C.LMSYS_DATASET,
        "revision": C.LMSYS_REVISION,
        "lang_filter": C.LANG_FILTER,
        "quotas": dict(sorted(quotas.items())),
        "scan_cap": scan_cap,
        "max_formatted_tokens": C.MAX_FORMATTED_TOKENS,
        "mathcode_keywords_sha": _sha16("|".join(C.MATHCODE_KEYWORDS)),
        "tokenizer_model": C.MODEL_IDS["R"],
        "filter_recipe_version": FILTER_RECIPE_VERSION,
        "probe": probe,
    }


def _fresh_rejects() -> dict[str, int]:
    return {
        "language": 0,
        "empty_conversation": 0,
        "single_structure": 0,
        "single_duplicate": 0,
        "single_quota_full": 0,
        "single_token_budget": 0,
        "multi_structure": 0,
        "multi_duplicate": 0,
        "multi_quota_full": 0,
        "multi_token_budget": 0,
    }


def _read_pool_rows(pool_path: Path, n_rows: int) -> list[dict]:
    """Read the first ``n_rows`` JSONL rows (text-mode line iteration — never
    ``.splitlines()``; #825/#950 U+2028 rule). A torn tail line beyond the
    meta-recorded count is dropped by construction."""
    rows: list[dict] = []
    with open(pool_path, encoding="utf-8") as f:
        for line in f:
            if len(rows) >= n_rows:
                break
            stripped = line.strip("\n")
            if stripped:
                rows.append(json.loads(stripped))
    if len(rows) != n_rows:
        raise RuntimeError(
            f"stream-cache pool {pool_path} holds {len(rows)} rows but meta records "
            f"{n_rows} — corrupt cache; delete it or pass --no-resume-stream"
        )
    return rows


def _write_checkpoint(
    pool_path: Path,
    meta_path: Path,
    fingerprint: dict,
    rows: list[dict],
    scanned: int,
    rejects: dict[str, int],
    kept_redacted: int,
    complete: bool,
) -> None:
    """Atomic checkpoint: pool JSONL FIRST, meta sidecar LAST (meta presence +
    matching fingerprint + row count == valid checkpoint)."""
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_pool = pool_path.with_suffix(pool_path.suffix + ".tmp")
    with open(tmp_pool, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    os.replace(tmp_pool, pool_path)
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fingerprint": fingerprint,
                "scanned": scanned,
                "kept_total": len(rows),
                "rejects": rejects,
                "kept_redacted": kept_redacted,
                "complete": complete,
            },
            f,
            indent=2,
        )
    os.replace(tmp_meta, meta_path)


def _bucket_counts(rows: list[dict]) -> dict[str, int]:
    counts = {BUCKET_SINGLE_GENERIC: 0, BUCKET_SINGLE_MATHCODE: 0, BUCKET_MULTI: 0}
    for r in rows:
        counts[r["bucket"]] += 1
    return counts


def _stream_lmsys(
    quotas: dict[str, int],
    scan_cap: int,
    cache_dir: Path,
    *,
    resume: bool = True,
    probe: bool = False,
    stats_out: dict | None = None,
) -> list[dict]:
    """One deterministic-order pass over LMSYS filling BOTH corpora's buckets.

    Kept rows persist to ``<cache_dir>/lmsys_pool.jsonl`` with a fingerprinted
    meta sidecar every CHECKPOINT_EVERY_SCANNED scanned rows (the #1092
    ``_stream_with_cache`` pattern, extended to mid-stream resume: on an exact
    fingerprint match the pool reloads and the stream skips the already-
    scanned rows). Row text is never logged.
    """
    fingerprint = _stream_fingerprint(quotas, scan_cap, probe)
    pool_path = cache_dir / "lmsys_pool.jsonl"
    meta_path = cache_dir / "lmsys_pool.meta.json"

    rows: list[dict] = []
    scanned = 0
    rejects = _fresh_rejects()
    kept_redacted = 0
    field_digests: list[dict] = []

    if not resume:
        logger.info("[stream] --no-resume-stream: ignoring any persisted pool")
    elif meta_path.exists() and pool_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        cached_fp = meta.get("fingerprint") or {}
        if cached_fp == fingerprint:
            rows = _read_pool_rows(pool_path, int(meta["kept_total"]))
            scanned = int(meta["scanned"])
            rejects.update(meta.get("rejects") or {})
            kept_redacted = int(meta.get("kept_redacted", 0))
            if meta.get("complete"):
                logger.info(
                    "[stream-cache] RESUMED COMPLETE pool: %d rows (%d scanned) — stream SKIPPED",
                    len(rows),
                    scanned,
                )
                if stats_out is not None:
                    stats_out.update(
                        {
                            "scanned": scanned,
                            "rejects": rejects,
                            "kept_redacted": kept_redacted,
                            "resumed_from_cache": True,
                            "bucket_counts": _bucket_counts(rows),
                        }
                    )
                return rows
            logger.info(
                "[stream-cache] RESUMING mid-stream: %d rows kept, skipping %d scanned rows",
                len(rows),
                scanned,
            )
        else:
            diff = sorted(
                k
                for k in set(fingerprint) | set(cached_fp)
                if cached_fp.get(k) != fingerprint.get(k)
            )
            logger.info("[stream-cache] fingerprint MISMATCH on keys %s — re-streaming", diff)
            rows, scanned = [], 0

    # Rebuild dedup sets from the (possibly resumed) kept rows.
    seen_single: set[str] = set()
    seen_multi: set[str] = set()
    for r in rows:
        if r["bucket"] == BUCKET_MULTI:
            seen_multi.add(r["prefix_sha"])
        else:
            seen_single.add(_sha16(r["query"]))
    counts = _bucket_counts(rows)

    def _quotas_met() -> bool:
        return all(counts[b] >= quotas[b] for b in counts)

    if _quotas_met():
        logger.info("[stream] all quotas already met from cache — stream SKIPPED")
    else:
        from datasets import load_dataset  # lazy import

        ds = load_dataset(
            C.LMSYS_DATASET,
            split="train",
            streaming=True,
            revision=C.LMSYS_REVISION,
        )
        if scanned:
            ds = ds.skip(scanned)

        t0 = time.time()
        row: dict | None = None
        for row in ds:
            if scanned >= scan_cap or _quotas_met():
                break
            source_index = scanned  # position in the pinned deterministic stream
            scanned += 1

            if probe and len(field_digests) < 3:
                field_digests.append(_structure_digest(row))

            if scanned % 50_000 == 0:
                logger.info(
                    "[stream] %d scanned, kept=%s, rejects=%s (%.0fs)",
                    scanned,
                    json.dumps(counts),
                    json.dumps(rejects),
                    time.time() - t0,
                )
            if scanned % CHECKPOINT_EVERY_SCANNED == 0:
                _write_checkpoint(
                    pool_path,
                    meta_path,
                    fingerprint,
                    rows,
                    scanned,
                    rejects,
                    kept_redacted,
                    complete=False,
                )

            # language filter — LMSYS stores full names ('English'); plan §4 P0.
            lang = (row.get("language") or "").strip().lower()
            if lang != C.LANG_FILTER.lower():
                rejects["language"] += 1
                continue

            turns = _normalize_turns(row.get("conversation") or [])
            if not turns:
                rejects["empty_conversation"] += 1
                continue
            redacted = bool(row.get("redacted"))

            # ── single-turn candidate ────────────────────────────────────────
            if turns[0]["role"] == "user" and turns[0]["content"].strip():
                query = turns[0]["content"].strip()
                if len(query) > FAST_REJECT_CHARS:
                    rejects["single_token_budget"] += 1
                elif _sha16(query) in seen_single:
                    rejects["single_duplicate"] += 1
                else:
                    mathcode = _is_mathcode(query)
                    if mathcode and counts[BUCKET_SINGLE_MATHCODE] < quotas[BUCKET_SINGLE_MATHCODE]:
                        bucket = BUCKET_SINGLE_MATHCODE
                    elif counts[BUCKET_SINGLE_GENERIC] < quotas[BUCKET_SINGLE_GENERIC]:
                        bucket = BUCKET_SINGLE_GENERIC
                    else:
                        bucket = None
                    if bucket is None:
                        rejects["single_quota_full"] += 1
                    else:
                        n_chat, n_plain = _formatted_token_counts(query)
                        if max(n_chat, n_plain) > C.MAX_FORMATTED_TOKENS:
                            rejects["single_token_budget"] += 1
                        else:
                            seen_single.add(_sha16(query))
                            counts[bucket] += 1
                            kept_redacted += int(redacted)
                            rows.append(
                                {
                                    "bucket": bucket,
                                    "query": query,
                                    "n_tokens_chat": n_chat,
                                    "n_tokens_plain": n_plain,
                                    "source_index": source_index,
                                }
                            )
            else:
                rejects["single_structure"] += 1

            # ── multi-turn candidate (same row may feed both corpora) ────────
            if counts[BUCKET_MULTI] >= quotas[BUCKET_MULTI]:
                rejects["multi_quota_full"] += 1
                continue
            pq = _multi_prefix_and_query(turns)
            if pq is None:
                rejects["multi_structure"] += 1
                continue
            prefix, query = pq
            prefix_sha = _sha16(
                json.dumps([(t["role"], t["content"]) for t in prefix], ensure_ascii=False)
            )
            if prefix_sha in seen_multi:
                rejects["multi_duplicate"] += 1
                continue
            total_chars = len(query) + sum(len(t["content"]) for t in prefix)
            if total_chars > FAST_REJECT_CHARS:
                rejects["multi_token_budget"] += 1
                continue
            n_chat, n_plain = _formatted_token_counts(query, prefix)
            if max(n_chat, n_plain) > C.MAX_FORMATTED_TOKENS:
                rejects["multi_token_budget"] += 1
                continue
            seen_multi.add(prefix_sha)
            counts[BUCKET_MULTI] += 1
            kept_redacted += int(redacted)
            rows.append(
                {
                    "bucket": BUCKET_MULTI,
                    "prefix_turns": prefix,
                    "query": query,
                    "n_prior_turns": len(prefix),
                    "prefix_sha": prefix_sha,
                    "n_tokens_chat": n_chat,
                    "n_tokens_plain": n_plain,
                    "source_index": source_index,
                }
            )

        # Release the streaming dataset deterministically (rc=134 SIGABRT
        # guard, #952: an IterableDataset surviving to shutdown aborts).
        del ds, row
        gc.collect()

    _write_checkpoint(
        pool_path, meta_path, fingerprint, rows, scanned, rejects, kept_redacted, complete=True
    )

    logger.info(
        "[stream] done: scanned=%d kept=%s rejects=%s kept_redacted=%d",
        scanned,
        json.dumps(counts),
        json.dumps(rejects),
        kept_redacted,
    )
    for b, quota in quotas.items():
        if counts[b] < quota:
            logger.warning(
                "[stream] bucket %s SHORT of quota: kept %d < %d (scan cap %d) — "
                "graceful-floor path (plan A9); shortfall recorded in stats",
                b,
                counts[b],
                quota,
                scan_cap,
            )

    if stats_out is not None:
        stats_out.update(
            {
                "scanned": scanned,
                "rejects": rejects,
                "kept_redacted": kept_redacted,
                "resumed_from_cache": False,
                "bucket_counts": counts,
                "field_digests": field_digests,
            }
        )
    return rows


# ── marked tier-2 strata (GSM8K + MBPP) ──────────────────────────────────────


def _build_marked_stratum(
    dataset: str,
    config: str,
    revision: str,
    field: str,
    cls: str,
    n_target: int,
    *,
    probe: bool = False,
) -> tuple[list[dict], dict]:
    """First-N deterministic draw from an established benchmark's train split."""
    from datasets import load_dataset  # lazy import

    ds = load_dataset(dataset, config, split="train", streaming=True, revision=revision)
    rows: list[dict] = []
    seen: set[str] = set()
    digest: dict = {}
    scanned = 0
    row: dict | None = None
    for row in ds:
        if len(rows) >= n_target:
            break
        if scanned == 0:
            digest = _structure_digest(row)
            if field not in row:
                raise RuntimeError(
                    f"{dataset} row is missing expected field {field!r}; "
                    f"observed structure: {json.dumps(digest)}"
                )
        scanned += 1
        query = (row.get(field) or "").strip()
        if not query:
            continue
        h = _sha16(query)
        if h in seen:
            continue
        n_chat, n_plain = _formatted_token_counts(query)
        if max(n_chat, n_plain) > C.MAX_FORMATTED_TOKENS:
            continue
        seen.add(h)
        rows.append(
            {
                "bucket": cls,
                "query": query,
                "n_tokens_chat": n_chat,
                "n_tokens_plain": n_plain,
                "source_index": scanned - 1,
            }
        )
    del ds, row
    gc.collect()
    if not rows:
        raise RuntimeError(f"{dataset} stratum kept 0 rows of {scanned} scanned — broken filter")
    logger.info("[stratum %s] kept %d of %d scanned", cls, len(rows), scanned)
    if probe:
        logger.info("[stratum %s] field-structure digest: %s", cls, json.dumps(digest))
    return rows, {"scanned": scanned, "kept": len(rows), "field_digest": digest}


# ── embeddings + clustering ──────────────────────────────────────────────────


def _embed_texts(texts: list[str], device: str, batch_size: int = C.EMBED_BATCH_SIZE):
    """Batched MiniLM embeddings (mean-pooled over the attention mask,
    L2-normalized — the standard sentence-transformers recipe for
    all-MiniLM-L6-v2, run through plain ``transformers``; the
    sentence-transformers package is not a project dependency)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.EMBED_MODEL_ID)
    model = AutoModel.from_pretrained(C.EMBED_MODEL_ID)
    model.to(device).eval()
    chunks = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            hidden = model(**enc).last_hidden_state  # (B, T, 384)
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            chunks.append(emb.cpu())
            if (i // batch_size) % 20 == 0:
                logger.info(
                    "[embed] %d/%d texts (%.0fs)",
                    min(i + batch_size, len(texts)),
                    len(texts),
                    time.time() - t0,
                )
    return torch.cat(chunks).numpy()


def _cluster_texts(texts: list[str], device: str) -> tuple[list[int], dict]:
    """MiniLM + k-means (k=40, seed 42). Returns (labels, clusters.json entry)."""
    from sklearn.cluster import KMeans

    if len(texts) < C.K_CLUSTERS:
        raise RuntimeError(
            f"cannot k-means {len(texts)} texts into k={C.K_CLUSTERS} clusters "
            "(production corpus too small — upstream filters are broken)"
        )
    embs = _embed_texts(texts, device=device)
    km = KMeans(n_clusters=C.K_CLUSTERS, random_state=C.KMEANS_SEED, n_init=10)
    labels = [int(x) for x in km.fit_predict(embs)]
    sizes: dict[str, int] = {}
    for lab in labels:
        sizes[str(lab)] = sizes.get(str(lab), 0) + 1
    entry = {
        "n_clustered": len(texts),
        "k": C.K_CLUSTERS,
        "seed": C.KMEANS_SEED,
        "inertia": float(km.inertia_),
        "sizes": dict(sorted(sizes.items(), key=lambda kv: int(kv[0]))),
        "centroids": [[float(v) for v in c] for c in km.cluster_centers_],
    }
    return labels, entry


# ── corpus assembly ──────────────────────────────────────────────────────────


def _assemble_single(
    lmsys_rows: list[dict],
    gsm8k_rows: list[dict],
    mbpp_rows: list[dict],
    clusters: list[int] | None,
) -> list[dict]:
    """Final single-turn corpus rows (schema: C.SINGLE_ROW_FIELDS). LMSYS rows
    carry k-means cluster groups; marked strata keep their own group labels
    and are excluded from generic clusters (cluster = UNCLUSTERED)."""
    out: list[dict] = []
    for i, r in enumerate(lmsys_rows):
        cluster = clusters[i] if clusters is not None else C.UNCLUSTERED
        group = f"cluster_{cluster}" if clusters is not None else "unclustered"
        cls = C.CLASS_MATHCODE if r["bucket"] == BUCKET_SINGLE_MATHCODE else C.CLASS_GENERIC
        out.append(
            {
                "id": "",  # assigned below
                "corpus": C.CORPUS_SINGLE,
                "class": cls,
                "group": group,
                "cluster": cluster,
                "query": r["query"],
                "n_tokens_chat": r["n_tokens_chat"],
                "n_tokens_plain": r["n_tokens_plain"],
                "source": "lmsys",
                "source_index": r["source_index"],
            }
        )
    for cls, source, rows in (
        (C.CLASS_GSM8K, "gsm8k", gsm8k_rows),
        (C.CLASS_MBPP, "mbpp", mbpp_rows),
    ):
        for r in rows:
            out.append(
                {
                    "id": "",
                    "corpus": C.CORPUS_SINGLE,
                    "class": cls,
                    "group": cls,
                    "cluster": C.UNCLUSTERED,
                    "query": r["query"],
                    "n_tokens_chat": r["n_tokens_chat"],
                    "n_tokens_plain": r["n_tokens_plain"],
                    "source": source,
                    "source_index": r["source_index"],
                }
            )
    for i, r in enumerate(out):
        r["id"] = f"single_{i:05d}"
        assert tuple(r.keys()) == C.SINGLE_ROW_FIELDS, f"schema drift: {tuple(r.keys())}"
    return out


def _assemble_multi(multi_rows: list[dict], clusters: list[int] | None) -> list[dict]:
    out: list[dict] = []
    for i, r in enumerate(multi_rows):
        cluster = clusters[i] if clusters is not None else C.UNCLUSTERED
        group = f"cluster_{cluster}" if clusters is not None else "unclustered"
        out.append(
            {
                "id": f"multi_{i:05d}",
                "corpus": C.CORPUS_MULTI,
                "class": C.CLASS_GENERIC,
                "group": group,
                "cluster": cluster,
                "prefix_turns": r["prefix_turns"],
                "query": r["query"],
                "n_prior_turns": r["n_prior_turns"],
                "n_tokens_chat": r["n_tokens_chat"],
                "n_tokens_plain": r["n_tokens_plain"],
                "source": "lmsys",
                "source_index": r["source_index"],
            }
        )
        assert tuple(out[-1].keys()) == C.MULTI_ROW_FIELDS, f"schema drift: {tuple(out[-1].keys())}"
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    os.replace(tmp, path)
    logger.info("[write] %d rows -> %s", len(rows), path)


# ── provenance / upload ──────────────────────────────────────────────────────


def _git_sha() -> str:
    """Repo sha for provenance; degrades on git-less lanes (fellows/SLURM rsync — #1902)."""
    env_sha = os.environ.get("EPS_GIT_SHA", "").strip()
    if env_sha:
        return env_sha
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    if out.returncode == 0:
        return out.stdout.strip()
    return "unavailable-no-git-checkout"


def _env_versions() -> dict[str, str]:
    import datasets
    import sklearn
    import transformers

    return {
        "datasets": datasets.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "python": sys.version.split()[0],
    }


def _upload_corpus(out_dir: Path, n_single: int, n_multi: int, git_sha: str) -> None:
    """Fail-loud bulk upload to the HF data repo (plan §4 P0). retry_transient
    bounds transient 5xx/429 retries and raises on exhaustion — a clean exit
    IS the upload contract."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    hub.assert_hub_dir_filecounts(out_dir, C.CORPUS_HF_PATH)
    api = HfApi()
    hub.retry_transient(
        lambda: api.upload_folder(
            folder_path=str(out_dir),
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=C.CORPUS_HF_PATH,
            commit_message=(
                f"issue1902 P0 corpus (single={n_single}, multi={n_multi}, git={git_sha[:8]})"
            ),
        ),
        what="upload_folder issue1902 corpus",
    )
    logger.info("[upload] corpus -> hf:%s/%s", C.HF_DATA_REPO, C.CORPUS_HF_PATH)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true", help="tiny-real streaming probe (bounded)")
    mode.add_argument("--full", action="store_true", help="production corpus build")
    ap.add_argument("--out-dir", type=Path, default=None, help="output dir (probe default: /tmp)")
    ap.add_argument("--cache-dir", type=Path, default=None, help="stream checkpoint dir")
    ap.add_argument("--scan-cap", type=int, default=None, help="override total streamed-row cap")
    ap.add_argument(
        "--kept-cap", type=int, default=PROBE_KEPT_CAP, help="probe per-corpus kept cap"
    )
    ap.add_argument("--embed-device", default="cpu", help="MiniLM device (cpu = deterministic)")
    ap.add_argument("--no-upload", action="store_true", help="skip the HF upload (full mode)")
    ap.add_argument("--no-resume-stream", action="store_true", help="ignore persisted stream pool")
    args = ap.parse_args()

    probe = bool(args.probe)
    t_start = time.time()

    if probe:
        out_dir = args.out_dir or Path("/tmp/issue1902_probe")
        scan_cap = args.scan_cap or PROBE_SCAN_CAP
        quotas = {
            BUCKET_SINGLE_GENERIC: args.kept_cap,
            BUCKET_SINGLE_MATHCODE: PROBE_MATHCODE_CAP,
            BUCKET_MULTI: args.kept_cap,
        }
        n_gsm8k, n_mbpp = PROBE_MARKED_CAP, PROBE_MARKED_CAP
    else:
        out_dir = args.out_dir or (PROJECT_ROOT / "data" / "issue_1902" / "corpus")
        scan_cap = args.scan_cap or C.SCAN_CAP
        quotas = {
            BUCKET_SINGLE_GENERIC: C.SINGLE_GENERIC_N,
            BUCKET_SINGLE_MATHCODE: C.SINGLE_MATHCODE_N,
            BUCKET_MULTI: C.MULTI_N,
        }
        n_gsm8k, n_mbpp = C.GSM8K_N, C.MBPP_N
    cache_dir = args.cache_dir or (
        out_dir / "stream_cache" if probe else PROJECT_ROOT / "data" / "issue_1902" / "stream_cache"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    git_sha = _git_sha()
    logger.info(
        "[P0] mode=%s out=%s scan_cap=%d quotas=%s git=%s",
        "probe" if probe else "full",
        out_dir,
        scan_cap,
        json.dumps(quotas),
        git_sha[:8],
    )

    # Resolve provenance shas early (loud failure before any long stream).
    from huggingface_hub import HfApi

    tokenizer_sha = str(HfApi().model_info(C.MODEL_IDS["R"]).sha)

    # ── marked tier-2 strata first (quick; surfaces field drift early — A16) ──
    gsm8k_rows, gsm8k_stats = _build_marked_stratum(
        C.GSM8K_DATASET,
        C.GSM8K_CONFIG,
        C.GSM8K_REVISION,
        "question",
        C.CLASS_GSM8K,
        n_gsm8k,
        probe=probe,
    )
    mbpp_rows, mbpp_stats = _build_marked_stratum(
        C.MBPP_DATASET,
        C.MBPP_CONFIG,
        C.MBPP_REVISION,
        "text",
        C.CLASS_MBPP,
        n_mbpp,
        probe=probe,
    )

    # ── LMSYS stream (both corpora, one pass) ────────────────────────────────
    stream_stats: dict[str, Any] = {}
    kept = _stream_lmsys(
        quotas,
        scan_cap,
        cache_dir,
        resume=not args.no_resume_stream and not probe,
        probe=probe,
        stats_out=stream_stats,
    )
    lmsys_single = [r for r in kept if r["bucket"] != BUCKET_MULTI]
    lmsys_multi = [r for r in kept if r["bucket"] == BUCKET_MULTI]

    if probe:
        for i, digest in enumerate(stream_stats.get("field_digests", [])):
            logger.info("[probe] LMSYS field-structure digest %d: %s", i + 1, json.dumps(digest))
        if not lmsys_single:
            raise RuntimeError(
                f"probe kept 0 single-turn rows — rejecting filters: "
                f"{json.dumps(stream_stats['rejects'])}"
            )
        if not lmsys_multi:
            raise RuntimeError(
                f"probe kept 0 multi-turn rows — rejecting filters: "
                f"{json.dumps(stream_stats['rejects'])}"
            )
        logger.info(
            "[probe] PASS: kept single=%d multi=%d of %d scanned",
            len(lmsys_single),
            len(lmsys_multi),
            stream_stats["scanned"],
        )

    # ── clustering (production only; probe rows stay unclustered) ────────────
    clusters_meta: dict[str, Any] = {
        "embedding_model": C.EMBED_MODEL_ID,
        "embed_device": args.embed_device,
        "embed_text": "plain_prompt_render_truncated_at_256_minilm_tokens",
        "k": C.K_CLUSTERS,
        "seed": C.KMEANS_SEED,
    }
    single_labels = multi_labels = None
    if not probe:
        single_labels, clusters_meta["single"] = _cluster_texts(
            [r["query"] for r in lmsys_single], device=args.embed_device
        )
        multi_labels, clusters_meta["multi"] = _cluster_texts(
            [C.render_plain_prompt(r["query"], r["prefix_turns"]) for r in lmsys_multi],
            device=args.embed_device,
        )

    # ── assemble + write ─────────────────────────────────────────────────────
    single_rows = _assemble_single(lmsys_single, gsm8k_rows, mbpp_rows, single_labels)
    multi_rows = _assemble_multi(lmsys_multi, multi_labels)
    _write_jsonl(out_dir / C.CORPUS_SINGLE_FILENAME, single_rows)
    _write_jsonl(out_dir / C.CORPUS_MULTI_FILENAME, multi_rows)
    with open(out_dir / C.CLUSTERS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(clusters_meta, f, indent=2)

    class_counts: dict[str, int] = {}
    for r in single_rows:
        class_counts[r["class"]] = class_counts.get(r["class"], 0) + 1
    stats = {
        "probe": probe,
        "git_sha": git_sha,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - t_start, 1),
        "env_versions": _env_versions(),
        "tokenizer_model": C.MODEL_IDS["R"],
        "tokenizer_sha_resolved": tokenizer_sha,
        "lmsys_dataset": C.LMSYS_DATASET,
        "lmsys_revision": C.LMSYS_REVISION,
        "gsm8k": {"revision": C.GSM8K_REVISION, **gsm8k_stats},
        "mbpp": {"revision": C.MBPP_REVISION, **mbpp_stats},
        "quotas": quotas,
        "scan_cap": scan_cap,
        "stream": {k: v for k, v in stream_stats.items() if k != "field_digests"},
        "n_single": len(single_rows),
        "n_multi": len(multi_rows),
        "single_class_counts": class_counts,
    }
    # gsm8k/mbpp field digests are structure-only (key names + types) — safe.
    with open(out_dir / "manifest_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if not probe and not args.no_upload:
        _upload_corpus(out_dir, len(single_rows), len(multi_rows), git_sha)
    elif args.no_upload:
        logger.info("[P0] --no-upload: skipping HF upload")

    logger.info(
        "[P0] done: single=%d (%s) multi=%d scanned=%d rejects=%s wall=%.0fs",
        len(single_rows),
        json.dumps(class_counts),
        len(multi_rows),
        stream_stats["scanned"],
        json.dumps(stream_stats["rejects"]),
        time.time() - t_start,
    )
    # Success path: every output is written (+ uploaded in --full) above, and
    # a bare sys.exit(0) still SIGABRTed rc=134 in interpreter finalization
    # (PyGILState_Release thread-state race across the combined tokenizers /
    # datasets / pyarrow teardown — the #654/#952/#1689 class; the #952
    # `del ds; gc.collect()` guard is in place at both stream sites and the
    # race persisted, measured 2026-07-31 on this script's own probe run).
    # Flush explicitly, then skip the poisoned finalize path entirely (the
    # #1689-sanctioned stronger alternative — safe here: no atexit-dependent
    # work). Any EXCEPTION before this line still propagates + exits non-zero.
    sys.stdout.flush()
    sys.stderr.flush()
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
