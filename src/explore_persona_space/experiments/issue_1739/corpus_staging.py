"""Corpus staging for issue #1739 (round B) — implements the registry stubs.

Staging contract (plan v3 Step 2a-bis + the round-B brief):

- STREAMING HF loads with per-chunk checkpointing + fingerprint-gated resume
  (`_stream_stage`, mirroring #1092's ``_stream_with_cache`` shape: atomic
  partial flushes, meta sidecar last, resume keyed on dataset revision-free
  fingerprint of every filter/recipe constant).
- Usable-text filters (removed/deleted/redacted/short) with PER-FILTER
  rejection counters surfaced in the ``done:`` log line + manifest.
- Deterministic seeded subsampling to the registry caps (over-collect a
  bounded usable pool, then one seeded ``rng.choice`` down to the cap).
- MinHash/LSH near-dup filter between train and EVERY eval rung
  (disjointness contract; eval rows near-dup'ing any train row are dropped
  and counted).

CONTENT HYGIENE (binding): several corpora are harmful-content / real-user
banks. This module NEVER logs, prints, or returns item text through any
logging channel — staged JSONL files on disk carry the text; logs and
manifests carry counts, ids, and field names only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import zlib
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# --- staging pins (plan v3 Step 2a-bis) -------------------------------------
DEFAULT_STAGED_ROOT = Path("data/issue_1739/staged")
OVERSAMPLE_FACTOR = 2  # collect <= factor*cap usable rows, then seeded subsample
FLUSH_EVERY_SCANNED = 2000
MIN_TEXT_CHARS = 16
MAX_TEXT_CHARS = 20_000  # crude pre-tokenizer bound; token budget enforced at generation
# MinHash near-dup parameters: 16 bands x 4 rows over 64 perms ~= Jaccard>=0.5 flagged.
MINHASH_N_PERM = 64
MINHASH_SHINGLE = 5
MINHASH_BANDS = 16
# Sycophancy REDDIT train/eval partition (round C2 fix): sha1(post_id) mod 10.
# Bucket SYC_EVAL_BUCKET -> eval-eligible; the other buckets -> train-eligible.
# BOTH sides apply the partition at stream time, so the train and eval slices
# are disjoint by post id BY CONSTRUCTION regardless of staging order; the
# MinHash near-dup filter stays as the safety net.
SYC_PARTITION_MOD = 10
SYC_EVAL_BUCKET = 9

_REMOVED_MARKERS = ("[removed]", "[deleted]", "[redacted]", "redacted")
_MERSENNE_P = np.uint64((1 << 61) - 1)


# ---------------------------------------------------------------------------
# text utils
# ---------------------------------------------------------------------------


def norm_text(text: str) -> str:
    """Lowercase + whitespace-collapsed normalization (dedup key space)."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


_FALSY_STRINGS = frozenset({"", "false", "f", "0", "no", "n", "none", "null"})
_TRUTHY_STRINGS = frozenset({"true", "t", "1", "yes", "y"})


def parse_bool_field(value: object) -> bool:
    """Schema-defensive bool parse for HF fields that ship bool OR string.

    HuggingFaceGECLM/REDDIT_submissions stores ``over_18`` as the STRING
    "False"/"True" (round-C1 bug: a bare truthiness check reads "False" as
    True and rejects every SFW row). Accepts bool / numeric / case-insensitive
    string; an unrecognized non-empty token parses True (conservative for an
    NSFW filter: reject, never keep). Never blind-truthy on unknown types.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value).strip().lower()
    if token in _FALSY_STRINGS:
        return False
    if token in _TRUTHY_STRINGS:
        return True
    return True


def usable_text(text: object, *, min_chars: int = MIN_TEXT_CHARS) -> str | None:
    """Return the reject reason for an unusable text, else None (usable).

    Rejects: non-string/empty, removed/deleted/redacted markers, too-short,
    absurdly long (pre-tokenizer bound; the generation-side token budget is
    the binding filter).
    """
    if not isinstance(text, str) or not text.strip():
        return "empty"
    t = text.strip()
    if norm_text(t) in _REMOVED_MARKERS or any(m == norm_text(t)[:10] for m in ()):
        return "removed_deleted"
    low = t.lower()
    if low.startswith(_REMOVED_MARKERS):
        return "removed_deleted"
    if len(t) < min_chars:
        return "too_short"
    if len(t) > MAX_TEXT_CHARS:
        return "too_long"
    return None


# ---------------------------------------------------------------------------
# MinHash near-dup filter (train vs eval disjointness contract)
# ---------------------------------------------------------------------------


def _shingle_hashes(text: str, shingle: int = MINHASH_SHINGLE) -> np.ndarray:
    """Deterministic char-shingle hash set (crc32 — never salted builtin hash)."""
    t = norm_text(text)
    if len(t) < shingle:
        t = t.ljust(shingle, "_")
    raw = {zlib.crc32(t[i : i + shingle].encode("utf-8")) for i in range(len(t) - shingle + 1)}
    return np.fromiter(raw, dtype=np.uint64, count=len(raw))


def minhash_signatures(
    texts: list[str], *, n_perm: int = MINHASH_N_PERM, seed: int = 0
) -> np.ndarray:
    """(n_texts, n_perm) uint64 MinHash signatures over char-5 shingles."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, int(_MERSENNE_P), size=n_perm, dtype=np.uint64)
    b = rng.integers(0, int(_MERSENNE_P), size=n_perm, dtype=np.uint64)
    sigs = np.empty((len(texts), n_perm), dtype=np.uint64)
    for i, text in enumerate(texts):
        h = _shingle_hashes(text)
        # (n_perm, n_shingles) universal hash, min over shingles.
        vals = (a[:, None] * h[None, :] + b[:, None]) % _MERSENNE_P
        sigs[i] = vals.min(axis=1)
    return sigs


def near_dup_mask(
    train_sigs: np.ndarray, eval_sigs: np.ndarray, *, bands: int = MINHASH_BANDS
) -> np.ndarray:
    """Boolean mask over eval rows: True = near-dup of SOME train row.

    LSH banding: any identical (band, band-signature) tuple between an eval
    row and any train row flags the eval row. 16 bands x 4 rows over 64
    perms flags Jaccard ~>= 0.5 with high probability.
    """
    assert train_sigs.shape[1] == eval_sigs.shape[1], (train_sigs.shape, eval_sigs.shape)
    n_perm = train_sigs.shape[1]
    rows_per_band = n_perm // bands
    train_bands: set[tuple] = set()
    for bi in range(bands):
        sl = slice(bi * rows_per_band, (bi + 1) * rows_per_band)
        for row in train_sigs[:, sl]:
            train_bands.add((bi, row.tobytes()))
    mask = np.zeros(eval_sigs.shape[0], dtype=bool)
    for ei in range(eval_sigs.shape[0]):
        for bi in range(bands):
            sl = slice(bi * rows_per_band, (bi + 1) * rows_per_band)
            if (bi, eval_sigs[ei, sl].tobytes()) in train_bands:
                mask[ei] = True
                break
    return mask


# ---------------------------------------------------------------------------
# streaming with per-chunk checkpoint + fingerprint-gated resume
# ---------------------------------------------------------------------------


def _fingerprint(**kwargs: object) -> str:
    """Stable fingerprint over every output-affecting staging constant."""
    return hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    """Text-mode line iteration (NEVER .splitlines() — U+2028/NEL, gotchas #825)."""
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _stream_stage(
    *,
    out_path: Path,
    fingerprint: str,
    row_iter_factory: Callable[[], Iterable[dict]],
    keep_fn: Callable[[dict], tuple[dict | None, str | None]],
    keep_cap: int | None,
    stream_cap: int | None,
    log_label: str,
) -> tuple[list[dict], dict]:
    """Stream-filter rows with chunked checkpointing + fingerprint resume.

    Returns (kept_rows, counters). Counters include ``scanned`` and per-filter
    reject counts. Asserts kept > 0 (data-ingestion probe contract).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_name(out_path.name + ".meta.json")
    partial_path = out_path.with_name(out_path.name + ".partial.jsonl")
    partial_meta = out_path.with_name(out_path.name + ".partial.meta.json")

    # Complete-pool resume (exact fingerprint match).
    if out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fingerprint:
            rows = read_jsonl(out_path)
            logger.info("[stage] %s: resume complete pool (%d rows)", log_label, len(rows))
            return rows, meta.get("counters", {})
        logger.info("[stage] %s: fingerprint changed; restaging", log_label)

    kept: list[dict] = []
    counters: dict[str, int] = {"scanned": 0}
    skip_scanned = 0
    if partial_path.exists() and partial_meta.exists():
        pmeta = json.loads(partial_meta.read_text())
        if pmeta.get("fingerprint") == fingerprint:
            kept = read_jsonl(partial_path)
            counters = dict(pmeta.get("counters", {"scanned": 0}))
            skip_scanned = int(pmeta.get("counters", {}).get("scanned", 0))
            logger.info(
                "[stage] %s: partial resume kept=%d scanned=%d", log_label, len(kept), skip_scanned
            )

    def _flush_partial() -> None:
        _write_jsonl_atomic(partial_path, kept)
        tmp = partial_meta.with_name(partial_meta.name + ".tmp")
        tmp.write_text(json.dumps({"fingerprint": fingerprint, "counters": counters}))
        os.replace(tmp, partial_meta)

    t0 = time.time()
    it: Iterator[dict] = iter(row_iter_factory())
    idx = 0
    for raw in it:
        idx += 1
        if idx <= skip_scanned:
            continue  # deterministic stream order: fast-forward past resumed rows
        counters["scanned"] += 1
        row, reject = keep_fn(raw)
        if reject is not None:
            counters[reject] = counters.get(reject, 0) + 1
        elif row is not None:
            kept.append(row)
        if counters["scanned"] % FLUSH_EVERY_SCANNED == 0:
            _flush_partial()
            logger.info(
                "[stage] %s: scanned=%d kept=%d elapsed=%.0fs",
                log_label,
                counters["scanned"],
                len(kept),
                time.time() - t0,
            )
        if keep_cap is not None and len(kept) >= keep_cap:
            break
        if stream_cap is not None and counters["scanned"] >= stream_cap:
            break

    if not kept:
        raise RuntimeError(
            f"[stage] {log_label}: kept 0 rows after scanning {counters['scanned']} "
            f"(rejects={ {k: v for k, v in counters.items() if k != 'scanned'} }); "
            "field-semantics or filter bug — fail loud (gotchas: real-corpus streaming filters)"
        )
    _write_jsonl_atomic(out_path, kept)
    meta = {
        "fingerprint": fingerprint,
        "counters": counters,
        "n_kept": len(kept),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, meta_path)
    partial_path.unlink(missing_ok=True)
    partial_meta.unlink(missing_ok=True)
    logger.info(
        "[stage] done: %s scanned=%d kept=%d rejects=%s",
        log_label,
        counters["scanned"],
        len(kept),
        {k: v for k, v in counters.items() if k != "scanned"},
    )
    return kept, counters


def _hf_stream(dataset_id: str, config: str | None, split: str, **kwargs) -> Iterable[dict]:
    from datasets import load_dataset

    if config is not None:
        return load_dataset(dataset_id, config, split=split, streaming=True, **kwargs)
    return load_dataset(dataset_id, split=split, streaming=True, **kwargs)


def _subsample(rows: list[dict], cap: int | None, seed: int) -> list[dict]:
    """Seeded subsample down to cap (order-preserving), identity when under cap."""
    if cap is None or len(rows) <= cap:
        return rows
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(rows), size=cap, replace=False))
    return [rows[i] for i in idx]


# ---------------------------------------------------------------------------
# component stagers (schemas verified against datasets-server /info, 2026-07-28)
# ---------------------------------------------------------------------------


def _stage_evil_prefixes(out_dir: Path, stream_cap: int | None) -> list[dict]:
    """in-the-wild jailbreak prompt corpus: config jailbreak_2023_12_25 (1,405 rows),
    field ``prompt``; dedupe on normalized text."""
    seen: set[str] = set()

    def keep(raw: dict) -> tuple[dict | None, str | None]:
        text = raw.get("prompt")
        reject = usable_text(text)
        if reject:
            return None, reject
        key = norm_text(text)
        if key in seen:
            return None, "dup_text"
        seen.add(key)
        return {"text": text, "source_id": str(raw.get("community_id") or len(seen))}, None

    rows, _ = _stream_stage(
        out_path=out_dir / "evil_prefix_pool.jsonl",
        fingerprint=_fingerprint(
            ds="TrustAIRLab/in-the-wild-jailbreak-prompts",
            config="jailbreak_2023_12_25",
            split="train",
            filters="usable+dedupe.v1",
            stream_cap=stream_cap,
        ),
        row_iter_factory=lambda: _hf_stream(
            "TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", "train"
        ),
        keep_fn=keep,
        keep_cap=None,
        stream_cap=stream_cap,
        log_label="evil-prefixes",
    )
    return rows


def _stage_evil_questions(out_dir: Path, stream_cap: int | None) -> list[dict]:
    """forbidden question set: 390 questions, fields question/content_policy_name."""
    seen: set[str] = set()

    def keep(raw: dict) -> tuple[dict | None, str | None]:
        text = raw.get("question")
        reject = usable_text(text, min_chars=8)
        if reject:
            return None, reject
        key = norm_text(text)
        if key in seen:
            return None, "dup_text"
        seen.add(key)
        return {
            "text": text,
            "source_id": str(raw.get("q_id")),
            "policy": str(raw.get("content_policy_name")),
        }, None

    rows, _ = _stream_stage(
        out_path=out_dir / "evil_question_pool.jsonl",
        fingerprint=_fingerprint(
            ds="TrustAIRLab/forbidden_question_set",
            split="train",
            filters="usable+dedupe.v1",
            stream_cap=stream_cap,
        ),
        row_iter_factory=lambda: _hf_stream("TrustAIRLab/forbidden_question_set", None, "train"),
        keep_fn=keep,
        keep_cap=None,
        stream_cap=stream_cap,
        log_label="evil-questions",
    )
    return rows


def build_evil_cross(
    prefixes: list[str], questions: list[str], cap: int | None, seed: int
) -> list[dict]:
    """Seeded (prefix x question) cross subsampled to cap; group_key = prefix.

    Pure function (unit-tested on synthetic fixtures). Group key is the
    PREFIX index — the plan's ~1,405 independent prefix groups.
    """
    n_p, n_q = len(prefixes), len(questions)
    assert n_p > 0 and n_q > 0, (n_p, n_q)
    n_pairs = n_p * n_q
    take = n_pairs if cap is None else min(cap, n_pairs)
    rng = np.random.default_rng(seed)
    flat = np.sort(rng.choice(n_pairs, size=take, replace=False))
    rows = []
    for i, f in enumerate(flat):
        pi, qi = int(f // n_q), int(f % n_q)
        rows.append(
            {
                "context_id": f"evil-train-cross-{i:06d}",
                "behavior": "evil",
                "split": "train",
                "rung": "train",
                "group_key": f"prefix{pi:04d}",
                "prefix_text": prefixes[pi],
                "query": questions[qi],
                "source_dataset": "in-the-wild-jailbreak-prompts x forbidden_question_set",
                "source_id": f"p{pi:04d}-q{qi:03d}",
            }
        )
    return rows


_HUMAN_TURN_RE = re.compile(r"\n\nHuman:\s*(.*?)\n\nAssistant:", re.DOTALL)


def first_human_turn(transcript: str) -> str | None:
    """First human turn of an hh-rlhf transcript (None when unparseable)."""
    if not isinstance(transcript, str):
        return None
    m = _HUMAN_TURN_RE.search("\n\n" + transcript.lstrip())
    return m.group(1).strip() if m else None


def _stage_hh_redteam(
    out_dir: Path, keep_cap: int, stream_cap: int | None, seed: int
) -> list[dict]:
    def keep(raw: dict) -> tuple[dict | None, str | None]:
        turn = first_human_turn(raw.get("transcript"))
        if turn is None:
            return None, "unparseable_transcript"
        reject = usable_text(turn)
        if reject:
            return None, reject
        return {"text": turn}, None

    rows, _ = _stream_stage(
        out_path=out_dir / "evil_eval_hhrt_pool.jsonl",
        fingerprint=_fingerprint(
            ds="Anthropic/hh-rlhf",
            data_dir="red-team-attempts",
            split="train",
            filters="first_human_turn+usable.v1",
            oversample=OVERSAMPLE_FACTOR * keep_cap,
            stream_cap=stream_cap,
        ),
        row_iter_factory=lambda: _hf_stream(
            "Anthropic/hh-rlhf", None, "train", data_dir="red-team-attempts"
        ),
        keep_fn=keep,
        keep_cap=OVERSAMPLE_FACTOR * keep_cap,
        stream_cap=stream_cap,
        log_label="evil-eval-hhrt",
    )
    return _subsample(rows, keep_cap, seed)


def _stage_toxicchat(out_dir: Path, keep_cap: int | None, stream_cap: int | None) -> list[dict]:
    """lmsys/toxic-chat toxicchat0124, train+test, toxicity==1 rows only."""

    def rows_iter() -> Iterator[dict]:
        yield from _hf_stream("lmsys/toxic-chat", "toxicchat0124", "train")
        yield from _hf_stream("lmsys/toxic-chat", "toxicchat0124", "test")

    def keep(raw: dict) -> tuple[dict | None, str | None]:
        if int(raw.get("toxicity") or 0) != 1:
            return None, "not_flagged"
        text = raw.get("user_input")
        reject = usable_text(text)
        if reject:
            return None, reject
        return {"text": text, "source_id": str(raw.get("conv_id", ""))[:24]}, None

    rows, _ = _stream_stage(
        out_path=out_dir / "evil_eval_toxicchat_pool.jsonl",
        fingerprint=_fingerprint(
            ds="lmsys/toxic-chat",
            config="toxicchat0124",
            splits="train+test",
            filters="toxicity1+usable.v1",
            stream_cap=stream_cap,
        ),
        row_iter_factory=rows_iter,
        keep_fn=keep,
        keep_cap=keep_cap,
        stream_cap=stream_cap,
        log_label="evil-eval-toxicchat",
    )
    return rows


def _clean_reddit_field(value: object) -> str:
    """One REDDIT text field, schema-defensively cleaned.

    Non-string -> "". A field that IS a removed/deleted/redacted sentinel
    (or starts with one — reddit ships literal "[removed]"/"[deleted]"
    selftext bodies) -> "" so the sentinel never pollutes the joined text.
    """
    if not isinstance(value, str):
        return ""
    t = value.strip()
    if not t or t.lower().startswith(_REMOVED_MARKERS):
        return ""
    return t


def reddit_text(raw: dict) -> str | None:
    """REDDIT_submissions row text: title + selftext (the live schema; the
    ``content`` column belongs to the DIFFERENT Value-Trade-off corpus and is
    kept only as a fallback), removed/deleted sentinels stripped per field
    (round-C1 bug fix: the old content-first read + unstripped "[removed]"
    bodies produced short/degenerate texts)."""
    title = _clean_reddit_field(raw.get("title"))
    body = _clean_reddit_field(raw.get("selftext"))
    text = (title + "\n\n" + body).strip()
    if text:
        return text
    return _clean_reddit_field(raw.get("content")) or None


def syc_post_bucket(post_id: str) -> int:
    """Deterministic sha1 partition bucket (0..SYC_PARTITION_MOD-1) of a post id."""
    return int(hashlib.sha1(str(post_id).encode("utf-8")).hexdigest(), 16) % SYC_PARTITION_MOD


def _stage_reddit(
    out_dir: Path,
    subreddit_split: str,
    keep_cap: int,
    stream_cap: int | None,
    seed: int,
    *,
    exclude_ids: set[str] | None = None,
    tag: str,
    partition: str | None = None,
) -> list[dict]:
    exclude_ids = exclude_ids or set()
    assert partition in (None, "train", "eval"), partition

    def keep(raw: dict) -> tuple[dict | None, str | None]:
        rid = str(raw.get("id"))
        if partition is not None:
            in_eval_bucket = syc_post_bucket(rid) == SYC_EVAL_BUCKET
            if (partition == "eval") != in_eval_bucket:
                return None, "hash_partition"
        if rid in exclude_ids:
            return None, "excluded_id"
        text = reddit_text(raw)
        reject = usable_text(text, min_chars=64)
        if reject:
            return None, reject
        if parse_bool_field(raw.get("over_18")):
            return None, "over_18"
        return {"text": text, "source_id": rid}, None

    rows, _ = _stream_stage(
        out_path=out_dir / f"syc_{tag}_{subreddit_split}_pool.jsonl",
        fingerprint=_fingerprint(
            ds="HuggingFaceGECLM/REDDIT_submissions",
            split=subreddit_split,
            # v3: sha1(post_id) mod-10 train/eval hash partition (C2 fix);
            # v2: string-typed over_18 + selftext-first text (C1 fix).
            filters="usable64+sfw.v3",
            partition=partition,
            partition_mod=SYC_PARTITION_MOD,
            eval_bucket=SYC_EVAL_BUCKET,
            oversample=OVERSAMPLE_FACTOR * keep_cap,
            n_excluded=len(exclude_ids),
            stream_cap=stream_cap,
        ),
        row_iter_factory=lambda: _hf_stream(
            "HuggingFaceGECLM/REDDIT_submissions", None, subreddit_split
        ),
        keep_fn=keep,
        keep_cap=OVERSAMPLE_FACTOR * keep_cap,
        stream_cap=stream_cap,
        log_label=f"syc-{tag}-{subreddit_split}",
    )
    return _subsample(rows, keep_cap, seed)


def _norm_answer_key(value: object) -> str:
    return norm_text(str(value))[:64] or "unknown"


def _stage_trivia(out_dir: Path, keep_cap: int, stream_cap: int | None, seed: int) -> list[dict]:
    def keep(raw: dict) -> tuple[dict | None, str | None]:
        q = raw.get("question")
        reject = usable_text(q, min_chars=8)
        if reject:
            return None, reject
        ans = raw.get("answer") or {}
        aliases = list(
            dict.fromkeys(
                [str(ans.get("value") or "")]
                + [str(a) for a in (ans.get("aliases") or [])]
                + [str(a) for a in (ans.get("normalized_aliases") or [])]
            )
        )
        aliases = [a for a in aliases if a.strip()]
        if not aliases:
            return None, "no_answer_aliases"
        return {
            "text": q,
            "source_id": str(raw.get("question_id")),
            "answer_aliases": aliases,
            "answer_key": _norm_answer_key(aliases[0]),
        }, None

    rows, _ = _stream_stage(
        out_path=out_dir / "halluc_train_trivia_pool.jsonl",
        fingerprint=_fingerprint(
            ds="mandarjoshi/trivia_qa",
            config="rc.nocontext",
            split="train",
            filters="usable+aliases.v1",
            oversample=OVERSAMPLE_FACTOR * keep_cap,
            stream_cap=stream_cap,
        ),
        row_iter_factory=lambda: _hf_stream("mandarjoshi/trivia_qa", "rc.nocontext", "train"),
        keep_fn=keep,
        keep_cap=OVERSAMPLE_FACTOR * keep_cap,
        stream_cap=stream_cap,
        log_label="halluc-train-trivia",
    )
    return _subsample(rows, keep_cap, seed)


def _stage_nq_open(out_dir: Path, keep_cap: int | None, stream_cap: int | None) -> list[dict]:
    def keep(raw: dict) -> tuple[dict | None, str | None]:
        q = raw.get("question")
        reject = usable_text(q, min_chars=8)
        if reject:
            return None, reject
        aliases = [str(a) for a in (raw.get("answer") or []) if str(a).strip()]
        if not aliases:
            return None, "no_answer_aliases"
        return {
            "text": q,
            "answer_aliases": aliases,
            "answer_key": _norm_answer_key(aliases[0]),
        }, None

    rows, _ = _stream_stage(
        out_path=out_dir / "halluc_eval_nqopen_pool.jsonl",
        fingerprint=_fingerprint(
            ds="google-research-datasets/nq_open",
            split="validation",
            filters="usable+aliases.v1",
            stream_cap=stream_cap,
        ),
        row_iter_factory=lambda: _hf_stream("google-research-datasets/nq_open", None, "validation"),
        keep_fn=keep,
        keep_cap=keep_cap,
        stream_cap=stream_cap,
        log_label="halluc-eval-nqopen",
    )
    return rows


def _stage_simpleqa(out_dir: Path, keep_cap: int | None, stream_cap: int | None) -> list[dict]:
    def keep(raw: dict) -> tuple[dict | None, str | None]:
        q = raw.get("problem")
        reject = usable_text(q, min_chars=8)
        if reject:
            return None, reject
        ans = str(raw.get("answer") or "")
        if not ans.strip():
            return None, "no_answer_aliases"
        return {"text": q, "answer_aliases": [ans], "answer_key": _norm_answer_key(ans)}, None

    rows, _ = _stream_stage(
        out_path=out_dir / "halluc_eval_simpleqa_pool.jsonl",
        fingerprint=_fingerprint(
            ds="basicv8vc/SimpleQA",
            split="test",
            filters="usable+answer.v1",
            stream_cap=stream_cap,
        ),
        row_iter_factory=lambda: _hf_stream("basicv8vc/SimpleQA", None, "test"),
        keep_fn=keep,
        keep_cap=keep_cap,
        stream_cap=stream_cap,
        log_label="halluc-eval-simpleqa",
    )
    return rows


def _resolve_elephant() -> tuple[str, str | None, str, str] | None:
    """Resolve the ELEPHANT AITA-YTA dataset (env override only — no known HF id).

    Returns (dataset_id, config, split, text_field) or None (unresolved).
    Concern `elephant-aita-unresolved` on #1739 records the 2026-07-28 probe:
    no resolvable HF id; the plan-registered fallback (assumption 8) is a
    held-out REDDIT socialskills slice disjoint from train by post id.
    """
    ds = os.environ.get("EPM_I1739_ELEPHANT_DATASET")
    if not ds:
        return None
    from huggingface_hub import HfApi

    if not HfApi().repo_exists(ds, repo_type="dataset"):
        raise RuntimeError(f"EPM_I1739_ELEPHANT_DATASET={ds!r} does not resolve on HF")
    return (
        ds,
        os.environ.get("EPM_I1739_ELEPHANT_CONFIG") or None,
        os.environ.get("EPM_I1739_ELEPHANT_SPLIT", "train"),
        os.environ.get("EPM_I1739_ELEPHANT_FIELD", "text"),
    )


def _stage_elephant_or_fallback(
    out_dir: Path, keep_cap: int, stream_cap: int | None, seed: int
) -> tuple[list[dict], dict]:
    resolved = _resolve_elephant()
    if resolved is not None:
        ds, config, split, field = resolved

        def keep(raw: dict) -> tuple[dict | None, str | None]:
            text = raw.get(field)
            reject = usable_text(text, min_chars=64)
            if reject:
                return None, reject
            return {"text": text}, None

        rows, _ = _stream_stage(
            out_path=out_dir / "syc_eval_elephant_pool.jsonl",
            fingerprint=_fingerprint(
                ds=ds,
                config=config,
                split=split,
                field=field,
                filters="usable64.v1",
                oversample=OVERSAMPLE_FACTOR * keep_cap,
                stream_cap=stream_cap,
            ),
            row_iter_factory=lambda: _hf_stream(ds, config, split),
            keep_fn=keep,
            keep_cap=OVERSAMPLE_FACTOR * keep_cap,
            stream_cap=stream_cap,
            log_label="syc-eval-elephant",
        )
        return _subsample(rows, keep_cap, seed), {
            "fallback_elephant_unresolved": False,
            "source": ds,
        }

    # Plan-registered fallback (assumption 8): held-out socialskills slice,
    # disjoint from train by a DETERMINISTIC HASH PARTITION of post ids applied
    # at stream time on BOTH sides (sha1 mod SYC_PARTITION_MOD; bucket
    # SYC_EVAL_BUCKET -> eval, others -> train) — order-independent, unlike the
    # old read-the-train-pool exclusion (round C2 fix). The MinHash near-dup
    # filter in stage_corpus stays as the safety net.
    rows = _stage_reddit(
        out_dir,
        "socialskills",
        keep_cap,
        stream_cap,
        seed + 1,
        tag="eval",
        partition="eval",
    )
    return rows, {"fallback_elephant_unresolved": True, "source": "REDDIT socialskills held-out"}


# ---------------------------------------------------------------------------
# top-level staging
# ---------------------------------------------------------------------------


def _to_contexts(
    rows: list[dict], *, behavior: str, split: str, rung: str, group_prefix: str | None = None
) -> list[dict]:
    """Wrap pool rows into the staged-context schema (bare-context corpora)."""
    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                "context_id": f"{behavior}-{split}-{rung}-{i:06d}",
                "behavior": behavior,
                "split": split,
                "rung": rung,
                "group_key": r.get("answer_key")
                or (
                    f"{group_prefix}-{r['source_id']}"
                    if group_prefix and r.get("source_id")
                    else f"{rung}-{i:06d}"
                ),
                "prefix_text": "",
                "query": r["text"],
                "source_dataset": rung,
                "source_id": str(r.get("source_id", i)),
                **({"answer_aliases": r["answer_aliases"]} if r.get("answer_aliases") else {}),
            }
        )
    return out


def _context_dedup_text(row: dict) -> str:
    prefix = row.get("prefix_text") or ""
    return (prefix + "\n\n" + row["query"]).strip() if prefix else row["query"]


def enforce_disjointness(train_rows: list[dict], eval_rows: list[dict]) -> tuple[list[dict], int]:
    """Drop eval rows near-dup'ing any train row (MinHash/LSH). Returns
    (kept_eval_rows, n_dropped)."""
    if not train_rows or not eval_rows:
        return eval_rows, 0
    train_sigs = minhash_signatures([_context_dedup_text(r) for r in train_rows])
    eval_sigs = minhash_signatures([_context_dedup_text(r) for r in eval_rows])
    mask = near_dup_mask(train_sigs, eval_sigs)
    kept = [r for r, dup in zip(eval_rows, mask, strict=True) if not dup]
    return kept, int(mask.sum())


def staged_context_path(out_dir: Path, behavior: str, split: str, rung: str) -> Path:
    return Path(out_dir) / f"{behavior}_{split}_{rung}.contexts.jsonl"


def stage_corpus(
    behavior: str,
    split: str,
    cap: int | None,
    seed: int,
    *,
    out_dir: Path | str | None = None,
    stream_cap: int | None = None,
) -> dict:
    """Stage one (behavior, split) corpus per the registry; returns the manifest.

    Writes per-rung context JSONLs (`{behavior}_{split}_{rung}.contexts.jsonl`)
    + a staging manifest. Eval staging ENSURES the train split is staged first
    (fingerprint-cached, idempotent) and applies the near-dup disjointness
    filter against it. ``stream_cap`` bounds the total scanned rows (probe
    mode); production leaves it None.
    """
    from explore_persona_space.experiments.issue_1739.corpus_registry import get_spec

    spec = get_spec(behavior, split)  # validates keys
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_STAGED_ROOT / behavior
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cap if cap is not None else spec.cap
    rungs: dict[str, list[dict]] = {}
    extra_manifest: dict = {}

    if behavior == "evil" and split == "train":
        prefixes = [r["text"] for r in _stage_evil_prefixes(out_dir, stream_cap)]
        questions = [r["text"] for r in _stage_evil_questions(out_dir, stream_cap)]
        rungs["train"] = build_evil_cross(prefixes, questions, cap, seed)
        extra_manifest["n_prefix_groups"] = len(prefixes)
        extra_manifest["n_questions"] = len(questions)
    elif behavior == "evil" and split == "eval":
        hh_cap = min(2_000, cap) if cap else 2_000
        rungs["hhrt"] = _to_contexts(
            _stage_hh_redteam(out_dir, hh_cap, stream_cap, seed),
            behavior=behavior,
            split=split,
            rung="hhrt",
        )
        rungs["toxicchat"] = _to_contexts(
            _stage_toxicchat(out_dir, None, stream_cap),
            behavior=behavior,
            split=split,
            rung="toxicchat",
        )
    elif behavior == "sycophancy" and split == "train":
        per_split = (cap or 16_000) // 2
        ra = _stage_reddit(
            out_dir,
            "relationship_advice",
            per_split,
            stream_cap,
            seed,
            tag="train",
            partition="train",
        )
        ss = _stage_reddit(
            out_dir, "socialskills", per_split, stream_cap, seed, tag="train", partition="train"
        )
        rows = [dict(r, subreddit="relationship_advice") for r in ra] + [
            dict(r, subreddit="socialskills") for r in ss
        ]
        contexts = []
        for i, r in enumerate(rows):
            contexts.append(
                {
                    "context_id": f"sycophancy-train-train-{i:06d}",
                    "behavior": behavior,
                    "split": split,
                    "rung": "train",
                    "group_key": f"{r['subreddit']}-{r['source_id']}",
                    "prefix_text": "",
                    "query": r["text"],
                    "source_dataset": f"REDDIT_submissions/{r['subreddit']}",
                    "source_id": r["source_id"],
                }
            )
        rungs["train"] = contexts
        extra_manifest["per_split_quota"] = per_split
    elif behavior == "sycophancy" and split == "eval":
        rows, flag = _stage_elephant_or_fallback(out_dir, cap or 2_000, stream_cap, seed)
        rungs["aita"] = _to_contexts(
            rows, behavior=behavior, split=split, rung="aita", group_prefix="post"
        )
        extra_manifest.update(flag)
    elif behavior == "hallucination" and split == "train":
        rungs["train"] = _to_contexts(
            _stage_trivia(out_dir, cap or 16_000, stream_cap, seed),
            behavior=behavior,
            split=split,
            rung="train",
        )
    elif behavior == "hallucination" and split == "eval":
        rungs["nqopen"] = _to_contexts(
            _stage_nq_open(out_dir, None, stream_cap),
            behavior=behavior,
            split=split,
            rung="nqopen",
        )
        rungs["simpleqa"] = _to_contexts(
            _stage_simpleqa(out_dir, None, stream_cap),
            behavior=behavior,
            split=split,
            rung="simpleqa",
        )
    else:  # pragma: no cover - registry validation makes this unreachable
        raise KeyError((behavior, split))

    dedup_drops: dict[str, int] = {}
    if split == "eval":
        # Disjointness contract: ensure train staged, then near-dup filter
        # every eval rung against it.
        train_spec = get_spec(behavior, "train")
        train_path = staged_context_path(out_dir, behavior, "train", "train")
        if not train_path.exists():
            stage_corpus(
                behavior, "train", train_spec.cap, seed, out_dir=out_dir, stream_cap=stream_cap
            )
        train_rows = read_jsonl(train_path)
        for rung, rows in list(rungs.items()):
            kept, dropped = enforce_disjointness(train_rows, rows)
            rungs[rung] = kept
            dedup_drops[rung] = dropped
            if not kept:
                raise RuntimeError(
                    f"disjointness filter dropped ALL {behavior}/{split}/{rung} rows"
                )

    for rung, rows in rungs.items():
        _write_jsonl_atomic(staged_context_path(out_dir, behavior, split, rung), rows)

    manifest = {
        "behavior": behavior,
        "split": split,
        "cap": cap,
        "seed": seed,
        "stream_cap": stream_cap,
        "rungs": {rung: len(rows) for rung, rows in rungs.items()},
        "n_groups": {rung: len({r["group_key"] for r in rows}) for rung, rows in rungs.items()},
        "near_dup_drops": dedup_drops,
        **extra_manifest,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = out_dir / f"{behavior}_{split}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("[stage] manifest %s: %s", manifest_path, manifest)
    return manifest
