#!/usr/bin/env python3
"""Issue #1738 — multi-turn prefix-arm corpus at ~100k: manifest + generate + capture.

Fork-and-extend of the #779 n1m driver (``issue779_ffc_n1m_generate_capture``,
MAIN-resident — plan §4 / §10 parent-lineage record: main ⊇ every parent branch).
The single manipulated variable vs the n1m parent is the CORPUS CONSTRUCTION:
first-turn single-turn prompts → natural multi-turn conversations (≥2 user
turns), with the prefix-end capture position added. Deltas (plan §4):

1. **Row extraction** — ``_multiturn_context(row)``: parse ``row["conversation"]``
   (role/content dicts, both corpora); require ≥2 user turns, strictly
   alternating roles starting with ``user``, all turns non-empty; ``messages`` =
   conversation up to and including the LAST user turn (a logged trailing
   assistant answer is dropped — we regenerate); ``prefix_messages`` =
   ``messages[:-1]``; record ``depth`` = n_user_turns + ``source_hash``.
2. **Dedup + near-dupe on the rendered context** (not the bare prompt): exact
   dedup on the normalized plain render; ``NearDupeGate`` (inherited) — or the
   pre-registered deterministic bottom-k=1024 MinHash sketch fallback — targets
   the fresh pinned val ∪ test ∪ holdout rendered contexts (~11,400 targets).
3. **Manifest + split in one pass, pre-registered per-corpus allocation** —
   REPLACES the parent's LMSYS-first-then-WildChat-top-up (quota-driven; would
   zero WildChat at this target). Stream BOTH corpora to EXHAUSTION, record
   E_L/E_W, then allocate n_W = clamp(round(N·E_W/(E_L+E_W)), min(30000, E_W),
   E_W), n_L = min(N − n_W, E_L) by seeded draw (RNG 1738); carve
   val 400 / test 1000 / holdout 10000 stratified by corpus × depth-band
   ({2, 3–4, ≥5} user turns); near-dupe-screen the remaining train pool against
   the carve (holdout excluded from every fit BY CONSTRUCTION).
4. **Capture: prefix-end read from the SAME context forward** (no third
   forward): one teacher-forced forward over the full context render captures
   ``cx_last`` (last token) AND ``px_last`` at index ``prefix_len − 1`` under
   the STRICT-TOKEN-PREFIX identity gate (``full_ids[:prefix_len] ==
   prefix_ids``, per-row; violation → row skipped + sidecar-recorded; pilot
   gate ≤0.5%). ``v_x`` via the parent's own ``capture_answer_vector``
   (full-template re-tokenization, answer span incl. end-of-turn tail).
5. **Chunk schema** — ``{px_last, cx_last, v_x (n, L, H) fp32, ci, prompts
   (=rendered messages json), response, depth, corpus, layers, shard_index,
   chunk}``; raw rollout text (full conversation + generated answer) rides in
   every chunk AND per-shard raw_completions JSONs uploaded before any reduce.

All parent ops fixes are inherited (imported, not copied): K=10 batched upload
commits, per-chunk upload→sha-verify→purge, over-length skip sidecars,
flock-serialized manifest download, external-stream checkpoint+fingerprint
resume, vLLM chunked generation + spawn workers + eager/prefix-caching knobs.

Refusal-safety: LMSYS + WildChat are unscreened real-user corpora. This driver
NEVER prints or logs conversation/rollout text — only counts, indices, corpus
tags, depths, and sha256s. Do not add such logging.

GPU (H100/A100) per capture shard; manifest build is CPU (cpu-mid lane). NO
judge/API calls. Fail loud — NaN never coerced.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# vLLM V1 fork-safety (#628): spawn BEFORE any vllm import in the process.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_collect as COL  # noqa: E402
import issue779_common as C  # noqa: E402
import numpy as np  # noqa: E402
import issue779_ffc_n1m_generate_capture as N1M  # noqa: E402
import issue779_ffc_n10k_generate_capture as N10  # noqa: E402
import issue779_ffc_n50k_generate_capture as N50  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1738_mt_gc")

# ── constants (parent-inherited unless the plan overrides) ────────────────────────
DEFAULT_MODEL = N1M.DEFAULT_MODEL  # Qwen/Qwen2.5-7B-Instruct
LMSYS_REPO = N1M.LMSYS_REPO
WILDCHAT_REPO = N1M.WILDCHAT_REPO
HF_PREFIX = "issue1738_multiturn"
MANIFEST_SUBDIR = N1M.MANIFEST_SUBDIR  # "sampling_manifest" (shared download helper)
CAPTURE_SUBDIR = "capture"
RAW_SUBDIR = "raw_completions"
KRESAMPLE_SUBDIR = "kresample"

CAPTURE_LAYERS = list(N1M.CAPTURE_LAYERS)  # [14, 19, 26]
MAX_MODEL_LEN = N1M.MAX_MODEL_LEN  # 8192
GEN_MAX_TOKENS = N1M.GEN_MAX_TOKENS  # 1024
PROMPT_TOKEN_BUDGET = N1M.PROMPT_TOKEN_BUDGET  # 7104 (parent frozen-manifest skip mechanism)
UPLOAD_BATCH = N1M.UPLOAD_BATCH  # K=10 batched upload commits (429-storm fix)

N_TARGET = 100_000  # plan §11: N = 100,000 (G0 re-sizes on shortfall)
G0_POOL_FLOOR = 111_400  # eligible pool >= this => N = 100,000; else N = pool
WILDCHAT_ALLOC_FLOOR = 30_000  # lower clamp min(30_000, E_W) (lmsys_transfer power)
N_VAL, N_TEST, N_HOLDOUT = 400, 1_000, 10_000
SPLIT_SEED = 1738
DEPTH_BANDS = ((2, 2), (3, 4), (5, 10_000))  # {2, 3–4, ≥5} user turns
MAX_CONTEXT_CHARS = 26_000  # cheap manifest-time prefilter (exact 7104-tok filter at gen)
STREAM_CHECKPOINT_EVERY = 20_000
FILTER_RECIPE_VERSION = "mt100k-v1"
MINHASH_K = 1024  # bottom-k sketch size (pre-registered near-dupe fallback)
PILOT_VIOLATION_RATE_MAX = 0.005  # G1: strict-token-prefix violations <= 0.5%
PILOT_PREFIX_COS_MAX = 0.999  # G1/K1: min pairwise px cos must be BELOW this
KRESAMPLE_SEEDS = (43, 44, 45, 46)


def _depth_band(depth: int) -> str:
    for lo, hi in DEPTH_BANDS:
        if lo <= depth <= hi:
            return f"{lo}-{hi}" if hi < 10_000 else f">={lo}"
    raise ValueError(f"depth {depth} outside every band")


# ── delta 1: multi-turn row extraction ────────────────────────────────────────────


def _multiturn_context(row) -> tuple[dict | None, str]:
    """Parse one corpus row into a multi-turn context, or (None, reject_reason).

    Keeps ``messages`` = turns up to and including the LAST user turn (any
    logged trailing assistant answer is dropped — the answer is regenerated
    on-policy). Requires: a list of role/content dicts, all non-empty, strictly
    alternating roles starting with ``user``, and ≥2 user turns in ``messages``.
    Returns ({"messages", "depth"}, "ok") on keep. Refusal-safe: no text logged.
    """
    conv = row.get("conversation")
    if not isinstance(conv, list) or len(conv) < 3:  # need at least u, a, u
        return None, "too_short_or_not_list"
    turns: list[dict] = []
    for t in conv:
        if not isinstance(t, dict):
            return None, "bad_turn"
        role = t.get("role")
        content = t.get("content") or t.get("value")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            return None, "bad_turn"
        turns.append({"role": role, "content": content.strip()})
    for i, t in enumerate(turns):
        if t["role"] != ("user" if i % 2 == 0 else "assistant"):
            return None, "non_alternating"
    last_user = len(turns) - 1 if turns[-1]["role"] == "user" else len(turns) - 2
    messages = turns[: last_user + 1]
    depth = sum(1 for t in messages if t["role"] == "user")
    if depth < 2:
        return None, "too_few_user_turns"
    return {"messages": messages, "depth": depth}, "ok"


def _plain_render(messages: list[dict]) -> str:
    """Deterministic tokenizer-free plain render (dedup/near-dupe/length surface)."""
    return "\n".join(f"{t['role']}: {t['content']}" for t in messages)


def _render_key(messages: list[dict]) -> str:
    """Normalized (lowercased, whitespace-collapsed) render — the exact-dedup key."""
    return " ".join(_plain_render(messages).lower().split())


def _source_hash(row, messages: list[dict]) -> str:
    ch = row.get("conversation_hash")  # WildChat carries this; LMSYS does not
    if isinstance(ch, str) and ch:
        return f"wc:{ch}"
    return "sha:" + hashlib.sha256(_render_key(messages).encode("utf-8")).hexdigest()[:32]


# ── delta 2: MinHash bottom-k near-dupe fallback (pre-registered, plan §4/§12(l)) ──


class DfFilteredNearDupeGate(N1M.NearDupeGate):
    """Parent exact gate + a document-frequency cap on the CANDIDATE index.

    On rendered multi-turn contexts the role-prefix grams (``user: `` /
    ``assistant: ``) index EVERY target, so the parent's inverted index prunes
    nothing (probe-measured 92 rows/s at 200 targets → ~1.6 rows/s extrapolated
    at 11,400). A gram indexing more than ``df_frac`` of targets is dropped from
    the candidate INDEX only — the Jaccard itself is still computed on the FULL
    gram sets, so a true near-dupe (J ≥ 0.8 shares ≥80% of the union, rare grams
    included) still surfaces via its rare shared grams."""

    def __init__(
        self, targets, ngram=N1M.NEAR_DUPE_NGRAM, thresh=N1M.NEAR_DUPE_JACCARD, df_frac=0.05
    ):
        super().__init__(targets, ngram, thresh)
        cap = max(2, int(df_frac * max(1, len(self.target_ngrams))))
        self.df_cap = cap
        self.inv = {g: s for g, s in self.inv.items() if len(s) <= cap}

    def stats(self) -> dict:
        return {**super().stats(), "impl": "exact_df_capped", "df_cap": self.df_cap}

    def dupe_kind(self, text: str) -> str | None:
        """'exact' | 'near' | None via counter snapshot (no parent-logic duplication)."""
        e0 = self.n_exact_drop
        if not self.is_dupe(text):
            return None
        return "exact" if self.n_exact_drop > e0 else "near"


# splitmix64 finalizer constants (vectorized 64-bit mix; deterministic, pure arithmetic)
_SM64_M1 = np.uint64(0xBF58476D1CE4E5B9)
_SM64_M2 = np.uint64(0x94D049BB133111EB)
_POLY64 = np.uint64(1099511628211)  # FNV-1a 64-bit prime (rolling-window combiner)


def _mix64(v: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64 finalizer over a uint64 array (wraps mod 2^64)."""
    v = v.copy()
    v ^= v >> np.uint64(30)
    v *= _SM64_M1
    v ^= v >> np.uint64(27)
    v *= _SM64_M2
    v ^= v >> np.uint64(31)
    return v


class MinHashNearDupeGate:
    """Deterministic bottom-k (k=1024) sketch over char-5-grams; Jaccard estimated
    as |bottomk(A∪B) ∩ A_sk ∩ B_sk| / k (the standard bottom-k estimator), with an
    inverted index over sketch values so a candidate is compared only against
    targets sharing ≥1 sketch hash. Same is_dupe/stats surface as NearDupeGate.

    r3 (#1738 compute-deviation): the sketch is fully VECTORIZED — code points via
    utf-32-le, a 5-pass polynomial rolling hash + splitmix64 finalize, np.unique
    bottom-k — replacing the per-gram Python crc32 loop the probe measured at
    ~37 rows/s (dominated by gram-string materialization + per-gram hashing).
    Deterministic across processes (pure arithmetic; no PYTHONHASHSEED exposure)."""

    def __init__(self, targets: list[str], ngram: int = 5, thresh: float = 0.8, k: int = MINHASH_K):
        self.ngram, self.thresh, self.k = int(ngram), float(thresh), int(k)
        self.exact: set[str] = set()
        self.sketches: list[np.ndarray] = []  # sorted-unique uint64, len <= k
        self.inv: dict[int, set[int]] = {}
        for ti, t in enumerate(targets):
            n = " ".join(t.lower().split())
            self.exact.add(n)
            sk = self._sketch(n)
            self.sketches.append(sk)
            for h in sk.tolist():
                self.inv.setdefault(h, set()).add(ti)
        # same df cap as the exact gate: a sketch hash indexing >5% of targets
        # prunes nothing (candidate index only; the estimator uses full sketches).
        cap = max(2, int(0.05 * max(1, len(targets))))
        self.inv = {h: s for h, s in self.inv.items() if len(s) <= cap}
        # per-target INDEXED sketch size (hashes surviving the df cap) — the
        # denominator for the hit-count prefilter in dupe_kind: near-identical
        # targets keep only a few indexed (discriminating) grams, so the floor
        # must scale with what the index can actually count, not the raw sketch.
        self.indexed_n = [sum(1 for h in sk.tolist() if h in self.inv) for sk in self.sketches]
        self.n_exact_drop = 0
        self.n_near_drop = 0

    def _sketch(self, norm_text: str) -> np.ndarray:
        """SORTED bottom-k uint64 sketch of the char-ngram set (vectorized)."""
        if not norm_text:
            return np.empty(0, dtype=np.uint64)
        cps = np.frombuffer(norm_text.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
        n_win = cps.size - self.ngram + 1
        if n_win <= 0:  # degenerate short text: one whole-text "gram" (parent convention)
            v = np.zeros(1, dtype=np.uint64)
            for j in range(cps.size):
                v = v * _POLY64 + cps[j : j + 1]
        else:
            v = np.zeros(n_win, dtype=np.uint64)
            for j in range(self.ngram):  # ngram vectorized passes over all windows
                v = v * _POLY64 + cps[j : j + n_win]
        return np.unique(_mix64(v))[: self.k]  # sorted unique -> bottom-k

    def dupe_kind(self, text: str) -> str | None:
        """'exact' | 'near' | None — pure (no counter mutation; parallel-safe)."""
        n = " ".join(text.lower().split())
        if n in self.exact:
            return "exact"
        sk = self._sketch(n)
        if sk.size == 0:
            return None
        # candidate HIT COUNTS from the inverted index (distinct shared indexed
        # sketch hashes per target) — cheap dict increments, no per-candidate numpy.
        counts: dict[int, int] = {}
        n_probe_indexed = 0
        inv = self.inv
        for h in sk.tolist():
            hit = inv.get(h)
            if hit:
                n_probe_indexed += 1
                for ti in hit:
                    counts[ti] = counts.get(ti, 0) + 1
        for ti, hits in counts.items():
            tsk = self.sketches[ti]
            # prefilter: a J>=thresh near-dupe shares ~thresh of the pair's
            # INDEXED (df-surviving) sketch hashes — the same conservative-index
            # assumption the exact gate's df cap documents; the floor scales
            # with the indexed sizes (a boilerplate-heavy target keeps only a
            # few indexed discriminating grams, so its floor is small), with a
            # 0.5 margin for asymmetric pruning. Low-hit candidates skip the
            # exact estimator (the per-candidate numpy calls that dominate on
            # gram-colliding pools).
            if hits < 0.5 * self.thresh * max(1, min(n_probe_indexed, self.indexed_n[ti])):
                continue
            inter = np.intersect1d(sk, tsk, assume_unique=True)
            if inter.size == 0:
                continue
            # bottom-k of the union + intersection count within it (standard
            # bottom-k estimator): union1d is sorted-unique, so the k-th smallest
            # union value bounds membership; searchsorted counts inter below it.
            u = np.union1d(sk, tsk)
            taken = min(self.k, u.size)
            if taken == 0:
                continue
            cutoff = u[taken - 1]
            n_in = int(np.searchsorted(inter, cutoff, side="right"))
            if n_in / taken >= self.thresh:
                return "near"
        return None

    def is_dupe(self, text: str) -> bool:
        kind = self.dupe_kind(text)
        if kind == "exact":
            self.n_exact_drop += 1
        elif kind == "near":
            self.n_near_drop += 1
        return kind is not None

    def stats(self) -> dict:
        return {
            "impl": "minhash_bottomk",
            "ngram": self.ngram,
            "jaccard_thresh": self.thresh,
            "k": self.k,
            "n_targets": len(self.sketches),
            "n_exact_drop": self.n_exact_drop,
            "n_near_drop": self.n_near_drop,
        }


def _make_gate(targets: list[str], impl: str):
    if impl == "minhash":
        return MinHashNearDupeGate(targets)
    return DfFilteredNearDupeGate(targets)


# ── delta 2 (cont., r3): parallel checkpointed near-dupe SCREEN (#1738 c-dev) ─────
# Fork-inherited worker state (set in the PARENT before Pool construction; texts
# + gate ride copy-on-write pages, so chunk args stay tiny index tuples).
_SCREEN_STATE: dict = {}


def _screen_chunk(bounds: tuple[int, int]) -> tuple[int, int, list[int], list[int]]:
    """Screen candidate positions [lo, hi) against the fork-inherited gate.

    Returns (lo, hi, exact_pool_ids, near_pool_ids). Decisions are per-row pure
    (target set fixed; no cross-row state), so chunk order / parallelism cannot
    change the result."""
    gate = _SCREEN_STATE["gate"]
    keys = _SCREEN_STATE["keys"]
    cand = _SCREEN_STATE["cand"]
    ex: list[int] = []
    nr: list[int] = []
    for j in range(bounds[0], bounds[1]):
        pool_id = cand[j]
        kind = gate.dupe_kind(keys[pool_id])
        if kind == "exact":
            ex.append(pool_id)
        elif kind == "near":
            nr.append(pool_id)
    return bounds[0], bounds[1], ex, nr


def _screen_fingerprint(keys: list[str], carve: dict[str, list[int]], gate, impl: str) -> str:
    """Resume-regime fingerprint for the screen checkpoint: every output-affecting
    key (recipe, seed, gate impl+params, pool size, carve shas) + a cheap content
    signature over ALL render keys (len + crc32 per row; text never hashed out)."""
    sig = {
        "recipe": FILTER_RECIPE_VERSION,
        "seed": SPLIT_SEED,
        "impl": impl,
        "gate": {
            k: v
            for k, v in gate.stats().items()
            if k in ("impl", "ngram", "jaccard_thresh", "k", "df_cap", "n_targets")
        },
        "n_pool": len(keys),
        "carve_sha": {name: _sha_int_list(sorted(carve[name])) for name in sorted(carve)},
    }
    h = hashlib.sha256(json.dumps(sig, sort_keys=True).encode())
    for k in keys:
        h.update(len(k).to_bytes(8, "little"))
        h.update(zlib.crc32(k.encode()).to_bytes(4, "little"))
    return h.hexdigest()


def _screen_candidates(
    keys: list[str],
    cand: list[int],
    gate,
    *,
    ckpt_path: Path,
    fingerprint: str,
    resume: bool = True,
    procs: int = 0,
    chunk: int = 4000,
    log_every: int = 10_000,
    ckpt_every: int = 50_000,
) -> tuple[list[int], list[int]]:
    """Near-dupe-screen ``keys[cand[i]]`` for all candidate positions, parallel +
    checkpointed. Returns (exact_ids, near_ids) as pool ids (ascending).

    - procs>1: fork Pool over contiguous candidate chunks (ordered imap; results
      identical to serial by per-row purity). procs=0 -> min(8, cpu_count).
    - progress: one log line per >=log_every screened rows (rate + ETA) — a
      silent multi-hour screen is the #1738 c-dev observability defect.
    - checkpoint: atomic JSON at ckpt_path every >=ckpt_every rows, keyed on the
      full regime fingerprint; a killed screen resumes, a regime change rebuilds."""
    procs = procs or max(1, min(8, os.cpu_count() or 1))
    n = len(cand)
    state = {"fingerprint": fingerprint, "n_done": 0, "exact_ids": [], "near_ids": []}
    if resume and ckpt_path.exists():
        old = json.loads(ckpt_path.read_text())
        if old.get("fingerprint") == fingerprint:
            state = old
            logger.info(
                "[neardupe] RESUMED checkpoint: %d/%d screened, %d dropped",
                int(state["n_done"]),
                n,
                len(state["exact_ids"]) + len(state["near_ids"]),
            )
        else:
            logger.info("[neardupe] checkpoint fingerprint MISMATCH — re-screening from scratch")
    start = int(state["n_done"])
    exact_ids = [int(x) for x in state["exact_ids"]]
    near_ids = [int(x) for x in state["near_ids"]]
    if start >= n:
        return exact_ids, near_ids
    _SCREEN_STATE.update({"gate": gate, "keys": keys, "cand": cand})
    bounds = [(lo, min(lo + chunk, n)) for lo in range(start, n, chunk)]
    t0 = time.time()
    last_log = start
    last_ckpt = start

    def _consume(lo: int, hi: int, ex: list[int], nr: list[int]) -> None:
        nonlocal last_log, last_ckpt
        exact_ids.extend(ex)
        near_ids.extend(nr)
        if hi - last_log >= log_every or hi == n:
            rate = (hi - start) / max(time.time() - t0, 1e-9)
            logger.info(
                "[neardupe] %d/%d screened (%.0f rows/s, %d dropped, est %.1f min remaining)",
                hi,
                n,
                rate,
                len(exact_ids) + len(near_ids),
                (n - hi) / max(rate, 1e-9) / 60.0,
            )
            last_log = hi
        if hi - last_ckpt >= ckpt_every or hi == n:
            N1M._atomic_write_json(
                ckpt_path,
                {
                    "fingerprint": fingerprint,
                    "n_done": hi,
                    "exact_ids": exact_ids,
                    "near_ids": near_ids,
                },
            )
            last_ckpt = hi

    if procs > 1 and len(bounds) > 1:
        ctx = multiprocessing.get_context("fork")  # CPU manifest path; no CUDA in-process
        with ctx.Pool(procs) as pool_obj:
            for lo, hi, ex, nr in pool_obj.imap(_screen_chunk, bounds):
                _consume(lo, hi, ex, nr)
    else:
        for b in bounds:
            lo, hi, ex, nr = _screen_chunk(b)
            _consume(lo, hi, ex, nr)
    return exact_ids, near_ids


# ── delta 3: multi-turn streaming (exhaustion, checkpointed, reject counters) ─────


def _stream_multiturn(
    repo: str,
    corpus_tag: str,
    cache_dir: Path,
    fingerprint: dict,
    *,
    resume: bool,
    smoke_stream=None,
    max_scan: int | None = None,
    keep_cap: int | None = None,
    extra_seen: set[str] | None = None,
) -> tuple[list[dict], dict]:
    """Stream one corpus keeping eligible multi-turn contexts, with the parent's
    on-disk checkpoint + fingerprint resume and PER-FILTER REJECT COUNTERS in the
    done-line (#1092 real-corpus probe class). ``max_scan`` caps TOTAL streamed
    rows and ``keep_cap`` caps kept rows (the bounded tiny-real probe); both
    None = stream to exhaustion. ``extra_seen`` = render keys already kept by an
    earlier corpus (cross-corpus exact dedup). Kept rows persist to
    ``{cache_dir}/{corpus_tag}.jsonl`` (the phase-0 text persistence for the
    manifest); text is never logged."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    pool_path = cache_dir / f"{corpus_tag}.jsonl"
    meta_path = cache_dir / f"{corpus_tag}.meta.json"
    kept: list[dict] = []
    consumed = 0
    counters: Counter = Counter()
    if resume and pool_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fingerprint:
            kept = N1M._read_jsonl(pool_path)
            consumed = int(meta.get("consumed", 0))
            counters = Counter(meta.get("counters", {}))
            if meta.get("complete"):
                logger.info(
                    "[stream %s] RESUMED complete cache: %d kept (stream skipped)",
                    corpus_tag,
                    len(kept),
                )
                return kept, dict(counters)
            logger.info(
                "[stream %s] RESUMED partial cache: %d kept, fast-forward %d rows",
                corpus_tag,
                len(kept),
                consumed,
            )
        else:
            logger.info("[stream %s] fingerprint MISMATCH; re-streaming from scratch", corpus_tag)
            kept, consumed, counters = [], 0, Counter()

    if smoke_stream is not None:
        it = iter(smoke_stream[consumed:])
        ds = None
    else:
        from datasets import load_dataset

        ds = load_dataset(repo, split="train", streaming=True)
        it = iter(ds.skip(consumed) if consumed else ds)

    def _flush(complete: bool) -> None:
        N1M._atomic_write_jsonl(pool_path, kept)
        N1M._atomic_write_json(
            meta_path,
            {
                "fingerprint": fingerprint,
                "consumed": consumed,
                "kept": len(kept),
                "complete": complete,
                "counters": dict(counters),
            },
        )

    seen: set[str] = {r["render_key"] for r in kept}
    row = None
    while (max_scan is None or consumed < max_scan) and (keep_cap is None or len(kept) < keep_cap):
        row = next(it, None)
        if row is None:
            break
        consumed += 1
        ctx, reason = _multiturn_context(row)
        if ctx is None:
            counters[reason] += 1
            continue
        plain = _plain_render(ctx["messages"])
        if len(plain) > MAX_CONTEXT_CHARS:
            counters["over_length_chars"] += 1
            continue
        key = _render_key(ctx["messages"])
        if key in seen or (extra_seen is not None and key in extra_seen):
            counters["exact_dupe"] += 1
            continue
        kept.append(
            {
                "messages": ctx["messages"],
                "depth": int(ctx["depth"]),
                "corpus": corpus_tag,
                "stream_pos": consumed - 1,
                "source_hash": _source_hash(row, ctx["messages"]),
                "n_chars": len(plain),
                "render_key": key,
            }
        )
        seen.add(key)
        counters["kept"] += 1
        if len(kept) % STREAM_CHECKPOINT_EVERY == 0:
            _flush(complete=False)
            logger.info(
                "[stream %s] checkpoint: %d kept / %d consumed", corpus_tag, len(kept), consumed
            )
    complete = (
        row is None
        or (keep_cap is not None and len(kept) >= keep_cap)
        or (max_scan is not None and consumed >= max_scan)
    )
    _flush(complete=complete)
    logger.info(
        "[stream %s] done: scanned=%d kept=%d rejects=%s",
        corpus_tag,
        consumed,
        len(kept),
        {k: v for k, v in sorted(counters.items()) if k != "kept"},
    )
    if ds is not None:  # release streaming dataset before shutdown (#952 rc=134 guard)
        del it, ds, row
        gc.collect()
    return kept, dict(counters)


def _depth_hist(rows: list[dict]) -> dict[str, int]:
    h: Counter = Counter(_depth_band(r["depth"]) for r in rows)
    return {k: int(v) for k, v in sorted(h.items())}


def run_probe(args, *, smoke_lmsys=None, smoke_wildchat=None) -> dict:
    """Bounded tiny-real streaming probe (#1092 class): run the PRODUCTION
    filter/dedup path on the first ``--probe-rows`` REAL rows of each corpus,
    assert kept > 0 per corpus, report keep-rate + depth histogram + near-dupe
    gate throughput (exact vs minhash) on the probe's own kept rows."""
    cache = args.out_dir / "probe_cache"
    lm, lm_ct = _stream_multiturn(
        LMSYS_REPO,
        "lmsys",
        cache,
        {"recipe": FILTER_RECIPE_VERSION, "probe_rows": args.probe_rows, "phase": "probe-lmsys"},
        resume=not args.no_resume_stream,
        smoke_stream=smoke_lmsys,
        max_scan=args.probe_rows,
    )
    wc, wc_ct = _stream_multiturn(
        WILDCHAT_REPO,
        "wildchat",
        cache,
        {"recipe": FILTER_RECIPE_VERSION, "probe_rows": args.probe_rows, "phase": "probe-wc"},
        resume=not args.no_resume_stream,
        smoke_stream=smoke_wildchat,
        max_scan=args.probe_rows,
        extra_seen={r["render_key"] for r in lm},
    )
    assert lm, "probe kept 0 LMSYS multi-turn rows — filter chain rejects everything (#1092 class)"
    assert wc, "probe kept 0 WildChat multi-turn rows — filter chain rejects everything"

    # near-dupe gate throughput on REAL probe rows (targets = a slice of kept rows).
    import numpy as np

    pool = lm + wc
    targets = [r["render_key"] for r in pool[: min(200, len(pool))]]
    probes = [r["render_key"] for r in pool]
    timing: dict[str, float] = {}
    decisions: dict[str, list[bool]] = {}
    for impl in ("exact", "minhash"):
        g = _make_gate(targets, impl)
        t0 = time.time()
        dec = [g.is_dupe(p) for p in probes]
        dt = max(time.time() - t0, 1e-9)
        timing[impl] = len(probes) / dt
        decisions[impl] = dec
        logger.info(
            "[probe] near-dupe %s: %.0f rows/s (%d/%d self-hits over %d targets)",
            impl,
            timing[impl],
            sum(dec),
            len(probes),
            len(targets),
        )
    # r3 equivalence leg (#1738 c-dev fix scope 3): MinHash drop set vs the primary
    # exact gate — (a) on the REAL probe rows; (b) on seeded planted perturbations
    # of the targets (char deletions at rates bracketing the J>=0.8 boundary), so
    # boundary agreement is quantified where random rows carry no near-dupes.
    # under_drop (exact drops, minhash keeps) is the RISK direction (screen is
    # conservative filtering; over-drop only costs pool rows). Digest-only: no text.
    ex_dec, mh_dec = decisions["exact"], decisions["minhash"]
    equiv_real = {
        "n": len(probes),
        "agree": sum(a == b for a, b in zip(ex_dec, mh_dec)),
        "under_drop": sum(a and not b for a, b in zip(ex_dec, mh_dec)),
        "over_drop": sum(b and not a for a, b in zip(ex_dec, mh_dec)),
        "exact_drops": sum(ex_dec),
        "minhash_drops": sum(mh_dec),
    }
    rng = np.random.default_rng(SPLIT_SEED)
    ge, gm = _make_gate(targets, "exact"), _make_gate(targets, "minhash")
    planted: dict[str, dict[str, int]] = {}
    for frac in (0.01, 0.02, 0.05, 0.10):
        row = {"n": 0, "exact_drops": 0, "minhash_drops": 0, "under_drop": 0, "over_drop": 0}
        for t in targets[: min(100, len(targets))]:
            keep = rng.random(len(t)) >= frac
            pert = "".join(c for c, k in zip(t, keep) if k)
            a, b = ge.is_dupe(pert), gm.is_dupe(pert)
            row["n"] += 1
            row["exact_drops"] += int(a)
            row["minhash_drops"] += int(b)
            row["under_drop"] += int(a and not b)
            row["over_drop"] += int(b and not a)
        planted[f"del_{frac}"] = row
    logger.info(
        "[probe] near-dupe equivalence: real %d/%d agree (under=%d over=%d); planted %s",
        equiv_real["agree"],
        equiv_real["n"],
        equiv_real["under_drop"],
        equiv_real["over_drop"],
        {k: (v["under_drop"], v["over_drop"], v["exact_drops"]) for k, v in planted.items()},
    )
    meta = {
        "probe_rows_per_corpus": int(args.probe_rows),
        "lmsys": {"kept": len(lm), "counters": lm_ct, "depth_hist": _depth_hist(lm)},
        "wildchat": {"kept": len(wc), "counters": wc_ct, "depth_hist": _depth_hist(wc)},
        "gate_rows_per_s": timing,
        "near_dupe_equiv": {"real_rows": equiv_real, "planted_char_del": planted},
        "recipe_version": FILTER_RECIPE_VERSION,
    }
    N1M._atomic_write_json(args.out_dir / "probe_meta.json", meta)
    logger.info(
        "[probe] done: %s", {k: meta[k] for k in ("probe_rows_per_corpus", "gate_rows_per_s")}
    )
    return meta


# ── delta 3 (cont.): allocation + stratified carve + near-dupe screen ─────────────


def allocate_counts(n_target: int, e_l: int, e_w: int) -> tuple[int, int, int]:
    """Pre-registered per-corpus allocation (plan §4 bullet 3 / §7 G0).

    N = n_target if pool >= G0_POOL_FLOOR else the whole pool (take-what-exists);
    n_W = clamp(round(N·E_W/(E_L+E_W)), min(30000, E_W), E_W); n_L = min(N−n_W, E_L).
    Returns (n_sel, n_l, n_w)."""
    pool = e_l + e_w
    assert pool > 0, "empty eligible pool"
    n_sel = n_target if pool >= G0_POOL_FLOOR else pool
    n_w_prop = round(n_sel * e_w / pool)
    n_w = min(max(n_w_prop, min(WILDCHAT_ALLOC_FLOOR, e_w)), e_w)
    n_l = min(n_sel - n_w, e_l)
    # if L cannot fill its share (tiny E_L), give the remainder back to W.
    short = n_sel - n_w - n_l
    if short > 0:
        n_w = min(n_w + short, e_w)
    return n_l + n_w, n_l, n_w


def _seeded_subset(n_pool: int, n_take: int, rng) -> list[int]:
    """Sorted seeded draw of n_take indices from range(n_pool) (stream order kept)."""
    if n_take >= n_pool:
        return list(range(n_pool))
    return sorted(rng.choice(n_pool, size=n_take, replace=False).tolist())


def stratified_carve(pool: list[dict], seed: int) -> dict[str, list[int]]:
    """Carve val/test/holdout GLOBAL indices stratified by corpus × depth-band
    (largest-remainder proportional allocation per stratum), seeded. The rest is
    the train pool. Holdout is excluded from every fit BY CONSTRUCTION."""
    import numpy as np

    strata: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(pool):
        strata.setdefault((r["corpus"], _depth_band(r["depth"])), []).append(i)
    n = len(pool)
    rng = np.random.default_rng(seed)
    out: dict[str, list[int]] = {"val": [], "test": [], "holdout": []}
    targets = {"val": N_VAL, "test": N_TEST, "holdout": N_HOLDOUT}
    # scale down carve targets when the pool is tiny (smoke / K2 shortfall regime)
    total_carve = sum(targets.values())
    if total_carve >= n:
        scale = n / (2 * total_carve)
        targets = {k: max(1, int(v * scale)) for k, v in targets.items()}
        logger.warning("[carve] pool %d < carve %d — scaled targets to %s", n, total_carve, targets)
    keys = sorted(strata.keys())
    shuffled = {k: rng.permutation(np.asarray(strata[k], dtype=np.int64)) for k in keys}
    cursor = dict.fromkeys(keys, 0)
    for set_name in ("val", "test", "holdout"):
        want = targets[set_name]
        quota = {k: want * len(strata[k]) / n for k in keys}
        alloc = {k: int(quota[k]) for k in keys}
        rem = want - sum(alloc.values())
        for k in sorted(keys, key=lambda k: quota[k] - int(quota[k]), reverse=True)[:rem]:
            alloc[k] += 1
        for k in keys:
            take = min(alloc[k], len(shuffled[k]) - cursor[k])
            out[set_name].extend(int(x) for x in shuffled[k][cursor[k] : cursor[k] + take])
            cursor[k] += take
    for s in out.values():
        s.sort()
    return out


def _sha_int_list(ids: list[int]) -> str:
    h = hashlib.sha256()
    for i in ids:
        h.update(str(int(i)).encode())
        h.update(b",")
    return h.hexdigest()


def build_manifest(args, *, smoke_lmsys=None, smoke_wildchat=None) -> dict:
    """Phase 0: probe → stream both corpora to exhaustion → allocate → carve →
    near-dupe-screen → write + upload manifest parts, meta, and split_1738.json."""
    import numpy as np

    C.phase("mt-probe")
    if not args.skip_probe:
        run_probe(args, smoke_lmsys=smoke_lmsys, smoke_wildchat=smoke_wildchat)

    C.phase("mt-stream")
    cache = args.out_dir / "stream_cache"
    fp = {"recipe": FILTER_RECIPE_VERSION, "max_chars": MAX_CONTEXT_CHARS, "phase": "full"}
    lmsys_pool, lm_ct = _stream_multiturn(
        LMSYS_REPO,
        "lmsys",
        cache,
        {**fp, "corpus": "lmsys"},
        resume=not args.no_resume_stream,
        smoke_stream=smoke_lmsys,
        max_scan=args.max_scan_rows,
    )
    wc_pool, wc_ct = _stream_multiturn(
        WILDCHAT_REPO,
        "wildchat",
        cache,
        {**fp, "corpus": "wildchat"},
        resume=not args.no_resume_stream,
        smoke_stream=smoke_wildchat,
        max_scan=args.max_scan_rows,
        extra_seen={r["render_key"] for r in lmsys_pool},
    )
    e_l, e_w = len(lmsys_pool), len(wc_pool)
    n_sel, n_l, n_w = allocate_counts(args.n_target, e_l, e_w)
    logger.info(
        "[alloc] E_L=%d E_W=%d -> N=%d (n_L=%d, n_W=%d; floor=%d, G0 pool floor=%d)",
        e_l,
        e_w,
        n_sel,
        n_l,
        n_w,
        WILDCHAT_ALLOC_FLOOR,
        G0_POOL_FLOOR,
    )
    transfer_descoped = e_w < 5_000  # plan §7 G0 shortfall degradation
    if transfer_descoped:
        logger.warning("[alloc] E_W=%d < 5000 — lmsys_transfer fold DESCOPED (reported)", e_w)

    rng = np.random.default_rng(SPLIT_SEED)
    sel_l = _seeded_subset(e_l, n_l, rng)
    sel_w = _seeded_subset(e_w, n_w, rng)
    pool = [lmsys_pool[i] for i in sel_l] + [wc_pool[i] for i in sel_w]

    C.phase("mt-carve")
    carve = stratified_carve(pool, SPLIT_SEED)
    carve_set = set(carve["val"]) | set(carve["test"]) | set(carve["holdout"])

    C.phase("mt-neardupe")
    targets = [pool[i]["render_key"] for i in sorted(carve_set)]
    # r3 fix (#1738 compute-deviation v1): "auto" previously resolved to the EXACT
    # gate — the pre-registered MinHash fallback was probe-timed + CLI-only, never
    # routed into the production screen, which then spun >5h on ~600k pool rows.
    # "auto" now routes to the registered bottom-k MinHash sketch (plan §8/§12(l));
    # --near-dupe-impl exact remains the explicit opt-in (equivalence reference).
    impl = "minhash" if args.near_dupe_impl == "auto" else args.near_dupe_impl
    gate = _make_gate(targets, impl)
    procs = args.screen_procs or max(1, min(8, os.cpu_count() or 1))
    logger.info(
        "[neardupe] impl=%s targets=%d procs=%d (production screen route; #1738 r3 fix)",
        gate.stats()["impl"],
        len(targets),
        procs,
    )
    keys = [r["render_key"] for r in pool]
    cand = [i for i in range(len(pool)) if i not in carve_set]
    fp_screen = _screen_fingerprint(keys, carve, gate, impl)
    t0 = time.time()
    exact_ids, near_ids = _screen_candidates(
        keys,
        cand,
        gate,
        ckpt_path=args.out_dir / "neardupe_ckpt.json",
        fingerprint=fp_screen,
        resume=not args.no_resume_stream,
        procs=procs,
    )
    dropped_set = set(exact_ids) | set(near_ids)
    dropped = len(dropped_set)
    train_ids = [i for i in cand if i not in dropped_set]
    logger.info(
        "[neardupe] screened %d train candidates in %.0fs (%s, procs=%d): "
        "%d dropped (%d exact, %d near)",
        len(cand),
        time.time() - t0,
        gate.stats()["impl"],
        procs,
        dropped,
        len(exact_ids),
        len(near_ids),
    )

    # manifest keeps carve + surviving train rows, re-indexed with global i.
    keep_ids = sorted(carve_set | set(train_ids))
    old_to_new = {old: new for new, old in enumerate(keep_ids)}
    manifest_rows = []
    for old in keep_ids:
        r = dict(pool[old])
        r.pop("render_key", None)  # dedup surface; not needed downstream
        r["i"] = old_to_new[old]
        r["split"] = (
            "val"
            if old in set(carve["val"])
            else "test"
            if old in set(carve["test"])
            else "holdout"
            if old in set(carve["holdout"])
            else "train"
        )
        manifest_rows.append(r)
    split_doc = {
        "seed": SPLIT_SEED,
        "n_manifest": len(manifest_rows),
        "recipe_version": FILTER_RECIPE_VERSION,
        "sets": {},
        "strata": {"depth_bands": ["2-2", "3-4", ">=5"], "by": "corpus x depth_band"},
        "transfer_descoped": bool(transfer_descoped),
    }
    for name in ("val", "test", "holdout"):
        ids = sorted(old_to_new[i] for i in carve[name])
        split_doc["sets"][name] = {"ci": ids, "n": len(ids), "sha256": _sha_int_list(ids)}
    train_ci = sorted(old_to_new[i] for i in train_ids)
    split_doc["sets"]["train"] = {
        "ci": train_ci,
        "n": len(train_ci),
        "sha256": _sha_int_list(train_ci),
    }

    meta = {
        "n_new": len(manifest_rows),  # read_manifest_pool contract
        "n_target": int(args.n_target),
        "n_selected": int(n_sel),
        "n_eligible": {"lmsys": e_l, "wildchat": e_w},
        "alloc": {"n_lmsys": n_l, "n_wildchat": n_w, "wildchat_floor": WILDCHAT_ALLOC_FLOOR},
        "g0": {
            "pool_floor": G0_POOL_FLOOR,
            "pool": e_l + e_w,
            "transfer_descoped": transfer_descoped,
        },
        "stream_counters": {"lmsys": lm_ct, "wildchat": wc_ct},
        # drop counts from the PARENT-side aggregation (worker gate counters do
        # not propagate across the fork boundary); gate.stats() keeps impl/params.
        "near_dupe": {
            **gate.stats(),
            "n_exact_drop": len(exact_ids),
            "n_near_drop": len(near_ids),
            "n_train_dropped": dropped,
            "screen_procs": procs,
        },
        "depth_hist": _depth_hist(manifest_rows),
        "split_shas": {k: v["sha256"] for k, v in split_doc["sets"].items()},
        "capture_layers": list(CAPTURE_LAYERS),
        "model": args.model,
        "recipe_version": FILTER_RECIPE_VERSION,
        "max_context_chars": MAX_CONTEXT_CHARS,
        "seed": SPLIT_SEED,
    }
    manifest_dir = args.out_dir / MANIFEST_SUBDIR
    n_parts = N1M._write_manifest_parts(manifest_dir, manifest_rows, meta)
    N1M._atomic_write_json(manifest_dir / "split_1738.json", split_doc)
    logger.info(
        "[manifest] wrote %d rows (%d lmsys + %d wildchat) in %d parts; carve=%s train=%d",
        len(manifest_rows),
        sum(1 for r in manifest_rows if r["corpus"] == "lmsys"),
        sum(1 for r in manifest_rows if r["corpus"] == "wildchat"),
        n_parts,
        {k: split_doc["sets"][k]["n"] for k in ("val", "test", "holdout")},
        len(train_ci),
    )
    if not args.no_upload:
        N1M._upload_manifest(manifest_dir, args.hf_prefix)
    C.phase("manifest-done")
    return meta


# ── delta 4: capture (cx_last + px_last from ONE forward; v_x parent-verbatim) ────


def _render_messages(tok, messages: list[dict], *, add_generation_prompt: bool) -> str:
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def _capture_context_and_prefix(hf, tok, messages: list[dict], layers: list[int]):
    """ONE teacher-forced forward over the context render captures cx_last (last
    token) AND px_last at index prefix_len − 1, under the STRICT-TOKEN-PREFIX
    identity gate. Returns (dict | None, "ok" | violation_reason)."""
    prefix_messages = messages[:-1]
    assert prefix_messages and prefix_messages[-1]["role"] == "assistant", (
        "prefix must end assistant"
    )
    ctx_text = _render_messages(tok, messages, add_generation_prompt=True)
    px_text = _render_messages(tok, prefix_messages, add_generation_prompt=False)
    ctx_ids = tok(ctx_text, return_tensors="pt", padding=False)["input_ids"]
    px_ids = tok(px_text, return_tensors="pt", padding=False)["input_ids"]
    prefix_len = int(px_ids.shape[1])
    full_len = int(ctx_ids.shape[1])
    if prefix_len >= full_len or ctx_ids[0, :prefix_len].tolist() != px_ids[0].tolist():
        return None, "prefix_token_mismatch"
    suffix = tok.decode(ctx_ids[0, -3:])
    assert suffix == C.GENERATION_SUFFIX, (
        f"context position assert failed: last-3 decode {suffix!r} != {C.GENERATION_SUFFIX!r}"
    )
    ctx_ids = ctx_ids.to(hf.device)
    captured = extract_layer_activations(hf, ctx_ids, layers)
    cx, px = [], []
    for li in layers:
        hs = captured[li][0]  # (T, H)
        cx.append(hs[-1, :].float().cpu())
        px.append(hs[prefix_len - 1, :].float().cpu())
    return (
        {
            "cx_last": torch.stack(cx),  # (L, H)
            "px_last": torch.stack(px),  # (L, H)
            "prompt_len": full_len,
            "prefix_len": prefix_len,
        },
        "ok",
    )


def _capture_shard_multiturn(hf, tok, rows, responses, layers):
    """Per-row capture for one chunk. ``rows`` are manifest rows (messages/depth/
    corpus/i) 1:1 with ``responses``. Returns (kept_row_dicts, violations)."""
    out, violations = [], []
    for r, resp in zip(rows, responses, strict=True):
        cap, reason = _capture_context_and_prefix(hf, tok, r["messages"], layers)
        if cap is None:
            violations.append({"ci": int(r["i"]), "reason": reason})
            continue
        av = COL.capture_answer_vector(
            hf, tok, r["messages"], resp, layers, {}, keep_per_token=False
        )
        if av is None:  # empty response
            continue
        out.append(
            {
                "ci": int(r["i"]),
                "messages": r["messages"],
                "response": resp,
                "depth": int(r["depth"]),
                "corpus": r["corpus"],
                "cx_last": cap["cx_last"],
                "px_last": cap["px_last"],
                "v_x": av["v_x"],
            }
        )
    return out, violations


def _stack_chunk_mt(rows, layers, shard_index, chunk_idx) -> dict:
    """Chunk bundle (plan §4 delta 5): px_last/cx_last/v_x (n, L, H) fp32 + text."""
    return {
        "px_last": torch.stack([r["px_last"] for r in rows]),
        "cx_last": torch.stack([r["cx_last"] for r in rows]),
        "v_x": torch.stack([r["v_x"] for r in rows]),
        "ci": [int(r["ci"]) for r in rows],
        "prompts": [json.dumps(r["messages"], ensure_ascii=False) for r in rows],
        "response": [r["response"] for r in rows],
        "depth": [int(r["depth"]) for r in rows],
        "corpus": [r["corpus"] for r in rows],
        "layers": list(layers),
        "shard_index": int(shard_index),
        "chunk": int(chunk_idx),
    }


def _flush_upload_batch_mt(scratch: Path, prefix: str, pt_names: list[str], raw_names: list[str]):
    """Parent's K-batched two-commit upload→verify→purge, with the #1738 subdirs
    (``capture/`` + ``raw_completions/``). Mirrors N1M._flush_upload_batch."""
    if not pt_names and not raw_names:
        return
    if pt_names:
        local_shas = {n: N50._sha256_file(scratch / n) for n in pt_names}
        url = hub._upload_folder_filtered(
            scratch,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{prefix}/{CAPTURE_SUBDIR}",
            allow_patterns=list(pt_names),
            expected_repo_paths=[f"{prefix}/{CAPTURE_SUBDIR}/{n}" for n in pt_names],
        )
        if not url:
            raise RuntimeError(
                f"batch upload of {len(pt_names)} capture .pt to {prefix}/{CAPTURE_SUBDIR} "
                "returned no URL"
            )
        remote = N50._remote_index(f"{prefix}/{CAPTURE_SUBDIR}")
        for n in pt_names:
            meta = remote.get(n)
            if meta is None:
                raise RuntimeError(f"{n} not present on Hub after batch upload (verify listing)")
            if meta["sha256"] is None or meta["sha256"] != local_shas[n]:
                raise RuntimeError(
                    f"{n} Hub LFS sha256 {meta['sha256']} != local {local_shas[n]} — upload corrupt"
                )
        for n in pt_names:
            (scratch / n).unlink()
        logger.info("[upload] batch of %d capture .pt verified (sha) + purged", len(pt_names))
    if raw_names:
        url = hub._upload_folder_filtered(
            scratch,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{prefix}/{RAW_SUBDIR}",
            allow_patterns=list(raw_names),
            expected_repo_paths=[f"{prefix}/{RAW_SUBDIR}/{n}" for n in raw_names],
        )
        if not url:
            raise RuntimeError(
                f"batch upload of {len(raw_names)} raw_completions to {prefix}/{RAW_SUBDIR} "
                "returned no URL"
            )
        for n in raw_names:
            (scratch / n).unlink()
        logger.info(
            "[upload] batch of %d raw_completions verified (presence) + purged", len(raw_names)
        )


def _filter_rows_overlength(
    rows: list[dict], tok_len_fn, budget: int
) -> tuple[list[dict], list[dict]]:
    """Partition manifest rows by rendered-prompt token length (the parent's
    frozen-manifest skip mechanism). Order-preserving in both partitions;
    refusal-safe (records ci + token count, never text)."""
    kept, skipped = [], []
    for r in rows:
        n_tok = int(tok_len_fn(r["messages"]))
        if n_tok > budget:
            skipped.append({"ci": int(r["i"]), "n_tokens": n_tok})
        else:
            kept.append(r)
    return kept, skipped


def _generate_multiturn(llm, tok, messages_list: list[list[dict]], *, seed: int = 42) -> list[str]:
    """1 rollout per multi-turn context under the parent decoding recipe
    (temp 1.0, top_p 0.95, max 1024 tok, engine/request seed). CPU-smoke path
    (llm None) returns stub responses through the SAME downstream capture code."""
    if llm is None:
        return ["This is a short stub response for the CPU capture smoke."] * len(messages_list)
    from vllm import SamplingParams

    sp = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=GEN_MAX_TOKENS, seed=seed)
    prompt_texts = [_render_messages(tok, m, add_generation_prompt=True) for m in messages_list]
    gen = COL._vllm_generate_chunked(llm, prompt_texts, sp)  # list[list[str]]
    return [g[0] for g in gen]


def _load_models(args):
    """Real path: parent load_models (GENERATION_SUFFIX assert). Tiny-CPU path
    (--tiny-model, smoke only): REAL Qwen tokenizer + from-config 2-layer Qwen2
    over the real vocab, so the chat-template/token-id convention transfers."""
    if not args.tiny_model:
        return N10.load_models(args.model, args.device)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    torch.manual_seed(0)
    hf = AutoModelForCausalLM.from_config(cfg)
    hf.eval()
    return tok, hf


def _min_pairwise_cos(px: torch.Tensor) -> dict[str, float]:
    """px (n, L, H) fp32 → per-layer min pairwise cosine (the K1/G1 sanity read)."""
    out = {}
    n = px.shape[0]
    for li in range(px.shape[1]):
        x = px[:, li, :].double()
        x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
        g = x @ x.T
        off = g[~torch.eye(n, dtype=torch.bool)]
        out[str(li)] = float(off.min().item()) if off.numel() else float("nan")
    return out


def run_capture(args) -> int:
    manifest_dir = _resolve_manifest_dir(args)
    pool, _meta = N1M.read_manifest_pool(manifest_dir)
    n_total = len(pool)
    start, end = N50._shard_range(n_total, args.num_shards, args.shard_index)
    shard_pool = pool[start:end]
    if args.pilot_cap and args.pilot_cap > 0:
        shard_pool = shard_pool[: args.pilot_cap]
    layers = [int(x) for x in args.capture_layers.split(",")]
    logger.info(
        "[shard %d/%d] range [%d, %d) = %d contexts (%d total pool)%s",
        args.shard_index,
        args.num_shards,
        start,
        end,
        len(shard_pool),
        n_total,
        f" PILOT cap {args.pilot_cap}" if args.pilot_cap else "",
    )
    if not shard_pool:
        logger.info("[shard %d] empty range; nothing to do", args.shard_index)
        C.phase("done")
        return 0

    scratch = args.out_dir / "shards"
    scratch.mkdir(parents=True, exist_ok=True)
    done_pt = (
        set(N50._remote_index(f"{args.hf_prefix}/{CAPTURE_SUBDIR}"))
        if not args.no_upload
        else set()
    )
    done_raw = (
        set(N50._remote_index(f"{args.hf_prefix}/{RAW_SUBDIR}")) if not args.no_upload else set()
    )

    C.phase("load_model")
    tok, hf = _load_models(args)
    llm = N1M._build_capture_engine(args) if args.device == "cuda" else None
    h_dim = int(hf.config.hidden_size)

    C.phase("capture")
    n_sub = (len(shard_pool) + args.shard_size - 1) // args.shard_size
    kept_total = 0
    t_start = time.time()
    pending_pt: list[str] = []
    pending_raw: list[str] = []
    violations_all: list[dict] = []
    skipped_all: list[dict] = []
    pilot_px: list[torch.Tensor] = []

    def _flush_pending() -> None:
        if args.no_upload or not pending_pt:
            return
        _flush_upload_batch_mt(scratch, args.hf_prefix, pending_pt, pending_raw)
        pending_pt.clear()
        pending_raw.clear()

    def _on_sigterm(signum, frame):
        raise SystemExit(f"SIGTERM ({signum}) received — flushing pending upload batch")

    prev_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        for ci_idx, s in enumerate(range(0, len(shard_pool), args.shard_size)):
            name = f"shard{args.shard_index:02d}_chunk{ci_idx:04d}.pt"
            raw_name = f"shard{args.shard_index:02d}_chunk{ci_idx:04d}.json"
            chunk = shard_pool[s : s + args.shard_size]
            # over-length filter on the EXACT generation render (parent mechanism)
            kept_rows, skipped = _filter_rows_overlength(
                chunk,
                lambda m: len(
                    tok(
                        _render_messages(tok, m, add_generation_prompt=True),
                        add_special_tokens=False,
                    )["input_ids"]
                ),
                PROMPT_TOKEN_BUDGET,
            )
            skipped_all.extend(skipped)
            if skipped:
                logger.warning(
                    "[shard %d] chunk %d/%d: skipped %d over-length (> %d tok); cis=%s",
                    args.shard_index,
                    ci_idx + 1,
                    n_sub,
                    len(skipped),
                    PROMPT_TOKEN_BUDGET,
                    [d["ci"] for d in skipped],
                )
            if name in done_pt and raw_name in done_raw:
                logger.info(
                    "[shard %d] chunk %d/%d already on Hub; skip",
                    args.shard_index,
                    ci_idx + 1,
                    n_sub,
                )
                continue
            if not kept_rows:
                logger.warning(
                    "[shard %d] chunk %d: 0 rows after length filter; skip",
                    args.shard_index,
                    ci_idx,
                )
                continue
            ts = time.time()
            responses = _generate_multiturn(llm, tok, [r["messages"] for r in kept_rows])
            rows, violations = _capture_shard_multiturn(hf, tok, kept_rows, responses, layers)
            violations_all.extend(violations)
            if not rows:
                logger.warning(
                    "[shard %d] chunk %d: 0 captured rows; skip", args.shard_index, ci_idx
                )
                continue
            for fld in ("px_last", "cx_last", "v_x"):
                for r in rows:
                    assert r[fld].shape == (len(layers), h_dim), (fld, r[fld].shape)
            torch.save(_stack_chunk_mt(rows, layers, args.shard_index, ci_idx), scratch / name)
            C.write_json_atomic(
                scratch / raw_name,
                {
                    "shard_index": args.shard_index,
                    "chunk": ci_idx,
                    "rows": [
                        {
                            "ci": int(r["ci"]),
                            "messages": r["messages"],
                            "response": r["response"],
                            "depth": int(r["depth"]),
                            "corpus": r["corpus"],
                        }
                        for r in rows
                    ],
                },
            )
            if args.pilot_cap:
                pilot_px.append(torch.stack([r["px_last"] for r in rows]))
            kept_total += len(rows)
            if not args.no_upload:
                pending_pt.append(name)
                pending_raw.append(raw_name)
                if len(pending_pt) >= UPLOAD_BATCH:
                    _flush_pending()
            logger.info(
                "[capture] chunk %d/%d shard=%d: %d/%d captured (%d viol, %d over-len, %.0fs)",
                ci_idx + 1,
                n_sub,
                args.shard_index,
                len(rows),
                len(chunk),
                len(violations),
                len(skipped),
                time.time() - ts,
            )
        _flush_pending()
    except BaseException:
        try:
            _flush_pending()
        except Exception:
            logger.exception(
                "[shard %d] best-effort pending-batch flush failed on exit", args.shard_index
            )
        raise
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)

    _write_sidecar(scratch, args, skipped_all, violations_all)
    wall_h = (time.time() - t_start) / 3600.0
    logger.info(
        "[shard %d] done: %d kept rows across %d chunks (%d over-length, %d prefix violations, %.2f h)",
        args.shard_index,
        kept_total,
        n_sub,
        len(skipped_all),
        len(violations_all),
        wall_h,
    )
    if args.pilot_cap:
        _write_pilot_meta(args, kept_total, wall_h, skipped_all, violations_all, pilot_px)
    C.phase("done")
    return 0


def _write_sidecar(scratch: Path, args, skipped_all: list[dict], violations_all: list[dict]):
    """Per-shard sidecar: over-length skips + strict-token-prefix violations
    (ci + counts only, never text). Uploaded beside raw_completions."""
    skip_name = f"shard{args.shard_index:02d}_skipped.json"
    C.write_json_atomic(
        scratch / skip_name,
        {
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
            "prompt_token_budget": PROMPT_TOKEN_BUDGET,
            "n_skipped": len(skipped_all),
            "skipped": skipped_all,
            "n_prefix_violations": len(violations_all),
            "prefix_violations": violations_all,
        },
    )
    if args.no_upload:
        return
    url = hub._upload_folder_filtered(
        scratch,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{args.hf_prefix}/{RAW_SUBDIR}",
        allow_patterns=[skip_name],
        expected_repo_paths=[f"{args.hf_prefix}/{RAW_SUBDIR}/{skip_name}"],
    )
    if not url:
        raise RuntimeError(f"skipped-sidecar upload of {skip_name} returned no URL")


def _write_pilot_meta(args, kept_total, wall_h, skipped_all, violations_all, pilot_px):
    """G1 pilot artifact: throughput + violation rate + prefix-variability sanity."""
    n_attempted = kept_total + len(violations_all)
    viol_rate = len(violations_all) / max(1, n_attempted)
    px = torch.cat(pilot_px, dim=0) if pilot_px else torch.zeros(0, 1, 1)
    min_cos = _min_pairwise_cos(px) if px.shape[0] >= 2 else {}
    layers = [int(x) for x in args.capture_layers.split(",")]
    min_cos_by_layer = {str(layers[int(k)]): v for k, v in min_cos.items()}
    ctx_per_gpu_h = kept_total / wall_h if wall_h > 0 else float("nan")
    verdict = {
        "violation_rate_ok": viol_rate <= PILOT_VIOLATION_RATE_MAX,
        "prefix_varies_ok": bool(min_cos_by_layer)
        and all(v < PILOT_PREFIX_COS_MAX for v in min_cos_by_layer.values()),
    }
    doc = {
        "n_captured": int(kept_total),
        "wall_h": float(wall_h),
        "ctx_per_gpu_h": float(ctx_per_gpu_h),
        "n_overlength_skipped": len(skipped_all),
        "n_prefix_violations": len(violations_all),
        "violation_rate": float(viol_rate),
        "violation_rate_max": PILOT_VIOLATION_RATE_MAX,
        "prefix_min_pairwise_cos_by_layer": min_cos_by_layer,
        "prefix_cos_max": PILOT_PREFIX_COS_MAX,
        "gate_g1": verdict,
        # DELIBERATE (review Minor 4): ctx_per_gpu_h is NOT a gate_g1 term — the
        # plan's G1 throughput clause is a RE-SIZE/descope decision the
        # orchestrator makes from this reported rate (fleet sizing), not a kill.
        "g1_throughput_note": "ctx_per_gpu_h feeds the orchestrator's fleet-sizing/"
        "descope decision (plan G1 re-size semantics); deliberately not a gate term",
        "shard_index": int(args.shard_index),
        "pilot_cap": int(args.pilot_cap),
    }
    out = args.out_dir / "pilot_meta.json"
    C.write_json_atomic(out, doc)
    logger.info(
        "[pilot] ctx/GPU-h=%.0f viol=%.4f min_cos=%s g1=%s",
        ctx_per_gpu_h,
        viol_rate,
        min_cos_by_layer,
        verdict,
    )
    if not args.no_upload:
        url = hub._upload(
            out,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=args.hf_prefix,
            upload_as_file=True,
        )
        if not url:
            raise RuntimeError("pilot_meta.json upload returned no URL")


def _resolve_manifest_dir(args) -> Path:
    local = args.out_dir / MANIFEST_SUBDIR
    if args.manifest_from_hf:
        return N1M._download_manifest(args.hf_prefix, local)
    if not (local / "meta.json").exists():
        raise SystemExit(
            f"manifest {local}/meta.json absent — run --build-sampling-manifest first, "
            "or pass --manifest-from-hf"
        )
    return local


# ── K-resample fresh-draw capture (Phase 4a; #1482 recipe, seeds 43–46) ──────────


def run_kresample(args) -> int:
    """K=4 fresh answers per subsampled holdout context (per-request seeds 43–46),
    v captured under the SAME full-template convention. The own seed-42 draw is
    STREAMED from Phase-2 chunks by the floor estimator — never recaptured
    (the #1482 93.1% convention-mismatch lesson)."""
    manifest_dir = _resolve_manifest_dir(args)
    pool, _meta = N1M.read_manifest_pool(manifest_dir)
    sub_path = Path(args.kresample_subsample)
    if not sub_path.exists():
        raise SystemExit(f"kresample subsample doc {sub_path} absent — run characterize first")
    sub = json.loads(sub_path.read_text())
    cis = [int(c) for c in sub["ci"]]
    got_sha = _sha_int_list(cis)
    if got_sha != sub["sha256"]:  # corrupted / hand-edited doc fails loud (Minor 6)
        raise SystemExit(f"kresample subsample sha mismatch: {got_sha} != {sub['sha256']}")
    seeds = [int(s) for s in args.seeds.split(",")]
    by_ci = {int(r["i"]): r for r in pool}
    rows = [by_ci[c] for c in cis]
    start, end = N50._shard_range(len(rows), args.num_shards, args.shard_index)
    rows = rows[start:end]
    layers = [int(x) for x in args.capture_layers.split(",")]
    logger.info(
        "[kresample] shard %d/%d: %d contexts x %d seeds",
        args.shard_index,
        args.num_shards,
        len(rows),
        len(seeds),
    )
    if not rows:
        C.phase("done")
        return 0

    scratch = args.out_dir / "kresample"
    scratch.mkdir(parents=True, exist_ok=True)
    name = f"kresample_shard{args.shard_index:02d}.pt"
    raw_name = f"kresample_shard{args.shard_index:02d}.json"
    if not args.no_upload and name in set(
        N50._remote_index(f"{args.hf_prefix}/{KRESAMPLE_SUBDIR}")
    ):
        logger.info("[kresample] shard %d already on Hub; skip", args.shard_index)
        C.phase("done")
        return 0

    C.phase("load_model")
    tok, hf = _load_models(args)
    llm = N1M._build_capture_engine(args) if args.device == "cuda" else None
    h_dim = int(hf.config.hidden_size)

    C.phase("kresample")
    v_by_seed: dict[int, list] = {s: [] for s in seeds}
    kept_ci: list[int] = []
    raw_rows: list[dict] = []
    msgs = [r["messages"] for r in rows]
    t_units0 = time.time()
    resp_by_seed = {s: _generate_multiturn(llm, tok, msgs, seed=s) for s in seeds}
    for j, r in enumerate(rows):
        vs = {}
        for s in seeds:
            av = COL.capture_answer_vector(
                hf, tok, r["messages"], resp_by_seed[s][j], layers, {}, keep_per_token=False
            )
            if av is None:
                break
            vs[s] = av["v_x"]
        if len(vs) != len(seeds):  # a seed produced an empty answer — drop the context
            continue
        kept_ci.append(int(r["i"]))
        for s in seeds:
            v_by_seed[s].append(vs[s])
        raw_rows.append(
            {
                "ci": int(r["i"]),
                "messages": r["messages"],
                "responses": {str(s): resp_by_seed[s][j] for s in seeds},
            }
        )
        # per-unit progress line (checkpoint-per-phase T2 convention): the
        # capture loop is the wall-clock phase; a silent loop wedges pollers.
        logger.info(
            "[kresample] unit %d/%d ci=%d elapsed=%.0fs",
            j + 1,
            len(rows),
            int(r["i"]),
            time.time() - t_units0,
        )
    if not kept_ci:
        logger.warning("[kresample] shard %d kept 0 contexts", args.shard_index)
        C.phase("done")
        return 0
    V = torch.stack([torch.stack(v_by_seed[s]) for s in seeds], dim=1)  # (n, K, L, H)
    assert V.shape[2:] == (len(layers), h_dim), V.shape
    torch.save(
        {"V": V.to(torch.float16), "ci": kept_ci, "seeds": seeds, "layers": layers},
        scratch / name,
    )
    C.write_json_atomic(scratch / raw_name, {"shard_index": args.shard_index, "rows": raw_rows})
    if not args.no_upload:
        for fn in (name, raw_name):
            url = hub._upload_folder_filtered(
                scratch,
                repo_id=C.HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=f"{args.hf_prefix}/{KRESAMPLE_SUBDIR}",
                allow_patterns=[fn],
                expected_repo_paths=[f"{args.hf_prefix}/{KRESAMPLE_SUBDIR}/{fn}"],
            )
            if not url:
                raise RuntimeError(f"kresample upload of {fn} returned no URL")
        for fn in (name, raw_name):
            (scratch / fn).unlink()
    logger.info(
        "[kresample] shard %d done: %d contexts x %d seeds",
        args.shard_index,
        len(kept_ci),
        len(seeds),
    )
    C.phase("done")
    return 0


# ── CPU logic smoke (synthetic streams; production code paths) ────────────────────


def _synth_conv(qi: int, n_user: int, *, wc: bool = False) -> dict:
    """Synthetic alternating conversation with n_user user turns + trailing answer."""
    conv = []
    for k in range(n_user):
        conv.append(
            {"role": "user", "content": f"synthetic question {qi} turn {k} about topic {qi % 7}"}
        )
        conv.append(
            {"role": "assistant", "content": f"synthetic answer {qi} turn {k} with details"}
        )
    row = {"conversation": conv}
    if wc:
        row["conversation_hash"] = f"hash{qi}"
    return row


def _smoke(args) -> int:
    """Model-free CPU logic smoke of the manifest path (production functions on
    synthetic streams) + the capture-indexing leg on the REAL Qwen tokenizer with
    a tiny from-config 2-layer Qwen2 (--tiny-model path). No network corpora."""
    logger.info("[smoke] multi-turn manifest + capture-indexing CPU smoke")

    # (1) row extraction: keeps + every reject reason.
    ctx, r = _multiturn_context(_synth_conv(1, 3))
    assert r == "ok" and ctx["depth"] == 3 and ctx["messages"][-1]["role"] == "user", (r, ctx)
    assert _multiturn_context({"conversation": "nope"})[1] == "too_short_or_not_list"
    # a single-user-turn conversation (u, a) rejects at the length gate; the
    # depth<2 branch is defensive (unreachable once alternation + len>=3 hold).
    assert _multiturn_context(_synth_conv(2, 1))[1] == "too_short_or_not_list"
    bad = _synth_conv(3, 2)
    bad["conversation"][1]["content"] = "  "
    assert _multiturn_context(bad)[1] == "bad_turn"
    swapped = _synth_conv(4, 2)
    swapped["conversation"][0]["role"] = "assistant"
    assert _multiturn_context(swapped)[1] == "non_alternating"

    # (2) allocation clamp arithmetic (proportional / floor / shortfall / take-all).
    n, nl, nw = allocate_counts(100_000, 300_000, 300_000)
    assert (n, nl, nw) == (100_000, 50_000, 50_000), (n, nl, nw)
    n, nl, nw = allocate_counts(100_000, 500_000, 20_000)  # floor binds at min(30k, E_W)=20k
    assert nw == 20_000 and nl == 80_000, (n, nl, nw)
    n, nl, nw = allocate_counts(100_000, 400_000, 40_000)  # prop=9091 < floor 30k -> 30k
    assert nw == 30_000 and nl == 70_000, (n, nl, nw)
    n, nl, nw = allocate_counts(100_000, 50_000, 30_000)  # pool 80k < 111.4k -> take all
    assert n == 80_000 and nl == 50_000 and nw == 30_000, (n, nl, nw)

    # (3) synthetic streams -> full manifest build (probe + carve + gate + parts).
    smoke_lmsys = [_synth_conv(i, 2 + (i % 5)) for i in range(400)]
    smoke_wc = [_synth_conv(1000 + i, 2 + (i % 4), wc=True) for i in range(300)]
    smoke_wc.append(smoke_lmsys[0])  # cross-corpus exact dupe -> must be dropped
    long_row = _synth_conv(9999, 2)
    long_row["conversation"][0]["content"] = "x " * 15_000  # 30k chars > MAX_CONTEXT_CHARS
    smoke_lmsys.append(long_row)  # over_length_chars reject branch must fire
    ns = argparse.Namespace(**{**vars(args), "no_upload": True, "n_target": 500, "probe_rows": 100})
    meta = build_manifest(ns, smoke_lmsys=smoke_lmsys, smoke_wildchat=smoke_wc)
    # r3 routing pin (#1738 c-dev): "auto" MUST resolve the PRODUCTION screen to the
    # MinHash gate — the pre-fix "auto"->exact routing is the 5h-spin incident.
    assert meta["near_dupe"]["impl"] == "minhash_bottomk", meta["near_dupe"]
    assert meta["stream_counters"]["wildchat"].get("exact_dupe", 0) >= 1, meta["stream_counters"]
    assert meta["stream_counters"]["lmsys"].get("over_length_chars", 0) >= 1, meta[
        "stream_counters"
    ]

    # generation-time over-length gate (degenerate probe, parent _smoke_length_filter
    # pattern): a fake token-length fn trips the budget on exactly one row.
    fr = [{"i": 7, "messages": [{"role": "user", "content": "a"}]} for _ in range(1)]
    fr += [{"i": 8, "messages": [{"role": "user", "content": "b"}]}]
    kept_p, skipped_p = _filter_rows_overlength(
        fr, lambda m: PROMPT_TOKEN_BUDGET + 1 if m[0]["content"] == "b" else 5, PROMPT_TOKEN_BUDGET
    )
    assert [r["i"] for r in kept_p] == [7] and [d["ci"] for d in skipped_p] == [8], (
        kept_p,
        skipped_p,
    )
    pool, meta2 = N1M.read_manifest_pool(ns.out_dir / MANIFEST_SUBDIR)
    assert meta2["n_new"] == len(pool)
    split = json.loads((ns.out_dir / MANIFEST_SUBDIR / "split_1738.json").read_text())
    all_ci = [r["i"] for r in pool]
    for s in ("val", "test", "holdout", "train"):
        ids = split["sets"][s]["ci"]
        assert split["sets"][s]["sha256"] == _sha_int_list(ids)
        assert set(ids).issubset(set(all_ci)), s
    sets = [set(split["sets"][s]["ci"]) for s in ("val", "test", "holdout", "train")]
    assert sum(len(s) for s in sets) == len(set().union(*sets)) == len(pool), (
        "split not a partition"
    )

    # determinism: a second build from the same streams is byte-identical.
    ns2 = argparse.Namespace(**{**vars(ns), "out_dir": args.out_dir / "_smoke_build2"})
    ns2.out_dir.mkdir(parents=True, exist_ok=True)
    build_manifest(ns2, smoke_lmsys=smoke_lmsys, smoke_wildchat=smoke_wc)
    a = (ns.out_dir / MANIFEST_SUBDIR / "part_00000.jsonl").read_bytes()
    b = (ns2.out_dir / MANIFEST_SUBDIR / "part_00000.jsonl").read_bytes()
    assert a == b, "manifest build not deterministic"

    # (4) minhash gate agrees with exact gate on planted near-dupes.
    tgt = [
        f"what is the capital of country number {i} in the atlas of the world" for i in range(30)
    ]
    ge, gm = _make_gate(tgt, "exact"), _make_gate(tgt, "minhash")
    for probe, want in (
        (tgt[3], True),
        (tgt[3] + "?", True),
        ("a distinct short cooking question", False),
    ):
        assert ge.is_dupe(probe) == want, ("exact", probe, want)
        assert gm.is_dupe(probe) == want, ("minhash", probe, want)

    # (4b) r3 parallel screen (#1738 c-dev): fork-Pool path == inline path; a
    # mid-run checkpoint resumes to the identical result; a fingerprint mismatch
    # re-screens from scratch (never reuses wrong cached rows, #722 r3 class).
    keys4 = [
        f"pool row {i} with shared filler text about topic {i % 11} and details" for i in range(300)
    ]
    keys4[7] = tgt[3]  # exact dupe of a target
    keys4[19] = tgt[5] + "!"  # near dupe (1 new 5-gram => J ~0.98 >= 0.8)
    cand4 = list(range(len(keys4)))
    ck = ns.out_dir / "_smoke_screen_ckpt.json"
    fp4 = "smoke-screen-fp-1"
    kw4 = dict(ckpt_path=ck, fingerprint=fp4, chunk=64, log_every=100, ckpt_every=128)
    ex_p, nr_p = _screen_candidates(
        keys4, cand4, _make_gate(tgt, "minhash"), resume=False, procs=2, **kw4
    )
    ex_s, nr_s = _screen_candidates(
        keys4, cand4, _make_gate(tgt, "minhash"), resume=False, procs=1, **kw4
    )
    assert (ex_p, nr_p) == (ex_s, nr_s) and ex_p == [7] and nr_p == [19], (ex_p, nr_p, ex_s, nr_s)
    partial = {
        "fingerprint": fp4,
        "n_done": 128,
        "exact_ids": [x for x in ex_p if x < 128],
        "near_ids": [x for x in nr_p if x < 128],
    }
    N1M._atomic_write_json(ck, partial)
    ex_r, nr_r = _screen_candidates(
        keys4, cand4, _make_gate(tgt, "minhash"), resume=True, procs=2, **kw4
    )
    assert (ex_r, nr_r) == (ex_p, nr_p), (ex_r, nr_r)
    N1M._atomic_write_json(ck, {**partial, "fingerprint": "other-regime"})
    ex_m, nr_m = _screen_candidates(
        keys4, cand4, _make_gate(tgt, "minhash"), resume=True, procs=1, **kw4
    )
    assert (ex_m, nr_m) == (ex_p, nr_p), (ex_m, nr_m)

    # (5) capture-indexing leg: REAL tokenizer + tiny 2-layer Qwen2 (--tiny-model),
    # exercising the strict-token-prefix gate + px/cx positions + v_x through the
    # PRODUCTION per-row capture path (stub generation).
    cargs = argparse.Namespace(
        **{
            **vars(args),
            "tiny_model": True,
            "device": "cpu",
            "capture_layers": "0,1",
            "no_upload": True,
            "num_shards": 4,
            "shard_index": 0,
            "shard_size": 3,
            "pilot_cap": 6,
            "manifest_from_hf": False,
            "out_dir": ns.out_dir,
        }
    )
    rc = run_capture(cargs)
    assert rc == 0
    chunk0 = sorted((ns.out_dir / "shards").glob("shard00_chunk*.pt"))
    assert chunk0, "capture smoke wrote no chunks"
    bundle = torch.load(chunk0[0], weights_only=False)
    for fld in ("px_last", "cx_last", "v_x"):
        assert bundle[fld].shape[1] == 2, (fld, bundle[fld].shape)
    assert bundle["depth"] and bundle["corpus"] and bundle["response"], "chunk text fields missing"
    assert not torch.allclose(bundle["px_last"], bundle["cx_last"]), "px == cx (indexing bug)"
    pm = json.loads((ns.out_dir / "pilot_meta.json").read_text())
    assert pm["gate_g1"]["violation_rate_ok"], pm
    assert pm["prefix_min_pairwise_cos_by_layer"], pm

    # violation branch: a context whose prefix is NOT a token-prefix is skipped +
    # recorded (forced by monkeypatching the prefix render to drift).
    tok, hf = _load_models(cargs)
    row = next(r for r in pool if r["depth"] >= 2)
    cap, reason = _capture_context_and_prefix(hf, tok, row["messages"], [0, 1])
    assert reason == "ok" and cap["prefix_len"] < cap["prompt_len"], (reason, cap and cap.keys())
    import unittest.mock as mock

    orig = _render_messages

    def _drift(tok_, messages, *, add_generation_prompt):
        txt = orig(tok_, messages, add_generation_prompt=add_generation_prompt)
        return txt + " DRIFT" if not add_generation_prompt else txt

    with mock.patch.object(sys.modules[__name__], "_render_messages", _drift):
        cap2, reason2 = _capture_context_and_prefix(hf, tok, row["messages"], [0, 1])
    assert cap2 is None and reason2 == "prefix_token_mismatch", reason2

    # (6) kresample path on the tiny model (stub generation, 2 contexts, K=2).
    sub_doc = ns.out_dir / "kresample_subsample.json"
    sub_cis = [int(pool[0]["i"]), int(pool[1]["i"])]
    N1M._atomic_write_json(
        sub_doc, {"ci": sub_cis, "seed": 173801, "sha256": _sha_int_list(sub_cis)}
    )
    kargs = argparse.Namespace(
        **{
            **vars(cargs),
            "kresample_subsample": str(sub_doc),
            "seeds": "43,44",
            "num_shards": 1,
            "shard_index": 0,
            "pilot_cap": 0,
        }
    )
    # degenerate probe (Minor-6 gate): a tampered subsample sha fails loud
    # BEFORE any model work.
    bad_doc = ns.out_dir / "kresample_subsample_bad.json"
    N1M._atomic_write_json(bad_doc, {"ci": sub_cis, "seed": 173801, "sha256": "deadbeef"})
    try:
        run_kresample(argparse.Namespace(**{**vars(kargs), "kresample_subsample": str(bad_doc)}))
        raise AssertionError("kresample subsample sha gate did not refuse")
    except SystemExit as e:
        assert "sha mismatch" in str(e.code), e.code
    rc = run_kresample(kargs)
    assert rc == 0
    kb = torch.load(ns.out_dir / "kresample" / "kresample_shard00.pt", weights_only=False)
    assert kb["V"].shape[:2] == (2, 2) and kb["seeds"] == [43, 44], kb["V"].shape

    logger.info("[smoke] OK — manifest/allocation/carve/gate/capture-indexing/kresample")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Issue #1738 multi-turn manifest + generate + capture."
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-target", type=int, default=N_TARGET)
    ap.add_argument("--num-shards", type=int, default=32)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=500, help="contexts per capture sub-chunk")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "issue_1738" / "mt100k")
    # UPLOAD_PREFIX_EXEMPT: this driver IS issue 1738's own store writer; a child issue reusing it must pass --hf-prefix explicitly (artifact-reuse check (i) covers reuse-time threading)
    ap.add_argument("--hf-prefix", default=HF_PREFIX)
    ap.add_argument("--capture-layers", default=",".join(str(x) for x in CAPTURE_LAYERS))
    ap.add_argument("--no-upload", action="store_true", help="stage locally, do NOT upload/purge")
    ap.add_argument(
        "--no-resume-stream", action="store_true", help="fresh corpus stream (no cache)"
    )
    ap.add_argument("--build-sampling-manifest", action="store_true")
    ap.add_argument("--manifest-from-hf", action="store_true")
    ap.add_argument("--probe-rows", type=int, default=2000, help="tiny-real probe scan cap/corpus")
    ap.add_argument("--probe-only", action="store_true", help="run the bounded probe, then exit")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument(
        "--max-scan-rows", type=int, default=None, help="cap TOTAL streamed rows (smoke)"
    )
    ap.add_argument("--near-dupe-impl", choices=["exact", "minhash", "auto"], default="auto")
    ap.add_argument(
        "--screen-procs",
        type=int,
        default=0,
        help="near-dupe screen worker processes (0 = min(8, cpu_count))",
    )
    ap.add_argument("--pilot-cap", type=int, default=0, help="G1 pilot: cap this shard's contexts")
    ap.add_argument("--kresample", action="store_true", help="K-resample fresh-draw capture mode")
    ap.add_argument("--kresample-subsample", default="", help="path to kresample_subsample.json")
    ap.add_argument("--seeds", default=",".join(str(s) for s in KRESAMPLE_SEEDS))
    ap.add_argument(
        "--tiny-model", action="store_true", help="SMOKE ONLY: 2-layer from-config model"
    )
    ap.add_argument("--smoke", action="store_true", help="CPU logic smoke (synthetic streams)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        rc = _smoke(args)
    elif args.probe_only:
        run_probe(args)
        rc = 0
    elif args.build_sampling_manifest:
        build_manifest(args)
        rc = 0
    elif args.kresample:
        rc = run_kresample(args)
    else:
        rc = run_capture(args)
    # heavy C-extension entrypoint: explicit exit dodges the finalize-time
    # PyGILState_Release atexit race (#1689 gotcha).
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)


if __name__ == "__main__":
    main()
