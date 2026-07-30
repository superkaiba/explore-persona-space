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

# ── bare-query arm constants (follow-up `bare-query`, plan §4.1) ──────────────────
BARE_BATCH_DEFAULT = 64  # right-padded forward batch (plan §11: ungrounded — pilot-gated)
BARE_CHUNK_DEFAULT = 2_000  # rows per output chunk (~86 MB fp32 at 3×3584)
BARE_FENCE_MIN_DEFAULT = 45.0  # G-B1 per-shard projected-wall fence (minutes)
BARE_PARITY_MIN_COS = 0.999  # G-B1: batched-vs-per-row per-layer cosine floor
RC_BARE_FENCE = 25  # G-B1 designed-halt rc (report written first; never a bare rc=1)
# unique sentinel for the one-time empty-system template split (never appears in
# a chat template); the split is what makes assert (b) STRING-level, not token-count
_BARE_SENTINEL = "EPMBARESENTINEL1738Q"

# ── crossed-multiturn-averaged round constants (follow-up fu3, plan v9 §4) ────────
CROSSED_HF_PREFIX = "issue1738_crossed"  # all fu3 writes ride this prefix (never parent)
# LOCAL dir name for the crossed manifest — deliberately NOT MANIFEST_SUBDIR: the
# builder READS the main manifest at {out_dir}/sampling_manifest and must never
# clobber it; the REMOTE path stays {crossed_hf_prefix}/sampling_manifest (the
# name is applied inside _upload_manifest/_download_manifest).
CROSSED_MANIFEST_LOCAL = "crossed_manifest"
CROSSED_N_PREFIXES = 5_000  # scope-approved: prefixes = averaged-arm sample size (never cut)
CROSSED_N_QUERIES = 20  # shared bank width (G1 ladder 20->12->10 via --queries-per-prefix)
CROSSED_G0_PREFIX_FLOOR = 5_500  # G0: eligible prefix pool floor (take-what-exists below)
CROSSED_G0_BANK_FLOOR = 200  # G0: bank candidate pool floor
CROSSED_BANK_QUERY_TOKEN_MAX = 256  # bank supply-side eligibility cap (meta-recorded;
# ungrounded per plan — a supply choice, not a measurement knob; G0 floor guards the pool)
CROSSED_SPLIT_PREFIXES = {"val": 20, "test": 50, "holdout": 500}  # train = remainder (4,430)
CROSSED_SEED = 42  # plan §10: draw/split seed
CROSSED_FENCE_GPU_H = 24.0  # G1: projected total past this => designed halt (query ladder)
CROSSED_S2_BOOK_GPU_H = 2.5  # plan §9 S2 book, added to the G1 projection
RC_CROSSED_FENCE = 28  # G1 designed-halt rc (report written first; never a bare rc=1)
RC_CROSSED_VIOLATION = 29  # fleet halt: strict-token-prefix violations > 0.5% (plan §4.2)
RC_CROSSED_G2 = 30  # G2 crossing-sanity FAIL (min pairwise px cos >= 0.999) — code bug halt
CROSSED_VIOLATION_MIN_ATTEMPTED = 1_000  # rate read is meaningless below this floor
CROSSED_SAE_LAYER = 19  # SAE fold-in layer (Source: #1482 via sae-arm round)
CROSSED_SAE_K = 64
CROSSED_SAE_FVE_MIN = 0.75  # plan §12.4 inherited bar (sae-arm G-S0)
CROSSED_SAE_L0_RANGE = (30.0, 120.0)
CROSSED_SAE_POOL_CAP = 150_000  # fitness token-pool cap (sae-arm _FvePool convention)
BARE_BANK_SUBDIR = "bare_bank"  # 20-query bare-arm capture destination under the prefix
CROSSED_PILOT_META_NAME = "crossed_pilot_meta.json"

# Engine-cap invariant (r5 crash fix): a budget-admitted prompt + max generation
# always fits the engine, so gating admission at PROMPT_TOKEN_BUDGET is what
# keeps llm_engine.add_request from ever seeing > MAX_MODEL_LEN. The attempt-3
# kresample crash was an UNGATED 14,217-token subsample row vs max_model_len
# 8,192 — engine-fatal instead of skip+record.
assert PROMPT_TOKEN_BUDGET + GEN_MAX_TOKENS <= MAX_MODEL_LEN, (
    PROMPT_TOKEN_BUDGET,
    GEN_MAX_TOKENS,
    MAX_MODEL_LEN,
)


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


def _gen_render_len(tok):
    """Token-length fn of the EXACT generation render (add_generation_prompt=True,
    tokenized with add_special_tokens=False) — the ONE budget arithmetic shared by
    the capture over-length filter and the kresample admission gate (r5 fix: a
    single source, so a K-draw's conditioning context can never be budgeted under
    a different convention than the primary draw's)."""

    def _n_tokens(messages: list[dict]) -> int:
        return len(
            tok(
                _render_messages(tok, messages, add_generation_prompt=True),
                add_special_tokens=False,
            )["input_ids"]
        )

    return _n_tokens


def _kresample_admission_gate(
    rows: list[dict], tok_len_fn, primary_cis: set[int] | None
) -> tuple[list[dict], list[dict]]:
    """Capture-parity admission for K-resample rows (the r5 crash fix).

    The subsample is drawn from the SPLIT-time holdout ci list, which predates
    Phase-2 capture — it can contain rows capture SKIPPED as over-length (5 of
    2,000; ci 98764 renders to 14,217 tok > max_model_len 8,192 and killed the
    engine at add_request) and rows with NO primary seed-42 draw in y_holdout
    (7 of 2,000; the pilot-era partial-chunk gap), which would trip the floor
    phase's designed rc-23 join gate. A row is admitted iff (a) its EXACT
    generation render fits PROMPT_TOKEN_BUDGET and (b) it has a primary draw
    when ``primary_cis`` is provided. Skips are RECORDED (ci + n_tokens +
    reasons; never text) — never engine-fatal. Returns (kept, skipped)."""
    kept, skipped = [], []
    for r in rows:
        ci = int(r["i"])
        n_tok = int(tok_len_fn(r["messages"]))
        reasons = []
        if n_tok > PROMPT_TOKEN_BUDGET:
            reasons.append("overlength")
        if primary_cis is not None and ci not in primary_cis:
            reasons.append("no_primary_draw")
        if reasons:
            skipped.append({"ci": ci, "n_tokens": n_tok, "reasons": reasons})
        else:
            kept.append(r)
    return kept, skipped


def _kresample_primary_cis(args) -> set[int] | None:
    """Resolve the primary-draw ci set (y_holdout membership) for the kresample
    admission gate. ``--kresample-primary-ci``: 'hf' (default) stages
    {hf_prefix}/analysis_tensors/y_holdout/L14.npz from the data repo and reads
    its 'ci' array (identical across L14/L19/L26 — verified 2026-07-28); a local
    path reads an .npz with a 'ci' array or a JSON list / {"ci": [...]} doc;
    'none' DISABLES the gate (loud warning — deliberate override only)."""
    spec = str(args.kresample_primary_ci)
    if spec == "none":
        logger.warning(
            "[kresample] primary-draw gate DISABLED (--kresample-primary-ci none) — "
            "contexts without a seed-42 primary draw will be generated and later "
            "fail the floor phase's rc-23 join gate"
        )
        return None
    if spec == "hf":
        dest = args.out_dir / "kresample" / "y_holdout_L14.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            hub.stage_hub_file(
                C.HF_DATA_REPO,
                f"{args.hf_prefix}/analysis_tensors/y_holdout/L14.npz",
                dest,
                repo_type="dataset",
            )
        path = dest
    else:
        path = Path(spec)
    if path.suffix == ".npz":
        cis = {int(c) for c in np.load(path)["ci"]}
    else:
        doc = json.loads(path.read_text())
        cis = {int(c) for c in (doc["ci"] if isinstance(doc, dict) else doc)}
    assert cis, f"primary-draw ci set from {path} is empty"
    logger.info("[kresample] primary-draw gate ON: %d cis from %s", len(cis), path.name)
    return cis


def _write_kresample_skipped(scratch: Path, args, skipped: list[dict]) -> None:
    """Kresample admission-gate record (capture-sidecar sibling): over-length +
    no-primary-draw skips, ci + token counts + reasons only (never text).
    Written + uploaded BEFORE generation so the record survives a later crash."""
    skip_name = f"kresample_shard{args.shard_index:02d}_skipped.json"
    C.write_json_atomic(
        scratch / skip_name,
        {
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
            "prompt_token_budget": PROMPT_TOKEN_BUDGET,
            "primary_gate": str(args.kresample_primary_ci) != "none",
            "n_skipped": len(skipped),
            "skipped": skipped,
        },
    )
    if args.no_upload:
        return
    url = hub._upload_folder_filtered(
        scratch,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{args.hf_prefix}/{KRESAMPLE_SUBDIR}",
        allow_patterns=[skip_name],
        expected_repo_paths=[f"{args.hf_prefix}/{KRESAMPLE_SUBDIR}/{skip_name}"],
    )
    if not url:
        raise RuntimeError(f"kresample skipped-sidecar upload of {skip_name} returned no URL")


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


# ── bare-query arm: empty-system render + batched forward-only capture ────────────
# (follow-up `bare-query`, plan §4.1; #1092 _render_bare_query_empty_system convention)


def _bare_template_parts(tok) -> tuple[str, str]:
    """Split the empty-system chat-template render around a unique sentinel query
    ONCE per process → (TEMPLATE_PREFIX, TEMPLATE_SUFFIX). The per-row identity
    assert is then STRING-level (`text == PREFIX + q + SUFFIX`) — BPE-seam-immune
    (plan §4.1.1: a leading-newline query merges across the template's `user\\n`
    seam, shifting token overhead 13→12, so a token-count identity false-fires)."""
    probe = tok.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "user", "content": _BARE_SENTINEL}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert probe.count(_BARE_SENTINEL) == 1, (
        f"bare template split failed: sentinel appears {probe.count(_BARE_SENTINEL)}x"
    )
    pre, _, suf = probe.partition(_BARE_SENTINEL)
    return pre, suf


def _render_bare_query(tok, q: str, parts: tuple[str, str]) -> str:
    """Render the FINAL user query with an explicit EMPTY system turn. Per-row
    HARD asserts (plan §4.1.1): (a) no injected Qwen default system prompt;
    (b) no-prefix-content identity at STRING level."""
    text = tok.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "user", "content": q}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if "You are Qwen" in text:  # (a) default-system injection
        raise RuntimeError("bare render carries the Qwen default system prompt (assert a)")
    if text != parts[0] + q + parts[1]:  # (b) exact-concatenation identity
        raise RuntimeError(
            "bare render != TEMPLATE_PREFIX + query + TEMPLATE_SUFFIX (assert b) — "
            "prefix-content / extra-turn leakage or a template content transform"
        )
    return text


def _bare_render_selftest(tok) -> tuple[str, str]:
    """B0 in-process self-test (plan §4.1.1): assert (b) PASSES on a probe query
    beginning with a literal newline (the BPE-seam case) and FAILS on a render
    with an injected extra turn; assert (a) has teeth (a system-less render on
    this tokenizer injects the Qwen default system prompt). Returns the parts."""
    parts = _bare_template_parts(tok)
    # (i) seam case: leading-"\n" query must PASS the string identity.
    _render_bare_query(tok, "\nWhat is the capital of France?", parts)
    # (ii) injected extra turn must FAIL the identity.
    q = "plain probe question"
    injected = tok.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": "an earlier turn"},
            {"role": "assistant", "content": "an earlier answer"},
            {"role": "user", "content": q},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert injected != parts[0] + q + parts[1], (
        "B0: injected-extra-turn render passed the string identity — assert (b) is toothless"
    )
    # (iii) assert (a) has teeth: messages WITHOUT a system turn inject the default.
    bare_default = tok.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
    )
    assert "You are Qwen" in bare_default, (
        "B0: system-less render does NOT inject the Qwen default system prompt — "
        "the empty-system convention needs re-derivation for this tokenizer"
    )
    logger.info("[bare-b0] render self-test OK (seam identity + injected-turn refusal)")
    return parts


def _capture_bare_batch(hf, tok, texts: list[str], layers: list[int], batch: int) -> torch.Tensor:
    """Right-padded batched teacher-forced forwards; last-REAL-token states at
    ``layers`` → (n, L, H) fp32 cpu. Right padding is causal-safe (later
    positions cannot influence earlier ones) and keeps real tokens at positions
    0..len-1, so no position_ids threading is needed; batching numerics are
    gated by the G-B1 parity probe (per-layer cos >= BARE_PARITY_MIN_COS)."""
    assert tok.pad_token_id is not None, "tokenizer has no pad token (right-padded batches)"
    outs = []
    prev_side = tok.padding_side
    tok.padding_side = "right"
    try:
        for s in range(0, len(texts), batch):
            enc = tok(texts[s : s + batch], return_tensors="pt", padding=True)
            ids = enc["input_ids"].to(hf.device)
            mask = enc["attention_mask"].to(hf.device)
            captured = extract_layer_activations(hf, ids, layers, attention_mask=mask)
            last = mask.sum(dim=1) - 1  # (B,) last real index per row
            b_idx = torch.arange(ids.shape[0])
            per_layer = []
            for li in layers:
                hs = captured[li]  # (B, T, H) on device
                per_layer.append(hs[b_idx.to(hs.device), last.to(hs.device), :].float().cpu())
            outs.append(torch.stack(per_layer, dim=1))  # (B, L, H)
    finally:
        tok.padding_side = prev_side
    return torch.cat(outs, dim=0)


def _bare_parity_probe(hf, tok, texts: list[str], layers: list[int], batch: int) -> dict:
    """G-B1 32-row parity probe: batched right-padded capture vs per-row unpadded
    forwards (the parent's per-row convention), per-layer MIN cosine. A miss is a
    capture-code bug — fail loud before the fleet proceeds (plan §7 G-B1)."""
    bq = _capture_bare_batch(hf, tok, texts, layers, batch)
    refs = []
    for t in texts:
        ids = tok(t, return_tensors="pt", padding=False)["input_ids"].to(hf.device)
        cap = extract_layer_activations(hf, ids, layers)
        refs.append(torch.stack([cap[li][0, -1, :].float().cpu() for li in layers]))
    ref = torch.stack(refs)  # (n, L, H)
    out: dict[str, float] = {}
    for k, li in enumerate(layers):
        cos = torch.nn.functional.cosine_similarity(bq[:, k, :].double(), ref[:, k, :].double(), 1)
        out[str(li)] = float(cos.min().item())
    return out


def _bare_fence_should_halt(
    elapsed_s: float, fresh_rows: int, pending_rows_total: int, fence_min: float
) -> bool:
    """G-B1 pure fence predicate: halt when the measured rate projects THIS
    shard's pending rows past ``fence_min`` minutes (plan §7: re-size, never a
    silent overrun)."""
    if fresh_rows <= 0:
        return False
    return (elapsed_s / fresh_rows) * pending_rows_total / 60.0 > fence_min


def _stack_chunk_bare(rows: list[dict], layers, shard_index: int, chunk_idx: int) -> dict:
    """Bare chunk bundle (plan §4.1.4): bq_last (n, L, H) fp32 + ci + renders."""
    return {
        "bq_last": torch.stack([r["bq_last"] for r in rows]),
        "ci": [int(r["ci"]) for r in rows],
        "bare_render": [r["bare_render"] for r in rows],
        "layers": list(layers),
        "shard_index": int(shard_index),
        "chunk": int(chunk_idx),
    }


def run_bare_capture(args) -> int:
    """Plan §4.1 ``--bare-query`` mode: forward-only BATCHED capture of the FINAL
    user query rendered with an explicit EMPTY system turn (#1092 convention).
    ALL manifest rows (parent over-length skips included — bare renders are
    short); no generation, no vLLM. Uploads ride ``--upload-prefix`` (default =
    ``--hf-prefix``) so the parent capture prefix is never clobbered."""
    # UPLOAD_PREFIX_EXEMPT: default = this issue's own --hf-prefix (issue1738_multiturn); the bare round passes --upload-prefix so the parent capture prefix is never written (plan v6 §4.1.4)
    up = args.upload_prefix or args.hf_prefix
    manifest_dir = _resolve_manifest_dir(args)  # manifest rides --hf-prefix (parent)
    pool, _meta = N1M.read_manifest_pool(manifest_dir)
    n_total = len(pool)
    start, end = N50._shard_range(n_total, args.num_shards, args.shard_index)
    shard_pool = pool[start:end]
    if args.pilot_cap and args.pilot_cap > 0:
        shard_pool = shard_pool[: args.pilot_cap]
    layers = [int(x) for x in args.capture_layers.split(",")]
    logger.info(
        "[bare shard %d/%d] range [%d, %d) = %d contexts (%d total)%s -> upload %s",
        args.shard_index,
        args.num_shards,
        start,
        end,
        len(shard_pool),
        n_total,
        f" PILOT cap {args.pilot_cap}" if args.pilot_cap else "",
        up,
    )

    C.phase("load_model")
    tok, hf = _load_models(args)
    C.phase("bare_selftest")
    parts = _bare_render_selftest(tok)  # B0 (hard asserts; plan §4.1.1)
    if not shard_pool:
        logger.info("[bare shard %d] empty range; nothing to do", args.shard_index)
        C.phase("done")
        return 0

    scratch = args.out_dir / "bare_shards"
    scratch.mkdir(parents=True, exist_ok=True)
    done_pt = set(N50._remote_index(f"{up}/{CAPTURE_SUBDIR}")) if not args.no_upload else set()

    C.phase("bare_capture")

    # render + over-length filter + deterministic sort over the WHOLE shard
    # (sorted by rendered token length -> near-uniform right-padded batches;
    # (n_tok, ci) tie-break keeps chunk composition deterministic for resume).
    _len_cache: dict[str, int] = {}  # memoized — the filter + the sort key share it

    def _bare_tok_len(messages: list[dict]) -> int:
        q = messages[-1]["content"]
        n = _len_cache.get(q)
        if n is None:
            n = len(tok(parts[0] + q + parts[1], add_special_tokens=False)["input_ids"])
            _len_cache[q] = n
        return n

    kept_rows, skipped_all = _filter_rows_overlength(shard_pool, _bare_tok_len, PROMPT_TOKEN_BUDGET)
    if skipped_all:  # expected 0 (plan §12 assumption 5) — recorded, never silent
        logger.warning(
            "[bare shard %d] %d over-length bare renders skipped (> %d tok); cis=%s",
            args.shard_index,
            len(skipped_all),
            PROMPT_TOKEN_BUDGET,
            [d["ci"] for d in skipped_all][:20],
        )
    kept_rows = sorted(kept_rows, key=lambda r: (_bare_tok_len(r["messages"]), int(r["i"])))
    n_sub = (len(kept_rows) + args.bare_chunk - 1) // args.bare_chunk
    chunk_specs = []  # (chunk_idx, name, rows, resumed)
    pending_rows_total = 0
    for ci_idx, s in enumerate(range(0, len(kept_rows), args.bare_chunk)):
        name = f"shard{args.shard_index:02d}_chunk{ci_idx:04d}.pt"
        rows = kept_rows[s : s + args.bare_chunk]
        resumed = name in done_pt
        if not resumed:
            pending_rows_total += len(rows)
        chunk_specs.append((ci_idx, name, rows, resumed))

    kept_total = 0
    fresh_rows = 0
    t_start = time.time()
    pending_pt: list[str] = []
    parity: dict | None = None
    pilot_written = False

    def _flush_pending() -> None:
        if args.no_upload or not pending_pt:
            return
        _flush_upload_batch_mt(scratch, up, pending_pt, [])
        pending_pt.clear()

    def _write_bare_pilot_meta() -> None:
        wall_h = (time.time() - t_start) / 3600.0
        rate = fresh_rows / wall_h if wall_h > 0 else float("nan")
        doc = {
            "n_pilot_rows": int(fresh_rows),
            "wall_h": float(wall_h),
            "rows_per_gpu_h": float(rate),
            "pending_rows_this_shard": int(pending_rows_total),
            "projected_shard_wall_min": float(
                (time.time() - t_start) / max(1, fresh_rows) * pending_rows_total / 60.0
            ),
            "bare_fence_min": float(args.bare_fence_min),
            # hard-assert semantics (plan §4.1.1): any render violation raises
            # per row, so reaching this write PROVES the count is 0.
            "n_render_violations": 0,
            "n_overlength_skipped": len(skipped_all),
            "parity_min_cos_by_layer": parity,
            "parity_min_cos_floor": BARE_PARITY_MIN_COS,
            "parity_ok": parity is not None
            and all(v >= BARE_PARITY_MIN_COS for v in parity.values()),
            "bare_batch": int(args.bare_batch),
            "bare_chunk": int(args.bare_chunk),
            # worked-example render, verbatim (plan §4.1.1) — a SYNTHETIC probe
            # query, never corpus text (refusal-safety: module docstring).
            "worked_example_render": _render_bare_query(
                tok, "What is the capital of France?", parts
            ),
            "shard_index": int(args.shard_index),
            "pilot_cap": int(args.pilot_cap),
        }
        out = args.out_dir / "bare_pilot_meta.json"
        C.write_json_atomic(out, doc)
        logger.info(
            "[bare-pilot] rows/GPU-h=%.0f projected_shard_wall=%.1f min parity=%s",
            rate,
            doc["projected_shard_wall_min"],
            parity,
        )
        if not args.no_upload:
            # upload_as_file=True uses path_in_repo as the FULL FILE destination
            # (hub._upload file branch: `path_in_repo or local_path.name`) — passing
            # the bare prefix here created a FILE at `<prefix>` that 400-blocked
            # every later `<prefix>/capture/*` chunk commit (fu1 B1 incident).
            url = hub._upload(
                out,
                repo_id=C.HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=f"{up}/{CAPTURE_SUBDIR}/bare_pilot_meta.json",
                upload_as_file=True,
            )
            if not url:
                raise RuntimeError("bare_pilot_meta.json upload returned no URL")

    def _on_sigterm(signum, frame):
        raise SystemExit(f"SIGTERM ({signum}) received — flushing pending upload batch")

    prev_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        for ci_idx, name, rows, resumed in chunk_specs:
            if resumed:
                logger.info(
                    "[bare shard %d] chunk %d/%d already on Hub; skip",
                    args.shard_index,
                    ci_idx + 1,
                    n_sub,
                )
                continue
            ts = time.time()
            texts = [_render_bare_query(tok, r["messages"][-1]["content"], parts) for r in rows]
            if parity is None:  # G-B1 32-row parity probe, first fresh chunk
                probe_n = min(32, len(texts))
                parity = _bare_parity_probe(hf, tok, texts[:probe_n], layers, args.bare_batch)
                bad = {k: v for k, v in parity.items() if v < BARE_PARITY_MIN_COS}
                if bad:
                    raise RuntimeError(
                        f"G-B1 parity probe FAILED (min cos < {BARE_PARITY_MIN_COS}): {parity} — "
                        "batched capture != per-row convention; fix before the fleet proceeds"
                    )
                logger.info("[bare-parity] %d rows, per-layer min cos %s", probe_n, parity)
            bq = _capture_bare_batch(hf, tok, texts, layers, args.bare_batch)
            h_dim = int(hf.config.hidden_size)
            assert bq.shape == (len(rows), len(layers), h_dim), bq.shape
            chunk_rows = [
                {"ci": int(r["i"]), "bare_render": t, "bq_last": bq[j]}
                for j, (r, t) in enumerate(zip(rows, texts, strict=True))
            ]
            torch.save(
                _stack_chunk_bare(chunk_rows, layers, args.shard_index, ci_idx), scratch / name
            )
            kept_total += len(rows)
            fresh_rows += len(rows)
            if not args.no_upload:
                pending_pt.append(name)
                if len(pending_pt) >= UPLOAD_BATCH:
                    _flush_pending()
            logger.info(
                "[bare-capture] chunk %d/%d shard=%d: %d rows (%.0fs elapsed=%.0fs)",
                ci_idx + 1,
                n_sub,
                args.shard_index,
                len(rows),
                time.time() - ts,
                time.time() - t_start,
            )
            if not pilot_written:
                # G-B1 in-run gate after the FIRST fresh chunk (plan §7): pilot
                # meta (shard 0 / pilot dispatch ONLY — on an 8-wide pod all
                # shards share out_dir + the HF meta path, so one writer) + the
                # per-shard projected-wall fence (designed halt, every shard).
                if args.shard_index == 0 or args.pilot_cap:
                    _write_bare_pilot_meta()
                pilot_written = True
                if _bare_fence_should_halt(
                    time.time() - t_start, fresh_rows, pending_rows_total, args.bare_fence_min
                ):
                    rep = {
                        "gate": "G-B1",
                        "shard_index": int(args.shard_index),
                        "fresh_rows": int(fresh_rows),
                        "pending_rows_this_shard": int(pending_rows_total),
                        "elapsed_s": time.time() - t_start,
                        "projected_shard_wall_min": (time.time() - t_start)
                        / max(1, fresh_rows)
                        * pending_rows_total
                        / 60.0,
                        "bare_fence_min": float(args.bare_fence_min),
                    }
                    C.write_json_atomic(args.out_dir / "bare_fence_report.json", rep)
                    logger.error("[G-B1] bare fence tripped: %s", rep)
                    _flush_pending()  # persist completed chunks before the halt
                    if not args.no_upload:
                        try:  # artifact-first halt routing: rc survives an upload failure
                            # UPLOAD_LOOP_EXEMPT: single bare_fence_report.json uploaded ONCE at the rc-25 fence halt — # UPLOAD_PREFIX_EXEMPT: dest defaults to this issue's own --hf-prefix (issue1738_multiturn); bare arm passes an explicit prefix (plan v6 §4.1.5)
                            url = hub._upload(
                                args.out_dir / "bare_fence_report.json",
                                repo_id=C.HF_DATA_REPO,
                                repo_type="dataset",
                                path_in_repo=up,
                                upload_as_file=True,
                            )
                            if not url:
                                logger.error("[G-B1] fence-report upload returned no URL")
                        except Exception:
                            logger.exception("[G-B1] fence-report upload failed (rc kept)")
                    sys.exit(RC_BARE_FENCE)
        _flush_pending()
    except BaseException:
        try:
            _flush_pending()
        except Exception:
            logger.exception(
                "[bare shard %d] best-effort pending-batch flush failed on exit",
                args.shard_index,
            )
        raise
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)

    _write_sidecar(scratch, args, skipped_all, [], prefix=up)
    wall_h = (time.time() - t_start) / 3600.0
    logger.info(
        "[bare shard %d] done: %d rows across %d chunks (%d over-length skips, %.2f h)",
        args.shard_index,
        kept_total,
        n_sub,
        len(skipped_all),
        wall_h,
    )
    C.phase("done")
    return 0


# ── crossed-multiturn-averaged round (follow-up fu3, plan v9 §4.1/§4.2) ───────────
# P0: build_crossed_manifest — 5,000 main-manifest prefixes × 20 shared real bank
# queries = 100,000 contexts, prefix-grouped 4-way split in the PARENT schema.
# S1: run_crossed_capture — shard BY PREFIX, one on-policy answer per context,
# px/cx/v_x at {14,19,26} + SAE L19 fold-in on already-materialized states.


def _crossed_split_prefix_targets(n_prefixes: int) -> dict[str, int]:
    """Prefix-count split targets: production 4,430/20/50/500 at P=5,000; tiny
    pools (smoke / G0 shortfall) scale to 1/1/1 + the remainder as train."""
    carve = dict(CROSSED_SPLIT_PREFIXES)
    if n_prefixes < sum(carve.values()) + 2:
        assert n_prefixes >= 5, f"crossed pool too small to split: {n_prefixes} prefixes"
        carve = {"val": 1, "test": 1, "holdout": 1}
        logger.warning("[crossed] tiny prefix pool %d — scaled carve to %s", n_prefixes, carve)
    return carve


def _q_norm(q: str) -> str:
    """Normalized query text — the bank near-dupe/dedup surface (render-key style)."""
    return " ".join(q.lower().split())


def _crossed_bank_candidates(pool: list[dict], tok, args) -> tuple[list[dict], dict]:
    """Seeded walk of the main-manifest pool -> >= bank-floor length-gated,
    mutually-non-near-dupe last-user-turn candidates (the #1092 bank
    construction). Returns (candidates [{q, source_ci, n_tokens}], stats)."""
    rng = np.random.default_rng(CROSSED_SEED)
    perm = rng.permutation(len(pool))
    cand: list[dict] = []
    n_len_reject = n_dupe_reject = 0
    for j in perm:
        r = pool[int(j)]
        q = r["messages"][-1]["content"]
        n_tok = len(tok(q, add_special_tokens=False)["input_ids"])
        if n_tok > args.crossed_bank_query_token_max:
            n_len_reject += 1
            continue
        if cand and _make_gate([_q_norm(c["q"]) for c in cand], "minhash").is_dupe(_q_norm(q)):
            n_dupe_reject += 1
            continue
        cand.append({"q": q, "source_ci": int(r["i"]), "n_tokens": n_tok})
        if len(cand) >= args.crossed_g0_bank_floor:
            break
    stats = {
        "n_candidates": len(cand),
        "bank_pool_floor": int(args.crossed_g0_bank_floor),
        "bank_pool_ok": len(cand) >= args.crossed_g0_bank_floor,
        "n_len_rejects": n_len_reject,
        "n_within_bank_dupe_rejects": n_dupe_reject,
        "bank_query_token_max": int(args.crossed_bank_query_token_max),
    }
    return cand, stats


def _bank_sha(bank: list[str]) -> str:
    return hashlib.sha256("\x00".join(bank).encode("utf-8")).hexdigest()


def build_crossed_manifest(args) -> dict:
    """P0 (plan v9 §4.1): crossed manifest = 5,000 prefixes (main-manifest rows,
    prefix_id = main ci) x 20-query shared bank, fully crossed by construction
    (row ci = prefix_pos * n_queries + q). Writes {meta.json, part_*.jsonl (the
    PREFIX rows), bank.json, split_1738_crossed.json} and uploads the folder to
    {crossed_hf_prefix}/sampling_manifest/ (the --manifest-from-hf loader's
    meta.json contract holds verbatim)."""
    from transformers import AutoTokenizer

    C.phase("crossed-manifest")
    main_dir = _resolve_manifest_dir(args)  # main manifest rides --hf-prefix (parent)
    pool, main_meta = N1M.read_manifest_pool(main_dir)
    tok = AutoTokenizer.from_pretrained(args.model)
    n_q = int(args.crossed_n_queries)

    # 1. bank candidate pool (G0 bank floor; length + within-bank near-dupe gates).
    cand, bank_stats = _crossed_bank_candidates(pool, tok, args)
    assert cand, "crossed bank: zero eligible candidates (length gate rejects everything?)"
    if not bank_stats["bank_pool_ok"]:
        logger.warning(
            "[crossed G0] bank pool %d < floor %d — take-what-exists",
            bank_stats["n_candidates"],
            bank_stats["bank_pool_floor"],
        )
    # LONGEST candidate bounds the length precheck so any prefix-clash replacement
    # (step 3) stays within the prechecked budget.
    longest_q = max(cand, key=lambda c: c["n_tokens"])["q"]
    bank_src_cis = {c["source_ci"] for c in cand}

    # 2. prefix eligibility walk (lazy length precheck against the longest candidate,
    #    stopping at the G0 floor — bounds tokenizer work to ~floor renders).
    rng = np.random.default_rng(CROSSED_SEED)
    perm = rng.permutation(len(pool))
    tok_len = _gen_render_len(tok)
    need = max(int(args.crossed_g0_prefix_floor), int(args.crossed_n_prefixes))
    eligible: list[int] = []
    n_pref_len_reject = 0
    n_walked = 0
    for j in perm:
        r = pool[int(j)]
        if int(r["i"]) in bank_src_cis:
            continue
        n_walked += 1
        prefix_messages = r["messages"][:-1]
        assert prefix_messages and prefix_messages[-1]["role"] == "assistant", r["i"]
        if tok_len([*prefix_messages, {"role": "user", "content": longest_q}]) > (
            PROMPT_TOKEN_BUDGET
        ):
            n_pref_len_reject += 1
            continue
        eligible.append(int(j))
        if len(eligible) >= need:
            break
    exhausted = len(eligible) < need
    prefix_pool_ok = len(eligible) >= int(args.crossed_g0_prefix_floor)
    if not prefix_pool_ok:
        logger.warning(
            "[crossed G0] eligible prefix pool %d < floor %d after exhausting the walk "
            "— take-what-exists (reported)",
            len(eligible),
            args.crossed_g0_prefix_floor,
        )
    take = min(int(args.crossed_n_prefixes), len(eligible))
    sel = eligible[:take]
    assert take >= 5, f"crossed: only {take} eligible prefixes"

    # 3. bank vs SELECTED-prefix near-dupe gate; replace trips from the candidate pool.
    gate_p = _make_gate([_render_key(pool[j]["messages"][:-1]) for j in sel], "minhash")
    bank: list[str] = []
    bank_ci: list[int] = []
    n_prefix_clash = 0
    for c in cand:
        if len(bank) >= n_q:
            break
        if gate_p.is_dupe(_q_norm(c["q"])):
            n_prefix_clash += 1
            continue
        bank.append(c["q"])
        bank_ci.append(c["source_ci"])
    assert len(bank) == min(n_q, len(cand) - n_prefix_clash), (len(bank), n_q)
    if len(bank) < n_q:
        logger.warning("[crossed G0] bank realized %d < %d — take-what-exists", len(bank), n_q)
    n_q = len(bank)

    # 4. prefix-grouped split (seeded permutation; all n_q rows of a prefix one side).
    carve = _crossed_split_prefix_targets(take)
    sperm = np.random.default_rng(CROSSED_SEED + 1).permutation(take)
    cursor = 0
    prefix_sets: dict[str, list[int]] = {}
    for name in ("val", "test", "holdout"):
        prefix_sets[name] = sorted(int(x) for x in sperm[cursor : cursor + carve[name]])
        cursor += carve[name]
    prefix_sets["train"] = sorted(int(x) for x in sperm[cursor:])
    split_doc: dict = {
        "seed": CROSSED_SEED,
        "n_manifest": take * n_q,
        "recipe_version": "crossed-mt-v1",
        "grid": {"n_prefixes": take, "n_queries": n_q},
        "sets": {},
        "prefix_sets": {},
        "transfer_descoped": False,
    }
    for name, pset in prefix_sets.items():
        rows = sorted(p * n_q + q for p in pset for q in range(n_q))
        split_doc["sets"][name] = {"ci": rows, "n": len(rows), "sha256": _sha_int_list(rows)}
        split_doc["prefix_sets"][name] = {
            "pi": pset,
            "n": len(pset),
            "sha256": _sha_int_list(pset),
        }

    # 5. prefix rows (the manifest parts) + meta + bank.json; upload the folder.
    prefix_rows = []
    for pos, j in enumerate(sel):
        r = pool[j]
        set_name = next(n for n, s in prefix_sets.items() if pos in set(s))
        prefix_rows.append(
            {
                "i": pos,
                "prefix_id": int(r["i"]),  # main-manifest ci — the mechanical join key
                "messages": r["messages"][:-1],  # PREFIX messages (ends assistant)
                "depth": int(r["depth"]),
                "corpus": r["corpus"],
                "split": set_name,
            }
        )
    meta = {
        "n_new": len(prefix_rows),  # read_manifest_pool contract (PREFIX rows)
        "kind": "crossed",
        "n_queries": n_q,
        "n_rows_crossed": take * n_q,
        "g0": {
            "prefix_pool_eligible_seen": len(eligible),
            "prefix_pool_floor": int(args.crossed_g0_prefix_floor),
            "prefix_pool_ok": prefix_pool_ok,
            "prefix_walk_exhausted": exhausted,
            "n_prefix_len_rejects": n_pref_len_reject,
            "n_prefixes_walked": n_walked,
            "bank": {**bank_stats, "n_prefix_clash_replaced": n_prefix_clash},
        },
        "overlap_fraction_main_manifest": 1.0,  # drawn FROM the main manifest by design
        "main_manifest_hf_prefix": args.hf_prefix,
        "main_manifest_recipe": main_meta.get("recipe_version"),
        "depth_hist": _depth_hist(prefix_rows),
        "corpus_counts": dict(Counter(r["corpus"] for r in prefix_rows)),
        "split_shas": {k: v["sha256"] for k, v in split_doc["sets"].items()},
        "prefix_split_shas": {k: v["sha256"] for k, v in split_doc["prefix_sets"].items()},
        "bank_sha256": _bank_sha(bank),  # G2 byte-identity assert surface
        "capture_layers": [int(x) for x in args.capture_layers.split(",")],
        "model": args.model,
        "prompt_token_budget": PROMPT_TOKEN_BUDGET,
        "recipe_version": "crossed-mt-v1",
        "seed": CROSSED_SEED,
    }
    crossed_dir = args.out_dir / CROSSED_MANIFEST_LOCAL
    n_parts = N1M._write_manifest_parts(crossed_dir, prefix_rows, meta)
    N1M._atomic_write_json(
        crossed_dir / "bank.json",
        {
            "queries": bank,
            "source_ci": bank_ci,
            "sha256": _bank_sha(bank),
            "n_tokens": [len(tok(q, add_special_tokens=False)["input_ids"]) for q in bank],
            **bank_stats,
        },
    )
    N1M._atomic_write_json(crossed_dir / "split_1738_crossed.json", split_doc)
    logger.info(
        "[crossed manifest] %d prefixes x %d queries = %d rows in %d parts; carve=%s",
        take,
        n_q,
        take * n_q,
        n_parts,
        {k: len(v) for k, v in prefix_sets.items()},
    )
    if not args.no_upload:
        N1M._upload_manifest(crossed_dir, args.crossed_hf_prefix)
    C.phase("crossed-manifest-done")
    return meta


def _load_crossed_manifest(args) -> tuple[list[dict], list[str], dict, dict]:
    """Resolve + load the crossed manifest (prefix rows, bank queries, split doc,
    meta). Local dir = {out_dir}/{CROSSED_MANIFEST_LOCAL}; --manifest-from-hf
    stages {crossed_hf_prefix}/sampling_manifest (bank + split ride the folder)."""
    local = args.out_dir / CROSSED_MANIFEST_LOCAL
    if args.manifest_from_hf:
        local = N1M._download_manifest(args.crossed_hf_prefix, local)
    prefix_rows, meta = N1M.read_manifest_pool(local)
    assert meta.get("kind") == "crossed", (
        f"{local} is not a crossed manifest (kind={meta.get('kind')!r})"
    )
    bank_doc = json.loads((local / "bank.json").read_text())
    bank = list(bank_doc["queries"])
    # G2 crossing-sanity half 2: the bank strings are byte-identical to the
    # manifest-time freeze (every realized row is built from THESE strings).
    got = _bank_sha(bank)
    assert got == meta["bank_sha256"], f"bank sha drift: {got} != {meta['bank_sha256']}"
    split_doc = json.loads((local / "split_1738_crossed.json").read_text())
    return prefix_rows, bank, split_doc, meta


def _crossed_rows(prefix_rows: list[dict], bank: list[str], n_q_grid: int, n_q: int):
    """Materialize crossed rows (p-major, q-minor) for the given prefix rows.
    Row ci = prefix_pos * n_q_grid + q (grid-stable under the G1 query ladder)."""
    out = []
    for r in prefix_rows:
        for q in range(n_q):
            out.append(
                {
                    "i": int(r["i"]) * n_q_grid + q,
                    "prefix_id": int(r["prefix_id"]),
                    "query_id": q,
                    "messages": [*r["messages"], {"role": "user", "content": bank[q]}],
                    "depth": int(r["depth"]),
                    "corpus": r["corpus"],
                }
            )
    return out


def _crossed_chunk_plan(
    n_rows: int, shard_size: int, shard_index: int, n_q: int, is_pilot: bool
) -> list[tuple[str, int, int]]:
    """Deterministic chunk plan for one crossed shard: [(basename, start, end)].

    r2 blocker-1 fix shape (a): the realized n_q is EMBEDDED in every chunk name
    (``shardSS_qQQ_chunkCCCC``), so a G1 query-ladder re-run at a reduced n_q can
    never name-collide with the stale family — the Hub resume skip and the reads
    scanner both key on the (shard, n_q, chunk) triple. Sibling A: a PILOT's
    trailing PARTIAL chunk (n_rows % shard_size != 0) is named ``pilotpartial``
    instead of ``chunk`` — the pilot's rows are a strict PREFIX of the fleet
    shard's rows (same prefix order, same n_q), so chunk k covers the identical
    row slice [k*ss, (k+1)*ss) in both runs for every FULL chunk (fleet-resumable
    by name), while the fleet's FULL chunk at the pilot's partial index gets a
    fresh capture instead of a silent (ss - partial)-row hole. The reads scanner
    consumes ``_qQQ_chunk`` names ONLY (pilotpartial rows are re-captured by the
    fleet; their raw text still uploads — persist-by-default)."""
    out = []
    for ci_idx, s in enumerate(range(0, n_rows, shard_size)):
        e = min(s + shard_size, n_rows)
        kind = "pilotpartial" if (is_pilot and (e - s) < shard_size) else "chunk"
        out.append((f"shard{shard_index:02d}_q{n_q:02d}_{kind}{ci_idx:04d}", s, e))
    return out


def _make_smoke_sae(act_dim: int, dict_size: int = 256, k: int = 4):
    """From-config tiny BatchTopK SAE over the tiny-model hidden dim (smoke only;
    the sae-arm smoke convention — real class, synthetic weights)."""
    import issue1482_sae as SAEMOD

    torch.manual_seed(1738)
    sd = {
        "b_dec": torch.zeros(act_dim),
        "k": torch.tensor(k),
        "threshold": torch.tensor(0.0),
        "decoder.weight": torch.randn(act_dim, dict_size) * 0.05,
        "encoder.weight": torch.randn(dict_size, act_dim) * 0.05,
        "encoder.bias": torch.zeros(dict_size),
    }
    return SAEMOD.BatchTopKSAE(sd, k=k, act_dim=act_dim, dict_size=dict_size)


def _resolve_crossed_sae(args, smoke_sae):
    """Fail-fast SAE staging for the fold-in (plan §4.2 preamble). Returns
    (sae | None). --no-sae => None (encode disabled outright, recorded)."""
    import issue1482_sae as SAEMOD

    if args.no_sae:
        return None
    if smoke_sae is not None:
        return smoke_sae
    cache = args.sae_cache_dir or (args.out_dir / "sae_cache")
    SAEMOD.BatchTopKSAE.ensure_downloaded(CROSSED_SAE_K, cache, layer=int(args.crossed_sae_layer))
    dev = "cuda" if args.device == "cuda" else "cpu"
    return SAEMOD.BatchTopKSAE.load(
        k=CROSSED_SAE_K, device=dev, cache_dir=cache, layer=int(args.crossed_sae_layer)
    )


def _crossed_sae_verdict_from_hub(up: str) -> bool:
    """Fleet shards read the PILOT's authoritative sae_enabled verdict from the
    Hub pilot meta (single decision point; launcher runs the pilot foreground
    first). Fail loud when absent — the launcher ordering guarantees presence."""
    from huggingface_hub import hf_hub_download

    try:
        p = hub.retry_transient(
            lambda: hf_hub_download(
                C.HF_DATA_REPO,
                f"{up}/{CAPTURE_SUBDIR}/{CROSSED_PILOT_META_NAME}",
                repo_type="dataset",
            ),
            what="crossed pilot meta fetch",
        )
    except Exception as e:  # noqa: BLE001 — re-raised with the operator recipe
        raise RuntimeError(
            f"crossed pilot meta absent on Hub ({up}/{CAPTURE_SUBDIR}/"
            f"{CROSSED_PILOT_META_NAME}) — run the launcher's foreground pilot first"
        ) from e
    doc = json.loads(Path(p).read_text())
    return bool(doc["sae"]["enabled"])


def _sae_encode_row(sae, span_states, px_state, cx_state, fve_pool):
    """SAE fold-in for ONE captured row: encode + pool inlier answer tokens
    (mean/max/frac trio) and the stored px/cx states (sae-arm helpers verbatim).
    Returns the per-row sparse pieces, or None when all tokens are outliers.

    Plan §6 with/without-mask robustness twin (r2 blocker-2): per-token states
    are a declared discard, so the UNMASKED pooled trio is computed HERE, on the
    same already-materialized states. ``encode`` is row-independent (thresholded
    ReLU per token), so masked == unmasked EXACTLY when no token is an outlier —
    the ``nm`` twin is stored ONLY for rows where the mask bites (n_inl < n_ans);
    the reads driver reconstructs equality for the rest (near-zero storage)."""
    import issue1482_sae as SAEMOD

    inl = SAEMOD.token_inlier_mask(span_states)
    span_in = span_states[inl]
    if span_in.shape[0] == 0:
        return None
    if fve_pool is not None:
        fve_pool["answer"].add(span_in)
        fve_pool["context"].add(px_state[None, :])
        fve_pool["context"].add(cx_state[None, :])
    f_all = sae.encode(span_states.to(sae.device))
    spd = SAEMOD.sparsify(SAEMOD.pool_answer_features(f_all[inl.to(f_all.device)]))
    out = {
        "idx": spd["idx"],
        "mean": spd["mean"],
        "max": spd["max"],
        "frac": spd["frac"],
        "n_ans": int(span_states.shape[0]),
        "n_inl": int(span_in.shape[0]),
        "nm": (
            SAEMOD.sparsify(SAEMOD.pool_answer_features(f_all))
            if int(span_in.shape[0]) < int(span_states.shape[0])
            else None  # no outliers: unmasked == masked by construction
        ),
    }
    for name, state in (("px", px_state), ("cx", cx_state)):
        v = sae.encode(state[None, :].to(sae.device))[0]
        nz = torch.nonzero(v, as_tuple=False).squeeze(-1)
        out[f"{name}_idx"] = nz.cpu().numpy().astype(np.int32)
        out[f"{name}_val"] = v[nz].float().cpu().numpy().astype(np.float16)
    return out


def _capture_crossed_rows(hf, tok, rows, responses, layers, sae, sae_layer_pos, fve_pool):
    """Per-row crossed capture: parent px/cx/v_x path + optional SAE fold-in on
    the answer-span states the SAME forward already materialized (plan §4.2).
    Returns (kept_row_dicts, violations, empty_response_cis)."""
    out, violations, empty_ci = [], [], []
    keep_tok = sae is not None
    for r, resp in zip(rows, responses, strict=True):
        cap, reason = _capture_context_and_prefix(hf, tok, r["messages"], layers)
        if cap is None:
            violations.append({"ci": int(r["i"]), "reason": reason})
            continue
        av = COL.capture_answer_vector(
            hf, tok, r["messages"], resp, layers, {}, keep_per_token=keep_tok
        )
        if av is None:  # empty response — a grid hole; counted for the 0.5% budget read
            empty_ci.append(int(r["i"]))
            continue
        row = {
            "ci": int(r["i"]),
            "prefix_id": int(r["prefix_id"]),
            "query_id": int(r["query_id"]),
            "messages": r["messages"],
            "response": resp,
            "depth": int(r["depth"]),
            "corpus": r["corpus"],
            "cx_last": cap["cx_last"],
            "px_last": cap["px_last"],
            "v_x": av["v_x"],
        }
        if keep_tok:
            row["sae"] = _sae_encode_row(
                sae,
                av["per_token"][:, sae_layer_pos, :],
                cap["px_last"][sae_layer_pos],
                cap["cx_last"][sae_layer_pos],
                fve_pool,
            )
        out.append(row)
    return out, violations, empty_ci


def _stack_chunk_crossed(rows, layers, shard_index, chunk_idx, *, sae, sae_layer):
    """Crossed chunk = parent bundle + {prefix_id, query_id} + (when the fold-in
    ran) the sae-arm chunk schema fields VERBATIM, so the sae-arm scan/CSR
    builders (`_scan_sae` / `_build_sae_matrices`) reuse without adaptation."""
    d = _stack_chunk_mt(rows, layers, shard_index, chunk_idx)
    d["prefix_id"] = [int(r["prefix_id"]) for r in rows]
    d["query_id"] = [int(r["query_id"]) for r in rows]
    d["sae_enabled"] = sae is not None
    if sae is None:
        return d

    def _cat(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    feat_idx, row_ptr = [], [0]
    vals: dict[str, list] = {"mean": [], "max": [], "frac": []}
    nm_idx, nm_ptr = [], [0]
    nm_vals: dict[str, list] = {"mean": [], "max": [], "frac": []}
    pxi, pxv, pxp = [], [], [0]
    cxi, cxv, cxp = [], [], [0]
    n_ans, n_inl, sae_skipped = [], [], []
    li = list(layers).index(sae_layer)
    for r in rows:
        s = r["sae"]
        if s is None:  # all answer tokens outliers — empty feature row, 1:1 kept
            sae_skipped.append(int(r["ci"]))
            row_ptr.append(row_ptr[-1])
            nm_ptr.append(nm_ptr[-1])
            pxp.append(pxp[-1])
            cxp.append(cxp[-1])
            n_ans.append(0)
            n_inl.append(0)
            continue
        feat_idx.append(s["idx"])
        row_ptr.append(row_ptr[-1] + len(s["idx"]))
        for p in ("mean", "max", "frac"):
            vals[p].append(s[p])
        # unmasked twin (plan §6 mask-robustness): stored ONLY where the mask
        # bit (n_inl < n_ans); empty row otherwise (== masked by construction)
        nm = s.get("nm")
        if nm is None:
            nm_ptr.append(nm_ptr[-1])
        else:
            nm_idx.append(nm["idx"])
            nm_ptr.append(nm_ptr[-1] + len(nm["idx"]))
            for p in ("mean", "max", "frac"):
                nm_vals[p].append(nm[p])
        pxi.append(s["px_idx"])
        pxv.append(s["px_val"])
        pxp.append(pxp[-1] + len(s["px_idx"]))
        cxi.append(s["cx_idx"])
        cxv.append(s["cx_val"])
        cxp.append(cxp[-1] + len(s["cx_idx"]))
        n_ans.append(s["n_ans"])
        n_inl.append(s["n_inl"])
    d.update(
        {
            "feat_idx": _cat(feat_idx, np.int32),
            "row_ptr": np.asarray(row_ptr, dtype=np.int64),
            "ans_mean": _cat(vals["mean"], np.float16),
            "ans_max": _cat(vals["max"], np.float16),
            "ans_frac": _cat(vals["frac"], np.float16),
            "nm_feat_idx": _cat(nm_idx, np.int32),
            "nm_row_ptr": np.asarray(nm_ptr, dtype=np.int64),
            "nm_mean": _cat(nm_vals["mean"], np.float16),
            "nm_max": _cat(nm_vals["max"], np.float16),
            "nm_frac": _cat(nm_vals["frac"], np.float16),
            "px_feat_idx": _cat(pxi, np.int32),
            "px_row_ptr": np.asarray(pxp, dtype=np.int64),
            "px_feat_val": _cat(pxv, np.float16),
            "cx_feat_idx": _cat(cxi, np.int32),
            "cx_row_ptr": np.asarray(cxp, dtype=np.int64),
            "cx_feat_val": _cat(cxv, np.float16),
            "px_dense19": torch.stack([r["px_last"][li] for r in rows]).to(torch.float16),
            "cx_dense19": torch.stack([r["cx_last"][li] for r in rows]).to(torch.float16),
            "n_ans_tokens": np.asarray(n_ans, dtype=np.int32),
            "n_inlier_tokens": np.asarray(n_inl, dtype=np.int32),
            "sae_skipped_ci": sae_skipped,
            "dropped_ci": [],  # _scan_sae contract: feature rows are 1:1 with ci here
            "sae": {
                "repo": None if sae is None else getattr(sae, "repo", None),
                "k": sae.k,
                "dict_size": sae.dict_size,
                "layer": int(sae_layer),
            },
        }
    )
    return d


def _run_bank_bare_capture(args, tok, hf, bank: list[str], layers, up: str) -> None:
    """The 20-query bare-bank capture (once, negligible — plan §4.2): B0 render
    self-test + batched empty-system capture of every realized bank query at the
    capture layers -> ONE bank_bare.pt under {up}/bare_bank/."""
    name = "bank_bare.pt"
    if not args.no_upload and name in N50._remote_index(f"{up}/{BARE_BANK_SUBDIR}"):
        logger.info("[crossed bank-bare] already on Hub; skip")
        return
    parts = _bare_render_selftest(tok)  # B0 hard asserts (bare-round convention)
    texts = [_render_bare_query(tok, q, parts) for q in bank]
    bq = _capture_bare_batch(hf, tok, texts, layers, args.bare_batch)
    h_dim = int(hf.config.hidden_size)
    assert bq.shape == (len(bank), len(layers), h_dim), bq.shape
    out = args.out_dir / "crossed_shards" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "bq_last": bq,
            "query_id": list(range(len(bank))),
            "bank_sha256": _bank_sha(bank),
            "bare_render": texts,
            "layers": list(layers),
        },
        out,
    )
    if not args.no_upload:
        url = hub._upload(
            out,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{up}/{BARE_BANK_SUBDIR}/{name}",
            upload_as_file=True,
        )
        if not url:
            raise RuntimeError("bank_bare.pt upload returned no URL")
    logger.info("[crossed bank-bare] captured %d bank queries -> %s", len(bank), out)


def _crossed_violation_should_halt(n_violations: int, n_attempted: int) -> bool:
    """Pure fleet-halt predicate: strict-token-prefix violations above the 0.5%
    budget once enough rows have been attempted (plan §4.2; rate below the
    attempt floor is statistically meaningless)."""
    return (
        n_attempted >= CROSSED_VIOLATION_MIN_ATTEMPTED
        and n_violations / n_attempted > PILOT_VIOLATION_RATE_MAX
    )


def _crossed_sae_gate(fve_pool, sae, *, force: str) -> dict:
    """Pilot SAE fitness re-check (plan §12.4): reference-parity fve_l0 on the
    accumulated inlier ANSWER token pool; bar FVE >= 0.75 / L0 in [30, 120].
    FAIL => the fleet PROCEEDS with the encode disabled (R6 skips cleanly).
    ``force`` ("on"/"off"/"") overrides the verdict (smoke: gate computed +
    recorded, verdict demoted to a log line — production-n gate calibration)."""
    pool = fve_pool["answer"].tensor()
    out: dict = {
        "fve_min": CROSSED_SAE_FVE_MIN,
        "l0_range": list(CROSSED_SAE_L0_RANGE),
        "n_pool_tokens": int(pool.shape[0]),
        "force": force,
    }
    if sae is None:
        out.update({"enabled": False, "reason": "--no-sae"})
        return out
    fve, l0, diag = sae.fve_l0(pool)
    lo, hi = CROSSED_SAE_L0_RANGE
    verdict = bool(fve >= CROSSED_SAE_FVE_MIN and lo <= l0 <= hi)
    out.update({"fve": float(fve), "l0": float(l0), "diag": diag, "gate_pass": verdict})
    if force == "on":
        out["enabled"] = True
        out["reason"] = "forced on (smoke gate demoted to informational)"
    elif force == "off":
        out["enabled"] = False
        out["reason"] = "forced off"
    else:
        out["enabled"] = verdict
        out["reason"] = "gate verdict"
    logger.info(
        "[crossed sae-gate] fve=%.4f l0=%.1f pass=%s enabled=%s", fve, l0, verdict, out["enabled"]
    )
    return out


def run_crossed_capture(args, *, smoke_sae=None) -> int:
    """S1 (plan v9 §4.2): sharded-BY-PREFIX crossed generation + capture + SAE
    fold-in + the once-only bank bare capture. Pilot (--pilot-cap, in PREFIXES)
    runs G1 (throughput fence, designed rc 28) + G2 (crossing sanity, rc 30) +
    the SAE fitness gate, and writes/uploads crossed_pilot_meta.json BEFORE the
    fleet detaches (launcher foreground, sae-arm pattern)."""
    up = args.crossed_hf_prefix
    prefix_rows, bank, split_doc, cmeta = _load_crossed_manifest(args)
    n_q_grid = int(cmeta["n_queries"])
    n_q = int(args.queries_per_prefix) or n_q_grid
    assert 1 <= n_q <= n_q_grid, (n_q, n_q_grid)
    if n_q != n_q_grid:
        logger.warning("[crossed] G1 query ladder: realizing %d/%d queries/prefix", n_q, n_q_grid)
    layers = [int(x) for x in args.capture_layers.split(",")]
    sae_layer = int(args.crossed_sae_layer)
    assert sae_layer in layers, (sae_layer, layers)
    sae_layer_pos = layers.index(sae_layer)

    P = len(prefix_rows)
    start, end = N50._shard_range(P, args.num_shards, args.shard_index)
    shard_prefixes = prefix_rows[start:end]
    is_pilot = bool(args.pilot_cap and args.pilot_cap > 0)
    if is_pilot:
        shard_prefixes = shard_prefixes[: args.pilot_cap]
    logger.info(
        "[crossed shard %d/%d] prefixes [%d, %d) = %d x %d queries = %d rows%s -> %s",
        args.shard_index,
        args.num_shards,
        start,
        end,
        len(shard_prefixes),
        n_q,
        len(shard_prefixes) * n_q,
        f" PILOT cap {args.pilot_cap} prefixes" if is_pilot else "",
        up,
    )

    scratch = args.out_dir / "crossed_shards"
    scratch.mkdir(parents=True, exist_ok=True)
    if args.no_upload:
        # smoke seam: a fake done index exercises the resume-skip path in-process
        smoke_done = set(getattr(args, "smoke_done_index", None) or [])
        done_pt = {n for n in smoke_done if n.endswith(".pt")}
        done_raw = {n for n in smoke_done if n.endswith(".json")}
    else:
        done_pt = set(N50._remote_index(f"{up}/{CAPTURE_SUBDIR}"))
        done_raw = set(N50._remote_index(f"{up}/{RAW_SUBDIR}"))
    qtag = f"_q{n_q:02d}_"
    foreign = sorted(n for n in done_pt if n.endswith(".pt") and qtag not in n)
    if foreign:
        logger.warning(
            "[crossed shard %d] %d foreign-n_q chunk(s) already under the prefix "
            "(G1 ladder residue, e.g. %s) — the resume skip ignores them and the "
            "reads scanner keys on the realized n_q from the pilot meta",
            args.shard_index,
            len(foreign),
            foreign[0],
        )

    C.phase("crossed_load_model")
    tok, hf = _load_models(args)
    llm = N1M._build_capture_engine(args) if args.device == "cuda" else None
    h_dim = int(hf.config.hidden_size)
    sae = _resolve_crossed_sae(args, smoke_sae)
    fve_pool = None
    if is_pilot or (args.no_upload and sae is not None):
        # pilot (or upload-less smoke): the gate is computed in-run from the pool
        import issue1738_sae_arm as SAEARM  # deferred: SAEARM imports this module

        fve_pool = {
            "answer": SAEARM._FvePool(cap=CROSSED_SAE_POOL_CAP),
            "context": SAEARM._FvePool(cap=CROSSED_SAE_POOL_CAP),
        }
        sae_gate = None
    elif sae is not None:
        # fleet shard: the PILOT's Hub-published verdict is authoritative
        enabled = _crossed_sae_verdict_from_hub(up)
        sae_gate = {"enabled": enabled, "reason": "pilot meta (Hub)"}
        if not enabled:
            logger.warning("[crossed] pilot verdict: SAE encode DISABLED — R6 will skip")
            sae = None
    else:
        sae_gate = {"enabled": False, "reason": "--no-sae"}

    if args.shard_index == 0:
        _run_bank_bare_capture(args, tok, hf, bank[:n_q], layers, up)

    C.phase("crossed_capture")
    rows_all = _crossed_rows(shard_prefixes, bank, n_q_grid, n_q)
    plan = _crossed_chunk_plan(len(rows_all), args.shard_size, args.shard_index, n_q, is_pilot)
    n_sub = len(plan)
    kept_total = 0
    t_start = time.time()
    pending_pt: list[str] = []
    pending_raw: list[str] = []
    violations_all: list[dict] = []
    skipped_all: list[dict] = []
    empty_all: list[int] = []
    pilot_px: list[torch.Tensor] = []

    def _flush_pending() -> None:
        if args.no_upload or not pending_pt:
            return
        _flush_upload_batch_mt(scratch, up, pending_pt, pending_raw)
        pending_pt.clear()
        pending_raw.clear()

    def _on_sigterm(signum, frame):
        raise SystemExit(f"SIGTERM ({signum}) received — flushing pending upload batch")

    prev_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        for ci_idx, (base, s, e) in enumerate(plan):
            name = f"{base}.pt"
            raw_name = f"{base}.json"
            chunk = rows_all[s:e]
            kept_rows, skipped = _filter_rows_overlength(
                chunk, _gen_render_len(tok), PROMPT_TOKEN_BUDGET
            )
            skipped_all.extend(skipped)
            if skipped:  # expected ~0: the manifest prechecked against the longest query
                logger.warning(
                    "[crossed shard %d] chunk %d: %d over-length skips (precheck leak?); cis=%s",
                    args.shard_index,
                    ci_idx,
                    len(skipped),
                    [d["ci"] for d in skipped][:20],
                )
            if name in done_pt and raw_name in done_raw:
                logger.info(
                    "[crossed shard %d] chunk %d/%d already on Hub; skip",
                    args.shard_index,
                    ci_idx + 1,
                    n_sub,
                )
                continue
            if not kept_rows:
                continue
            ts = time.time()
            responses = _generate_multiturn(llm, tok, [r["messages"] for r in kept_rows])
            rows, violations, empties = _capture_crossed_rows(
                hf, tok, kept_rows, responses, layers, sae, sae_layer_pos, fve_pool
            )
            violations_all.extend(violations)
            empty_all.extend(empties)
            # fleet halt: strict-token-prefix violations above the 0.5% budget
            attempted = kept_total + len(rows) + len(violations_all)
            if _crossed_violation_should_halt(len(violations_all), attempted):
                rep = {
                    "gate": "crossed-violation",
                    "shard_index": int(args.shard_index),
                    "n_attempted": attempted,
                    "n_violations": len(violations_all),
                    "rate": len(violations_all) / attempted,
                    "rate_max": PILOT_VIOLATION_RATE_MAX,
                }
                C.write_json_atomic(args.out_dir / "crossed_violation_report.json", rep)
                logger.error("[crossed] violation fleet-halt: %s", rep)
                _flush_pending()
                sys.exit(RC_CROSSED_VIOLATION)
            if not rows:
                logger.warning(
                    "[crossed shard %d] chunk %d: 0 captured rows; skip", args.shard_index, ci_idx
                )
                continue
            for fld in ("px_last", "cx_last", "v_x"):
                for r in rows:
                    assert r[fld].shape == (len(layers), h_dim), (fld, r[fld].shape)
            torch.save(
                _stack_chunk_crossed(
                    rows, layers, args.shard_index, ci_idx, sae=sae, sae_layer=sae_layer
                ),
                scratch / name,
            )
            C.write_json_atomic(
                scratch / raw_name,
                {
                    "shard_index": args.shard_index,
                    "chunk": ci_idx,
                    "rows": [
                        {
                            "ci": int(r["ci"]),
                            "prefix_id": int(r["prefix_id"]),
                            "query_id": int(r["query_id"]),
                            "messages": r["messages"],
                            "response": r["response"],
                            "depth": int(r["depth"]),
                            "corpus": r["corpus"],
                        }
                        for r in rows
                    ],
                },
            )
            if is_pilot:
                pilot_px.append(torch.stack([r["px_last"] for r in rows]))
            kept_total += len(rows)
            if not args.no_upload:
                pending_pt.append(name)
                pending_raw.append(raw_name)
                if len(pending_pt) >= UPLOAD_BATCH:
                    _flush_pending()
            logger.info(
                "[crossed-capture] chunk %d/%d shard=%d: %d/%d captured (%d viol, %d empty, "
                "%.0fs elapsed=%.0fs)",
                ci_idx + 1,
                n_sub,
                args.shard_index,
                len(rows),
                len(chunk),
                len(violations),
                len(empties),
                time.time() - ts,
                time.time() - t_start,
            )
        _flush_pending()
    except BaseException:
        try:
            _flush_pending()
        except Exception:
            logger.exception(
                "[crossed shard %d] best-effort pending-batch flush failed on exit",
                args.shard_index,
            )
        raise
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)

    _write_sidecar(scratch, args, skipped_all, violations_all, prefix=up)
    wall_h = (time.time() - t_start) / 3600.0
    logger.info(
        "[crossed shard %d] done: %d rows / %d chunks (%d over-length, %d violations, "
        "%d empty-response drops, %.2f h)",
        args.shard_index,
        kept_total,
        n_sub,
        len(skipped_all),
        len(violations_all),
        len(empty_all),
        wall_h,
    )
    if is_pilot or (args.no_upload and fve_pool is not None):
        sae_gate = _crossed_sae_gate(fve_pool, sae, force=args.crossed_sae_force)
    if is_pilot:
        rc = _write_crossed_pilot_meta(
            args,
            up,
            kept_total=kept_total,
            wall_h=wall_h,
            n_rows_total=P * n_q,
            skipped_all=skipped_all,
            violations_all=violations_all,
            pilot_px=pilot_px,
            sae_gate=sae_gate,
            n_q=n_q,
            n_q_grid=n_q_grid,
            n_empty=len(empty_all),
        )
        if rc != 0:
            sys.exit(rc)
    C.phase("done")
    return 0


def _write_crossed_pilot_meta(
    args,
    up,
    *,
    kept_total,
    wall_h,
    n_rows_total,
    skipped_all,
    violations_all,
    pilot_px,
    sae_gate,
    n_q,
    n_q_grid,
    n_empty=0,
) -> int:
    """G1/G2 pilot artifact + designed-halt routing (report FIRST, then rc).

    G1: projected total GPU-h (measured rate x full grid + the S2 book) past the
    24 GPU-h fence => rc 28 (the orchestrator re-sizes the query ladder; never a
    silent overspend). G2: per-layer min pairwise prefix-end cosine must sit
    BELOW 0.999 (the multi-turn prefixes genuinely vary) => rc 30 on FAIL."""
    n_attempted = kept_total + len(violations_all)
    viol_rate = len(violations_all) / max(1, n_attempted)
    px = torch.cat(pilot_px, dim=0) if pilot_px else torch.zeros(0, 1, 1)
    min_cos = _min_pairwise_cos(px) if px.shape[0] >= 2 else {}
    layers = [int(x) for x in args.capture_layers.split(",")]
    min_cos_by_layer = {str(layers[int(k)]): v for k, v in min_cos.items()}
    rate = kept_total / wall_h if wall_h > 0 else float("nan")
    projected_gpu_h = (n_rows_total / rate if rate > 0 else float("inf")) + (CROSSED_S2_BOOK_GPU_H)
    g2_prefix_varies = bool(min_cos_by_layer) and all(
        v < PILOT_PREFIX_COS_MAX for v in min_cos_by_layer.values()
    )
    doc = {
        "n_captured": int(kept_total),
        "wall_h": float(wall_h),
        "ctx_per_gpu_h": float(rate),
        "n_rows_total_grid": int(n_rows_total),
        "queries_per_prefix_realized": int(n_q),
        "queries_per_prefix_grid": int(n_q_grid),
        "projected_total_gpu_h": float(projected_gpu_h),
        "fence_gpu_h": float(args.crossed_fence_gpu_h),
        "n_overlength_skipped": len(skipped_all),
        "n_prefix_violations": len(violations_all),
        "n_empty_response_dropped": int(n_empty),  # grid holes vs the <0.5% budget (r2 minor)
        "violation_rate": float(viol_rate),
        "violation_rate_max": PILOT_VIOLATION_RATE_MAX,
        "prefix_min_pairwise_cos_by_layer": min_cos_by_layer,
        "prefix_cos_max": PILOT_PREFIX_COS_MAX,
        "gate_g1": {"projected_within_fence": projected_gpu_h <= args.crossed_fence_gpu_h},
        "gate_g2": {
            "prefix_varies_ok": g2_prefix_varies,
            "bank_byte_identity_ok": True,  # asserted fail-loud at manifest load
            "violation_rate_ok": viol_rate <= PILOT_VIOLATION_RATE_MAX,
        },
        "sae": sae_gate,
        "shard_index": int(args.shard_index),
        "pilot_cap_prefixes": int(args.pilot_cap),
    }
    out = args.out_dir / CROSSED_PILOT_META_NAME
    C.write_json_atomic(out, doc)
    logger.info(
        "[crossed-pilot] ctx/GPU-h=%.0f projected=%.1f GPU-h (fence %.0f) viol=%.4f "
        "min_cos=%s sae=%s",
        rate,
        projected_gpu_h,
        args.crossed_fence_gpu_h,
        viol_rate,
        min_cos_by_layer,
        sae_gate.get("enabled"),
    )
    if not args.no_upload:
        # under capture/ — a FILE at the bare prefix 400-blocks later folder
        # commits under the prefix (the fu1 B1 incident, commit dd9a615c22)
        url = hub._upload(
            out,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{up}/{CAPTURE_SUBDIR}/{CROSSED_PILOT_META_NAME}",
            upload_as_file=True,
        )
        if not url:
            raise RuntimeError("crossed_pilot_meta.json upload returned no URL")
    if not doc["gate_g2"]["prefix_varies_ok"] or not doc["gate_g2"]["violation_rate_ok"]:
        logger.error("[crossed G2] FAIL: %s — halt fleet (code bug, not science)", doc["gate_g2"])
        return RC_CROSSED_G2
    if not doc["gate_g1"]["projected_within_fence"]:
        logger.error(
            "[crossed G1] projected %.1f GPU-h > fence %.0f — designed halt (query ladder "
            "20->12->10 is the orchestrator's re-size; prefixes are never reduced)",
            projected_gpu_h,
            args.crossed_fence_gpu_h,
        )
        return RC_CROSSED_FENCE
    return 0


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
            # over-length filter on the EXACT generation render (parent mechanism;
            # r5: shared _gen_render_len — the kresample admission gate uses the
            # SAME arithmetic, single source)
            kept_rows, skipped = _filter_rows_overlength(
                chunk, _gen_render_len(tok), PROMPT_TOKEN_BUDGET
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


def _write_sidecar(
    scratch: Path, args, skipped_all: list[dict], violations_all: list[dict], *, prefix: str = ""
):
    """Per-shard sidecar: over-length skips + strict-token-prefix violations
    (ci + counts only, never text). Uploaded beside raw_completions. ``prefix``
    overrides the upload prefix (the bare arm's ``--upload-prefix``; default =
    ``args.hf_prefix``, the parent behavior)."""
    # UPLOAD_PREFIX_EXEMPT: default = this issue's own --hf-prefix (issue1738_multiturn, parent behavior); bare arm passes an explicit prefix (plan v6 §4.1.4)
    up = prefix or args.hf_prefix
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
    # UPLOAD_PREFIX_EXEMPT: dest defaults to this issue's own --hf-prefix (issue1738_multiturn, parent behavior); the bare arm passes an explicit prefix (plan v6 §4.1.4)
    url = hub._upload_folder_filtered(
        scratch,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{up}/{RAW_SUBDIR}",
        allow_patterns=[skip_name],
        expected_repo_paths=[f"{up}/{RAW_SUBDIR}/{skip_name}"],
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
    # r5 admission gate (BEFORE the engine build): capture-parity over-length
    # skip + primary-draw membership — skip+record, never engine-fatal.
    primary_cis = _kresample_primary_cis(args)
    rows, gate_skipped = _kresample_admission_gate(rows, _gen_render_len(tok), primary_cis)
    _write_kresample_skipped(scratch, args, gate_skipped)
    if gate_skipped:
        logger.warning(
            "[kresample] skipped %d rows (%d over-length > %d tok, %d no-primary-draw); cis=%s",
            len(gate_skipped),
            sum(1 for d in gate_skipped if "overlength" in d["reasons"]),
            PROMPT_TOKEN_BUDGET,
            sum(1 for d in gate_skipped if "no_primary_draw" in d["reasons"]),
            [d["ci"] for d in gate_skipped],
        )
    if not rows:
        logger.warning(
            "[kresample] shard %d kept 0 contexts after admission gate", args.shard_index
        )
        C.phase("done")
        return 0
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
    # r5: an over-length-TOKENS row UNDER the 26k-char manifest prefilter — it
    # survives the build and must be caught by the generation-time budget gates
    # (capture filter + kresample admission gate; the ci-98764 crash class).
    long_tok_row = _synth_conv(8888, 2)
    long_tok_row["conversation"][0]["content"] = " ".join(f"q{i}x" for i in range(1800))
    smoke_lmsys.append(long_tok_row)
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

    # (6) kresample path on the tiny model (stub generation, K=2), r5 shape:
    # 2 admitted contexts + 1 over-length-TOKENS row (the ci-98764 crash class)
    # + 1 no-primary-draw row — the admission gate must SKIP+RECORD both, never
    # hand them to the engine.
    long_tok_ci = next(
        int(r["i"]) for r in pool if any(len(m["content"]) > 8_000 for m in r["messages"])
    )
    keep_cis = [int(pool[0]["i"]), int(pool[1]["i"])]
    gate_out_ci = next(int(r["i"]) for r in pool if int(r["i"]) not in {*keep_cis, long_tok_ci})
    sub_doc = ns.out_dir / "kresample_subsample.json"
    sub_cis = [*keep_cis, long_tok_ci, gate_out_ci]
    N1M._atomic_write_json(
        sub_doc, {"ci": sub_cis, "seed": 173801, "sha256": _sha_int_list(sub_cis)}
    )
    # primary-draw set: everything EXCEPT gate_out_ci (long_tok_ci included, so
    # its skip is attributable to "overlength" alone).
    primary_npz = ns.out_dir / "kresample_primary.npz"
    np.savez(primary_npz, ci=np.asarray([*keep_cis, long_tok_ci], dtype=np.int64))
    kargs = argparse.Namespace(
        **{
            **vars(cargs),
            "kresample_subsample": str(sub_doc),
            "kresample_primary_ci": str(primary_npz),
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
    # degenerate probe (r5 kept-0 gate): a subsample whose rows are ALL gated out
    # exits 0 with the skip record written and NO shard tensor.
    zero_doc = ns.out_dir / "kresample_subsample_zero.json"
    N1M._atomic_write_json(
        zero_doc, {"ci": [gate_out_ci], "seed": 173801, "sha256": _sha_int_list([gate_out_ci])}
    )
    rc = run_kresample(argparse.Namespace(**{**vars(kargs), "kresample_subsample": str(zero_doc)}))
    assert rc == 0
    assert not (ns.out_dir / "kresample" / "kresample_shard00.pt").exists(), (
        "kept-0 kresample run must not write a shard tensor"
    )
    zrec = json.loads((ns.out_dir / "kresample" / "kresample_shard00_skipped.json").read_text())
    assert zrec["n_skipped"] == 1 and zrec["skipped"][0]["reasons"] == ["no_primary_draw"], zrec
    # main run: 2 admitted, 2 skipped (1 overlength, 1 no_primary_draw).
    rc = run_kresample(kargs)
    assert rc == 0
    kb = torch.load(ns.out_dir / "kresample" / "kresample_shard00.pt", weights_only=False)
    assert kb["V"].shape[:2] == (2, 2) and kb["seeds"] == [43, 44], kb["V"].shape
    assert kb["ci"] == keep_cis, (kb["ci"], keep_cis)
    krec = json.loads((ns.out_dir / "kresample" / "kresample_shard00_skipped.json").read_text())
    by_reason = {tuple(d["reasons"]): d for d in krec["skipped"]}
    assert krec["n_skipped"] == 2 and krec["prompt_token_budget"] == PROMPT_TOKEN_BUDGET, krec
    assert by_reason[("overlength",)]["ci"] == long_tok_ci, krec
    assert by_reason[("overlength",)]["n_tokens"] > PROMPT_TOKEN_BUDGET, krec
    assert by_reason[("no_primary_draw",)]["ci"] == gate_out_ci, krec

    # (7) bare-query arm (plan §4.1.6): B0 render self-test + a tiny batched bare
    # pass through the PRODUCTION entrypoint (run_bare_capture; tiny 2-layer
    # model, REAL Qwen tokenizer). Render asserts + shapes + bq != cx sanity.
    bargs = argparse.Namespace(
        **{
            **vars(cargs),
            "bare_query": True,
            "upload_prefix": "",
            "bare_batch": 2,
            "bare_chunk": 3,
            "bare_fence_min": 10_000.0,
            "pilot_cap": 6,
            "num_shards": 4,
            "shard_index": 0,
        }
    )
    rc = run_bare_capture(bargs)
    assert rc == 0
    bchunks = sorted((ns.out_dir / "bare_shards").glob("shard00_chunk*.pt"))
    assert bchunks, "bare capture smoke wrote no chunks"
    bq_by_ci: dict[int, torch.Tensor] = {}
    for bc in bchunks:
        bb = torch.load(bc, weights_only=False)
        assert bb["bq_last"].shape[1] == 2, bb["bq_last"].shape
        assert len(bb["bare_render"]) == len(bb["ci"]) == bb["bq_last"].shape[0], bc
        for j, c in enumerate(bb["ci"]):
            bq_by_ci[int(c)] = bb["bq_last"][j]
    cx_by_ci: dict[int, torch.Tensor] = {}
    for pc in sorted((ns.out_dir / "shards").glob("shard00_chunk*.pt")):
        pb = torch.load(pc, weights_only=False)
        for j, c in enumerate(pb["ci"]):
            cx_by_ci[int(c)] = pb["cx_last"][j]
    shared = sorted(set(bq_by_ci) & set(cx_by_ci))
    assert shared, "no shared ci between bare and context smoke chunks"
    for c in shared:
        assert not torch.allclose(bq_by_ci[c], cx_by_ci[c]), (
            f"bq == cx at ci {c} — bare render not distinct from the context render"
        )
    bpm = json.loads((ns.out_dir / "bare_pilot_meta.json").read_text())
    assert bpm["parity_ok"] and bpm["n_render_violations"] == 0, bpm
    assert bpm["worked_example_render"].startswith("<|im_start|>system"), bpm[
        "worked_example_render"
    ]
    # degenerate probes (data-dependent-gates duty): the G-B1 fence predicate
    # fires past the budget + stays quiet under it; a fence-tripping run takes
    # the DESIGNED halt rc (report JSON written first).
    assert _bare_fence_should_halt(3600.0, 100, 10_000, 45.0)
    assert not _bare_fence_should_halt(10.0, 100, 200, 45.0)
    try:
        run_bare_capture(argparse.Namespace(**{**vars(bargs), "bare_fence_min": 1e-9}))
        raise AssertionError("G-B1 bare fence did not halt")
    except SystemExit as e:
        assert e.code == RC_BARE_FENCE, e.code
    frep = json.loads((ns.out_dir / "bare_fence_report.json").read_text())
    assert frep["gate"] == "G-B1" and frep["projected_shard_wall_min"] > 0, frep

    # (8) crossed-multiturn-averaged leg (fu3, plan v9 S0): tiny crossed manifest
    # (from a DIVERSE-query smoke main manifest — synthetic near-identical queries
    # would near-dupe-collapse the bank pool) → crossed capture (tiny model + tiny
    # from-config SAE) → the FULL reads driver on the local chunks. Proves the
    # meta.json contract, the chunk schema incl. the sae fields, and split reuse
    # — all through the PRODUCTION entrypoints (PASS_UNIFIED).
    _smoke_crossed(args, ns, cargs)

    logger.info(
        "[smoke] OK — manifest/allocation/carve/gate/capture-indexing/kresample/bare-query/crossed"
    )
    return 0


_SMOKE_DIVERSE_QUERIES = (
    "Explain how ocean tides form along irregular coastlines",
    "What are solid beginner openings to study in chess",
    "Draft a short toast for my sister's graduation dinner",
    "Compare compost bins and worm farms for a small balcony",
    "Why does bread dough need a second rise before baking",
    "Summarize the plot of a heist film without naming it",
    "How do I split rent fairly with roommates of unequal rooms",
    "Give me stretches for wrist pain from typing all day",
    "What is the difference between espresso and moka pot coffee",
    "Plan a rainy weekend itinerary for two days in a small city",
    "How should I prune an overgrown rosemary bush in spring",
    "Explain latency versus bandwidth using a postal analogy",
)


def _smoke_crossed(args, ns, cargs) -> None:
    """The fu3 S0 leg (plan v9 §4.3): P0 build (incl. the G0 shortfall degenerate
    probe) → S1 capture (both shards; pilot gates G1/G2/violation probed) → S2
    reads (all six reads on the local chunks; R6 skip branch probed)."""
    import numpy as np

    xbase = ns.out_dir / "_smoke_crossed"
    xbase.mkdir(parents=True, exist_ok=True)
    # diverse-last-user-turn corpus so the bank's near-dupe gates keep a pool
    div_lm = [_synth_conv(i, 2 + (i % 4)) for i in range(80)]
    div_wc = [_synth_conv(500 + i, 2, wc=True) for i in range(30)]
    for i, conv in enumerate(div_lm + div_wc):
        q = _SMOKE_DIVERSE_QUERIES[i % len(_SMOKE_DIVERSE_QUERIES)]
        conv["conversation"][-2]["content"] = f"{q} case {i}"
    xmain = argparse.Namespace(
        **{
            **vars(args),
            "no_upload": True,
            "n_target": 60,
            "probe_rows": 40,
            "skip_probe": True,
            "out_dir": xbase,
        }
    )
    build_manifest(xmain, smoke_lmsys=div_lm, smoke_wildchat=div_wc)

    xns = argparse.Namespace(
        **{
            **vars(cargs),
            "out_dir": xbase,
            "crossed_hf_prefix": "issue1738_crossed_smoke",
            "crossed_n_prefixes": 5,
            "crossed_n_queries": 3,
            "crossed_g0_prefix_floor": 5,
            "crossed_g0_bank_floor": 6,
            "crossed_bank_query_token_max": CROSSED_BANK_QUERY_TOKEN_MAX,
            "queries_per_prefix": 0,
            # G1's 24 GPU-h fence is calibrated at production THROUGHPUT — a tiny
            # CPU pilot projects past any real fence by construction (the smoke-
            # gate-calibration law): demote to a non-binding bound here, and probe
            # the fence BRANCH with a degenerate tiny fence below.
            "crossed_fence_gpu_h": 1e9,
            "crossed_sae_layer": 1,
            "no_sae": False,
            "crossed_sae_force": "on",  # gate computed + recorded; verdict demoted (random SAE)
            "sae_cache_dir": None,
            "manifest_from_hf": False,
            "num_shards": 2,
            "shard_index": 0,
            "shard_size": 4,
            "pilot_cap": 3,
        }
    )
    # G0 shortfall degenerate probe FIRST (overwritten by the real build below):
    # an unreachable prefix floor takes-what-exists + reports, never all-or-nothing.
    short_meta = build_crossed_manifest(
        argparse.Namespace(**{**vars(xns), "crossed_g0_prefix_floor": 10_000})
    )
    assert short_meta["g0"]["prefix_pool_ok"] is False, short_meta["g0"]
    assert short_meta["g0"]["prefix_walk_exhausted"] is True, short_meta["g0"]
    assert short_meta["n_new"] == 5, short_meta["n_new"]

    xmeta = build_crossed_manifest(xns)
    assert xmeta["n_new"] == 5 and xmeta["n_queries"] == 3, (xmeta["n_new"], xmeta["n_queries"])
    assert xmeta["overlap_fraction_main_manifest"] == 1.0
    xdir = xbase / CROSSED_MANIFEST_LOCAL
    assert (xdir / "meta.json").exists(), (
        "crossed meta.json contract broken (loader hard-requires it)"
    )
    xsplit = json.loads((xdir / "split_1738_crossed.json").read_text())
    for s in ("val", "test", "holdout", "train"):
        ids = xsplit["sets"][s]["ci"]
        assert xsplit["sets"][s]["sha256"] == _sha_int_list(ids), s
        pset = set(xsplit["prefix_sets"][s]["pi"])
        assert all(i // 3 in pset for i in ids), f"{s}: rows not prefix-grouped"
    row_sets = [set(xsplit["sets"][s]["ci"]) for s in ("val", "test", "holdout", "train")]
    assert sum(len(x) for x in row_sets) == len(set().union(*row_sets)) == 15, (
        "crossed split not a partition of the 5x3 grid"
    )

    # S1 capture: pilot shard 0 (G1/G2/SAE gates) + fleet shards 0+1 -> full grid.
    # Pilot: 3 prefixes x 3 q = 9 rows @ shard_size 4 -> chunk0000, chunk0001 +
    # pilotpartial0002 (blocker-1 sibling A: the trailing partial is NOT
    # fleet-resumable by name, so the fleet re-captures that range fully).
    tiny_sae = _make_smoke_sae(64, dict_size=256, k=4)
    rc = run_crossed_capture(xns, smoke_sae=tiny_sae)
    assert rc == 0
    xscratch = xbase / "crossed_shards"
    assert (xscratch / "shard00_q03_pilotpartial0002.pt").exists(), sorted(
        p.name for p in xscratch.iterdir()
    )
    # fleet shard 0 against a fake done index = everything the pilot "uploaded":
    # full chunks SKIP by name (mtime unchanged); the partial range re-captures
    # under the fleet's `_chunk` name — the resume-skip path, exercised in-process.
    q3_pilot_files = sorted(p.name for p in xscratch.iterdir() if "_q03_" in p.name)
    mt_full = (xscratch / "shard00_q03_chunk0000.pt").stat().st_mtime_ns
    rc = run_crossed_capture(
        argparse.Namespace(**{**vars(xns), "pilot_cap": 0, "smoke_done_index": q3_pilot_files}),
        smoke_sae=tiny_sae,
    )
    assert rc == 0
    assert (xscratch / "shard00_q03_chunk0000.pt").stat().st_mtime_ns == mt_full, (
        "fleet re-captured a full pilot chunk the done index already covers"
    )
    assert (xscratch / "shard00_q03_chunk0002.pt").exists(), (
        "fleet did not re-capture the pilot-partial row range (silent row hole)"
    )
    rc = run_crossed_capture(
        argparse.Namespace(**{**vars(xns), "shard_index": 1, "pilot_cap": 0}),
        smoke_sae=tiny_sae,
    )
    assert rc == 0
    seen_ci: list[int] = []
    nm_extra_bytes = 0
    for xc in sorted(xscratch.glob("shard*_chunk*.pt")):
        xb = torch.load(xc, weights_only=False)
        assert xb["sae_enabled"], xc
        assert int(xb["row_ptr"][-1]) == len(xb["feat_idx"]), "sae CSR row_ptr misaligned"
        # mask-twin schema (r2 blocker-2): nm CSR 1:1 with rows; nm rows are
        # non-empty ONLY where the inlier mask bit (n_inl < n_ans)
        assert len(xb["nm_row_ptr"]) == len(xb["ci"]) + 1, xc
        assert int(xb["nm_row_ptr"][-1]) == len(xb["nm_feat_idx"]), "nm CSR misaligned"
        nm_off = np.diff(np.asarray(xb["nm_row_ptr"]))
        differs = np.asarray(xb["n_inlier_tokens"]) < np.asarray(xb["n_ans_tokens"])
        assert bool(np.all((nm_off > 0) <= differs)), "nm row stored for a mask-equal row"
        nm_extra_bytes += 4 * len(xb["nm_feat_idx"]) + 2 * sum(
            len(xb[k]) for k in ("nm_mean", "nm_max", "nm_frac")
        )
        assert len(xb["prefix_id"]) == len(xb["ci"]) == len(xb["query_id"])
        assert xb["px_dense19"].shape == (len(xb["ci"]), 64), xb["px_dense19"].shape
        for j, c in enumerate(xb["ci"]):
            assert int(c) % 3 == int(xb["query_id"][j]) and 0 <= int(c) // 3 < 5, (c, j)
        seen_ci.extend(int(c) for c in xb["ci"])
    assert sorted(seen_ci) == list(range(15)), f"grid incomplete: {sorted(seen_ci)}"
    logger.info("[smoke crossed] nm mask-twin payload across q3 chunks: %d bytes", nm_extra_bytes)
    assert (xscratch / "bank_bare.pt").exists(), "bank bare capture missing"
    xpm = json.loads((xbase / CROSSED_PILOT_META_NAME).read_text())
    assert xpm["gate_g2"]["prefix_varies_ok"] and xpm["gate_g2"]["violation_rate_ok"], xpm
    assert xpm["sae"]["enabled"] and xpm["sae"]["force"] == "on", xpm["sae"]
    assert "gate_pass" in xpm["sae"] and "fve" in xpm["sae"], "sae gate not COMPUTED"
    assert xpm["queries_per_prefix_realized"] == 3, xpm

    # degenerate gate probes (data-dependent-gates duty):
    # (a) violation fleet-halt predicate fires past the budget, stays quiet under it
    assert _crossed_violation_should_halt(10, 1_000)
    assert not _crossed_violation_should_halt(1, 1_000)
    assert not _crossed_violation_should_halt(500, 999)  # below the attempt floor
    # (b) G1 fence: a tiny fence takes the DESIGNED halt rc 28 (report written first)
    try:
        run_crossed_capture(
            argparse.Namespace(**{**vars(xns), "crossed_fence_gpu_h": 1e-9}),
            smoke_sae=tiny_sae,
        )
        raise AssertionError("crossed G1 fence did not halt")
    except SystemExit as e:
        assert e.code == RC_CROSSED_FENCE, e.code
    frep = json.loads((xbase / CROSSED_PILOT_META_NAME).read_text())
    assert frep["gate_g1"]["projected_within_fence"] is False, frep["gate_g1"]
    # (c) G2 sanity: identical prefix-end states (min cos 1.0) -> designed rc 30
    rc_g2 = _write_crossed_pilot_meta(
        xns,
        "unused-prefix",
        kept_total=4,
        wall_h=0.01,
        n_rows_total=15,
        skipped_all=[],
        violations_all=[],
        pilot_px=[torch.ones(4, 2, 64)],
        sae_gate={"enabled": True},
        n_q=3,
        n_q_grid=3,
    )
    assert rc_g2 == RC_CROSSED_G2, rc_g2
    # restore the PASSING pilot meta for the reads leg (deterministic re-run)
    rc = run_crossed_capture(xns, smoke_sae=tiny_sae)
    assert rc == 0
    xpm = json.loads((xbase / CROSSED_PILOT_META_NAME).read_text())
    assert xpm["gate_g2"]["prefix_varies_ok"] and xpm["sae"]["enabled"], xpm

    # S2: the full reads driver on the local chunks (deferred import: the reads
    # module imports THIS module at top level).
    import issue1738_crossed_reads as CR

    rargs = argparse.Namespace(
        hf_prefix="issue1738_crossed_smoke",
        local_capture_dir=str(xscratch),
        local_manifest_dir=str(xdir),
        pilot_meta=str(xbase / CROSSED_PILOT_META_NAME),
        out_eval=xbase / "crossed_eval",
        out_local=xbase / "crossed_reads_local",
        mm_dir=xbase / "crossed_reads_local" / "mm",
        layers="0,1",
        device="cpu",
        ridge_block=4096,
        # G3's 2x first-cell fence assumes production-homogeneous cell walls;
        # millisecond smoke cells are jitter-dominated (smoke-gate-calibration
        # law) — demote here; the PREDICATE is probed degenerately below.
        fence_mult=100.0,
        coverage_floor=0.95,
        n_boot=25,
        pca_dirs=2,
        inner_cv_folds=2,
        n_operator_nulls=4,
        sae_feature_block=64,
        rb_tail_n=8,
        skip_rb_align=True,
        queries_per_prefix=0,
        kresample_summary=str(xbase / "_missing_kresample.json"),
        no_upload=True,
    )
    rc = CR.run_reads(rargs)
    assert rc == 0
    # G3 fence PREDICATE degenerate probe (the fence itself is smoke-demoted above)
    import issue1738_multiturn_fits as MTF

    assert MTF._fence_should_halt(100.0, 1.0, 10, 2.0)
    assert not MTF._fence_should_halt(10.0, 1.0, 10, 2.0)
    xf = json.loads((rargs.out_eval / "crossed_fits.json").read_text())
    assert len(xf["cells"]) == 4 * 2, sorted(xf["cells"])  # 4 arms x 2 smoke layers
    assert xf["cells"]["avg_L1"]["selection"].startswith("inner CV"), xf["cells"]["avg_L1"]
    assert "gap_induced_minus_independent_r2" in xf["induced"]["L0"], xf["induced"]
    assert ("skipped" in xf["transfer"]) or ("context" in xf["transfer"]["cells"])
    xa = json.loads((rargs.out_eval / "anova.json").read_text())
    sh = xa["per_layer"]["0"]["overall"]
    assert abs(sh["share_prefix"] + sh["share_query"] + sh["share_inter"] - 1.0) < 1e-4, sh
    assert xa["kresample_reference"].get("skipped"), "missing-comparator branch not recorded"
    assert len(xa["per_layer"]["0"]["per_direction"]["share_prefix"]) == 2  # pca_dirs
    ib = json.loads((rargs.out_eval / "mapping_baselines.json").read_text())
    assert ib["cells"]["stitch_L0"]["identity_bias"].get("status") == "inapplicable", (
        "R4 identity+bias dimension-mismatch not recorded EXPLICITLY"
    )
    assert "identity_bias" in ib["cells"]["context_L0"] and (
        "holdout_r2" in ib["cells"]["context_L0"]["identity_bias"]
    )
    xo = json.loads((rargs.out_eval / "operator_geometry.json").read_text())
    assert "context|prefix" in xo["per_layer"]["0"]["pairs"], sorted(xo["per_layer"]["0"]["pairs"])
    assert (rargs.out_eval / "stitch.json").exists()
    xsae = json.loads((rargs.out_eval / "sae_perfeature.json").read_text())
    assert not xsae.get("skipped"), xsae
    assert "induced_averaged" in xsae["feature_maps"], xsae["feature_maps"]
    mr = xsae["mask_robustness"]  # plan §6 with/without-mask twin (r2 blocker-2)
    assert "n_rows_mask_equal" in mr and "n_rows_mask_differs" in mr, mr
    assert mr["n_rows_mask_equal"] + mr["n_rows_mask_differs"] > 0, mr
    assert (rargs.out_eval / "perfeature_crossed_summary.csv").exists()
    # R6 clean-skip branch (sae_enabled=false verdict) — separate out dir so the
    # PASSING artifacts above are never clobbered
    skip_dir = xbase / "crossed_eval_skip"
    skip_dir.mkdir(exist_ok=True)
    CR._phase_sae(
        rargs,
        Path(rargs.mm_dir),
        np.zeros(0, np.int64),
        3,
        3,
        {"train": np.zeros(0, np.int64)},
        np.zeros(0, np.int64),
        np.zeros((0, 3), np.int64),
        {},
        torch.device("cpu"),
        {"enabled": False},
        skip_dir,
        xbase / "crossed_reads_local" / "pf_skip",
        {},
    )
    assert json.loads((skip_dir / "sae_perfeature.json").read_text())["skipped"] is True
    # _stack_chunk_crossed empty-feature-row alignment (all-tokens-outlier shape)
    dummy = [
        {
            "ci": 0,
            "prefix_id": 0,
            "query_id": 0,
            "messages": [{"role": "user", "content": "q"}],
            "response": "r",
            "depth": 2,
            "corpus": "lmsys",
            "cx_last": torch.zeros(2, 64),
            "px_last": torch.zeros(2, 64),
            "v_x": torch.zeros(2, 64),
            "sae": None,
        }
    ]
    dch = _stack_chunk_crossed(dummy, [0, 1], 0, 0, sae=tiny_sae, sae_layer=1)
    assert dch["sae_skipped_ci"] == [0] and dch["row_ptr"].tolist() == [0, 0], dch["row_ptr"]
    assert dch["nm_row_ptr"].tolist() == [0, 0], dch["nm_row_ptr"]

    # ── mask-twin pure probe (r2 blocker-2): synthetic sidecar with one
    # mask-equal row + one differing row of KNOWN cosine 16/25 = 0.64 exactly
    # (masked mean [3,4] on idx [0,3]; unmasked [4,3] on idx [3,8]).
    fake_side = xbase / "fake_side"
    fake_side.mkdir(exist_ok=True)
    two_nm = np.asarray([4.0, 3.0], np.float16)
    fake = {
        "n_ans_tokens": np.asarray([2, 3], np.int32),
        "n_inlier_tokens": np.asarray([2, 2], np.int32),
        "sae_skipped_ci": [],
        "row_ptr": np.asarray([0, 2, 4], np.int64),
        "feat_idx": np.asarray([0, 1, 0, 3], np.int32),
        "ans_mean": np.asarray([0.5, 0.5, 3.0, 4.0], np.float16),
        "ans_max": np.asarray([0.5, 0.5, 3.0, 4.0], np.float16),
        "ans_frac": np.asarray([0.5, 0.5, 3.0, 4.0], np.float16),
        "nm_row_ptr": np.asarray([0, 0, 2], np.int64),
        "nm_feat_idx": np.asarray([3, 8], np.int32),
        "nm_mean": two_nm,
        "nm_max": two_nm,
        "nm_frac": two_nm,
    }
    torch.save(fake, fake_side / "f0.pt")
    mrp = CR._mask_robustness([fake_side / "f0.pt"])
    assert mrp["n_rows_mask_equal"] == 1 and mrp["n_rows_mask_differs"] == 1, mrp
    for pool in ("mean", "max", "frac"):
        assert abs(mrp[pool]["cos_median"] - 0.64) < 1e-3, (pool, mrp[pool])
    legacy = {k: v for k, v in fake.items() if not k.startswith("nm_")}
    torch.save(legacy, fake_side / "legacy.pt")
    assert "skipped" in CR._mask_robustness([fake_side / "legacy.pt"]), (
        "pre-r2 chunks (no nm_*) must skip cleanly (backward tolerance)"
    )

    # ── blocker-1 sibling-A EXACT arithmetic: pilot rows are a strict PREFIX of
    # the fleet shard's rows, so every SHARED name covers the identical slice and
    # the pilot's trailing partial NEVER collides with the fleet's full chunk.
    pplan = _crossed_chunk_plan(6, 4, 0, 2, True)  # pilot: 3 prefixes x 2 q
    assert [b for b, _, _ in pplan] == [
        "shard00_q02_chunk0000",
        "shard00_q02_pilotpartial0001",
    ], pplan
    fplan = {b: (s, e) for b, s, e in _crossed_chunk_plan(10, 4, 0, 2, False)}  # full shard
    for b, s, e in pplan:
        if "pilotpartial" in b:
            assert b not in fplan, b  # the partial can never be skip-matched
        else:
            assert fplan[b] == (s, e), (b, fplan[b], (s, e))  # identical slice => resumable
    # ladder families never collide (the q tag is in every name); an exactly-
    # divisible pilot has NO partial (all chunks fleet-resumable)
    p_q3 = {b for b, _, _ in _crossed_chunk_plan(9, 4, 0, 3, True)}
    p_q2 = {b for b, _, _ in _crossed_chunk_plan(6, 4, 0, 2, True)}
    assert not (p_q3 & p_q2), p_q3 & p_q2
    assert all("_chunk" in b for b, _, _ in _crossed_chunk_plan(8, 4, 0, 2, True))
    # ladder-aware coverage view (blocker-1(b)): a clean q<2 capture of the 5x3
    # grid reads 10/10 under the view (PASS) vs 10/15 = 0.67 < 0.95 raw (FAIL)
    cov_doc = {"sets": {"train": {"n": 15, "ci": list(range(15))}}}
    lv = CR._ladder_split_view(cov_doc, 3, 2)
    assert lv["sets"]["train"]["n"] == 10, lv
    assert CR._ladder_split_view(cov_doc, 3, 3) is cov_doc  # identity off-ladder
    assert MTF._coverage_shortfalls({"train": np.arange(10)}, cov_doc, 0.95)
    assert not MTF._coverage_shortfalls({"train": np.arange(10)}, lv, 0.95)

    # ── G1 query-ladder e2e (r2 blocker-1): descoped pilot (n_q=2) against a
    # fake done index holding the FULL-grid q3 family — under the pre-r2 naming
    # those entries collide, every chunk skips, pilot_px stays empty, and the
    # pilot dies as a spurious G2 rc-30; q-tagged names keep the re-run fresh.
    assert q3_pilot_files, "expected q03-family files from the full-grid leg"
    xl = argparse.Namespace(
        **{**vars(xns), "queries_per_prefix": 2, "smoke_done_index": q3_pilot_files}
    )
    rc = run_crossed_capture(xl, smoke_sae=tiny_sae)
    assert rc == 0, "ladder pilot skip-collided with the stale q3 family (the rc-30 chain)"
    lpm = json.loads((xbase / CROSSED_PILOT_META_NAME).read_text())
    assert lpm["queries_per_prefix_realized"] == 2 and lpm["queries_per_prefix_grid"] == 3, lpm
    assert lpm["gate_g2"]["prefix_varies_ok"], lpm["gate_g2"]
    assert (xscratch / "shard00_q02_pilotpartial0001.pt").exists()  # 6 rows @ shard_size 4
    # fleet shard 0 at the SAME n_q: full pilot chunk skips by name; the partial
    # range re-captures under the fleet's `_chunk` name (no row hole).
    q2_pilot_files = sorted(p.name for p in xscratch.iterdir() if "_q02_" in p.name)
    mt_q2 = (xscratch / "shard00_q02_chunk0000.pt").stat().st_mtime_ns
    rc = run_crossed_capture(
        argparse.Namespace(
            **{
                **vars(xns),
                "queries_per_prefix": 2,
                "pilot_cap": 0,
                "smoke_done_index": q3_pilot_files + q2_pilot_files,
            }
        ),
        smoke_sae=tiny_sae,
    )
    assert rc == 0
    assert (xscratch / "shard00_q02_chunk0000.pt").stat().st_mtime_ns == mt_q2
    assert (xscratch / "shard00_q02_chunk0001.pt").exists(), "ladder fleet left a row hole"
    rc = run_crossed_capture(
        argparse.Namespace(
            **{**vars(xns), "queries_per_prefix": 2, "shard_index": 1, "pilot_cap": 0}
        ),
        smoke_sae=tiny_sae,
    )
    assert rc == 0
    # ladder reads: scratch now holds BOTH families + pilotpartial files; the
    # scanner keys on realized n_q=2 (pilot meta) and coverage on the ladder view.
    lr = argparse.Namespace(
        **{
            **vars(rargs),
            "out_eval": xbase / "crossed_eval_ladder",
            "out_local": xbase / "crossed_reads_ladder",
            "mm_dir": xbase / "crossed_reads_ladder" / "mm",
        }
    )
    rc = CR.run_reads(lr)
    assert rc == 0, "ladder reads must pass the 0.95 coverage floor on a clean q2 capture"
    lcur = json.loads((Path(lr.mm_dir) / "cursor.json").read_text())
    assert lcur["n_rows"] == 10, lcur  # 5 prefixes x 2 realized queries; q3 chunks excluded
    la = json.loads((Path(lr.out_eval) / "anova.json").read_text())
    assert la["n_queries"] == 2, la["n_queries"]
    lf = json.loads((Path(lr.out_eval) / "crossed_fits.json").read_text())
    assert len(lf["cells"]) == 4 * 2, sorted(lf["cells"])
    logger.info(
        "[smoke] crossed leg OK — manifest -> capture (2 shards; resume-skip + "
        "pilot-partial re-capture) -> all six reads -> G1 ladder (q3 -> q2: no "
        "collision, no rc-30, ladder coverage PASS, mixed-family store excluded)"
    )


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
    ap.add_argument(
        "--kresample-primary-ci",
        default="hf",
        help="primary-draw ci source for the kresample admission gate: 'hf' (stage "
        "y_holdout L14.npz from the data repo; production default), a local .npz/JSON "
        "path, or 'none' (disable — loud override)",
    )
    ap.add_argument("--seeds", default=",".join(str(s) for s in KRESAMPLE_SEEDS))
    ap.add_argument(
        "--tiny-model", action="store_true", help="SMOKE ONLY: 2-layer from-config model"
    )
    # ── bare-query arm (follow-up `bare-query`, plan §4.1) ────────────────────────
    ap.add_argument(
        "--bare-query",
        action="store_true",
        help="bare-arm mode: batched forward-only capture of the FINAL user query "
        "under the empty-system render (#1092 convention); no generation",
    )
    ap.add_argument(
        "--upload-prefix",
        default="",
        help="HF prefix for THIS mode's uploads (default = --hf-prefix); the bare "
        "arm passes issue1738_multiturn/bare_query so the parent capture prefix "
        "is never clobbered (plan §4.1.4)",
    )
    ap.add_argument("--bare-batch", type=int, default=BARE_BATCH_DEFAULT)
    ap.add_argument("--bare-chunk", type=int, default=BARE_CHUNK_DEFAULT)
    ap.add_argument(
        "--bare-fence-min",
        type=float,
        default=BARE_FENCE_MIN_DEFAULT,
        help="G-B1: designed halt when the measured rate projects this shard past N minutes",
    )
    # ── crossed-multiturn-averaged round (follow-up fu3, plan v9 §4.1/§4.2) ───────
    ap.add_argument(
        "--build-crossed-manifest",
        action="store_true",
        help="P0: build + upload the crossed manifest (prefixes x shared bank)",
    )
    ap.add_argument(
        "--crossed-capture",
        action="store_true",
        help="S1: crossed generation + capture + SAE fold-in (shard BY PREFIX)",
    )
    # UPLOAD_PREFIX_EXEMPT: the crossed round's OWN self-contained prefix (plan v9 §10); parent prefixes are never written by these modes
    ap.add_argument("--crossed-hf-prefix", default=CROSSED_HF_PREFIX)
    ap.add_argument("--crossed-n-prefixes", type=int, default=CROSSED_N_PREFIXES)
    ap.add_argument("--crossed-n-queries", type=int, default=CROSSED_N_QUERIES)
    ap.add_argument("--crossed-g0-prefix-floor", type=int, default=CROSSED_G0_PREFIX_FLOOR)
    ap.add_argument("--crossed-g0-bank-floor", type=int, default=CROSSED_G0_BANK_FLOOR)
    ap.add_argument(
        "--crossed-bank-query-token-max", type=int, default=CROSSED_BANK_QUERY_TOKEN_MAX
    )
    ap.add_argument(
        "--queries-per-prefix",
        type=int,
        default=0,
        help="G1 ladder subset (20->12->10); 0 = the full manifest grid",
    )
    ap.add_argument("--crossed-fence-gpu-h", type=float, default=CROSSED_FENCE_GPU_H)
    ap.add_argument("--crossed-sae-layer", type=int, default=CROSSED_SAE_LAYER)
    ap.add_argument("--no-sae", action="store_true", help="crossed: skip the SAE fold-in outright")
    ap.add_argument(
        "--crossed-sae-force",
        choices=["", "on", "off"],
        default="",
        help="override the pilot SAE fitness verdict (smoke: 'on' — gate demoted to a log)",
    )
    ap.add_argument("--sae-cache-dir", type=Path, default=None)
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
    elif args.build_crossed_manifest:
        build_crossed_manifest(args)
        rc = 0
    elif args.crossed_capture:
        rc = run_crossed_capture(args)
    elif args.kresample:
        rc = run_kresample(args)
    elif args.bare_query:
        rc = run_bare_capture(args)
    else:
        rc = run_capture(args)
    # heavy C-extension entrypoint: explicit exit dodges the finalize-time
    # PyGILState_Release atexit race (#1689 gotcha).
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)


if __name__ == "__main__":
    main()
