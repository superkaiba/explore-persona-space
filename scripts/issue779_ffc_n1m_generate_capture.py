#!/usr/bin/env python3
"""Issue #779 inline follow-up (``fitter-fair-comparison-n1m``): corpus extension
to n_train ~= 1,000,000 + TRIMMED combined capture, MULTI-POD / MULTI-GPU SHARDED.

Extends the n50k round (``issue779_ffc_n50k_generate_capture.py``) along the SINGLE
variable ``n`` from 46,600 new contexts to a ~960,000-context NEW pool, spanning
TWO real-user corpora, with an HF-hosted sampling manifest so K capture shards
across M pods all read the SAME deterministic context list.

Everything below is the n50k recipe verbatim EXCEPT the four n1m-specific deltas:

1. **Corpus: LMSYS-exhaustion (phase A) + WildChat-1M balance (phase B).**
   ``build_new_pool`` re-derives the round-1 (5000) + n10k (6500) + n50k (46600)
   USED set from the deterministic LMSYS stream (via ``N50.sample_disjoint_n50k``),
   then (phase A) continues the SAME LMSYS corpus keeping every NEW disjoint
   first-turn to EXHAUSTION, then (phase B) tops up from ``allenai/WildChat-1M``
   (same non-empty-first-turn filter + string-disjointness) until the total NEW
   pool reaches ``--n-new`` (default 960,000). Per-context provenance
   (``lmsys`` | ``wildchat``) + per-corpus counts + stream positions land in the
   manifest. The kept prompt TEXT of every context persists to HF (the manifest
   IS that text-persistence for the prompts; per-chunk ``raw_completions`` persist
   the rollouts). External-stream checkpoint+resume per corpus (#1092): atomic
   pool file + fingerprint sidecar, periodic partial checkpoints, ``.skip()``
   fast-forward resume — a mid-stream crash never loses the whole kept pool.

2. **Near-dupe contamination gate.** Before a context is kept, ``NearDupeGate``
   rejects it if its normalized prompt EXACTLY matches, or has char-5-gram
   Jaccard >= 0.8 against, ANY of the 1,400 pinned val/test prompts (recovered
   DETERMINISTICALLY from the re-derived round-1 5,000 + the exact ``fixed_split``;
   the HF pass_b bundle is tensors-only). Dropped
   counts (exact vs near) land in the manifest meta. CPU, part of manifest build.

3. **Trimmed capture recipe.** Captures ONLY ``cx_last`` (last prompt token) +
   ``v_x`` (mean-response) at layers ``{14, 19, 26}`` — no ``cx_mean``, no
   pass-1/pass-2 summaries (the n50k fields the n1m fits never read). The two
   capture forwards REUSE the round-1 ``capture_context_vector`` /
   ``capture_answer_vector`` VERBATIM (same tokenization, same positions), passing
   ``layers=[14,19,26]`` so only those layers are materialized. ~86 KB/context fp32
   (2 fields x 3 layers x 3584 x 4 bytes) => ~82 GB total across ~960k contexts.
   Same ``.pt`` shard-chunk format + upload->verify->PURGE cycle as n50k; HF prefix
   ``issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture/``.

4. **Multi-pod sharding.** The manifest is built ONCE (``--build-sampling-manifest``,
   which ALSO uploads it to HF) and every shard reads it (``--manifest-from-hf``
   downloads it). ``--num-shards K --shard-index i`` slices the ordered new-pool
   into K contiguous ranges (``N50._shard_range``); the launcher's ``--shard-offset``
   maps pod k of M to global shard indices ``[k*8, k*8+8)`` of ``--num-shards 32``.
   Nothing but the manifest is shared across shards.

FAITHFUL-REUSE DEVIATION (recorded in metadata.deviations, one line): cx_last / v_x
are captured per-row (batch-1 forwards) via the round-1 capture functions, NOT
batched — the same deviation the n50k / n10k drivers recorded (a batched rewrite
would risk left-pad/bf16 padded-batch numeric divergence from the round-1 corpus,
the #779 r12 equivalence-gate class, breaking the "n is the only variable"
constraint). vLLM generation IS chunked (``N10._generate`` -> chunk 500).

Refusal-safety: LMSYS + WildChat are unscreened real-user corpora. This driver
NEVER prints or logs example context/rollout text — only counts, indices, corpus
tags, and sha256s. Do not add such logging.

GPU (H100/A100) per shard. NO judge/API calls. Fail loud — NaN never coerced.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
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
import issue779_ffc_n10k_generate_capture as N10  # noqa: E402
import issue779_ffc_n50k_generate_capture as N50  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue779_ffc_n1m_gc")

N_LAYERS = C.EXPECTED_LAYERS  # 28 (full capture range before trimming)
H_DIM = C.EXPECTED_HIDDEN  # 3584
DEFAULT_MODEL = N50.DEFAULT_MODEL  # Qwen/Qwen2.5-7B-Instruct (round-1 pass-B model)
LMSYS_REPO = N10.LMSYS_REPO  # lmsys/lmsys-chat-1m
WILDCHAT_REPO = "allenai/WildChat-1M"
HF_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m"
MANIFEST_SUBDIR = "sampling_manifest"  # HF: {HF_PREFIX}/sampling_manifest/{part_*.jsonl, meta.json}

# Chunk-upload batching (#779 n1m fleet HF-429 fix): accumulate K chunks locally,
# then upload the pending group as ONE upload_folder commit per artifact kind (2
# commits per batch instead of 2 per chunk). 32 shards x 2 commits x (chunks/K)
# keeps the fleet under HF's 256-commits/hr cap. Peak local footprint ~K*43 MB
# (.pt) + raw < 1 GB/shard, trivially under the RunPod MooseFS quota.
UPLOAD_BATCH = int(os.environ.get("EPM_N1M_UPLOAD_BATCH", "10"))

# The three prior phases to re-derive + EXCLUDE (n is the only variable).
N_ROUND1 = N50.N_ROUND1  # 5000
N_N10K = N50.N_N10K  # 6500
N_N50K = N50.N_N50K  # 46600
N_NEW_TARGET = 960_000  # total NEW pool (margin over the ~950k needed for the 1M fits)

# Trimmed capture: cx_last + v_x at these three layers ONLY.
CAPTURE_LAYERS = [14, 19, 26]

# pass_b bundle (val/test prompt source) — local cache + HF fallback (the brief's
# named path; the fits loader shares it).
PASS_B_LOCAL = PROJECT_ROOT / "data" / "issue_779" / "pass_b" / "train_context_vectors.pt"
PASS_B_HF_PATH = "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"

# Byte budget per manifest JSONL part (<9 MB => non-LFS on the HF data repo).
MANIFEST_PART_BYTES = 8 * 1024 * 1024
STREAM_CHECKPOINT_EVERY = 50_000  # partial-pool checkpoint cadence (kept rows)
NEAR_DUPE_NGRAM = 5
NEAR_DUPE_JACCARD = 0.8
# Bumped when the selection/filter recipe changes so a stale stream cache re-streams.
FILTER_RECIPE_VERSION = "n1m-v1"


# ── HF pass_b fallback + val/test prompt recovery (near-dupe targets) ────────────


def _load_pass_b_bundle(local_path: Path):
    """mmap-load the pass_b bundle; fetch from HF (the analysis_tensors path) if
    absent locally, cross-device-safe os.replace into the local cache."""
    if not local_path.exists():
        from huggingface_hub import hf_hub_download

        logger.info("[pass_b] %s absent; fetching %s from HF", local_path, PASS_B_HF_PATH)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        got = Path(
            hf_hub_download(
                C.HF_DATA_REPO,
                filename=PASS_B_HF_PATH,
                repo_type="dataset",
                local_dir=local_path.parent,
            )
        )
        if got != local_path:
            os.replace(got, local_path)
    import issue779_fitter_fair_comparison as F

    return F._mmap_load(local_path)


def _valtest_prompts_from_round1(round1: list[str], *, check_ctx0: bool = True) -> list[str]:
    """The 1,400 pinned val+test prompt strings (the near-dupe targets), recovered
    DETERMINISTICALLY from the re-derived original 5,000 LMSYS round-1 contexts.

    The HF ``analysis_tensors/pass_b`` bundle is tensors-only — its ``prompts`` field
    is stripped on the analysis-tensors upload path (``_ANALYSIS_TENSORS_TEXT_FIELDS``
    in issue779_collect), and ``source``/``metadata`` carry no per-row text — so the
    val/test prompt TEXT cannot come from the bundle. Instead ``round1`` (returned by
    ``N50.sample_disjoint_n50k``, which streams those rows anyway before excluding them)
    IS the pass_b prompt list in pass_b row order; applying the ORIGINAL round's
    ``fixed_split(5000, 3600, 400, 1000, 42)`` val/test INDICES to it yields exactly the
    1,400 pinned targets. ``check_ctx0`` guards the stream-ordering re-derivation on the
    real path (mirrors N50.build_manifest's assert); the CPU smoke passes it a synthetic
    round1 and sets it False.
    """
    import issue779_fitter_fair_comparison as F

    n = len(round1)
    assert n == N_ROUND1, f"round1 has {n} prompts but the fixed_split anchor is {N_ROUND1}"
    if check_ctx0:
        norm = " ".join(round1[0].lower().split()).rstrip(".?!,")
        assert norm == N10.EXPECTED_CTX0_PROMPT, (
            f"round-1 ctx0 re-derivation drift: got {round1[0][:80]!r} — the LMSYS stream "
            "ordering changed; the val/test prompt re-derivation is no longer trustworthy"
        )
    _r1, val, test = F.fixed_split(N_ROUND1, N_ROUND1 - 400 - 1000, 400, 1000, F.SPLIT_SEED)
    idx = list(val) + list(test)
    assert len(idx) == 1400, len(idx)
    return [round1[i] for i in idx]


# ── near-dupe gate (exact-normalized + char-ngram Jaccard, inverted index) ───────


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _char_ngrams(norm_text: str, n: int) -> frozenset[str]:
    if len(norm_text) < n:
        return frozenset([norm_text]) if norm_text else frozenset()
    return frozenset(norm_text[i : i + n] for i in range(len(norm_text) - n + 1))


class NearDupeGate:
    """Reject a candidate prompt that is an exact (normalized) match or a
    char-ngram Jaccard >= ``thresh`` near-duplicate of ANY target (val/test)
    prompt. Direct Jaccard, made feasible over 1,400 targets by an ngram
    inverted index (a candidate is compared only against targets sharing >=1
    ngram). Refusal-safe: stores/returns no prompt TEXT, only counts."""

    def __init__(
        self, targets: list[str], ngram: int = NEAR_DUPE_NGRAM, thresh: float = NEAR_DUPE_JACCARD
    ):
        self.ngram = int(ngram)
        self.thresh = float(thresh)
        self.exact: set[str] = set()
        self.target_ngrams: list[frozenset[str]] = []
        self.inv: dict[str, set[int]] = defaultdict(set)
        for ti, t in enumerate(targets):
            n = _norm(t)
            self.exact.add(n)
            g = _char_ngrams(n, self.ngram)
            self.target_ngrams.append(g)
            for ng in g:
                self.inv[ng].add(ti)
        self.n_exact_drop = 0
        self.n_near_drop = 0

    def is_dupe(self, prompt: str) -> bool:
        n = _norm(prompt)
        if n in self.exact:
            self.n_exact_drop += 1
            return True
        g = _char_ngrams(n, self.ngram)
        if not g:
            return False
        cand: set[int] = set()
        for ng in g:
            cand |= self.inv.get(ng, set())
        for ti in cand:
            tg = self.target_ngrams[ti]
            inter = len(g & tg)
            if inter == 0:
                continue
            union = len(g) + len(tg) - inter
            if union and inter / union >= self.thresh:
                self.n_near_drop += 1
                return True
        return False

    def stats(self) -> dict:
        return {
            "ngram": self.ngram,
            "jaccard_thresh": self.thresh,
            "n_targets": len(self.target_ngrams),
            "n_exact_drop": self.n_exact_drop,
            "n_near_drop": self.n_near_drop,
        }


# ── external-stream checkpoint + fingerprint resume (#1092) ──────────────────────


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict]:
    # text-mode iteration, never .splitlines() (#825/#950 U+2028 rule).
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip("\n")
            if s:
                out.append(json.loads(s))
    return out


def _stream_corpus(
    repo: str,
    corpus_tag: str,
    keep_pred,
    target: int | None,
    cache_dir: Path,
    fingerprint: dict,
    *,
    resume: bool,
    smoke_stream=None,
) -> list[dict]:
    """Stream one corpus keeping first-turn prompts that pass ``keep_pred``, with a
    per-corpus on-disk checkpoint + fingerprint resume.

    ``target`` = keep until this many kept (None = stream to EXHAUSTION). Persists
    the kept pool to ``{cache_dir}/{corpus_tag}.jsonl`` + a ``.meta.json`` sidecar
    carrying the fingerprint, consumed-row count, and a ``complete`` flag; partial
    checkpoints every ``STREAM_CHECKPOINT_EVERY`` kept rows. On startup an EXACT
    fingerprint match reloads the pool and either returns (complete) or
    ``.skip(consumed)``-resumes the stream. ``smoke_stream`` (an in-memory list of
    row dicts) injects a synthetic corpus for the CPU smoke (no real data).

    Refusal-safe: only the kept first-turn prompt text is persisted (the manifest
    IS its required text-persistence), never logged.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    pool_path = cache_dir / f"{corpus_tag}.jsonl"
    meta_path = cache_dir / f"{corpus_tag}.meta.json"
    kept: list[dict] = []
    consumed = 0
    if resume and pool_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fingerprint:
            kept = _read_jsonl(pool_path)
            consumed = int(meta.get("consumed", 0))
            if meta.get("complete"):
                logger.info(
                    "[stream %s] RESUMED complete cache: %d kept (stream skipped)",
                    corpus_tag,
                    len(kept),
                )
                return kept
            logger.info(
                "[stream %s] RESUMED partial cache: %d kept, fast-forward %d rows",
                corpus_tag,
                len(kept),
                consumed,
            )
        else:
            logger.info("[stream %s] fingerprint MISMATCH; re-streaming from scratch", corpus_tag)
            kept, consumed = [], 0

    if smoke_stream is not None:
        it = iter(smoke_stream)
        ds = None
    else:
        from datasets import load_dataset

        ds = load_dataset(repo, split="train", streaming=True)
        it = iter(ds.skip(consumed) if consumed else ds)

    def _flush(complete: bool) -> None:
        _atomic_write_jsonl(pool_path, kept)
        _atomic_write_json(
            meta_path,
            {
                "fingerprint": fingerprint,
                "consumed": consumed,
                "kept": len(kept),
                "complete": complete,
            },
        )

    seen: set[str] = {r["prompt"] for r in kept}
    row = None
    while target is None or len(kept) < target:
        row = next(it, None)
        if row is None:  # exhaustion
            break
        consumed += 1
        p = N10._first_user_turn(row)
        if p and p not in seen and keep_pred(p):
            kept.append({"prompt": p, "corpus": corpus_tag, "stream_pos": consumed - 1})
            seen.add(p)
            if len(kept) % STREAM_CHECKPOINT_EVERY == 0:
                _flush(complete=False)
                logger.info(
                    "[stream %s] checkpoint: %d kept / %d consumed", corpus_tag, len(kept), consumed
                )
    _flush(complete=True)
    logger.info("[stream %s] done: %d kept / %d consumed", corpus_tag, len(kept), consumed)
    # release the streaming dataset + last row before shutdown (#952 rc=134 guard)
    if ds is not None:
        del it, ds, row
        gc.collect()
    return kept


# ── manifest build (once) + read (per shard) ─────────────────────────────────────


def build_new_pool(args, *, smoke_lmsys=None, smoke_wildchat=None) -> tuple[list[dict], dict]:
    """The ordered NEW-context pool + build meta.

    Phase A: re-derive the round-1 + n10k + n50k USED set, then stream LMSYS to
    EXHAUSTION keeping new disjoint non-near-dupe first-turns. Phase B: stream
    WildChat until the total pool reaches ``args.n_new``. Provenance per context.
    """
    # Phase 0: re-derive the used set (round-1 + n10k + n50k) FIRST — exact by
    # construction. round1 doubles as the pass_b prompt list (pass_b row order), from
    # which the 1,400 near-dupe val/test targets are recovered deterministically (the
    # HF pass_b bundle is tensors-only; its prompts field is stripped at upload).
    if smoke_lmsys is not None:
        used_man = N50.sample_disjoint_n50k(
            args.skip_round1, args.n_n10k, args.n_n50k, stream_iter=list(smoke_lmsys)
        )
    else:
        used_man = N50.sample_disjoint_n50k(args.skip_round1, args.n_n10k, args.n_n50k)
    used: set[str] = set(used_man["round1"]) | set(used_man["n10k"]) | set(used_man["new"])
    logger.info("[pool] used set (round1+n10k+n50k) = %d prompts", len(used))

    # val/test near-dupe targets: real path re-derives from round1; smoke uses injected.
    if smoke_lmsys is None:
        valtest = _valtest_prompts_from_round1(used_man["round1"])
    else:
        valtest = list(args.smoke_valtest or [])
    gate = NearDupeGate(valtest)

    used_fp = {
        "round1_sha": used_man["round1_prompt_sha256"],
        "n10k_sha": used_man["n10k_prompt_sha256"],
        "n50k_sha": used_man["new_prompt_sha256"],
        "recipe": FILTER_RECIPE_VERSION,
        "near_dupe": {"ngram": NEAR_DUPE_NGRAM, "thresh": NEAR_DUPE_JACCARD},
    }
    cache_dir = args.out_dir / "stream_cache"

    # Phase A: LMSYS to exhaustion.
    def keep_lmsys(p: str) -> bool:
        return p not in used and not gate.is_dupe(p)

    lmsys_pool = _stream_corpus(
        LMSYS_REPO,
        "lmsys",
        keep_lmsys,
        None,  # to exhaustion
        cache_dir,
        {**used_fp, "phase": "A-lmsys"},
        resume=not args.no_resume_stream,
        smoke_stream=smoke_lmsys,
    )
    lmsys_set = {r["prompt"] for r in lmsys_pool}

    # Phase B: WildChat top-up to reach n_new total.
    remaining = max(0, args.n_new - len(lmsys_pool))

    def keep_wildchat(p: str) -> bool:
        return p not in used and p not in lmsys_set and not gate.is_dupe(p)

    wildchat_pool: list[dict] = []
    if remaining > 0:
        wildchat_pool = _stream_corpus(
            WILDCHAT_REPO,
            "wildchat",
            keep_wildchat,
            remaining,
            cache_dir,
            {**used_fp, "phase": "B-wildchat", "lmsys_kept": len(lmsys_pool)},
            resume=not args.no_resume_stream,
            smoke_stream=smoke_wildchat,
        )
    else:
        logger.info("[pool] LMSYS alone filled the %d target; no WildChat top-up", args.n_new)

    pool = lmsys_pool + wildchat_pool
    for i, r in enumerate(pool):
        r["i"] = i  # global new-context index (== chunk `ci`)
    meta = {
        "n_new": len(pool),
        "n_new_target": int(args.n_new),
        "n_lmsys": len(lmsys_pool),
        "n_wildchat": len(wildchat_pool),
        "used_set_size": len(used),
        "near_dupe": gate.stats(),
        "used_shas": {
            "round1": used_man["round1_prompt_sha256"],
            "n10k": used_man["n10k_prompt_sha256"],
            "n50k": used_man["new_prompt_sha256"],
        },
        "new_prompt_sha256": N10._sha_prompts([r["prompt"] for r in pool]),
        "corpora": {"lmsys": LMSYS_REPO, "wildchat": WILDCHAT_REPO},
        "capture_layers": list(CAPTURE_LAYERS),
        "model": args.model,
        "recipe_version": FILTER_RECIPE_VERSION,
    }
    return pool, meta


def _write_manifest_parts(manifest_dir: Path, pool: list[dict], meta: dict) -> int:
    """Write the ordered pool as byte-budgeted JSONL parts + meta.json. Returns
    the number of parts (each <9 MB => non-LFS on the HF data repo)."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for old in manifest_dir.glob("part_*.jsonl"):
        old.unlink()
    part_idx = 0
    buf: list[str] = []
    buf_bytes = 0

    def _flush_part() -> None:
        nonlocal part_idx, buf, buf_bytes
        if not buf:
            return
        (manifest_dir / f"part_{part_idx:05d}.jsonl").write_text("".join(buf), encoding="utf-8")
        part_idx += 1
        buf, buf_bytes = [], 0

    for r in pool:
        line = json.dumps(r, ensure_ascii=False) + "\n"
        b = len(line.encode("utf-8"))
        if buf and buf_bytes + b > MANIFEST_PART_BYTES:
            _flush_part()
        buf.append(line)
        buf_bytes += b
    _flush_part()
    meta = {**meta, "n_parts": part_idx, "part_bytes_budget": MANIFEST_PART_BYTES}
    _atomic_write_json(manifest_dir / "meta.json", meta)
    return part_idx


def read_manifest_pool(manifest_dir: Path) -> tuple[list[dict], dict]:
    """Read the ordered new-context pool + meta from a local manifest dir (parts
    in filename order == global index order). Shared by the fits driver."""
    meta_path = manifest_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"manifest meta {meta_path} absent — build the manifest first")
    meta = json.loads(meta_path.read_text())
    pool: list[dict] = []
    for part in sorted(manifest_dir.glob("part_*.jsonl")):
        pool.extend(_read_jsonl(part))
    assert len(pool) == meta["n_new"], (len(pool), meta["n_new"])
    # global-index alignment invariant (parts are contiguous, filename-ordered)
    for expected, r in enumerate(pool):
        if r.get("i") != expected:
            raise SystemExit(
                f"manifest pool index drift at row {expected}: stored i={r.get('i')} "
                "(parts out of order or corrupt)"
            )
    return pool, meta


def _upload_manifest(manifest_dir: Path, hf_prefix: str) -> None:
    url = hub._upload(
        manifest_dir,
        repo_id=C.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{hf_prefix}/{MANIFEST_SUBDIR}",
    )
    if not url:
        raise RuntimeError(f"manifest upload to {hf_prefix}/{MANIFEST_SUBDIR} returned no URL")
    logger.info("[manifest] uploaded %s -> %s/%s", manifest_dir, hf_prefix, MANIFEST_SUBDIR)


def _manifest_complete_locally(dest: Path) -> bool:
    """True iff ``dest`` holds meta.json + all ``n_parts`` part files it names.

    The completeness predicate for the fleet-safe download short-circuit: a dir
    is complete only when meta.json parses AND the ``part_*.jsonl`` count matches
    meta's ``n_parts`` (so a half-written dir never reads as complete)."""
    meta_path = dest / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    n_parts = meta.get("n_parts")
    if not isinstance(n_parts, int) or n_parts < 0:
        return False
    return len(list(dest.glob("part_*.jsonl"))) == n_parts


def _download_manifest(hf_prefix: str, dest: Path) -> Path:
    """Download the HF-hosted manifest folder (parts + meta) to ``dest``.

    Fleet-safe: N shards per pod may call this concurrently against the SAME
    local dir (8 shards raced the same 88 files at fleet launch — the winner's
    os.replace moved each source into place, the losers' os.replace then hit
    FileNotFoundError and 21/32 shards died). Three guards: (1) short-circuit
    if the manifest is already complete locally (no lock, no network); (2)
    serialize download+move behind an exclusive flock on ``<dest>/.download.lock``
    with a post-acquire re-check (the shard that blocked on the lock returns the
    winner's materialized dir instead of re-downloading); (3) tolerate a missing
    ``got`` when ``target`` already exists (another process moved it)."""
    import fcntl

    from huggingface_hub import HfApi, hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)

    # (1) fast path: already complete locally — no lock, no network.
    if _manifest_complete_locally(dest):
        logger.info("[manifest] already complete locally at %s; skipping download", dest)
        return dest

    prefix = f"{hf_prefix}/{MANIFEST_SUBDIR}"
    lock_path = dest / ".download.lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        # (2) re-check under the lock — a concurrent shard we blocked on may have
        #     finished the full download while we waited.
        if _manifest_complete_locally(dest):
            logger.info("[manifest] completed by a concurrent shard; using %s", dest)
            return dest

        names = [
            f.path
            for f in HfApi().list_repo_tree(
                C.HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
            )
            if getattr(f, "size", None) is not None
        ]
        if not names:
            raise SystemExit(
                f"no manifest files under HF {prefix} — build + upload the manifest first"
            )
        for name in names:
            base = name.rsplit("/", 1)[-1]
            got = Path(
                hf_hub_download(
                    C.HF_DATA_REPO, filename=name, repo_type="dataset", local_dir=dest.parent
                )
            )
            target = dest / base
            if got != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                # (3) tolerate a missing source when the target already landed
                #     (another process moved it before we ran).
                try:
                    os.replace(got, target)
                except FileNotFoundError:
                    if not target.exists():
                        raise
        logger.info("[manifest] downloaded %d files from HF %s -> %s", len(names), prefix, dest)
        return dest


def build_manifest(args) -> dict:
    C.phase("sample")
    pool, meta = build_new_pool(args)
    manifest_dir = args.out_dir / MANIFEST_SUBDIR
    n_parts = _write_manifest_parts(manifest_dir, pool, meta)
    logger.info(
        "[manifest] wrote %d contexts (%d lmsys + %d wildchat) in %d parts; near-dupe drops: %s",
        meta["n_new"],
        meta["n_lmsys"],
        meta["n_wildchat"],
        n_parts,
        meta["near_dupe"],
    )
    if not args.no_upload:
        _upload_manifest(manifest_dir, args.hf_prefix)
    C.phase("manifest-done")
    return meta


def _resolve_manifest_dir(args) -> Path:
    local = args.out_dir / MANIFEST_SUBDIR
    if args.manifest_from_hf:
        return _download_manifest(args.hf_prefix, local)
    if not (local / "meta.json").exists():
        raise SystemExit(
            f"manifest {local}/meta.json absent — run --build-sampling-manifest first, "
            "or pass --manifest-from-hf to fetch the HF-hosted manifest"
        )
    return local


# ── trimmed per-row capture (cx_last + v_x at CAPTURE_LAYERS) ─────────────────────


def _capture_shard_trimmed(hf, tok, prompts, responses, ci_base_global, cis, layers):
    """Capture cx_last + v_x at ``layers`` for one chunk (per-row forwards via the
    round-1 capture functions). ``cis`` are the GLOBAL new-context indices (manifest
    order) aligned to ``prompts``. Keeps only rows with a non-empty response
    (v_x computable), matching run_pass_b / N10 kept_idx."""
    rows = []
    for p, resp, ci in zip(prompts, responses, cis, strict=True):
        msgs = [{"role": "user", "content": p}]
        cx = COL.capture_context_vector(hf, tok, msgs, layers)
        av = COL.capture_answer_vector(hf, tok, msgs, resp, layers, {}, keep_per_token=False)
        if av is None:  # empty response
            continue
        rows.append(
            {
                "ci": int(ci),
                "prompt": p,
                "response": resp,
                "cx_last": cx["last"],  # (len(layers), H)
                "v_x": av["v_x"],  # (len(layers), H)
            }
        )
    return rows


def _stack_chunk(rows, layers, shard_index, chunk_idx) -> dict:
    """Stack per-row trimmed capture dicts into one mmap-slice-friendly bundle."""
    return {
        "cx_last": torch.stack([r["cx_last"] for r in rows]),  # (n, 3, H)
        "v_x": torch.stack([r["v_x"] for r in rows]),  # (n, 3, H)
        "ci": [int(r["ci"]) for r in rows],  # GLOBAL new-context index (manifest order)
        "prompts": [r["prompt"] for r in rows],
        "layers": list(layers),
        "shard_index": int(shard_index),
        "chunk": int(chunk_idx),
    }


def _flush_upload_batch(
    scratch: Path, prefix: str, pt_names: list[str], raw_names: list[str]
) -> None:
    """Upload a pending batch of chunk files as ONE ``upload_folder`` commit per
    artifact kind, verify, then purge the local files.

    The 429-fix for the 32-shard fleet: instead of two Hub commits per chunk
    (capture .pt + raw .json), accumulate ``UPLOAD_BATCH`` chunks and flush the
    group in TWO commits total (one per kind). ``hub._upload_folder_filtered``
    composes exactly ONE ``create_commit`` per kind (no per-file recursive
    tree-listing) AND does an EXACT expected-set membership verify on a fresh
    scoped listing — so a partial commit returns ``""`` and we raise BEFORE any
    purge. The .pt kind additionally sha256-verifies each file against the Hub
    LFS metadata (corruption guard; the raw-json kind is non-LFS text, so the
    exact-set presence verify is the check, matching N50._upload_raw). Purge
    happens ONLY after the (verified) non-empty URL, so a failed flush loses
    nothing — the caller keeps the local files and the resume-skip re-uploads
    the same chunk next run. Chunk filenames are IDENTICAL to the unbatched
    path (``shard{SI:02d}_chunk{ci:04d}.{pt,json}``), so resume compatibility
    with already-uploaded chunks is preserved.

    No-ops on empty batches (both lists empty). Does NOT clear the input lists —
    the caller clears them after this returns (so a raise leaves the pending set
    intact for a retry/report)."""
    if not pt_names and not raw_names:
        return

    # (1) capture .pt batch -> ONE commit under final_token_capture/, sha-verified.
    if pt_names:
        local_shas = {n: N50._sha256_file(scratch / n) for n in pt_names}
        url = hub._upload_folder_filtered(
            scratch,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{prefix}/final_token_capture",
            allow_patterns=list(pt_names),
            expected_repo_paths=[f"{prefix}/final_token_capture/{n}" for n in pt_names],
        )
        if not url:  # _upload_folder_filtered fail-soft returns "" — fail loud here
            raise RuntimeError(
                f"batch upload of {len(pt_names)} capture .pt to "
                f"{prefix}/final_token_capture returned no URL"
            )
        remote = N50._remote_index(f"{prefix}/final_token_capture")
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

    # (2) raw json batch -> ONE commit under raw_completions/ (text is NEVER
    #     discardable; the exact-set presence verify inside the helper is the check).
    if raw_names:
        url = hub._upload_folder_filtered(
            scratch,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{prefix}/raw_completions",
            allow_patterns=list(raw_names),
            expected_repo_paths=[f"{prefix}/raw_completions/{n}" for n in raw_names],
        )
        if not url:
            raise RuntimeError(
                f"batch upload of {len(raw_names)} raw_completions to "
                f"{prefix}/raw_completions returned no URL"
            )
        for n in raw_names:
            (scratch / n).unlink()
        logger.info(
            "[upload] batch of %d raw_completions verified (presence) + purged", len(raw_names)
        )


def _build_capture_engine(args):
    """Build the vLLM capture engine with the default-off H100 long-prompt
    hang/IMA mitigation knobs (byte-identical engine args when the flags are
    unset; same one-config-for-the-round discipline as n50k, gotchas #1092)."""
    from explore_persona_space.eval.generation import create_vllm_engine

    llm_kwargs: dict = {}
    if os.environ.get("EPM_VLLM_ENFORCE_EAGER") == "1":
        llm_kwargs["enforce_eager"] = True
    if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING") == "1":
        llm_kwargs["enable_prefix_caching"] = False
    if llm_kwargs:
        logger.info("[engine-knobs] %s", llm_kwargs)
    return create_vllm_engine(args.model, max_model_len=8192, seed=42, **llm_kwargs)


def _capture_stage_chunk(
    hf, tok, llm, chunk, layers, scratch, name, raw_name, shard_index, chunk_idx, global_base
) -> int:
    """Generate + trimmed-capture one chunk and stage its .pt + raw json into
    ``scratch`` (filenames ``name`` / ``raw_name``). Returns the kept-row count
    (0 = all-empty responses; nothing written, caller skips this chunk)."""
    chunk_prompts = [r["prompt"] for r in chunk]
    chunk_cis = [int(r["i"]) for r in chunk]
    responses = N10._generate(llm, tok, chunk_prompts)
    rows = _capture_shard_trimmed(hf, tok, chunk_prompts, responses, global_base, chunk_cis, layers)
    if not rows:
        return 0
    for fld in ("cx_last", "v_x"):
        for r in rows:
            assert r[fld].shape == (len(layers), H_DIM), (fld, r[fld].shape)
    torch.save(_stack_chunk(rows, layers, shard_index, chunk_idx), scratch / name)
    C.write_json_atomic(
        scratch / raw_name,
        {
            "shard_index": shard_index,
            "chunk": chunk_idx,
            "rows": [
                {"ci": int(r["ci"]), "prompt": r["prompt"], "response": r["response"]} for r in rows
            ],
        },
    )
    return len(rows)


def run_capture(args) -> int:
    manifest_dir = _resolve_manifest_dir(args)
    pool, _meta = read_manifest_pool(manifest_dir)
    n_total = len(pool)
    start, end = N50._shard_range(n_total, args.num_shards, args.shard_index)
    shard_pool = pool[start:end]
    layers = list(CAPTURE_LAYERS)
    logger.info(
        "[shard %d/%d] range [%d, %d) = %d contexts (%d total pool)",
        args.shard_index,
        args.num_shards,
        start,
        end,
        len(shard_pool),
        n_total,
    )
    if not shard_pool:
        logger.info("[shard %d] empty range; nothing to do", args.shard_index)
        C.phase("done")
        return 0

    scratch = args.out_dir / "shards"
    scratch.mkdir(parents=True, exist_ok=True)

    # Resume: chunks whose .pt AND raw json are already on the Hub are skipped.
    done_pt = set(N50._remote_index(f"{args.hf_prefix}/final_token_capture"))
    done_raw = set(N50._remote_index(f"{args.hf_prefix}/raw_completions"))

    C.phase("load_model")
    tok, hf = N50.N10.load_models(args.model, args.device)
    llm = _build_capture_engine(args) if args.device == "cuda" else None

    C.phase("capture")
    n_sub = (len(shard_pool) + args.shard_size - 1) // args.shard_size
    kept_total = 0
    # Batch pending chunk files; flush every UPLOAD_BATCH chunks + at shard end
    # (2 Hub commits per batch instead of 2 per chunk — the fleet 429 fix).
    pending_pt: list[str] = []
    pending_raw: list[str] = []

    def _flush_pending() -> None:
        if args.no_upload or not pending_pt:
            return
        _flush_upload_batch(scratch, args.hf_prefix, pending_pt, pending_raw)
        pending_pt.clear()
        pending_raw.clear()

    def _on_sigterm(signum, frame):
        # Raise so the try/except below best-effort-flushes the pending batch
        # before the process exits (pod stop / watcher kill).
        raise SystemExit(f"SIGTERM ({signum}) received — flushing pending upload batch")

    prev_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        for ci_idx, s in enumerate(range(0, len(shard_pool), args.shard_size)):
            name = f"shard{args.shard_index:02d}_chunk{ci_idx:04d}.pt"
            raw_name = f"shard{args.shard_index:02d}_chunk{ci_idx:04d}.json"
            if name in done_pt and raw_name in done_raw:
                logger.info(
                    "[shard %d] chunk %d/%d already on Hub; skip",
                    args.shard_index,
                    ci_idx + 1,
                    n_sub,
                )
                continue
            chunk = shard_pool[s : s + args.shard_size]
            ts = time.time()
            n_kept = _capture_stage_chunk(
                hf,
                tok,
                llm,
                chunk,
                layers,
                scratch,
                name,
                raw_name,
                args.shard_index,
                ci_idx,
                start + s,
            )
            if not n_kept:
                logger.warning(
                    "[shard %d] chunk %d: 0 kept rows (all empty responses); skip",
                    args.shard_index,
                    ci_idx,
                )
                continue
            kept_total += n_kept
            if not args.no_upload:
                pending_pt.append(name)
                pending_raw.append(raw_name)
                if len(pending_pt) >= UPLOAD_BATCH:
                    _flush_pending()
            logger.info(
                "[shard %d] chunk %d/%d: %d/%d kept (%.0fs)",
                args.shard_index,
                ci_idx + 1,
                n_sub,
                n_kept,
                len(chunk),
                time.time() - ts,
            )
        # Final partial batch — fail-loud (a flush failure here IS a real failure).
        _flush_pending()
    except BaseException:
        # Best-effort persist of the pending batch before propagating (SIGTERM /
        # crash); never mask the original exception with a flush failure.
        try:
            _flush_pending()
        except Exception:
            logger.exception(
                "[shard %d] best-effort pending-batch flush failed on exit", args.shard_index
            )
        raise
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)

    logger.info(
        "[shard %d] done: %d kept rows across %d chunks", args.shard_index, kept_total, n_sub
    )
    C.phase("done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Issue #779 n1m corpus extension + trimmed capture (multi-pod sharded)."
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip-round1", type=int, default=N_ROUND1)
    ap.add_argument("--n-n10k", type=int, default=N_N10K)
    ap.add_argument("--n-n50k", type=int, default=N_N50K)
    ap.add_argument("--n-new", type=int, default=N_NEW_TARGET)
    ap.add_argument("--num-shards", type=int, default=32)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=500, help="contexts per capture sub-chunk")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--pass-b", type=Path, default=PASS_B_LOCAL)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "issue_779" / "ffc_n1m")
    ap.add_argument("--hf-prefix", default=HF_PREFIX)
    ap.add_argument("--no-upload", action="store_true", help="capture locally, do NOT upload/purge")
    ap.add_argument(
        "--no-resume-stream", action="store_true", help="force a fresh manifest stream (no cache)"
    )
    ap.add_argument(
        "--build-sampling-manifest",
        action="store_true",
        help="stream + write + upload the sampling manifest, then exit (no capture)",
    )
    ap.add_argument(
        "--manifest-from-hf",
        action="store_true",
        help="download the HF-hosted manifest instead of requiring a local build",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny CPU logic smoke (synthetic streams)")
    args = ap.parse_args()
    args.smoke_valtest = None

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        return _smoke(args)
    if args.build_sampling_manifest:
        build_manifest(args)
        return 0
    return run_capture(args)


def _smoke_upload_batching(args) -> list[int]:
    """CPU smoke for chunk-upload batching (extracted from _smoke). Asserts
    ``_flush_upload_batch`` does ONE upload_folder commit per artifact kind with
    the EXACT batch filenames (unchanged from the unbatched path), verify+purges
    ONLY after the commit succeeds, and that the run_capture cadence flushes at
    ``UPLOAD_BATCH`` and again for the end-of-loop remainder. Returns the observed
    flush-size cadence."""
    from unittest import mock

    batch_dir = args.out_dir / "_smoke_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    bprefix = "smoke_prefix"
    pt_names = [f"shard00_chunk{i:04d}.pt" for i in range(3)]
    raw_names = [f"shard00_chunk{i:04d}.json" for i in range(3)]
    for n in pt_names + raw_names:
        (batch_dir / n).write_text("x")  # tiny stand-ins; content irrelevant to the commit shape

    folder_calls: list[tuple[str, tuple[str, ...]]] = []

    def _fake_folder_filtered(
        local_dir, *, repo_id, repo_type, path_in_repo, allow_patterns, expected_repo_paths, **kw
    ):
        folder_calls.append((path_in_repo, tuple(allow_patterns)))
        return f"{repo_id}/{path_in_repo}"  # non-empty == verified success

    # N50._remote_index is called ONLY for the .pt kind's sha cross-check; return
    # each pt's fake sha so the verify loop passes.
    def _fake_remote_index(prefix):
        if prefix.endswith("final_token_capture"):
            return {n: {"size": 1, "sha256": "deadbeef"} for n in pt_names}
        return {}

    with (
        mock.patch.object(hub, "_upload_folder_filtered", _fake_folder_filtered),
        mock.patch.object(N50, "_sha256_file", lambda _p: "deadbeef"),
        mock.patch.object(N50, "_remote_index", _fake_remote_index),
    ):
        _flush_upload_batch(batch_dir, bprefix, list(pt_names), list(raw_names))
    # two commits: one per artifact kind, each carrying the exact batch filenames.
    assert len(folder_calls) == 2, folder_calls
    assert folder_calls[0] == (f"{bprefix}/final_token_capture", tuple(pt_names)), folder_calls[0]
    assert folder_calls[1] == (f"{bprefix}/raw_completions", tuple(raw_names)), folder_calls[1]
    # verify+purge AFTER the commit: every batched file is unlinked.
    assert not any((batch_dir / n).exists() for n in pt_names + raw_names), "batch files not purged"

    # cadence: accumulate then flush at UPLOAD_BATCH and again for the remainder.
    flushed_sizes: list[int] = []
    pend: list[str] = []

    def _fake_flush() -> None:
        if pend:
            flushed_sizes.append(len(pend))
            pend.clear()

    for i in range(2 * UPLOAD_BATCH + 3):  # exact multiples + a partial remainder
        pend.append(f"shard00_chunk{i:04d}.pt")
        if len(pend) >= UPLOAD_BATCH:
            _fake_flush()
    _fake_flush()  # end-of-loop remainder
    assert flushed_sizes == [UPLOAD_BATCH, UPLOAD_BATCH, 3], flushed_sizes
    return flushed_sizes


def _smoke(args) -> int:
    """MODEL-FREE CPU logic smoke (no real corpora, no 7B model, refusal-safe).

    The capture forwards are GPU-bound (Qwen-2.5-7B) — per the GPU-bound-phase
    carve-out, the smoke exercises only the CPU-runnable portion: the near-dupe
    gate (planted dupe + near-dupe), the LMSYS-exhaustion + WildChat-top-up
    selection on synthetic streams, manifest determinism (two independent builds
    agree byte-for-byte), the manifest part/read roundtrip + global-index
    invariant, the shard-range partition-coverage invariant, and a signature
    check on the reused capture entrypoints. Real generation + capture run only
    on a GPU shard via the launcher."""
    import inspect

    logger.info(
        "[smoke] model-free CPU logic smoke (near-dupe + selection + manifest + shard-range)"
    )

    # (1) near-dupe gate: exact + near dupe dropped, distinct kept.
    targets = [f"what is the capital of country number {i} in the world atlas" for i in range(20)]
    gate = NearDupeGate(targets)
    assert gate.is_dupe(targets[3]), "exact val/test prompt must be dropped"
    near = targets[3] + "?"  # 1 new char-5-gram => Jaccard ~0.98, well over the 0.8 gate
    assert gate.is_dupe(near), "near-dupe (Jaccard>=0.8) must be dropped"
    assert not gate.is_dupe("a totally unrelated short question about cooking pasta"), (
        "a distinct prompt must be kept"
    )
    assert gate.n_exact_drop == 1 and gate.n_near_drop == 1, gate.stats()

    # (1b) FIX: the 1,400 val/test near-dupe targets are recovered from round1 (the HF
    #      pass_b bundle is tensors-only), NOT from a pass_b ``prompts`` field. Exercise
    #      the real-path recovery on a synthetic 5,000-row round1 (check_ctx0=False), and
    #      confirm the ctx0 guard rejects a drifted round1[0].
    synth_round1 = [f"round-1 context number {i}" for i in range(N_ROUND1)]
    vt = _valtest_prompts_from_round1(synth_round1, check_ctx0=False)
    assert len(vt) == 1400 and len(set(vt)) == 1400, len(vt)
    assert set(vt).issubset(synth_round1), "val/test targets must be round1 rows"
    guarded = False
    try:
        _valtest_prompts_from_round1(["a drifted first prompt", *synth_round1[1:]], check_ctx0=True)
    except AssertionError:
        guarded = True
    assert guarded, "ctx0 guard must reject a drifted round1[0]"

    # (2) selection on synthetic streams: re-derive used, LMSYS phase A to exhaustion,
    #     WildChat phase B top-up. Build the synthetic LMSYS stream so the first
    #     (r1 + n10k + n50k) non-empty first-turns re-derive as the used set, then a
    #     small tail of NEW lmsys, then WildChat fills to the target.
    def _row(text):
        return {"conversation": [{"content": text, "role": "user"}]}

    n_r1, n_n10, n_n50 = 4, 3, 2
    a = argparse.Namespace(
        skip_round1=n_r1,
        n_n10k=n_n10,
        n_n50k=n_n50,
        n_new=12,
        out_dir=args.out_dir / "_smoke",
        pass_b=args.pass_b,
        model="SYNTHETIC-SMOKE",
        no_resume_stream=True,
        smoke_valtest=list(targets),
    )
    a.out_dir.mkdir(parents=True, exist_ok=True)
    # ctx0 assert in N50.sample_disjoint_n50k only fires in build_manifest, not here.
    used_texts = [f"used prompt number {i}" for i in range(n_r1 + n_n10 + n_n50)]
    new_lmsys_texts = [f"new lmsys prompt number {i}" for i in range(6)]
    lmsys_stream = [_row(t) for t in used_texts + new_lmsys_texts]
    wildchat_stream = [_row(f"wildchat prompt number {i}") for i in range(20)]
    pool, meta = build_new_pool(a, smoke_lmsys=lmsys_stream, smoke_wildchat=wildchat_stream)
    assert meta["n_lmsys"] == 6, meta  # 6 new lmsys after the 9 used
    assert meta["n_new"] == 12 and meta["n_wildchat"] == 6, meta  # topped up to 12
    assert [r["corpus"] for r in pool[:6]] == ["lmsys"] * 6, "lmsys precede wildchat"
    assert [r["corpus"] for r in pool[6:]] == ["wildchat"] * 6, "wildchat top-up"

    # (3) manifest determinism: a SECOND independent build agrees byte-for-byte.
    a2 = argparse.Namespace(**{**vars(a), "out_dir": args.out_dir / "_smoke2"})
    a2.out_dir.mkdir(parents=True, exist_ok=True)
    pool2, meta2 = build_new_pool(a2, smoke_lmsys=lmsys_stream, smoke_wildchat=wildchat_stream)
    assert meta2["new_prompt_sha256"] == meta["new_prompt_sha256"], (
        "manifest build not deterministic"
    )
    assert [r["prompt"] for r in pool2] == [r["prompt"] for r in pool], "pool order differs"

    # (4) manifest parts write/read roundtrip + global-index invariant.
    md = a.out_dir / MANIFEST_SUBDIR
    _write_manifest_parts(md, pool, meta)
    rpool, rmeta = read_manifest_pool(md)
    assert [r["prompt"] for r in rpool] == [r["prompt"] for r in pool], (
        "manifest roundtrip mismatch"
    )
    assert rmeta["n_new"] == meta["n_new"]

    # (5) shard-range partition coverage: every context covered exactly once, in order.
    for k in (2, 3, 4):
        covered: list[int] = []
        for i in range(k):
            s, e = N50._shard_range(len(rpool), k, i)
            covered.extend(range(s, e))
        assert covered == list(range(len(rpool))), (k, covered)

    # (6) signature check on the reused capture entrypoints (call-site ABI).
    cap_params = list(inspect.signature(COL.capture_context_vector).parameters)
    assert cap_params[:4] == ["model", "tokenizer", "messages", "layers"], cap_params
    av_params = list(inspect.signature(COL.capture_answer_vector).parameters)
    assert av_params[:5] == ["model", "tokenizer", "messages", "response", "layers"], av_params
    gen_params = list(inspect.signature(N10._generate).parameters)
    assert gen_params[:3] == ["llm", "tok", "prompts"], gen_params

    # (7) fleet-safe manifest download: two concurrent shards (threads) against a
    #     monkeypatched hf_hub_download serialize via the flock — ONE downloads,
    #     both return the complete dir (no os.replace FileNotFoundError race).
    import shutil
    import threading
    from unittest import mock

    dl_prefix = "smoke_prefix"
    dl_dest = args.out_dir / "_smoke_dl" / MANIFEST_SUBDIR
    # Start clean each run — a prior smoke leaves a COMPLETE manifest here, which
    # would make both threads short-circuit on the fast path (0 downloads).
    shutil.rmtree(dl_dest.parent, ignore_errors=True)
    remote_prefix = f"{dl_prefix}/{MANIFEST_SUBDIR}"
    remote_files = {
        f"{remote_prefix}/meta.json": json.dumps({"n_new": 2, "n_parts": 2}),
        f"{remote_prefix}/part_0000.jsonl": '{"i": 0}\n',
        f"{remote_prefix}/part_0001.jsonl": '{"i": 1}\n',
    }
    dl_calls: list[str] = []
    dl_calls_lock = threading.Lock()

    class _FakeTreeEntry:
        def __init__(self, path):
            self.path = path
            self.size = 1

    class _FakeHfApi:
        def list_repo_tree(self, repo_id, path_in_repo=None, repo_type=None, recursive=False):
            return [_FakeTreeEntry(p) for p in remote_files]

    def _fake_hf_hub_download(repo_id, filename=None, repo_type=None, local_dir=None):
        with dl_calls_lock:
            dl_calls.append(filename)
        time.sleep(0.02)  # widen the race window so both threads reach the flock
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(remote_files[filename])
        return str(out)

    dl_results: list[Path] = []
    dl_errors: list[BaseException] = []

    def _dl_worker():
        try:
            dl_results.append(_download_manifest(dl_prefix, dl_dest))
        except BaseException as e:
            dl_errors.append(e)

    # ONE patch context spans both threads (per-thread mock.patch would race the
    # module-attr restore); both threads see the fakes for their whole lifetime.
    with (
        mock.patch("huggingface_hub.HfApi", _FakeHfApi),
        mock.patch("huggingface_hub.hf_hub_download", _fake_hf_hub_download),
    ):
        dl_threads = [threading.Thread(target=_dl_worker) for _ in range(2)]
        for t in dl_threads:
            t.start()
        for t in dl_threads:
            t.join()
    assert not dl_errors, f"concurrent _download_manifest raised: {dl_errors}"
    assert len(dl_results) == 2 and all(r == dl_dest for r in dl_results), dl_results
    assert _manifest_complete_locally(dl_dest), "manifest incomplete after concurrent download"
    # exactly ONE thread ran the full 3-file download; the other short-circuited under the lock.
    assert len(dl_calls) == len(remote_files), (
        f"expected one full download ({len(remote_files)} files), got {len(dl_calls)}: {dl_calls}"
    )

    # (8) chunk-upload batching (extracted to keep _smoke under the complexity cap).
    batch_cadence = _smoke_upload_batching(args)

    logger.info(
        "[smoke] PASS: near-dupe (1 exact + 1 near drop); valtest-from-round1 (1400 targets + "
        "ctx0 guard); lmsys-exhaust=6 + wildchat-topup=6 = 12; manifest deterministic (sha match) "
        "+ roundtrip + global-index; shard-range k in {2,3,4}; capture signatures match; "
        "concurrent manifest download serialized (1 download, both complete); "
        f"upload batching (2 commits/batch, verify+purge, cadence {batch_cadence})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
