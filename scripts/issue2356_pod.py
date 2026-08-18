#!/usr/bin/env python
"""issue2356 P1+P2 — GPU pod dispatcher (generation + teacher-forced capture).

Phases (subprocess-per-shard fan-out across the ALLOCATED GPUs):

- ``gen``     P1: vLLM batched generation. Arm A + Arm B: N_SAMPLES samples/prompt
              (10 production / 2 under ``--smoke``) @ temp 0.9, top_p 0.95,
              max_new_tokens 2048, seed 42, PLUS 1 greedy. generic: 1 greedy.
              Raw completions land in the plan-parity split layout (R3-7):
              ``raw_completions/{corpus}/{greedy,samples}/shard*.json`` chunk
              files, records carrying the ENGINE's completion ``token_ids``
              (capture consumes those, never a text re-encode). Cap-hit
              (``finish_reason == "length"``) > 2% per corpus — computed
              CORPUS-WIDE across shards via a gen_counts barrier (R3-5) ⇒
              re-generate that corpus's flagged rows (greedy AND samples, per
              kind) on a FRESH engine at ``max_model_len=8192`` /
              ``max_new_tokens=4096`` (M2). Rollout TEXT is uploaded to the HF
              data repo at the END of this phase, BEFORE any capture (#779).
- ``capture`` P2: teacher-forced bf16 HF forwards, BATCHED (token-budget batching,
              ~16 rows/GPU cap), per-npz-shard intra-phase resume, and a
              ``capture_rejects.shard<N>.jsonl`` (digest-only) for empty
              completions (keyed on empty persisted ``token_ids``). Prompt AND
              completion segments reuse the PERSISTED token ids from the gen
              chunks (bit-identical to what generation produced; #1092 recipe).
              Sampled v_A rows persist their ORIGINAL draw index
              (``v_A_sample_idx``) beside the compacted ``v_A_sample_k`` stack.
              Two P2-entry gates: a two-bar batched-vs-serial cosine equivalence
              gate over the LIVE ``_forward_batch`` + an exact render-identity
              gate against the persisted gen-side prompt token ids.
- ``means``   cross-shard merge into CONSOLIDATED ``.npz`` shards
              (≤ MEANS_SHARD_SIZE prompts each; the HF Hub rejects >10k files
              per repo dir) + ``row_index.jsonl``. Entry guards (R3-6): every
              shard's fingerprint-matching capture sentinel MUST exist, and the
              gen-manifest sha set (minus greedy-kind rejects) MUST be ⊆ the
              captured index — a partial capture never merges silently.
- ``upload``  one bulk folder commit for the consolidated stores + exact-set
              verify; re-verifies the raw-completion uploads phase_gen performed.
              Idempotent (R3-8a): a fingerprint-matching upload sentinel skips
              the repeat network uploads on restart.

Row-sharding: ``--shard {0..K-1} --n-shards K`` selects rows by
``index % n_shards == shard``; the dispatcher (no ``--shard``) parses the
INHERITED ``CUDA_VISIBLE_DEVICES`` into an allocation list and pins
``ALLOC[shard]`` per child in the LAUNCHER env (the #545 import-time-cuInit
clobber; the #1336 restricted-allocation ordinal escape). Children write
per-shard log files under ``<out_root>/logs/``; the dispatcher tails a failed
child's log into the main log (#1315 rule iii). The literal ``[phase=done]``
line is RESERVED for the dispatcher's terminal emission on ``--phase all``.

Sentinels: RESUME sentinels live at ``<out_root>/.sentinels`` (OUTSIDE any
poller-drained glob; pod-side-reporting.md requirement 3). The VM-poller-facing
RESULTS sentinel conforms to ``poll_pipeline._SENTINEL_REQUIRED_KEYS`` and is
written to ``--sentinel-dir`` (default ``/workspace/logs``) as
``issue-2356-<kind_slug>-<epoch>.json`` with kind ``epm:results``
(``epm:smoke-result`` under ``--smoke``), version hardcoded 1 (drain-side
max+1 rewrite, #1095), emitted BEFORE the ``[phase=done]`` line.

Smoke (``--smoke``): source-STRATIFIED row caps per corpus (armA 12 / armB 12 /
generic 32 — stratified so no judge-pilot arm goes empty), ``N_SAMPLES=2``, a
forced-truncation dial on 1 armA row (tiny ``max_tokens`` greedy →
``finish_reason == "length"``) so the M2 8192 re-gen branch is smoke-reachable,
and a ``*_smoke`` out_root + HF prefix (no smoke output can clobber production).

Pod-side contract: this driver NEVER shells out to ``scripts/task.py`` (pods run
on ``issue-<N>`` branches; task.py branch-guards to main). Progress is
``[phase=...]`` log breadcrumbs + the sentinel files above.

Content hygiene: consumes harmful (Arm A) + over-refusal (Arm B) + raw real-user
(generic) prompts and the model's own completions. NEVER prints prompt/response
text — digest-only logging (shas, counts, token lengths, cap-hit fractions).
"""

from __future__ import annotations

# load_dotenv BEFORE torch/vllm import (creds; the wrapper fails open on pods so
# no VM thread caps are applied here — dedicated GPUs keep full width).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import os  # noqa: E402

# vLLM V1 EngineCore silent-death guard (#628): set the multiproc method to spawn
# BEFORE anything imports vllm (vllm reads the var at import time).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from explore_persona_space.analysis.extraction import (  # noqa: E402
    extract_layer_activations,
)
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger("issue2356_pod")

# ---------------------------------------------------------------------------
# Constants (plan §10)
# ---------------------------------------------------------------------------
ISSUE = 2356
SLUG = "refusalpred"
HF_PREFIX = f"issue{ISSUE}_{SLUG}"
HF_PREFIX_SMOKE = f"issue{ISSUE}_{SLUG}_smoke"
DATA_REPO = hub.DEFAULT_DATASET_REPO

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAIN_MAX_MODEL_LEN = 4096
MAIN_MAX_NEW_TOKENS = 2048
REGEN_MAX_MODEL_LEN = 8192
REGEN_MAX_NEW_TOKENS = 4096
PROMPT_BUDGET = MAIN_MAX_MODEL_LEN - MAIN_MAX_NEW_TOKENS  # 2048

N_SAMPLES = 10
SAMPLE_TEMPERATURE = 0.9
SAMPLE_TOP_P = 0.95
GLOBAL_SEED = 42
CAP_HIT_REGEN_THRESHOLD = 0.02  # >2% per corpus ⇒ re-gen that corpus (#1332/#1426)

N_HIDDEN_STATES = 29  # Qwen2.5-7B: embedding + 28 decoder blocks
LAYERS = [-1, *range(28)]  # EMBED_LAYER=-1 -> hs[0]; blocks 0..27 -> hs[1..28]
# Capture batching (B8): rows per forward bounded BOTH by a padded-token budget
# (bounds the 29-layer detached hidden-state footprint) and a row cap (~16/GPU,
# plan Step C). 16384 tokens ≈ 3.4 GB of bf16 hidden states per batch at H=3584.
CAPTURE_BATCH_SIZE = 16
CAPTURE_TOKEN_BUDGET = int(os.environ.get("EPM_CAPTURE_TOKEN_BUDGET", "16384"))
NPZ_SHARD_SIZE = 500  # ≤500 prompts per capture .npz shard (intra-phase resume grain)
MEANS_SHARD_SIZE = 500  # ≤500 prompts per CONSOLIDATED store shard (B1; Hub 10k/dir cap)
GEN_CHUNK = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
# raw_completions chunking (R3-7): plan requires ≥1 shard per 500 prompts per arm;
# completion token ids inflate rows well past text-only size, so samples files
# chunk smaller to stay under the ~9.5 MB non-LFS Hub file budget (upload-policy).
RAW_GREEDY_CHUNK = 250  # prompts per {corpus}/greedy/shard*.json file
RAW_SAMPLES_CHUNK = 100  # prompts per {corpus}/samples/shard*.json file
RAW_KINDS = ("greedy", "samples")
# R3-5: cross-shard cap-hit barrier wait (the >2% M2 re-gen decision is
# corpus-WIDE, so each shard blocks until every shard's counters exist).
GEN_COUNTS_WAIT_S = int(os.environ.get("EPM_GEN_COUNTS_WAIT_S", "7200"))

# Two-bar equivalence gate (#779 bf16 single-position calibration).
EQUIV_EARLY_LAYERS = (0, 1, 2, 3)
EQUIV_EARLY_BAR = 0.999
EQUIV_FLAT_BAR = 0.995
EQUIV_N_ROWS = 8
RENDER_IDENTITY_N_ROWS = 100

# Smoke dials (B5): per-corpus row caps (source-stratified), reduced samples,
# and a forced-truncation greedy on 1 armA row so the M2 re-gen branch runs.
SMOKE_ROW_CAPS = {"armA": 12, "armB": 12, "generic": 32}
SMOKE_N_SAMPLES = 2
SMOKE_TRUNC_MAX_TOKENS = 16

ARMS_MULTI = ("armA", "armB")  # multi-sample arms
CORPORA = ("armA", "armB", "generic")

DEFAULT_OUT_ROOT = "/workspace/issue2356"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hf_prefix(args: argparse.Namespace) -> str:
    return HF_PREFIX_SMOKE if args.smoke else HF_PREFIX


def _out_root(args: argparse.Namespace) -> Path:
    """Smoke runs get their own out_root so no smoke output clobbers production."""
    if args.smoke and args.out_root == DEFAULT_OUT_ROOT:
        return Path(DEFAULT_OUT_ROOT + "_smoke")
    return Path(args.out_root)


def _n_samples(args: argparse.Namespace) -> int:
    return SMOKE_N_SAMPLES if args.smoke else N_SAMPLES


def _code_sha() -> str:
    return git_provenance().commit_sha


_CORPUS_SHA_CACHE: dict[str, dict[str, str]] = {}


def _corpus_shas(args: argparse.Namespace) -> dict[str, str]:
    """sha256 per staged corpus JSONL — keys the phase fingerprints to the INPUT corpus."""
    prefix = _hf_prefix(args)
    if prefix in _CORPUS_SHA_CACHE:
        return _CORPUS_SHA_CACHE[prefix]
    dest = _out_root(args) / "corpus_staged"
    dest.mkdir(parents=True, exist_ok=True)
    hub.stage_hub_prefix(DATA_REPO, f"{prefix}/corpus", dest, repo_type="dataset")
    shas: dict[str, str] = {}
    for corpus in CORPORA:
        jl = dest / prefix / "corpus" / f"{corpus}.jsonl"
        shas[corpus] = hashlib.sha256(jl.read_bytes()).hexdigest()
    _CORPUS_SHA_CACHE[prefix] = shas
    return shas


def _flag_fingerprint(args: argparse.Namespace, phase: str, shard: int | None) -> str:
    """Resume fingerprint: input corpus shas + every output-affecting flag + code sha."""
    payload = {
        "phase": phase,
        "shard": shard,
        "n_shards": args.n_shards,
        "smoke": bool(args.smoke),
        "corpus_prefix": _hf_prefix(args),
        "corpus_shas": _corpus_shas(args),
        "main_max_model_len": MAIN_MAX_MODEL_LEN,
        "main_max_new_tokens": MAIN_MAX_NEW_TOKENS,
        "regen_max_model_len": REGEN_MAX_MODEL_LEN,
        "regen_max_new_tokens": REGEN_MAX_NEW_TOKENS,
        "n_samples": _n_samples(args),
        "smoke_row_caps": SMOKE_ROW_CAPS if args.smoke else None,
        "smoke_trunc_max_tokens": SMOKE_TRUNC_MAX_TOKENS if args.smoke else None,
        "npz_shard_size": NPZ_SHARD_SIZE,
        "means_shard_size": MEANS_SHARD_SIZE,
        "seed": GLOBAL_SEED,
        "code_sha": _code_sha(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _sentinel_path(args: argparse.Namespace, phase: str, shard: int | None) -> Path:
    """RESUME sentinels — kept under out_root/.sentinels, OUTSIDE the poller's glob."""
    sent = _out_root(args) / ".sentinels"
    name = f"{phase}.done.json" if shard is None else f"{phase}.shard{shard}.done.json"
    return sent / name


def _write_json_atomic(path: Path, rec: dict[str, Any]) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _write_sentinel(path: Path, fingerprint: str, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "input_fingerprint": fingerprint,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _code_sha(),
        **extra,
    }
    _write_json_atomic(path, rec)


def _sentinel_ok(path: Path, fingerprint: str, resume: bool) -> bool:
    if not resume or not path.exists():
        return False
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return rec.get("input_fingerprint") == fingerprint


def _load_corpus_rows(args: argparse.Namespace, corpus: str) -> list[dict[str, Any]]:
    """Stage the corpus text JSONL from HF and return its rows (prompt text).

    Under ``--smoke``, rows are capped per corpus (SMOKE_ROW_CAPS) with a
    SOURCE-STRATIFIED round-robin: the corpus JSONLs are source-ordered, so a
    head-slice would silently drop a whole source (and empty its judge-pilot
    arm — the round-1 armB_orbench empty-arm crash class).
    """
    prefix = _hf_prefix(args)
    dest = _out_root(args) / "corpus_staged"
    dest.mkdir(parents=True, exist_ok=True)
    # stage_hub_prefix mirrors <repo prefix> under dest: dest/<prefix>/corpus/<c>.jsonl
    hub.stage_hub_prefix(DATA_REPO, f"{prefix}/corpus", dest, repo_type="dataset")
    jl = dest / prefix / "corpus" / f"{corpus}.jsonl"
    rows: list[dict[str, Any]] = []
    with open(jl, encoding="utf-8") as fh:
        for line in fh:  # text-mode iteration, never .splitlines() (#950)
            s = line.strip("\n")
            if s:
                rows.append(json.loads(s))
    if args.smoke:
        cap = min(SMOKE_ROW_CAPS[corpus], len(rows))
        by_source: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_source.setdefault(str(r.get("source", "unknown")), []).append(r)
        sources = sorted(by_source)
        picked: list[dict[str, Any]] = []
        idx = 0
        while len(picked) < cap and any(by_source[s] for s in sources):
            src = sources[idx % len(sources)]
            if by_source[src]:
                picked.append(by_source[src].pop(0))
            idx += 1
        rows = picked
        logger.info(
            "[corpus %s] smoke cap: %d rows across %d sources", corpus, len(rows), len(sources)
        )
    return rows


def _shard_rows(rows: list[dict[str, Any]], shard: int, n_shards: int) -> list[dict[str, Any]]:
    return [r for i, r in enumerate(rows) if i % n_shards == shard]


def _assert_generation_window(
    max_prompt_tokens: int, max_new_tokens: int, engine_max_model_len: int
) -> None:
    """M2: max_prompt_tokens + max_new_tokens ≤ max_model_len, on EVERY branch."""
    total = max_prompt_tokens + max_new_tokens
    if total > engine_max_model_len:
        raise ValueError(
            f"generation-window invariant violated: prompt {max_prompt_tokens} + "
            f"new {max_new_tokens} = {total} > max_model_len {engine_max_model_len}"
        )


def _raw_chunk_files(kind_dir: Path, shard: int) -> list[Path]:
    """This GPU-shard's chunk files under a {corpus}/{greedy|samples}/ dir."""
    return sorted(kind_dir.glob(f"shard{shard}_*.json")) if kind_dir.exists() else []


def _write_raw_chunks(
    kind_dir: Path, shard: int, rows: list[dict[str, Any]], chunk_size: int
) -> None:
    """Chunked raw-completions writer (R3-7 plan-glob parity:
    ``{corpus}/{greedy,samples}/shard*.json``). Removes any stale chunk files a
    prior differently-chunked run of THIS shard left (they would merge into the
    read-side glob), then writes ``shard{N}_{chunk:03d}.json`` atomically."""
    kind_dir.mkdir(parents=True, exist_ok=True)
    for old in _raw_chunk_files(kind_dir, shard):
        old.unlink()
    for ci, start in enumerate(range(0, len(rows), chunk_size)):
        chunk = rows[start : start + chunk_size]
        out = kind_dir / f"shard{shard}_{ci:03d}.json"
        tmp = Path(str(out) + ".tmp")
        tmp.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, out)


def _read_raw_shard(kind_dir: Path, shard: int) -> tuple[list[dict[str, Any]], list[Path]]:
    """All entries of this GPU-shard's chunk files (ordered), plus the files."""
    files = _raw_chunk_files(kind_dir, shard)
    entries: list[dict[str, Any]] = []
    for f in files:
        entries.extend(json.loads(f.read_text(encoding="utf-8")))
    return entries, files


def _await_gen_counts(args: argparse.Namespace, fp: str) -> tuple[dict[str, int], dict[str, int]]:
    """R3-5 cross-shard barrier: block until EVERY shard's ``gen_counts`` file
    (fingerprint-matching) exists, then return corpus-WIDE (cap_hit, gen) sums.

    The >2% M2 re-gen decision keys on the corpus-wide cap-hit fraction — a
    shard-local fraction mis-estimates it whenever shards see different row
    subsets. n_shards=1 passes trivially (own file just written). Bounded wait
    (``EPM_GEN_COUNTS_WAIT_S``, default 7200 s) then fail-loud naming the
    missing shards."""
    counts_dir = _out_root(args) / ".sentinels"
    deadline = time.time() + GEN_COUNTS_WAIT_S
    last_log = 0.0
    while True:
        recs: list[dict[str, Any]] = []
        missing: list[int] = []
        for s in range(args.n_shards):
            p = counts_dir / f"gen_counts.shard{s}.json"
            rec: dict[str, Any] | None = None
            if p.exists():
                try:
                    rec = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    rec = None
            if rec and rec.get("input_fingerprint") == fp:
                recs.append(rec)
            else:
                missing.append(s)
        if not missing:
            cap: dict[str, int] = {}
            gen: dict[str, int] = {}
            for rec in recs:
                for c, v in (rec.get("cap_hit_total") or {}).items():
                    cap[c] = cap.get(c, 0) + int(v)
                for c, v in (rec.get("gen_total") or {}).items():
                    gen[c] = gen.get(c, 0) + int(v)
            return cap, gen
        if time.time() > deadline:
            raise RuntimeError(
                f"[gen] cross-shard cap-hit barrier timed out after {GEN_COUNTS_WAIT_S}s; "
                f"missing gen_counts from shards {missing}"
            )
        if time.time() - last_log > 300:
            logger.info("[gen] cap-hit barrier waiting on gen_counts from shards %s", missing)
            last_log = time.time()
        time.sleep(10)


# ---------------------------------------------------------------------------
# Phase gen (P1)
# ---------------------------------------------------------------------------


def _render_chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )


def _max_prompt_tokens(tokenizer, rendered: list[str]) -> int:
    """Max token count over RENDERED chat-template strings — the strings the
    engine actually consumes (raw prompt counts under-count template overhead)."""
    return max(len(tokenizer.encode(p, add_special_tokens=False)) for p in rendered)


def _generate_chunked(engine, prompts: list[str], sampling_params) -> list[Any]:
    """Order-preserving chunked generate; use_tqdm=False (both the large-batch
    deadlock prevention #664 and the tqdm ZeroDivision guard #613)."""
    outputs: list[Any] = []
    for start in range(0, len(prompts), GEN_CHUNK):
        chunk = prompts[start : start + GEN_CHUNK]
        logger.info("[vllm-chunk] gen chunk %d..%d / %d", start, start + len(chunk), len(prompts))
        outputs.extend(engine.generate(chunk, sampling_params, use_tqdm=False))
    return outputs


def phase_gen(args: argparse.Namespace, shard: int) -> None:
    from vllm import SamplingParams  # deferred (vllm imported once, spawn set above)

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    fp = _flag_fingerprint(args, "gen", shard)
    sent = _sentinel_path(args, "gen", shard)
    if _sentinel_ok(sent, fp, resume=not args.no_resume):
        logger.info("[gen shard %d] resume-skip (fingerprint match)", shard)
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    engine = create_vllm_engine(MODEL, max_model_len=MAIN_MAX_MODEL_LEN, seed=GLOBAL_SEED)
    raw_root = _out_root(args) / "eval_results" / _hf_prefix(args) / "raw_completions"
    n_samples = _n_samples(args)

    cap_hit_total: dict[str, int] = {}
    gen_total: dict[str, int] = {}
    regen_flagged: list[dict[str, Any]] = []

    for corpus in CORPORA:
        rows = _shard_rows(_load_corpus_rows(args, corpus), shard, args.n_shards)
        if not rows:
            continue
        rendered = [_render_chat(tokenizer, r["prompt"]) for r in rows]
        mpt = _max_prompt_tokens(tokenizer, rendered)

        # greedy pass (all corpora)
        _assert_generation_window(mpt, MAIN_MAX_NEW_TOKENS, MAIN_MAX_MODEL_LEN)
        greedy_sp = SamplingParams(
            temperature=0.0, max_tokens=MAIN_MAX_NEW_TOKENS, seed=GLOBAL_SEED
        )
        greedy_out = _generate_chunked(engine, rendered, greedy_sp)

        sampled_out: list[Any] | None = None
        if corpus in ARMS_MULTI:
            _assert_generation_window(mpt, MAIN_MAX_NEW_TOKENS, MAIN_MAX_MODEL_LEN)
            sampled_sp = SamplingParams(
                n=n_samples,
                temperature=SAMPLE_TEMPERATURE,
                top_p=SAMPLE_TOP_P,
                max_tokens=MAIN_MAX_NEW_TOKENS,
                seed=GLOBAL_SEED,
            )
            sampled_out = _generate_chunked(engine, rendered, sampled_sp)

        cap_hit = 0
        n_gen = 0
        greedy_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        for i, r in enumerate(rows):
            greedy_comp = greedy_out[i].outputs[0]
            n_gen += 1
            if greedy_comp.finish_reason == "length":
                cap_hit += 1
                regen_flagged.append(
                    {"corpus": corpus, "prompt_sha": r["prompt_sha"], "kind": "greedy"}
                )
            entry: dict[str, Any] = {
                "prompt_sha": r["prompt_sha"],
                "prompt": r["prompt"],
                # corpus provenance carried into gen shards: judge.py pilots stratify
                # arms on `source` (round-1 crash: missing source emptied armB_orbench).
                "source": r["source"],
                # the ENGINE's realized prompt token ids (render-identity gate input, B2)
                "prompt_token_ids": list(greedy_out[i].prompt_token_ids),
                # completion token ids persisted so capture consumes the ENGINE's
                # realized completion tokens, never a re-encode of decoded text
                # (completion-token-identity; the #1092 recipe on the answer side).
                "greedy": {
                    "text": greedy_comp.text,
                    "finish_reason": greedy_comp.finish_reason,
                    "token_ids": list(greedy_comp.token_ids),
                },
            }
            for opt_key in ("base_id", "axis", "category"):
                if opt_key in r:
                    entry[opt_key] = r[opt_key]
            greedy_rows.append(entry)
            if sampled_out is not None:
                samples = []
                for k, comp in enumerate(sampled_out[i].outputs):
                    n_gen += 1
                    if comp.finish_reason == "length":
                        cap_hit += 1
                        regen_flagged.append(
                            {"corpus": corpus, "prompt_sha": r["prompt_sha"], "kind": f"sample{k}"}
                        )
                    samples.append(
                        {
                            "text": comp.text,
                            "finish_reason": comp.finish_reason,
                            "token_ids": list(comp.token_ids),
                        }
                    )
                sample_rows.append(
                    {
                        "prompt_sha": r["prompt_sha"],
                        "prompt": r["prompt"],
                        "source": r["source"],
                        "samples": samples,
                    }
                )

        # Forced-truncation smoke dial (B5): regenerate armA row 0 greedy at a tiny
        # cap so finish_reason == "length" genuinely occurs and the M2 8192 re-gen
        # branch below is exercised end-to-end by the pod smoke.
        if args.smoke and corpus == "armA" and greedy_rows:
            trunc_sp = SamplingParams(
                temperature=0.0, max_tokens=SMOKE_TRUNC_MAX_TOKENS, seed=GLOBAL_SEED
            )
            trunc_out = _generate_chunked(engine, rendered[:1], trunc_sp)
            tcomp = trunc_out[0].outputs[0]
            greedy_rows[0]["greedy"] = {
                "text": tcomp.text,
                "finish_reason": tcomp.finish_reason,
                "token_ids": list(tcomp.token_ids),
            }
            if tcomp.finish_reason == "length":
                cap_hit += 1
                flag = {
                    "corpus": corpus,
                    "prompt_sha": greedy_rows[0]["prompt_sha"],
                    "kind": "greedy",
                }
                if flag not in regen_flagged:
                    regen_flagged.append(flag)
                logger.info(
                    "[smoke-forced-trunc] armA row0 greedy truncated at %d tokens",
                    SMOKE_TRUNC_MAX_TOKENS,
                )

        cap_hit_total[corpus] = cap_hit
        gen_total[corpus] = n_gen
        frac = cap_hit / max(1, n_gen)
        logger.info(
            "[gen shard %d] corpus=%s gens=%d cap_hit=%d frac=%.4f",
            shard,
            corpus,
            n_gen,
            cap_hit,
            frac,
        )

        _write_raw_chunks(raw_root / corpus / "greedy", shard, greedy_rows, RAW_GREEDY_CHUNK)
        if sample_rows:
            _write_raw_chunks(raw_root / corpus / "samples", shard, sample_rows, RAW_SAMPLES_CHUNK)

    cleanup_vllm(engine)

    # R3-5: publish THIS shard's counters, then block on every shard's counters —
    # the >2% M2 re-gen decision is CORPUS-WIDE across shards, never shard-local
    # (two GPU workers each seeing half the corpus mis-estimate the true fraction).
    counts_fp = _flag_fingerprint(args, "gen-counts", None)  # shard-INDEPENDENT
    counts_dir = _out_root(args) / ".sentinels"
    counts_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        counts_dir / f"gen_counts.shard{shard}.json",
        {
            "input_fingerprint": counts_fp,
            "shard": shard,
            "cap_hit_total": cap_hit_total,
            "gen_total": gen_total,
        },
    )
    global_cap, global_gen = _await_gen_counts(args, counts_fp)

    # Cap-hit re-gen on a FRESH 8192 engine at 4096 new tokens (M2). PER-CORPUS
    # gating on the GLOBAL fraction: only corpora whose corpus-wide cap-hit
    # fraction exceeds the threshold are re-generated (the round-1 `any(...)`
    # gate re-generated EVERY flagged row); each shard re-gens its OWN rows.
    local_frac = {c: cap_hit_total.get(c, 0) / max(1, gen_total.get(c, 1)) for c in cap_hit_total}
    per_corpus_frac = {c: global_cap.get(c, 0) / max(1, global_gen.get(c, 1)) for c in global_gen}
    logger.info(
        "[gen shard %d] cap-hit fractions local=%s global=%s", shard, local_frac, per_corpus_frac
    )
    regen_targets = [
        f for f in regen_flagged if per_corpus_frac.get(f["corpus"], 0.0) > CAP_HIT_REGEN_THRESHOLD
    ]
    regen_meta: dict[str, Any] = {
        "regen_ran": False,
        "n_flagged": len(regen_flagged),
        "n_targeted": len(regen_targets),
    }
    if regen_targets:
        regen_meta.update(_regen_flagged_rows(args, shard, regen_targets, tokenizer))

    # Persist rollout TEXT to HF BEFORE any capture (#779, B3): a pod death during
    # capture must not lose the generation output.
    _upload_raw_shard(args, shard)

    _write_sentinel(
        sent,
        fp,
        {
            "phase": "gen",
            "shard": shard,
            "n_samples": n_samples,
            "cap_hit_total": cap_hit_total,
            "gen_total": gen_total,
            "per_corpus_frac_local": local_frac,
            "per_corpus_frac_global": per_corpus_frac,
            "regen": regen_meta,
        },
    )
    logger.info("[phase=gen shard=%d complete]", shard)


def _regen_flagged_rows(
    args: argparse.Namespace, shard: int, flagged: list[dict[str, Any]], tokenizer
) -> dict[str, Any]:
    """M2 re-gen: greedy flags → greedy_regen8192; sampleK flags → samples[k].regen8192.

    A sample-only flag never touches the (healthy) greedy record (round-1 B4
    clobber), and flagged samples are re-generated at SAMPLE params, grouped by
    sample index k with a per-k decorrelated seed (one engine call per k)."""
    from vllm import SamplingParams

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    logger.info(
        "[regen shard %d] re-generating %d flagged rows on fresh 8192 engine", shard, len(flagged)
    )
    engine = create_vllm_engine(MODEL, max_model_len=REGEN_MAX_MODEL_LEN, seed=GLOBAL_SEED)
    raw_root = _out_root(args) / "eval_results" / _hf_prefix(args) / "raw_completions"

    residual = 0
    total = 0
    by_corpus: dict[str, list[dict[str, Any]]] = {}
    for f in flagged:
        by_corpus.setdefault(f["corpus"], []).append(f)

    for corpus, flags in by_corpus.items():
        # split-layout RMW (R3-7): entries keep a ref to their chunk FILE so only
        # touched chunk files are rewritten (atomically) after the re-gen.
        g_by_file: dict[Path, list[dict[str, Any]]] = {}
        g_by_sha: dict[str, tuple[Path, dict[str, Any]]] = {}
        for f in _raw_chunk_files(raw_root / corpus / "greedy", shard):
            ents = json.loads(f.read_text(encoding="utf-8"))
            g_by_file[f] = ents
            for e in ents:
                g_by_sha[e["prompt_sha"]] = (f, e)
        s_by_file: dict[Path, list[dict[str, Any]]] = {}
        s_by_sha: dict[str, tuple[Path, dict[str, Any]]] = {}
        for f in _raw_chunk_files(raw_root / corpus / "samples", shard):
            ents = json.loads(f.read_text(encoding="utf-8"))
            s_by_file[f] = ents
            for e in ents:
                s_by_sha[e["prompt_sha"]] = (f, e)
        touched: set[Path] = set()

        # greedy re-gen (greedy params)
        greedy_shas = sorted({f["prompt_sha"] for f in flags if f["kind"] == "greedy"})
        if greedy_shas:
            rendered = [_render_chat(tokenizer, g_by_sha[s][1]["prompt"]) for s in greedy_shas]
            mpt = _max_prompt_tokens(tokenizer, rendered)
            _assert_generation_window(mpt, REGEN_MAX_NEW_TOKENS, REGEN_MAX_MODEL_LEN)
            sp = SamplingParams(temperature=0.0, max_tokens=REGEN_MAX_NEW_TOKENS, seed=GLOBAL_SEED)
            outs = _generate_chunked(engine, rendered, sp)
            for sha, out in zip(greedy_shas, outs):
                comp = out.outputs[0]
                total += 1
                if comp.finish_reason == "length":
                    residual += 1
                fpath, entry = g_by_sha[sha]
                entry["greedy_regen8192"] = {
                    "text": comp.text,
                    "finish_reason": comp.finish_reason,
                    "token_ids": list(comp.token_ids),
                }
                touched.add(fpath)

        # sample re-gen (sample params), grouped per sample index k
        by_k: dict[int, list[str]] = {}
        for f in flags:
            if f["kind"].startswith("sample"):
                by_k.setdefault(int(f["kind"][len("sample") :]), []).append(f["prompt_sha"])
        for k, shas_k in sorted(by_k.items()):
            shas_k = sorted(set(shas_k))
            rendered = [_render_chat(tokenizer, s_by_sha[s][1]["prompt"]) for s in shas_k]
            mpt = _max_prompt_tokens(tokenizer, rendered)
            _assert_generation_window(mpt, REGEN_MAX_NEW_TOKENS, REGEN_MAX_MODEL_LEN)
            sp = SamplingParams(
                n=1,
                temperature=SAMPLE_TEMPERATURE,
                top_p=SAMPLE_TOP_P,
                max_tokens=REGEN_MAX_NEW_TOKENS,
                seed=GLOBAL_SEED + 1000 + k,  # decorrelated per k (same seed ⇒ same draw)
            )
            outs = _generate_chunked(engine, rendered, sp)
            for sha, out in zip(shas_k, outs):
                comp = out.outputs[0]
                total += 1
                if comp.finish_reason == "length":
                    residual += 1
                fpath, entry = s_by_sha[sha]
                entry["samples"][k]["regen8192"] = {
                    "text": comp.text,
                    "finish_reason": comp.finish_reason,
                    "token_ids": list(comp.token_ids),
                }
                touched.add(fpath)

        for fpath in sorted(touched):
            ents = g_by_file.get(fpath, s_by_file.get(fpath))
            tmp = Path(str(fpath) + ".tmp")
            tmp.write_text(json.dumps(ents, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, fpath)

    cleanup_vllm(engine)
    logger.info("[regen shard %d] residual_truncated=%d/%d at 8192/4096", shard, residual, total)
    return {"regen_ran": True, "n_regen": total, "residual_truncated": residual}


def _upload_raw_shard(args: argparse.Namespace, shard: int) -> None:
    """Upload THIS shard's rollout-text JSONs to the HF data repo (end of phase_gen).

    Bounded per-file loop over this shard's {corpus}/{greedy,samples}/ chunk files
    (a few dozen at production scale — far under the ~256 commits/hr Hub throttle,
    #591); verified with the retried exact-set helper. phase_upload later
    re-verifies the full set."""
    prefix = _hf_prefix(args)
    raw_root = _out_root(args) / "eval_results" / prefix / "raw_completions"
    uploaded: list[str] = []
    for corpus in CORPORA:
        for kind in RAW_KINDS:
            for f in _raw_chunk_files(raw_root / corpus / kind, shard):
                # UPLOAD_LOOP_EXEMPT: bounded per-shard chunk-file loop (tens of files)
                base_url = hub._upload(
                    local_path=f,
                    repo_id=DATA_REPO,
                    repo_type="dataset",
                    path_in_repo=f"{prefix}/raw_completions/{corpus}/{kind}/{f.name}",
                    raise_on_error=True,
                    upload_as_file=True,
                )
                if not base_url:
                    raise RuntimeError(
                        f"[gen shard {shard}] _upload returned no path for "
                        f"{corpus}/{kind}/{f.name} (silent durability loss — missing "
                        "HF_TOKEN / absent local path / failed verify)"
                    )
                uploaded.append(f"{corpus}/{kind}/{f.name}")
    if not uploaded:
        raise RuntimeError(f"[gen shard {shard}] no raw completion files to upload")
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        DATA_REPO,
        uploaded,
        path_in_repo=f"{prefix}/raw_completions",
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"raw_completions shard upload incomplete; missing: {missing}")
    logger.info("[gen shard %d] rollout text uploaded + verified (%d files)", shard, len(uploaded))


# ---------------------------------------------------------------------------
# Phase capture (P2) — teacher-forced summaries
# ---------------------------------------------------------------------------

# #1092 recipe: forward the CONCATENATED PER-SEGMENT TOKEN IDS (never re-tokenize
# the concatenated string). The prompt segment is the PERSISTED engine-side
# prompt_token_ids from the gen shard (bit-identical to what generation consumed);
# the answer span is read at the true token boundary.


def _capture_row_ids_and_positions(
    prompt_ids: list[int], completion_ids: list[int], max_len: int
) -> tuple[list[int], dict[str, int]]:
    prompt_ids = list(prompt_ids)
    completion_ids = list(completion_ids)
    if not prompt_ids:
        raise ValueError("empty prompt_ids")
    if not completion_ids:
        raise ValueError("empty completion_ids (caller must reject empty completions)")
    row_ids = prompt_ids + completion_ids
    n_total = len(row_ids)
    if n_total > max_len:
        raise ValueError(f"capture row {n_total} tokens exceeds max_len {max_len}")
    return row_ids, {
        "n_total": n_total,
        "n_prompt": len(prompt_ids),
        "context_end": len(prompt_ids) - 1,
        "answer_start": len(prompt_ids),
        "answer_end": n_total,
    }


def _forward_hidden_states(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, layers: list[int]
) -> dict[int, torch.Tensor]:
    """Hook-based per-layer hidden states. ``return_logits=False`` makes
    ``extract_layer_activations`` apply ``_logits_to_keep_kwargs`` INTERNALLY
    (extraction.py:252) so the unread full-vocab logits are never materialized
    (#779)."""
    return extract_layer_activations(
        model,
        input_ids,
        layers,
        attention_mask=attention_mask,
        return_logits=False,
        detach_to_cpu=True,
    )


def _summarize_row(
    hs: dict[int, torch.Tensor], b: int, pos: dict[str, int], layers: list[int]
) -> dict[str, np.ndarray]:
    """v_C (last prompt token) + v_A (mean over answer span), per layer, fp16."""
    v_c = np.stack(
        [hs[layer][b, pos["context_end"]].float().numpy() for layer in layers], axis=0
    ).astype(np.float16)
    a0, a1 = pos["answer_start"], pos["answer_end"]
    v_a = np.stack(
        [hs[layer][b, a0:a1].float().mean(dim=0).numpy() for layer in layers], axis=0
    ).astype(np.float16)
    return {"v_C": v_c, "v_A": v_a}


def _forward_batch(
    model, pad_id: int, items: list[tuple[list[int], dict[str, int]]], layers: list[int]
) -> list[dict[str, np.ndarray]]:
    """ONE right-padded batched teacher-forced forward → per-item fp16 summaries.

    THE live capture hot path (B8). Right padding keeps real-token positions
    identical to the unpadded sequence under the causal mask, so batch-1 and
    batch-B agree up to kernel numerics — asserted by ``_equivalence_core``,
    which runs THIS function on both sides (no hollow gate)."""
    maxlen = max(len(ids) for ids, _ in items)
    batch = torch.full((len(items), maxlen), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(items), maxlen), dtype=torch.long)
    for i, (ids, _) in enumerate(items):
        batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, : len(ids)] = 1
    with torch.no_grad():
        hs = _forward_hidden_states(model, batch.to(model.device), mask.to(model.device), layers)
    outs = [_summarize_row(hs, i, pos, layers) for i, (_, pos) in enumerate(items)]
    del hs
    return outs


def _iter_capture_batches(items: list) -> Any:
    """Token-budget + row-cap batching; the longest item alone still forms a batch."""
    batch: list = []
    maxlen = 0
    for it in items:
        n = len(it[1])
        new_max = max(maxlen, n)
        too_big = new_max * (len(batch) + 1) > CAPTURE_TOKEN_BUDGET
        if batch and (too_big or len(batch) >= CAPTURE_BATCH_SIZE):
            yield batch
            batch, maxlen = [], 0
            new_max = n
        batch.append(it)
        maxlen = new_max
    if batch:
        yield batch


def _equivalence_core(
    model, pad_id: int, items: list[tuple[list[int], dict[str, int]]], layers: list[int]
) -> tuple[float, float]:
    """Two-bar batched-vs-serial cosine gate (#779) over the LIVE ``_forward_batch``:
    batch-1 calls are the serial reference, one batch-B call is the tested path."""
    serial = [_forward_batch(model, pad_id, [it], layers)[0] for it in items]
    batched = _forward_batch(model, pad_id, items, layers)
    early_rows = [li for li, layer in enumerate(layers) if layer in EQUIV_EARLY_LAYERS]
    early_min = 1.0
    flat: list[float] = []
    for s, bt in zip(serial, batched):
        for key in ("v_C", "v_A"):
            sv = s[key].astype(np.float32)
            bv = bt[key].astype(np.float32)
            num = (sv * bv).sum(axis=1)
            den = np.linalg.norm(sv, axis=1) * np.linalg.norm(bv, axis=1) + 1e-12
            cos = num / den
            flat.extend(cos.tolist())
            for li in early_rows:
                early_min = min(early_min, float(cos[li]))
    flat_mean = float(np.mean(flat))
    if early_min < EQUIV_EARLY_BAR:
        raise ValueError(f"equivalence gate early-layer cosine {early_min:.6f} < {EQUIV_EARLY_BAR}")
    if flat_mean < EQUIV_FLAT_BAR:
        raise ValueError(f"equivalence gate flat cosine {flat_mean:.6f} < {EQUIV_FLAT_BAR}")
    return early_min, flat_mean


def _equivalence_gate(model, tokenizer) -> None:
    """P2-entry gate: unequal-length rows (padding genuinely fires) through
    ``_equivalence_core`` on the production model."""
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    items = []
    for i in range(EQUIV_N_ROWS):
        text = f"Question {i}: describe topic {i} briefly."
        comp = " A short answer." + " More detail follows here." * (i % 3)
        p_ids = tokenizer.encode(_render_chat(tokenizer, text), add_special_tokens=False)
        c_ids = tokenizer.encode(comp, add_special_tokens=False)
        row_ids, pos = _capture_row_ids_and_positions(p_ids, c_ids, REGEN_MAX_MODEL_LEN)
        items.append((row_ids, pos))
    early_min, flat_mean = _equivalence_core(model, int(pad_id), items, LAYERS)
    logger.info("[equiv-gate] early_min=%.6f flat_mean=%.6f", early_min, flat_mean)


def _equivalence_selftest() -> int:
    """CPU fp32 tiny-Qwen2 batched-vs-serial equivalence check of ``_forward_batch``
    (the batched-rewrite equivalence duty for B8). Random weights + synthetic token
    ids — no download, no GPU; B≥2 with UNEQUAL lengths so right-padding fires."""
    from transformers import Qwen2Config

    torch.manual_seed(0)
    cfg = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=512,
        max_position_embeddings=512,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    layers = [-1, 0, 1, 2, 3]
    rng = np.random.default_rng(0)
    items = []
    for _ in range(6):
        p_ids = rng.integers(1, 512, size=int(rng.integers(8, 24))).tolist()
        c_ids = rng.integers(1, 512, size=int(rng.integers(2, 12))).tolist()
        row_ids, pos = _capture_row_ids_and_positions(p_ids, c_ids, 512)
        items.append((row_ids, pos))
    early_min, flat_mean = _equivalence_core(model, 0, items, layers)
    logger.info(
        "[equiv-selftest] early_min=%.6f flat_mean=%.6f (tiny fp32 CPU Qwen2)",
        early_min,
        flat_mean,
    )
    return 0


def _render_identity_gate(tokenizer, rows: list[dict[str, Any]]) -> None:
    """Gen↔capture render seam (B2): capture's re-render token ids must equal the
    PERSISTED engine-side ``prompt_token_ids`` from the gen shard — a genuine
    cross-stage comparison (the round-1 version compared two identical
    re-encodes and could never fail)."""
    n = min(RENDER_IDENTITY_N_ROWS, len(rows))
    for r in rows[:n]:
        gen_ids = r.get("prompt_token_ids")
        if gen_ids is None:
            raise ValueError(
                f"gen shard entry {r['prompt_sha']} lacks prompt_token_ids — "
                "re-run phase gen with the r2 dispatcher"
            )
        cap_ids = tokenizer.encode(_render_chat(tokenizer, r["prompt"]), add_special_tokens=False)
        if list(gen_ids) != cap_ids:
            raise ValueError(f"render-identity mismatch for prompt_sha {r['prompt_sha']}")
    logger.info("[render-identity] capture render == persisted engine ids on %d rows", n)


def _greedy_record(entry: dict[str, Any]) -> dict[str, Any]:
    """greedy_regen8192 is written ONLY when the greedy itself was flagged (B4)."""
    return entry.get("greedy_regen8192", entry["greedy"])


def _sample_record(sample: dict[str, Any]) -> dict[str, Any]:
    return sample.get("regen8192", sample)


def phase_capture(args: argparse.Namespace, shard: int) -> None:
    fp = _flag_fingerprint(args, "capture", shard)
    sent = _sentinel_path(args, "capture", shard)
    if _sentinel_ok(sent, fp, resume=not args.no_resume):
        logger.info("[capture shard %d] resume-skip (fingerprint match)", shard)
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_root = _out_root(args) / "eval_results" / _hf_prefix(args) / "raw_completions"
    store_root = _out_root(args) / "stores" / f"shard{shard}"
    store_root.mkdir(parents=True, exist_ok=True)

    # B9: validate ALL inputs (chunk files parse, prompt_token_ids present,
    # render-identity gate) BEFORE the expensive 7B model load. The split
    # greedy/samples records (R3-7) are joined by prompt_sha into one merged
    # per-prompt view (the shape _capture_corpus consumes).
    corpora_entries: dict[str, list[dict[str, Any]]] = {}
    for corpus in CORPORA:
        g_entries, _ = _read_raw_shard(raw_root / corpus / "greedy", shard)
        if not g_entries:
            continue
        _render_identity_gate(tokenizer, g_entries)
        s_entries, _ = _read_raw_shard(raw_root / corpus / "samples", shard)
        s_by_sha = {e["prompt_sha"]: e for e in s_entries}
        orphans = sorted(set(s_by_sha) - {e["prompt_sha"] for e in g_entries})
        if orphans:
            raise RuntimeError(
                f"[capture shard {shard}] {corpus}: {len(orphans)} samples entries "
                f"lack a greedy entry (first: {orphans[:3]})"
            )
        for e in g_entries:
            if e["prompt_sha"] in s_by_sha:
                e["samples"] = s_by_sha[e["prompt_sha"]]["samples"]
        corpora_entries[corpus] = g_entries
    if not corpora_entries:
        raise RuntimeError(
            f"[capture shard {shard}] no gen shard files under {raw_root} — run gen first"
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    _equivalence_gate(model, tokenizer)

    rejects: list[dict[str, Any]] = []
    n_written = 0
    for corpus, entries in corpora_entries.items():
        n_written += _capture_corpus(
            fp, args, model, tokenizer, corpus, entries, store_root, rejects
        )

    # digest-only reject record (sha / kind / reason — never text)
    rej = store_root / f"capture_rejects.shard{shard}.jsonl"
    rej_tmp = store_root / f"capture_rejects.shard{shard}.jsonl.tmp"
    with open(rej_tmp, "w", encoding="utf-8") as fh:
        for r in rejects:
            fh.write(json.dumps(r))
            fh.write("\n")
    os.replace(rej_tmp, rej)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _write_sentinel(
        sent,
        fp,
        {"phase": "capture", "shard": shard, "n_prompts": n_written, "n_rejects": len(rejects)},
    )
    logger.info("[phase=capture shard=%d complete] rejects=%d", shard, len(rejects))


def _capture_corpus(
    fp: str,
    args: argparse.Namespace,
    model,
    tokenizer,
    corpus: str,
    entries: list[dict[str, Any]],
    store_root: Path,
    rejects: list[dict[str, Any]],
) -> int:
    """Batched capture in ≤NPZ_SHARD_SIZE-prompt npz shards with per-shard resume."""
    pad_id = int(
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    n_groups = (len(entries) + NPZ_SHARD_SIZE - 1) // NPZ_SHARD_SIZE
    n_done = 0
    for shard_start in range(0, len(entries), NPZ_SHARD_SIZE):
        t0 = time.time()
        group = entries[shard_start : shard_start + NPZ_SHARD_SIZE]
        gi = shard_start // NPZ_SHARD_SIZE
        out = store_root / f"{corpus}.shard{gi:04d}.npz"
        done = store_root / f"{corpus}.shard{gi:04d}.done.json"
        if not args.no_resume and out.exists() and _sentinel_ok(done, fp, resume=True):
            logger.info("[capture] resume-skip %s (%d prompts)", out.name, len(group))
            n_done += len(group)
            continue

        pending: list[tuple[str, list[int], dict[str, int]]] = []
        for e in group:
            sha = e["prompt_sha"]
            p_ids = e.get("prompt_token_ids")
            if p_ids is None:
                raise ValueError(f"gen shard entry {sha} lacks prompt_token_ids (re-run gen)")
            # completion-token-identity: consume the PERSISTED engine-side
            # completion token ids, never a re-encode of the decoded text (a
            # re-encode can diverge at BPE seams; empty-completion rejects key
            # on empty token_ids).
            g_rec = _greedy_record(e)
            g_ids = g_rec.get("token_ids")
            if g_ids is None:
                raise ValueError(f"gen greedy record {sha} lacks token_ids (re-run gen)")
            if not g_ids:
                rejects.append(
                    {
                        "corpus": corpus,
                        "prompt_sha": sha,
                        "kind": "greedy",
                        "reason": "empty_completion",
                    }
                )
                continue
            row_ids, pos = _capture_row_ids_and_positions(p_ids, g_ids, REGEN_MAX_MODEL_LEN)
            pending.append((f"{sha}__greedy", row_ids, pos))
            for k, s in enumerate(e.get("samples", [])):
                s_rec = _sample_record(s)
                s_ids = s_rec.get("token_ids")
                if s_ids is None:
                    raise ValueError(f"gen sample record {sha}.s{k} lacks token_ids (re-run gen)")
                if not s_ids:
                    rejects.append(
                        {
                            "corpus": corpus,
                            "prompt_sha": sha,
                            "kind": f"sample{k}",
                            "reason": "empty_completion",
                        }
                    )
                    continue
                row_ids_k, pos_k = _capture_row_ids_and_positions(p_ids, s_ids, REGEN_MAX_MODEL_LEN)
                pending.append((f"{sha}__sample{k}", row_ids_k, pos_k))

        results: dict[str, dict[str, np.ndarray]] = {}
        n_batches = 0
        for batch in _iter_capture_batches(pending):
            outs = _forward_batch(model, pad_id, [(ids, pos) for _, ids, pos in batch], LAYERS)
            for (key, _, _), summ in zip(batch, outs):
                results[key] = summ
            n_batches += 1
            if n_batches % 25 == 0:
                logger.info(
                    "[capture] %s.shard%04d batch %d (%d/%d rows) elapsed=%.0fs",
                    corpus,
                    gi,
                    n_batches,
                    len(results),
                    len(pending),
                    time.time() - t0,
                )

        payload: dict[str, np.ndarray] = {}
        for e in group:
            sha = e["prompt_sha"]
            gkey = f"{sha}__greedy"
            if gkey not in results:
                continue  # rejected row (empty greedy) — recorded in rejects
            payload[f"{sha}__v_C"] = results[gkey]["v_C"]
            payload[f"{sha}__v_A_greedy"] = results[gkey]["v_A"]
            if "samples" in e:
                # capture-sample-index-compaction fix: persist the ORIGINAL draw
                # index k per stacked row (v_A_sample_idx) so downstream joins by
                # draw index survive rejected-sample compaction.
                idxs = [k for k in range(len(e["samples"])) if f"{sha}__sample{k}" in results]
                if not idxs:
                    rejects.append(
                        {
                            "corpus": corpus,
                            "prompt_sha": sha,
                            "kind": "rollout_mean",
                            "reason": "all_samples_empty",
                        }
                    )
                else:
                    stacked = np.stack(
                        [results[f"{sha}__sample{k}"]["v_A"] for k in idxs], axis=0
                    )  # (K_kept, L, H)
                    payload[f"{sha}__v_A_sample_k"] = stacked
                    payload[f"{sha}__v_A_sample_idx"] = np.asarray(idxs, dtype=np.int64)
                    payload[f"{sha}__v_A_rollout_mean"] = (
                        stacked.astype(np.float32).mean(axis=0).astype(np.float16)
                    )

        # np.savez appends .npz to any name not ending .npz — keep suffix .npz on the tmp.
        tmp = store_root / f"{corpus}.shard{gi:04d}.tmp.npz"
        np.savez(tmp, **payload)
        os.replace(tmp, out)
        _write_sentinel(done, fp, {"n_rows": len(group), "n_captured_keys": len(payload)})
        n_done += len(group)
        logger.info(
            "[capture] unit %d/%d %s.shard%04d rows=%d elapsed=%.0fs",
            gi + 1,
            n_groups,
            corpus,
            gi,
            len(group),
            time.time() - t0,
        )
    return n_done


# ---------------------------------------------------------------------------
# Phase means — cross-shard merge into CONSOLIDATED stores (B1)
# ---------------------------------------------------------------------------


def phase_means(args: argparse.Namespace) -> None:
    """Merge per-gen-shard capture npzs into consolidated ≤MEANS_SHARD_SIZE-prompt
    shards + row_index.jsonl (the HF Hub rejects >10k files per repo dir — the
    round-1 per-sha layout wrote ~13k files into one dir)."""
    fp = _flag_fingerprint(args, "means", None)
    sent = _sentinel_path(args, "means", None)
    if _sentinel_ok(sent, fp, resume=not args.no_resume):
        logger.info("[means] resume-skip (fingerprint match)")
        return

    stores_root = _out_root(args) / "stores"
    merged = _out_root(args) / "stores_merged"
    merged.mkdir(parents=True, exist_ok=True)

    # R3-6 guard 1: a standalone `--phase means` after a PARTIALLY-completed
    # capture must not silently merge (and later upload) a partial store set —
    # require a fingerprint-matching capture sentinel for EVERY shard.
    for s in range(args.n_shards):
        cap_fp = _flag_fingerprint(args, "capture", s)
        if not _sentinel_ok(_sentinel_path(args, "capture", s), cap_fp, resume=True):
            raise RuntimeError(
                f"[means] capture shard {s}/{args.n_shards} incomplete (no fingerprint-"
                "matching sentinel) — run capture to completion first"
            )

    # index: corpus -> prompt_sha -> source capture npz
    index: dict[str, dict[str, Path]] = {}
    for shard_dir in sorted(stores_root.glob("shard*")):
        for npz in sorted(shard_dir.glob("*.shard*.npz")):
            corpus = npz.name.split(".")[0]
            with np.load(npz) as data:
                shas = sorted({k.split("__")[0] for k in data.files})
            for sha in shas:
                index.setdefault(corpus, {})[sha] = npz
    if not index:
        raise RuntimeError(f"[means] no capture npz shards under {stores_root} — run capture first")

    # R3-6 guard 2: expected-sha reconciliation — every gen-manifest prompt sha
    # (minus greedy-kind capture rejects, which drop the whole prompt) must be
    # present in the captured index.
    raw_root = _out_root(args) / "eval_results" / _hf_prefix(args) / "raw_completions"
    rejected: dict[str, set[str]] = {}
    for rej_file in sorted(stores_root.glob("shard*/capture_rejects.shard*.jsonl")):
        with open(rej_file, encoding="utf-8") as fh:
            for line in fh:  # text-mode iteration, never .splitlines() (#950)
                s_line = line.strip()
                if not s_line:
                    continue
                r = json.loads(s_line)
                if r.get("kind") == "greedy":
                    rejected.setdefault(r["corpus"], set()).add(r["prompt_sha"])
    for corpus in CORPORA:
        gen_shas: set[str] = set()
        greedy_dir = raw_root / corpus / "greedy"
        for f in sorted(greedy_dir.glob("shard*_*.json")) if greedy_dir.exists() else []:
            gen_shas.update(e["prompt_sha"] for e in json.loads(f.read_text(encoding="utf-8")))
        if not gen_shas:
            continue
        want = gen_shas - rejected.get(corpus, set())
        have = set(index.get(corpus, {}))
        missing_shas = sorted(want - have)
        if missing_shas:
            raise RuntimeError(
                f"[means] corpus {corpus}: {len(missing_shas)}/{len(want)} gen prompts "
                f"missing from capture stores (first 5: {missing_shas[:5]}) — capture "
                "is incomplete"
            )

    row_index: list[dict[str, Any]] = []
    n_merged = 0
    for corpus in sorted(index):
        shas = sorted(index[corpus])
        n_shards_c = (len(shas) + MEANS_SHARD_SIZE - 1) // MEANS_SHARD_SIZE
        for gi, start in enumerate(range(0, len(shas), MEANS_SHARD_SIZE)):
            t0 = time.time()
            group = shas[start : start + MEANS_SHARD_SIZE]
            shard_name = f"{corpus}.means{gi:04d}.npz"
            by_src: dict[Path, list[str]] = {}
            for sha in group:
                by_src.setdefault(index[corpus][sha], []).append(sha)
            payload: dict[str, np.ndarray] = {}
            for src, src_shas in by_src.items():
                with np.load(src) as data:
                    wanted = set(src_shas)
                    for key in data.files:
                        if key.split("__")[0] in wanted:
                            payload[key] = data[key]
            tmp = merged / f"{corpus}.means{gi:04d}.tmp.npz"
            np.savez(tmp, **payload)
            os.replace(tmp, merged / shard_name)
            for sha in group:
                row_index.append({"prompt_sha": sha, "corpus": corpus, "shard_file": shard_name})
            n_merged += len(group)
            logger.info(
                "[means] unit %d/%d %s rows=%d elapsed=%.0fs",
                gi + 1,
                n_shards_c,
                shard_name,
                len(group),
                time.time() - t0,
            )

    ri = merged / "row_index.jsonl"
    ri_tmp = merged / "row_index.jsonl.tmp"
    with open(ri_tmp, "w", encoding="utf-8") as fh:
        for r in row_index:
            fh.write(json.dumps(r))
            fh.write("\n")
    os.replace(ri_tmp, ri)
    _write_sentinel(sent, fp, {"phase": "means", "n_merged": n_merged})
    logger.info("[phase=means complete] merged=%d prompts", n_merged)


# ---------------------------------------------------------------------------
# Phase upload — consolidated stores (bulk, verified) + raw re-verify
# ---------------------------------------------------------------------------


def phase_upload(args: argparse.Namespace) -> None:
    # R3-8a: idempotency — a restart after a completed upload must not repeat
    # both network uploads (fingerprint-matching sentinel skips).
    fp = _flag_fingerprint(args, "upload", None)
    sent = _sentinel_path(args, "upload", None)
    if _sentinel_ok(sent, fp, resume=not args.no_resume):
        logger.info("[upload] resume-skip (fingerprint match)")
        return

    # B9: completion guard — never upload from an incomplete means phase.
    means_fp = _flag_fingerprint(args, "means", None)
    if not _sentinel_ok(_sentinel_path(args, "means", None), means_fp, resume=True):
        raise RuntimeError("upload: means phase incomplete (no fingerprint-matching sentinel)")

    prefix = _hf_prefix(args)

    # Raw completions: phase_gen already uploaded per shard (#779). Re-upload is a
    # cheap idempotent backstop; the exact-set verify below is the binding check.
    expected: list[str] = []
    raw_root = _out_root(args) / "eval_results" / prefix / "raw_completions"
    if raw_root.exists():
        base_url = hub._upload(
            local_path=raw_root,
            repo_id=DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{prefix}/raw_completions",
            raise_on_error=True,
        )
        if not base_url:
            raise RuntimeError(f"raw_completions upload returned no path ({prefix})")
        expected = [
            f"{c}/{kind}/{f.name}"
            for c in CORPORA
            for kind in RAW_KINDS
            for f in sorted((raw_root / c / kind).glob("shard*_*.json"))
        ]
        if not expected:
            raise RuntimeError(f"[upload] raw_completions dir {raw_root} holds no chunk files")
        missing = hub.verify_repo_paths_uploaded(
            HfApi(),
            DATA_REPO,
            expected,
            path_in_repo=f"{prefix}/raw_completions",
            repo_type="dataset",
        )
        if missing:
            raise RuntimeError(f"raw_completions upload incomplete; missing: {missing}")
        logger.info("[upload] raw_completions verified %d files", len(expected))

    merged = _out_root(args) / "stores_merged"
    if not merged.exists():
        raise RuntimeError(f"[upload] consolidated store dir {merged} missing — run means first")
    # Consolidated shards: ~(n_prompts / MEANS_SHARD_SIZE) files + row_index.jsonl,
    # far below the Hub's 10k-files-per-dir cap by construction (B1).
    base_url = hub._upload(
        local_path=merged,
        repo_id=DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{prefix}/summary_stores",
        raise_on_error=True,
    )
    if not base_url:
        raise RuntimeError(f"summary_stores upload returned no path ({prefix})")
    expected_stores = sorted(p.name for p in merged.glob("*.npz")) + ["row_index.jsonl"]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        DATA_REPO,
        expected_stores,
        path_in_repo=f"{prefix}/summary_stores",
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"summary_stores upload incomplete; missing: {missing}")
    logger.info("[upload] summary_stores verified %d files", len(expected_stores))

    # provenance sidecar
    meta = {"issue": ISSUE, "prefix": prefix, **as_metadata_dict(git_provenance())}
    (_out_root(args) / "upload_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_sentinel(
        sent,
        fp,
        {"phase": "upload", "n_raw_files": len(expected), "n_store_files": len(expected_stores)},
    )
    logger.info("[phase=upload complete]")


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


def _resolve_gpu_alloc(n_shards: int) -> list[str]:
    """Physical GPU ids for the shard fan-out, honoring an INHERITED restricted
    allocation (#1336: `CUDA_VISIBLE_DEVICES=str(shard)` escapes a restricted CVD
    like "2,3"). Precedence: inherited CVD > SLURM allocation env > 0..N-1."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        alloc = [t for t in (p.strip() for p in cvd.split(",")) if t]
    elif os.environ.get("SLURM_JOB_ID"):
        raw = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")
        if raw:
            alloc = [t for t in (p.strip() for p in raw.split(",")) if t]
        else:
            # gpu-allocation-chain fix: SLURM_GPUS_ON_NODE is a COUNT-only var
            # some SLURM configs export instead of the id lists; a whole-node
            # count allocation exposes ordinals 0..N-1.
            n_on_node = os.environ.get("SLURM_GPUS_ON_NODE")
            if not n_on_node:
                raise RuntimeError(
                    "SLURM job with no CUDA_VISIBLE_DEVICES / SLURM_JOB_GPUS / "
                    "SLURM_STEP_GPUS / SLURM_GPUS_ON_NODE — cannot derive the GPU "
                    "allocation (never fall back to the physical count)"
                )
            alloc = [str(i) for i in range(int(n_on_node))]
    else:
        alloc = [str(i) for i in range(n_shards)]
    if len(alloc) < n_shards:
        raise RuntimeError(f"n_shards={n_shards} exceeds the GPU allocation {alloc}")
    return alloc


def _tail_lines(path: Path, n: int = 120) -> list[str]:
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            fh.seek(max(0, size - 65536))
            data = fh.read()
        return data.decode("utf-8", errors="replace").split("\n")[-n:]
    except OSError:
        return ["<log unreadable>"]


def _fan_out(args: argparse.Namespace, phase: str) -> int:
    """Launch one subprocess per shard with ALLOC[shard] pinned in the LAUNCHER env
    (defeats the #545 import-time-cuInit clobber), each child writing its OWN log
    file (a child inheriting dispatcher stdout could land `[phase=...]` lines —
    or a stray `[phase=done]` — in the poller-scanned main log, #545/#1315),
    wait, echo a failed child's log tail, then write a merged done-sentinel."""
    alloc = _resolve_gpu_alloc(args.n_shards)
    log_dir = _out_root(args) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[int, subprocess.Popen, Any, Path]] = []
    for shard in range(args.n_shards):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": alloc[shard]}
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--phase",
            phase,
            "--shard",
            str(shard),
            "--n-shards",
            str(args.n_shards),
            "--out-root",
            str(_out_root(args)),
            "--sentinel-dir",
            args.sentinel_dir,
        ]
        if args.smoke:
            cmd.append("--smoke")
        if args.no_resume:
            cmd.append("--no-resume")
        log_path = log_dir / f"{phase}.shard{shard}.log"
        lf = open(log_path, "ab")
        logger.info("[fan-out] %s shard %d CVD=%s log=%s", phase, shard, alloc[shard], log_path)
        procs.append(
            (
                shard,
                subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT),
                lf,
                log_path,
            )
        )

    rc = 0
    for shard, p, lf, log_path in procs:
        code = p.wait()
        lf.close()
        logger.info("[fan-out] %s shard %d exited rc=%d", phase, shard, code)
        if code != 0:
            for line in _tail_lines(log_path):
                logger.error("[shard %d tail] %s", shard, line)
        rc = rc or code
    if rc != 0:
        return rc

    fp = _flag_fingerprint(args, phase, None)
    _write_sentinel(
        _sentinel_path(args, phase, None),
        fp,
        {"phase": phase, "n_shards": args.n_shards, "merged": True},
    )
    return 0


# ---------------------------------------------------------------------------
# Poller-facing results sentinel (B7)
# ---------------------------------------------------------------------------


def _write_poller_sentinel(args: argparse.Namespace) -> None:
    """Results sentinel conforming to ``poll_pipeline._SENTINEL_REQUIRED_KEYS``,
    written to the drained namespace (``--sentinel-dir``, default /workspace/logs)
    as ``issue-<N>-<kind_slug>-<epoch>.json``. Smoke runs write kind
    ``epm:smoke-result`` (never a smoke flag on ``epm:results``); version is
    hardcoded 1 (drain-side max+1 rewrite, #1095). Written ONCE, before the
    reserved ``[phase=done]`` line."""
    kind = "epm:smoke-result" if args.smoke else "epm:results"
    sdir = Path(args.sentinel_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    note = {
        "phase_arg": args.phase,
        "smoke": bool(args.smoke),
        "out_root": str(_out_root(args)),
        "hf_data_repo": DATA_REPO,
        "hf_prefix": _hf_prefix(args),
        "raw_completions": f"{_hf_prefix(args)}/raw_completions/",
        "summary_stores": f"{_hf_prefix(args)}/summary_stores/",
        **as_metadata_dict(git_provenance()),
    }
    payload = {
        "sentinel_schema_version": 1,
        "task_id": ISSUE,
        "kind": kind,
        "version": 1,
        "smoke": bool(args.smoke),
        "gate": None,
        "blocks_pipeline": False,
        "note": json.dumps(note),
        "by": "issue2356_pod",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    slug = kind.replace(":", "_")
    path = sdir / f"issue-{ISSUE}-{slug}-{int(time.time())}.json"
    _write_json_atomic(path, payload)
    logger.info("[sentinel] wrote %s", path)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="issue2356 pod dispatcher (gen/capture/means/upload)")
    ap.add_argument("--phase", choices=["gen", "capture", "means", "upload", "all"], default="all")
    ap.add_argument("--shard", type=int, default=None, help="single-shard worker (else fan out)")
    ap.add_argument("--n-shards", type=int, default=2, help="row-shard / GPU count")
    ap.add_argument("--smoke", action="store_true", help="tiny stratified slice; smoke HF prefix")
    ap.add_argument("--no-resume", action="store_true", help="ignore sentinels; recompute")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="pod-local output root")
    ap.add_argument(
        "--sentinel-dir",
        default="/workspace/logs",
        help="poller-drained sentinel dir (poll_pipeline glob issue-<N>-*.json)",
    )
    ap.add_argument(
        "--import-check", action="store_true", help="verify imports + args attrs; exit 0"
    )
    ap.add_argument(
        "--equivalence-check",
        action="store_true",
        help="CPU fp32 tiny-model batched-vs-serial equivalence check of _forward_batch; exit 0",
    )
    return ap


def _run_phase(args: argparse.Namespace, phase: str) -> int:
    if phase in ("means", "upload"):
        logger.info("[phase=%s]", phase)
        (phase_means if phase == "means" else phase_upload)(args)
        return 0
    # gen / capture are sharded
    if args.shard is not None:
        logger.info("[phase=%s shard=%d]", phase, args.shard)
        (phase_gen if phase == "gen" else phase_capture)(args, args.shard)
        return 0
    return _fan_out(args, phase)


def main() -> int:
    args = build_argparser().parse_args()

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # execute the deferred imports so import-check verifies them (#606)
        from transformers import Qwen2Config  # noqa: F401  (deferred: _equivalence_selftest)

        from explore_persona_space.eval.generation import (  # noqa: F401
            cleanup_vllm,
            create_vllm_engine,
        )

        logger.info("[import-check] imports + args attributes OK")
        return 0

    if args.equivalence_check:
        rc = _equivalence_selftest()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)

    phases = ["gen", "capture", "means", "upload"] if args.phase == "all" else [args.phase]
    try:
        for phase in phases:
            rc = _run_phase(args, phase)
            if rc != 0:
                logger.error("[phase=%s FAILED rc=%d]", phase, rc)
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(rc)
    except BaseException:
        # R3-8b: an exception propagating through interpreter finalization with
        # live vLLM engine/worker children DEADLOCKS (#1739/#2149) — a crashed
        # shard would then hang on a billing pod. Log the traceback, flush,
        # hard-exit.
        logger.exception("[phase-dispatch] unhandled exception; hard-exit rc=1")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    if args.shard is None and args.phase == "all":
        # Reserved terminal emission: DISPATCHER full-pipeline runs only (#545) —
        # sentinel FIRST, then the [phase=done] line (pod-side-reporting.md).
        _write_poller_sentinel(args)
        logger.info("[phase=done]")
    elif args.shard is None:
        logger.info(
            "[phase=%s complete] (single-phase dispatcher run; no terminal line)", args.phase
        )
    else:
        logger.info("[shard-worker %d exit ok]", args.shard)
    # os._exit after flush: a vLLM/torch generation driver deadlocks at
    # finalization on unreaped engine/worker children (#1739/#2149).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
