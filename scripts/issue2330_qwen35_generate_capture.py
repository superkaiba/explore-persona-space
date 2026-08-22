#!/usr/bin/env python3
"""Task #2330 P1/P2: Qwen3.5-9B generate + trimmed capture (STANDALONE port).

Port of ``scripts/issue1491_ladder_generate_capture.py`` (plan §4 P1: same
``_render_prompt`` / ``_sampling_params`` / ``_segment_token_ids`` / capture /
upload machinery) with every ``explore_persona_space`` / ``issue779_*`` /
``issue1491_*`` import DROPPED so it runs in the pod-side fresh venv
(``uv venv /workspace/venvs/qwen35 && uv pip install vllm==0.27.1`` — brings a
qwen3_5-supporting transformers >= 5.2). Deps: stdlib + torch + transformers +
huggingface_hub (+ vllm, deferred to generation paths only).

Deviations from the parent driver (each plan-mandated):
  - ``enable_thinking=False`` threaded into EVERY ``apply_chat_template`` call
    (plan §11; Qwen3.5 has no /nothink soft switch).
  - Template pin: the rendered prompt must END with the empty-think-block
    header ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`` (token-level
    assert on EVERY row via ``_segment_token_ids`` — the v_C position assert,
    plan §4 P1 gates 1+6). ``--expect-suffix plain`` selects the Qwen2.5 shape
    for the capture-port parity leg.
  - Splits come from ``eval_results/issue_2330/split_ids.json`` (P0 output,
    the single source every count pin reads): ``SPLIT_TO_MANIFEST`` maps split
    -> (#1491 manifest file @ pin 815ff6d, split_ids key, generation seed).
  - fp32 capture by default (``--capture-dtype float32``; plan §4 P2 / §12
    A15 — 9B fp32 weights ~36 GB fit one H200). The parity-vs-banked leg runs
    ``--capture-dtype bfloat16`` (the parent computed the banked captures in
    bf16, so bf16 recompute minimizes dtype-mismatch noise under the plan's
    cosine >= 0.999 tolerance).
  - The #1491 first-chunk self-gate (plan-§7-of-1491 machinery) is NOT ported:
    #2330's validity gates are the P1 convention gates below + P3's port-parity
    anchor gate (plan §4/§7 enumerate them; the ladder self-gate is not among
    them).
  - vLLM engine teardown routes through ``_reap_vllm_engine`` and the process
    terminal is ``os._exit`` after explicit flushes whenever an engine was
    constructed (gotchas.md: ``sys.exit(0)`` is NOT a terminal for a vLLM
    generation driver — finalize-time multiprocessing cleanup can deadlock on
    surviving engine children).

P1 convention gates (``--gate``, plan §4 P1 steps 1-6, all fail-loud):
  template_pin  step 1: render 3 probe prompts, assert the empty-think-block
                suffix (text + token ids), record realized header ids to run
                meta.
  length_scan   step 2: tokenize ALL distinct consumed prompts (train_10k +
                val_400 + test_1000 + wc_test_1k = 12,399 nominal) under the
                Qwen3.5 tokenizer; over-budget (> 7,104) rows are DROPPED from
                the split lists in split_ids.json (field ``dropped_overlength``
                appended, shas + counts recomputed); > 0.5% over-budget HALTS
                (exit 4) without mutating split_ids.
  emit_spans    support for step 5(a): tokenizer-only segmentation of the
                pinned banked 7B rows, spans written to ``--spans-out``. Run
                once in the REPO env (transformers 4.57.6 == the parent's
                stack) to produce the reference the fresh-venv gate compares
                against.
  parity7b      step 5: teacher-forced CAPTURE on Qwen/Qwen2.5-7B-Instruct
                over the pinned banked ``scale7_refit/train_25k`` rows @
                815ff6d; asserts (a) exact token-id spans vs
                ``--expected-spans`` and (b) per-row cx_last/v_x cosine >=
                0.999 vs the banked captures at layers {14,19,26}. Miss =>
                halt (failure_class: code — the capture port is broken).
  hook_probe    step 6: on 4 fixed probe rows through the production capture
                entrypoint, assert hooked block-k output == hidden_states[k+1]
                (exact shape, rel <= 1e-5) on the 9B at blocks {16,22,30}, plus
                the v_C-position / header-suffix assert; results persisted to
                run meta (closes plan §12 A4).
  (step 3 — the 500-row smoke shard — is NOT a gate mode: smoke IS this
  driver's production entrypoint at ``--num-shards 20 --shard-index 0
  --shard-size 500 --no-upload`` per the plan's smoke/sweep-parity clause.)
  (step 4 — fits-shape smoke — is the unit-2 fits script's job; ``--fits-smoke``
  is a documented fail-loud stub until unit 2 lands.)

Production (P2), one process per GPU with CUDA_VISIBLE_DEVICES pinned in the
LAUNCHER env (this script never sets CVD itself):

  # per split in {train_10k, val_400, test_1000, wc_test_1k,
  #               ceiling_draw_43, ceiling_draw_44}, shards 0/1:
  CUDA_VISIBLE_DEVICES=0 python scripts/issue2330_qwen35_generate_capture.py \
      --split train_10k --capture-mode phase_split_gen \
      --hf-prefix issue2330_matched/qwen35_9b --h-dim 4096 \
      --num-shards 2 --shard-index 0 -v
  # then the capture wave: --capture-mode phase_split_capture (same args).

Uploads land at ``<hf_prefix>/<split>/{final_token_capture,raw_completions}/``
(ceiling draws: ``<hf_prefix>/ceiling_draws/seed{43,44}/...`` — the banked 7B
layout P3 consumes symmetrically). Rollout TEXT uploads unconditionally
(persist-by-default); per-split cap-hit and <think>-leak fractions are
reported in the shard digest, the latter with a hard < 1% assert
(thinking-off actually engaged — plan §7 validity assert).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent

# vLLM v1 EngineCore dies silently under fork() when the parent touched
# tokenizers/transformers before LLM() — set spawn BEFORE any vllm import
# (gotchas.md #628; vllm reads the var at import time).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _load_dotenv() -> None:
    """Minimal .env loader (standalone — no explore_persona_space import).

    setdefault semantics (never clobbers explicit env); first existing path
    wins: $EPM_DOTENV_PATH, <repo_root>/.env, /workspace/explore-persona-space/.env.
    """
    candidates = [
        os.environ.get("EPM_DOTENV_PATH"),
        str(_REPO_ROOT / ".env"),
        "/workspace/explore-persona-space/.env",
    ]
    for cand in candidates:
        if not cand or not Path(cand).is_file():
            continue
        for line in Path(cand).read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key:
                os.environ.setdefault(key, val)
        return


_load_dotenv()

import torch  # noqa: E402

logger = logging.getLogger("issue2330_qwen35_generate_capture")

# ---------------------------------------------------------------------------
# Constants (parent parity + #2330 plan §11)
# ---------------------------------------------------------------------------

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

# Pinned #1491 reuse sources (plan §10 reproducibility card).
MANIFEST_HF_PREFIX = "issue1491_scale_ladder/manifest"
MANIFEST_REVISION = "815ff6d976c686af8672b27cfdfb1ce6b419c02c"
PARITY_BANKED_PREFIX = "issue1491_scale_ladder/scale7_refit/train_25k"

# vLLM engine limits (parent parity: issue1491_ladder_generate_capture.py:136-139).
MAX_MODEL_LEN = 8192
GEN_MAX_TOKENS = 1024
LENGTH_MARGIN = 64
PROMPT_TOKEN_BUDGET = MAX_MODEL_LEN - GEN_MAX_TOKENS - LENGTH_MARGIN
assert PROMPT_TOKEN_BUDGET == 7104, f"budget drift: {PROMPT_TOKEN_BUDGET} != 7104 (plan §4 P1)"


def _apply_gen_max_tokens(n: int) -> tuple[int, int, int]:
    """Rebind the generation cap (cap2048 follow-up): GEN_MAX_TOKENS <- n with
    PROMPT_TOKEN_BUDGET held INVARIANT at 7104 and MAX_MODEL_LEN derived
    (budget + cap + margin). Holding the BUDGET fixed keeps the admitted row
    set byte-identical to the cap-1024 originals (a shrunk budget would drop
    tokenizer-dependent long-prompt tails and break the matched-ID row
    alignment); raising max_model_len instead is the CLAUDE.md inherited-rig
    rule for a raised cap (#505/#601). Returns the realized
    (gen_max_tokens, max_model_len, prompt_token_budget) triple; asserts the
    arithmetic re-derives (fail-loud, never a silent drift)."""
    global GEN_MAX_TOKENS, MAX_MODEL_LEN
    assert n >= 1, f"--gen-max-tokens must be >= 1, got {n}"
    GEN_MAX_TOKENS = int(n)
    MAX_MODEL_LEN = PROMPT_TOKEN_BUDGET + GEN_MAX_TOKENS + LENGTH_MARGIN
    assert PROMPT_TOKEN_BUDGET == MAX_MODEL_LEN - GEN_MAX_TOKENS - LENGTH_MARGIN
    return GEN_MAX_TOKENS, MAX_MODEL_LEN, PROMPT_TOKEN_BUDGET


# Sampling params (recipe identity with the banked 7B targets — plan §11,
# grounded at issue1491_ladder_generate_capture.py:148-150).
GEN_TEMP = 1.0
GEN_TOP_P = 0.95
GEN_SEED_DEFAULT = 42

# Assistant turn-end tail appended after the response inside the v_x span
# (parent parity — the #779/#1491 v(x) convention includes it).
IM_END_TAIL = "<|im_end|>\n"

# Rendered-prompt suffixes. "think" = Qwen3.5 with enable_thinking=False
# (plan §12 A2: the template appends the empty think block); "plain" = the
# Qwen2.5-Instruct shape (capture-port parity leg).
THINK_SUFFIX_TEXT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
PLAIN_SUFFIX_TEXT = "<|im_start|>assistant\n"

# Pre-registered cap-hit reporting trigger (CLAUDE.md cap-hit accounting; the
# #2330 disposition is the #1491 truncation-restriction control, not re-gen —
# plan §11 — so this only WARNs).
CAP_HIT_REGEN_TRIGGER = 0.02

# Cap-hit aggregate schema token (round 2, cap-hit-control-unwired; round 3
# bumps v1→v2: exact-coverage invariants + the expected-id-set sha256
# fingerprint + logical-split/store-split routing fields — consumers REQUIRE
# the new fields, so pre-fingerprint v1 aggregates fail loud here instead of
# KeyError-ing downstream). Produced by --aggregate-cap-hit, consumed
# FAIL-LOUD by issue2330_matched_fits.py (--cap-hit-dir,
# truncation-restriction read+refit control) and issue2330_figures.py
# (--cap-hit-dir, fig_cap_hit). Keep the three literals in sync.
CAP_HIT_SCHEMA = "issue2330_cap_hit_v2"

# <think>-leak validity assert (plan §7: < 1% of responses open with <think>).
THINK_SCAN_MAX_FRAC = 0.01

DEFAULT_SHARD_SIZE = 500
UPLOAD_BATCH = int(os.environ.get("EPM_I2330_UPLOAD_BATCH", "20"))
VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))

# split -> (#1491 manifest split file key, split_ids.json key, generation seed).
# Ceiling draws re-render the SAME test_1000 prompts at seeds 43/44 (plan §4 P2).
SPLIT_TO_MANIFEST = {
    "train_10k": ("train_25k", "train_10k", GEN_SEED_DEFAULT),
    "val_400": ("val_400", "val_400", GEN_SEED_DEFAULT),
    "test_1000": ("test_1000", "test_1000", GEN_SEED_DEFAULT),
    "wc_test_1k": ("wc_test_1k", "wc_test_1k", GEN_SEED_DEFAULT),
    "ceiling_draw_43": ("test_1000", "test_1000", 43),
    "ceiling_draw_44": ("test_1000", "test_1000", 44),
}

MANIFEST_SPLIT_FILES = {
    "train_25k": "train_25k.jsonl",
    "val_400": "val_400.jsonl",
    "test_1000": "test_1000.jsonl",
    "wc_test_1k": "wc_test_1k.jsonl",
}


def store_subpath_for_split(split: str) -> str:
    """Logical split -> store subpath under an hf-prefix root.

    THE single routing table (round 3, cap-hit-control-unwired FIX 1b): both
    ``run_capture``'s stage_prefix (the WRITE side) and the cap-hit
    aggregator's chunk prefix (the READ side) compose paths through this
    function, so the two can never drift. Ceiling draws land under
    ``ceiling_draws/seed{S}`` (generation re-renders test_1000 at seeds
    43/44); every other split lives under its own name. The banked 7B
    train store is the one case where the STORE subpath differs from the
    logical split's canonical subpath (train_25k holds the train_10k rows)
    — the aggregator's ``--cap-hit-store-split`` override + split-ID
    subsetting handles it (never a new logical split)."""
    if split.startswith("ceiling_draw_"):
        _, _, gen_seed = SPLIT_TO_MANIFEST[split]
        return f"ceiling_draws/seed{gen_seed}"
    return split


# Split_ids keys the length scan covers = every DISTINCT consumed prompt set
# (ceiling draws reuse test_1000's renders; plan §12 A5 count fix: 12,399).
LENGTH_SCAN_KEYS = ("train_10k", "val_400", "test_1000", "wc_test_1k")

_ENGINE_CONSTRUCTED = False  # set by _build_engine; drives the os._exit terminal
_LIVE_ENGINE = None  # last-constructed engine handle (the __main__ exception-teardown guard)

# Required P1 gate run_meta keys for the plan-§9 P1 completion sentinel
# (round 2, p1-sentinel-no-writer). emit_spans is a support mode on the 7B
# leg (its output is parity7b's input), so parity7b subsumes it.
P1_SENTINEL_REQUIRED = (
    "template_pin",
    "length_scan",
    "smoke_shard",
    "fits_smoke",
    "parity7b",
    "hook_probe",
)


# ---------------------------------------------------------------------------
# Small standalone utilities (ports of orchestrate.hub / issue779_common bits)
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Tolerant provenance sha (#1902: never check=True — a git-less lane must
    degrade to the literal, not kill the workload)."""
    env_sha = os.environ.get("EPS_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_SCRIPTS),
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable-no-git-checkout"
    if proc.returncode == 0:
        return proc.stdout.strip()
    return "unavailable-no-git-checkout"


def _phase(name: str) -> None:
    """poll_pipeline-parseable phase breadcrumb ([phase=done] is terminal-only)."""
    print(f"[phase={name}]", flush=True)


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    tmp.replace(path)


def _update_run_meta(path: Path, key: str, record: dict) -> None:
    """Read-modify-write the accumulating run-meta JSON (repro card carrier)."""
    meta: dict = {}
    if path.exists():
        meta = json.loads(path.read_text(encoding="utf-8"))
    meta[key] = record
    meta.setdefault("_meta", {})
    meta["_meta"]["git_commit"] = _git_sha()
    meta["_meta"]["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json_atomic(path, meta)
    logger.info("[run-meta] updated %s (%s)", path, key)


def _is_transient_hub_error(e: Exception) -> bool:
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status in (408, 409, 425, 429, 500, 502, 503, 504):
        # 409: "Another commit operation is in progress" — sibling shard's
        # concurrent commit to the same data repo; clears in seconds.
        return True
    names = {t.__name__ for t in type(e).__mro__}
    return bool(
        {
            "Timeout",
            "ConnectTimeout",
            "ReadTimeout",
            "ConnectionError",
            "ChunkedEncodingError",
            "ProtocolError",
        }
        & names
    )


def _retry_transient(fn, *, what: str, max_attempts: int = 6):
    """Standalone port of hub.retry_transient: bounded attempts + wall budget,
    Retry-After honored, non-transient errors re-raise immediately."""
    budget_s = float(os.environ.get("EPM_HF_RETRY_BUDGET_S", "1800"))
    start = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — predicate-filtered, re-raised when non-transient
            if not _is_transient_hub_error(e):
                raise
            now = time.monotonic()
            within_attempts = attempt < max_attempts
            within_budget = budget_s > 0 and (now - start) < budget_s
            if not (within_attempts or within_budget):
                logger.warning("%s: transient-retry exhausted after %d calls", what, attempt)
                raise
            retry_after = getattr(getattr(e, "response", None), "headers", {}) or {}
            try:
                sleep_s = min(float(retry_after.get("Retry-After", "")), 900.0)
            except (TypeError, ValueError):
                sleep_s = min(180.0, 10.0 * 2.0 ** min(attempt - 1, 6)) * (
                    1.0 + random.random() * 0.25
                )
            logger.warning(
                "%s: transient error (%s: %s) — retry %d in %.0fs",
                what,
                type(e).__name__,
                e,
                attempt,
                sleep_s,
            )
            time.sleep(sleep_s)


def _hf_api():
    from huggingface_hub import HfApi

    return HfApi()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _remote_index(prefix: str, revision: str | None = None) -> dict[str, dict]:
    """{basename: {size, sha256}} for a data-repo prefix — SCOPED list_repo_tree
    only (never a bare full-repo listing on the ~1M-file data repo), retried on
    transients (a transient must NOT read as 'nothing uploaded', which would
    disable resume). 404 on a not-yet-created prefix -> empty; a missing REPO
    (RepositoryNotFoundError) stays loud."""
    from huggingface_hub.errors import (
        EntryNotFoundError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    api = _hf_api()

    def _list():
        # Materialize INSIDE the retry: list_repo_tree is a LAZY generator.
        return list(
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in the module's own _retry_transient port
            api.list_repo_tree(
                repo_id=HF_DATA_REPO,
                path_in_repo=prefix,
                repo_type="dataset",
                revision=revision,
                recursive=True,
            )
        )

    try:
        entries = _retry_transient(_list, what=f"list_repo_tree {prefix}")
    except RepositoryNotFoundError:
        raise  # subclass of HfHubHTTPError — must stay loud
    except EntryNotFoundError:
        return {}
    except HfHubHTTPError as e:
        if getattr(getattr(e, "response", None), "status_code", None) == 404:
            return {}
        raise
    out: dict[str, dict] = {}
    for f in entries:
        if f.path.endswith("/") or getattr(f, "size", None) is None:
            continue
        lfs = getattr(f, "lfs", None)
        out[f.path.rsplit("/", 1)[-1]] = {
            "size": f.size,
            "sha256": getattr(lfs, "sha256", None) if lfs else None,
        }
    return out


def _hub_download(filename: str, cache_dir: Path, revision: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        _retry_transient(
            # NO_RETRY: wrapped in the module's own _retry_transient port (standalone file)
            lambda: hf_hub_download(
                repo_id=HF_DATA_REPO,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=str(cache_dir),
            ),
            what=f"hf_hub_download {filename}",
        )
    )


# ---------------------------------------------------------------------------
# Manifest + split_ids loading
# ---------------------------------------------------------------------------


def _download_manifest_split(manifest_key: str, cache_dir: Path) -> list[dict]:
    """Download + read one #1491 manifest split at the PINNED revision.

    Text-mode line iteration (never splitlines() — real-user text can carry
    U+2028/U+2029, gotchas.md)."""
    fname = MANIFEST_SPLIT_FILES[manifest_key]
    local = _hub_download(f"{MANIFEST_HF_PREFIX}/{fname}", cache_dir, MANIFEST_REVISION)
    rows: list[dict] = []
    with open(local, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_split_ids(path: Path) -> dict:
    assert path.is_file(), (
        f"split_ids.json missing at {path} — run scripts/issue2330_split_ids.py (P0) first "
        "and make sure the issue branch carrying it is checked out"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "splits" in payload and "sha256" in payload, (
        "split_ids.json schema drift",
        sorted(payload.keys()),
    )
    return payload


def _subset_rows(manifest_rows: list[dict], ids: list[int], ids_key: str) -> list[dict]:
    """Subset manifest rows to the split_ids id list, in id-list order.

    Every id MUST resolve (fail loud — a miss means manifest/split_ids drift);
    ladder_local_id is REQUIRED on every row (schema probed at the pin)."""
    by_id: dict[int, dict] = {}
    for r in manifest_rows:
        assert "ladder_local_id" in r, ("manifest row missing ladder_local_id", sorted(r.keys()))
        by_id[int(r["ladder_local_id"])] = r
    missing = [i for i in ids if i not in by_id]
    assert not missing, (
        f"{ids_key}: {len(missing)} split_ids ids absent from the manifest "
        f"(first: {missing[:10]}) — manifest/split_ids drift"
    )
    return [by_id[i] for i in ids]


# ---------------------------------------------------------------------------
# Tokenizer / render / segmentation (parent machinery + think-suffix pin)
# ---------------------------------------------------------------------------


def _render_prompt(tok, prompt: str) -> str:
    """The EXACT prompt render vLLM generation consumes (and the render the
    over-length filter budgets against). ``enable_thinking=False`` is inert on
    templates that never reference it (Qwen2.5) and disables the think block
    on Qwen3.5 (plan §11)."""
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _expected_suffix(expect_suffix: str) -> str:
    assert expect_suffix in ("think", "plain"), expect_suffix
    return THINK_SUFFIX_TEXT if expect_suffix == "think" else PLAIN_SUFFIX_TEXT


def _load_tokenizer(model_id: str, expect_suffix: str):
    """Tokenizer load + template pin (plan §4 P1 gate 1 semantics, enforced at
    EVERY load): the rendered probe prompt must end with the expected
    assistant-header suffix, text-level AND token-level.

    Returns (tok, suffix_text, suffix_ids)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    want = _expected_suffix(expect_suffix)
    probe = _render_prompt(tok, "hi")
    assert probe.endswith(want), (
        f"template pin FAIL: rendered prompt does not end with the expected "
        f"{expect_suffix!r} suffix; realized tail: {probe[-80:]!r}"
    )
    suffix_ids = tok(want, add_special_tokens=False)["input_ids"]
    probe_ids = tok(probe, add_special_tokens=False)["input_ids"]
    assert len(suffix_ids) >= 1 and probe_ids[-len(suffix_ids) :] == suffix_ids, (
        "template pin FAIL: rendered prompt token ids do not end with the suffix tokenization",
        suffix_ids,
        probe_ids[-len(suffix_ids) :],
    )
    return tok, want, suffix_ids


def _rendered_prompt_token_len(tok, prompt: str) -> int:
    """Token length of the render EXACTLY as generation consumes it (the length
    vLLM validates against max_model_len)."""
    return len(tok(_render_prompt(tok, prompt), add_special_tokens=False)["input_ids"])


def _filter_overlength_prompts(prompts, cis, token_len_fn, budget):
    """Partition (prompts, cis) into kept vs skipped by rendered token length
    (verbatim parent port — over-length rows are ENGINE-FATAL at vLLM
    add_request, so the filter is load-bearing even post-length-scan).
    Refusal-safe: records ci + token count, never text."""
    kept_prompts, kept_cis, skipped = [], [], []
    for p, ci in zip(prompts, cis, strict=True):
        n = token_len_fn(p)
        if n > budget:
            skipped.append({"ci": int(ci), "n_tokens": int(n)})
        else:
            kept_prompts.append(p)
            kept_cis.append(int(ci))
    return kept_prompts, kept_cis, skipped


def _is_empty_response(resp: str) -> bool:
    return not resp.strip()


def _opens_with_think(resp: str) -> bool:
    """<think>-leak scan predicate (plan §7: thinking-off actually engaged)."""
    return resp.lstrip().startswith("<think>")


def _segment_token_ids(
    tok, prompt: str, response: str, suffix_ids: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-segment token ids for the teacher-forced capture input.

    NEVER re-tokenize the concatenated string (gotchas.md BPE-seam rule): the
    forward input is torch.cat([prompt_ids, resp_ids, tail_ids]), so the
    prompt segment is bit-identical to what vLLM generation consumed BY
    CONSTRUCTION and cx_last sits at p_len-1 exactly.

    The prompt segment must END with the realized assistant-header suffix
    (``suffix_ids`` from the template pin) — this is the per-row v_C position
    assert (plan §4 P1 gate 6b), enforced on EVERY captured row."""
    prompt_text = _render_prompt(tok, prompt)
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    r_ids = tok(response, add_special_tokens=False)["input_ids"]
    t_ids = tok(IM_END_TAIL, add_special_tokens=False)["input_ids"]
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    assert t_ids and t_ids[0] == im_end_id, ("turn-end tail tokenization drift", t_ids)
    assert p_ids[-len(suffix_ids) :] == suffix_ids, (
        "v_C position assert FAIL: rendered prompt ids do not end with the "
        "realized assistant-header suffix (template drift mid-run?)"
    )
    return (
        torch.tensor(p_ids, dtype=torch.long),
        torch.tensor(r_ids, dtype=torch.long),
        torch.tensor(t_ids, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Layer extraction (standalone port of analysis/extraction.py, trimmed)
# ---------------------------------------------------------------------------


def _logits_to_keep_kwargs(model) -> dict:
    """OOM guard: skip full-vocab logits when the caller never reads them —
    only when the forward names an EXPLICIT logits_to_keep parameter."""
    import inspect

    fn = getattr(model, "forward", None)
    if fn is None:
        return {}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    p = params.get("logits_to_keep")
    if p is None or p.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        return {}
    return {"logits_to_keep": 1}


def _unwrap(output):
    return output[0] if isinstance(output, tuple) else output


def _resolve_decoder_blocks(model):
    """Walk the wrapper chain (``.model`` / ``.language_model``, depth 1..4) to
    the module exposing ``.layers``. Returns (blocks, depth) or (None, 0)."""
    inner = model
    for depth in range(1, 5):
        nxt = getattr(inner, "model", None)
        if nxt is None:
            nxt = getattr(inner, "language_model", None)
        if nxt is None:
            return None, 0
        inner = nxt
        blocks = getattr(inner, "layers", None)
        if blocks is not None:
            return blocks, depth
    return None, 0


@torch.no_grad()
def _extract_layer_activations(
    model, input_ids: torch.Tensor, layers: list[int], attention_mask: torch.Tensor
) -> dict[int, torch.Tensor]:
    """{block index L: (B, T, H)} via forward hooks (block-L-output semantics ==
    output_hidden_states[L+1] for every non-last block — the parent
    convention; #2330's layers {16,22,30} of 32 / {14,19,26} of 28 never
    include the last block, so the pre-/post-final-norm divergence is moot).
    Falls back to the full-tuple read when no block chain resolves (CPU test
    stubs)."""
    blocks, _depth = _resolve_decoder_blocks(model)
    ltk = _logits_to_keep_kwargs(model)
    if blocks is None:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **ltk,
        )
        return {int(L): out.hidden_states[int(L) + 1].detach() for L in layers}

    n_blocks = len(blocks)
    for L in layers:
        assert 0 <= int(L) < n_blocks - 1, (
            f"layer {L} out of the hook-safe range [0, {n_blocks - 2}] — the LAST block's "
            "hook output diverges from hidden_states[-1] (pre- vs post-final-norm)"
        )
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(L: int):
        def _hook(_module, _inp, output):
            captured[L] = _unwrap(output).detach()

        return _hook

    for L in layers:
        handles.append(blocks[int(L)].register_forward_hook(_make_hook(int(L))))
    try:
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            **ltk,
        )
    finally:
        for h in handles:
            h.remove()
    missing = [L for L in layers if L not in captured]
    assert not missing, f"hooks captured nothing for layers {missing}"
    return captured


# ---------------------------------------------------------------------------
# Model / engine setup
# ---------------------------------------------------------------------------


def _resolve_h_dim(model_id: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    hidden = getattr(cfg, "hidden_size", None)
    if hidden is None:
        text_cfg = getattr(cfg, "text_config", None)
        hidden = getattr(text_cfg, "hidden_size", None)
    assert hidden is not None, (
        f"could not resolve hidden_size for {model_id} (checked cfg.hidden_size + "
        "cfg.text_config.hidden_size) — pass --h-dim explicitly"
    )
    return int(hidden)


_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def _load_capture_model(model_id: str, device: str, dtype_str: str):
    """HF capture model load. fp32 default per plan §4 P2 (9B fp32 ~36 GB fits
    one H200); the parity-vs-banked 7B leg passes bfloat16 (banked dtype)."""
    from transformers import AutoModelForCausalLM

    dtype = _DTYPES[dtype_str] if device == "cuda" else torch.float32
    device_map = {"": 0} if device == "cuda" else None
    try:
        hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map)
    except TypeError:
        # transformers 4.57.x repo-env fallback (dtype= landed as the canonical
        # name later; NOT a silent failure — any other exception propagates).
        logger.info("[load] dtype= kwarg rejected; retrying with torch_dtype= (transformers<5)")
        hf = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device_map
        )
    hf.eval()
    return hf


def _build_engine(model_id: str, seed: int):
    """vLLM engine (standalone port of create_vllm_engine defaults + the
    env-gated hang/IMA mitigation knobs — the launch script sets them)."""
    global _ENGINE_CONSTRUCTED, _LIVE_ENGINE
    from vllm import LLM

    llm_kwargs: dict = {}
    if os.environ.get("EPM_VLLM_ENFORCE_EAGER") == "1":
        llm_kwargs["enforce_eager"] = True
    if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING") == "1":
        llm_kwargs["enable_prefix_caching"] = False
    gpu_mem = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))
    logger.info("[engine-knobs] %s engine_seed=%d gpu_mem=%.2f", llm_kwargs, seed, gpu_mem)
    _ENGINE_CONSTRUCTED = True
    _LIVE_ENGINE = LLM(
        model=model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=64,
        seed=int(seed),
        **llm_kwargs,
    )
    return _LIVE_ENGINE


def _reap_vllm_engine(llm) -> None:
    """vLLM v1 teardown reap (gotchas.md recipe): engine_core.shutdown() ->
    destroy_process_group -> gc + empty_cache + ipc_collect + settle sleep.
    Clears _LIVE_ENGINE so the __main__ exception guard never double-reaps."""
    global _LIVE_ENGINE
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if shutdown is None:
        shutdown = getattr(getattr(engine, "model_executor", None), "shutdown", None)
    if callable(shutdown):
        shutdown()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(1.0)
    if _LIVE_ENGINE is llm:
        _LIVE_ENGINE = None


def _sampling_params(gen_seed: int):
    """#779 pass-B sampling recipe with the SPLIT's seed threaded (ceiling
    draws ride 43/44); realized-seed assert kept (plan binding convention)."""
    from vllm import SamplingParams

    sp = SamplingParams(
        n=1, temperature=GEN_TEMP, top_p=GEN_TOP_P, max_tokens=GEN_MAX_TOKENS, seed=int(gen_seed)
    )
    assert sp.seed == int(gen_seed), ("realized sampling seed drift", sp.seed, gen_seed)
    return sp


def _generate_seeded(llm, tok, prompts, gen_seed: int) -> tuple[list[str], list[str]]:
    """1 rollout per prompt, chunked (large single generate() calls can
    deadlock the v1 EngineCore — gotchas.md). Returns (responses,
    finish_reasons). CPU-smoke path (llm None) returns stub responses through
    the SAME downstream capture code."""
    if llm is None:
        return (
            ["This is a short stub response for the CPU capture smoke."] * len(prompts),
            ["stop"] * len(prompts),
        )
    sp = _sampling_params(gen_seed)
    logger.info(
        "[gen] realized sampling seed=%s temp=%s top_p=%s max_tokens=%s",
        sp.seed,
        sp.temperature,
        sp.top_p,
        sp.max_tokens,
    )
    prompt_texts = [_render_prompt(tok, p) for p in prompts]
    texts: list[str] = []
    finish: list[str] = []
    n_chunks = (len(prompt_texts) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
    for i in range(0, len(prompt_texts), VLLM_CHUNK_SIZE):
        chunk = prompt_texts[i : i + VLLM_CHUNK_SIZE]
        logger.info(
            "[vllm-chunk] chunk %d/%d (%d prompts, seed=%s)",
            i // VLLM_CHUNK_SIZE + 1,
            n_chunks,
            len(chunk),
            sp.seed,
        )
        chunk_out = llm.generate(chunk, sp, use_tqdm=False)
        for o in chunk_out:
            texts.append(o.outputs[0].text)
            finish.append(str(o.outputs[0].finish_reason))
    return texts, finish


# ---------------------------------------------------------------------------
# Capture: per-row (parity oracle + fallback) and batched (parent machinery)
# ---------------------------------------------------------------------------


def _reduce_row(captured: dict, row_i: int, p_len: int, f_len: int, layers, h_dim):
    """cx_last (v_C: hidden state at the LAST prompt token, index p_len-1) +
    v_x (v_A: mean over response + turn-end tail positions p_len..f_len-1) —
    parent parity."""
    cx_last_stack: list[torch.Tensor] = []
    v_x_stack: list[torch.Tensor] = []
    for li in layers:
        hs = captured[li][row_i]  # (T, H); right-pad positions >= f_len never read
        cx_last_stack.append(hs[p_len - 1, :].float().cpu())
        v_x_stack.append(hs[p_len:f_len, :].float().cpu().mean(dim=0))
    cx_last = torch.stack(cx_last_stack)
    v_x = torch.stack(v_x_stack)
    assert cx_last.shape == (len(layers), h_dim), ("cx_last", cx_last.shape)
    assert v_x.shape == (len(layers), h_dim), ("v_x", v_x.shape)
    return cx_last, v_x


def _capture_perrow(hf, tok, prompts, responses, cis, layers, h_dim, suffix_ids):
    """Per-row capture (parity oracle + safe fallback for the batched path)."""
    rows: list[dict] = []
    dropped: list[int] = []
    for p, resp, ci in zip(prompts, responses, cis, strict=True):
        if _is_empty_response(resp):
            dropped.append(int(ci))
            continue
        p_ids, r_ids, t_ids = _segment_token_ids(tok, p, resp, suffix_ids)
        assert r_ids.shape[0] >= 1, ("non-empty response tokenized to 0 tokens", ci)
        input_ids = torch.cat([p_ids, r_ids, t_ids]).unsqueeze(0).to(hf.device)
        attn = torch.ones_like(input_ids)
        captured = _extract_layer_activations(hf, input_ids, layers, attn)
        p_len = int(p_ids.shape[0])
        f_len = int(input_ids.shape[1])
        cx_last, v_x = _reduce_row(captured, 0, p_len, f_len, layers, h_dim)
        rows.append({"ci": int(ci), "prompt": p, "response": resp, "cx_last": cx_last, "v_x": v_x})
    return rows, dropped


def _capture_batched(hf, tok, prompts, responses, cis, layers, h_dim, batch_size, suffix_ids):
    """Batched teacher-forced capture: length-sorted RIGHT-padded batching over
    the same shared segment/reduce helpers (parent machinery)."""
    rows: list[dict] = []
    dropped: list[int] = []
    if not prompts:
        return rows, dropped

    seg: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for k, (p, resp) in enumerate(zip(prompts, responses, strict=True)):
        if _is_empty_response(resp):
            dropped.append(int(cis[k]))
            continue
        p_ids, r_ids, t_ids = _segment_token_ids(tok, p, resp, suffix_ids)
        assert r_ids.shape[0] >= 1, ("non-empty response tokenized to 0 tokens", cis[k])
        seg.append((k, p_ids, r_ids, t_ids))
    if not seg:
        return rows, dropped

    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id

    order = sorted(
        range(len(seg)),
        key=lambda i: int(seg[i][1].shape[0] + seg[i][2].shape[0] + seg[i][3].shape[0]),
    )
    for bs in range(0, len(order), batch_size):
        batch = [seg[i] for i in order[bs : bs + batch_size]]
        full_ids = [torch.cat([p_ids, r_ids, t_ids]) for _, p_ids, r_ids, t_ids in batch]
        p_lens = [int(p_ids.shape[0]) for _, p_ids, _r, _t in batch]
        f_lens = [int(x.shape[0]) for x in full_ids]

        max_len = max(f_lens)
        padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row_i, ids in enumerate(full_ids):
            padded[row_i, : ids.shape[0]] = ids
            attn[row_i, : ids.shape[0]] = 1

        padded = padded.to(hf.device)
        attn = attn.to(hf.device)
        captured = _extract_layer_activations(hf, padded, layers, attn)

        for row_i, (k, _p, _r, _t) in enumerate(batch):
            cx_last, v_x = _reduce_row(captured, row_i, p_lens[row_i], f_lens[row_i], layers, h_dim)
            rows.append(
                {
                    "ci": int(cis[k]),
                    "prompt": prompts[k],
                    "response": responses[k],
                    "cx_last": cx_last,
                    "v_x": v_x,
                }
            )
    return rows, dropped


def _assert_fp32_probe_feasible(hf) -> None:
    """Fail LOUD when the in-place fp32 cast for the parity probe cannot fit
    (parent port; only reached when production capture dtype is not fp32)."""
    dev = next(hf.parameters()).device
    if dev.type != "cuda":
        return
    cur_bytes = sum(p.numel() * p.element_size() for p in hf.parameters())
    cur_bytes += sum(b.numel() * b.element_size() for b in hf.buffers())
    fp32_bytes = sum(p.numel() * 4 for p in hf.parameters())
    fp32_bytes += sum(b.numel() * 4 for b in hf.buffers())
    margin_bytes = int(float(os.environ.get("EPM_I2330_FP32_PROBE_MARGIN_GIB", "8")) * (1 << 30))
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(dev)
    need = (fp32_bytes - cur_bytes) + margin_bytes
    if free < need:
        raise RuntimeError(
            "fp32 parity probe infeasible on this GPU: needs "
            f"~{(fp32_bytes - cur_bytes) / (1 << 30):.1f} GiB extra + "
            f"{margin_bytes / (1 << 30):.1f} GiB margin, free "
            f"{free / (1 << 30):.1f}/{total / (1 << 30):.1f} GiB. Refusing a bf16 probe "
            "or bar retune — use a larger GPU or lower EPM_I2330_FP32_PROBE_MARGIN_GIB "
            "only with a measured activation footprint."
        )


def _batched_capture_parity_gate(hf, tok, prompts, responses, cis, layers, h_dim, batch_size, sfx):
    """Run-start batched-vs-per-row parity gate (parent port): 32 probe rows,
    per-field cosine > 0.9999 and max rel-L2 < 1e-3 in fp32. On failure the
    caller falls back to per-row capture with a fail-loud WARN. Buffers are
    snapshot-restored around the cast (fp32 buffers like rotary inv_freq do
    NOT round-trip through bf16)."""
    n = min(32, len(prompts))
    if n == 0:
        return True, "empty probe (nothing to check)"
    p = prompts[:n]
    r = responses[:n]
    ci = cis[:n]
    orig_dtype = next(hf.parameters()).dtype
    cast = orig_dtype != torch.float32
    saved_buffers: dict[str, torch.Tensor] = {}
    if cast:
        _assert_fp32_probe_feasible(hf)
        saved_buffers = {
            name: buf.detach().clone()
            for name, buf in hf.named_buffers()
            if buf.is_floating_point()
        }
    try:
        if cast:
            hf.to(torch.float32)
        try:
            rows_serial, drop_serial = _capture_perrow(hf, tok, p, r, ci, layers, h_dim, sfx)
            rows_batched, drop_batched = _capture_batched(
                hf, tok, p, r, ci, layers, h_dim, batch_size, sfx
            )
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                "fp32 parity probe OOM despite the feasibility check — raise "
                "EPM_I2330_FP32_PROBE_MARGIN_GIB or use a larger GPU; do NOT revert "
                f"the probe to bf16. Original: {e}"
            ) from e
        except Exception as e:  # noqa: BLE001 — reported to caller, which fails loud to per-row
            return False, f"probe crashed: {type(e).__name__}: {e}"
    finally:
        if cast:
            hf.to(orig_dtype)
            for name, buf in saved_buffers.items():
                parent_path, _, leaf = name.rpartition(".")
                owner = hf.get_submodule(parent_path) if parent_path else hf
                setattr(owner, leaf, buf.to(device=getattr(owner, leaf).device))
            if next(hf.parameters()).is_cuda:
                torch.cuda.empty_cache()

    if set(drop_serial) != set(drop_batched):
        return False, (
            f"empty-drop mismatch: serial={sorted(drop_serial)} batched={sorted(drop_batched)}"
        )
    by_ci_batched = {row["ci"]: row for row in rows_batched}
    matched = 0
    max_cos_dev = 0.0
    max_rel_l2 = 0.0
    for rs in rows_serial:
        rb = by_ci_batched.get(rs["ci"])
        if rb is None:
            continue
        for field in ("cx_last", "v_x"):
            a = rs[field].float().flatten()
            b = rb[field].float().flatten()
            cos = float((a * b).sum()) / (float(a.norm()) * float(b.norm()) + 1e-30)
            rel = float((a - b).norm()) / (float(a.norm()) + 1e-30)
            max_cos_dev = max(max_cos_dev, 1.0 - cos)
            max_rel_l2 = max(max_rel_l2, rel)
        matched += 1
    dtype_note = f"probe_dtype=float32 (cast={cast}, restored={orig_dtype})"
    if matched == 0:
        return False, f"no matching rows between serial + batched probes [{dtype_note}]"
    if 1.0 - max_cos_dev < 0.9999:
        return False, f"cosine gate FAIL: min cos={1.0 - max_cos_dev:.6f} < 0.9999 [{dtype_note}]"
    if max_rel_l2 >= 1e-3:
        return False, f"rel-L2 gate FAIL: max rel-L2={max_rel_l2:.3e} >= 1e-3 [{dtype_note}]"
    return (
        True,
        f"PASS: {matched} rows, min cos={1.0 - max_cos_dev:.6f}, "
        f"max rel-L2={max_rel_l2:.3e} [{dtype_note}]",
    )


# ---------------------------------------------------------------------------
# Upload machinery (standalone port of _flush_upload_batch + exact-set verify)
# ---------------------------------------------------------------------------


def _stack_chunk(rows, layers, shard_index, chunk_idx) -> dict:
    """Stack per-row trimmed capture dicts into one bundle (parent layout —
    the P3 fits streamer consumes exactly this shape)."""
    return {
        "cx_last": torch.stack([r["cx_last"] for r in rows]),  # (n, L, H)
        "v_x": torch.stack([r["v_x"] for r in rows]),  # (n, L, H)
        "ci": [int(r["ci"]) for r in rows],
        "prompts": [r["prompt"] for r in rows],
        "layers": list(layers),
        "shard_index": int(shard_index),
        "chunk": int(chunk_idx),
    }


def _upload_names_once(scratch: Path, path_in_repo: str, names: list[str], verify_sha: bool):
    """ONE upload_folder commit for ``names`` (allow_patterns-scoped), then an
    exact-set presence verify on a fresh SCOPED listing (+ LFS sha256 verify
    for .pt). Raises on any miss — the caller purges only after this returns."""
    api = _hf_api()
    local_shas = {n: _sha256_file(scratch / n) for n in names} if verify_sha else {}
    _retry_transient(
        # HUB_DIR_FILECOUNT_EXEMPT: shard dirs hold <= ~40 files, far below the 10k cap
        lambda: api.upload_folder(  # NO_RETRY: wrapped in the module's own _retry_transient port
            folder_path=str(scratch),
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            allow_patterns=list(names),
            commit_message=f"task #2330: {len(names)} files -> {path_in_repo}",
        ),
        what=f"upload_folder {path_in_repo}",
    )
    remote = _remote_index(path_in_repo)
    missing = [n for n in names if n not in remote]
    if missing:
        raise RuntimeError(
            f"{len(missing)} files missing on Hub after batch upload to {path_in_repo} "
            f"(first: {missing[:5]}) — refusing to purge local copies"
        )
    if verify_sha:
        for n in names:
            sha = remote[n].get("sha256")
            if sha is None or sha != local_shas[n]:
                raise RuntimeError(
                    f"{n}: Hub LFS sha256 {sha} != local {local_shas[n]} — upload corrupt"
                )


def _flush_upload_batch(scratch: Path, prefix: str, pt_names: list[str], raw_names: list[str]):
    """Upload pending chunk batches (one commit per artifact kind), verify,
    purge. Raw text is NEVER discardable; purge only after verified presence.
    Does NOT clear the input lists (caller clears after return)."""
    if pt_names:
        _upload_names_once(scratch, f"{prefix}/final_token_capture", pt_names, verify_sha=True)
        for n in pt_names:
            (scratch / n).unlink()
        logger.info("[upload] batch of %d capture .pt verified (sha) + purged", len(pt_names))
    if raw_names:
        _upload_names_once(scratch, f"{prefix}/raw_completions", raw_names, verify_sha=False)
        for n in raw_names:
            (scratch / n).unlink()
        logger.info("[upload] batch of %d raw_completions verified + purged", len(raw_names))


# ---------------------------------------------------------------------------
# Gen-wave raw-chunk loaders (parent ports — resume/salvage/join contracts)
# ---------------------------------------------------------------------------


def _assert_raw_payload_matches(
    payload: dict,
    raw_name: str,
    *,
    expect_split: str,
    expect_seed: int,
    expect_shard_index: int,
    expect_chunk: int,
) -> None:
    """Wave-alignment asserts: the consumer must iterate the SAME manifest
    slice under the SAME shard arithmetic + split seed as the gen wave."""
    assert int(payload["shard_index"]) == expect_shard_index, (
        "gen/capture shard mismatch",
        raw_name,
        payload["shard_index"],
        expect_shard_index,
    )
    assert int(payload["chunk"]) == expect_chunk, (
        "gen/capture chunk mismatch",
        raw_name,
        payload["chunk"],
        expect_chunk,
    )
    assert payload["split"] == expect_split, (
        "gen/capture split mismatch",
        raw_name,
        payload["split"],
        expect_split,
    )
    assert int(payload["seed"]) == expect_seed, (
        "gen/capture seed mismatch",
        raw_name,
        payload["seed"],
        expect_seed,
    )


def _load_local_raw_salvage(
    scratch: Path,
    raw_name: str,
    *,
    expect_split: str,
    expect_seed: int,
    expect_shard_index: int,
    expect_chunk: int,
) -> dict | None:
    """Local gen raw-chunk payload for salvage, or None. A local chunk that
    never reached the Hub is re-UPLOADED verbatim on gen resume — NEVER
    regenerated (a fresh temperature-1.0 draw would publish text diverging
    from any .pt already captured from the local text)."""
    local = scratch / raw_name
    if not local.exists():
        return None
    with open(local, encoding="utf-8") as fh:
        payload = json.load(fh)
    _assert_raw_payload_matches(
        payload,
        raw_name,
        expect_split=expect_split,
        expect_seed=expect_seed,
        expect_shard_index=expect_shard_index,
        expect_chunk=expect_chunk,
    )
    return payload


def _load_persisted_gen_chunk(
    scratch: Path,
    stage_prefix: str,
    raw_name: str,
    cache_dir: Path,
    done_raw: set[str],
    *,
    expect_split: str,
    expect_seed: int,
    expect_shard_index: int,
    expect_chunk: int,
    allow_local_only: bool = False,
) -> dict[int, dict]:
    """Load ONE gen-wave raw chunk for phase_split_capture. HUB-REQUIRED
    unless --no-upload: a local-only chunk means the gen wave died before
    flushing — capturing from it would ship .pt whose published source text a
    later gen resume could replace. Returns {ci: row}."""
    local = scratch / raw_name
    on_hub = raw_name in done_raw
    if not on_hub and not allow_local_only:
        if local.exists():
            raise RuntimeError(
                f"phase_split_capture: {raw_name} exists locally ({local}) but is NOT on the "
                f"Hub under {stage_prefix}/raw_completions — the gen wave died before flushing. "
                "Re-run phase_split_gen (its resume re-uploads local raw chunks verbatim), "
                "then re-run this capture wave."
            )
        raise RuntimeError(
            f"phase_split_capture: gen-wave raw completions missing for {raw_name} — neither "
            f"local ({local}) nor on Hub under {stage_prefix}/raw_completions. Run the "
            "phase_split_gen wave to completion first."
        )
    if not local.exists():
        if not on_hub:
            raise RuntimeError(
                f"phase_split_capture (--no-upload): {raw_name} not in local scratch {scratch} "
                "and not on the Hub — run the --no-upload gen wave first."
            )
        local = _hub_download(f"{stage_prefix}/raw_completions/{raw_name}", cache_dir)
    with open(local, encoding="utf-8") as fh:
        payload = json.load(fh)
    _assert_raw_payload_matches(
        payload,
        raw_name,
        expect_split=expect_split,
        expect_seed=expect_seed,
        expect_shard_index=expect_shard_index,
        expect_chunk=expect_chunk,
    )
    rows = {int(r["ci"]): r for r in payload["rows"]}
    assert len(rows) == len(payload["rows"]), f"{raw_name}: duplicate ci in gen rows"
    return rows


# ---------------------------------------------------------------------------
# Shard arithmetic (parent parity: N50._shard_range semantics)
# ---------------------------------------------------------------------------


def _split_shard_range(n_total: int, num_shards: int, shard_index: int) -> tuple[int, int]:
    """Contiguous [start, end) — even split, remainder to the first shards."""
    assert 0 <= shard_index < num_shards, (shard_index, num_shards)
    base, rem = divmod(n_total, num_shards)
    start = shard_index * base + min(shard_index, rem)
    size = base + (1 if shard_index < rem else 0)
    return start, start + size


def _resolve_layers_arg(layers_arg: str) -> list[int]:
    """Comma-separated block indices, each element an int OR an inclusive
    ``A-B`` range (dense-sweep follow-up: ``--layers 0-30``). Duplicates are
    rejected (a duplicated hook layer would silently double rows in the
    stacked (n, L, H) bundle)."""
    ints: list[int] = []
    for part in (p.strip() for p in layers_arg.split(",")):
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if sep and lo.strip() and hi.strip():
            a, b = int(lo), int(hi)
            if a > b:
                raise ValueError(f"--layers range {part!r} is inverted (want A<=B)")
            ints.extend(range(a, b + 1))
        else:
            ints.append(int(part))
    if not ints:
        raise ValueError(f"--layers must be non-empty, got {layers_arg!r}")
    if len(set(ints)) != len(ints):
        raise ValueError(f"--layers contains duplicates: {layers_arg!r}")
    return ints


# ---------------------------------------------------------------------------
# P1 convention gates (plan §4 P1 steps 1/2/5/6 + the emit_spans support mode)
# ---------------------------------------------------------------------------

_TEMPLATE_PIN_PROBES = ["hi", "Explain how rain forms.", "What is 2 + 2?"]


def _maybe_write_p1_sentinel(args) -> None:
    """Plan-§9 P1 completion sentinel writer (round 2, p1-sentinel-no-writer).

    Called at the PASS end of every P1 gate step. Once EVERY required gate
    (P1_SENTINEL_REQUIRED) has a ``passed: true`` record in run_meta, writes
    the fingerprinted sentinel to --sentinel-path (default the plan's
    /workspace/logs/issue-2330-p1-smoke.json) atomically and logs the path;
    otherwise logs which gates are still pending. Idempotent — a later gate
    re-run re-writes the sentinel from the fresh run_meta. Fingerprints:
    target model id + its HF repo sha, code git SHA, split_ids file sha256
    (+ the payload's per-split id-list shas — post-drop when length_scan
    dropped rows), and the per-gate PASS records verbatim. P2/P3 do NOT
    hard-gate on the sentinel (reconciler scope: write + log is the
    contract); the orchestrator's poller consumes the path."""
    meta: dict = {}
    if args.run_meta_out.exists():
        meta = json.loads(args.run_meta_out.read_text(encoding="utf-8"))
    missing = [k for k in P1_SENTINEL_REQUIRED if not meta.get(k, {}).get("passed")]
    if missing:
        print(f"[p1-sentinel] pending — P1 gates without a PASS record yet: {missing}", flush=True)
        return
    split_ids_path = Path(args.split_ids)
    split_payload = _load_split_ids(split_ids_path)
    # The 9B production model (template_pin runs on the 9B leg by construction;
    # args.model at write time may be the 7B parity leg's).
    target_model = str(meta["template_pin"]["model"])
    model_sha = _retry_transient(
        lambda: getattr(_hf_api().model_info(target_model), "sha", None),
        what=f"model_info {target_model}",
    )
    sentinel = {
        "schema": "issue2330_p1_smoke_v1",
        "issue": 2330,
        "phase": "P1",
        "status": "PASS",
        "model": target_model,
        "model_hf_sha": model_sha,
        "code_git_sha": _git_sha(),
        "split_ids_path": str(split_ids_path),
        "split_ids_file_sha256": _sha256_file(split_ids_path),
        "split_ids_sha256_per_split": split_payload.get("sha256"),
        "gates": {k: meta[k] for k in P1_SENTINEL_REQUIRED},
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sentinel_path = Path(args.sentinel_path)
    _write_json_atomic(sentinel_path, sentinel)
    print(
        f"[p1-sentinel] WROTE {sentinel_path} — all {len(P1_SENTINEL_REQUIRED)} P1 gates PASS",
        flush=True,
    )


def gate_template_pin(args) -> int:
    """Gate step 1: render 3 probe prompts with enable_thinking=False; assert
    the empty-think-block suffix; record realized header token ids (repro
    card) into run meta."""
    _phase("gate_template_pin")
    tok, suffix_text, suffix_ids = _load_tokenizer(args.model, args.expect_suffix)
    for probe in _TEMPLATE_PIN_PROBES:
        rendered = _render_prompt(tok, probe)
        assert rendered.endswith(suffix_text), ("template pin probe FAIL", probe)
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        assert ids[-len(suffix_ids) :] == suffix_ids, ("template pin token FAIL", probe)
    record = {
        "model": args.model,
        "expect_suffix": args.expect_suffix,
        "suffix_text_repr": repr(suffix_text),
        "suffix_token_ids": [int(i) for i in suffix_ids],
        "n_probe_prompts": len(_TEMPLATE_PIN_PROBES),
        "tokenizer_name_or_path": str(getattr(tok, "name_or_path", args.model)),
        "vocab_size": int(len(tok)),
        "passed": True,  # asserts above precede the record write
    }
    _update_run_meta(args.run_meta_out, "template_pin", record)
    print(f"[gate] template_pin PASS: suffix ids={record['suffix_token_ids']}")
    _maybe_write_p1_sentinel(args)
    _phase("done")
    return 0


def gate_length_scan(args) -> int:
    """Gate step 2: tokenize ALL distinct consumed prompts under the target
    tokenizer; over-budget rows are dropped from split_ids.json (recorded in
    dropped_overlength, shas/counts recomputed); > max-over-budget-frac HALTS
    without mutating split_ids (exit 4)."""
    _phase("gate_length_scan")
    tok, _suffix_text, _suffix_ids = _load_tokenizer(args.model, args.expect_suffix)
    split_ids_path = Path(args.split_ids)
    payload = _load_split_ids(split_ids_path)
    cache_dir = args.out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_cache: dict[str, list[dict]] = {}
    drops: dict[str, list[dict]] = {}
    total_scanned = 0
    total_over = 0
    max_len_seen = 0
    for key in LENGTH_SCAN_KEYS:
        manifest_key = {
            "train_10k": "train_25k",
            "val_400": "val_400",
            "test_1000": "test_1000",
            "wc_test_1k": "wc_test_1k",
        }[key]
        if manifest_key not in manifest_cache:
            manifest_cache[manifest_key] = _download_manifest_split(manifest_key, cache_dir)
        rows = _subset_rows(manifest_cache[manifest_key], payload["splits"][key], key)
        for j, row in enumerate(rows):
            n = _rendered_prompt_token_len(tok, row["prompt"])
            max_len_seen = max(max_len_seen, n)
            total_scanned += 1
            if n > PROMPT_TOKEN_BUDGET:
                drops.setdefault(key, []).append(
                    {"id": int(row["ladder_local_id"]), "n_tokens": int(n)}
                )
                total_over += 1
            if (j + 1) % 2000 == 0:
                print(f"[length-scan] {key}: {j + 1}/{len(rows)} scanned", flush=True)
        print(
            f"[length-scan] {key}: {len(rows)} scanned, {len(drops.get(key, []))} over budget",
            flush=True,
        )

    expected_total = sum(len(payload["splits"][k]) for k in LENGTH_SCAN_KEYS)
    assert total_scanned == expected_total, (total_scanned, expected_total)
    frac = total_over / max(total_scanned, 1)
    record = {
        "model": args.model,
        "budget": PROMPT_TOKEN_BUDGET,
        "scanned": total_scanned,
        "over_budget": total_over,
        "over_budget_frac": frac,
        "max_rendered_len": max_len_seen,
        "drops_per_split": {k: len(v) for k, v in drops.items()},
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # The record persists on the HALT branch too (audit trail) — the P1
        # sentinel keys on this flag, not on key presence.
        "passed": total_over == 0 or frac <= args.max_over_budget_frac,
    }
    _update_run_meta(args.run_meta_out, "length_scan", record)

    if total_over == 0:
        print(
            f"[gate] length_scan PASS: 0/{total_scanned} over the {PROMPT_TOKEN_BUDGET}-token "
            f"budget (max rendered len {max_len_seen})"
        )
        _maybe_write_p1_sentinel(args)
        _phase("done")
        return 0
    if frac > args.max_over_budget_frac:
        print(
            f"[gate] length_scan HALT: {total_over}/{total_scanned} = {frac:.4%} over budget "
            f"exceeds the {args.max_over_budget_frac:.2%} re-scope band (plan §7) — "
            "split_ids.json NOT mutated; re-scope required.",
            file=sys.stderr,
        )
        _phase("done")
        return 4

    # Drop-from-both path: remove the ids from the split lists (both models
    # subset by these lists, so removal drops the rows from BOTH cells),
    # record them, recompute shas + counts, re-write atomically.
    import hashlib as _hashlib

    def _sha_ids(ids: list[int]) -> str:
        return _hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()

    for key, entries in drops.items():
        drop_ids = {e["id"] for e in entries}
        payload["splits"][key] = [i for i in payload["splits"][key] if i not in drop_ids]
        if key == "train_10k" and "train_5k" in payload["splits"]:
            payload["splits"]["train_5k"] = [
                i for i in payload["splits"]["train_5k"] if i not in drop_ids
            ]
        payload.setdefault("dropped_overlength", {}).setdefault(key, []).extend(entries)
    payload["sha256"] = {k: _sha_ids(v) for k, v in payload["splits"].items()}
    payload["counts"] = {k: len(v) for k, v in payload["splits"].items()}
    payload["length_scan"] = record
    _write_json_atomic(split_ids_path, payload)
    print(
        f"[gate] length_scan: DROPPED {total_over} over-budget rows "
        f"({frac:.4%} <= {args.max_over_budget_frac:.2%}); split_ids.json re-written — "
        f"post-drop counts {payload['counts']} (commit + push the update before P2/P3)"
    )
    _maybe_write_p1_sentinel(args)
    _phase("done")
    return 0


def _select_parity_rows(args, cache_dir: Path) -> tuple[str, list[dict], Path | None]:
    """Deterministic banked-row selection shared by emit_spans + parity7b:
    the lexicographically-first raw_completions chunk of the pinned banked
    prefix; first --parity-rows rows with a non-empty response, payload order.

    Returns (chunk_stem, selected_rows, pt_local_path_or_None)."""
    raw_prefix = f"{args.parity_banked_prefix}/raw_completions"
    pt_prefix = f"{args.parity_banked_prefix}/final_token_capture"
    raw_index = _remote_index(raw_prefix, revision=args.parity_banked_revision)
    assert raw_index, f"no raw_completions under {raw_prefix} at the pin"
    raw_name = sorted(n for n in raw_index if n.endswith(".json"))[0]
    stem = raw_name[: -len(".json")]
    raw_local = _hub_download(f"{raw_prefix}/{raw_name}", cache_dir, args.parity_banked_revision)
    with open(raw_local, encoding="utf-8") as fh:
        payload = json.load(fh)
    selected = [r for r in payload["rows"] if not _is_empty_response(r["response"])]
    selected = selected[: args.parity_rows]
    assert len(selected) == args.parity_rows, (
        f"banked chunk {raw_name} holds only {len(selected)} non-empty rows "
        f"< --parity-rows {args.parity_rows}"
    )
    pt_local: Path | None = None
    if args.gate == "parity7b":
        pt_index = _remote_index(pt_prefix, revision=args.parity_banked_revision)
        pt_name = f"{stem}.pt"
        assert pt_name in pt_index, f"{pt_name} absent under {pt_prefix} at the pin"
        pt_local = _hub_download(f"{pt_prefix}/{pt_name}", cache_dir, args.parity_banked_revision)
    return stem, selected, pt_local


def gate_emit_spans(args) -> int:
    """Support mode for gate step 5(a): tokenizer-only segmentation of the
    pinned banked rows, spans written to --spans-out. Run in the REPO env
    (transformers 4.57.6 — the parent's stack) to freeze the reference the
    fresh-venv parity7b gate compares against."""
    _phase("gate_emit_spans")
    assert args.spans_out, "--spans-out required for --gate emit_spans"
    assert args.expect_suffix == "plain", "emit_spans runs on the Qwen2.5 leg (plain suffix)"
    tok, _suffix_text, suffix_ids = _load_tokenizer(args.model, args.expect_suffix)
    cache_dir = args.out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem, selected, _pt = _select_parity_rows(args, cache_dir)

    import transformers

    rows_out = []
    for r in selected:
        p_ids, r_ids, t_ids = _segment_token_ids(tok, r["prompt"], r["response"], suffix_ids)
        rows_out.append(
            {
                "ci": int(r["ci"]),
                "p_ids": [int(i) for i in p_ids.tolist()],
                "r_ids": [int(i) for i in r_ids.tolist()],
                "t_ids": [int(i) for i in t_ids.tolist()],
            }
        )
    out = {
        "model": args.model,
        "banked_prefix": args.parity_banked_prefix,
        "banked_revision": args.parity_banked_revision,
        "chunk_stem": stem,
        "transformers_version": transformers.__version__,
        "suffix_token_ids": [int(i) for i in suffix_ids],
        "rows": rows_out,
        "git_commit": _git_sha(),
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json_atomic(Path(args.spans_out), out)
    print(f"[gate] emit_spans: wrote {len(rows_out)} row spans -> {args.spans_out}")
    _phase("done")
    return 0


def gate_parity7b(args) -> int:
    """Gate step 5: capture-port parity vs the parent driver on banked 7B rows.

    (a) exact token-id spans vs --expected-spans (emit_spans output from the
        repo env — the parent's transformers stack);
    (b) per-row cx_last/v_x cosine >= --parity-cos-min vs the banked captures
        at the banked layers. A miss HALTS (failure_class: code)."""
    _phase("gate_parity7b")
    assert args.expected_spans, "--expected-spans required for --gate parity7b (run emit_spans)"
    assert args.expect_suffix == "plain", "parity7b runs on the Qwen2.5 leg (plain suffix)"
    layers = _resolve_layers_arg(args.layers)
    tok, _suffix_text, suffix_ids = _load_tokenizer(args.model, args.expect_suffix)
    h_dim = _resolve_h_dim(args.model, args.h_dim)
    cache_dir = args.out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem, selected, pt_local = _select_parity_rows(args, cache_dir)
    assert pt_local is not None

    expected = json.loads(Path(args.expected_spans).read_text(encoding="utf-8"))
    assert expected["chunk_stem"] == stem, (
        "expected-spans chunk mismatch",
        expected["chunk_stem"],
        stem,
    )
    exp_by_ci = {int(r["ci"]): r for r in expected["rows"]}

    # (a) exact token-id span agreement, this venv vs the reference venv.
    for r in selected:
        ci = int(r["ci"])
        assert ci in exp_by_ci, f"ci {ci} absent from --expected-spans (row-selection drift)"
        p_ids, r_ids, t_ids = _segment_token_ids(tok, r["prompt"], r["response"], suffix_ids)
        exp = exp_by_ci[ci]
        for name, got, want in (
            ("p_ids", p_ids.tolist(), exp["p_ids"]),
            ("r_ids", r_ids.tolist(), exp["r_ids"]),
            ("t_ids", t_ids.tolist(), exp["t_ids"]),
        ):
            assert [int(i) for i in got] == [int(i) for i in want], (
                f"parity7b span FAIL: ci={ci} {name} diverges from the reference "
                f"segmentation (len {len(got)} vs {len(want)}) — cross-venv tokenizer drift"
            )
    print(f"[gate] parity7b: token-id spans EXACT-match for {len(selected)} rows")

    # (b) capture parity vs the banked vectors.
    # weights_only=False: sha-pinned self-produced bundle at a pinned revision
    # (torch>=2.6 default flip; the bundle carries plain dict/list/tensor).
    bundle = torch.load(pt_local, map_location="cpu", weights_only=False)
    assert list(bundle["layers"]) == layers, (
        "banked layer list mismatch — pass --layers matching the banked capture",
        bundle["layers"],
        layers,
    )
    ci_to_idx = {int(c): i for i, c in enumerate(bundle["ci"])}
    hf = _load_capture_model(args.model, args.device, args.capture_dtype)
    prompts = [r["prompt"] for r in selected]
    responses = [r["response"] for r in selected]
    cis = [int(r["ci"]) for r in selected]
    rows_c, dropped = _capture_perrow(hf, tok, prompts, responses, cis, layers, h_dim, suffix_ids)
    assert not dropped, ("parity7b: unexpected empty-response drops", dropped)

    worst = {"cos": 1.0, "ci": None, "field": None, "layer": None}
    n_checked = 0
    for row in rows_c:
        ci = int(row["ci"])
        assert ci in ci_to_idx, f"ci {ci} absent from the banked bundle"
        b_idx = ci_to_idx[ci]
        for field, banked_key in (("cx_last", "cx_last"), ("v_x", "v_x")):
            for li, layer in enumerate(layers):
                a = row[field][li].float()
                b = bundle[banked_key][b_idx][li].float()
                assert a.shape == b.shape, (field, layer, a.shape, b.shape)
                cos = float((a * b).sum()) / (float(a.norm()) * float(b.norm()) + 1e-30)
                n_checked += 1
                if cos < worst["cos"]:
                    worst = {"cos": cos, "ci": ci, "field": field, "layer": layer}
    record = {
        "model": args.model,
        "capture_dtype": args.capture_dtype,
        "banked_prefix": args.parity_banked_prefix,
        "banked_revision": args.parity_banked_revision,
        "chunk_stem": stem,
        "n_rows": len(rows_c),
        "n_vector_checks": n_checked,
        "cos_min_required": args.parity_cos_min,
        "worst_cos": worst["cos"],
        "worst_at": {k: worst[k] for k in ("ci", "field", "layer")},
        "max_cos_deviation": 1.0 - worst["cos"],
        # Persisted on the HALT branch too — the P1 sentinel keys on this flag.
        "passed": worst["cos"] >= args.parity_cos_min,
    }
    _update_run_meta(args.run_meta_out, "parity7b", record)
    if worst["cos"] < args.parity_cos_min:
        print(
            f"[gate] parity7b FAIL: worst cosine {worst['cos']:.6f} < {args.parity_cos_min} at "
            f"ci={worst['ci']} field={worst['field']} layer={worst['layer']} — the capture "
            "port is broken (failure_class: code); HALT.",
            file=sys.stderr,
        )
        _phase("done")
        return 5
    print(
        f"[gate] parity7b PASS: {len(rows_c)} rows x 2 fields x {len(layers)} layers, "
        f"worst cosine {worst['cos']:.6f} (>= {args.parity_cos_min})"
    )
    _maybe_write_p1_sentinel(args)
    _phase("done")
    return 0


def gate_hook_probe(args) -> int:
    """Gate step 6: hook-vs-tuple index probe + v_C position assert on the 9B.

    On --hook-probe-rows fixed rows (first kept rows of the split subset,
    stub responses — the probe tests INDEX conventions, not content): register
    a forward hook on each named block {16,22,30} and assert
    hidden_states[k+1] equals the hooked block-k output (exact shape;
    rel <= --hook-rel-tol); assert tuple length == n_blocks + 1; assert each
    stored v_C index is the FINAL rendered-prompt token whose ids END with the
    realized empty-think-block suffix. Results persisted in run meta (closes
    plan §12 A4)."""
    _phase("gate_hook_probe")
    layers = _resolve_layers_arg(args.layers)
    split = args.split or "train_10k"
    manifest_key, ids_key, _seed = SPLIT_TO_MANIFEST[split]
    tok, _suffix_text, suffix_ids = _load_tokenizer(args.model, args.expect_suffix)
    h_dim = _resolve_h_dim(args.model, args.h_dim)
    cache_dir = args.out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_payload = _load_split_ids(Path(args.split_ids))
    manifest_rows = _download_manifest_split(manifest_key, cache_dir)
    rows = _subset_rows(manifest_rows, split_payload["splits"][ids_key], ids_key)
    kept_prompts, kept_cis, _skipped = _filter_overlength_prompts(
        [r["prompt"] for r in rows[: 4 * args.hook_probe_rows]],
        [int(r["ladder_local_id"]) for r in rows[: 4 * args.hook_probe_rows]],
        lambda p: _rendered_prompt_token_len(tok, p),
        PROMPT_TOKEN_BUDGET,
    )
    probe_prompts = kept_prompts[: args.hook_probe_rows]
    probe_cis = kept_cis[: args.hook_probe_rows]
    assert len(probe_prompts) == args.hook_probe_rows, "not enough kept rows for the probe"
    stub_response = "This is a short stub response for the hook-vs-tuple index probe."

    hf = _load_capture_model(args.model, args.device, args.capture_dtype)
    blocks, depth = _resolve_decoder_blocks(hf)
    assert blocks is not None, (
        "decoder blocks did not resolve on this architecture — the hook path is unavailable; "
        "the capture convention cannot be verified (fail loud, do not fall back silently)"
    )
    n_blocks = len(blocks)
    ltk = _logits_to_keep_kwargs(hf)

    per_layer_max_rel = {int(L): 0.0 for L in layers}
    tuple_len = None
    for p, ci in zip(probe_prompts, probe_cis, strict=True):
        p_ids, r_ids, t_ids = _segment_token_ids(tok, p, stub_response, suffix_ids)
        input_ids = torch.cat([p_ids, r_ids, t_ids]).unsqueeze(0).to(hf.device)
        attn = torch.ones_like(input_ids)
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def _mk(L: int, store: dict):
            def _hook(_m, _i, output):
                store[L] = _unwrap(output).detach()

            return _hook

        for L in layers:
            handles.append(blocks[int(L)].register_forward_hook(_mk(int(L), captured)))
        try:
            with torch.no_grad():
                out = hf(
                    input_ids=input_ids,
                    attention_mask=attn,
                    output_hidden_states=True,
                    **ltk,
                )
        finally:
            for h in handles:
                h.remove()
        hs = out.hidden_states
        tuple_len = len(hs)
        assert tuple_len == n_blocks + 1, (
            f"hidden_states tuple length {tuple_len} != n_blocks+1 = {n_blocks + 1} "
            "(plan §12 A4 tuple-shape premise violated)"
        )
        for L in layers:
            a = captured[int(L)]
            b = hs[int(L) + 1]
            assert a.shape == b.shape, ("hook-vs-tuple shape mismatch", L, a.shape, b.shape)
            assert a.shape[-1] == h_dim, ("h_dim mismatch", a.shape, h_dim)
            rel = float((a.float() - b.float()).norm()) / (float(a.float().norm()) + 1e-30)
            per_layer_max_rel[int(L)] = max(per_layer_max_rel[int(L)], rel)
            assert rel <= args.hook_rel_tol, (
                f"hook-vs-tuple index probe FAIL: block {L} vs hidden_states[{L + 1}] "
                f"rel={rel:.3e} > {args.hook_rel_tol} (ci={ci}) — the hidden_states index "
                "convention does NOT hold for this architecture"
            )
        # v_C position assert: cx_last reads index p_len-1, the FINAL rendered
        # prompt token; the prompt ids END with the realized header suffix
        # (asserted inside _segment_token_ids; re-asserted here explicitly).
        assert p_ids.tolist()[-len(suffix_ids) :] == suffix_ids

    record = {
        "model": args.model,
        "capture_dtype": args.capture_dtype,
        "n_probe_rows": len(probe_prompts),
        "probe_cis": probe_cis,
        "split": split,
        "n_blocks": n_blocks,
        "wrapper_depth": depth,
        "hidden_states_tuple_len": tuple_len,
        "layer_index_mapping": {int(L): int(L) + 1 for L in layers},
        "per_layer_max_rel": per_layer_max_rel,
        "rel_tol": args.hook_rel_tol,
        "h_dim": h_dim,
        "v_c_convention": "cx_last = hidden state at index p_len-1 (final rendered-prompt token)",
        "suffix_token_ids": [int(i) for i in suffix_ids],
        "passed": True,  # asserts above precede the record write
    }
    _update_run_meta(args.run_meta_out, "hook_probe", record)
    print(
        f"[gate] hook_probe PASS: {len(probe_prompts)} rows, blocks {layers} == "
        f"hidden_states[k+1] (max rel {max(per_layer_max_rel.values()):.2e}), "
        f"tuple len {tuple_len} = {n_blocks}+1, h_dim {h_dim}"
    )
    _maybe_write_p1_sentinel(args)
    _phase("done")
    return 0


# ---------------------------------------------------------------------------
# Run capture (P2 production path — parent run_capture port)
# ---------------------------------------------------------------------------


def run_capture(args) -> int:
    """Generation + trimmed capture for ONE (split, shard). Emits per-chunk
    .pt + raw JSON into out_dir/shards/, uploads in K-batches under
    ``<hf_prefix>/<split>/...`` (ceiling draws: ``.../ceiling_draws/seed{S}``)."""
    layers = _resolve_layers_arg(args.layers)
    h_dim = _resolve_h_dim(args.model, args.h_dim)
    manifest_key, ids_key, gen_seed = SPLIT_TO_MANIFEST[args.split]
    logger.info(
        "[i2330] model=%s split=%s (manifest=%s ids=%s seed=%d) layers=%s H=%d shard=%d/%d "
        "prefix=%s mode=%s",
        args.model,
        args.split,
        manifest_key,
        ids_key,
        gen_seed,
        layers,
        h_dim,
        args.shard_index,
        args.num_shards,
        args.hf_prefix,
        args.capture_mode,
    )

    # 1. Manifest subset per split_ids (the P0/P1 single source).
    cache_dir = args.out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_payload = _load_split_ids(Path(args.split_ids))
    manifest_rows = _download_manifest_split(manifest_key, cache_dir)
    all_rows = _subset_rows(manifest_rows, split_payload["splits"][ids_key], ids_key)
    n_total = len(all_rows)
    start, end = _split_shard_range(n_total, args.num_shards, args.shard_index)
    shard_rows = all_rows[start:end]
    if not shard_rows:
        logger.info("[shard %d] empty range; nothing to do", args.shard_index)
        _phase("done")
        return 0

    # Round 3 (FIX 1b): the WRITE side of the shared logical-split → store-
    # subpath table — the cap-hit aggregator READS through the same function.
    stage_prefix = f"{args.hf_prefix}/{store_subpath_for_split(args.split)}"

    # Dense re-capture follow-up: phase_split_capture may read the gen wave's
    # raw chunks from a DIFFERENT (banked) prefix than this run's write prefix
    # (asserted phase_split_capture-only in main; identical otherwise).
    gen_stage_prefix = stage_prefix
    if args.gen_source_prefix:
        gen_stage_prefix = f"{args.gen_source_prefix}/{store_subpath_for_split(args.split)}"
        logger.info(
            "[i2330] gen-source prefix: %s (capture .pt writes stay under %s)",
            gen_stage_prefix,
            stage_prefix,
        )

    scratch = args.out_dir / "shards" / args.split.replace("ceiling_draw_", "cdraw_")
    scratch.mkdir(parents=True, exist_ok=True)

    # 2. Resume — chunks already on the Hub are skipped (mode-scoped below).
    done_pt = set(_remote_index(f"{stage_prefix}/final_token_capture"))
    done_raw = set(_remote_index(f"{gen_stage_prefix}/raw_completions"))

    # 3. Load models (mode governs which we hold at once — phase_split keeps
    # the 9B HF fp32 model and the vLLM engine on SEPARATE invocations).
    _phase("load_model")
    tok, _suffix_text, suffix_ids = _load_tokenizer(args.model, args.expect_suffix)
    hf = None
    if args.capture_mode in ("coresident", "phase_split_capture"):
        hf = _load_capture_model(args.model, args.device, args.capture_dtype)

    llm = None
    if args.capture_mode in ("coresident", "phase_split_gen"):
        llm = _build_engine(args.model, gen_seed) if args.device == "cuda" else None

    # 4. Capture method selection: batched (default) with parity fallback.
    capture_fn_choice = "perrow"
    if args.capture_batch_size > 1 and args.capture_mode in ("coresident", "phase_split_capture"):
        n_cand = min(64, len(shard_rows))
        if args.capture_mode == "phase_split_capture":
            n_cand = min(n_cand, args.shard_size)
        cand = shard_rows[:n_cand]
        cand_prompts = [r["prompt"] for r in cand]
        cand_cis = [int(r["ladder_local_id"]) for r in cand]
        kept_probe_prompts, kept_probe_cis, probe_skipped = _filter_overlength_prompts(
            cand_prompts,
            cand_cis,
            lambda p: _rendered_prompt_token_len(tok, p),
            PROMPT_TOKEN_BUDGET,
        )
        if probe_skipped:
            logger.info("[i2330] parity probe: %d over-length rows excluded", len(probe_skipped))
        probe_prompts = kept_probe_prompts[:32]
        probe_cis = kept_probe_cis[:32]
        if not probe_cis:
            probe_responses: list[str] = []
        elif args.capture_mode == "phase_split_capture":
            probe_raw_name = f"shard{args.shard_index:02d}_chunk0000.json"
            probe_map = _load_persisted_gen_chunk(
                scratch,
                gen_stage_prefix,
                probe_raw_name,
                cache_dir,
                done_raw,
                expect_split=args.split,
                expect_seed=gen_seed,
                expect_shard_index=args.shard_index,
                expect_chunk=0,
                allow_local_only=args.no_upload,
            )
            probe_missing = [c for c in probe_cis if c not in probe_map]
            assert not probe_missing, (
                "parity probe: gen-wave rows missing (shard config drift?)",
                probe_raw_name,
                probe_missing[:8],
            )
            probe_responses = [probe_map[c]["response"] for c in probe_cis]
        else:
            probe_responses, _probe_finish = _generate_seeded(llm, tok, probe_prompts, gen_seed)
        gate_pass, gate_reason = _batched_capture_parity_gate(
            hf,
            tok,
            probe_prompts,
            probe_responses,
            probe_cis,
            layers,
            h_dim,
            args.capture_batch_size,
            suffix_ids,
        )
        logger.info(
            "[i2330] batched-capture parity gate: %s (%s)",
            "PASS" if gate_pass else "FAIL",
            gate_reason,
        )
        if gate_pass:
            capture_fn_choice = "batched"
        else:
            logger.warning(
                "[i2330] batched-capture parity gate FAILED — per-row fallback. Reason: %s",
                gate_reason,
            )

    def _do_capture(prompts_i, responses_i, cis_i, _hf=hf, _tok=tok):
        if capture_fn_choice == "batched":
            return _capture_batched(
                _hf,
                _tok,
                prompts_i,
                responses_i,
                cis_i,
                layers,
                h_dim,
                args.capture_batch_size,
                suffix_ids,
            )
        return _capture_perrow(_hf, _tok, prompts_i, responses_i, cis_i, layers, h_dim, suffix_ids)

    # 5. Main loop across chunks.
    _phase("capture")
    n_sub = (len(shard_rows) + args.shard_size - 1) // args.shard_size
    kept_total = 0
    pending_pt: list[str] = []
    pending_raw: list[str] = []

    def _flush_pending() -> None:
        # Key on EITHER pending kind: phase_split_gen fills only pending_raw
        # and phase_split_capture only pending_pt.
        if args.no_upload or (not pending_pt and not pending_raw):
            return
        _flush_upload_batch(scratch, stage_prefix, pending_pt, pending_raw)
        pending_pt.clear()
        pending_raw.clear()

    def _on_sigterm(signum, frame):
        raise SystemExit(f"SIGTERM ({signum}) — flushing pending upload batch")

    prev_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
    skipped_all: list[dict] = []
    dropped_empty_all: list[int] = []
    cap_hit_total = 0
    think_total = 0
    gen_total = 0

    try:
        for ci_idx, s in enumerate(range(0, len(shard_rows), args.shard_size)):
            name = f"shard{args.shard_index:02d}_chunk{ci_idx:04d}.pt"
            raw_name = f"shard{args.shard_index:02d}_chunk{ci_idx:04d}.json"
            chunk = shard_rows[s : s + args.shard_size]
            kept_prompts, kept_cis, skipped = _filter_overlength_prompts(
                [r["prompt"] for r in chunk],
                [int(r["ladder_local_id"]) for r in chunk],
                lambda p: _rendered_prompt_token_len(tok, p),
                PROMPT_TOKEN_BUDGET,
            )
            skipped_all.extend(skipped)
            if skipped:
                # Post-length-scan this should be EMPTY; a non-empty set means
                # the P1 scan did not run (or tokenizer drift) — the P3 count
                # pins will fail loud on the shortfall either way.
                logger.warning(
                    "[shard %d] chunk %d: %d over-length rows skipped DESPITE the P1 "
                    "length scan — investigate before P3 (cis %s)",
                    args.shard_index,
                    ci_idx,
                    len(skipped),
                    [x["ci"] for x in skipped][:10],
                )
            # Resume predicate is MODE-scoped (parent parity).
            if args.capture_mode == "phase_split_gen":
                chunk_done = raw_name in done_raw
            elif args.capture_mode == "phase_split_capture":
                chunk_done = name in done_pt
            else:
                chunk_done = name in done_pt and raw_name in done_raw
            if chunk_done:
                logger.info(
                    "[shard %d] chunk %d/%d already on Hub; skip",
                    args.shard_index,
                    ci_idx + 1,
                    n_sub,
                )
                continue
            if not kept_prompts:
                logger.warning(
                    "[shard %d] chunk %d: all rows over-length; skip", args.shard_index, ci_idx
                )
                continue

            ts = time.time()
            if args.capture_mode == "phase_split_capture":
                raw_map = _load_persisted_gen_chunk(
                    scratch,
                    gen_stage_prefix,
                    raw_name,
                    cache_dir,
                    done_raw,
                    expect_split=args.split,
                    expect_seed=gen_seed,
                    expect_shard_index=args.shard_index,
                    expect_chunk=ci_idx,
                    allow_local_only=args.no_upload,
                )
                missing = [c for c in kept_cis if c not in raw_map]
                if missing:
                    raise RuntimeError(
                        f"phase_split_capture: {len(missing)} kept cis absent from gen-wave "
                        f"{raw_name} (first: {missing[:10]}) — the gen wave ran under a "
                        "different shard config / split_ids; refusing a partial join."
                    )
                extra_cis = sorted(set(raw_map) - set(kept_cis))
                if extra_cis:
                    logger.warning(
                        "[shard %d] chunk %d: %d gen-wave rows absent from this run's kept "
                        "set (first: %s) — admission drift; those rows stay raw-only",
                        args.shard_index,
                        ci_idx,
                        len(extra_cis),
                        extra_cis[:10],
                    )
                for c, p in zip(kept_cis, kept_prompts, strict=True):
                    assert raw_map[c]["prompt"] == p, (
                        "prompt drift between manifest row and gen-wave row",
                        c,
                    )
                responses = [raw_map[c]["response"] for c in kept_cis]
                n_cap_hit = 0  # cap-hit accounting belongs to the gen wave
                n_think = 0
            else:
                # SALVAGE-FIRST: a local raw chunk from a prior run that died
                # before flushing is re-uploaded verbatim, never regenerated.
                salvaged = None
                if raw_name not in done_raw:
                    salvaged = _load_local_raw_salvage(
                        scratch,
                        raw_name,
                        expect_split=args.split,
                        expect_seed=gen_seed,
                        expect_shard_index=args.shard_index,
                        expect_chunk=ci_idx,
                    )
                if salvaged is not None:
                    sal_rows = {int(r["ci"]): r for r in salvaged["rows"]}
                    if set(sal_rows) != set(kept_cis):
                        raise RuntimeError(
                            f"gen salvage: local {raw_name} row set diverges from this run's "
                            f"kept set (local-only: "
                            f"{sorted(set(sal_rows) - set(kept_cis))[:10]}, kept-only: "
                            f"{sorted(set(kept_cis) - set(sal_rows))[:10]}) — refusing to "
                            "reuse OR regenerate."
                        )
                    for c, p in zip(kept_cis, kept_prompts, strict=True):
                        assert sal_rows[c]["prompt"] == p, (
                            "prompt drift between manifest row and salvaged gen row",
                            c,
                        )
                    responses = [sal_rows[c]["response"] for c in kept_cis]
                    finish_reasons = [str(sal_rows[c]["finish_reason"]) for c in kept_cis]
                    n_cap_hit = int(
                        salvaged.get("n_cap_hit", sum(1 for f in finish_reasons if f == "length"))
                    )
                    n_think = sum(1 for r in responses if _opens_with_think(r))
                    cap_hit_total += n_cap_hit
                    think_total += n_think
                    gen_total += len(responses)
                    logger.warning(
                        "[shard %d] chunk %d: SALVAGED %d rows from local %s — text reused "
                        "verbatim, NOT regenerated",
                        args.shard_index,
                        ci_idx,
                        len(responses),
                        raw_name,
                    )
                else:
                    responses, finish_reasons = _generate_seeded(llm, tok, kept_prompts, gen_seed)
                    n_cap_hit = sum(1 for f in finish_reasons if f == "length")
                    n_think = sum(1 for r in responses if _opens_with_think(r))
                    cap_hit_total += n_cap_hit
                    think_total += n_think
                    gen_total += len(responses)
                    # Persist raw completions FIRST (persist-by-default).
                    _write_json_atomic(
                        scratch / raw_name,
                        {
                            "shard_index": args.shard_index,
                            "chunk": ci_idx,
                            "split": args.split,
                            "seed": gen_seed,
                            "sampling_seed": gen_seed,
                            "engine_seed": gen_seed,
                            "gen_max_tokens": GEN_MAX_TOKENS,
                            "n_cap_hit": n_cap_hit,
                            "n_think_open": n_think,
                            "model": args.model,
                            "rows": [
                                {"ci": int(c), "prompt": p, "response": r, "finish_reason": f}
                                for c, p, r, f in zip(
                                    kept_cis, kept_prompts, responses, finish_reasons, strict=True
                                )
                            ],
                        },
                    )

            # Trimmed capture (skipped in phase_split_gen — gen only).
            if args.capture_mode == "phase_split_gen":
                n_kept = len(kept_prompts)
                pending_raw.append(raw_name)
                n_dropped_empty = 0
            else:
                rows, dropped_cis = _do_capture(kept_prompts, responses, kept_cis)
                dropped_empty_all.extend(dropped_cis)
                n_dropped_empty = len(dropped_cis)
                if dropped_cis:
                    logger.info(
                        "[shard %d] chunk %d: dropped %d empty-response rows (cis %s%s)",
                        args.shard_index,
                        ci_idx,
                        len(dropped_cis),
                        dropped_cis[:20],
                        "..." if len(dropped_cis) > 20 else "",
                    )
                if rows:
                    bundle = _stack_chunk(rows, layers, args.shard_index, ci_idx)
                    bundle["dropped_empty_cis"] = [int(c) for c in dropped_cis]
                    # Provenance: which prefix the source completions came from
                    # (== --hf-prefix except on a --gen-source-prefix re-capture).
                    bundle["gen_source_prefix"] = args.gen_source_prefix or args.hf_prefix
                    torch.save(bundle, scratch / name)
                    pending_pt.append(name)
                else:
                    logger.warning(
                        "[shard %d] chunk %d: 0 captured rows (all empty responses)",
                        args.shard_index,
                        ci_idx,
                    )
                n_kept = len(rows)
                if args.capture_mode != "phase_split_capture":
                    pending_raw.append(raw_name)

            kept_total += n_kept
            if not args.no_upload and max(len(pending_pt), len(pending_raw)) >= UPLOAD_BATCH:
                _flush_pending()

            logger.info(
                "[shard %d] chunk %d/%d: %d/%d kept (%d over-length, %d empty-dropped, "
                "%d cap-hit, %d think-open, %.0fs) [%s]",
                args.shard_index,
                ci_idx + 1,
                n_sub,
                n_kept,
                len(chunk),
                len(skipped),
                n_dropped_empty,
                n_cap_hit,
                n_think if args.capture_mode != "phase_split_capture" else 0,
                time.time() - ts,
                capture_fn_choice if args.capture_mode != "phase_split_gen" else "gen-only",
            )

        _flush_pending()
    except BaseException:
        try:
            _flush_pending()
        except Exception:  # noqa: BLE001 — best-effort persist on the way out; original raises
            logger.exception(
                "[shard %d] best-effort pending-batch flush failed on exit", args.shard_index
            )
        raise
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)

    logger.info(
        "[shard %d] done: %d kept rows across %d chunks (%d over-length skipped, "
        "%d empty-response dropped)",
        args.shard_index,
        kept_total,
        n_sub,
        len(skipped_all),
        len(dropped_empty_all),
    )
    # Cap-hit digest (report-only here; the #2330 disposition is the #1491
    # truncation-restriction control at P3, plan §11 — not the 2% re-gen).
    if gen_total > 0:
        cap_frac = cap_hit_total / gen_total
        logger.info(
            "[shard %d] cap-hit: %d/%d = %.4f (finish_reason=='length', gen_max_tokens=%d)",
            args.shard_index,
            cap_hit_total,
            gen_total,
            cap_frac,
            GEN_MAX_TOKENS,
        )
        if cap_frac > CAP_HIT_REGEN_TRIGGER:
            logger.warning(
                "[shard %d] cap-hit fraction %.2f%% exceeds %.0f%% — report per split; the "
                "plan-registered disposition is the truncation-restriction control at P3",
                args.shard_index,
                100.0 * cap_frac,
                100.0 * CAP_HIT_REGEN_TRIGGER,
            )
        # <think>-leak validity assert (plan §7: thinking-off actually engaged).
        think_frac = think_total / gen_total
        logger.info(
            "[shard %d] think-scan: %d/%d = %.4f responses open with <think>",
            args.shard_index,
            think_total,
            gen_total,
            think_frac,
        )
        if think_frac >= THINK_SCAN_MAX_FRAC:
            raise RuntimeError(
                f"<think>-leak scan FAIL: {think_total}/{gen_total} = {think_frac:.4f} >= "
                f"{THINK_SCAN_MAX_FRAC} of responses open with <think> — enable_thinking=False "
                "is not reaching the template (plan §8 risk row); fix the render call. "
                "Raw completions for this shard are already persisted."
            )

    # Free GPU state before exit (helps a chained follow-up invocation).
    if hf is not None:
        del hf
    if llm is not None:
        _reap_vllm_engine(llm)
        del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.no_upload:
        # P1 step-3 smoke shard (local-only run; production never passes
        # --no-upload): record realized values for the P1 completion sentinel.
        _update_run_meta(
            args.run_meta_out,
            "smoke_shard",
            {
                "split": args.split,
                "capture_mode": args.capture_mode,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "shard_size": args.shard_size,
                "kept_rows": int(kept_total),
                "n_chunks": int(n_sub),
                "overlength_skipped": len(skipped_all),
                "empty_response_dropped": len(dropped_empty_all),
                "gen_rows": int(gen_total),
                "cap_hit": int(cap_hit_total),
                "think_open": int(think_total),
                "capture_fn": capture_fn_choice,
                "passed": True,  # the think-leak gate above raises before this point
            },
        )
        _maybe_write_p1_sentinel(args)

    _phase("done")
    return 0


def _aggregate_cap_hit_core(
    chunks: list[tuple[str, dict]],
    expected_ids: list[int],
    *,
    subset_mode: bool,
    prefix: str,
) -> dict:
    """Pure cap-hit aggregation core over downloaded chunk payloads (round 3,
    FIX 2a — exact-coverage invariants; exercised by ``--selftest-cap-hit``).

    ``expected_ids`` is the LOGICAL split's committed id list (split_ids.json).
    Coverage is EXACT over that set: a missing chunk / missing ci raises, and
    in same-split mode an extra ci raises too. ``subset_mode`` (the banked
    train_25k store read for the train_10k logical split) tolerates store rows
    OUTSIDE the expected set — skipped + counted, never silently mixed into
    the totals. Per-row finish_reason is the ground truth; a chunk whose
    ``n_cap_hit`` metadata disagrees raises, and a row LACKING the field
    raises KeyError (round 4 — a missing field must never silently classify
    as uncapped). Returns the aggregate count fields; the caller adds
    routing + fingerprint + repro metadata."""
    expected = {int(i) for i in expected_ids}
    per_chunk: list[dict] = []
    cap_cis: list[int] = []
    seen: set[int] = set()
    outside_cis: list[int] = []
    n_store_rows = 0
    gen_max: int | None = None
    think_total: int | None = None
    for name, payload in chunks:
        rows = payload["rows"]
        chunk_cap_all = [int(r["ci"]) for r in rows if str(r["finish_reason"]) == "length"]
        meta_n = payload.get("n_cap_hit")
        if meta_n is not None and int(meta_n) != len(chunk_cap_all):
            raise RuntimeError(
                f"{name}: chunk metadata n_cap_hit={meta_n} != per-row finish_reason=='length' "
                f"count {len(chunk_cap_all)} — refusing to aggregate inconsistent metadata"
            )
        n_covered = 0
        for r in rows:
            ci = int(r["ci"])
            if ci in seen:
                raise RuntimeError(f"{name}: duplicate ci {ci} across chunks of {prefix}")
            seen.add(ci)
            if ci in expected:
                n_covered += 1
            else:
                outside_cis.append(ci)
        gm = payload.get("gen_max_tokens")
        if gm is not None:
            assert gen_max is None or int(gm) == gen_max, ("gen_max_tokens drift", gen_max, gm)
            gen_max = int(gm)
        nt = payload.get("n_think_open")
        if nt is not None:
            think_total = (think_total or 0) + int(nt)
        n_store_rows += len(rows)
        cap_cis.extend(c for c in chunk_cap_all if c in expected)
        per_chunk.append(
            {
                "name": name,
                "n_rows": len(rows),
                "n_covered": n_covered,
                "n_cap_hit_store": len(chunk_cap_all),
            }
        )
        print(
            f"[cap-hit] {name}: {len(chunk_cap_all)}/{len(rows)} finish_reason=='length'",
            flush=True,
        )
    missing = sorted(expected - seen)
    if missing:
        raise RuntimeError(
            f"cap-hit coverage INCOMPLETE under {prefix}: {len(missing)}/{len(expected)} "
            f"expected cis absent from the store chunks (first: {missing[:10]}) — a missing "
            "chunk or a partial store; refusing to write a partial aggregate (a missing row "
            "must NEVER read downstream as uncapped)"
        )
    if outside_cis and not subset_mode:
        raise RuntimeError(
            f"cap-hit coverage EXTRA rows under {prefix}: {len(outside_cis)} store cis outside "
            f"the expected id set in same-split mode (first: {sorted(outside_cis)[:10]}) — "
            "store/split_ids drift; refusing to aggregate"
        )
    total = len(expected)  # == covered rows: coverage exact + duplicates impossible
    return {
        "total": total,
        "cap_hit": len(cap_cis),
        "cap_hit_frac": len(cap_cis) / max(total, 1),
        "cap_hit_cis": sorted(cap_cis),
        "n_chunks": len(chunks),
        "n_store_rows": n_store_rows,
        "n_rows_outside_expected": len(outside_cis),
        "gen_max_tokens": gen_max,
        "n_think_open": think_total,
        "per_chunk": per_chunk,
    }


def run_aggregate_cap_hit(args) -> int:
    """Cap-hit aggregator (round 2 FIX 2i; round 3 coverage + routing rework).

    Aggregates per-chunk cap-hit metadata for ONE LOGICAL split into
    ``cap_hit_<split>.json`` (schema ``issue2330_cap_hit_v2``): split totals +
    fraction and the per-context ``cap_hit_cis`` list (finish_reason ==
    'length') the P3 truncation-restriction read+refit control joins on.

    Routing (FIX 1b): chunks are read from
    ``<root>/<store subpath>/raw_completions`` where the store subpath comes
    from the SAME ``store_subpath_for_split`` table run_capture writes with
    (ceiling draws → ``ceiling_draws/seed{S}``); ``--cap-hit-store-split``
    overrides it for the banked 7B train case (``--split train_10k
    --cap-hit-store-split train_25k``), which enters SUBSET mode: store rows
    outside the committed train_10k id set are skipped + counted.

    Coverage (FIX 2a): the committed split_ids.json id set for the logical
    split is loaded and coverage must be EXACT — a missing chunk, a missing
    ci, or (same-split mode) an extra ci each raise; the expected-set sha256
    (copied from split_ids.json's own per-split digest — same producer domain,
    ``issue2330_split_ids._sha256_id_list``) is fingerprinted into the
    aggregate so the fits consumer can refuse an aggregate produced against a
    drifted split_ids. Runnable over the BANKED 7B store (``--cap-hit-root
    issue1491_scale_ladder/scale7_refit --cap-hit-revision <pin>``) AND this
    issue's own 9B prefix (``--hf-prefix``). Uploads the aggregate next to the
    STORE subpath for issue-owned roots only; a foreign banked root is never
    written to (the JSON is committed to git instead)."""
    _phase("aggregate_cap_hit")
    assert args.split, "--split is required for --aggregate-cap-hit"
    root = args.cap_hit_root or args.hf_prefix
    assert root, "--cap-hit-root or --hf-prefix is required for --aggregate-cap-hit"
    split_payload = _load_split_ids(Path(args.split_ids))
    _, ids_key, _ = SPLIT_TO_MANIFEST[args.split]
    expected_ids = [int(i) for i in split_payload["splits"][ids_key]]
    expected_sha = str(split_payload["sha256"][ids_key])
    canonical_subpath = store_subpath_for_split(args.split)
    store_split = args.cap_hit_store_split or canonical_subpath
    subset_mode = store_split != canonical_subpath
    prefix = f"{root}/{store_split}/raw_completions"
    revision = args.cap_hit_revision
    cache_dir = args.out_dir / ".cache_caphit"
    index = _remote_index(prefix, revision=revision)
    names = sorted(n for n in index if n.endswith(".json"))
    if not names:
        raise RuntimeError(f"no raw_completions chunks under {prefix} (revision={revision})")
    print(
        f"[cap-hit] {root}/{args.split}: store_split={store_split} subset_mode={subset_mode} "
        f"expected_n={len(expected_ids)} chunks={len(names)}",
        flush=True,
    )
    chunks: list[tuple[str, dict]] = []
    for name in names:
        local = _hub_download(f"{prefix}/{name}", cache_dir, revision)
        with open(local, encoding="utf-8") as fh:
            chunks.append((name, json.load(fh)))
    agg = _aggregate_cap_hit_core(chunks, expected_ids, subset_mode=subset_mode, prefix=prefix)
    out = {
        "schema": CAP_HIT_SCHEMA,
        "root": root,
        "split": args.split,  # LOGICAL split (the key consumers join on)
        "store_split": store_split,  # store subpath actually read
        "subset_mode": subset_mode,
        "revision": revision,
        "expected_ids_key": ids_key,
        "expected_n": len(expected_ids),
        "expected_ids_sha256": expected_sha,
        "split_ids_path": str(args.split_ids),
        **agg,
        "git_commit": _git_sha(),
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = (
        Path(os.path.expanduser(str(args.cap_hit_out)))
        if args.cap_hit_out
        else (args.out_dir / f"cap_hit_{args.split}.json")
    )
    _write_json_atomic(out_path, out)
    print(
        f"[cap-hit] {root}/{args.split}: {out['cap_hit']}/{out['total']} = "
        f"{100.0 * out['cap_hit'] / max(out['total'], 1):.2f}% cap-hit "
        f"(store rows {out['n_store_rows']}, outside-expected "
        f"{out['n_rows_outside_expected']}) -> {out_path}",
        flush=True,
    )
    if args.no_upload:
        print("[cap-hit] --no-upload: aggregate NOT uploaded (local/smoke path)", flush=True)
    elif args.hf_prefix and root == args.hf_prefix:
        # "Uploaded with the split": one scoped upload_folder commit + verify,
        # next to the STORE subpath the chunks live under.
        _upload_names_once(out_path.parent, f"{root}/{store_split}", [out_path.name], False)
        print(f"[cap-hit] uploaded {out_path.name} -> {root}/{store_split}/", flush=True)
    else:
        print(
            "[cap-hit] foreign/banked root — aggregate NOT uploaded to the banked prefix; "
            "commit the JSON under eval_results/issue_2330/cap_hit/ (git) instead",
            flush=True,
        )
    _phase("done")
    return 0


def _expect_raise_capture(fn, needle: str, what: str) -> None:
    """Selftest helper: fn() must raise RuntimeError containing ``needle``."""
    try:
        fn()
    except RuntimeError as e:
        assert needle in str(e), (what, needle, str(e)[:200])
        print(f"[selftest-cap-hit] PASS {what}: raised as designed ({needle!r})", flush=True)
        return
    raise AssertionError(f"[selftest-cap-hit] FAIL {what}: no RuntimeError raised")


def run_selftest_cap_hit() -> int:
    """CPU selftest for the cap-hit routing table + aggregation core (round 3):
    table-driven logical-split → store-subpath resolution (ceiling path +
    train_25k subset), exact-coverage rejection (missing chunk / missing CI /
    extra CI), metadata mismatch, duplicate ci, subset-mode filtering; round 4
    adds the missing-finish_reason rejection (hard access ⇒ KeyError)."""
    # 1. Routing table: every logical split resolves through the ONE function
    #    run_capture writes with (ceiling draws → ceiling_draws/seed{S}).
    expect_subpath = {
        "train_10k": "train_10k",
        "val_400": "val_400",
        "test_1000": "test_1000",
        "wc_test_1k": "wc_test_1k",
        "ceiling_draw_43": "ceiling_draws/seed43",
        "ceiling_draw_44": "ceiling_draws/seed44",
    }
    assert sorted(expect_subpath) == sorted(SPLIT_TO_MANIFEST), "routing table drift"
    for split, want in expect_subpath.items():
        got = store_subpath_for_split(split)
        assert got == want, (split, got, want)
    print("[selftest-cap-hit] PASS store_subpath_for_split table (6 logical splits)", flush=True)

    def _chunk(name: str, cis: list[int], capped: set[int], meta_n: int | None = None) -> tuple:
        rows = [{"ci": c, "finish_reason": ("length" if c in capped else "stop")} for c in cis]
        n = len([c for c in cis if c in capped]) if meta_n is None else meta_n
        return (name, {"rows": rows, "n_cap_hit": n, "gen_max_tokens": GEN_MAX_TOKENS})

    expected = list(range(10, 20))  # the logical split's committed id list

    # 2. Happy path, same-split mode: exact coverage; totals at the logical grain.
    chunks = [
        _chunk("c0.json", expected[:5], {11, 13}),
        _chunk("c1.json", expected[5:], {17}),
    ]
    agg = _aggregate_cap_hit_core(chunks, expected, subset_mode=False, prefix="selftest")
    assert agg["total"] == 10 and agg["cap_hit"] == 3, agg
    assert agg["cap_hit_cis"] == [11, 13, 17], agg["cap_hit_cis"]
    assert agg["n_store_rows"] == 10 and agg["n_rows_outside_expected"] == 0, agg
    print("[selftest-cap-hit] PASS same-split happy path (exact coverage)", flush=True)

    # 3. Missing chunk (5 expected cis never seen) ⇒ raise.
    _expect_raise_capture(
        lambda: _aggregate_cap_hit_core(chunks[:1], expected, subset_mode=False, prefix="st"),
        "coverage INCOMPLETE",
        "missing chunk",
    )
    # 3b. Missing single CI ⇒ raise.
    holed = [_chunk("c0.json", expected[:5], set()), _chunk("c1.json", expected[5:9], set())]
    _expect_raise_capture(
        lambda: _aggregate_cap_hit_core(holed, expected, subset_mode=False, prefix="st"),
        "coverage INCOMPLETE",
        "missing single ci",
    )
    # 3c. Extra CI in same-split mode ⇒ raise.
    extra = [_chunk("c0.json", expected[:5] + [99], set()), _chunk("c1.json", expected[5:], set())]
    _expect_raise_capture(
        lambda: _aggregate_cap_hit_core(extra, expected, subset_mode=False, prefix="st"),
        "EXTRA rows",
        "extra ci (same-split mode)",
    )
    # 3d. Duplicate ci across chunks ⇒ raise.
    dup = [_chunk("c0.json", expected[:5], set()), _chunk("c1.json", expected[4:], set())]
    _expect_raise_capture(
        lambda: _aggregate_cap_hit_core(dup, expected, subset_mode=False, prefix="st"),
        "duplicate ci",
        "duplicate ci",
    )
    # 3e. Chunk-metadata mismatch ⇒ raise.
    bad_meta = [
        _chunk("c0.json", expected[:5], {11}, meta_n=3),
        _chunk("c1.json", expected[5:], set()),
    ]
    _expect_raise_capture(
        lambda: _aggregate_cap_hit_core(bad_meta, expected, subset_mode=False, prefix="st"),
        "n_cap_hit",
        "chunk metadata mismatch",
    )

    # 3f. Row LACKING finish_reason ⇒ KeyError (round 4: hard access — a row
    #     without the field must never silently classify as uncapped).
    missing_fr = [_chunk("c0.json", expected[:5], {11}), _chunk("c1.json", expected[5:], set())]
    del missing_fr[1][1]["rows"][0]["finish_reason"]
    try:
        _aggregate_cap_hit_core(missing_fr, expected, subset_mode=False, prefix="st")
    except KeyError as e:
        assert "finish_reason" in str(e), e
        print(
            "[selftest-cap-hit] PASS missing finish_reason: KeyError raised as designed",
            flush=True,
        )
    else:
        raise AssertionError("[selftest-cap-hit] FAIL missing finish_reason: no KeyError raised")

    # 4. Subset mode (banked train_25k-style superset store): outside rows
    #    skipped + counted; cap set FILTERED to the expected ids; totals at
    #    the LOGICAL grain.
    superset = [
        _chunk("s0.json", expected[:5] + [100, 101], {11, 100}),
        _chunk("s1.json", expected[5:] + [102], {17, 102}),
    ]
    agg_sub = _aggregate_cap_hit_core(superset, expected, subset_mode=True, prefix="selftest")
    assert agg_sub["total"] == 10 and agg_sub["cap_hit"] == 2, agg_sub
    assert agg_sub["cap_hit_cis"] == [11, 17], agg_sub["cap_hit_cis"]
    assert agg_sub["n_store_rows"] == 13 and agg_sub["n_rows_outside_expected"] == 3, agg_sub
    print("[selftest-cap-hit] PASS subset mode (outside rows skipped+counted)", flush=True)
    # 4b. Subset mode still refuses a MISSING expected ci.
    _expect_raise_capture(
        lambda: _aggregate_cap_hit_core(superset[:1], expected, subset_mode=True, prefix="st"),
        "coverage INCOMPLETE",
        "subset mode missing ci",
    )

    print("[selftest-cap-hit] ALL PASS", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Task #2330 P1/P2: Qwen3.5-9B generate + trimmed capture (standalone port)"
    )
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="HF model id")
    ap.add_argument(
        "--layers",
        default="16,22,30",
        help="comma-separated block indices (9B: 16,22,30; the parity7b gate passes 14,19,26)",
    )
    ap.add_argument(
        "--h-dim",
        type=int,
        default=None,
        help="hidden dim (production passes 4096 explicitly — qwen3_5 keeps it in text_config)",
    )
    ap.add_argument(
        "--split",
        choices=sorted(SPLIT_TO_MANIFEST.keys()),
        default=None,
        help="split to process (required for run mode; hook_probe defaults to train_10k)",
    )
    ap.add_argument(
        "--split-ids",
        default=str(_REPO_ROOT / "eval_results" / "issue_2330" / "split_ids.json"),
        help="path to the P0 split_ids.json (the single-source id lists)",
    )
    ap.add_argument(
        "--hf-prefix",
        default=None,
        help="HF data-repo prefix, e.g. issue2330_matched/qwen35_9b (REQUIRED for run mode; "
        "no default — upload-prefix defaults are the #1005 clobber shape)",
    )
    ap.add_argument(
        "--capture-mode",
        default="coresident",
        choices=["coresident", "phase_split_gen", "phase_split_capture"],
        help="production P2 runs phase_split_gen then phase_split_capture (fp32 9B capture "
        "must not co-reside with the vLLM engine)",
    )
    ap.add_argument(
        "--capture-dtype",
        default="float32",
        choices=sorted(_DTYPES.keys()),
        help="HF capture model dtype (plan §4 P2: fp32 for the 9B; the parity7b gate passes "
        "bfloat16 — the banked 7B captures were computed in bf16)",
    )
    ap.add_argument("--capture-batch-size", type=int, default=8)
    ap.add_argument(
        "--gen-max-tokens",
        type=int,
        default=GEN_MAX_TOKENS,
        help="vLLM generation cap (cap2048 follow-up; default 1024 keeps every existing "
        "invocation byte-identical). A non-default value re-derives MAX_MODEL_LEN with "
        "PROMPT_TOKEN_BUDGET held at 7104 (same admitted row set as the originals)",
    )
    ap.add_argument(
        "--gen-source-prefix",
        default=None,
        help="phase_split_capture only (dense re-capture follow-up): HF prefix holding the "
        "ALREADY-BANKED gen wave's raw_completions to capture from (default --hf-prefix). "
        "Lets a dense re-capture read the original seed-42 completions while writing its "
        ".pt chunks under a NEW --hf-prefix (the original store is never clobbered)",
    )
    ap.add_argument("--num-shards", type=int, default=2, help="plan §4 P2: 2-way sharding")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "EPM_I2330_OUT_DIR",
                os.path.expanduser("~/data/issue_2330/qwen35_generate_capture"),
            )
        ),
        help="scratch/output root (pod launches set EPM_I2330_OUT_DIR under /workspace — "
        "the container overlay HOME is wiped on stop/resume)",
    )
    ap.add_argument(
        "--no-upload", action="store_true", help="local-only (the 500-row smoke shard path)"
    )
    ap.add_argument(
        "--expect-suffix",
        choices=["think", "plain"],
        default="think",
        help="rendered-prompt suffix pin: think = Qwen3.5 empty-think-block (production); "
        "plain = Qwen2.5 (parity7b/emit_spans legs)",
    )
    ap.add_argument(
        "--gate",
        choices=["template_pin", "length_scan", "emit_spans", "parity7b", "hook_probe"],
        default=None,
        help="run ONE P1 convention gate and exit (plan §4 P1 steps 1/2/5/6)",
    )
    ap.add_argument(
        "--max-over-budget-frac",
        type=float,
        default=0.005,
        help="length_scan halt band (plan §7: >0.5%% over budget => re-scope)",
    )
    ap.add_argument(
        "--parity-banked-prefix",
        default=PARITY_BANKED_PREFIX,
        help="banked 7B store prefix for the parity7b/emit_spans gates",
    )
    ap.add_argument(
        "--parity-banked-revision",
        default=MANIFEST_REVISION,
        help="pinned revision of the banked 7B store",
    )
    ap.add_argument("--parity-rows", type=int, default=32)
    ap.add_argument("--parity-cos-min", type=float, default=0.999)
    ap.add_argument(
        "--expected-spans", default=None, help="emit_spans output (required for parity7b)"
    )
    ap.add_argument("--spans-out", default=None, help="output path for emit_spans")
    ap.add_argument("--hook-probe-rows", type=int, default=4)
    ap.add_argument("--hook-rel-tol", type=float, default=1e-5)
    ap.add_argument(
        "--run-meta-out",
        type=Path,
        default=None,
        help="accumulating run-meta JSON (default <out-dir>/run_meta.json)",
    )
    ap.add_argument(
        "--sentinel-path",
        type=Path,
        default=Path("/workspace/logs/issue-2330-p1-smoke.json"),
        help="plan-§9 P1 completion sentinel (written once EVERY P1 gate has a "
        "passed=true run-meta record; default = the plan's pod path — local "
        "smokes pass an explicit /tmp path)",
    )
    ap.add_argument(
        "--aggregate-cap-hit",
        action="store_true",
        help="aggregate per-chunk cap-hit metadata for ONE (root, split) into "
        "cap_hit_<split>.json (schema issue2330_cap_hit_v2; consumed by the P3 "
        "truncation-restriction control + fig_cap_hit)",
    )
    ap.add_argument(
        "--cap-hit-root",
        default=None,
        help="aggregate mode: store root holding <root>/<split>/raw_completions/ "
        "(default --hf-prefix; pass the banked 7B root "
        f"{PARITY_BANKED_PREFIX.rsplit('/', 1)[0]} + --cap-hit-revision for the 7B side)",
    )
    ap.add_argument(
        "--cap-hit-revision",
        default=None,
        help="aggregate mode: pinned data-repo revision for a banked root (7B: the "
        "plan-§10 store pin); None = main (the issue's own 9B prefix)",
    )
    ap.add_argument(
        "--cap-hit-store-split",
        default=None,
        help="aggregate mode: store subpath override when the chunks live under a "
        "DIFFERENT subpath than the logical split's canonical one (the banked 7B "
        "train case: --split train_10k --cap-hit-store-split train_25k — enters "
        "SUBSET mode, filtering store rows to the committed split_ids id set); "
        "default = store_subpath_for_split(--split)",
    )
    ap.add_argument(
        "--selftest-cap-hit",
        action="store_true",
        help="CPU selftest of the cap-hit routing table + aggregation core (round 3: "
        "ceiling/train_25k routing, exact-coverage rejection, subset filtering)",
    )
    ap.add_argument(
        "--cap-hit-out",
        default=None,
        help="aggregate mode: output JSON path (default <out-dir>/cap_hit_<split>.json)",
    )
    ap.add_argument(
        "--fits-smoke",
        action="store_true",
        help="plan §4 P1 step 4 — invoke the P3 fits port (scripts/issue2330_matched_fits.py, "
        "repo venv) on the local 500-row smoke chunk (count pins opted out, labeled smoke, CUDA)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attribute completeness + deferred-import resolution (fresh-venv "
        "pre-flight; imports vllm+transformers, so run it in the target venv)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def _assert_args_attrs_defined() -> None:
    """Whole-module argparse-attribute completeness check (standalone port of
    orchestrate.argcheck — a never-smoked branch must not ship an
    ``args.<attr>`` AttributeError)."""
    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    used: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
            and isinstance(node.ctx, ast.Load)
        ):
            used.add(node.attr)
    defined = {a.dest for a in _build_parser()._actions}  # noqa: SLF001
    missing = sorted(used - defined)
    assert not missing, f"args attributes read but never defined by the parser: {missing}"


def _run_import_check() -> int:
    """Execute every deferred import (the smoke-architecture Axis 1 command)."""
    _assert_args_attrs_defined()
    from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
    from huggingface_hub.errors import (  # noqa: F401
        EntryNotFoundError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )
    from transformers import (  # noqa: F401
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from vllm import LLM, SamplingParams  # noqa: F401

    print("[import-check] OK: argparse attrs complete; deferred imports resolved")
    return 0


def _run_fits_smoke(args) -> int:
    """Plan §4 P1 step 4 — run the REAL P3 fits port on the local 500-row smoke chunk.

    Subprocess into the REPO venv (`uv run` from the repo root — the fits port
    imports `explore_persona_space`, which this driver's fresh qwen35 venv
    deliberately lacks). expected_split_n=None semantics (count pins opted out
    — the documented smoke downgrade, plan §4 smoke-enumeration item (d)),
    labeled smoke, CUDA (artifact-reuse check (m): the 9B d=4096 shape + the
    subset/matched-ID code exercised on the production device class)."""
    chunk_dir = args.out_dir / "shards" / (args.split or "train_10k")
    if not chunk_dir.is_dir() or not sorted(chunk_dir.glob("*.pt")):
        print(
            f"[fits-smoke] no local capture chunks under {chunk_dir} — run the P1 step-3 "
            "smoke shard first (--split train_10k --shard-size 500 --no-upload)",
            file=sys.stderr,
        )
        return 2
    fits_script = _SCRIPTS / "issue2330_matched_fits.py"
    assert fits_script.is_file(), f"P3 fits port missing: {fits_script}"
    out_json = args.out_dir / "fits_smoke.json"
    cmd = [
        "uv",
        "run",
        "python",
        str(fits_script),
        "--smoke-chunk-dir",
        str(chunk_dir),
        "--device",
        "cuda",
        "--h-dim",
        "4096",
        "--out-json",
        str(out_json),
    ]
    print(f"[fits-smoke] invoking P3 fits port (repo venv): {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(_SCRIPTS.parent), env={**os.environ}, check=False)
    if proc.returncode != 0:
        print(f"[fits-smoke] FAIL: fits port exited rc={proc.returncode}", file=sys.stderr)
        return proc.returncode
    assert out_json.is_file(), f"fits smoke exited 0 but wrote no {out_json}"
    _update_run_meta(
        args.run_meta_out,
        "fits_smoke",
        {
            "chunk_dir": str(chunk_dir),
            "out_json": str(out_json),
            "out_json_sha256": _sha256_file(out_json),
            "passed": True,  # rc==0 + artifact-present asserts precede the record
        },
    )
    _maybe_write_p1_sentinel(args)
    print(f"[fits-smoke] OK — {out_json}", flush=True)
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    if args.import_check:
        return _run_import_check()

    if args.gen_max_tokens != GEN_MAX_TOKENS:
        gm, mml, budget = _apply_gen_max_tokens(args.gen_max_tokens)
        print(
            f"[gen-cap-tokens] gen_max_tokens={gm} max_model_len={mml} "
            f"prompt_token_budget={budget} (non-default cap; budget invariant)",
            flush=True,
        )
    if args.gen_source_prefix:
        assert args.capture_mode == "phase_split_capture", (
            "--gen-source-prefix is only meaningful for --capture-mode phase_split_capture "
            f"(got {args.capture_mode!r}) — the gen wave always writes under its own --hf-prefix"
        )

    args.out_dir = Path(os.path.expanduser(str(args.out_dir)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.run_meta_out is None:
        args.run_meta_out = args.out_dir / "run_meta.json"

    if args.selftest_cap_hit:
        # Local-only synthetic-chunk selftest: runs BEFORE the token assert below.
        return run_selftest_cap_hit()

    if args.fits_smoke:
        # Local-only (no HF token needed): runs BEFORE the token assert below.
        return _run_fits_smoke(args)

    # Every mode touches the private data repo (manifest / banked-store /
    # upload paths) — fail fast on a missing token rather than mid-phase.
    assert os.environ.get("HF_TOKEN"), (
        "HF_TOKEN missing — .env not loaded (set EPM_DOTENV_PATH or run from a checkout "
        "with .env present)"
    )

    if args.aggregate_cap_hit:
        return run_aggregate_cap_hit(args)

    if args.gate:
        return {
            "template_pin": gate_template_pin,
            "length_scan": gate_length_scan,
            "emit_spans": gate_emit_spans,
            "parity7b": gate_parity7b,
            "hook_probe": gate_hook_probe,
        }[args.gate](args)

    assert args.split, "--split is required for run mode"
    assert args.hf_prefix, "--hf-prefix is required for run mode (no default by design)"
    return run_capture(args)


if __name__ == "__main__":
    import traceback

    # Round-2 vllm-exception-teardown guard (unanimous concern): an exception
    # must NEVER reach interpreter finalization with an engine constructed —
    # finalize-time multiprocessing cleanup deadlocks on engine children
    # (gotchas.md "sys.exit() is NOT a terminal for a vLLM generation driver").
    try:
        _rc = main()
    except SystemExit as e:  # argparse exits / SIGTERM handler — preserve the code
        _code = e.code
        if isinstance(_code, int) or _code is None:
            _rc = 0 if _code is None else int(_code)
        else:
            print(_code, file=sys.stderr)
            _rc = 1
    except BaseException:
        traceback.print_exc()
        _rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    if _ENGINE_CONSTRUCTED:
        if _LIVE_ENGINE is not None:
            try:  # best-effort reap on the exception path; the traceback is already printed
                _reap_vllm_engine(_LIVE_ENGINE)
            except Exception:
                traceback.print_exc()
        # gotchas.md: sys.exit() is NOT a terminal for a vLLM generation driver —
        # finalize-time multiprocessing cleanup can deadlock on engine children.
        # All durables (uploads verified before purge, atomic raw writes,
        # [phase=done]) have landed by this point.
        os._exit(_rc)
    sys.exit(_rc)
