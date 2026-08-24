#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
"""Issue #2546 generation + parse + capture rig (plan v4 §4.2 P1-P4).

Ported from ``scripts/issue928_extract_thinking_store.py`` keeping its phase
structure (gate -> Generate -> Parse -> Batch-capture -> Upload) and its
load-bearing capture mechanics: char-offset -> token-position span mapping
(``issue928_common.char_span_to_token_span``, the #825 zero-width-span-safe
form), left-padded batched teacher-forced forwards with explicit
``position_ids``, ``logits_to_keep=1`` introspection guard (#779), in-forward
streaming reduction (#666/#772 — the token x layer grid is never
materialized), and length-sorted token-budget batch packing.

Phases (CLI contract pinned by ``scripts/issue2546_dispatch.sh``):

- default (no ``--phase``) + ``--smoke``  -> P1 rig smoke (arm 1: 240 GSM8K-test
  rows + 5/other-corpus + 8 T=0.6 rows; arms 2/3: 60-row lite incl. the arm-3
  think-off leg). Gates G-A/G-B/G-C/G-D/G-F evaluated in-script; rc=0 all
  PASS, rc=4 declared-fallback band (report names the flag), rc=3 hard FAIL.
- ``--phase pilot``     -> P2a (arm 1 only): full gsm8k_test1319 post+pre gen
  + capture of both; measures production walls (gen tok/s, capture rows/s,
  one-shard serialize+upload wall) into the pilot report.
- ``--phase gen-post``  -> P2: post-side / think-on greedy generation over the
  arm's full row set (cap 8,192; forced 2x re-gen of finish_reason=="length"
  rows; residual >2%/cell drop-and-count) + T=0.6/top_p .95 x4 reliability
  draws at the plan quotas. Rollout TEXT persists locally per corpus and
  uploads at phase end, BEFORE any reduction.
- ``--phase gen-short`` -> P3: pre-side / think-off generation (+500x4
  reliability draws).
- ``--phase capture``   -> P4: teacher-forced capture of both sides, bf16
  shards of <=500 rows per (model-or-mode, corpus) stem, finiteness asserted,
  per-corpus upload-then-free; G-F determinism gate (fresh-load 2-row
  re-capture, two-bar 0.999 early / 0.9999 flat); per-row metadata incl.
  exact-match correctness (boxed/letter/native, exact code — zero LLM-judge
  calls) and the arm-3 toggle-necessity + arms-1/2 pair_necessity labels.

Port deviations from #928 (each plan-registered):

1. Store codec is bf16, NOT fp16 (plan divergence 5 — fp16 overflows residual
   outliers; #825/#1336 parity).
2. No ``<|im_end|>`` + ``\\n`` boundary feed after the completion: the 2546
   kind set (cx_last / cot_mean / cot_boundary / ans_mean / out_mean /
   think_t10..t90 / pre-side cx_last+ans_mean) reads no post-completion
   boundary position, so the teacher-forced input is prompt + completion ids.
3. Per-arm segment semantics (emergent / prefill / off) replace
   ``issue928_common.segment_completion``'s rung semantics; the char->token
   mapping + repeated-4-gram screen are reused verbatim from issue928_common.
4. Store schema is kind-tensor shards per (side, corpus) stem, not the #928
   per-context ``percq_summaries`` blobs (the #2546 fit cells consume
   (n, kind, layer, H) tensors).

Parallelism: one vLLM engine / one HF capture model per GPU; rows sharded by
row-id hash across workers spawned with per-process ``CUDA_VISIBLE_DEVICES``
pinned in the LAUNCHER env (never ``+gpu_id``; gotchas.md CVD family). The
parent process never touches CUDA.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))
# vLLM v1 EngineCore dies silently under fork() when the parent touched
# CUDA-adjacent code before LLM() (gotchas.md #628) — set BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Cross-script helper imports hoisted to module top so a missing symbol crashes
# at process start, never inside a smoke-skipped branch (gotchas.md #606).
from issue928_common import char_span_to_token_span, repeated_4gram_fraction  # noqa: E402

from explore_persona_space.orchestrate.hub import (  # noqa: E402
    DEFAULT_DATASET_REPO,
    _upload,
    stage_hub_prefix,
)

logger = logging.getLogger("issue2546_gen_capture")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Constants (plan v4 §4.2 / §10 / §11)
# ---------------------------------------------------------------------------

RECIPE_VERSION = "issue2546-genrig-v1"
SEED = 2546
CORPORA_PREFIX = "issue2546_cotmap/corpora_v1"
RAW_PREFIX = "issue2546_cotmap/raw_completions"
STORE_PREFIX = "issue2546_cotmap/analysis_tensors/thinkstore"

MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.85
VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
REL_TEMPERATURE, REL_TOP_P, REL_DRAWS = 0.6, 0.95, 4
T_GRID = tuple(round(0.1 * i, 1) for i in range(1, 10))  # think_t10 .. think_t90
SHARD_ROWS = 500
SHORT_THINK_TOKENS = 10  # rows with think-span < 10 keep collided t-positions, fraction reported

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
PREFILL_TEXT = THINK_OPEN + "\n"

# Gates (plan §7)
GATE_GA_FLOOR = 0.99  # think-emission well-formed floor (per post-like side)
GATE_GA_PREFILL_BAND = 0.90  # 90-99% -> declared prefill-fallback rung
GATE_GB_USABLE_FLOOR = 0.90  # usable floor on the GSM8K slice
GATE_GB_OFFENDER_MAX = 0.10  # repetition offenders at > 0.50 repeated-4-gram
REPEAT_4GRAM_MAX_FRAC = 0.50
CAPHIT_TRIGGER = 0.02  # production per-cell trigger (post re-gen residual -> drop-and-count)
SMOKE_CAPHIT_GB3 = 0.05  # arm-3 smoke cap-hit band for the declared T=0.6 fallback (G-B3)

# G-F capture-determinism two-bar gate (Source: #1005, bf16-calibrated)
GF_EARLY_LAYERS = (0, 1, 2, 3)
GF_EARLY_BAR = 0.999
GF_FLAT_BAR = 0.9999

# Designed artifact-routed halts (never a bare rc=1; gotchas.md pilot-gate entry)
RC_GATE_FAIL = 3
RC_FALLBACK_BAND = 4

CAPTURE_BATCH_ROWS = int(os.environ.get("EPM_I2546_BATCH_ROWS", "64"))
CAPTURE_TOKEN_BUDGET = int(os.environ.get("EPM_I2546_TOKEN_BUDGET", "16384"))

CORPUS_ORDER = [
    "gsm8k_test",
    "gsm8k_train",
    "math",
    "contexthub",
    "mmlu",
    "arc_challenge",
    "csqa",
    "piqa",
]

# MATH gold join source (the #1336 math7500 producer: rows with dataset=="MATH"
# of the RLVR mix; src_index indexes its train split at this pinned revision).
RLVR_DATASET = "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints"
RLVR_REV = "7dbd180f5440"

# Reliability-draw quotas, arm-1 post base (plan §4.2 P2); other totals scale
# by largest-remainder allocation in scaled_quota().
REL_BASE = {
    "gsm8k_train:kbin": 150,  # per realized k-bin
    "math": 200,
    "contexthub:cell": 50,  # per (type x level) cell
    "mmlu": 100,
    "arc_challenge": 50,
    "csqa": 50,
    "piqa": 100,
}
REL_TOTAL_POST = {1: 1500, 2: 1000, 3: 1000}
REL_TOTAL_SHORT = 500  # every arm (plan §4.2 P3)


# ---------------------------------------------------------------------------
# Arm registry (plan §4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SideSpec:
    side: str  # post | pre | think_on | think_off
    model: str
    parse_mode: str  # emergent | prefill | off
    cap: int
    regen_cap: int
    post_like: bool  # captures 5 full kinds + t-kinds; else cx_last + ans_mean
    stage: str  # raw-completions stage name (plan §10)
    read_point: str  # prompt_last | pre_think | assist_start
    enable_thinking: bool | None = None  # arm-3 apply_chat_template kwarg
    open_ids: tuple[int, ...] | None = None  # pinned think-open encoding (asserted)
    close_ids: tuple[int, ...] | None = None  # pinned think-close encoding (asserted)


@dataclass(frozen=True)
class ArmSpec:
    arm: int
    n_layers: int
    hidden: int
    frozen: tuple[int, int, int]
    sides: tuple[SideSpec, ...]  # (post-like, short-like)
    render_identity: str  # identical | content_only | single_model (G-D form)


ARMS: dict[int, ArmSpec] = {
    1: ArmSpec(
        arm=1,
        n_layers=28,
        hidden=3584,
        frozen=(14, 19, 26),
        render_identity="identical",
        sides=(
            SideSpec(
                side="post",
                model="open-thoughts/OpenThinker3-7B",
                parse_mode="emergent",
                cap=8192,
                regen_cap=16384,
                post_like=True,
                stage="post_greedy_a1",
                read_point="prompt_last",
                open_ids=(13708, 766, 29),
                close_ids=(522, 26865, 29),
            ),
            SideSpec(
                side="pre",
                model="Qwen/Qwen2.5-7B-Instruct",
                parse_mode="off",
                cap=2048,
                regen_cap=4096,
                post_like=False,
                stage="pre_greedy_a1",
                read_point="prompt_last",
            ),
        ),
    ),
    2: ArmSpec(
        arm=2,
        n_layers=28,
        hidden=3584,
        frozen=(14, 19, 26),
        render_identity="content_only",
        sides=(
            SideSpec(
                side="post",
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                parse_mode="prefill",  # template prefills <think>\n (#1005 semantics)
                cap=8192,
                regen_cap=16384,
                post_like=True,
                stage="post_greedy_a2",
                read_point="pre_think",
                open_ids=(151648,),
                close_ids=(151649,),
            ),
            SideSpec(
                side="pre",
                model="Qwen/Qwen2.5-Math-7B",
                parse_mode="off",
                cap=4096,  # Math-7B's default template elicits stepwise answers (plan §11)
                regen_cap=8192,
                post_like=False,
                stage="pre_greedy_a2",
                read_point="prompt_last",
            ),
        ),
    ),
    3: ArmSpec(
        arm=3,
        n_layers=36,
        hidden=4096,
        frozen=(18, 24, 33),
        render_identity="single_model",
        sides=(
            SideSpec(
                side="think_on",
                model="Qwen/Qwen3-8B",
                parse_mode="emergent",
                cap=8192,
                regen_cap=16384,
                post_like=True,
                stage="thinkon_a3",
                read_point="assist_start",
                enable_thinking=True,
                open_ids=(151667,),
                close_ids=(151668,),
            ),
            SideSpec(
                side="think_off",
                model="Qwen/Qwen3-8B",
                parse_mode="off",
                cap=2048,
                regen_cap=4096,
                post_like=False,
                stage="thinkoff_a3",
                read_point="assist_start",
                enable_thinking=False,
            ),
        ),
    ),
}

KINDS_POST = ("cx_last", "cot_mean", "cot_boundary", "ans_mean", "out_mean")
KINDS_SHORT = ("cx_last", "ans_mean")
KINDS_T = tuple(f"think_t{int(round(t * 100))}" for t in T_GRID)


def phase_line(name: str) -> None:
    """Emit a poll_pipeline.py-parseable phase breadcrumb."""
    print(f"[phase={name}]", flush=True)


def _git_sha() -> str:
    """Tolerant commit sha (git-less scratch trees degrade, never crash; #1902)."""
    env_sha = os.environ.get("EPS_GIT_SHA")
    if env_sha:
        return env_sha
    p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return p.stdout.strip() if p.returncode == 0 else "unavailable-no-git-checkout"


def repro_meta(phase: str) -> dict:
    """Reproducibility metadata block for every result artifact (CLAUDE.md rule)."""
    import transformers

    return {
        "task": 2546,
        "phase": phase,
        "recipe_version": RECIPE_VERSION,
        "git_commit": _git_sha(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": transformers.__version__,
        },
    }


def _norm(text: str) -> str:
    """Whitespace-collapsed join key (matches issue2546_stage_corpora._norm)."""
    return " ".join(text.split())


def _slot_of(row_id: str, num_workers: int) -> int:
    return int(hashlib.sha1(row_id.encode("utf-8")).hexdigest(), 16) % num_workers


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict]:
    # Text-mode iteration, never .splitlines() (U+2028-in-strings shred; gotchas.md #950).
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Corpus staging + per-arm row sets
# ---------------------------------------------------------------------------


def stage_corpora(out_root: Path, corpora_dir: str | None) -> tuple[dict[str, list[dict]], dict]:
    """Load the P0 corpora bundle (local-first, else HF-staged; fail loud).

    Search order: ``--corpora-dir`` override -> ``data/issue_2546/corpora_v1``
    (VM-local P0 output) -> stage from ``issue2546_cotmap/corpora_v1/`` on the
    data repo via ``stage_hub_prefix`` (dest is a MIRROR ROOT — files land at
    ``mirror/<repo-relative path>``; #1774 gotcha, arithmetic asserted).
    """
    candidates = []
    if corpora_dir:
        candidates.append(Path(corpora_dir))
    candidates.append(PROJECT_ROOT / "data" / "issue_2546" / "corpora_v1")
    mirror = out_root / "hf_dl" / "corpora_mirror"
    staged_dir = mirror / CORPORA_PREFIX
    candidates.append(staged_dir)
    src = next((c for c in candidates if (c / "corpora_manifest.json").is_file()), None)
    if src is None:
        logger.info("[stage] corpora not local — staging %s from HF", CORPORA_PREFIX)
        staged = stage_hub_prefix(DEFAULT_DATASET_REPO, CORPORA_PREFIX, mirror)
        assert (staged_dir / "corpora_manifest.json").is_file(), (
            f"stage_hub_prefix mirror arithmetic broke: corpora_manifest.json not at "
            f"{staged_dir} (staged {len(staged)} files)"
        )
        src = staged_dir
    manifest = json.loads((src / "corpora_manifest.json").read_text())
    corpora: dict[str, list[dict]] = {}
    for c in CORPUS_ORDER:
        single = src / f"{c}.jsonl"
        man = src / f"{c}.manifest.json"
        if single.is_file():
            corpora[c] = _read_jsonl(single)
        elif man.is_file():
            parts = json.loads(man.read_text())["parts"]
            rows: list[dict] = []
            for p in parts:
                rows.extend(_read_jsonl(src / p))
            corpora[c] = rows
        else:
            raise FileNotFoundError(f"corpus {c!r}: neither {single} nor {man} exists")
        if not corpora[c]:
            raise RuntimeError(f"corpus {c!r} loaded EMPTY from {src} — refusing (fail loud)")
        logger.info("[stage] %s: %d rows", c, len(corpora[c]))
    return corpora, manifest


def arm_rows(corpora: dict[str, list[dict]], arm: int) -> dict[str, list[dict]]:
    key = "in_arm3" if arm == 3 else "in_arm12"
    out = {c: [r for r in rows if r[key]] for c, rows in corpora.items()}
    empty = [c for c, rows in out.items() if not rows]
    if empty:
        raise RuntimeError(f"arm {arm}: corpora with ZERO in-arm rows: {empty}")
    return out


def smoke_slice(rows_by_corpus: dict[str, list[dict]], arm: int) -> dict[str, list[dict]]:
    """P1 smoke slice: arm 1 = 240 gsm8k_test + 5/other corpus; arms 2/3 = 40 + 5."""
    n_gsm = 240 if arm == 1 else 40
    rng = np.random.default_rng(SEED)
    out: dict[str, list[dict]] = {}
    for c, rows in rows_by_corpus.items():
        n = n_gsm if c == "gsm8k_test" else 5
        idx = rng.permutation(len(rows))[: min(n, len(rows))]
        out[c] = [rows[int(i)] for i in sorted(idx)]
    return out


def scaled_quota(total_target: int, strata_sizes: dict[str, int]) -> dict[str, int]:
    """Largest-remainder allocation of ``total_target`` across strata,
    proportional to the arm-1 base quotas (REL_BASE), capped at stratum size."""
    base: dict[str, float] = {}
    for name in strata_sizes:
        if name.startswith("gsm8k_train:"):
            base[name] = REL_BASE["gsm8k_train:kbin"]
        elif name.startswith("contexthub:"):
            base[name] = REL_BASE["contexthub:cell"]
        else:
            base[name] = REL_BASE[name]
    tot = sum(base.values())
    raw = {k: total_target * v / tot for k, v in base.items()}
    alloc = {k: min(int(raw[k]), strata_sizes[k]) for k in raw}
    rem = total_target - sum(alloc.values())
    order = sorted(raw, key=lambda k: raw[k] - int(raw[k]), reverse=True)
    i = 0
    while rem > 0 and i < 10 * len(order):
        k = order[i % len(order)]
        if alloc[k] < strata_sizes[k]:
            alloc[k] += 1
            rem -= 1
        i += 1
    return alloc


def reliability_row_ids(rows_by_corpus: dict[str, list[dict]], total_target: int) -> list[str]:
    """Seeded stratified reliability-draw selection (plan §4.2 P2/P3 quotas)."""
    strata: dict[str, list[str]] = {}
    for c, rows in rows_by_corpus.items():
        for r in rows:
            if c == "gsm8k_train":
                key = f"gsm8k_train:{r['k_bin']}"
            elif c == "contexthub":
                key = f"contexthub:{r['ch_type']}_L{r['level']}"
            elif c == "gsm8k_test":
                continue  # eval-only corpus: excluded from reliability quotas
            else:
                key = c
            strata.setdefault(key, []).append(r["row_id"])
    sizes = {k: len(v) for k, v in strata.items()}
    alloc = scaled_quota(total_target, sizes)
    rng = np.random.default_rng(SEED + 77)
    picked: list[str] = []
    for k in sorted(strata):
        ids = sorted(strata[k])
        take = alloc[k]
        idx = rng.permutation(len(ids))[:take]
        picked.extend(ids[int(i)] for i in idx)
    return picked


# ---------------------------------------------------------------------------
# Prompt render + read-point (plan §4.1 per-arm v_C convention)
# ---------------------------------------------------------------------------


def render_prompt(tok, user_text: str, side: SideSpec, prefill_fallback: bool) -> str:
    kwargs = {}
    if side.enable_thinking is not None:
        kwargs["enable_thinking"] = side.enable_thinking
    text = tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    if prefill_fallback and side.parse_mode == "emergent" and side.post_like:
        # Declared G-A fallback rung: assistant-prefix <think>\n prefill with
        # #1005 parser semantics (exactly-one </think>). Never a silent switch.
        text = text + PREFILL_TEXT
    return text


def effective_parse_mode(side: SideSpec, prefill_fallback: bool) -> str:
    if prefill_fallback and side.parse_mode == "emergent" and side.post_like:
        return "prefill"
    return side.parse_mode


def _find_last_subseq(ids: list[int], sub: tuple[int, ...]) -> int:
    n, m = len(ids), len(sub)
    for start in range(n - m, -1, -1):
        if tuple(ids[start : start + m]) == sub:
            return start
    return -1


def compute_read_idx(
    side: SideSpec,
    prompt_ids: list[int],
    *,
    prefill_fallback: bool,
    on_prompt_len: int | None = None,
    tpl_prompt_len: int | None = None,
) -> int:
    """Registered per-arm v_C read point (plan §4.1), as an index into prompt_ids.

    prompt_last  -> last generation-prompt token (arms 1/2 pre; arm-1 post).
    pre_think    -> last token PRECEDING the prefilled <think> (arm-2 post; the
                    #1005 prefill-rung convention). Under the arm-1 prefill
                    FALLBACK the read point is the last TEMPLATED-prompt token
                    (tpl_prompt_len - 1) — the same "before any think token"
                    semantics.
    assist_start -> the shared assistant-start position of the arm-3 mode pair
                    (= last token of the THINK-ON render, which ends at the
                    assistant tag; the off-mode prompt continues past it).
    """
    if side.read_point == "assist_start":
        assert on_prompt_len is not None and 0 < on_prompt_len <= len(prompt_ids)
        return on_prompt_len - 1
    if prefill_fallback and side.parse_mode == "emergent" and side.post_like:
        assert tpl_prompt_len is not None and 0 < tpl_prompt_len <= len(prompt_ids)
        return tpl_prompt_len - 1
    if side.read_point == "pre_think":
        assert side.open_ids is not None
        start = _find_last_subseq(prompt_ids, side.open_ids)
        assert start > 0, (
            f"arm-2 prefill read point: think-open ids {side.open_ids} not found in the "
            f"generation prompt (len {len(prompt_ids)}) — template drifted, refusing"
        )
        return start - 1
    return len(prompt_ids) - 1


def assert_think_pins(tok, side: SideSpec) -> None:
    """Assert the pinned think-delimiter encodings on this side's tokenizer (P0 pin)."""
    if side.open_ids is not None:
        got = tuple(tok.encode(THINK_OPEN, add_special_tokens=False))
        assert got == side.open_ids, f"{side.model}: {THINK_OPEN!r} -> {got} != {side.open_ids}"
    if side.close_ids is not None:
        got = tuple(tok.encode(THINK_CLOSE, add_special_tokens=False))
        assert got == side.close_ids, f"{side.model}: {THINK_CLOSE!r} -> {got} != {side.close_ids}"


def resolve_stop_ids(model: str, revision: str | None) -> list[int]:
    """Stop ids from the model's own generation_config.json (asserted; plan §10)."""
    from transformers import GenerationConfig

    gc = GenerationConfig.from_pretrained(model, revision=revision)
    eos = gc.eos_token_id
    ids = [eos] if isinstance(eos, int) else list(eos or [])
    assert ids and all(isinstance(i, int) for i in ids), (
        f"{model}: generation_config.json eos_token_id unusable: {eos!r}"
    )
    return ids


def resolve_revision(model: str, out_root: Path) -> str:
    """Resolve + pin the model revision sha once per out-root (write-once assert)."""
    from huggingface_hub import HfApi

    pin_path = out_root / "revisions.json"
    pins = json.loads(pin_path.read_text()) if pin_path.is_file() else {}
    if model in pins:
        return pins[model]
    sha = HfApi().model_info(model).sha
    assert sha, f"could not resolve revision for {model}"
    pins[model] = sha
    _atomic_write_json(pin_path, pins)
    logger.info("[pin] %s @ %s", model, sha)
    return sha


# ---------------------------------------------------------------------------
# Parse (per-arm segment semantics; char offsets — the #928 exact-offset parser)
# ---------------------------------------------------------------------------


def _strip_span(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def segment_completion_arm(
    text: str, mode: str
) -> tuple[bool, str, tuple[int, int], tuple[int, int]]:
    """(well_formed, reason, cot_char_span, ans_char_span) under per-arm semantics.

    emergent: exactly one <think> (nothing but whitespace before it) and exactly
    one </think>, open before close (arm 1 / arm-3 think-on).
    prefill:  the prompt carries the <think>; well-formed iff exactly one
    </think> and NO open tag in the completion (#1005; arm-2 post + fallback).
    off:      no think block — the answer span is the whole generated text
    (arm-3 think-off / plain pre-side completions).
    """
    if mode == "off":
        s, e = _strip_span(text, 0, len(text))
        return (e > s), ("empty_answer" if e <= s else ""), (0, 0), (s, e)
    n_open, n_close = text.count(THINK_OPEN), text.count(THINK_CLOSE)
    zero = ((0, 0), (0, 0))
    if mode == "prefill":
        if n_close != 1:
            return False, f"close_count_{n_close}", *zero
        if n_open != 0:
            return False, "unexpected_open_tag", *zero
        c = text.index(THINK_CLOSE)
        cot = _strip_span(text, 0, c)
    elif mode == "emergent":
        if n_open != 1 or n_close != 1:
            return False, f"open{n_open}_close{n_close}", *zero
        o, c = text.index(THINK_OPEN), text.index(THINK_CLOSE)
        if c < o:
            return False, "close_before_open", *zero
        if text[:o].strip():
            return False, "text_before_open", *zero
        cot = _strip_span(text, o + len(THINK_OPEN), c)
    else:
        raise ValueError(f"unknown parse mode {mode!r}")
    ans = _strip_span(text, c + len(THINK_CLOSE), len(text))
    if cot[1] <= cot[0]:
        return False, "empty_think", cot, ans
    if ans[1] <= ans[0]:
        return False, "empty_answer", cot, ans
    return True, "", cot, ans


def parse_generation(row: dict, mode: str) -> dict:
    """Parse one merged rollout row -> parse record (drop-and-count classes)."""
    text, fr = row["text"], row["finish_reason"]
    wf, reason, cot, ans = segment_completion_arm(text, mode)
    if not wf and fr == "length" and mode != "off" and THINK_CLOSE not in text:
        reason = "truncated_no_close"
    if mode == "off" and fr == "length":
        # Residual cap-hit after the forced re-gen: drop-and-count (plan G-C).
        wf, reason = False, "truncated_residual"
    rep = repeated_4gram_fraction(text)
    if wf and rep > REPEAT_4GRAM_MAX_FRAC:
        wf, reason = False, "degenerate_repetition"
    if wf and mode != "off" and fr == "length":
        wf, reason = False, "truncated_residual"
    return {
        "well_formed": wf,
        "reason": reason,
        "cot_char_span": list(cot),
        "ans_char_span": list(ans),
        "rep_frac": rep,
        "finish_reason": fr,
        "structural_ok": segment_completion_arm(text, mode)[0],
    }


# ---------------------------------------------------------------------------
# Exact-match correctness (exact code; zero LLM-judge calls — plan §4.3)
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\s*\{")
_LETTER_RE = re.compile(r"\b([A-J])\b")


def extract_boxed(text: str) -> str | None:
    """Content of the LAST \\boxed{...} (brace-balanced)."""
    last = None
    for m in _BOXED_RE.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[m.end() : i - 1]
    return last


def _norm_math(ans: str) -> str:
    s = ans.strip().replace(" ", "").replace(",", "").replace("$", "")
    s = s.removeprefix("\\text{").removesuffix("}") if s.startswith("\\text{") else s
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _norm_free(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def exact_match_correct(corpus: str, ans_text: str, gold: str | None) -> bool | None:
    """Deterministic exact-match correctness per corpus family (plan §6 MF-4 classes)."""
    if gold is None:
        return None
    if corpus in ("gsm8k_test", "gsm8k_train", "math"):
        boxed = extract_boxed(ans_text)
        return _norm_math(boxed) == _norm_math(gold) if boxed is not None else False
    if corpus in ("mmlu", "arc_challenge", "csqa", "piqa"):
        m = _LETTER_RE.search(ans_text)
        return (m.group(1) == gold.strip().upper()) if m else False
    if corpus == "contexthub":
        lines = [ln for ln in ans_text.strip().splitlines() if ln.strip()]
        last = _norm_free(lines[-1]) if lines else ""
        g = _norm_free(gold)
        return bool(g) and (last == g or g in last.split() or f" {g} " in f" {last} ")
    raise ValueError(f"unknown corpus {corpus!r}")


def stage_math_golds(rows: list[dict]) -> dict[str, str]:
    """Join MATH golds from the RLVR mix by src_index (capture-metadata time).

    The #1336 math7500 rows carry src_index into the RLVR train split @
    RLVR_REV; ``ground_truth`` is the verifiable gold. Grain-checked per row:
    the staged question must be contained in the source first user turn.
    """
    from datasets import load_dataset

    ds = load_dataset(RLVR_DATASET, split="train", revision=RLVR_REV)
    out: dict[str, str] = {}
    for r in rows:
        srow = ds[int(r["src_index"])]
        turn = next((m["content"] for m in srow["messages"] if m["role"] == "user"), None)
        assert turn is not None, f"math src_index {r['src_index']}: no user turn in RLVR row"
        assert _norm(r["question"]) in _norm(turn), (
            f"math src_index {r['src_index']}: staged question not contained in the RLVR "
            "first user turn — join grain broke, refusing"
        )
        gold = srow.get("ground_truth")
        assert gold is not None and str(gold).strip(), (
            f"math src_index {r['src_index']}: RLVR row lacks ground_truth"
        )
        out[r["row_id"]] = str(gold)
    return out


# ---------------------------------------------------------------------------
# Gates (plan §7) — computed on parse records
# ---------------------------------------------------------------------------


def gate_report(parses: list[dict], smoke_n_cap: int) -> dict:
    n = len(parses)
    wf_structural = sum(1 for p in parses if p["structural_ok"])
    usable = sum(1 for p in parses if p["well_formed"])
    offenders = sum(1 for p in parses if p["rep_frac"] > REPEAT_4GRAM_MAX_FRAC)
    caphit = sum(1 for p in parses if p["finish_reason"] == "length")
    reasons: dict[str, int] = {}
    for p in parses:
        if not p["well_formed"]:
            reasons[p["reason"]] = reasons.get(p["reason"], 0) + 1
    return {
        "n_rows": n,
        "emission_rate": wf_structural / max(1, n),
        "usable_rate": usable / max(1, n),
        "offender_rate": offenders / max(1, n),
        "caphit_rate": caphit / max(1, n),
        "cap": smoke_n_cap,
        "malformed_reasons": reasons,
    }


# ---------------------------------------------------------------------------
# vLLM generation (worker side)
# ---------------------------------------------------------------------------


def build_engine(model: str, revision: str | None):
    from vllm import LLM

    return LLM(
        model=model,
        revision=revision,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        seed=SEED,
    )


def sampling_params(cap: int, stop_ids: list[int], *, greedy: bool, seed: int | None = None):
    from vllm import SamplingParams

    if greedy:
        return SamplingParams(temperature=0.0, max_tokens=cap, stop_token_ids=stop_ids)
    return SamplingParams(
        temperature=REL_TEMPERATURE,
        top_p=REL_TOP_P,
        seed=seed,
        max_tokens=cap,
        stop_token_ids=stop_ids,
    )


def generate_chunked(llm, prompts: list[str], sp, tag: str) -> list[tuple[str, str, int]]:
    """Chunked order-preserving generate -> [(text, finish_reason, n_gen_tokens)].

    Per-chunk INFO logs keep the poller's stall detection fed (gotchas.md #664);
    ``use_tqdm=False`` (#613 ZeroDivision + GCE line-length traps).
    """
    out: list[tuple[str, str, int]] = []
    n_chunks = (len(prompts) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
    for i in range(0, len(prompts), VLLM_CHUNK_SIZE):
        chunk = prompts[i : i + VLLM_CHUNK_SIZE]
        logger.info(
            "[vllm-chunk] %s chunk %d/%d (%d prompts)",
            tag,
            i // VLLM_CHUNK_SIZE + 1,
            n_chunks,
            len(chunk),
        )
        for o in llm.generate(chunk, sp, use_tqdm=False):
            out.append((o.outputs[0].text, o.outputs[0].finish_reason, len(o.outputs[0].token_ids)))
    return out


def run_gen_worker(args) -> None:
    """One GPU's generation slot: primary -> forced re-gen -> reliability draws."""
    work = json.loads(Path(args.work_file).read_text())
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(work["model"], revision=work["revision"])
    side = SideSpec(**work["side_spec"])
    assert_think_pins(tok, side)
    llm = build_engine(work["model"], work["revision"])
    rows = work["rows"]
    prompts = [r["prompt"] for r in rows]
    stop_ids = work["stop_ids"]
    greedy = not work["decode_fallback"]
    sp = sampling_params(work["cap"], stop_ids, greedy=greedy, seed=None if greedy else SEED)
    prim = generate_chunked(llm, prompts, sp, f"primary-{side.side}")
    out_rows = []
    regen_idx = [i for i, (_t, fr, _n) in enumerate(prim) if fr == "length"]
    regen_out: dict[int, tuple[str, str, int]] = {}
    if regen_idx:
        logger.info(
            "[regen] %d/%d rows hit cap %d -> forced %d",
            len(regen_idx),
            len(rows),
            work["cap"],
            work["regen_cap"],
        )
        sp2 = sampling_params(
            work["regen_cap"], stop_ids, greedy=greedy, seed=None if greedy else SEED
        )
        regen_texts = generate_chunked(llm, [prompts[i] for i in regen_idx], sp2, "regen")
        regen_out = dict(zip(regen_idx, regen_texts, strict=True))
    for i, r in enumerate(rows):
        text, fr, ntok = prim[i]
        rec = {
            "row_id": r["row_id"],
            "corpus": r["corpus"],
            "kind": "primary",
            "text": text,
            "finish_reason": fr,
            "n_gen_tokens": ntok,
            "regen": False,
            "n_prompt_tokens": r["n_prompt_tokens"],
            "read_idx": r["read_idx"],
        }
        if i in regen_out:
            # Persist the superseded truncated primary too (persist-by-default).
            out_rows.append({**rec, "kind": "superseded_primary"})
            t2, fr2, n2 = regen_out[i]
            rec = {**rec, "text": t2, "finish_reason": fr2, "n_gen_tokens": n2, "regen": True}
        out_rows.append(rec)
    rel_ids = set(work["rel_row_ids"])
    rel_rows = [(i, r) for i, r in enumerate(rows) if r["row_id"] in rel_ids]
    for draw in range(work["rel_draws"]):
        if not rel_rows:
            break
        spd = sampling_params(work["cap"], stop_ids, greedy=False, seed=SEED * 100 + draw)
        outs = generate_chunked(llm, [prompts[i] for i, _ in rel_rows], spd, f"rel-d{draw}")
        for (_, r), (text, fr, ntok) in zip(rel_rows, outs, strict=True):
            out_rows.append(
                {
                    "row_id": r["row_id"],
                    "corpus": r["corpus"],
                    "kind": "reliability",
                    "draw": draw,
                    "text": text,
                    "finish_reason": fr,
                    "n_gen_tokens": ntok,
                    "regen": False,
                    "n_prompt_tokens": r["n_prompt_tokens"],
                    "read_idx": r["read_idx"],
                }
            )
    _write_jsonl(Path(work["out_file"]), out_rows)
    logger.info(
        "[gen-worker] slot %s wrote %d rows -> %s",
        args.worker_slot,
        len(out_rows),
        work["out_file"],
    )
    sys.stdout.flush()
    sys.stderr.flush()
    # A vLLM generation driver's terminal is os._exit(0) after flushes + durable
    # writes — sys.exit deadlocks on surviving engine children (gotchas.md #1739/#2149).
    os._exit(0)


# ---------------------------------------------------------------------------
# Teacher-forced capture (worker side) — ported #928 mechanics
# ---------------------------------------------------------------------------


def _logits_to_keep_kwargs(model) -> dict:
    """``logits_to_keep=1`` when the forward names it EXPLICITLY (gotcha #779)."""
    import inspect

    fwd = getattr(model, "forward", None) or model.__call__
    try:
        params = inspect.signature(fwd).parameters
    except (TypeError, ValueError):
        return {}
    return {"logits_to_keep": 1} if "logits_to_keep" in params else {}


def build_capture_row(tok, wrow: dict, post_like: bool) -> tuple[dict | None, str]:
    """One teacher-forced row: ids + spans + positions, or (None, counted reason).

    Token spans derive from ``return_offsets_mapping`` over the completion text
    (robust to BPE merges; #825 zero-width spans drop with a counted reason).
    The teacher-forced input is prompt_ids + completion_ids CONCATENATED AS IDS
    (never a re-tokenize of the concatenated string — the #1092 seam rule).
    """
    prompt_ids = tok(wrow["prompt"], add_special_tokens=False)["input_ids"]
    assert len(prompt_ids) == wrow["n_prompt_tokens"], (
        f"{wrow['row_id']}: prompt re-tokenization drifted "
        f"({len(prompt_ids)} != {wrow['n_prompt_tokens']})"
    )
    enc = tok(wrow["text"], add_special_tokens=False, return_offsets_mapping=True)
    comp_ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if not comp_ids:
        return None, "empty_completion_tokens"
    prompt_len = len(prompt_ids)
    ans_tok = char_span_to_token_span(offsets, tuple(wrow["ans_char_span"]))
    if ans_tok == (0, 0):
        return None, "empty_ans_token_span"
    spans = {
        "ans": (prompt_len + ans_tok[0], prompt_len + ans_tok[1]),
        "out": (prompt_len, prompt_len + len(comp_ids)),
    }
    positions = {"cx_last": wrow["read_idx"]}
    t_positions: list[int] = []
    short_think = False
    if post_like:
        cot_tok = char_span_to_token_span(offsets, tuple(wrow["cot_char_span"]))
        if cot_tok == (0, 0):
            return None, "empty_cot_token_span"
        spans["cot"] = (prompt_len + cot_tok[0], prompt_len + cot_tok[1])
        close_char = wrow["text"].index(THINK_CLOSE) + len(THINK_CLOSE) - 1
        close_tok = char_span_to_token_span(offsets, (close_char, close_char + 1))
        if close_tok == (0, 0):
            return None, "empty_close_token_span"
        positions["cot_boundary"] = prompt_len + close_tok[1] - 1
        n_think = cot_tok[1] - cot_tok[0]
        short_think = n_think < SHORT_THINK_TOKENS
        # token index = round(t * (L_think - 1)); collided positions kept (plan §4.2 P4)
        t_positions = [prompt_len + cot_tok[0] + int(round(t * (n_think - 1))) for t in T_GRID]
    full_ids = prompt_ids + list(comp_ids)
    n_total = len(full_ids)
    for name, (s, e) in spans.items():
        assert 0 <= s < e <= n_total, (name, s, e, n_total)
    for name, p in positions.items():
        assert 0 <= p < n_total, (name, p, n_total)
    for p in t_positions:
        assert prompt_len <= p < n_total, ("think_t", p, n_total)
    return {
        "row_id": wrow["row_id"],
        "full_ids": torch.tensor(full_ids, dtype=torch.long),
        "spans": spans,
        "positions": positions,
        "t_positions": t_positions,
        "short_think": short_think,
    }, ""


def pack_batches(rows: list[dict], batch_rows: int, token_budget: int) -> list[list[int]]:
    """Length-sorted token-budget packing (#928): <= batch_rows rows AND
    <= token_budget padded tokens (B x max_len) per batch."""
    order = sorted(range(len(rows)), key=lambda i: -int(rows[i]["full_ids"].shape[0]))
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        length = int(rows[i]["full_ids"].shape[0])
        new_max = max(cur_max, length)
        if cur and (len(cur) + 1 > batch_rows or new_max * (len(cur) + 1) > token_budget):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = length
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def reduce_forward_batch_2546(
    model, capture, batch_rows: list[dict], arm: ArmSpec, post_like: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """ONE left-padded forward -> (full, tk) bf16 CPU tensors.

    full: (B, K_full, L_all, H) — K_full = 5 post-like / 2 short-like kinds.
    tk:   (B, 9, 3, H) think_t10..t90 at the arm's frozen layers (post-like only).
    Explicit ``position_ids`` (RoPE under left-pad silently diverges without
    them); fp32 reduction of the bf16 hook captures; finiteness asserted before
    the bf16 cast (plan §4.2 P4).
    """
    device = model.device
    kinds = KINDS_POST if post_like else KINDS_SHORT
    B = len(batch_rows)
    pad_id = model.config.eos_token_id
    if isinstance(pad_id, list):
        pad_id = pad_id[0]
    max_len = max(int(r["full_ids"].shape[0]) for r in batch_rows)
    input_ids = torch.full((B, max_len), int(pad_id), dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    mean_parts = ("cot", "ans", "out") if post_like else ("ans",)
    part_masks = {p: torch.zeros((B, max_len), dtype=torch.bool) for p in mean_parts}
    pos_names = ["cx_last", "cot_boundary"] if post_like else ["cx_last"]
    pos_idx = torch.zeros((B, len(pos_names)), dtype=torch.long)
    t_idx = torch.zeros((B, len(T_GRID)), dtype=torch.long) if post_like else None
    for bi, r in enumerate(batch_rows):
        length = int(r["full_ids"].shape[0])
        pad = max_len - length  # LEFT-pad: real tokens at [pad, max_len)
        input_ids[bi, pad:] = r["full_ids"]
        attn[bi, pad:] = 1
        for p in mean_parts:
            s, e = r["spans"][p]
            part_masks[p][bi, pad + s : pad + e] = True
        for pi, name in enumerate(pos_names):
            pos_idx[bi, pi] = pad + r["positions"][name]
        if post_like:
            for ti, p in enumerate(r["t_positions"]):
                t_idx[bi, ti] = pad + p
    input_ids = input_ids.to(device)
    attn = attn.to(device)
    position_ids = (attn.long().cumsum(dim=1) - 1).clamp(min=0).to(device)
    pos_idx_dev = pos_idx.to(device)
    t_idx_dev = t_idx.to(device) if t_idx is not None else None
    masks_dev = {p: m.to(device) for p, m in part_masks.items()}
    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=position_ids,
            **_logits_to_keep_kwargs(model),
        )
    H = arm.hidden
    frozen_pos = {layer: k for k, layer in enumerate(arm.frozen)}
    per_layer_full: list[torch.Tensor] = []
    tk_layers: list[torch.Tensor | None] = [None] * len(arm.frozen)
    for li in range(arm.n_layers):
        hs = capture.latest[li].float()  # (B, T, H) fp32 reduce of the bf16 capture
        assert hs.shape == (B, max_len, H), (hs.shape, (B, max_len, H))
        by_name: dict[str, torch.Tensor] = {}
        for p in mean_parts:
            m = masks_dev[p].unsqueeze(-1)
            cnt = masks_dev[p].sum(dim=1).clamp(min=1).unsqueeze(-1)
            by_name[f"{p}_mean"] = (hs * m).sum(dim=1) / cnt
        picked = torch.gather(hs, 1, pos_idx_dev.unsqueeze(-1).expand(B, len(pos_names), H))
        for pi, name in enumerate(pos_names):
            by_name[name] = picked[:, pi]
        stacked = torch.stack([by_name[k] for k in kinds], dim=1)  # (B, K, H)
        assert torch.isfinite(stacked).all(), f"non-finite capture at layer {li}"
        per_layer_full.append(stacked.cpu())
        if post_like and li in frozen_pos:
            tpick = torch.gather(hs, 1, t_idx_dev.unsqueeze(-1).expand(B, len(T_GRID), H))
            assert torch.isfinite(tpick).all(), f"non-finite t-kind capture at layer {li}"
            tk_layers[frozen_pos[li]] = tpick.cpu()
    capture.latest.clear()
    full = torch.stack(per_layer_full, dim=2)  # (B, K, L, H) fp32 CPU
    tk = torch.stack([t for t in tk_layers], dim=2) if post_like else None  # (B, 9, 3, H)
    return full.to(torch.bfloat16), (tk.to(torch.bfloat16) if tk is not None else None)


def _capture_rows_to_tensors(
    model, capture, rows: list[dict], arm: ArmSpec, post_like: bool, tag: str
) -> tuple[torch.Tensor, torch.Tensor | None, list[int]]:
    """Capture ``rows`` in packed batches; returns tensors in ROW order + order idx."""
    batches = pack_batches(rows, CAPTURE_BATCH_ROWS, CAPTURE_TOKEN_BUDGET)
    full_parts: dict[int, torch.Tensor] = {}
    tk_parts: dict[int, torch.Tensor] = {}
    t0 = time.time()
    for bnum, batch in enumerate(batches):
        f, tk = reduce_forward_batch_2546(model, capture, [rows[i] for i in batch], arm, post_like)
        for j, i in enumerate(batch):
            full_parts[i] = f[j]
            if tk is not None:
                tk_parts[i] = tk[j]
        print(
            f"[capture] {tag} batch {bnum + 1}/{len(batches)} rows={len(batch)} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    order = list(range(len(rows)))
    full = torch.stack([full_parts[i] for i in order])
    tk_t = torch.stack([tk_parts[i] for i in order]) if post_like else None
    return full, tk_t, order


def _cosine_by_layer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B, K, L, H) x2 -> (L,) mean cosine over rows x kinds, fp32."""
    af, bf = a.float(), b.float()
    num = (af * bf).sum(-1)
    den = af.norm(dim=-1) * bf.norm(dim=-1)
    cos = num / den.clamp(min=1e-12)
    return cos.mean(dim=(0, 1))


def run_capture_worker(args) -> None:
    """One GPU's capture slot: build rows, forward, shard-write, optional G-F gate."""
    work = json.loads(Path(args.work_file).read_text())
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from issue594_extract_context_vectors import LayerCapture

    arm = ARMS[work["arm"]]
    side = SideSpec(**work["side_spec"])
    tok = AutoTokenizer.from_pretrained(work["model"], revision=work["revision"])
    assert_think_pins(tok, side)

    def _load_model():
        m = AutoModelForCausalLM.from_pretrained(
            work["model"], revision=work["revision"], torch_dtype=torch.bfloat16
        ).to("cuda")
        m.eval()
        assert m.config.hidden_size == arm.hidden, (m.config.hidden_size, arm.hidden)
        assert m.config.num_hidden_layers == arm.n_layers, (
            m.config.num_hidden_layers,
            arm.n_layers,
        )
        return m

    model = _load_model()
    capture = LayerCapture(model, arm.n_layers)
    built: list[dict] = []
    metas: list[dict] = []
    drops: Counter[str] = Counter()
    for wrow in work["rows"]:
        row, reason = build_capture_row(tok, wrow, side.post_like)
        if row is None:
            drops[reason] += 1
            continue
        built.append(row)
        metas.append({**wrow["meta"], "row_id": wrow["row_id"], "short_think": row["short_think"]})
    stem_dir = Path(work["stem_dir"])
    stem_dir.mkdir(parents=True, exist_ok=True)
    n_shards = 0
    for s0 in range(0, len(built), SHARD_ROWS):
        chunk = built[s0 : s0 + SHARD_ROWS]
        chunk_meta = metas[s0 : s0 + SHARD_ROWS]
        full, tk, _ = _capture_rows_to_tensors(
            model, capture, chunk, arm, side.post_like, f"{work['stem']}/slot{args.worker_slot}"
        )
        shard = {
            "task": 2546,
            "arm": arm.arm,
            "side": side.side,
            "model": work["model"],
            "revision": work["revision"],
            "corpus": work["corpus"],
            "kinds_full": list(KINDS_POST if side.post_like else KINDS_SHORT),
            "kinds_t": list(KINDS_T) if side.post_like else [],
            "layers_all": list(range(arm.n_layers)),
            "frozen_layers": list(arm.frozen),
            "hidden": arm.hidden,
            "full": full,
            "tk": tk,
            "row_ids": [r["row_id"] for r in chunk],
            "meta": chunk_meta,
            "repro": repro_meta("p4_capture"),
        }
        dest = stem_dir / f"slot{args.worker_slot}.shard{n_shards:03d}.pt"
        tmp = stem_dir / (dest.name + ".tmp.pt")
        torch.save(shard, tmp)
        os.replace(tmp, dest)
        n_shards += 1
        print(
            f"[capture] {work['stem']} slot{args.worker_slot} shard {n_shards} "
            f"({len(chunk)} rows) -> {dest}",
            flush=True,
        )
    gf_result = None
    if work.get("gf_gate") and built:
        # G-F determinism gate: fresh model load, 2-row re-capture, two-bar cosine.
        gf_rows = built[: min(2, len(built))]
        ref_full, _, _ = _capture_rows_to_tensors(
            model, capture, gf_rows, arm, side.post_like, "gf-ref"
        )
        capture.remove()
        del model
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        model = _load_model()
        capture = LayerCapture(model, arm.n_layers)
        new_full, _, _ = _capture_rows_to_tensors(
            model, capture, gf_rows, arm, side.post_like, "gf-fresh"
        )
        by_layer = _cosine_by_layer(ref_full, new_full)
        early_min = float(by_layer[list(GF_EARLY_LAYERS)].min())
        flat_mean = float(by_layer.mean())
        gf_result = {
            "early_min_cosine": early_min,
            "flat_mean_cosine": flat_mean,
            "early_bar": GF_EARLY_BAR,
            "flat_bar": GF_FLAT_BAR,
            "pass": early_min >= GF_EARLY_BAR and flat_mean >= GF_FLAT_BAR,
            "n_rows": len(gf_rows),
        }
    capture.remove()
    _atomic_write_json(
        Path(work["summary_file"]),
        {
            "slot": args.worker_slot,
            "stem": work["stem"],
            "n_captured": len(built),
            "n_dropped": sum(drops.values()),
            "drop_reasons": dict(drops),
            "n_shards": n_shards,
            "short_think_frac": (sum(1 for m in metas if m["short_think"]) / max(1, len(metas))),
            "gf": gf_result,
        },
    )
    sys.stdout.flush()
    sys.stderr.flush()
    # Heavy-C-extension entrypoints exit explicitly (PyGILState_Release atexit
    # race turns a COMPLETED phase into rc!=0 under set -euo pipefail; gotchas.md).
    sys.exit(0)


# ---------------------------------------------------------------------------
# Parent orchestration
# ---------------------------------------------------------------------------


def gpu_count() -> int:
    # nvidia-smi subprocess, never torch (the CVD-clobber family; gotchas.md).
    p = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, check=False)
    if p.returncode != 0:
        return 0
    return len([ln for ln in p.stdout.split("\n") if ln.strip()])


def spawn_workers(script_args: list[str], work_files: list[Path], out_root: Path, tag: str) -> None:
    """Spawn one CVD-pinned worker per work file; fail loud on any rc != 0."""
    procs = []
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for slot, wf in enumerate(work_files):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(slot)}
        log_path = log_dir / f"{tag}.slot{slot}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            *script_args,
            "--worker-slot",
            str(slot),
            "--work-file",
            str(wf),
        ]
        logger.info("[spawn] slot %d (CVD=%d) -> %s (log %s)", slot, slot, wf.name, log_path)
        with log_path.open("w") as lf:
            procs.append(
                (
                    slot,
                    log_path,
                    subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env),
                )
            )
    failed = []
    for slot, log_path, p in procs:
        rc = p.wait()
        if rc != 0:
            tail = "".join(log_path.read_text().split("\n")[-120:])
            logger.error("[spawn] slot %d FAILED rc=%d; log tail:\n%s", slot, rc, tail)
            failed.append((slot, rc))
    if failed:
        raise RuntimeError(f"{tag}: worker slots failed: {failed}")


def prompt_budget(side: SideSpec) -> int:
    # Regen-aware length budget (verify_plan c69 arithmetic: max_model_len - 2x cap).
    return MAX_MODEL_LEN - 2 * side.cap


def compose_prompts(
    tok,
    tok_on,
    side: SideSpec,
    rows_by_corpus: dict[str, list[dict]],
    prefill_fallback: bool,
) -> tuple[dict[str, list[dict]], dict]:
    """Render + tokenize prompts, compute read points, drop overlong rows (digest-only).

    ``tok_on`` is the same tokenizer, used to derive the arm-3 shared
    assistant-start read point from the THINK-ON render (asserted prefix).
    """
    out: dict[str, list[dict]] = {}
    dropped: dict[str, list[dict]] = {}
    budget = prompt_budget(side)
    for c, rows in rows_by_corpus.items():
        keep = []
        drop = []
        for r in rows:
            text = render_prompt(tok, r["user_text"], side, prefill_fallback)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) > budget:
                drop.append({"row_id": r["row_id"], "n_tokens": len(ids)})
                continue
            on_len = None
            tpl_len = None
            if side.read_point == "assist_start":
                on_text = tok_on.apply_chat_template(
                    [{"role": "user", "content": r["user_text"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                on_ids = tok_on(on_text, add_special_tokens=False)["input_ids"]
                assert ids[: len(on_ids)] == on_ids, (
                    f"{r['row_id']}: arm-3 mode renders do NOT share the assistant-start "
                    "prefix — template drifted, refusing (plan §4.1 read-point convention)"
                )
                on_len = len(on_ids)
            if prefill_fallback and side.parse_mode == "emergent" and side.post_like:
                tpl_text = render_prompt(tok, r["user_text"], side, False)
                tpl_ids = tok(tpl_text, add_special_tokens=False)["input_ids"]
                assert ids[: len(tpl_ids)] == tpl_ids, f"{r['row_id']}: prefill prefix drifted"
                tpl_len = len(tpl_ids)
            read_idx = compute_read_idx(
                side,
                ids,
                prefill_fallback=prefill_fallback,
                on_prompt_len=on_len,
                tpl_prompt_len=tpl_len,
            )
            keep.append(
                {
                    "row_id": r["row_id"],
                    "corpus": c,
                    "prompt": text,
                    "n_prompt_tokens": len(ids),
                    "read_idx": read_idx,
                    "read_distance": len(ids) - 1 - read_idx,
                }
            )
        out[c] = keep
        if drop:
            dropped[c] = drop
            logger.warning(
                "[compose] %s: dropped %d overlong prompts (budget %d)", c, len(drop), budget
            )
    return out, {
        "budget_tokens": budget,
        "dropped": {c: len(d) for c, d in dropped.items()},
        "dropped_detail": dropped,
    }


def gen_fingerprint(arm: ArmSpec, side: SideSpec, revision: str, flags: dict) -> dict:
    return {
        "recipe_version": RECIPE_VERSION,
        "arm": arm.arm,
        "side": side.side,
        "model": side.model,
        "revision": revision,
        "cap": side.cap,
        "regen_cap": side.regen_cap,
        "parse_mode": side.parse_mode,
        **flags,
    }


def run_generation(
    args,
    arm: ArmSpec,
    side: SideSpec,
    rows_by_corpus: dict[str, list[dict]],
    out_root: Path,
    rel_total: int,
    num_workers: int,
) -> dict[str, Path]:
    """Parent gen driver: compose -> spawn workers -> merge per corpus -> meta."""
    from transformers import AutoTokenizer

    revision = resolve_revision(side.model, out_root)
    tok = AutoTokenizer.from_pretrained(side.model, revision=revision)
    assert_think_pins(tok, side)
    stop_ids = resolve_stop_ids(side.model, revision)
    flags = {
        "decode_fallback": bool(args.decode_fallback and side.post_like),
        "prefill_fallback": bool(args.prefill_fallback),
    }
    fp = gen_fingerprint(arm, side, revision, flags)
    stage_dir = out_root / "rollouts" / side.stage
    rel_dir = out_root / "rollouts" / f"reliability_a{arm.arm}"
    merged: dict[str, Path] = {}
    pending = {}
    for c, rows in rows_by_corpus.items():
        dest = stage_dir / f"{c}.jsonl"
        meta_p = stage_dir / f"{c}.meta.json"
        if dest.is_file() and meta_p.is_file():
            old = json.loads(meta_p.read_text())
            if old.get("fingerprint") == fp:
                logger.info("[gen] %s/%s: resume-skip (fingerprint match)", side.stage, c)
                merged[c] = dest
                continue
            raise RuntimeError(
                f"{dest}: existing rollouts carry a DIFFERENT regime fingerprint "
                f"({old.get('fingerprint')} != {fp}) — refusing to mix; use a fresh out-root"
            )
        pending[c] = rows
    if not pending:
        return merged
    composed, compose_report = compose_prompts(tok, tok, side, pending, args.prefill_fallback)
    rel_ids = set(reliability_row_ids(composed, rel_total)) if rel_total > 0 else set()
    all_rows = [r for c in sorted(composed) for r in composed[c]]
    work_dir = out_root / "work" / f"{args.phase}_{side.side}"
    work_dir.mkdir(parents=True, exist_ok=True)
    work_files = []
    for slot in range(num_workers):
        slot_rows = [r for r in all_rows if _slot_of(r["row_id"], num_workers) == slot]
        wf = work_dir / f"slot{slot}.json"
        _atomic_write_json(
            wf,
            {
                "model": side.model,
                "revision": revision,
                "side_spec": asdict(side),
                "cap": side.cap,
                "regen_cap": side.regen_cap,
                "stop_ids": stop_ids,
                "decode_fallback": flags["decode_fallback"],
                "rel_draws": REL_DRAWS if rel_total > 0 else 0,
                "rel_row_ids": sorted(rid for rid in rel_ids),
                "rows": slot_rows,
                "out_file": str(work_dir / f"slot{slot}.out.jsonl"),
            },
        )
        work_files.append(wf)
    base_args = [
        "--arm",
        str(arm.arm),
        "--phase",
        args.phase,
        "--out-root",
        str(out_root),
        "--worker-kind",
        "gen",
    ]
    if args.smoke:
        base_args.append("--smoke")
    if args.decode_fallback:
        base_args.append("--decode-fallback")
    if args.prefill_fallback:
        base_args.append("--prefill-fallback")
    spawn_workers(base_args, work_files, out_root, f"gen-{side.stage}")
    slot_rows_out: list[dict] = []
    for slot in range(num_workers):
        slot_rows_out.extend(_read_jsonl(work_dir / f"slot{slot}.out.jsonl"))
    by_corpus: dict[str, dict[str, list[dict]]] = {}
    for rec in slot_rows_out:
        by_corpus.setdefault(rec["corpus"], {}).setdefault(rec["kind"], []).append(rec)
    for c in sorted(pending):
        recs = by_corpus.get(c, {})
        primaries = recs.get("primary", [])
        expect = len(composed[c])
        assert len(primaries) == expect, (
            f"{side.stage}/{c}: primary coverage {len(primaries)} != composed {expect}"
        )
        caphit = sum(1 for r in primaries if r["finish_reason"] == "length") / max(1, expect)
        dest = stage_dir / f"{c}.jsonl"
        _write_jsonl(dest, sorted(primaries, key=lambda r: r["row_id"]))
        sup = recs.get("superseded_primary", [])
        if sup:
            _write_jsonl(
                out_root / "rollouts" / f"regen16k_a{arm.arm}" / f"{side.side}__{c}.jsonl",
                sorted(sup, key=lambda r: r["row_id"]),
            )
        rel = recs.get("reliability", [])
        if rel:
            _write_jsonl(
                rel_dir / f"{side.side}__{c}.jsonl",
                sorted(rel, key=lambda r: (r["row_id"], r["draw"])),
            )
        _atomic_write_json(
            stage_dir / f"{c}.meta.json",
            {
                "fingerprint": fp,
                "n_rows": expect,
                "caphit_residual_rate": caphit,
                "caphit_trigger": CAPHIT_TRIGGER,
                "caphit_over_trigger": caphit > CAPHIT_TRIGGER,
                "n_regen": sum(1 for r in primaries if r.get("regen")),
                "n_reliability": len(rel),
                "compose": {k: v for k, v in compose_report.items() if k != "dropped_detail"},
                "repro": repro_meta(args.phase),
            },
        )
        print(
            f"[gen] {side.stage}/{c}: {expect} rows merged (cap-hit residual {caphit:.4f})",
            flush=True,
        )
        merged[c] = dest
    if compose_report["dropped"]:
        _atomic_write_json(stage_dir / "_overlong_dropped.json", compose_report)
    return merged


def upload_stage(out_root: Path, rel_stage: str, skip: bool, smoke: bool) -> None:
    """Upload one local rollout stage dir to the HF data repo (fail loud)."""
    local = out_root / "rollouts" / rel_stage
    if skip or not local.is_dir():
        if skip:
            logger.info("[upload] SKIPPED (--skip-upload): %s", rel_stage)
        return
    dest_stage = f"smoke_{rel_stage}" if smoke else rel_stage
    dest = f"{RAW_PREFIX}/{dest_stage}"
    res = _upload(local, DEFAULT_DATASET_REPO, "dataset", dest, raise_on_error=True)
    assert res, f"rollout upload returned empty result for {dest} (HF_TOKEN missing?)"
    logger.info("[upload] rollouts %s -> %s", rel_stage, res)


def load_rollouts(out_root: Path, side: SideSpec, corpus: str) -> list[dict]:
    p = out_root / "rollouts" / side.stage / f"{corpus}.jsonl"
    if not p.is_file():
        raise FileNotFoundError(
            f"rollouts missing for {side.stage}/{corpus}: {p} — run the generation phase first"
        )
    return _read_jsonl(p)


def run_capture(
    args,
    arm: ArmSpec,
    rows_by_corpus: dict[str, list[dict]],
    out_root: Path,
    num_workers: int,
    corpora_filter: list[str] | None = None,
) -> dict:
    """Parent capture driver (P4): parse -> metadata -> workers -> shard verify ->
    per-corpus upload-then-free -> necessity labels."""
    row_index = {r["row_id"]: r for rows in rows_by_corpus.values() for r in rows}
    math_golds: dict[str, str] = {}
    if any(r["corpus"] == "math" for rows in rows_by_corpus.values() for r in rows) and (
        corpora_filter is None or "math" in corpora_filter
    ):
        math_rows = [r for r in rows_by_corpus.get("math", [])]
        if math_rows:
            logger.info("[capture] joining %d MATH golds from %s", len(math_rows), RLVR_DATASET)
            math_golds = stage_math_golds(math_rows)
    correctness: dict[str, dict[str, bool | None]] = {s.side: {} for s in arm.sides}
    reports: dict[str, dict] = {}
    gf_results: dict[str, dict] = {}
    for side in arm.sides:
        revision = resolve_revision(side.model, out_root)
        mode = effective_parse_mode(side, args.prefill_fallback)
        corpora = corpora_filter or [c for c in CORPUS_ORDER if c in rows_by_corpus]
        for ci, c in enumerate(corpora):
            stem = f"{side.side}__{c}"
            stem_dir = out_root / "store" / f"arm{arm.arm}" / stem
            done_p = stem_dir / "_complete.json"
            fp = gen_fingerprint(arm, side, revision, {"stage": "capture"})
            if done_p.is_file() and json.loads(done_p.read_text()).get("fingerprint") == fp:
                logger.info("[capture] %s: resume-skip", stem)
                prior = json.loads(done_p.read_text())
                for rid, corr in prior.get("correctness", {}).items():
                    correctness[side.side][rid] = corr
                reports[stem] = prior["report"]
                continue
            rollouts = load_rollouts(out_root, side, c)
            wrows = []
            parse_counts: Counter[str] = Counter()
            corr_this: dict[str, bool | None] = {}
            for rec in rollouts:
                parse = parse_generation(rec, mode)
                src = row_index[rec["row_id"]]
                gold = math_golds.get(rec["row_id"], src.get("gold_answer"))
                ans_text = rec["text"][parse["ans_char_span"][0] : parse["ans_char_span"][1]]
                correct = exact_match_correct(c, ans_text, gold) if parse["well_formed"] else None
                corr_this[rec["row_id"]] = correct
                if not parse["well_formed"]:
                    parse_counts[parse["reason"]] += 1
                    continue
                parse_counts["usable"] += 1
                wrows.append(
                    {
                        "row_id": rec["row_id"],
                        "corpus": c,
                        "prompt": None,  # filled below (re-render, deterministic)
                        "text": rec["text"],
                        "n_prompt_tokens": rec["n_prompt_tokens"],
                        "read_idx": rec["read_idx"],
                        "cot_char_span": parse["cot_char_span"],
                        "ans_char_span": parse["ans_char_span"],
                        "meta": {
                            "corpus": c,
                            "side": side.side,
                            "finish_reason": rec["finish_reason"],
                            "regen": rec.get("regen", False),
                            "n_gen_tokens": rec["n_gen_tokens"],
                            "read_distance": rec["n_prompt_tokens"] - 1 - rec["read_idx"],
                            "k_bin": src.get("k_bin"),
                            "k": src.get("k"),
                            "level": src.get("level"),
                            "ch_type": src.get("ch_type"),
                            "rescue_rate": src.get("rescue_rate"),
                            "correct": correct,
                            "parse_reason": "",
                        },
                    }
                )
            usable_rate = parse_counts["usable"] / max(1, len(rollouts))
            if usable_rate < GATE_GB_USABLE_FLOOR:
                # Per-corpus usable floor FLAGS, never silently drops (plan §4.3).
                logger.warning(
                    "[capture] FLAG: %s usable rate %.3f < %.2f (reasons: %s) — "
                    "captured rows proceed; the fit-side denominator is revised",
                    stem,
                    usable_rate,
                    GATE_GB_USABLE_FLOOR,
                    dict(parse_counts),
                )
            # Re-render prompts deterministically (the rollout file stores no
            # prompt text; render is a pure function of user_text + template).
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(side.model, revision=revision)
            for w in wrows:
                w["prompt"] = render_prompt(
                    tok, row_index[w["row_id"]]["user_text"], side, args.prefill_fallback
                )
            work_dir = out_root / "work" / f"capture_{stem}"
            work_dir.mkdir(parents=True, exist_ok=True)
            work_files = []
            for slot in range(num_workers):
                slot_rows = [w for w in wrows if _slot_of(w["row_id"], num_workers) == slot]
                wf = work_dir / f"slot{slot}.json"
                _atomic_write_json(
                    wf,
                    {
                        "arm": arm.arm,
                        "model": side.model,
                        "revision": revision,
                        "side_spec": asdict(side),
                        "corpus": c,
                        "stem": stem,
                        "stem_dir": str(stem_dir),
                        "rows": slot_rows,
                        "summary_file": str(work_dir / f"slot{slot}.summary.json"),
                        # G-F once per side, on the first captured corpus, slot 0.
                        "gf_gate": bool(ci == 0 and slot == 0),
                    },
                )
                work_files.append(wf)
            base_args = [
                "--arm",
                str(arm.arm),
                "--phase",
                args.phase,
                "--out-root",
                str(out_root),
                "--worker-kind",
                "capture",
            ]
            if args.smoke:
                base_args.append("--smoke")
            if args.prefill_fallback:
                base_args.append("--prefill-fallback")
            t_ser0 = time.time()
            spawn_workers(base_args, work_files, out_root, f"capture-{stem}")
            summaries = [
                json.loads((work_dir / f"slot{s}.summary.json").read_text())
                for s in range(num_workers)
            ]
            n_captured = sum(s["n_captured"] for s in summaries)
            assert n_captured == len(wrows), (
                f"{stem}: captured {n_captured} != composed {len(wrows)} "
                f"(drops: {[s['drop_reasons'] for s in summaries]})"
            )
            for s in summaries:
                if s.get("gf") is not None:
                    gf_results[side.side] = s["gf"]
            report = {
                "stem": stem,
                "n_rollout_rows": len(rollouts),
                "n_captured": n_captured,
                "usable_rate": usable_rate,
                "parse_counts": dict(parse_counts),
                "span_drop_reasons": dict(
                    sum((Counter(s["drop_reasons"]) for s in summaries), Counter())
                ),
                "short_think_frac": float(np.mean([s["short_think_frac"] for s in summaries])),
                "wall_s": time.time() - t_ser0,
            }
            reports[stem] = report
            correctness[side.side].update(corr_this)
            # Per-corpus upload-then-free (bounds arm-3 peak disk; plan §4.2 P4).
            if not args.skip_upload:
                dest = f"{STORE_PREFIX}/arm{arm.arm}/{stem}"
                if args.smoke:
                    dest = f"{STORE_PREFIX}/smoke_arm{arm.arm}/{stem}"
                t_up = time.time()
                res = _upload(stem_dir, DEFAULT_DATASET_REPO, "dataset", dest, raise_on_error=True)
                assert res, f"store upload returned empty result for {dest}"
                report["upload_wall_s"] = time.time() - t_up
                if not args.smoke and args.phase == "capture":
                    shard_files = sorted(stem_dir.glob("slot*.shard*.pt"))
                    for f in shard_files:
                        f.unlink()
                    logger.info(
                        "[capture] %s: freed %d local shards post-upload", stem, len(shard_files)
                    )
            _atomic_write_json(
                done_p,
                {
                    "fingerprint": fp,
                    "report": report,
                    "correctness": corr_this,
                    "repro": repro_meta(args.phase),
                },
            )
            print(f"[capture] {stem}: complete ({n_captured} rows)", flush=True)
    necessity = compute_necessity(arm, correctness, row_index, out_root)
    return {"reports": reports, "gf": gf_results, "necessity_summary": necessity}


def compute_necessity(
    arm: ArmSpec,
    correctness: dict[str, dict[str, bool | None]],
    row_index: dict[str, dict],
    out_root: Path,
) -> dict:
    """Toggle necessity (arm 3) + pair_necessity (arms 1-2) from exact-match labels."""
    post_side, short_side = arm.sides[0].side, arm.sides[1].side
    post_c, short_c = correctness[post_side], correctness[short_side]
    shared = sorted(set(post_c) & set(short_c))
    labels: dict[str, str] = {}
    for rid in shared:
        p, s = post_c[rid], short_c[rid]
        if p is None or s is None:
            labels[rid] = "unknown"
        elif p and not s:
            labels[rid] = "necessary"
        elif p and s:
            labels[rid] = "both_correct"
        elif not p and not s:
            labels[rid] = "both_wrong"
        else:
            labels[rid] = "rescued_by_no_think" if arm.arm == 3 else "pre_only_correct"
    by_corpus: dict[str, Counter] = {}
    for rid, lab in labels.items():
        by_corpus.setdefault(row_index[rid]["corpus"], Counter())[lab] += 1
    summary = {c: dict(v) for c, v in by_corpus.items()}
    payload = {
        "arm": arm.arm,
        "definition": (
            "necessary(q) := exact-match correct(think/post) AND NOT correct(no-think/pre); "
            "both modes greedy; exact code, no LLM judge (plan §4.1)"
        ),
        "labels": labels,
        "question_by_row_id": {rid: row_index[rid]["question"] for rid in labels},
        "class_sizes": summary,
        "repro": repro_meta("necessity"),
    }
    name = "qwen3_toggle_labels.json" if arm.arm == 3 else f"pair_necessity_a{arm.arm}.json"
    _atomic_write_json(out_root / "out" / "necessity" / name, payload)
    return summary


# ---------------------------------------------------------------------------
# Phase drivers
# ---------------------------------------------------------------------------


def post_side(arm: ArmSpec) -> SideSpec:
    return arm.sides[0]


def short_side(arm: ArmSpec) -> SideSpec:
    return arm.sides[1]


def gd_render_asserts(
    arm: ArmSpec, rows_by_corpus: dict[str, list[dict]], out_root: Path, n_sample: int = 100
) -> dict:
    """Pod-side G-D render-identity re-assert (blocking; per-arm form, plan §7)."""
    from transformers import AutoTokenizer

    rng = np.random.default_rng(SEED + 5)
    toks = {}
    for s in arm.sides:
        rev = resolve_revision(s.model, out_root)
        toks[s.side] = AutoTokenizer.from_pretrained(s.model, revision=rev)
    a, b = arm.sides[0], arm.sides[1]
    report: dict = {"form": arm.render_identity, "n_sample_per_corpus": n_sample}
    for c, rows in rows_by_corpus.items():
        idx = rng.permutation(len(rows))[: min(n_sample, len(rows))]
        sample = [rows[int(i)] for i in idx]
        if arm.render_identity == "identical":
            for r in sample:
                ia = toks[a.side](
                    render_prompt(toks[a.side], r["user_text"], a, False), add_special_tokens=False
                )["input_ids"]
                ib = toks[b.side](
                    render_prompt(toks[b.side], r["user_text"], b, False), add_special_tokens=False
                )["input_ids"]
                assert ia == ib, (
                    f"G-D1 FAIL: {r['row_id']}: pre/post render token ids differ — a template "
                    "mismatch invalidates cells E/H (BLOCKING, plan §7)"
                )
        elif arm.render_identity == "content_only":
            for r in sample:
                ia = toks[a.side](r["user_text"], add_special_tokens=False)["input_ids"]
                ib = toks[b.side](r["user_text"], add_special_tokens=False)["input_ids"]
                assert ia == ib, (
                    f"G-D2 FAIL: {r['row_id']}: question-CONTENT token ids differ under the "
                    "two arm-2 tokenizers (shared Qwen2 BPE assumption broke)"
                )
        else:  # single_model: cross-MODE shared-prefix probe (asserted in compose too)
            tok = toks[a.side]
            for r in sample:
                on = tok(render_prompt(tok, r["user_text"], a, False), add_special_tokens=False)[
                    "input_ids"
                ]
                off = tok(render_prompt(tok, r["user_text"], b, False), add_special_tokens=False)[
                    "input_ids"
                ]
                assert off[: len(on)] == on, (
                    f"G-D3 FAIL: {r['row_id']}: arm-3 mode renders do not share the "
                    "assistant-start prefix"
                )
        report[c] = "PASS"
    if arm.render_identity == "content_only":
        # Full-render hashes recorded, EXPECTED to differ (arm-2 render confound).
        r0 = rows_by_corpus[next(iter(rows_by_corpus))][0]
        report["full_render_sha_post"] = hashlib.sha256(
            render_prompt(toks[a.side], r0["user_text"], a, False).encode()
        ).hexdigest()[:16]
        report["full_render_sha_pre"] = hashlib.sha256(
            render_prompt(toks[b.side], r0["user_text"], b, False).encode()
        ).hexdigest()[:16]
    return report


def phase_smoke(args, arm: ArmSpec, corpora: dict[str, list[dict]], out_root: Path) -> int:
    """P1 rig smoke: gates G-A..G-D + G-F on the production entrypoint."""
    phase_line("p1_smoke_rig")
    rows = smoke_slice(arm_rows(corpora, arm.arm), arm.arm)
    gd = gd_render_asserts(arm, rows, out_root, n_sample=5)
    sides = list(arm.sides) if arm.arm == 3 else [post_side(arm), short_side(arm)]
    gen_reports: dict[str, dict] = {}
    for side in sides:
        rel_total = 8 if side.post_like else 0  # reliability-rung exercise (plan §4.2 P1)
        run_generation(args, arm, side, rows, out_root, rel_total, num_workers=1)
        mode = effective_parse_mode(side, args.prefill_fallback)
        parses = []
        for c in rows:
            for rec in load_rollouts(out_root, side, c):
                p = parse_generation(rec, mode)
                p["corpus"] = c
                parses.append(p)
        gsm = [p for p in parses if p["corpus"] == "gsm8k_test"]
        gen_reports[side.side] = gate_report(gsm, side.cap)
        gen_reports[side.side]["all_corpora"] = gate_report(parses, side.cap)
    cap_res = run_capture(args, arm, rows, out_root, num_workers=1)
    for stage in [s.stage for s in sides] + [f"reliability_a{arm.arm}"]:
        upload_stage(out_root, stage, args.skip_upload, smoke=True)

    verdicts: dict[str, dict] = {}
    rc = 0
    fallback = None
    post = post_side(arm)
    g = gen_reports[post.side]
    emission = g["emission_rate"]
    if post.parse_mode == "emergent" and not args.prefill_fallback:
        ga_pass = emission >= GATE_GA_FLOOR
        if not ga_pass and emission >= GATE_GA_PREFILL_BAND:
            fallback, rc = "prefill", RC_FALLBACK_BAND
        elif not ga_pass:
            rc = RC_GATE_FAIL
        verdicts["G-A"] = {
            "emission_rate": emission,
            "floor": GATE_GA_FLOOR,
            "band": GATE_GA_PREFILL_BAND,
            "pass": ga_pass,
        }
    else:  # prefill guaranteed (arm 2) or fallback engaged: the #1005 form
        ga_pass = emission >= GATE_GA_FLOOR
        if not ga_pass:
            rc = RC_GATE_FAIL
        verdicts["G-A"] = {"emission_rate": emission, "floor": GATE_GA_FLOOR, "pass": ga_pass}
    gb_pass = (
        g["usable_rate"] >= GATE_GB_USABLE_FLOOR and g["offender_rate"] <= GATE_GB_OFFENDER_MAX
    )
    if not gb_pass:
        if arm.arm == 3 and not args.decode_fallback:
            fallback, rc = "decode", max(rc, RC_FALLBACK_BAND)
        else:
            rc = RC_GATE_FAIL
    verdicts["G-B"] = {
        "usable_rate": g["usable_rate"],
        "usable_floor": GATE_GB_USABLE_FLOOR,
        "offender_rate": g["offender_rate"],
        "offender_max": GATE_GB_OFFENDER_MAX,
        "pass": gb_pass,
    }
    verdicts["G-C"] = {
        "caphit_rate": g["caphit_rate"],
        "production_trigger": CAPHIT_TRIGGER,
        "informational_at_smoke": True,
    }
    if arm.arm == 3 and not args.decode_fallback and g["caphit_rate"] > SMOKE_CAPHIT_GB3:
        fallback, rc = "decode", max(rc, RC_FALLBACK_BAND)
        verdicts["G-B3"] = {
            "caphit_rate": g["caphit_rate"],
            "band": SMOKE_CAPHIT_GB3,
            "fallback": "T=0.6/top_p 0.95 primary (model-card decode)",
        }
    verdicts["G-D"] = gd
    gf = cap_res["gf"]
    gf_pass = bool(gf) and all(v["pass"] for v in gf.values())
    if not gf_pass:
        rc = RC_GATE_FAIL
    verdicts["G-F"] = {"per_side": gf, "pass": gf_pass}
    report = {
        "arm": arm.arm,
        "smoke": True,
        "gates": verdicts,
        "gen": gen_reports,
        "capture": cap_res["reports"],
        "fallback_available": fallback,
        "rc": rc,
        "repro": repro_meta("p1_smoke"),
    }
    _atomic_write_json(out_root / "out" / "reports" / f"smoke_a{arm.arm}.json", report)
    print(
        f"[smoke] arm {arm.arm} rc={rc} fallback={fallback} gates="
        f"{ {k: v.get('pass', 'info') for k, v in verdicts.items()} }",
        flush=True,
    )
    return rc


def phase_pilot(
    args, arm: ArmSpec, corpora: dict[str, list[dict]], out_root: Path, num_workers: int
) -> int:
    """P2a (arm 1): full gsm8k_test1319 both sides + capture; measures walls."""
    assert arm.arm == 1, "p2a pilot is arm-1 only (dispatcher guards this too)"
    phase_line("p2a_pilot_rig")
    rows = arm_rows(corpora, arm.arm)
    pilot_rows = {"gsm8k_test": rows["gsm8k_test"]}
    if args.smoke:
        pilot_rows = {"gsm8k_test": rows["gsm8k_test"][:24]}
    walls = {}
    for side in arm.sides:
        t0 = time.time()
        run_generation(args, arm, side, pilot_rows, out_root, rel_total=0, num_workers=num_workers)
        wall = time.time() - t0
        toks = sum(r["n_gen_tokens"] for r in load_rollouts(out_root, side, "gsm8k_test"))
        walls[f"gen_{side.side}"] = {
            "wall_s": wall,
            "gen_tokens": toks,
            "tok_per_s": toks / max(1.0, wall),
        }
    t0 = time.time()
    cap = run_capture(
        args, arm, pilot_rows, out_root, num_workers=num_workers, corpora_filter=["gsm8k_test"]
    )
    walls["capture"] = {
        "wall_s": time.time() - t0,
        "reports": cap["reports"],
        "upload_wall_s": {k: v.get("upload_wall_s") for k, v in cap["reports"].items()},
    }
    report = {
        "arm": 1,
        "pilot": "gsm8k_test1319",
        "walls": walls,
        "gf": cap["gf"],
        "smoke": bool(args.smoke),
        "repro": repro_meta("p2a_pilot"),
    }
    _atomic_write_json(out_root / "out" / "reports" / "pilot_a1.json", report)
    print(f"[pilot] walls: { {k: round(v['wall_s'], 1) for k, v in walls.items()} }", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if args.parser_selftest:
        run_parser_selftest()
        raise SystemExit(0)
    if args.worker_slot is not None:
        assert args.work_file, "--worker-slot requires --work-file"
        if args.worker_kind == "gen":
            run_gen_worker(args)
        elif args.worker_kind == "capture":
            run_capture_worker(args)
        raise SystemExit(f"unknown --worker-kind {args.worker_kind!r}")

    arm = ARMS[args.arm]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ngpu = gpu_count()
    if ngpu < 1:
        raise SystemExit("no GPUs visible — this entrypoint's phases are GPU-bound")
    num_workers = 1 if (args.smoke and args.phase == "smoke") else min(ngpu, args.num_workers)
    logger.info(
        "[main] arm=%d phase=%s smoke=%s workers=%d out_root=%s",
        arm.arm,
        args.phase,
        bool(args.smoke),
        num_workers,
        out_root,
    )
    corpora, _manifest = stage_corpora(out_root, args.corpora_dir)

    if args.phase == "smoke":
        return phase_smoke(args, arm, corpora, out_root)
    if args.phase == "pilot":
        return phase_pilot(args, arm, corpora, out_root, num_workers)

    rows = arm_rows(corpora, arm.arm)
    if args.smoke:
        rows = smoke_slice(rows, arm.arm)
    if args.phase == "gen-post":
        phase_line("p2_gen_post_rig")
        side = post_side(arm)
        rel = 0 if args.smoke else REL_TOTAL_POST[arm.arm]
        run_generation(args, arm, side, rows, out_root, rel, num_workers)
        upload_stage(out_root, side.stage, args.skip_upload, smoke=bool(args.smoke))
        upload_stage(out_root, f"reliability_a{arm.arm}", args.skip_upload, smoke=bool(args.smoke))
        upload_stage(out_root, f"regen16k_a{arm.arm}", args.skip_upload, smoke=bool(args.smoke))
        return 0
    if args.phase == "gen-short":
        phase_line("p3_gen_short_rig")
        side = short_side(arm)
        rel = 0 if args.smoke else REL_TOTAL_SHORT
        run_generation(args, arm, side, rows, out_root, rel, num_workers)
        upload_stage(out_root, side.stage, args.skip_upload, smoke=bool(args.smoke))
        upload_stage(out_root, f"reliability_a{arm.arm}", args.skip_upload, smoke=bool(args.smoke))
        upload_stage(out_root, f"regen16k_a{arm.arm}", args.skip_upload, smoke=bool(args.smoke))
        return 0
    if args.phase == "capture":
        phase_line("p4_capture_rig")
        res = run_capture(args, arm, rows, out_root, num_workers)
        gf = res["gf"]
        if gf and not all(v["pass"] for v in gf.values()):
            _atomic_write_json(
                out_root / "out" / "reports" / f"gf_fail_a{arm.arm}.json",
                {"gf": gf, "repro": repro_meta("p4_capture")},
            )
            print(f"[capture] G-F determinism gate FAILED: {gf}", flush=True)
            return RC_GATE_FAIL
        _atomic_write_json(
            out_root / "out" / "reports" / f"capture_a{arm.arm}.json",
            {
                "reports": res["reports"],
                "gf": gf,
                "necessity_summary": res["necessity_summary"],
                "repro": repro_meta("p4_capture"),
            },
        )
        return 0
    raise SystemExit(f"unknown --phase {args.phase!r}")


# ---------------------------------------------------------------------------
# CPU parser self-test (offline; the VM-side smoke for this unit)
# ---------------------------------------------------------------------------


def run_parser_selftest() -> None:
    """Synthetic-string checks of the per-arm parse semantics + span mapping.

    Offline (no tokenizer, no model): asserts segment_completion_arm spans per
    arm delimiter shape, char->token mapping on synthetic offsets, quota
    arithmetic, and the exact-match correctness rules.
    """
    # emergent (arm 1 / arm-3 think-on)
    t = "<think>\nreason a b c\n</think>\n\nThe answer is \\boxed{42}."
    wf, reason, cot, ans = segment_completion_arm(t, "emergent")
    assert wf and reason == "", (wf, reason)
    assert t[cot[0] : cot[1]] == "reason a b c", t[cot[0] : cot[1]]
    assert t[ans[0] : ans[1]].startswith("The answer"), t[ans[0] : ans[1]]
    wf, reason, *_ = segment_completion_arm("no think block at all", "emergent")
    assert not wf and reason == "open0_close0"
    wf, reason, *_ = segment_completion_arm("preface <think>x</think> y", "emergent")
    assert not wf and reason == "text_before_open"
    wf, reason, *_ = segment_completion_arm("<think>x</think>", "emergent")
    assert not wf and reason == "empty_answer"
    # prefill (arm-2 post / arm-1 fallback): prompt carries <think>; one close only
    t2 = "step 1 then step 2\n</think>\nFinal: \\boxed{7}"
    wf, reason, cot, ans = segment_completion_arm(t2, "prefill")
    assert wf and t2[cot[0] : cot[1]] == "step 1 then step 2"
    assert t2[ans[0] : ans[1]] == "Final: \\boxed{7}"
    wf, reason, *_ = segment_completion_arm("<think>bad</think>ans", "prefill")
    assert not wf and reason == "unexpected_open_tag"
    wf, reason, *_ = segment_completion_arm("no close tag here", "prefill")
    assert not wf and reason == "close_count_0"
    # off (arm-3 think-off / plain pre-side)
    wf, reason, cot, ans = segment_completion_arm("  Just the answer.  ", "off")
    assert wf and cot == (0, 0) and "  Just the answer.  "[ans[0] : ans[1]] == "Just the answer."
    # truncation reclass
    p = parse_generation({"text": "<think>going on and on", "finish_reason": "length"}, "emergent")
    assert not p["well_formed"] and p["reason"] == "truncated_no_close"
    # degenerate repetition reclass
    loop = " ".join(["repeat the same loop words"] * 20)
    p = parse_generation(
        {"text": f"<think>{loop}</think>\n\nAnswer: \\boxed{{1}}", "finish_reason": "stop"},
        "emergent",
    )
    assert not p["well_formed"] and p["reason"] == "degenerate_repetition"
    # char->token mapping on synthetic offsets (BPE-merge-robust half-open spans)
    offsets = [(0, 4), (4, 9), (9, 12), (12, 20)]
    assert char_span_to_token_span(offsets, (4, 12)) == (1, 3)
    assert char_span_to_token_span(offsets, (5, 6)) == (1, 2)
    assert char_span_to_token_span(offsets, (20, 25)) == (0, 0)  # no overlap -> counted drop
    # t-grid arithmetic: round(t * (L-1)) with collided positions kept
    n_think = 4
    tpos = [int(round(tt * (n_think - 1))) for tt in T_GRID]
    assert tpos == [0, 1, 1, 1, 2, 2, 2, 2, 3], tpos
    # read-point conventions
    side2 = ARMS[2].sides[0]
    ids = [1, 2, 3, 151648, 11]
    assert compute_read_idx(side2, ids, prefill_fallback=False) == 2
    side3 = ARMS[3].sides[1]
    assert compute_read_idx(side3, list(range(10)), prefill_fallback=False, on_prompt_len=6) == 5
    # correctness rules
    assert exact_match_correct("math", "so \\boxed{42}", "42") is True
    assert exact_match_correct("gsm8k_test", "\\boxed{1,000}", "1000") is True
    assert exact_match_correct("math", "no box here", "42") is False
    assert exact_match_correct("mmlu", "The answer is B.", "B") is True
    assert exact_match_correct("mmlu", "The answer is B.", "C") is False
    assert exact_match_correct("contexthub", "Therefore:\nTrue", "true") is True
    assert exact_match_correct("math", "x", None) is None
    # quota arithmetic: arm-1 base realizes the plan totals at 4 k-bins + 8 cells
    sizes = {f"gsm8k_train:k{i}": 10_000 for i in range(4)}
    sizes.update({f"contexthub:c{i}": 10_000 for i in range(8)})
    sizes.update(
        {"math": 10_000, "mmlu": 10_000, "arc_challenge": 10_000, "csqa": 10_000, "piqa": 10_000}
    )
    alloc = scaled_quota(1500, sizes)
    assert sum(alloc.values()) == 1500
    assert alloc["math"] == 200 and alloc["gsm8k_train:k0"] == 150
    assert alloc["contexthub:c0"] == 50 and alloc["mmlu"] == 100
    alloc2 = scaled_quota(1000, sizes)
    assert sum(alloc2.values()) == 1000
    # boundary-char arithmetic for cot_boundary
    text = "<think>x</think>ans"
    close_char = text.index(THINK_CLOSE) + len(THINK_CLOSE) - 1
    assert text[close_char] == ">"
    print(
        "[parser-selftest] PASS: segment semantics, span mapping, t-grid, "
        "read points, correctness rules, quota arithmetic",
        flush=True,
    )


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument(
        "--phase",
        default="smoke",
        choices=("smoke", "pilot", "gen-post", "gen-short", "capture"),
        help="dispatcher contract: absent --phase + --smoke = the P1 rig smoke",
    )
    ap.add_argument("--out-root", default="/workspace/issue2546")
    ap.add_argument("--smoke", action="store_true", help="tiny-slice rehearsal mode")
    ap.add_argument("--corpora-dir", default=None, help="local corpora_v1 bundle override")
    ap.add_argument("--num-workers", type=int, default=4, help="max GPU workers (capped at ngpu)")
    ap.add_argument("--skip-upload", action="store_true", help="skip HF uploads (local smoke)")
    ap.add_argument(
        "--prefill-fallback",
        action="store_true",
        help="G-A declared fallback rung: assistant-prefix <think>\\n prefill with #1005 "
        "parser semantics (dispatcher sets this after the smoke gate; never silent)",
    )
    ap.add_argument(
        "--decode-fallback",
        action="store_true",
        help="G-B3 declared arm-3 fallback: T=0.6/top_p 0.95 PRIMARY decode "
        "(model-card recommendation; recorded as an arm-3 scope deviation)",
    )
    ap.add_argument("--worker-slot", type=int, default=None, help="internal: GPU worker slot")
    ap.add_argument(
        "--worker-kind",
        default="gen",
        choices=("gen", "capture"),
        help="internal: worker entrypoint kind",
    )
    ap.add_argument("--work-file", default=None, help="internal: worker worklist JSON")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attribute completeness + helper-call bind check",
    )
    ap.add_argument(
        "--parser-selftest",
        action="store_true",
        help="offline CPU self-test of parse/span/quota/correctness logic",
    )
    return ap


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
